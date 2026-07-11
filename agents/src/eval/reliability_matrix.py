# agents/src/eval/reliability_matrix.py
"""
Per-(source, class) confusion-matrix data model for P3 reliability fitting.

Built from audit report JSONs against ground-truth labels. Each cell is the
2×2 confusion matrix of "did source X emit a positive signal for class Y"
vs "is class Y in the contract's ground-truth labels".

Sources today: ml, slither, aderyn (per consensus_verdict[cls] fields:
    ml_signal (0/1), slither_match (int), aderyn_match (int)).
Future P8 sources (halmos, gigahorse, taint, …) extend the source set —
the matrix builder picks up new sources automatically if their per-class
positive signal is exposed in the report.

This module is pure data + (de)serialisation; the builder lives in
`agents/scripts/build_reliability_matrix.py` and the shrinkage fitter in
`agents/src/eval/reliability_fit.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import isnan
from pathlib import Path
from typing import Any

# Decision-number — the three sources the consensus_verdict schema exposes
# today. P8 channels will add to the union of sources discovered per-report.
KNOWN_SOURCES: tuple[str, ...] = ("ml", "slither", "aderyn")


@dataclass
class Cell:
    """One 2×2 confusion-matrix cell for (source, class)."""
    source: str
    cls: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def measured_precision(self) -> float:
        """tp / (tp + fp); 0.0 when undefined (no predicted positives)."""
        denom = self.tp + self.fp
        if denom == 0:
            return 0.0
        return self.tp / denom

    def as_dict(self) -> dict[str, Any]:
        p = self.measured_precision
        return {
            "source": self.source,
            "cls": self.cls,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "n": self.n,
            "measured_precision": 0.0 if isnan(p) else p,
        }


@dataclass
class Matrix:
    """The full (source × class) confusion grid + provenance."""
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    n_contracts: int = 0
    classes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    reports_dir: str = ""
    corpus_dir: str = ""
    # Per-tool: list of contract stems that the tool did NOT run on (excluded
    # from that tool's TP+FP+FN+TN counts). Per Rule 5C (CLAUDE.md, 2026-06-25),
    # the matrix must distinguish "tool ran and found nothing" from "tool was
    # absent" — an empty positive signal is a lie when the tool didn't run.
    excluded_contracts: dict[str, list[str]] = field(default_factory=dict)

    def cell(self, source: str, cls: str) -> Cell:
        key = (source, cls)
        if key not in self.cells:
            self.cells[key] = Cell(source=source, cls=cls)
        return self.cells[key]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_contracts": self.n_contracts,
            "classes": sorted(self.classes),
            "sources": sorted(self.sources),
            "reports_dir": self.reports_dir,
            "corpus_dir": self.corpus_dir,
            "excluded_contracts": {
                tool: sorted(stems) for tool, stems in self.excluded_contracts.items()
            },
            "cells": [
                self.cells[k].as_dict()
                for k in sorted(self.cells.keys())
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Matrix":
        m = cls(
            n_contracts=int(data.get("n_contracts", 0)),
            classes=list(data.get("classes", [])),
            sources=list(data.get("sources", [])),
            reports_dir=str(data.get("reports_dir", "")),
            corpus_dir=str(data.get("corpus_dir", "")),
        )
        # excluded_contracts — older matrices may not have this field; default {}.
        for tool, stems in (data.get("excluded_contracts") or {}).items():
            m.excluded_contracts[tool] = list(stems)
        for c in data.get("cells", []):
            cell = Cell(
                source=c["source"], cls=c["cls"],
                tp=int(c["tp"]), fp=int(c["fp"]),
                fn=int(c["fn"]), tn=int(c["tn"]),
            )
            m.cells[(cell.source, cell.cls)] = cell
        return m

    @classmethod
    def load(cls, path: Path) -> "Matrix":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _tool_ran(row: Any, source: str) -> bool:
    """
    Per Rule 5C: a tool that did NOT run on a contract must be excluded from
    that tool's confusion-matrix counts. `row.tool_status` is the dataclass
    field loaded from `report["tool_status"]`; if it's missing (older rows)
    or the source isn't in it, assume the tool ran (legacy behaviour).
    """
    ts = getattr(row, "tool_status", None) or {}
    entry = ts.get(source)
    if not isinstance(entry, dict):
        return True  # no entry → assume ran (legacy compat)
    return entry.get("ran") is not False


def build_matrix(
    rows: list[Any],
    classes: list[str],
    sources: tuple[str, ...] = KNOWN_SOURCES,
    reports_dir: str = "",
    corpus_dir: str = "",
) -> Matrix:
    """
    Build the (source × class) confusion matrix from labelled ContractEval rows.

    A source's positive signal for class `cls` on contract `row`:
        ml      : row.consensus_verdict[cls].ml_signal == 1
        slither : row.consensus_verdict[cls].slither_match > 0
        aderyn  : row.consensus_verdict[cls].aderyn_match > 0

    Ground truth: `cls in row.labels`.

    Per cell:
        TP = positive signal AND class in labels
        FP = positive signal AND class NOT in labels
        FN = no signal   AND class in labels
        TN = no signal   AND class NOT in labels

    Rule 5C (CLAUDE.md, 2026-06-25): contracts where a tool did NOT run
    (e.g., ML service failed on 22 specific contracts) are EXCLUDED from
    that tool's TP+FP+FN+TN counts. They're recorded in
    `matrix.excluded_contracts[source]` with the contract stems. The
    total `n_contracts` is the count of all rows (including the excluded
    ones); the per-cell `n` reflects only the contracts where the tool
    ran.
    """
    m = Matrix(
        n_contracts=len(rows),
        classes=list(classes),
        sources=list(sources),
        reports_dir=reports_dir,
        corpus_dir=corpus_dir,
    )

    def _positive(source: str, cv_cls: dict | None) -> bool:
        if cv_cls is None:
            return False
        if source == "ml":
            return int(cv_cls.get("ml_signal", 0)) == 1
        if source == "slither":
            return int(cv_cls.get("slither_match", 0)) > 0
        if source == "aderyn":
            return int(cv_cls.get("aderyn_match", 0)) > 0
        # Future sources: off by default until their report schema is wired.
        return False

    for row in rows:
        labels = set(row.labels)
        for source in sources:
            if not _tool_ran(row, source):
                # Rule 5C: tool didn't run on this contract — exclude.
                m.excluded_contracts.setdefault(source, []).append(row.stem)
                continue
            for cls in classes:
                cv_cls = row.consensus_verdict.get(cls) if row.consensus_verdict else None
                in_labels = cls in labels
                pred_pos = _positive(source, cv_cls)
                cell = m.cell(source, cls)
                if pred_pos and in_labels:
                    cell.tp += 1
                elif pred_pos and not in_labels:
                    cell.fp += 1
                elif not pred_pos and in_labels:
                    cell.fn += 1
                else:
                    cell.tn += 1

    return m