"""
benchmarks.py — Contract corpus loader for the eval framework.

A `Benchmark` represents one named evaluation set (e.g. the 88-contract
WS0 corpus). It knows:
  - Where the .sol contracts live
  - Where the ground-truth labels live (json sidecar OR `// expect:` header)
  - Where the audit reports live (paired by stem)

It produces a list of `ContractMetrics` (one per contract) that the
PipelineMetrics class can then aggregate.

Label resolution order (per `_derive_labels_for_stem`):
  1. `<stem>.json` sidecar in the corpus dir (benchmark v0.1 + edge cases)
  2. `// expect:` header in the .sol file itself (manual_hand_written_contracts/)

`ground_truth` is "vulnerable" if the labels list is non-empty, else "safe".

Usage:
    bench = Benchmark(
        name="ws0_88",
        corpus_dir=Path("agents/eval/corpus_combined"),
        reports_dir=Path("agents/eval/runs/2026-06-22-llm-on"),
    )
    contracts = bench.load()
    metrics = metrics_from_contracts(contracts)
    print(metrics.macro_f1)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.pipeline_metrics import ContractMetrics


@dataclass
class Benchmark:
    """
    One evaluation set.

    Fields are intentionally simple so this can be constructed from CLI
    args, a config file, or a notebook.
    """
    name:       str
    corpus_dir: Path
    reports_dir: Path
    positive_verdicts: frozenset[str] = field(
        default_factory=lambda: frozenset({"CONFIRMED", "LIKELY"})
    )

    # ----- Sidecar + sol index (private) -----

    def _build_sidecar_index(self) -> dict[str, Path]:
        """
        Index every <stem>.json sidecar under corpus_dir by stem.
        Stems are unique within the benchmark (filenames are designed
        to be unique across class dirs).

        Excludes benchmark-level manifest files (tier_a_manifest.json,
        contamination_check.json) that aren't per-contract labels.

        Uses os.walk(followlinks=True) so symlinked corpus dirs work.
        """
        index: dict[str, Path] = {}
        for root, _dirs, files in os.walk(self.corpus_dir, followlinks=True):
            for f in files:
                if not f.endswith(".json"):
                    continue
                if f in ("tier_a_manifest.json", "contamination_check.json"):
                    continue
                index[Path(f).stem] = Path(root) / f
        return index

    def _build_sol_index(self) -> dict[str, Path]:
        """Index every <stem>.sol under corpus_dir by stem."""
        index: dict[str, Path] = {}
        for root, _dirs, files in os.walk(self.corpus_dir, followlinks=True):
            for f in files:
                if f.endswith(".sol"):
                    index[Path(f).stem] = Path(root) / f
        return index

    @staticmethod
    def _parse_expect_header(sol_path: Path) -> tuple[list[str], str] | None:
        """
        Parse `// expect:` labels from the first few lines of a .sol file.

        Format (from manual_hand_written_contracts):
            // expect: Reentrancy,LIKELY  ExternalBug,SAFE
        (or the variants the comparator already supports)

        Returns (labels, ground_truth) where ground_truth is
        "vulnerable" if any class has a non-SAFE verdict in the header,
        else "safe".
        """
        try:
            lines = sol_path.read_text(encoding="utf-8", errors="replace").splitlines()[:30]
        except OSError:
            return None

        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("//"):
                continue
            text = stripped[2:].strip()
            lower = text.lower()
            if not lower.startswith("expect"):
                continue
            # Accept a few formats: "expect: X", "expected: X", "expects: X".
            if ":" not in text:
                continue
            body = text.split(":", 1)[1].strip()
            if not body:
                return None

            labels: list[str] = []
            # Parse comma- or whitespace-separated entries. Each entry can
            # be either a bare class name ("Reentrancy") or a class+verdict
            # pair ("Reentrancy,CONFIRMED"). Verdict, if present, is informational
            # only — the COMPARE_PATH class set is the labels.
            for token in body.replace("\t", " ").split(","):
                tok = token.strip()
                if not tok:
                    continue
                parts = tok.split()
                # Heuristic: if first part is a known class name, keep it.
                # Otherwise, treat the whole token as a class name.
                cls_name = parts[0] if parts else tok
                if cls_name and cls_name not in ("//",):
                    labels.append(cls_name)
            if not labels:
                return None
            # Reasonable default: if labels is non-empty, ground truth is
            # "vulnerable". The "safe" class is implicit — if a contract
            # has no labels, it's a safe baseline.
            return labels, "vulnerable"
        return None

    def _derive_labels_for_stem(
        self,
        stem: str,
        sol_path: Path | None,
        sidecar_index: dict[str, Path],
    ) -> tuple[list[str], str, Path | None, str] | None:
        """
        Resolve ground truth for one contract.

        Returns (labels, ground_truth, label_source_path, label_format) or
        None if no label source could be found (in which case the
        contract is excluded from the benchmark — can't evaluate it).
        """
        # 1. JSON sidecar (benchmark v0.1 + edge cases).
        sidecar = sidecar_index.get(stem)
        if sidecar is not None:
            try:
                data = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                pass
            else:
                # Accept both {"vulnerabilities": [...]} and {"labels": [...]} shapes.
                labels_raw = data.get("vulnerabilities") or data.get("labels") or []
                labels = [str(x) for x in labels_raw]
                gt = "vulnerable" if labels else "safe"
                return labels, gt, sidecar, "json"

        # 2. // expect: header in the .sol file (manual_hand_written_contracts/).
        if sol_path is not None:
            parsed = self._parse_expect_header(sol_path)
            if parsed is not None:
                labels, gt = parsed
                return labels, gt, sol_path, "expect_header"

        return None

    def _load_report(self, report_path: Path) -> dict[str, Any]:
        """Load a report JSON, returning {} on any error."""
        try:
            return json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _build_contract_metrics(
        self,
        report_path: Path,
        sidecar_index: dict[str, Path],
        sol_index: dict[str, Path],
    ) -> ContractMetrics | None:
        """
        Build one ContractMetrics from a report file, matching it to its
        corpus entry. Returns None if no label source is found.
        """
        stem = report_path.stem.removesuffix("_report")
        sol_path = sol_index.get(stem)
        label_info = self._derive_labels_for_stem(stem, sol_path, sidecar_index)
        if label_info is None:
            return None
        labels, gt, _, _ = label_info

        raw = self._load_report(report_path)
        ml_result = raw.get("ml_result", {}) or {}
        probabilities = ml_result.get("probabilities", {}) or {}
        final_report = raw.get("final_report", {}) or {}

        # Verdicts live in two places depending on whether cross_validator ran.
        verdicts: dict[str, str] = (
            raw.get("verdicts")
            or final_report.get("verdicts")
            or {}
        )
        verdicts = {str(k): str(v) for k, v in verdicts.items()}

        return ContractMetrics(
            stem=stem,
            report_path=str(report_path),
            labels=labels,
            ground_truth=gt,
            verdicts=verdicts,
            probabilities={str(k): float(v) for k, v in probabilities.items()},
            overall_verdict=final_report.get("overall_verdict"),
            path_taken=final_report.get("path_taken", "unknown"),
            error=raw.get("error"),
        )

    # ----- Public entry point -----

    def load(self) -> list[ContractMetrics]:
        """
        Walk the reports dir, match each report to its benchmark entry, and
        return a list of ContractMetrics. Reports with no matching label
        source are silently skipped (with a logger.info if you wire one in).
        """
        if not self.reports_dir.is_dir():
            raise FileNotFoundError(f"reports_dir not found: {self.reports_dir}")
        if not self.corpus_dir.is_dir():
            raise FileNotFoundError(f"corpus_dir not found: {self.corpus_dir}")

        sidecar_index = self._build_sidecar_index()
        sol_index = self._build_sol_index()

        contracts: list[ContractMetrics] = []
        for report_path in sorted(self.reports_dir.glob("*_report.json")):
            row = self._build_contract_metrics(report_path, sidecar_index, sol_index)
            if row is not None:
                contracts.append(row)
        return contracts
