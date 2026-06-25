"""
regression.py — Baseline save/load + comparison.

A `RegressionBaseline` is a snapshot of a `PipelineMetrics` result
serialised to JSON. It lets you:
  1. Capture the "known good" state (e.g. the pre-redesign baseline).
  2. Compare a fresh run against that baseline to detect regressions.

Format matches what `PipelineMetrics.as_dict()` produces, so a baseline
file is just the JSON dict of a previous run. The existing
`agents/eval/baselines/pre_redesign.json` was produced by the WS0
script; this class can both read and write that format.

Usage:
    baseline = RegressionBaseline.load(Path("eval/baselines/pre_redesign.json"))
    current = PipelineMetrics(contracts).compute()  # compute_aggregates() inside
    diff = baseline.compare(current)
    if diff.regressed:
        sys.exit(1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.pipeline_metrics import PipelineMetrics


@dataclass
class RegressionResult:
    """
    The result of comparing a current run against a baseline.

    `regressed` is True if the current run is worse than the baseline on
    macro-F1 (the headline metric). `metric_deltas` lists every headline
    metric that changed, with its current/baseline/delta values.
    `per_class_deltas` lists the per-class F1 deltas (always populated,
    even for classes that appear in only one of baseline/current).
    """
    baseline_macro_f1:    float
    current_macro_f1:     float
    baseline_macro_fbeta: float
    current_macro_fbeta:  float
    baseline_micro_f1:    float
    current_micro_f1:     float
    regressed:            bool
    metric_deltas:        dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-class F1 deltas (baseline_f1, current_f1, delta). Populated for
    # every class that appears in baseline OR current.
    per_class_deltas:     dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-class F1 regressions (current < baseline by >min_delta_pp).
    regressed_classes:    list[str] = field(default_factory=list)
    # Per-class F1 improvements (current > baseline by >min_delta_pp).
    improved_classes:     list[str] = field(default_factory=list)


class RegressionBaseline:
    """
    Read/write a stored baseline (JSON file).

    The baseline format is exactly what `PipelineMetrics.as_dict()`
    produces — so you can dump a run, mark it as the baseline, and
    later compare any future run against it.
    """

    def __init__(self, data: dict[str, Any], source_path: Path | None = None):
        self.data = data
        self.source_path = source_path

    # ----- Convenience accessors -----

    @property
    def macro_f1(self) -> float:
        return float(self.data.get("macro_f1", 0.0))

    @property
    def macro_fbeta(self) -> float:
        return float(self.data.get("macro_fbeta", self.macro_f1))

    @property
    def micro_f1(self) -> float:
        return float(self.data.get("micro_f1", 0.0))

    @property
    def contract_count(self) -> int:
        return int(self.data.get("contract_count", 0))

    @property
    def per_class(self) -> dict[str, dict[str, Any]]:
        return self.data.get("per_class", {}) or {}

    # ----- IO -----

    @staticmethod
    def load(path: Path) -> "RegressionBaseline":
        """Load a baseline from a JSON file."""
        if not path.is_file():
            raise FileNotFoundError(f"baseline file not found: {path}")
        data = json.loads(path.read_text())
        return RegressionBaseline(data, source_path=path)

    @staticmethod
    def from_metrics(metrics: PipelineMetrics) -> "RegressionBaseline":
        """Capture a current run as a baseline (in memory, not saved yet)."""
        return RegressionBaseline(metrics.as_dict())

    def save(self, path: Path) -> None:
        """Persist this baseline to a JSON file (creates parent dirs)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2, default=str))

    # ----- Comparison -----

    def compare(
        self,
        current: PipelineMetrics,
        *,
        min_delta_pp: float = 0.01,
    ) -> RegressionResult:
        """
        Compare `current` against this baseline.

        `min_delta_pp` (default 1pp) is the threshold for "regressed"/
        "improved" class-level deltas. Smaller deltas are still
        recorded in `metric_deltas` but don't trigger a pass/fail.

        A run is "regressed" iff current macro-F1 < baseline macro-F1.
        """
        cur = current.as_dict()
        cur_macro = float(cur.get("macro_f1", 0.0))
        cur_macro_fbeta = float(cur.get("macro_fbeta", cur_macro))
        cur_micro = float(cur.get("micro_f1", 0.0))

        regressed = cur_macro < self.macro_f1

        # Per-class F1 deltas.
        regressed_classes: list[str] = []
        improved_classes:  list[str] = []
        base_per_class = self.per_class
        cur_per_class  = cur.get("per_class", {}) or {}
        all_classes = set(base_per_class.keys()) | set(cur_per_class.keys())
        per_class_deltas: dict[str, dict[str, float]] = {}
        for cls in sorted(all_classes):
            base_f1 = float(base_per_class.get(cls, {}).get("f1", 0.0))
            cur_f1  = float(cur_per_class.get(cls,  {}).get("f1", 0.0))
            delta = cur_f1 - base_f1
            per_class_deltas[cls] = {
                "baseline_f1": base_f1,
                "current_f1":  cur_f1,
                "delta":       delta,
            }
            if delta < -min_delta_pp:
                regressed_classes.append(cls)
            elif delta > min_delta_pp:
                improved_classes.append(cls)

        # Headline metric deltas.
        metric_deltas: dict[str, dict[str, float]] = {
            "macro_f1": {
                "baseline": self.macro_f1,
                "current":  cur_macro,
                "delta":    cur_macro - self.macro_f1,
            },
            "macro_fbeta": {
                "baseline": self.macro_fbeta,
                "current":  cur_macro_fbeta,
                "delta":    cur_macro_fbeta - self.macro_fbeta,
            },
            "micro_f1": {
                "baseline": self.micro_f1,
                "current":  cur_micro,
                "delta":    cur_micro - self.micro_f1,
            },
            "contract_accuracy_loose": {
                "baseline": float(self.data.get("contract_accuracy_loose", 0.0)),
                "current":  float(cur.get("contract_accuracy_loose", 0.0)),
                "delta":    float(cur.get("contract_accuracy_loose", 0.0)) - float(self.data.get("contract_accuracy_loose", 0.0)),
            },
        }

        return RegressionResult(
            baseline_macro_f1=self.macro_f1,
            current_macro_f1=cur_macro,
            baseline_macro_fbeta=self.macro_fbeta,
            current_macro_fbeta=cur_macro_fbeta,
            baseline_micro_f1=self.micro_f1,
            current_micro_f1=cur_micro,
            regressed=regressed,
            metric_deltas=metric_deltas,
            per_class_deltas=per_class_deltas,
            regressed_classes=regressed_classes,
            improved_classes=improved_classes,
        )
