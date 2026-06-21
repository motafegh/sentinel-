"""
threshold_sensitivity.py — Per-class threshold sensitivity analysis.

WHY THIS EXISTS

A model that has F1=0.88 at threshold 0.35 (the default) but F1=0.0 at
threshold 0.50 is gaming the threshold. The model isn't actually
discriminating the class well — it's relying on the threshold to
squeeze out a good F1.

ExternalBug on Run 12 had F1=0.88 at the tuned threshold. The
behavioral probes (C.2.4) caught the FP, but a threshold sensitivity
analysis would also catch it: the ExternalBug probability is high
(0.85) on safe contracts, so changing the threshold doesn't help.

WHAT IT DOES

For each class:
1. Run the model on a reference benchmark (e.g., v0.1 honest benchmark)
2. For each candidate threshold (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
   - Compute F1, precision, recall
3. Flag classes where:
   - F1 swings by > 0.20 across thresholds (gaming)
   - F1 is high at all thresholds (>0.80) — likely over-prediction
   - F1 is low at all thresholds (<0.20) — model doesn't learn this class

USAGE

    # Default: Run 12 FINAL on v0.1 benchmark
    python ml/testing_specs/threshold_sensitivity.py \\
        --checkpoint ml/checkpoints/Run12_FINAL.pt \\
        --benchmark data_module/benchmarks/v0_1_honest/ \\
        --output ml/checkpoints/Run12_threshold_sensitivity.json \\
        --exit-on-fail
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger

# Default thresholds to sweep
DEFAULT_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Flags
MAX_F1_SWING = 0.20  # WARN if F1 swings by more than this
MAX_F1_AT_ALL_THRESHOLDS = 0.80  # WARN if F1 is high at all thresholds (over-prediction)
MIN_F1_AT_ALL_THRESHOLDS = 0.20  # WARN if F1 is low at all thresholds (under-learning)
MIN_N_POSITIVES = 5  # Don't flag "F1 < X at all thresholds" if there are fewer positives than this.
                      # F1 is unstable with n_positives < 5; flags would be noise.


@dataclass
class ThresholdResult:
    """Threshold sensitivity result for one class."""

    class_name: str
    n_samples: int
    n_positives: int  # Ground truth positive count
    n_predicted_positives: dict  # threshold -> count
    f1_at_thresholds: dict  # threshold -> F1
    precision_at_thresholds: dict  # threshold -> precision
    recall_at_thresholds: dict  # threshold -> recall
    f1_swing: float  # max F1 - min F1 across thresholds
    flags: list  # list of issue flags (e.g., "gaming", "over_predicting")

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute F1, precision, recall for binary labels."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def analyze_class(
    class_name: str,
    y_true: list[int],
    probs: list[float],
    thresholds: list[float] = None,
) -> ThresholdResult:
    """Analyze threshold sensitivity for one class."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    n_samples = len(y_true)
    n_positives = sum(y_true)

    f1_at = {}
    precision_at = {}
    recall_at = {}
    n_predicted = {}

    for t in thresholds:
        y_pred = [1 if p >= t else 0 for p in probs]
        m = _compute_metrics(y_true, y_pred)
        f1_at[t] = m["f1"]
        precision_at[t] = m["precision"]
        recall_at[t] = m["recall"]
        n_predicted[t] = sum(y_pred)

    f1_swing = max(f1_at.values()) - min(f1_at.values()) if f1_at else 0.0

    flags = []
    if f1_swing > MAX_F1_SWING:
        flags.append(f"F1 gaming: swings by {f1_swing:.2f} across thresholds")
    if n_positives >= MIN_N_POSITIVES:
        if all(f1 > MAX_F1_AT_ALL_THRESHOLDS for f1 in f1_at.values()):
            flags.append(f"F1 > {MAX_F1_AT_ALL_THRESHOLDS} at all thresholds — over-prediction")
        if all(f1 < MIN_F1_AT_ALL_THRESHOLDS for f1 in f1_at.values()):
            flags.append(f"F1 < {MIN_F1_AT_ALL_THRESHOLDS} at all thresholds — under-learning")
    else:
        # Don't flag based on F1 with too few positives. F1 is unstable at
        # n_positives < MIN_N_POSITIVES. Note this so the user knows.
        if any(f1 > MAX_F1_AT_ALL_THRESHOLDS for f1 in f1_at.values()):
            # Still flag over-prediction, since that's a recall-irrelevant signal
            # that the model is producing high probabilities for everything.
            if all(p >= MAX_F1_AT_ALL_THRESHOLDS for p in probs):
                flags.append(
                    f"over-prediction: all {n_samples} probs >= {MAX_F1_AT_ALL_THRESHOLDS} "
                    f"(n_positives={n_positives} < {MIN_N_POSITIVES} — F1 not evaluated)"
                )

    return ThresholdResult(
        class_name=class_name,
        n_samples=n_samples,
        n_positives=n_positives,
        n_predicted_positives=n_predicted,
        f1_at_thresholds=f1_at,
        precision_at_thresholds=precision_at,
        recall_at_thresholds=recall_at,
        f1_swing=f1_swing,
        flags=flags,
    )


def load_benchmark(benchmark_dir: Path) -> list[dict]:
    """Load benchmark contracts with ground truth labels.

    Expected format: a directory with .sol files and a manifest.json
    mapping filename -> {class: [label1, label2, ...]}.
    """
    manifest_path = benchmark_dir / "manifest.json"
    if not manifest_path.exists():
        # Try the v0.1 benchmark format
        return _load_v01_benchmark(benchmark_dir)

    with manifest_path.open() as f:
        manifest = json.load(f)

    contracts = []
    for entry in manifest.get("contracts", []):
        path = benchmark_dir / entry["file"]
        if path.exists():
            contracts.append({
                "path": str(path),
                "source": path.read_text(),
                "labels": entry.get("labels", []),
            })
    return contracts


def _load_v01_benchmark(benchmark_dir: Path) -> list[dict]:
    """Load the v0.1 honest benchmark format.

    Format: each .sol file has a corresponding .json sidecar with metadata.
    """
    contracts = []
    for sol_path in sorted(benchmark_dir.rglob("*.sol")):
        json_path = sol_path.with_suffix(".json")
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text())
            except Exception:
                continue
            contracts.append({
                "path": str(sol_path),
                "source": sol_path.read_text(),
                "labels": meta.get("labels", []),
            })
    return contracts


def run_sensitivity(
    predictor: Any,
    benchmark_contracts: list[dict],
    class_names: list[str] = None,
    thresholds: list[float] = None,
) -> dict[str, Any]:
    """Run threshold sensitivity analysis on a benchmark.

    Args:
        predictor: Object with .predict(source_code) -> {probabilities: {class: float}}.
        benchmark_contracts: List of {path, source, labels} dicts.
        class_names: List of class names. Default: SENTINEL 10 classes.
        thresholds: List of thresholds to sweep. Default: 0.10 to 0.90 step 0.10.

    Returns:
        Dict with per-class results and overall summary.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    # Default class list — should be loaded from somewhere but this is a fallback
    class_names = class_names or [
        "Reentrancy", "CallToUnknown", "Timestamp", "ExternalBug",
        "GasException", "DenialOfService", "IntegerUO", "UnusedReturn",
        "MishandledException", "TransactionOrderDependence",
    ]

    # Get predictions for all contracts
    print(f"Running inference on {len(benchmark_contracts)} contracts...")
    all_probs: dict[str, list[float]] = {name: [] for name in class_names}
    all_truth: dict[str, list[int]] = {name: [] for name in class_names}

    for i, c in enumerate(benchmark_contracts):
        if i % 10 == 0:
            print(f"  [{i+1}/{len(benchmark_contracts)}]", end="\r")
        try:
            r = predictor.predict({"source_code": c["source"]})
        except Exception as e:
            logger.warning(f"Inference failed for {c['path']}: {e}")
            continue
        probs = r.get("probabilities", {})
        labels = c.get("labels", [])
        for name in class_names:
            all_probs[name].append(probs.get(name, 0.0))
            all_truth[name].append(1 if name in labels else 0)

    print(f"  [{len(benchmark_contracts)}/{len(benchmark_contracts)}]")
    print()

    # Analyze each class
    results: list[ThresholdResult] = []
    n_flagged = 0
    for name in class_names:
        r = analyze_class(name, all_truth[name], all_probs[name], thresholds)
        results.append(r)
        if r.flags:
            n_flagged += 1

    return {
        "summary": {
            "n_contracts": len(benchmark_contracts),
            "n_classes": len(class_names),
            "n_flagged": n_flagged,
            "thresholds": thresholds,
        },
        "per_class": [r.to_dict() for r in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="threshold_sensitivity",
        description=(
            "Per-class threshold sensitivity analysis. Catches models "
            "that game a single threshold to get a good F1, and classes "
            "that over- or under-predict at all thresholds."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Path to benchmark directory with .sol files (recursive).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write report to this path.")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds, e.g., '0.1,0.3,0.5,0.7,0.9'")
    parser.add_argument("--exit-on-fail", action="store_true",
                        help="Exit 1 if any class has flags.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1
    if not args.benchmark.exists():
        print(f"ERROR: benchmark dir not found: {args.benchmark}")
        return 1

    # Build predictor
    from ml.src.inference.predictor import Predictor
    predictor = Predictor(str(args.checkpoint))

    # Wrap predict to take our internal format
    class _Adapter:
        def __init__(self, p):
            self._p = p
        def predict(self, payload):
            r = self._p.predict_source(payload["source_code"])
            return {"probabilities": r.get("probabilities", {})}
    adapter = _Adapter(predictor)

    # Load benchmark
    contracts = load_benchmark(args.benchmark)
    if not contracts:
        print(f"ERROR: no contracts found in {args.benchmark}")
        return 1
    print(f"Loaded {len(contracts)} contracts from {args.benchmark}")

    # Parse thresholds
    thresholds = DEFAULT_THRESHOLDS
    if args.thresholds:
        try:
            thresholds = [float(t) for t in args.thresholds.split(",")]
        except ValueError:
            print(f"ERROR: invalid thresholds: {args.thresholds}")
            return 1

    results = run_sensitivity(adapter, contracts, thresholds=thresholds)
    summary = results["summary"]

    # Print results
    print("=" * 70)
    print(f"THRESHOLD SENSITIVITY: {summary['n_flagged']}/{summary['n_classes']} classes flagged")
    print("=" * 70)
    print()
    print(f"Per-class F1 across thresholds:")
    for r in results["per_class"]:
        print(f"\n  {r['class_name']} (n_pos={r['n_positives']}/{r['n_samples']}):")
        for t in thresholds:
            f1 = r['f1_at_thresholds'].get(str(t), r['f1_at_thresholds'].get(t, '?'))
            marker = "⚠" if f1 != "?" and abs(max(r['f1_at_thresholds'].values()) - min(r['f1_at_thresholds'].values())) > MAX_F1_SWING else " "
            print(f"    {marker} thresh={t:.2f}  F1={f1:.3f}" if f1 != "?" else f"    thresh={t:.2f}  F1=?")
        if r['flags']:
            print(f"    FLAGS: {r['flags']}")
        print(f"    F1 swing: {r['f1_swing']:.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nReport written to: {args.output}")

    if args.exit_on_fail and summary["n_flagged"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
