"""
label_quality.py — Pre-launch label quality audit (F.1.0).

WHY THIS EXISTS

Run 12 had ExternalBug labeled positive on 75% of training contracts
(16,638 / 22,493). The DIVE crosswalk maps the "Access Control" folder
to ExternalBug, which is too broad — it includes benign owner patterns.
The model then learned the wrong feature and gave ExternalBug=0.85 on a
safe_storage-style contract.

A pre-launch label quality check would have caught this BEFORE training
started.

WHAT IT DOES

Checks the training labels for:
1. Per-class positive rate (FAIL if any class > 50% or < 1%)
2. Per-source positive rate per class (FAIL if a single source is > 80%)
3. Class co-occurrence (FLAG if any pair > 0.60)
4. Class definition sanity (WARN if a class has 0 positives in val)

USAGE

    # Default: v3 export
    python ml/testing_specs/label_quality.py --exit-on-fail

    # Explicit
    python ml/testing_specs/label_quality.py \\
        --labels data_module/data/exports/<my_export>/labels.parquet \\
        --output ml/checkpoints/Run_label_quality.json \\
        --exit-on-fail
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Same class order as synthetic_probes.py
# NOTE: This is the SENTINEL class order, but the parquet's class_0..class_9
# columns use a DIFFERENT order. The mapping is in the export's manifest.json
# under `label_class_columns`. ALWAYS load class order from the manifest, never
# hardcode it. The defaults below are a fallback only.
DEFAULT_CLASSES = [
    "Reentrancy",
    "CallToUnknown",
    "Timestamp",
    "ExternalBug",
    "GasException",
    "DenialOfService",
    "IntegerUO",
    "UnusedReturn",
    "MishandledException",
    "TransactionOrderDependence",
]


def _load_class_order_from_manifest(labels_path: Path) -> list[str] | None:
    """Try to load the canonical class order from the export's manifest.json.

    The manifest.json is in the same directory as labels.parquet and has a
    `label_class_columns` field with the canonical class order.
    """
    manifest_path = labels_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        cols = data.get("label_class_columns")
        if cols and isinstance(cols, list) and len(cols) == 10:
            return cols
    except Exception:
        pass
    return None


@dataclass
class QualityResult:
    """Result of one label quality check."""

    name: str
    status: str  # "PASS" | "FAIL" | "WARN"
    message: str
    value: Any = None

    def to_dict(self) -> dict:
        return asdict(self)


def _per_class_positive_rate(labels: list[dict], class_names: list[str]) -> dict[str, dict]:
    """Compute per-class positive count and rate."""
    n = len(labels)
    out: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        col = f"class_{i}"
        pos = sum(1 for r in labels if r.get(col) == 1)
        out[name] = {
            "positive": pos,
            "rate": pos / n if n else 0,
        }
    return out


def _per_source_per_class(labels: list[dict], class_names: list[str]) -> dict:
    """For each (source, class), compute positive count and rate.

    Helps detect if a single source dominates a class (e.g., DIVE has
    99% of ExternalBug positives).
    """
    by_source_class: dict[str, dict[str, int]] = {}
    for r in labels:
        src = r.get("source", "unknown")
        if src not in by_source_class:
            by_source_class[src] = {f"class_{i}": 0 for i in range(len(class_names))}
        for i in range(len(class_names)):
            col = f"class_{i}"
            if r.get(col) == 1:
                by_source_class[src][col] += 1

    out: dict[str, dict[str, dict]] = {}
    for src, counts in by_source_class.items():
        out[src] = {}
        for i, name in enumerate(class_names):
            col = f"class_{i}"
            total_pos = sum(
                by_source_class[s].get(col, 0) for s in by_source_class
            )
            out[src][name] = {
                "positive": counts.get(col, 0),
                "share": counts.get(col, 0) / total_pos if total_pos else 0,
            }
    return out


def _co_occurrence(labels: list[dict], class_names: list[str]) -> dict[str, float]:
    """Compute co-occurrence rate between every pair of classes.

    For two classes A and B: how often are they both positive?
    Jaccard-style: |A ∩ B| / |A ∪ B|

    Returns dict with string keys "{a}+{b}" because JSON can't serialize
    tuples as dict keys.
    """
    pos: dict[int, set] = {i: set() for i in range(len(class_names))}
    for j, r in enumerate(labels):
        for i in range(len(class_names)):
            if r.get(f"class_{i}") == 1:
                pos[i].add(j)

    coocc: dict[str, float] = {}
    for i in range(len(class_names)):
        for k in range(i + 1, len(class_names)):
            union = pos[i] | pos[k]
            if not union:
                continue
            inter = pos[i] & pos[k]
            coocc[f"{class_names[i]}+{class_names[k]}"] = len(inter) / len(union)
    return coocc


def _val_test_positives(labels: list[dict], class_names: list[str]) -> dict[str, dict]:
    """Count positives per split (train/val/test) per class.

    Flags classes with 0 positives in val (model can't learn to detect
    them on val data).
    """
    out: dict[str, dict[str, int]] = {name: {"train": 0, "val": 0, "test": 0} for name in class_names}
    for r in labels:
        split = r.get("split", "unknown")
        if split not in ("train", "val", "test"):
            continue
        for i, name in enumerate(class_names):
            if r.get(f"class_{i}") == 1:
                out[name][split] += 1
    return out


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

# Thresholds
MAX_POSITIVE_RATE = 0.50       # FAIL if any class > 50% positive
MIN_POSITIVE_RATE = 0.01       # FAIL if any class < 1% positive
MAX_SOURCE_DOMINANCE = 0.80    # FAIL if a single source > 80% of a class's positives
MAX_COOCCURRENCE = 0.60        # WARN if any pair > 0.60 co-occurrence


def check_per_class_rate(
    per_class: dict[str, dict],
) -> list[QualityResult]:
    """Check per-class positive rate is in [1%, 50%]."""
    results: list[QualityResult] = []
    for name, info in per_class.items():
        rate = info["rate"]
        pos = info["positive"]
        if rate > MAX_POSITIVE_RATE:
            results.append(QualityResult(
                name=f"per_class_rate:{name}",
                status="FAIL",
                message=(
                    f"{name} positive rate {rate:.3f} > {MAX_POSITIVE_RATE:.2f} threshold. "
                    f"{pos} contracts labeled positive — likely over-labeled."
                ),
                value={"rate": rate, "positive": pos},
            ))
        elif rate < MIN_POSITIVE_RATE:
            results.append(QualityResult(
                name=f"per_class_rate:{name}",
                status="FAIL",
                message=(
                    f"{name} positive rate {rate:.3f} < {MIN_POSITIVE_RATE:.2f} threshold. "
                    f"Only {pos} contracts labeled — model cannot learn this class."
                ),
                value={"rate": rate, "positive": pos},
            ))
        else:
            results.append(QualityResult(
                name=f"per_class_rate:{name}",
                status="PASS",
                message=f"{name} rate {rate:.3f} OK (in [{MIN_POSITIVE_RATE:.2f}, {MAX_POSITIVE_RATE:.2f}])",
                value={"rate": rate, "positive": pos},
            ))
    return results


def check_per_source_dominance(
    per_source: dict[str, dict[str, dict]],
    per_class: dict[str, dict],
) -> list[QualityResult]:
    """Check that no single source dominates a class's positives (> 80%)."""
    results: list[QualityResult] = []
    class_names = list(per_class.keys())
    for src, class_data in per_source.items():
        for name in class_names:
            data = class_data.get(name, {})
            share = data.get("share", 0)
            pos = data.get("positive", 0)
            if pos == 0:
                continue  # Source has no positives for this class
            if share > MAX_SOURCE_DOMINANCE:
                results.append(QualityResult(
                    name=f"source_dominance:{src}:{name}",
                    status="FAIL",
                    message=(
                        f"Source '{src}' has {share:.1%} of {name} positives ({pos} contracts). "
                        f"Threshold: {MAX_SOURCE_DOMINANCE:.0%}. Class may be over-fit to source labels."
                    ),
                    value={"source": src, "share": share, "positive": pos},
                ))
            else:
                results.append(QualityResult(
                    name=f"source_dominance:{src}:{name}",
                    status="PASS",
                    message=f"Source '{src}' share of {name} is {share:.1%} (≤ {MAX_SOURCE_DOMINANCE:.0%})",
                    value={"source": src, "share": share, "positive": pos},
                ))
    return results


def check_co_occurrence(
    coocc: dict[str, float],
) -> list[QualityResult]:
    """Flag suspicious co-occurrence patterns (Jaccard > 0.60)."""
    results: list[QualityResult] = []
    for key, score in coocc.items():
        # key is "A+B" string
        a, b = key.split("+", 1)
        if score > MAX_COOCCURRENCE:
            results.append(QualityResult(
                name=f"co_occurrence:{a}+{b}",
                status="WARN",
                message=(
                    f"{a} + {b} co-occurrence is {score:.2f} (> {MAX_COOCCURRENCE:.2f}). "
                    f"These classes may be conflated in labels."
                ),
                value={"a": a, "b": b, "jaccard": score},
            ))
    return results


def check_split_coverage(
    val_test: dict[str, dict[str, int]],
) -> list[QualityResult]:
    """Warn if a class has 0 positives in val (model can't validate it)."""
    results: list[QualityResult] = []
    for name, splits in val_test.items():
        if splits["val"] == 0:
            results.append(QualityResult(
                name=f"val_coverage:{name}",
                status="WARN",
                message=(
                    f"{name} has 0 positives in val split. "
                    f"Model will not be evaluated on this class during training."
                ),
                value=splits,
            ))
        else:
            results.append(QualityResult(
                name=f"val_coverage:{name}",
                status="PASS",
                message=f"{name} has {splits['val']} positives in val (validated)",
                value=splits,
            ))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_quality_checks(
    labels: list[dict],
    class_names: list[str] = None,
) -> dict[str, Any]:
    """Run all label quality checks.

    Args:
        labels: List of label dicts (from labels.parquet).
        class_names: List of class names. Default: SENTINEL 10 classes.

    Returns:
        Dict with checks, summary, and metadata.
    """
    class_names = class_names or DEFAULT_CLASSES
    n = len(labels)

    per_class = _per_class_positive_rate(labels, class_names)
    per_source = _per_source_per_class(labels, class_names)
    coocc = _co_occurrence(labels, class_names)  # already string-keyed
    val_test = _val_test_positives(labels, class_names)

    checks: list[QualityResult] = []
    checks.extend(check_per_class_rate(per_class))
    checks.extend(check_per_source_dominance(per_source, per_class))
    checks.extend(check_co_occurrence(coocc))
    checks.extend(check_split_coverage(val_test))

    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_warn = sum(1 for c in checks if c.status == "WARN")
    n_pass = sum(1 for c in checks if c.status == "PASS")

    return {
        "summary": {
            "total_contracts": n,
            "total_checks": len(checks),
            "passed": n_pass,
            "warned": n_warn,
            "failed": n_fail,
            "all_passed": n_fail == 0,
        },
        "per_class": per_class,
        "per_source": per_source,
        "co_occurrence_top10": dict(
            sorted(coocc.items(), key=lambda x: -x[1])[:10]
        ),
        "val_test_coverage": val_test,
        "checks": [c.to_dict() for c in checks],
    }


def _load_parquet(path: Path) -> list[dict]:
    """Load labels from a parquet file."""
    import pandas as pd
    df = pd.read_parquet(path)
    return df.to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="label_quality",
        description=(
            "Audit training labels BEFORE launching a run. Catches over-"
            "labeled classes (e.g., ExternalBug at 75% positive in Run 12). "
            "Exits non-zero if any check FAILS."
        ),
    )
    parser.add_argument(
        "--labels", type=Path, default=None,
        help="Path to labels.parquet. Default: v3 export.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write results JSON to this path.",
    )
    parser.add_argument(
        "--exit-on-fail", action="store_true",
        help="Exit with code 1 if any check FAILS. Default: always 0.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every check (not just FAILs).",
    )
    args = parser.parse_args()

    if args.labels:
        labels_path = args.labels
    else:
        # Default: v3 export
        candidates = [
            Path("data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/labels.parquet"),
            Path("/home/motafeq/projects/sentinel/data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/labels.parquet"),
        ]
        labels_path = next((c for c in candidates if c.exists()), None)
        if not labels_path:
            print("ERROR: No labels.parquet found. Pass --labels.")
            return 1

    # Try to load the canonical class order from the manifest
    class_names = _load_class_order_from_manifest(labels_path) or DEFAULT_CLASSES
    if class_names == DEFAULT_CLASSES:
        print("WARNING: Could not load class order from manifest. Using fallback DEFAULT_CLASSES.")
        print("WARNING: The fallback order may NOT match the parquet's class_0..class_9 columns.")
        print("WARNING: ALWAYS use a parquet that has a manifest.json with label_class_columns.")
    else:
        print(f"Loaded {len(class_names)} class names from manifest: {class_names}")

    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        return 1

    print(f"Loading labels from: {labels_path}")
    labels = _load_parquet(labels_path)
    print(f"Loaded {len(labels)} contracts")

    results = run_quality_checks(labels, class_names=class_names)
    summary = results["summary"]

    # Print summary
    print()
    print("=" * 70)
    print(f"LABEL QUALITY: {summary['passed']} PASS, {summary['warned']} WARN, {summary['failed']} FAIL")
    print("=" * 70)
    print()
    print("Per-class positive rate:")
    for name, info in results["per_class"].items():
        rate = info["rate"]
        marker = "✓" if rate <= MAX_POSITIVE_RATE and rate >= MIN_POSITIVE_RATE else "✗"
        print(f"  {marker} {name:30} {info['positive']:>5} / {summary['total_contracts']:>5}  ({rate:>5.1%})")
    print()
    print("Per-source dominance (FAIL if any source > 80% of a class's positives):")
    for src, class_data in results["per_source"].items():
        for name, info in class_data.items():
            if info["positive"] > 0 and info["share"] > 0.3:
                marker = "⚠" if info["share"] > MAX_SOURCE_DOMINANCE else " "
                print(f"  {marker} {src:15} {name:25} {info['share']:>5.1%} ({info['positive']} contracts)")
    print()
    print("Co-occurrence (top 10):")
    for key, score in list(results["co_occurrence_top10"].items())[:5]:
        a, b = key.split("+", 1)
        marker = "⚠" if score > MAX_COOCCURRENCE else " "
        print(f"  {marker} {a:25} + {b:25} {score:.2f}")
    print()

    # Print individual checks
    print("=" * 70)
    print("INDIVIDUAL CHECKS:")
    print("=" * 70)
    for c in results["checks"]:
        if c["status"] == "FAIL" or args.verbose:
            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[c["status"]]
            print(f"  [{icon}] {c['name']}: {c['message']}")
    if not args.verbose:
        n_fail = summary["failed"]
        n_warn = summary["warned"]
        if n_fail == 0 and n_warn == 0:
            print("  (all checks PASS — no warnings)")
        elif n_fail == 0:
            print(f"  (no FAILs, {n_warn} warnings — use --verbose to see)")

    # Write JSON
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults written to: {args.output}")

    if args.exit_on_fail and not summary["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
