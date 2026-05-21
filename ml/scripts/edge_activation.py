"""
edge_activation.py — GATE-3A-0: per-edge-type activation analysis.

For each edge type in the v8 schema, reports:
  - overall_pct    : % of split graphs that have ≥1 edge of that type
  - per_class_pct  : for each vulnerability class, % of class-positive graphs
                     in the split that have ≥1 edge of that type

Then applies the GATE-3A-0 required thresholds and prints a PASS / FAIL
summary.  Non-zero edge counts in the full dataset (PLAN-2I) do not guarantee
that edges concentrate on the *right* contracts — this script closes that gap.

Usage
─────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/edge_activation.py \\
        --cache    ml/data/cached_dataset_v8.pkl \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \\
        --splits-dir ml/data/splits/deduped \\
        --split    train \\
        --out      ml/logs/edge_activation_train.json

    # Run on val split instead:
    PYTHONPATH=. python ml/scripts/edge_activation.py --split val ...

Exit codes
──────────
    0  all GATE-3A-0 required checks PASS
    1  one or more required checks FAIL or script error
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from ml.src.preprocessing.graph_schema import EDGE_TYPES, NUM_EDGE_TYPES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]

# Human-readable edge type names, indexed by type ID.
# NUM_EDGE_TYPES=11 in v8; REVERSE_CONTAINS(7) is added at runtime by the
# dataset and never stored on disk, so it will show zero coverage here.
_EDGE_NAMES: dict[int, str] = {v: k for k, v in EDGE_TYPES.items()}

# ── GATE-3A-0 required checks ─────────────────────────────────────────────────
# Each entry: (description, edge_type_id, class_filter, threshold_pct)
# class_filter=None  → check over all graphs in the split
# class_filter="X"   → check only over graphs where class X label == 1
REQUIRED_CHECKS: list[tuple[str, int, str | None, float]] = [
    ("CALL_ENTRY present in ≥30% of all training graphs",    8, None,          30.0),
    ("RETURN_TO  present in ≥30% of all training graphs",    9, None,          30.0),
    ("DEF_USE    present in ≥50% of all training graphs",   10, None,          50.0),
    ("CALL_ENTRY present in ≥40% of Reentrancy=1 graphs",   8, "Reentrancy",  40.0),
    ("CALL_ENTRY present in ≥40% of ExternalBug=1 graphs",  8, "ExternalBug", 40.0),
    ("DEF_USE    present in ≥60% of IntegerUO=1 graphs",   10, "IntegerUO",   60.0),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_cache(cache_path: Path) -> dict:
    log.info(f"Loading cache: {cache_path} ({cache_path.stat().st_size / 1e9:.2f} GB)")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    log.info(f"Cache loaded: {len(cache):,} entries")
    return cache


def _load_split_stems(
    splits_dir: Path,
    split: str,
    label_csv: Path,
) -> tuple[list[str], pd.DataFrame]:
    """
    Return (stems_in_split, label_df_for_split).

    The split .npy files contain row indices into the label CSV.
    stems_in_split preserves the split order.
    """
    npy_path = splits_dir / f"{split}_indices.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Split file not found: {npy_path}")

    indices = np.load(npy_path)
    df_full = pd.read_csv(label_csv)
    df_split = df_full.iloc[indices].reset_index(drop=True)
    stems = df_split["md5_stem"].tolist()
    log.info(f"Split '{split}': {len(stems):,} samples (from {label_csv.name})")
    return stems, df_split


def _analyse(
    stems: list[str],
    df_split: pd.DataFrame,
    cache: dict,
) -> dict:
    """
    For each edge type, compute overall and per-class activation rates.

    Returns a nested dict:
        {
          edge_type_id: {
            "name": str,
            "overall": {"graphs_with_type": int, "total_graphs": int, "pct": float},
            "per_class": {
              class_name: {"graphs_with_type": int, "total_graphs": int, "pct": float},
              ...
            }
          },
          ...
        }
    """
    # Build stem→label-row lookup from split df
    stem_to_labels: dict[str, dict[str, int]] = {}
    for _, row in df_split.iterrows():
        stem_to_labels[row["md5_stem"]] = {c: int(row[c]) for c in CLASS_NAMES}

    # Counters: edge_type → {overall, per-class}
    total_seen       = 0          # graphs that exist in cache and split
    type_overall     = defaultdict(int)   # edge_type → count of graphs with ≥1 edge
    type_per_class   = {           # edge_type → class_name → (with_type, total_positive)
        t: defaultdict(lambda: [0, 0]) for t in range(NUM_EDGE_TYPES)
    }

    # __schema_version__ is a metadata key, not a stem — skip it gracefully.
    missing_from_cache = 0
    for stem in stems:
        if stem not in cache:
            missing_from_cache += 1
            continue

        graph, _token = cache[stem]
        labels = stem_to_labels.get(stem, {})
        if not labels:
            continue  # stem not in df (shouldn't happen but guard anyway)

        total_seen += 1
        edge_attr = graph.edge_attr  # 1-D int64 tensor

        # Which edge types appear in this graph?
        types_present: set[int] = set(edge_attr.unique().tolist())

        for etype in range(NUM_EDGE_TYPES):
            has_type = int(etype in types_present)
            type_overall[etype] += has_type
            for cls in CLASS_NAMES:
                is_positive = labels.get(cls, 0)
                counts = type_per_class[etype][cls]
                counts[1] += is_positive         # total positives for this class
                counts[0] += has_type * is_positive  # positives that also have this edge type

    if missing_from_cache > 0:
        log.warning(f"{missing_from_cache:,} stems not found in cache (excluded from analysis)")

    log.info(f"Analysed {total_seen:,} graphs from split")

    # Build result dict
    result: dict = {}
    for etype in range(NUM_EDGE_TYPES):
        name = _EDGE_NAMES.get(etype, f"UNKNOWN_{etype}")
        overall_count = type_overall[etype]
        overall_pct   = 100.0 * overall_count / total_seen if total_seen else 0.0

        per_class: dict = {}
        for cls in CLASS_NAMES:
            with_type, total_pos = type_per_class[etype][cls]
            pct = 100.0 * with_type / total_pos if total_pos > 0 else 0.0
            per_class[cls] = {
                "graphs_with_type": with_type,
                "total_positive_graphs": total_pos,
                "pct": round(pct, 2),
            }

        result[etype] = {
            "name": name,
            "overall": {
                "graphs_with_type": overall_count,
                "total_graphs": total_seen,
                "pct": round(overall_pct, 2),
            },
            "per_class": per_class,
        }

    return result


def _run_checks(stats: dict) -> list[dict]:
    """
    Apply GATE-3A-0 required checks.  Returns list of result dicts with
    fields: description, edge_type, class_filter, threshold, actual, passed.
    """
    results = []
    for desc, etype, cls_filter, threshold in REQUIRED_CHECKS:
        entry = stats.get(etype, {})
        if cls_filter is None:
            actual = entry.get("overall", {}).get("pct", 0.0)
        else:
            actual = entry.get("per_class", {}).get(cls_filter, {}).get("pct", 0.0)

        results.append({
            "description": desc,
            "edge_type": etype,
            "edge_name": _EDGE_NAMES.get(etype, "?"),
            "class_filter": cls_filter,
            "threshold_pct": threshold,
            "actual_pct": round(actual, 2),
            "passed": actual >= threshold,
        })
    return results


def _print_summary(stats: dict, checks: list[dict], split: str) -> None:
    # Overall activation table
    print(f"\n{'─'*72}")
    print(f"  Edge activation — split='{split}'")
    print(f"{'─'*72}")
    print(f"  {'Type':>4}  {'Name':<20}  {'Overall %':>10}  {'With type':>10}  {'Total':>8}")
    print(f"  {'----':>4}  {'----':<20}  {'---------':>10}  {'---------':>10}  {'-----':>8}")
    for etype, info in stats.items():
        ov = info["overall"]
        print(
            f"  {etype:>4}  {info['name']:<20}  "
            f"{ov['pct']:>9.1f}%  {ov['graphs_with_type']:>10,}  {ov['total_graphs']:>8,}"
        )

    # Per-class breakdown for ICFG + DFG types
    focus_types = [8, 9, 10]
    focus_classes = ["Reentrancy", "ExternalBug", "IntegerUO", "GasException", "CallToUnknown"]
    print(f"\n{'─'*72}")
    print(f"  Per-class breakdown (edge types: CALL_ENTRY=8, RETURN_TO=9, DEF_USE=10)")
    print(f"{'─'*72}")
    hdr = f"  {'Class':<28}" + "".join(f"  ET{t:>2}%" for t in focus_types)
    print(hdr)
    print(f"  {'-'*28}" + "  -----" * len(focus_types))
    for cls in focus_classes:
        row = f"  {cls:<28}"
        for etype in focus_types:
            pct = stats[etype]["per_class"].get(cls, {}).get("pct", 0.0)
            row += f"  {pct:>5.1f}"
        print(row)

    # Gate check results
    print(f"\n{'─'*72}")
    print(f"  GATE-3A-0 required checks")
    print(f"{'─'*72}")
    all_pass = True
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_pass = False
        print(f"  [{status}]  {c['actual_pct']:>5.1f}% >= {c['threshold_pct']:.0f}%  {c['description']}")

    print(f"{'─'*72}")
    if all_pass:
        print("  GATE-3A-0: ALL CHECKS PASSED — proceed to GATE-3A-1")
    else:
        failed = sum(1 for c in checks if not c["passed"])
        print(f"  GATE-3A-0: {failed} CHECK(S) FAILED — do not launch PLAN-3A")
        print("  See docs/ACTIVE_PLAN.md § GATE-3A-0 for failure actions.")
    print(f"{'─'*72}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GATE-3A-0: per-edge-type activation analysis on a split."
    )
    p.add_argument(
        "--cache",
        default="ml/data/cached_dataset_v8.pkl",
        help="Path to pre-built v8 cache pickle (default: ml/data/cached_dataset_v8.pkl)",
    )
    p.add_argument(
        "--label-csv",
        default="ml/data/processed/multilabel_index_cleaned.csv",
        help="Multi-label index CSV (default: multilabel_index_cleaned.csv)",
    )
    p.add_argument(
        "--splits-dir",
        default="ml/data/splits/deduped",
        help="Directory containing train/val/test_indices.npy (default: ml/data/splits/deduped)",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to analyse (default: train)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Path to write JSON report (default: ml/logs/edge_activation_<split>.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cache_path  = Path(args.cache)
    label_csv   = Path(args.label_csv)
    splits_dir  = Path(args.splits_dir)
    out_path    = Path(args.out) if args.out else Path(f"ml/logs/edge_activation_{args.split}.json")

    # ── sanity checks ──────────────────────────────────────────────────────────
    for p, name in [(cache_path, "cache"), (label_csv, "label_csv"), (splits_dir, "splits_dir")]:
        if not p.exists():
            log.error(f"{name} not found: {p}")
            return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────────
    cache = _load_cache(cache_path)

    # Schema version is stored as cache["__schema_version__"] by create_cache.py.
    # (The token dict carries a stale 'v7' tag — BUG-M6 — and is not authoritative.)
    schema_ver = cache.get("__schema_version__", "<missing>")
    if schema_ver != "v8":
        log.warning(f"Cache schema version is '{schema_ver}' (expected 'v8') — results may be unreliable")
    else:
        log.info(f"Cache schema version: {schema_ver}  OK")

    stems, df_split = _load_split_stems(splits_dir, args.split, label_csv)

    # ── analyse ────────────────────────────────────────────────────────────────
    log.info("Computing per-edge-type activation rates …")
    stats = _analyse(stems, df_split, cache)

    # ── gate checks ────────────────────────────────────────────────────────────
    checks = _run_checks(stats)
    _print_summary(stats, checks, args.split)

    # ── save JSON ──────────────────────────────────────────────────────────────
    report = {
        "split": args.split,
        "cache": str(cache_path),
        "label_csv": str(label_csv),
        "splits_dir": str(splits_dir),
        "schema_version": schema_ver,
        "edge_type_stats": {str(k): v for k, v in stats.items()},
        "gate_checks": checks,
        "gate_passed": all(c["passed"] for c in checks),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {out_path}")

    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
