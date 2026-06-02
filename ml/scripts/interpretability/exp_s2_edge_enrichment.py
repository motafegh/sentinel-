"""
exp_s2_edge_enrichment.py — Layer 1, P0: Edge Presence Enrichment Ratio Per Vulnerability Class

PURPOSE
───────
Extends the logic from ml/scripts/edge_activation.py which computes per-class
edge presence rates but does NOT compute the positive/negative enrichment ratio.
This script adds that ratio to reveal which edge types are disproportionately
concentrated in contracts that carry a specific vulnerability label.

For each edge type 0-10, for each vulnerability class:
  class_pct    = % of class-positive graphs that have ≥1 edge of this type
  baseline_pct = % of ALL graphs in split that have ≥1 edge of this type
  enrichment_ratio = class_pct / baseline_pct  (0 if baseline_pct == 0)

LAYER / PRIORITY
─────────────────
Layer 1, Priority 0 — must pass before graph-structure assumptions are trusted.

PASS CRITERIA
─────────────
Named enrichment checks (ratio >= threshold):
  CONTROL_FLOW(6) enriched for Reentrancy           >= 1.3
  CONTROL_FLOW(6) enriched for IntegerUO            >= 1.3
  CALL_ENTRY(8)   enriched for Reentrancy           >= 1.3
  CALL_ENTRY(8)   enriched for ExternalBug          >= 1.3
  RETURN_TO(9)    enriched for Reentrancy           >= 1.3
  DEF_USE(10)     enriched for IntegerUO            >= 1.3
  DEF_USE(10)     enriched for UnusedReturn         >= 1.3
  READS(1)        enriched for TransactionOrderDep  >= 1.2

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_s2_edge_enrichment.py \\
        --cache ml/data/cached_dataset_v9.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --split train \\
        --out ml/logs/interpretability/s2_edge_enrichment.json

OUTPUT
──────
1. Full enrichment matrix (11 edge types × 10 classes) printed as table
2. PASS/FAIL for each named enrichment check
3. JSON file with all stats at --out path
4. PNG heatmap of the enrichment matrix alongside the JSON

EXIT CODES
──────────
    0  all named checks pass
    1  one or more named checks fail or script error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.src.preprocessing.graph_schema import EDGE_TYPES, NUM_EDGE_TYPES
from ml.scripts.interpretability.utils import (
    add_common_args,
    load_val_split,
    CLASS_NAMES,
    plot_class_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Human-readable edge names ─────────────────────────────────────────────────

_EDGE_NAMES: dict[int, str] = {v: k for k, v in EDGE_TYPES.items()}

# ── Named enrichment checks ───────────────────────────────────────────────────
# (description, edge_type_id, class_name, min_ratio)
ENRICHMENT_CHECKS = [
    ("CONTROL_FLOW(6) enriched for Reentrancy",   6, "Reentrancy",                1.3),
    ("CONTROL_FLOW(6) enriched for IntegerUO",     6, "IntegerUO",                 1.3),
    ("CALL_ENTRY(8) enriched for Reentrancy",      8, "Reentrancy",                1.3),
    ("CALL_ENTRY(8) enriched for ExternalBug",     8, "ExternalBug",               1.3),
    ("RETURN_TO(9) enriched for Reentrancy",       9, "Reentrancy",                1.3),
    ("DEF_USE(10) enriched for IntegerUO",        10, "IntegerUO",                 1.3),
    ("DEF_USE(10) enriched for UnusedReturn",     10, "UnusedReturn",              1.3),
    ("READS(1) enriched for TransactionOrderDep",  1, "TransactionOrderDependence", 1.2),
]


# ── Core analysis ─────────────────────────────────────────────────────────────

def _compute_enrichment(
    stems: list[str],
    df_split,
    cache: dict,
) -> dict:
    """
    Compute per-edge-type, per-class enrichment ratios.

    Returns:
        {
          edge_type_id: {
            "name": str,
            "baseline_pct": float,
            "class_stats": {
              class_name: {
                "class_pct": float,
                "baseline_pct": float,
                "enrichment_ratio": float,
                "class_positive_graphs": int,
                "class_positive_with_edge": int,
              }, ...
            }
          }, ...
        }
    """
    stem_to_labels: dict[str, dict[str, int]] = {}
    for _, row in df_split.iterrows():
        stem_to_labels[row["md5_stem"]] = {c: int(row[c]) for c in CLASS_NAMES}

    total_graphs = 0
    # edge_type -> count of graphs that have >=1 of this edge type (all graphs)
    overall_has: dict[int, int] = defaultdict(int)
    # edge_type -> class_name -> [graphs_with_edge_and_positive, total_positive]
    class_counts: dict[int, dict[str, list[int]]] = {
        t: {c: [0, 0] for c in CLASS_NAMES} for t in range(NUM_EDGE_TYPES)
    }

    missing = 0
    for stem in stems:
        if stem not in cache:
            missing += 1
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 2:
            missing += 1
            continue

        graph, _ = entry
        labels = stem_to_labels.get(stem)
        if labels is None:
            continue

        total_graphs += 1
        edge_attr = graph.edge_attr
        present: set[int] = set(edge_attr.unique().tolist())

        for etype in range(NUM_EDGE_TYPES):
            has = int(etype in present)
            overall_has[etype] += has
            for cls in CLASS_NAMES:
                is_pos = labels.get(cls, 0)
                class_counts[etype][cls][1] += is_pos
                class_counts[etype][cls][0] += has * is_pos

    if missing:
        log.warning(f"{missing:,} stems not found in cache — excluded")
    log.info(f"Analysed {total_graphs:,} graphs")

    result: dict = {}
    for etype in range(NUM_EDGE_TYPES):
        baseline_pct = (
            100.0 * overall_has[etype] / total_graphs if total_graphs else 0.0
        )
        class_stats: dict[str, dict] = {}
        for cls in CLASS_NAMES:
            with_edge, total_pos = class_counts[etype][cls]
            class_pct = (
                100.0 * with_edge / total_pos if total_pos > 0 else 0.0
            )
            enrichment = (
                class_pct / baseline_pct if baseline_pct > 0.0 else 0.0
            )
            class_stats[cls] = {
                "class_pct": round(class_pct, 3),
                "baseline_pct": round(baseline_pct, 3),
                "enrichment_ratio": round(enrichment, 4),
                "class_positive_graphs": total_pos,
                "class_positive_with_edge": with_edge,
            }
        result[etype] = {
            "name": _EDGE_NAMES.get(etype, f"UNKNOWN_{etype}"),
            "baseline_pct": round(baseline_pct, 3),
            "overall_graphs_with_edge": overall_has[etype],
            "total_graphs": total_graphs,
            "class_stats": class_stats,
        }

    return result


def _run_enrichment_checks(stats: dict) -> list[dict]:
    results = []
    for desc, etype, cls, min_ratio in ENRICHMENT_CHECKS:
        actual = stats.get(etype, {}).get("class_stats", {}).get(cls, {}).get(
            "enrichment_ratio", 0.0
        )
        results.append({
            "description": desc,
            "edge_type": etype,
            "edge_name": _EDGE_NAMES.get(etype, "?"),
            "class": cls,
            "min_ratio": min_ratio,
            "actual_ratio": round(actual, 4),
            "passed": actual >= min_ratio,
        })
    return results


def _print_matrix(stats: dict, split: str) -> None:
    print(f"\n{'─'*100}")
    print(f"  Edge enrichment matrix — split='{split}'  (ratio = class_pct / baseline_pct)")
    print(f"{'─'*100}")
    # Header: class names abbreviated
    abbrev = [c[:9] for c in CLASS_NAMES]
    hdr = f"  {'ET':>3}  {'Name':<20}  {'Base%':>6}" + "".join(f"  {a:>9}" for a in abbrev)
    print(hdr)
    print(f"  {'---':>3}  {'----':<20}  {'-----':>6}" + "  ---------" * len(CLASS_NAMES))
    for etype in range(NUM_EDGE_TYPES):
        info = stats[etype]
        row = f"  {etype:>3}  {info['name']:<20}  {info['baseline_pct']:>5.1f}%"
        for cls in CLASS_NAMES:
            r = info["class_stats"][cls]["enrichment_ratio"]
            row += f"  {r:>9.3f}"
        print(row)
    print(f"{'─'*100}\n")


def _print_checks(checks: list[dict]) -> bool:
    all_pass = True
    print(f"\n{'─'*80}")
    print("  Named enrichment checks")
    print(f"{'─'*80}")
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_pass = False
        print(
            f"  [{status}]  ratio={c['actual_ratio']:.4f} >= {c['min_ratio']:.1f}"
            f"  {c['description']}"
        )
    print(f"{'─'*80}")
    if all_pass:
        print("  ALL enrichment checks PASSED")
    else:
        n_fail = sum(1 for c in checks if not c["passed"])
        print(f"  {n_fail} check(s) FAILED")
    print(f"{'─'*80}\n")
    return all_pass


def _build_matrix(stats: dict) -> np.ndarray:
    """Build 11×10 enrichment ratio matrix (rows=edge types, cols=classes)."""
    mat = np.zeros((NUM_EDGE_TYPES, len(CLASS_NAMES)), dtype=np.float32)
    for etype in range(NUM_EDGE_TYPES):
        for j, cls in enumerate(CLASS_NAMES):
            mat[etype, j] = stats[etype]["class_stats"][cls]["enrichment_ratio"]
    return mat


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="exp_s2: Edge presence enrichment ratio per vulnerability class"
    )
    add_common_args(p, require_checkpoint=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cache_path  = Path(args.cache)
    label_csv   = Path(args.label_csv)
    splits_dir  = Path(args.splits_dir)
    out_path    = (
        Path(args.out) if args.out
        else Path(f"ml/logs/interpretability/s2_edge_enrichment_{args.split}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path.with_suffix(".png")

    # ── load ──────────────────────────────────────────────────────────────────
    try:
        stems, df_split, cache = load_val_split(
            cache_path, label_csv, splits_dir, split=args.split
        )
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    # ── analyse ───────────────────────────────────────────────────────────────
    log.info("Computing edge enrichment ratios …")
    stats = _compute_enrichment(stems, df_split, cache)

    # ── print + check ─────────────────────────────────────────────────────────
    _print_matrix(stats, args.split)
    checks = _run_enrichment_checks(stats)
    all_pass = _print_checks(checks)

    # ── heatmap ───────────────────────────────────────────────────────────────
    matrix = _build_matrix(stats)
    row_labels = [
        f"{t}: {_EDGE_NAMES.get(t, 'UNKNOWN')}" for t in range(NUM_EDGE_TYPES)
    ]
    plot_class_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=CLASS_NAMES,
        title=f"Edge Enrichment Ratio — split={args.split}",
        output_path=png_path,
        fmt=".2f",
        cmap="RdYlGn",
    )

    # ── JSON ──────────────────────────────────────────────────────────────────
    report = {
        "split": args.split,
        "cache": str(cache_path),
        "label_csv": str(label_csv),
        "splits_dir": str(splits_dir),
        "enrichment_checks": checks,
        "all_checks_passed": all_pass,
        "edge_enrichment": {
            str(k): v for k, v in stats.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON saved: {out_path}")
    log.info(f"PNG saved:  {png_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
