"""
exp_a1_pooling_audit.py — Phase 3 Pooling Node-Type Audit

PURPOSE
───────
Layer 3, P0 audit.  Verifies that the SentinelModel's function-level mean
pooling (which restricts global mean pool to FUNCTION/MODIFIER/FALLBACK/
RECEIVE/CONSTRUCTOR node types) has adequate coverage across the val split.
If a graph has no FUNCTION-like nodes the model falls back to all-node pooling,
which is an architectural edge case worth quantifying.

This script does NOT require a checkpoint — it operates on graphs from the
cached dataset alone.

LAYER
─────
Layer 3 — static graph audit (no forward pass required).

PASS CRITERIA
─────────────
• ≥95% of graphs have ≥1 FUNCTION-like node (types 1, 2, 4, 5, 6).
• <5% of graphs trigger the all-node fallback path.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_a1_pooling_audit.py \\
        --cache ml/data/cached_dataset_v9.pkl \\
        --out ml/interpretability_results/exp_a1 \\
        --n-contracts 2000

OUTPUT
──────
  <out>/exp_a1_pooling_audit.json   — audit results with pass/fail flags
  stdout                            — summary table + pass/fail
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    load_val_split,
    add_common_args,
    CLASS_NAMES,
    get_node_type_tensor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Node types treated as FUNCTION-like by SentinelModel._FUNC_TYPE_IDS
FUNC_TYPE_IDS = {1, 2, 4, 5, 6}  # FUNCTION, MODIFIER, FALLBACK, RECEIVE, CONSTRUCTOR

FUNC_TYPE_NAMES = {
    1: "FUNCTION",
    2: "MODIFIER",
    4: "FALLBACK",
    5: "RECEIVE",
    6: "CONSTRUCTOR",
}

# Histogram buckets for FUNCTION-like node counts
HIST_BUCKETS = [(0, 0), (1, 1), (2, 5), (6, 10), (11, None)]
HIST_LABELS  = ["0", "1", "2-5", "6-10", ">10"]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit FUNCTION-like node coverage for SentinelModel pooling.",
    )
    # Use add_common_args but checkpoint is optional (not needed)
    add_common_args(p, require_checkpoint=False)
    p.set_defaults(n_contracts=2000)
    return p


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out) if args.out else Path("ml/interpretability_results/exp_a1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────────
    stems, df_split, cache = load_val_split(
        cache_path=Path(args.cache),
        label_csv=Path(args.label_csv),
        splits_dir=Path(args.splits_dir),
        split=args.split,
    )

    rng = np.random.default_rng(args.seed)
    if args.n_contracts and len(stems) > args.n_contracts:
        idx    = rng.choice(len(stems), size=args.n_contracts, replace=False)
        stems  = [stems[i] for i in idx]
        df_split = df_split.iloc[idx].reset_index(drop=True)

    log.info(f"Auditing {len(stems):,} graphs for FUNCTION-like node coverage...")

    # ── Per-graph audit ──────────────────────────────────────────────────────────
    counts_funclike: list[int]   = []
    total_nodes:     list[int]   = []
    fractions:       list[float] = []
    has_any:         list[bool]  = []

    # Type-level breakdown: how often each FUNC type appears
    type_count = {t: 0 for t in FUNC_TYPE_IDS}

    n_skipped = 0

    for stem in stems:
        if stem not in cache:
            n_skipped += 1
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 1:
            n_skipped += 1
            continue

        graph = entry[0]

        try:
            type_ids  = get_node_type_tensor(graph)  # [N]
            N         = int(type_ids.shape[0])
            func_mask = sum(type_ids == t for t in FUNC_TYPE_IDS).bool()
            n_func    = int(func_mask.sum().item())

            counts_funclike.append(n_func)
            total_nodes.append(N)
            fractions.append(n_func / N if N > 0 else 0.0)
            has_any.append(n_func > 0)

            for t in FUNC_TYPE_IDS:
                type_count[t] += int((type_ids == t).sum().item())

        except Exception as exc:
            log.debug(f"Skipping {stem}: {exc}")
            n_skipped += 1
            continue

    n_audited = len(counts_funclike)
    if n_audited == 0:
        log.error("No graphs audited — check cache path.")
        return 1

    counts_arr   = np.array(counts_funclike)
    total_arr    = np.array(total_nodes)
    fraction_arr = np.array(fractions)
    has_any_arr  = np.array(has_any)

    pct_has_any  = 100.0 * has_any_arr.mean()
    pct_fallback = 100.0 - pct_has_any
    mean_frac    = float(fraction_arr.mean())
    mean_count   = float(counts_arr.mean())

    # Histogram
    hist_counts = []
    for lo, hi in HIST_BUCKETS:
        if hi is None:
            hist_counts.append(int((counts_arr > lo - 1).sum()))
        elif lo == hi:
            hist_counts.append(int((counts_arr == lo).sum()))
        else:
            hist_counts.append(int(((counts_arr >= lo) & (counts_arr <= hi)).sum()))

    # ── Print summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP-A1: Pooling Node-Type Audit (FUNCTION-like coverage)")
    print("=" * 70)
    print(f"  Graphs audited:               {n_audited:,}")
    print(f"  Graphs skipped (cache miss):  {n_skipped:,}")
    print(f"  Total node-type tokens seen:  {int(total_arr.sum()):,}")
    print()
    print(f"  ≥1 FUNCTION-like node:   {pct_has_any:6.2f}%  (pass criterion: ≥95%)")
    print(f"  Fallback to all-nodes:   {pct_fallback:6.2f}%  (pass criterion: <5%)")
    print(f"  Mean FUNCTION-like/graph: {mean_count:.2f} nodes")
    print(f"  Mean fraction func-like:  {mean_frac:.4f}  ({mean_frac*100:.2f}%)")
    print()
    print("  Node count histogram (FUNCTION-like nodes per graph):")
    for label, cnt in zip(HIST_LABELS, hist_counts):
        bar = "#" * min(int(cnt / max(n_audited, 1) * 50), 50)
        print(f"    {label:>5}: {cnt:>6} ({100*cnt/n_audited:5.1f}%)  {bar}")
    print()
    print("  FUNCTION-like type breakdown (total occurrences across all graphs):")
    for t, name in FUNC_TYPE_NAMES.items():
        print(f"    type {t} ({name:<12}): {type_count[t]:>8,}")
    print()

    # Pass/fail
    crit1_pass = pct_has_any >= 95.0
    crit2_pass = pct_fallback < 5.0
    overall    = "PASS" if (crit1_pass and crit2_pass) else "FAIL"

    print(f"  CRITERION 1: ≥95% graphs have ≥1 func-like node   → {'PASS' if crit1_pass else 'FAIL'} ({pct_has_any:.2f}%)")
    print(f"  CRITERION 2: <5% graphs trigger fallback           → {'PASS' if crit2_pass else 'FAIL'} ({pct_fallback:.2f}%)")
    print(f"\n  OVERALL: {overall}")
    print("=" * 70 + "\n")

    # ── JSON output ──────────────────────────────────────────────────────────────
    json_data = {
        "experiment":             "exp_a1_pooling_audit",
        "n_audited":              n_audited,
        "n_skipped":              n_skipped,
        "pct_has_funclike_node":  float(pct_has_any),
        "pct_fallback":           float(pct_fallback),
        "mean_funclike_count":    float(mean_count),
        "mean_funclike_fraction": float(mean_frac),
        "histogram": {
            label: int(cnt)
            for label, cnt in zip(HIST_LABELS, hist_counts)
        },
        "type_breakdown": {
            FUNC_TYPE_NAMES[t]: int(type_count[t])
            for t in FUNC_TYPE_IDS
        },
        "criterion1_pass": bool(crit1_pass),
        "criterion2_pass": bool(crit2_pass),
        "overall":         overall,
        "pass_criteria": {
            "pct_has_funclike_min":  95.0,
            "pct_fallback_max":       5.0,
        },
    }
    json_path = out_dir / "exp_a1_pooling_audit.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    log.info(f"JSON saved: {json_path}")

    return 0 if overall == "PASS" else 1


def main() -> None:
    parser = _build_argparser()
    args   = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
