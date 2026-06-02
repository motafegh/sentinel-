"""
exp_s3_feature_distribution.py — Layer 1, P1: Graph Size and Feature Distribution Per Class

PURPOSE
───────
For each vulnerability class, compares the distribution of 7 graph-level metrics
between contracts that carry the label (positives) and those that do not
(negatives).  The goal is to expose statistical shortcuts the model could exploit
instead of learning genuine structural patterns.

METRICS (7 per graph)
─────────────────────
1. total_cfg_nodes        — nodes with type_id in {8,9,10,11,12}
2. cfg_call_count         — nodes with type_id == 8  (CFG_NODE_CALL)
3. function_count         — nodes with type_id == 1  (FUNCTION)
4. def_use_edge_count     — edges with edge_attr == 10
5. ext_call_count_sum     — sum of feature dim 10 (raw external_call_count)
                            across FUNCTION nodes (type_id==1)
6. mean_return_ignored_func — mean of feature dim 7 (return_ignored)
                              across FUNCTION nodes (type_id==1)
                              NOTE: return_ignored is a function-level feature —
                              it is intentionally 0.0 on CFG_NODE_* nodes.
                              Prior runs that computed this over CFG nodes
                              produced an artifactual "dead feature" finding.
7. total_nodes            — total node count (graph size)

LAYER / PRIORITY
─────────────────
Layer 1, Priority 1 — diagnostic only, no pass/fail threshold.
A Cohen's d > 1.5 on any metric is flagged as:
  "STRONG SHORTCUT — model may exploit this without graph structure"

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_s3_feature_distribution.py \\
        --cache ml/data/cached_dataset_v9.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --split train \\
        --n-contracts 5000 \\
        --out ml/logs/interpretability/s3_feature_distribution.json

    # No --checkpoint needed.

OUTPUT
──────
Per-class tables printed to stdout.
JSON with all per-class, per-metric statistics.
Exit 0 always (diagnostic script).

EXIT CODES
──────────
    0  always (diagnostic, no hard pass/fail)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES
from ml.scripts.interpretability.utils import (
    add_common_args,
    load_val_split,
    CLASS_NAMES,
    get_node_type_tensor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── CFG node type IDs ─────────────────────────────────────────────────────────
_CFG_TYPE_IDS: set[int] = {8, 9, 10, 11, 12}
_FUNCTION_TYPE_ID: int = NODE_TYPES.get("FUNCTION", 1)
_CFG_CALL_TYPE_ID: int = NODE_TYPES.get("CFG_NODE_CALL", 8)
_DEF_USE_EDGE_ID:  int = EDGE_TYPES.get("DEF_USE", 10)

SHORTCUT_THRESHOLD: float = 1.5

METRIC_NAMES = [
    "total_cfg_nodes",
    "cfg_call_count",
    "function_count",
    "def_use_edge_count",
    "ext_call_count_sum",
    "mean_return_ignored_func",
    "total_nodes",
]


# ── Per-graph metric extraction ───────────────────────────────────────────────

def _extract_metrics(graph) -> dict[str, float]:
    """Return a dict of the 7 metrics for a single graph."""
    node_types = get_node_type_tensor(graph)  # [N] int
    N = node_types.shape[0]

    cfg_mask      = sum((node_types == t) for t in _CFG_TYPE_IDS).bool()
    func_mask     = (node_types == _FUNCTION_TYPE_ID)
    cfg_call_mask = (node_types == _CFG_CALL_TYPE_ID)

    total_cfg  = int(cfg_mask.sum().item())
    cfg_calls  = int(cfg_call_mask.sum().item())
    func_cnt   = int(func_mask.sum().item())

    # DEF_USE edge count
    edge_attr = graph.edge_attr  # [E] int64
    def_use_cnt = int((edge_attr == _DEF_USE_EDGE_ID).sum().item())

    # External call count sum over FUNCTION nodes (feature dim 10, raw)
    x = graph.x  # [N, 11] float
    if func_mask.any():
        ext_call_sum = float(x[func_mask, 10].sum().item())
    else:
        ext_call_sum = 0.0

    # Mean return_ignored over FUNCTION nodes (feature dim 7).
    # return_ignored is a function-level feature: non-Function nodes always have
    # x[:,7]=0.0 by design (graph_extractor.py). Computing this over CFG nodes
    # produces a misleading "dead feature" artifact — use FUNCTION nodes only.
    if func_mask.any():
        mean_return_ignored = float(x[func_mask, 7].mean().item())
    else:
        mean_return_ignored = 0.0

    return {
        "total_cfg_nodes":          float(total_cfg),
        "cfg_call_count":           float(cfg_calls),
        "function_count":           float(func_cnt),
        "def_use_edge_count":       float(def_use_cnt),
        "ext_call_count_sum":       ext_call_sum,
        "mean_return_ignored_func": mean_return_ignored,
        "total_nodes":              float(N),
    }


# ── Cohen's d ────────────────────────────────────────────────────────────────

def _cohens_d(pos_vals: list[float], neg_vals: list[float]) -> float:
    """Compute Cohen's d = (mean_pos - mean_neg) / pooled_std."""
    if len(pos_vals) < 2 or len(neg_vals) < 2:
        return 0.0
    m_p, m_n = np.mean(pos_vals), np.mean(neg_vals)
    v_p = np.var(pos_vals, ddof=1)
    v_n = np.var(neg_vals, ddof=1)
    n_p, n_n = len(pos_vals), len(neg_vals)
    pooled_var = ((n_p - 1) * v_p + (n_n - 1) * v_n) / (n_p + n_n - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-9
    return float((m_p - m_n) / pooled_std)


# ── Main analysis ─────────────────────────────────────────────────────────────

def _analyse(
    stems: list[str],
    df_split,
    cache: dict,
    n_contracts: int,
    seed: int,
) -> dict:
    """
    For each class, accumulate per-graph metrics into positives/negatives lists.

    Returns nested dict:
        { class_name: { metric_name: {
              "mean_pos": float, "std_pos": float, "n_pos": int,
              "mean_neg": float, "std_neg": float, "n_neg": int,
              "cohens_d": float,
              "shortcut_flag": bool,
        } } }
    """
    # Subsample for speed
    if n_contracts < len(stems):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(stems), size=n_contracts, replace=False)
        stems = [stems[i] for i in sorted(idx)]

    stem_to_labels: dict[str, dict[str, int]] = {}
    for _, row in df_split.iterrows():
        stem_to_labels[row["md5_stem"]] = {c: int(row[c]) for c in CLASS_NAMES}

    # Accumulator: class -> metric -> [pos_list, neg_list]
    accum: dict[str, dict[str, tuple[list, list]]] = {
        cls: {m: ([], []) for m in METRIC_NAMES} for cls in CLASS_NAMES
    }

    missing = skipped = analysed = 0
    for stem in stems:
        if stem not in cache:
            missing += 1
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 2:
            skipped += 1
            continue
        graph, _ = entry
        labels = stem_to_labels.get(stem)
        if labels is None:
            skipped += 1
            continue

        try:
            metrics = _extract_metrics(graph)
        except Exception as exc:
            log.debug(f"Skipping {stem}: {exc}")
            skipped += 1
            continue

        analysed += 1
        for cls in CLASS_NAMES:
            is_pos = labels.get(cls, 0)
            for m, val in metrics.items():
                pos_list, neg_list = accum[cls][m]
                if is_pos:
                    pos_list.append(val)
                else:
                    neg_list.append(val)

    if missing:
        log.warning(f"{missing:,} stems not found in cache")
    log.info(f"Analysed {analysed:,} graphs (skipped {skipped})")

    # Summarise
    result: dict = {}
    for cls in CLASS_NAMES:
        result[cls] = {}
        for m in METRIC_NAMES:
            pos_list, neg_list = accum[cls][m]
            if not pos_list:
                result[cls][m] = {
                    "mean_pos": 0.0, "std_pos": 0.0, "n_pos": 0,
                    "mean_neg": 0.0, "std_neg": 0.0, "n_neg": len(neg_list),
                    "cohens_d": 0.0, "shortcut_flag": False,
                }
                continue
            d = _cohens_d(pos_list, neg_list)
            result[cls][m] = {
                "mean_pos": round(float(np.mean(pos_list)), 4),
                "std_pos":  round(float(np.std(pos_list, ddof=1) if len(pos_list) > 1 else 0.0), 4),
                "n_pos":    len(pos_list),
                "mean_neg": round(float(np.mean(neg_list)) if neg_list else 0.0, 4),
                "std_neg":  round(float(np.std(neg_list, ddof=1) if len(neg_list) > 1 else 0.0), 4),
                "n_neg":    len(neg_list),
                "cohens_d": round(d, 4),
                "shortcut_flag": abs(d) > SHORTCUT_THRESHOLD,
            }
    return result


def _print_results(stats: dict, split: str) -> None:
    print(f"\n{'═'*90}")
    print(f"  Feature distribution analysis — split='{split}'")
    print(f"  Cohen's d > {SHORTCUT_THRESHOLD} flagged as STRONG SHORTCUT")
    print(f"{'═'*90}")
    for cls in CLASS_NAMES:
        has_shortcut = any(
            stats[cls][m]["shortcut_flag"] for m in METRIC_NAMES
        )
        flag_str = "  ** SHORTCUTS DETECTED **" if has_shortcut else ""
        print(f"\n  Class: {cls}{flag_str}")
        print(f"  {'Metric':<22}  {'mean_pos':>9}  {'std_pos':>9}  "
              f"{'n_pos':>6}  {'mean_neg':>9}  {'std_neg':>9}  "
              f"{'n_neg':>6}  {'Cohen_d':>8}  {'Flag'}")
        print(f"  {'------':<22}  {'---------':>9}  {'---------':>9}  "
              f"{'------':>6}  {'---------':>9}  {'---------':>9}  "
              f"{'------':>6}  {'-------':>8}  {'----'}")
        for m in METRIC_NAMES:
            s = stats[cls][m]
            flag = "SHORTCUT" if s["shortcut_flag"] else ""
            print(
                f"  {m:<22}  {s['mean_pos']:>9.3f}  {s['std_pos']:>9.3f}  "
                f"{s['n_pos']:>6}  {s['mean_neg']:>9.3f}  {s['std_neg']:>9.3f}  "
                f"{s['n_neg']:>6}  {s['cohens_d']:>8.3f}  {flag}"
            )
    print(f"\n{'═'*90}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="exp_s3: Graph size and feature distribution per vulnerability class"
    )
    add_common_args(p, require_checkpoint=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cache_path = Path(args.cache)
    label_csv  = Path(args.label_csv)
    splits_dir = Path(args.splits_dir)
    out_path   = (
        Path(args.out) if args.out
        else Path(
            f"ml/logs/interpretability/s3_feature_distribution_{args.split}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    try:
        stems, df_split, cache = load_val_split(
            cache_path, label_csv, splits_dir, split=args.split
        )
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    # ── analyse ───────────────────────────────────────────────────────────────
    log.info(
        f"Computing feature distributions (n_contracts={args.n_contracts}) …"
    )
    stats = _analyse(stems, df_split, cache, args.n_contracts, args.seed)

    # ── print ─────────────────────────────────────────────────────────────────
    _print_results(stats, args.split)

    # ── summary shortcut table ─────────────────────────────────────────────────
    any_shortcut = any(
        stats[cls][m]["shortcut_flag"]
        for cls in CLASS_NAMES
        for m in METRIC_NAMES
    )
    if any_shortcut:
        print("  SHORTCUT SUMMARY")
        print(f"  {'Class':<30}  {'Metric':<22}  {'Cohen_d':>8}")
        print(f"  {'-----':<30}  {'------':<22}  {'-------':>8}")
        for cls in CLASS_NAMES:
            for m in METRIC_NAMES:
                if stats[cls][m]["shortcut_flag"]:
                    d = stats[cls][m]["cohens_d"]
                    print(f"  {cls:<30}  {m:<22}  {d:>8.3f}")
        print()

    # ── JSON ──────────────────────────────────────────────────────────────────
    report = {
        "split": args.split,
        "cache": str(cache_path),
        "label_csv": str(label_csv),
        "splits_dir": str(splits_dir),
        "n_contracts_requested": args.n_contracts,
        "shortcut_threshold": SHORTCUT_THRESHOLD,
        "metrics": METRIC_NAMES,
        "class_stats": stats,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
