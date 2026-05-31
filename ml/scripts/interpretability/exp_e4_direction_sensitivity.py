"""
exp_e4_direction_sensitivity.py — Layer 2, P2: Direction Sensitivity:
Directed vs Undirected CFG

PURPOSE
───────
Tests whether directed CONTROL_FLOW edges provide more distinguishing power
than undirected edges, using the Typed Directed WL test from exp_e2.  If
making CONTROL_FLOW bidirectional (undirected) does not reduce WL
distinguishability, the directionality of the CFG is not helping the GNN
separate positive from negative contracts.

LAYER / PRIORITY
─────────────────
Layer 2, Priority 2 — Edge direction ablation study.

HOW IT WORKS
────────────
1. For each reentrancy positive/negative pair (matched by node count):
   a. Directed graph: CONTROL_FLOW edges kept as-is (A->B only).
   b. Undirected graph: for every CONTROL_FLOW edge A->B, also add B->A with
      the same edge type.
2. Run typed directed WL (8 rounds) on both versions and measure % of pairs
   that are WL-distinguishable (NOT equivalent) at each round.
3. Compare directed vs undirected distinguishability curves.

WL functions are imported from exp_e2 (compute_wl_hashes, graph_wl_hash).

PASS CRITERIA
─────────────
  Directed graph produces >= 10% more distinguishable pairs than undirected
  at round 8.
  If difference < 5%: direction is NOT helping — warn user.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_e4_direction_sensitivity.py \\
        --cache ml/data/cached_dataset_v8.pkl \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \\
        --splits-dir ml/data/splits/deduped \\
        --out ml/logs/interpretability/e4_direction_sensitivity.json \\
        --n-contracts 100

    # No --checkpoint needed.

OUTPUT
──────
Directed vs undirected distinguishability comparison table, difference plot,
JSON report.

EXIT CODES
──────────
    0  directed >= undirected + 10% (direction is helping)
    1  difference < 10% (direction not clearly helping)
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    add_common_args,
    load_val_split,
    CLASS_NAMES,
    get_node_type_tensor,
)
from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES
from ml.scripts.interpretability.exp_e2_wl_distinguishability import (
    compute_wl_hashes,
    graph_wl_hash,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_CONTROL_FLOW = EDGE_TYPES["CONTROL_FLOW"]  # 6
WL_ROUNDS     = 8


# ── Graph manipulation ────────────────────────────────────────────────────────

def _make_cf_undirected(
    edge_index: torch.Tensor,
    edge_attr:  torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return a new (edge_index, edge_attr) where every CONTROL_FLOW edge A->B
    has a reverse edge B->A with the same edge type added.

    Non-CONTROL_FLOW edges are left unchanged.
    Duplicate reverse edges (already present) are NOT deduplicated — the WL
    hash is order-invariant via sorted(), so duplicates only affect hash speed,
    not correctness.
    """
    if edge_index.numel() == 0 or edge_attr is None:
        return edge_index, edge_attr

    cf_mask = edge_attr == _CONTROL_FLOW
    if not cf_mask.any():
        return edge_index, edge_attr

    cf_ei   = edge_index[:, cf_mask]              # [2, E_cf]
    cf_rev  = torch.stack([cf_ei[1], cf_ei[0]], dim=0)  # reversed
    cf_ea   = edge_attr[cf_mask]

    new_ei = torch.cat([edge_index, cf_rev], dim=1)
    new_ea = torch.cat([edge_attr,  cf_ea],  dim=0)
    return new_ei, new_ea


# ── Pairing ───────────────────────────────────────────────────────────────────

def _pair_by_node_count(
    pos_graphs: list,
    neg_graphs: list,
) -> list[tuple]:
    """Pair positive and negative graphs by closest node count (greedy)."""
    if not pos_graphs or not neg_graphs:
        return []

    neg_pool  = list(neg_graphs)
    used_neg  = [False] * len(neg_pool)
    pairs: list[tuple] = []

    for pg in pos_graphs:
        pn = pg.x.shape[0]
        best_idx  = -1
        best_diff = float("inf")
        for i, ng in enumerate(neg_pool):
            if used_neg[i]:
                continue
            diff = abs(ng.x.shape[0] - pn)
            if diff < best_diff:
                best_diff = diff
                best_idx  = i
        if best_idx >= 0:
            used_neg[best_idx] = True
            pairs.append((pg, neg_pool[best_idx]))

    return pairs


# ── WL distinguishability for a set of pairs ─────────────────────────────────

def _distinguishability_curve(
    pairs: list[tuple],
    undirected: bool,
    n_rounds: int = WL_ROUNDS,
) -> dict[int, float]:
    """
    Compute % of pairs that are WL-DISTINGUISHABLE (not equivalent) at each
    round.

    Args:
        pairs:      List of (pos_graph, neg_graph) pairs.
        undirected: If True, make CONTROL_FLOW edges undirected before WL.
        n_rounds:   Number of WL rounds.

    Returns:
        {round: distinguishable_pct}
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {r: 0.0 for r in range(1, n_rounds + 1)}

    distinguishable = {r: 0 for r in range(1, n_rounds + 1)}

    for pg, ng in pairs:
        p_types = get_node_type_tensor(pg).tolist()
        n_types = get_node_type_tensor(ng).tolist()

        p_ei = pg.edge_index
        p_ea = pg.edge_attr if pg.edge_attr is not None else torch.zeros(p_ei.shape[1], dtype=torch.long)
        n_ei = ng.edge_index
        n_ea = ng.edge_attr if ng.edge_attr is not None else torch.zeros(n_ei.shape[1], dtype=torch.long)

        if undirected:
            p_ei, p_ea = _make_cf_undirected(p_ei, p_ea)
            n_ei, n_ea = _make_cf_undirected(n_ei, n_ea)

        p_rounds = compute_wl_hashes(p_types, p_ei, p_ea, n_rounds)
        n_rounds_hashes = compute_wl_hashes(n_types, n_ei, n_ea, n_rounds)

        for r in range(1, n_rounds + 1):
            p_gh = graph_wl_hash(p_rounds[r - 1])
            n_gh = graph_wl_hash(n_rounds_hashes[r - 1])
            if p_gh != n_gh:
                distinguishable[r] += 1

    return {
        r: round(100.0 * distinguishable[r] / n_pairs, 2)
        for r in range(1, n_rounds + 1)
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(
    stems: list[str],
    df_split: pd.DataFrame,
    cache: dict,
    n_contracts: int,
    seed: int = 42,
) -> dict:
    """Run directed vs undirected CFG WL distinguishability comparison."""
    rng = np.random.default_rng(seed)
    stem_to_row = {row["md5_stem"]: row for _, row in df_split.iterrows()}

    pos_stems = [s for s in stems if s in stem_to_row and int(stem_to_row[s].get("Reentrancy", 0)) == 1]
    neg_stems = [s for s in stems if s in stem_to_row and int(stem_to_row[s].get("Reentrancy", 0)) == 0]

    if len(pos_stems) > n_contracts:
        pos_stems = rng.choice(pos_stems, size=n_contracts, replace=False).tolist()
    if len(neg_stems) > n_contracts:
        neg_stems = rng.choice(neg_stems, size=n_contracts, replace=False).tolist()

    def _load(stem_list: list[str]) -> list:
        graphs = []
        for s in stem_list:
            if s not in cache:
                continue
            entry = cache[s]
            if not isinstance(entry, tuple):
                continue
            g, _ = entry
            if g is None or g.x is None or g.x.shape[0] == 0:
                continue
            graphs.append(g)
        return graphs

    pos_graphs = _load(pos_stems)
    neg_graphs = _load(neg_stems)
    log.info(f"Reentrancy: {len(pos_graphs)} pos / {len(neg_graphs)} neg graphs loaded")

    pairs = _pair_by_node_count(pos_graphs, neg_graphs)
    log.info(f"Formed {len(pairs)} pos/neg pairs")

    if not pairs:
        return {
            "n_pairs":            0,
            "directed":           {},
            "undirected":         {},
            "difference":         {},
            "final_diff_pct":     None,
            "overall_pass":       False,
            "warning":            "no pairs formed",
        }

    log.info("Computing directed WL curves...")
    directed_curve   = _distinguishability_curve(pairs, undirected=False)

    log.info("Computing undirected CF WL curves...")
    undirected_curve = _distinguishability_curve(pairs, undirected=True)

    diff_curve = {
        r: round(directed_curve[r] - undirected_curve[r], 2)
        for r in range(1, WL_ROUNDS + 1)
    }

    final_diff = diff_curve[WL_ROUNDS]
    passed = final_diff >= 10.0
    warn   = final_diff < 5.0

    return {
        "n_pairs":        len(pairs),
        "directed":       directed_curve,
        "undirected":     undirected_curve,
        "difference":     diff_curve,
        "final_directed_pct":   directed_curve[WL_ROUNDS],
        "final_undirected_pct": undirected_curve[WL_ROUNDS],
        "final_diff_pct": final_diff,
        "pass_criterion": "directed >= undirected + 10% distinguishable at round 8",
        "overall_pass":   passed,
        "warning": (
            "Direction is NOT helping: directed and undirected CF curves within 5%"
            if warn else None
        ),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(results: dict) -> None:
    print(f"\n{'═'*72}")
    print("  EXP-E4: Direction Sensitivity (Directed vs Undirected CFG)")
    print(f"{'═'*72}")
    print(f"\n  Pairs: {results['n_pairs']}")
    print(f"\n  {'Round':>6}  {'Directed':>12}  {'Undirected':>12}  {'Diff':>8}")
    print(f"  {'------':>6}  {'------------':>12}  {'------------':>12}  {'--------':>8}")
    for r in range(1, WL_ROUNDS + 1):
        d  = results["directed"].get(r, 0.0)
        u  = results["undirected"].get(r, 0.0)
        di = results["difference"].get(r, 0.0)
        print(f"  {r:>6}  {d:>11.1f}%  {u:>11.1f}%  {di:>7.1f}%")

    print(f"\n  Pass criterion: directed - undirected >= 10% at round 8")
    print(f"  Final diff: {results.get('final_diff_pct', 'N/A'):.1f}%"
          if results.get("final_diff_pct") is not None else "  Final diff: N/A")
    print(f"  Result: {'PASS' if results['overall_pass'] else 'FAIL'}")

    if results.get("warning"):
        print(f"\n  WARNING: {results['warning']}")

    print(f"\n{'─'*72}")
    print(f"  EXP-E4 RESULT: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print(f"{'═'*72}\n")


def _save_plot(results: dict, out_dir: Path) -> None:
    rounds = list(range(1, WL_ROUNDS + 1))
    dir_vals  = [results["directed"].get(r, 0.0)   for r in rounds]
    undir_vals = [results["undirected"].get(r, 0.0) for r in rounds]
    diff_vals  = [results["difference"].get(r, 0.0) for r in rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rounds, dir_vals,   marker="o", label="Directed CFG")
    ax1.plot(rounds, undir_vals, marker="s", label="Undirected CFG")
    ax1.set_xlabel("WL round")
    ax1.set_ylabel("% distinguishable pairs")
    ax1.set_title("Distinguishability: Directed vs Undirected")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(rounds, diff_vals, color=["green" if v >= 10 else "orange" if v >= 5 else "red" for v in diff_vals])
    ax2.axhline(10, color="green", linestyle="--", linewidth=0.8, label="10% pass threshold")
    ax2.axhline(5,  color="orange", linestyle=":", linewidth=0.8, label="5% warn threshold")
    ax2.set_xlabel("WL round")
    ax2.set_ylabel("Directed - Undirected (%)")
    ax2.set_title("Direction Benefit (diff)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("E4: CFG Direction Sensitivity")
    plt.tight_layout()
    plot_path = out_dir / "e4_direction_sensitivity.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-E4: Direction Sensitivity (Directed vs Undirected CFG)"
    )
    add_common_args(p, require_checkpoint=False)
    p.set_defaults(n_contracts=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else Path("ml/logs/interpretability/e4_direction_sensitivity.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stems, df_split, cache = load_val_split(
            cache_path  = Path(args.cache),
            label_csv   = Path(args.label_csv),
            splits_dir  = Path(args.splits_dir),
            split       = args.split,
        )
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    results = run_analysis(
        stems       = stems,
        df_split    = df_split,
        cache       = cache,
        n_contracts = args.n_contracts,
        seed        = args.seed,
    )

    _print_report(results)
    _save_plot(results, out_path.parent)

    report = {
        "args": {
            "cache":       str(args.cache),
            "label_csv":   str(args.label_csv),
            "splits_dir":  str(args.splits_dir),
            "split":       args.split,
            "n_contracts": args.n_contracts,
            "seed":        args.seed,
        },
        "results":      results,
        "overall_pass": results["overall_pass"],
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {out_path}")

    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
