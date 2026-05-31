"""
exp_e1_receptive_field.py — Layer 2, P1: K-Hop Receptive Field Analysis

PURPOSE
───────
For each contract graph, compute the set of nodes reachable within k hops
following only the edges used in each GNN phase.  This verifies that the
graph topology actually gives the GNN the structural reach it needs to detect
vulnerability patterns.

LAYER / PRIORITY
─────────────────
Layer 2, Priority 1 — Receptive-field topology verification.

HOW IT WORKS
────────────
Three analyses are run:

  Analysis 1 — CEI pattern reachability (Reentrancy)
    Phase 2 edges (CONTROL_FLOW=6, CALL_ENTRY=8, RETURN_TO=9, DEF_USE=10).
    For each reentrancy-positive contract: is there at least one
    CFG_NODE_WRITE (type 9) reachable from any CFG_NODE_CALL (type 8)?
    Reported at k=1..8.

  Analysis 2 — FUNCTION→CFG child coverage (CONTAINS only)
    CONTAINS=5 edge only (REVERSE_CONTAINS=7 is runtime-only, never stored).
    For all contracts: what fraction of FUNCTION nodes have ≥1 CFG_NODE
    child via CONTAINS within 1 hop?  Tests CFG extraction completeness.

  Analysis 3 — Intra-contract CALLS connectivity
    Phase 1 CALLS=0 edges only.
    For all contracts: what fraction of FUNCTION nodes can reach ≥1 other
    FUNCTION node via CALLS within 2 hops?  Tests call-graph connectivity.

NOTE: Original Analysis 2 used REVERSE_CONTAINS which is runtime-only
(built by GNNEncoder Phase 3, never in .pt files).  Original Analysis 3
checked CONTRACT→FUNCTION reachability, but no such edge type exists in
v8 schema — all Phase 1 edges originate from FUNCTION or CONTRACT nodes,
but CONTAINS goes FUNCTION→CFG_NODE, not CONTRACT→FUNCTION.

PASS CRITERIA
─────────────
  1. Reentrancy positives: >= 50% of contracts have CFG_WRITE reachable
     from CFG_CALL via Phase 2 edges within 8 hops.
  2. FUNCTION→CFG coverage: >= 80% of FUNCTION nodes have ≥1 CFG child.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_e1_receptive_field.py \\
        --cache ml/data/cached_dataset_v8.pkl \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \\
        --splits-dir ml/data/splits/deduped \\
        --out ml/logs/interpretability/e1_receptive_field.json \\
        --n-contracts 200

    # No --checkpoint needed.

OUTPUT
──────
Per-k reachability rates table, summary statistics, JSON report.

EXIT CODES
──────────
    0  all pass criteria satisfied
    1  one or more pass criteria failed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Phase edge-type sets (v8 schema: CALLS=0, CONTAINS=5, CONTROL_FLOW=6,
#   REVERSE_CONTAINS=7 runtime-only, CALL_ENTRY=8, RETURN_TO=9, DEF_USE=10)
PHASE1_EDGE_TYPES = {
    EDGE_TYPES["CALLS"],      # 0 — function → internally-called function
    EDGE_TYPES["READS"],      # 1
    EDGE_TYPES["WRITES"],     # 2
    EDGE_TYPES["EMITS"],      # 3
    EDGE_TYPES["INHERITS"],   # 4
    EDGE_TYPES["CONTAINS"],   # 5 — function → CFG_NODE children
}
PHASE2_EDGE_TYPES = {
    EDGE_TYPES["CONTROL_FLOW"],  # 6
    EDGE_TYPES["CALL_ENTRY"],    # 8
    EDGE_TYPES["RETURN_TO"],     # 9
    EDGE_TYPES["DEF_USE"],       # 10 — def→use data-flow (IntegerUO, UnusedReturn)
}
PHASE3_EDGE_TYPES = {
    EDGE_TYPES["REVERSE_CONTAINS"],  # 7 — runtime-only; never in stored .pt files
    EDGE_TYPES["CONTAINS"],          # 5
}
# CONTAINS-only subset used for Analysis 2 (REVERSE_CONTAINS not in cache)
_CONTAINS_ONLY = {EDGE_TYPES["CONTAINS"]}
# CALLS-only subset used for Analysis 3
_CALLS_ONLY = {EDGE_TYPES["CALLS"]}

# Node type IDs
_CFG_NODE_CALL  = NODE_TYPES["CFG_NODE_CALL"]   # 8
_CFG_NODE_WRITE = NODE_TYPES["CFG_NODE_WRITE"]  # 9
_FUNCTION       = NODE_TYPES["FUNCTION"]         # 1
_CONTRACT       = NODE_TYPES["CONTRACT"]         # 7

# CFG node types (for Phase 3 coverage check)
_CFG_NODE_TYPES = {
    NODE_TYPES.get("CFG_NODE_CALL",   8),
    NODE_TYPES.get("CFG_NODE_WRITE",  9),
    NODE_TYPES.get("CFG_NODE_READ",  10),
    NODE_TYPES.get("CFG_NODE_CHECK", 11),
    NODE_TYPES.get("CFG_NODE_OTHER", 12),
}


# ── Core BFS ──────────────────────────────────────────────────────────────────

def k_hop_neighbors(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    allowed_edge_types: set,
    start_nodes: set,
    k_max: int = 8,
) -> dict:
    """BFS from start_nodes following only edges with type in allowed_edge_types.

    Returns:
        dict {k: set of ALL reachable node indices up to and including hop k}
        for k = 1 .. k_max.  k=0 corresponds to start_nodes themselves.
    """
    # Build adjacency filtered by edge type
    adj: dict[int, set] = defaultdict(set)
    if edge_index.numel() > 0 and edge_attr is not None:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        ets  = edge_attr.tolist()
        for src, dst, et in zip(srcs, dsts, ets):
            if int(et) in allowed_edge_types:
                adj[src].add(dst)

    frontier = set(start_nodes)
    visited  = set(start_nodes)
    by_hop: dict[int, set] = {}
    for k in range(1, k_max + 1):
        next_frontier: set = set()
        for node in frontier:
            next_frontier |= adj[node] - visited
        visited        |= next_frontier
        frontier        = next_frontier
        by_hop[k]       = set(visited)
    return by_hop


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _analysis1_cei_reachability(graph, k_max: int = 8) -> dict[int, bool]:
    """
    Returns {k: bool} — True if CFG_NODE_WRITE reachable from any CFG_NODE_CALL
    within k hops following Phase 2 edges.
    """
    node_types = get_node_type_tensor(graph)
    call_nodes  = set(torch.where(node_types == _CFG_NODE_CALL)[0].tolist())
    write_nodes = set(torch.where(node_types == _CFG_NODE_WRITE)[0].tolist())

    if not call_nodes or not write_nodes:
        return {k: False for k in range(1, k_max + 1)}

    by_hop = k_hop_neighbors(
        graph.edge_index, graph.edge_attr, PHASE2_EDGE_TYPES, call_nodes, k_max
    )
    result: dict[int, bool] = {}
    for k in range(1, k_max + 1):
        reachable = by_hop.get(k, set())
        result[k] = bool(reachable & write_nodes)
    return result


def _analysis2_function_cfg_coverage(graph) -> list[bool]:
    """
    For each FUNCTION node: does it have >=1 CFG_NODE child via CONTAINS
    within 1 hop?  Tests CFG extraction completeness.

    REVERSE_CONTAINS (type 7) is NOT used — it is runtime-only and never
    stored in .pt graph files.

    Returns a list of bool — one per FUNCTION node.
    """
    node_types = get_node_type_tensor(graph)
    func_nodes = torch.where(node_types == _FUNCTION)[0].tolist()
    cfg_set: set[int] = set()
    for t in _CFG_NODE_TYPES:
        cfg_set |= set(torch.where(node_types == t)[0].tolist())

    if not func_nodes:
        return []

    results: list[bool] = []
    for fn in func_nodes:
        by_hop = k_hop_neighbors(
            graph.edge_index, graph.edge_attr, _CONTAINS_ONLY, {fn}, k_max=1
        )
        reachable = by_hop.get(1, set())
        results.append(bool(reachable & cfg_set))
    return results


def _analysis3_calls_connectivity(graph, k_max: int = 2) -> list[bool]:
    """
    For each FUNCTION node: can it reach >=1 other FUNCTION node via
    CALLS edges within k_max hops?  Tests intra-contract call graph
    connectivity.

    (Original Analysis 3 checked CONTRACT→FUNCTION reachability, but no
    such edge type exists in v8 schema — replaced with this check.)

    Returns a list of bool — one per FUNCTION node.
    """
    node_types = get_node_type_tensor(graph)
    func_nodes = torch.where(node_types == _FUNCTION)[0].tolist()
    func_set   = set(func_nodes)

    if len(func_nodes) < 2:
        return []

    results: list[bool] = []
    for fn in func_nodes:
        by_hop = k_hop_neighbors(
            graph.edge_index, graph.edge_attr, _CALLS_ONLY, {fn}, k_max
        )
        reachable = by_hop.get(k_max, set())
        results.append(bool(reachable & (func_set - {fn})))
    return results


# ── Main analysis loop ────────────────────────────────────────────────────────

def run_analyses(
    stems: list[str],
    df_split: pd.DataFrame,
    cache: dict,
    n_contracts: int,
    seed: int = 42,
) -> dict:
    """Run all three analyses and return aggregated results."""
    reentrancy_col = "Reentrancy"
    stem_to_row = {row["md5_stem"]: row for _, row in df_split.iterrows()}

    rng = np.random.default_rng(seed)

    # Sample reentrancy positives
    pos_stems = [s for s in stems if s in stem_to_row and int(stem_to_row[s].get(reentrancy_col, 0)) == 1]
    neg_stems = [s for s in stems if s in stem_to_row and int(stem_to_row[s].get(reentrancy_col, 0)) == 0]

    if len(pos_stems) > n_contracts:
        pos_stems = rng.choice(pos_stems, size=n_contracts, replace=False).tolist()
    if len(neg_stems) > n_contracts:
        neg_stems = rng.choice(neg_stems, size=n_contracts, replace=False).tolist()

    # All stems for analyses 2 and 3
    all_sample = list({*pos_stems, *neg_stems})
    if len(all_sample) > n_contracts * 2:
        all_sample = rng.choice(all_sample, size=n_contracts * 2, replace=False).tolist()

    K_MAX = 8

    # Analysis 1: CEI reachability per k
    a1_pos_found = {k: 0 for k in range(1, K_MAX + 1)}  # reentrancy positives
    a1_pos_total = 0
    a1_neg_found = {k: 0 for k in range(1, K_MAX + 1)}  # reentrancy negatives
    a1_neg_total = 0

    log.info(f"Analysis 1: CEI reachability ({len(pos_stems)} pos, {len(neg_stems)} neg)")
    for stem in pos_stems:
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple):
            continue
        graph, _ = entry
        if graph.edge_attr is None or graph.edge_index.numel() == 0:
            continue
        found_by_k = _analysis1_cei_reachability(graph, K_MAX)
        a1_pos_total += 1
        for k in range(1, K_MAX + 1):
            if found_by_k[k]:
                a1_pos_found[k] += 1

    for stem in neg_stems:
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple):
            continue
        graph, _ = entry
        if graph.edge_attr is None or graph.edge_index.numel() == 0:
            continue
        found_by_k = _analysis1_cei_reachability(graph, K_MAX)
        a1_neg_total += 1
        for k in range(1, K_MAX + 1):
            if found_by_k[k]:
                a1_neg_found[k] += 1

    a1_pos_rates = {
        k: round(100.0 * a1_pos_found[k] / a1_pos_total, 2) if a1_pos_total > 0 else 0.0
        for k in range(1, K_MAX + 1)
    }
    a1_neg_rates = {
        k: round(100.0 * a1_neg_found[k] / a1_neg_total, 2) if a1_neg_total > 0 else 0.0
        for k in range(1, K_MAX + 1)
    }

    # Analysis 2: FUNCTION→CFG child coverage via CONTAINS (on-disk only)
    log.info(f"Analysis 2: FUNCTION→CFG coverage via CONTAINS ({len(all_sample)} contracts)")
    a2_results: list[bool] = []
    for stem in all_sample:
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple):
            continue
        graph, _ = entry
        if graph.edge_attr is None:
            continue
        a2_results.extend(_analysis2_function_cfg_coverage(graph))

    a2_total      = len(a2_results)
    a2_pass_count = sum(a2_results)
    a2_pass_rate  = round(100.0 * a2_pass_count / a2_total, 2) if a2_total > 0 else 0.0

    # Analysis 3: intra-contract CALLS connectivity
    log.info(f"Analysis 3: CALLS connectivity ({len(all_sample)} contracts)")
    a3_results: list[bool] = []
    for stem in all_sample:
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple):
            continue
        graph, _ = entry
        if graph.edge_attr is None:
            continue
        a3_results.extend(_analysis3_calls_connectivity(graph, k_max=2))

    a3_total      = len(a3_results)
    a3_pass_count = sum(a3_results)
    a3_pass_rate  = round(100.0 * a3_pass_count / a3_total, 2) if a3_total > 0 else 0.0

    return {
        "analysis1_cei_reachability": {
            "description": "CFG_WRITE reachable from CFG_CALL via Phase2 edges",
            "reentrancy_positive": {
                "n_contracts":  a1_pos_total,
                "rates_by_hop": a1_pos_rates,
            },
            "reentrancy_negative": {
                "n_contracts":  a1_neg_total,
                "rates_by_hop": a1_neg_rates,
            },
            "pass_criterion":  ">=50% positive contracts have CFG_WRITE reachable at k<=8",
            "pass": a1_pos_rates.get(K_MAX, 0.0) >= 50.0,
        },
        "analysis2_function_cfg_coverage": {
            "description": "FUNCTION nodes with >=1 CFG child via CONTAINS (1-hop; REVERSE_CONTAINS excluded — runtime-only)",
            "n_function_nodes":  a2_total,
            "pass_count":        a2_pass_count,
            "pass_rate_pct":     a2_pass_rate,
            "pass_criterion":    ">=80% of FUNCTION nodes have >=1 CFG child via CONTAINS",
            "pass":              a2_pass_rate >= 80.0,
        },
        "analysis3_calls_connectivity": {
            "description": "FUNCTION nodes that can reach >=1 other FUNCTION via CALLS within 2 hops",
            "n_function_nodes": a3_total,
            "pass_count":       a3_pass_count,
            "pass_rate_pct":    a3_pass_rate,
        },
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(results: dict) -> None:
    a1 = results["analysis1_cei_reachability"]
    a2 = results["analysis2_function_cfg_coverage"]
    a3 = results["analysis3_calls_connectivity"]

    print(f"\n{'═'*72}")
    print("  EXP-E1: K-Hop Receptive Field Analysis")
    print(f"{'═'*72}")

    print("\n  Analysis 1: CEI Reachability (Phase 2 edges)")
    print(f"  Reentrancy positives: {a1['reentrancy_positive']['n_contracts']} contracts")
    print(f"  Reentrancy negatives: {a1['reentrancy_negative']['n_contracts']} contracts")
    print(f"\n  {'k':>4}  {'Pos rate':>10}  {'Neg rate':>10}")
    print(f"  {'----':>4}  {'---------':>10}  {'---------':>10}")
    for k in range(1, 9):
        pr = a1["reentrancy_positive"]["rates_by_hop"].get(k, 0.0)
        nr = a1["reentrancy_negative"]["rates_by_hop"].get(k, 0.0)
        print(f"  {k:>4}  {pr:>9.1f}%  {nr:>9.1f}%")
    print(f"\n  A1 PASS criterion: >=50% positives at k=8: "
          f"{'PASS' if a1['pass'] else 'FAIL'} "
          f"({a1['reentrancy_positive']['rates_by_hop'].get(8, 0.0):.1f}%)")

    print(f"\n  Analysis 2: FUNCTION→CFG Coverage via CONTAINS (1-hop)")
    print(f"  Total FUNCTION nodes sampled: {a2['n_function_nodes']}")
    print(f"  Have >=1 CFG child: {a2['pass_count']} ({a2['pass_rate_pct']:.1f}%)")
    print(f"  A2 PASS criterion: >=80%: {'PASS' if a2['pass'] else 'FAIL'}")

    print(f"\n  Analysis 3: CALLS Connectivity (k=2)")
    print(f"  Total FUNCTION nodes sampled: {a3['n_function_nodes']}")
    print(f"  Can reach another FUNCTION via CALLS: {a3['pass_count']} ({a3['pass_rate_pct']:.1f}%)")

    overall = a1["pass"] and a2["pass"]
    print(f"\n{'─'*72}")
    print(f"  EXP-E1 RESULT: {'PASS' if overall else 'FAIL'}")
    print(f"{'═'*72}\n")


def _save_plot(results: dict, out_dir: Path) -> None:
    a1 = results["analysis1_cei_reachability"]
    pos_rates = [a1["reentrancy_positive"]["rates_by_hop"].get(k, 0.0) for k in range(1, 9)]
    neg_rates = [a1["reentrancy_negative"]["rates_by_hop"].get(k, 0.0) for k in range(1, 9)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(range(1, 9))
    ax.plot(ks, pos_rates, marker="o", label="Reentrancy positive")
    ax.plot(ks, neg_rates, marker="s", label="Reentrancy negative")
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, label="50% pass threshold")
    ax.set_xlabel("k (hop depth)")
    ax.set_ylabel("% contracts with CFG_WRITE reachable from CFG_CALL")
    ax.set_title("E1: CEI Reachability via Phase 2 Edges")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "e1_cei_reachability.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-E1: K-Hop Receptive Field Analysis"
    )
    add_common_args(p, require_checkpoint=False)
    p.set_defaults(n_contracts=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_path = Path(args.out) if args.out else Path("ml/logs/interpretability/e1_receptive_field.json")
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

    results = run_analyses(
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
            "cache":        str(args.cache),
            "label_csv":    str(args.label_csv),
            "splits_dir":   str(args.splits_dir),
            "split":        args.split,
            "n_contracts":  args.n_contracts,
            "seed":         args.seed,
        },
        "results":       results,
        "overall_pass":  (
            results["analysis1_cei_reachability"]["pass"]
            and results["analysis2_function_aggregation"]["pass"]
        ),
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {out_path}")

    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
