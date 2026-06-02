"""
exp_a2_cfg_inheritance.py — CFG Node Feature Inheritance Validation

PURPOSE
───────
Verifies that BUG-C3 (the fix applied during v8 graph extraction that makes
CFG nodes inherit dims [1,3,4,5,9] from their parent FUNCTION node) is actually
present in the extracted graphs stored in the cache.

Specifically: for a CFG_NODE_* (type_id 8–12) that lives inside a payable
function, the node must have feature dim[4] (payable) = 1.0.  Likewise, a
CFG_NODE inside a modifier-accessed function should have dim[5] (is_modifier)
consistent with its parent.

LAYER
─────
Layer 1 (Structure) — testable without any trained model, purely from the
graph features as stored in cached_dataset_v9.pkl.

WHAT IT TESTS
─────────────
1. Inheritance coverage rate: what fraction of CFG nodes inside payable
   FUNCTION nodes have dim[4] (payable) = 1.0?
   → Expected: ≈1.0 if BUG-C3 was applied; near 0 if not.

2. Inheritance dimension coverage: for each inherited dim
   [1]=visibility, [3]=has_loop, [4]=payable, [5]=is_modifier, [9]=has_state_write
   check consistency between parent FUNCTION and its child CFG nodes.

3. Spot-check mode: detailed print of 20 CFG nodes across 5 contracts,
   showing parent FUNCTION features vs. child CFG node features.

PASS CRITERIA
─────────────
- ≥90% of CFG nodes inside payable FUNCTION nodes have dim[4]=1.0
- ≥80% of CFG nodes have dim[4] matching their parent FUNCTION's dim[4]
  (across all functions, not just payable ones)
If either criterion fails: BUG-C3 is not correctly applied; the extractor
needs to be re-run with the fix active before training is meaningful.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_a2_cfg_inheritance.py \\
        --cache ml/data/cached_dataset_v9.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --out ml/logs/interpretability/exp_a2_cfg_inheritance.json

OUTPUT
──────
- Printed summary: inheritance rates per dimension, spot-check table
- JSON at --out with detailed stats
- Exit 0 if pass criteria met, 1 otherwise

NOTES
─────
- Uses CONTAINS edges (edge_attr==5 in v8 schema) to find parent FUNCTION of each CFG node.
- CFG nodes have type_id 8–12 (CFG_NODE_CALL=8, CFG_NODE_WRITE=9,
  CFG_NODE_READ=10, CFG_NODE_CHECK=11, CFG_NODE_OTHER=12).
- FUNCTION-like nodes have type_id 1–6 (FUNCTION=1, MODIFIER=2, EVENT=3,
  FALLBACK=4, RECEIVE=5, CONSTRUCTOR=6).  For inheritance purposes we care
  about FUNCTION(1), FALLBACK(4), RECEIVE(5), CONSTRUCTOR(6).
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

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import add_common_args, load_val_split, get_node_type_tensor
from ml.src.preprocessing.graph_schema import EDGE_TYPES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# CFG node type IDs
CFG_TYPE_IDS = {8, 9, 10, 11, 12}  # CALL, WRITE, READ, CHECK, OTHER

# Function-like node type IDs (owners of CFG subgraphs)
FUNC_TYPE_IDS = {1, 4, 5, 6}  # FUNCTION, FALLBACK, RECEIVE, CONSTRUCTOR

# Inherited feature dimensions (per BUG-C3 description)
INHERITED_DIMS = {
    1: "visibility_norm",
    3: "has_loop",
    4: "payable",
    5: "is_modifier",
    9: "has_state_write",
}

_MAX_TYPE_ID = 12.0
_CONTAINS_EDGE = EDGE_TYPES["CONTAINS"]  # 5 in v8 schema


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _build_contains_parent_map(graph) -> dict[int, int]:
    """
    For each node, find its parent via CONTAINS edge (parent CONTAINS child).
    Returns: {child_idx: parent_idx} for CONTAINS relationships.
    Only maps children of FUNCTION-like parents.
    """
    edge_index = graph.edge_index  # [2, E]
    edge_attr  = graph.edge_attr   # [E]
    node_types = get_node_type_tensor(graph)

    parent_of: dict[int, int] = {}
    for i in range(edge_attr.shape[0]):
        if int(edge_attr[i]) == _CONTAINS_EDGE:
            src = int(edge_index[0, i])  # parent
            dst = int(edge_index[1, i])  # child
            src_type = int(node_types[src])
            dst_type = int(node_types[dst])
            if src_type in FUNC_TYPE_IDS and dst_type in CFG_TYPE_IDS:
                parent_of[dst] = src
    return parent_of


def _analyse_graph(graph) -> dict:
    """
    For one graph, compute inheritance consistency statistics.
    Returns per-dim consistency counts.
    """
    x          = graph.x.float()   # [N, 11]
    node_types = get_node_type_tensor(graph)
    parent_of  = _build_contains_parent_map(graph)

    if not parent_of:
        return {}  # no CFG parent relationships found

    stats: dict = {
        "cfg_nodes_with_parent": len(parent_of),
        "per_dim": {},
    }

    for dim, dim_name in INHERITED_DIMS.items():
        match_count = 0
        total       = 0
        payable_parent_count = 0
        payable_child_match  = 0

        for child_idx, parent_idx in parent_of.items():
            child_val  = float(x[child_idx, dim])
            parent_val = float(x[parent_idx, dim])
            total += 1
            if abs(child_val - parent_val) < 0.01:  # float equality with tolerance
                match_count += 1

            # Special check: payable function → payable CFG node
            if dim == 4:
                if parent_val > 0.5:  # parent is payable
                    payable_parent_count += 1
                    if child_val > 0.5:
                        payable_child_match += 1

        consistency_rate = match_count / total if total > 0 else 0.0
        stats["per_dim"][dim] = {
            "name":             dim_name,
            "consistency_rate": consistency_rate,
            "match":            match_count,
            "total":            total,
        }
        if dim == 4:
            payable_rate = payable_child_match / payable_parent_count if payable_parent_count > 0 else None
            stats["per_dim"][dim]["payable_parent_count"] = payable_parent_count
            stats["per_dim"][dim]["payable_child_match"]  = payable_child_match
            stats["per_dim"][dim]["payable_inheritance_rate"] = payable_rate

    return stats


def _spot_check(stems: list[str], cache: dict, n_contracts: int = 5, n_nodes_each: int = 4) -> None:
    """Print a detailed table of N CFG nodes per contract showing parent vs child features."""
    printed = 0
    print(f"\n{'─'*90}")
    print(f"  CFG Node Feature Inheritance Spot-Check (first {n_contracts} contracts with CFG)")
    print(f"{'─'*90}")

    hdr_dim = " ".join(f"d{d:02d}" for d in range(11))
    print(f"  {'Role':<12} {'NodeType':<15} {'idx':>4}  {hdr_dim}")
    print(f"  {'-'*12} {'-'*15} {'----':>4}  " + "  ".join(["----"]*11))

    for stem in stems:
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple):
            continue
        graph = entry[0]
        parent_of = _build_contains_parent_map(graph)
        if not parent_of:
            continue

        shown = 0
        for child_idx, parent_idx in list(parent_of.items())[:n_nodes_each]:
            x = graph.x.float()
            node_types = get_node_type_tensor(graph)
            child_type  = int(node_types[child_idx])
            parent_type = int(node_types[parent_idx])
            parent_row  = " ".join(f"{float(x[parent_idx, d]):.2f}" for d in range(11))
            child_row   = " ".join(f"{float(x[child_idx, d]):.2f}" for d in range(11))
            type_names  = {1:"FUNCTION", 4:"FALLBACK", 5:"RECEIVE", 6:"CONSTRUCTOR",
                           8:"CFG_CALL", 9:"CFG_WRITE", 10:"CFG_READ", 11:"CFG_CHECK", 12:"CFG_OTHER"}
            print(f"  {'PARENT':<12} {type_names.get(parent_type, str(parent_type)):<15} {parent_idx:>4}  {parent_row}")
            print(f"  {'CHILD':<12} {type_names.get(child_type,  str(child_type)) :<15} {child_idx:>4}  {child_row}")
            # Highlight inheritance dimensions
            mismatch_dims = [d for d in INHERITED_DIMS if abs(float(x[parent_idx, d]) - float(x[child_idx, d])) >= 0.01]
            if mismatch_dims:
                print(f"  {'':12} {'':15} {'':4}  *** MISMATCH on dims: {mismatch_dims} ***")
            print()
            shown += 1

        printed += 1
        if printed >= n_contracts:
            break

    print(f"{'─'*90}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-A2: CFG node feature inheritance validation.")
    add_common_args(p, require_checkpoint=False)
    p.add_argument("--n-spot", type=int, default=5,
                   help="Contracts to show in spot-check table (default: 5)")
    return p.parse_args()


def main() -> int:
    args   = parse_args()
    out    = Path(args.out) if args.out else Path("ml/logs/interpretability/exp_a2_cfg_inheritance.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    stems, df_split, cache = load_val_split(
        Path(args.cache), Path(args.label_csv), Path(args.splits_dir), args.split
    )

    # Subsample
    rng = np.random.default_rng(args.seed)
    if args.n_contracts and len(stems) > args.n_contracts:
        idx   = rng.choice(len(stems), size=args.n_contracts, replace=False)
        stems = [stems[i] for i in sorted(idx)]

    log.info(f"Analysing {len(stems):,} contracts for CFG inheritance consistency …")

    aggregated: dict[int, dict] = {d: {"match":0,"total":0,"payable_parent":0,"payable_match":0}
                                    for d in INHERITED_DIMS}
    graphs_with_any_parents = 0
    graphs_analysed          = 0

    for stem in stems:
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple):
            continue
        graph = entry[0]
        stats = _analyse_graph(graph)
        graphs_analysed += 1
        if not stats:
            continue
        graphs_with_any_parents += 1
        for dim, info in stats.get("per_dim", {}).items():
            aggregated[dim]["match"]  += info["match"]
            aggregated[dim]["total"]  += info["total"]
            if dim == 4:
                aggregated[dim]["payable_parent"] += info.get("payable_parent_count", 0)
                aggregated[dim]["payable_match"]  += info.get("payable_child_match",  0)

    # ── Compute rates ─────────────────────────────────────────────────────────
    results: dict = {}
    print(f"\n{'─'*72}")
    print(f"  EXP-A2: CFG Feature Inheritance Validation (n={graphs_analysed:,} graphs)")
    print(f"{'─'*72}")
    print(f"  Graphs with ≥1 CFG→FUNCTION parent relationship: "
          f"{graphs_with_any_parents:,} / {graphs_analysed:,} "
          f"({100*graphs_with_any_parents/max(graphs_analysed,1):.1f}%)")
    print()
    print(f"  {'Dim':>3}  {'Feature':<22}  {'Consistency%':>13}  {'Total CFG nodes':>16}")
    print(f"  {'---':>3}  {'-------':<22}  {'------------':>13}  {'---------------':>16}")

    all_pass = True
    for dim, name in INHERITED_DIMS.items():
        agg  = aggregated[dim]
        rate = agg["match"] / agg["total"] if agg["total"] > 0 else 0.0
        results[dim] = {"dim": dim, "name": name, "consistency_rate": round(rate, 4),
                        "match": agg["match"], "total": agg["total"]}
        if dim == 4:
            pp = agg["payable_parent"]
            pm = agg["payable_match"]
            payable_rate = pm / pp if pp > 0 else None
            results[dim]["payable_parent_count"] = pp
            results[dim]["payable_child_match"]  = pm
            results[dim]["payable_inheritance_rate"] = round(payable_rate, 4) if payable_rate else None
            payable_str = f"  payable-specific: {pm}/{pp} = {payable_rate*100:.1f}%" if pp > 0 else ""
            print(f"  {dim:>3}  {name:<22}  {rate*100:>12.1f}%  {agg['total']:>16,}{payable_str}")
        else:
            print(f"  {dim:>3}  {name:<22}  {rate*100:>12.1f}%  {agg['total']:>16,}")

        if dim == 4:
            if payable_rate is not None and payable_rate < 0.90:
                all_pass = False
            if rate < 0.80:
                all_pass = False
        else:
            if rate < 0.80:
                all_pass = False

    # ── Gate checks ───────────────────────────────────────────────────────────
    dim4     = results.get(4, {})
    payable_r = dim4.get("payable_inheritance_rate")
    overall_r = dim4.get("consistency_rate", 0.0)

    print(f"\n{'─'*72}")
    print(f"  Pass Criteria")
    print(f"{'─'*72}")

    c1 = payable_r is not None and payable_r >= 0.90
    c2 = overall_r >= 0.80
    print(f"  [{'PASS' if c1 else 'FAIL'}]  Payable CFG inheritance rate ≥90%: "
          f"{f'{payable_r*100:.1f}%' if payable_r is not None else 'N/A'}")
    print(f"  [{'PASS' if c2 else 'FAIL'}]  Overall dim[4] consistency rate ≥80%: {overall_r*100:.1f}%")

    if not c1 or not c2:
        print("\n  *** WARNING: BUG-C3 may not be applied. Re-run graph extractor "
              "before trusting payable-based vulnerability detection. ***")
    else:
        print("\n  BUG-C3 feature inheritance is correctly applied. ✓")
    print(f"{'─'*72}\n")

    # ── Spot-check ────────────────────────────────────────────────────────────
    _spot_check(stems, cache, n_contracts=args.n_spot)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report = {
        "graphs_analysed":           graphs_analysed,
        "graphs_with_parents":       graphs_with_any_parents,
        "inheritance_stats":         {str(k): v for k, v in results.items()},
        "pass_payable_specific":     bool(c1),
        "pass_overall_consistency":  bool(c2),
        "passed":                    bool(c1 and c2),
    }
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {out}")

    return 0 if (c1 and c2) else 1


if __name__ == "__main__":
    sys.exit(main())
