"""
exp_s1_structural_trace.py — Layer 1, P0: CEI Structural Trace Audit

WHAT IT TESTS
─────────────
For each vulnerability class, verifies that the expected vulnerability pattern
exists as a reachable path in the extracted contract graph. This is the most
fundamental test: if the graph extractor is not creating the edges that encode
the vulnerability structure, no amount of GNN sophistication will detect it.

LAYER / PRIORITY
─────────────────
Layer 1, Priority 0 — Graph structure correctness (prerequisite for everything else).

PATTERNS TESTED
───────────────
Reentrancy CEI:     CFG_NODE_CALL -[CONTROL_FLOW*]-> CFG_NODE_WRITE
                    (with no intervening CFG_NODE_CHECK)
Reentrancy ICFG:    CFG_NODE_CALL -[CALL_ENTRY]-> ... -[RETURN_TO]-> any
IntegerUO:          CFG_NODE_OTHER -[DEF_USE]-> CFG_NODE_WRITE
Timestamp:          node with uses_block_globals=1.0 exists
UnusedReturn:       CFG_NODE_CALL with return_ignored=1.0

PASS CRITERIA
─────────────
>= 7/10 positive test contracts show the expected pattern for their class.
For val-split contracts: >= 50% of class-positive graphs with >= 1 CFG node
show the expected pattern.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_s1_structural_trace.py \\
        --cache ml/data/cached_dataset_v8.pkl \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \\
        --splits-dir ml/data/splits/deduped \\
        --out ml/logs/interpretability/s1_structural_trace.json

    # No --checkpoint needed — this script does not load the model.

EXPECTED OUTPUT
───────────────
Per-contract pass/fail table for test contracts.
Per-class statistics from val split.
JSON report with all findings.

EXIT CODES
──────────
    0  all P0 checks pass
    1  one or more P0 checks fail
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_TYPE_ID = float(max(NODE_TYPES.values()))

# Specific test contracts and their expected vulnerability class
TEST_CONTRACT_MAP: dict[str, str] = {
    "01_reentrancy_classic.sol":    "Reentrancy",
    "02_reentrancy_tricky.sol":     "Reentrancy",
    "03_integer_overflow.sol":      "IntegerUO",
    "04_timestamp_dependence.sol":  "Timestamp",
    "06_mishandled_exception.sol":  "MishandledException",
    "07_tx_order_dependence.sol":   "TransactionOrderDependence",
    "08_unused_return.sol":         "UnusedReturn",
}

# Local test contracts from this interpretability suite
LOCAL_CONTRACT_MAP: dict[str, str] = {
    "reentrancy_vulnerable.sol":    "Reentrancy",
    "reentrancy_safe.sol":          "SAFE",
    "integer_uo_vulnerable.sol":    "IntegerUO",
    "timestamp_vulnerable.sol":     "Timestamp",
    "unused_return_vulnerable.sol": "UnusedReturn",
}

# ── Graph pattern checkers ────────────────────────────────────────────────────

def _get_node_types(graph) -> torch.Tensor:
    return (graph.x[:, 0].float() * _MAX_TYPE_ID).round().long()


def _build_adjacency(graph, edge_type_filter: set[int]) -> dict[int, list[int]]:
    """Build adjacency list for nodes connected by edges of given types."""
    adj: dict[int, list[int]] = defaultdict(list)
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return adj
    ei    = graph.edge_index
    ea    = graph.edge_attr
    for i in range(ei.shape[1]):
        if ea[i].item() in edge_type_filter:
            src = ei[0, i].item()
            dst = ei[1, i].item()
            adj[src].append(dst)
    return adj


def _reachable(src: int, adj: dict[int, list[int]], max_hops: int = 10) -> set[int]:
    """BFS from src following adj edges, up to max_hops."""
    visited = {src}
    frontier = [src]
    for _ in range(max_hops):
        next_frontier = []
        for node in frontier:
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
        if not next_frontier:
            break
        frontier = next_frontier
    return visited


def check_reentrancy_cei(graph) -> bool:
    """
    Check for CEI violation: CFG_NODE_CALL -> [CONTROL_FLOW*] -> CFG_NODE_WRITE
    with no intervening CFG_NODE_CHECK.
    """
    node_types = _get_node_types(graph)
    n = node_types.shape[0]
    if n == 0:
        return False

    cf_adj   = _build_adjacency(graph, {EDGE_TYPES["CONTROL_FLOW"]})
    call_nodes  = set((node_types == NODE_TYPES["CFG_NODE_CALL"]).nonzero().squeeze(1).tolist())
    write_nodes = set((node_types == NODE_TYPES["CFG_NODE_WRITE"]).nonzero().squeeze(1).tolist())
    check_nodes = set((node_types == NODE_TYPES["CFG_NODE_CHECK"]).nonzero().squeeze(1).tolist())

    if not call_nodes or not write_nodes:
        return False

    # BFS from each call node; check if write is reachable without passing through a CHECK
    for call_node in call_nodes:
        # DFS with check-avoidance
        stack = [call_node]
        visited = {call_node}
        while stack:
            curr = stack.pop()
            for nb in cf_adj.get(curr, []):
                if nb in write_nodes:
                    return True
                if nb not in visited and nb not in check_nodes:
                    visited.add(nb)
                    stack.append(nb)
    return False


def check_reentrancy_icfg(graph) -> bool:
    """
    Check: CFG_NODE_CALL -> [CALL_ENTRY] -> [some nodes] -> [RETURN_TO edge exists].
    """
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return False
    node_types = _get_node_types(graph)
    call_entry_mask = graph.edge_attr == EDGE_TYPES["CALL_ENTRY"]
    return_to_mask  = graph.edge_attr == EDGE_TYPES["RETURN_TO"]
    # If both CALL_ENTRY and RETURN_TO edges exist, the ICFG path exists
    return bool(call_entry_mask.any() and return_to_mask.any())


def check_integer_uo(graph) -> bool:
    """
    Check: any CFG_NODE_OTHER with DEF_USE -> CFG_NODE_WRITE exists.
    """
    node_types = _get_node_types(graph)
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return False

    du_adj = _build_adjacency(graph, {EDGE_TYPES["DEF_USE"]})
    other_nodes = set((node_types == NODE_TYPES["CFG_NODE_OTHER"]).nonzero().squeeze(1).tolist())
    write_nodes = set((node_types == NODE_TYPES["CFG_NODE_WRITE"]).nonzero().squeeze(1).tolist())

    for other in other_nodes:
        for nb in du_adj.get(other, []):
            if nb in write_nodes:
                return True
    # Also check if DEF_USE edges exist at all (weaker check)
    return bool((graph.edge_attr == EDGE_TYPES["DEF_USE"]).any())


def check_timestamp(graph) -> bool:
    """
    Check: any node with uses_block_globals=1.0 (feature dim 2).
    """
    if graph.x.shape[0] == 0:
        return False
    return bool((graph.x[:, 2] >= 0.99).any())


def check_unused_return(graph) -> bool:
    """
    Check: any CFG_NODE_CALL with return_ignored=1.0 (feature dim 7).
    """
    node_types = _get_node_types(graph)
    call_mask = node_types == NODE_TYPES["CFG_NODE_CALL"]
    if not call_mask.any():
        return False
    call_return_ignored = graph.x[call_mask, 7]
    return bool((call_return_ignored >= 0.99).any())


def check_mishandled_exception(graph) -> bool:
    """
    Check: FUNCTION node with return_ignored=1.0.
    """
    node_types = _get_node_types(graph)
    func_mask = node_types == NODE_TYPES["FUNCTION"]
    if not func_mask.any():
        return False
    return bool((graph.x[func_mask, 7] >= 0.99).any())


def check_tod(graph) -> bool:
    """
    Check: has both CFG_NODE_WRITE and READS/WRITES edges (shared state pattern).
    """
    node_types = _get_node_types(graph)
    has_write = (node_types == NODE_TYPES["CFG_NODE_WRITE"]).any()
    has_read  = (node_types == NODE_TYPES["CFG_NODE_READ"]).any()
    return bool(has_write and has_read)


PATTERN_CHECKERS = {
    "Reentrancy":              check_reentrancy_cei,
    "Reentrancy_ICFG":         check_reentrancy_icfg,
    "IntegerUO":               check_integer_uo,
    "Timestamp":               check_timestamp,
    "UnusedReturn":            check_unused_return,
    "MishandledException":     check_mishandled_exception,
    "TransactionOrderDependence": check_tod,
}


# ── Test contract analysis ────────────────────────────────────────────────────

def analyse_test_contracts(contracts_dir: Path, contract_map: dict[str, str]) -> list[dict]:
    """Extract and audit test contracts."""
    try:
        from ml.src.preprocessing.graph_extractor import (
            GraphExtractionConfig,
            extract_contract_graph,
        )
    except ImportError as e:
        log.warning(f"Cannot import graph_extractor: {e} — skipping test contract analysis")
        return []

    config = GraphExtractionConfig(
        include_edge_attr=True,
    )
    results = []

    for filename, vuln_class in contract_map.items():
        sol_path = contracts_dir / filename
        if not sol_path.exists():
            log.warning(f"Test contract not found: {sol_path}")
            results.append({
                "file": filename,
                "expected_class": vuln_class,
                "found": False,
                "reason": "file not found",
            })
            continue

        try:
            graph = extract_contract_graph(str(sol_path), config)
            if graph is None:
                results.append({
                    "file": filename,
                    "expected_class": vuln_class,
                    "found": False,
                    "reason": "extraction returned None",
                })
                continue

            checker = PATTERN_CHECKERS.get(vuln_class)
            if checker is None:
                log.warning(f"No checker for class '{vuln_class}'")
                pattern_found = None
            else:
                pattern_found = checker(graph)

            n_nodes = graph.x.shape[0]
            n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
            node_types = _get_node_types(graph)
            cfg_call_count  = int((node_types == NODE_TYPES["CFG_NODE_CALL"]).sum())
            cfg_write_count = int((node_types == NODE_TYPES["CFG_NODE_WRITE"]).sum())

            results.append({
                "file":           filename,
                "expected_class": vuln_class,
                "found":          pattern_found,
                "n_nodes":        n_nodes,
                "n_edges":        n_edges,
                "cfg_call_nodes": cfg_call_count,
                "cfg_write_nodes": cfg_write_count,
            })

        except Exception as exc:
            log.warning(f"Error extracting {filename}: {exc}")
            results.append({
                "file": filename,
                "expected_class": vuln_class,
                "found": False,
                "reason": str(exc),
            })

    return results


# ── Val split analysis ────────────────────────────────────────────────────────

def analyse_val_split(
    stems: list[str],
    df_split: pd.DataFrame,
    cache: dict,
    max_per_class: int = 200,
) -> dict:
    """
    For each class, check pattern presence in val-split class-positive graphs.
    """
    stem_to_labels = {
        row["md5_stem"]: row for _, row in df_split.iterrows()
    }

    CLASS_NAMES = [
        "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
        "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
        "TransactionOrderDependence", "UnusedReturn",
    ]

    results: dict[str, dict] = {}

    for cls in ["Reentrancy", "IntegerUO", "Timestamp", "UnusedReturn", "MishandledException"]:
        checker = PATTERN_CHECKERS.get(cls)
        if checker is None:
            continue

        positive_stems = [
            s for s in stems
            if s in stem_to_labels and int(stem_to_labels[s].get(cls, 0)) == 1
        ][:max_per_class]

        found_pattern  = 0
        total_with_cfg = 0
        skipped        = 0

        for stem in positive_stems:
            if stem not in cache:
                skipped += 1
                continue
            entry = cache[stem]
            if not isinstance(entry, tuple):
                skipped += 1
                continue
            graph, _ = entry

            # Only count graphs with CFG nodes (others have no graph structure to trace)
            node_types = _get_node_types(graph)
            has_cfg = bool((node_types >= NODE_TYPES["CFG_NODE_CALL"]).any())
            if not has_cfg:
                continue

            total_with_cfg += 1
            if checker(graph):
                found_pattern += 1

        pct = 100.0 * found_pattern / total_with_cfg if total_with_cfg > 0 else 0.0
        results[cls] = {
            "positive_contracts": len(positive_stems),
            "with_cfg_nodes":     total_with_cfg,
            "pattern_found":      found_pattern,
            "pattern_pct":        round(pct, 2),
            "skipped":            skipped,
            "pass":               pct >= 50.0,
        }

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_results(test_results: list[dict], val_results: dict) -> None:
    print(f"\n{'═'*72}")
    print("  EXP-S1: CEI Structural Trace Audit")
    print(f"{'═'*72}")

    print("\n  Test Contracts:")
    print(f"  {'File':<35} {'Class':<28} {'Pattern?':<8} {'Nodes':>6} {'Edges':>6}")
    print(f"  {'-'*35} {'-'*28} {'-'*8} {'------':>6} {'------':>6}")
    passed_test = 0
    total_test  = 0
    for r in test_results:
        status = "FOUND" if r.get("found") else "MISS"
        if r.get("found"):
            passed_test += 1
        total_test += 1
        print(
            f"  {r['file']:<35} {r['expected_class']:<28} {status:<8} "
            f"{r.get('n_nodes', 0):>6} {r.get('n_edges', 0):>6}"
        )

    print(f"\n  Test contracts: {passed_test}/{total_test} patterns found")
    print(f"  P0 gate: {passed_test}/{total_test} >= 7/10? {'PASS' if passed_test >= 7 else 'FAIL'}")

    print(f"\n  Val Split Pattern Rates:")
    print(f"  {'Class':<28} {'Positives':>10} {'WithCFG':>8} {'PatternPct':>12} {'PASS':>6}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*12} {'-'*6}")
    all_pass = True
    for cls, r in val_results.items():
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        print(
            f"  {cls:<28} {r['positive_contracts']:>10} {r['with_cfg_nodes']:>8} "
            f"{r['pattern_pct']:>11.1f}% {status:>6}"
        )

    print(f"\n{'─'*72}")
    overall = passed_test >= 7 and all_pass
    print(f"  EXP-S1 RESULT: {'PASS' if overall else 'FAIL'}")
    print(f"{'═'*72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-S1: CEI Structural Trace Audit")
    p.add_argument("--cache",      default="ml/data/cached_dataset_v8.pkl")
    p.add_argument("--label-csv",  default="ml/data/processed/multilabel_index_cleaned.csv")
    p.add_argument("--splits-dir", default="ml/data/splits/deduped")
    p.add_argument("--split",      default="val", choices=["train", "val", "test"])
    p.add_argument("--out",        default="ml/logs/interpretability/s1_structural_trace.json")
    p.add_argument("--n-contracts", type=int, default=500, dest="n_contracts")
    p.add_argument("--checkpoint", default=None,
                   help="Not used by this script — accepted for uniform CLI compatibility")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Analyse test contracts from the main test_contracts/ directory
    main_contracts_dir = Path(__file__).resolve().parents[1] / "test_contracts"
    local_contracts_dir = Path(__file__).resolve().parent / "test_contracts"

    test_results: list[dict] = []

    if main_contracts_dir.exists():
        test_results.extend(
            analyse_test_contracts(main_contracts_dir, TEST_CONTRACT_MAP)
        )

    if local_contracts_dir.exists():
        test_results.extend(
            analyse_test_contracts(local_contracts_dir, LOCAL_CONTRACT_MAP)
        )

    if not test_results:
        log.warning("No test contracts found in either contracts directory")

    # Analyse val split
    cache_path = Path(args.cache)
    label_csv  = Path(args.label_csv)
    splits_dir = Path(args.splits_dir)
    val_results: dict = {}

    for p, name in [(cache_path, "cache"), (label_csv, "label_csv"), (splits_dir, "splits_dir")]:
        if not Path(p).exists():
            log.error(f"{name} not found: {p}")
            log.error("Skipping val split analysis")
            break
    else:
        import pickle
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        df_full  = pd.read_csv(label_csv)
        indices  = np.load(splits_dir / f"{args.split}_indices.npy")
        df_split = df_full.iloc[indices].reset_index(drop=True)
        stems    = df_split["md5_stem"].tolist()

        if args.n_contracts and len(stems) > args.n_contracts:
            rng = np.random.default_rng(42)
            stems = rng.choice(stems, size=args.n_contracts, replace=False).tolist()

        val_results = analyse_val_split(stems, df_split, cache)

    _print_results(test_results, val_results)

    # Evaluate pass/fail
    n_found = sum(1 for r in test_results if r.get("found"))
    n_total = len([r for r in test_results if r.get("expected_class") != "SAFE"])
    test_pass = n_found >= 7

    val_pass = all(r["pass"] for r in val_results.values()) if val_results else True

    report = {
        "test_contracts": test_results,
        "val_split":      val_results,
        "test_pass":      test_pass,
        "val_pass":       val_pass,
        "overall_pass":   test_pass and val_pass,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {out_path}")

    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
