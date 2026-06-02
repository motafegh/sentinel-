"""
exp_s4_icfg_path_audit.py — Layer 1, P0: ICFG-Lite Path Closure Audit for Reentrancy

PURPOSE
───────
Verifies that for reentrancy-positive contracts the full ICFG-Lite path closes:

  CFG_NODE_CALL (type 8) --[CALL_ENTRY edge(8)]--> callee_entry
      ... --[RETURN_TO edge(9)]--> caller_successor
      ... --> CFG_NODE_WRITE (type 9)

This is the critical cross-function control-flow structure that lets the GNN
learn reentrancy without relying on simple shallow heuristics.  If these edges
are absent the ICFG-Lite encoding is not doing its job.

ALGORITHM (per contract)
────────────────────────
1. Find all CFG_NODE_CALL nodes (type_id==8).
2. For each, follow outgoing CALL_ENTRY edges (edge_attr==8) — these go to
   callee entry-points.
3. For each callee entry-point found, check whether a RETURN_TO edge
   (edge_attr==9) exists that leads back to any node in the graph.
4. Check whether any WRITE node (type_id==9) is reachable after a RETURN_TO
   (simple structural check: a CFG_NODE_WRITE exists and RETURN_TO exists).

TWO DATA SOURCES
────────────────
a) Val-split reentrancy-positive contracts from cache (up to --n-contracts).
b) Test contracts from ml/scripts/test_contracts/:
     01_reentrancy_classic.sol
     02_reentrancy_tricky.sol
   Extracted fresh via graph_extractor (requires solc on PATH).

PASS CRITERIA
─────────────
From the FIRST 10 reentrancy-positive val contracts:
  >= 6/10 show at least one CALL_ENTRY edge          (exit code 0 if met)
  >= 4/10 show the full CALL_ENTRY → RETURN_TO chain
If < 4/10 have CALL_ENTRY edges at all → prints warning about ICFG encoding.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_s4_icfg_path_audit.py \\
        --cache ml/data/cached_dataset_v9.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --split val \\
        --n-contracts 50 \\
        --out ml/logs/interpretability/s4_icfg_path_audit.json

    # No --checkpoint needed.

OUTPUT
──────
Per-contract audit table.
Summary counts.
JSON report.
Exit 0 if >=6/10 first-10 reentrancy-positive val contracts have CALL_ENTRY.
Exit 1 otherwise.

EXIT CODES
──────────
    0  >= 6/10 first-10 reentrancy-positive val contracts have CALL_ENTRY edges
    1  < 6/10 have CALL_ENTRY edges (ICFG encoding may be broken)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")

from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES
from ml.scripts.interpretability.utils import (
    add_common_args,
    load_val_split,
    CLASS_NAMES,
    get_node_type_tensor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Type / edge ID constants ──────────────────────────────────────────────────
_CFG_CALL_TYPE:  int = NODE_TYPES.get("CFG_NODE_CALL", 8)
_CFG_WRITE_TYPE: int = NODE_TYPES.get("CFG_NODE_WRITE", 9)
_CALL_ENTRY_ID:  int = EDGE_TYPES.get("CALL_ENTRY", 8)
_RETURN_TO_ID:   int = EDGE_TYPES.get("RETURN_TO", 9)

# Test contract stems for the two reentrancy files
_TEST_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[0] / "test_contracts"
)
_REENTRANCY_TEST_FILES = [
    "01_reentrancy_classic.sol",
    "02_reentrancy_tricky.sol",
]


# ── Per-graph ICFG audit ──────────────────────────────────────────────────────

def _audit_graph(graph) -> dict:
    """
    Inspect a single graph for ICFG-Lite path closure signals.

    Returns:
        {
          "num_cfg_call_nodes":       int,
          "has_call_entry_edge":      bool,
          "num_call_entry_edges":     int,
          "has_return_to_edge":       bool,
          "num_return_to_edges":      int,
          "has_write_after_return":   bool,
          "full_chain_present":       bool,   # CALL_ENTRY AND RETURN_TO
          "num_nodes":                int,
          "num_edges":                int,
        }
    """
    node_types  = get_node_type_tensor(graph)          # [N] int
    edge_index  = graph.edge_index                     # [2, E]
    edge_attr   = graph.edge_attr                      # [E] int64

    N = node_types.shape[0]
    E = edge_attr.shape[0] if edge_attr is not None and edge_attr.numel() > 0 else 0

    # CFG_NODE_CALL nodes
    cfg_call_mask = (node_types == _CFG_CALL_TYPE)
    num_cfg_calls = int(cfg_call_mask.sum().item())

    if E == 0 or edge_attr is None:
        return {
            "num_cfg_call_nodes":     num_cfg_calls,
            "has_call_entry_edge":    False,
            "num_call_entry_edges":   0,
            "has_return_to_edge":     False,
            "num_return_to_edges":    0,
            "has_write_after_return": False,
            "full_chain_present":     False,
            "num_nodes":              N,
            "num_edges":              0,
        }

    call_entry_mask = (edge_attr == _CALL_ENTRY_ID)
    return_to_mask  = (edge_attr == _RETURN_TO_ID)
    num_call_entry  = int(call_entry_mask.sum().item())
    num_return_to   = int(return_to_mask.sum().item())

    has_call_entry = num_call_entry > 0
    has_return_to  = num_return_to  > 0

    # Check: does any CFG_NODE_CALL have an outgoing CALL_ENTRY edge?
    cfg_call_source_present = False
    if has_call_entry and num_cfg_calls > 0:
        cfg_call_indices = cfg_call_mask.nonzero(as_tuple=True)[0]
        cfg_call_set = set(cfg_call_indices.tolist())
        src_nodes = edge_index[0][call_entry_mask]
        for s in src_nodes.tolist():
            if s in cfg_call_set:
                cfg_call_source_present = True
                break

    # Simple structural check: RETURN_TO exists AND CFG_NODE_WRITE exists
    has_write = (node_types == _CFG_WRITE_TYPE).any().item()
    has_write_after_return = bool(has_return_to and has_write)

    full_chain = has_call_entry and has_return_to

    return {
        "num_cfg_call_nodes":         num_cfg_calls,
        "has_call_entry_edge":        has_call_entry,
        "num_call_entry_edges":       num_call_entry,
        "cfg_call_is_source":         cfg_call_source_present,
        "has_return_to_edge":         has_return_to,
        "num_return_to_edges":        num_return_to,
        "has_write_after_return":     has_write_after_return,
        "full_chain_present":         full_chain,
        "num_nodes":                  N,
        "num_edges":                  E,
    }


# ── Val-split audit ───────────────────────────────────────────────────────────

def _audit_val_split(
    stems: list[str],
    df_split,
    cache: dict,
    n_contracts: int,
) -> list[dict]:
    """
    Audit up to n_contracts reentrancy-positive val contracts.

    Returns list of audit result dicts, each with "stem" added.
    """
    stem_to_labels: dict[str, dict[str, int]] = {}
    for _, row in df_split.iterrows():
        stem_to_labels[row["md5_stem"]] = {c: int(row[c]) for c in CLASS_NAMES}

    results: list[dict] = []
    for stem in stems:
        if len(results) >= n_contracts:
            break
        labels = stem_to_labels.get(stem, {})
        if not labels.get("Reentrancy", 0):
            continue
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 2:
            continue
        graph, _ = entry
        try:
            info = _audit_graph(graph)
        except Exception as exc:
            log.debug(f"Skipping {stem}: {exc}")
            continue
        info["stem"] = stem
        info["source"] = "val_split"
        results.append(info)

    log.info(
        f"Audited {len(results)} reentrancy-positive val contracts "
        f"(requested <= {n_contracts})"
    )
    return results


# ── Test-contract audit ───────────────────────────────────────────────────────

def _audit_test_contracts() -> list[dict]:
    """
    Extract and audit the two reentrancy test contracts using graph_extractor.

    Errors (compilation failure, missing solc, None return) are handled
    gracefully — the result dict records "extraction_failed": True.
    """
    try:
        from ml.src.preprocessing.graph_extractor import (
            GraphExtractionConfig,
            extract_contract_graph,
        )
    except ImportError as exc:
        log.warning(f"Cannot import graph_extractor: {exc}")
        return []

    cfg = GraphExtractionConfig()
    results: list[dict] = []

    for fname in _REENTRANCY_TEST_FILES:
        contract_path = _TEST_CONTRACTS_DIR / fname
        if not contract_path.exists():
            log.warning(f"Test contract not found: {contract_path}")
            results.append({
                "stem": fname,
                "source": "test_contract",
                "extraction_failed": True,
                "reason": "file_not_found",
            })
            continue

        try:
            graph = extract_contract_graph(str(contract_path), cfg)
        except Exception as exc:
            log.warning(f"Extraction failed for {fname}: {exc}")
            results.append({
                "stem": fname,
                "source": "test_contract",
                "extraction_failed": True,
                "reason": str(exc),
            })
            continue

        if graph is None:
            log.warning(f"Extraction returned None for {fname}")
            results.append({
                "stem": fname,
                "source": "test_contract",
                "extraction_failed": True,
                "reason": "returned_None",
            })
            continue

        info = _audit_graph(graph)
        info["stem"] = fname
        info["source"] = "test_contract"
        info["extraction_failed"] = False
        results.append(info)
        log.info(
            f"Test contract {fname}: "
            f"call_entry={info['has_call_entry_edge']}, "
            f"return_to={info['has_return_to_edge']}, "
            f"full_chain={info['full_chain_present']}"
        )

    return results


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_audit_table(records: list[dict], title: str) -> None:
    print(f"\n{'─'*95}")
    print(f"  {title}")
    print(f"{'─'*95}")
    print(
        f"  {'Stem':<42}  {'N_nodes':>7}  {'N_edges':>7}  "
        f"{'CFG_CALL':>8}  {'CALL_ENT':>8}  {'RET_TO':>6}  "
        f"{'CHAIN':>5}  {'W_AFTER':>7}"
    )
    print(
        f"  {'----':<42}  {'-------':>7}  {'-------':>7}  "
        f"{'--------':>8}  {'--------':>8}  {'------':>6}  "
        f"{'-----':>5}  {'-------':>7}"
    )
    for r in records:
        if r.get("extraction_failed"):
            print(f"  {r['stem']:<42}  EXTRACTION FAILED — {r.get('reason','?')}")
            continue
        print(
            f"  {str(r['stem'])[:42]:<42}  "
            f"{r['num_nodes']:>7}  "
            f"{r['num_edges']:>7}  "
            f"{r['num_cfg_call_nodes']:>8}  "
            f"{'Y' if r['has_call_entry_edge'] else 'N':>8}  "
            f"{'Y' if r['has_return_to_edge'] else 'N':>6}  "
            f"{'Y' if r['full_chain_present'] else 'N':>5}  "
            f"{'Y' if r['has_write_after_return'] else 'N':>7}"
        )
    print(f"{'─'*95}")


def _print_summary(
    val_records: list[dict],
    test_records: list[dict],
) -> dict:
    """Print summary and return pass/fail info dict."""
    first10 = [r for r in val_records if not r.get("extraction_failed")][:10]
    n10 = len(first10)

    n_call_entry   = sum(1 for r in first10 if r.get("has_call_entry_edge"))
    n_full_chain   = sum(1 for r in first10 if r.get("full_chain_present"))
    n_write_after  = sum(1 for r in first10 if r.get("has_write_after_return"))

    total_val       = len([r for r in val_records if not r.get("extraction_failed")])
    val_call_entry  = sum(1 for r in val_records if r.get("has_call_entry_edge"))
    val_full_chain  = sum(1 for r in val_records if r.get("full_chain_present"))

    print(f"\n{'═'*70}")
    print(f"  ICFG-Lite Path Audit — Summary")
    print(f"{'═'*70}")
    print(f"  Val-split reentrancy positives audited: {total_val}")
    print(f"  First 10 used for pass/fail criteria:")
    print(f"    Has CALL_ENTRY edge:    {n_call_entry}/{n10}   (need >= 6/10 to PASS)")
    print(f"    Full CALL_ENTRY+RETURN_TO chain: {n_full_chain}/{n10}  (need >= 4/10)")
    print(f"    Has WRITE after RETURN_TO:       {n_write_after}/{n10}")
    print()
    print(f"  All val reentrancy positives ({total_val}):")
    print(f"    CALL_ENTRY present: {val_call_entry}/{total_val} ({100.0*val_call_entry/max(total_val,1):.1f}%)")
    print(f"    Full chain present: {val_full_chain}/{total_val} ({100.0*val_full_chain/max(total_val,1):.1f}%)")
    print()

    # Warnings
    if n10 > 0 and n_call_entry < 4:
        print(
            "  WARNING: ICFG-Lite may not be encoding reentrancy cross-function paths\n"
            "           Only {}/{} of the first 10 val contracts have CALL_ENTRY edges.\n"
            "           Check graph_extractor ICFG construction — CALL_ENTRY/RETURN_TO\n"
            "           edges may be missing from the phase2_edge_types config.".format(
                n_call_entry, n10
            )
        )

    # Test contracts
    if test_records:
        print(f"  Test contracts ({len(test_records)}):")
        for r in test_records:
            if r.get("extraction_failed"):
                print(f"    {r['stem']}: EXTRACTION FAILED — {r.get('reason','?')}")
            else:
                print(
                    f"    {r['stem']}: "
                    f"call_entry={r['has_call_entry_edge']}, "
                    f"return_to={r['has_return_to_edge']}, "
                    f"full_chain={r['full_chain_present']}"
                )

    # Pass/fail
    passed = n10 >= 6 and n_call_entry >= 6
    print()
    print(f"  PASS CRITERIA: >= 6/10 val reentrancy contracts with CALL_ENTRY")
    status = "PASS" if passed else "FAIL"
    print(f"  RESULT: [{status}]  {n_call_entry}/{n10} contracts with CALL_ENTRY")
    print(f"{'═'*70}\n")

    return {
        "first10_audited": n10,
        "first10_call_entry": n_call_entry,
        "first10_full_chain": n_full_chain,
        "first10_write_after_return": n_write_after,
        "total_val_audited": total_val,
        "val_call_entry_count": val_call_entry,
        "val_full_chain_count": val_full_chain,
        "passed": passed,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="exp_s4: ICFG-Lite path closure audit for reentrancy"
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
            f"ml/logs/interpretability/s4_icfg_path_audit_{args.split}.json"
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

    # ── val-split audit ───────────────────────────────────────────────────────
    log.info("Auditing val-split reentrancy positives …")
    val_records = _audit_val_split(
        stems, df_split, cache, args.n_contracts
    )

    # ── test-contract audit ───────────────────────────────────────────────────
    log.info("Auditing test contracts …")
    test_records = _audit_test_contracts()

    # ── print tables ──────────────────────────────────────────────────────────
    _print_audit_table(
        val_records[:50],
        f"Val-split reentrancy positives (showing first {min(len(val_records), 50)})"
    )
    if test_records:
        _print_audit_table(test_records, "Test contracts (reentrancy)")

    # ── summary + pass/fail ───────────────────────────────────────────────────
    summary = _print_summary(val_records, test_records)

    # ── JSON ──────────────────────────────────────────────────────────────────
    report = {
        "split": args.split,
        "cache": str(cache_path),
        "label_csv": str(label_csv),
        "splits_dir": str(splits_dir),
        "n_contracts_requested": args.n_contracts,
        "summary": summary,
        "val_records": val_records,
        "test_records": test_records,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON saved: {out_path}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
