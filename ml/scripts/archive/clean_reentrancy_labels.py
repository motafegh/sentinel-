"""
clean_reentrancy_labels.py — Sol-1: CEI-order Reentrancy label cleaning

PURPOSE
───────
Removes Reentrancy=1 labels from contracts where the CFG shows safe CEI order
(Checks → Effects → Interactions). These contracts have external calls AND state
writes but in a safe order — the exploitable pattern requires a write AFTER a
call in the control flow graph.

⚠ RELIABILITY: LOW — READ BEFORE RUNNING ⚠
──────────────────────────────────────────
This script is a heuristic approximation with known gaps:

1. MISSES CROSS-FUNCTION REENTRANCY. The BFS only traverses CONTROL_FLOW edges
   within a single function's CFG. It cannot see patterns where the attacker
   re-enters via a *different* function (e.g. fallback calls a state-writing
   helper). Those contracts will incorrectly have their label removed.

2. CFG_NODE_WRITE COVERAGE UNCERTAIN. Whether all state writes become
   CFG_NODE_WRITE nodes depends on the graph extractor's IR handling. Contracts
   whose writes are not captured will appear CEI-safe when they are not.

3. DOES NOT REPLACE SLITHER. Slither's built-in `reentrancy-eth` detector
   handles cross-function taint, all call types, and read-only reentrancy.
   This script does not.

RECOMMENDED USAGE
─────────────────
1. Run with --dry-run first to see what would be removed.
2. Sample 20 stems from the removal list.
3. Cross-check each against: slither <file> --detect reentrancy-eth
4. If error rate > 10% (>2 of 20 are genuinely vulnerable), DO NOT run the
   script. Instead, skip Sol-1 and rely on E1 (CEI aux loss in trainer.py)
   to force the model to learn the correct CFG pattern from data directly.

EXPECTED REMOVAL: ~200–400 Reentrancy=1 labels (if validation passes)

USAGE
─────
    source ml/.venv/bin/activate
    python -m ml.scripts.clean_reentrancy_labels \
        --cache ml/data/cached_dataset_v8.pkl \
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \
        --out ml/data/processed/multilabel_index_sol1.csv \
        --dry-run          # always run this first

OUTPUT
──────
    CSV: updated multilabel_index with Reentrancy labels removed for CFG-CEI-safe contracts
    JSON: audit log with removed stems (review this before committing)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.src.preprocessing.graph_schema import EDGE_TYPES, NODE_TYPES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_MAX_TYPE_ID = float(max(NODE_TYPES.values()))  # 12.0
_CALL_NORM   = NODE_TYPES["CFG_NODE_CALL"]  / _MAX_TYPE_ID   # 8/12
_WRITE_NORM  = NODE_TYPES["CFG_NODE_WRITE"] / _MAX_TYPE_ID   # 9/12
_EDGE_CF     = EDGE_TYPES["CONTROL_FLOW"]    # 6
_MAX_BFS_DEPTH = 30


def check_reentrancy_cei_order(data) -> bool:
    """
    Returns True if any CFG execution path has a state WRITE reachable AFTER
    an external CALL via CONTROL_FLOW edges (CEI violation → label is plausible).
    Returns False if no such path exists (→ label candidate for removal).

    Uses CONTROL_FLOW edges (type 6) and node types:
      CFG_NODE_CALL  (type 8 → 8/12 = 0.6667)
      CFG_NODE_WRITE (type 9 → 9/12 = 0.75)

    KNOWN LIMITATIONS:
    - Only traverses CONTROL_FLOW within a single function. Cross-function
      reentrancy (CALL in fn A → reenter fn B which writes state) is invisible.
    - Returns False (→ label removed) when CFG_NODE_CALL or CFG_NODE_WRITE
      counts are zero — which may indicate extractor coverage gaps, not safety.
    - BFS depth capped at _MAX_BFS_DEPTH=30; very long CFG chains may be missed.
    """
    if data.edge_index.size(1) == 0 or data.edge_attr is None:
        return False

    x  = data.x
    ei = data.edge_index
    ea = data.edge_attr
    if ea.dim() > 1:
        ea = ea.squeeze(-1)

    type_col = x[:, 0]

    # CALL nodes (external calls)
    call_nodes = set(
        i for i in range(x.shape[0])
        if abs(type_col[i].item() - _CALL_NORM) < 0.01
    )
    # WRITE nodes (state writes)
    write_nodes = set(
        i for i in range(x.shape[0])
        if abs(type_col[i].item() - _WRITE_NORM) < 0.01
    )

    if not call_nodes or not write_nodes:
        return False

    # Build CONTROL_FLOW adjacency
    cf_mask = (ea == _EDGE_CF)
    cf_src  = ei[0, cf_mask].tolist()
    cf_dst  = ei[1, cf_mask].tolist()
    adj: dict[int, list[int]] = {}
    for s, d in zip(cf_src, cf_dst):
        adj.setdefault(int(s), []).append(int(d))

    # For each CALL node, BFS to see if any WRITE node is reachable downstream
    for call_node in call_nodes:
        visited: set[int] = set()
        queue: list[tuple[int, int]] = [(int(call_node), 0)]
        while queue:
            curr, depth = queue.pop()
            if curr in visited or depth > _MAX_BFS_DEPTH:
                continue
            visited.add(curr)
            if curr in write_nodes:
                return True  # CEI violation: write after call
            for nxt in adj.get(curr, []):
                queue.append((nxt, depth + 1))

    return False  # all writes precede calls — safe CEI order


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sol-1: CEI-order Reentrancy label cleaning")
    p.add_argument("--cache", default="ml/data/cached_dataset_v8.pkl")
    p.add_argument("--label-csv", default="ml/data/processed/multilabel_index_cleaned.csv")
    p.add_argument("--out", default="ml/data/processed/multilabel_index_sol1.csv")
    p.add_argument("--audit-json", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-contracts", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    label_csv  = Path(args.label_csv)
    cache_path = Path(args.cache)
    out_path   = Path(args.out)
    audit_path = Path(args.audit_json) if args.audit_json else out_path.with_suffix(".audit.json")

    if not label_csv.exists():
        log.error(f"Label CSV not found: {label_csv}")
        return 1
    if not cache_path.exists():
        log.error(f"Cache not found: {cache_path}")
        return 1

    log.info(f"Loading cache: {cache_path}...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    log.info(f"Cache: {len(cache):,} entries")

    log.info(f"Loading labels: {label_csv}")
    df = pd.read_csv(label_csv)

    re_positive = df[df["Reentrancy"] == 1]["md5_stem"].tolist()
    log.info(f"Reentrancy=1 contracts: {len(re_positive):,}")

    if args.max_contracts:
        re_positive = re_positive[:args.max_contracts]

    cei_safe:   list[str] = []
    cei_unsafe: list[str] = []
    no_cache:   list[str] = []

    for i, stem in enumerate(re_positive):
        if i % 200 == 0:
            log.info(f"  {i}/{len(re_positive)} checked, {len(cei_safe)} to remove...")

        if stem not in cache:
            no_cache.append(stem)
            continue

        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 1:
            no_cache.append(stem)
            continue

        has_violation = check_reentrancy_cei_order(entry[0])
        if has_violation:
            cei_unsafe.append(stem)
        else:
            cei_safe.append(stem)

    log.info(f"\n=== Sol-1 Reentrancy CEI-order results ===")
    log.info(f"Reentrancy+ checked: {len(re_positive):,}")
    log.info(f"  CEI violation (keep): {len(cei_unsafe):,}")
    log.info(f"  CEI safe (remove):    {len(cei_safe):,}")
    log.info(f"  Not in cache (keep):  {len(no_cache):,}")

    audit = {
        "n_reentrancy_positive":  len(re_positive),
        "n_cei_violation_kept":   len(cei_unsafe),
        "n_cei_safe_removed":     len(cei_safe),
        "n_no_cache_kept":        len(no_cache),
        "cei_safe_stems":         cei_safe,
    }

    if args.dry_run:
        log.info("DRY RUN — no files written")
        return 0

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    log.info(f"Audit → {audit_path}")

    df_out = df.copy()
    mask = df_out["md5_stem"].isin(set(cei_safe))
    df_out.loc[mask, "Reentrancy"] = 0
    log.info(f"Updated {mask.sum()} rows: Reentrancy 1→0")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    log.info(f"Updated CSV → {out_path}")

    log.info(f"Reentrancy labels: {int(df['Reentrancy'].sum())} → {int(df_out['Reentrancy'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
