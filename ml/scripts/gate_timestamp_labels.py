"""
gate_timestamp_labels.py — Sol-3: Timestamp CFG-path gating filter

PURPOSE
───────
Audits Timestamp=1 labels and removes labels for contracts where block.timestamp
is NOT in a branch condition that gates an external call or ETH transfer.

The dangerous pattern this requires:
    [CFG_NODE_CHECK with uses_block_globals=1] →(CONTROL_FLOW)→ [CFG_NODE_CALL]

Contracts where timestamp only appears in assignments or event emissions
(without the gating check) get their Timestamp label removed.

⚠ RELIABILITY: MEDIUM — READ BEFORE RUNNING ⚠
──────────────────────────────────────────────
Known gaps:

1. INDIRECT TIMESTAMP PROPAGATION MISSED. `uses_block_globals` is set at the
   function level. A CFG_NODE_CHECK that reads a STATE_VAR which was *set by*
   block.timestamp in a prior transaction will NOT have uses_block_globals=1
   on that check node. Those contracts will be incorrectly removed.

2. BLOCK.NUMBER AS TIME PROXY. block.number used as a time proxy is also
   flagged by uses_block_globals, but the vulnerability semantics differ.
   This script does not distinguish the two.

RECOMMENDED USAGE
─────────────────
1. Run with --dry-run first.
2. Sample 20 stems from the removal list, inspect their source for
   block.timestamp usage context.
3. If error rate > 10%, do NOT run the script.

EXPECTED REMOVAL: ~100–200 Timestamp=1 labels (if validation passes)

USAGE
─────
    source ml/.venv/bin/activate
    python -m ml.scripts.gate_timestamp_labels \
        --cache ml/data/cached_dataset_v8.pkl \
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \
        --out ml/data/processed/multilabel_index_sol3.csv \
        --dry-run          # always run this first

OUTPUT
──────
    CSV: updated multilabel_index with Timestamp labels removed for non-gated contracts
    JSON: audit log — which contracts were gated and why
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.src.preprocessing.graph_schema import EDGE_TYPES, NODE_TYPES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_TYPE_ID  = float(max(NODE_TYPES.values()))  # 12.0
_CHECK_NORM   = NODE_TYPES["CFG_NODE_CHECK"] / _MAX_TYPE_ID   # 11/12 = 0.9167
_CALL_NORM    = NODE_TYPES["CFG_NODE_CALL"]  / _MAX_TYPE_ID   # 8/12  = 0.6667
_EDGE_CF      = EDGE_TYPES["CONTROL_FLOW"]    # 6
_EDGE_CONTAINS = EDGE_TYPES["CONTAINS"]       # 5

# BFS depth limit — keeps runtime bounded for large contracts
_MAX_BFS_DEPTH = 30


# ── CFG-path check ─────────────────────────────────────────────────────────────

def check_timestamp_gated_path(data) -> bool:
    """
    Returns True if block.timestamp appears in a CFG CHECK node with a
    CONTROL_FLOW path to a CFG CALL node (dangerous pattern: timestamp gates a call).
    Returns False otherwise (→ label candidate for removal).

    Feature layout (v8 schema):
      x[:, 0] = type_id_norm = NODE_TYPES[kind] / 12.0
      x[:, 2] = uses_block_globals (1.0 if reads block.timestamp/number/etc)
      EDGE_TYPES["CONTROL_FLOW"] = 6

    KNOWN LIMITATIONS:
    - `uses_block_globals` is a function-level feature; individual CFG_NODE_CHECK
      nodes do NOT have it set per-node. A check that reads a STATE_VAR which
      was previously assigned from block.timestamp will not be detected.
    - Returns False (→ label removed) when no CFG_NODE_CHECK with
      uses_block_globals=1 exists, even if the contract is genuinely vulnerable
      through an indirect path.
    """
    if data.edge_index.size(1) == 0 or data.edge_attr is None:
        return False

    x  = data.x          # [N, 11]
    ei = data.edge_index  # [2, E]
    ea = data.edge_attr   # [E] or [E, 1]
    if ea.dim() > 1:
        ea = ea.squeeze(-1)

    type_col    = x[:, 0]
    globals_col = x[:, 2]

    # CHECK nodes that also read block globals
    check_with_ts = [
        i for i in range(x.shape[0])
        if abs(type_col[i].item() - _CHECK_NORM) < 0.01
        and globals_col[i].item() > 0.5
    ]
    if not check_with_ts:
        return False

    # Build CONTROL_FLOW adjacency list
    cf_mask = (ea == _EDGE_CF)
    cf_src  = ei[0, cf_mask].tolist()
    cf_dst  = ei[1, cf_mask].tolist()
    adj: dict[int, list[int]] = {}
    for s, d in zip(cf_src, cf_dst):
        adj.setdefault(int(s), []).append(int(d))

    # CALL nodes
    call_nodes = set(
        i for i in range(x.shape[0])
        if abs(type_col[i].item() - _CALL_NORM) < 0.01
    )
    if not call_nodes:
        return False

    # BFS from each timestamp-gated CHECK node; hit a CALL → True
    for check_node in check_with_ts:
        visited: set[int] = set()
        queue: list[tuple[int, int]] = [(int(check_node), 0)]
        while queue:
            curr, depth = queue.pop()
            if curr in visited or depth > _MAX_BFS_DEPTH:
                continue
            visited.add(curr)
            if curr in call_nodes:
                return True
            for nxt in adj.get(curr, []):
                queue.append((nxt, depth + 1))

    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sol-3: Timestamp CFG-path gating filter")
    p.add_argument("--cache", default="ml/data/cached_dataset_v8.pkl",
                   help="Path to cached_dataset_v8.pkl")
    p.add_argument("--label-csv", default="ml/data/processed/multilabel_index_cleaned.csv",
                   help="Input label CSV")
    p.add_argument("--out", default="ml/data/processed/multilabel_index_sol3.csv",
                   help="Output label CSV (Timestamp labels updated)")
    p.add_argument("--audit-json", default=None,
                   help="Path to write audit log JSON (default: <out>.audit.json)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print counts without writing output")
    p.add_argument("--max-contracts", type=int, default=None,
                   help="Limit to first N Timestamp-positive contracts (for testing)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    label_csv = Path(args.label_csv)
    cache_path = Path(args.cache)
    out_path   = Path(args.out)
    audit_path = Path(args.audit_json) if args.audit_json else out_path.with_suffix(".audit.json")

    if not label_csv.exists():
        log.error(f"Label CSV not found: {label_csv}")
        return 1
    if not cache_path.exists():
        log.error(f"Cache not found: {cache_path}")
        return 1

    log.info(f"Loading cache: {cache_path} ({cache_path.stat().st_size / 1e9:.2f} GB)...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    log.info(f"Cache loaded: {len(cache):,} entries")

    log.info(f"Loading labels: {label_csv}")
    df = pd.read_csv(label_csv)
    n_total = len(df)
    log.info(f"Total contracts: {n_total:,}")

    ts_positive = df[df["Timestamp"] == 1]["md5_stem"].tolist()
    log.info(f"Timestamp=1 contracts: {len(ts_positive):,}")

    if args.max_contracts:
        ts_positive = ts_positive[:args.max_contracts]
        log.info(f"Limited to first {args.max_contracts} for testing")

    # Audit each Timestamp+ contract
    n_checked  = 0
    n_in_cache = 0
    gated_out: list[str] = []     # contracts where label should be removed
    kept_in:   list[str] = []     # contracts where label is valid (genuine gated pattern)
    no_cache:  list[str] = []     # contracts not in cache (keep by default)

    for stem in ts_positive:
        n_checked += 1
        if n_checked % 100 == 0:
            log.info(f"  {n_checked}/{len(ts_positive)} checked, {len(gated_out)} gated out so far...")

        if stem not in cache:
            no_cache.append(stem)
            continue

        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 1:
            no_cache.append(stem)
            continue

        graph = entry[0]
        n_in_cache += 1

        has_gated_path = check_timestamp_gated_path(graph)
        if has_gated_path:
            kept_in.append(stem)
        else:
            gated_out.append(stem)

    log.info(f"\n=== Sol-3 Timestamp gating results ===")
    log.info(f"Timestamp+ contracts checked: {len(ts_positive):,}")
    log.info(f"  Found in cache:             {n_in_cache:,}")
    log.info(f"  Not in cache (kept):        {len(no_cache):,}")
    log.info(f"  Valid gated pattern (kept): {len(kept_in):,}")
    log.info(f"  No gated path (remove):     {len(gated_out):,}")

    # Build audit log
    audit = {
        "n_timestamp_positive":  len(ts_positive),
        "n_in_cache":            n_in_cache,
        "n_no_cache_kept":       len(no_cache),
        "n_valid_kept":          len(kept_in),
        "n_gated_removed":       len(gated_out),
        "gated_removed_stems":   gated_out,
        "valid_kept_stems":      kept_in[:100],  # sample only
        "no_cache_stems":        no_cache[:100],
    }

    if args.dry_run:
        log.info("DRY RUN — no files written")
        log.info(f"Would remove {len(gated_out)} Timestamp labels")
        log.info(f"Would write audit → {audit_path}")
        log.info(f"Would write updated CSV → {out_path}")
        return 0

    # Write audit JSON
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    log.info(f"Audit log → {audit_path}")

    # Update label CSV
    df_out = df.copy()
    gated_set = set(gated_out)
    mask = df_out["md5_stem"].isin(gated_set)
    df_out.loc[mask, "Timestamp"] = 0
    log.info(f"Updated {mask.sum()} rows: Timestamp 1→0")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    log.info(f"Updated CSV → {out_path} ({len(df_out):,} rows)")

    # Verify
    ts_after = int(df_out["Timestamp"].sum())
    ts_before = int(df["Timestamp"].sum())
    log.info(f"Timestamp labels: {ts_before} → {ts_after} (Δ={ts_after-ts_before})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
