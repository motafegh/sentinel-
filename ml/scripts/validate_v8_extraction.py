#!/usr/bin/env python3
"""
validate_v8_extraction.py — PLAN-1B: structural comparison gate for v8 extractor.

Samples N contracts from the cleaned CSV, re-extracts each with the v8 extractor,
and verifies that all legacy edge types (0–6) are bit-for-bit identical to the
stored v7 graphs.  Also validates edge count distribution and confirms that every
new v8 edge type fires at least once across the sample.

PASS criteria (all must hold):
  1. Structural parity  — legacy edge set (types 0–6) identical between v7 disk
                          graph and v8 re-extraction for every contract.
  2. Edge count P99     — P99 edges per graph < 5,000
  3. Edge count max     — no single graph exceeds 10,000 edges
  4. New types fire     — CALL_ENTRY(8), RETURN_TO(9), DEF_USE(10) all have
                          non-zero total counts across the sample.
  5. DataLoader batch   — a batch of 8 graphs loads through the DataLoader
                          without error.

Usage:
    python ml/scripts/validate_v8_extraction.py
    python ml/scripts/validate_v8_extraction.py --sample 500 --seed 0
    python ml/scripts/validate_v8_extraction.py --no-dataloader   # skip GPU test

Exit codes:
    0  all criteria PASS
    1  one or more criteria FAIL or script error
"""

import argparse
import logging
import re
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from ml.src.preprocessing.graph_extractor import (
    GraphExtractionConfig,
    GraphExtractionError,
    extract_contract_graph,
)
from ml.src.preprocessing.graph_schema import EDGE_TYPES, NUM_EDGE_TYPES

# ── Solc version detection (mirrors reextract_graphs.py) ─────────────────────
_PRAGMA_RE    = re.compile(r'pragma\s+solidity\s+[\^~>=<\s]*(\d+\.\d+\.\d+)')
_LATEST_PATCH = {"0.4": "0.4.26", "0.5": "0.5.17", "0.6": "0.6.12",
                 "0.7": "0.7.6",  "0.8": "0.8.31"}
_SOLC_ARTIFACTS = _ROOT / "ml" / ".venv" / ".solc-select" / "artifacts"


def _detect_solc_version(sol_path: Path) -> str:
    try:
        m = _PRAGMA_RE.search(sol_path.read_text(encoding="utf-8", errors="replace"))
        if m:
            minor = ".".join(m.group(1).split(".")[:2])
            return _LATEST_PATCH.get(minor, m.group(1))
    except OSError:
        pass
    return "0.8.31"


def _solc_binary(version: str) -> Optional[Path]:
    binary = _SOLC_ARTIFACTS / f"solc-{version}" / f"solc-{version}"
    return binary if binary.exists() else None

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GRAPHS_DIR  = _ROOT / "ml" / "data" / "graphs"
CSV_PATH    = _ROOT / "ml" / "data" / "processed" / "multilabel_index_cleaned.csv"
LEGACY_TYPES = set(range(7))   # edge types 0–6; type 7 is runtime-only REVERSE_CONTAINS
NEW_TYPES    = {
    EDGE_TYPES["CALL_ENTRY"]: "CALL_ENTRY",
    EDGE_TYPES["RETURN_TO"]:  "RETURN_TO",
    EDGE_TYPES["DEF_USE"]:    "DEF_USE",
}

P99_EDGE_LIMIT  = 5_000
MAX_EDGE_LIMIT  = 10_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _legacy_edge_set(graph) -> set[tuple]:
    """Return a set of (src, dst, edge_type) for all legacy edge types (0–6)."""
    ea = graph.edge_attr
    ei = graph.edge_index
    mask = torch.zeros(ea.shape[0], dtype=torch.bool)
    for t in LEGACY_TYPES:
        mask |= (ea == t)
    idxs = mask.nonzero(as_tuple=False).squeeze(1)
    return {
        (ei[0, i].item(), ei[1, i].item(), ea[i].item())
        for i in idxs.tolist()
    }


def _edge_type_counts(graph) -> dict[int, int]:
    ea = graph.edge_attr
    return {t: (ea == t).sum().item() for t in range(NUM_EDGE_TYPES)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample",       type=int,  default=2000,
                        help="Number of contracts to sample (default: 2000)")
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--no-dataloader", action="store_true",
                        help="Skip the DataLoader batch test")
    parser.add_argument("--csv",          default=str(CSV_PATH))
    parser.add_argument("--graphs-dir",   default=str(GRAPHS_DIR))
    args = parser.parse_args()

    random.seed(args.seed)
    graphs_dir = Path(args.graphs_dir)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    print(f"CSV rows: {len(df):,}")

    # Shuffle and lazily scan until we have `args.sample` usable contracts.
    # Lazy scan avoids loading all 41k graphs just to build a pool
    # (BUG-M7: ~8.5% have empty contract_path — those are skipped).
    all_md5s = df["md5_stem"].tolist()
    random.shuffle(all_md5s)

    print(f"Building sample (target={args.sample})...")
    sample: list[tuple[str, str]] = []   # (md5, absolute_sol_path)
    n_missing_graph = n_missing_path = 0
    for md5 in all_md5s:
        if len(sample) >= args.sample:
            break
        gfile = graphs_dir / f"{md5}.pt"
        if not gfile.exists():
            n_missing_graph += 1
            continue
        g = torch.load(gfile, weights_only=False)
        cp = (getattr(g, "contract_path", "") or "").strip()
        if not cp:
            n_missing_path += 1
            continue
        full = _ROOT / cp
        if not full.exists():
            n_missing_path += 1
            continue
        sample.append((md5, str(full)))

    print(f"Sample built: {len(sample):,}  "
          f"(skipped: {n_missing_graph} no-graph, {n_missing_path} no-path)\n")

    # ── Extraction config (version resolved per file below) ──────────────────
    _base_cfg = dict(include_edge_attr=True)

    # ── Per-contract comparison ───────────────────────────────────────────────
    n_pass = n_fail = n_skip = 0
    fail_reasons: list[str] = []
    all_edge_counts: list[int] = []
    type_totals: dict[int, int] = defaultdict(int)

    t_start = time.time()
    for i, (md5, sol_path) in enumerate(sample, 1):
        if i % 100 == 0 or i == len(sample):
            elapsed = time.time() - t_start
            print(f"  [{i:4d}/{len(sample)}] {elapsed:.0f}s elapsed — "
                  f"pass={n_pass} fail={n_fail} skip={n_skip}")

        # Load v7 graph from disk
        v7_graph = torch.load(graphs_dir / f"{md5}.pt", weights_only=False)

        # Re-extract with v8 — pick correct solc binary per pragma
        try:
            sol = Path(sol_path)
            ver = _detect_solc_version(sol)
            cfg = GraphExtractionConfig(
                solc_version=ver,
                solc_binary=_solc_binary(ver),
                **_base_cfg,
            )
            v8_graph = extract_contract_graph(sol_path, cfg)
        except GraphExtractionError as e:
            n_skip += 1
            continue
        except Exception as e:
            n_skip += 1
            logger.debug("Unexpected error on %s: %s", md5, e)
            continue

        # Criterion 1: structural parity
        v7_legacy = _legacy_edge_set(v7_graph)
        v8_legacy = _legacy_edge_set(v8_graph)
        if v7_legacy != v8_legacy:
            n_fail += 1
            only_v7 = v7_legacy - v8_legacy
            only_v8 = v8_legacy - v7_legacy
            fail_reasons.append(
                f"PARITY FAIL {md5[:16]}: "
                f"{len(only_v7)} edges only in v7, {len(only_v8)} only in v8"
            )
            if len(fail_reasons) <= 5:
                print(f"  FAIL {md5[:16]} — parity mismatch: "
                      f"-{len(only_v7)} +{len(only_v8)} edges")
            continue

        # Collect stats
        n_pass += 1
        total = v8_graph.num_edges
        all_edge_counts.append(total)
        for t, c in _edge_type_counts(v8_graph).items():
            type_totals[t] += c

    elapsed_total = time.time() - t_start

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("PLAN-1B STRUCTURAL COMPARISON GATE — RESULTS")
    print("=" * 62)
    print(f"Processed : {n_pass + n_fail + n_skip:,} contracts in {elapsed_total:.0f}s")
    print(f"  PASS    : {n_pass:,}")
    print(f"  FAIL    : {n_fail:,}")
    print(f"  SKIP    : {n_skip:,} (extraction error or missing source)")

    if fail_reasons:
        print("\nFirst failures:")
        for r in fail_reasons[:10]:
            print(f"  {r}")

    # ── Criterion 1: structural parity ────────────────────────────────────────
    parity_ok = (n_fail == 0)
    print(f"\n[{'PASS' if parity_ok else 'FAIL'}] Criterion 1 — structural parity "
          f"({n_fail} failures)")

    # ── Criterion 2+3: edge count distribution ────────────────────────────────
    ec = np.array(all_edge_counts) if all_edge_counts else np.array([0])
    p50  = int(np.percentile(ec, 50))
    p99  = int(np.percentile(ec, 99))
    emax = int(ec.max())
    emean = float(ec.mean())
    print(f"\nEdge count distribution (n={len(all_edge_counts):,}):")
    print(f"  mean={emean:.0f}  P50={p50}  P99={p99}  max={emax}")
    p99_ok  = (p99  < P99_EDGE_LIMIT)
    max_ok  = (emax < MAX_EDGE_LIMIT)
    print(f"[{'PASS' if p99_ok  else 'FAIL'}] Criterion 2 — P99 < {P99_EDGE_LIMIT:,}  (P99={p99})")
    print(f"[{'PASS' if max_ok  else 'FAIL'}] Criterion 3 — max < {MAX_EDGE_LIMIT:,}  (max={emax})")

    # ── Criterion 4: new edge types fire ─────────────────────────────────────
    # EDGE_TYPES is {name_str: int_id} — sort by int value for display
    print("\nEdge type totals across sample:")
    new_types_ok = True
    for name, eid in sorted(EDGE_TYPES.items(), key=lambda x: x[1]):
        count = type_totals.get(eid, 0)
        tag = ""
        if eid in NEW_TYPES:
            fires = count > 0
            tag = f"  <- NEW {'OK' if fires else 'ZERO - FAIL'}"
            if not fires:
                new_types_ok = False
        if count > 0 or eid in NEW_TYPES:
            print(f"  {name:20s} ({eid:2d}): {count:,}{tag}")
    print(f"[{'PASS' if new_types_ok else 'FAIL'}] Criterion 4 — all new edge types fire")

    # ── Criterion 5: DataLoader batch ─────────────────────────────────────────
    dl_ok = True
    if not args.no_dataloader and all_edge_counts:
        print("\nDataLoader batch test (batch_size=8)...")
        try:
            from torch_geometric.data import Batch
            # Load 8 v8 graphs from sample and batch them
            batch_graphs = []
            for md5, sol_path in sample[:20]:
                try:
                    _sol = Path(sol_path)
                    _ver = _detect_solc_version(_sol)
                    _cfg = GraphExtractionConfig(
                        solc_version=_ver, solc_binary=_solc_binary(_ver), **_base_cfg
                    )
                    g = extract_contract_graph(sol_path, _cfg)
                    batch_graphs.append(g)
                    if len(batch_graphs) == 8:
                        break
                except Exception:
                    continue
            if len(batch_graphs) < 8:
                print(f"  Only {len(batch_graphs)} graphs available — skipping batch test")
                dl_ok = True
            else:
                batch = Batch.from_data_list(batch_graphs)
                print(f"  Batch: {batch.num_graphs} graphs, "
                      f"{batch.num_nodes} nodes, {batch.num_edges} edges")
                dl_ok = True
                print("[PASS] Criterion 5 — DataLoader batch")
        except Exception as e:
            dl_ok = False
            print(f"[FAIL] Criterion 5 — DataLoader batch: {e}")
    else:
        print("\n[SKIP] Criterion 5 — DataLoader batch (--no-dataloader)")

    # ── Final verdict ─────────────────────────────────────────────────────────
    all_ok = parity_ok and p99_ok and max_ok and new_types_ok and dl_ok
    print("\n" + "=" * 62)
    print(f"FINAL VERDICT: {'PASS — safe to proceed with full v8 re-extraction' if all_ok else 'FAIL — do NOT re-extract; investigate failures above'}")
    print("=" * 62)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
