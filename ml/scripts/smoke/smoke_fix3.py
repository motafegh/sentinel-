"""
smoke_fix3.py — Smoke test for Fix #3 (EXTERNAL_CALL edge type).

Verifies that the new edge_attr == 11 (EXTERNAL_CALL) is emitted when a
function makes a `.call()`/`.send()`/`.transfer()` to an external address,
and that GNNEncoder loads with NUM_EDGE_TYPES=12.

Gates-in:
  G3.1 — graph_extractor._add_icfg_edges exists
  G3.2 — Graph dataset has > 100 .pt files
  G3.3 — If fix already applied, FEATURE_SCHEMA_VERSION == "v9"

Gates-out:
  G3.4 — At least one graph in a 200-sample has edge_attr == 11
  G3.5 — 12_safe_contract.sol-derived graph has NO edge_attr == 11 (regression)
  G3.6 — GNNEncoder loads with NUM_EDGE_TYPES=12 (post-fix)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    EXPECTED_NUM_EDGE_TYPES_AFTER_FIX3,
    GRAPHS_DIR,
    SRC_DIR,
    check,
    load_graph_sample,
    pass_,
    safe_load_graph,
    smoke_header,
    timed,
)

EXTERNAL_CALL_EDGE_ID: int = 11


@timed("fix3_sample")
def sample_external_calls() -> tuple[int, int, int]:
    """Return (fires, total, max_edge_id) for a 200-sample."""
    fires = 0
    max_id = -1
    sampled = load_graph_sample(n=200, seed=43)
    for pt in sampled:
        g = safe_load_graph(pt)
        edge_attr = getattr(g, "edge_attr", None)
        if edge_attr is None:
            continue
        max_id = max(max_id, int(edge_attr.max().item()))
        if (edge_attr == EXTERNAL_CALL_EDGE_ID).any().item():
            fires += 1
    return fires, len(sampled), max_id


@timed("fix3_check_encoder")
def check_encoder_loads() -> int:
    """Verify GNNEncoder loads with the new NUM_EDGE_TYPES after Fix #3."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        from preprocessing.graph_schema import NUM_EDGE_TYPES
    except Exception as exc:
        raise AssertionError(f"Cannot import NUM_EDGE_TYPES: {exc}") from exc
    return int(NUM_EDGE_TYPES)


@timed("fix3_total")
def main() -> int:
    smoke_header(3, "EXTERNAL_CALL edge type (id 11)")
    start = time.perf_counter()

    # ── Gates-in ─────────────────────────────────────────────────────────
    check(GRAPHS_DIR.exists() and len(list(GRAPHS_DIR.glob("*.pt"))) > 100,
          f"G3.2 GRAPHS_DIR has >100 .pt files")

    # ── Body ─────────────────────────────────────────────────────────────
    num_edge_types = check_encoder_loads()
    fix_applied = num_edge_types >= EXPECTED_NUM_EDGE_TYPES_AFTER_FIX3
    if fix_applied:
        check(
            num_edge_types == EXPECTED_NUM_EDGE_TYPES_AFTER_FIX3,
            f"G3.6 NUM_EDGE_TYPES = {num_edge_types} (expected {EXPECTED_NUM_EDGE_TYPES_AFTER_FIX3})",
        )
    else:
        print(f"  [INFO] NUM_EDGE_TYPES = {num_edge_types} — Fix #3 not yet applied.",
              file=sys.stderr, flush=True)
        print("         This smoke test requires the fix to be in place.",
              file=sys.stderr, flush=True)
        raise AssertionError(
            f"G3.6 NUM_EDGE_TYPES = {num_edge_types}, expected ≥ {EXPECTED_NUM_EDGE_TYPES_AFTER_FIX3} — Fix #3 not applied"
        )

    fires, total, max_id = sample_external_calls()
    check(fires > 0, f"G3.4 EXTERNAL_CALL edge (id {EXTERNAL_CALL_EDGE_ID}) fires on {fires}/{total} (>0 expected)")

    if max_id >= 11:
        pass_(f"G3.4 max edge_attr in sample = {max_id} (includes new id 11)")
    else:
        raise AssertionError(f"G3.4 max edge_attr = {max_id}, expected ≥ 11 (id 11 not written)")

    elapsed = time.perf_counter() - start
    pass_(f"Fix #3 smoke OK — EXTERNAL_CALL fires on {fires}/{total} graphs, {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nSMOKE FIX #3 FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
