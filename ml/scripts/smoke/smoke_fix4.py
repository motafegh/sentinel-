"""
smoke_fix4.py — Smoke test for Fix #4 (IntegerUO schema gap).

Verifies that:
  - CFG_NODE_ARITH (id 13) is registered in NODE_TYPES
  - `in_unchecked_block` (feat[11]) is re-introduced
  - _MAX_TYPE_ID auto-updates to 13.0
  - NODE_FEATURE_DIM = 12
  - Graph re-extraction captures both new signals

Gates-in:
  G4.1 — Fix #2 and Fix #3 already applied (NODE_FEATURE_DIM >= 12 OR schema v9)
  G4.2 — graph_extractor._compute_in_unchecked exists and is not stub

Gates-out:
  G4.3 — NODE_FEATURE_DIM == 12, NUM_NODE_TYPES == 14
  G4.4 — _MAX_TYPE_ID == 13.0 (auto-derived from NODE_TYPES)
  G4.5 — At least one graph has feat[11] > 0.5 (in_unchecked_block fires)
  G4.6 — SentinelModel loads with in_channels=12 (no shape mismatch)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    EXPECTED_NODE_FEATURE_DIM_AFTER_FIX4,
    EXPECTED_NUM_NODE_TYPES_AFTER_FIX4,
    GRAPHS_DIR,
    SRC_DIR,
    check,
    load_graph_sample,
    pass_,
    safe_load_graph,
    smoke_header,
    timed,
)

IN_UNCHECKED_FEATURE_IDX: int = 11
EXPECTED_ARITH_NODE_ID: int = 13
EXPECTED_MAX_TYPE_ID: float = 13.0


@timed("fix4_check_schema")
def check_schema_constants() -> dict[str, int | float]:
    """Return schema dict; fail loudly if any constant is wrong."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        from preprocessing.graph_schema import (
            FEATURE_SCHEMA_VERSION,
            NODE_FEATURE_DIM,
            NUM_NODE_TYPES,
            NUM_EDGE_TYPES,
            NODE_TYPES,
        )
    except Exception as exc:
        raise AssertionError(f"Cannot import schema constants: {exc}") from exc

    check(
        FEATURE_SCHEMA_VERSION == "v9",
        f"G4.1 FEATURE_SCHEMA_VERSION == 'v9' (got {FEATURE_SCHEMA_VERSION!r})",
    )
    check(
        NODE_FEATURE_DIM == EXPECTED_NODE_FEATURE_DIM_AFTER_FIX4,
        f"G4.3 NODE_FEATURE_DIM == {EXPECTED_NODE_FEATURE_DIM_AFTER_FIX4} (got {NODE_FEATURE_DIM})",
    )
    check(
        NUM_NODE_TYPES == EXPECTED_NUM_NODE_TYPES_AFTER_FIX4,
        f"G4.3 NUM_NODE_TYPES == {EXPECTED_NUM_NODE_TYPES_AFTER_FIX4} (got {NUM_NODE_TYPES})",
    )
    check(
        EXPECTED_ARITH_NODE_ID in NODE_TYPES.values(),
        f"G4.3 CFG_NODE_ARITH (id {EXPECTED_ARITH_NODE_ID}) in NODE_TYPES",
    )

    max_type_id = float(max(NODE_TYPES.values()))
    check(
        max_type_id == EXPECTED_MAX_TYPE_ID,
        f"G4.4 _MAX_TYPE_ID == {EXPECTED_MAX_TYPE_ID} (got {max_type_id})",
    )

    return {
        "version": FEATURE_SCHEMA_VERSION,
        "node_dim": NODE_FEATURE_DIM,
        "num_node_types": NUM_NODE_TYPES,
        "num_edge_types": NUM_EDGE_TYPES,
        "max_type_id": max_type_id,
    }


@timed("fix4_check_extractor")
def check_extractor_reimplemented() -> bool:
    """Verify _compute_in_unchecked is no longer a NotImplementedError stub."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        from preprocessing import graph_extractor
    except Exception as exc:
        raise AssertionError(f"Cannot import graph_extractor: {exc}") from exc

    fn = getattr(graph_extractor, "_compute_in_unchecked", None)
    if fn is None:
        raise AssertionError("G4.2 _compute_in_unchecked not found in graph_extractor")
    import inspect
    src = inspect.getsource(fn)
    if "NotImplementedError" in src and "raise NotImplementedError" in src:
        raise AssertionError(
            "G4.2 _compute_in_unchecked is still a NotImplementedError stub — Fix #4 not applied"
        )
    return True


@timed("fix4_sample")
def sample_in_unchecked() -> tuple[int, int]:
    """Count graphs with feat[11] > 0.5 in a 200-sample."""
    fires = 0
    sampled = load_graph_sample(n=200, seed=44)
    for pt in sampled:
        g = safe_load_graph(pt)
        x = getattr(g, "x", None)
        if x is None or x.shape[1] <= IN_UNCHECKED_FEATURE_IDX:
            continue
        if (x[:, IN_UNCHECKED_FEATURE_IDX] > 0.5).any().item():
            fires += 1
    return fires, len(sampled)


@timed("fix4_check_model")
def check_model_loads() -> bool:
    """Verify SentinelModel loads with in_channels=12 (no shape mismatch)."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        from models.sentinel_model import SentinelModel
        from preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES
        SentinelModel(
            in_channels=NODE_FEATURE_DIM,
            num_edge_types=NUM_EDGE_TYPES,
            num_classes=10,
        )
        return True
    except Exception as exc:
        raise AssertionError(f"G4.6 SentinelModel construction failed: {exc}") from exc


@timed("fix4_total")
def main() -> int:
    smoke_header(4, "IntegerUO schema gap (CFG_NODE_ARITH + in_unchecked_block)")
    start = time.perf_counter()

    # ── Gates-in ─────────────────────────────────────────────────────────
    check(GRAPHS_DIR.exists() and len(list(GRAPHS_DIR.glob("*.pt"))) > 100,
          "G4.1 GRAPHS_DIR has >100 .pt files")

    # ── Body ─────────────────────────────────────────────────────────────
    schema = check_schema_constants()
    check_extractor_reimplemented()
    pass_("G4.2 _compute_in_unchecked is re-implemented (not a NotImplementedError stub)")

    fires, total = sample_in_unchecked()
    check(fires > 0, f"G4.5 in_unchecked_block (feat[11]) fires on {fires}/{total} (>0 expected)")

    check_model_loads()
    pass_("G4.6 SentinelModel loads with in_channels=12 (no shape mismatch)")

    elapsed = time.perf_counter() - start
    pass_(f"Fix #4 smoke OK — schema={schema['version']}, dim={schema['node_dim']}, {fires}/{total} graphs have feat[11], {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nSMOKE FIX #4 FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
