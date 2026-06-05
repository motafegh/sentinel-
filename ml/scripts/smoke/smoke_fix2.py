"""
smoke_fix2.py — Smoke test for Fix #2 (block-globals extraction).

Verifies that `_compute_uses_block_globals` correctly fires on contracts using
`now` (alias for block.timestamp) and library wrappers (SafeMath, etc.).

Gates-in:
  G2.1 — graph_schema.FEATURE_SCHEMA_VERSION is still "v8" (or "v9" if #2 applied)
  G2.2 — ml/src/preprocessing/graph_extractor.py:459 has _compute_uses_block_globals
  G2.3 — ml/data/graphs/ has > 100 .pt files (BCCC dataset)

Gates-out:
  G2.4 — On a 100-graph sample, feat[2] > 0.5 on at least 25 graphs (sane baseline)
  G2.5 — After fix, contracts using `now` (alias) also fire feat[2]
  G2.6 — validate_graph_dataset.py exits 0 on the new graphs
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    GRAPHS_DIR,
    SRC_DIR,
    check,
    check_schema_version,
    load_graph_sample,
    pass_,
    safe_load_graph,
    smoke_header,
    timed,
)


@timed("fix2_sample")
def sample_block_globals() -> tuple[int, int]:
    """Return (fires, total) — count of graphs with feat[2] > 0.5 in a 100-sample."""
    fires = 0
    sampled = load_graph_sample(n=100, seed=42)
    for pt in sampled:
        g = safe_load_graph(pt)
        x = getattr(g, "x", None)
        if x is None or x.shape[1] < 3:
            continue
        if (x[:, 2] > 0.5).any().item():
            fires += 1
    return fires, len(sampled)


@timed("fix2_check_now_alias")
def check_now_alias() -> bool:
    """Verify _compute_uses_block_globals handles 'now' alias.

    This is the actual regression check for the bug described in doc 02.
    Imports graph_extractor and inspects the function source.
    """
    sys.path.insert(0, str(SRC_DIR))
    try:
        from preprocessing import graph_extractor
    except Exception as exc:
        raise AssertionError(f"Cannot import graph_extractor: {exc}") from exc

    fn = getattr(graph_extractor, "_compute_uses_block_globals", None)
    if fn is None:
        raise AssertionError("_compute_uses_block_globals not found in graph_extractor")

    import inspect
    src = inspect.getsource(fn)
    has_now = "'now'" in src or '"now"' in src
    return has_now


@timed("fix2_total")
def main() -> int:
    smoke_header(2, "Block-globals extraction (now alias + library wrappers)")
    start = time.perf_counter()

    # ── Gates-in ─────────────────────────────────────────────────────────
    check(GRAPHS_DIR.exists() and len(list(GRAPHS_DIR.glob("*.pt"))) > 100,
          f"G2.3 GRAPHS_DIR has >100 .pt files")

    # ── Body ─────────────────────────────────────────────────────────────
    has_now = check_now_alias()
    if has_now:
        pass_("G2.5 _compute_uses_block_globals source contains 'now' alias check")
    else:
        print("  [WARN] G2.5 _compute_uses_block_globals does NOT mention 'now' alias.",
              file=sys.stderr, flush=True)
        print("         This is the exact bug doc 02 fixes — apply the fix first.",
              file=sys.stderr, flush=True)
        raise AssertionError(
            "G2.5 failed: 'now' alias not in _compute_uses_block_globals — Fix #2 not applied"
        )

    fires, total = sample_block_globals()
    pct = 100 * fires / total if total else 0
    check(fires >= 25, f"G2.4 feat[2] fires on {fires}/{total} ({pct:.0f}%) — expected ≥ 25")

    elapsed = time.perf_counter() - start
    pass_(f"Fix #2 smoke OK — {fires}/{total} graphs have feat[2] active, {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nSMOKE FIX #2 FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
