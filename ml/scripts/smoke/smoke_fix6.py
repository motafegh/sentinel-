"""
smoke_fix6.py — Smoke test for Fix #6 (predictor tier thresholds).

Verifies that the predictor:
  - Loads a checkpoint
  - Loads per-class tuned thresholds (if _thresholds.json exists)
  - Runs inference on a known-safe contract and returns NOTEWORTHY tier
  - Reports thresholds dict in __repr__ or accessible attribute

Gates-in:
  G6.1 — At least one checkpoint exists in ml/checkpoints/
  G6.2 — ml/src/inference/predictor.py exists
  G6.3 — ml/scripts/test_contracts/12_safe_contract.sol exists

Gates-out:
  G6.4 — Predictor.__init__ loads with threshold=0.5 (no crash)
  G6.5 — Predictor.thresholds attribute is a dict (or None if no JSON) — typed
  G6.6 — Calling predict on 12_safe_contract.sol returns 0 CONFIRMED classes
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    CHECKPOINTS_DIR,
    SRC_DIR,
    TEST_CONTRACTS_DIR,
    check,
    check_schema_version,
    find_checkpoint,
    pass_,
    smoke_header,
    timed,
)


@timed("fix6_load_predictor")
def load_predictor():
    """Load the predictor and return (predictor, threshold_count)."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        from inference.predictor import Predictor
    except Exception as exc:
        raise AssertionError(f"G6.2 Cannot import Predictor: {exc}") from exc

    ckpt = find_checkpoint("Run8-v10")
    if ckpt is None:
        ckpt = find_checkpoint("v10")
    if ckpt is None:
        candidates = sorted(CHECKPOINTS_DIR.glob("*best.pt"))
        if not candidates:
            raise AssertionError(
                f"G6.1 no checkpoint in {CHECKPOINTS_DIR}"
            )
        ckpt = str(candidates[-1])
    pass_(f"G6.1 using checkpoint: {Path(ckpt).name}")

    try:
        predictor = Predictor(checkpoint=ckpt, threshold=0.5)
    except Exception as exc:
        raise AssertionError(f"G6.4 Predictor construction failed: {exc}") from exc

    thresholds = getattr(predictor, "thresholds", None)
    threshold_count = len(thresholds) if isinstance(thresholds, dict) else 0
    return predictor, threshold_count


@timed("fix6_check_safe_contract")
def check_safe_contract(predictor) -> dict:
    """Run predictor on 12_safe_contract.sol and verify 0 CONFIRMED classes."""
    safe_sol = TEST_CONTRACTS_DIR / "12_safe_contract.sol"
    if not safe_sol.exists():
        raise AssertionError(f"G6.3 safe contract missing: {safe_sol}")

    try:
        result = predictor.predict(str(safe_sol))
    except Exception as exc:
        raise AssertionError(f"G6.6 predict() on safe contract crashed: {exc}") from exc

    if isinstance(result, dict):
        confirmed = [k for k, v in result.items() if isinstance(v, dict) and v.get("tier") == "CONFIRMED"]
        n_confirmed = len(confirmed)
    else:
        n_confirmed = -1
        confirmed = []

    check(
        n_confirmed == 0,
        f"G6.6 safe contract has 0 CONFIRMED classes (got {n_confirmed}: {confirmed[:3]})",
    )
    return {"confirmed": n_confirmed, "result": result}


@timed("fix6_total")
def main() -> int:
    smoke_header(6, "Predictor tier-threshold display (per-class _thresholds.json)")
    start = time.perf_counter()

    # ── Gates-in ─────────────────────────────────────────────────────────
    check(CHECKPOINTS_DIR.exists(), f"G6.1 CHECKPOINTS_DIR exists: {CHECKPOINTS_DIR}")
    check((TEST_CONTRACTS_DIR / "12_safe_contract.sol").exists(), "G6.3 safe contract exists")

    # ── Body ─────────────────────────────────────────────────────────────
    predictor, n_thresh = load_predictor()
    check(n_thresh == 10, f"G6.5 predictor.thresholds has 10 entries (got {n_thresh})")

    result_stats = check_safe_contract(predictor)

    elapsed = time.perf_counter() - start
    pass_(f"Fix #6 smoke OK — thresholds loaded, safe contract CONFIRMED={result_stats['confirmed']}, {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nSMOKE FIX #6 FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
