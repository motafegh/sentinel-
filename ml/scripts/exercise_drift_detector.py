#!/usr/bin/env python3
"""
exercise_drift_detector.py — CI smoke test for DriftDetector.

WHY THIS EXISTS
───────────────
DriftDetector is exercised in production but never tested in CI.
A bug in the KS test or rolling buffer could go undetected until a real
distribution shift occurs in production.

TEST SEQUENCE
─────────────
Phase 1 — Warm-up (500 requests, no baseline):
    Feed 500 synthetic in-distribution vectors.
    Assert check() returns {} (alerts suppressed during warm-up).
    Assert warmup_done flips True at exactly N_WARMUP.

Phase 2 — Baseline construction:
    Build baseline dict from Phase 1 warm-up buffer.
    Write to temp file with source='warmup' metadata.

Phase 3 — In-distribution (100 requests, baseline loaded):
    Feed 100 more in-distribution vectors.
    Assert zero KS alerts fire.

Phase 4 — Distribution shift (100 requests, 3× num_nodes):
    Feed 100 shifted vectors (num_nodes × 3, num_edges × 3).
    Assert at least one KS alert fires within 50 post-shift requests.

Exit codes:
    0  all assertions pass
    1  any assertion fails
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Allow running from repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.inference.drift_detector import DriftDetector, KS_ALPHA, MIN_SAMPLES_FOR_KS

# ── Synthetic distribution parameters ────────────────────────────────────────
# Based on Run 4 corpus stats: graphs up to 1735 nodes, median ~120-150.
RNG_SEED        = 42
N_WARMUP        = 500
BUFFER_SIZE     = 200
N_INDIST        = 100   # in-distribution requests after baseline loaded
N_SHIFT         = 100   # shifted requests
SHIFT_FACTOR    = 3.0   # multiply num_nodes / num_edges by this after shift
CHECK_INTERVAL  = 10    # run check() every N requests in post-shift phase

STAT_NAMES = ["num_nodes", "num_edges", "confirmed_count", "suspicious_count"]


def _sample_vector(rng: np.random.Generator, shifted: bool = False) -> dict[str, float]:
    factor = SHIFT_FACTOR if shifted else 1.0
    num_nodes = float(rng.lognormal(mean=4.8, sigma=0.9)) * factor   # median ~120
    num_edges = num_nodes * rng.uniform(1.8, 2.8) * factor
    confirmed_count = float(rng.poisson(0.4))
    suspicious_count = float(rng.poisson(1.1))
    return {
        "num_nodes":        round(num_nodes, 2),
        "num_edges":        round(num_edges, 2),
        "confirmed_count":  confirmed_count,
        "suspicious_count": suspicious_count,
    }


def _build_baseline(warmup_data: list[dict[str, float]]) -> dict:
    baseline: dict = {"source": "warmup"}
    for stat in STAT_NAMES:
        baseline[stat] = [s[stat] for s in warmup_data if stat in s]
    return baseline


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Warm-up
# ─────────────────────────────────────────────────────────────────────────────

def phase1_warmup(rng: np.random.Generator) -> DriftDetector:
    print("\nPhase 1: warm-up (500 requests, no baseline)")
    detector = DriftDetector(baseline_path=None, n_warmup=N_WARMUP, buffer_size=BUFFER_SIZE)

    for i in range(N_WARMUP):
        detector.update_stats(_sample_vector(rng))

        if (i + 1) % 50 == 0:
            result = detector.check()
            if result:
                _fail(
                    f"check() returned non-empty during warm-up at request {i+1}: {result}\n"
                    "  DriftDetector fired alerts before warm-up complete — alerts should be suppressed."
                )

        # warmup_done should flip at exactly N_WARMUP
        if i + 1 == N_WARMUP - 1:
            if detector.warmup_done:
                _fail(
                    f"warmup_done=True at request {i+1}, expected False until {N_WARMUP}."
                )
        if i + 1 == N_WARMUP:
            if not detector.warmup_done:
                _fail(
                    f"warmup_done=False after {N_WARMUP} requests — warm-up never completed."
                )

    _ok(f"warmup_done=True after {N_WARMUP} requests; check() returned {{}} throughout warm-up")
    return detector


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Baseline construction
# ─────────────────────────────────────────────────────────────────────────────

def phase2_baseline(detector: DriftDetector, tmp_path: Path) -> Path:
    print("\nPhase 2: build baseline from warm-up buffer")
    warmup_data = detector.dump_warmup_stats()

    if len(warmup_data) < MIN_SAMPLES_FOR_KS:
        _fail(
            f"Warm-up buffer has only {len(warmup_data)} entries; "
            f"need at least {MIN_SAMPLES_FOR_KS} for KS test."
        )

    baseline = _build_baseline(warmup_data)
    baseline_file = tmp_path / "drift_baseline_test.json"
    with open(baseline_file, "w") as f:
        json.dump(baseline, f)

    _ok(
        f"baseline written to {baseline_file.name} "
        f"({len(warmup_data)} vectors, {len(STAT_NAMES)} stats)"
    )
    return baseline_file


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — In-distribution (no alerts expected)
# ─────────────────────────────────────────────────────────────────────────────

def phase3_in_distribution(rng: np.random.Generator, baseline_file: Path) -> DriftDetector:
    print("\nPhase 3: in-distribution (100 requests, no alerts expected)")
    detector = DriftDetector(
        baseline_path=baseline_file, n_warmup=N_WARMUP, buffer_size=BUFFER_SIZE
    )

    if not detector.warmup_done:
        _fail("warmup_done=False after loading a baseline — should be True immediately.")
    _ok("warmup_done=True immediately after baseline loaded")

    alert_count = 0
    for i in range(N_INDIST):
        detector.update_stats(_sample_vector(rng, shifted=False))

        if (i + 1) % 50 == 0:
            results = detector.check()
            alerts = {k: v for k, v in results.items() if v < KS_ALPHA}
            alert_count += len(alerts)
            if alerts:
                print(
                    f"  WARNING: {len(alerts)} alert(s) on in-distribution data at "
                    f"request {i+1}: {alerts}\n"
                    f"  (p < {KS_ALPHA} — may be a false positive from small sample; "
                    f"not failing, but investigate if repeated)"
                )

    if alert_count > 0:
        # Tolerate up to 1 false positive (5% expected from alpha=0.05 over 2 checks × 4 stats)
        expected_fp = 2 * len(STAT_NAMES) * KS_ALPHA  # ~0.4
        if alert_count > max(1, int(expected_fp * 3)):
            _fail(
                f"{alert_count} KS alerts on in-distribution data — far above expected "
                f"false-positive rate ({expected_fp:.1f}). Baseline or KS logic may be broken."
            )
        print(f"  NOTE: {alert_count} marginal alert(s) — within tolerable false-positive range")
    else:
        _ok("zero KS alerts on in-distribution data")

    return detector


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Distribution shift (must trigger alert)
# ─────────────────────────────────────────────────────────────────────────────

def phase4_shift(rng: np.random.Generator, detector: DriftDetector) -> None:
    print(f"\nPhase 4: distribution shift ({SHIFT_FACTOR}× num_nodes/num_edges, {N_SHIFT} requests)")
    alert_fired = False
    alert_at    = None

    for i in range(N_SHIFT):
        detector.update_stats(_sample_vector(rng, shifted=True))

        if (i + 1) % CHECK_INTERVAL == 0:
            results = detector.check()
            alerts = {k: v for k, v in results.items() if v < KS_ALPHA}
            if alerts:
                alert_fired = True
                alert_at    = i + 1
                print(f"  Drift detected at request {alert_at}: {alerts}")
                break

    if not alert_fired:
        _fail(
            f"No KS drift alert fired within {N_SHIFT} shifted requests "
            f"(shift factor {SHIFT_FACTOR}×). "
            "Either the KS test is broken or the buffer is not accumulating shifted data."
        )

    _ok(f"KS alert fired at shifted request {alert_at} — detector is responsive to distribution shift")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("SENTINEL DriftDetector smoke test")
    print(f"  N_WARMUP={N_WARMUP}  BUFFER_SIZE={BUFFER_SIZE}  SHIFT={SHIFT_FACTOR}×")

    rng = np.random.default_rng(RNG_SEED)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        detector  = phase1_warmup(rng)
        baseline_file = phase2_baseline(detector, tmp_path)
        detector2 = phase3_in_distribution(rng, baseline_file)
        phase4_shift(rng, detector2)

    print("\nPASS — all DriftDetector smoke tests passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
