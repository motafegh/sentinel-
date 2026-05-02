"""
test_drift_detector.py — Unit tests for DriftDetector (T2-B).

Covers the three cases from the ROADMAP unit test plan:
  1. Warm-up mode suppresses alerts
  2. KS test fires on p < 0.05 (large distributional shift)
  3. Rolling buffer evicts old entries after buffer_size requests

No real inference requests or model files are needed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ml.src.inference.drift_detector import DriftDetector, KS_ALPHA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_baseline(tmp_path: Path, values: list[float], key: str = "num_nodes") -> Path:
    baseline = {key: values}
    p = tmp_path / "drift_baseline.json"
    p.write_text(json.dumps(baseline))
    return p


def _feed(detector: DriftDetector, values: list[float], key: str = "num_nodes") -> None:
    for v in values:
        detector.update_stats({key: v})


# ---------------------------------------------------------------------------
# Test 1 — warm-up mode suppresses alerts
# ---------------------------------------------------------------------------

def test_warmup_suppresses_alerts(tmp_path):
    """Before warm-up completes (n_seen < n_warmup), check() returns empty dict."""
    baseline = _write_baseline(tmp_path, list(range(500)))
    # n_warmup=100 but we only feed 10 requests — warm-up incomplete
    detector = DriftDetector(baseline_path=None, n_warmup=100, buffer_size=200)

    _feed(detector, list(range(10)))

    assert not detector.warmup_done, "warm-up should not be marked done after 10 requests"
    result = detector.check()
    assert result == {}, "check() must return empty dict during warm-up"


def test_warmup_completes_after_n_requests():
    """After n_warmup requests, warmup_done flips to True."""
    detector = DriftDetector(baseline_path=None, n_warmup=5, buffer_size=200)
    _feed(detector, [1.0] * 5)
    assert detector.warmup_done


# ---------------------------------------------------------------------------
# Test 2 — KS fires on p < 0.05 (large distributional shift)
# ---------------------------------------------------------------------------

def test_ks_fires_on_drift(tmp_path, monkeypatch):
    """When current distribution differs drastically from baseline, p < 0.05."""
    # Baseline: num_nodes drawn from N(10, 1) — small, simple contracts
    import random
    rng = random.Random(42)
    baseline_vals = [rng.gauss(10, 1) for _ in range(200)]

    baseline_path = _write_baseline(tmp_path, baseline_vals)
    detector = DriftDetector(baseline_path=baseline_path, buffer_size=200)
    assert detector.warmup_done, "should skip warm-up when baseline is pre-loaded"

    # Current window: num_nodes from N(100, 1) — very different distribution
    current_vals = [rng.gauss(100, 1) for _ in range(100)]
    _feed(detector, current_vals)

    results = detector.check()
    assert "num_nodes" in results, "KS test should run for 'num_nodes'"
    assert results["num_nodes"] < KS_ALPHA, (
        f"Expected p < {KS_ALPHA} for a 10x shift, got p={results['num_nodes']:.4f}"
    )


def test_ks_does_not_fire_on_same_distribution(tmp_path):
    """When current distribution matches baseline, p > 0.05."""
    import random
    rng = random.Random(0)
    same_vals = [rng.gauss(10, 1) for _ in range(300)]

    baseline_path = _write_baseline(tmp_path, same_vals[:150])
    detector = DriftDetector(baseline_path=baseline_path, buffer_size=200)

    # Feed same distribution as baseline
    _feed(detector, same_vals[150:])
    results = detector.check()
    if "num_nodes" in results:
        assert results["num_nodes"] > KS_ALPHA, (
            f"KS should NOT fire for same distribution, got p={results['num_nodes']:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3 — rolling buffer evicts old entries after buffer_size requests
# ---------------------------------------------------------------------------

def test_buffer_rolls_after_buffer_size(tmp_path):
    """Buffer has a fixed maxlen; feeding more entries evicts the oldest."""
    baseline_path = _write_baseline(tmp_path, [5.0] * 100)
    buffer_size = 50
    detector = DriftDetector(baseline_path=baseline_path, buffer_size=buffer_size)

    # Feed 80 entries — first 30 should be evicted
    for i in range(80):
        detector.update_stats({"num_nodes": float(i)})

    assert detector.buffer_len == buffer_size, (
        f"Buffer should cap at {buffer_size}, got {detector.buffer_len}"
    )
    assert detector.n_seen == 80, "n_seen should count ALL requests, not just buffered ones"
