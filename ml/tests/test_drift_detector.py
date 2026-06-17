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


# ---------------------------------------------------------------------------
# Test 4 — placeholder baseline triggers warm-up mode (Q4 MLOps Phase A.1)
# ---------------------------------------------------------------------------

def test_placeholder_baseline_triggers_warmup_mode(tmp_path):
    """A baseline JSON with no known stat names forces warm-up mode (alerts off).

    The shipped placeholder JSON has only `source/status/note` keys. Loading
    it must NOT silently disable drift monitoring; the detector should log
    a warning and enter warm-up mode.
    """
    placeholder = tmp_path / "placeholder.json"
    placeholder.write_text(json.dumps({
        "source": "warmup", "status": "PLACEHOLDER", "note": "not real"
    }))

    detector = DriftDetector(baseline_path=placeholder)
    assert not detector.warmup_done, (
        "Placeholder baseline (no known stat names) must force warm-up mode"
    )
    assert detector._baseline is None


def test_non_dict_baseline_triggers_warmup_mode(tmp_path):
    """A baseline that is not a JSON object forces warm-up mode."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2, 3]))  # list, not dict

    detector = DriftDetector(baseline_path=bad)
    assert not detector.warmup_done
    assert detector._baseline is None


def test_partial_baseline_recognises_known_stats(tmp_path):
    """A baseline with at least one known stat name is accepted."""
    mixed = {"source": "warmup", "unrelated_key": [1, 2], "num_nodes": [10, 20, 30]}
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps(mixed))

    detector = DriftDetector(baseline_path=p)
    assert detector.warmup_done, "Should accept baseline that has at least one known stat"
    assert "num_nodes" in detector._baseline


# ---------------------------------------------------------------------------
# Test 5 — dump_warmup_to_jsonl (Q4 MLOps Phase B.4)
# ---------------------------------------------------------------------------

def test_dump_warmup_to_jsonl_writes_valid_jsonl(tmp_path):
    """dump_warmup_to_jsonl writes one JSON record per line, all fields preserved."""
    detector = DriftDetector(baseline_path=None, n_warmup=5, buffer_size=200)
    for i in range(10):
        detector.update_stats({
            "num_nodes":        float(i * 10),
            "num_edges":        float(i * 25),
            "confirmed_count":  float(i % 3),
            "suspicious_count": float((i + 1) % 2),
        })

    out = tmp_path / "warmup.jsonl"
    n = detector.dump_warmup_to_jsonl(out)
    assert n == 10, f"Expected 10 records written, got {n}"
    assert out.exists()

    records = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(records) == 10
    # Verify the first and last records preserve all stat fields
    assert set(records[0].keys())  == {"num_nodes", "num_edges", "confirmed_count", "suspicious_count"}
    assert records[0]["num_nodes"]  == 0.0
    assert records[9]["num_nodes"]  == 90.0


def test_dump_warmup_to_jsonl_creates_parent_dirs(tmp_path):
    """dump_warmup_to_jsonl creates missing parent directories."""
    detector = DriftDetector(baseline_path=None, n_warmup=2, buffer_size=200)
    detector.update_stats({"num_nodes": 1.0})
    detector.update_stats({"num_nodes": 2.0})

    nested = tmp_path / "deep" / "nested" / "warmup.jsonl"
    detector.dump_warmup_to_jsonl(nested)
    assert nested.exists(), "Should create nested parent dirs"


def test_dump_warmup_to_jsonl_empty_buffer(tmp_path):
    """dump_warmup_to_jsonl on an empty buffer writes an empty file (0 lines)."""
    detector = DriftDetector(baseline_path=None, n_warmup=5, buffer_size=200)
    out = tmp_path / "empty.jsonl"
    n = detector.dump_warmup_to_jsonl(out)
    assert n == 0
    assert out.exists()
    assert out.read_text() == ""


# ---------------------------------------------------------------------------
# Test 6 — B.4 real baseline file (synthetic warmup)
# ---------------------------------------------------------------------------

def test_real_baseline_file_loads_into_active_mode():
    """The shipped drift_baseline_run12.json loads into active mode (4 stats)."""
    baseline_path = Path("ml/data/drift_baseline_run12.json")
    if not baseline_path.exists():
        pytest.skip(
            "ml/data/drift_baseline_run12.json not present — "
            "run: python ml/scripts/compute_drift_baseline.py --source warmup "
            "--warmup-log ml/data/warmup_run12.jsonl "
            "--output ml/data/drift_baseline_run12.json"
        )

    detector = DriftDetector(baseline_path=baseline_path)
    assert detector.warmup_done, "Real baseline should put detector in active mode"
    assert detector._baseline is not None
    expected_stats = {"num_nodes", "num_edges", "confirmed_count", "suspicious_count"}
    actual_stats = set(detector._baseline.keys()) & expected_stats
    assert actual_stats == expected_stats, (
        f"Expected 4 known stats, got {actual_stats}"
    )
    for stat in expected_stats:
        assert len(detector._baseline[stat]) >= 30, (
            f"{stat} has {len(detector._baseline[stat])} samples; need >=30 for KS"
        )
