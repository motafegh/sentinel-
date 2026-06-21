# ml/tests/test_framework_gates.py
"""
Unit tests for the testing framework gates (ml/testing_specs/framework/gates.py).

Covers the gate classes added in the 2026-06-17 testing suite overhaul:
- JSONKeyGate
- ReproducibilityGate
- StaleCheckpointsGate
- CompositeGate (aggregation logic)
- GateStatus enum

These tests are project-agnostic — they don't need a model checkpoint,
the API, or any ML dependencies. Just the gates module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.testing_specs.framework.gates import (
    CompositeGate,
    F1Gate,
    FileExistsGate,
    Gate,
    GateResult,
    GateStatus,
    JSONFileGate,
    JSONKeyGate,
    ReproducibilityGate,
    StaleCheckpointsGate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_json(tmp_path: Path) -> Path:
    """Helper to write a small JSON file in tmp_path."""
    f = tmp_path / "data.json"
    return f


# ---------------------------------------------------------------------------
# JSONKeyGate
# ---------------------------------------------------------------------------


class TestJSONKeyGate:
    """Test JSONKeyGate — supports ==, !=, >, >=, <, <= on any JSON key."""

    def test_equal_op_pass(self, tmp_json):
        tmp_json.write_text(json.dumps({"summary": {"failed": 0, "n_flagged": 0}}))
        g = JSONKeyGate("g", tmp_json, key="summary.failed", op="==", expected=0)
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS
        assert "0" in r.message

    def test_equal_op_fail(self, tmp_json):
        tmp_json.write_text(json.dumps({"summary": {"failed": 13}}))
        g = JSONKeyGate("g", tmp_json, key="summary.failed", op="==", expected=0)
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL
        assert "13" in r.message

    def test_inequality_ops(self, tmp_json):
        tmp_json.write_text(json.dumps({"summary": {"max_disagreement": 0.45}}))
        # Test <=
        g = JSONKeyGate("g", tmp_json, key="summary.max_disagreement", op="<=", expected=0.50)
        assert g.run_with_timing().status == GateStatus.PASS
        # Test >
        g = JSONKeyGate("g", tmp_json, key="summary.max_disagreement", op=">", expected=0.30)
        assert g.run_with_timing().status == GateStatus.PASS
        # Test < (0.45 < 0.30 = False)
        g = JSONKeyGate("g", tmp_json, key="summary.max_disagreement", op="<", expected=0.30)
        assert g.run_with_timing().status == GateStatus.FAIL

    def test_nested_key_path(self, tmp_json):
        tmp_json.write_text(json.dumps({"a": {"b": {"c": "hello"}}}))
        g = JSONKeyGate("g", tmp_json, key="a.b.c", op="==", expected="hello")
        assert g.run_with_timing().status == GateStatus.PASS

    def test_missing_key_is_fail(self, tmp_json):
        tmp_json.write_text(json.dumps({"summary": {"n_flagged": 0}}))
        g = JSONKeyGate("g", tmp_json, key="summary.failed", op="==", expected=0)
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL
        assert "not found" in r.message

    def test_missing_file_is_fail(self, tmp_path):
        g = JSONKeyGate("g", tmp_path / "missing.json", key="x", op="==", expected=0)
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL
        assert "not found" in r.message

    def test_unparseable_json_is_fail(self, tmp_json):
        tmp_json.write_text("not valid json")
        g = JSONKeyGate("g", tmp_json, key="x", op="==", expected=0)
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL
        assert "Could not parse" in r.message

    def test_unknown_op_is_fail(self, tmp_json):
        tmp_json.write_text(json.dumps({"x": 1}))
        g = JSONKeyGate("g", tmp_json, key="x", op="~~", expected=0)
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL
        assert "Unknown op" in r.message

    def test_run_with_timing_measures(self, tmp_json):
        tmp_json.write_text(json.dumps({"x": 1}))
        g = JSONKeyGate("g", tmp_json, key="x", op="==", expected=1)
        r = g.run_with_timing()
        assert r.duration_s >= 0
        assert r.gate_name == "g"


# ---------------------------------------------------------------------------
# ReproducibilityGate
# ---------------------------------------------------------------------------


class TestReproducibilityGate:
    """Test ReproducibilityGate — checks auto_reproducibility_check.py output."""

    def test_pass(self, tmp_json):
        tmp_json.write_text(json.dumps({
            "checkpoint": "/path/to/ckpt.pt",
            "model_state_hash": "abc123",
            "git_commit": "12b449249",
            "poetry_lock_hash": "deadbeef",
            "transformers_offline": False,
            "result": "PASS",
        }))
        g = ReproducibilityGate("g", tmp_json, expected_result="PASS")
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS
        assert r.value["git_commit"] == "12b449249"

    def test_fail(self, tmp_json):
        tmp_json.write_text(json.dumps({
            "checkpoint": "/path/to/ckpt.pt",
            "result": "FAIL",
        }))
        g = ReproducibilityGate("g", tmp_json, expected_result="PASS")
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL
        assert "FAIL" in r.message

    def test_missing_file_is_unverified(self, tmp_path):
        """No reference = unverified, not failed (we don't have a reference to compare to)."""
        g = ReproducibilityGate("g", tmp_path / "missing.json", expected_result="PASS")
        r = g.run_with_timing()
        assert r.status == GateStatus.UNVERIFIED
        assert "not found" in r.message

    def test_unparseable_is_fail(self, tmp_json):
        tmp_json.write_text("garbage")
        g = ReproducibilityGate("g", tmp_json, expected_result="PASS")
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL

    def test_default_expected_result(self, tmp_json):
        """Default expected_result is 'PASS'."""
        tmp_json.write_text(json.dumps({"result": "PASS"}))
        g = ReproducibilityGate("g", tmp_json)
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS


# ---------------------------------------------------------------------------
# StaleCheckpointsGate
# ---------------------------------------------------------------------------


class TestStaleCheckpointsGate:
    """Test StaleCheckpointsGate — checks check_stale_checkpoints.py output."""

    def test_zero_stale_passes(self, tmp_json):
        tmp_json.write_text(json.dumps({
            "summary": {"total": 5, "stale_or_orphan": 0, "all_clean": True},
        }))
        g = StaleCheckpointsGate("g", tmp_json, max_stale=0)
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS

    def test_one_stale_with_default_max_fails(self, tmp_json):
        """Default max_stale=0 means even 1 stale is a fail."""
        tmp_json.write_text(json.dumps({
            "summary": {"total": 5, "stale_or_orphan": 1, "all_clean": False},
        }))
        g = StaleCheckpointsGate("g", tmp_json)
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL

    def test_max_stale_allows_some(self, tmp_json):
        """max_stale=3 allows up to 3 stale checkpoints."""
        tmp_json.write_text(json.dumps({
            "summary": {"total": 5, "stale_or_orphan": 2, "all_clean": False},
        }))
        g = StaleCheckpointsGate("g", tmp_json, max_stale=3)
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS

    def test_above_max_fails(self, tmp_json):
        tmp_json.write_text(json.dumps({
            "summary": {"total": 5, "stale_or_orphan": 4, "all_clean": False},
        }))
        g = StaleCheckpointsGate("g", tmp_json, max_stale=3)
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL

    def test_missing_file_is_unverified(self, tmp_path):
        g = StaleCheckpointsGate("g", tmp_path / "missing.json")
        r = g.run_with_timing()
        assert r.status == GateStatus.UNVERIFIED

    def test_empty_summary(self, tmp_json):
        """Empty summary block (no stale_or_orphan key) should not crash."""
        tmp_json.write_text(json.dumps({"summary": {}}))
        g = StaleCheckpointsGate("g", tmp_json, max_stale=0)
        r = g.run_with_timing()
        # stale_or_phan defaults to 0
        assert r.status == GateStatus.PASS


# ---------------------------------------------------------------------------
# CompositeGate
# ---------------------------------------------------------------------------


class _StubGate(Gate):
    """Gate that returns a pre-set status. Used in CompositeGate tests."""

    def __init__(self, name: str, status: GateStatus, msg: str = ""):
        super().__init__(name)
        self._status = status
        self._msg = msg

    def run(self) -> GateResult:
        return GateResult(
            gate_name=self.name,
            status=self._status,
            message=self._msg or f"stub {self._status.value}",
            duration_s=0.0,
        )


class TestCompositeGate:
    """Test CompositeGate aggregation rules."""

    def test_all_pass(self):
        g = CompositeGate("c", [
            _StubGate("a", GateStatus.PASS),
            _StubGate("b", GateStatus.PASS),
        ])
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS
        assert "2 PASS" in r.message

    def test_any_fail_dominates(self):
        g = CompositeGate("c", [
            _StubGate("a", GateStatus.PASS),
            _StubGate("b", GateStatus.FAIL),
            _StubGate("c", GateStatus.PASS),
        ])
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL
        assert "1 FAIL" in r.message
        assert "b" in r.message  # failure named in message

    def test_unverified_dominates_over_warn(self):
        g = CompositeGate("c", [
            _StubGate("a", GateStatus.WARN),
            _StubGate("b", GateStatus.UNVERIFIED),
        ])
        r = g.run_with_timing()
        assert r.status == GateStatus.UNVERIFIED

    def test_warn_with_require_all(self):
        g = CompositeGate("c", [
            _StubGate("a", GateStatus.PASS),
            _StubGate("b", GateStatus.WARN),
        ], require_all=True)
        r = g.run_with_timing()
        assert r.status == GateStatus.WARN

    def test_warn_without_require_all_passes(self):
        g = CompositeGate("c", [
            _StubGate("a", GateStatus.PASS),
            _StubGate("b", GateStatus.WARN),
        ], require_all=False)
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS


# ---------------------------------------------------------------------------
# FileExistsGate / JSONFileGate / F1Gate (sanity tests, no behavior change)
# ---------------------------------------------------------------------------


class TestExistingGates:
    """Smoke tests for the original gates (added 2026-06-17)."""

    def test_file_exists_pass(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("hi")
        g = FileExistsGate("g", f)
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS
        assert r.value["size"] == 2

    def test_file_exists_fail(self, tmp_path):
        g = FileExistsGate("g", tmp_path / "missing.txt")
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL

    def test_json_file_pass(self, tmp_json):
        tmp_json.write_text(json.dumps({"x": 5}))
        g = JSONFileGate("g", tmp_json, key="x", expected=5)
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS

    def test_f1_unverified_without_prior(self):
        g = F1Gate("g", current_f1=0.7, prior_f1=None)
        r = g.run_with_timing()
        assert r.status == GateStatus.UNVERIFIED

    def test_f1_pass(self):
        g = F1Gate("g", current_f1=0.75, prior_f1=0.70)
        r = g.run_with_timing()
        assert r.status == GateStatus.PASS
        assert r.value["delta"] == pytest.approx(0.05)

    def test_f1_fail_when_regression(self):
        g = F1Gate("g", current_f1=0.65, prior_f1=0.70)
        r = g.run_with_timing()
        assert r.status == GateStatus.FAIL

    def test_f1_warn_when_small_improvement(self):
        g = F1Gate("g", current_f1=0.701, prior_f1=0.70, min_improvement=0.01)
        r = g.run_with_timing()
        assert r.status == GateStatus.WARN


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


class TestGateResult:
    """Test the GateResult dataclass and its helpers."""

    def test_to_dict(self):
        r = GateResult(gate_name="g", status=GateStatus.PASS, message="ok", value=42)
        d = r.to_dict()
        assert d["gate_name"] == "g"
        assert d["status"] == "PASS"
        assert d["value"] == 42

    def test_passed_and_failed_properties(self):
        r = GateResult(gate_name="g", status=GateStatus.PASS, message="ok")
        assert r.passed is True
        assert r.failed is False
        r2 = GateResult(gate_name="g", status=GateStatus.FAIL, message="bad")
        assert r2.passed is False
        assert r2.failed is True


# ---------------------------------------------------------------------------
# GateStatus enum
# ---------------------------------------------------------------------------


class TestGateStatus:
    """Test the GateStatus enum (4 outcomes: PASS, FAIL, WARN, UNVERIFIED)."""

    def test_all_four_statuses(self):
        statuses = {s.value for s in GateStatus}
        assert statuses == {"PASS", "FAIL", "WARN", "UNVERIFIED"}

    def test_str_enum_inherits_str(self):
        # GateStatus is a str enum — it can be used as a string
        assert GateStatus.PASS == "PASS"
        assert str(GateStatus.PASS) == "GateStatus.PASS"
