"""Tests for the verification gate (Stage 4)."""
import json
from pathlib import Path

import pytest

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.class_auditor import run_audit, AuditResult
from sentinel_data.verification.gate import run_gate, GateResult, Verdict
from sentinel_data.verification.semantic_checker import run_semantic_check, SemanticCheckResult


_DATA_DIR = Path("Data/data")


def _skip_if_no_merged():
    merged = _DATA_DIR / "labels" / "merged"
    if not merged.exists() or not any(merged.glob("*.labels.json")):
        pytest.skip("Merged labels not found — run merger first")


class TestGateUnit:
    def _empty_audit(self) -> AuditResult:
        return AuditResult(per_class={cls: __import__(
            "sentinel_data.verification.class_auditor", fromlist=["ClassStats"]
        ).ClassStats(class_name=cls) for cls in class_names()})

    def _empty_semantic(self) -> SemanticCheckResult:
        from sentinel_data.verification.semantic_checker import ClassCheckSummary
        return SemanticCheckResult(by_class={cls: ClassCheckSummary(class_name=cls) for cls in class_names()})

    def test_runs_without_error_empty(self):
        audit = self._empty_audit()
        sem = self._empty_semantic()
        result = run_gate(audit, sem)
        assert isinstance(result, GateResult)

    def test_no_positives_is_provisional(self):
        audit = self._empty_audit()
        sem = self._empty_semantic()
        result = run_gate(audit, sem)
        for cls in class_names():
            assert result.verdicts[cls].verdict == Verdict.PROVISIONAL

    def test_t0_with_no_failures_is_verified(self):
        from sentinel_data.verification.class_auditor import ClassStats
        audit = AuditResult(
            per_class={
                "Reentrancy": ClassStats(
                    class_name="Reentrancy",
                    total_positives=100,
                    total_contracts=200,
                    by_tier={"T0": 100},
                    by_source={"solidifi": 100},
                ),
                **{cls: ClassStats(class_name=cls) for cls in class_names() if cls != "Reentrancy"}
            }
        )
        from sentinel_data.verification.semantic_checker import ClassCheckSummary
        sem = SemanticCheckResult(by_class={
            "Reentrancy": ClassCheckSummary(class_name="Reentrancy", pass_count=10, fail_count=0, positives_skipped=5),
            **{cls: ClassCheckSummary(class_name=cls) for cls in class_names() if cls != "Reentrancy"}
        })
        result = run_gate(audit, sem)
        assert result.verdicts["Reentrancy"].verdict == Verdict.VERIFIED

    def test_high_fail_rate_is_fail(self):
        from sentinel_data.verification.class_auditor import ClassStats
        audit = AuditResult(
            per_class={
                "CallToUnknown": ClassStats(
                    class_name="CallToUnknown",
                    total_positives=100,
                    total_contracts=200,
                    by_tier={"T3": 100},
                    by_source={"bccc": 100},
                ),
                **{cls: ClassStats(class_name=cls) for cls in class_names() if cls != "CallToUnknown"}
            }
        )
        from sentinel_data.verification.semantic_checker import ClassCheckSummary
        sem = SemanticCheckResult(by_class={
            "CallToUnknown": ClassCheckSummary(class_name="CallToUnknown", pass_count=5, fail_count=95),
            **{cls: ClassCheckSummary(class_name=cls) for cls in class_names() if cls != "CallToUnknown"}
        })
        result = run_gate(audit, sem)
        assert result.verdicts["CallToUnknown"].verdict == Verdict.FAIL
        assert "CallToUnknown" in result.hard_fails

    def test_gate_passed_means_no_hard_fails(self):
        audit = self._empty_audit()
        sem = self._empty_semantic()
        result = run_gate(audit, sem)
        assert result.gate_passed == (len(result.hard_fails) == 0)

    def test_str_representation_works(self):
        audit = self._empty_audit()
        sem = self._empty_semantic()
        result = run_gate(audit, sem)
        report = str(result)
        assert "Verification Gate" in report
        assert "PASS" in report or "FAIL" in report


class TestGateIntegration:
    def test_smoke_on_real_corpus(self):
        _skip_if_no_merged()
        audit = run_audit(_DATA_DIR)
        semantic = run_semantic_check(_DATA_DIR, limit_per_class=10)
        result = run_gate(audit, semantic)
        assert isinstance(result, GateResult)
        assert len(result.verdicts) == 10

    def test_no_hard_fails_on_verified_corpus(self):
        """SolidiFI+DIVE corpus should have no FAIL classes."""
        _skip_if_no_merged()
        audit = run_audit(_DATA_DIR)
        semantic = run_semantic_check(_DATA_DIR, limit_per_class=20)
        result = run_gate(audit, semantic)
        assert len(result.hard_fails) == 0, (
            f"Unexpected FAIL classes: {result.hard_fails}"
        )
