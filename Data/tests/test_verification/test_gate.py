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


class TestGateWithToolValidation:
    def _build_audit_semantic(self, cls="CallToUnknown", fail_count=2, pass_count=8):
        from sentinel_data.verification.class_auditor import ClassStats
        from sentinel_data.verification.semantic_checker import ClassCheckSummary
        audit = AuditResult(
            per_class={
                cls: ClassStats(
                    class_name=cls, total_positives=10, total_contracts=20,
                    by_tier={"T3": 10}, by_source={"bccc": 10},
                ),
                **{c: ClassStats(class_name=c) for c in class_names() if c != cls}
            }
        )
        sem = SemanticCheckResult(by_class={
            cls: ClassCheckSummary(class_name=cls, pass_count=pass_count, fail_count=fail_count),
            **{c: ClassCheckSummary(class_name=c) for c in class_names() if c != cls}
        })
        return audit, sem

    def test_tool_agreement_low_downgrades_with_coflag(self):
        from sentinel_data.verification.tool_validator import (
            ToolValidationResult, ClassAgreementStats,
        )
        audit, sem = self._build_audit_semantic(pass_count=10, fail_count=0)
        # Add a co-occurrence flag for CallToUnknown
        from sentinel_data.verification.class_auditor import CoOccurrencePair
        audit.flagged_pairs = [CoOccurrencePair(
            class_a="CallToUnknown", class_b="DoS", count=5, count_a=10,
            rate=0.5, flagged=True,
        )]
        # Tool agreement is very low (10%) — should downgrade
        tv = ToolValidationResult(by_class={
            "CallToUnknown": ClassAgreementStats(
                class_name="CallToUnknown", positives_total=10,
                agree=1, disagree=9, checkable=10,
            ),
            **{c: ClassAgreementStats(class_name=c) for c in class_names() if c != "CallToUnknown"}
        })
        result = run_gate(audit, sem, tool_validation=tv)
        assert result.verdicts["CallToUnknown"].verdict.value in ("PROVISIONAL", "BEST-EFFORT")
        # Without co-flag, low tool agreement alone should NOT downgrade
        audit.flagged_pairs = []
        result2 = run_gate(audit, sem, tool_validation=tv)
        assert result2.verdicts["CallToUnknown"].verdict.value == "VERIFIED"


class TestGateWithFPEstimation:
    def test_fp_rate_above_threshold_fails(self):
        from sentinel_data.verification.class_auditor import ClassStats
        from sentinel_data.verification.semantic_checker import ClassCheckSummary
        from sentinel_data.verification.fp_estimator import (
            FPEstimationResult, ClassFPStats,
        )
        audit = AuditResult(per_class={
            "Reentrancy": ClassStats(class_name="Reentrancy", total_positives=50,
                                     total_contracts=100, by_tier={"T3": 50},
                                     by_source={"bccc": 50}),
            **{c: ClassStats(class_name=c) for c in class_names() if c != "Reentrancy"}
        })
        sem = SemanticCheckResult(by_class={
            "Reentrancy": ClassCheckSummary(class_name="Reentrancy", pass_count=45, fail_count=5),
            **{c: ClassCheckSummary(class_name=c) for c in class_names() if c != "Reentrancy"}
        })
        # High semantic pass rate (90%) — but FP rate is 50% → FAIL wins
        fp = FPEstimationResult(by_class={
            "Reentrancy": ClassFPStats(class_name="Reentrancy", sampled=10,
                                      likely_fp=5, errored=0),
            **{c: ClassFPStats(class_name=c) for c in class_names() if c != "Reentrancy"}
        })
        result = run_gate(audit, sem, fp_estimation=fp)
        assert result.verdicts["Reentrancy"].verdict.value == "FAIL"
        assert "FP rate" in result.verdicts["Reentrancy"].reason

    def test_fp_rate_below_threshold_does_not_fail(self):
        from sentinel_data.verification.class_auditor import ClassStats
        from sentinel_data.verification.semantic_checker import ClassCheckSummary
        from sentinel_data.verification.fp_estimator import (
            FPEstimationResult, ClassFPStats,
        )
        audit = AuditResult(per_class={
            "Reentrancy": ClassStats(class_name="Reentrancy", total_positives=50,
                                     total_contracts=100, by_tier={"T3": 50},
                                     by_source={"bccc": 50}),
            **{c: ClassStats(class_name=c) for c in class_names() if c != "Reentrancy"}
        })
        sem = SemanticCheckResult(by_class={
            "Reentrancy": ClassCheckSummary(class_name="Reentrancy", pass_count=45, fail_count=5),
            **{c: ClassCheckSummary(class_name=c) for c in class_names() if c != "Reentrancy"}
        })
        # Low FP rate (20%) — should NOT trigger FAIL
        fp = FPEstimationResult(by_class={
            "Reentrancy": ClassFPStats(class_name="Reentrancy", sampled=10,
                                      likely_fp=2, errored=0),
            **{c: ClassFPStats(class_name=c) for c in class_names() if c != "Reentrancy"}
        })
        result = run_gate(audit, sem, fp_estimation=fp)
        assert result.verdicts["Reentrancy"].verdict.value == "VERIFIED"


class TestGateWithNegativeCheck:
    def test_neg_check_fail_blocks_export(self):
        from sentinel_data.verification.negative_checker import NonVulnResult
        result = run_gate(
            self._empty_audit(), self._empty_semantic(),
            negative_check=NonVulnResult(total_checked=100, total_hits=20, total_errored=0),
        )
        assert result.negative_check_status == "FAIL"
        assert "__neg_check__" in result.hard_fails
        assert result.gate_passed is False

    def test_neg_check_warn_does_not_block(self):
        from sentinel_data.verification.negative_checker import NonVulnResult
        result = run_gate(
            self._empty_audit(), self._empty_semantic(),
            negative_check=NonVulnResult(total_checked=100, total_hits=7, total_errored=0),
        )
        assert result.negative_check_status == "WARN"
        assert "__neg_check__" not in result.hard_fails
        assert result.gate_passed is True

    def test_neg_check_ok_does_not_block(self):
        from sentinel_data.verification.negative_checker import NonVulnResult
        result = run_gate(
            self._empty_audit(), self._empty_semantic(),
            negative_check=NonVulnResult(total_checked=100, total_hits=3, total_errored=0),
        )
        assert result.negative_check_status == "OK"
        assert result.gate_passed is True

    def _empty_audit(self):
        from sentinel_data.verification.class_auditor import ClassStats
        return AuditResult(per_class={c: ClassStats(class_name=c) for c in class_names()})

    def _empty_semantic(self):
        from sentinel_data.verification.semantic_checker import ClassCheckSummary
        return SemanticCheckResult(by_class={c: ClassCheckSummary(class_name=c) for c in class_names()})
