"""Tests for P8a formal verification (Halmos) node."""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from src.orchestration.verdict.evidence import Evidence, Polarity, Kind
from src.orchestration.verdict.emit import emit_halmos_evidence


class TestEmitHalmosEvidence:
    """Test the emit_halmos_evidence function."""

    def test_violation_emits_supports(self):
        """A violated invariant should emit SUPPORTS evidence."""
        findings = [
            {
                "tool": "halmos",
                "vulnerability_class": "Reentrancy",
                "invariant": "reentrancy",
                "proven": False,
                "counterexample": '{"msg.sender": "0xdead"}',
            }
        ]
        evidence = emit_halmos_evidence(findings)
        assert len(evidence) == 1
        assert evidence[0].source == "halmos"
        assert evidence[0].vuln_class == "Reentrancy"
        assert evidence[0].polarity == Polarity.SUPPORTS
        assert evidence[0].kind == Kind.FORMAL
        assert evidence[0].deterministic is True
        assert evidence[0].strength == 1.0

    def test_proven_emits_refutes(self):
        """A proven invariant should emit REFUTES evidence."""
        findings = [
            {
                "tool": "halmos",
                "vulnerability_class": "Reentrancy",
                "invariant": "reentrancy",
                "proven": True,
                "counterexample": "",
            }
        ]
        evidence = emit_halmos_evidence(findings)
        assert len(evidence) == 1
        assert evidence[0].polarity == Polarity.REFUTES

    def test_non_halmos_findings_ignored(self):
        """Findings from other tools should be ignored."""
        findings = [
            {"tool": "z3", "vulnerability_class": "Reentrancy", "proven": False},
        ]
        evidence = emit_halmos_evidence(findings)
        assert len(evidence) == 0

    def test_empty_findings(self):
        """Empty findings should return empty list."""
        assert emit_halmos_evidence([]) == []

    def test_multiple_findings(self):
        """Multiple findings should produce multiple evidence items."""
        findings = [
            {"tool": "halmos", "vulnerability_class": "Reentrancy", "invariant": "reentrancy", "proven": False, "counterexample": ""},
            {"tool": "halmos", "vulnerability_class": "IntegerUO", "invariant": "arithmetic", "proven": True, "counterexample": ""},
        ]
        evidence = emit_halmos_evidence(findings)
        assert len(evidence) == 2
        assert evidence[0].vuln_class == "Reentrancy"
        assert evidence[0].polarity == Polarity.SUPPORTS
        assert evidence[1].vuln_class == "IntegerUO"
        assert evidence[1].polarity == Polarity.REFUTES


class TestEvidenceFormalConstructor:
    """Test the Evidence.formal() constructor."""

    def test_supports_violation(self):
        """Formal evidence for a violation should have high strength."""
        ev = Evidence.formal(
            source="halmos",
            vuln_class="Reentrancy",
            polarity=Polarity.SUPPORTS,
            invariant="reentrancy",
            proven=False,
            counterexample="0xdead",
        )
        assert ev.source == "halmos"
        assert ev.kind == Kind.FORMAL
        assert ev.deterministic is True
        assert ev.strength == 1.0
        assert ev.detail["invariant"] == "reentrancy"
        assert ev.detail["proven"] is False

    def test_refutes_safety(self):
        """Formal evidence for safety should have 0.9 strength."""
        ev = Evidence.formal(
            source="halmos",
            vuln_class="Reentrancy",
            polarity=Polarity.REFUTES,
            invariant="reentrancy",
            proven=True,
        )
        assert ev.strength == 0.9
        assert ev.detail["proven"] is True


class TestParseHalmosOutput:
    """Test the Halmos JSON output parser."""

    def test_parse_pass(self):
        """A passing test should produce proven=True."""
        from src.orchestration.nodes.formal_verification import _parse_halmos_output
        json_str = json.dumps({
            "results": [
                {"name": "check_reentrancy()", "status": "pass"},
            ]
        })
        findings = _parse_halmos_output(json_str)
        assert len(findings) == 1
        assert findings[0]["proven"] is True
        assert findings[0]["vulnerability_class"] == "Reentrancy"

    def test_parse_fail(self):
        """A failing test should produce proven=False with counterexample."""
        from src.orchestration.nodes.formal_verification import _parse_halmos_output
        json_str = json.dumps({
            "results": [
                {
                    "name": "check_reentrancy()",
                    "status": "fail",
                    "counterexample": {"msg.sender": "0xdead"},
                },
            ]
        })
        findings = _parse_halmos_output(json_str)
        assert len(findings) == 1
        assert findings[0]["proven"] is False
        assert "0xdead" in findings[0]["counterexample"]

    def test_parse_empty(self):
        """Empty results should return empty list."""
        from src.orchestration.nodes.formal_verification import _parse_halmos_output
        json_str = json.dumps({"results": []})
        findings = _parse_halmos_output(json_str)
        assert len(findings) == 0

    def test_parse_invalid_json(self):
        """Invalid JSON should return empty list."""
        from src.orchestration.nodes.formal_verification import _parse_halmos_output
        findings = _parse_halmos_output("not json")
        assert len(findings) == 0

    def test_parse_unknown_invariant_skipped(self):
        """Unknown invariant names should be skipped."""
        from src.orchestration.nodes.formal_verification import _parse_halmos_output
        json_str = json.dumps({
            "results": [
                {"name": "check_unknown_thing()", "status": "pass"},
            ]
        })
        findings = _parse_halmos_output(json_str)
        assert len(findings) == 0


class TestFormalVerificationNode:
    """Test the formal_verification node."""

    @pytest.mark.asyncio
    async def test_skip_deterministic_mode(self):
        """Node should skip when SENTINEL_DETERMINISTIC=1."""
        import os
        from src.orchestration.nodes.formal_verification import formal_verification

        with patch.dict(os.environ, {"SENTINEL_DETERMINISTIC": "1"}):
            result = await formal_verification({"contract_code": "contract Test {}"})
            assert result["symbolic_findings"] == []

    @pytest.mark.asyncio
    async def test_skip_no_contract_code(self):
        """Node should skip when no contract code."""
        from src.orchestration.nodes.formal_verification import formal_verification

        result = await formal_verification({})
        assert result["symbolic_findings"] == []

    @pytest.mark.asyncio
    async def test_fail_soft_on_missing_tools(self):
        """Node should fail-soft when halmos/forge not installed."""
        from src.orchestration.nodes.formal_verification import formal_verification

        with patch("shutil.which", return_value=None):
            result = await formal_verification({
                "contract_code": "contract Test {}",
                "contract_address": "0xTEST",
            })
            assert result["symbolic_findings"] == []
            assert result["tool_status"]["halmos"]["ran"] is False
