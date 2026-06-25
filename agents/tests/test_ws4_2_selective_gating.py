"""
Tests for WS4.2 (2026-06-22): selective debate gating in cross_validator.

What WS4.2 does:
  Skip the LLM debate when MULTIPLE INDEPENDENT TOOLS (2+ of {ML, Slither,
  Aderyn}) already agree every flagged class is CONFIRMED by consensus. The
  consensus verdict (which already incorporates the tool corroboration) is
  used directly as the LLM verdict.

Why this is asymmetric per the FN/FP principle:
  The cheap signals (ML is discounted 50%; Slither/Aderyn are syntactic
  pattern-matchers with documented per-class blind spots) are LEAST trustworthy
  in the "looks safe" direction. We skip only on multi-tool "looks vulnerable"
  agreement. We NEVER skip on "looks safe by cheap signals" — that's exactly
  where a missed vulnerability would hide.

These tests verify the gating logic at the cross_validator level, using
mocked LLMs to prove the debate IS or IS NOT called based on the
consensus_verdict_state shape.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.nodes import cross_validator


def _make_state(
    flagged: list[dict],
    consensus_verdict: dict | None = None,
    static_findings: list | None = None,
    contract_code: str = "pragma solidity ^0.8.0;",
) -> dict:
    """Build a minimal AuditState for cross_validator."""
    return {
        "ml_result": {
            "confirmed": [v for v in flagged if v.get("tier") == "CONFIRMED"],
            "suspicious": [v for v in flagged if v.get("tier") == "SUSPICIOUS"],
            "vulnerabilities": [v for v in flagged if v.get("tier") in (None, "CONFIRMED")],
        },
        "static_findings": static_findings or [],
        "rag_results": [],
        "audit_history": [],
        "contract_code": contract_code,
        "consensus_verdict": consensus_verdict or {},
    }


def _mock_llm_with_response(verdict_json: str) -> MagicMock:
    """A mock LLM whose .invoke() returns a fixed string (used to detect if the debate ran)."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=verdict_json)
    return mock


# ---------------------------------------------------------------------------
# Test 1: debate is SKIPPED when all classes are CONFIRMED with ≥2 tools
# ---------------------------------------------------------------------------

class TestDebateSkipped:
    """
    The asymmetric skip case: 2+ tools agree on CONFIRMED for every flagged
    class. The debate is skipped and the consensus verdict is used directly.
    """

    @pytest.mark.asyncio
    async def test_debate_skipped_when_all_confirmed_with_two_tools(self, monkeypatch):
        """
        Setup: Reentrancy flagged, consensus CONFIRMED, ml_signal=1, slither_match=1.
        Tool count = 2 → satisfies the "multi-tool" condition.
        Expected: debate is skipped (LLM not called), evidence_list has CONFIRMED.
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            consensus_verdict={
                "Reentrancy": {
                    "verdict": "CONFIRMED",
                    "ml_signal": 1,
                    "slither_match": 1,
                    "aderyn_match": 0,
                    "confidence": 0.78,
                },
            },
        )

        # If the debate runs, this mock would be called 3 times. If skipped, never.
        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            mock_get_fast.return_value = _mock_llm_with_response(
                '{"Reentrancy": "DISPUTED"}'  # if debate ran, this would be the result
            )
            mock_get_strong.return_value = _mock_llm_with_response(
                '{"Reentrancy": "DISPUTED"}'
            )
            out = await cross_validator(state)

        # Debate was NOT called.
        assert mock_get_fast.return_value.invoke.call_count == 0
        # But the evidence_list was still emitted (CONFIRMED, from consensus).
        assert "evidence_list" in out
        reent_ev = [e for e in out["evidence_list"] if e.vuln_class == "Reentrancy"]
        assert len(reent_ev) == 1
        assert reent_ev[0].detail.get("debate_verdict") == "CONFIRMED"
        # No debate_transcript because the debate didn't run.
        assert "debate_transcript" not in out

    @pytest.mark.asyncio
    async def test_debate_skipped_with_all_three_tools(self, monkeypatch):
        """
        Strongest agreement: ML + Slither + Aderyn all vote CONFIRMED.
        Still satisfies the multi-tool condition.
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.95, "tier": "CONFIRMED"}],
            consensus_verdict={
                "Reentrancy": {
                    "verdict": "CONFIRMED",
                    "ml_signal": 1, "slither_match": 1, "aderyn_match": 1,
                    "confidence": 1.0,
                },
            },
        )

        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            mock_get_fast.return_value = _mock_llm_with_response("X")
            mock_get_strong.return_value = _mock_llm_with_response("X")
            out = await cross_validator(state)

        assert mock_get_fast.return_value.invoke.call_count == 0
        assert "evidence_list" in out
        reent_ev = [e for e in out["evidence_list"] if e.vuln_class == "Reentrancy"]
        assert len(reent_ev) == 1
        assert reent_ev[0].detail.get("debate_verdict") == "CONFIRMED"

    @pytest.mark.asyncio
    async def test_debate_skipped_with_multiple_classes(self, monkeypatch):
        """
        Setup: TWO flagged classes, BOTH CONFIRMED with 2+ tools each.
        Expected: both debate-skipped, both evidence shows CONFIRMED.
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        state = _make_state(
            flagged=[
                {"vulnerability_class": "Reentrancy",  "probability": 0.85, "tier": "CONFIRMED"},
                {"vulnerability_class": "IntegerUO",  "probability": 0.80, "tier": "CONFIRMED"},
            ],
            consensus_verdict={
                "Reentrancy": {"verdict": "CONFIRMED", "ml_signal": 1, "slither_match": 1, "aderyn_match": 0, "confidence": 0.78},
                "IntegerUO":  {"verdict": "CONFIRMED", "ml_signal": 1, "slither_match": 0, "aderyn_match": 1, "confidence": 0.75},
            },
        )

        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            mock_get_fast.return_value = _mock_llm_with_response("X")
            mock_get_strong.return_value = _mock_llm_with_response("X")
            out = await cross_validator(state)

        assert mock_get_fast.return_value.invoke.call_count == 0
        assert "evidence_list" in out
        reent_ev = [e for e in out["evidence_list"] if e.vuln_class == "Reentrancy"]
        intuo_ev = [e for e in out["evidence_list"] if e.vuln_class == "IntegerUO"]
        assert len(reent_ev) == 1 and reent_ev[0].detail.get("debate_verdict") == "CONFIRMED"
        assert len(intuo_ev) == 1 and intuo_ev[0].detail.get("debate_verdict") == "CONFIRMED"


# ---------------------------------------------------------------------------
# Test 2: debate RUNS when ANY class is below CONFIRMED (asymmetric)
# ---------------------------------------------------------------------------

class TestDebateRuns:
    """
    The asymmetric non-skip cases. The debate MUST run whenever any
    flagged class is below CONFIRMED — even if other classes are CONFIRMED
    with multi-tool agreement. This is the FN/FP asymmetry rule.
    """

    @pytest.mark.asyncio
    async def test_debate_runs_when_one_class_is_likely(self, monkeypatch):
        """
        Setup: one CONFIRMED-with-2-tools, one LIKELY.
        Expected: debate runs (the LIKELY class is below CONFIRMED).
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        state = _make_state(
            flagged=[
                {"vulnerability_class": "Reentrancy",  "probability": 0.85, "tier": "CONFIRMED"},
                {"vulnerability_class": "Timestamp",   "probability": 0.60, "tier": "SUSPICIOUS"},
            ],
            consensus_verdict={
                "Reentrancy": {"verdict": "CONFIRMED", "ml_signal": 1, "slither_match": 1, "aderyn_match": 0, "confidence": 0.78},
                "Timestamp":  {"verdict": "LIKELY",    "ml_signal": 1, "slither_match": 1, "aderyn_match": 0, "confidence": 0.55},
            },
        )

        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            # Mock the debate to return a final verdict.
            mock_get_fast.return_value = _mock_llm_with_response(
                '{"Reentrancy": "CONFIRMED", "Timestamp": "DISPUTED"}'
            )
            mock_get_strong.return_value = _mock_llm_with_response(
                '{"Reentrancy": "CONFIRMED", "Timestamp": "DISPUTED"}'
            )
            out = await cross_validator(state)

        # Debate ran (3 LLM calls: prosecutor, defender, judge).
        assert mock_get_fast.return_value.invoke.call_count == 3
        # Both verdicts came from the LLM, via evidence_list.
        assert "evidence_list" in out
        reent_ev = [e for e in out["evidence_list"] if e.vuln_class == "Reentrancy"]
        ts_ev = [e for e in out["evidence_list"] if e.vuln_class == "Timestamp"]
        assert len(reent_ev) == 1 and reent_ev[0].detail.get("debate_verdict") == "CONFIRMED"
        assert len(ts_ev) == 1 and ts_ev[0].detail.get("debate_verdict") == "DISPUTED"

    @pytest.mark.asyncio
    async def test_debate_runs_when_one_class_disputed(self, monkeypatch):
        """
        Setup: one CONFIRMED-with-2-tools, one DISPUTED.
        DISPUTED is below CONFIRMED → debate runs.
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        state = _make_state(
            flagged=[
                {"vulnerability_class": "Reentrancy",  "probability": 0.85, "tier": "CONFIRMED"},
                {"vulnerability_class": "ExternalBug", "probability": 0.55, "tier": "SUSPICIOUS"},
            ],
            consensus_verdict={
                "Reentrancy":  {"verdict": "CONFIRMED", "ml_signal": 1, "slither_match": 1, "aderyn_match": 0, "confidence": 0.78},
                "ExternalBug": {"verdict": "DISPUTED",  "ml_signal": 1, "slither_match": 0, "aderyn_match": 0, "confidence": 0.40},
            },
        )

        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            mock_get_fast.return_value = _mock_llm_with_response(
                '{"Reentrancy": "CONFIRMED", "ExternalBug": "DISPUTED"}'
            )
            mock_get_strong.return_value = _mock_llm_with_response(
                '{"Reentrancy": "CONFIRMED", "ExternalBug": "DISPUTED"}'
            )
            out = await cross_validator(state)

        assert mock_get_fast.return_value.invoke.call_count == 3
        assert "evidence_list" in out
        ebug_ev = [e for e in out["evidence_list"] if e.vuln_class == "ExternalBug"]
        assert len(ebug_ev) == 1
        assert ebug_ev[0].detail.get("debate_verdict") == "DISPUTED"

    @pytest.mark.asyncio
    async def test_debate_runs_when_consensus_says_safe(self, monkeypatch):
        """
        Asymmetric rule: NEVER skip because cheap signals say "safe".
        If consensus voted SAFE for a flagged class, the debate MUST still
        run — that's exactly where a missed vulnerability could be hiding.
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        # Flagged class with consensus SAFE (no tools, ML below threshold).
        state = _make_state(
            flagged=[{"vulnerability_class": "Timestamp", "probability": 0.45, "tier": "SUSPICIOUS"}],
            consensus_verdict={
                "Timestamp": {
                    "verdict": "SAFE",
                    "ml_signal": 0, "slither_match": 0, "aderyn_match": 0,
                    "confidence": 0.20,
                    "overridden_from_safe": False,
                },
            },
        )

        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            mock_get_fast.return_value = _mock_llm_with_response(
                '{"Timestamp": "DISPUTED"}'
            )
            mock_get_strong.return_value = _mock_llm_with_response(
                '{"Timestamp": "DISPUTED"}'
            )
            out = await cross_validator(state)

        # Debate ran even though consensus said SAFE.
        assert mock_get_fast.return_value.invoke.call_count == 3
        # And the LLM chose DISPUTED (not SAFE) — surfacing uncertainty.

    @pytest.mark.asyncio
    async def test_debate_runs_when_only_one_tool_agrees(self, monkeypatch):
        """
        Setup: consensus says CONFIRMED, but only 1 tool agreed (ml_signal=1,
        slither_match=0, aderyn_match=0). The "multiple independent tools"
        condition is NOT met → debate runs.

        This is the safety margin: a single tool's CONFIRMED verdict is
        not enough to skip the debate.
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        state = _make_state(
            flagged=[{"vulnerability_class": "Timestamp", "probability": 0.85, "tier": "CONFIRMED"}],
            consensus_verdict={
                "Timestamp": {
                    "verdict": "CONFIRMED",
                    "ml_signal": 1, "slither_match": 0, "aderyn_match": 0,
                    "confidence": 0.70,
                },
            },
        )

        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            mock_get_fast.return_value = _mock_llm_with_response('{"Timestamp": "CONFIRMED"}')
            mock_get_strong.return_value = _mock_llm_with_response('{"Timestamp": "CONFIRMED"}')
            await cross_validator(state)

        # Debate ran — only 1 tool agreed, not 2.
        assert mock_get_fast.return_value.invoke.call_count == 3

    @pytest.mark.asyncio
    async def test_debate_runs_when_no_consensus_state(self, monkeypatch):
        """
        No consensus_verdict_state at all → debate runs (no information to skip on).
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            consensus_verdict={},  # empty
        )

        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            mock_get_fast.return_value = _mock_llm_with_response('{"Reentrancy": "CONFIRMED"}')
            mock_get_strong.return_value = _mock_llm_with_response('{"Reentrancy": "CONFIRMED"}')
            await cross_validator(state)

        assert mock_get_fast.return_value.invoke.call_count == 3


# ---------------------------------------------------------------------------
# Test 3: invariants — what the skipped-debate verdict looks like
# ---------------------------------------------------------------------------

class TestSkippedDebateVerdictShape:
    """
    When the debate is skipped, the verdict still flows downstream correctly
    so the rest of the pipeline (synthesizer, etc.) works unchanged.
    """

    @pytest.mark.asyncio
    async def test_skipped_debate_emits_evidence_list(self, monkeypatch):
        """
        The skipped-debate path emits evidence_list with an Evidence item
        per class (confirmations/contradictions are now derived by
        synthesizer from evidence_list).
        """
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")

        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            consensus_verdict={
                "Reentrancy": {"verdict": "CONFIRMED", "ml_signal": 1, "slither_match": 1, "aderyn_match": 0, "confidence": 0.78},
            },
        )

        with patch("src.llm.client.get_fast_llm") as mock_get_fast, \
             patch("src.llm.client.get_strong_llm") as mock_get_strong:
            mock_get_fast.return_value = _mock_llm_with_response("X")
            mock_get_strong.return_value = _mock_llm_with_response("X")
            out = await cross_validator(state)

        assert "evidence_list" in out
        reent_ev = [e for e in out["evidence_list"] if e.vuln_class == "Reentrancy"]
        assert len(reent_ev) == 1
        assert reent_ev[0].source == "debate"
        assert reent_ev[0].detail.get("debate_verdict") == "CONFIRMED"
