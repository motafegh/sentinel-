"""Tests for A.3 reflection agent + A.4 debate (src/orchestration/nodes.py)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.nodes import reflection, cross_validator


def _report_with_verdicts():
    return {
        "final_report": {
            "vulnerability_verdicts": [
                {"vulnerability_class": "Reentrancy", "probability": 0.8, "verdict": "LIKELY"},
                {"vulnerability_class": "ExternalBug", "probability": 0.6, "verdict": "DISPUTED"},
            ],
            "ml_truncated": False,
            "verdicts": {"Reentrancy": "LIKELY", "ExternalBug": "DISPUTED"},
        },
        "static_findings": [
            {"tool": "slither", "detector": "timestamp", "impact": "High"},
        ],
        "rag_results": [{"metadata": {"vulnerability_type": "Reentrancy"}, "score": 0.7}],
        "contradictions": {"ExternalBug": ["ml_flagged but no corroboration"]},
        "confidence_by_class": {"Reentrancy": 0.85, "ExternalBug": 0.4},
    }


class TestReflectionRuleBased:
    @pytest.mark.asyncio
    async def test_rule_based_when_llm_disabled(self, monkeypatch):
        monkeypatch.setenv("AGENTS_DISABLE_LLM", "1")
        out = await reflection(_report_with_verdicts())
        notes = out["reflection_notes"]
        assert notes["llm_used"] is False
        # ExternalBug DISPUTED + low confidence → uncertain
        assert any("ExternalBug" in u for u in notes["uncertain_verdicts"])
        # ExternalBug present → failure-mode note about ML over-prediction
        assert any("ExternalBug" in f for f in notes["failure_modes"])
        # contradiction surfaced
        assert any("ExternalBug" in c for c in notes["contradictions"])
        # uncited High static finding (timestamp) flagged as unused
        assert notes["unused_evidence"]
        assert notes["summary"]

    @pytest.mark.asyncio
    async def test_truncated_adds_failure_mode(self, monkeypatch):
        monkeypatch.setenv("AGENTS_DISABLE_LLM", "1")
        state = _report_with_verdicts()
        state["final_report"]["ml_truncated"] = True
        out = await reflection(state)
        assert any("512" in f or "truncat" in f.lower() for f in out["reflection_notes"]["failure_modes"])

    @pytest.mark.asyncio
    async def test_empty_state_is_safe(self, monkeypatch):
        monkeypatch.setenv("AGENTS_DISABLE_LLM", "1")
        out = await reflection({})
        assert "reflection_notes" in out
        assert out["reflection_notes"]["summary"]


class TestReflectionLLM:
    @pytest.mark.asyncio
    async def test_llm_summary_used_when_enabled(self, monkeypatch):
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "The ExternalBug verdict is likely a false positive given no call evidence."
        mock_llm.invoke.return_value = mock_resp
        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            out = await reflection(_report_with_verdicts())
        notes = out["reflection_notes"]
        assert notes["llm_used"] is True
        assert "false positive" in notes["summary"]

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self, monkeypatch):
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        with patch("src.llm.client.get_strong_llm", side_effect=RuntimeError("down")):
            out = await reflection(_report_with_verdicts())
        notes = out["reflection_notes"]
        assert notes["llm_used"] is False
        assert notes["summary"]  # rule-based summary still present


class TestDebateMode:
    @pytest.mark.asyncio
    async def test_debate_runs_three_roles(self, monkeypatch):
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")
        monkeypatch.setenv("CROSS_VALIDATOR_LLM_MODEL", "strong")
        mock_llm = MagicMock()
        # prosecutor, defender, judge — judge must be parseable JSON
        responses = [
            MagicMock(content="It is vulnerable because X."),
            MagicMock(content="It may be a false positive because Y."),
            MagicMock(content='{"Reentrancy": "CONFIRMED"}'),
        ]
        mock_llm.invoke.side_effect = responses
        state = {
            "ml_result": {"confirmed": [{"vulnerability_class": "Reentrancy", "probability": 0.9, "tier": "CONFIRMED"}],
                          "suspicious": []},
            "static_findings": [], "rag_results": [], "audit_history": [],
            "contract_code": "contract C { function f() public { msg.sender.call(''); } }",
        }
        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            out = await cross_validator(state)
        assert mock_llm.invoke.call_count == 3
        assert out["verdicts"]["Reentrancy"] == "CONFIRMED"
        assert "debate_transcript" in out
        assert set(out["debate_transcript"]) == {"prosecutor", "defender", "judge"}

    @pytest.mark.asyncio
    async def test_llm_disabled_skips_debate(self, monkeypatch):
        monkeypatch.setenv("AGENTS_DISABLE_LLM", "1")
        state = {
            "ml_result": {"confirmed": [{"vulnerability_class": "Reentrancy", "probability": 0.9, "tier": "CONFIRMED"}],
                          "suspicious": []},
        }
        out = await cross_validator(state)
        assert out == {}

    @pytest.mark.asyncio
    async def test_debate_bounded_by_one_outer_timeout(self, monkeypatch):
        # Real-audit finding (2026-06-21): an earlier version applied
        # CROSS_VALIDATOR_TIMEOUT_S to EACH of the 3 sequential debate calls
        # (up to 3x90s=270s worst case), which exceeded external script
        # timeouts. Verify the debate now respects ONE outer DEBATE_TIMEOUT_S
        # regardless of how slow individual calls are.
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "on")
        monkeypatch.setenv("DEBATE_TIMEOUT_S", "0.2")
        monkeypatch.setenv("CROSS_VALIDATOR_LLM_MODEL", "strong")

        def _slow_invoke(_messages):
            import time
            time.sleep(1)  # slower than the 0.2s outer debate budget
            return MagicMock(content='{"Reentrancy": "CONFIRMED"}')

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _slow_invoke
        state = {
            "ml_result": {"confirmed": [{"vulnerability_class": "Reentrancy", "probability": 0.9, "tier": "CONFIRMED"}],
                          "suspicious": []},
        }
        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            out = await cross_validator(state)
        # Times out on the FIRST call (prosecutor) well before 3x90s would
        # have elapsed under the old per-call scheme — falls back to {}.
        assert out == {}
