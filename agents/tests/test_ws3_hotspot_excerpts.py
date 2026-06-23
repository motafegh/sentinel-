"""
Tests for WS3 (2026-06-22): hotspot-guided excerpts + D4 per-eye clues in
cross_validator.

What WS3 does:
  - Replaces the raw `contract_code[:2000]` char truncation in cross_validator
    with a `graph_explain`-guided excerpt: for each flagged class, pull the
    specific source lines from `ml_hotspots` and present them as the
    primary code view (so the debate reasons over the actual vulnerable
    lines, not a blind prefix).
  - Falls back to the old `[:2000]` truncation when `ml_hotspots` is empty,
    with a note about ML sliding-window count.
  - Adds per-eye clues to the per-class evidence block: each class shows
    its per-eye probability (gnn=X, transformer=Y, fused=Z, phase2=W) and
    which eye is driving suspicion — as discountable hints, NOT votes
    (per the FN/FP asymmetry rule: don't quadruple-count the ML signal).
  - Always includes the full contract source as a reference block (capped
    at 4000 chars) below the focused excerpt.

These tests inspect the prompt the LLM receives, using `mock.invoke.call_args`
to capture the messages and assert on their content.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.nodes import cross_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    flagged: list[dict],
    ml_hotspots: list[dict] | None = None,
    eye_predictions: dict | None = None,
    static_findings: list | None = None,
    contract_code: str = "pragma solidity ^0.8.0;\ncontract C {}\n",
) -> dict:
    ml_result: dict = {
        "confirmed": [v for v in flagged if v.get("tier") == "CONFIRMED"],
        "suspicious": [v for v in flagged if v.get("tier") == "SUSPICIOUS"],
        "vulnerabilities": [v for v in flagged if v.get("tier") in (None, "CONFIRMED")],
    }
    if eye_predictions is not None:
        ml_result["eye_predictions"] = eye_predictions
    return {
        "ml_result":      ml_result,
        "static_findings": static_findings or [],
        "rag_results":     [],
        "audit_history":   [],
        "contract_code":   contract_code,
        "ml_hotspots":     ml_hotspots or [],
    }


def _capture_prompts(monkeypatch, state) -> tuple[str, list[str]]:
    """
    Run cross_validator and return (system_prompt, [prosecutor_prompt,
    defender_prompt, judge_prompt]). Pulled from the LLM's call_args.

    For the prosecutor, the prompt is the second message in the
    HumanMessage/SystemMessage pair passed to llm.invoke.

    Note: the judge prompt only contains the prior prosecutor+defender
    outputs + the evidence block — NOT the source code. Tests that
    check the source/excerpts should look at prosecutor + defender only.
    """
    monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
    monkeypatch.setenv("DEBATE_MODE", "on")

    # Mock the LLM. We need it to return a valid JSON verdict for the
    # judge so the parser doesn't fall over. Prosecutor/defender content
    # doesn't matter for these tests.
    mock_llm = MagicMock()
    responses = [
        MagicMock(content="Prosecutor argument"),
        MagicMock(content="Defender argument"),
        MagicMock(content='{"Reentrancy": "CONFIRMED"}'),
    ]
    mock_llm.invoke.side_effect = responses

    with patch("src.llm.client.get_fast_llm", return_value=mock_llm), \
         patch("src.llm.client.get_strong_llm", return_value=mock_llm):
        import asyncio
        asyncio.run(cross_validator(state))

    # 3 calls: prosecutor, defender, judge
    assert mock_llm.invoke.call_count == 3, \
        f"expected 3 LLM calls, got {mock_llm.invoke.call_count}"
    system = mock_llm.invoke.call_args_list[0][0][0][0].content
    prosecutor = mock_llm.invoke.call_args_list[0][0][0][1].content
    defender   = mock_llm.invoke.call_args_list[1][0][0][1].content
    judge      = mock_llm.invoke.call_args_list[2][0][0][1].content
    return system, {"prosecutor": prosecutor, "defender": defender, "judge": judge}


# ---------------------------------------------------------------------------
# Hotspot excerpts (the main WS3.1/D3 fix)
# ---------------------------------------------------------------------------

class TestHotspotExcerpts:
    """
    The core WS3 fix: the debate should see the flagged functions, not
    a blind `[:2000]` prefix.
    """

    def test_hotspot_excerpts_replace_raw_prefix(self, monkeypatch):
        """
        With ml_hotspots available, the prosecutor prompt should contain
        a "Focused code excerpts" block, NOT just a 2000-char prefix.
        """
        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            ml_hotspots=[
                {
                    "class": "Reentrancy",
                    "fn_name": "withdraw",
                    "lines": [10, 11, 12],
                    "score": 0.85,
                    "signals": ["external_call", "state_write_after_call"],
                },
            ],
            contract_code="pragma solidity ^0.8.0;\n// line 2\n// line 3\n// line 4\n// line 5\n// line 6\n// line 7\n// line 8\n// line 9\nfunction withdraw() public { msg.sender.call(\"\"); balances[msg.sender] = 0; }\n",
        )
        _, prompts = _capture_prompts(monkeypatch, state)

        # Hotspot excerpt is present in prosecutor + defender prompts
        for prompt in (prompts["prosecutor"], prompts["defender"]):
            assert "Focused code excerpts" in prompt
            assert "Reentrancy" in prompt
            assert "withdraw" in prompt
            # The specific source lines should appear
            assert "msg.sender.call" in prompt
            # And the source-line numbers in the excerpt
            assert "10:" in prompt or "11:" in prompt or "12:" in prompt

    def test_hotspot_excerpts_grouped_by_class(self, monkeypatch):
        """
        Multiple classes with hotspots → each gets its own labeled section.
        """
        state = _make_state(
            flagged=[
                {"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"},
                {"vulnerability_class": "Timestamp",  "probability": 0.75, "tier": "CONFIRMED"},
            ],
            ml_hotspots=[
                {"class": "Reentrancy", "fn_name": "withdraw", "lines": [10, 11, 12], "score": 0.85, "signals": []},
                {"class": "Timestamp",  "fn_name": "getPrice", "lines": [20, 21],    "score": 0.72, "signals": []},
            ],
            contract_code=(
                "pragma solidity ^0.8.0;\n"
                + "\n".join(f"// filler line {i}" for i in range(30))
                + "\nfunction withdraw() public { msg.sender.call(\"\"); balances[msg.sender] = 0; }\n"
                + "function getPrice() public view returns (uint) { return uint(block.timestamp) % 100; }\n"
            ),
        )
        _, prompts = _capture_prompts(monkeypatch, state)
        # Both class headers present in the prosecutor prompt
        assert "── Reentrancy ──" in prompts["prosecutor"]
        assert "── Timestamp ──" in prompts["prosecutor"]

    def test_full_source_included_as_reference(self, monkeypatch):
        """
        Even with hotspot excerpts, the full source is included below as
        reference (capped at 4000 chars).
        """
        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            ml_hotspots=[
                {"class": "Reentrancy", "fn_name": "withdraw", "lines": [10, 11, 12], "score": 0.85, "signals": []},
            ],
            contract_code="pragma solidity ^0.8.0;\n" + "\n".join(f"// line {i}" for i in range(50)) + "\nfunction withdraw() public { msg.sender.call(\"\"); }\n",
        )
        _, prompts = _capture_prompts(monkeypatch, state)
        # Source/excerpts are seen by prosecutor + defender (not judge)
        for prompt in (prompts["prosecutor"], prompts["defender"]):
            assert "Full contract source (for reference)" in prompt
            assert "pragma solidity" in prompt

    def test_fallback_to_raw_truncation_when_no_hotspots(self, monkeypatch):
        """
        When ml_hotspots is empty, fall back to the old [:2000] truncation
        approach (with a note about ML windowing if applicable).
        """
        long_source = "pragma solidity ^0.8.0;\n" + "\n".join(f"// filler {i}" for i in range(50))
        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            ml_hotspots=[],  # no hotspots
            contract_code=long_source,
        )
        # Tell the ML result that the contract was truncated into multiple windows,
        # so the fallback note is triggered.
        state["ml_result"]["windows_used"] = 3
        state["ml_result"]["truncated"] = True

        _, prompts = _capture_prompts(monkeypatch, state)
        # Source/excerpts are seen by prosecutor + defender (not judge)
        for prompt in (prompts["prosecutor"], prompts["defender"]):
            # No "Focused code excerpts" header
            assert "Focused code excerpts" not in prompt
            # Old "Contract source" header is used
            assert "Contract source" in prompt
            # Note about sliding windows
            assert "sliding window" in prompt

    def test_no_fallback_note_when_single_window(self, monkeypatch):
        """
        When ml_hotspots is empty AND ML used 1 window, the fallback note
        is suppressed (no useful info to add).
        """
        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            ml_hotspots=[],
            contract_code="pragma solidity ^0.8.0;\ncontract C {}",
        )
        state["ml_result"]["windows_used"] = 1

        _, prompts = _capture_prompts(monkeypatch, state)
        # Source/excerpts are seen by prosecutor + defender (not judge)
        for prompt in (prompts["prosecutor"], prompts["defender"]):
            assert "Focused code excerpts" not in prompt
            assert "Contract source" in prompt
            # No "sliding window" note when there was 1 window
            assert "sliding window" not in prompt


# ---------------------------------------------------------------------------
# D4 per-eye clues
# ---------------------------------------------------------------------------

class TestD4EyeClues:
    """
    D4: per-eye auxiliary predictions (gnn, transformer, fused, phase2) as
    discountable clues — hints, not votes. They tell the debate WHICH
    reasoning drives the model's suspicion.
    """

    def test_eye_clues_appear_in_evidence_block(self, monkeypatch):
        """
        When eye_predictions is available, each class line should include
        an "Eye clues:" line showing the per-eye probabilities and which
        eye is driving.
        """
        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            eye_predictions={
                "gnn":         {"Reentrancy": 0.81, "Timestamp": 0.12},
                "transformer": {"Reentrancy": 0.32, "Timestamp": 0.45},
                "fused":       {"Reentrancy": 0.72, "Timestamp": 0.21},
                "phase2":      {"Reentrancy": 0.55, "Timestamp": 0.10},
            },
        )
        _, prompts = _capture_prompts(monkeypatch, state)
        for prompt in prompts.values():
            # "Eye clues:" prefix is added
            assert "Eye clues:" in prompt
            # All 4 eyes shown
            assert "gnn=" in prompt
            assert "transformer=" in prompt
            assert "fused=" in prompt
            assert "phase2=" in prompt
            # The driving eye is marked (gnn=0.81 is highest → "gnn eye driving")
            assert "gnn eye driving" in prompt

    def test_no_eye_clues_when_predictions_missing(self, monkeypatch):
        """
        When eye_predictions is absent from ml_result, the Eye clues: line
        is omitted (graceful degradation, not a crash).
        """
        state = _make_state(
            flagged=[{"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"}],
            # eye_predictions NOT in ml_result
        )
        _, prompts = _capture_prompts(monkeypatch, state)
        for prompt in prompts.values():
            assert "Eye clues:" not in prompt

    def test_eye_clues_only_for_relevant_class(self, monkeypatch):
        """
        Each class gets its own eye-clue line with the probabilities for
        THAT class (not all 10 classes' worth of data per class).
        """
        state = _make_state(
            flagged=[
                {"vulnerability_class": "Reentrancy", "probability": 0.85, "tier": "CONFIRMED"},
                {"vulnerability_class": "Timestamp",  "probability": 0.60, "tier": "SUSPICIOUS"},
            ],
            eye_predictions={
                "gnn":         {"Reentrancy": 0.81, "Timestamp": 0.12},
                "transformer": {"Reentrancy": 0.32, "Timestamp": 0.45},
                "fused":       {"Reentrancy": 0.72, "Timestamp": 0.21},
                "phase2":      {"Reentrancy": 0.55, "Timestamp": 0.10},
            },
        )
        _, prompts = _capture_prompts(monkeypatch, state)
        for prompt in prompts.values():
            # The Reentrancy class line should mention gnn=0.81
            assert "0.81" in prompt
            # And the Timestamp class line should mention transformer=0.45
            assert "0.45" in prompt
            # The driving eyes are different per class
            assert "gnn eye driving" in prompt   # for Reentrancy
            assert "transformer eye driving" in prompt  # for Timestamp
