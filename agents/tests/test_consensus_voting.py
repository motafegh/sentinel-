"""
Tests for A.6 tool consensus voting (src/orchestration/consensus.py) and the
consensus_engine node, including Ali's ML-as-a-hint down-weighting directive.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.consensus import (
    consensus_vote,
    get_weights,
    ACCURACY_WEIGHTS,
)
from src.orchestration.nodes import consensus_engine


class TestConsensusVote:
    def test_all_agree_confirms(self):
        r = consensus_vote(0.9, True, True, "Reentrancy")
        assert r["verdict"] == "CONFIRMED"
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["ml_signal"] == 1 and r["slither_match"] == 1 and r["aderyn_match"] == 1

    def test_ml_alone_cannot_confirm(self):
        # Ali directive: ML is a hint — ML-only must not reach CONFIRMED.
        for cls in ACCURACY_WEIGHTS:
            r = consensus_vote(0.99, False, False, cls)
            assert r["verdict"] != "CONFIRMED", f"{cls} confirmed on ML alone"

    def test_ml_plus_one_strong_tool_escalates(self):
        ml_only = consensus_vote(0.9, False, False, "Reentrancy")["confidence"]
        ml_slither = consensus_vote(0.9, True, False, "Reentrancy")["confidence"]
        assert ml_slither > ml_only

    def test_no_signals_is_safe(self):
        r = consensus_vote(0.1, False, False, "Timestamp")
        assert r["verdict"] == "SAFE"
        assert r["confidence"] == 0.0

    def test_confidence_bounds(self):
        for prob in (0.0, 0.5, 1.0):
            for s in (True, False):
                for a in (True, False):
                    r = consensus_vote(prob, s, a, "ExternalBug")
                    assert 0.0 <= r["confidence"] <= 1.0

    def test_unknown_class_uses_default_weights(self):
        r = consensus_vote(0.9, True, True, "TotallyNewClass")
        assert r["verdict"] in {"CONFIRMED", "LIKELY", "DISPUTED", "SAFE"}
        assert set(r["weights"]) == {"ml", "slither", "aderyn"}

    def test_ml_weight_scaled_down(self, monkeypatch):
        monkeypatch.setenv("ML_WEIGHT_SCALE", "0.5")
        w = get_weights("Reentrancy")
        assert w["ml"] == pytest.approx(ACCURACY_WEIGHTS["Reentrancy"]["ml"] * 0.5, abs=1e-3)
        # slither/aderyn untouched
        assert w["slither"] == ACCURACY_WEIGHTS["Reentrancy"]["slither"]

    def test_ml_scale_one_restores_weight(self, monkeypatch):
        monkeypatch.setenv("ML_WEIGHT_SCALE", "1.0")
        w = get_weights("Reentrancy")
        assert w["ml"] == pytest.approx(ACCURACY_WEIGHTS["Reentrancy"]["ml"], abs=1e-3)


class TestConsensusEngineNode:
    @pytest.mark.asyncio
    async def test_emits_rows_for_flagged_and_tool_hits(self):
        state = {
            "ml_result": {"probabilities": {"Reentrancy": 0.8, "Timestamp": 0.1}},
            "static_findings": [
                {"tool": "slither", "detector": "reentrancy-eth", "impact": "High"},
            ],
        }
        out = await consensus_engine(state)
        assert "consensus_verdict" in out
        assert "Reentrancy" in out["consensus_verdict"]
        # Timestamp at 0.1 with no tool hit → not emitted
        assert "Timestamp" not in out["consensus_verdict"]
        assert "confidence_by_class" in out
        assert 0.0 <= out["confidence_by_class"]["Reentrancy"] <= 1.0

    @pytest.mark.asyncio
    async def test_no_probabilities_returns_empty(self):
        out = await consensus_engine({"ml_result": {}})
        assert out == {}

    @pytest.mark.asyncio
    async def test_falls_back_to_flagged_list(self):
        state = {
            "ml_result": {
                "confirmed": [{"vulnerability_class": "Reentrancy", "probability": 0.9}],
                "suspicious": [],
            },
            "static_findings": [
                {"tool": "slither", "detector": "reentrancy-eth", "impact": "High"},
            ],
        }
        out = await consensus_engine(state)
        assert "Reentrancy" in out["consensus_verdict"]
