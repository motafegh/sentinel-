"""Tests for A.8 metric attribution (src/orchestration/attribution.py) + explainer node."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.attribution import attribute_verdict
from src.orchestration.nodes import explainer


class TestAttributeVerdict:
    def test_percentages_sum_to_100(self):
        a = attribute_verdict(0.8, True, 0.6)
        assert a["ml_pct"] + a["slither_pct"] + a["rag_pct"] == pytest.approx(100.0, abs=0.1)

    def test_no_evidence_returns_zeros(self):
        a = attribute_verdict(0.0, False, 0.0)
        assert a == {"ml_pct": 0.0, "slither_pct": 0.0, "rag_pct": 0.0}

    def test_slither_only(self):
        a = attribute_verdict(0.0, True, 0.0)
        assert a["slither_pct"] == 100.0
        assert a["ml_pct"] == 0.0

    def test_subfloor_rag_ignored(self):
        a = attribute_verdict(0.5, False, 0.2)  # rag below 0.30 floor
        assert a["rag_pct"] == 0.0
        assert a["ml_pct"] == 100.0

    def test_ml_dominant_when_strong(self):
        a = attribute_verdict(0.9, False, 0.0)
        assert a["ml_pct"] == 100.0


class TestExplainerNode:
    @pytest.mark.asyncio
    async def test_attributes_and_folds_into_report(self):
        state = {
            "final_report": {
                "vulnerability_verdicts": [
                    {"vulnerability_class": "Reentrancy", "probability": 0.8, "verdict": "LIKELY"},
                ],
            },
            "static_findings": [
                {"tool": "slither", "detector": "reentrancy-eth", "impact": "High"},
            ],
            "rag_results": [
                {"metadata": {"vulnerability_type": "Reentrancy"}, "score": 0.7},
            ],
            "confidence_by_class": {"Reentrancy": 0.85},
            "consensus_verdict": {"Reentrancy": {"verdict": "LIKELY"}},
            "reflection_notes": {"summary": "ok"},
        }
        out = await explainer(state)
        assert "metric_attribution" in out
        assert "Reentrancy" in out["metric_attribution"]
        report = out["final_report"]
        assert report["confidence_by_class"]["Reentrancy"] == 0.85
        assert report["metric_attribution"]["Reentrancy"]["slither_pct"] > 0
        # verdict row annotated in place
        assert report["vulnerability_verdicts"][0]["attribution"]
        assert report["vulnerability_verdicts"][0]["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_empty_report_is_safe(self):
        out = await explainer({})
        assert out["metric_attribution"] == {}
