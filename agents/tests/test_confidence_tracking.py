"""Tests for A.7 staged confidence tracking (src/orchestration/confidence.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.confidence import track_confidence, confidence_band


class TestTrackConfidence:
    def test_ml_only_returns_clamped_prob(self):
        assert track_confidence(0.6) == 0.6
        assert track_confidence(1.5) == 1.0
        assert track_confidence(-0.2) == 0.0

    def test_slither_agreement_boosts(self):
        base = track_confidence(0.5)
        boosted = track_confidence(0.5, slither_found=True)
        assert boosted > base

    def test_slither_disagreement_shrinks(self):
        base = track_confidence(0.5)
        reduced = track_confidence(0.5, slither_found=False)
        assert reduced < base

    def test_rag_boost_only_above_relevance(self):
        no_rag = track_confidence(0.5, slither_found=True)
        weak_rag = track_confidence(0.5, slither_found=True, rag_score=0.4)
        strong_rag = track_confidence(0.5, slither_found=True, rag_score=0.8)
        assert weak_rag == no_rag  # below relevance floor → no change
        assert strong_rag > no_rag

    def test_always_in_bounds(self):
        for prob in (0.0, 0.3, 0.7, 1.0):
            v = track_confidence(prob, slither_found=True, aderyn_found=True, rag_score=0.9)
            assert 0.0 <= v <= 1.0

    def test_pipeline_flow_ml_slither_rag(self):
        conf = track_confidence(0.7, slither_found=True, aderyn_found=True, rag_score=0.75)
        assert conf > 0.7  # corroboration pushes it up

    def test_band_labels(self):
        assert confidence_band(0.9) == "high"
        assert confidence_band(0.6) == "medium"
        assert confidence_band(0.4) == "low"
        assert confidence_band(0.1) == "negligible"
