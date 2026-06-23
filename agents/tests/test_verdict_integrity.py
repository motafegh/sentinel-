"""
WS1 regression tests — verdict integrity & FN/FP safety net.

Tests the core WS1 guarantees:
  1. compute_verdict() returns INCONCLUSIVE (not SAFE) for flagged classes
  2. compute_verdict() returns SAFE for below-threshold classes
  3. consensus_engine votes on every flagged class (no silent skip)
  4. consensus_engine overrides SAFE → DISPUTED for flagged classes
  5. quick_screen Aderyn escalates on Medium/Critical (not just High)
  6. OVERALL_VERDICT_RANK includes INCONCLUSIVE + WATCH
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.routing import (
    compute_verdict,
    compute_overall_verdict,
    OVERALL_VERDICT_RANK,
    DEEP_THRESHOLDS,
)
from src.orchestration.nodes import consensus_engine


# ── compute_verdict tests ────────────────────────────────────────────────────

class TestComputeVerdictWS1:
    def test_flagged_no_evidence_is_inconclusive_not_safe(self):
        """A class above DEEP_THRESHOLD with no corroboration → INCONCLUSIVE,
        not SAFE. This is the core WS1 fix for Finding #9."""
        verdict, sources = compute_verdict(
            "Reentrancy", 0.42, [], [], "deep"
        )
        assert verdict == "INCONCLUSIVE", (
            f"Expected INCONCLUSIVE for flagged class (prob=0.42 >= "
            f"DEEP_THRESHOLD={DEEP_THRESHOLDS['Reentrancy']}), got {verdict}"
        )

    def test_below_threshold_is_safe(self):
        """A class below DEEP_THRESHOLD → SAFE (genuinely not flagged)."""
        verdict, sources = compute_verdict(
            "Reentrancy", 0.20, [], [], "deep"
        )
        assert verdict == "SAFE"

    def test_flagged_with_slither_is_confirmed(self):
        """A flagged class with Slither corroboration → CONFIRMED (unchanged)."""
        findings = [{"detector": "reentrancy-eth", "impact": "High"}]
        verdict, sources = compute_verdict(
            "Reentrancy", 0.60, findings, [], "deep"
        )
        assert verdict == "CONFIRMED"

    def test_flagged_no_evidence_higher_band_is_inconclusive(self):
        """A class at 0.48 (above 0.35, below 0.50) → INCONCLUSIVE."""
        verdict, _ = compute_verdict(
            "Reentrancy", 0.48, [], [], "deep"
        )
        assert verdict == "INCONCLUSIVE"

    def test_fast_path_still_likely(self):
        """Fast path → LIKELY (unchanged by WS1)."""
        verdict, _ = compute_verdict("Reentrancy", 0.60, [], [], "fast")
        assert verdict == "LIKELY"

    def test_just_at_deep_threshold_is_inconclusive(self):
        """A class exactly at DEEP_THRESHOLD with no evidence → INCONCLUSIVE
        (>= is the condition, so the boundary is included)."""
        threshold = DEEP_THRESHOLDS["Reentrancy"]
        verdict, _ = compute_verdict(
            "Reentrancy", threshold, [], [], "deep"
        )
        assert verdict == "INCONCLUSIVE"

    def test_just_below_deep_threshold_is_safe(self):
        """A class just below DEEP_THRESHOLD → SAFE."""
        threshold = DEEP_THRESHOLDS["Reentrancy"]
        verdict, _ = compute_verdict(
            "Reentrancy", threshold - 0.01, [], [], "deep"
        )
        assert verdict == "SAFE"


# ── OVERALL_VERDICT_RANK tests ───────────────────────────────────────────────

class TestOverallVerdictRank:
    def test_inconclusive_present(self):
        assert "INCONCLUSIVE" in OVERALL_VERDICT_RANK

    def test_watch_present(self):
        assert "WATCH" in OVERALL_VERDICT_RANK

    def test_inconclusive_ranks_above_safe(self):
        assert OVERALL_VERDICT_RANK["INCONCLUSIVE"] > OVERALL_VERDICT_RANK["SAFE"]

    def test_inconclusive_ranks_below_disputed(self):
        assert OVERALL_VERDICT_RANK["INCONCLUSIVE"] < OVERALL_VERDICT_RANK["DISPUTED"]

    def test_overall_with_inconclusive_not_safe(self):
        """If any class is INCONCLUSIVE, overall verdict should not be SAFE."""
        verdicts = {"Reentrancy": "INCONCLUSIVE", "IntegerUO": "SAFE"}
        overall = compute_overall_verdict(verdicts)
        assert overall == "INCONCLUSIVE"

    def test_overall_disputed_beats_inconclusive(self):
        verdicts = {"Reentrancy": "DISPUTED", "IntegerUO": "INCONCLUSIVE"}
        overall = compute_overall_verdict(verdicts)
        assert overall == "DISPUTED"


# ── consensus_engine WS1 tests ───────────────────────────────────────────────

class TestConsensusEngineWS1:
    @pytest.mark.asyncio
    async def test_votes_on_borderline_no_corroboration(self):
        """WS1: class at 0.42 (above DEEP_THRESHOLD 0.35) with no tools → gets a vote."""
        state = {
            "ml_result": {"probabilities": {"Reentrancy": 0.42}},
            "static_findings": [],
        }
        out = await consensus_engine(state)
        assert "Reentrancy" in out["consensus_verdict"]

    @pytest.mark.asyncio
    async def test_overrides_safe_to_disputed(self):
        """WS1: consensus_vote returns SAFE but class is flagged → DISPUTED."""
        state = {
            "ml_result": {"probabilities": {"Reentrancy": 0.42}},
            "static_findings": [],
        }
        out = await consensus_engine(state)
        vote = out["consensus_verdict"]["Reentrancy"]
        assert vote["verdict"] == "DISPUTED"
        assert vote.get("overridden_from_safe") is True

    @pytest.mark.asyncio
    async def test_does_not_override_when_below_threshold(self):
        """WS1: class below DEEP_THRESHOLD with no tools → skipped (not overridden)."""
        state = {
            "ml_result": {"probabilities": {"Reentrancy": 0.20}},
            "static_findings": [],
        }
        out = await consensus_engine(state)
        assert "Reentrancy" not in out.get("consensus_verdict", {})

    @pytest.mark.asyncio
    async def test_tool_hit_below_threshold_still_votes(self):
        """WS1: class below DEEP_THRESHOLD but with a tool hit → still gets a vote
        (tool corroboration is an independent signal)."""
        state = {
            "ml_result": {"probabilities": {"Reentrancy": 0.20}},
            "static_findings": [
                {"tool": "slither", "detector": "reentrancy-eth", "impact": "High"},
            ],
        }
        out = await consensus_engine(state)
        assert "Reentrancy" in out["consensus_verdict"]

    @pytest.mark.asyncio
    async def test_all_classes_above_threshold_get_votes(self):
        """WS1: every class above its DEEP_THRESHOLD gets a vote, even without tools."""
        state = {
            "ml_result": {
                "probabilities": {
                    "Reentrancy": 0.42,
                    "Timestamp": 0.36,
                    "ExternalBug": 0.41,
                }
            },
            "static_findings": [],
        }
        out = await consensus_engine(state)
        cv = out["consensus_verdict"]
        assert "Reentrancy" in cv
        assert "Timestamp" in cv  # 0.36 >= DEEP_THRESHOLD 0.35
        assert "ExternalBug" in cv  # 0.41 >= DEEP_THRESHOLD 0.40
        # All should be DISPUTED (flagged but no corroboration)
        for cls in ("Reentrancy", "Timestamp", "ExternalBug"):
            assert cv[cls]["verdict"] == "DISPUTED", (
                f"{cls} should be DISPUTED (flagged, no corroboration), "
                f"got {cv[cls]['verdict']}"
            )


# ── quick_screen Aderyn impact test ──────────────────────────────────────────

class TestQuickScreenAderynImpact:
    def test_aderyn_medium_impact_in_escalation_levels(self):
        """WS1: Aderyn Medium-impact findings should trigger quick_screen escalation,
        matching Slither's High/Medium/Critical levels (Finding #11 fix)."""
        # Read the source to verify the condition includes Medium
        import inspect
        from src.orchestration import nodes as nodes_mod
        source = inspect.getsource(nodes_mod.quick_screen)
        # The Aderyn escalation condition should include "Medium" and "Critical"
        assert '("High", "Medium", "Critical")' in source, (
            "Aderyn escalation should check High/Medium/Critical (Finding #11 fix)"
        )
