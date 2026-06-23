"""
WS1.5 regression tests — verdict reconciliation.

Tests the 8-case reconciliation function (_reconcile_verdicts) that replaced
the old "debate wins, then consensus, then compute_verdict" priority chain.
The core invariant: the debate can UPGRADE but can only DOWNGRADE to DISPUTED,
never to SAFE, when consensus voted non-SAFE.

See docs/plan/agents/2026-06-21-agents-redesign/05_VERDICT_RECONCILIATION_PLAN.md
for the full 8-case table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.nodes import _reconcile_verdicts


# ── Helpers ──────────────────────────────────────────────────────────────────

def _consensus(verdict: str, confidence: float = 0.5, **extra) -> dict:
    """Build a consensus_verdict[cls] dict."""
    base = {
        "verdict": verdict,
        "confidence": confidence,
        "ml_signal": 1,
        "slither_match": 0,
        "aderyn_match": 0,
        "score": 0.0,
        "weights": {},
    }
    base.update(extra)
    return base


# ── The 8 reconciliation cases ───────────────────────────────────────────────

class TestReconciliationCases:
    """One test per case in the 8-case table."""

    def test_case1_consensus_confirmed_debate_cannot_clear(self):
        """Case 1a: consensus CONFIRMED (conf≥0.70) + debate SAFE → stays CONFIRMED.
        All tools agreed — the debate cannot CLEAR a unanimously tool-corroborated class."""
        result, sources = _reconcile_verdicts(
            "Reentrancy", 0.82,
            _consensus("CONFIRMED", confidence=1.0, slither_match=1, aderyn_match=1),
            "SAFE", [], [], "deep",
        )
        assert result == "CONFIRMED"
        assert "consensus_confirmed" in " ".join(sources)

    def test_case1b_confirmed_vs_disputed_surfaces_uncertainty(self):
        """Case 1b: consensus CONFIRMED + debate DISPUTED → DISPUTED.
        The debate read the source and expresses uncertainty — surface it,
        don't hide it. DISPUTED is not "cleared," it's "uncertain, investigate."
        Fixes the FP on 05_unexpected_revert_dos/Reentrancy."""
        result, sources = _reconcile_verdicts(
            "Reentrancy", 0.82,
            _consensus("CONFIRMED", confidence=1.0, slither_match=1, aderyn_match=1),
            "DISPUTED", [], [], "deep",
        )
        assert result == "DISPUTED"
        assert "uncertainty" in " ".join(sources)

    def test_case1c_confirmed_vs_watch_stays_confirmed(self):
        """Case 1a variant: consensus CONFIRMED + debate WATCH → CONFIRMED.
        WATCH = "weak signal, monitor only" — the debate had nothing useful."""
        result, _ = _reconcile_verdicts(
            "Reentrancy", 0.82,
            _consensus("CONFIRMED", confidence=0.85),
            "WATCH", [], [], "deep",
        )
        assert result == "CONFIRMED"

    def test_case1d_confirmed_vs_inconclusive_stays_confirmed(self):
        """Case 1a variant: consensus CONFIRMED + debate INCONCLUSIVE → CONFIRMED.
        The debate couldn't adjudicate — consensus stands."""
        result, _ = _reconcile_verdicts(
            "Reentrancy", 0.82,
            _consensus("CONFIRMED", confidence=0.85),
            "INCONCLUSIVE", [], [], "deep",
        )
        assert result == "CONFIRMED"

    def test_case2_likely_vs_safe_disputes(self):
        """Case 2: consensus LIKELY + debate SAFE → DISPUTED (surface disagreement)."""
        result, sources = _reconcile_verdicts(
            "ExternalBug", 0.64,
            _consensus("LIKELY", confidence=0.57),
            "SAFE", [], [], "deep",
        )
        assert result == "DISPUTED"
        assert "disagreement" in " ".join(sources)

    def test_case3_likely_vs_disputed_agrees_on_disputed(self):
        """Case 3: consensus LIKELY + debate DISPUTED → DISPUTED."""
        result, _ = _reconcile_verdicts(
            "ExternalBug", 0.64,
            _consensus("LIKELY", confidence=0.57),
            "DISPUTED", [], [], "deep",
        )
        assert result == "DISPUTED"

    def test_case4_disputed_vs_safe_stays_disputed(self):
        """Case 4: consensus DISPUTED + debate SAFE → DISPUTED (uncorroborated ≠ cleared)."""
        result, sources = _reconcile_verdicts(
            "IntegerUO", 0.42,
            _consensus("DISPUTED", confidence=0.30),
            "SAFE", [], [], "deep",
        )
        assert result == "DISPUTED"
        assert "not_cleared" in " ".join(sources)

    def test_case5_disputed_vs_confirmed_upgrades(self):
        """Case 5: consensus DISPUTED + debate CONFIRMED → CONFIRMED (debate upgrade).
        The debate found something the tools didn't — this is the debate's value."""
        result, sources = _reconcile_verdicts(
            "Reentrancy", 0.45,
            _consensus("DISPUTED", confidence=0.30),
            "CONFIRMED", [], [], "deep",
        )
        assert result == "CONFIRMED"
        assert "debate_upgrade" in " ".join(sources)

    def test_case5b_disputed_vs_likely_upgrades(self):
        """Case 5 variant: consensus DISPUTED + debate LIKELY → LIKELY."""
        result, _ = _reconcile_verdicts(
            "Reentrancy", 0.45,
            _consensus("DISPUTED", confidence=0.30),
            "LIKELY", [], [], "deep",
        )
        assert result == "LIKELY"

    def test_case6_consensus_only_when_debate_silent(self):
        """Case 6: consensus voted, debate was None (empty/timeout) → consensus stands."""
        result, sources = _reconcile_verdicts(
            "Reentrancy", 0.82,
            _consensus("LIKELY", confidence=0.57),
            None, [], [], "deep",
        )
        assert result == "LIKELY"
        assert "consensus" in " ".join(sources)

    def test_case7_debate_only_when_consensus_silent(self):
        """Case 7: no consensus vote → debate is the only signal."""
        result, sources = _reconcile_verdicts(
            "Reentrancy", 0.30,
            None,
            "CONFIRMED", [], [], "deep",
        )
        assert result == "CONFIRMED"
        assert "debate" in " ".join(sources)

    def test_case8_compute_verdict_when_neither(self):
        """Case 8: neither consensus nor debate voted → compute_verdict (last resort).
        For a below-threshold class, compute_verdict returns SAFE."""
        result, _ = _reconcile_verdicts(
            "Reentrancy", 0.20,
            None,
            None, [], [], "deep",
        )
        assert result == "SAFE"

    def test_case8b_compute_verdict_inconclusive_for_flagged(self):
        """Case 8 variant: neither voted + class is flagged (above DEEP_THRESHOLD)
        → compute_verdict returns INCONCLUSIVE (WS1's change)."""
        result, _ = _reconcile_verdicts(
            "Reentrancy", 0.42,
            None,
            None, [], [], "deep",
        )
        assert result == "INCONCLUSIVE"


# ── The core invariants ──────────────────────────────────────────────────────

class TestReconciliationInvariants:
    """The rules that must ALWAYS hold, regardless of specific case."""

    def test_debate_cannot_safe_a_consensus_flagged_class(self):
        """CORE INVARIANT: when consensus says non-SAFE, debate can never make it SAFE.
        This is the FN/FP asymmetry principle's enforcement at the reconciliation point.
        The debate CAN downgrade to DISPUTED (uncertainty) but NEVER to SAFE (clearing)."""
        for cv_verdict in ("CONFIRMED", "LIKELY", "DISPUTED"):
            for cv_conf in (0.30, 0.50, 0.70, 1.0):
                result, _ = _reconcile_verdicts(
                    "Reentrancy", 0.60,
                    _consensus(cv_verdict, confidence=cv_conf),
                    "SAFE", [], [], "deep",
                )
                assert result != "SAFE", (
                    f"consensus={cv_verdict}(conf={cv_conf}) + debate=SAFE → "
                    f"got {result}, must NOT be SAFE (debate cannot clear a "
                    f"consensus-flagged class)"
                )

    def test_confidence_1_0_never_cleared_to_safe(self):
        """Finding #14 worst case: consensus CONFIRMED at confidence=1.0 (all tools
        agreed) must NEVER be cleared to SAFE by the debate. But the debate CAN
        express uncertainty (DISPUTED) — that's allowed, just not clearing."""
        for debate in ("SAFE", "WATCH", "INCONCLUSIVE"):
            result, _ = _reconcile_verdicts(
                "ExternalBug", 0.82,
                _consensus("CONFIRMED", confidence=1.0, slither_match=1, aderyn_match=1),
                debate, [], [], "deep",
            )
            assert result == "CONFIRMED", (
                f"consensus CONFIRMED conf=1.0 + debate={debate} → "
                f"got {result}, must stay CONFIRMED (debate cannot clear)"
            )
        # DISPUTED is allowed through (surfaces uncertainty, not clearing)
        result, _ = _reconcile_verdicts(
            "ExternalBug", 0.82,
            _consensus("CONFIRMED", confidence=1.0, slither_match=1, aderyn_match=1),
            "DISPUTED", [], [], "deep",
        )
        assert result == "DISPUTED", (
            f"consensus CONFIRMED conf=1.0 + debate=DISPUTED → "
            f"got {result}, should be DISPUTED (uncertainty surfaced)"
        )

    def test_both_agree_returns_agreement(self):
        """When consensus and debate agree, return that verdict."""
        for agreed in ("CONFIRMED", "LIKELY", "DISPUTED", "SAFE", "WATCH", "INCONCLUSIVE"):
            result, _ = _reconcile_verdicts(
                "Reentrancy", 0.60,
                _consensus(agreed, confidence=0.50),
                agreed, [], [], "deep",
            )
            assert result == agreed

    def test_debate_can_upgrade_disputed_to_confirmed(self):
        """The debate CAN upgrade — this is its value (it reads the source)."""
        result, _ = _reconcile_verdicts(
            "Reentrancy", 0.45,
            _consensus("DISPUTED", confidence=0.30),
            "CONFIRMED", [], [], "deep",
        )
        assert result == "CONFIRMED"

    def test_debate_can_upgrade_safe_to_confirmed_when_consensus_safe(self):
        """If consensus said SAFE (below threshold, genuinely not flagged) and the
        debate says CONFIRMED → take the debate (it found something). This is the
        "debate is the only signal that found it" path — case 7/default."""
        result, _ = _reconcile_verdicts(
            "Reentrancy", 0.20,
            _consensus("SAFE", confidence=0.0),
            "CONFIRMED", [], [], "deep",
        )
        # cv_rank for SAFE is 0, debate_rank for CONFIRMED is 5 → debate wins
        assert result == "CONFIRMED"

    def test_more_severe_wins_on_unhandled_combination(self):
        """Default rule: for combinations not explicitly in the 8 cases, the more
        severe (higher rank) verdict wins. E.g. consensus WATCH + debate CONFIRMED
        → CONFIRMED."""
        result, _ = _reconcile_verdicts(
            "Reentrancy", 0.30,
            _consensus("WATCH", confidence=0.20),
            "CONFIRMED", [], [], "deep",
        )
        assert result == "CONFIRMED"
