from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.routing import compute_verdict, OVERALL_VERDICT_RANK


def _reconcile_verdicts(
    cls: str,
    prob: float,
    consensus_vote: dict | None,
    debate_verdict: str | None,
    static_findings: list,
    rag_results: list,
    path_taken: str,
) -> tuple[str, list[str]]:
    """
    Reconcile consensus_engine's vote and the debate's verdict into one final
    verdict. Implements the FN/FP asymmetry principle (WS1.5, 2026-06-21):
    the debate can UPGRADE (e.g. DISPUTED → CONFIRMED) but can only DOWNGRADE
    to DISPUTED, never to SAFE, when consensus voted non-SAFE. The only way
    a flagged class reaches SAFE is if BOTH consensus AND debate agree it's
    safe (or if neither voted and compute_verdict() says SAFE for a
    below-threshold class).

    See `docs/plan/agents/2026-06-21-agents-redesign/05_VERDICT_RECONCILIATION_PLAN.md`
    for the 8-case table + the full design rationale.

    Args:
        cls:             vulnerability class name
        prob:            ML probability for this class
        consensus_vote:  consensus_verdict[cls] dict or None (consensus didn't vote)
        debate_verdict:  pre_verdicts[cls] string or None (debate was silent)
        static_findings: full Slither/Aderyn findings list
        rag_results:     RAG chunks from state
        path_taken:      "fast" or "deep"

    Returns:
        (verdict_str, evidence_sources_list)
    """
    cv_verdict = consensus_vote.get("verdict") if consensus_vote else None
    cv_conf = consensus_vote.get("confidence", 0.0) if consensus_vote else 0.0

    # Case 7: no consensus vote → debate is the only signal
    if consensus_vote is None:
        if debate_verdict is not None:
            return debate_verdict, [f"ml:{prob:.3f}", "debate"]
        # Case 8: neither → compute_verdict (last resort, returns INCONCLUSIVE
        # for flagged classes per WS1's change to routing.py)
        return compute_verdict(cls, prob, static_findings, rag_results, path_taken)

    # Case 6: consensus voted, debate was silent (empty/timeout) → consensus stands
    if debate_verdict is None:
        return cv_verdict, [f"ml:{prob:.3f}", f"consensus:{cv_verdict}(conf={cv_conf:.2f})"]

    # Both voted — apply the reconciliation rules
    sources = [
        f"ml:{prob:.3f}",
        f"consensus:{cv_verdict}(conf={cv_conf:.2f})",
        f"debate:{debate_verdict}",
    ]

    # Case 1a: consensus CONFIRMED (conf >= 0.70, all tools agreed) + debate
    # SAFE/WATCH/INCONCLUSIVE → keep CONFIRMED. The debate cannot CLEAR or
    # ignore a unanimously tool-corroborated class. SAFE = "I checked, it's
    # not a bug" (blocked — the debate was systematically wrong at this per
    # Finding #14). WATCH/INCONCLUSIVE = "weak/no signal" (the debate had
    # nothing useful to add — consensus stands).
    if cv_verdict == "CONFIRMED" and debate_verdict in ("SAFE", "WATCH", "INCONCLUSIVE"):
        return cv_verdict, sources + ["rule:consensus_confirmed_debate_cannot_clear"]

    # Case 1b: consensus CONFIRMED + debate DISPUTED → DISPUTED. The debate
    # read the source and expresses uncertainty — this is valuable semantic
    # signal the tools can't provide (they're syntactic pattern matchers;
    # they fire on "state change after external call" whether the state is a
    # balance or an index). DISPUTED is NOT "cleared" — the class is still
    # flagged, still in the report, still contributes to overall_verdict.
    # This surfaces the disagreement honestly: "tools agree, but the
    # source-reading debate is uncertain — investigate further."
    # Fixes the FP on 05_unexpected_revert_dos/Reentrancy (syntactic CEI on
    # a non-balance index — all 3 tools agreed CONFIRMED, debate correctly
    # said DISPUTED, Case 1a would have kept CONFIRMED).
    if cv_verdict == "CONFIRMED" and debate_verdict == "DISPUTED":
        return "DISPUTED", sources + ["rule:confirmed_surfaces_debate_uncertainty"]

    # Case 5: consensus DISPUTED + debate CONFIRMED/LIKELY → take debate (upgrade)
    if cv_verdict == "DISPUTED" and debate_verdict in ("CONFIRMED", "LIKELY"):
        return debate_verdict, sources + ["rule:debate_upgrade"]

    # Case 2: consensus LIKELY + debate SAFE → DISPUTED (surface the disagreement)
    if cv_verdict == "LIKELY" and debate_verdict == "SAFE":
        return "DISPUTED", sources + ["rule:disagreement_surfaces_as_disputed"]

    # Case 3: consensus LIKELY + debate DISPUTED → DISPUTED (agreement on "not confirmed")
    if cv_verdict == "LIKELY" and debate_verdict == "DISPUTED":
        return "DISPUTED", sources + ["rule:likely_downgraded_to_disputed"]

    # Case 4: consensus DISPUTED + debate SAFE → DISPUTED (uncorroborated ≠ cleared)
    if cv_verdict == "DISPUTED" and debate_verdict == "SAFE":
        return "DISPUTED", sources + ["rule:disputed_not_cleared_by_debate"]

    # Both agree (same verdict) → return the agreement
    if cv_verdict == debate_verdict:
        return cv_verdict, sources + ["rule:both_agree"]

    # Default: take the more severe (higher rank). Covers cases not explicitly
    # above (e.g. consensus WATCH + debate CONFIRMED → CONFIRMED; consensus
    # SAFE + debate CONFIRMED → CONFIRMED; consensus INCONCLUSIVE + debate
    # LIKELY → LIKELY). This is the "debate can upgrade" path.
    cv_rank = OVERALL_VERDICT_RANK.get(cv_verdict, 0)
    debate_rank = OVERALL_VERDICT_RANK.get(debate_verdict, 0)
    if debate_rank > cv_rank:
        return debate_verdict, sources + ["rule:more_severe_wins_debate"]
    return cv_verdict, sources + ["rule:more_severe_wins_consensus"]
