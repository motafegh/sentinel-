"""
confidence.py — Staged confidence tracking (Phase A, A.7).

A verdict's confidence should not be a single number pulled from the ML model;
it should *evolve* as independent evidence arrives. We start from the ML
probability and nudge it multiplicatively as each downstream signal agrees or
disagrees — a lightweight Bayesian-style update that stays in [0, 1].

Design choices:
  - Multiplicative nudges (not additive) so the value cannot leave [0, 1] in
    spirit and we still clamp defensively.
  - Agreement boosts, disagreement shrinks. Magnitudes are deliberately small
    (±10% Slither, ±5% RAG) — evidence refines, it does not overrule the model.
  - Pure functions, no I/O. The `consensus_engine` / `cross_validator` nodes
    call `track_confidence()` per class and store the result in
    state["confidence_by_class"] and final_report["confidence_by_class"].
"""

from __future__ import annotations

# Multiplicative nudge factors. Tunable; kept conservative on purpose.
SLITHER_AGREE = 1.10
SLITHER_DISAGREE = 0.90
ADERYN_AGREE = 1.05
ADERYN_DISAGREE = 0.97
RAG_AGREE = 1.05          # applied when rag_score >= RAG_RELEVANCE
RAG_RELEVANCE = 0.70


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def track_confidence(
    ml_prob: float,
    *,
    slither_found: bool | None = None,
    aderyn_found: bool | None = None,
    rag_score: float | None = None,
) -> float:
    """
    Update a per-class confidence starting from the ML probability.

    Args:
        ml_prob:       base confidence (ML probability for the class), [0, 1].
        slither_found: True/False if Slither evidence is available (None=skip).
        aderyn_found:  True/False if Aderyn evidence is available (None=skip).
        rag_score:     best RAG similarity for the class (None=skip).

    Returns:
        Updated confidence in [0, 1], rounded to 4 dp.

    Each available signal applies one multiplicative nudge. Absent signals
    (None) are skipped so a fast-path verdict with only ML evidence simply
    returns the (clamped) ML probability unchanged.
    """
    conf = _clamp(ml_prob)

    if slither_found is not None:
        conf *= SLITHER_AGREE if slither_found else SLITHER_DISAGREE

    if aderyn_found is not None:
        conf *= ADERYN_AGREE if aderyn_found else ADERYN_DISAGREE

    if rag_score is not None and rag_score >= RAG_RELEVANCE:
        conf *= RAG_AGREE

    return round(_clamp(conf), 4)


def confidence_band(conf: float) -> str:
    """Map a confidence value to a coarse label for human-readable reports."""
    if conf >= 0.70:
        return "high"
    if conf >= 0.50:
        return "medium"
    if conf >= 0.30:
        return "low"
    return "negligible"
