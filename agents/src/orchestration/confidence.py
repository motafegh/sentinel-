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


def __getattr__(name: str):
    from src.config import get_config as _get_cfg

    _map = {
        "SLITHER_AGREE":    lambda c: c.confidence.slither_agree,
        "SLITHER_DISAGREE": lambda c: c.confidence.slither_disagree,
        "ADERYN_AGREE":     lambda c: c.confidence.aderyn_agree,
        "ADERYN_DISAGREE":  lambda c: c.confidence.aderyn_disagree,
        "RAG_AGREE":        lambda c: c.confidence.rag_agree,
        "RAG_RELEVANCE":    lambda c: c.confidence.rag_relevance,
    }
    if name in _map:
        return _map[name](_get_cfg())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    """
    from src.config import get_config as _get_cfg

    cfg = _get_cfg()
    conf = _clamp(ml_prob)

    if slither_found is not None:
        conf *= cfg.confidence.slither_agree if slither_found else cfg.confidence.slither_disagree

    if aderyn_found is not None:
        conf *= cfg.confidence.aderyn_agree if aderyn_found else cfg.confidence.aderyn_disagree

    if rag_score is not None and rag_score >= cfg.confidence.rag_relevance:
        conf *= cfg.confidence.rag_agree

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
