"""
attribution.py — Metric attribution (Phase A, A.8).

Explains *where* a verdict's evidence came from, LIME-style:
"60% from the ML model, 30% from Slither, 10% from RAG". This makes the
pipeline auditable — a reviewer can see whether a CONFIRMED verdict rests on a
strong model signal, on a corroborating static-analysis finding, or on a
retrieved historical exploit pattern.

The contribution of each source is its (normalised) evidence mass:
  - ML:      the class probability (continuous signal).
  - Slither: 1.0 if a mapped detector fired, else 0.0 (binary signal).
  - RAG:     only the portion of similarity above a relevance floor counts
             (a 0.31 match is noise; a 0.85 match is real evidence).

Percentages are normalised to sum to 100. Pure logic — the `explainer` node
calls `attribute_verdict()` per class and stores the result in
state["metric_attribution"] and final_report.
"""

from __future__ import annotations

# Only RAG similarity above this floor counts as evidence (below = noise).
RAG_RELEVANCE_FLOOR: float = 0.30


def attribute_verdict(
    ml_prob: float,
    slither_match: bool,
    rag_score: float,
) -> dict[str, float]:
    """
    Attribute a verdict's evidence across {ML, Slither, RAG}.

    Args:
        ml_prob:       ML probability for the class, [0, 1].
        slither_match: True if a Slither detector for this class fired.
        rag_score:     best RAG similarity for the class, [0, 1].

    Returns:
        {ml_pct, slither_pct, rag_pct} rounded to 1 dp, summing to ~100.0.
        If NO source contributed (no ML mass, no Slither, sub-floor RAG),
        returns all zeros — the caller can treat that as "no attributable
        evidence" rather than dividing by zero.
    """
    ml_contrib = max(0.0, min(1.0, ml_prob))
    slither_contrib = 1.0 if slither_match else 0.0
    rag_contrib = max(0.0, rag_score - RAG_RELEVANCE_FLOOR)

    total = ml_contrib + slither_contrib + rag_contrib
    if total <= 0:
        return {"ml_pct": 0.0, "slither_pct": 0.0, "rag_pct": 0.0}

    ml_pct = round(100.0 * ml_contrib / total, 1)
    slither_pct = round(100.0 * slither_contrib / total, 1)
    # Make the three sum to exactly 100.0 despite rounding by deriving the last.
    rag_pct = round(100.0 - ml_pct - slither_pct, 1)

    return {"ml_pct": ml_pct, "slither_pct": slither_pct, "rag_pct": rag_pct}
