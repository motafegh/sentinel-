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


def __getattr__(name: str):
    from src.config import get_config as _get_cfg

    _map = {
        "RAG_RELEVANCE_FLOOR": lambda c: c.attribution.rag_relevance_floor,
    }
    if name in _map:
        return _map[name](_get_cfg())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def attribute_verdict(
    ml_prob: float,
    slither_match: bool,
    rag_score: float,
) -> dict[str, float]:
    from src.config import get_config as _get_cfg

    floor = _get_cfg().attribution.rag_relevance_floor
    ml_contrib = max(0.0, min(1.0, ml_prob))
    slither_contrib = 1.0 if slither_match else 0.0
    rag_contrib = max(0.0, rag_score - floor)

    total = ml_contrib + slither_contrib + rag_contrib
    if total <= 0:
        return {"ml_pct": 0.0, "slither_pct": 0.0, "rag_pct": 0.0}

    ml_pct = round(100.0 * ml_contrib / total, 1)
    slither_pct = round(100.0 * slither_contrib / total, 1)
    # Make the three sum to exactly 100.0 despite rounding by deriving the last.
    rag_pct = round(100.0 - ml_pct - slither_pct, 1)

    return {"ml_pct": ml_pct, "slither_pct": slither_pct, "rag_pct": rag_pct}
