"""
reliability.py — Per-(source, class) reliability lookup (P2 placeholder → P3 learned).

P3 will fit these from a confusion matrix on the labeled benchmark (B-3).
Until then, returns the hand-set accuracy_weights from the versioned config.
"""

from __future__ import annotations


def load_reliability(config=None) -> dict[tuple[str, str], float]:
    """
    Build the reliability lookup table: {(source, class): weight}.

    Sources: ml, slither, aderyn.  Values come from config.consensus.accuracy_weights
    (per-class ml/slither/aderyn weights) with the ML weight already discounted
    by ml_weight_scale (mirroring consensus.py's get_weights pattern).

    This function exists as the P3 extension point: when called with a config
    that has L3-fitted values, it returns data-derived reliability. For now,
    it returns the L1 hand-set values.
    """
    if config is None:
        from src.config import get_config
        config = get_config()

    table: dict[tuple[str, str], float] = {}
    acc = config.consensus.accuracy_weights
    scale = config.consensus.ml_weight_scale
    defaults = config.consensus.default_weights

    all_classes = set(acc.keys())
    for cls in all_classes:
        w = acc.get(cls, defaults)
        table[("ml", cls)] = round(w["ml"] * scale, 4)
        table[("slither", cls)] = w["slither"]
        table[("aderyn", cls)] = w["aderyn"]

    return table


def get_reliability(source: str, cls: str, config=None) -> float:
    """Single lookup with a fallback to the default weight."""
    table = load_reliability(config)

    if (source, cls) in table:
        return table[(source, cls)]

    # Fallback: use default weights from config
    if config is None:
        from src.config import get_config
        config = get_config()

    defaults = config.consensus.default_weights
    scale = config.consensus.ml_weight_scale
    if source == "ml":
        return round(defaults.get("ml", 0.60) * scale, 4)
    if source in defaults:
        return defaults[source]
    # Hard fallback for unknown sources (RAG, debate, etc.)
    _source_defaults = {"rag": 0.50, "debate": 0.55, "quick_screen": 0.40}
    return _source_defaults.get(source, 0.50)
