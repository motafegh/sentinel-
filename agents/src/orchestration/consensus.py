"""
consensus.py — Tool consensus voting (Phase A, A.6).

When the ML model, Slither, and Aderyn disagree on whether a contract has a
given vulnerability class, a flat majority vote is wrong: the three tools have
very different per-class reliability. Slither's `reentrancy-eth` detector is
near-authoritative for Reentrancy; the ML model is authoritative for Timestamp
(tools miss `block.timestamp` business-logic misuse by design); and ML is
*known to be FP-prone* for ExternalBug (Run 12: s_Form001 scored p=0.96 on a
26-line KV store — a clear false positive; see MEMORY.md DIVE crosswalk audit).

`consensus_vote()` therefore computes a WEIGHTED vote: each tool's signal is
multiplied by its per-class reliability weight, summed, and normalised into a
[0, 1] confidence, which then maps to a verdict band.

This module is pure logic (no I/O, no LLM) — fully unit-testable. The
`consensus_engine` node in nodes.py is a thin wrapper that pulls signals out of
AuditState and calls `consensus_vote()` per flagged class.
"""

from __future__ import annotations

from typing import Any

# ── Config-backed module "constants" ───────────────────────────────────────
# All decision-numbers are read from the YAML config at call time (not import
# time) so existing imports like `consensus.CONFIRMED_BAND` still work and
# tests can monkeypatch get_config()'s return value.
#
# P1 (2026-06-23): old env-var override ML_WEIGHT_SCALE removed — decision-
# numbers come from config now (reproducibility > convenience).


def __getattr__(name: str):
    # Lazy import to avoid circular dep at import time.
    from src.config import get_config as _get_cfg

    _map = {
        "ACCURACY_WEIGHTS":     lambda c: c.consensus.accuracy_weights,
        "DEFAULT_WEIGHTS":      lambda c: c.consensus.default_weights,
        "DEFAULT_ML_WEIGHT_SCALE": lambda c: c.consensus.ml_weight_scale,
        "ML_POSITIVE_THRESHOLD": lambda c: c.consensus.ml_positive_threshold,
        "CONFIRMED_BAND":       lambda c: c.consensus.confirmed_band,
        "LIKELY_BAND":          lambda c: c.consensus.likely_band,
        "DISPUTED_BAND":        lambda c: c.consensus.disputed_band,
    }
    if name in _map:
        return _map[name](_get_cfg())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_weights(class_name: str) -> dict[str, float]:
    """
    Return the {ml, slither, aderyn} reliability weights for a class, with the
    ML weight discounted by ml_weight_scale (ML treated as a hint, not authority).
    """
    from src.config import get_config as _get_cfg

    cfg = _get_cfg()
    base = cfg.consensus.accuracy_weights.get(
        class_name, cfg.consensus.default_weights
    )
    scale = cfg.consensus.ml_weight_scale
    return {
        "ml":      round(base["ml"] * scale, 4),
        "slither": base["slither"],
        "aderyn":  base["aderyn"],
    }


def consensus_vote(
    ml_prob: float,
    slither_found: bool,
    aderyn_found: bool,
    class_name: str,
    *,
    ml_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Weighted per-class consensus over {ML, Slither, Aderyn}.

    Args:
        ml_prob:       ML probability for this class, in [0, 1].
        slither_found: True if Slither flagged a detector mapped to this class.
        aderyn_found:  True if Aderyn flagged a rule mapped to this class.
        class_name:    vulnerability class (selects reliability weights).
        ml_threshold:  ML probability at/above which ML votes positive.

    Returns:
        dict with:
            ml_signal     — int 0/1 (did ML vote positive)
            slither_match — int 0/1
            aderyn_match  — int 0/1
            score         — weighted positive mass (raw)
            confidence    — score normalised by total weight, in [0, 1]
            verdict       — CONFIRMED | LIKELY | DISPUTED | SAFE
            weights       — the weights used (for auditability/attribution)

    The confidence is the fraction of *available reliability mass* that voted
    positive, so a lone but highly-reliable tool can still confirm, while two
    weak agreeing tools may not. Confidence is always clamped to [0, 1].
    """
    from src.config import get_config as _get_cfg

    if ml_threshold is None:
        ml_threshold = _get_cfg().consensus.ml_positive_threshold
    cfg = _get_cfg()
    w = get_weights(class_name)

    signals = {
        "ml":      1 if ml_prob >= ml_threshold else 0,
        "slither": 1 if slither_found else 0,
        "aderyn":  1 if aderyn_found else 0,
    }

    total_weight = sum(w.values())
    score = sum(signals[k] * w[k] for k in signals)
    confidence = 0.0 if total_weight <= 0 else max(0.0, min(1.0, score / total_weight))

    if confidence >= cfg.consensus.confirmed_band:
        verdict = "CONFIRMED"
    elif confidence >= cfg.consensus.likely_band:
        verdict = "LIKELY"
    elif confidence >= cfg.consensus.disputed_band:
        verdict = "DISPUTED"
    else:
        verdict = "SAFE"

    return {
        "ml_signal":     signals["ml"],
        "slither_match": signals["slither"],
        "aderyn_match":  signals["aderyn"],
        "score":         round(score, 4),
        "confidence":    round(confidence, 4),
        "verdict":       verdict,
        "weights":       w,
    }
