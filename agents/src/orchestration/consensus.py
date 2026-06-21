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

import os
from typing import Any

# ---------------------------------------------------------------------------
# Per-class reliability weights for {ml, slither, aderyn}.
#
# These are PRINCIPLED DEFAULTS derived from SENTINEL Run 12 evaluation
# findings (MEMORY.md: 47K SmartBugs-Wild eval + manual inspection + DIVE
# crosswalk audit), NOT a fitted confusion-matrix table. Rationale per class:
#
#   Reentrancy   — ML true-positive confirmed; Slither reentrancy-eth strong;
#                  Aderyn corroborates but is mostly a Slither superset.
#   Timestamp    — ML authoritative (static tools miss business-logic misuse
#                  of block.timestamp by design) → up-weight ML, down tools.
#   ExternalBug  — ML FP-prone (s_Form001 p=0.96 FP); DIVE/Slither/Aderyn give
#                  NO independent precision signal (3-way precision 3.0%) →
#                  flatten all three; require strong agreement to confirm.
#   IntegerUO    — Slither/Aderyn syntactic detectors reliable; ML moderate.
#   The rest     — balanced defaults; refine when a per-tool benchmark exists.
#
# Weights need not sum to 1; consensus_vote normalises by the active sum.
# ---------------------------------------------------------------------------

ACCURACY_WEIGHTS: dict[str, dict[str, float]] = {
    "Reentrancy":                 {"ml": 0.78, "slither": 0.82, "aderyn": 0.60},
    "IntegerUO":                  {"ml": 0.62, "slither": 0.80, "aderyn": 0.70},
    "GasException":               {"ml": 0.40, "slither": 0.65, "aderyn": 0.55},
    "Timestamp":                  {"ml": 0.80, "slither": 0.45, "aderyn": 0.40},
    "TransactionOrderDependence": {"ml": 0.70, "slither": 0.60, "aderyn": 0.45},
    "ExternalBug":                {"ml": 0.45, "slither": 0.50, "aderyn": 0.45},
    "CallToUnknown":              {"ml": 0.60, "slither": 0.70, "aderyn": 0.60},
    "MishandledException":        {"ml": 0.55, "slither": 0.72, "aderyn": 0.62},
    "UnusedReturn":               {"ml": 0.55, "slither": 0.75, "aderyn": 0.65},
    "DenialOfService":            {"ml": 0.65, "slither": 0.55, "aderyn": 0.50},
}

# Used when a class is not in the table above (forward-compatible with new
# classes / a renamed taxonomy). Balanced, slightly ML-leaning.
DEFAULT_WEIGHTS: dict[str, float] = {"ml": 0.60, "slither": 0.65, "aderyn": 0.55}

# ── ML-as-a-HINT discount (Ali directive 2026-06-21) ────────────────────────
# SENTINEL's Run 12 ML model is not yet reliable (known FP behaviour, e.g.
# ExternalBug). The agent layer must do its OWN analysis (static tools + LLM
# debate) and treat the ML prediction as a clue, NOT an authority. We therefore
# scale every class's ML reliability weight by ML_WEIGHT_SCALE (default 0.5)
# before voting, so the ML signal ALONE can never reach the CONFIRMED band —
# corroboration from Slither/Aderyn (and the LLM debate downstream) is required.
# Raise this toward 1.0 once a better-calibrated model ships.
DEFAULT_ML_WEIGHT_SCALE: float = 0.5


def _ml_scale() -> float:
    """ML weight discount, read at call time so it is tunable / monkeypatchable."""
    try:
        return float(os.getenv("ML_WEIGHT_SCALE", str(DEFAULT_ML_WEIGHT_SCALE)))
    except ValueError:
        return DEFAULT_ML_WEIGHT_SCALE

# ML probability at/above which the ML signal counts as a positive vote.
ML_POSITIVE_THRESHOLD: float = 0.50

# Confidence → verdict band boundaries.
CONFIRMED_BAND: float = 0.70
LIKELY_BAND: float = 0.50
DISPUTED_BAND: float = 0.30  # below this → SAFE


def get_weights(class_name: str) -> dict[str, float]:
    """
    Return the {ml, slither, aderyn} reliability weights for a class, with the
    ML weight discounted by ML_WEIGHT_SCALE (ML treated as a hint, not authority).
    """
    base = ACCURACY_WEIGHTS.get(class_name, DEFAULT_WEIGHTS)
    return {
        "ml":      round(base["ml"] * _ml_scale(), 4),
        "slither": base["slither"],
        "aderyn":  base["aderyn"],
    }


def consensus_vote(
    ml_prob: float,
    slither_found: bool,
    aderyn_found: bool,
    class_name: str,
    *,
    ml_threshold: float = ML_POSITIVE_THRESHOLD,
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
    w = get_weights(class_name)

    signals = {
        "ml":      1 if ml_prob >= ml_threshold else 0,
        "slither": 1 if slither_found else 0,
        "aderyn":  1 if aderyn_found else 0,
    }

    total_weight = sum(w.values())
    score = sum(signals[k] * w[k] for k in signals)
    confidence = 0.0 if total_weight <= 0 else max(0.0, min(1.0, score / total_weight))

    if confidence >= CONFIRMED_BAND:
        verdict = "CONFIRMED"
    elif confidence >= LIKELY_BAND:
        verdict = "LIKELY"
    elif confidence >= DISPUTED_BAND:
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
