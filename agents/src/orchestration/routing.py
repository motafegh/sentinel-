# agents/src/orchestration/routing.py
"""
Per-class routing logic for the SENTINEL audit graph.

Replaces the single global `_is_high_risk(ml_result) > 0.70` threshold with:
  - DEEP_THRESHOLDS  : per-class probability that triggers deep analysis
  - ROUTING_RULES    : which tools activate per flagged class
  - CLASS_TO_DETECTORS: maps vulnerability class → relevant Slither detector names
  - compute_active_tools() : returns set of tool node names to fan-out to
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str):
    from src.config import get_config as _get_cfg

    _map = {
        "DEEP_THRESHOLDS":     lambda c: c.routing.deep_thresholds,
        "ROUTING_RULES":       lambda c: c.routing.routing_rules,
        "OVERALL_VERDICT_RANK": lambda c: c.routing.overall_verdict_rank,
    }
    if name in _map:
        return _map[name](_get_cfg())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ---------------------------------------------------------------------------
# Slither detector name → vulnerability class mapping.
# Used by static_analysis node (detector scoping) and by synthesizer
# (matching Slither findings to ML-flagged classes for verdict computation).
# ---------------------------------------------------------------------------

CLASS_TO_DETECTORS: dict[str, list[str]] = {
    "Reentrancy": [
        "reentrancy-eth",
        "reentrancy-no-eth",
        # 2026-06-21: renamed from "reentrancy-events-and-order" in current
        # Slither (0.11.5) — verified via direct detector enumeration.
        "reentrancy-events",
        "reentrancy-benign",
    ],
    "IntegerUO": [
        # 2026-06-21: "integer-overflow" and "toctou" do NOT exist as Slither
        # 0.11.5 detector ARGUMENT values — verified via direct enumeration of
        # slither.detectors.all_detectors (101 detectors, neither name present).
        # Both were removed upstream (Solidity >=0.8 has built-in checked
        # arithmetic; Slither dropped the dedicated overflow detector with no
        # direct replacement). Previously these were silent dead entries —
        # IntegerUO got zero real corroboration from either name.
        "unchecked-lowlevel",
    ],
    "GasException": [
        "costly-loop",
        "calls-loop",
        "incorrect-exp",
    ],
    "Timestamp": [
        "timestamp",
    ],
    "TransactionOrderDependence": [
        "tx-origin",
        "controlled-delegatecall",
        "msg-value-loop",
    ],
    "ExternalBug": [
        "arbitrary-send-eth",
        "low-level-calls",
        "unchecked-send",
        "controlled-delegatecall",
    ],
    "CallToUnknown": [
        "low-level-calls",
        "controlled-delegatecall",
        "delegatecall-loop",
    ],
    "MishandledException": [
        "unchecked-send",
        "unchecked-lowlevel",
        "unchecked-transfer",
        "return-bomb",
    ],
    "UnusedReturn": [
        "unused-return",
    ],
    "DenialOfService": [
        "calls-loop",
        "costly-loop",
        "msg-value-loop",
    ],
}

# Inverted map: detector name → list of classes it supports.
# Built once at import time; used by synthesizer for verdict computation.
DETECTOR_TO_CLASSES: dict[str, list[str]] = {}
for _cls, _dets in CLASS_TO_DETECTORS.items():
    for _det in _dets:
        DETECTOR_TO_CLASSES.setdefault(_det, []).append(_cls)


# ---------------------------------------------------------------------------
# Core routing function
# ---------------------------------------------------------------------------

def _iter_class_probs(ml_result: dict[str, Any]):
    """
    Yield (class_name, probability) pairs from ml_result.

    Prefers the `probabilities` dict (three-tier schema — all 10 classes present).
    Falls back to `vulnerabilities` list (legacy schema — above-threshold only).
    """
    probabilities = ml_result.get("probabilities")
    if probabilities:
        yield from probabilities.items()
    else:
        for vuln in ml_result.get("vulnerabilities", []):
            yield vuln.get("vulnerability_class", ""), vuln.get("probability", 0.0)


def compute_active_tools(ml_result: dict[str, Any]) -> list[str]:
    """
    Given ml_result, return the list of tool node names that should run.
    """
    from src.config import get_config as _get_cfg

    cfg = _get_cfg()
    active: set[str] = set()
    for cls, prob in _iter_class_probs(ml_result):
        if prob >= cfg.routing.deep_thresholds.get(cls, 0.40):
            active.update(cfg.routing.routing_rules.get(cls, []))
    return sorted(active)


def build_routing_decisions(ml_result: dict[str, Any]) -> list[str]:
    """
    Build human-readable routing decision strings for AuditState.routing_decisions.
    """
    from src.config import get_config as _get_cfg

    cfg = _get_cfg()
    decisions: list[str] = []
    active_any = False

    for cls, prob in _iter_class_probs(ml_result):
        thr = cfg.routing.deep_thresholds.get(cls, 0.40)
        tools = cfg.routing.routing_rules.get(cls, [])

        if prob >= thr:
            active_any = True
            decisions.append(
                f"{cls} prob={prob:.3f} >= threshold={thr} → {'+'.join(tools)}"
            )
        else:
            decisions.append(
                f"{cls} prob={prob:.3f} < threshold={thr} → skip"
            )

    if not active_any:
        decisions.append("all classes below per-class threshold → fast path")

    return decisions


def compute_verdict(
    cls: str,
    prob: float,
    static_findings: list[dict],
    rag_results: list[dict],
    path_taken: str,
) -> tuple[str, list[str]]:
    """
    Compute a single-class verdict and evidence source list.

    Verdict hierarchy (rule-based, no LLM cost):
        CONFIRMED  ← prob >= cutoff AND (slither match OR rag >= confirmed_cutoff)
        LIKELY     ← prob >= cutoff AND rag >= likely_cutoff
        DISPUTED   ← prob >= cutoff AND no corroborating evidence
        SAFE       ← prob < DEEP_THRESHOLDS[cls]

    Decision-numbers from config (routing.compute_verdict_*).
    """
    from src.config import get_config as _get_cfg

    cfg = _get_cfg()
    cutoff = cfg.routing.compute_verdict_prob_cutoff
    rag_confirmed = cfg.routing.compute_verdict_rag_confirmed_cutoff
    rag_likely = cfg.routing.compute_verdict_rag_likely_cutoff

    if path_taken == "fast":
        return "LIKELY", [f"ml:{prob:.3f}"]

    relevant_detectors = set(CLASS_TO_DETECTORS.get(cls, []))
    slither_match = any(
        f.get("detector", "") in relevant_detectors
        and f.get("impact", "") in ("High", "Medium")
        for f in static_findings
    )

    rag_score = max((r.get("score", 0.0) for r in rag_results), default=0.0)

    sources: list[str] = [f"ml:{prob:.3f}"]
    if slither_match:
        matching = [
            f.get("detector", "")
            for f in static_findings
            if f.get("detector", "") in relevant_detectors
        ]
        sources.append(f"slither:{','.join(matching[:2])}")
    if rag_score >= 0.50:
        sources.append(f"rag:{rag_score:.3f}")

    if prob >= cutoff and slither_match:
        return "CONFIRMED", sources
    if prob >= cutoff and rag_score >= rag_confirmed:
        return "CONFIRMED", sources
    if prob >= cutoff and rag_score >= rag_likely:
        return "LIKELY", sources
    if prob >= cutoff:
        return "DISPUTED", sources
    if prob >= cfg.routing.deep_thresholds.get(cls, 0.40):
        return "INCONCLUSIVE", sources
    return "SAFE", sources


def prob_to_severity(prob: float) -> str:
    from src.config import get_config as _get_cfg

    thresholds = _get_cfg().routing.prob_to_severity
    sorted_labels = sorted(thresholds, key=thresholds.get, reverse=True)
    for label in sorted_labels:
        if prob >= thresholds[label]:
            return label
    return "INFO"


def compute_overall_verdict(verdicts: dict[str, str]) -> str:
    """Return the highest-severity verdict across all classes."""
    from src.config import get_config as _get_cfg

    rank = _get_cfg().routing.overall_verdict_rank
    if not verdicts:
        return "SAFE"
    return max(verdicts.values(), key=lambda v: rank.get(v, 0))
