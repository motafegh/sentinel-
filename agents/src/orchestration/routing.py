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

# ---------------------------------------------------------------------------
# Per-class probability thresholds for deep analysis.
# Deliberately lower than the inference threshold (0.50) — we want to
# investigate borderline cases, not skip them.
# DenialOfService uses 0.30 because the class is rare and under-represented.
# ---------------------------------------------------------------------------

DEEP_THRESHOLDS: dict[str, float] = {
    "Reentrancy":          0.35,
    "IntegerUO":           0.35,
    "GasException":        0.40,
    "Timestamp":           0.35,
    "TOD":                 0.35,
    "ExternalBug":         0.40,
    "CallToUnknown":       0.40,
    "MishandledException": 0.40,
    "UnusedReturn":        0.45,
    "DenialOfService":     0.30,
}

# ---------------------------------------------------------------------------
# Which tool nodes to activate per flagged class.
# static_analysis always runs when any class triggers (Slither is fast).
# rag_research for classes with documented exploit history in DeFiHackLabs.
# econ_assessment: Phase 3 — not yet wired as a node.
# ---------------------------------------------------------------------------

ROUTING_RULES: dict[str, list[str]] = {
    "Reentrancy":          ["static_analysis", "rag_research"],
    "IntegerUO":           ["static_analysis", "rag_research"],
    "GasException":        ["static_analysis"],
    "Timestamp":           ["static_analysis", "rag_research"],
    "TOD":                 ["static_analysis", "rag_research"],
    "ExternalBug":         ["static_analysis", "rag_research"],
    "CallToUnknown":       ["static_analysis", "rag_research"],
    "MishandledException": ["static_analysis"],
    "UnusedReturn":        ["static_analysis"],
    "DenialOfService":     ["static_analysis", "rag_research"],
}

# ---------------------------------------------------------------------------
# Slither detector name → vulnerability class mapping.
# Used by static_analysis node (detector scoping) and by synthesizer
# (matching Slither findings to ML-flagged classes for verdict computation).
# ---------------------------------------------------------------------------

CLASS_TO_DETECTORS: dict[str, list[str]] = {
    "Reentrancy": [
        "reentrancy-eth",
        "reentrancy-no-eth",
        "reentrancy-events-and-order",
        "reentrancy-benign",
    ],
    "IntegerUO": [
        "integer-overflow",
        "toctou",
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
    "TOD": [
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

def compute_active_tools(ml_result: dict[str, Any]) -> list[str]:
    """
    Given ml_result, return the list of tool node names that should run.

    Returns empty list → fast path directly to synthesizer.
    Returns non-empty list → fan-out to those nodes in parallel.

    The set deduplication ensures each tool appears at most once even if
    multiple classes trigger the same tool (e.g. Reentrancy + ExternalBug
    both trigger static_analysis and rag_research).
    """
    active: set[str] = set()
    for vuln in ml_result.get("vulnerabilities", []):
        cls  = vuln.get("vulnerability_class", "")
        prob = vuln.get("probability", 0.0)
        if prob >= DEEP_THRESHOLDS.get(cls, 0.40):
            active.update(ROUTING_RULES.get(cls, []))
    return sorted(active)  # sorted for deterministic order in logs


def build_routing_decisions(ml_result: dict[str, Any]) -> list[str]:
    """
    Build human-readable routing decision strings for AuditState.routing_decisions.
    One entry per vulnerability class — logged for full auditability.
    """
    decisions: list[str] = []
    active_any = False

    for vuln in ml_result.get("vulnerabilities", []):
        cls   = vuln.get("vulnerability_class", "?")
        prob  = vuln.get("probability", 0.0)
        thr   = DEEP_THRESHOLDS.get(cls, 0.40)
        tools = ROUTING_RULES.get(cls, [])

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
        CONFIRMED  ← prob >= 0.50 AND (slither match OR rag score >= 0.80)
        LIKELY     ← prob >= 0.50 AND rag score >= 0.50
        DISPUTED   ← prob >= 0.50 AND no corroborating evidence
        SAFE       ← prob < DEEP_THRESHOLDS[cls]  (shouldn't reach here)

    Args:
        cls:             vulnerability class name
        prob:            ML probability for this class
        static_findings: full Slither findings list from state
        rag_results:     RAG chunks from state
        path_taken:      "fast" or "deep" — fast path skips evidence

    Returns:
        (verdict_str, evidence_sources_list)
    """
    if path_taken == "fast":
        return "LIKELY", [f"ml:{prob:.3f}"]

    # Check whether any Slither finding matches this class.
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

    if prob >= 0.50 and slither_match:
        return "CONFIRMED", sources
    if prob >= 0.50 and rag_score >= 0.80:
        return "CONFIRMED", sources
    if prob >= 0.50 and rag_score >= 0.50:
        return "LIKELY", sources
    if prob >= 0.50:
        return "DISPUTED", sources
    return "SAFE", sources


def prob_to_severity(prob: float) -> str:
    if prob >= 0.85: return "CRITICAL"
    if prob >= 0.70: return "HIGH"
    if prob >= 0.50: return "MEDIUM"
    if prob >= 0.35: return "LOW"
    return "INFO"


OVERALL_VERDICT_RANK = {"CONFIRMED": 4, "LIKELY": 3, "DISPUTED": 2, "SAFE": 1}


def compute_overall_verdict(verdicts: dict[str, str]) -> str:
    """Return the highest-severity verdict across all classes."""
    if not verdicts:
        return "SAFE"
    return max(verdicts.values(), key=lambda v: OVERALL_VERDICT_RANK.get(v, 0))
