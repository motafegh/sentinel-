"""Versioned report projection, separate from historical fusion scores.

No output of this policy is a confirmed dataset negative or security certificate.
"""
from __future__ import annotations

import math
from typing import Any

from src.orchestration.routing import CLASS_TO_DETECTORS, compute_active_tools, compute_overall_verdict
from src.orchestration.verdict.evidence import Evidence, Polarity
from src.orchestration.verdict.fuse import fuse

POLICY = "evidence-adequacy-v1"


def project_assessment(
    state: dict[str, Any], *, ml_result: dict, ml_eligible: bool,
    evidence: list[Evidence], raw_verdicts: dict[str, str], classes: set[str],
) -> tuple[dict[str, str], str, dict[str, Any]]:
    statuses = state.get("tool_status") or {}
    hits = state.get("quick_screen_hits") or {}
    statuses = statuses if isinstance(statuses, dict) else {}
    hits = hits if isinstance(hits, dict) else {}

    def ran(tool: str) -> bool:
        status = statuses.get(tool)
        return isinstance(status, dict) and status.get("ran") is True

    # Context, derived votes and generic formal harness findings remain visible
    # in raw fusion but cannot independently establish target-specific findings.
    direct = [
        e for e in evidence
        if (e.source == "ml" and ml_eligible)
        or (e.source in ("slither", "aderyn") and ran(e.source))
    ]
    direct_fused = fuse(direct)
    supports = {e.vuln_class for e in direct if e.polarity == Polarity.SUPPORTS
                and e.strength > 0 and e.reliability > 0}
    refutes = {e.vuln_class for e in direct if e.polarity == Polarity.REFUTES
               and e.strength > 0 and e.reliability > 0}
    projected = {}
    for cls in sorted(classes | set(raw_verdicts)):
        if cls not in supports:
            projected[cls] = "INCONCLUSIVE"
        elif cls in refutes:
            projected[cls] = "DISPUTED"
        else:
            verdict = direct_fused[cls].verdict_full
            projected[cls] = "WATCH" if verdict == "SAFE" else verdict

    probs = ml_result.get("probabilities")
    vector_complete = (
        isinstance(probs, dict) and set(probs) == set(CLASS_TO_DETECTORS)
        and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                for v in probs.values())
    )
    reasons = []
    if not ml_eligible:
        reasons.append("ml_evidence_unavailable_or_ineligible")
    if not vector_complete:
        reasons.append("ml_class_vector_incomplete_or_invalid")
    if ml_result.get("truncated") is not False:
        reasons.append("ml_input_coverage_not_established")
    if state.get("analysis_path") != "fast":
        reasons.append("completed_fast_screen_scope_not_established")
    for tool in ("slither", "aderyn"):
        if not ran(tool):
            reasons.append(f"{tool}_completion_not_established")
        if not isinstance(hits.get(tool), list) or hits[tool]:
            reasons.append(f"{tool}_empty_screen_not_established")
    if vector_complete and compute_active_tools(ml_result):
        reasons.append("ml_investigation_threshold_reached")
    if classes or evidence or state.get("static_findings") or state.get("error"):
        reasons.append("signals_or_errors_require_review")
    if ml_result.get("label") != "safe":
        reasons.append("ml_no_flag_label_not_established")

    # Deep-run completeness is deliberately not certified by this first policy.
    coverage = "INCOMPLETE" if reasons else "SCREEN_SCOPE_COMPLETED"
    if supports:
        overall = compute_overall_verdict(projected)
    elif not reasons:
        overall = "NO_FINDINGS_IN_SCREEN_SCOPE"
    else:
        overall = "INCONCLUSIVE"
    return projected, overall, {
        "policy": POLICY,
        "coverage": coverage,
        "coverage_reasons": reasons,
        "abstained": overall == "INCONCLUSIVE",
        "direct_signal_classes": sorted(supports),
        "raw_fusion_verdicts": dict(raw_verdicts),
        "direct_signal_scores": {cls: cv.confidence for cls, cv in direct_fused.items()},
        "direct_signal_sources": {cls: sorted({e.source for e in direct if e.vuln_class == cls})
                                  for cls in direct_fused},
        "assurance": "NOT_ESTABLISHED",
        "scope": "Historical ML and static screening signals; not class-negative evidence.",
        "risk_probability_scope": "Historical ML flagged-class score; not calibrated contract risk.",
    }
