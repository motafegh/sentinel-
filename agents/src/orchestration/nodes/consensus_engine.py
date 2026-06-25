from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.nodes._helpers import _signals_for_class, _best_rag_score


async def consensus_engine(state: AuditState) -> dict[str, Any]:
    """
    Tool consensus voting + staged confidence (A.6 + A.7).

    Runs after audit_check (Slither/Aderyn available), before cross_validator.
    For each flagged class (ML ≥ threshold OR a static-tool hit), computes a
    weighted ML/Slither/Aderyn vote and a Bayesian-updated confidence.

    State updates:
        consensus_verdict   → {class: {ml_signal, slither_match, aderyn_match,
                                       score, confidence, verdict, weights}}
        confidence_by_class → {class: confidence in [0,1]}
    """
    from src.orchestration.consensus import consensus_vote
    from src.orchestration.confidence import track_confidence
    from src.orchestration.routing import DEEP_THRESHOLDS

    ml_result = state.get("ml_result", {})
    probabilities: dict[str, float] = ml_result.get("probabilities", {}) or {}
    static_findings = state.get("static_findings", []) or []
    rag_results = state.get("rag_results", []) or []

    # Fall back to flagged-class probabilities if the full vector is absent.
    if not probabilities:
        for v in (ml_result.get("confirmed", []) + ml_result.get("suspicious", [])
                  or ml_result.get("vulnerabilities", [])):
            probabilities[v.get("vulnerability_class", "?")] = v.get("probability", 0.0)

    if not probabilities:
        logger.info("consensus_engine | no ML probabilities — skipping")
        return {}

    consensus_verdict: dict[str, dict[str, Any]] = {}
    confidence_by_class: dict[str, float] = {}

    for cls, prob in probabilities.items():
        slither_found, aderyn_found = _signals_for_class(cls, static_findings)
        deep_threshold = DEEP_THRESHOLDS.get(cls, 0.40)
        # WS1 (2026-06-21): consensus_engine is the sole verdict authority.
        # Vote on EVERY class that warranted investigation (crossed its
        # DEEP_THRESHOLD) OR has a tool hit. Previously skipped classes in
        # the 0.35-0.49 band with no tools — causing them to fall through to
        # compute_verdict() which silently SAFEd them (Finding #9/#10).
        if prob < deep_threshold and not slither_found and not aderyn_found:
            continue  # genuinely not flagged — below investigation threshold
        vote = consensus_vote(float(prob), slither_found, aderyn_found, cls)
        # WS1 (2026-06-21): a flagged class (crossed DEEP_THRESHOLD) that got
        # no positive signals is NOT "cleared" — it's "uncorroborated." The
        # FN/FP asymmetry principle says we must not silently SAFE it. Override
        # SAFE → DISPUTED so downstream consumers see the distinction. This is
        # the cheap-signal verdict; the debate (if it runs) can still clear it.
        if vote["verdict"] == "SAFE" and float(prob) >= deep_threshold:
            vote["verdict"] = "DISPUTED"
            vote["overridden_from_safe"] = True
        consensus_verdict[cls] = vote
        confidence_by_class[cls] = track_confidence(
            float(prob),
            slither_found=slither_found,
            aderyn_found=aderyn_found,
            rag_score=_best_rag_score(cls, rag_results),
        )

    logger.info(
        "consensus_engine complete | voted {} class(es)", len(consensus_verdict)
    )
    # ── P2 dual-write: emit evidence alongside legacy verdicts ──
    from src.orchestration.verdict.emit import (
        emit_ml_evidence, emit_static_evidence, emit_consensus_evidence,
    )
    evidence_list: list[Any] = []
    evidence_list.extend(emit_ml_evidence(ml_result))
    evidence_list.extend(emit_static_evidence(static_findings))
    # Emit consensus vote evidence for EVERY voted class (including below-ML-threshold
    # classes with tool corroboration that consensus_engine overrides SAFE→DISPUTED).
    evidence_list.extend(emit_consensus_evidence(consensus_verdict))

    return {
        "consensus_verdict":   consensus_verdict,
        "confidence_by_class": confidence_by_class,
        "evidence_list":       evidence_list,
    }
