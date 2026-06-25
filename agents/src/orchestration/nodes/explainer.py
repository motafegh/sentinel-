from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.nodes._helpers import _signals_for_class, _best_rag_score


async def explainer(state: AuditState) -> dict[str, Any]:
    """
    Metric attribution (A.8) — runs after reflection.

    For each verdict, attributes the evidence LIME-style across ML / Slither /
    RAG (percentages sum to ~100). Also folds confidence_by_class,
    consensus_verdict, metric_attribution and reflection_notes INTO the
    final_report so a single artifact carries the full Phase-A enrichment.

    State updates:
        metric_attribution → {class: {ml_pct, slither_pct, rag_pct}}
        final_report       → enriched copy
    """
    from src.orchestration.attribution import attribute_verdict

    report = dict(state.get("final_report", {}) or {})
    vuln_verdicts = report.get("vulnerability_verdicts", []) or []
    static_findings = state.get("static_findings", []) or []
    rag_results = state.get("rag_results", []) or []
    confidence_by_class = state.get("confidence_by_class", {}) or {}

    attribution: dict[str, dict[str, float]] = {}
    for v in vuln_verdicts:
        cls = v.get("vulnerability_class", "?")
        prob = float(v.get("probability", 0.0) or 0.0)
        slither_found, _ = _signals_for_class(cls, static_findings)
        rag_score = _best_rag_score(cls, rag_results)
        attribution[cls] = attribute_verdict(prob, slither_found, rag_score)
        # annotate the verdict row in place for the report
        v["attribution"] = attribution[cls]
        if cls in confidence_by_class:
            v["confidence"] = confidence_by_class[cls]

    report["metric_attribution"] = attribution
    report["confidence_by_class"] = confidence_by_class
    report["consensus_verdict"] = state.get("consensus_verdict", {})
    report["reflection_notes"] = state.get("reflection_notes", {})

    logger.info("explainer complete | attributed {} verdict(s)", len(attribution))
    return {"metric_attribution": attribution, "final_report": report}
