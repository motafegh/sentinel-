from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.provenance import eligible_ml_result
from src.orchestration.routing import build_routing_decisions, choose_analysis_route
from src.orchestration.state import AuditState


async def evidence_router(state: AuditState) -> dict[str, Any]:
    """
    Compute per-class routing and log decisions to AuditState.routing_decisions.

    Reads both ml_result AND quick_screen_hits — if quick_screen found anything,
    the routing note records the escalation so the final report is fully auditable.
    Actual branching logic lives in _route_from_evidence_router (graph.py); this
    node only logs; it never raises.

    State updates:
        routing_decisions → list of human-readable routing decision strings
    """
    ml_result = eligible_ml_result(state, purpose="routing")
    decisions = build_routing_decisions(ml_result)
    target, reason = choose_analysis_route(state)
    path = "fast" if target == "synthesizer" else "deep"
    decisions.append(f"analysis_path={path}: {path} path selected: {reason}")
    logger.info("evidence_router | path={} | reason={}", path, reason)
    return {"routing_decisions": decisions, "analysis_path": path}
