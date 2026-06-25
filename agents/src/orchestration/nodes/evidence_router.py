from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.routing import build_routing_decisions, compute_active_tools


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
    ml_result = state.get("ml_result", {})
    decisions = build_routing_decisions(ml_result)

    # Log quick_screen signal alongside ML per-class decisions.
    quick_screen_hits = state.get("quick_screen_hits", {})
    slither_hits = quick_screen_hits.get("slither", [])
    aderyn_hits  = quick_screen_hits.get("aderyn",  [])

    active = compute_active_tools(ml_result)
    if slither_hits or aderyn_hits:
        escalation = (
            f"quick_screen: slither={slither_hits[:3]} aderyn={aderyn_hits[:3]}"
            f" → escalate to deep path (overrides fast-path even if ML safe)"
        )
        decisions.append(escalation)
        logger.info("evidence_router | {}", escalation)
    elif not active:
        decisions.append("quick_screen: no hits, ML safe → fast path confirmed")

    logger.info(
        "evidence_router | active_tools={} | quick_screen_slither={} | quick_screen_aderyn={}",
        active or ["fast-path"],
        len(slither_hits),
        len(aderyn_hits),
    )

    return {"routing_decisions": decisions}
