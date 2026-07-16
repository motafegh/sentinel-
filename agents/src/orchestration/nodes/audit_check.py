from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import src.orchestration.nodes._helpers as _h
from src.contracts.execution import (
    ExecutionState,
    failure_status,
    parse_status,
    require_eligible_payload,
)
from src.orchestration.state import AuditState

# MCP server URL — overridable via agents/.env
_AUDIT_URL: str = os.getenv("MCP_AUDIT_URL", "http://localhost:8012/sse")


async def audit_check(state: AuditState) -> dict[str, Any]:
    """
    Query AuditRegistry for prior on-chain audit records for this contract.

    RECALL — what this node does:
        Calls sentinel-audit:get_audit_history for the contract_address.
        Returns all prior audits in reverse-chronological order.
        The synthesizer uses this to answer: "Has this contract been audited
        before? Did prior audits flag it? Has the risk score improved?"

    Skipped gracefully if contract_address is empty or not a valid address.
    Sets audit_history=[] in that case — synthesizer handles missing data.

    State updates:
        audit_history → list of AuditResult dicts (may be empty list)
        error         → set on failure (appends, does not replace existing)
    """
    contract_address = state.get("contract_address", "").strip()

    if not contract_address:
        logger.info("audit_check | no contract_address — skipping on-chain lookup")
        status = failure_status(
            ExecutionState.SKIPPED_POLICY,
            dependency="audit_registry",
            reason_code="missing_contract_address",
            detail="on-chain history requires a contract address",
            attempted=False,
        )
        return {"audit_history": [], "tool_status": {"audit_registry": status}}

    logger.info("audit_check | address={}", contract_address)

    try:
        result = await _h._call_mcp_tool(
            server_url=_AUDIT_URL,
            tool_name="get_audit_history",
            arguments={"contract_address": contract_address, "limit": 10},
        )

        if "error" in result:
            logger.warning("audit_check | registry error: {}", result["error"])
            try:
                status = parse_status(result.get("execution_status")).model_dump(mode="json")
            except (TypeError, ValueError):
                status = failure_status(
                    ExecutionState.FAILED,
                    dependency="audit_registry",
                    reason_code="invalid_provenance",
                    detail="registry error response omitted a valid execution status",
                    attempted=True,
                )
            return {
                "audit_history": [],
                "error": f"audit_check: {result.get('error')}",
                "tool_status": {"audit_registry": status},
            }

        try:
            status = require_eligible_payload(
                result,
                purpose="audit history orchestration",
                input_payload={
                    "operation": "get_audit_history",
                    "arguments": {"contract_address": contract_address, "limit": 10},
                },
            ).model_dump(mode="json")
        except (TypeError, ValueError) as exc:
            try:
                status = parse_status(result.get("execution_status")).model_dump(mode="json")
            except (TypeError, ValueError):
                status = failure_status(
                    ExecutionState.FAILED,
                    dependency="audit_registry",
                    reason_code="invalid_provenance",
                    detail=str(exc),
                    attempted=True,
                )
            return {
                "audit_history": [],
                "error": f"audit_check: ineligible provenance — {exc}",
                "tool_status": {"audit_registry": status},
            }

        records = result.get("records", [])
        logger.info("audit_check complete | {} prior audit(s) found", len(records))
        return {
            "audit_history": records,
            "tool_status": {"audit_registry": status},
        }

    except Exception as exc:
        logger.error("audit_check failed: {}", exc)
        status = failure_status(
            ExecutionState.UNAVAILABLE,
            dependency="audit_registry",
            reason_code="transport_exception",
            detail=str(exc),
            attempted=True,
        )
        return {
            "audit_history": [],
            "error": f"audit_check: {exc}",
            "tool_status": {"audit_registry": status},
        }
