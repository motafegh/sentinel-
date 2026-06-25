from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
import src.orchestration.nodes._helpers as _h

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
        return {"audit_history": []}

    logger.info("audit_check | address={}", contract_address)

    try:
        result = await _h._call_mcp_tool(
            server_url=_AUDIT_URL,
            tool_name="get_audit_history",
            arguments={"contract_address": contract_address, "limit": 10},
        )

        if "error" in result:
            logger.warning("audit_check | registry error: {}", result["error"])
            return {
                "audit_history": [],
                "error": f"audit_check: {result.get('error')}",
            }

        records = result.get("records", [])
        logger.info(
            "audit_check complete | {} prior audit(s) found", len(records)
        )
        return {"audit_history": records}

    except Exception as exc:
        logger.error("audit_check failed: {}", exc)
        return {
            "audit_history": [],
            "error": f"audit_check: {exc}",
        }
