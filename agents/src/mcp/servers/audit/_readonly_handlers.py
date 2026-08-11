# agents/src/mcp/servers/audit/_readonly_handlers.py
"""Live read-only MCP dispatcher for the SENTINEL audit service.

R0 removed transaction signing from the analysis/MCP security domain. The
historical `_handlers.py` module still contains a compatibility-only
`_handle_submit_audit` function that can drive the legacy V2 proof pipeline,
but that path must not be reachable through the live MCP dispatcher.

This module reuses the existing, tested read handlers while registering a
separate MCP `Server` whose call surface contains exactly the three advertised
read operations. Any attempt to call `submit_audit` (or any other undeclared
name) fails before `_submit.py` is imported or executed.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.types import TextContent, Tool

from ._handlers import (
    _handle_check_audit_exists,
    _handle_get_audit_history,
    _handle_get_latest_audit,
    _validate_address,
)

READ_ONLY_TOOLS = frozenset(
    {"get_latest_audit", "get_audit_history", "check_audit_exists"}
)

server = Server("sentinel-audit")


def _shim():
    from src.mcp.servers import audit_server as _as

    return _as


def _rejected_tool(name: str) -> list[TextContent]:
    reason = (
        "runtime audit submission is not exposed by the analysis MCP service; "
        "legacy V2 submission is policy-ineligible and V3 signing belongs to "
        "the isolated policy-signer domain"
    )
    return [
        TextContent(
            type="text",
            text=json.dumps(
                {
                    "status": "policy_rejected",
                    "failed_step": "tool_dispatch",
                    "reason_code": "read_only_service",
                    "reason": reason,
                    "tool": name,
                    "attempted": False,
                }
            ),
        )
    ]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the complete live audit-MCP surface."""
    audit_shim = _shim()
    return [
        Tool(
            name="get_latest_audit",
            description=(
                "Get the most recent historical audit record for a smart-contract "
                "address from the configured AuditRegistry."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_address": {
                        "type": "string",
                        "description": "Ethereum contract address.",
                    }
                },
                "required": ["contract_address"],
            },
        ),
        Tool(
            name="get_audit_history",
            description="Get historical audit records for a contract, newest first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_address": {
                        "type": "string",
                        "description": "Ethereum contract address.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": audit_shim._DEFAULT_HISTORY_LIMIT,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["contract_address"],
            },
        ),
        Tool(
            name="check_audit_exists",
            description="Check whether historical audit records exist for a contract.",
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_address": {
                        "type": "string",
                        "description": "Ethereum contract address.",
                    }
                },
                "required": ["contract_address"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch only explicitly registered read operations."""
    logger.info("Audit read-only tool called: {} | args keys: {}", name, list(arguments.keys()))

    if name not in READ_ONLY_TOOLS:
        logger.warning("Rejected non-read audit tool at MCP boundary: {}", name)
        return _rejected_tool(name)
    if name == "get_latest_audit":
        return await _handle_get_latest_audit(arguments)
    if name == "get_audit_history":
        return await _handle_get_audit_history(arguments)
    return await _handle_check_audit_exists(arguments)


__all__ = [
    "READ_ONLY_TOOLS",
    "_handle_check_audit_exists",
    "_handle_get_audit_history",
    "_handle_get_latest_audit",
    "_validate_address",
    "call_tool",
    "list_tools",
    "server",
]
