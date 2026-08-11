# agents/src/mcp/servers/audit/_readonly_handlers.py
"""Live read-only, version-aware MCP dispatcher for SENTINEL AuditRegistry.

R0 removed transaction signing from the analysis/MCP security domain. The live
service therefore exposes exactly three read operations and rejects every write
name before the historical V2 submission module is imported.

The query names are protocol-neutral and now match that promise: they observe
V1, V2, and V3 registry storage rather than silently reading only the legacy V1
scalar mapping.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.types import TextContent, Tool

from ._handlers import _validate_address
from ._status import explicit_mock_result, failed_result, live_result, unavailable_result
from ._versioned_reads import (
    count_by_protocol,
    history_across_protocols,
    latest_across_protocols,
)

READ_ONLY_TOOLS = frozenset(
    {"get_latest_audit", "get_audit_history", "check_audit_exists"}
)

server = Server("sentinel-audit")


def _shim():
    from src.mcp.servers import audit_server as _as

    return _as


def _text(payload: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload))]


def _rejected_tool(name: str) -> list[TextContent]:
    reason = (
        "runtime audit submission is not exposed by the analysis MCP service; "
        "legacy V2 submission is policy-ineligible and V3 signing belongs to "
        "the isolated policy-signer domain"
    )
    return _text(
        {
            "status": "policy_rejected",
            "failed_step": "tool_dispatch",
            "reason_code": "read_only_service",
            "reason": reason,
            "tool": name,
            "attempted": False,
        }
    )


def _invalid_result(operation: str, arguments: dict[str, Any], detail: str) -> list[TextContent]:
    return _text(
        failed_result(
            {"error": "invalid_request", "detail": detail},
            operation=operation,
            arguments=arguments,
            reason_code="invalid_request",
            detail=detail,
            attempted=False,
        )
    )


def _mock_v3_record(contract_address: str, *, timestamp: int = 1786480000) -> dict[str, Any]:
    return {
        "protocol_version": "v3",
        "submission_protocol": "context_attested_v3",
        "proof_scope": "legacy_proxy_only_unbound",
        "contract_address": contract_address,
        "class_score_felts": [1000 + i * 100 for i in range(10)],
        "proof_hash": "0x" + "11" * 32,
        "request_digest": "0x" + "22" * 32,
        "public_signals_hash": "0x" + "33" * 32,
        "contract_code_hash": "0x" + "44" * 32,
        "teacher_model_hash": "0x" + "55" * 32,
        "proxy_bundle_hash": "0x" + "66" * 32,
        "data_version_hash": "0x" + "77" * 32,
        "class_schema_hash": "0x" + "88" * 32,
        "round_id": 7,
        "timestamp": timestamp,
        "timestamp_iso": "2026-08-11T20:26:40+00:00",
        "agent": "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
        "policy_signer": "0x0000000000000000000000000000000000000002",
        "verifier": "0x0000000000000000000000000000000000000003",
        "verified": True,
    }


def _mock_latest(contract_address: str) -> dict[str, Any]:
    latest = _mock_v3_record(contract_address)
    return {
        **latest,
        "exists": True,
        "total_count": 3,
        "counts_by_protocol": {"v3": 1, "v2": 1, "v1": 1},
    }


def _mock_history(contract_address: str, limit: int) -> dict[str, Any]:
    records = [_mock_v3_record(contract_address)]
    if limit >= 2:
        records.append(
            {
                "protocol_version": "v2",
                "proof_scope": "legacy_proxy_only_unbound",
                "contract_address": contract_address,
                "class_score_felts": [900 + i * 50 for i in range(10)],
                "proof_hash": "0x" + "aa" * 32,
                "model_hash": "0x" + "bb" * 32,
                "timestamp": 1786470000,
                "timestamp_iso": "2026-08-11T17:40:00+00:00",
                "agent": "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
                "verified": True,
            }
        )
    if limit >= 3:
        records.append(
            {
                "protocol_version": "v1",
                "contract_address": contract_address,
                "score": 0.7314,
                "score_field_element": 5993,
                "label": "vulnerable",
                "threshold": 0.5,
                "proof_hash": "0x" + "cc" * 32,
                "timestamp": 1786460000,
                "timestamp_iso": "2026-08-11T14:53:20+00:00",
                "agent": "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
                "verified": True,
            }
        )
    counts = {"v3": 1, "v2": 1, "v1": 1}
    return {
        "contract_address": contract_address,
        "total_count": 3,
        "returned": min(limit, 3),
        "counts_by_protocol": counts,
        "records": records[:limit],
    }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the complete live audit-MCP surface."""
    audit_shim = _shim()
    return [
        Tool(
            name="get_latest_audit",
            description=(
                "Get the newest persisted audit across AuditRegistry V3, V2, and V1. "
                "The response includes protocol_version explicitly."
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
            description=(
                "Get merged V3/V2/V1 audit history for a contract, sorted by on-chain "
                "timestamp newest first. Each record includes protocol_version."
            ),
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
            description=(
                "Check whether any V3/V2/V1 audit exists and return counts by protocol."
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
    ]


async def _handle_get_latest_audit(arguments: dict[str, Any]) -> list[TextContent]:
    operation = "get_latest_audit"
    raw_address = arguments.get("contract_address", "")
    try:
        address = _validate_address(raw_address)
    except ValueError as exc:
        return _invalid_result(operation, arguments, str(exc))

    audit_shim = _shim()
    if audit_shim._MOCK_MODE:
        return _text(
            explicit_mock_result(
                _mock_latest(address), operation=operation, arguments=arguments
            )
        )
    if audit_shim._registry is None:
        return _text(
            unavailable_result(
                {"error": "rpc_unavailable", "contract_address": address},
                operation=operation,
                arguments=arguments,
                reason_code="rpc_not_ready",
                detail="AuditRegistry client is not initialized",
                attempted=False,
            )
        )

    started = time.monotonic()
    try:
        result = await latest_across_protocols(audit_shim._registry, address)
        latest = result.pop("latest")
        payload = {**result, **latest} if latest is not None else result
        if latest is None:
            payload["message"] = "No V1/V2/V3 audit exists for this contract address."
        return _text(
            live_result(
                payload,
                operation=operation,
                arguments=arguments,
                duration_ms=(time.monotonic() - started) * 1000,
                clean=latest is None,
            )
        )
    except Exception as exc:
        logger.error("version-aware latest-audit RPC error | address={} | error={}", address, exc)
        return _text(
            unavailable_result(
                {
                    "error": "rpc_error",
                    "contract_address": address,
                    "detail": str(exc),
                },
                operation=operation,
                arguments=arguments,
                reason_code="versioned_registry_read_failed",
                detail=str(exc),
                attempted=True,
                duration_ms=(time.monotonic() - started) * 1000,
            )
        )


async def _handle_get_audit_history(arguments: dict[str, Any]) -> list[TextContent]:
    operation = "get_audit_history"
    raw_address = arguments.get("contract_address", "")
    try:
        address = _validate_address(raw_address)
        limit = int(arguments.get("limit", _shim()._DEFAULT_HISTORY_LIMIT))
        if limit < 1:
            raise ValueError("limit must be at least 1")
        limit = min(limit, 50)
    except (TypeError, ValueError) as exc:
        return _invalid_result(operation, arguments, str(exc))

    audit_shim = _shim()
    if audit_shim._MOCK_MODE:
        return _text(
            explicit_mock_result(
                _mock_history(address, limit), operation=operation, arguments=arguments
            )
        )
    if audit_shim._registry is None:
        return _text(
            unavailable_result(
                {"error": "rpc_unavailable", "contract_address": address},
                operation=operation,
                arguments=arguments,
                reason_code="rpc_not_ready",
                detail="AuditRegistry client is not initialized",
                attempted=False,
            )
        )

    started = time.monotonic()
    try:
        payload = await history_across_protocols(
            audit_shim._registry, address, limit=limit
        )
        if payload["total_count"] == 0:
            payload["message"] = "No V1/V2/V3 audit history exists for this address."
        return _text(
            live_result(
                payload,
                operation=operation,
                arguments=arguments,
                duration_ms=(time.monotonic() - started) * 1000,
                clean=payload["total_count"] == 0,
            )
        )
    except Exception as exc:
        logger.error("version-aware audit-history RPC error | address={} | error={}", address, exc)
        return _text(
            unavailable_result(
                {
                    "error": "rpc_error",
                    "contract_address": address,
                    "detail": str(exc),
                },
                operation=operation,
                arguments=arguments,
                reason_code="versioned_registry_read_failed",
                detail=str(exc),
                attempted=True,
                duration_ms=(time.monotonic() - started) * 1000,
            )
        )


async def _handle_check_audit_exists(arguments: dict[str, Any]) -> list[TextContent]:
    operation = "check_audit_exists"
    raw_address = arguments.get("contract_address", "")
    try:
        address = _validate_address(raw_address)
    except ValueError as exc:
        return _invalid_result(operation, arguments, str(exc))

    audit_shim = _shim()
    if audit_shim._MOCK_MODE:
        return _text(
            explicit_mock_result(
                {
                    "contract_address": address,
                    "exists": True,
                    "total_count": 3,
                    "counts_by_protocol": {"v3": 1, "v2": 1, "v1": 1},
                },
                operation=operation,
                arguments=arguments,
            )
        )
    if audit_shim._registry is None:
        return _text(
            unavailable_result(
                {"error": "rpc_unavailable", "contract_address": address},
                operation=operation,
                arguments=arguments,
                reason_code="rpc_not_ready",
                detail="AuditRegistry client is not initialized",
                attempted=False,
            )
        )

    started = time.monotonic()
    try:
        counts = await count_by_protocol(audit_shim._registry, address)
        total = sum(counts.values())
        payload = {
            "contract_address": address,
            "exists": total > 0,
            "total_count": total,
            "counts_by_protocol": counts,
        }
        return _text(
            live_result(
                payload,
                operation=operation,
                arguments=arguments,
                duration_ms=(time.monotonic() - started) * 1000,
                clean=total == 0,
            )
        )
    except Exception as exc:
        logger.error("version-aware audit-count RPC error | address={} | error={}", address, exc)
        return _text(
            unavailable_result(
                {
                    "error": "rpc_error",
                    "contract_address": address,
                    "detail": str(exc),
                },
                operation=operation,
                arguments=arguments,
                reason_code="versioned_registry_read_failed",
                detail=str(exc),
                attempted=True,
                duration_ms=(time.monotonic() - started) * 1000,
            )
        )


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
