# agents/src/mcp/servers/audit/_handlers.py
"""
MCP tool declarations + call dispatch for the sentinel-audit server.

Defines the `server` instance (used by decorators) and the three tool
handlers. Handlers read the mutable runtime state (`_MOCK_MODE`, `_registry`)
via the audit_server.py shim module attribute at call time, NOT via a
copied import binding — so test monkeypatches on
`src.mcp.servers.audit_server._MOCK_MODE` take effect.

RECALL — AuditRegistry.sol tools (read-only phase):
    get_latest_audit(contract_address)             → latest AuditResult
    get_audit_history(contract_address, limit=10)  → list[AuditResult]
    check_audit_exists(contract_address)           → {exists, count}
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.types import TextContent, Tool

# Single State access point: the shim module. We import it lazily inside
# each handler to avoid load-order issues and to always observe the latest
# runtime/monkeypatched values. (Importing the module object, not its
# current attribute values, is the key — `from X import Y` would snapshot
# `_MOCK_MODE` and miss test rebinds.)


def _shim():
    """Return the audit_server shim module (state holder)."""
    from src.mcp.servers import audit_server as _as
    return _as


server = Server("sentinel-audit")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the tools this server exposes."""
    _as = _shim()
    return [
        Tool(
            name="get_latest_audit",
            description=(
                "Get the most recent audit record for a smart contract from the "
                "AuditRegistry on Sepolia. Returns the vulnerability score, ZK proof "
                "hash, timestamp, and whether the proof was verified on-chain. "
                "Returns null if no audit has ever been submitted for this address."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_address": {
                        "type": "string",
                        "description": (
                            "Ethereum address of the contract to look up. "
                            "Must be a valid checksummed or lowercase hex address."
                        ),
                    },
                },
                "required": ["contract_address"],
            },
        ),
        Tool(
            name="get_audit_history",
            description=(
                "Get the full audit history for a smart contract from the "
                "AuditRegistry on Sepolia. Returns a list of all past audits, "
                "newest first. Use this to track how risk score changed over time "
                "or to verify audit continuity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_address": {
                        "type": "string",
                        "description": "Ethereum address of the contract.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Maximum number of records to return. "
                            f"Default: {_as._DEFAULT_HISTORY_LIMIT}. Max: 50."
                        ),
                        "default": _as._DEFAULT_HISTORY_LIMIT,
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
                "Quickly check whether a contract address has any audit record on-chain. "
                "Cheaper than get_latest_audit — use this first to gate further lookups. "
                "Returns {exists: bool, count: int}."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_address": {
                        "type": "string",
                        "description": "Ethereum address of the contract.",
                    },
                },
                "required": ["contract_address"],
            },
        ),
        Tool(
            name="submit_audit",
            description=(
                "Submit the current audit result on-chain via AuditRegistry.submitAuditV2. "
                "Generates a ZK proof from the contract's source code and ML fusion embedding, "
                "then sends the signed transaction to Sepolia. Requires SENTINEL_OPERATOR_KEY "
                "env var to be set with a funded, staked operator account. "
                "Returns {status, tx_hash, class_scores, proof_hash, model_hash}."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_code": {
                        "type": "string",
                        "description": "Raw Solidity source code of the audited contract.",
                    },
                    "contract_address": {
                        "type": "string",
                        "description": "0x-prefixed on-chain address of the deployed contract.",
                    },
                    "model_hash": {
                        "type": "string",
                        "description": "SHA-256 of the teacher checkpoint (64 hex chars).",
                    },
                },
                "required": ["source_code", "contract_address", "model_hash"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool call dispatcher
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to their handlers."""
    logger.info("Tool called: {} | args keys: {}", name, list(arguments.keys()))

    if name == "get_latest_audit":
        return await _handle_get_latest_audit(arguments)
    elif name == "get_audit_history":
        return await _handle_get_audit_history(arguments)
    elif name == "check_audit_exists":
        return await _handle_check_audit_exists(arguments)
    elif name == "submit_audit":
        return await _handle_submit_audit(arguments)
    else:
        logger.error("call_tool received unknown tool name: {}", name)
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}),
        )]


# ---------------------------------------------------------------------------
# Address validation
# ---------------------------------------------------------------------------

def _validate_address(raw: str) -> str:
    """
    Return checksummed address or raise ValueError.

    web3.py contract functions require EIP-55 checksummed addresses.
    Agents often pass lowercase hex — checksum it here transparently.

    Raises:
        ValueError: if the string is not a valid 20-byte Ethereum address.
    """
    try:
        from web3 import Web3
        return Web3.to_checksum_address(raw)
    except Exception:
        raise ValueError(
            f"Invalid Ethereum address: '{raw}'. "
            "Must be a 0x-prefixed 20-byte hex string."
        )


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _handle_get_latest_audit(
    arguments: dict[str, Any],
) -> list[TextContent]:
    """
    Fetch the latest audit for a contract address.

    Returns null_result if no audit exists (timestamp == 0 sentinel).
    Never raises — catches all RPC errors and returns structured error dict.
    """
    # Lazy import: the decode helpers are stateless, but importing here keeps
    # the import graph acyclic (decode imports config; handlers import decode
    # via the shim, not via a top-level import that could cycle with _config).
    from ._decode import _decode_audit_result, _mock_audit_result
    _as = _shim()

    raw_address: str = arguments["contract_address"]

    try:
        address = _validate_address(raw_address)
    except ValueError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    if _as._MOCK_MODE:
        logger.debug("get_latest_audit | mock mode | address={}", address)
        result = _mock_audit_result(address)
        return [TextContent(type="text", text=json.dumps(result))]

    try:
        # RECALL — getLatestAudit returns AuditResult tuple:
        # (scoreFieldElement, proofHash, timestamp, agent, verified)
        # If no audit exists, all fields are zero (Solidity default value).
        raw = await _as._registry.functions.getLatestAudit(address).call()

        # Detect "no audit" by checking timestamp == 0 (Solidity zero default)
        if raw[2] == 0:
            logger.info("get_latest_audit | no audit found | address={}", address)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "contract_address": address,
                    "exists": False,
                    "message": "No audit has been submitted for this contract address.",
                }),
            )]

        decoded = _decode_audit_result(raw, address)
        logger.info(
            "get_latest_audit | address={} | score={} | label={}",
            address,
            decoded["score"],
            decoded["label"],
        )
        return [TextContent(type="text", text=json.dumps(decoded))]

    except Exception as exc:
        logger.error("get_latest_audit RPC error | address={} | error: {}", address, exc)
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": "rpc_error",
                "contract_address": address,
                "detail": str(exc),
                "suggestion": "Check SEPOLIA_RPC_URL and network connectivity.",
            }),
        )]


async def _handle_get_audit_history(
    arguments: dict[str, Any],
) -> list[TextContent]:
    """
    Fetch full audit history for a contract address, newest first.

    The contract stores audits in insertion order. We reverse the list
    so the most recent audit is at index 0 (natural reading order).
    """
    from ._decode import _decode_audit_result, _mock_history
    _as = _shim()

    raw_address: str = arguments["contract_address"]
    limit: int = min(
        arguments.get("limit", _as._DEFAULT_HISTORY_LIMIT),
        50,  # hard cap — prevents agents pulling enormous histories
    )

    try:
        address = _validate_address(raw_address)
    except ValueError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    if _as._MOCK_MODE:
        logger.debug("get_audit_history | mock mode | address={} | limit={}", address, limit)
        records = _mock_history(address, limit)
        return [TextContent(
            type="text",
            text=json.dumps({"contract_address": address, "count": len(records), "records": records}),
        )]

    try:
        # getAuditHistory returns AuditResult[] — all audits for the address.
        raw_list = await _as._registry.functions.getAuditHistory(address).call()

        if not raw_list:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "contract_address": address,
                    "count": 0,
                    "records": [],
                    "message": "No audit history found for this contract address.",
                }),
            )]

        # Reverse so newest is first (contract appends in submission order)
        decoded_list = [_decode_audit_result(r, address) for r in reversed(raw_list)]
        limited = decoded_list[:limit]

        logger.info(
            "get_audit_history | address={} | total={} | returned={}",
            address,
            len(raw_list),
            len(limited),
        )
        return [TextContent(
            type="text",
            text=json.dumps({
                "contract_address": address,
                "count":            len(raw_list),
                "returned":         len(limited),
                "records":          limited,
            }),
        )]

    except Exception as exc:
        logger.error("get_audit_history RPC error | address={} | error: {}", address, exc)
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": "rpc_error",
                "contract_address": address,
                "detail": str(exc),
            }),
        )]


async def _handle_check_audit_exists(
    arguments: dict[str, Any],
) -> list[TextContent]:
    """
    Quick existence check — cheaper than get_latest_audit.

    Calls hasAudit() (single bool SLOAD) and getAuditCount() (single SLOAD).
    Use this to gate further lookups before paying for a full history read.
    """
    _as = _shim()

    raw_address: str = arguments["contract_address"]

    try:
        address = _validate_address(raw_address)
    except ValueError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    if _as._MOCK_MODE:
        return [TextContent(
            type="text",
            text=json.dumps({"contract_address": address, "exists": True, "count": 2}),
        )]

    try:
        # Two read calls — both are cheap SLOAD operations.
        # Run sequentially (not concurrently) for simplicity; they're fast.
        exists: bool = await _as._registry.functions.hasAudit(address).call()
        count:  int  = await _as._registry.functions.getAuditCount(address).call()

        logger.info(
            "check_audit_exists | address={} | exists={} | count={}",
            address, exists, count,
        )
        return [TextContent(
            type="text",
            text=json.dumps({
                "contract_address": address,
                "exists":           exists,
                "count":            int(count),
            }),
        )]

    except Exception as exc:
        logger.error("check_audit_exists RPC error | address={} | error: {}", address, exc)
        return [TextContent(
            type="text",
            text=json.dumps({
                "error":            "rpc_error",
                "contract_address": address,
                "detail":           str(exc),
            }),
        )]


async def _handle_submit_audit(arguments: dict[str, Any]) -> list[TextContent]:
    """Run the on-chain audit submission pipeline and return structured result."""
    from ._submit import _run_submit

    source_code = arguments.get("source_code", "")
    contract_address = arguments.get("contract_address", "")
    model_hash = arguments.get("model_hash", "")

    if not source_code:
        return [TextContent(type="text", text=json.dumps({
            "status": "failed", "failed_step": "validation",
            "reason": "source_code is required",
        }))]
    if not contract_address:
        return [TextContent(type="text", text=json.dumps({
            "status": "failed", "failed_step": "validation",
            "reason": "contract_address is required",
        }))]
    if not model_hash or len(model_hash) != 64:
        return [TextContent(type="text", text=json.dumps({
            "status": "failed", "failed_step": "validation",
            "reason": "model_hash must be a 64-character hex string",
        }))]

    logger.info("submit_audit — contract: {}, model: {}...",
                contract_address[:18], model_hash[:16])

    result = _run_submit(source_code, contract_address, model_hash)
    return [TextContent(type="text", text=json.dumps(result))]