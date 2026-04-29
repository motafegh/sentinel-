# agents/src/mcp/servers/audit_server.py
"""
MCP server — sentinel-audit
Transport: SSE (HTTP)

Exposes AuditRegistry.sol as MCP tools so any MCP-compatible agent
(LangGraph, Claude Desktop, Cursor) can query or submit on-chain audit
records without knowing the Web3 / Solidity details.

RECALL — AuditRegistry.sol architecture:
    Deployed as a UUPS proxy on Sepolia.
    Proxy address:     0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf
    submitAudit()  — stores score + ZK proof for a contract address
    getLatestAudit()   — returns most recent AuditResult for an address
    getAuditHistory()  — returns full AuditResult[] for an address

RECALL — AuditResult struct fields:
    scoreFieldElement  uint256    BN254 field element encoding the score
    proofHash          bytes32    keccak256 of the ZK proof bytes
    timestamp          uint256    Unix timestamp of submission
    agent              address    Submitter's wallet address
    verified           bool       True if ZK proof passed on-chain verify

RECALL — score field element decoding:
    score = scoreFieldElement / 8192.0
    8192 = 2^13 — the EZKL scale factor from calibration.
    Example: scoreFieldElement=4497 → 4497/8192 = 0.5490 (vulnerable)
    This is the same decoding used in run_proof.py and extract_calldata.py.

Tools exposed (read-only phase — submitAudit added after Track 3):
    get_latest_audit(contract_address)             → latest AuditResult
    get_audit_history(contract_address, limit=10)  → list[AuditResult]
    check_audit_exists(contract_address)           → {exists, count}

Usage:
    cd ~/projects/sentinel
    poetry run python agents/src/mcp/servers/audit_server.py
    → http://localhost:8012/health
    → http://localhost:8012/sse  (MCP SSE endpoint)

FIX (2026-04-29):
    Bug 2 — ABI was loaded at module import time (_load_abi() called unconditionally
             at module level). Any environment without compiled contracts
             (CI, mock mode, any dev box that hasn’t run forge build) crashed
             with FileNotFoundError before the server could start.
             Fix: _ABI=None at module level; loaded lazily in _on_startup()
             only when _MOCK_MODE is False. Mock mode starts cleanly with no
             compiled contracts present.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from loguru import logger
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

# ── sys.path — make agents/ importable when started from project root ───────────
# __file__ = agents/src/mcp/servers/audit_server.py
# parents[3] = agents/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration — all values overridable via agents/.env
# ---------------------------------------------------------------------------

_SERVER_PORT: int = int(os.getenv("MCP_AUDIT_PORT", "8012"))

# Sepolia RPC — any JSON-RPC endpoint works: Alchemy, Infura, or a local node.
# Example: https://eth-sepolia.g.alchemy.com/v2/<key>
_RPC_URL: str = os.getenv("SEPOLIA_RPC_URL", "")

# AuditRegistry proxy address on Sepolia.
# UUPS proxy — same address even after implementation upgrades.
_REGISTRY_ADDRESS: str = os.getenv(
    "AUDIT_REGISTRY_ADDRESS",
    "0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf",
)

# Default results returned for get_audit_history.
# Agents can override per call via the `limit` argument.
_DEFAULT_HISTORY_LIMIT: int = int(os.getenv("AUDIT_HISTORY_DEFAULT_LIMIT", "10"))

# Mock mode — return realistic fake on-chain data when RPC is not configured.
# Set AUDIT_MOCK=true in agents/.env during development / CI.
# Must be false in M6 production.
_MOCK_MODE: bool = (
    os.getenv("AUDIT_MOCK", "false").lower() == "true"
    or not _RPC_URL  # auto-mock if no RPC configured at all
)

# ---------------------------------------------------------------------------
# ABI — path resolution only at module level; actual load deferred to startup
# ---------------------------------------------------------------------------
# Bug 2 fix: _load_abi() was previously called at module level unconditionally.
# This crashed with FileNotFoundError in any environment without compiled contracts
# (CI, mock mode, dev boxes that haven’t run `forge build`) before the server
# could even start. _ABI is now None at import time and populated lazily inside
# _on_startup() only when _MOCK_MODE is False.

_PROJECT_ROOT = Path(__file__).resolve().parents[4]   # sentinel/
_ABI_PATH = _PROJECT_ROOT / "contracts/out/AuditRegistry.sol/AuditRegistry.json"


def _load_abi() -> list:
    """Load and return the AuditRegistry ABI from Foundry build output.

    Called lazily from _on_startup() only when _MOCK_MODE is False.
    Never called at module import time.

    Raises:
        FileNotFoundError: if contracts/ haven't been compiled yet.
                           Run: cd contracts && forge build
    """
    if not _ABI_PATH.exists():
        raise FileNotFoundError(
            f"AuditRegistry ABI not found at: {_ABI_PATH}\n"
            "Compile the contracts first:\n"
            "  cd contracts && forge build"
        )
    with open(_ABI_PATH) as f:
        artifact = json.load(f)
    # Foundry artifact format: top-level "abi" key contains the ABI array
    return artifact["abi"]


# Bug 2 fix — was: _ABI: list = _load_abi()  (unconditional module-level call)
# Now: None at import time; set inside _on_startup() only in real mode.
_ABI: list | None = None

# ---------------------------------------------------------------------------
# Web3 client — initialised on startup, shared across all tool calls
# ---------------------------------------------------------------------------
# Using AsyncWeb3 for non-blocking contract calls that don't stall the
# event loop. The contract object is created once and reused — web3.py
# caches the ABI, so repeated calls don't re-parse the JSON.

_w3: Any = None           # AsyncWeb3 instance
_registry: Any = None     # AsyncWeb3 contract object

EZKL_SCALE_FACTOR = 8192  # 2^13 — must match calibration from setup_circuit.py

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server = Server("sentinel-audit")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the tools this server exposes."""
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
                            f"Default: {_DEFAULT_HISTORY_LIMIT}. Max: 50."
                        ),
                        "default": _DEFAULT_HISTORY_LIMIT,
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
    ]


# ---------------------------------------------------------------------------
# Startup / shutdown — Web3 client lifecycle
# ---------------------------------------------------------------------------

async def _on_startup() -> None:
    """Initialise the AsyncWeb3 client and contract object at server start.

    Bug 2 fix: _load_abi() is now called here, inside the mock guard,
    not at module import time. Mock mode starts without any compiled contracts.
    """
    global _w3, _registry, _MOCK_MODE, _ABI

    if _MOCK_MODE:
        logger.info(
            "Audit server starting in MOCK MODE — "
            "no RPC calls will be made (AUDIT_MOCK=true or SEPOLIA_RPC_URL not set)"
        )
        return  # _ABI stays None — mock handlers never use it

    try:
        # Bug 2 fix — ABI loaded here, only in real mode, after mock guard.
        _ABI = _load_abi()

        # Lazy import — only needed when running in real mode.
        # web3 v7 — AsyncWeb3 for non-blocking HTTP RPC calls.
        from web3 import AsyncWeb3

        _w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(_RPC_URL))

        # Verify connectivity — raises if the RPC is unreachable.
        # chain_id = 11155111 for Sepolia.
        chain_id = await _w3.eth.chain_id
        if chain_id != 11155111:
            logger.warning(
                "Unexpected chain ID: {} (expected 11155111 for Sepolia). "
                "Check SEPOLIA_RPC_URL.",
                chain_id,
            )

        # Convert to checksummed address — web3.py requires EIP-55 checksum.
        checksum_address = AsyncWeb3.to_checksum_address(_REGISTRY_ADDRESS)
        _registry = _w3.eth.contract(address=checksum_address, abi=_ABI)

        logger.info(
            "Web3 client ready — chain={} | registry={} | rpc={}",
            chain_id,
            checksum_address,
            _RPC_URL[:40] + "…" if len(_RPC_URL) > 40 else _RPC_URL,
        )

    except Exception as exc:
        # Don't crash the server — log the error and switch to mock mode.
        # This lets CI and offline development work without a live RPC.
        logger.error(
            "Failed to initialise Web3 client: {} — switching to mock mode", exc
        )
        _MOCK_MODE = True


async def _on_shutdown() -> None:
    """Clean up Web3 resources on server stop."""
    global _w3, _registry
    # AsyncHTTPProvider doesn't hold persistent connections — nothing to close.
    _w3 = None
    _registry = None
    logger.info("Audit server shutdown — Web3 client released")


# ---------------------------------------------------------------------------
# Score decoding helpers
# ---------------------------------------------------------------------------

def _decode_audit_result(
    result: tuple,
    contract_address: str,
) -> dict[str, Any]:
    """
    Convert a raw AuditResult tuple from the contract to a clean dict.

    AuditResult tuple layout (indices):
        0  scoreFieldElement  uint256  BN254 field element
        1  proofHash          bytes32  keccak256(proof bytes)
        2  timestamp          uint256  Unix epoch seconds
        3  agent              address  Submitter's wallet
        4  verified           bool     On-chain ZK proof passed

    Score decoding:
        score = scoreFieldElement / EZKL_SCALE_FACTOR (= 2^13 = 8192)
        This is the same factor used in run_proof.py and extract_calldata.py.
        Example: 4497 / 8192 = 0.5490 → "vulnerable"
    """
    score_field_element: int  = int(result[0])
    proof_hash_bytes:    bytes = result[1]
    timestamp:           int  = int(result[2])
    agent:               str  = result[3]
    verified:            bool = bool(result[4])

    # Decode score from field element
    score: float = score_field_element / EZKL_SCALE_FACTOR
    label: str   = "vulnerable" if score >= 0.50 else "safe"

    # Convert timestamp to ISO string for readability
    timestamp_iso: str = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        if timestamp > 0 else "never"
    )

    # Convert bytes32 proof hash to hex string
    proof_hash_hex: str = "0x" + proof_hash_bytes.hex() if proof_hash_bytes else "0x" + "0" * 64

    return {
        "contract_address":    contract_address,
        "score":               round(score, 4),
        "score_field_element": score_field_element,
        "label":               label,
        "threshold":           0.50,       # binary phase threshold
        "proof_hash":          proof_hash_hex,
        "timestamp":           timestamp,
        "timestamp_iso":       timestamp_iso,
        "agent":               agent,
        "verified":            verified,
    }


def _mock_audit_result(contract_address: str) -> dict[str, Any]:
    """
    Realistic fake audit result for development and CI.

    Mirrors _decode_audit_result() output shape exactly — swapping
    mock → real requires zero changes to callers.
    """
    return {
        "contract_address":    contract_address,
        "score":               0.7314,
        "score_field_element": 5993,       # 5993 / 8192 ≈ 0.7314
        "label":               "vulnerable",
        "threshold":           0.50,
        "proof_hash":          "0x" + "ab" * 32,
        "timestamp":           1713200000,
        "timestamp_iso":       "2026-04-15T12:00:00+00:00",
        "agent":               "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
        "verified":            True,
    }


def _mock_history(contract_address: str, limit: int) -> list[dict[str, Any]]:
    """Realistic fake audit history — two entries to exercise pagination."""
    if limit == 0:
        return []
    records = [
        {
            **_mock_audit_result(contract_address),
            "timestamp":     1713200000,
            "timestamp_iso": "2026-04-15T12:00:00+00:00",
            "score":         0.7314,
            "label":         "vulnerable",
        },
    ]
    if limit >= 2:
        records.append({
            **_mock_audit_result(contract_address),
            "timestamp":     1712900000,
            "timestamp_iso": "2026-04-12T03:20:00+00:00",
            "score":         0.4102,
            "score_field_element": 3362,
            "label":         "safe",
        })
    return records[:limit]


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
    else:
        logger.error("call_tool received unknown tool name: {}", name)
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}),
        )]


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


async def _handle_get_latest_audit(
    arguments: dict[str, Any],
) -> list[TextContent]:
    """
    Fetch the latest audit for a contract address.

    Returns null_result if no audit exists (timestamp == 0 sentinel).
    Never raises — catches all RPC errors and returns structured error dict.
    """
    raw_address: str = arguments["contract_address"]

    try:
        address = _validate_address(raw_address)
    except ValueError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    if _MOCK_MODE:
        logger.debug("get_latest_audit | mock mode | address={}", address)
        result = _mock_audit_result(address)
        return [TextContent(type="text", text=json.dumps(result))]

    try:
        # RECALL — getLatestAudit returns AuditResult tuple:
        # (scoreFieldElement, proofHash, timestamp, agent, verified)
        # If no audit exists, all fields are zero (Solidity default value).
        raw = await _registry.functions.getLatestAudit(address).call()

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
    raw_address: str = arguments["contract_address"]
    limit: int = min(
        arguments.get("limit", _DEFAULT_HISTORY_LIMIT),
        50,  # hard cap — prevents agents pulling enormous histories
    )

    try:
        address = _validate_address(raw_address)
    except ValueError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    if _MOCK_MODE:
        logger.debug("get_audit_history | mock mode | address={} | limit={}", address, limit)
        records = _mock_history(address, limit)
        return [TextContent(
            type="text",
            text=json.dumps({"contract_address": address, "count": len(records), "records": records}),
        )]

    try:
        # getAuditHistory returns AuditResult[] — all audits for the address.
        raw_list = await _registry.functions.getAuditHistory(address).call()

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
    raw_address: str = arguments["contract_address"]

    try:
        address = _validate_address(raw_address)
    except ValueError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    if _MOCK_MODE:
        return [TextContent(
            type="text",
            text=json.dumps({"contract_address": address, "exists": True, "count": 2}),
        )]

    try:
        # Two read calls — both are cheap SLOAD operations.
        # Run sequentially (not concurrently) for simplicity; they're fast.
        exists: bool = await _registry.functions.hasAudit(address).call()
        count:  int  = await _registry.functions.getAuditCount(address).call()

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


# ---------------------------------------------------------------------------
# SSE server entrypoint
# ---------------------------------------------------------------------------

def run_server() -> None:
    """
    Wire up the MCP server to SSE transport and start uvicorn.

    Same SSE architecture as inference_server.py and rag_server.py:
        SseServerTransport → /sse + /messages/
        Starlette app      → ASGI router
        uvicorn            → ASGI server
    """
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        """Accept a new SSE client connection and run the MCP session."""
        logger.info("New MCP client connected from {}", request.client)
        async with sse_transport.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
        return Response()

    async def health(request: Request) -> JSONResponse:
        """Liveness probe — used by Docker Compose and monitoring."""
        return JSONResponse({
            "status":             "ok",
            "server":             "sentinel-audit",
            "mock_mode":          _MOCK_MODE,
            "registry_address":   _REGISTRY_ADDRESS,
            "rpc_configured":     bool(_RPC_URL),
            "tools":              ["get_latest_audit", "get_audit_history", "check_audit_exists"],
        })

    starlette_app = Starlette(
        on_startup=[_on_startup],
        on_shutdown=[_on_shutdown],
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
            Route("/health", endpoint=health),
        ],
    )

    logger.info(
        "Starting sentinel-audit MCP server | port={} | mock={} | registry={}",
        _SERVER_PORT,
        _MOCK_MODE,
        _REGISTRY_ADDRESS,
    )
    uvicorn.run(starlette_app, host="0.0.0.0", port=_SERVER_PORT)


if __name__ == "__main__":
    run_server()
