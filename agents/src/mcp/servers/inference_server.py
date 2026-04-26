# agents/src/mcp/servers/inference_server.py
"""
MCP server — sentinel-inference
Transport: SSE (HTTP)

Exposes Module 1's /predict endpoint as MCP tools so any
MCP-compatible agent (LangGraph, Claude Desktop, Cursor) can
call them without knowing the HTTP implementation details.

Tools:
  predict(contract_code)              → single contract risk assessment
  batch_predict(contracts)            → list of contracts, parallel scoring

stdio alternative: replace run_server() body with:
  from mcp.server.stdio import stdio_server
  async with stdio_server() as (r, w):
      await server.run(r, w, server.create_initialization_options())
Same tool logic, different transport. Use stdio for subprocess tools,
SSE for anything that needs to be deployed or serve multiple clients.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import uvicorn
from loguru import logger
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response   # Response is new
from starlette.routing import Mount, Route

# ---------------------------------------------------------------------------
# Configuration — all values overridable via agents/.env
# ---------------------------------------------------------------------------

# Where Module 1's FastAPI inference server lives.
# In M6 this becomes http://ml-server:8001 inside Docker Compose.
_MODULE1_URL: str = os.getenv("MODULE1_INFERENCE_URL", "http://localhost:8001")

# Port this MCP server listens on.
# 8010 chosen to avoid collision with Module 1 (8001) and future API (8000).
_SERVER_PORT: int = int(os.getenv("MCP_INFERENCE_PORT", "8010"))

# Timeout for calls to Module 1. Inference can take 5-15s on CPU.
_REQUEST_TIMEOUT: float = float(os.getenv("MODULE1_TIMEOUT", "30.0"))

# Mock mode — return realistic fake responses when Module 1 is not running.
# Set MODULE1_MOCK=true in agents/.env during M4 development.
# Flip to false in M6 when the real inference API is live.
_MOCK_MODE: bool = os.getenv("MODULE1_MOCK", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------

# "sentinel-inference" is the server's identity in the MCP handshake.
# Appears in agent logs and error messages — keep it stable across deploys.
server = Server("sentinel-inference")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# @server.list_tools() is called once per client connection during the
# MCP initialization handshake. The client caches the result and uses
# the inputSchema to validate arguments before sending call_tool requests.
# JSON Schema format — same spec as OpenAI function calling.
@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the tools this server exposes."""
    return [
        Tool(
            name="predict",
            description=(
                "Analyse a single Solidity smart contract and return a risk "
                "score plus per-vulnerability probabilities. Use this when you "
                "have the contract source code and need an ML-based assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_code": {
                        "type": "string",
                        "description": "Full Solidity source code of the contract to analyse.",
                    },
                    "contract_address": {
                        "type": "string",
                        "description": (
                            "Optional on-chain address for traceability. "
                            "Does not affect the prediction."
                        ),
                    },
                },
                # contract_code is the only required field —
                # address is metadata, prediction works without it.
                "required": ["contract_code"],
            },
        ),
        Tool(
            name="batch_predict",
            description=(
                "Analyse multiple Solidity contracts in a single call. "
                "Returns a risk assessment for each contract. Use this when "
                "auditing a multi-contract protocol to avoid multiple round-trips."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contracts": {
                        "type": "array",
                        "description": "List of contracts to analyse.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "contract_code": {"type": "string"},
                                "contract_address": {"type": "string"},
                            },
                            "required": ["contract_code"],
                        },
                        # Soft cap — Module 1 inference is GPU-bound.
                        # At RTX 3070 throughput, 10 contracts ≈ 60-150s.
                        # Enforce a hard cap in the handler below.
                        "minItems": 1,
                        "maxItems": 20,
                    }
                },
                "required": ["contracts"],
            },
        ),
    ]

# ---------------------------------------------------------------------------
# Shared HTTP client (A-20)
# ---------------------------------------------------------------------------
# Created once at server startup, reused across all tool calls.
# A new client per call = a new TCP+TLS handshake every time (~20-50ms).
# This shared client keeps the connection alive and amortises that cost.
# Initialised in _on_startup(), closed in _on_shutdown().

_http_client: httpx.AsyncClient | None = None


async def _on_startup() -> None:
    """Create the shared HTTP client when the ASGI server starts."""
    global _http_client
    _http_client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)
    logger.info(
        "Shared HTTP client ready — Module 1 URL: {} | timeout: {}s",
        _MODULE1_URL,
        _REQUEST_TIMEOUT,
    )


async def _on_shutdown() -> None:
    """Close the shared HTTP client when the ASGI server stops."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        logger.info("Shared HTTP client closed")


# ---------------------------------------------------------------------------
# HTTP bridge to Module 1
# ---------------------------------------------------------------------------

async def _call_inference_api(contract_code: str, contract_address: str = "") -> dict[str, Any]:
    """
    Call Module 1's POST /predict endpoint.

    Returns a dict matching Track 3 PredictResponse (2026-04-17):
      label            str    "vulnerable" | "safe"
      vulnerabilities  list   [{"vulnerability_class": str, "probability": float}, ...]
                               sorted desc by probability, only classes >= threshold.
                               Empty list = safe.
      threshold        float  decision boundary used
      truncated        bool   True if source > 512 tokens (tail not analysed)
      num_nodes        int    AST node count
      num_edges        int    AST edge count

    Falls back to _mock_prediction() when _MOCK_MODE is True or Module 1
    is unreachable. Caller does not need to handle the fallback — it is
    transparent from the tool handler's perspective.
    """
    if _MOCK_MODE:
        logger.debug("Mock mode active — skipping Module 1 HTTP call")
        return _mock_prediction(contract_code)

    payload = {
        "source_code": contract_code,
        # contract_address has no field in PredictRequest — drop it from payload.
        # It's metadata for traceability only — log it server-side, don't send it.
    }

    try:
        # A-20 fix: use the shared module-level AsyncClient (initialised in _on_startup)
        # instead of creating a new client per call.
        # Benefit: TCP connection reuse — each new client triggers a full TCP+TLS handshake
        # (~20-50ms), shared client amortises that across requests.
        # The `async with` pattern is dropped — the shared client stays alive for the
        # server's lifetime and is closed in _on_shutdown().
        response = await _http_client.post(
            f"{_MODULE1_URL}/predict",
            json=payload,
        )
        # Raise immediately on 4xx/5xx — the caller handles the exception
        # and converts it to a structured error TextContent.
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException:
        logger.warning(
            "Module 1 inference timed out after {}s — falling back to mock",
            _REQUEST_TIMEOUT,
        )
        return _mock_prediction(contract_code)

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Module 1 returned HTTP {} for predict call",
            exc.response.status_code,
        )
        # Don't silently fall back on 4xx — that signals a bug in our payload,
        # not a transient network failure. Re-raise so call_tool returns an error.
        raise

    except httpx.RequestError as exc:
        # Connection refused, DNS failure, etc. — Module 1 is simply not running.
        logger.warning("Module 1 unreachable ({}), falling back to mock", exc)
        return _mock_prediction(contract_code)


def _mock_prediction(contract_code: str) -> dict[str, Any]:
    """
    Realistic fake prediction for development and testing (Track 3 schema, 2026-04-17).

    Returns multi-label format matching Track 3 PredictResponse.
    Values are plausible for a medium-risk contract — not random,
    so demo output looks coherent during development and interviews.
    Structure exactly mirrors what Module 1 returns, so swapping mock → real
    requires zero changes to the tool handlers.
    """
    code_lower = contract_code.lower()
    has_reentrancy_pattern = "call.value" in code_lower or "transfer(" in code_lower

    vulnerabilities = []
    if has_reentrancy_pattern:
        vulnerabilities = [
            {"vulnerability_class": "Reentrancy",  "probability": 0.72},
            {"vulnerability_class": "IntegerUO",   "probability": 0.54},
        ]

    return {
        "label":           "vulnerable" if vulnerabilities else "safe",
        "vulnerabilities": vulnerabilities,
        "threshold":       0.50,
        "truncated":       False,
        "num_nodes":       42,
        "num_edges":       58,
    }

# ---------------------------------------------------------------------------
# Tool call dispatcher
# ---------------------------------------------------------------------------

# @server.call_tool() receives every tool invocation from every connected client.
# The SDK has already validated that `name` is in the list returned by list_tools()
# and that `arguments` passes the inputSchema — by the time we're here, the
# input is structurally valid. We still validate semantics (e.g. batch size cap).
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to their handler functions."""
    logger.info("Tool called: {} | args keys: {}", name, list(arguments.keys()))

    if name == "predict":
        return await _handle_predict(arguments)
    elif name == "batch_predict":
        return await _handle_batch_predict(arguments)
    else:
        # Should never reach here — SDK enforces name is from list_tools().
        # Defensive branch for future tools added to list_tools() without
        # a matching handler — surfaces as a clear error, not a silent no-op.
        logger.error("call_tool received unknown tool name: {}", name)
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}),
        )]


async def _handle_predict(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle a single-contract predict call."""
    contract_code: str = arguments["contract_code"]
    contract_address: str = arguments.get("contract_address", "")

    try:
        result = await _call_inference_api(contract_code, contract_address)
        logger.info(
            "predict complete | address={} | label={} | detected={} class(es)",
            contract_address or "unknown",
            result.get("label", "unknown"),
            len(result.get("vulnerabilities", [])),
        )
        return [TextContent(type="text", text=json.dumps(result))]

    except httpx.HTTPStatusError as exc:
        error = {
            "error": "inference_api_error",
            "status_code": exc.response.status_code,
            "detail": exc.response.text[:200],  # truncate — could be HTML
        }
        return [TextContent(type="text", text=json.dumps(error))]


async def _handle_batch_predict(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle a batch predict call.

    Runs contracts sequentially — not concurrently — because Module 1
    is GPU-bound. Concurrent requests would serialize on the GPU anyway
    and add queue overhead. If Module 1 adds a native batch endpoint in
    M6, replace the loop with a single batched HTTP call.
    """
    contracts: list[dict] = arguments["contracts"]

    # A-03 fix: JSON Schema validation IS enforced in MCP 1.27.0+.
    # This cap is defence-in-depth in case a client sends a pre-validated
    # request that bypasses the SDK layer (e.g. raw HTTP test).
    if len(contracts) > 20:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "batch size exceeds maximum of 20"}),
        )]

    results = []
    for i, contract in enumerate(contracts):
        code = contract["contract_code"]
        address = contract.get("contract_address", "")
        try:
            result = await _call_inference_api(code, address)
            results.append({
                "index": i,
                "contract_address": address,
                **result,
            })
        except httpx.HTTPStatusError as exc:
            # Don't abort the whole batch on one failure —
            # record the error for this index and continue.
            results.append({
                "index": i,
                "contract_address": address,
                "error": f"HTTP {exc.response.status_code}",
            })

    logger.info("batch_predict complete | {} contracts processed", len(results))
    return [TextContent(type="text", text=json.dumps({"results": results}))]

# ---------------------------------------------------------------------------
# SSE server entrypoint
# ---------------------------------------------------------------------------

def run_server() -> None:
    """
    Wire up the MCP server to SSE transport and start uvicorn.

    Architecture:
      SseServerTransport  — handles /sse (persistent event stream)
                            and /messages (JSON-RPC POST endpoint)
      Starlette app       — ASGI router that mounts both routes
                            plus a /health route for Docker healthchecks
      uvicorn             — ASGI server that runs the Starlette app

    The Server object (our tool registry) is connected to the transport
    via server.run() inside the SSE connection handler. Each new client
    connection spawns an independent run() coroutine — multiple agents
    can connect simultaneously without shared state issues.
    """
    # SseServerTransport manages the /messages POST endpoint internally.
    # We pass the mount path so it generates correct URLs in SSE events.
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:  # A-04: was -> None (wrong — we return Response())
        """Accept a new SSE client connection and run the MCP session."""
        logger.info("New MCP client connected from {}", request.client)
        async with sse_transport.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            # server.run() drives the full MCP session:
            # initialization handshake → list_tools → call_tool calls
            # It returns when the client disconnects.
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
        return Response()  # ← this line. 200 OK, empty body, tells Starlette we're done.

    async def health(request: Request) -> JSONResponse:
        """Liveness probe — used by Docker Compose and monitoring."""
        return JSONResponse({
            "status": "ok",
            "server": "sentinel-inference",
            "mock_mode": _MOCK_MODE,
            "module1_url": _MODULE1_URL,
        })

    starlette_app = Starlette(
        on_startup=[_on_startup],    # creates shared HTTP client
        on_shutdown=[_on_shutdown],  # closes it cleanly on Ctrl+C / SIGTERM
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
            Route("/health", endpoint=health),
        ]
    )

    logger.info(
        "Starting sentinel-inference MCP server | port={} | mock={}",
        _SERVER_PORT,
        _MOCK_MODE,
    )
    uvicorn.run(starlette_app, host="0.0.0.0", port=_SERVER_PORT)


if __name__ == "__main__":
    run_server()