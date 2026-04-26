# agents/src/mcp/servers/rag_server.py
"""
MCP server — sentinel-rag
Transport: SSE (HTTP)

Exposes Module 4's HybridRetriever as an MCP tool so any
MCP-compatible agent (LangGraph, Claude Desktop, Cursor) can
query the RAG knowledge base without knowing the retriever internals.

Tools:
    search(query, k, filters) → list of relevant exploit chunks

Chunk shape returned per result:
    {
        "chunk_id":   str,
        "content":    str,
        "doc_id":     str,
        "chunk_index": int,
        "total_chunks": int,
        "metadata": {
            "vuln_type":    str | None,
            "date":         str | None,
            "loss_usd":     float | None,
            "source":       str,
            "has_summary":  bool,
            ...
        },
        "score": float   # RRF score — higher is more relevant
    }

Transport note:
    Same SSE pattern as inference_server.py.
    In M6 this becomes http://rag-server:8011 inside Docker Compose.
    No transport changes needed — only the env var changes.

stdio alternative: replace run_server() body with:
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())
"""

from __future__ import annotations

import dataclasses
import json
import os
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
from dotenv import load_dotenv
load_dotenv(override=True)

# A-15 fix: replace relative import with absolute path-anchored import.
# `from src.rag.retriever import ...` assumes the process cwd is agents/,
# which breaks when the server is started from the project root or from Docker.
# Adding agents/ to sys.path at module load time makes the import work
# from any working directory.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))  # → agents/
from src.rag.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Configuration — all values overridable via agents/.env
# ---------------------------------------------------------------------------

# Port this MCP server listens on.
# 8011 avoids collision with Module 1 (8001), inference MCP (8010).
_SERVER_PORT: int = int(os.getenv("MCP_RAG_PORT", "8011"))

# Default number of chunks returned per search call.
# Agents can override per call via the `k` argument.
_DEFAULT_K: int = int(os.getenv("RAG_DEFAULT_K", "5"))

# Hard cap on k — prevents agents from accidentally pulling the entire index.
_MAX_K: int = 20

# ---------------------------------------------------------------------------
# Retriever — loaded once at startup, shared across all tool calls
# ---------------------------------------------------------------------------

# HybridRetriever.__init__ loads FAISS + BM25 + chunks from disk.
# This happens once when the server starts, not per request.
# If the index is missing or corrupt, it raises RuntimeError here —
# which is the correct behaviour: fail fast, don't serve broken results.
logger.info("Loading HybridRetriever index…")
_retriever = HybridRetriever()
logger.info("HybridRetriever ready — {} chunks indexed", len(_retriever.chunks))

# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------

# "sentinel-rag" is the server's identity in the MCP handshake.
# Matches the key used in M5's SERVER_CONFIG — keep it stable.
server = Server("sentinel-rag")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the tools this server exposes."""
    return [
        Tool(
            name="search",
            description=(
                "Search the SENTINEL RAG knowledge base for past DeFi exploits "
                "similar to a given query. Returns ranked chunks from post-mortems "
                "and audit reports. Use this to ground ML risk scores in historical "
                "evidence and retrieve specific vulnerability patterns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language query describing the vulnerability or "
                            "pattern to search for. E.g. 'reentrancy in withdraw function' "
                            "or 'flash loan price oracle manipulation'."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": (
                            f"Number of chunks to return. "
                            f"Default {_DEFAULT_K}, max {_MAX_K}."
                        ),
                        "default": _DEFAULT_K,
                        "minimum": 1,
                        "maximum": _MAX_K,
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional metadata filters to narrow results. "
                            "All fields are optional — omit to search the full index."
                        ),
                        "properties": {
                            "vuln_type": {
                                "type": "string",
                                "description": (
                                    "Filter by vulnerability class. "
                                    "Known values: reentrancy, flash_loan, access_control, "
                                    "oracle_manipulation, logic_error, other."
                                ),
                            },
                            "date_gte": {
                                "type": "string",
                                "description": "ISO date string — only return exploits on or after this date. E.g. '2023-01-01'.",
                            },
                            "loss_gte": {
                                "type": "number",
                                "description": "Minimum loss in USD — only return exploits above this threshold.",
                            },
                            "source": {
                                "type": "string",
                                "description": "Filter by data source. E.g. 'DeFiHackLabs'.",
                            },
                            "has_summary": {
                                "type": "boolean",
                                "description": "If true, only return chunks that include a written summary.",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["query"],
            },
        )
    ]

# ---------------------------------------------------------------------------
# Tool call dispatcher
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to their handler functions."""
    logger.info("Tool called: {} | args keys: {}", name, list(arguments.keys()))

    if name == "search":
        return await _handle_search(arguments)
    else:
        # Defensive — SDK enforces name is from list_tools(), but surface clearly.
        logger.error("call_tool received unknown tool name: {}", name)
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}),
        )]

# ---------------------------------------------------------------------------
# Search handler
# ---------------------------------------------------------------------------

async def _handle_search(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle a search call.

    Retriever.search() is synchronous (FAISS + BM25 are CPU-bound, no I/O).
    Calling it directly in an async handler is fine at this scale — it
    completes in < 100ms on the RTX 3070. If latency becomes a problem in
    M6, wrap with asyncio.run_in_executor to avoid blocking the event loop.
    """
    query: str = arguments["query"]
    k: int = arguments.get("k", _DEFAULT_K)
    filters: dict[str, Any] = arguments.get("filters", {})

    # Enforce hard cap — schema declares maximum: _MAX_K.
    # Defence-in-depth: MCP 1.27.0+ enforces this via JSON Schema but a direct
    # HTTP client could bypass that layer.
    if k > _MAX_K:
        k = _MAX_K
        logger.warning("k capped at {} (requested {})", _MAX_K, arguments.get("k"))

    # A-08: full query at DEBUG level, truncated at INFO level.
    # DEBUG is off by default — enable for troubleshooting retrieval quality.
    # INFO shows only [:60] to keep logs readable during normal operation.
    logger.debug("search | full_query='{}'", query)

    try:
        chunks = _retriever.search(query=query, k=k, filters=filters or None)

        # Chunk is a dataclass — json.dumps() can't serialise it natively.
        # dataclasses.asdict() recursively converts to plain dicts.
        serialised = [dataclasses.asdict(chunk) for chunk in chunks]

        logger.info(
            "search complete | query='{}' | k={} | filters={} | results={}",
            query[:60],
            k,
            filters,
            len(serialised),
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "query": query,
                "k_requested": k,
                "k_returned": len(serialised),
                "filters_applied": filters,
                "results": serialised,
            }),
        )]

    except Exception as exc:
        # Don't let a retriever failure crash the MCP session.
        # Log the full traceback server-side; return a structured error to the agent.
        logger.exception("search failed | query='{}' | error: {}", query[:60], exc)
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": "retriever_error",
                "detail": str(exc),
                "query": query,
            }),
        )]

# ---------------------------------------------------------------------------
# SSE server entrypoint — identical wiring to inference_server.py
# ---------------------------------------------------------------------------

def run_server() -> None:
    """
    Wire up the MCP server to SSE transport and start uvicorn.

    Architecture:
        SseServerTransport — handles /sse (persistent event stream)
                             and /messages (JSON-RPC POST endpoint)
        Starlette app      — ASGI router: /sse, /messages/, /health
        uvicorn            — ASGI server

    M5 SERVER_CONFIG entry for this server:
        "sentinel-rag": {
            "transport": "sse",
            "url": "http://localhost:8011/sse",
            "timeout": 30.0,
        }
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
        # Return Response() so Starlette doesn't raise a TypeError
        # when it tries to call handle_sse as a response callable.
        # Same fix applied to inference_server.py (Starlette teardown bug).
        return Response()

    async def health(request: Request) -> JSONResponse:
        """Liveness probe — used by Docker Compose and monitoring."""
        return JSONResponse({
            "status": "ok",
            "server": "sentinel-rag",
            "chunks_indexed": len(_retriever.chunks),
            "default_k": _DEFAULT_K,
        })

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
            Route("/health", endpoint=health),
        ]
    )

    logger.info(
        "Starting sentinel-rag MCP server | port={} | chunks={}",
        _SERVER_PORT,
        len(_retriever.chunks),
    )

    uvicorn.run(starlette_app, host="0.0.0.0", port=_SERVER_PORT)


if __name__ == "__main__":
    run_server()
