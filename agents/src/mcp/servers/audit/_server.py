# agents/src/mcp/servers/audit/_server.py
"""
SSE transport wiring + uvicorn entrypoint for the sentinel-audit server.

Same SSE architecture as inference_server.py and rag_server.py:
    SseServerTransport → /sse + /messages/
    Starlette app      → ASGI router
    uvicorn            → ASGI server

The `server` MCP instance is imported from _handlers (where the
@list_tools / @call_tool decorators register tools). Lifecycle hooks
(_on_startup/_on_shutdown) come from _lifecycle. Mutable state stays on
the audit_server shim module.
"""

from __future__ import annotations

import uvicorn
from loguru import logger
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route


def _shim():
    """Return the audit_server shim module (state holder + config consts)."""
    from src.mcp.servers import audit_server as _as
    return _as


def run_server() -> None:
    """Wire up the MCP server to SSE transport and start uvicorn."""
    _as = _shim()
    from ._handlers import server
    from ._lifecycle import _on_shutdown, _on_startup

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
            "mock_mode":          _as._MOCK_MODE,
            "registry_address":   _as._REGISTRY_ADDRESS,
            "rpc_configured":     bool(_as._RPC_URL),
            "tools":              ["get_latest_audit", "get_audit_history", "check_audit_exists"],
        })

    # Starlette >= 1.0 removed on_startup/on_shutdown kwargs in favor of lifespan.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        await _on_startup()
        try:
            yield
        finally:
            await _on_shutdown()

    starlette_app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
            Route("/health", endpoint=health),
        ],
    )

    logger.info(
        "Starting sentinel-audit MCP server | port={} | mock={} | registry={}",
        _as._SERVER_PORT,
        _as._MOCK_MODE,
        _as._REGISTRY_ADDRESS,
    )
    uvicorn.run(starlette_app, host="0.0.0.0", port=_as._SERVER_PORT)