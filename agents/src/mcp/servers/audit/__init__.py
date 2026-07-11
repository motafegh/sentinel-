# agents/src/mcp/servers/audit/__init__.py
"""sentinel-audit MCP server package.

Split (P2.5, 2026-06-25) from the original `audit_server.py` god-file
(717 LOC) into focused modules:
    _config.py     — env-config + ABI path + mutable runtime state
    _lifecycle.py  — Web3 client startup/shutdown + ABI load
    _decode.py     — AuditResult tuple decoding + mock data
    _handlers.py   — MCP `server` instance + tool declarations + dispatch
    _server.py     — Starlette/SSE transport + uvicorn entrypoint

The thin `audit_server.py` re-exports the public API so existing imports
(`from src.mcp.servers.audit_server import _handle_get_latest_audit, …`)
and monkeypatch paths (`monkeypatch.setattr("...audit_server._MOCK_MODE")`)
keep working unchanged.
"""