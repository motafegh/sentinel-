# agents/src/mcp/servers/audit_server.py
"""
sentinel-audit MCP server — public API shim.

DEPRECATED entrypoint location (P2.5, 2026-06-25). The implementation now
lives in the `audit/` package (see `audit/__init__.py`). This file is kept
as a thin re-export so existing imports and test monkeypatch paths keep
resolving unchanged:

    from src.mcp.servers.audit_server import _handle_get_latest_audit, …
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)

Run the server:
    cd ~/projects/sentinel
    poetry run python agents/src/mcp/servers/audit_server.py
    → http://localhost:8012/health
    → http://localhost:8012/sse  (MCP SSE endpoint)

Import order matters: the mutable runtime state (_MOCK_MODE, _ABI, _w3,
_registry) is bound into this shim's namespace FIRST (from _config), so
when _handlers later imports this module by name to read `_as._MOCK_MODE`
it sees the live attribute — and a test `monkeypatch.setattr` on it is
observed by every handler.
"""

from __future__ import annotations

# ── 1. Config + runtime state (re-imported into this module's namespace) ─────
#     These bindings become rebindable attributes of `audit_server`:
#     tests do `monkeypatch.setattr("...audit_server._MOCK_MODE", False)` and
#     handlers observe the rebound value via `_as._MOCK_MODE` at call time.
from .audit._config import (
    EZKL_SCALE_FACTOR,
    _ABI,
    _ABI_PATH,
    _DEFAULT_HISTORY_LIMIT,
    _MOCK_MODE,
    _PROJECT_ROOT,
    _REGISTRY_ADDRESS,
    _RPC_URL,
    _SERVER_PORT,
    _w3,
    _registry,
)

# ── 2. Pure decode helpers ─────────────────────────────────────────────────────
from .audit._decode import (
    _decode_audit_result,
    _mock_audit_result,
    _mock_history,
)

# ── 3. MCP `server` instance + tool declarations + handlers ──────────────────
from .audit._handlers import (
    call_tool,
    list_tools,
    server,
    _handle_check_audit_exists,
    _handle_get_audit_history,
    _handle_get_latest_audit,
    _validate_address,
)

# ── 4. Web3 lifecycle ─────────────────────────────────────────────────────────
from .audit._lifecycle import (
    _load_abi,
    _on_shutdown,
    _on_startup,
)

# ── 5. Server entrypoint ──────────────────────────────────────────────────────
from .audit._server import run_server


if __name__ == "__main__":
    run_server()