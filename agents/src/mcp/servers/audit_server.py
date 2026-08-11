# agents/src/mcp/servers/audit_server.py
"""sentinel-audit MCP server — public API shim.

The live runtime surface is deliberately read-only. Historical submission code
remains under `audit/_handlers.py` + `audit/_submit.py` for compatibility and
forensic tests, but it is not registered on the MCP server. V3 signing belongs
to the isolated policy-signer security domain.

Run the server:
    cd ~/projects/sentinel
    poetry run python agents/src/mcp/servers/audit_server.py
    → http://localhost:8012/health
    → http://localhost:8012/sse

Mutable runtime state is bound into this shim first so tests can continue to
monkeypatch `_MOCK_MODE`, `_registry`, and related state at the established
public import path.
"""

from __future__ import annotations

from .audit._config import (
    _ABI,
    _ABI_PATH,
    _DEFAULT_HISTORY_LIMIT,
    _MOCK_MODE,
    _PROJECT_ROOT,
    _REGISTRY_ADDRESS,
    _RPC_URL,
    _SERVER_PORT,
    EZKL_SCALE_FACTOR,
    _execution_status,
    _registry,
    _w3,
)

from .audit._decode import _decode_audit_result, _mock_audit_result, _mock_history

# The live MCP dispatcher is read-only. Re-export the established read-handler
# names so existing tests/importers keep their public API without exposing the
# historical `_handle_submit_audit` compatibility function.
from .audit._readonly_handlers import (
    _handle_check_audit_exists,
    _handle_get_audit_history,
    _handle_get_latest_audit,
    _validate_address,
    call_tool,
    list_tools,
    server,
)

from .audit._lifecycle import _load_abi, _on_shutdown, _on_startup
from .audit._server import run_server

if __name__ == "__main__":
    run_server()
