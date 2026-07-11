# agents/src/mcp/servers/audit/_config.py
"""
Configuration + runtime state for the sentinel-audit MCP server.

Constants (read once at import): server port, RPC URL, registry address,
default history limit, ABI path, EZKL scale factor, project root.

Mutable runtime state (initialised here, mutated by _lifecycle at startup
and read by _handlers at call time): _ABI, _w3, _registry. _MOCK_MODE is
fixed at import but switchable at runtime by _on_startup on Web3 failure.

The mutable names are re-imported into the audit_server.py shim namespace
so tests can monkeypatch `src.mcp.servers.audit_server._MOCK_MODE` etc. and
the handlers see the rebound value (handlers access them via the shim
module attribute at call time, not via a copied import binding).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ── sys.path — make agents/ importable when started from project root ───────────
# __file__ = agents/src/mcp/servers/audit/_config.py
# parents[0]=audit  [1]=servers  [2]=mcp  [3]=src  [4]=agents
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

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
# This crashed with FileNotFoundError in any environment without compiled
# contracts (CI, mock mode, dev boxes that haven't run `forge build`) before the
# server could even start. _ABI is now None at import time and populated lazily
# inside _on_startup() only when _MOCK_MODE is False.

# __file__ = agents/src/mcp/servers/audit/_config.py
# parents[0]=audit [1]=servers [2]=mcp [3]=src [4]=agents [5]=sentinel/
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_ABI_PATH = _PROJECT_ROOT / "contracts/out/AuditRegistry.sol/AuditRegistry.json"

EZKL_SCALE_FACTOR = 8192  # 2^13 — must match calibration from setup_circuit.py

# ---------------------------------------------------------------------------
# Web3 client — initialised on startup, shared across all tool calls.
# Using AsyncWeb3 for non-blocking contract calls that don't stall the event
# loop. Mutated by _lifecycle._on_startup/_on_shutdown; read by _handlers.
# These bindings are re-imported into audit_server.py so tests can monkeypatch
# `src.mcp.servers.audit_server._registry` and the handlers observe the mock.
# ---------------------------------------------------------------------------
_ABI: list | None = None
_w3: Any | None = None
_registry: Any | None = None

# ---------------------------------------------------------------------------
# V2 / submission configuration (P11, 2026-07)
# ---------------------------------------------------------------------------
# Operator private key — hex-encoded (no 0x prefix). Used to sign on-chain
# submitAuditV2 transactions. Must have Sepolia ETH for gas + >= MIN_STAKE SNTL.
_OPERATOR_KEY: str = os.getenv("SENTINEL_OPERATOR_KEY", "")

# ABI for AuditRegistry V2 (includes submitAuditV2, AuditResultV2 tuple).
# Same contract address — UUPS, V1 and V2 coexist on the same proxy.
_ABI_V2_PATH = _PROJECT_ROOT / "contracts/out/AuditRegistry.sol/AuditRegistry.json"
_ABI_V2: list | None = None

# ML inference API — used to fetch fusion embeddings for proof generation.
_ML_API_URL: str = os.getenv("SENTINEL_ML_API_URL", "http://localhost:8001")

# Proxy model checkpoint (for computing proxy scores locally).
_PROXY_CHECKPOINT = _PROJECT_ROOT / "zkml/models/proxy_best.pt"

# EZKL paths (for proof generation subprocess).
_EZKL_RUN_PROOF = _PROJECT_ROOT / "zkml/src/ezkl/run_proof.py"
_EZKL_CALLDATA  = _PROJECT_ROOT / "zkml/src/ezkl/extract_calldata.py"

# Minimum block confirmations to wait after submission.
_SUBMIT_CONFIRM_BLOCKS: int = int(os.getenv("SENTINEL_CONFIRM_BLOCKS", "2"))