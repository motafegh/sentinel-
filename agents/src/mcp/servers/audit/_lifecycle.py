# agents/src/mcp/servers/audit/_lifecycle.py
"""
Web3 client startup/shutdown lifecycle for the sentinel-audit server.

_on_startup initialises the AsyncWeb3 client + contract object at server
start (mock mode short-circuits). On RPC failure it logs and switches to
mock mode so the server still starts. _on_shutdown releases the client.

State (_ABI, _w3, _registry, _MOCK_MODE) is mutated on the audit_server.py
shim module attribute (read by _handlers at call time). The shim imports
these from _config; writing `_as._registry = ...` rebinds the shim's
attribute, which is what handlers observe.
"""

from __future__ import annotations

from loguru import logger


def _shim():
    """Return the audit_server shim module (state holder)."""
    from src.mcp.servers import audit_server as _as
    return _as


async def _on_startup() -> None:
    """Initialise the AsyncWeb3 client and contract object at server start.

    Bug 2 fix: _load_abi() is now called here, inside the mock guard,
    not at module import time. Mock mode starts without any compiled contracts.
    """
    _as = _shim()

    if _as._MOCK_MODE:
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

        _w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(_as._RPC_URL))

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
        checksum_address = AsyncWeb3.to_checksum_address(_as._REGISTRY_ADDRESS)
        _registry = _w3.eth.contract(address=checksum_address, abi=_ABI)

        # Publish the runtime state to the shim module so handlers observe it.
        _as._ABI = _ABI
        _as._w3 = _w3
        _as._registry = _registry

        logger.info(
            "Web3 client ready — chain={} | registry={} | rpc={}",
            chain_id,
            checksum_address,
            _as._RPC_URL[:40] + "…" if len(_as._RPC_URL) > 40 else _as._RPC_URL,
        )

    except Exception as exc:
        # Don't crash the server — log the error and switch to mock mode.
        # This lets CI and offline development work without a live RPC.
        logger.error(
            "Failed to initialise Web3 client: {} — switching to mock mode", exc
        )
        _as._MOCK_MODE = True


async def _on_shutdown() -> None:
    """Clean up Web3 resources on server stop."""
    _as = _shim()
    # AsyncHTTPProvider doesn't hold persistent connections — nothing to close.
    _as._w3 = None
    _as._registry = None
    logger.info("Audit server shutdown — Web3 client released")


def _load_abi() -> list:
    """Load and return the AuditRegistry ABI from Foundry build output.

    Called lazily from _on_startup() only when _MOCK_MODE is False.
    Never called at module import time.

    Raises:
        FileNotFoundError: if contracts/ haven't been compiled yet.
                           Run: cd contracts && forge build
    """
    _as = _shim()
    if not _as._ABI_PATH.exists():
        raise FileNotFoundError(
            f"AuditRegistry ABI not found at: {_as._ABI_PATH}\n"
            "Compile the contracts first:\n"
            "  cd contracts && forge build"
        )
    import json
    with open(_as._ABI_PATH) as f:
        artifact = json.load(f)
    # Foundry artifact format: top-level "abi" key contains the ABI array
    return artifact["abi"]