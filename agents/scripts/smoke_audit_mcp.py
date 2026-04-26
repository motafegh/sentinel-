"""
smoke_audit_mcp.py — Quick connectivity test for the sentinel-audit MCP server.

Starts the server in the background (subprocess), waits for /health,
then exercises all three tools via the MCP SSE client.

Usage:
    cd ~/projects/sentinel
    poetry run python agents/scripts/smoke_audit_mcp.py

Expected output:
    [PASS] /health responded — mock_mode=True
    [PASS] check_audit_exists → exists=True
    [PASS] get_latest_audit   → score=0.7314 label=vulnerable
    [PASS] get_audit_history  → 2 records returned
    All smoke tests passed.

RECALL — this script only tests the MCP server layer.
It does NOT test the Sepolia RPC path (that requires a live SEPOLIA_RPC_URL).
For on-chain testing, set AUDIT_MOCK=false and provide a real RPC URL.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# ── make agents/ importable ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

# ── config ───────────────────────────────────────────────────────────────────
SERVER_PORT    = int(__import__("os").getenv("MCP_AUDIT_PORT", "8012"))
SERVER_URL     = f"http://localhost:{SERVER_PORT}"
HEALTH_URL     = f"{SERVER_URL}/health"
SSE_URL        = f"{SERVER_URL}/sse"
STARTUP_WAIT_S = 5.0   # seconds to wait for uvicorn to start

# A known DeFi contract — used just to exercise the tools (mock returns canned data)
TEST_ADDRESS = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"  # Uniswap V3 NonfungiblePositionManager


async def _wait_for_server(timeout: float = STARTUP_WAIT_S) -> None:
    """Poll /health until the server is up or timeout expires."""
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(HEALTH_URL)
                if r.status_code == 200:
                    data = r.json()
                    print(f"[PASS] /health responded — mock_mode={data.get('mock_mode')}")
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(0.3)
    raise TimeoutError(f"Server did not start within {timeout}s")


async def _run_smoke_tests() -> None:
    """Connect via MCP SSE client and call all three tools."""
    async with sse_client(SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── Tool 1: check_audit_exists ────────────────────────────────
            r1 = await session.call_tool(
                "check_audit_exists",
                {"contract_address": TEST_ADDRESS},
            )
            data1 = json.loads(r1.content[0].text)
            assert "exists" in data1, f"check_audit_exists missing 'exists' key: {data1}"
            print(f"[PASS] check_audit_exists → exists={data1['exists']} count={data1.get('count')}")

            # ── Tool 2: get_latest_audit ─────────────────────────────────
            r2 = await session.call_tool(
                "get_latest_audit",
                {"contract_address": TEST_ADDRESS},
            )
            data2 = json.loads(r2.content[0].text)
            assert "score" in data2, f"get_latest_audit missing 'score' key: {data2}"
            assert "label" in data2, f"get_latest_audit missing 'label' key: {data2}"
            assert 0.0 <= data2["score"] <= 1.0, f"score out of range: {data2['score']}"
            print(
                f"[PASS] get_latest_audit   → "
                f"score={data2['score']} label={data2['label']} "
                f"verified={data2.get('verified')}"
            )

            # ── Tool 3: get_audit_history ─────────────────────────────────
            r3 = await session.call_tool(
                "get_audit_history",
                {"contract_address": TEST_ADDRESS, "limit": 5},
            )
            data3 = json.loads(r3.content[0].text)
            assert "records" in data3, f"get_audit_history missing 'records' key: {data3}"
            records = data3["records"]
            print(f"[PASS] get_audit_history  → {len(records)} record(s) returned")
            if records:
                # Verify first record has the expected fields
                first = records[0]
                required = {"score", "label", "timestamp", "agent", "verified"}
                missing = required - set(first.keys())
                assert not missing, f"Missing keys in history record: {missing}"

            # ── Tool 4: bad address — validate error handling ─────────────
            r4 = await session.call_tool(
                "get_latest_audit",
                {"contract_address": "not-an-address"},
            )
            data4 = json.loads(r4.content[0].text)
            assert "error" in data4, f"Expected error for bad address: {data4}"
            print(f"[PASS] bad address → error returned (not a crash): {data4['error']}")


async def main() -> None:
    """Start server subprocess, run smoke tests, tear down."""
    # Start the server as a subprocess.
    # Using sys.executable so we get the same Poetry venv interpreter.
    proc = subprocess.Popen(
        [sys.executable, "-m", "agents.src.mcp.servers.audit_server"],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        await _wait_for_server()
        await _run_smoke_tests()
        print("\nAll smoke tests passed. ✓")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
