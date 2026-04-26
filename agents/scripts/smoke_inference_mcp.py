"""
Smoke test for sentinel-inference MCP server over SSE transport.
Verifies: connect → discover tools → call predict → parse response.

Run with:
    cd ~/projects/sentinel/agents
    poetry run python scripts/smoke_inference_mcp.py
"""

import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession
from loguru import logger

SERVER_URL = "http://localhost:8010/sse"

# Minimal Solidity contract — real enough to trigger mock heuristics.
TEST_CONTRACT = """
pragma solidity ^0.8.0;
contract Vault {
    mapping(address => uint) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() external {
        uint amount = balances[msg.sender];
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] = 0;  // state update AFTER external call = reentrancy
    }
}
"""

async def run_smoke_test() -> None:
    logger.info("Connecting to {} ...", SERVER_URL)

    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:

            # ── Step 1: handshake ─────────────────────────────
            await session.initialize()
            logger.info("MCP handshake OK")

            # ── Step 2: discover tools ────────────────────────
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]
            logger.info("Tools discovered: {}", tool_names)

            assert "predict" in tool_names, \
                f"'predict' missing — got {tool_names}"
            assert "batch_predict" in tool_names, \
                f"'batch_predict' missing — got {tool_names}"
            logger.info("Tool discovery PASSED")

            # ── Step 3: call predict ──────────────────────────
            logger.info("Calling predict ...")
            result = await session.call_tool(
                "predict",
                {"contract_code": TEST_CONTRACT}
            )

            raw = result.content[0].text
            parsed = json.loads(raw)
            logger.info("Raw response: {}", json.dumps(parsed, indent=2))

            # ── Step 4: validate response shape ──────────────
            assert "label" in parsed, \
                f"'label' missing from response: {parsed.keys()}"
            assert "confidence" in parsed, \
                f"'confidence' missing from response: {parsed.keys()}"
            assert parsed["label"] in ("vulnerable", "safe"), \
                f"unexpected label value: {parsed['label']}"
            assert 0.0 <= parsed["confidence"] <= 1.0, \
                f"confidence out of range: {parsed['confidence']}"
            assert "truncated" in parsed, \
                f"'truncated' missing from response: {parsed.keys()}"
            assert "num_nodes" in parsed, \
                f"'num_nodes' missing from response: {parsed.keys()}"

            # mock flag must be absent on real inference
            assert parsed.get("mock") is not True, \
                "Got mock=True — is MODULE1_MOCK still set in .env? Restart the MCP server."

            confidence = parsed["confidence"]
            logger.info(
                "predict PASSED | label={} | confidence={} | nodes={} | truncated={}",
                parsed["label"],
                confidence,
                parsed.get("num_nodes"),
                parsed.get("truncated"),
            )

            # ── Step 5: reentrancy contract should score higher than safe ──
            # The test contract has call{value: amount} pattern — real model should flag it.
            logger.info(
                "Contract has reentrancy pattern — confidence={} (expect > 0.5 for vulnerable)",
                confidence,
            )
if __name__ == "__main__":
    asyncio.run(run_smoke_test())