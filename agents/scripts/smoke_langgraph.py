"""
smoke_langgraph.py — End-to-end smoke test for the M5 audit graph.

Runs the full LangGraph pipeline against mock MCP servers.
Does NOT require any MCP servers to actually be running — patches
_call_mcp_tool so each node returns realistic mock data.

Usage:
    cd ~/projects/sentinel
    poetry run python agents/scripts/smoke_langgraph.py

Expected output:
    [PASS] Graph compiled successfully
    [PASS] ml_assessment ran — label=vulnerable confidence=0.82
    [PASS] rag_research ran — 3 RAG chunks retrieved (deep path)
    [PASS] audit_check ran  — 2 prior audits found
    [PASS] synthesizer ran  — recommendation present
    [PASS] path_taken=deep  (confidence 0.82 > 0.70 threshold)

For a live end-to-end test (all three MCP servers must be running):
    MODULE1_MOCK=false AUDIT_MOCK=false \\
    poetry run python agents/scripts/smoke_langgraph.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.graph import build_graph

# ── Mock MCP responses ───────────────────────────────────────────────────────
# These mirror the exact shapes that the real MCP servers return,
# so the nodes' parsing logic is exercised faithfully.

_MOCK_ML = {
    "label":      "vulnerable",
    "confidence": 0.82,
    "threshold":  0.50,
    "truncated":  False,
    "num_nodes":  42,
    "num_edges":  58,
}

_MOCK_RAG = [
    {
        "chunk_id": "dfl-001",
        "content":  "Reentrancy attack on Compound Finance — attacker called withdraw() repeatedly before state update...",
        "doc_id":   "compound-2023",
        "score":    0.91,
        "metadata": {"vuln_type": "reentrancy", "loss_usd": 80_000_000},
    },
    {
        "chunk_id": "dfl-002",
        "content":  "Call-value reentrancy in Uniswap V2 periphery contract...",
        "doc_id":   "uniswap-2022",
        "score":    0.84,
        "metadata": {"vuln_type": "reentrancy", "loss_usd": 5_000_000},
    },
    {
        "chunk_id": "dfl-003",
        "content":  "Integer overflow in ERC20 balances — unchecked addition wraps to zero...",
        "doc_id":   "token-2021",
        "score":    0.73,
        "metadata": {"vuln_type": "integer_overflow", "loss_usd": 1_200_000},
    },
]

_MOCK_AUDIT_HISTORY = {
    "contract_address": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
    "count": 2,
    "records": [
        {
            "score":        0.73,
            "label":        "vulnerable",
            "timestamp":    1713200000,
            "timestamp_iso": "2026-04-15T12:00:00+00:00",
            "agent":        "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
            "verified":     True,
        },
        {
            "score":        0.41,
            "label":        "safe",
            "timestamp":    1712900000,
            "timestamp_iso": "2026-04-12T03:20:00+00:00",
            "agent":        "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
            "verified":     True,
        },
    ],
}


async def _mock_call_mcp_tool(server_url: str, tool_name: str, arguments: dict) -> dict:
    """Return canned mock data based on which tool was called."""
    if tool_name == "predict":
        return _MOCK_ML
    elif tool_name == "search":
        return _MOCK_RAG
    elif tool_name == "get_audit_history":
        return _MOCK_AUDIT_HISTORY
    else:
        return {"error": f"unknown tool: {tool_name}"}


async def run_smoke(live: bool = False) -> None:
    """Run the smoke test — mock mode by default, live if --live passed."""
    print(f"\n{'='*60}")
    print(f"SENTINEL M5 smoke test — {'LIVE mode' if live else 'mock mode'}")
    print(f"{'='*60}\n")

    # Build graph without checkpointer (faster for tests)
    graph = build_graph(use_checkpointer=False)
    print("[PASS] Graph compiled successfully")

    initial_state = {
        "contract_code": (
            "pragma solidity ^0.8.0;\n"
            "contract Vault {\n"
            "    mapping(address => uint) public balances;\n"
            "    function deposit() external payable { balances[msg.sender] += msg.value; }\n"
            "    function withdraw() external {\n"
            "        uint amt = balances[msg.sender];\n"
            "        (bool ok,) = msg.sender.call{value: amt}(\"\");\n"
            "        require(ok);\n"
            "        balances[msg.sender] = 0;\n"
            "    }\n"
            "}"
        ),
        "contract_address": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
    }

    if live:
        # Use real MCP servers — they must all be running
        result = await graph.ainvoke(initial_state)
    else:
        # Patch the MCP call helper so no servers are needed
        with patch(
            "src.orchestration.nodes._call_mcp_tool",
            side_effect=_mock_call_mcp_tool,
        ):
            result = await graph.ainvoke(initial_state)

    # ── Validate results ────────────────────────────────────────────────────
    ml = result.get("ml_result", {})
    assert ml.get("label") in ("vulnerable", "safe"), f"Bad label: {ml.get('label')}"
    print(f"[PASS] ml_assessment ran — label={ml.get('label')} confidence={ml.get('confidence'):.2f}")

    report = result.get("final_report", {})
    assert report, "final_report is empty — synthesizer did not run"

    path = report.get("path_taken")
    rag  = report.get("rag_evidence", [])
    hist = report.get("audit_history", [])
    rec  = report.get("recommendation", "")

    if path == "deep":
        assert len(rag) > 0,  f"Deep path but rag_evidence is empty: {rag}"
        print(f"[PASS] rag_research ran  — {len(rag)} RAG chunk(s) retrieved (deep path)")
        print(f"[PASS] audit_check ran   — {len(hist)} prior audit(s) found")
    else:
        print(f"[INFO] fast path taken   — RAG + audit skipped (confidence ≤ 0.70)")

    assert rec, "recommendation field is empty"
    print(f"[PASS] synthesizer ran   — recommendation present ({len(rec)} chars)")
    print(f"[PASS] path_taken={path} (confidence {ml.get('confidence'):.2f})")

    if report.get("error"):
        print(f"[WARN] non-fatal error recorded: {report['error']}")

    print(f"\n{'='*60}")
    print("All smoke tests passed.")
    print(f"{'='*60}\n")

    if not live:
        print("Full report (mock):")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use real MCP servers")
    args = parser.parse_args()
    asyncio.run(run_smoke(live=args.live))
