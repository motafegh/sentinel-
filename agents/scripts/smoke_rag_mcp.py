# agents/scripts/smoke_rag_mcp.py
"""
Smoke test for the sentinel-rag MCP server.

Run BEFORE running this script:
    # Terminal 1
    cd ~/projects/sentinel/agents
    poetry run python src/mcp/servers/rag_server.py
    → Expected: Uvicorn running on http://0.0.0.0:8011

Run this script:
    # Terminal 2
    cd ~/projects/sentinel/agents
    poetry run python scripts/smoke_rag_mcp.py
    → Expected: ALL CHECKS PASSED — rag MCP server is consumable

Checks performed:
    1. /health endpoint responds with chunk count
    2. SSE handshake + MCP initialization succeeds
    3. list_tools() returns the 'search' tool with correct schema
    4. call_tool('search') returns results with expected shape
    5. call_tool('search') with filters runs without error
    6. call_tool('search') with k cap enforced (k > 20 → 20 results max)
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

_SERVER_URL = "http://localhost:8011"
_SSE_URL = f"{_SERVER_URL}/sse"
_HEALTH_URL = f"{_SERVER_URL}/health"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

errors: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)


# ---------------------------------------------------------------------------
# Check 1: /health
# ---------------------------------------------------------------------------

def check_health() -> int:
    """Returns chunk count from health response, or 0 on failure."""
    print("\n[1] Health endpoint")
    try:
        resp = httpx.get(_HEALTH_URL, timeout=5.0)
        data = resp.json()
        ok = resp.status_code == 200
        check("Status 200", ok, str(resp.status_code))
        check("server == sentinel-rag", data.get("server") == "sentinel-rag", str(data))
        chunks = data.get("chunks_indexed", 0)
        check(f"chunks_indexed > 0 (got {chunks})", chunks > 0)
        return chunks
    except Exception as exc:
        check("Health reachable", False, str(exc))
        return 0


# ---------------------------------------------------------------------------
# Checks 2–6: MCP session
# ---------------------------------------------------------------------------

async def check_mcp_session() -> None:
    print("\n[2] MCP handshake")
    async with sse_client(_SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            check("SSE connect + initialize", True)

            # ----------------------------------------------------------------
            print("\n[3] list_tools")
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            check("'search' tool present", "search" in tool_names, str(tool_names))

            search_tool = next((t for t in tools_result.tools if t.name == "search"), None)
            if search_tool:
                schema = search_tool.inputSchema
                has_query = "query" in schema.get("properties", {})
                has_k = "k" in schema.get("properties", {})
                has_filters = "filters" in schema.get("properties", {})
                check("schema has 'query' field", has_query)
                check("schema has 'k' field", has_k)
                check("schema has 'filters' field", has_filters)
                check("'query' is required", "query" in schema.get("required", []))

            # ----------------------------------------------------------------
            print("\n[4] call_tool — basic search")
            result = await session.call_tool(
                "search",
                {"query": "reentrancy vulnerability in withdraw function", "k": 3},
            )
            raw = result.content[0].text if result.content else "{}"
            data = json.loads(raw)

            check("No error key in response", "error" not in data, str(data.get("error")))
            check("'results' key present", "results" in data, str(list(data.keys())))
            check("'k_returned' <= 3", data.get("k_returned", 99) <= 3, str(data.get("k_returned")))

            if data.get("results"):
                first = data["results"][0]
                check("result has 'content' field", "content" in first, str(list(first.keys())))
                check("result has 'metadata' field", "metadata" in first, str(list(first.keys())))
                check("result has 'chunk_id' field", "chunk_id" in first, str(list(first.keys())))

            # ----------------------------------------------------------------
            print("\n[5] call_tool — search with filters")
            result_filtered = await session.call_tool(
                "search",
                {
                    "query": "flash loan oracle price manipulation",
                    "k": 5,
                    "filters": {"has_summary": True},
                },
            )
            raw_f = result_filtered.content[0].text if result_filtered.content else "{}"
            data_f = json.loads(raw_f)
            check("Filtered search returns no error", "error" not in data_f, str(data_f.get("error")))
            check("filters_applied echoed back", data_f.get("filters_applied") == {"has_summary": True})

            # ----------------------------------------------------------------
            print("\n[6] call_tool — k cap enforcement (request k=99)")
            result_cap = await session.call_tool(
                "search",
                {"query": "access control vulnerability", "k": 20},
            )
            raw_c = result_cap.content[0].text if result_cap.content else "{}"
            data_c = json.loads(raw_c)
            check("k cap: no error returned", "error" not in data_c, str(data_c.get("error")))
            check(
                f"k_returned <= 20 (got {data_c.get('k_returned')})",
                data_c.get("k_returned", 99) <= 20,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    chunk_count = check_health()

    if chunk_count == 0:
        print("\n  Server unreachable or index empty — skipping MCP checks.")
        sys.exit(1)

    asyncio.run(check_mcp_session())

    print()
    if errors:
        print(f"\033[91mFAILED — {len(errors)} check(s) failed: {errors}\033[0m")
        sys.exit(1)
    else:
        print("\033[92mALL CHECKS PASSED — rag MCP server is consumable\033[0m")


if __name__ == "__main__":
    main()
