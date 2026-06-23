"""
MCP server — sentinel-representation (Phase 2, WS5 2026-06-22)
Transport: SSE (HTTP) on port 8014

WHAT THIS IS
────────────
WS5 (per docs/plan/agents/2026-06-21-agents-redesign/01_MASTER_PLAN.md)
wraps data_module's existing graph/CFG extraction code as MCP tools the
agents module can call. Reuses tested code rather than re-inventing
AST/CFG parsing inside agents.

This server exposes:
  - get_function_cfgs(contract_code) → per-function CFG + derived signals
    (CEI violation detection, call/write/loop counts, max depth)

The underlying implementation is `data_module/sentinel_data/representation/
cfg_builder.py::build_cfg` — the same code path the ML training pipeline
uses. By exposing it as an MCP tool, agents get:
  - Direct access to per-function control-flow graphs (CALL/WRITE/READ/CHECK
    node types) without going through the ML model's interpretation.
  - Explicit CEI violation detection (CALL nodes appearing AFTER WRITE
    nodes in the same function — the classic reentrancy pattern).
  - Loop and depth metrics for GasException / DoS classes.

This complements (does NOT replace) graph_inspector_server's hotspot
analysis: graph_inspector uses ML attention weights, this server returns
the raw structural data.

ANALYSIS MODES
──────────────
  "data_module_cfgs"  — build_cfg() returned valid per-function CFGs
  "slither_error"     — build_cfg() returned an error (solc/Slither failed);
                        functions list is empty + the error string is included
  "data_module_unavailable"  — the data_module import failed; caller should
                               fall back to a Slither-only analysis
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import asdict
from typing import Any

import uvicorn
from loguru import logger
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from src.orchestration.timeouts import DEFAULT_GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT_S
from src.orchestration.timing import step_timer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SERVER_PORT: int = int(os.getenv("MCP_REPRESENTATION_PORT", "8014"))
_MOCK_MODE:   bool = os.getenv("REPRESENTATION_MOCK", "false").lower() == "true"

# Reuse the same default timeout shape as graph_inspector for consistency.
_FUNCTION_CFGS_TIMEOUT: float = float(os.getenv(
    "REPRESENTATION_FUNCTION_CFGS_TIMEOUT",
    str(DEFAULT_GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT_S),
))

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

server: Server = Server("sentinel-representation")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_function_cfgs",
            description=(
                "Return per-function control-flow graphs (CFG) for a Solidity "
                "contract, with derived vulnerability signals. Reuses data_module's "
                "build_cfg() (the same code path as the ML training pipeline). "
                "Each function gets: a node-type summary (CALL/WRITE/READ/CHECK/"
                "ARITH counts), structural metrics (num_loops, max_depth), and "
                "an explicit list of CEI violations (CALL nodes appearing after "
                "WRITE nodes — the classic reentrancy pattern). Use this when the "
                "agents need direct structural control-flow data without ML "
                "interpretation. For ML-based hotspot analysis, use "
                "graph_inspector_server's get_graph_hotspots instead."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_code": {
                        "type": "string",
                        "description": "The full Solidity source code of the contract.",
                    },
                    "solc_version": {
                        "type": "string",
                        "description": (
                            "Solidity compiler version string (e.g. '0.8.19'). "
                            "Defaults to whatever solc is on PATH in the agents venv."
                        ),
                        "default": "",
                    },
                },
                "required": ["contract_code"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# CEI violation detection
# ---------------------------------------------------------------------------

# Node-type labels from cfg_builder._cfg_node_type_str. Kept as a constant
# here (not imported) so the server is self-contained and survives a data_module
# refactor.
_NODE_CALL:  str = "CFG_NODE_CALL"
_NODE_WRITE: str = "CFG_NODE_WRITE"
_NODE_READ:  str = "CFG_NODE_READ"
_NODE_CHECK: str = "CFG_NODE_CHECK"
_NODE_ARITH: str = "CFG_NODE_ARITH"
_NODE_OTHER: str = "CFG_NODE_OTHER"

# A control-flow edge src→dst means "src executes before dst" (in linear order
# of the sorted node list). A CEI violation = a WRITE node reachable to a CALL
# node in the same function (i.e. a state write happens before an external call,
# but the call's effects can re-enter the function and find the write undone).
#
# Detection here uses a simple forward DFS from each WRITE node: does the WRITE
# reach a CALL? If yes, that's a CEI violation. This is structurally what
# matter (not "is the call on the same line as the write" — that's the ML
# model's job via its phase2 CEI aux head). The CfgArtifact gives us the
# explicit edges, so the DFS is precise.
def _detect_cei_violations(edges: list[dict], nodes: list[dict]) -> list[dict]:
    """
    Find write→call CEI violations in a function's CFG.

    Returns a list of {write_index, call_index, write_lines, call_lines, path}
    dicts. Path is the list of node indices from WRITE to CALL (inclusive of both
    ends), useful for debugging.
    """
    if not edges or not nodes:
        return []

    # Build adjacency list (forward direction).
    adj: dict[int, list[int]] = {n["index"]: [] for n in nodes}
    for e in edges:
        if e["src"] in adj:
            adj[e["src"]].append(e["dst"])

    # Index nodes by type for O(1) lookup.
    writes = [n for n in nodes if n["type"] == _NODE_WRITE]
    calls  = [n for n in nodes if n["type"] == _NODE_CALL]
    if not writes or not calls:
        return []

    violations: list[dict] = []

    def _dfs(start: int, target_set: set[int], max_depth: int = 64) -> list[list[int]]:
        """Find all paths from `start` to any node in `target_set`, capped at max_depth."""
        results: list[list[int]] = []
        stack: list[tuple[int, list[int]]] = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            if cur in target_set and cur != start:
                results.append(path)
                continue  # don't extend past the first target hit per path
            if len(path) >= max_depth:
                continue
            for nxt in adj.get(cur, []):
                if nxt in path:  # cycle guard
                    continue
                stack.append((nxt, path + [nxt]))
        return results

    write_indices = {w["index"]: w for w in writes}
    call_indices  = {c["index"]: c for c in calls}
    call_set      = set(call_indices.keys())

    for w_idx, w_node in write_indices.items():
        paths = _dfs(w_idx, call_set)
        for path in paths:
            call_idx = path[-1]
            call_node = call_indices[call_idx]
            violations.append({
                "write_index":      w_idx,
                "call_index":       call_idx,
                "write_source_lines": w_node.get("source_lines", []),
                "call_source_lines":  call_node.get("source_lines", []),
                "path":             path,
            })
    return violations


def _summarise_function(cfg_fn: dict) -> dict:
    """
    Reduce a CfgFunction dict to a function-level summary.

    Input is the output of dataclasses.asdict(CfgFunction) — has keys
    canonical_name, nodes, edges, num_loops, max_depth.

    Returns a richer dict with:
      - node_type_counts: how many of each CFG node type
      - cei_violations: write→call violations
      - has_external_call: bool
      - has_state_write: bool
      - has_arithmetic: bool
      - cfg_complexity_score: heuristic (loops*3 + max_depth + num_calls + num_writes)
    """
    nodes = cfg_fn.get("nodes", [])
    edges = cfg_fn.get("edges", [])

    type_counts: dict[str, int] = {}
    for n in nodes:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1

    cei = _detect_cei_violations(edges, nodes)

    has_call   = type_counts.get(_NODE_CALL,  0) > 0
    has_write  = type_counts.get(_NODE_WRITE, 0) > 0
    has_arith  = type_counts.get(_NODE_ARITH, 0) > 0
    num_loops  = cfg_fn.get("num_loops", 0)
    max_depth  = cfg_fn.get("max_depth", 0)

    # Heuristic structural complexity score — used as a cheap ranking signal
    # for the debate / hotspot targeting. Not a learned model — just a
    # transparent formula: loops dominate (DoS/Gas), then depth (complexity),
    # then the number of state-touching operations (Reentrancy/CEI).
    cfg_complexity_score = (
        num_loops * 3.0
        + max_depth
        + type_counts.get(_NODE_CALL, 0) * 1.0
        + type_counts.get(_NODE_WRITE, 0) * 0.5
    )

    return {
        "canonical_name":       cfg_fn.get("canonical_name", "?"),
        "num_nodes":            len(nodes),
        "num_edges":            len(edges),
        "num_loops":            num_loops,
        "max_depth":            max_depth,
        "node_type_counts":     type_counts,
        "has_external_call":    has_call,
        "has_state_write":      has_write,
        "has_arithmetic":       has_arith,
        "cfg_complexity_score": round(cfg_complexity_score, 3),
        "cei_violations":       cei,
        "cei_violation_count":  len(cei),
    }


# ---------------------------------------------------------------------------
# build_cfg call
# ---------------------------------------------------------------------------

def _call_build_cfg(
    contract_code: str,
    solc_version: str = "",
) -> dict[str, Any]:
    """
    Write contract_code to a temp file, call data_module's build_cfg, return
    a JSON-serialisable dict with the per-function CFG summaries.
    """
    from data_module.sentinel_data.representation.cfg_builder import build_cfg
    from data_module.sentinel_data.representation.graph_extractor import (
        GraphExtractionConfig,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".sol", prefix="sentinel_repr_", mode="w",
        encoding="utf-8", delete=False,
    ) as tmp:
        tmp.write(contract_code)
        tmp_path = tmp.name

    try:
        config = GraphExtractionConfig(
            solc_binary="solc",  # PATH resolution — agents venv has solc symlinked
            solc_version=solc_version or None,
            allow_paths=[],
        )
        artifact = build_cfg(sol_path=tmp_path, config=config, source="agents_mcp")
        # build_cfg returns a CfgArtifact. asdict() converts it fully
        # (the per-function .nodes lists of dataclasses are also converted).
        artifact_dict = asdict(artifact)
    except Exception as exc:
        logger.error("representation | build_cfg failed: {}", exc)
        return {
            "error":         f"build_cfg failed: {exc}",
            "analysis_mode": "data_module_unavailable",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Build per-function summaries
    function_summaries: list[dict] = []
    total_cei = 0
    for cfg_fn in artifact_dict.get("functions", []):
        summary = _summarise_function(cfg_fn)
        function_summaries.append(summary)
        total_cei += summary["cei_violation_count"]

    # Sort by complexity score (descending) — most suspicious first.
    function_summaries.sort(
        key=lambda s: (s["cfg_complexity_score"], s["cei_violation_count"]),
        reverse=True,
    )

    if artifact_dict.get("error"):
        mode = "slither_error"
    else:
        mode = "data_module_cfgs"

    return {
        "analysis_mode":   mode,
        "schema_version":  artifact_dict.get("schema_version", "?"),
        "extractor_version": artifact_dict.get("extractor_version", "?"),
        "solc_version":    artifact_dict.get("solc_version", "") or "default",
        "error":           artifact_dict.get("error"),
        "num_functions":   len(function_summaries),
        "total_cei_violations": total_cei,
        "functions":       function_summaries,
    }


# ---------------------------------------------------------------------------
# Mock (for testing / dev without solc)
# ---------------------------------------------------------------------------

def _mock_function_cfgs(contract_code: str) -> dict[str, Any]:
    return {
        "analysis_mode":   "mock",
        "schema_version":  "v9",
        "extractor_version": "mock",
        "solc_version":    "0.8.19",
        "error":           None,
        "num_functions":   0,
        "total_cei_violations": 0,
        "functions":       [],
    }


# ---------------------------------------------------------------------------
# Tool call dispatcher
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "get_function_cfgs":
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown tool: {name}",
        }))]

    contract_code = arguments.get("contract_code", "")
    solc_version  = arguments.get("solc_version", "")

    if not contract_code.strip():
        return [TextContent(type="text", text=json.dumps({
            "error": "contract_code is required and must not be empty.",
        }))]

    with step_timer("representation.get_function_cfgs", len_chars=len(contract_code)):
        if _MOCK_MODE:
            result = _mock_function_cfgs(contract_code)
        else:
            try:
                result = await asyncio.to_thread(
                    _call_build_cfg, contract_code, solc_version,
                )
            except NameError:
                # asyncio not imported — call directly (test contexts)
                result = _call_build_cfg(contract_code, solc_version)

        logger.info(
            "get_function_cfgs | mode={} | fns={} | cei_violations={}",
            result.get("analysis_mode"),
            result.get("num_functions", 0),
            result.get("total_cei_violations", 0),
        )
    return [TextContent(type="text", text=json.dumps(result))]


# ---------------------------------------------------------------------------
# Health + SSE
# ---------------------------------------------------------------------------

async def handle_sse(request: Request) -> Response:
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send,
    ) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
    return Response()


async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status":   "ok",
        "server":   "sentinel-representation",
        "port":     _SERVER_PORT,
        "phase":    "2",
        "backends": ["data_module_cfgs", "slither_error", "mock"],
    })


# ---------------------------------------------------------------------------
# ASGI app + entry point
# ---------------------------------------------------------------------------

sse_transport = SseServerTransport("/messages/")

app = Starlette(routes=[
    Route("/health",    health),
    Route("/sse",       handle_sse),
    Mount("/messages/", app=sse_transport.handle_post_message),
])


def run_server() -> None:
    logger.info("sentinel-representation MCP server starting on port {}", _SERVER_PORT)
    uvicorn.run(app, host="0.0.0.0", port=_SERVER_PORT, log_level="warning")


if __name__ == "__main__":
    run_server()
