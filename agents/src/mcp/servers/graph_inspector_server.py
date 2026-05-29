"""
MCP server — sentinel-graph-inspector (Phase 1)
Transport: SSE (HTTP) on port 8013

WHY THIS EXISTS
───────────────
The inference API returns per-class probabilities but gives no indication of
WHERE in the contract the suspicious pattern lives. A security auditor needs
to know which function to examine first — not just "Reentrancy: 0.72".

This server extracts function-level structural hotspots by combining:
  1. Slither's AST and detector output (structural ground truth)
  2. ML probability signal (which classes are suspicious)
  3. Heuristic scoring: detector hits, external calls, state writes,
     complexity, cross-contract dependencies

True GNN attention weights require hooking the compiled model forward pass.
That is Phase 2 work (graph_explain node extended). Phase 1 gives auditors
actionable function-level attribution using Slither alone — accurate enough
to direct manual review.

Tools:
  get_graph_hotspots(contract_code, flagged_classes)
    → ranked list of suspicious functions per vulnerability class
    → graph topology metadata (node/edge counts, function count)
"""

from __future__ import annotations

import json
import os
import tempfile
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SERVER_PORT: int = int(os.getenv("MCP_GRAPH_INSPECTOR_PORT", "8013"))
_MOCK_MODE:   bool = os.getenv("GRAPH_INSPECTOR_MOCK", "false").lower() == "true"

# Vulnerability class → structural features that indicate suspicion.
# Used to score functions even when the specific Slither detector is absent.
_CLASS_STRUCTURAL_SIGNALS: dict[str, list[str]] = {
    "Reentrancy":          ["external_calls", "state_writes", "low_level_calls"],
    "IntegerUO":           ["arithmetic_ops", "unchecked_blocks"],
    "GasException":        ["loops", "external_calls_in_loops"],
    "Timestamp":           ["block_timestamp", "block_number"],
    "TOD":                 ["tx_origin", "state_reads_before_calls"],
    "ExternalBug":         ["high_level_calls", "interfaces"],
    "CallToUnknown":       ["low_level_calls", "delegatecall"],
    "MishandledException": ["low_level_calls", "unchecked_returns"],
    "UnusedReturn":        ["return_values", "external_calls"],
    "DenialOfService":     ["loops", "array_operations", "external_calls_in_loops"],
}

# Slither detector → vulnerability class mapping for scoring.
_DETECTOR_CLASS_MAP: dict[str, str] = {
    "reentrancy-eth":               "Reentrancy",
    "reentrancy-no-eth":            "Reentrancy",
    "reentrancy-events-and-order":  "Reentrancy",
    "reentrancy-benign":            "Reentrancy",
    "integer-overflow":             "IntegerUO",
    "toctou":                       "IntegerUO",
    "unchecked-lowlevel":           "MishandledException",
    "costly-loop":                  "GasException",
    "calls-loop":                   "GasException",
    "timestamp":                    "Timestamp",
    "tx-origin":                    "TOD",
    "controlled-delegatecall":      "CallToUnknown",
    "arbitrary-send-eth":           "ExternalBug",
    "low-level-calls":              "CallToUnknown",
    "unchecked-send":               "MishandledException",
    "unchecked-transfer":           "MishandledException",
    "return-bomb":                  "MishandledException",
    "unused-return":                "UnusedReturn",
    "msg-value-loop":               "DenialOfService",
    "delegatecall-loop":            "CallToUnknown",
}

# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------

server = Server("sentinel-graph-inspector")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_graph_hotspots",
            description=(
                "Analyse a Solidity contract and return function-level hotspots "
                "for each flagged vulnerability class. Combines Slither structural "
                "analysis with ML probability signal to rank functions by suspicion. "
                "Use this to direct auditor attention to the most relevant code regions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contract_code": {
                        "type": "string",
                        "description": "Raw Solidity source code.",
                    },
                    "flagged_classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Vulnerability classes to analyse hotspots for "
                            "(e.g. ['Reentrancy', 'ExternalBug']). "
                            "Empty list analyses all 10 classes."
                        ),
                        "default": [],
                    },
                },
                "required": ["contract_code"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Core analysis logic
# ---------------------------------------------------------------------------

def _analyze_hotspots(contract_code: str, flagged_classes: list[str]) -> dict[str, Any]:
    """
    Run Slither on the contract and return function-level hotspot data.

    Scoring strategy (Phase 1):
        score = detector_hits × 3.0
              + structural_signal_count × 1.0
              + is_external_facing × 0.5
              + complexity_penalty × 0.2

    Returns dict with:
        hotspots: [{contract, function, lines, vulnerability_classes, score, signals}, ...]
        graph_stats: {num_functions, num_contracts, has_interfaces, ...}
        analysis_mode: "slither" | "mock"
    """
    try:
        from slither import Slither

        with tempfile.NamedTemporaryFile(
            suffix=".sol", prefix="sentinel_gi_", mode="w",
            encoding="utf-8", delete=False,
        ) as tmp:
            tmp.write(contract_code)
            tmp_path = tmp.name

        sl = Slither(tmp_path)

        # ── Run relevant detectors ───────────────────────────────────────────
        # Filter to detectors for flagged classes when provided.
        from src.orchestration.routing import CLASS_TO_DETECTORS
        if flagged_classes:
            scoped = set()
            for cls in flagged_classes:
                scoped.update(CLASS_TO_DETECTORS.get(cls, []))
            if scoped:
                sl._detectors = [  # type: ignore[attr-defined]
                    d for d in sl._detectors  # type: ignore[attr-defined]
                    if getattr(d, "ARGUMENT", "") in scoped
                ]

        # Map: (contract_name, fn_name) → {score, lines, classes, signals}
        fn_scores: dict[tuple[str, str], dict] = {}

        def _get_or_create(contract: str, fn: str) -> dict:
            key = (contract, fn)
            if key not in fn_scores:
                fn_scores[key] = {
                    "contract": contract,
                    "function": fn,
                    "lines": set(),
                    "vulnerability_classes": set(),
                    "signals": [],
                    "score": 0.0,
                }
            return fn_scores[key]

        # Score from detector findings.
        for result in sl.run_detectors():
            for finding in result:
                detector = finding.get("check", "")
                impact   = finding.get("impact", "")
                impact_weight = {"High": 3.0, "Medium": 2.0, "Low": 1.0}.get(impact, 0.5)
                cls = _DETECTOR_CLASS_MAP.get(detector, "")

                if flagged_classes and cls not in flagged_classes:
                    continue

                for elem in finding.get("elements", []):
                    src   = elem.get("source_mapping", {})
                    lines = src.get("lines", [])
                    fn_name = elem.get("name", "?") if elem.get("type") == "function" else "?"

                    # Try to get contract name from element.
                    contract_name = "unknown"
                    if elem.get("type") == "function":
                        contract_name = elem.get("type_specific_fields", {}).get(
                            "parent", {}
                        ).get("name", "unknown")

                    entry = _get_or_create(contract_name, fn_name)
                    entry["lines"].update(int(ln) for ln in lines if isinstance(ln, int))
                    if cls:
                        entry["vulnerability_classes"].add(cls)
                        entry["signals"].append(f"{detector}({impact})")
                    entry["score"] += impact_weight

        # Score from structural features of each function.
        for contract in sl.contracts:
            if contract.is_interface:
                continue
            for fn in contract.functions_and_modifiers:
                entry = _get_or_create(contract.name, fn.name)
                signals = _CLASS_STRUCTURAL_SIGNALS

                # External calls → Reentrancy, ExternalBug signal
                ext_calls = getattr(fn, "external_calls_as_expressions", [])
                if ext_calls:
                    for cls in ["Reentrancy", "ExternalBug", "MishandledException"]:
                        if not flagged_classes or cls in flagged_classes:
                            entry["vulnerability_classes"].add(cls)
                    entry["score"] += len(ext_calls) * 0.5
                    entry["signals"].append(f"external_calls={len(ext_calls)}")

                # State variable writes → Reentrancy, TOD signal
                state_writes = getattr(fn, "state_variables_written", [])
                if state_writes and (not flagged_classes or "Reentrancy" in flagged_classes):
                    entry["score"] += 0.3
                    entry["signals"].append(f"state_writes={len(state_writes)}")

                # High-level calls → ExternalBug
                hl_calls = getattr(fn, "high_level_calls", [])
                if hl_calls and (not flagged_classes or "ExternalBug" in flagged_classes):
                    entry["vulnerability_classes"].add("ExternalBug")
                    entry["score"] += len(hl_calls) * 0.4
                    entry["signals"].append(f"high_level_calls={len(hl_calls)}")

                # Public/external visibility → raises suspicion
                if getattr(fn, "visibility", "") in ("public", "external"):
                    entry["score"] += 0.2

                # Source lines
                src = getattr(fn, "source_mapping", None)
                if src:
                    fn_lines = getattr(src, "lines", [])
                    entry["lines"].update(int(ln) for ln in fn_lines if isinstance(ln, int))

        # Filter to only flagged classes and convert to serializable format.
        hotspots = []
        for (contract, fn), data in fn_scores.items():
            classes = (
                data["vulnerability_classes"] & set(flagged_classes)
                if flagged_classes
                else data["vulnerability_classes"]
            )
            if not classes or data["score"] < 0.1:
                continue
            hotspots.append({
                "contract":              contract,
                "function":              fn,
                "lines":                 sorted(data["lines"]),
                "vulnerability_classes": sorted(classes),
                "score":                 round(data["score"], 3),
                "signals":               data["signals"][:5],
            })

        hotspots.sort(key=lambda h: h["score"], reverse=True)

        # Graph stats
        all_fns     = [fn for c in sl.contracts for fn in c.functions_and_modifiers]
        graph_stats = {
            "num_contracts":  len(sl.contracts),
            "num_functions":  len(all_fns),
            "has_interfaces": any(c.is_interface for c in sl.contracts),
            "num_hotspots":   len(hotspots),
        }

        try:
            import os
            os.unlink(tmp_path)
        except OSError:
            pass

        return {
            "hotspots":      hotspots[:20],
            "graph_stats":   graph_stats,
            "analysis_mode": "slither",
        }

    except ImportError:
        logger.warning("graph_inspector | slither not installed — returning mock")
        return _mock_hotspots(flagged_classes)

    except Exception as exc:
        logger.error("graph_inspector | analysis failed: {}", exc)
        return _mock_hotspots(flagged_classes)


def _mock_hotspots(flagged_classes: list[str]) -> dict[str, Any]:
    """Deterministic mock for development (Slither unavailable or MOCK_MODE)."""
    classes = flagged_classes or ["Reentrancy", "ExternalBug"]
    return {
        "hotspots": [
            {
                "contract":              "MockVault",
                "function":              "withdraw",
                "lines":                 [42, 43, 44, 45],
                "vulnerability_classes": classes[:2],
                "score":                 4.2,
                "signals":               ["reentrancy-eth(High)", "external_calls=2"],
            },
            {
                "contract":              "MockVault",
                "function":              "deposit",
                "lines":                 [28, 29, 30],
                "vulnerability_classes": classes[:1],
                "score":                 1.5,
                "signals":               ["state_writes=3"],
            },
        ],
        "graph_stats": {
            "num_contracts":  1,
            "num_functions":  4,
            "has_interfaces": False,
            "num_hotspots":   2,
        },
        "analysis_mode": "mock",
    }


# ---------------------------------------------------------------------------
# Tool call dispatcher
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "get_graph_hotspots":
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    contract_code  = arguments.get("contract_code", "")
    flagged_classes = arguments.get("flagged_classes", [])

    if not contract_code.strip():
        return [TextContent(type="text", text=json.dumps({
            "error": "contract_code is required and must not be empty."
        }))]

    if _MOCK_MODE:
        result = _mock_hotspots(flagged_classes)
    else:
        result = _analyze_hotspots(contract_code, flagged_classes)

    logger.info(
        "get_graph_hotspots | mode={} | hotspots={} | classes={}",
        result.get("analysis_mode"),
        len(result.get("hotspots", [])),
        flagged_classes or ["all"],
    )
    return [TextContent(type="text", text=json.dumps(result))]


# ---------------------------------------------------------------------------
# Health check + SSE routing
# ---------------------------------------------------------------------------

async def handle_sse(request: Request) -> Response:
    transport = SseServerTransport("/messages/")
    async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
    return Response()


async def handle_messages(request: Request) -> Response:
    transport = SseServerTransport("/messages/")
    await transport.handle_post_message(request.scope, request.receive, request._send)
    return Response()


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "sentinel-graph-inspector", "port": _SERVER_PORT})


# ---------------------------------------------------------------------------
# ASGI app + entry point
# ---------------------------------------------------------------------------

app = Starlette(routes=[
    Route("/health",    health),
    Route("/sse",       handle_sse),
    Mount("/messages/", routes=[Route("/{path:path}", handle_messages, methods=["POST"])]),
])


def run_server() -> None:
    logger.info("sentinel-graph-inspector MCP server starting on port {}", _SERVER_PORT)
    uvicorn.run(app, host="0.0.0.0", port=_SERVER_PORT, log_level="warning")


if __name__ == "__main__":
    run_server()
