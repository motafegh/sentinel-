"""
MCP server — sentinel-graph-inspector (Phase 2)
Transport: SSE (HTTP) on port 8013

WHAT CHANGED FROM PHASE 1
──────────────────────────
Phase 1 scored functions using Slither structural signals (detector hits,
external call count, state writes). The scores were Slither-proxy signals,
not actual model attention.

Phase 2 calls the ML inference API's /hotspots endpoint, which returns
per-function GNN embedding-norm scores — the real signal the model used to
generate its predictions. This makes the hotspot data ground-truth attribution
rather than a correlated proxy.

Fallback chain (in order):
  1. ML API /hotspots  — real GNN attention scores  (preferred)
  2. Slither analysis  — structural proxy scoring    (if ML API unreachable)
  3. Mock data         — deterministic stub          (if Slither unavailable or MOCK_MODE)

ANALYSIS MODES
──────────────
  "gnn_attention"  — /hotspots returned valid data; scores are real model signal
  "slither"        — ML API unreachable; Slither structural scoring used
  "mock"           — both unavailable; deterministic stub returned

Tools:
  get_graph_hotspots(contract_code, flagged_classes)
    → ranked list of suspicious functions per vulnerability class
    → graph topology metadata (node/edge counts, function count)
    → analysis_mode indicating which backend scored the hotspots
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import httpx
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

_SERVER_PORT:      int  = int(os.getenv("MCP_GRAPH_INSPECTOR_PORT", "8013"))
_MOCK_MODE:        bool = os.getenv("GRAPH_INSPECTOR_MOCK", "false").lower() == "true"

# ML inference API base URL — override via env var if running on a different host.
_ML_API_URL:       str  = os.getenv("SENTINEL_ML_API_URL", "http://localhost:8000")
_HOTSPOTS_TIMEOUT: float = float(os.getenv("GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT", "60"))

# Vulnerability class → structural features that indicate suspicion.
# Used by the Slither fallback path only.
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

# Slither detector → vulnerability class (Slither fallback path).
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
                "for each flagged vulnerability class. "
                "Phase 2: uses real GNN attention scores from the ML model (not Slither proxy). "
                "Falls back to Slither structural analysis if the ML API is unreachable. "
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
                            "Vulnerability classes to filter hotspots for "
                            "(e.g. ['Reentrancy', 'ExternalBug']). "
                            "Empty list returns hotspots for all flagged classes."
                        ),
                        "default": [],
                    },
                },
                "required": ["contract_code"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Phase 2: GNN attention hotspots via ML API /hotspots
# ---------------------------------------------------------------------------

async def _analyze_hotspots_gnn(
    contract_code: str,
    flagged_classes: list[str],
) -> dict[str, Any] | None:
    """
    Call ML API POST /hotspots and transform the response into the tool's
    standard output format.

    Returns None if the ML API is unreachable or returns an error — the
    caller falls back to Slither analysis.

    The /hotspots endpoint returns GNN embedding-norm scores: the L2 norm of
    each function-level node's embedding after the GNN encoder. Higher norm =
    the GNN concentrated more structural message-passing signal on that node.
    This is the real model signal, not a Slither-based proxy.
    """
    try:
        async with httpx.AsyncClient(timeout=_HOTSPOTS_TIMEOUT) as client:
            resp = await client.post(
                f"{_ML_API_URL}/hotspots",
                json={"source_code": contract_code},
            )
        if resp.status_code != 200:
            logger.warning(
                "graph_inspector | /hotspots returned {} — falling back to Slither",
                resp.status_code,
            )
            return None

        data = resp.json()
    except httpx.RequestError as exc:
        logger.warning("graph_inspector | ML API unreachable ({}): {} — falling back to Slither", type(exc).__name__, exc)
        return None

    raw_hotspots = data.get("hotspots", [])
    if not raw_hotspots:
        # Model returned no function nodes (e.g. interface-only contract).
        # Still return a valid result rather than falling back.
        pass

    # Filter to flagged classes when provided.
    # GNN hotspots are class-agnostic (embedding norm is per-function, not per-class).
    # We annotate each hotspot with the ML-flagged classes for the investigator.
    ml_flagged = set(
        v["vulnerability_class"]
        for v in data.get("confirmed",  []) + data.get("suspicious", [])
    )
    classes_of_interest = set(flagged_classes) if flagged_classes else ml_flagged

    hotspots = []
    for h in raw_hotspots:
        hotspots.append({
            "contract":              _infer_contract_name(h["fn_name"]),
            "function":              _short_fn_name(h["fn_name"]),
            "fn_name_canonical":     h["fn_name"],
            "lines":                 h.get("lines", []),
            "vulnerability_classes": sorted(classes_of_interest),
            "score":                 h["score"],
            "node_id":               h["node_id"],
            "node_type":             h.get("node_type", "FUNCTION"),
            "signals":               [f"gnn_norm={h['score']:.3f}"],
            "source":                "gnn_attention",
        })

    stats = data.get("hotspot_stats", {})
    return {
        "hotspots": hotspots[:20],
        "graph_stats": {
            "num_functions":  stats.get("total_function_nodes", len(hotspots)),
            "num_nodes":      stats.get("num_nodes", 0),
            "num_contracts":  1,
            "has_interfaces": False,
            "num_hotspots":   len(hotspots),
            # Also surface the ML verdict so callers have full context
            "ml_label":       data.get("label", "unknown"),
            "ml_confirmed":   [v["vulnerability_class"] for v in data.get("confirmed",  [])],
            "ml_suspicious":  [v["vulnerability_class"] for v in data.get("suspicious", [])],
        },
        "analysis_mode": "gnn_attention",
    }


def _infer_contract_name(canonical_name: str) -> str:
    """Extract contract name from 'ContractName.functionName' canonical form."""
    if "." in canonical_name:
        return canonical_name.split(".")[0]
    return "unknown"


def _short_fn_name(canonical_name: str) -> str:
    """Extract bare function name from 'ContractName.functionName' canonical form."""
    if "." in canonical_name:
        return canonical_name.split(".", 1)[1]
    return canonical_name


# ---------------------------------------------------------------------------
# Slither fallback analysis (Phase 1 logic, kept as fallback)
# ---------------------------------------------------------------------------

def _analyze_hotspots_slither(
    contract_code: str,
    flagged_classes: list[str],
) -> dict[str, Any]:
    """
    Slither-based hotspot scoring — Phase 1 logic, used as fallback when the
    ML API /hotspots endpoint is unreachable.

    Scoring:
        score = detector_hits × impact_weight
              + external_call_count × 0.5
              + state_write_count   × 0.3
              + high_level_calls    × 0.4
              + is_external_facing  × 0.2
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

        # Scope Slither to relevant detectors when flagged_classes provided.
        try:
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
        except ImportError:
            pass  # routing module not available outside agents env

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
                    "source": "slither",
                }
            return fn_scores[key]

        # Score from detector findings
        for result in sl.run_detectors():
            for finding in result:
                detector     = finding.get("check", "")
                impact       = finding.get("impact", "")
                impact_weight = {"High": 3.0, "Medium": 2.0, "Low": 1.0}.get(impact, 0.5)
                cls          = _DETECTOR_CLASS_MAP.get(detector, "")

                if flagged_classes and cls not in flagged_classes:
                    continue

                for elem in finding.get("elements", []):
                    src       = elem.get("source_mapping", {})
                    lines     = src.get("lines", [])
                    fn_name   = elem.get("name", "?") if elem.get("type") == "function" else "?"
                    con_name  = elem.get("type_specific_fields", {}).get("parent", {}).get("name", "unknown")

                    entry = _get_or_create(con_name, fn_name)
                    entry["lines"].update(int(ln) for ln in lines if isinstance(ln, int))
                    if cls:
                        entry["vulnerability_classes"].add(cls)
                        entry["signals"].append(f"{detector}({impact})")
                    entry["score"] += impact_weight

        # Score from function structural features
        for contract in sl.contracts:
            if contract.is_interface:
                continue
            for fn in contract.functions_and_modifiers:
                entry = _get_or_create(contract.name, fn.name)

                ext_calls = getattr(fn, "external_calls_as_expressions", [])
                if ext_calls:
                    for cls in ["Reentrancy", "ExternalBug", "MishandledException"]:
                        if not flagged_classes or cls in flagged_classes:
                            entry["vulnerability_classes"].add(cls)
                    entry["score"]  += len(ext_calls) * 0.5
                    entry["signals"].append(f"external_calls={len(ext_calls)}")

                state_writes = getattr(fn, "state_variables_written", [])
                if state_writes and (not flagged_classes or "Reentrancy" in flagged_classes):
                    entry["score"]  += 0.3
                    entry["signals"].append(f"state_writes={len(state_writes)}")

                hl_calls = getattr(fn, "high_level_calls", [])
                if hl_calls and (not flagged_classes or "ExternalBug" in flagged_classes):
                    entry["vulnerability_classes"].add("ExternalBug")
                    entry["score"]  += len(hl_calls) * 0.4
                    entry["signals"].append(f"high_level_calls={len(hl_calls)}")

                if getattr(fn, "visibility", "") in ("public", "external"):
                    entry["score"] += 0.2

                src = getattr(fn, "source_mapping", None)
                if src:
                    fn_lines = getattr(src, "lines", [])
                    entry["lines"].update(int(ln) for ln in fn_lines if isinstance(ln, int))

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
                "source":                "slither",
            })

        hotspots.sort(key=lambda h: h["score"], reverse=True)
        all_fns = [fn for c in sl.contracts for fn in c.functions_and_modifiers]

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        return {
            "hotspots":      hotspots[:20],
            "graph_stats": {
                "num_contracts":  len(sl.contracts),
                "num_functions":  len(all_fns),
                "has_interfaces": any(c.is_interface for c in sl.contracts),
                "num_hotspots":   len(hotspots),
            },
            "analysis_mode": "slither",
        }

    except ImportError:
        logger.warning("graph_inspector | slither not installed — returning mock")
        return _mock_hotspots(flagged_classes)
    except Exception as exc:
        logger.error("graph_inspector | Slither analysis failed: {}", exc)
        return _mock_hotspots(flagged_classes)


def _mock_hotspots(flagged_classes: list[str]) -> dict[str, Any]:
    """Deterministic mock — returned when both GNN and Slither are unavailable."""
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
                "source":                "mock",
            },
            {
                "contract":              "MockVault",
                "function":              "deposit",
                "lines":                 [28, 29, 30],
                "vulnerability_classes": classes[:1],
                "score":                 1.5,
                "signals":               ["state_writes=3"],
                "source":                "mock",
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

    contract_code   = arguments.get("contract_code", "")
    flagged_classes = arguments.get("flagged_classes", [])

    if not contract_code.strip():
        return [TextContent(type="text", text=json.dumps({
            "error": "contract_code is required and must not be empty."
        }))]

    if _MOCK_MODE:
        result = _mock_hotspots(flagged_classes)
    else:
        # Try GNN attention first (Phase 2), fall back to Slither (Phase 1).
        result = await _analyze_hotspots_gnn(contract_code, flagged_classes)
        if result is None:
            result = _analyze_hotspots_slither(contract_code, flagged_classes)

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
    return JSONResponse({
        "status":   "ok",
        "server":   "sentinel-graph-inspector",
        "port":     _SERVER_PORT,
        "phase":    "2",
        "backends": ["gnn_attention", "slither", "mock"],
    })


# ---------------------------------------------------------------------------
# ASGI app + entry point
# ---------------------------------------------------------------------------

app = Starlette(routes=[
    Route("/health",    health),
    Route("/sse",       handle_sse),
    Mount("/messages/", routes=[Route("/{path:path}", handle_messages, methods=["POST"])]),
])


def run_server() -> None:
    logger.info("sentinel-graph-inspector MCP server starting on port {} (Phase 2)", _SERVER_PORT)
    uvicorn.run(app, host="0.0.0.0", port=_SERVER_PORT, log_level="warning")


if __name__ == "__main__":
    run_server()
