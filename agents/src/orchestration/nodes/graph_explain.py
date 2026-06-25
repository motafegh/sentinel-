from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
import src.orchestration.nodes._helpers as _h

_GRAPH_INSPECTOR_URL: str = os.getenv("MCP_GRAPH_INSPECTOR_URL", "http://localhost:8013/sse")


async def graph_explain(state: AuditState) -> dict[str, Any]:
    """
    Call sentinel-graph-inspector to get function-level hotspot attribution.

    WHY THIS EXISTS
    ───────────────
    The ML model returns per-class probabilities but no indication of WHERE
    in the contract the suspicious pattern lives. graph_explain bridges this gap
    by combining Slither structural analysis with the ML probability signal to
    produce ranked function-level hotspots — directing auditor attention to the
    most suspicious code regions.

    Phase 1 (current): Slither structural proxy for attention — detector hits,
        external calls, state writes, complexity, cross-contract dependencies.
    Phase 2 (future): true GNN attention weights via forward-pass hooking.

    State updates:
        ml_hotspots       → ranked list of suspicious functions per vuln class
        graph_explanations → per-class hotspot breakdown + graph topology stats
    """
    contract_code = state.get("contract_code", "")
    if not contract_code or not contract_code.strip():
        logger.info("graph_explain | contract_code empty — skipping")
        return {"ml_hotspots": [], "graph_explanations": {}}

    ml_result  = state.get("ml_result", {})
    confirmed  = ml_result.get("confirmed",  [])
    suspicious = ml_result.get("suspicious", [])
    flagged    = confirmed + suspicious or ml_result.get("vulnerabilities", [])
    flagged_classes = [
        v.get("vulnerability_class", "")
        for v in flagged
        if v.get("vulnerability_class")
    ]

    logger.info("graph_explain | classes={}", flagged_classes or ["all"])

    try:
        result = await _h._call_mcp_tool(
            server_url=_GRAPH_INSPECTOR_URL,
            tool_name="get_graph_hotspots",
            arguments={
                "contract_code":   contract_code,
                "flagged_classes": flagged_classes,
            },
        )

        if "error" in result:
            logger.warning("graph_explain | inspector error: {}", result["error"])
            return {"ml_hotspots": [], "graph_explanations": {}}

        hotspots    = result.get("hotspots",    [])
        graph_stats = result.get("graph_stats", {})
        mode        = result.get("analysis_mode", "unknown")

        # Build per-class breakdown for state.graph_explanations
        hotspots_by_class: dict[str, list[dict]] = {}
        for cls in flagged_classes:
            hotspots_by_class[cls] = [
                h for h in hotspots
                if cls in h.get("vulnerability_classes", [])
            ]

        graph_explanations: dict[str, Any] = {
            "graph_stats":       graph_stats,
            "analysis_mode":     mode,
            "hotspots_by_class": hotspots_by_class,
        }

        # ml_hotspots: flat list matching AuditState schema
        ml_hotspots = [
            {
                "class":   h["vulnerability_classes"][0] if h["vulnerability_classes"] else "?",
                "fn_name": h["function"],
                "lines":   h["lines"],
                "node_ids": [],  # Phase 2: populated from GNN attention
                "score":   h["score"],
                "signals": h.get("signals", []),
            }
            for h in hotspots
        ]

        logger.info(
            "graph_explain complete | mode={} | hotspots={} | contracts={}",
            mode,
            len(hotspots),
            graph_stats.get("num_contracts", 0),
        )
        return {"ml_hotspots": ml_hotspots, "graph_explanations": graph_explanations}

    except Exception as exc:
        logger.error("graph_explain failed: {}", exc)
        return {"ml_hotspots": [], "graph_explanations": {}}
