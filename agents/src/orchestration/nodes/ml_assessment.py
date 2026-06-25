from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
import src.orchestration.nodes._helpers as _h

_INFERENCE_URL: str = os.getenv("MCP_INFERENCE_URL", "http://localhost:8010/sse")


async def ml_assessment(state: AuditState) -> dict[str, Any]:
    """
    Call sentinel-inference to get a vulnerability assessment for the contract.

    RECALL — what this node does:
        POSTs the contract source to Module 1 via MCP.
        Module 1 runs the full dual-path model:
            raw Solidity → Slither AST → GNNEncoder(8-dim features)
                        → CodeBERT tokens → TransformerEncoder
                        → CrossAttentionFusion → per-class sigmoid → thresholds
        Returns (Track 3): label, vulnerabilities, threshold,
                           truncated, num_nodes, num_edges.
        NOTE: NO "confidence" field — removed in Track 3. Use
              max(v["probability"]) across vulnerabilities instead.

    State updates:
        ml_result → full predict response dict
        error     → set on MCP failure (graph still continues to synthesizer)
    """
    logger.info("ml_assessment | contract_address={}", state.get("contract_address", "unknown"))

    try:
        result = await _h._call_mcp_tool(
            server_url=_INFERENCE_URL,
            tool_name="predict",
            arguments={"contract_code": state["contract_code"]},
        )

        # Guard: tool might return an error dict instead of a prediction.
        # This happens if Module 1 is running but returns HTTP 4xx/5xx.
        if "error" in result:
            logger.warning("ml_assessment | inference error: {}", result["error"])
            return {
                "ml_result": {},
                "error": f"ml_assessment: {result.get('error')} — {result.get('detail', '')}",
            }

        # Log top class from confirmed tier, then suspicious, then legacy field.
        confirmed_list  = result.get("confirmed",  [])
        suspicious_list = result.get("suspicious", [])
        top_tier = confirmed_list or suspicious_list or result.get("vulnerabilities", [])
        if top_tier:
            top = max(top_tier, key=lambda v: v.get("probability", 0.0))
            logger.info(
                "ml_assessment complete | label={} | top_vuln={} ({}) | prob={:.3f} | nodes={}",
                result.get("label"),
                top.get("vulnerability_class"),
                top.get("tier", "CONFIRMED"),
                top.get("probability", 0.0),
                result.get("num_nodes"),
            )
        else:
            logger.info(
                "ml_assessment complete | label={} | no vulnerabilities detected | nodes={}",
                result.get("label"),
                result.get("num_nodes"),
            )

        return {"ml_result": result}

    except Exception as exc:
        # Don't abort the graph — synthesizer will note the missing ml_result.
        logger.error("ml_assessment failed: {}", exc)
        return {
            "ml_result": {},
            "error": f"ml_assessment: {exc}",
        }
