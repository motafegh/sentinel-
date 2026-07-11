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


def _ml_unavailable_result(error_kind: str, detail: str) -> dict[str, Any]:
    """
    Build the standard "ML unavailable" payload (Rule 5C contract).

    Returns:
        dict with three keys:
            ml_result: {"ran": False, "reason": ..., "detail": ...}
                — distinct from a successful run's full result, AND
                distinct from `{}` (which was the pre-fix silent-skip).
            error:    human-readable string for the report's top-level `error`.
            tool_status: dict for `state["tool_status"]["ml"]` so the
                _merge_tool_status reducer carries the status forward.

    Per Rule 5C: an empty `ml_result` IS a lie — downstream cannot tell
    "ML ran and found nothing" from "ML never ran." The `ran=False`
    marker carries the failure explicitly.
    """
    return {
        "ml_result": {
            "ran":    False,
            "reason": error_kind,
            "detail": detail,
        },
        "error":      f"ml_assessment: {error_kind} — {detail}",
        "tool_status": {"ml": {"ran": False, "reason": error_kind, "detail": detail}},
    }


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

    Rule 5C contract (CLAUDE.md, 2026-06-25):
        On any failure (ML service error, MCP transport error, etc.) the
        node returns `ml_result = {"ran": False, "reason": ..., "detail": ...}`
        — NOT `ml_result = {}` — so downstream can distinguish "ML ran and
        produced no findings" from "ML never ran." The same `tool_status`
        field is also written (mirrors the Aderyn fix in static_analysis /
        quick_screen). The node stays non-fatal at the graph level
        (intentional) — the failure is visible in state + report.

    State updates:
        ml_result  → full predict response dict on success; {ran: False, ...}
                     on failure (Rule 5C)
        error      → human-readable string on failure (top-level report)
        tool_status → {"ml": {"ran": bool, "reason": str, ...}} — Rule 5C
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
        # Rule 5C: do NOT return ml_result={} on error — return the explicit
        # unavailable payload so the eval layer can distinguish.
        if "error" in result:
            err_kind = str(result.get("error", "unknown_error"))
            err_detail = str(result.get("detail", ""))
            logger.warning("ml_assessment | inference error: {} — {}", err_kind, err_detail)
            return _ml_unavailable_result(err_kind, err_detail)

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

        return {
            "ml_result":   result,
            "model_hash":  result.get("model_hash", ""),
            "tool_status": {"ml": {"ran": True, "label": result.get("label", "?")}},
        }

    except Exception as exc:
        # Don't abort the graph — synthesizer will note the missing ml_result.
        # Rule 5C: surface the failure explicitly via the unavailable payload.
        logger.error("ml_assessment failed: {}", exc)
        return _ml_unavailable_result("exception", str(exc))
