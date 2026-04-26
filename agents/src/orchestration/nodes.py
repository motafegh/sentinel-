# agents/src/orchestration/nodes.py
"""
LangGraph node functions for the SENTINEL audit graph.

Each function is an async LangGraph node:
    - Receives the full AuditState dict
    - Returns a PARTIAL dict with only the keys it updated
    - Never raises — errors are captured into state["error"]

RECALL — node contract with LangGraph:
    Nodes are pure functions from state → partial state.
    LangGraph calls them with the current snapshot and merges
    the returned dict back. Nodes must NOT mutate the input dict.

RECALL — MCP client pattern:
    Each node that needs an MCP tool opens a short-lived SSE connection,
    calls exactly one tool, and closes the connection.
    _call_mcp_tool() is the shared helper for this.
    Connection-per-call is slightly slower than pooling but simpler
    for M5. In M6 the connection can be promoted to a module-level
    persistent client if latency becomes an issue.

Node execution order:
    ml_assessment
        ├─ [confidence > 0.70] → rag_research → audit_check → synthesizer
        └─ [confidence ≤ 0.70] → synthesizer  (fast path)

Nodes (M5):
    ml_assessment   — calls sentinel-inference: predict
    rag_research    — calls sentinel-rag: search
    audit_check     — calls sentinel-audit: get_audit_history
    synthesizer     — assembles final_report from available state

Nodes added in M6:
    static_analysis — Slither + Mythril (no MCP, direct call)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

# ── sys.path is already set by graph.py before nodes is imported ──────────
from src.orchestration.state import AuditState

# ---------------------------------------------------------------------------
# MCP server URLs — overridable via agents/.env
# ---------------------------------------------------------------------------
# The /sse suffix is the SSE endpoint path, NOT the base URL.
# Pattern: http://host:port/sse

_INFERENCE_URL: str = os.getenv("MCP_INFERENCE_URL", "http://localhost:8010/sse")
_RAG_URL:       str = os.getenv("MCP_RAG_URL",       "http://localhost:8011/sse")
_AUDIT_URL:     str = os.getenv("MCP_AUDIT_URL",     "http://localhost:8012/sse")

# Default number of RAG chunks to retrieve for deep-path analysis.
_RAG_K: int = int(os.getenv("AUDIT_RAG_K", "5"))

# ---------------------------------------------------------------------------
# Risk routing helper
# ---------------------------------------------------------------------------

def _is_high_risk(ml_result: dict[str, Any]) -> bool:
    """
    Return True if the ML result warrants deep analysis.

    BINARY PHASE (current):
        Checks ml_result["confidence"] > 0.70.
        Threshold 0.70 deliberately higher than inference threshold (0.50):
        we want deep analysis for high-confidence vulnerable contracts,
        not every contract that crosses the binary boundary.

    MULTI-LABEL PHASE (Track 3 swap):
        Replace the confidence check with:
            max(v["probability"] for v in ml_result["vulnerabilities"]) > 0.70
        This function is the single place to update — no other code changes.

    Args:
        ml_result: dict from ml_assessment node (label, confidence, …)

    Returns:
        True  → deep path: rag_research → audit_check → synthesizer
        False → fast path: synthesizer only
    """
    return ml_result.get("confidence", 0.0) > 0.70


# ---------------------------------------------------------------------------
# Shared MCP call helper
# ---------------------------------------------------------------------------

async def _call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Open a short-lived SSE connection, call one MCP tool, return parsed JSON.

    RECALL — why a fresh connection per call:
        MCP SSE connections are stateless between calls — the server holds
        no session state after the tool call completes. A fresh connection
        costs one TCP handshake (~1ms on localhost) but avoids stale-connection
        bugs during long-running graphs. In M6, promote to persistent client
        if RTT measurements show this is a bottleneck.

    Args:
        server_url: Full SSE URL, e.g. "http://localhost:8010/sse"
        tool_name:  MCP tool name exactly as declared in list_tools()
        arguments:  Tool arguments dict (must match the tool's inputSchema)

    Returns:
        Parsed dict from the tool's TextContent response.

    Raises:
        RuntimeError: if the tool response cannot be parsed as JSON.
        Any exception from the MCP/network layer propagates to the caller.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            raw = result.content[0].text
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"MCP tool '{tool_name}' returned non-JSON response: {raw[:200]}"
                ) from exc


# ---------------------------------------------------------------------------
# Node: ml_assessment
# ---------------------------------------------------------------------------

async def ml_assessment(state: AuditState) -> dict[str, Any]:
    """
    Call sentinel-inference to get a vulnerability score for the contract.

    RECALL — what this node does:
        POSTs the contract source to Module 1 via MCP.
        Module 1 runs the full dual-path model:
            raw Solidity → Slither AST → GNNEncoder(8-dim features)
                        → CodeBERT tokens → TransformerEncoder
                        → FusionLayer → Sigmoid → score [0,1]
        Returns: label, confidence, threshold, truncated, num_nodes, num_edges.

    State updates:
        ml_result → full predict response dict
        error     → set on MCP failure (graph still continues to synthesizer)
    """
    logger.info("ml_assessment | contract_address={}", state.get("contract_address", "unknown"))

    try:
        result = await _call_mcp_tool(
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

        logger.info(
            "ml_assessment complete | label={} | confidence={:.3f} | nodes={}",
            result.get("label"),
            result.get("confidence", 0.0),
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


# ---------------------------------------------------------------------------
# Node: rag_research
# ---------------------------------------------------------------------------

async def rag_research(state: AuditState) -> dict[str, Any]:
    """
    Search the RAG knowledge base for exploit patterns matching this contract.

    RECALL — what this node does:
        Builds a query from the ML result + a snippet of the contract code.
        Calls sentinel-rag:search to retrieve the most relevant DeFiHackLabs
        exploit write-ups.
        The synthesizer uses these chunks to enrich the audit report with
        real-world precedent for the detected vulnerability pattern.

    Query construction:
        The query combines the vulnerability label with a leading code snippet.
        The snippet (first 200 chars) gives the RAG model lexical context —
        it helps distinguish "reentrancy in a vault" from "reentrancy in a DEX".

    State updates:
        rag_results → list of ranked exploit chunk dicts
        error       → set on failure (does not replace existing error)
    """
    ml_result = state.get("ml_result", {})
    label     = ml_result.get("label", "unknown")
    contract_snippet = state.get("contract_code", "")[:200].strip()

    # Build a natural-language query that the embedding model can match against
    # DeFiHackLabs exploit write-ups. "vulnerability" and the label give the
    # topic; the code snippet anchors it to the specific pattern.
    query = (
        f"smart contract vulnerability {label} "
        f"exploit attack pattern: {contract_snippet}"
    )
    logger.info("rag_research | query_prefix='{}…'", query[:80])

    try:
        result = await _call_mcp_tool(
            server_url=_RAG_URL,
            tool_name="search",
            arguments={"query": query, "k": _RAG_K},
        )

        if "error" in result:
            logger.warning("rag_research | RAG error: {}", result["error"])
            return {
                "rag_results": [],
                "error": f"rag_research: {result.get('error')}",
            }

        chunks = result if isinstance(result, list) else result.get("results", [])
        logger.info("rag_research complete | {} chunks retrieved", len(chunks))
        return {"rag_results": chunks}

    except Exception as exc:
        logger.error("rag_research failed: {}", exc)
        return {
            "rag_results": [],
            "error": f"rag_research: {exc}",
        }


# ---------------------------------------------------------------------------
# Node: audit_check
# ---------------------------------------------------------------------------

async def audit_check(state: AuditState) -> dict[str, Any]:
    """
    Query AuditRegistry for prior on-chain audit records for this contract.

    RECALL — what this node does:
        Calls sentinel-audit:get_audit_history for the contract_address.
        Returns all prior audits in reverse-chronological order.
        The synthesizer uses this to answer: "Has this contract been audited
        before? Did prior audits flag it? Has the risk score improved?"

    Skipped gracefully if contract_address is empty or not a valid address.
    Sets audit_history=[] in that case — synthesizer handles missing data.

    State updates:
        audit_history → list of AuditResult dicts (may be empty list)
        error         → set on failure (appends, does not replace existing)
    """
    contract_address = state.get("contract_address", "").strip()

    if not contract_address:
        logger.info("audit_check | no contract_address — skipping on-chain lookup")
        return {"audit_history": []}

    logger.info("audit_check | address={}", contract_address)

    try:
        result = await _call_mcp_tool(
            server_url=_AUDIT_URL,
            tool_name="get_audit_history",
            arguments={"contract_address": contract_address, "limit": 10},
        )

        if "error" in result:
            logger.warning("audit_check | registry error: {}", result["error"])
            return {
                "audit_history": [],
                "error": f"audit_check: {result.get('error')}",
            }

        records = result.get("records", [])
        logger.info(
            "audit_check complete | {} prior audit(s) found", len(records)
        )
        return {"audit_history": records}

    except Exception as exc:
        logger.error("audit_check failed: {}", exc)
        return {
            "audit_history": [],
            "error": f"audit_check: {exc}",
        }


# ---------------------------------------------------------------------------
# Node: synthesizer
# ---------------------------------------------------------------------------

async def synthesizer(state: AuditState) -> dict[str, Any]:
    """
    Assemble the final audit report from all available node outputs.

    RECALL — what this node does (M5 vs M6):
        M5 (current):  pure data assembly — no LLM call.
            Structured JSON report from available state fields.
            The "recommendation" field is rule-based for M5.

        M6 (next):     call the LLM synthesizer agent.
            The synthesizer agent uses STRONG model (qwen3.5-9b-ud) to
            produce a natural-language audit summary + recommendation.
            The M5 structured report becomes the LLM's context.

    Report schema (binary phase):
        contract_address:  str
        overall_label:     str          "vulnerable" | "safe" | "unknown"
        confidence:        float        ML score in [0,1]
        threshold:         float        decision boundary used
        ml_truncated:      bool         True if contract exceeded 512 tokens
        num_nodes:         int          AST node count
        num_edges:         int          AST edge count
        rag_evidence:      list         matched exploit chunks (deep path)
        audit_history:     list         prior on-chain audit records
        static_findings:   dict | None  reserved for M6
        recommendation:    str          rule-based in M5, LLM-generated in M6
        error:             str | None   any non-fatal error during the run
        path_taken:        str          "deep" or "fast" — for observability

    NOTE — After Track 3 (multi-label):
        overall_label → most likely class name (or "safe" if all < 0.5)
        confidence    → max(probability across 11 vulnerability classes)
        Add:  vulnerabilities: list[{class, probability}]

    State updates:
        final_report → complete report dict
    """
    ml_result    : dict = state.get("ml_result",    {})
    rag_results  : list = state.get("rag_results",  [])
    audit_history: list = state.get("audit_history", [])
    error        : str | None = state.get("error")

    label      = ml_result.get("label",      "unknown")
    confidence = ml_result.get("confidence", 0.0)
    threshold  = ml_result.get("threshold",  0.50)
    truncated  = ml_result.get("truncated",  False)
    num_nodes  = ml_result.get("num_nodes",  0)
    num_edges  = ml_result.get("num_edges",  0)

    # Determine which path was taken — useful for debugging and observability.
    # If rag_results is populated, we went deep. If empty, fast path.
    path_taken = "deep" if rag_results else "fast"

    # ── Rule-based recommendation (M5) ──────────────────────────────────────
    # M6 will replace this with an LLM-generated narrative.
    # Keep this block clearly separated so it's easy to swap out.
    if not ml_result:
        recommendation = (
            "ML assessment failed — manual review required. "
            "Check that the inference server (port 8001) is running."
        )
    elif label == "vulnerable" and confidence >= 0.70:
        rag_count = len(rag_results)
        prior_count = len(audit_history)
        recommendation = (
            f"HIGH RISK — ML confidence {confidence:.1%}. "
            f"{rag_count} similar exploit pattern(s) found in DeFiHackLabs corpus. "
            f"{prior_count} prior on-chain audit(s). "
            "Recommend full manual audit before deployment."
        )
    elif label == "vulnerable":
        recommendation = (
            f"MODERATE RISK — ML confidence {confidence:.1%} (above 0.50 threshold). "
            "Contract crossed the vulnerability boundary but below high-confidence threshold. "
            "Recommend targeted review of flagged patterns."
        )
    else:
        recommendation = (
            f"LOW RISK — ML confidence {confidence:.1%} below vulnerability threshold ({threshold:.0%}). "
            "Standard due diligence recommended."
        )

    if truncated:
        recommendation += (
            " NOTE: Contract exceeded 512 CodeBERT tokens — tail code was not analysed. "
            "For large contracts, manual review of the unanalysed portion is recommended."
        )

    report = {
        "contract_address": state.get("contract_address", ""),
        "overall_label":    label,
        "confidence":       round(confidence, 4),
        "threshold":        threshold,
        "ml_truncated":     truncated,
        "num_nodes":        num_nodes,
        "num_edges":        num_edges,
        "rag_evidence":     rag_results,
        "audit_history":    audit_history,
        "static_findings":  state.get("static_findings"),  # None until M6
        "recommendation":   recommendation,
        "error":            error,
        "path_taken":       path_taken,
    }

    logger.info(
        "synthesizer complete | label={} | confidence={:.3f} | path={} | "
        "rag_chunks={} | prior_audits={}",
        label,
        confidence,
        path_taken,
        len(rag_results),
        len(audit_history),
    )

    return {"final_report": report}
