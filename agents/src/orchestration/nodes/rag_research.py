"""
rag_research node — Search the RAG knowledge base for exploit patterns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ── sys.path — make agents/ importable regardless of cwd ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from loguru import logger

from src.orchestration.state import AuditState
import src.orchestration.nodes._helpers as _h


_RAG_URL: str = os.getenv("MCP_RAG_URL", "http://localhost:8011/sse")
_RAG_K: int = int(os.getenv("AUDIT_RAG_K", "5"))


async def rag_research(state: AuditState) -> dict[str, Any]:
    """
    Search the RAG knowledge base for exploit patterns matching this contract.

    RECALL — what this node does:
        Builds a query from the ML result + a snippet of the contract code.
        Calls sentinel-rag:search to retrieve the most relevant DeFiHackLabs
        exploit write-ups.
        The synthesizer uses these chunks to enrich the audit report with
        real-world precedent for the detected vulnerability pattern.

    Query construction (Track 3):
        Uses the top vulnerability class name (highest probability) as the
        primary topic anchor. Falls back to label if vulnerabilities empty.
        The code snippet (first 200 chars) gives lexical context.

    State updates:
        rag_results → list of ranked exploit chunk dicts
        error       → set on failure (does not replace existing error)
    """
    ml_result  = state.get("ml_result", {})
    confirmed  = ml_result.get("confirmed",  [])
    suspicious = ml_result.get("suspicious", [])
    # Prefer highest-probability confirmed class, then suspicious, then legacy field.
    flagged = confirmed or suspicious or ml_result.get("vulnerabilities", [])

    if flagged:
        top_class = max(flagged, key=lambda v: v.get("probability", 0.0))
        topic = top_class.get("vulnerability_class", ml_result.get("label", "unknown"))
    else:
        topic = ml_result.get("label", "unknown")

    contract_snippet = state.get("contract_code", "")[:200].strip()

    # ExternalBug enrichment: include callee contracts/functions so the RAG
    # query targets the specific oracle/price-feed pattern rather than generic
    # "ExternalBug" (which retrieves low-signal chunks).
    ext_calls = state.get("external_call_summary", [])
    if topic == "ExternalBug" and ext_calls:
        call_str = "; ".join(
            f"{c['caller_function']}→{c['callee_contract']}.{c['callee_function']}"
            for c in ext_calls[:6]
        )
        query = (
            f"smart contract ExternalBug oracle manipulation price manipulation "
            f"external dependency vulnerability: {call_str}"
        )
    else:
        query = (
            f"smart contract {topic} vulnerability "
            f"exploit attack pattern: {contract_snippet}"
        )
    logger.info("rag_research | query_prefix='{}…'", query[:80])

    try:
        result = await _h._call_mcp_tool(
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
