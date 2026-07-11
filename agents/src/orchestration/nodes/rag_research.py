"""
rag_research node — Search the RAG knowledge base for exploit patterns.

P7 (2026-06-26): Fixed zero-match issue.
  - Skip RAG when no classes flagged (topic="unknown")
  - Map ML class names to RAG-friendly keywords
  - Remove Solidity code from query (text embedder can't handle code)
  - Add fallback query if first query returns 0 results
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

_VULN_CLASS_TO_RAG_KEYWORDS: dict[str, str] = {
    "Reentrancy":              "reentrancy reentrant call",
    "IntegerUO":               "integer overflow underflow arithmetic",
    "GasException":            "gas limit denial of service",
    "Timestamp":               "timestamp manipulation time dependence",
    "TransactionOrderDependence": "transaction ordering front running MEV",
    "ExternalBug":             "external call oracle price manipulation",
    "CallToUnknown":           "unknown external call untrusted contract",
    "MishandledException":     "exception handling error propagation",
    "UnusedReturn":            "unchecked return value ignored",
    "DenialOfService":         "denial of service DoS griefing",
}


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

    P5 (2026-06-26): Skip RAG when SENTINEL_DETERMINISTIC=1.
        RAG uses embedding models which are non-deterministic. In deterministic
        mode, we skip RAG to ensure reproducible verdicts.

    P7 (2026-06-26): Fixed zero-match issue.
        - Skip RAG entirely when no classes are flagged (topic="unknown")
        - Map ML class names to RAG-friendly keywords
        - Remove Solidity code from query (confuses text embedder)
        - Add fallback query if first query returns 0 results

    State updates:
        rag_results → list of ranked exploit chunk dicts
        error       → set on failure (does not replace existing error)
    """
    if os.getenv("SENTINEL_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
        logger.info("rag_research | skipped (SENTINEL_DETERMINISTIC mode)")
        return {"rag_results": []}

    ml_result  = state.get("ml_result", {})
    confirmed  = ml_result.get("confirmed",  [])
    suspicious = ml_result.get("suspicious", [])
    flagged = confirmed or suspicious or ml_result.get("vulnerabilities", [])

    if not flagged:
        logger.info("rag_research | skipped (no flagged classes — nothing to search for)")
        return {"rag_results": []}

    top_class = max(flagged, key=lambda v: v.get("probability", 0.0))
    topic = top_class.get("vulnerability_class", ml_result.get("label", "unknown"))

    if topic == "unknown":
        logger.info("rag_research | skipped (topic unknown — nothing to search for)")
        return {"rag_results": []}

    rag_keywords = _VULN_CLASS_TO_RAG_KEYWORDS.get(topic, topic)

    ext_calls = state.get("external_call_summary", [])
    if topic == "ExternalBug" and ext_calls:
        call_str = "; ".join(
            f"{c['caller_function']}→{c['callee_contract']}.{c['callee_function']}"
            for c in ext_calls[:6]
        )
        query = (
            f"smart contract external dependency oracle manipulation "
            f"price feed vulnerability: {call_str}"
        )
    else:
        query = (
            f"smart contract {rag_keywords} "
            f"vulnerability exploit attack pattern"
        )
    logger.info("rag_research | query='{}…'", query[:80])

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

        if not chunks:
            logger.info("rag_research | zero results — trying fallback query")
            fallback_query = f"{topic} vulnerability exploit"
            fallback_result = await _h._call_mcp_tool(
                server_url=_RAG_URL,
                tool_name="search",
                arguments={"query": fallback_query, "k": _RAG_K},
            )
            if "error" not in fallback_result:
                chunks = fallback_result if isinstance(fallback_result, list) else fallback_result.get("results", [])
                logger.info("rag_research fallback | {} chunks retrieved", len(chunks))

        return {"rag_results": chunks}

    except Exception as exc:
        logger.error("rag_research failed: {}", exc)
        return {
            "rag_results": [],
            "error": f"rag_research: {exc}",
        }
