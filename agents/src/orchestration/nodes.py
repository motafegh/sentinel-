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

Node execution order (Track 3 / multi-label):
    ml_assessment
        ├─ [max(probability) > 0.70] → rag_research → audit_check → synthesizer
        └─ [max(probability) ≤ 0.70] → synthesizer  (fast path)

Nodes (M5):
    ml_assessment   — calls sentinel-inference: predict
    rag_research    — calls sentinel-rag: search
    audit_check     — calls sentinel-audit: get_audit_history
    synthesizer     — assembles final_report from available state

Nodes added in M6:
    static_analysis — Slither + Mythril (no MCP, direct call)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

# ── sys.path is already set by graph.py before nodes is imported ──────────
from src.orchestration.state import AuditState

# BRIDGE (Issue #1): import REPORTS_DIR so synthesizer can persist the
# final_report for feedback_loop.py to read back via contract_address.
from src.ingestion.pipeline import REPORTS_DIR

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

    Track 3 (multi-label, current):
        Uses max(v["probability"]) across all detected vulnerabilities.
        Threshold 0.70 is deliberately higher than the per-class inference
        threshold (0.50): we want deep analysis only for high-confidence
        detections, not every contract that crosses any class boundary.
        Safe contracts have an empty vulnerabilities list → max() returns
        0.0 → fast path. Correct behaviour.

    Args:
        ml_result: dict from ml_assessment node.
                   Track 3 schema: label, vulnerabilities, threshold,
                   truncated, num_nodes, num_edges.
                   NO "confidence" field — removed in Track 3.

    Returns:
        True  → deep path: rag_research → audit_check → synthesizer
        False → fast path: synthesizer only
    """
    vulns = ml_result.get("vulnerabilities", [])
    if not vulns:
        return False
    return max(v.get("probability", 0.0) for v in vulns) > 0.70


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

        # Bug 8 fix: "confidence" no longer exists in Track 3 schema.
        # Log top vulnerability class and its probability instead.
        vulns = result.get("vulnerabilities", [])
        if vulns:
            top = max(vulns, key=lambda v: v.get("probability", 0.0))
            logger.info(
                "ml_assessment complete | label={} | top_vuln={} | prob={:.3f} | nodes={}",
                result.get("label"),
                top.get("vulnerability_class"),
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

    Query construction (Track 3):
        Uses the top vulnerability class name (highest probability) as the
        primary topic anchor. Falls back to label if vulnerabilities empty.
        The code snippet (first 200 chars) gives lexical context.

    State updates:
        rag_results → list of ranked exploit chunk dicts
        error       → set on failure (does not replace existing error)
    """
    ml_result = state.get("ml_result", {})
    vulns     = ml_result.get("vulnerabilities", [])

    # Use the top detected vulnerability class as the query topic.
    # More precise than the binary label — "Reentrancy exploit" retrieves
    # better RAG results than "vulnerable exploit".
    if vulns:
        top_class = max(vulns, key=lambda v: v.get("probability", 0.0))
        topic = top_class.get("vulnerability_class", ml_result.get("label", "unknown"))
    else:
        topic = ml_result.get("label", "unknown")

    contract_snippet = state.get("contract_code", "")[:200].strip()

    query = (
        f"smart contract {topic} vulnerability "
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
# Node: static_analysis
# ---------------------------------------------------------------------------

async def static_analysis(state: AuditState) -> dict[str, Any]:
    """
    Run Slither directly on the contract source and return per-finding dicts.

    RECALL — why direct Slither call, not MCP:
        Slither is a Python library installed in this process.
        Spawning it via MCP would add latency for no benefit.
        The result is merged into state alongside rag_research output — both
        run in parallel in the deep path (LangGraph fan-out semantics).

    RECALL — what this node produces:
        Each finding is:
            {
                "tool":        "slither",
                "detector":    str   — detector name (e.g. "reentrancy-eth"),
                "impact":      str   — "High" | "Medium" | "Low" | "Informational",
                "confidence":  str   — "High" | "Medium" | "Low",
                "description": str   — human-readable finding description,
                "lines":       list  — source line numbers affected (may be []),
            }

    RECALL — why Slither in detectors_to_run default (all detectors):
        For calibration we want everything — the synthesizer filters by impact.
        The ML model already caught what it caught; Slither catches rule-based
        patterns the GNN/BERT may have missed (e.g. tx.origin misuse).

    State updates:
        static_findings → list of finding dicts (may be empty)
        error           → set on failure (non-fatal; node returns empty list)
    """
    contract_code = state.get("contract_code", "")
    if not contract_code or not contract_code.strip():
        logger.warning("static_analysis | contract_code empty — skipping")
        return {"static_findings": []}

    logger.info(
        "static_analysis | running Slither | contract_address={}",
        state.get("contract_address", "unknown"),
    )

    tmp_path: str | None = None
    try:
        from slither import Slither
        from slither.core.declarations import Function

        # Slither requires a real file path — write to temp file.
        with tempfile.NamedTemporaryFile(
            suffix=".sol",
            prefix="sentinel_static_",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(contract_code)
            tmp_path = tmp.name

        # Run all detectors (default). detectors_to_run=[] disables them;
        # omitting the argument runs all available detectors.
        sl = Slither(tmp_path)

        findings: list[dict] = []
        for result in sl.run_detectors():
            for finding in result:
                # Normalise: extract line numbers from source mappings when present.
                elements = finding.get("elements", [])
                lines: list[int] = []
                for elem in elements:
                    src = elem.get("source_mapping", {})
                    elem_lines = src.get("lines", [])
                    if isinstance(elem_lines, list):
                        lines.extend(int(ln) for ln in elem_lines if isinstance(ln, int))

                findings.append({
                    "tool":        "slither",
                    "detector":    finding.get("check", "unknown"),
                    "impact":      finding.get("impact", "Unknown"),
                    "confidence":  finding.get("confidence", "Unknown"),
                    "description": finding.get("description", ""),
                    "lines":       sorted(set(lines)),
                })

        logger.info(
            "static_analysis complete | {} finding(s) | contract_address={}",
            len(findings),
            state.get("contract_address", "unknown"),
        )
        return {"static_findings": findings}

    except ImportError:
        logger.warning("static_analysis | slither not installed — skipping")
        return {"static_findings": []}

    except Exception as exc:
        logger.error("static_analysis failed: {}", exc)
        return {
            "static_findings": [],
            "error": f"static_analysis: {exc}",
        }

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("static_analysis | failed to delete temp file {}: {}", tmp_path, e)


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

    Report schema (Track 3 / multi-label) — matches SENTINEL-SPEC §8.1:
        contract_address:   str
        overall_label:      str          "vulnerable" | "safe" | "unknown"
        risk_probability:   float        max(probability) across detected vulns
        top_vulnerability:  str | None   class name with highest probability
        vulnerabilities:    list         [{vulnerability_class, probability}, ...]
        threshold:          float        per-class decision boundary used
        ml_truncated:       bool         True if contract exceeded 512 tokens
        num_nodes:          int          AST node count
        num_edges:          int          AST edge count
        rag_evidence:       list         matched exploit chunks (deep path)
        audit_history:      list         prior on-chain audit records
        static_findings:    dict | None  reserved for M6
        recommendation:     str          rule-based in M5, LLM-generated in M6
        error:              str | None   any non-fatal error during the run
        path_taken:         str          "deep" or "fast" — for observability

    BRIDGE (Issue #1):
        If contract_address is known, the final_report is persisted to
        data/reports/{contract_address}.json BEFORE this node returns.
        feedback_loop.py reads this file by contract_address when it
        processes the on-chain AuditSubmitted event, and uses
        report["top_vulnerability"] as the vuln_type metadata field.
        This replaces the hardcoded vuln_type="unknown" that made all
        on-chain RAG findings invisible to filtered searches.

    State updates:
        final_report → complete report dict
    """
    ml_result      : dict      = state.get("ml_result",      {})
    rag_results    : list      = state.get("rag_results",    [])
    audit_history  : list      = state.get("audit_history",  [])
    static_findings: list      = state.get("static_findings", [])
    error          : str | None = state.get("error")

    label     = ml_result.get("label",     "unknown")
    vulns     = ml_result.get("vulnerabilities", [])
    threshold = ml_result.get("threshold", 0.50)
    truncated = ml_result.get("truncated", False)
    num_nodes = ml_result.get("num_nodes", 0)
    num_edges = ml_result.get("num_edges", 0)

    # Derive risk_probability and top_vulnerability from the vulnerabilities list.
    # These replace the binary-era "confidence" field (removed in Track 3).
    if vulns:
        top_vuln      = max(vulns, key=lambda v: v.get("probability", 0.0))
        risk_prob     = round(top_vuln.get("probability", 0.0), 4)
        top_vuln_name = top_vuln.get("vulnerability_class")
    else:
        risk_prob     = 0.0
        top_vuln_name = None

    # Determine which path was taken — useful for debugging and observability.
    # If rag_results or static_findings is populated, we went deep.
    path_taken = "deep" if (rag_results or static_findings) else "fast"

    # ── Rule-based recommendation (fallback) ─────────────────────────────────
    # Used when the LLM is unavailable or times out.
    if not ml_result:
        recommendation = (
            "ML assessment failed — manual review required. "
            "Check that the inference server (port 8001) is running."
        )
    elif label == "vulnerable" and risk_prob >= 0.70:
        rag_count    = len(rag_results)
        prior_count  = len(audit_history)
        slither_high = sum(
            1 for f in static_findings if f.get("impact") in ("High", "Medium")
        )
        recommendation = (
            f"HIGH RISK — top vulnerability: {top_vuln_name} "
            f"(probability {risk_prob:.1%}). "
            f"{rag_count} similar exploit pattern(s) found in DeFiHackLabs corpus. "
            f"{slither_high} Slither High/Medium finding(s). "
            f"{prior_count} prior on-chain audit(s). "
            "Recommend full manual audit before deployment."
        )
    elif label == "vulnerable":
        recommendation = (
            f"MODERATE RISK — top vulnerability: {top_vuln_name} "
            f"(probability {risk_prob:.1%}, above per-class threshold). "
            "Contract crossed the vulnerability boundary but below high-confidence threshold. "
            "Recommend targeted review of flagged patterns."
        )
    else:
        recommendation = (
            f"LOW RISK — no vulnerability exceeded per-class threshold "
            f"(max probability: {risk_prob:.1%}, threshold: {threshold:.0%}). "
            "Standard due diligence recommended."
        )

    # ── LLM narrative (T3-A / Move 5) ────────────────────────────────────────
    # Attempt a structured Markdown security narrative from the strong LLM.
    # Falls back silently to the rule-based recommendation above on any failure
    # (LLM unavailable, timeout, malformed response).
    narrative: str | None = None
    if ml_result:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from src.llm.client import get_strong_llm

            vuln_lines = "\n".join(
                f"  - {v.get('vulnerability_class', '?')}: {v.get('probability', 0.0):.1%}"
                for v in vulns
            ) or "  (none detected)"

            rag_lines = "\n".join(
                f"  [{i + 1}] {c.get('metadata', {}).get('protocol', 'unknown')}: "
                f"{c.get('content', '')[:120]}..."
                for i, c in enumerate(rag_results[:3])
            ) if rag_results else "  (no matching exploit evidence retrieved)"

            slither_lines = "\n".join(
                f"  [{f.get('impact', '')}] {f.get('detector', '')}: "
                f"{f.get('description', '')[:100]}"
                for f in static_findings[:5]
                if f.get("impact") in ("High", "Medium")
            ) if static_findings else "  (no High/Medium static analysis findings)"

            code_snippet = state.get("contract_code", "")[:500].strip()

            system_msg = SystemMessage(content=(
                "You are a senior smart contract security auditor. "
                "Produce a concise, structured Markdown security assessment with exactly "
                "these four sections:\n"
                "## Severity\n"
                "ONE of: CRITICAL | HIGH | MEDIUM | LOW | INFORMATIONAL\n"
                "## Vulnerability Summary\n"
                "2–3 sentences describing what was detected and why it is dangerous.\n"
                "## Exploit Pattern\n"
                "How an attacker could exploit this — reference exploit evidence if available.\n"
                "## Recommended Fix\n"
                "Concrete, actionable mitigation steps specific to the detected vulnerability.\n"
                "Be concise. Output only the Markdown, no preamble."
            ))

            user_msg = HumanMessage(content=(
                f"**Contract address:** {state.get('contract_address', 'unknown')}\n"
                f"**ML model assessment:** {label} (threshold: {threshold:.0%})\n\n"
                f"**Detected vulnerabilities:**\n{vuln_lines}\n\n"
                f"**RAG exploit evidence:**\n{rag_lines}\n\n"
                f"**Static analysis findings (High/Medium):**\n{slither_lines}\n\n"
                f"**Contract code snippet (first 500 chars):**\n```solidity\n{code_snippet}\n```"
            ))

            llm = get_strong_llm()
            response = await asyncio.wait_for(
                asyncio.to_thread(llm.invoke, [system_msg, user_msg]),
                timeout=45.0,
            )
            narrative = response.content.strip()
            logger.info("synthesizer | LLM narrative generated ({} chars)", len(narrative))

        except Exception as _llm_exc:
            logger.warning(
                "synthesizer | LLM narrative failed (using rule-based fallback): {}",
                _llm_exc,
            )
            narrative = None

    truncated_note = (
        "\n\n> **NOTE:** Contract exceeded 512 CodeBERT tokens — "
        "tail code was not analysed. Manual review of the unanalysed portion is recommended."
    ) if truncated else ""

    final_recommendation = (narrative or recommendation) + truncated_note

    report = {
        "contract_address":  state.get("contract_address", ""),
        "overall_label":     label,
        "risk_probability":  risk_prob,
        "top_vulnerability": top_vuln_name,
        "vulnerabilities":   vulns,
        "threshold":         threshold,
        "ml_truncated":      truncated,
        "num_nodes":         num_nodes,
        "num_edges":         num_edges,
        "rag_evidence":      rag_results,
        "audit_history":     audit_history,
        "static_findings":   static_findings,
        "recommendation":    final_recommendation,
        "narrative":         narrative,
        "error":             error,
        "path_taken":        path_taken,
    }

    # ── BRIDGE (Issue #1): persist report for feedback_loop.py ──────────────
    # feedback_loop.py has no access to in-memory state — it runs as a
    # separate process listening to on-chain events. Writing the report to
    # disk by contract_address gives it the vulnerability_class it needs to
    # index on-chain findings with a meaningful vuln_type instead of "unknown".
    #
    # Only write if contract_address is known (it may be empty in test runs).
    # Failures are logged but never raise — the report is still returned
    # to the caller; a missing file only degrades RAG quality, not correctness.
    contract_address = state.get("contract_address", "").strip()
    if contract_address:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = REPORTS_DIR / f"{contract_address}.json"
            report_path.write_text(json.dumps(report, indent=2))
            logger.debug("synthesizer | report persisted → {}", report_path)
        except Exception as exc:
            logger.warning(
                "synthesizer | could not persist report for bridge (non-fatal): {}", exc
            )

    logger.info(
        "synthesizer complete | label={} | risk_prob={:.3f} | top_vuln={} | "
        "path={} | rag_chunks={} | prior_audits={}",
        label,
        risk_prob,
        top_vuln_name,
        path_taken,
        len(rag_results),
        len(audit_history),
    )

    return {"final_report": report}
