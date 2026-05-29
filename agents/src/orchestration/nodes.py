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

Graph topology (Phase 1):
    ml_assessment
        ↓
    evidence_router          ← logs routing_decisions to state
        ├─ (deep path) → rag_research ──────┐
        │               static_analysis ─────┤→ audit_check → cross_validator → synthesizer
        │               graph_explain ───────┘
        └─ (fast path) ──────────────────────────────────────────────────→ synthesizer

Nodes:
    ml_assessment   — calls sentinel-inference: predict
    evidence_router — computes per-class routing; logs to routing_decisions
    rag_research    — calls sentinel-rag: search (deep path only)
    static_analysis — Slither direct call, scoped to flagged classes (deep path)
    graph_explain   — calls sentinel-graph-inspector: function-level hotspots (deep path)
    audit_check     — calls sentinel-audit: get_audit_history (deep path only)
    cross_validator — LLM-adjudicated per-class verdicts (deep path only)
    synthesizer     — assembles final_report; uses cross_validator verdicts or rule-based fallback
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
from src.orchestration.routing import (
    CLASS_TO_DETECTORS,
    build_routing_decisions,
    compute_active_tools,
    compute_overall_verdict,
    compute_verdict,
    prob_to_severity,
)

# BRIDGE (Issue #1): import REPORTS_DIR so synthesizer can persist the
# final_report for feedback_loop.py to read back via contract_address.
from src.ingestion.pipeline import REPORTS_DIR

# ---------------------------------------------------------------------------
# MCP server URLs — overridable via agents/.env
# ---------------------------------------------------------------------------
# The /sse suffix is the SSE endpoint path, NOT the base URL.
# Pattern: http://host:port/sse

_INFERENCE_URL:       str = os.getenv("MCP_INFERENCE_URL",       "http://localhost:8010/sse")
_RAG_URL:             str = os.getenv("MCP_RAG_URL",             "http://localhost:8011/sse")
_AUDIT_URL:           str = os.getenv("MCP_AUDIT_URL",           "http://localhost:8012/sse")
_GRAPH_INSPECTOR_URL: str = os.getenv("MCP_GRAPH_INSPECTOR_URL", "http://localhost:8013/sse")

# Default number of RAG chunks to retrieve for deep-path analysis.
_RAG_K: int = int(os.getenv("AUDIT_RAG_K", "5"))


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
# Node: quick_screen  (Tier 0 — runs on EVERY contract, before routing)
# ---------------------------------------------------------------------------

# High/Critical-impact Slither detectors worth escalating on even if ML is safe.
# Informational/Low detectors are intentionally excluded to limit false escalations.
_SCREEN_SLITHER_DETECTORS: frozenset[str] = frozenset({
    "reentrancy-eth",
    "reentrancy-no-eth",
    "arbitrary-send-eth",
    "controlled-delegatecall",
    "delegatecall-loop",
    "integer-overflow",
    "msg-value-loop",
    "unchecked-send",
    "unchecked-transfer",
    "calls-loop",
    "tx-origin",
    "suicidal",
    "uninitialized-local",
    "uninitialized-state",
    "write-after-write",
})

# Aderyn rule IDs that indicate High/Critical findings worth escalating.
# Only a subset — informational rules are ignored.
_SCREEN_ADERYN_HIGH_IDS: frozenset[str] = frozenset({
    "H-1", "H-2", "H-3", "H-4", "H-5",
    "C-1", "C-2", "C-3",
})


async def quick_screen(state: AuditState) -> dict[str, Any]:
    """
    Tier 0 screen — runs Slither + Aderyn on every contract before routing.

    PURPOSE: Closes the ML blind spot where all class probabilities fall below
    DEEP_THRESHOLDS. A contract scoring "safe" on ML is still escalated to deep
    path if either static tool fires a High/Critical finding.

    Two independent signals are required to agree before fast-path is allowed:
        Signal 1: ML — all class probabilities below DEEP_THRESHOLDS
        Signal 2: quick_screen — zero High/Critical findings from Slither+Aderyn

    Slither: subset of High-impact detectors (_SCREEN_SLITHER_DETECTORS).
    Aderyn:  subprocess call; parses JSON output; looks for H-* / C-* rule IDs.
             Non-fatal if aderyn is not installed — only slither result used.

    State updates:
        quick_screen_hits → {"slither": [detector_name, ...], "aderyn": [rule_id, ...]}
                           Empty lists in each key when nothing fires or tool absent.
    """
    contract_code = state.get("contract_code", "")
    if not contract_code or not contract_code.strip():
        logger.warning("quick_screen | contract_code empty — skipping")
        return {"quick_screen_hits": {"slither": [], "aderyn": []}}

    logger.info(
        "quick_screen | running Tier-0 screen | contract_address={}",
        state.get("contract_address", "unknown"),
    )

    slither_hits: list[str] = []
    aderyn_hits:  list[str] = []
    tmp_path: str | None    = None

    try:
        from slither import Slither

        with tempfile.NamedTemporaryFile(
            suffix=".sol",
            prefix="sentinel_screen_",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(contract_code)
            tmp_path = tmp.name

        sl = Slither(tmp_path)

        # Scope to High-impact detectors only — avoids escalating on noise.
        sl._detectors = [  # type: ignore[attr-defined]
            d for d in sl._detectors  # type: ignore[attr-defined]
            if getattr(d, "ARGUMENT", "") in _SCREEN_SLITHER_DETECTORS
        ]

        for result in sl.run_detectors():
            for finding in result:
                impact = finding.get("impact", "")
                if impact in ("High", "Medium", "Critical"):
                    detector = finding.get("check", "unknown")
                    if detector not in slither_hits:
                        slither_hits.append(detector)

        logger.info(
            "quick_screen | slither done | hits={} | contract_address={}",
            slither_hits,
            state.get("contract_address", "unknown"),
        )

    except ImportError:
        logger.warning("quick_screen | slither not installed — skipping slither screen")

    except Exception as exc:
        logger.warning("quick_screen | slither error (non-fatal): {}", exc)

    # ── Aderyn ────────────────────────────────────────────────────────────────
    if tmp_path:
        try:
            import json as _json
            import subprocess

            aderyn_result = subprocess.run(
                ["aderyn", "--output", "json", tmp_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if aderyn_result.returncode == 0 and aderyn_result.stdout.strip():
                aderyn_data = _json.loads(aderyn_result.stdout)
                # Aderyn JSON: {"high": [...], "low": [...]} or {"findings": [...]}
                high_findings = aderyn_data.get("high", aderyn_data.get("findings", []))
                for finding in high_findings:
                    rule_id = finding.get("id", finding.get("rule_id", ""))
                    if rule_id and rule_id not in aderyn_hits:
                        aderyn_hits.append(rule_id)
            elif aderyn_result.returncode != 0:
                logger.debug(
                    "quick_screen | aderyn exited {} — stderr: {}",
                    aderyn_result.returncode,
                    aderyn_result.stderr[:200],
                )

        except FileNotFoundError:
            logger.debug("quick_screen | aderyn not installed — skipping")

        except Exception as exc:
            logger.warning("quick_screen | aderyn error (non-fatal): {}", exc)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError as e:
            logger.warning("quick_screen | failed to delete temp file {}: {}", tmp_path, e)

    hits = {"slither": slither_hits, "aderyn": aderyn_hits}
    logger.info(
        "quick_screen complete | slither_hits={} | aderyn_hits={} | contract_address={}",
        len(slither_hits),
        len(aderyn_hits),
        state.get("contract_address", "unknown"),
    )
    return {"quick_screen_hits": hits}


# ---------------------------------------------------------------------------
# Node: evidence_router
# ---------------------------------------------------------------------------

async def evidence_router(state: AuditState) -> dict[str, Any]:
    """
    Compute per-class routing and log decisions to AuditState.routing_decisions.

    Reads both ml_result AND quick_screen_hits — if quick_screen found anything,
    the routing note records the escalation so the final report is fully auditable.
    Actual branching logic lives in _route_from_evidence_router (graph.py); this
    node only logs; it never raises.

    State updates:
        routing_decisions → list of human-readable routing decision strings
    """
    ml_result = state.get("ml_result", {})
    decisions = build_routing_decisions(ml_result)

    # Log quick_screen signal alongside ML per-class decisions.
    quick_screen_hits = state.get("quick_screen_hits", {})
    slither_hits = quick_screen_hits.get("slither", [])
    aderyn_hits  = quick_screen_hits.get("aderyn",  [])

    active = compute_active_tools(ml_result)
    if slither_hits or aderyn_hits:
        escalation = (
            f"quick_screen: slither={slither_hits[:3]} aderyn={aderyn_hits[:3]}"
            f" → escalate to deep path (overrides fast-path even if ML safe)"
        )
        decisions.append(escalation)
        logger.info("evidence_router | {}", escalation)
    elif not active:
        decisions.append("quick_screen: no hits, ML safe → fast path confirmed")

    logger.info(
        "evidence_router | active_tools={} | quick_screen_slither={} | quick_screen_aderyn={}",
        active or ["fast-path"],
        len(slither_hits),
        len(aderyn_hits),
    )

    return {"routing_decisions": decisions}


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
# ExternalBug helper
# ---------------------------------------------------------------------------

def _extract_external_call_summary(sl: Any) -> list[dict[str, Any]]:
    """
    Extract inter-contract call graph from a Slither object.

    WHY THIS EXISTS
    ───────────────
    The GNN encodes call_target_typed=1.00 for all typed interface calls,
    making them look structurally "safe" to the model. This is the primary
    reason ExternalBug is hard to detect from graph features alone — an
    oracle price manipulation call through an interface looks identical to
    a safe library call at the graph level.

    This function extracts the actual callee contracts and functions so that:
    - rag_research can build a targeted query: "ExternalBug ChainlinkOracle
      getLatestPrice stale price manipulation"
    - synthesizer can include the call graph in the LLM prompt, enabling
      it to reason about oracle dependency risks explicitly.

    Args:
        sl: Slither instance (already initialised on the contract file)

    Returns:
        List of call dicts: {caller_contract, caller_function,
                             callee_contract, callee_function,
                             callee_is_interface}
    """
    calls: list[dict[str, Any]] = []
    try:
        for contract in sl.contracts:
            if contract.is_interface:
                continue
            for fn in contract.functions_and_modifiers:
                for callee_contract, callee_fn in getattr(fn, "high_level_calls", []):
                    callee_name = (
                        callee_fn.name
                        if hasattr(callee_fn, "name")
                        else str(callee_fn)
                    )
                    calls.append({
                        "caller_contract":  contract.name,
                        "caller_function":  fn.name,
                        "callee_contract":  callee_contract.name,
                        "callee_function":  callee_name,
                        "callee_is_interface": getattr(callee_contract, "is_interface", False),
                    })
    except Exception as exc:
        logger.debug("_extract_external_call_summary | partial failure (non-fatal): {}", exc)
    return calls


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

    RECALL — scoped detectors:
        Slither is run with only the detectors relevant to ML-flagged classes.
        CLASS_TO_DETECTORS in routing.py defines the mapping.
        This reduces runtime 3–8× vs running all 90+ detectors on large contracts.
        Any class above DEEP_THRESHOLDS contributes its detectors to the active set.
        If ml_result is empty or no class is flagged, all detectors run (safe fallback).

    State updates:
        static_findings → list of finding dicts (may be empty)
        error           → set on failure (non-fatal; node returns empty list)
    """
    contract_code = state.get("contract_code", "")
    if not contract_code or not contract_code.strip():
        logger.warning("static_analysis | contract_code empty — skipping")
        return {"static_findings": []}

    # Collect detector names relevant to classes above DEEP_THRESHOLDS.
    # Prefer probabilities dict (all 10 classes) over legacy vulnerabilities list.
    ml_result     = state.get("ml_result", {})
    probabilities = ml_result.get("probabilities", {})
    if probabilities:
        flagged_classes = {cls for cls, prob in probabilities.items() if prob >= 0.35}
    else:
        flagged_classes = {
            v["vulnerability_class"]
            for v in ml_result.get("vulnerabilities", [])
            if v.get("probability", 0.0) >= 0.35
        }
    scoped_detectors: set[str] = set()
    for cls in flagged_classes:
        scoped_detectors.update(CLASS_TO_DETECTORS.get(cls, []))

    logger.info(
        "static_analysis | running Slither | address={} | classes={} | detectors={}",
        state.get("contract_address", "unknown"),
        sorted(flagged_classes) or ["all"],
        sorted(scoped_detectors) or ["all"],
    )

    tmp_path: str | None = None
    try:
        from slither import Slither

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

        sl = Slither(tmp_path)

        # If we have a scoped detector set, filter _detectors before running.
        # Slither stores detector classes in sl._detectors (list).
        # Filtering by ARGUMENT attribute (the detector's CLI name) is stable
        # across Slither versions as it's part of the public detector interface.
        if scoped_detectors:
            sl._detectors = [  # type: ignore[attr-defined]
                d for d in sl._detectors  # type: ignore[attr-defined]
                if getattr(d, "ARGUMENT", "") in scoped_detectors
            ]

        # ExternalBug structural gap fix: GNN sees call_target_typed=1.00 for
        # typed interface calls (looks safe). Extract inter-contract call graph
        # so rag_research and synthesizer can reason about oracle/price manipulation.
        external_calls: list[dict] = []
        if "ExternalBug" in flagged_classes:
            external_calls = _extract_external_call_summary(sl)
            if external_calls:
                logger.info(
                    "static_analysis | ExternalBug: {} inter-contract call(s) extracted",
                    len(external_calls),
                )

        findings: list[dict] = []
        for result in sl.run_detectors():
            for finding in result:
                elements = finding.get("elements", [])
                lines: list[int] = []
                fn_names: list[str] = []
                for elem in elements:
                    src = elem.get("source_mapping", {})
                    elem_lines = src.get("lines", [])
                    if isinstance(elem_lines, list):
                        lines.extend(int(ln) for ln in elem_lines if isinstance(ln, int))
                    if elem.get("type") == "function":
                        fn_names.append(elem.get("name", ""))

                detector_name = finding.get("check", "unknown")
                findings.append({
                    "tool":           "slither",
                    "detector":       detector_name,
                    "impact":         finding.get("impact", "Unknown"),
                    "confidence":     finding.get("confidence", "Unknown"),
                    "description":    finding.get("description", ""),
                    "lines":          sorted(set(lines)),
                    "function_names": fn_names,
                })

        logger.info(
            "static_analysis complete | {} finding(s) | {} external call(s) | contract_address={}",
            len(findings),
            len(external_calls),
            state.get("contract_address", "unknown"),
        )
        return {"static_findings": findings, "external_call_summary": external_calls}

    except ImportError:
        logger.warning("static_analysis | slither not installed — skipping")
        return {"static_findings": [], "external_call_summary": []}

    except Exception as exc:
        logger.error("static_analysis failed: {}", exc)
        return {
            "static_findings": [],
            "external_call_summary": [],
            "error": f"static_analysis: {exc}",
        }

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("static_analysis | failed to delete temp file {}: {}", tmp_path, e)


# ---------------------------------------------------------------------------
# Node: graph_explain
# ---------------------------------------------------------------------------

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
        result = await _call_mcp_tool(
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


# ---------------------------------------------------------------------------
# Node: cross_validator
# ---------------------------------------------------------------------------

async def cross_validator(state: AuditState) -> dict[str, Any]:
    """
    LLM-adjudicated per-class verdicts (Phase 2 deep path).

    Runs after audit_check (fan-in) and before synthesizer.
    For each ML-flagged class, prompts the strong LLM with all available
    evidence (ML tier + probability, Slither findings, RAG chunks, prior audits)
    and returns a structured per-class verdict.

    Verdict scale:
        CONFIRMED  — ML ≥ 0.55 AND Slither finding(s) agree
        LIKELY     — ML ≥ 0.35 AND at least one corroborating signal
        DISPUTED   — ML flagged but corroboration absent or contradictory
        WATCH      — ML 0.25–0.34, single weak signal; monitor only
        SAFE       — evidence points to false positive

    Falls back silently to empty dict on LLM failure — synthesizer then
    computes rule-based verdicts as before.

    State updates:
        verdicts       → {class: verdict_str}
        confirmations  → {class: [evidence_source, ...]}
        contradictions → {class: [description, ...]}
    """
    ml_result       = state.get("ml_result",       {})
    static_findings = state.get("static_findings",  [])
    rag_results     = state.get("rag_results",      [])
    audit_history   = state.get("audit_history",    [])

    confirmed  = ml_result.get("confirmed",  [])
    suspicious = ml_result.get("suspicious", [])
    all_flagged = confirmed + suspicious or ml_result.get("vulnerabilities", [])

    if not all_flagged:
        logger.info("cross_validator | no flagged classes — skipping")
        return {}

    logger.info("cross_validator | adjudicating {} class(es)", len(all_flagged))

    # Build Slither-finding index keyed by vuln class (without importing routing again)
    from src.orchestration.routing import CLASS_TO_DETECTORS

    slither_by_class: dict[str, list[str]] = {}
    for finding in static_findings:
        detector = finding.get("detector", "")
        for cls, detectors in CLASS_TO_DETECTORS.items():
            if detector in detectors:
                slither_by_class.setdefault(cls, []).append(
                    f"{finding.get('impact', '')} {detector}: "
                    f"{finding.get('description', '')[:80]}"
                )

    rag_topics = [
        c.get("metadata", {}).get("vulnerability_type", "")
        for c in rag_results[:5]
        if c.get("metadata", {}).get("vulnerability_type")
    ]
    prior_count = len(audit_history)

    class_lines = []
    for vuln in all_flagged:
        cls    = vuln.get("vulnerability_class", "?")
        prob   = vuln.get("probability", 0.0)
        tier   = vuln.get("tier", "CONFIRMED")
        slither_hits = slither_by_class.get(cls, ["(no Slither findings)"])
        class_lines.append(
            f"- {cls} [{tier}] prob={prob:.3f}\n"
            f"  Slither: {'; '.join(slither_hits[:3])}\n"
            f"  RAG topics matched: {', '.join(rag_topics) or '(none)'}\n"
            f"  Prior audits: {prior_count}"
        )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.llm.client import get_strong_llm

        system_msg = SystemMessage(content=(
            "You are a smart contract security auditor. "
            "For each vulnerability class below, return a JSON object mapping "
            "class name → verdict string.\n"
            "Verdict options: CONFIRMED | LIKELY | DISPUTED | WATCH | SAFE\n"
            "Rules:\n"
            "  CONFIRMED: ML ≥ 0.55 AND Slither finding(s) agree\n"
            "  LIKELY:    ML ≥ 0.35 AND at least one corroborating signal\n"
            "  DISPUTED:  ML flagged but Slither/RAG contradicts or is absent\n"
            "  WATCH:     ML 0.25–0.34, single weak signal; monitor only\n"
            "  SAFE:      evidence points to false positive\n"
            "Return ONLY valid JSON, no markdown fences, no explanation.\n"
            'Example: {"Reentrancy": "CONFIRMED", "IntegerUO": "LIKELY"}'
        ))
        user_msg = HumanMessage(content=(
            "Vulnerability evidence:\n" + "\n".join(class_lines)
        ))

        llm = get_strong_llm()
        response = await asyncio.wait_for(
            asyncio.to_thread(llm.invoke, [system_msg, user_msg]),
            timeout=30.0,
        )

        raw = response.content.strip()
        # Strip markdown fences if the model wraps the JSON anyway
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        llm_verdicts: dict[str, str] = json.loads(raw)

        valid_verdicts = {"CONFIRMED", "LIKELY", "DISPUTED", "WATCH", "SAFE"}
        flagged_class_set = {v.get("vulnerability_class", "") for v in all_flagged}
        verdicts: dict[str, str] = {
            cls: (v if v in valid_verdicts else "DISPUTED")
            for cls, v in llm_verdicts.items()
            if cls in flagged_class_set
        }

        # Build confirmations / contradictions
        confirmations:  dict[str, list[str]] = {}
        contradictions: dict[str, list[str]] = {}
        for cls, verdict in verdicts.items():
            prob = next(
                (v["probability"] for v in all_flagged
                 if v.get("vulnerability_class") == cls),
                0.0,
            )
            sources = [f"ml:{prob:.3f}"]
            if slither_by_class.get(cls):
                sources.append(f"slither:{len(slither_by_class[cls])} finding(s)")
            if rag_topics:
                sources.append(f"rag:{len(rag_topics)} relevant chunk(s)")

            if verdict in ("CONFIRMED", "LIKELY"):
                confirmations[cls]  = sources
            elif verdict == "DISPUTED":
                contradictions[cls] = [
                    f"ml_flagged (prob={prob:.3f}) but insufficient corroboration"
                ]
                confirmations[cls]  = sources[:1]
            else:
                confirmations[cls]  = sources[:1]

        logger.info("cross_validator complete | verdicts={}", verdicts)
        return {
            "verdicts":       verdicts,
            "confirmations":  confirmations,
            "contradictions": contradictions,
        }

    except Exception as exc:
        logger.warning(
            "cross_validator | failed (synthesizer will use rule-based fallback): {}", exc
        )
        return {}


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
    ml_result       : dict       = state.get("ml_result",       {})
    rag_results     : list       = state.get("rag_results",     [])
    audit_history   : list       = state.get("audit_history",   [])
    static_findings : list       = state.get("static_findings",  [])
    routing_decisions: list      = state.get("routing_decisions", [])
    error           : str | None = state.get("error")

    label      = ml_result.get("label",     "unknown")
    confirmed  = ml_result.get("confirmed",  [])
    suspicious = ml_result.get("suspicious", [])
    # All flagged classes (CONFIRMED + SUSPICIOUS) — used for verdicts and report.
    # Falls back to legacy `vulnerabilities` field if three-tier keys absent.
    all_flagged = confirmed + suspicious or ml_result.get("vulnerabilities", [])
    threshold  = ml_result.get("threshold", 0.50)
    truncated  = ml_result.get("truncated", False)
    num_nodes  = ml_result.get("num_nodes", 0)
    num_edges  = ml_result.get("num_edges", 0)

    # Derive risk_probability and top_vulnerability from CONFIRMED tier first.
    risk_source = confirmed or suspicious or ml_result.get("vulnerabilities", [])
    if risk_source:
        top_vuln      = max(risk_source, key=lambda v: v.get("probability", 0.0))
        risk_prob     = round(top_vuln.get("probability", 0.0), 4)
        top_vuln_name = top_vuln.get("vulnerability_class")
    else:
        risk_prob     = 0.0
        top_vuln_name = None

    # Determine which path was taken.
    path_taken = "deep" if (rag_results or static_findings) else "fast"

    # ── Per-class verdict computation ────────────────────────────────────────
    # Use cross_validator pre-computed verdicts (deep path, LLM-adjudicated).
    # Fall back to rule-based compute_verdict() when cross_validator didn't run
    # (fast path) or when it failed (returned empty dict).
    pre_verdicts:      dict[str, str]       = state.get("verdicts",      {})
    pre_confirmations: dict[str, list[str]] = state.get("confirmations", {})

    verdicts:      dict[str, str]       = {}
    confirmations: dict[str, list[str]] = {}
    vuln_verdicts: list[dict]           = []

    for vuln in all_flagged:
        cls  = vuln.get("vulnerability_class", "?")
        prob = vuln.get("probability", 0.0)
        if cls in pre_verdicts:
            verdict = pre_verdicts[cls]
            sources = pre_confirmations.get(cls, [f"ml:{prob:.3f}"])
        else:
            verdict, sources = compute_verdict(
                cls, prob, static_findings, rag_results, path_taken
            )
        verdicts[cls]      = verdict
        confirmations[cls] = sources
        vuln_verdicts.append({
            "vulnerability_class": cls,
            "probability":         prob,
            "verdict":             verdict,
            "evidence_sources":    sources,
            "severity":            prob_to_severity(prob),
        })

    overall_verdict = compute_overall_verdict(verdicts)

    # ── Rule-based recommendation (fallback) ─────────────────────────────────
    # Used when the LLM is unavailable or times out.
    if not ml_result:
        recommendation = (
            "ML assessment failed — manual review required. "
            "Check that the inference server (port 8001) is running."
        )
    elif label in ("confirmed_vulnerable", "vulnerable") and risk_prob >= 0.70:
        rag_count    = len(rag_results)
        prior_count  = len(audit_history)
        slither_high = sum(
            1 for f in static_findings if f.get("impact") in ("High", "Medium")
        )
        recommendation = (
            f"HIGH RISK — top vulnerability: {top_vuln_name} "
            f"(probability {risk_prob:.1%}, CONFIRMED tier). "
            f"{rag_count} similar exploit pattern(s) found in DeFiHackLabs corpus. "
            f"{slither_high} Slither High/Medium finding(s). "
            f"{prior_count} prior on-chain audit(s). "
            "Recommend full manual audit before deployment."
        )
    elif label in ("confirmed_vulnerable", "suspicious", "vulnerable"):
        tier_note = (
            f"{len(confirmed)} confirmed, {len(suspicious)} suspicious"
            if (confirmed or suspicious)
            else f"probability {risk_prob:.1%}"
        )
        recommendation = (
            f"MODERATE RISK — top vulnerability: {top_vuln_name} "
            f"({tier_note}). "
            "Recommend targeted review of flagged patterns."
        )
    else:
        recommendation = (
            f"LOW RISK — no vulnerability exceeded detection threshold "
            f"(max probability: {risk_prob:.1%}). "
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
                f"  - [{v.get('tier', 'CONFIRMED')}] "
                f"{v.get('vulnerability_class', '?')}: {v.get('probability', 0.0):.1%}"
                for v in all_flagged
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

            # ExternalBug: add inter-contract call graph to prompt so LLM can
            # reason about oracle/price-feed dependency risks explicitly.
            ext_calls = state.get("external_call_summary", [])
            ext_flagged = any(
                v.get("vulnerability_class") == "ExternalBug" for v in all_flagged
            )
            ext_call_lines = ""
            if ext_flagged and ext_calls:
                ext_call_lines = "\n**Inter-contract call graph (ExternalBug context):**\n" + "\n".join(
                    f"  {c['caller_function']}({c['caller_contract']}) "
                    f"→ {c['callee_contract']}.{c['callee_function']}"
                    + (" [INTERFACE]" if c.get("callee_is_interface") else "")
                    for c in ext_calls[:8]
                )

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

            tier_summary = (
                f"{len(confirmed)} CONFIRMED (≥0.55), {len(suspicious)} SUSPICIOUS (0.25–0.54)"
                if (confirmed or suspicious) else ""
            )
            user_msg = HumanMessage(content=(
                f"**Contract address:** {state.get('contract_address', 'unknown')}\n"
                f"**ML model assessment:** {label}"
                + (f" — {tier_summary}" if tier_summary else "") + "\n\n"
                f"**Detected vulnerabilities (tier: class: probability):**\n{vuln_lines}\n\n"
                f"**RAG exploit evidence:**\n{rag_lines}\n\n"
                f"**Static analysis findings (High/Medium):**\n{slither_lines}\n"
                + ext_call_lines + "\n\n"
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
        "contract_address":       state.get("contract_address", ""),
        "overall_label":          label,
        "overall_verdict":        overall_verdict,
        "risk_probability":       risk_prob,
        "top_vulnerability":      top_vuln_name,
        "confirmed":              confirmed,
        "suspicious":             suspicious,
        "vulnerabilities":        all_flagged,
        "probabilities":          ml_result.get("probabilities", {}),
        "tier_thresholds":        ml_result.get("tier_thresholds", {}),
        "vulnerability_verdicts": vuln_verdicts,
        "threshold":              threshold,
        "ml_truncated":           truncated,
        "num_nodes":              num_nodes,
        "num_edges":              num_edges,
        "rag_evidence":           rag_results,
        "audit_history":          audit_history,
        "static_findings":        static_findings,
        "external_call_summary":  state.get("external_call_summary", []),
        "routing_decisions":      routing_decisions,
        "recommendation":         final_recommendation,
        "narrative":              narrative,
        "error":                  error,
        "path_taken":             path_taken,
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
        "synthesizer complete | label={} | verdict={} | risk_prob={:.3f} | "
        "top_vuln={} | confirmed={} | suspicious={} | path={} | "
        "rag_chunks={} | prior_audits={} | static_findings={}",
        label,
        overall_verdict,
        risk_prob,
        top_vuln_name,
        len(confirmed),
        len(suspicious),
        path_taken,
        len(rag_results),
        len(audit_history),
        len(static_findings),
    )

    return {
        "final_report":  report,
        "verdicts":      verdicts,
        "confirmations": confirmations,
    }
