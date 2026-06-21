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
from src.orchestration.timing import step_timer
from src.orchestration.timeouts import (
    ENV_ADERYN_TIMEOUT_S,
    ENV_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
    ENV_DEBATE_TIMEOUT_S,
    ENV_REFLECTION_TIMEOUT_S,
    ENV_SYNTHESIZER_NARRATIVE_TIMEOUT_S,
    DEFAULT_ADERYN_TIMEOUT_S,
    DEFAULT_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
    DEFAULT_DEBATE_TIMEOUT_S,
    DEFAULT_REFLECTION_TIMEOUT_S,
    DEFAULT_SYNTHESIZER_NARRATIVE_TIMEOUT_S,
    get_timeout,
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
# LLM enable guard (Phase A) — lets the test session disable all LLM calls so
# the graph runs fast and deterministically without a live LM Studio, while
# real runs (run_real_audit.py) leave it enabled. LLM-calling nodes
# (cross_validator, synthesizer narrative, reflection) consult this first.
# ---------------------------------------------------------------------------

def _llm_enabled() -> bool:
    """False when AGENTS_DISABLE_LLM is truthy — nodes then use rule-based fallback."""
    return os.getenv("AGENTS_DISABLE_LLM", "").strip().lower() not in ("1", "true", "yes")


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
    # "integer-overflow" removed 2026-06-21 — does not exist as a Slither 0.11.5
    # detector ARGUMENT (removed upstream; Solidity >=0.8 has checked arithmetic).
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

# Removed 2026-06-21: _SCREEN_ADERYN_HIGH_IDS (a frozenset of "H-1".."C-3" labels)
# was dead code — never referenced anywhere. It also encoded a wrong assumption:
# those labels are per-report table-of-contents positions (H-1, H-2, ...), not
# stable detector identifiers — they'd renumber per contract. quick_screen now
# escalates on any Aderyn finding with impact=="High" directly (see below),
# matching real detector_name values from the verified JSON schema.


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
        import inspect

        from slither import Slither
        from slither.detectors import all_detectors
        from slither.detectors.abstract_detector import AbstractDetector

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

        # CRITICAL (found 2026-06-21, same bug as static_analysis): Slither()
        # registers ZERO detectors on construction — must register the full set
        # before scoping/running, or quick_screen NEVER escalates on anything.
        for detector_cls in (
            d for d in (getattr(all_detectors, name) for name in dir(all_detectors))
            if inspect.isclass(d) and issubclass(d, AbstractDetector)
        ):
            sl.register_detector(detector_cls)

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
    # Delegates to _run_aderyn_on_file (defined below) — fixed 2026-06-21 to use
    # a real directory ROOT + a real --output file path + the actual JSON schema
    # (see that function's docstring). Previously this block had its own
    # independent invocation that failed identically (file-not-directory error,
    # silently swallowed) — Aderyn never escalated anything here either.
    try:
        for finding in _run_aderyn_on_file(contract_code):
            if finding["impact"] == "High" and finding["detector"] not in aderyn_hits:
                aderyn_hits.append(finding["detector"])
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

def _parse_aderyn_report(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Normalise Aderyn 0.6.8's real JSON report schema into SENTINEL finding dicts.

    REAL schema (verified 2026-06-21 by manually running `aderyn --output
    <path>.json <dir>` and inspecting the file — the PREVIOUS parser assumed a
    schema that does not exist and silently produced zero findings forever):
        {"high_issues": {"issues": [{"detector_name", "title", "description",
                                      "instances": [{"contract_path", "line_no",
                                                      "src", "src_char", "hint"?}]}]},
         "low_issues":  {"issues": [...]}}
    There is no "medium" bucket in this Aderyn version. Instances carry NO
    function-name field (only contract_path/line_no/src/src_char/hint) — do not
    fabricate one.
    """
    findings: list[dict[str, Any]] = []
    for impact_label, bucket_key in [("High", "high_issues"), ("Low", "low_issues")]:
        for issue in data.get(bucket_key, {}).get("issues", []):
            instances = issue.get("instances", [])
            lines = sorted({
                inst["line_no"] for inst in instances
                if isinstance(inst.get("line_no"), int)
            })
            findings.append({
                "tool":           "aderyn",
                "detector":       issue.get("detector_name", issue.get("title", "unknown")),
                "impact":         impact_label,
                "confidence":     "Medium",  # Aderyn does not report a confidence score
                "description":    issue.get("description", issue.get("title", "")),
                "lines":          lines,
                "function_names": [],  # not present in this Aderyn JSON schema
            })
    return findings


def _run_aderyn_on_file(contract_code: str) -> list[dict[str, Any]]:
    """
    Run Aderyn on contract source and return a normalised findings list.

    Called by static_analysis (deep path) to augment Slither findings with Aderyn's
    detector set. The two tools have different false-negative profiles — running both
    on the deep path increases coverage without slowing down the fast path.

    CRITICAL (found 2026-06-21 via manual verification): Aderyn 0.6.8 requires a
    DIRECTORY as its [ROOT] argument ("Not a directory (os error 20)" on a file
    path) and `--output` takes a REAL FILE PATH ending in .json/.md/.sarif, not a
    bare format word. The previous invocation
    (`aderyn --output json <file>`) failed BOTH requirements simultaneously,
    exit code 1, silently swallowed by the `returncode != 0` early-return below
    — Aderyn has never produced a finding through this path before this fix.

    Non-fatal: any failure (not installed, timeout, bad JSON) returns an empty list.

    Returns:
        List of finding dicts: {tool="aderyn", detector, impact, confidence,
                                description, lines, function_names}
        Empty list on any error.
    """
    import json as _json
    import shutil
    import subprocess

    findings: list[dict[str, Any]] = []
    tmpdir: str | None = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="sentinel_aderyn_")
        sol_path = Path(tmpdir) / "contract.sol"
        sol_path.write_text(contract_code)
        report_path = Path(tmpdir) / "report.json"

        result = subprocess.run(
            ["aderyn", "--output", str(report_path), tmpdir],
            capture_output=True,
            text=True,
            timeout=get_timeout(ENV_ADERYN_TIMEOUT_S, DEFAULT_ADERYN_TIMEOUT_S),
        )
        if result.returncode != 0 or not report_path.exists():
            logger.debug(
                "_run_aderyn_on_file | exit={} stderr={}",
                result.returncode,
                result.stderr[:200],
            )
            return []

        data = _json.loads(report_path.read_text())
        findings = _parse_aderyn_report(data)

    except FileNotFoundError:
        logger.debug("_run_aderyn_on_file | aderyn not installed — skipping")
    except subprocess.TimeoutExpired:
        logger.warning("_run_aderyn_on_file | aderyn timed out after 90s")
    except Exception as exc:
        logger.warning("_run_aderyn_on_file | error (non-fatal): {}", exc)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return findings


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
        Two tools run on the same temp file; findings are combined in one list.

        Slither (scoped to ML-flagged classes):
            {tool="slither", detector, impact, confidence, description, lines, function_names}

        Aderyn (full scan — Aderyn has no per-class scoping):
            {tool="aderyn", detector, impact, confidence, description, lines, function_names}
            Non-fatal: silently skipped if Aderyn is not installed.

        Having both tool names in the findings lets cross_validator and synthesizer
        reason about corroboration: "Slither AND Aderyn both found X" is stronger
        evidence than either tool alone.

    RECALL — scoped detectors (Slither only):
        Slither is run with only the detectors relevant to ML-flagged classes.
        CLASS_TO_DETECTORS in routing.py defines the mapping.
        This reduces runtime 3–8× vs running all 90+ detectors on large contracts.
        Any class above DEEP_THRESHOLDS contributes its detectors to the active set.
        If ml_result is empty or no class is flagged, all detectors run (safe fallback).
        Aderyn always runs its full detector set — it has no equivalent scope API.

    State updates:
        static_findings → combined Slither + Aderyn findings (may be empty)
        error           → set on Slither failure (non-fatal; returns empty list)
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
        import inspect

        from slither import Slither
        from slither.detectors import all_detectors
        from slither.detectors.abstract_detector import AbstractDetector

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

        # CRITICAL (found 2026-06-21 via manual real-contract verification):
        # The Slither() constructor registers ZERO detectors — sl._detectors starts
        # empty. The CLI (slither/__main__.py) explicitly registers every detector
        # class via slither.register_detector() before calling run_detectors();
        # this in-process API call must do the same or static_analysis silently
        # finds NOTHING on every contract, regardless of vulnerability. Confirmed by
        # direct comparison: `slither contract.sol` on the CLI found reentrancy-eth
        # on a textbook reentrant Vault; this node (pre-fix) found 0.
        all_detector_classes = [
            d for d in (getattr(all_detectors, name) for name in dir(all_detectors))
            if inspect.isclass(d) and issubclass(d, AbstractDetector)
        ]
        for detector_cls in all_detector_classes:
            sl.register_detector(detector_cls)

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

        # Run Aderyn on the same source — adds findings with tool="aderyn".
        # Non-fatal: _run_aderyn_on_file returns [] on any failure.
        aderyn_findings = _run_aderyn_on_file(contract_code)
        findings.extend(aderyn_findings)

        logger.info(
            "static_analysis complete | slither={} aderyn={} external_calls={} | contract_address={}",
            len(findings) - len(aderyn_findings),
            len(aderyn_findings),
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

    if not _llm_enabled():
        logger.info("cross_validator | LLM disabled (AGENTS_DISABLE_LLM) — rule-based fallback")
        return {}

    # Real-audit finding (2026-06-21): ambiguous/small contracts can spread weak
    # "suspicious" signal (tier threshold 0.25) across most of the 10 classes.
    # Adjudicating all of them — especially with the 3-call debate plus the
    # contract source in every prompt — risks the FAST model overrunning
    # CROSS_VALIDATOR_TIMEOUT_S (3 sequential calls) on a tiny RTX 3070 box.
    # Cap to the top-N most probable classes; the rest fall back to rule-based
    # verdicts in synthesizer (they're weak signals anyway).
    _max_classes = int(os.getenv("CROSS_VALIDATOR_MAX_CLASSES", "5"))
    if len(all_flagged) > _max_classes:
        all_flagged = sorted(
            all_flagged, key=lambda v: v.get("probability", 0.0), reverse=True
        )[:_max_classes]

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

    debate_transcript: dict[str, str] = {}
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        # 2026-06-17: Use FAST model (gemma-4-e2b-it) instead of STRONG.
        # Verdict picking from 5 options is a simple classification task.
        # STRONG (qwen3.5-9b-ud) was taking 94s+ for 9 classes (TIMED OUT at 90s).
        # FAST runs ~3x faster, finishes in ~20-30s, quality is sufficient.
        # Override via CROSS_VALIDATOR_LLM_MODEL env var (model ID string).
        from src.llm.client import get_fast_llm, get_strong_llm
        _cv_model = os.getenv("CROSS_VALIDATOR_LLM_MODEL", "fast").lower()
        if _cv_model == "strong":
            llm = get_strong_llm()
        else:
            llm = get_fast_llm()

        evidence_block = "\n".join(class_lines)
        # Ali directive (2026-06-21): the ML model is only a HINT. Give the
        # debate the actual SOURCE so the LLM forms an INDEPENDENT judgement
        # from the code, rather than rubber-stamping the ML/tool summary.
        _code_for_debate = (state.get("contract_code", "") or "")[:2000].strip()
        code_block = (
            f"\n\nContract source (analyse it yourself — the ML signal is only a hint):\n"
            f"```solidity\n{_code_for_debate}\n```" if _code_for_debate else ""
        )
        # ── A.4 Multi-LLM debate (Prosecutor → Defender → Judge) ─────────────
        # DEBATE_MODE (default on) runs three role-specific passes so the final
        # verdict reflects an adversarial exchange rather than one classification
        # call. Any failure raises and is caught below → silent rule-based
        # fallback in synthesizer.
        #
        # Real-audit finding (2026-06-21): an EARLIER version applied
        # CROSS_VALIDATOR_TIMEOUT_S (90s) PER CALL inside _ask(). With 3
        # sequential calls that allowed up to 270s worst case — which blew
        # past the calling script's own timeout (observed: a 200s script
        # timeout killed the process mid-debate, abandoning an
        # asyncio.to_thread() call that keeps running in its OS thread even
        # after cancellation, since to_thread cannot be cancelled).
        # FIX: ONE outer timeout bounds the entire debate (or single-pass
        # call) as a unit — DEBATE_TIMEOUT_S (default 240s) when debate is
        # on, CROSS_VALIDATOR_TIMEOUT_S (default 90s) for single-pass.
        _debate_on = os.getenv("DEBATE_MODE", "on").strip().lower() in ("1", "true", "on", "yes")
        _address = state.get("contract_address", "unknown")

        async def _ask(role: str, system: str, user: str) -> str:
            # Per-role timing (2026-06-21): each debate role is individually
            # logged so a live run shows exactly which of the 3 sequential
            # calls is slow, rather than only the aggregate debate duration.
            with step_timer(f"cross_validator.{role}", address=_address):
                resp = await asyncio.to_thread(
                    llm.invoke,
                    [SystemMessage(content=system), HumanMessage(content=user)],
                )
            return resp.content.strip()

        judge_system = (
            "You are the JUDGE in a smart contract security review. "
            "You have heard the prosecutor (argues vulnerable) and the defender "
            "(argues false-positive). For each vulnerability class below, return a "
            "JSON object mapping class name → verdict string.\n"
            "Verdict options: CONFIRMED | LIKELY | DISPUTED | WATCH | SAFE\n"
            "Rules:\n"
            "  CONFIRMED: ML ≥ 0.55 AND Slither finding(s) agree\n"
            "  LIKELY:    ML ≥ 0.35 AND at least one corroborating signal\n"
            "  DISPUTED:  ML flagged but Slither/RAG contradicts or is absent\n"
            "  WATCH:     ML 0.25–0.34, single weak signal; monitor only\n"
            "  SAFE:      evidence points to false positive\n"
            "Return ONLY valid JSON, no markdown fences, no explanation.\n"
            'Example: {"Reentrancy": "CONFIRMED", "IntegerUO": "LIKELY"}'
        )

        async def _run_debate() -> tuple[str, dict[str, str]]:
            prosecutor = await _ask(
                "prosecutor",
                "You are a security PROSECUTOR. Read the contract source yourself and "
                "argue concisely why it HAS the vulnerabilities below. Ground each "
                "claim in the actual code; cite supporting evidence (Slither detector, "
                "RAG match) where it agrees. Treat the ML probability as a weak hint "
                "only — do not assert a vulnerability the code does not support.",
                "Vulnerability evidence:\n" + evidence_block + code_block,
            )
            defender = await _ask(
                "defender",
                "You are a skeptical DEFENDER. Read the contract source yourself. "
                "Given the prosecutor's case, argue concisely why these findings may be "
                "false positives or low severity (e.g. typed interface calls, benign "
                "timestamp use, guarded external calls). The ML model is known to "
                "over-predict — challenge claims the code does not justify.",
                f"Prosecutor's case:\n{prosecutor}\n\nEvidence:\n{evidence_block}{code_block}",
            )
            judge_raw = await _ask(
                "judge",
                judge_system,
                f"Prosecutor:\n{prosecutor}\n\nDefender:\n{defender}\n\n"
                f"Evidence:\n{evidence_block}",
            )
            transcript = {"prosecutor": prosecutor, "defender": defender, "judge": judge_raw}
            return judge_raw, transcript

        if _debate_on:
            _debate_timeout = get_timeout(ENV_DEBATE_TIMEOUT_S, DEFAULT_DEBATE_TIMEOUT_S)
            with step_timer("cross_validator.debate_total", address=_address, budget_s=_debate_timeout):
                raw, debate_transcript = await asyncio.wait_for(_run_debate(), timeout=_debate_timeout)
            logger.info("cross_validator | debate complete (3 roles)")
        else:
            # Legacy single-pass classification (retained for perf/tests).
            _single_pass_timeout = get_timeout(
                ENV_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
                DEFAULT_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
            )
            raw = await asyncio.wait_for(
                _ask("single_pass", judge_system, "Vulnerability evidence:\n" + evidence_block),
                timeout=_single_pass_timeout,
            )

        raw = raw.strip()
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
        result: dict[str, Any] = {
            "verdicts":       verdicts,
            "confirmations":  confirmations,
            "contradictions": contradictions,
        }
        if debate_transcript:
            result["debate_transcript"] = debate_transcript
        return result

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
    # Verdict source priority (fixed 2026-06-21 — see "Verdict fallback gap"
    # incident, docs/changes/2026-06-21-agents-manual-verification-real-bugs-found.md):
    #   1. cross_validator's debate verdict — read the actual source, strictly
    #      the most informed source when it succeeds.
    #   2. consensus_engine's vote — ML-discounted (ML_WEIGHT_SCALE), already
    #      tool-corroborated, computed on EVERY deep-path run regardless of LLM
    #      availability. This is the correct fallback when the debate fails —
    #      NOT compute_verdict() (see #3), which predates the ML-as-hint design
    #      and disagreed with consensus_engine in production (verified live:
    #      consensus_engine said ExternalBug=SAFE on safe_storage.sol while
    #      compute_verdict() said DISPUTED for the identical evidence).
    #   3. compute_verdict() — last resort, for classes consensus_engine never
    #      scored (it only votes when ML prob >= 0.50 or a tool hit exists;
    #      weaker SUSPICIOUS-tier noise below that bar still needs an answer).
    pre_verdicts:      dict[str, str]       = state.get("verdicts",      {})
    pre_confirmations: dict[str, list[str]] = state.get("confirmations", {})
    consensus_verdict: dict[str, dict]      = state.get("consensus_verdict", {})

    verdicts:      dict[str, str]       = {}
    confirmations: dict[str, list[str]] = {}
    vuln_verdicts: list[dict]           = []

    for vuln in all_flagged:
        cls  = vuln.get("vulnerability_class", "?")
        prob = vuln.get("probability", 0.0)
        if cls in pre_verdicts:
            verdict = pre_verdicts[cls]
            sources = pre_confirmations.get(cls, [f"ml:{prob:.3f}"])
        elif cls in consensus_verdict:
            verdict = consensus_verdict[cls]["verdict"]
            sources = [
                f"ml:{prob:.3f}",
                f"consensus:confidence={consensus_verdict[cls]['confidence']:.2f}",
            ]
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
    if ml_result and _llm_enabled():
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from src.llm.client import get_strong_llm

            # Fix applied 2026-06-21 ("narrative hallucination" incident, see
            # docs/changes/2026-06-21-agents-manual-verification-real-bugs-found.md):
            # vuln_lines used to list every ML-flagged class (down to weak
            # SUSPICIOUS-tier noise that never even reached the deep-path
            # threshold) with NO verdict attached. The model had no signal that
            # a listed class had actually been cleared, and wrote about it as
            # if it were a real finding — observed live: a narrative described
            # a "Reentrancy risk" on a contract whose Reentrancy verdict was
            # SAFE and which contains zero external calls. `verdicts` (computed
            # just above) is now attached to every line, mirroring the pattern
            # reflection's prompt already used correctly.
            vuln_lines = "\n".join(
                f"  - [{v.get('tier', 'CONFIRMED')}] "
                f"{v.get('vulnerability_class', '?')}: {v.get('probability', 0.0):.1%} "
                f"→ verdict: {verdicts.get(v.get('vulnerability_class', ''), 'PENDING')}"
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
                "2–3 sentences describing what was detected and why it is dangerous. "
                "Only discuss classes whose verdict below is CONFIRMED or LIKELY — "
                "if a class's verdict is SAFE or DISPUTED, do NOT describe it as a "
                "real risk even though it appears in the list, and do NOT introduce "
                "a vulnerability class that is not in the list at all.\n"
                "## Exploit Pattern\n"
                "How an attacker could exploit the CONFIRMED/LIKELY class(es) above — "
                "the RAG section below is general background on similar historical "
                "exploits, not necessarily evidence about THIS contract; only cite it "
                "if it genuinely matches a CONFIRMED/LIKELY class.\n"
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
                f"**ML-flagged classes (tier: class: probability: verdict):**\n{vuln_lines}\n\n"
                f"**RAG retrieved exploit patterns (general historical reference — "
                f"NOT necessarily about this contract; only use if it matches a "
                f"CONFIRMED/LIKELY class above):**\n{rag_lines}\n\n"
                f"**Static analysis findings (High/Medium):**\n{slither_lines}\n"
                + ext_call_lines + "\n\n"
                f"**Contract code snippet (first 500 chars):**\n```solidity\n{code_snippet}\n```"
            ))

            llm = get_strong_llm(max_tokens=int(os.getenv("SYNTHESIZER_MAX_TOKENS", "4096")))
            # Timeout configurable via SYNTHESIZER_TIMEOUT_S env var.
            # 2026-06-17: bumped default 45s → 120s. The narrative prompt is
            # longer than cross_validator (full contract snippet + multi-class
            # summary). 120s gives 2.5× headroom. Failure → narrative=None
            # (synthesizer rule-based fallback handles the rest).
            # FIX-17 (2026-06-17): pass max_tokens=4096 to LLM. Without this,
            # LM Studio's default ~2K is too small for the 4-section narrative
            # + reasoning content (model returns content="" → empty exception).
            _synthesizer_timeout = get_timeout(
                ENV_SYNTHESIZER_NARRATIVE_TIMEOUT_S, DEFAULT_SYNTHESIZER_NARRATIVE_TIMEOUT_S
            )
            with step_timer(
                "synthesizer.narrative",
                address=state.get("contract_address", "unknown"),
                budget_s=_synthesizer_timeout,
            ):
                response = await asyncio.wait_for(
                    asyncio.to_thread(llm.invoke, [system_msg, user_msg]),
                    timeout=_synthesizer_timeout,
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


# ===========================================================================
# Extended Capability — Phase A nodes (2026-06-21)
# consensus_engine (A.6/A.7) · reflection (A.3) · explainer (A.8) · visualizer (A.9)
# All additive, all fail-soft (never raise — errors captured into state).
# ===========================================================================


def _signals_for_class(class_name: str, static_findings: list[dict]) -> tuple[bool, bool]:
    """
    Did Slither and/or Aderyn corroborate `class_name`?

    Slither findings carry detector names that map directly via CLASS_TO_DETECTORS.
    Aderyn uses its own detector ids, so we match by token overlap with the
    class's Slither detector names (e.g. Aderyn "reentrancy-state-change" shares
    the "reentrancy" token with Slither "reentrancy-eth"). Best-effort but robust.
    """
    detectors = CLASS_TO_DETECTORS.get(class_name, [])
    tokens = {tok for det in detectors for tok in det.split("-") if len(tok) > 3}

    slither_found = False
    aderyn_found = False
    for f in static_findings:
        det = str(f.get("detector", "")).lower()
        tool = f.get("tool", "")
        if tool == "slither" and det in detectors:
            slither_found = True
        elif tool == "aderyn" and any(tok in det for tok in tokens):
            aderyn_found = True
    return slither_found, aderyn_found


def _best_rag_score(class_name: str, rag_results: list[dict]) -> float:
    """Best RAG similarity whose metadata vulnerability_type matches the class."""
    best = 0.0
    for c in rag_results:
        vt = c.get("metadata", {}).get("vulnerability_type", "")
        if vt == class_name:
            best = max(best, float(c.get("score", 0.0) or 0.0))
    return best


async def consensus_engine(state: AuditState) -> dict[str, Any]:
    """
    Tool consensus voting + staged confidence (A.6 + A.7).

    Runs after audit_check (Slither/Aderyn available), before cross_validator.
    For each flagged class (ML ≥ threshold OR a static-tool hit), computes a
    weighted ML/Slither/Aderyn vote and a Bayesian-updated confidence.

    State updates:
        consensus_verdict   → {class: {ml_signal, slither_match, aderyn_match,
                                       score, confidence, verdict, weights}}
        confidence_by_class → {class: confidence in [0,1]}
    """
    from src.orchestration.consensus import consensus_vote
    from src.orchestration.confidence import track_confidence

    ml_result = state.get("ml_result", {})
    probabilities: dict[str, float] = ml_result.get("probabilities", {}) or {}
    static_findings = state.get("static_findings", []) or []
    rag_results = state.get("rag_results", []) or []

    # Fall back to flagged-class probabilities if the full vector is absent.
    if not probabilities:
        for v in (ml_result.get("confirmed", []) + ml_result.get("suspicious", [])
                  or ml_result.get("vulnerabilities", [])):
            probabilities[v.get("vulnerability_class", "?")] = v.get("probability", 0.0)

    if not probabilities:
        logger.info("consensus_engine | no ML probabilities — skipping")
        return {}

    consensus_verdict: dict[str, dict[str, Any]] = {}
    confidence_by_class: dict[str, float] = {}

    for cls, prob in probabilities.items():
        slither_found, aderyn_found = _signals_for_class(cls, static_findings)
        # Only emit a row when at least one tool has a positive signal.
        if prob < 0.50 and not slither_found and not aderyn_found:
            continue
        vote = consensus_vote(float(prob), slither_found, aderyn_found, cls)
        consensus_verdict[cls] = vote
        confidence_by_class[cls] = track_confidence(
            float(prob),
            slither_found=slither_found,
            aderyn_found=aderyn_found,
            rag_score=_best_rag_score(cls, rag_results),
        )

    logger.info(
        "consensus_engine complete | voted {} class(es)", len(consensus_verdict)
    )
    return {
        "consensus_verdict":   consensus_verdict,
        "confidence_by_class": confidence_by_class,
    }


async def reflection(state: AuditState) -> dict[str, Any]:
    """
    Self-critique pass (A.3) — runs after synthesizer.

    Checks the assembled audit for internal consistency:
        - unused_evidence:    collected (RAG/static) but not reflected in verdicts
        - contradictions:     tools disagreeing (from cross_validator)
        - uncertain_verdicts: DISPUTED/WATCH or confidence < 0.7
        - failure_modes:      what could make this audit wrong

    Uses the strong LLM when available for a narrative `summary`; always
    computes the structured lists rule-based so the node is meaningful even
    with no LLM. Never raises.

    State updates:
        reflection_notes → {unused_evidence, contradictions, uncertain_verdicts,
                            failure_modes, summary, llm_used}
    """
    report = state.get("final_report", {}) or {}
    verdicts: dict[str, str] = report.get("verdicts", {}) or state.get("verdicts", {}) or {}
    vuln_verdicts = report.get("vulnerability_verdicts", []) or []
    static_findings = state.get("static_findings", []) or []
    rag_results = state.get("rag_results", []) or []
    contradictions = state.get("contradictions", {}) or {}
    confidence_by_class = state.get("confidence_by_class", {}) or {}
    truncated = report.get("ml_truncated", False)

    cited_classes = {v.get("vulnerability_class") for v in vuln_verdicts}

    # ── Rule-based structured critique ───────────────────────────────────────
    unused_evidence: list[str] = []
    if rag_results and not cited_classes:
        unused_evidence.append(
            f"{len(rag_results)} RAG exploit chunk(s) retrieved but no class was adjudicated."
        )
    uncited_static = [
        f.get("detector") for f in static_findings
        if f.get("impact") in ("High", "Medium")
        and not any(d in CLASS_TO_DETECTORS.get(c, []) for c in cited_classes
                    for d in [f.get("detector")])
    ]
    if uncited_static:
        unused_evidence.append(
            f"{len(uncited_static)} High/Medium static finding(s) not tied to a verdict: "
            f"{', '.join(str(d) for d in uncited_static[:5])}"
        )

    uncertain: list[str] = []
    for v in vuln_verdicts:
        cls = v.get("vulnerability_class", "?")
        verdict = v.get("verdict", "")
        conf = confidence_by_class.get(cls)
        if verdict in ("DISPUTED", "WATCH"):
            uncertain.append(f"{cls}: verdict {verdict}")
        elif isinstance(conf, (int, float)) and conf < 0.70:
            uncertain.append(f"{cls}: confidence {conf:.0%} below 0.70")

    failure_modes: list[str] = []
    if truncated:
        failure_modes.append(
            "Contract exceeded 512 CodeBERT tokens — tail code unanalysed; "
            "verdicts may miss vulnerabilities in the truncated region."
        )
    if any(v.get("vulnerability_class") == "ExternalBug" for v in vuln_verdicts):
        failure_modes.append(
            "ExternalBug flagged — the ML model is known to over-predict this class "
            "(Run 12 class-definition mismatch); treat as lower confidence unless "
            "corroborated by an inter-contract call finding."
        )
    if not static_findings:
        failure_modes.append(
            "No static-analysis findings available (tool unavailable or fast path) — "
            "verdicts rest on the ML signal alone."
        )

    contradiction_list = [
        f"{cls}: {'; '.join(reasons)}" for cls, reasons in contradictions.items()
    ]

    notes: dict[str, Any] = {
        "unused_evidence":    unused_evidence,
        "contradictions":     contradiction_list,
        "uncertain_verdicts": uncertain,
        "failure_modes":      failure_modes,
        "summary":            "",
        "llm_used":           False,
    }

    # ── Optional LLM narrative summary ───────────────────────────────────────
    if _llm_enabled() and vuln_verdicts:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from src.llm.client import get_strong_llm

            verdict_lines = "\n".join(
                f"  - {v.get('vulnerability_class')}: {v.get('verdict')} "
                f"(prob {v.get('probability', 0.0):.1%}, "
                f"conf {confidence_by_class.get(v.get('vulnerability_class'), 0.0):.0%})"
                for v in vuln_verdicts
            )
            system_msg = SystemMessage(content=(
                "You are a senior auditor performing a SELF-CRITIQUE of an audit "
                "already produced. In 3-5 sentences, flag any internal "
                "inconsistencies, over/under-confidence, or evidence that was "
                "ignored. Be skeptical and concrete. Output plain prose only."
            ))
            user_msg = HumanMessage(content=(
                f"Verdicts:\n{verdict_lines}\n\n"
                f"Unused evidence: {unused_evidence or 'none'}\n"
                f"Contradictions: {contradiction_list or 'none'}\n"
                f"Uncertain: {uncertain or 'none'}\n"
                f"Failure modes: {failure_modes or 'none'}"
            ))
            _timeout = get_timeout(ENV_REFLECTION_TIMEOUT_S, DEFAULT_REFLECTION_TIMEOUT_S)
            llm = get_strong_llm(max_tokens=int(os.getenv("REFLECTION_MAX_TOKENS", "1024")))
            with step_timer(
                "reflection.llm_summary",
                address=state.get("contract_address", "unknown"),
                budget_s=_timeout,
            ):
                resp = await asyncio.wait_for(
                    asyncio.to_thread(llm.invoke, [system_msg, user_msg]),
                    timeout=_timeout,
                )
            notes["summary"] = resp.content.strip()
            notes["llm_used"] = True
            logger.info("reflection | LLM self-critique generated ({} chars)", len(notes["summary"]))
        except Exception as exc:
            logger.warning("reflection | LLM failed (rule-based notes kept): {}", exc)

    if not notes["summary"]:
        parts = []
        if uncertain:
            parts.append(f"{len(uncertain)} verdict(s) are uncertain")
        if unused_evidence:
            parts.append(f"{len(unused_evidence)} evidence item(s) unused")
        if failure_modes:
            parts.append(f"{len(failure_modes)} failure mode(s) noted")
        notes["summary"] = (
            "Self-critique: " + "; ".join(parts) + "."
            if parts else "Self-critique: audit is internally consistent; no concerns flagged."
        )

    logger.info(
        "reflection complete | uncertain={} unused={} failure_modes={} llm={}",
        len(uncertain), len(unused_evidence), len(failure_modes), notes["llm_used"],
    )
    return {"reflection_notes": notes}


async def explainer(state: AuditState) -> dict[str, Any]:
    """
    Metric attribution (A.8) — runs after reflection.

    For each verdict, attributes the evidence LIME-style across ML / Slither /
    RAG (percentages sum to ~100). Also folds confidence_by_class,
    consensus_verdict, metric_attribution and reflection_notes INTO the
    final_report so a single artifact carries the full Phase-A enrichment.

    State updates:
        metric_attribution → {class: {ml_pct, slither_pct, rag_pct}}
        final_report       → enriched copy
    """
    from src.orchestration.attribution import attribute_verdict

    report = dict(state.get("final_report", {}) or {})
    vuln_verdicts = report.get("vulnerability_verdicts", []) or []
    static_findings = state.get("static_findings", []) or []
    rag_results = state.get("rag_results", []) or []
    confidence_by_class = state.get("confidence_by_class", {}) or {}

    attribution: dict[str, dict[str, float]] = {}
    for v in vuln_verdicts:
        cls = v.get("vulnerability_class", "?")
        prob = float(v.get("probability", 0.0) or 0.0)
        slither_found, _ = _signals_for_class(cls, static_findings)
        rag_score = _best_rag_score(cls, rag_results)
        attribution[cls] = attribute_verdict(prob, slither_found, rag_score)
        # annotate the verdict row in place for the report
        v["attribution"] = attribution[cls]
        if cls in confidence_by_class:
            v["confidence"] = confidence_by_class[cls]

    report["metric_attribution"] = attribution
    report["confidence_by_class"] = confidence_by_class
    report["consensus_verdict"] = state.get("consensus_verdict", {})
    report["reflection_notes"] = state.get("reflection_notes", {})

    logger.info("explainer complete | attributed {} verdict(s)", len(attribution))
    return {"metric_attribution": attribution, "final_report": report}


async def visualizer(state: AuditState) -> dict[str, Any]:
    """
    Hotspot attribution visualization (A.9) — last node before END.

    Generates a self-contained interactive HTML report (source + verdict panel
    with confidence and attribution bars) and writes it to
    data/reports/{address}_hotspot.html. Never raises.

    State updates:
        hotspot_visualization → HTML string
    """
    from src.orchestration.visualizer import generate_hotspot_html

    try:
        html_str = generate_hotspot_html(dict(state))
    except Exception as exc:
        logger.warning("visualizer | HTML generation failed (non-fatal): {}", exc)
        return {"hotspot_visualization": None}

    address = (state.get("contract_address", "") or "").strip()
    if address:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out = REPORTS_DIR / f"{address}_hotspot.html"
            out.write_text(html_str)
            logger.info("visualizer | hotspot HTML written → {}", out)
        except Exception as exc:
            logger.warning("visualizer | could not persist hotspot HTML (non-fatal): {}", exc)

    logger.info("visualizer complete | html={} chars", len(html_str))
    return {"hotspot_visualization": html_str}
