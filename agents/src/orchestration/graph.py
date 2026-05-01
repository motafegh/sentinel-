# agents/src/orchestration/graph.py
"""
M5 — SENTINEL audit graph (LangGraph).

Wires ml_assessment → [rag_research ‖ static_analysis → audit_check →] synthesizer
into a stateful, checkpointed LangGraph.

RECALL — LangGraph execution model:
    StateGraph compiles to a Pregel-style message-passing graph.
    Each node is a Python coroutine. LangGraph calls nodes in topological
    order, passing the current state snapshot and merging returned updates.
    MemorySaver checkpoints after EVERY node — if the process crashes
    mid-graph, it can be resumed from the last completed node by
    providing the same thread_id in the config.

RECALL — why conditional routing:
    ML assessment is cheap (~5-15s GPU). RAG + on-chain lookup is slower.
    High-confidence vulnerable contracts get full deep analysis.
    Low-confidence or borderline results get the fast path to synthesizer.
    The routing function (_route_after_ml) is the single place to update
    for Track 3 (multi-label) — see nodes._is_high_risk() docstring.

RECALL — parallel branches (deep path):
    _route_after_ml returns a list ["rag_research", "static_analysis"] for
    high-risk contracts. LangGraph fans out to both nodes in the same
    superstep (parallel execution). audit_check waits for BOTH to complete
    before running — LangGraph's fan-in semantics handle this automatically.

Graph topology:
    START
      │
      ▼
    ml_assessment
      │
      ├─ high risk (deep)──► rag_research ──┐
      │                   ├─ static_analysis ──► audit_check ──► synthesizer
      │
      └─ low risk (fast)──────────────────────────────────────► synthesizer
                                                                     │
                                                                    END

Usage (standalone):
    from src.orchestration.graph import build_graph

    graph = build_graph()

    result = await graph.ainvoke(
        {
            "contract_code":    "<solidity source>",
            "contract_address": "0x...",
        },
        config={"configurable": {"thread_id": "audit-001"}},
    )
    print(result["final_report"])

Usage (with resume from checkpoint):
    # If a previous run with thread_id="audit-001" was interrupted,
    # ainvoke with the same thread_id resumes from the last completed node.
    result = await graph.ainvoke(
        None,  # pass None to resume — state is loaded from checkpointer
        config={"configurable": {"thread_id": "audit-001"}},
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── sys.path — make agents/ importable regardless of cwd ──────────────────
# __file__ is agents/src/orchestration/graph.py → parents[2] = agents/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from src.orchestration.state import AuditState
from src.orchestration.nodes import (
    audit_check,
    ml_assessment,
    rag_research,
    static_analysis,
    synthesizer,
    _is_high_risk,
)


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def _route_after_ml(state: AuditState) -> str | list[str]:
    """
    Decide which path to take after ml_assessment completes.

    Returns:
        list["rag_research", "static_analysis"]
            → fan out to both nodes in parallel (deep path)
        "synthesizer"
            → skip directly to synthesizer (fast path)

    RECALL — parallel fan-out semantics:
        Returning a list from a LangGraph conditional edge function causes
        LangGraph to execute all listed nodes concurrently in the same
        superstep. Both nodes write to independent state keys (rag_results
        and static_findings), so there are no write conflicts.
        audit_check will NOT run until BOTH branches complete — LangGraph's
        fan-in handles the synchronisation automatically.

    NOTE — error fallback:
        If ml_assessment failed (ml_result is empty), route to fast path.
        The synthesizer produces a partial report noting the failure.
    """
    ml_result = state.get("ml_result", {})

    if not ml_result:
        logger.warning("_route_after_ml | ml_result empty — using fast path")
        return "synthesizer"

    if _is_high_risk(ml_result):
        vulns = ml_result.get("vulnerabilities", [])
        top_prob = max((v.get("probability", 0.0) for v in vulns), default=0.0)
        logger.info(
            "_route_after_ml | label={} | top_prob={:.3f} | path=deep (parallel)",
            ml_result.get("label"),
            top_prob,
        )
        return ["rag_research", "static_analysis"]

    logger.info(
        "_route_after_ml | label={} | path=fast",
        ml_result.get("label"),
    )
    return "synthesizer"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(use_checkpointer: bool = True) -> Any:
    """
    Build and compile the SENTINEL audit StateGraph.

    Args:
        use_checkpointer: If True (default), attach a MemorySaver so the
            graph can be resumed after interruption. Set False in unit tests
            to avoid the overhead of checkpointing.

    Returns:
        Compiled LangGraph (CompiledStateGraph) ready for .ainvoke() or
        .invoke() calls.

    RECALL — MemorySaver vs Redis:
        MemorySaver: in-process dict. Fast, zero setup. State is lost on
            process restart. Fine for M5 (single-process, single-machine).
        Redis (langgraph-checkpoint-redis): persistent across restarts.
            Required for M6 production (multi-replica Docker Compose).
            Swap by replacing MemorySaver() with RedisSaver(redis_url=...).

    RECALL — compile() locks the graph topology.
        After compile(), add_node/add_edge are no longer valid.
        To modify the graph, rebuild from scratch with build_graph().
    """
    graph = StateGraph(AuditState)

    # ── Register nodes ──────────────────────────────────────────────────────
    graph.add_node("ml_assessment",  ml_assessment)
    graph.add_node("rag_research",   rag_research)
    graph.add_node("static_analysis", static_analysis)
    graph.add_node("audit_check",    audit_check)
    graph.add_node("synthesizer",    synthesizer)

    # ── Entry point ─────────────────────────────────────────────────────────
    graph.set_entry_point("ml_assessment")

    # ── Conditional routing after ml_assessment ─────────────────────────────
    # _route_after_ml returns either:
    #   ["rag_research", "static_analysis"] → LangGraph fans out to BOTH in parallel
    #   "synthesizer"                       → skip directly to synthesizer
    # No path_map — function returns node name(s) directly.
    graph.add_conditional_edges("ml_assessment", _route_after_ml)

    # ── Deep path fan-in: both parallel branches converge at audit_check ─────
    # LangGraph waits for ALL nodes with edges pointing to audit_check before
    # executing it — so audit_check sees both rag_results and static_findings.
    graph.add_edge("rag_research",    "audit_check")
    graph.add_edge("static_analysis", "audit_check")
    graph.add_edge("audit_check",     "synthesizer")

    # ── Terminal edge ────────────────────────────────────────────────────────
    graph.add_edge("synthesizer", END)

    # ── Compile ─────────────────────────────────────────────────────────────
    checkpointer = MemorySaver() if use_checkpointer else None
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "Audit graph compiled | checkpointer={} | nodes={}",
        "MemorySaver" if use_checkpointer else "None",
        ["ml_assessment", "rag_research", "static_analysis", "audit_check", "synthesizer"],
    )
    return compiled


# ---------------------------------------------------------------------------
# Module-level default instance
# ---------------------------------------------------------------------------
# Importing this module gives you a ready-to-use graph instance.
# For tests, call build_graph(use_checkpointer=False) directly.

audit_graph = build_graph()
