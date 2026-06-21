# agents/src/orchestration/graph.py
"""
M5/Phase1 — SENTINEL audit graph (LangGraph).

Phase 1 topology:
    START → ml_assessment → quick_screen → evidence_router → [fan-out] → audit_check → cross_validator → synthesizer → END

Conditional routing after evidence_router:
    Deep path  → rag_research ──────┐
                 static_analysis ────┤→ audit_check → cross_validator → synthesizer
                 graph_explain ──────┘
    Fast path  → synthesizer directly (ML safe AND quick_screen clean)

    Two-signal gate (A3): fast path requires BOTH ML AND quick_screen to agree.
    If quick_screen fires (Slither/Aderyn High/Critical) despite ML safe → deep path.

RECALL — LangGraph execution model:
    StateGraph compiles to a Pregel-style message-passing graph.
    Each node is a Python coroutine. LangGraph calls nodes in topological
    order, passing the current state snapshot and merging returned updates.
    SqliteSaver checkpoints after EVERY node — if the process crashes
    mid-graph, it can be resumed from the last completed node by
    providing the same thread_id in the config.

RECALL — parallel branches (deep path):
    _route_from_evidence_router returns a list when deep analysis is needed.
    LangGraph fans out to all listed nodes in the same superstep.
    audit_check waits for ALL to complete before running — LangGraph's
    fan-in semantics handle this automatically.

RECALL — routing split:
    evidence_router NODE  → logs routing_decisions to AuditState
    _route_from_evidence_router FUNCTION → returns node names for LangGraph edges
    Both call compute_active_tools(ml_result). The node call is for state logging;
    the function call is for graph branching. Slight redundancy is intentional —
    keeps the node pure (no branching) and the function stateless (no state update).

Usage (standalone):
    from src.orchestration.graph import build_graph

    graph = build_graph()
    result = await graph.ainvoke(
        {"contract_code": "<solidity source>", "contract_address": "0x..."},
        config={"configurable": {"thread_id": "audit-001"}},
    )
    print(result["final_report"])

Usage (resume from checkpoint):
    result = await graph.ainvoke(
        None,  # state loaded from SqliteSaver by thread_id
        config={"configurable": {"thread_id": "audit-001"}},
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── sys.path — make agents/ importable regardless of cwd ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, StateGraph
from loguru import logger

from src.orchestration.state import AuditState
from src.orchestration.routing import compute_active_tools
from src.orchestration.timing import timed_node
from src.orchestration.nodes import (
    audit_check,
    consensus_engine,
    cross_validator,
    evidence_router,
    explainer,
    graph_explain,
    ml_assessment,
    quick_screen,
    rag_research,
    reflection,
    static_analysis,
    synthesizer,
    visualizer,
)


# ---------------------------------------------------------------------------
# Routing function (conditional edge — not a node)
# ---------------------------------------------------------------------------

def _route_from_evidence_router(state: AuditState) -> str | list[str]:
    """
    Decide which path to take after evidence_router has logged its decisions.

    Returns:
        list[str]    — node names to fan out to in parallel (deep path)
        "synthesizer" — fast path: skip directly to synthesizer

    Two-signal fast-path gate (A3):
        Fast path requires BOTH signals to agree it is safe:
          1. ML — all class probabilities below DEEP_THRESHOLDS
          2. quick_screen — zero High/Critical Slither or Aderyn hits

        If quick_screen fires but ML did not, we still go deep (with at least
        static_analysis), because two independent tools disagreeing warrants
        human-level scrutiny.

    RECALL — evidence_router node already called compute_active_tools and
    stored results in routing_decisions for auditability. This function calls
    it again (cheap — pure dict lookup) to produce the LangGraph branch target.
    The two calls must be consistent; routing.py is the single source of truth.
    """
    ml_result = state.get("ml_result", {})
    active    = compute_active_tools(ml_result)

    # Check quick_screen escalation signal.
    quick_hits = state.get("quick_screen_hits", {})
    has_screen_hits = bool(quick_hits.get("slither") or quick_hits.get("aderyn"))

    if not active and not has_screen_hits:
        logger.info("_route_from_evidence_router | fast path (ML safe + screen clean)")
        return "synthesizer"

    # quick_screen fired but ML did not → minimal deep path (static_analysis only).
    if not active and has_screen_hits:
        logger.info(
            "_route_from_evidence_router | screen-escalated deep path "
            "(ML safe but quick_screen hit: slither={} aderyn={})",
            quick_hits.get("slither", []),
            quick_hits.get("aderyn",  []),
        )
        active = ["static_analysis"]

    # graph_explain always joins the deep-path fan-out (Phase 1 hotspot analysis).
    deep_nodes = sorted(set(active + ["graph_explain"]))
    logger.info("_route_from_evidence_router | deep path → {}", deep_nodes)
    return deep_nodes


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(use_checkpointer: bool = True) -> Any:
    """
    Build and compile the SENTINEL audit StateGraph.

    Args:
        use_checkpointer: If True (default), attach a SqliteSaver so the
            graph persists state across restarts. Set False in unit tests.

    Returns:
        Compiled LangGraph (CompiledStateGraph) ready for .ainvoke() calls.

    RECALL — SqliteSaver vs MemorySaver:
        MemorySaver: in-process dict. State lost on restart. Was used in M5.
        SqliteSaver: persists to agents/data/checkpoints.db. Resume-from-node
            on crash. Full audit trail inspectable via sqlite3. Required for
            production and for debugging mid-graph state.
        To upgrade to PostgresSaver for M6 multi-replica: swap one import.
    """
    graph = StateGraph(AuditState)

    # ── Register nodes ──────────────────────────────────────────────────────
    # Every node is wrapped with `timed_node` (2026-06-21) so a uniform
    # START/DONE+elapsed log pair is produced for EVERY invocation, in EVERY
    # context this graph runs in (production MCP-driven server,
    # run_real_audit.py, ad-hoc scripts) — not only when a caller happens to
    # add its own ad-hoc timing wrapper. See src/orchestration/timing.py.
    graph.add_node("ml_assessment",   timed_node("ml_assessment", ml_assessment))
    graph.add_node("quick_screen",    timed_node("quick_screen", quick_screen))
    graph.add_node("evidence_router", timed_node("evidence_router", evidence_router))
    graph.add_node("rag_research",    timed_node("rag_research", rag_research))
    graph.add_node("static_analysis", timed_node("static_analysis", static_analysis))
    graph.add_node("graph_explain",   timed_node("graph_explain", graph_explain))
    graph.add_node("audit_check",     timed_node("audit_check", audit_check))
    graph.add_node("consensus_engine", timed_node("consensus_engine", consensus_engine))  # A.6/A.7
    graph.add_node("cross_validator", timed_node("cross_validator", cross_validator))
    graph.add_node("synthesizer",     timed_node("synthesizer", synthesizer))
    graph.add_node("reflection",      timed_node("reflection", reflection))               # A.3
    graph.add_node("explainer",       timed_node("explainer", explainer))                 # A.8
    graph.add_node("visualizer",      timed_node("visualizer", visualizer))                # A.9

    # ── Entry point ─────────────────────────────────────────────────────────
    graph.set_entry_point("ml_assessment")

    # ── ml_assessment → quick_screen → evidence_router (always) ─────────────
    # quick_screen runs on EVERY contract (Tier 0) to catch cases where ML
    # is below all DEEP_THRESHOLDS but static analysis finds High/Critical issues.
    graph.add_edge("ml_assessment", "quick_screen")
    graph.add_edge("quick_screen",  "evidence_router")

    # ── evidence_router → conditional fan-out ───────────────────────────────
    # _route_from_evidence_router returns either:
    #   ["rag_research", "static_analysis"] → parallel fan-out (deep path)
    #   ["static_analysis"]                 → single tool (depends on class)
    #   "synthesizer"                       → fast path
    graph.add_conditional_edges("evidence_router", _route_from_evidence_router)

    # ── Deep path fan-in: all parallel branches converge at audit_check ──────
    graph.add_edge("rag_research",    "audit_check")
    graph.add_edge("static_analysis", "audit_check")
    graph.add_edge("graph_explain",   "audit_check")

    # ── Deep path: audit_check → consensus_engine → cross_validator → synthesizer
    # consensus_engine (A.6/A.7) weights ML/Slither/Aderyn per class and tracks
    # Bayesian confidence; cross_validator then runs the Prosecutor/Defender/Judge
    # debate (A.4). Both fail-soft → rule-based verdicts when the LLM is absent.
    graph.add_edge("audit_check",      "consensus_engine")
    graph.add_edge("consensus_engine", "cross_validator")
    graph.add_edge("cross_validator",  "synthesizer")

    # ── Post-synthesis enrichment: reflection → explainer → visualizer → END ──
    # A.3 self-critique → A.8 metric attribution (+ folds confidence/consensus
    # into final_report) → A.9 interactive hotspot HTML. The fast path also
    # reaches synthesizer, so every run gets the enrichment chain.
    graph.add_edge("synthesizer", "reflection")
    graph.add_edge("reflection",  "explainer")
    graph.add_edge("explainer",   "visualizer")
    graph.add_edge("visualizer",  END)

    # ── Checkpointer ────────────────────────────────────────────────────────
    checkpointer = None
    if use_checkpointer:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = Path(__file__).parents[2] / "data" / "checkpoints.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            # SqliteSaver.from_conn_string() is a context manager in langgraph >= 1.2.
            # Use sqlite3.connect() directly to avoid requiring a `with` block.
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.debug("graph | checkpointer=SqliteSaver | db={}", db_path)
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            logger.warning(
                "graph | langgraph-checkpoint-sqlite not installed — "
                "falling back to MemorySaver (state lost on restart). "
                "Run: pip install langgraph-checkpoint-sqlite"
            )

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "Audit graph compiled | checkpointer={} | nodes={}",
        type(checkpointer).__name__ if checkpointer else "None",
        ["ml_assessment", "quick_screen", "evidence_router", "rag_research",
         "static_analysis", "graph_explain", "audit_check", "consensus_engine",
         "cross_validator", "synthesizer", "reflection", "explainer", "visualizer"],
    )
    return compiled


# ---------------------------------------------------------------------------
# Module-level default instance (LAZY — A.1 graph cleanup)
# ---------------------------------------------------------------------------
# Historically this module ran `audit_graph = build_graph()` at import time.
# That side effect compiled the graph (and opened a SqliteSaver connection)
# on every `import`, even for callers that only wanted `build_graph` or a
# single node — slowing test collection and forcing checkpointer I/O.
#
# PEP 562 module-level __getattr__ defers the build until the FIRST time
# `audit_graph` is actually accessed, then caches it. Existing callers
# (`from src.orchestration.graph import audit_graph`, smoke scripts) keep
# working unchanged; importers that never touch `audit_graph` pay nothing.
# Tests should still call build_graph(use_checkpointer=False) directly.

_audit_graph_singleton: Any = None


def __getattr__(name: str) -> Any:
    """Lazily construct the default `audit_graph` on first attribute access."""
    if name == "audit_graph":
        global _audit_graph_singleton
        if _audit_graph_singleton is None:
            _audit_graph_singleton = build_graph()
        return _audit_graph_singleton
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
