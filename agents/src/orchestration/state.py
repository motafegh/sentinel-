"""
AuditState — the shared state dict that flows through every LangGraph node.

RECALL — LangGraph state design:
    Every node receives the full current state and returns a PARTIAL dict
    containing only the keys it updated. LangGraph merges the partial
    dict back into the state — unchanged keys are preserved automatically.
    Nodes never need to pass through fields they didn't touch.

RECALL — reducers:
    By default LangGraph replaces each key on update.
    `Annotated[list, operator.add]` gives append semantics — used for
    routing_decisions so every node can append without overwriting prior entries.

Field lifecycle:
    contract_code      — set by caller, never mutated
    contract_address   — set by caller, never mutated
    ml_result          — set by ml_assessment node
    routing_decisions  — set by evidence_router, appended by any node (reducer)
    rag_results        — set by rag_research node (deep path only)
    audit_history      — set by audit_check node (deep path only)
    static_findings    — set by static_analysis node (deep path only)
    verdicts           — set by synthesizer (rule-based) or cross_validator (Phase 2)
    confirmations      — set by synthesizer / cross_validator
    contradictions     — set by cross_validator (Phase 2)
    final_report       — set by synthesizer node
    error              — set by any node on failure; synthesizer reads it
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AuditState(TypedDict, total=False):
    """
    Shared mutable state for the SENTINEL audit graph.

    `total=False` means every field is optional — nodes only need to
    return the fields they actually computed. LangGraph's merge semantics
    handle the rest.

    Fields that ARE required by the graph entry point (contract_code,
    contract_address) are enforced in graph.py via input validation,
    not here in the TypedDict, so the rest of the graph can use
    `total=False` cleanly.
    """

    # ── Input (set by caller, never changed by nodes) ──────────────────────
    contract_code:    str           # raw Solidity source to audit
    contract_address: str           # on-chain address (for audit registry lookup)

    # ── ML evidence ─────────────────────────────────────────────────────────
    ml_result: dict[str, Any]
    # Set by ml_assessment.
    # Three-tier schema (2026-05-27):
    #   label:           str        — "safe" | "suspicious" | "confirmed_vulnerable"
    #   probabilities:   dict       — {class: float}  full 10-class vector
    #   confirmed:       list       — [{vulnerability_class, probability, tier="CONFIRMED"}, ...]
    #   suspicious:      list       — [{vulnerability_class, probability, tier="SUSPICIOUS"}, ...]
    #   vulnerabilities: list       — legacy alias for confirmed (backward compat)
    #   tier_thresholds: dict       — {"confirmed": 0.55, "suspicious": 0.25, ...}
    #   thresholds:      list[float] — per-class tuned decision thresholds
    #   truncated:       bool       — True if contract exceeded 512 CodeBERT tokens
    #   windows_used:    int        — token windows scored (>1 for long contracts)
    #   num_nodes:       int        — AST node count
    #   num_edges:       int        — AST edge count

    ml_hotspots: list[dict[str, Any]]
    # Phase 1 — set by ml_assessment (extended) or graph_explain node.
    # Each item: {class, fn_name, lines, node_ids, score}
    # GNN attention + CodeBERT gradient × attention per flagged class.

    # ── Routing trace ────────────────────────────────────────────────────────
    routing_decisions: Annotated[list[str], operator.add]
    # Append-reducer: every node can append without overwriting prior entries.
    # Evidence_router writes the primary routing decisions.
    # Example entries:
    #   "Reentrancy prob=0.872 >= threshold=0.35 → static_analysis+rag_research"
    #   "GasException prob=0.290 < threshold=0.40 → skip"

    # ── Graph explanation (Phase 1) ──────────────────────────────────────────
    graph_explanations: dict[str, Any]
    # {class: {subgraph_json, feature_descriptions, node_ids}}
    # Set by graph_explain node once graph_inspector_server :8013 is built.

    # ── Static analysis ──────────────────────────────────────────────────────
    static_findings: list[dict[str, Any]]
    # Set by static_analysis node (deep path only).
    # Each item: {tool, detector, impact, confidence, description, lines}

    external_call_summary: list[dict[str, Any]]
    # Set by static_analysis node when ExternalBug is flagged.
    # Each item: {caller_contract, caller_function, callee_contract,
    #             callee_function, callee_is_interface}
    # Used by rag_research (enriched query) and synthesizer (LLM prompt).
    # Addresses the GNN structural gap: call_target_typed=1.00 makes typed
    # interface calls look safe — agent layer must compensate with Slither.

    # ── RAG evidence ─────────────────────────────────────────────────────────
    rag_results: list[dict[str, Any]]
    # Set by rag_research node (deep path only).
    # Each item: a ranked RAG chunk with content, metadata, score.

    # ── Economic simulation (Phase 3) ────────────────────────────────────────
    econ_scenarios: list[dict[str, Any]]
    # Set by econ_assessment node (Phase 3).
    # Each item: {name, inputs, outcome, exploitable: bool, description}

    # ── Cross-validation ─────────────────────────────────────────────────────
    verdicts: dict[str, str]
    # {vulnerability_class: "CONFIRMED" | "LIKELY" | "DISPUTED" | "SAFE"}
    # Set by synthesizer (rule-based, Phase 0) or cross_validator (Phase 2).

    confirmations: dict[str, list[str]]
    # {vulnerability_class: [evidence_source, ...]}
    # Tools that corroborate the ML signal for this class.
    # Example: {"Reentrancy": ["ml:0.872", "slither:reentrancy-eth", "rag:0.81"]}

    contradictions: dict[str, list[str]]
    # {vulnerability_class: [description, ...]}
    # Phase 2 — populated by cross_validator when tools disagree.
    # Example: {"IntegerUO": ["ml_flagged", "slither_clean", "rag_nomatch"]}

    # ── On-chain history ─────────────────────────────────────────────────────
    audit_history: list[dict[str, Any]]
    # Set by audit_check node (deep path only).
    # Each item: one historical AuditResult from AuditRegistry.

    # ── Final output ─────────────────────────────────────────────────────────
    final_report: dict[str, Any]
    # Set by synthesizer.
    # Schema (extended Phase 0):
    #   contract_address, overall_label, overall_verdict,
    #   risk_probability, top_vulnerability,
    #   vulnerabilities, vulnerability_verdicts,
    #   threshold, ml_truncated, num_nodes, num_edges,
    #   rag_evidence, audit_history, static_findings,
    #   routing_decisions, recommendation, narrative,
    #   error, path_taken

    narrative: str | None
    # Set by synthesizer when the LLM narrative call succeeds.
    # Structured Markdown with sections: Severity, Vulnerability Summary,
    # Exploit Pattern, Recommended Fix. None when LLM is unavailable.

    error: str | None
    # Set by any node that encounters a non-fatal error.
    # The synthesizer includes it in the final report if set.
    # A node setting error does NOT stop the graph — the synthesizer
    # still runs and produces a partial report.
