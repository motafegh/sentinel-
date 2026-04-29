"""
AuditState — the shared state dict that flows through every LangGraph node.

RECALL — LangGraph state design:
    Every node receives the full current state and returns a PARTIAL dict
    containing only the keys it updated. LangGraph merges the partial
    dict back into the state — unchanged keys are preserved automatically.
    Nodes never need to pass through fields they didn't touch.

RECALL — reducers:
    By default LangGraph replaces each key on update.
    If a key needs list-append semantics (e.g. a log), annotate it with
    Annotated[list, operator.add]. We keep all fields as replace-semantics
    for now — the synthesizer assembles the final report from snapshots,
    not accumulating streams.

Field lifecycle:
    contract_code     — set by caller, never mutated
    contract_address  — set by caller, never mutated
    ml_result         — set by ml_assessment node
    rag_results       — set by rag_research node (deep path only)
    audit_history     — set by audit_check node (deep path only)
    static_findings   — placeholder (static_analysis node, M6)
    final_report      — set by synthesizer node
    error             — set by any node on failure; synthesizer reads it
"""

from __future__ import annotations

from typing import Any, TypedDict


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

    # ── Node outputs ────────────────────────────────────────────────────────
    ml_result: dict[str, Any]
    # Set by ml_assessment.
    # Track 3 (multi-label) schema — NO "confidence" field (removed in Track 3):
    #   label:           str   — "vulnerable" | "safe"
    #   vulnerabilities: list  — [{vulnerability_class: str, probability: float}, ...]
    #                            empty list when label == "safe"
    #   threshold:       float — per-class decision boundary (default 0.50)
    #   truncated:       bool  — True if contract exceeded 512 CodeBERT tokens
    #   num_nodes:       int   — AST node count
    #   num_edges:       int   — AST edge count
    # Example:
    #   {
    #     "label": "vulnerable",
    #     "vulnerabilities": [
    #       {"vulnerability_class": "Reentrancy", "probability": 0.91},
    #       {"vulnerability_class": "AccessControl", "probability": 0.43},
    #     ],
    #     "threshold": 0.50,
    #     "truncated": False,
    #     "num_nodes": 142,
    #     "num_edges": 187,
    #   }

    rag_results: list[dict[str, Any]]
    # Set by rag_research (deep path only).
    # Each item: a ranked RAG chunk with content, metadata, score.
    # Empty list if rag_research was skipped (fast path).

    audit_history: list[dict[str, Any]]
    # Set by audit_check (deep path only).
    # Each item: one historical AuditResult from AuditRegistry.
    # Empty list if the contract has never been audited on-chain.

    static_findings: dict[str, Any]
    # Reserved for M6 static_analyzer agent (Slither + Mythril).
    # None until M6 is built.

    final_report: dict[str, Any]
    # Set by synthesizer.
    # Track 3 report schema (matches SENTINEL-SPEC §8.1):
    #   contract_address, overall_label, risk_probability, top_vulnerability,
    #   vulnerabilities, threshold, ml_truncated, num_nodes, num_edges,
    #   rag_evidence, audit_history, static_findings, recommendation,
    #   error, path_taken

    error: str | None
    # Set by any node that encounters a non-fatal error.
    # The synthesizer includes it in the final report if set.
    # A node setting error does NOT stop the graph — the synthesizer
    # still runs and produces a partial report.
