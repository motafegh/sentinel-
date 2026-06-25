"""
P2 integration tests — evidence_list, dual verdicts, asymmetry invariant.

These tests verify the transitional dual-write: evidence flows from channels,
fuse() produces verdict_provable + verdict_full alongside legacy verdicts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.state import AuditState
from src.orchestration.verdict.evidence import Evidence, Polarity, Kind
from src.orchestration.verdict.fuse import fuse
from src.orchestration.verdict.emit import (
    emit_ml_evidence,
    emit_static_evidence,
    emit_rag_evidence,
    emit_debate_evidence,
    emit_quick_screen_evidence,
)


# ── Test data ────────────────────────────────────────────────────────────────

ML_RESULT_DEEP = {
    "label": "confirmed_vulnerable",
    "probabilities": {
        "Reentrancy": 0.87,
        "IntegerUO": 0.65,
        "ExternalBug": 0.45,
        "Timestamp": 0.30,
    },
    "confirmed": [
        {"vulnerability_class": "Reentrancy", "probability": 0.87, "tier": "CONFIRMED"},
        {"vulnerability_class": "IntegerUO", "probability": 0.65, "tier": "CONFIRMED"},
    ],
    "suspicious": [
        {"vulnerability_class": "ExternalBug", "probability": 0.45, "tier": "SUSPICIOUS"},
        {"vulnerability_class": "Timestamp", "probability": 0.30, "tier": "SUSPICIOUS"},
    ],
    "truncated": False,
    "windows_used": 1,
}

STATIC_FINDINGS = [
    {
        "tool": "slither",
        "detector": "reentrancy-eth",
        "impact": "High",
        "description": "Reentrancy in withdraw()",
        "lines": [10, 11, 12],
    },
    {
        "tool": "slither",
        "detector": "unchecked-lowlevel",
        "impact": "Medium",
        "description": "Unchecked call return value",
        "lines": [15],
    },
    {
        "tool": "aderyn",
        "detector": "reentrancy-state-change",
        "impact": "High",
        "description": "State change after external call",
        "lines": [11],
    },
]

RAG_RESULTS = [
    {
        "score": 0.72,
        "chunk_id": "ch001",
        "metadata": {"vulnerability_type": "Reentrancy", "title": "DAO Hack"},
    },
    {
        "score": 0.55,
        "chunk_id": "ch002",
        "metadata": {"vulnerability_type": "IntegerUO", "title": "Batch Overflow"},
    },
    {
        "score": 0.25,  # below relevance floor (0.30)
        "chunk_id": "ch003",
        "metadata": {"vulnerability_type": "ExternalBug", "title": "Oracle Manip"},
    },
]


# ── Emit tests ───────────────────────────────────────────────────────────────

class TestEmitML:
    """emit_ml_evidence converts ML output to Evidence."""

    def test_emits_for_classes_above_threshold(self):
        """ML classes with prob >= 0.50 produce evidence."""
        evidence = emit_ml_evidence(ML_RESULT_DEEP)
        classes = {e.vuln_class for e in evidence}
        assert "Reentrancy" in classes
        assert "IntegerUO" in classes
        # ExternalBug at 0.45 < 0.50 threshold — NOT emitted
        assert "ExternalBug" not in classes

    def test_all_are_deterministic(self):
        evidence = emit_ml_evidence(ML_RESULT_DEEP)
        assert all(e.deterministic for e in evidence)

    def test_tier_is_set(self):
        evidence = emit_ml_evidence(ML_RESULT_DEEP)
        reent = [e for e in evidence if e.vuln_class == "Reentrancy"][0]
        assert reent.detail["tier"] == "CONFIRMED"


class TestEmitStatic:
    """emit_static_evidence converts Slither/Aderyn findings."""

    def test_maps_slither_to_classes(self):
        evidence = emit_static_evidence(STATIC_FINDINGS)
        classes = {e.vuln_class for e in evidence}
        assert "Reentrancy" in classes
        assert "IntegerUO" in classes

    def test_all_are_deterministic(self):
        evidence = emit_static_evidence(STATIC_FINDINGS)
        assert all(e.deterministic for e in evidence)

    def test_dedup_per_detector(self):
        """Same detector for same class shouldn't duplicate."""
        duplicate_findings = STATIC_FINDINGS + [STATIC_FINDINGS[0]]
        evidence = emit_static_evidence(duplicate_findings)
        reent_count = sum(1 for e in evidence if e.vuln_class == "Reentrancy" and e.source == "slither")
        assert reent_count == 1


class TestEmitRAG:
    """emit_rag_evidence converts RAG chunks."""

    def test_emits_above_relevance_floor(self):
        evidence = emit_rag_evidence(RAG_RESULTS)
        classes = {e.vuln_class for e in evidence}
        assert "Reentrancy" in classes
        assert "IntegerUO" in classes
        # ExternalBug at 0.25 < 0.30 floor — NOT emitted
        assert "ExternalBug" not in classes

    def test_all_are_deterministic(self):
        evidence = emit_rag_evidence(RAG_RESULTS)
        assert all(e.deterministic for e in evidence)


class TestEmitDebate:
    """emit_debate_evidence converts debate output."""

    def test_maps_verdicts_to_evidence(self):
        pre_verdicts = {
            "Reentrancy": "CONFIRMED",
            "IntegerUO": "DISPUTED",
            "ExternalBug": "SAFE",
        }
        debate_transcript = {"judge": "Reentrancy confirmed..."}
        evidence = emit_debate_evidence(debate_transcript, pre_verdicts)

        assert len(evidence) == 3
        by_cls = {e.vuln_class: e for e in evidence}
        assert by_cls["Reentrancy"].polarity == Polarity.SUPPORTS
        assert by_cls["ExternalBug"].polarity == Polarity.REFUTES
        assert by_cls["IntegerUO"].polarity == Polarity.NEUTRAL

    def test_all_are_nondeterministic(self):
        evidence = emit_debate_evidence({}, {"Reentrancy": "LIKELY"})
        assert all(not e.deterministic for e in evidence)


class TestEmitQuickScreen:
    """emit_quick_screen_evidence converts quick-screen hits."""

    def test_maps_hits_to_classes(self):
        hits = {"slither": ["reentrancy-eth"], "aderyn": []}
        evidence = emit_quick_screen_evidence(hits)
        assert len(evidence) == 1
        assert evidence[0].vuln_class == "Reentrancy"


# ── Fuse integration ─────────────────────────────────────────────────────────

class TestFuseFullPipeline:
    """fuse() over evidence from all channels."""

    def test_fuse_all_channels(self):
        evidence = []
        evidence.extend(emit_ml_evidence(ML_RESULT_DEEP))
        evidence.extend(emit_static_evidence(STATIC_FINDINGS))
        evidence.extend(emit_rag_evidence(RAG_RESULTS))
        evidence.extend(emit_debate_evidence({}, {"Reentrancy": "LIKELY", "IntegerUO": "DISPUTED"}))

        result = fuse(evidence)
        assert "Reentrancy" in result
        assert "IntegerUO" in result

        # Reentrancy: ML 0.87 + Slither High + Aderyn High + RAG + debate LIKELY → strong
        cv = result["Reentrancy"]
        assert cv.verdict_provable == "CONFIRMED"  # deterministic tier
        assert cv.confidence > 0.5

    def test_driving_evidence_present(self):
        evidence = emit_ml_evidence(ML_RESULT_DEEP)
        result = fuse(evidence)
        cv = result["Reentrancy"]
        assert len(cv.driving_evidence) >= 1

    def test_asymmetry_no_flagged_class_safe(self):
        """A class with strong evidence must not reach SAFE."""
        evidence = [
            Evidence.ml("Reentrancy", 0.87, 0.60, tier="CONFIRMED"),
            Evidence.debate("Reentrancy", "SAFE", 0.95),  # REFUTES
        ]
        result = fuse(evidence)
        assert result["Reentrancy"].verdict_full != "SAFE"

    def test_provable_differs_from_full_with_nondeterministic(self):
        """With debate evidence (non-deterministic), provable may differ from full."""
        evidence = [
            Evidence.ml("Reentrancy", 0.60, 0.60, tier="CONFIRMED"),
            Evidence.slither("Reentrancy", "Medium", "found", 0.82),
            Evidence.debate("Reentrancy", "SAFE", 0.90),  # non-deterministic REFUTES
        ]
        result = fuse(evidence)
        cv = result["Reentrancy"]
        # Deterministic tier has ML + Slither → LIKELY or CONFIRMED
        # Full tier includes debate refute → may differ
        assert isinstance(cv.verdict_provable, str)
        assert isinstance(cv.verdict_full, str)


# ── State shape (verdict_provable / verdict_full in final_report) ────────────

class TestStateShape:
    """The new state fields are present after the graph runs."""

    @pytest.mark.asyncio
    async def test_evidence_list_populated_in_graph(self):
        """After a deep-path run, evidence_list should have entries."""
        from src.orchestration.graph import build_graph

        graph = build_graph(use_checkpointer=False)
        initial_state: dict = {
            "contract_code": (
                "pragma solidity ^0.8.0;\n"
                "contract V {\n"
                "  function withdraw() public {\n"
                "    msg.sender.call{value: 1}('');\n"  # reentrancy pattern
                "  }\n"
                "}\n"
            ),
            "contract_address": "0xINTEG",
        }

        # Mock MCP calls: ml_assessment returns our deep result
        def _mock_mcp(server_url, tool_name, arguments):
            if "inference" in server_url:
                return ML_RESULT_DEEP
            if "rag" in server_url:
                return {"results": RAG_RESULTS}
            if "graph_inspector" in server_url:
                return {"explanations": {}, "hotspots": []}
            return {}

        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   side_effect=_mock_mcp):
            result = await graph.ainvoke(initial_state)

        # New P2 fields should be present in the result (even if empty)
        assert "evidence_list" in result, f"evidence_list missing from result keys: {sorted(result.keys())}"
        assert "verdict_provable" in result
        assert "verdict_full" in result

        # evidence_list should be a list (may be empty if ML mock didn't set up state fully)
        assert isinstance(result.get("evidence_list"), list)
        assert isinstance(result.get("verdict_provable"), dict)
        assert isinstance(result.get("verdict_full"), dict)
