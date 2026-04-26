# agents/tests/test_graph_routing.py
"""
Unit tests for the M5 LangGraph orchestration layer.

Coverage:
    - _is_high_risk()         — routing helper (binary threshold, edge cases)
    - _route_after_ml()       — conditional routing function
    - build_graph()           — graph compiles without errors
    - ml_assessment node      — happy path + MCP error + exception
    - rag_research node       — happy path + list vs dict response + failure
    - audit_check node        — happy path + missing address + failure
    - synthesizer node        — deep path / fast path / ml-failure / truncated
    - Full graph (mocked)     — deep path, fast path, ml-failure path

Running:
    cd ~/projects/sentinel/agents
    poetry run pytest tests/test_graph_routing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

# ── Make agents/ importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.graph import build_graph, _route_after_ml
from src.orchestration.nodes import (
    _is_high_risk,
    ml_assessment,
    rag_research,
    audit_check,
    synthesizer,
)
from src.orchestration.state import AuditState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ml_high() -> dict:
    """ML result that triggers the deep path (confidence > 0.70)."""
    return {"label": "vulnerable", "confidence": 0.82, "threshold": 0.50,
            "truncated": False, "num_nodes": 42, "num_edges": 58}


@pytest.fixture
def ml_low() -> dict:
    """ML result that triggers the fast path (confidence ≤ 0.70)."""
    return {"label": "vulnerable", "confidence": 0.65, "threshold": 0.50,
            "truncated": False, "num_nodes": 10, "num_edges": 12}


@pytest.fixture
def ml_safe() -> dict:
    """ML result for a safe contract."""
    return {"label": "safe", "confidence": 0.20, "threshold": 0.50,
            "truncated": False, "num_nodes": 5, "num_edges": 4}


@pytest.fixture
def rag_chunks() -> list:
    return [
        {"chunk_id": "x-1", "content": "Reentrancy in Compound…", "score": 0.91},
        {"chunk_id": "x-2", "content": "Integer overflow…",       "score": 0.73},
    ]


@pytest.fixture
def audit_records() -> list:
    return [
        {"score": 0.73, "label": "vulnerable", "timestamp": 1713200000,
         "timestamp_iso": "2026-04-15T12:00:00+00:00",
         "agent": "0xDead", "verified": True},
    ]


@pytest.fixture
def base_state() -> AuditState:
    return {
        "contract_code": "pragma solidity ^0.8.0;\ncontract T {}",
        "contract_address": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
    }


# ---------------------------------------------------------------------------
# _is_high_risk — routing helper
# ---------------------------------------------------------------------------

class TestIsHighRisk:
    def test_above_threshold_is_high_risk(self, ml_high):
        assert _is_high_risk(ml_high) is True

    def test_at_threshold_is_not_high_risk(self):
        # boundary: exactly 0.70 is NOT > 0.70 → fast path
        assert _is_high_risk({"confidence": 0.70}) is False

    def test_below_threshold_is_not_high_risk(self, ml_low):
        assert _is_high_risk(ml_low) is False

    def test_safe_label_low_confidence(self, ml_safe):
        assert _is_high_risk(ml_safe) is False

    def test_missing_confidence_defaults_to_zero(self):
        assert _is_high_risk({}) is False

    def test_zero_confidence_is_not_high_risk(self):
        assert _is_high_risk({"confidence": 0.0}) is False

    def test_max_confidence_is_high_risk(self):
        assert _is_high_risk({"confidence": 1.0}) is True


# ---------------------------------------------------------------------------
# _route_after_ml — conditional routing
# ---------------------------------------------------------------------------

class TestRouteAfterMl:
    def test_high_confidence_routes_deep(self, ml_high):
        state: AuditState = {"ml_result": ml_high}
        assert _route_after_ml(state) == "deep"

    def test_low_confidence_routes_fast(self, ml_low):
        state: AuditState = {"ml_result": ml_low}
        assert _route_after_ml(state) == "fast"

    def test_empty_ml_result_routes_fast(self):
        state: AuditState = {"ml_result": {}}
        assert _route_after_ml(state) == "fast"

    def test_missing_ml_result_key_routes_fast(self):
        # ml_result not even in state (could happen on cold resume)
        state: AuditState = {}
        assert _route_after_ml(state) == "fast"


# ---------------------------------------------------------------------------
# build_graph — compilation
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_builds_without_checkpointer(self):
        graph = build_graph(use_checkpointer=False)
        assert graph is not None

    def test_builds_with_checkpointer(self):
        graph = build_graph(use_checkpointer=True)
        assert graph is not None

    def test_graph_has_correct_nodes(self):
        graph = build_graph(use_checkpointer=False)
        # CompiledStateGraph exposes node names via .nodes dict
        node_names = set(graph.nodes.keys())
        assert "ml_assessment" in node_names
        assert "rag_research"  in node_names
        assert "audit_check"   in node_names
        assert "synthesizer"   in node_names


# ---------------------------------------------------------------------------
# ml_assessment node
# ---------------------------------------------------------------------------

class TestMlAssessmentNode:
    @pytest.mark.asyncio
    async def test_happy_path(self, base_state, ml_high):
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(return_value=ml_high)):
            result = await ml_assessment(base_state)

        assert result["ml_result"] == ml_high
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_mcp_error_dict_sets_error(self, base_state):
        error_response = {"error": "inference timeout", "detail": "GPU busy"}
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(return_value=error_response)):
            result = await ml_assessment(base_state)

        assert result["ml_result"] == {}
        assert "ml_assessment" in result["error"]
        assert "inference timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_sets_error_and_empty_result(self, base_state):
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(side_effect=ConnectionError("MCP server down"))):
            result = await ml_assessment(base_state)

        assert result["ml_result"] == {}
        assert "ml_assessment" in result["error"]
        assert "MCP server down" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_contract_code_still_calls_mcp(self, ml_high):
        state: AuditState = {"contract_code": "", "contract_address": "0xabc"}
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(return_value=ml_high)):
            result = await ml_assessment(state)
        assert result["ml_result"] == ml_high


# ---------------------------------------------------------------------------
# rag_research node
# ---------------------------------------------------------------------------

class TestRagResearchNode:
    @pytest.mark.asyncio
    async def test_happy_path_list_response(self, base_state, ml_high, rag_chunks):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(return_value=rag_chunks)):
            result = await rag_research(state)

        assert result["rag_results"] == rag_chunks
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_happy_path_dict_with_results_key(self, base_state, ml_high, rag_chunks):
        state = {**base_state, "ml_result": ml_high}
        response = {"results": rag_chunks}
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(return_value=response)):
            result = await rag_research(state)

        assert result["rag_results"] == rag_chunks

    @pytest.mark.asyncio
    async def test_rag_error_dict(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(return_value={"error": "index empty"})):
            result = await rag_research(state)

        assert result["rag_results"] == []
        assert "rag_research" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_empty_results(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(side_effect=RuntimeError("FAISS died"))):
            result = await rag_research(state)

        assert result["rag_results"] == []
        assert "rag_research" in result["error"]


# ---------------------------------------------------------------------------
# audit_check node
# ---------------------------------------------------------------------------

class TestAuditCheckNode:
    @pytest.mark.asyncio
    async def test_happy_path(self, base_state, audit_records):
        state = {**base_state}
        response = {"contract_address": state["contract_address"],
                    "count": 1, "records": audit_records}
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(return_value=response)):
            result = await audit_check(state)

        assert result["audit_history"] == audit_records

    @pytest.mark.asyncio
    async def test_missing_address_skips_lookup(self):
        state: AuditState = {"contract_code": "...", "contract_address": ""}
        result = await audit_check(state)
        assert result["audit_history"] == []
        # MCP should NOT have been called
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_returns_empty_history(self, base_state):
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(side_effect=ConnectionError("Sepolia RPC down"))):
            result = await audit_check(base_state)

        assert result["audit_history"] == []
        assert "audit_check" in result["error"]

    @pytest.mark.asyncio
    async def test_registry_error_dict(self, base_state):
        with patch("src.orchestration.nodes._call_mcp_tool",
                   new=AsyncMock(return_value={"error": "contract not found"})):
            result = await audit_check(base_state)

        assert result["audit_history"] == []
        assert "audit_check" in result["error"]


# ---------------------------------------------------------------------------
# synthesizer node
# ---------------------------------------------------------------------------

class TestSynthesizerNode:
    @pytest.mark.asyncio
    async def test_deep_path_recommendation(self, base_state, ml_high, rag_chunks, audit_records):
        state = {
            **base_state,
            "ml_result":     ml_high,
            "rag_results":   rag_chunks,
            "audit_history": audit_records,
        }
        result = await synthesizer(state)
        report = result["final_report"]

        assert report["overall_label"] == "vulnerable"
        assert report["path_taken"]    == "deep"
        assert len(report["rag_evidence"])  == 2
        assert len(report["audit_history"]) == 1
        assert "HIGH RISK" in report["recommendation"]
        assert report["error"] is None

    @pytest.mark.asyncio
    async def test_fast_path_no_rag(self, base_state, ml_low):
        state = {**base_state, "ml_result": ml_low, "rag_results": [], "audit_history": []}
        result = await synthesizer(state)
        report = result["final_report"]

        assert report["path_taken"] == "fast"
        assert report["rag_evidence"] == []
        assert "MODERATE RISK" in report["recommendation"]

    @pytest.mark.asyncio
    async def test_ml_failure_path(self, base_state):
        state = {
            **base_state,
            "ml_result":  {},
            "rag_results": [],
            "audit_history": [],
            "error": "ml_assessment: Connection refused",
        }
        result = await synthesizer(state)
        report = result["final_report"]

        assert report["overall_label"] == "unknown"
        assert "ML assessment failed" in report["recommendation"]
        assert report["error"] is not None

    @pytest.mark.asyncio
    async def test_truncated_flag_appends_note(self, base_state):
        ml = {"label": "vulnerable", "confidence": 0.82, "threshold": 0.50,
              "truncated": True, "num_nodes": 999, "num_edges": 1200}
        state = {**base_state, "ml_result": ml, "rag_results": [], "audit_history": []}
        result = await synthesizer(state)
        assert "NOTE" in result["final_report"]["recommendation"]
        assert "512" in result["final_report"]["recommendation"]

    @pytest.mark.asyncio
    async def test_safe_label_recommendation(self, base_state, ml_safe):
        state = {**base_state, "ml_result": ml_safe, "rag_results": [], "audit_history": []}
        result = await synthesizer(state)
        assert "LOW RISK" in result["final_report"]["recommendation"]

    @pytest.mark.asyncio
    async def test_report_has_all_required_fields(self, base_state, ml_high, rag_chunks, audit_records):
        state = {**base_state, "ml_result": ml_high,
                 "rag_results": rag_chunks, "audit_history": audit_records}
        result = await synthesizer(state)
        report = result["final_report"]

        required_fields = [
            "contract_address", "overall_label", "confidence", "threshold",
            "ml_truncated", "num_nodes", "num_edges", "rag_evidence",
            "audit_history", "static_findings", "recommendation", "error", "path_taken",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Full graph integration (all nodes mocked)
# ---------------------------------------------------------------------------

_MOCK_ML = {"label": "vulnerable", "confidence": 0.82, "threshold": 0.50,
            "truncated": False, "num_nodes": 42, "num_edges": 58}

_MOCK_RAG = [{"chunk_id": "c-1", "content": "Reentrancy…", "doc_id": "d-1", "score": 0.91}]

_MOCK_AUDIT = {"contract_address": "0xABC", "count": 1,
               "records": [{"score": 0.73, "label": "vulnerable", "timestamp": 1713200000,
                             "timestamp_iso": "2026-04-15T12:00:00+00:00",
                             "agent": "0xDead", "verified": True}]}


async def _mock_mcp(server_url: str, tool_name: str, arguments: dict) -> dict:
    if tool_name == "predict":         return _MOCK_ML
    if tool_name == "search":          return _MOCK_RAG
    if tool_name == "get_audit_history": return _MOCK_AUDIT
    return {"error": f"unexpected tool: {tool_name}"}


class TestFullGraphIntegration:
    @pytest.mark.asyncio
    async def test_deep_path_end_to_end(self):
        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code": "pragma solidity ^0.8.0;\ncontract Vault {}",
            "contract_address": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        }
        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=_mock_mcp):
            result = await graph.ainvoke(initial_state)

        report = result["final_report"]
        assert report["path_taken"]    == "deep"
        assert report["overall_label"] == "vulnerable"
        assert len(report["rag_evidence"]) > 0
        assert report["recommendation"]

    @pytest.mark.asyncio
    async def test_fast_path_end_to_end(self):
        ml_low = {**_MOCK_ML, "confidence": 0.55}
        async def mock_fast(url, tool, args):
            if tool == "predict": return ml_low
            return {"error": "should not be called on fast path"}

        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code": "pragma solidity ^0.8.0;\ncontract Safe {}",
            "contract_address": "0xDEAD",
        }
        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=mock_fast):
            result = await graph.ainvoke(initial_state)

        report = result["final_report"]
        assert report["path_taken"]    == "fast"
        assert report["rag_evidence"]  == []

    @pytest.mark.asyncio
    async def test_ml_failure_still_produces_report(self):
        async def mock_failing(url, tool, args):
            raise ConnectionError("inference server unreachable")

        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code": "contract X {}",
            "contract_address": "0xFACE",
        }
        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=mock_failing):
            result = await graph.ainvoke(initial_state)

        # Graph must not crash — synthesizer still runs
        report = result.get("final_report")
        assert report is not None
        assert "ML assessment failed" in report["recommendation"]
        assert report["error"] is not None
