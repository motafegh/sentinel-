# agents/tests/test_graph_routing.py
"""
Unit tests for the LangGraph orchestration layer (three-tier schema, 2026-05-27).

Coverage:
    - _route_from_evidence_router() — conditional routing function
    - build_graph()                 — graph compiles without errors
    - ml_assessment node            — happy path + MCP error + exception
    - rag_research node             — happy path + list vs dict response + failure
    - audit_check node              — happy path + missing address + failure
    - synthesizer node              — deep path / fast path / ml-failure / truncated
    - Full graph (mocked)           — deep path, fast path, ml-failure path

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

from src.orchestration.graph import build_graph, _route_from_evidence_router
from src.orchestration.nodes import (
    ml_assessment,
    rag_research,
    audit_check,
    synthesizer,
)
from src.orchestration.state import AuditState


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _ml_three_tier(confirmed: list[tuple[str, float]] = (), suspicious: list[tuple[str, float]] = ()) -> dict:
    """Build a three-tier ml_result from (class, prob) pairs."""
    CONF_THR = 0.55
    SUSP_THR = 0.25
    all_items = list(confirmed) + list(suspicious)
    probs = dict(all_items)

    confirmed_list = [
        {"vulnerability_class": c, "probability": p, "tier": "CONFIRMED"}
        for c, p in confirmed
    ]
    suspicious_list = [
        {"vulnerability_class": c, "probability": p, "tier": "SUSPICIOUS"}
        for c, p in suspicious
    ]

    if confirmed_list:     label = "confirmed_vulnerable"
    elif suspicious_list:  label = "suspicious"
    else:                  label = "safe"

    return {
        "label":           label,
        "probabilities":   probs,
        "confirmed":       confirmed_list,
        "suspicious":      suspicious_list,
        "vulnerabilities": [{"vulnerability_class": c, "probability": p} for c, p in confirmed],
        "tier_thresholds": {"confirmed": CONF_THR, "suspicious": SUSP_THR, "noteworthy": 0.10},
        "thresholds":      [0.5] * 10,
        "truncated":       False,
        "windows_used":    1,
        "num_nodes":       42,
        "num_edges":       58,
    }


@pytest.fixture
def ml_high():
    """ML result with high-probability CONFIRMED class — triggers deep path."""
    return _ml_three_tier(
        confirmed=[("Reentrancy", 0.82), ("IntegerUO", 0.61)],
    )


@pytest.fixture
def ml_suspicious():
    """ML result with only SUSPICIOUS classes — triggers deep path via DEEP_THRESHOLDS."""
    return _ml_three_tier(
        suspicious=[("Timestamp", 0.38), ("MishandledException", 0.42)],
    )


@pytest.fixture
def ml_low():
    """ML result where all classes are below DEEP_THRESHOLDS — triggers fast path."""
    return _ml_three_tier(
        suspicious=[("Reentrancy", 0.10)],
    )


@pytest.fixture
def ml_safe():
    """ML result for a completely safe contract."""
    return _ml_three_tier(confirmed=[], suspicious=[])


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
# _route_from_evidence_router — conditional routing
# ---------------------------------------------------------------------------

class TestRouteFromEvidenceRouter:
    def test_high_prob_class_routes_deep(self, ml_high):
        state: AuditState = {"ml_result": ml_high}
        # Reentrancy at 0.82 > DEEP_THRESHOLD[0.35] → should activate tools
        result = _route_from_evidence_router(state)
        assert result != "synthesizer"  # deep path returns tool list

    def test_all_below_threshold_routes_to_synthesizer(self, ml_low):
        # ml_low has Reentrancy at 0.10 — below DEEP_THRESHOLD[0.35]
        state: AuditState = {"ml_result": ml_low}
        result = _route_from_evidence_router(state)
        assert result == "synthesizer"

    def test_empty_ml_result_routes_to_synthesizer(self):
        state: AuditState = {"ml_result": {}}
        result = _route_from_evidence_router(state)
        assert result == "synthesizer"

    def test_missing_ml_result_key_routes_to_synthesizer(self):
        state: AuditState = {}
        result = _route_from_evidence_router(state)
        assert result == "synthesizer"

    def test_suspicious_above_deep_threshold_routes_deep(self, ml_suspicious):
        # Timestamp at 0.38 > DEEP_THRESHOLD[0.35] → deep path
        state: AuditState = {"ml_result": ml_suspicious}
        result = _route_from_evidence_router(state)
        assert result != "synthesizer"


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
        node_names = set(graph.nodes.keys())
        assert "ml_assessment"  in node_names
        assert "evidence_router" in node_names
        assert "rag_research"   in node_names
        assert "audit_check"    in node_names
        assert "synthesizer"    in node_names


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

    @pytest.mark.asyncio
    async def test_uses_confirmed_class_for_rag_query(self, base_state, ml_high, rag_chunks):
        # ml_high has Reentrancy confirmed — query should reference it
        state = {**base_state, "ml_result": ml_high}
        captured = {}
        async def capture(server_url, tool_name, arguments):
            captured.update(arguments)
            return rag_chunks
        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=capture):
            await rag_research(state)
        assert "Reentrancy" in captured.get("query", "")


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

        assert report["overall_label"] == "confirmed_vulnerable"
        assert report["path_taken"]    == "deep"
        assert len(report["rag_evidence"])  == 2
        assert len(report["audit_history"]) == 1
        assert "HIGH RISK" in report["recommendation"]
        assert report["error"] is None

    @pytest.mark.asyncio
    async def test_suspicious_path_recommendation(self, base_state, ml_suspicious):
        state = {**base_state, "ml_result": ml_suspicious, "rag_results": [], "audit_history": []}
        result = await synthesizer(state)
        report = result["final_report"]

        assert report["overall_label"] == "suspicious"
        assert "MODERATE RISK" in report["recommendation"]

    @pytest.mark.asyncio
    async def test_fast_path_no_rag(self, base_state, ml_low):
        state = {**base_state, "ml_result": ml_low, "rag_results": [], "audit_history": []}
        result = await synthesizer(state)
        report = result["final_report"]

        assert report["path_taken"] == "fast"
        assert report["rag_evidence"] == []

    @pytest.mark.asyncio
    async def test_safe_label_recommendation(self, base_state, ml_safe):
        state = {**base_state, "ml_result": ml_safe, "rag_results": [], "audit_history": []}
        result = await synthesizer(state)
        assert "LOW RISK" in result["final_report"]["recommendation"]

    @pytest.mark.asyncio
    async def test_ml_failure_path(self, base_state):
        state = {
            **base_state,
            "ml_result":   {},
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
        ml = {**_ml_three_tier(confirmed=[("IntegerUO", 0.65)]), "truncated": True}
        state = {**base_state, "ml_result": ml, "rag_results": [], "audit_history": []}
        result = await synthesizer(state)
        assert "NOTE" in result["final_report"]["recommendation"]
        assert "512" in result["final_report"]["recommendation"]

    @pytest.mark.asyncio
    async def test_report_has_all_required_fields(self, base_state, ml_high, rag_chunks, audit_records):
        state = {**base_state, "ml_result": ml_high,
                 "rag_results": rag_chunks, "audit_history": audit_records}
        result = await synthesizer(state)
        report = result["final_report"]

        required_fields = [
            "contract_address", "overall_label", "risk_probability",
            "top_vulnerability", "confirmed", "suspicious", "probabilities",
            "tier_thresholds", "threshold", "ml_truncated", "num_nodes", "num_edges",
            "rag_evidence", "audit_history", "static_findings", "recommendation",
            "error", "path_taken",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_confirmed_and_suspicious_in_report(self, base_state, rag_chunks):
        ml = _ml_three_tier(
            confirmed=[("Reentrancy", 0.72)],
            suspicious=[("Timestamp", 0.35)],
        )
        state = {**base_state, "ml_result": ml, "rag_results": rag_chunks, "audit_history": []}
        result = await synthesizer(state)
        report = result["final_report"]

        assert len(report["confirmed"]) == 1
        assert report["confirmed"][0]["vulnerability_class"] == "Reentrancy"
        assert len(report["suspicious"]) == 1
        assert report["suspicious"][0]["vulnerability_class"] == "Timestamp"


# ---------------------------------------------------------------------------
# Full graph integration (all nodes mocked)
# ---------------------------------------------------------------------------

_MOCK_ML = _ml_three_tier(confirmed=[("Reentrancy", 0.82)])
_MOCK_RAG = [{"chunk_id": "c-1", "content": "Reentrancy…", "doc_id": "d-1", "score": 0.91}]
_MOCK_AUDIT = {
    "contract_address": "0xABC",
    "count": 1,
    "records": [{"score": 0.73, "label": "vulnerable", "timestamp": 1713200000,
                 "timestamp_iso": "2026-04-15T12:00:00+00:00",
                 "agent": "0xDead", "verified": True}],
}


async def _mock_mcp(server_url: str, tool_name: str, arguments: dict) -> dict:
    if tool_name == "predict":           return _MOCK_ML
    if tool_name == "search":            return _MOCK_RAG
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
        assert report["overall_label"] == "confirmed_vulnerable"
        assert len(report["rag_evidence"]) > 0
        assert report["recommendation"]

    @pytest.mark.asyncio
    async def test_fast_path_end_to_end(self):
        # ml_fast has all classes below DEEP_THRESHOLDS → fast path
        ml_fast = _ml_three_tier(suspicious=[("Reentrancy", 0.10)])

        async def mock_fast(url, tool, args):
            if tool == "predict": return ml_fast
            return {"error": "should not be called on fast path"}

        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code": "pragma solidity ^0.8.0;\ncontract Safe {}",
            "contract_address": "0xDEAD",
        }
        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=mock_fast):
            result = await graph.ainvoke(initial_state)

        report = result["final_report"]
        assert report["path_taken"]   == "fast"
        assert report["rag_evidence"] == []

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

        report = result.get("final_report")
        assert report is not None
        assert "ML assessment failed" in report["recommendation"]
        assert report["error"] is not None
