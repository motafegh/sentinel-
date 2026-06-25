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
    audit_check,
    cross_validator,
    graph_explain,
    ml_assessment,
    quick_screen,
    rag_research,
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
        result = _route_from_evidence_router(state)
        assert result != "synthesizer"

    def test_deep_path_always_includes_graph_explain(self, ml_high):
        state: AuditState = {"ml_result": ml_high}
        result = _route_from_evidence_router(state)
        assert isinstance(result, list)
        assert "graph_explain" in result

    def test_all_below_threshold_routes_to_synthesizer(self, ml_low):
        # ml_low has Reentrancy at 0.10 — below DEEP_THRESHOLD[0.35], no screen hits
        state: AuditState = {"ml_result": ml_low, "quick_screen_hits": {}}
        result = _route_from_evidence_router(state)
        assert result == "synthesizer"

    def test_empty_ml_result_routes_to_synthesizer(self):
        state: AuditState = {"ml_result": {}, "quick_screen_hits": {}}
        result = _route_from_evidence_router(state)
        assert result == "synthesizer"

    def test_missing_ml_result_key_routes_to_synthesizer(self):
        state: AuditState = {}
        result = _route_from_evidence_router(state)
        assert result == "synthesizer"

    def test_quick_screen_escalates_ml_safe_to_deep(self, ml_low):
        # ML says safe but quick_screen found a High-impact Slither hit → deep path
        state: AuditState = {
            "ml_result": ml_low,
            "quick_screen_hits": {"slither": ["reentrancy-eth"], "aderyn": []},
        }
        result = _route_from_evidence_router(state)
        assert result != "synthesizer"
        assert isinstance(result, list)
        assert "static_analysis" in result

    def test_quick_screen_escalated_path_includes_graph_explain(self, ml_low):
        state: AuditState = {
            "ml_result": ml_low,
            "quick_screen_hits": {"slither": ["arbitrary-send-eth"], "aderyn": []},
        }
        result = _route_from_evidence_router(state)
        assert isinstance(result, list)
        assert "graph_explain" in result

    def test_aderyn_hit_alone_escalates(self, ml_low):
        state: AuditState = {
            "ml_result": ml_low,
            "quick_screen_hits": {"slither": [], "aderyn": ["H-2"]},
        }
        result = _route_from_evidence_router(state)
        assert result != "synthesizer"

    def test_empty_quick_screen_plus_ml_safe_is_fast_path(self, ml_low):
        state: AuditState = {
            "ml_result": ml_low,
            "quick_screen_hits": {"slither": [], "aderyn": []},
        }
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
        for name in ("ml_assessment", "quick_screen", "evidence_router", "rag_research",
                     "static_analysis", "graph_explain", "audit_check",
                     "cross_validator", "synthesizer"):
            assert name in node_names, f"missing node: {name}"


# ---------------------------------------------------------------------------
# ml_assessment node
# ---------------------------------------------------------------------------

class TestMlAssessmentNode:
    @pytest.mark.asyncio
    async def test_happy_path(self, base_state, ml_high):
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(return_value=ml_high)):
            result = await ml_assessment(base_state)

        assert result["ml_result"] == ml_high
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_mcp_error_dict_sets_error(self, base_state):
        error_response = {"error": "inference timeout", "detail": "GPU busy"}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(return_value=error_response)):
            result = await ml_assessment(base_state)

        assert result["ml_result"] == {}
        assert "ml_assessment" in result["error"]
        assert "inference timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_sets_error_and_empty_result(self, base_state):
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(side_effect=ConnectionError("MCP server down"))):
            result = await ml_assessment(base_state)

        assert result["ml_result"] == {}
        assert "ml_assessment" in result["error"]
        assert "MCP server down" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_contract_code_still_calls_mcp(self, ml_high):
        state: AuditState = {"contract_code": "", "contract_address": "0xabc"}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
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
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(return_value=rag_chunks)):
            result = await rag_research(state)

        assert result["rag_results"] == rag_chunks
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_happy_path_dict_with_results_key(self, base_state, ml_high, rag_chunks):
        state = {**base_state, "ml_result": ml_high}
        response = {"results": rag_chunks}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(return_value=response)):
            result = await rag_research(state)

        assert result["rag_results"] == rag_chunks

    @pytest.mark.asyncio
    async def test_rag_error_dict(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(return_value={"error": "index empty"})):
            result = await rag_research(state)

        assert result["rag_results"] == []
        assert "rag_research" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_empty_results(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
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
        with patch("src.orchestration.nodes._helpers._call_mcp_tool", side_effect=capture):
            await rag_research(state)
        assert "Reentrancy" in captured.get("query", "")


# ---------------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_externalbug_uses_call_summary_for_rag_query(self, base_state, rag_chunks):
        # When ExternalBug is the top class AND external_call_summary is present,
        # the RAG query should reference the callee contracts/functions, not the
        # generic contract snippet.
        ml = _ml_three_tier(confirmed=[("ExternalBug", 0.71)])
        ext_calls = [
            {"caller_contract": "Vault", "caller_function": "getPrice",
             "callee_contract": "ChainlinkOracle", "callee_function": "latestRoundData",
             "callee_is_interface": True},
        ]
        state = {**base_state, "ml_result": ml, "external_call_summary": ext_calls}
        captured = {}
        async def capture(server_url, tool_name, arguments):
            captured.update(arguments)
            return rag_chunks
        with patch("src.orchestration.nodes._helpers._call_mcp_tool", side_effect=capture):
            await rag_research(state)
        # Query must reference ExternalBug and the oracle contract name
        assert "ExternalBug" in captured.get("query", "")
        assert "ChainlinkOracle" in captured.get("query", "")

    @pytest.mark.asyncio
    async def test_non_externalbug_uses_generic_query(self, base_state, rag_chunks):
        # Non-ExternalBug class should use the generic snippet-based query even
        # if external_call_summary happens to be populated.
        ml = _ml_three_tier(confirmed=[("Reentrancy", 0.80)])
        ext_calls = [
            {"caller_contract": "X", "caller_function": "f",
             "callee_contract": "Y", "callee_function": "g",
             "callee_is_interface": False},
        ]
        state = {**base_state, "ml_result": ml, "external_call_summary": ext_calls}
        captured = {}
        async def capture(server_url, tool_name, arguments):
            captured.update(arguments)
            return rag_chunks
        with patch("src.orchestration.nodes._helpers._call_mcp_tool", side_effect=capture):
            await rag_research(state)
        assert "Reentrancy" in captured.get("query", "")
        assert "ChainlinkOracle" not in captured.get("query", "")


# audit_check node
# ---------------------------------------------------------------------------

class TestAuditCheckNode:
    @pytest.mark.asyncio
    async def test_happy_path(self, base_state, audit_records):
        state = {**base_state}
        response = {"contract_address": state["contract_address"],
                    "count": 1, "records": audit_records}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
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
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(side_effect=ConnectionError("Sepolia RPC down"))):
            result = await audit_check(base_state)

        assert result["audit_history"] == []
        assert "audit_check" in result["error"]

    @pytest.mark.asyncio
    async def test_registry_error_dict(self, base_state):
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
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

    @pytest.mark.asyncio
    async def test_narrative_prompt_grounds_each_class_with_its_verdict(
        self, base_state, ml_high, rag_chunks, audit_records, monkeypatch,
    ):
        # Fix 2 (2026-06-21, "narrative hallucination" incident — see
        # docs/changes/2026-06-21-agents-manual-verification-real-bugs-found.md):
        # the narrative prompt used to list every ML-flagged class with NO
        # verdict attached, so the model had no signal a class had been
        # cleared — observed live: it wrote about a "Reentrancy risk" on a
        # contract whose Reentrancy verdict was SAFE. Verify the prompt sent
        # to the LLM now attaches each class's verdict, and the system prompt
        # explicitly instructs the model to only discuss CONFIRMED/LIKELY
        # classes and to treat RAG content as general background.
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "## Severity\nHIGH\n## Vulnerability Summary\n...\n## Exploit Pattern\n...\n## Recommended Fix\n..."
        mock_llm.invoke.return_value = mock_resp

        state = {
            **base_state,
            "ml_result":     ml_high,
            "rag_results":   rag_chunks,
            "audit_history": audit_records,
        }
        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            await synthesizer(state)

        sent_messages = mock_llm.invoke.call_args[0][0]
        system_text = sent_messages[0].content
        user_text   = sent_messages[1].content

        # System prompt grounds the model against hallucinating ungated classes.
        assert "CONFIRMED or LIKELY" in system_text
        assert "do NOT describe it as" in system_text or "do NOT introduce" in system_text

        # User prompt attaches "→ verdict: ..." to every listed class.
        assert "→ verdict:" in user_text

        # RAG section is explicitly labeled as general background, not
        # site-specific evidence.
        assert "general historical reference" in user_text or "NOT necessarily about this contract" in user_text

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
        with patch("src.orchestration.nodes._helpers._call_mcp_tool", side_effect=_mock_mcp):
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
        with patch("src.orchestration.nodes._helpers._call_mcp_tool", side_effect=mock_fast):
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
        with patch("src.orchestration.nodes._helpers._call_mcp_tool", side_effect=mock_failing):
            result = await graph.ainvoke(initial_state)

        report = result.get("final_report")
        assert report is not None
        assert "ML assessment failed" in report["recommendation"]
        assert report["error"] is not None


# ---------------------------------------------------------------------------
# graph_explain node
# ---------------------------------------------------------------------------

class TestGraphExplainNode:
    _MOCK_INSPECTOR_RESPONSE = {
        "hotspots": [
            {
                "contract": "Vault",
                "function": "withdraw",
                "lines": [42, 43, 44],
                "vulnerability_classes": ["Reentrancy", "ExternalBug"],
                "score": 4.2,
                "signals": ["reentrancy-eth(High)", "external_calls=2"],
            },
            {
                "contract": "Vault",
                "function": "deposit",
                "lines": [28, 29],
                "vulnerability_classes": ["Reentrancy"],
                "score": 1.5,
                "signals": ["state_writes=3"],
            },
        ],
        "graph_stats": {"num_contracts": 1, "num_functions": 4, "has_interfaces": False, "num_hotspots": 2},
        "analysis_mode": "slither",
    }

    @pytest.mark.asyncio
    async def test_happy_path_returns_hotspots(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(return_value=self._MOCK_INSPECTOR_RESPONSE)):
            result = await graph_explain(state)

        assert "ml_hotspots" in result
        assert len(result["ml_hotspots"]) == 2
        assert result["ml_hotspots"][0]["fn_name"] == "withdraw"
        assert result["ml_hotspots"][0]["score"] == 4.2

    @pytest.mark.asyncio
    async def test_returns_graph_explanations(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(return_value=self._MOCK_INSPECTOR_RESPONSE)):
            result = await graph_explain(state)

        ge = result["graph_explanations"]
        assert ge["analysis_mode"] == "slither"
        assert ge["graph_stats"]["num_contracts"] == 1
        assert "Reentrancy" in ge["hotspots_by_class"]

    @pytest.mark.asyncio
    async def test_inspector_error_returns_empty(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(return_value={"error": "slither failed"})):
            result = await graph_explain(state)

        assert result["ml_hotspots"] == []
        assert result["graph_explanations"] == {}

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        with patch("src.orchestration.nodes._helpers._call_mcp_tool",
                   new=AsyncMock(side_effect=ConnectionError("inspector down"))):
            result = await graph_explain(state)

        assert result["ml_hotspots"] == []
        assert result["graph_explanations"] == {}

    @pytest.mark.asyncio
    async def test_empty_contract_code_skips(self, ml_high):
        state: AuditState = {"contract_code": "", "ml_result": ml_high}
        result = await graph_explain(state)
        assert result["ml_hotspots"] == []

    @pytest.mark.asyncio
    async def test_passes_flagged_classes_to_inspector(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high}
        captured_args: list = []

        async def capture(server_url, tool_name, arguments):
            captured_args.append(arguments)
            return self._MOCK_INSPECTOR_RESPONSE

        with patch("src.orchestration.nodes._helpers._call_mcp_tool", side_effect=capture):
            await graph_explain(state)

        assert len(captured_args) == 1
        fc = captured_args[0]["flagged_classes"]
        assert "Reentrancy" in fc
        assert "IntegerUO" in fc


# ---------------------------------------------------------------------------
# cross_validator node
# ---------------------------------------------------------------------------

class TestCrossValidatorNode:
    @pytest.fixture(autouse=True)
    def _enable_llm_single_pass(self, monkeypatch):
        # These tests exercise the LLM verdict-parsing path with a mocked LLM,
        # so re-enable LLM (conftest disables it globally) and force single-pass
        # (debate off) for deterministic one-call parsing assertions.
        monkeypatch.delenv("AGENTS_DISABLE_LLM", raising=False)
        monkeypatch.setenv("DEBATE_MODE", "off")
        monkeypatch.setenv("CROSS_VALIDATOR_LLM_MODEL", "strong")

    @pytest.mark.asyncio
    async def test_no_flagged_classes_returns_empty(self, base_state, ml_safe):
        state = {**base_state, "ml_result": ml_safe}
        result = await cross_validator(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high, "static_findings": [], "rag_results": []}
        with patch("src.llm.client.get_strong_llm",
                   side_effect=ImportError("LLM not configured")):
            result = await cross_validator(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_happy_path_returns_evidence_list(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high, "static_findings": [], "rag_results": [], "audit_history": []}
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"Reentrancy": "CONFIRMED", "IntegerUO": "LIKELY"}'
        mock_llm.invoke.return_value = mock_response

        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            result = await cross_validator(state)

        assert "evidence_list" in result
        evidence_list = result["evidence_list"]
        assert len(evidence_list) == 2
        reent_ev = [e for e in evidence_list if e.vuln_class == "Reentrancy"]
        intuo_ev = [e for e in evidence_list if e.vuln_class == "IntegerUO"]
        assert len(reent_ev) == 1
        assert len(intuo_ev) == 1
        assert reent_ev[0].detail.get("debate_verdict") == "CONFIRMED"
        assert intuo_ev[0].detail.get("debate_verdict") == "LIKELY"

    @pytest.mark.asyncio
    async def test_invalid_verdict_coerced_to_disputed(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high, "static_findings": [], "rag_results": [], "audit_history": []}
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"Reentrancy": "MAYBE", "IntegerUO": "CONFIRMED"}'
        mock_llm.invoke.return_value = mock_response

        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            result = await cross_validator(state)

        assert "evidence_list" in result
        evidence_list = result["evidence_list"]
        reent_ev = [e for e in evidence_list if e.vuln_class == "Reentrancy"]
        intuo_ev = [e for e in evidence_list if e.vuln_class == "IntegerUO"]
        assert len(reent_ev) == 1
        assert len(intuo_ev) == 1
        assert reent_ev[0].detail.get("debate_verdict") == "DISPUTED"
        assert intuo_ev[0].detail.get("debate_verdict") == "CONFIRMED"

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high, "static_findings": [], "rag_results": [], "audit_history": []}
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"Reentrancy": "CONFIRMED"}\n```'
        mock_llm.invoke.return_value = mock_response

        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            result = await cross_validator(state)

        assert "evidence_list" in result
        evidence_list = result["evidence_list"]
        reent_ev = [e for e in evidence_list if e.vuln_class == "Reentrancy"]
        assert len(reent_ev) == 1
        assert reent_ev[0].detail.get("debate_verdict") == "CONFIRMED"

    @pytest.mark.asyncio
    async def test_caps_classes_sent_to_llm(self, base_state, monkeypatch):
        # Real-audit finding (2026-06-21): ambiguous contracts can flag most of
        # the 10 classes as "suspicious". Adjudicating all of them risked the
        # FAST model overrunning the timeout in the 3-call debate. Verify the
        # cap keeps only the top-N most probable classes in the LLM prompt.
        monkeypatch.setenv("CROSS_VALIDATOR_MAX_CLASSES", "3")
        many_suspicious = [
            {"vulnerability_class": f"Class{i}", "probability": 0.30 + i * 0.01, "tier": "SUSPICIOUS"}
            for i in range(9)
        ]
        state = {
            **base_state,
            "ml_result": {"confirmed": [], "suspicious": many_suspicious},
            "static_findings": [], "rag_results": [], "audit_history": [],
        }
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "{}"
        mock_llm.invoke.return_value = mock_response

        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            await cross_validator(state)

        sent_prompt = mock_llm.invoke.call_args[0][0][1].content
        # Only the top-3 by probability (Class6, Class7, Class8) should appear.
        assert "Class8" in sent_prompt and "Class7" in sent_prompt and "Class6" in sent_prompt
        assert "Class0" not in sent_prompt

    @pytest.mark.asyncio
    async def test_llm_json_parse_error_returns_empty(self, base_state, ml_high):
        state = {**base_state, "ml_result": ml_high, "static_findings": [], "rag_results": [], "audit_history": []}
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I cannot determine the verdicts."
        mock_llm.invoke.return_value = mock_response

        with patch("src.llm.client.get_strong_llm", return_value=mock_llm):
            result = await cross_validator(state)

        assert result == {}


# ---------------------------------------------------------------------------
# Synthesizer uses pre-computed verdicts from cross_validator
# ---------------------------------------------------------------------------

class TestSynthesizerUsesPreComputedVerdicts:
    @pytest.mark.asyncio
    async def test_synthesizer_uses_cross_validator_verdicts(self, base_state, ml_high):
        from src.orchestration.verdict.evidence import Evidence
        evidence_list = [
            Evidence.ml("Reentrancy", 0.82, 0.39),
            Evidence.debate("Reentrancy", "CONFIRMED", 0.85),
            Evidence.ml("IntegerUO", 0.61, 0.31),
            Evidence.debate("IntegerUO", "LIKELY", 0.65),
        ]
        state = {
            **base_state,
            "ml_result":       ml_high,
            "rag_results":     [],
            "audit_history":   [],
            "static_findings": [],
            "routing_decisions": [],
            "evidence_list":   evidence_list,
        }
        with patch("src.llm.client.get_strong_llm",
                   side_effect=ImportError("no LLM")):
            result = await synthesizer(state)

        report = result["final_report"]
        vv = {v["vulnerability_class"]: v["verdict"] for v in report["vulnerability_verdicts"]}
        assert vv["Reentrancy"] == "CONFIRMED"
        assert vv["IntegerUO"]  == "LIKELY"

    @pytest.mark.asyncio
    async def test_consensus_evidece_respected_when_debate_fails(
        self, base_state, ml_high,
    ):
        # Fix 1 (2026-06-21, "verdict fallback gap") — Shape A (T2.7):
        # when cross_validator's debate fails, synthesizer uses fuse() over
        # the accumulated evidence_list which includes consensus evidence
        # (emitted by consensus_engine node). The consensus verdict's
        # contribution is incorporated naturally via fuse() aggregation.
        from src.orchestration.verdict.evidence import Evidence, Polarity, Kind

        evidence_list = [
            Evidence.ml("Reentrancy", 0.82, 0.39),
            Evidence.ml("IntegerUO", 0.61, 0.31),
            Evidence(
                source="consensus", vuln_class="Reentrancy",
                polarity=Polarity.NEUTRAL, strength=0.19, reliability=0.85,
                kind=Kind.STATISTICAL, deterministic=True,
                detail={"consensus_verdict": "SAFE", "consensus_confidence": 0.19},
            ),
            Evidence(
                source="consensus", vuln_class="IntegerUO",
                polarity=Polarity.SUPPORTS, strength=0.55, reliability=0.85,
                kind=Kind.STATISTICAL, deterministic=True,
                detail={"consensus_verdict": "LIKELY", "consensus_confidence": 0.55},
            ),
        ]
        state = {
            **base_state,
            "ml_result":         ml_high,
            "rag_results":       [],
            "audit_history":     [],
            "static_findings":   [],
            "routing_decisions": [],
            "evidence_list":     evidence_list,
            "consensus_verdict": {
                "Reentrancy": {"verdict": "SAFE", "confidence": 0.19},
                "IntegerUO":  {"verdict": "LIKELY", "confidence": 0.55},
            },
        }
        with patch("src.llm.client.get_strong_llm", side_effect=ImportError("no LLM")):
            result = await synthesizer(state)

        vv = {v["vulnerability_class"]: v["verdict"] for v in result["final_report"]["vulnerability_verdicts"]}
        assert vv["IntegerUO"]  == "LIKELY"
        assert vv["Reentrancy"] in ("DISPUTED", "SAFE")

    @pytest.mark.asyncio
    async def test_compute_verdict_is_last_resort_when_consensus_engine_didnt_score_class(
        self, base_state,
    ):
        # A class consensus_engine never voted on (e.g. weak SUSPICIOUS-tier
        # noise below its scoring bar) still needs an answer — compute_verdict()
        # remains the correct final fallback for THOSE classes only.
        ml_result = {
            "confirmed":  [],
            "suspicious": [{"vulnerability_class": "Timestamp", "probability": 0.30, "tier": "SUSPICIOUS"}],
        }
        state = {
            **base_state,
            "ml_result":         ml_result,
            "rag_results":       [],
            "audit_history":     [],
            "static_findings":   [],
            "routing_decisions": [],
            "consensus_verdict": {},  # did not score Timestamp at all
        }
        with patch("src.llm.client.get_strong_llm", side_effect=ImportError("no LLM")):
            result = await synthesizer(state)

        vv = {v["vulnerability_class"]: v["verdict"] for v in result["final_report"]["vulnerability_verdicts"]}
        # compute_verdict()'s rule for prob < 0.50: falls through its own logic
        # (not asserting the exact label here, only that it didn't crash and
        # produced SOME verdict via the compute_verdict() path).
        assert "Timestamp" in vv


# ---------------------------------------------------------------------------
# quick_screen node
# ---------------------------------------------------------------------------

class TestQuickScreenNode:
    """
    Tests for the quick_screen (Tier 0) node.

    All Slither/subprocess calls are mocked — these are unit tests.
    The quick_screen node must be non-fatal: every test branch must return
    {"quick_screen_hits": {"slither": [...], "aderyn": [...]}} even on errors.
    """

    @pytest.mark.asyncio
    async def test_empty_contract_code_returns_empty_hits(self):
        state: AuditState = {"contract_code": "", "contract_address": "0xABC"}
        result = await quick_screen(state)
        assert result["quick_screen_hits"] == {"slither": [], "aderyn": []}

    @pytest.mark.asyncio
    async def test_slither_not_installed_returns_empty_hits(self, base_state):
        # Simulate ImportError for slither.
        with patch.dict("sys.modules", {"slither": None}):
            result = await quick_screen(base_state)
        assert "quick_screen_hits" in result
        assert isinstance(result["quick_screen_hits"]["slither"], list)
        assert isinstance(result["quick_screen_hits"]["aderyn"],  list)

    @pytest.mark.asyncio
    async def test_slither_exception_is_non_fatal(self, base_state):
        # Slither raises an unexpected exception → node returns empty hits, no raise.
        with patch("src.orchestration.nodes.quick_screen.tempfile.NamedTemporaryFile",
                   side_effect=OSError("disk full")):
            result = await quick_screen(base_state)
        assert "quick_screen_hits" in result
        assert result["quick_screen_hits"]["slither"] == []

    @pytest.mark.asyncio
    async def test_slither_high_impact_finding_captured(self, base_state):
        """
        When Slither returns a High-impact finding for a screen detector,
        the detector name must appear in quick_screen_hits["slither"].
        """
        from unittest.mock import MagicMock

        mock_finding = {
            "check": "reentrancy-eth",
            "impact": "High",
            "confidence": "High",
            "description": "Reentrancy in Vault.withdraw",
            "elements": [],
        }

        # Build a fake Slither instance with enough API surface for quick_screen.
        mock_detector = MagicMock()
        mock_detector.ARGUMENT = "reentrancy-eth"

        mock_sl = MagicMock()
        mock_sl._detectors = [mock_detector]
        mock_sl.run_detectors.return_value = [[mock_finding]]

        mock_slither_cls = MagicMock(return_value=mock_sl)

        # Patch slither.Slither at the import location used inside quick_screen.
        import slither as _slither_module
        with patch.object(_slither_module, "Slither", mock_slither_cls), \
             patch("subprocess.run", side_effect=FileNotFoundError("aderyn")):
            result = await quick_screen(base_state)

        hits = result["quick_screen_hits"]
        assert "reentrancy-eth" in hits["slither"]

    @pytest.mark.asyncio
    async def test_aderyn_not_installed_is_non_fatal(self, base_state):
        # FileNotFoundError from aderyn subprocess → ignored, returns empty aderyn list.
        with patch("src.orchestration.nodes.quick_screen.tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("os.unlink"), \
             patch("subprocess.run", side_effect=FileNotFoundError("aderyn not found")):
            mock_file = type("MockTmpFile", (), {"name": "/tmp/x.sol", "write": lambda s, t: None})()
            mock_tmp.return_value.__enter__ = lambda s: mock_file
            mock_tmp.return_value.__exit__  = lambda s, *a: None
            # Also stub out slither import to avoid needing it installed
            with patch.dict("sys.modules", {"slither": None}):
                result = await quick_screen(base_state)

        assert result["quick_screen_hits"]["aderyn"] == []

    @pytest.mark.asyncio
    async def test_result_always_has_both_keys(self, base_state):
        """quick_screen_hits must always contain both 'slither' and 'aderyn' keys."""
        with patch.dict("sys.modules", {"slither": None}), \
             patch("subprocess.run", side_effect=FileNotFoundError("aderyn")):
            result = await quick_screen(base_state)

        hits = result["quick_screen_hits"]
        assert "slither" in hits
        assert "aderyn"  in hits
        assert isinstance(hits["slither"], list)
        assert isinstance(hits["aderyn"],  list)

    @pytest.mark.asyncio
    async def test_synthesizer_falls_back_without_precomputed_verdicts(self, base_state, ml_high):
        state = {
            **base_state,
            "ml_result":       ml_high,
            "rag_results":     [],
            "audit_history":   [],
            "static_findings": [],
            "routing_decisions": [],
        }
        with patch("src.llm.client.get_strong_llm",
                   side_effect=ImportError("no LLM")):
            result = await synthesizer(state)

        report = result["final_report"]
        assert len(report["vulnerability_verdicts"]) > 0
        for vv in report["vulnerability_verdicts"]:
            assert vv["verdict"] in ("CONFIRMED", "LIKELY", "DISPUTED", "WATCH", "SAFE",
                                     "CORROBORATED", "UNCONFIRMED", "ML_ONLY")
