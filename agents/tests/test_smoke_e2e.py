# agents/tests/test_smoke_e2e.py
"""
End-to-end smoke test for the full SENTINEL audit graph (A5).

This tests the complete LangGraph topology with all nodes:
    ml_assessment → quick_screen → evidence_router → [deep/fast] → synthesizer

All MCP and external calls are mocked. Slither and Aderyn run in-process
(or are skipped gracefully if not available). The test verifies:
    1. Deep path is triggered for a reentrancy contract (ML + screen both fire)
    2. quick_screen_hits is populated in the final state
    3. final_report contains the required Phase 1 fields
    4. graph_explanations and verdicts are present after deep path
    5. Screen-escalated path: ML safe but quick_screen fires → still goes deep
    6. Fast path: ML safe + screen clean → synthesizer directly

These tests do NOT require any MCP server to be running. All network calls
are intercepted via unittest.mock.patch.

IMPORTANT: These are integration tests (full graph + real Slither when available).
They run slower than unit tests (~30s each). Run selectively with:
    poetry run pytest tests/test_smoke_e2e.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.graph import build_graph
from src.orchestration.state import AuditState


# ---------------------------------------------------------------------------
# Fixtures — contract sources and ML responses
# ---------------------------------------------------------------------------

VAULT_CONTRACT = """
pragma solidity ^0.8.0;
contract Vault {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;
    }
}
""".strip()

SAFE_CONTRACT = """
pragma solidity ^0.8.0;
contract SafeStorage {
    uint256 public value;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function setValue(uint256 _value) external {
        require(msg.sender == owner, "Not owner");
        value = _value;
    }

    function getValue() external view returns (uint256) {
        return value;
    }
}
""".strip()


def _ml_result_vault() -> dict:
    """Three-tier ML result for Vault — Reentrancy confirmed."""
    return {
        "label": "confirmed_vulnerable",
        "probabilities": {
            "Reentrancy": 0.82, "IntegerUO": 0.15, "GasException": 0.05,
            "Timestamp": 0.08, "TransactionOrderDependence": 0.04, "ExternalBug": 0.11,
            "CallToUnknown": 0.09, "MishandledException": 0.06,
            "UnusedReturn": 0.03, "DenialOfService": 0.04,
        },
        "confirmed":  [{"vulnerability_class": "Reentrancy", "probability": 0.82, "tier": "CONFIRMED"}],
        "suspicious": [],
        "vulnerabilities": [{"vulnerability_class": "Reentrancy", "probability": 0.82, "tier": "CONFIRMED"}],
        "tier_thresholds": {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10},
        "thresholds": [0.5] * 10,
        "truncated": False,
        "windows_used": 1,
        "num_nodes": 42,
        "num_edges": 58,
    }


def _ml_result_safe() -> dict:
    """Three-tier ML result — all classes well below DEEP_THRESHOLDS."""
    probs = {
        "Reentrancy": 0.04, "IntegerUO": 0.03, "GasException": 0.02,
        "Timestamp": 0.05, "TransactionOrderDependence": 0.01, "ExternalBug": 0.03,
        "CallToUnknown": 0.02, "MishandledException": 0.02,
        "UnusedReturn": 0.01, "DenialOfService": 0.01,
    }
    return {
        "label": "safe",
        "probabilities": probs,
        "confirmed":  [],
        "suspicious": [],
        "vulnerabilities": [],
        "tier_thresholds": {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10},
        "thresholds": [0.5] * 10,
        "truncated": False,
        "windows_used": 1,
        "num_nodes": 18,
        "num_edges": 22,
    }


_MOCK_RAG_CHUNKS = [
    {"chunk_id": "r-1", "content": "Reentrancy in Compound…", "doc_id": "d-1", "score": 0.88},
]

_MOCK_AUDIT_RESPONSE = {
    "contract_address": "0xVAULT",
    "count": 0,
    "records": [],
}

_MOCK_INSPECTOR_RESPONSE = {
    "hotspots": [
        {
            "contract": "Vault",
            "function": "withdraw",
            "lines": [9, 10, 11],
            "vulnerability_classes": ["Reentrancy"],
            "score": 0.91,
            "signals": ["reentrancy-eth(High)"],
        }
    ],
    "graph_stats": {"num_contracts": 1, "num_functions": 2, "has_interfaces": False, "num_hotspots": 1},
    "analysis_mode": "gnn",
}


async def _mock_mcp(server_url: str, tool_name: str, arguments: dict) -> dict:
    if tool_name == "predict":           return _ml_result_vault()
    if tool_name == "search":            return _MOCK_RAG_CHUNKS
    if tool_name == "get_audit_history": return _MOCK_AUDIT_RESPONSE
    if tool_name == "get_graph_hotspots": return _MOCK_INSPECTOR_RESPONSE
    return {"error": f"unexpected tool: {tool_name}"}


async def _mock_mcp_safe(server_url: str, tool_name: str, arguments: dict) -> dict:
    if tool_name == "predict":           return _ml_result_safe()
    if tool_name == "search":            return []
    if tool_name == "get_audit_history": return _MOCK_AUDIT_RESPONSE
    if tool_name == "get_graph_hotspots": return _MOCK_INSPECTOR_RESPONSE
    return {"error": f"unexpected tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------

class TestSmokePaths:
    """
    Full-graph smoke tests. All MCP calls mocked; Slither runs in-process
    (results vary by environment), Aderyn skipped if not installed.
    """

    @pytest.mark.asyncio
    async def test_deep_path_vault_produces_final_report(self):
        """
        Vault contract with high Reentrancy ML score → deep path → full report.
        Verifies all Phase 1 required fields are present in final_report.
        """
        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code":    VAULT_CONTRACT,
            "contract_address": "0xVAULT",
        }

        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=_mock_mcp):
            result = await graph.ainvoke(initial_state)

        report = result.get("final_report")
        assert report is not None, "final_report must be set"
        assert report["path_taken"] == "deep"
        assert report["overall_label"] == "confirmed_vulnerable"

        # Required Phase 1 fields
        required = [
            "contract_address", "overall_label", "risk_probability",
            "top_vulnerability", "confirmed", "suspicious", "probabilities",
            "tier_thresholds", "threshold", "ml_truncated", "num_nodes", "num_edges",
            "rag_evidence", "audit_history", "static_findings", "recommendation",
            "error", "path_taken",
        ]
        for f in required:
            assert f in report, f"Missing required field: {f}"

        assert report["error"] is None

    @pytest.mark.asyncio
    async def test_quick_screen_hits_in_final_state(self):
        """
        After a full graph run, quick_screen_hits must be set in state.
        Its content may be empty (Slither/Aderyn not available in CI) but
        the key must exist — the node always runs.
        """
        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code":    VAULT_CONTRACT,
            "contract_address": "0xVAULT",
        }

        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=_mock_mcp):
            result = await graph.ainvoke(initial_state)

        assert "quick_screen_hits" in result, "quick_screen_hits must be in final state"
        hits = result["quick_screen_hits"]
        assert "slither" in hits
        assert "aderyn"  in hits
        assert isinstance(hits["slither"], list)
        assert isinstance(hits["aderyn"],  list)

    @pytest.mark.asyncio
    async def test_deep_path_graph_explanations_present(self):
        """graph_explanations must be non-empty after a deep-path run."""
        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code":    VAULT_CONTRACT,
            "contract_address": "0xVAULT",
        }

        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=_mock_mcp):
            result = await graph.ainvoke(initial_state)

        assert "graph_explanations" in result
        assert result["graph_explanations"] != {}

    @pytest.mark.asyncio
    async def test_routing_decisions_logged(self):
        """
        routing_decisions must contain at least one entry documenting
        why the path was chosen. This proves evidence_router ran and logged.
        """
        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code":    VAULT_CONTRACT,
            "contract_address": "0xVAULT",
        }

        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=_mock_mcp):
            result = await graph.ainvoke(initial_state)

        decisions = result.get("routing_decisions", [])
        assert len(decisions) > 0, "routing_decisions must be populated"
        # At least one entry should mention Reentrancy (the confirmed class)
        joined = " ".join(decisions)
        assert "Reentrancy" in joined

    @pytest.mark.asyncio
    async def test_fast_path_safe_contract(self):
        """
        A safe contract with ML safe and quick_screen clean → fast path.
        No rag_results, no graph_explanations, path_taken == "fast".
        """
        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code":    SAFE_CONTRACT,
            "contract_address": "0xSAFE",
        }

        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=_mock_mcp_safe), \
             patch("src.orchestration.nodes.tempfile.NamedTemporaryFile") as mock_tmp, \
             patch("src.orchestration.nodes.os.unlink"):
            # Stub quick_screen temp file to avoid real Slither on safe contract
            mock_file = type("TmpFile", (), {"name": "/tmp/sentinel_smoke_safe.sol", "write": lambda s, t: None})()
            mock_tmp.return_value.__enter__ = lambda s: mock_file
            mock_tmp.return_value.__exit__  = lambda s, *a: None
            # Stub Slither import so quick_screen returns clean screen
            with patch.dict("sys.modules", {"slither": None}), \
                 patch("subprocess.run", side_effect=FileNotFoundError("aderyn")):
                result = await graph.ainvoke(initial_state)

        report = result.get("final_report")
        assert report is not None
        assert report["path_taken"] == "fast"
        assert report["rag_evidence"] == []

    @pytest.mark.asyncio
    async def test_screen_escalated_path_when_ml_safe_but_screen_fires(self):
        """
        ML says safe (all probs below DEEP_THRESHOLDS) but quick_screen finds
        a High-impact Slither hit → goes deep (screen-escalated path).

        The final report must NOT show path_taken=="fast".
        """
        from unittest.mock import MagicMock
        import slither as _slither_module

        # ML says safe
        ml_safe = _ml_result_safe()

        async def mock_safe_ml(server_url, tool_name, arguments):
            if tool_name == "predict":           return ml_safe
            if tool_name == "search":            return []
            if tool_name == "get_audit_history": return _MOCK_AUDIT_RESPONSE
            if tool_name == "get_graph_hotspots": return _MOCK_INSPECTOR_RESPONSE
            return {}

        # Slither finds a reentrancy-eth hit despite ML saying safe
        mock_finding = {
            "check": "reentrancy-eth", "impact": "High",
            "confidence": "High", "description": "Reentrancy found", "elements": [],
        }
        mock_detector = MagicMock()
        mock_detector.ARGUMENT = "reentrancy-eth"
        mock_sl = MagicMock()
        mock_sl._detectors = [mock_detector]
        mock_sl.run_detectors.return_value = [[mock_finding]]
        mock_slither_cls = MagicMock(return_value=mock_sl)

        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code":    SAFE_CONTRACT,
            "contract_address": "0xTRICKY",
        }

        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=mock_safe_ml), \
             patch.object(_slither_module, "Slither", mock_slither_cls), \
             patch("subprocess.run", side_effect=FileNotFoundError("aderyn")):
            result = await graph.ainvoke(initial_state)

        hits = result.get("quick_screen_hits", {})
        assert "reentrancy-eth" in hits.get("slither", []), (
            "quick_screen must record the Slither hit"
        )
        report = result.get("final_report")
        assert report is not None
        assert report["path_taken"] != "fast", (
            "Screen-escalated path must NOT be a fast path even though ML said safe"
        )

    @pytest.mark.asyncio
    async def test_ml_failure_still_produces_report(self):
        """Graph must produce a final report even when ML inference is unavailable."""
        async def ml_down(url, tool, args):
            raise ConnectionError("inference server unreachable")

        graph = build_graph(use_checkpointer=False)
        initial_state = {
            "contract_code":    VAULT_CONTRACT,
            "contract_address": "0xERROR",
        }

        with patch("src.orchestration.nodes._call_mcp_tool", side_effect=ml_down), \
             patch.dict("sys.modules", {"slither": None}), \
             patch("subprocess.run", side_effect=FileNotFoundError("aderyn")):
            result = await graph.ainvoke(initial_state)

        report = result.get("final_report")
        assert report is not None, "Must produce a report even with ML failure"
        assert "ML assessment failed" in report["recommendation"]
        assert report["error"] is not None
