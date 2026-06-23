"""
Unit tests for the sentinel-representation MCP server (WS5, 2026-06-22).

Strategy: no live HTTP calls, no running server process, no solc needed.
  - The pure-logic helpers (_detect_cei_violations, _summarise_function)
    are tested directly with hand-constructed CfgFunction dicts.
  - The tool-call dispatcher is tested with a mocked _call_build_cfg
    to verify the JSON shape and error handling.
  - The MCP server (Starlette app) is built and route-validated to
    confirm the SSE/health endpoints are wired up.

These tests run without solc / Slither / data_module, so they pass even
when the full data pipeline isn't available. The live server is exercised
in scripts/smoke_representation_mcp.py (separate).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from mcp.types import TextContent

from src.mcp.servers.representation_server import (
    _NODE_ARITH,
    _NODE_CALL,
    _NODE_CHECK,
    _NODE_OTHER,
    _NODE_READ,
    _NODE_WRITE,
    _detect_cei_violations,
    _mock_function_cfgs,
    _summarise_function,
    app,
    call_tool,
    health,
    list_tools,
)


# ---------------------------------------------------------------------------
# _detect_cei_violations
# ---------------------------------------------------------------------------

class TestCEIViolationDetection:
    """
    The classic CEI (Checks-Effects-Interactions) pattern: a state write
    that happens BEFORE an external call means the function is
    reentrancy-vulnerable. _detect_cei_violations finds every
    write→call path in a function's CFG.
    """

    def test_empty_inputs_returns_empty(self):
        assert _detect_cei_violations([], []) == []

    def test_no_writes_returns_empty(self):
        nodes = [
            {"index": 0, "type": _NODE_CALL,  "source_lines": [10], "expression": "call"},
            {"index": 1, "type": _NODE_READ,  "source_lines": [11], "expression": "read"},
        ]
        edges = [{"src": 0, "dst": 1}]
        assert _detect_cei_violations(edges, nodes) == []

    def test_no_calls_returns_empty(self):
        nodes = [
            {"index": 0, "type": _NODE_WRITE, "source_lines": [10], "expression": "write"},
            {"index": 1, "type": _NODE_READ,  "source_lines": [11], "expression": "read"},
        ]
        edges = [{"src": 0, "dst": 1}]
        assert _detect_cei_violations(edges, nodes) == []

    def test_classic_reentrancy_pattern(self):
        """
        W: balances[msg.sender] = 0   (line 44)
        C: msg.sender.call{value:..}   (line 43 — but CFG node order is what matters)

        With CFG edges 0→1 (W→C), a write reaches a call → CEI violation.
        """
        nodes = [
            {"index": 0, "type": _NODE_WRITE, "source_lines": [44], "expression": "write"},
            {"index": 1, "type": _NODE_CALL,  "source_lines": [43], "expression": "call"},
        ]
        edges = [{"src": 0, "dst": 1}]
        violations = _detect_cei_violations(edges, nodes)
        assert len(violations) == 1
        v = violations[0]
        assert v["write_index"] == 0
        assert v["call_index"] == 1
        assert v["write_source_lines"] == [44]
        assert v["call_source_lines"] == [43]
        assert v["path"] == [0, 1]

    def test_call_before_write_is_not_a_violation(self):
        """
        If a CALL happens before any WRITE, the write doesn't reach the call
        (the call can't be reentered to find the state unchanged). So no violation.
        """
        nodes = [
            {"index": 0, "type": _NODE_CALL,  "source_lines": [10], "expression": "call"},
            {"index": 1, "type": _NODE_WRITE, "source_lines": [11], "expression": "write"},
        ]
        edges = [{"src": 0, "dst": 1}]
        violations = _detect_cei_violations(edges, nodes)
        assert violations == []

    def test_multiple_writes_one_call(self):
        """
        Two writes both reach the same call — should produce 2 violations
        (one per write→call path).
        """
        nodes = [
            {"index": 0, "type": _NODE_WRITE, "source_lines": [40], "expression": "w1"},
            {"index": 1, "type": _NODE_WRITE, "source_lines": [42], "expression": "w2"},
            {"index": 2, "type": _NODE_CALL,  "source_lines": [44], "expression": "call"},
        ]
        edges = [
            {"src": 0, "dst": 2},
            {"src": 1, "dst": 2},
        ]
        violations = _detect_cei_violations(edges, nodes)
        assert len(violations) == 2
        write_indices = sorted(v["write_index"] for v in violations)
        assert write_indices == [0, 1]
        for v in violations:
            assert v["call_index"] == 2
            assert v["call_source_lines"] == [44]

    def test_cycle_is_handled_without_infinite_loop(self):
        """
        Cycle guard: a write → ... → write (cycle) should not loop forever.
        """
        nodes = [
            {"index": 0, "type": _NODE_WRITE, "source_lines": [10], "expression": "w1"},
            {"index": 1, "type": _NODE_CALL,  "source_lines": [11], "expression": "call"},
            {"index": 2, "type": _NODE_WRITE, "source_lines": [12], "expression": "w2"},
        ]
        edges = [
            {"src": 0, "dst": 1},
            {"src": 0, "dst": 2},
            {"src": 1, "dst": 0},  # cycle
            {"src": 2, "dst": 0},  # cycle
        ]
        violations = _detect_cei_violations(edges, nodes)
        # Both writes can reach node 1 (call). Each produces one violation.
        assert len(violations) == 2
        for v in violations:
            assert v["call_index"] == 1


# ---------------------------------------------------------------------------
# _summarise_function
# ---------------------------------------------------------------------------

class TestFunctionSummary:
    def test_empty_function(self):
        cfg_fn = {
            "canonical_name": "Empty.test",
            "nodes": [],
            "edges": [],
            "num_loops": 0,
            "max_depth": 0,
        }
        summary = _summarise_function(cfg_fn)
        assert summary["canonical_name"] == "Empty.test"
        assert summary["num_nodes"] == 0
        assert summary["num_edges"] == 0
        assert summary["cei_violations"] == []
        assert summary["cei_violation_count"] == 0
        assert summary["has_external_call"] is False
        assert summary["has_state_write"] is False
        assert summary["has_arithmetic"] is False
        assert summary["cfg_complexity_score"] == 0

    def test_function_with_cei_violation(self):
        """Classic reentrancy pattern: write reaches call."""
        cfg_fn = {
            "canonical_name": "Vault.withdraw",
            "nodes": [
                {"index": 0, "type": _NODE_READ,  "source_lines": [42], "expression": "r"},
                {"index": 1, "type": _NODE_WRITE, "source_lines": [44], "expression": "w"},
                {"index": 2, "type": _NODE_CALL,  "source_lines": [43], "expression": "c"},
            ],
            "edges": [
                {"src": 0, "dst": 1},
                {"src": 1, "dst": 2},
            ],
            "num_loops": 0,
            "max_depth": 3,
        }
        summary = _summarise_function(cfg_fn)
        assert summary["has_external_call"] is True
        assert summary["has_state_write"] is True
        assert summary["has_arithmetic"] is False
        assert summary["num_loops"] == 0
        assert summary["max_depth"] == 3
        assert summary["cei_violation_count"] == 1
        assert summary["node_type_counts"] == {
            _NODE_READ: 1, _NODE_WRITE: 1, _NODE_CALL: 1,
        }
        # complexity = loops*3 + depth + calls*1 + writes*0.5
        # = 0 + 3 + 1 + 0.5 = 4.5
        assert summary["cfg_complexity_score"] == 4.5

    def test_function_with_loops_and_arithmetic(self):
        """GasException-relevant: loops + arithmetic raise complexity."""
        cfg_fn = {
            "canonical_name": "Token.transferBatch",
            "nodes": [
                {"index": 0, "type": _NODE_CHECK, "source_lines": [10], "expression": "if"},
                {"index": 1, "type": _NODE_ARITH, "source_lines": [11], "expression": "+"},
                {"index": 2, "type": _NODE_OTHER, "source_lines": [12], "expression": "loop_body"},
            ],
            "edges": [
                {"src": 0, "dst": 1},
                {"src": 1, "dst": 2},
                {"src": 2, "dst": 1},  # loop back-edge
            ],
            "num_loops": 1,
            "max_depth": 3,
        }
        summary = _summarise_function(cfg_fn)
        assert summary["num_loops"] == 1
        assert summary["has_arithmetic"] is True
        assert summary["has_external_call"] is False
        # No CEI violation (no calls)
        assert summary["cei_violation_count"] == 0
        # complexity = 1*3 + 3 + 0 + 0 = 6
        assert summary["cfg_complexity_score"] == 6


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

class TestMock:
    def test_mock_function_cfgs_shape(self):
        result = _mock_function_cfgs("pragma solidity ^0.8.0; contract X {}")
        assert result["analysis_mode"] == "mock"
        assert result["schema_version"] == "v9"
        assert result["num_functions"] == 0
        assert result["total_cei_violations"] == 0
        assert result["functions"] == []


# ---------------------------------------------------------------------------
# MCP tool dispatcher
# ---------------------------------------------------------------------------

class TestToolDispatcher:
    def test_list_tools_has_get_function_cfgs(self):
        tools = asyncio.run(list_tools())
        names = [t.name for t in tools]
        assert "get_function_cfgs" in names

    def test_get_function_cfgs_with_mock(self):
        """When _MOCK_MODE is on, the mock data path is used."""
        with patch("src.mcp.servers.representation_server._MOCK_MODE", True):
            result = asyncio.run(call_tool("get_function_cfgs", {
                "contract_code": "pragma solidity ^0.8.0; contract X {}",
            }))
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        parsed = json.loads(result[0].text)
        assert parsed["analysis_mode"] == "mock"
        assert parsed["num_functions"] == 0

    def test_get_function_cfgs_rejects_empty(self):
        result = asyncio.run(call_tool("get_function_cfgs", {
            "contract_code": "   \n  \t",
        }))
        parsed = json.loads(result[0].text)
        assert "error" in parsed
        assert "required" in parsed["error"].lower()

    def test_unknown_tool_returns_error(self):
        result = asyncio.run(call_tool("get_nonexistent_tool", {}))
        parsed = json.loads(result[0].text)
        assert "error" in parsed
        assert "Unknown tool" in parsed["error"]

    def test_get_function_cfgs_with_mocked_build_cfg(self):
        """Mock _call_build_cfg to verify the real (non-mock) path wires up."""
        with patch(
            "src.mcp.servers.representation_server._MOCK_MODE", False,
        ), patch(
            "src.mcp.servers.representation_server._call_build_cfg",
            return_value={
                "analysis_mode": "data_module_cfgs",
                "schema_version": "v9",
                "extractor_version": "test",
                "solc_version": "0.8.19",
                "error": None,
                "num_functions": 1,
                "total_cei_violations": 1,
                "functions": [
                    {
                        "canonical_name": "Test.vuln",
                        "num_nodes": 3,
                        "num_edges": 2,
                        "num_loops": 0,
                        "max_depth": 2,
                        "node_type_counts": {"CFG_NODE_CALL": 1, "CFG_NODE_WRITE": 1},
                        "has_external_call": True,
                        "has_state_write": True,
                        "has_arithmetic": False,
                        "cfg_complexity_score": 2.5,
                        "cei_violations": [
                            {
                                "write_index": 0,
                                "call_index": 1,
                                "write_source_lines": [5],
                                "call_source_lines": [4],
                                "path": [0, 1],
                            }
                        ],
                        "cei_violation_count": 1,
                    }
                ],
            },
        ):
            result = asyncio.run(call_tool("get_function_cfgs", {
                "contract_code": "pragma solidity ^0.8.0; contract Test { function vuln() public { x = 1; other.call(); } }",
            }))
        parsed = json.loads(result[0].text)
        assert parsed["analysis_mode"] == "data_module_cfgs"
        assert parsed["num_functions"] == 1
        assert parsed["total_cei_violations"] == 1
        assert parsed["functions"][0]["cei_violation_count"] == 1


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = asyncio.run(health(None))
        # Starlette's JSONResponse stores the body as bytes
        body = json.loads(bytes(response.body).decode("utf-8"))
        assert body["status"] == "ok"
        assert body["server"] == "sentinel-representation"
        assert body["port"] == 8014
        assert "data_module_cfgs" in body["backends"]


# ---------------------------------------------------------------------------
# ASGI app routes
# ---------------------------------------------------------------------------

class TestAppRoutes:
    def test_health_route_registered(self):
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/health" in paths
        assert "/sse" in paths
