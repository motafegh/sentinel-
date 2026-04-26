# agents/tests/test_inference_server.py
"""
Unit tests for the sentinel-inference MCP server (Track 3, 2026-04-17).

Strategy: no live HTTP calls, no running server process.
  - list_tools() is a pure async function — call it directly
  - call_tool() is tested by calling the registered handler directly
  - _call_inference_api() is patched at the module level so tool
    handlers run their full logic without touching Module 1

Why patch at module level rather than mocking httpx:
  Patching _call_inference_api isolates tool handler logic from
  HTTP transport logic. If httpx changes its API, these tests don't
  break. Each layer is tested independently.

WHAT CHANGED FROM BINARY (Track 3, 2026-04-17):
  - MOCK_PREDICTION_RESULT: uses Track 3 schema (label, vulnerabilities list)
  - _mock_prediction tests: assert new schema fields (label, vulnerabilities)
  - Tests asserting "risk_score" or "confidence" → updated to "label"/"vulnerabilities"
  - Assert "mock" key is GONE (A-13 fix was already in the implementation)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# Import the server object and the functions we test directly.
# We do NOT import run_server — that would start uvicorn.
from src.mcp.servers.inference_server import (
    _handle_batch_predict,
    _handle_predict,
    _mock_prediction,
    list_tools,
    call_tool,
)
from mcp.types import TextContent
import httpx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONTRACT = (
    "pragma solidity ^0.8.0;\n"
    "contract Safe {\n"
    "    mapping(address => uint256) public balances;\n"
    "}"
)

REENTRANCY_CONTRACT = (
    "pragma solidity ^0.8.0;\n"
    "contract Vulnerable {\n"
    "    function withdraw() public {\n"
    "        msg.sender.call.value(balance)(\"\");\n"
    "    }\n"
    "}"
)

MOCK_PREDICTION_RESULT: dict[str, Any] = {
    "label":           "safe",
    "vulnerabilities": [],
    "threshold":       0.50,
    "truncated":       False,
    "num_nodes":       42,
    "num_edges":       58,
}


# ---------------------------------------------------------------------------
# list_tools — tool registration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools_returns_two_tools():
    """Server must always expose exactly predict + batch_predict."""
    tools = await list_tools()
    names = {t.name for t in tools}
    assert names == {"predict", "batch_predict"}


@pytest.mark.asyncio
async def test_predict_tool_schema_has_required_field():
    """predict tool must declare contract_code as required."""
    tools = await list_tools()
    predict = next(t for t in tools if t.name == "predict")
    assert "contract_code" in predict.inputSchema["required"]


@pytest.mark.asyncio
async def test_predict_tool_schema_contract_address_optional():
    """contract_address must NOT be in required — it's optional metadata."""
    tools = await list_tools()
    predict = next(t for t in tools if t.name == "predict")
    assert "contract_address" not in predict.inputSchema["required"]


@pytest.mark.asyncio
async def test_batch_predict_tool_schema_contracts_required():
    """batch_predict tool must declare contracts as required."""
    tools = await list_tools()
    batch = next(t for t in tools if t.name == "batch_predict")
    assert "contracts" in batch.inputSchema["required"]


# ---------------------------------------------------------------------------
# _mock_prediction — mock heuristic
# ---------------------------------------------------------------------------

def test_mock_prediction_safe_contract_returns_safe():
    """Contracts without reentrancy patterns should return label='safe', empty list."""
    result = _mock_prediction(SAMPLE_CONTRACT)
    assert result["label"] == "safe"
    assert result["vulnerabilities"] == []
    assert "mock" not in result   # A-13: no mock key in production-mirror schema


def test_mock_prediction_reentrancy_pattern_high_risk():
    """Contracts with call.value pattern should return label='vulnerable' + vuln list."""
    result = _mock_prediction(REENTRANCY_CONTRACT)
    assert result["label"] == "vulnerable"
    assert len(result["vulnerabilities"]) > 0
    vuln_map = {v["vulnerability_class"]: v["probability"] for v in result["vulnerabilities"]}
    assert "Reentrancy" in vuln_map
    assert vuln_map["Reentrancy"] == 0.72


def test_mock_prediction_result_structure():
    """Mock result must have exact keys Module 1 returns — Track 3 schema contract."""
    result = _mock_prediction(SAMPLE_CONTRACT)
    expected_keys = {"label", "vulnerabilities", "threshold", "truncated", "num_nodes", "num_edges"}
    assert set(result.keys()) == expected_keys
    assert "mock" not in result   # A-13: must NOT have mock key


# ---------------------------------------------------------------------------
# _handle_predict — single contract tool handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_predict_returns_text_content():
    """Handler must return list[TextContent] — the MCP return type contract."""
    with patch(
        "src.mcp.servers.inference_server._call_inference_api",
        new=AsyncMock(return_value=MOCK_PREDICTION_RESULT),
    ):
        result = await _handle_predict({"contract_code": SAMPLE_CONTRACT})

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].type == "text"


@pytest.mark.asyncio
async def test_handle_predict_result_is_valid_json():
    """TextContent.text must be parseable JSON — agents call json.loads() on it."""
    with patch(
        "src.mcp.servers.inference_server._call_inference_api",
        new=AsyncMock(return_value=MOCK_PREDICTION_RESULT),
    ):
        result = await _handle_predict({"contract_code": SAMPLE_CONTRACT})

    parsed = json.loads(result[0].text)
    assert "label" in parsed
    assert "vulnerabilities" in parsed


@pytest.mark.asyncio
async def test_handle_predict_passes_contract_address():
    """contract_address arg must be forwarded to _call_inference_api."""
    mock_api = AsyncMock(return_value=MOCK_PREDICTION_RESULT)
    with patch("src.mcp.servers.inference_server._call_inference_api", new=mock_api):
        await _handle_predict({
            "contract_code": SAMPLE_CONTRACT,
            "contract_address": "0xabc123",
        })

    mock_api.assert_called_once_with(SAMPLE_CONTRACT, "0xabc123")


@pytest.mark.asyncio
async def test_handle_predict_http_error_returns_error_content():
    """4xx from Module 1 must produce error TextContent, not raise."""
    mock_response = AsyncMock()
    mock_response.status_code = 422
    mock_response.text = "Unprocessable Entity"

    with patch(
        "src.mcp.servers.inference_server._call_inference_api",
        new=AsyncMock(side_effect=httpx.HTTPStatusError(
            "422", request=AsyncMock(), response=mock_response
        )),
    ):
        result = await _handle_predict({"contract_code": SAMPLE_CONTRACT})

    parsed = json.loads(result[0].text)
    assert "error" in parsed
    assert parsed["status_code"] == 422


# ---------------------------------------------------------------------------
# _handle_batch_predict — batch handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_predict_processes_all_contracts():
    """All contracts in the batch must produce a result entry."""
    contracts = [
        {"contract_code": SAMPLE_CONTRACT},
        {"contract_code": REENTRANCY_CONTRACT},
    ]
    with patch(
        "src.mcp.servers.inference_server._call_inference_api",
        new=AsyncMock(return_value=MOCK_PREDICTION_RESULT),
    ):
        result = await _handle_batch_predict({"contracts": contracts})

    parsed = json.loads(result[0].text)
    assert len(parsed["results"]) == 2


@pytest.mark.asyncio
async def test_batch_predict_index_field_present():
    """Each result must include index field for agent-side mapping."""
    contracts = [{"contract_code": SAMPLE_CONTRACT}]
    with patch(
        "src.mcp.servers.inference_server._call_inference_api",
        new=AsyncMock(return_value=MOCK_PREDICTION_RESULT),
    ):
        result = await _handle_batch_predict({"contracts": contracts})

    parsed = json.loads(result[0].text)
    assert parsed["results"][0]["index"] == 0


@pytest.mark.asyncio
async def test_batch_predict_partial_failure_continues():
    """
    One contract failing with HTTPStatusError must not abort the whole batch.
    Other contracts must still appear in results.
    """
    mock_response = AsyncMock()
    mock_response.status_code = 500

    # First call succeeds, second raises HTTPStatusError
    mock_api = AsyncMock(side_effect=[
        MOCK_PREDICTION_RESULT,
        httpx.HTTPStatusError("500", request=AsyncMock(), response=mock_response),
    ])

    contracts = [
        {"contract_code": SAMPLE_CONTRACT},
        {"contract_code": REENTRANCY_CONTRACT},
    ]
    with patch("src.mcp.servers.inference_server._call_inference_api", new=mock_api):
        result = await _handle_batch_predict({"contracts": contracts})

    parsed = json.loads(result[0].text)
    # Both contracts must appear — one with result, one with error
    assert len(parsed["results"]) == 2
    assert "error" in parsed["results"][1]
    assert "label" in parsed["results"][0]


@pytest.mark.asyncio
async def test_batch_predict_enforces_size_cap():
    """Batches over 20 must be rejected before calling inference API."""
    contracts = [{"contract_code": SAMPLE_CONTRACT}] * 21

    mock_api = AsyncMock(return_value=MOCK_PREDICTION_RESULT)
    with patch("src.mcp.servers.inference_server._call_inference_api", new=mock_api):
        result = await _handle_batch_predict({"contracts": contracts})

    parsed = json.loads(result[0].text)
    assert "error" in parsed
    # Inference must not have been called at all
    mock_api.assert_not_called()


# ---------------------------------------------------------------------------
# call_tool dispatcher
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_tool_routes_predict():
    """call_tool('predict') must route to _handle_predict and return TextContent."""
    with patch(
        "src.mcp.servers.inference_server._call_inference_api",
        new=AsyncMock(return_value=MOCK_PREDICTION_RESULT),
    ):
        result = await call_tool("predict", {"contract_code": SAMPLE_CONTRACT})

    assert isinstance(result[0], TextContent)
    parsed = json.loads(result[0].text)
    assert "label" in parsed
    assert "vulnerabilities" in parsed


@pytest.mark.asyncio
async def test_call_tool_routes_batch_predict():
    """call_tool('batch_predict') must route to _handle_batch_predict."""
    contracts = [{"contract_code": SAMPLE_CONTRACT}]
    with patch(
        "src.mcp.servers.inference_server._call_inference_api",
        new=AsyncMock(return_value=MOCK_PREDICTION_RESULT),
    ):
        result = await call_tool("batch_predict", {"contracts": contracts})

    parsed = json.loads(result[0].text)
    assert "results" in parsed


@pytest.mark.asyncio
async def test_call_tool_unknown_name_returns_error_not_raises():
    """
    Unknown tool name must return error TextContent — never raise.
    The MCP SDK expects a result, not an exception, from call_tool.
    """
    result = await call_tool("nonexistent_tool", {})

    assert isinstance(result, list)
    assert len(result) == 1
    parsed = json.loads(result[0].text)
    assert "error" in parsed
    assert "nonexistent_tool" in parsed["error"]