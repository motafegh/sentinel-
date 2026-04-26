"""
test_audit_server.py — Unit tests for the sentinel-audit MCP server.

Tests the tool handlers directly (no HTTP, no SSE, no subprocess).
The registry object is mocked so these tests pass without a live RPC.

Run:
    cd ~/projects/sentinel
    cd agents && poetry run pytest tests/test_audit_server.py -v

All tests must pass without SEPOLIA_RPC_URL being set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── sys.path — make agents/ importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set mock mode BEFORE importing audit_server so the module-level
# _MOCK_MODE = True check triggers correctly.
os.environ.setdefault("AUDIT_MOCK", "true")
os.environ.setdefault("SEPOLIA_RPC_URL", "")

from src.mcp.servers.audit_server import (
    _decode_audit_result,
    _handle_check_audit_exists,
    _handle_get_audit_history,
    _handle_get_latest_audit,
    _mock_audit_result,
    _mock_history,
    _validate_address,
    list_tools,
    EZKL_SCALE_FACTOR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ADDRESS   = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
INVALID_ADDRESS = "not-an-address"

# A realistic AuditResult tuple as returned by web3.py from the contract.
# Layout: (scoreFieldElement, proofHash, timestamp, agent, verified)
SAMPLE_TUPLE = (
    5993,                              # scoreFieldElement (5993/8192 ≈ 0.7314)
    bytes.fromhex("ab" * 32),          # proofHash (bytes32)
    1713200000,                        # timestamp
    "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",  # agent
    True,                              # verified
)

ZERO_TUPLE = (0, b"\x00" * 32, 0, "0x0000000000000000000000000000000000000000", False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(content_list) -> dict:
    """Extract and parse the JSON from a list[TextContent] tool response."""
    assert len(content_list) == 1, f"Expected 1 TextContent, got {len(content_list)}"
    return json.loads(content_list[0].text)


# ---------------------------------------------------------------------------
# list_tools — schema tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools_returns_three_tools():
    """Audit server must expose exactly 3 tools."""
    tools = await list_tools()
    names = {t.name for t in tools}
    assert names == {"get_latest_audit", "get_audit_history", "check_audit_exists"}


@pytest.mark.asyncio
async def test_list_tools_schemas_have_required():
    """Every tool schema must declare which arguments are required."""
    tools = await list_tools()
    for tool in tools:
        schema = tool.inputSchema
        assert "required" in schema, f"{tool.name} missing 'required' in schema"
        assert "contract_address" in schema["required"], (
            f"{tool.name} does not require contract_address"
        )


# ---------------------------------------------------------------------------
# _validate_address
# ---------------------------------------------------------------------------

def test_validate_address_accepts_valid():
    """Valid checksummed address should pass without error."""
    result = _validate_address(VALID_ADDRESS)
    assert result == VALID_ADDRESS


def test_validate_address_lowercases():
    """Lowercase addresses should be checksummed automatically."""
    lower = VALID_ADDRESS.lower()
    result = _validate_address(lower)
    assert result == VALID_ADDRESS


def test_validate_address_rejects_garbage():
    """Non-address strings must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid Ethereum address"):
        _validate_address(INVALID_ADDRESS)


# ---------------------------------------------------------------------------
# _decode_audit_result
# ---------------------------------------------------------------------------

def test_decode_audit_result_score():
    """Score must be scoreFieldElement / EZKL_SCALE_FACTOR."""
    decoded = _decode_audit_result(SAMPLE_TUPLE, VALID_ADDRESS)
    expected_score = round(5993 / EZKL_SCALE_FACTOR, 4)
    assert decoded["score"] == expected_score


def test_decode_audit_result_label_vulnerable():
    """Score ≥ 0.50 must produce label='vulnerable'."""
    decoded = _decode_audit_result(SAMPLE_TUPLE, VALID_ADDRESS)
    assert decoded["label"] == "vulnerable"


def test_decode_audit_result_label_safe():
    """Score < 0.50 must produce label='safe'."""
    safe_tuple = (3000, b"\x00" * 32, 1713200000, VALID_ADDRESS, True)
    decoded = _decode_audit_result(safe_tuple, VALID_ADDRESS)
    assert decoded["label"] == "safe"


def test_decode_audit_result_required_fields():
    """Decoded dict must contain all fields expected by downstream agents."""
    decoded = _decode_audit_result(SAMPLE_TUPLE, VALID_ADDRESS)
    required = {
        "contract_address", "score", "score_field_element", "label",
        "threshold", "proof_hash", "timestamp", "timestamp_iso",
        "agent", "verified",
    }
    missing = required - set(decoded.keys())
    assert not missing, f"Missing fields: {missing}"


def test_decode_audit_result_proof_hash_hex():
    """proof_hash must be a 0x-prefixed hex string of length 66."""
    decoded = _decode_audit_result(SAMPLE_TUPLE, VALID_ADDRESS)
    assert decoded["proof_hash"].startswith("0x")
    assert len(decoded["proof_hash"]) == 66  # "0x" + 64 hex chars


# ---------------------------------------------------------------------------
# _handle_get_latest_audit (mock mode)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_latest_audit_mock_returns_result():
    """In mock mode, get_latest_audit must return a valid decoded result."""
    result = await _handle_get_latest_audit({"contract_address": VALID_ADDRESS})
    data = _parse(result)
    assert "score" in data
    assert "label" in data
    assert 0.0 <= data["score"] <= 1.0


@pytest.mark.asyncio
async def test_get_latest_audit_bad_address():
    """Invalid address must return an error dict, not raise."""
    result = await _handle_get_latest_audit({"contract_address": INVALID_ADDRESS})
    data = _parse(result)
    assert "error" in data


@pytest.mark.asyncio
async def test_get_latest_audit_live_no_audit(monkeypatch):
    """
    In live mode, if the contract returns a zero-timestamp tuple,
    get_latest_audit must return {exists: False, message: ...}.
    """
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)

    mock_contract = MagicMock()
    mock_contract.functions.getLatestAudit.return_value.call = AsyncMock(
        return_value=ZERO_TUPLE
    )
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", mock_contract)

    result = await _handle_get_latest_audit({"contract_address": VALID_ADDRESS})
    data = _parse(result)
    assert data.get("exists") is False
    assert "message" in data


@pytest.mark.asyncio
async def test_get_latest_audit_live_rpc_error(monkeypatch):
    """
    In live mode, an RPC exception must return a structured error dict,
    not propagate the exception to the MCP session.
    """
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)

    mock_contract = MagicMock()
    mock_contract.functions.getLatestAudit.return_value.call = AsyncMock(
        side_effect=ConnectionError("RPC timeout")
    )
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", mock_contract)

    result = await _handle_get_latest_audit({"contract_address": VALID_ADDRESS})
    data = _parse(result)
    assert data["error"] == "rpc_error"
    assert "detail" in data


# ---------------------------------------------------------------------------
# _handle_get_audit_history (mock mode)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_audit_history_mock_returns_list():
    """In mock mode, history must contain at least one record."""
    result = await _handle_get_audit_history({
        "contract_address": VALID_ADDRESS,
        "limit": 5,
    })
    data = _parse(result)
    assert "records" in data
    assert isinstance(data["records"], list)
    assert len(data["records"]) >= 1


@pytest.mark.asyncio
async def test_get_audit_history_respects_limit(monkeypatch):
    """
    History must not return more records than the requested limit.
    Even in live mode with 100 on-chain records, limit=2 must return ≤ 2.
    """
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)

    many_records = [SAMPLE_TUPLE] * 100
    mock_contract = MagicMock()
    mock_contract.functions.getAuditHistory.return_value.call = AsyncMock(
        return_value=many_records
    )
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", mock_contract)

    result = await _handle_get_audit_history({
        "contract_address": VALID_ADDRESS,
        "limit": 2,
    })
    data = _parse(result)
    assert len(data["records"]) <= 2


@pytest.mark.asyncio
async def test_get_audit_history_live_empty(monkeypatch):
    """Empty on-chain history must return count=0 and empty records list."""
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)

    mock_contract = MagicMock()
    mock_contract.functions.getAuditHistory.return_value.call = AsyncMock(return_value=[])
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", mock_contract)

    result = await _handle_get_audit_history({"contract_address": VALID_ADDRESS})
    data = _parse(result)
    assert data["count"] == 0
    assert data["records"] == []


# ---------------------------------------------------------------------------
# _handle_check_audit_exists (mock mode)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_audit_exists_mock():
    """In mock mode, check_audit_exists must return {exists, count}."""
    result = await _handle_check_audit_exists({"contract_address": VALID_ADDRESS})
    data = _parse(result)
    assert "exists" in data
    assert "count" in data
    assert isinstance(data["exists"], bool)
    assert isinstance(data["count"], int)


@pytest.mark.asyncio
async def test_check_audit_exists_bad_address():
    """Invalid address must return error dict."""
    result = await _handle_check_audit_exists({"contract_address": "0xinvalid"})
    data = _parse(result)
    assert "error" in data


@pytest.mark.asyncio
async def test_check_audit_exists_live(monkeypatch):
    """Live mode must correctly reflect contract hasAudit + getAuditCount."""
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)

    mock_contract = MagicMock()
    mock_contract.functions.hasAudit.return_value.call   = AsyncMock(return_value=True)
    mock_contract.functions.getAuditCount.return_value.call = AsyncMock(return_value=7)
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", mock_contract)

    result = await _handle_check_audit_exists({"contract_address": VALID_ADDRESS})
    data = _parse(result)
    assert data["exists"] is True
    assert data["count"] == 7


# ---------------------------------------------------------------------------
# Hard cap enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_history_limit_capped_at_50(monkeypatch):
    """
    Even if the agent requests limit=999, must cap at 50.
    This verifies the hard cap in _handle_get_audit_history.
    """
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)

    many_records = [SAMPLE_TUPLE] * 200
    mock_contract = MagicMock()
    mock_contract.functions.getAuditHistory.return_value.call = AsyncMock(
        return_value=many_records
    )
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", mock_contract)

    result = await _handle_get_audit_history({
        "contract_address": VALID_ADDRESS,
        "limit": 999,   # way over the cap
    })
    data = _parse(result)
    assert len(data["records"]) <= 50
