"""Tests for the live read-only, version-aware SENTINEL audit MCP."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("AUDIT_MOCK", "true")
os.environ.setdefault("SEPOLIA_RPC_URL", "")

from src.mcp.servers.audit._server import _readiness_payload
from src.mcp.servers.audit._versioned_reads import decode_v1, decode_v2, decode_v3
from src.mcp.servers.audit_server import (
    _handle_check_audit_exists,
    _handle_get_audit_history,
    _handle_get_latest_audit,
    _on_startup,
    _validate_address,
    list_tools,
)

VALID_ADDRESS = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
INVALID_ADDRESS = "not-an-address"
AGENT = "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF"
SIGNER = "0x0000000000000000000000000000000000000002"
VERIFIER = "0x0000000000000000000000000000000000000003"

V1_TUPLE = (5993, bytes.fromhex("11" * 32), 100, AGENT, True)
V2_TUPLE = (
    tuple(1000 + i for i in range(10)),
    bytes.fromhex("22" * 32),
    bytes.fromhex("33" * 32),
    200,
    AGENT,
    True,
)
V3_TUPLE = (
    tuple(2000 + i for i in range(10)),
    bytes.fromhex("44" * 32),
    bytes.fromhex("45" * 32),
    bytes.fromhex("46" * 32),
    bytes.fromhex("47" * 32),
    bytes.fromhex("48" * 32),
    bytes.fromhex("49" * 32),
    bytes.fromhex("4a" * 32),
    bytes.fromhex("4b" * 32),
    7,
    300,
    AGENT,
    SIGNER,
    VERIFIER,
    True,
)


@pytest.fixture(autouse=True)
def _isolate_mock_mode(monkeypatch):
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", True)


def _parse(content_list) -> dict:
    assert len(content_list) == 1
    return json.loads(content_list[0].text)


def _versioned_registry(
    *,
    counts=(1, 1, 1),
    v3_latest=V3_TUPLE,
    v2_latest=V2_TUPLE,
    v1_latest=V1_TUPLE,
    v3_history=None,
    v2_history=None,
    v1_history=None,
):
    registry = MagicMock()
    v3_count, v2_count, v1_count = counts
    registry.functions.getAuditCountV3.return_value.call = AsyncMock(return_value=v3_count)
    registry.functions.getAuditCountV2.return_value.call = AsyncMock(return_value=v2_count)
    registry.functions.getAuditCount.return_value.call = AsyncMock(return_value=v1_count)
    registry.functions.getLatestAuditV3.return_value.call = AsyncMock(return_value=v3_latest)
    registry.functions.getLatestAuditV2.return_value.call = AsyncMock(return_value=v2_latest)
    registry.functions.getLatestAudit.return_value.call = AsyncMock(return_value=v1_latest)
    registry.functions.getAuditHistoryV3.return_value.call = AsyncMock(
        return_value=[v3_latest] if v3_history is None else v3_history
    )
    registry.functions.getAuditHistoryV2.return_value.call = AsyncMock(
        return_value=[v2_latest] if v2_history is None else v2_history
    )
    registry.functions.getAuditHistory.return_value.call = AsyncMock(
        return_value=[v1_latest] if v1_history is None else v1_history
    )
    return registry


@pytest.mark.asyncio
async def test_list_tools_returns_exactly_three_read_only_tools():
    tools = await list_tools()
    assert {tool.name for tool in tools} == {
        "get_latest_audit",
        "get_audit_history",
        "check_audit_exists",
    }


@pytest.mark.asyncio
async def test_tool_descriptions_are_protocol_aware():
    tools = {tool.name: tool for tool in await list_tools()}
    assert "V3" in tools["get_latest_audit"].description
    assert "V2" in tools["get_latest_audit"].description
    assert "protocol_version" in tools["get_latest_audit"].description


def test_validate_address_accepts_lowercase_and_rejects_garbage():
    assert _validate_address(VALID_ADDRESS.lower()) == VALID_ADDRESS
    with pytest.raises(ValueError, match="Invalid Ethereum address"):
        _validate_address(INVALID_ADDRESS)


def test_decode_v1_is_explicitly_versioned():
    record = decode_v1(V1_TUPLE, VALID_ADDRESS)
    assert record["protocol_version"] == "v1"
    assert record["score_field_element"] == 5993


def test_decode_v2_preserves_raw_class_score_felts():
    record = decode_v2(V2_TUPLE, VALID_ADDRESS)
    assert record["protocol_version"] == "v2"
    assert record["proof_scope"] == "legacy_proxy_only_unbound"
    assert record["class_score_felts"] == list(V2_TUPLE[0])
    assert "score" not in record
    assert "label" not in record


def test_decode_v3_exposes_bound_provenance_without_fake_probability():
    record = decode_v3(V3_TUPLE, VALID_ADDRESS)
    assert record["protocol_version"] == "v3"
    assert record["submission_protocol"] == "context_attested_v3"
    assert record["round_id"] == 7
    assert record["teacher_model_hash"] == "0x" + "48" * 32
    assert record["policy_signer"] == SIGNER
    assert record["verifier"] == VERIFIER
    assert "score" not in record
    assert "label" not in record


@pytest.mark.asyncio
async def test_latest_mock_reports_v3_shape():
    result = await _handle_get_latest_audit({"contract_address": VALID_ADDRESS})
    data = _parse(result)
    assert data["protocol_version"] == "v3"
    assert len(data["class_score_felts"]) == 10
    assert data["counts_by_protocol"] == {"v3": 1, "v2": 1, "v1": 1}
    assert data["execution_status"]["status"] == "MOCK"


@pytest.mark.asyncio
async def test_latest_live_selects_newest_timestamp_not_highest_protocol(monkeypatch):
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)
    # Make V2 newer than V3 to prove selection is timestamp-based.
    v2_newer = list(V2_TUPLE)
    v2_newer[3] = 400
    registry = _versioned_registry(v2_latest=tuple(v2_newer))
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", registry)

    data = _parse(await _handle_get_latest_audit({"contract_address": VALID_ADDRESS}))
    assert data["protocol_version"] == "v2"
    assert data["timestamp"] == 400
    assert data["total_count"] == 3
    assert data["execution_status"]["status"] == "SUCCEEDED"


@pytest.mark.asyncio
async def test_latest_live_with_no_versions_is_clean(monkeypatch):
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)
    registry = _versioned_registry(counts=(0, 0, 0))
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", registry)

    data = _parse(await _handle_get_latest_audit({"contract_address": VALID_ADDRESS}))
    assert data["exists"] is False
    assert data["total_count"] == 0
    assert data["counts_by_protocol"] == {"v3": 0, "v2": 0, "v1": 0}
    assert data["execution_status"]["status"] == "CLEAN"


@pytest.mark.asyncio
async def test_history_live_merges_versions_and_sorts_timestamp(monkeypatch):
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)
    registry = _versioned_registry()
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", registry)

    data = _parse(
        await _handle_get_audit_history(
            {"contract_address": VALID_ADDRESS, "limit": 10}
        )
    )
    assert [row["protocol_version"] for row in data["records"]] == ["v3", "v2", "v1"]
    assert [row["timestamp"] for row in data["records"]] == [300, 200, 100]
    assert data["counts_by_protocol"] == {"v3": 1, "v2": 1, "v1": 1}


@pytest.mark.asyncio
async def test_history_limit_is_capped_at_50(monkeypatch):
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)
    many_v3 = []
    for i in range(60):
        row = list(V3_TUPLE)
        row[10] = 1000 + i
        many_v3.append(tuple(row))
    registry = _versioned_registry(v3_history=many_v3, v2_history=[], v1_history=[])
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", registry)

    data = _parse(
        await _handle_get_audit_history(
            {"contract_address": VALID_ADDRESS, "limit": 999}
        )
    )
    assert len(data["records"]) == 50
    assert data["returned"] == 50


@pytest.mark.asyncio
async def test_check_exists_returns_counts_by_protocol(monkeypatch):
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)
    registry = _versioned_registry(counts=(2, 3, 4))
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", registry)

    data = _parse(await _handle_check_audit_exists({"contract_address": VALID_ADDRESS}))
    assert data["exists"] is True
    assert data["total_count"] == 9
    assert data["counts_by_protocol"] == {"v3": 2, "v2": 3, "v1": 4}


@pytest.mark.asyncio
async def test_versioned_rpc_failure_is_structured_unavailable(monkeypatch):
    monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)
    registry = _versioned_registry()
    registry.functions.getAuditCountV3.return_value.call = AsyncMock(
        side_effect=ConnectionError("RPC timeout")
    )
    monkeypatch.setattr("src.mcp.servers.audit_server._registry", registry)

    data = _parse(await _handle_get_latest_audit({"contract_address": VALID_ADDRESS}))
    assert data["error"] == "rpc_error"
    assert data["execution_status"]["status"] == "UNAVAILABLE"
    assert data["execution_status"]["ran"] is False


@pytest.mark.asyncio
async def test_invalid_address_is_structured_failure():
    data = _parse(await _handle_get_latest_audit({"contract_address": INVALID_ADDRESS}))
    assert data["error"] == "invalid_request"
    assert data["execution_status"]["ran"] is False


@pytest.mark.asyncio
async def test_missing_rpc_stays_unavailable_and_never_auto_enables_mock(monkeypatch):
    import src.mcp.servers.audit_server as audit_server

    monkeypatch.setattr(audit_server, "_MOCK_MODE", False)
    monkeypatch.setattr(audit_server, "_RPC_URL", "")
    await _on_startup()
    readiness = _readiness_payload()
    assert audit_server._MOCK_MODE is False
    assert readiness["status"] == "unavailable"
    assert readiness["dependency"]["reason_code"] == "rpc_not_configured"


@pytest.mark.asyncio
async def test_explicit_mock_startup_reports_mock_not_ready(monkeypatch):
    import src.mcp.servers.audit_server as audit_server

    monkeypatch.setattr(audit_server, "_MOCK_MODE", True)
    await _on_startup()
    readiness = _readiness_payload()
    assert readiness["status"] == "mock"
    assert readiness["ready"] is False
