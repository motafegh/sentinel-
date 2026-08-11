from __future__ import annotations

import asyncio
import json
import sys

from src.mcp.servers.audit._readonly_handlers import READ_ONLY_TOOLS, call_tool


def test_live_audit_tool_set_is_exactly_read_only() -> None:
    assert READ_ONLY_TOOLS == {
        "get_latest_audit",
        "get_audit_history",
        "check_audit_exists",
    }
    assert "submit_audit" not in READ_ONLY_TOOLS


def test_submit_audit_is_rejected_before_legacy_submit_module_import() -> None:
    sys.modules.pop("src.mcp.servers.audit._submit", None)

    result = asyncio.run(
        call_tool(
            "submit_audit",
            {
                "source_code": "pragma solidity ^0.8.20; contract T {}",
                "contract_address": "0x0000000000000000000000000000000000000001",
                "model_hash": "00" * 32,
            },
        )
    )

    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["status"] == "policy_rejected"
    assert payload["failed_step"] == "tool_dispatch"
    assert payload["reason_code"] == "read_only_service"
    assert payload["attempted"] is False
    assert payload["tool"] == "submit_audit"
    assert "src.mcp.servers.audit._submit" not in sys.modules


def test_unknown_write_like_name_is_also_rejected() -> None:
    result = asyncio.run(call_tool("submit_audit_v3", {}))
    payload = json.loads(result[0].text)
    assert payload["status"] == "policy_rejected"
    assert payload["attempted"] is False
