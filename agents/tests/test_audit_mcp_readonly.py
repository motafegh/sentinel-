from __future__ import annotations

import asyncio
import json
import sys

from src.mcp.servers.audit import _readonly_handlers as readonly


def test_live_audit_tool_set_is_exactly_read_only() -> None:
    assert readonly.READ_ONLY_TOOLS == {
        "get_latest_audit",
        "get_audit_history",
        "check_audit_exists",
    }
    assert "submit_audit" not in readonly.READ_ONLY_TOOLS


def test_public_shim_exports_the_read_only_server() -> None:
    from src.mcp.servers import audit_server

    assert audit_server.server is readonly.server
    assert audit_server.call_tool is readonly.call_tool
    assert audit_server.list_tools is readonly.list_tools
    assert not hasattr(audit_server, "_handle_submit_audit")


def test_submit_audit_is_rejected_before_legacy_submit_module_import() -> None:
    sys.modules.pop("src.mcp.servers.audit._submit", None)

    result = asyncio.run(
        readonly.call_tool(
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
    result = asyncio.run(readonly.call_tool("submit_audit_v3", {}))
    payload = json.loads(result[0].text)
    assert payload["status"] == "policy_rejected"
    assert payload["attempted"] is False
