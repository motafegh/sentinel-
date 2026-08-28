from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("p8_probe_v10_cfg_write_evidence.py")
SPEC = importlib.util.spec_from_file_location("p8_probe_v10_cfg_write_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _node(*, expression_writes, state_written, ir_lvalues):
    return {
        "name": "EXPRESSION item.value = 1",
        "source_lines": [10],
        "function": "Fixture.write()",
        "node_type": "NodeType.EXPRESSION",
        "variable_declaration": None,
        "expression_writes": expression_writes,
        "state_variables_written": state_written,
        "state_variables_read": [],
        "ir_lvalues": ir_lvalues,
    }


def test_duplicate_node_views_merge_as_sorted_unions() -> None:
    identifier = {"class": "Identifier", "text": "item", "root_variable": None}
    member = {"class": "MemberAccess", "text": "item.value", "root_variable": None}
    assignment = {"operation": "Assignment", "lvalue": "REF_1"}
    member_ir = {"operation": "Member", "lvalue": "REF_1"}

    merged = MODULE._merge_node_record(
        _node(
            expression_writes=[member, identifier],
            state_written=[],
            ir_lvalues=[member_ir],
        ),
        _node(
            expression_writes=[identifier],
            state_written=["Fixture.items"],
            ir_lvalues=[assignment, member_ir],
        ),
    )

    assert merged["expression_writes"] == MODULE._canonical_records(
        [identifier, member]
    )
    assert merged["ir_lvalues"] == MODULE._canonical_records([assignment, member_ir])
    assert merged["state_variables_written"] == ["Fixture.items"]


def test_duplicate_node_conflict_fails_closed() -> None:
    left = _node(expression_writes=[], state_written=[], ir_lvalues=[])
    right = {**left, "function": "Other.write()"}

    with pytest.raises(ValueError, match="conflicting duplicate-node field function"):
        MODULE._merge_node_record(left, right)


def test_stable_state_writes_ignore_ir_alias_observations() -> None:
    rows = [
        {
            "class": "Identifier",
            "text": "items",
            "root_variable": {
                "class": "StateVariable",
                "name": "items",
                "location": None,
                "is_storage": True,
            },
        },
        {
            "class": "MemberAccess",
            "text": "item.value",
            "root_variable": {
                "class": "LocalVariable",
                "name": "item",
                "location": "storage",
                "is_storage": True,
            },
        },
    ]

    assert MODULE._stable_state_variables_written(rows) == ["items"]
