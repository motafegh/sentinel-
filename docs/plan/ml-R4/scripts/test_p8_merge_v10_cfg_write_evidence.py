from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT = SCRIPT_DIR / "p8_merge_v10_cfg_write_evidence.py"
SPEC = importlib.util.spec_from_file_location(
    "p8_merge_v10_cfg_write_evidence",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def _node(name: str, line: int, *, root_name: str = "alias") -> dict:
    return {
        "name": name,
        "source_lines": [line],
        "function": "Fixture.f()",
        "node_type": "EXPRESSION",
        "variable_declaration": None,
        "expression_writes": [
            {
                "class": "MemberAccess",
                "text": name,
                "root_variable": {
                    "class": "LocalVariable",
                    "name": root_name,
                    "location": "storage",
                    "is_storage": True,
                },
            }
        ],
        "state_variables_written": [],
        "state_variables_read": [],
        "ir_lvalues": [],
    }


def _report(logical: str, nodes: list[dict], *, slither: str = "0.10.0") -> dict:
    return {
        "schema": merge.SCHEMA,
        "slither_analyzer": slither,
        "contracts_requested": 1,
        "contracts": [
            {
                "contract": logical,
                "requested_nodes": len(nodes),
                "observed_nodes": len(nodes),
                "missing_nodes": [],
                "nodes": nodes,
            }
        ],
        "all_requested_nodes_found": True,
        "physical_acceptance": False,
        "training_authorized": False,
    }


def _write(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def test_merges_new_nodes_into_existing_contract(tmp_path: Path) -> None:
    logical = "dive/fixture"
    base = tmp_path / "base.json"
    extension = tmp_path / "extension.json"
    _write(base, _report(logical, [_node("EXPRESSION alias.a = 1", 10)]))
    _write(extension, _report(logical, [_node("EXPRESSION alias.b = 2", 11)]))

    result = merge.merge_reports([base, extension])

    assert result["contracts_requested"] == 1
    contract = result["contracts"][0]
    assert contract["requested_nodes"] == 2
    assert contract["observed_nodes"] == 2
    assert [row["source_lines"] for row in contract["nodes"]] == [[10], [11]]
    assert len(result["source_reports"]) == 2


def test_identical_duplicate_evidence_is_deduplicated(tmp_path: Path) -> None:
    logical = "dive/fixture"
    row = _node("EXPRESSION alias.a = 1", 10)
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write(left, _report(logical, [row]))
    _write(right, _report(logical, [row]))

    result = merge.merge_reports([left, right])
    assert result["contracts"][0]["requested_nodes"] == 1


def test_conflicting_duplicate_evidence_fails_closed(tmp_path: Path) -> None:
    logical = "dive/fixture"
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write(left, _report(logical, [_node("EXPRESSION alias.a = 1", 10)]))
    _write(
        right,
        _report(
            logical,
            [_node("EXPRESSION alias.a = 1", 10, root_name="different_alias")],
        ),
    )

    with pytest.raises(ValueError, match="conflicting evidence"):
        merge.merge_reports([left, right])


def test_wrong_slither_version_fails_closed(tmp_path: Path) -> None:
    logical = "dive/fixture"
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    _write(good, _report(logical, [_node("EXPRESSION alias.a = 1", 10)]))
    _write(
        bad,
        _report(
            logical,
            [_node("EXPRESSION alias.b = 2", 11)],
            slither="0.11.5",
        ),
    )

    with pytest.raises(ValueError, match="exact Slither 0.10.0"):
        merge.merge_reports([good, bad])
