"""Focused tests for V9-to-V10 transition-audit diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).with_name("p8_audit_v10_transition.py")
SPEC = importlib.util.spec_from_file_location("p8_audit_v10_transition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_only_source_syntax_hits_are_explicitly_diagnostic(tmp_path: Path) -> None:
    source = tmp_path / "fixture.sol"
    source.write_text(
        """contract Fixture {
    function f(address payable target) public {
        target.call("");
        target.call.value(1)("");
        target.delegatecall("");
        target.transfer(1);
        target.send(1);
        new Fixture();
    }
}
""",
        encoding="utf-8",
    )

    assert MODULE._parse_only_source_syntax_hits(source) == {
        "raw_low_level": 3,
        "ether_transfer": 1,
        "ether_send": 1,
        "contract_creation": 1,
    }


def test_historical_v9_normal_mode_defaults_to_full_analysis() -> None:
    assert MODULE._historical_v9_extraction_mode({}) == "slither_full_analysis"
    assert (
        MODULE._historical_v9_extraction_mode(
            {"graph_extraction_mode": "slither_parse_only"}
        )
        == "slither_parse_only"
    )


def test_unchanged_edge_topology_comparison_is_order_independent() -> None:
    left = SimpleNamespace(
        edge_attr=torch.tensor([8, 6, 8, 12]),
        edge_index=torch.tensor([[1, 0, 1, 3], [2, 1, 2, 3]]),
    )
    right = SimpleNamespace(
        edge_attr=torch.tensor([12, 8, 6, 8]),
        edge_index=torch.tensor([[3, 1, 0, 1], [3, 2, 1, 2]]),
    )

    assert MODULE._edge_topology_equal_through(left, right, 10) is True
    right.edge_index[1, 3] = 4
    assert MODULE._edge_topology_equal_through(left, right, 10) is False
