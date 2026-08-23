"""Focused R4-D-010 tests for versioned call-kind graph semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentinel_data.representation.graph_extractor import (
    _CallOperationTypes,
    _add_icfg_edges,
    _classify_v10_call_kinds,
    _compute_has_cei_path,
    _unclassified_v10_call_ir,
)
from sentinel_data.representation.graph_schema_versions import (
    V9_SCHEMA,
    V10_SCHEMA,
    get_graph_schema,
)


def test_v9_vocabulary_remains_immutable() -> None:
    assert V9_SCHEMA.num_edge_types == 12
    assert V9_SCHEMA.edge_types["EXTERNAL_CALL"] == 11
    assert "LOW_LEVEL_CALL" not in V9_SCHEMA.edge_types


def test_v10_call_vocabulary_is_explicit_and_contiguous() -> None:
    assert V10_SCHEMA.num_edge_types == 17
    assert V10_SCHEMA.edge_types["HIGH_LEVEL_CALL"] == 11
    assert V10_SCHEMA.edge_types["LOW_LEVEL_CALL"] == 12
    assert V10_SCHEMA.edge_types["ETHER_TRANSFER"] == 13
    assert V10_SCHEMA.edge_types["ETHER_SEND"] == 14
    assert V10_SCHEMA.edge_types["LIBRARY_CALL"] == 15
    assert V10_SCHEMA.edge_types["CONTRACT_CREATION"] == 16
    assert V10_SCHEMA.call_to_unknown_signal_edge_names == (
        "LOW_LEVEL_CALL",
        "ETHER_SEND",
    )


def test_unknown_schema_fails_closed() -> None:
    try:
        get_graph_schema("v11")
    except ValueError as exc:
        assert "unsupported graph schema" in str(exc)
    else:
        raise AssertionError("unknown graph schema was accepted")


def test_v10_classifier_distinguishes_library_subclass_first() -> None:
    class HighLevel:
        pass

    class Library(HighLevel):
        pass

    class LowLevel:
        pass

    class Transfer:
        pass

    class Send:
        pass

    types = _CallOperationTypes(
        high_level=HighLevel,
        low_level=LowLevel,
        transfer=Transfer,
        send=Send,
        library=Library,
    )
    kinds = _classify_v10_call_kinds(
        [Library(), HighLevel(), LowLevel(), Transfer(), Send(), Library()],
        operation_types=types,
    )
    assert kinds == (
        "HIGH_LEVEL_CALL",
        "LOW_LEVEL_CALL",
        "ETHER_TRANSFER",
        "ETHER_SEND",
        "LIBRARY_CALL",
    )


def test_v10_classifier_emits_contract_creation_explicitly() -> None:
    class NewContract:
        pass

    types = _CallOperationTypes(
        high_level=type("HighLevel", (), {}),
        low_level=type("LowLevel", (), {}),
        transfer=type("Transfer", (), {}),
        send=type("Send", (), {}),
        library=type("Library", (), {}),
        new_contract=NewContract,
    )

    assert _classify_v10_call_kinds(
        [NewContract()], operation_types=types
    ) == ("CONTRACT_CREATION",)


def test_v10_reports_unknown_call_ir_instead_of_relabeling_it() -> None:
    class Call:
        pass

    class Known(Call):
        pass

    class Unknown(Call):
        pass

    types = _CallOperationTypes(
        high_level=Known,
        low_level=Known,
        transfer=Known,
        send=Known,
        library=Known,
        call=Call,
    )
    assert _unclassified_v10_call_ir([Known(), Unknown()], types) == (
        f"{Unknown.__module__}.{Unknown.__qualname__}",
    )


def test_v10_reports_calls_even_when_function_cfg_map_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Node:
        node_id = 7
        irs = [object()]

    class Function:
        canonical_name = "Leaf.callback(address)"
        name = "callback"
        nodes = [Node()]

    class Contract:
        functions = [Function()]

    monkeypatch.setattr(
        "sentinel_data.representation.graph_extractor._classify_v10_call_kinds",
        lambda operations: ("HIGH_LEVEL_CALL",),
    )
    monkeypatch.setattr(
        "sentinel_data.representation.graph_extractor._unclassified_v10_call_ir",
        lambda operations: ("fixture.UnknownCall",),
    )
    names = (
        "HIGH_LEVEL_CALL",
        "LOW_LEVEL_CALL",
        "ETHER_TRANSFER",
        "ETHER_SEND",
        "LIBRARY_CALL",
        "CONTRACT_CREATION",
    )
    classified = {name: 0 for name in names}
    emitted = {name: 0 for name in names}
    unknown: list[dict] = []
    mapping_errors: list[dict] = []
    _add_icfg_edges(
        Contract(),
        {},
        {},
        {},
        [],
        [],
        V10_SCHEMA,
        unknown,
        classified,
        emitted,
        mapping_errors,
    )
    assert unknown[0]["operation_type"] == "fixture.UnknownCall"
    assert classified["HIGH_LEVEL_CALL"] == 1
    assert emitted["HIGH_LEVEL_CALL"] == 0
    assert mapping_errors[0]["reason"] == "missing_function_cfg_map"


def _cei_fixture(interaction_edge_id: int) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    metadata = [
        {"type": "CFG_NODE_CALL"},
        {"type": "CFG_NODE_WRITE"},
    ]
    edge_index = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [interaction_edge_id, V10_SCHEMA.edge_types["CONTROL_FLOW"]],
        dtype=torch.long,
    )
    return metadata, edge_index, edge_attr


def test_v10_cei_excludes_library_call() -> None:
    args = _cei_fixture(V10_SCHEMA.edge_types["LIBRARY_CALL"])
    assert _compute_has_cei_path(*args, graph_schema_version="v10") == 0


def test_v10_cei_includes_real_external_handoff() -> None:
    for edge_name in V10_SCHEMA.external_handoff_edge_names:
        args = _cei_fixture(V10_SCHEMA.edge_types[edge_name])
        assert _compute_has_cei_path(*args, graph_schema_version="v10") == 1


def test_v9_cei_behavior_is_preserved() -> None:
    metadata, edge_index, _ = _cei_fixture(V10_SCHEMA.edge_types["LIBRARY_CALL"])
    edge_attr = torch.tensor(
        [V9_SCHEMA.edge_types["EXTERNAL_CALL"], V9_SCHEMA.edge_types["CONTROL_FLOW"]]
    )
    assert _compute_has_cei_path(
        metadata,
        edge_index,
        edge_attr,
        graph_schema_version="v9",
    ) == 1


def test_v10_real_slither_ir_emits_all_call_kinds(tmp_path: Path) -> None:
    solc_binary = (
        Path.home()
        / ".solc-select/artifacts/solc-0.5.7/solc-0.5.7"
    )
    if not solc_binary.is_file():
        pytest.skip("solc 0.5.7 is unavailable")

    source = tmp_path / "call_kinds.sol"
    source.write_text(
        """pragma solidity ^0.5.7;
library MathLib {
    function add(uint a, uint b) public pure returns (uint) { return a + b; }
}
interface ITarget { function ping() external; }
contract Created { uint public value; }
contract CallKinds {
    ITarget public typedTarget;
    address payable public recipient;
    uint public total;
    function exercise(address payable rawTarget) public {
        total = MathLib.add(total, 1);
        typedTarget.ping();
        recipient.transfer(1);
        recipient.send(1);
    }
    function rawCall(address rawTarget) public { rawTarget.call(""); }
    function rawDelegatecall(address rawTarget) public { rawTarget.delegatecall(""); }
    function rawStaticcall(address rawTarget) public { rawTarget.staticcall(""); }
    function createContract() public { new Created(); }
}
""",
        encoding="utf-8",
    )

    from sentinel_data.representation.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
    )

    graph = extract_contract_graph(
        source,
        GraphExtractionConfig(
            solc_binary=solc_binary,
            solc_version="0.5.7",
            allow_paths=str(tmp_path),
            graph_schema_version="v10",
        ),
    )
    assert graph.graph_schema_version == "v10"
    assert graph.representation_extractor_version == V10_SCHEMA.extractor_version
    assert graph.unclassified_call_ir == []
    assert graph.call_mapping_errors == []
    assert graph.classified_call_ir_counts == graph.emitted_call_edge_counts
    observed = set(graph.edge_attr.tolist())
    for edge_name in (
        "HIGH_LEVEL_CALL",
        "LOW_LEVEL_CALL",
        "ETHER_TRANSFER",
        "ETHER_SEND",
        "LIBRARY_CALL",
        "CONTRACT_CREATION",
    ):
        assert V10_SCHEMA.edge_types[edge_name] in observed, edge_name
    assert int(
        (graph.edge_attr == V10_SCHEMA.edge_types["LOW_LEVEL_CALL"]).sum()
    ) == 3


def test_v10_does_not_swallow_unexpected_icfg_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solc_binary = Path.home() / ".solc-select/artifacts/solc-0.5.7/solc-0.5.7"
    if not solc_binary.is_file():
        pytest.skip("solc 0.5.7 is unavailable")
    source = tmp_path / "fail_closed.sol"
    source.write_text(
        "pragma solidity ^0.5.7; contract FailClosed { function f() public {} }\n",
        encoding="utf-8",
    )

    from sentinel_data.representation.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
    )

    def fail_icfg(*args, **kwargs):
        raise RuntimeError("synthetic ICFG failure")

    monkeypatch.setattr(
        "sentinel_data.representation.graph_extractor._add_icfg_edges",
        fail_icfg,
    )
    with pytest.raises(RuntimeError, match="synthetic ICFG failure"):
        extract_contract_graph(
            source,
            GraphExtractionConfig(
                solc_binary=solc_binary,
                solc_version="0.5.7",
                allow_paths=str(tmp_path),
                graph_schema_version="v10",
            ),
        )


def test_v10_legacy_callcode_maps_to_low_level_call(tmp_path: Path) -> None:
    solc_binary = Path.home() / ".solc-select/artifacts/solc-0.4.25/solc-0.4.25"
    if not solc_binary.is_file():
        pytest.skip("solc 0.4.25 is unavailable")
    source = tmp_path / "callcode.sol"
    source.write_text(
        """pragma solidity ^0.4.25;
contract LegacyCallcode {
    function invoke(address target) public { target.callcode(""); }
}
""",
        encoding="utf-8",
    )
    from sentinel_data.representation.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
    )

    graph = extract_contract_graph(
        source,
        GraphExtractionConfig(
            solc_binary=solc_binary,
            solc_version="0.4.25",
            allow_paths=str(tmp_path),
            graph_schema_version="v10",
        ),
    )
    assert int(
        (graph.edge_attr == V10_SCHEMA.edge_types["LOW_LEVEL_CALL"]).sum()
    ) == 1


def test_v10_imported_using_for_maps_to_library_call(tmp_path: Path) -> None:
    solc_binary = Path.home() / ".solc-select/artifacts/solc-0.5.7/solc-0.5.7"
    if not solc_binary.is_file():
        pytest.skip("solc 0.5.7 is unavailable")

    (tmp_path / "math.sol").write_text(
        """pragma solidity ^0.5.7;
library ImportedMath {
    function plus(uint a, uint b) public pure returns (uint) { return a + b; }
}
""",
        encoding="utf-8",
    )
    source = tmp_path / "using_for.sol"
    source.write_text(
        """pragma solidity ^0.5.7;
import "./math.sol";
contract UsesImportedLibrary {
    using ImportedMath for uint;
    uint public total;
    function increment() public { total = total.plus(1); }
}
""",
        encoding="utf-8",
    )

    from sentinel_data.representation.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
    )

    graph = extract_contract_graph(
        source,
        GraphExtractionConfig(
            solc_binary=solc_binary,
            solc_version="0.5.7",
            allow_paths=str(tmp_path),
            graph_schema_version="v10",
        ),
    )
    assert V10_SCHEMA.edge_types["LIBRARY_CALL"] in set(graph.edge_attr.tolist())
    assert V10_SCHEMA.edge_types["HIGH_LEVEL_CALL"] not in set(graph.edge_attr.tolist())


def test_v10_inherited_typed_callback_is_not_lost_to_canonical_map_collision(
    tmp_path: Path,
) -> None:
    solc_binary = Path.home() / ".solc-select/artifacts/solc-0.4.25/solc-0.4.25"
    if not solc_binary.is_file():
        pytest.skip("solc 0.4.25 is unavailable")
    source = tmp_path / "inherited_callback.sol"
    source.write_text(
        """pragma solidity ^0.4.25;
contract Recipient { function notify(uint value) public; }
contract Base {
    function callback(address destination, uint value) public {
        Recipient(destination).notify(value);
    }
}
contract Leaf is Base { uint public marker; }
""",
        encoding="utf-8",
    )
    from sentinel_data.representation.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
    )

    graph = extract_contract_graph(
        source,
        GraphExtractionConfig(
            solc_binary=solc_binary,
            solc_version="0.4.25",
            allow_paths=str(tmp_path),
            multi_contract_policy="by_name",
            target_contract_name="Leaf",
            graph_schema_version="v10",
        ),
    )
    assert int(
        (graph.edge_attr == V10_SCHEMA.edge_types["HIGH_LEVEL_CALL"]).sum()
    ) >= 1
