"""Focused tests for the R4 V10 deterministic CFG classification seam."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sentinel_data.representation import graph_extractor
from sentinel_data.representation.v10_cfg_determinism import (
    _call_mutates_persistent_storage,
    _expression_writes_persistent_storage,
    v10_deterministic_cfg_classification,
)


def _identifier(variable):
    from slither.core.expressions.identifier import Identifier

    return Identifier(variable)


def _member(variable, name: str = "field"):
    from slither.core.expressions.member_access import MemberAccess

    return MemberAccess(name, "uint256", _identifier(variable))


def _node(*written, declaration=None):
    return SimpleNamespace(
        expression=None,
        variables_written_as_expression=list(written),
        variable_declaration=declaration,
    )


def _state_variable(name: str = "state"):
    from slither.core.variables.state_variable import StateVariable

    variable = StateVariable()
    variable.name = name
    return variable


def _member_call(variable, method: str = "push"):
    from slither.core.expressions.call_expression import CallExpression
    from slither.core.expressions.member_access import MemberAccess

    receiver = _member(variable, "items")
    called = MemberAccess(method, "function ()", receiver)
    return CallExpression(called, [], "tuple()")


def _call_node(variable, method: str = "push"):
    return SimpleNamespace(
        expression=_member_call(variable, method),
        variables_written_as_expression=[],
        variable_declaration=None,
    )


def _local_variable(name: str, location: str):
    from slither.core.variables.local_variable import LocalVariable

    variable = LocalVariable()
    variable.name = name
    variable.set_location(location)
    return variable


def test_direct_state_identifier_is_persistent_write() -> None:
    state = _state_variable()
    assert _expression_writes_persistent_storage(_node(_identifier(state))) is True


def test_state_member_is_persistent_write() -> None:
    state = _state_variable()
    assert _expression_writes_persistent_storage(_node(_member(state))) is True


def test_storage_local_member_is_persistent_write() -> None:
    alias = _local_variable("alias", "storage")
    assert _expression_writes_persistent_storage(_node(_member(alias))) is True


def test_memory_local_member_is_not_persistent_write() -> None:
    local = _local_variable("copy", "memory")
    assert _expression_writes_persistent_storage(_node(_member(local))) is False


def test_storage_reference_declaration_is_not_state_mutation() -> None:
    alias = _local_variable("alias", "storage")
    assert (
        _expression_writes_persistent_storage(
            _node(_identifier(alias), declaration=alias)
        )
        is False
    )


def test_bare_storage_local_rebinding_is_not_promoted_to_write() -> None:
    alias = _local_variable("alias", "storage")
    assert _expression_writes_persistent_storage(_node(_identifier(alias))) is False


@pytest.mark.parametrize("method", ["push", "pop"])
def test_storage_collection_mutator_is_persistent_write(method: str) -> None:
    alias = _local_variable("alias", "storage")
    assert _call_mutates_persistent_storage(_call_node(alias, method)) is True


def test_state_collection_mutator_is_persistent_write() -> None:
    state = _state_variable("items")
    assert _call_mutates_persistent_storage(_call_node(state)) is True


def test_memory_collection_receiver_is_not_promoted() -> None:
    local = _local_variable("copy", "memory")
    assert _call_mutates_persistent_storage(_call_node(local)) is False


def test_arbitrary_storage_member_method_is_not_promoted() -> None:
    alias = _local_variable("alias", "storage")
    assert _call_mutates_persistent_storage(_call_node(alias, "append")) is False


def test_v10_guard_promotes_storage_collection_mutator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = _local_variable("alias", "storage")
    node = _call_node(alias)
    arithmetic_type = graph_extractor.NODE_TYPES["CFG_NODE_ARITH"]
    write_type = graph_extractor.NODE_TYPES["CFG_NODE_WRITE"]

    def unstable_classifier(_node):
        return arithmetic_type

    monkeypatch.setattr(graph_extractor, "_cfg_node_type", unstable_classifier)
    with v10_deterministic_cfg_classification():
        assert graph_extractor._cfg_node_type(node) == write_type


def test_v10_guard_promotes_stable_storage_member_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = _local_variable("alias", "storage")
    node = _node(_member(alias))
    read_type = graph_extractor.NODE_TYPES["CFG_NODE_READ"]
    write_type = graph_extractor.NODE_TYPES["CFG_NODE_WRITE"]

    def unstable_classifier(_node):
        return read_type

    monkeypatch.setattr(graph_extractor, "_cfg_node_type", unstable_classifier)
    with v10_deterministic_cfg_classification():
        assert graph_extractor._cfg_node_type(node) == write_type
    assert graph_extractor._cfg_node_type is unstable_classifier


def test_v10_guard_preserves_call_priority_and_restores_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state_variable()
    node = _node(_identifier(state))
    call_type = graph_extractor.NODE_TYPES["CFG_NODE_CALL"]

    def call_classifier(_node):
        return call_type

    monkeypatch.setattr(graph_extractor, "_cfg_node_type", call_classifier)
    with pytest.raises(RuntimeError, match="fixture failure"):
        with v10_deterministic_cfg_classification():
            assert graph_extractor._cfg_node_type(node) == call_type
            raise RuntimeError("fixture failure")
    assert graph_extractor._cfg_node_type is call_classifier
