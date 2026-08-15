"""Tests for fail-closed R4 graph target selection."""

from __future__ import annotations

import pytest

from sentinel_data.representation.target_selector import (
    TargetSelectionError,
    declarations,
    resolve_file_graph_targets,
    resolve_target_contract,
)


def test_safe_math_library_is_not_selected_over_application_contract():
    source = """
    library SafeMath { function add(uint a, uint b) internal pure returns(uint){return a+b;} }
    contract Vault { function deposit() public {} }
    """
    assert resolve_target_contract(source) == "Vault"


def test_interface_and_library_only_file_fails_closed():
    source = "interface IToken { } library SafeMath { }"
    with pytest.raises(TargetSelectionError, match="no application contract"):
        resolve_target_contract(source)


def test_multiple_application_contracts_require_explicit_provenance():
    source = "contract A {} contract B {}"
    with pytest.raises(TargetSelectionError, match="multiple application contracts"):
        resolve_target_contract(source)


def test_explicit_target_selects_requested_application_contract():
    source = "contract A {} contract B {}"
    assert resolve_target_contract(source, explicit_target="A") == "A"


def test_unknown_explicit_target_fails_instead_of_falling_back():
    source = "library SafeMath {} contract Vault {}"
    with pytest.raises(TargetSelectionError, match="not found"):
        resolve_target_contract(source, explicit_target="Missing")


def test_explicit_library_target_is_rejected():
    source = "library SafeMath {} contract Vault {}"
    with pytest.raises(TargetSelectionError, match="library"):
        resolve_target_contract(source, explicit_target="SafeMath")


def test_unique_provenance_contract_name_resolves_multiple_contract_file():
    source = "contract A {} contract B {}"
    assert resolve_target_contract(
        source,
        provenance_contract_names=("B",),
    ) == "B"


def test_unique_inheritance_leaf_resolves_application_contract():
    source = "contract Context {} contract Ownable is Context {} contract Vault is Ownable {}"
    assert resolve_target_contract(source) == "Vault"


def test_inheritance_constructor_arguments_do_not_split_base_names():
    source = "contract A {} contract B {} contract C is A(1, 2), B {}"
    items = declarations(source)
    assert items[-1].base_names == ("A", "B")
    assert resolve_target_contract(source) == "C"


def test_file_graph_retains_all_unrelated_application_leaves():
    source = "contract Helper {} contract Main {}"
    assert resolve_file_graph_targets(source) == ("Helper", "Main")


def test_library_only_file_retains_executable_library():
    source = "interface IToken {} library Math { function add() internal {} }"
    assert resolve_file_graph_targets(source) == ("Math",)


def test_declaration_words_in_comments_and_strings_are_ignored():
    source = '''
    // contract FakeComment {}
    contract Real {
        string constant S = "library FakeString {}";
    }
    '''
    items = declarations(source)
    assert [(item.kind, item.name) for item in items] == [("contract", "Real")]
