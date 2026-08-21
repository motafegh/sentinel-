from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("p8_audit_external_call_semantics.py")
SPEC = importlib.util.spec_from_file_location("p8_external_call_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_declared_library_names_are_unique_and_sorted() -> None:
    source = "library Zed {} library SafeMath {} library Zed {}"
    assert MODULE.declared_library_names(source) == ("SafeMath", "Zed")


def test_classify_edge_name_distinguishes_library_and_transfer() -> None:
    library = MODULE.classify_edge_name(
        "EXPRESSION total = SafeMath.add(total,amount)", ("SafeMath",)
    )
    assert library == {
        "declared_library": True,
        "transfer": False,
        "send": False,
        "raw_low_level": False,
    }

    transfer = MODULE.classify_edge_name(
        "EXPRESSION msg.sender.transfer(amount)", ("SafeMath",)
    )
    assert transfer["declared_library"] is False
    assert transfer["transfer"] is True
    assert transfer["send"] is False
    assert transfer["raw_low_level"] is False


def test_classify_edge_name_recognizes_raw_low_level_forms() -> None:
    for name in (
        "EXPRESSION target.call(data)",
        "EXPRESSION target.call.value(amount)(data)",
        "EXPRESSION target.delegatecall(data)",
        "EXPRESSION target.send(amount)",
    ):
        assert MODULE.classify_edge_name(name, ())["raw_low_level"] is True
