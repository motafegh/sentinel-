"""Focused tests for the protected V10 generation runtime boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("p8_generate_v10_candidate.py")
SPEC = importlib.util.spec_from_file_location("p8_generate_v10_candidate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_slither_runtime_rejects_stale_generation_environment(monkeypatch) -> None:
    versions = {"slither-analyzer": "0.11.5", "crytic-compile": "0.3.11"}
    monkeypatch.setattr(
        MODULE.importlib.metadata, "version", lambda name: versions[name]
    )

    with pytest.raises(RuntimeError, match="exact slither-analyzer 0.10.0; found 0.11.5"):
        MODULE._require_slither_runtime("0.10.0")


def test_slither_runtime_records_accepted_generation_environment(monkeypatch) -> None:
    versions = {"slither-analyzer": "0.10.0", "crytic-compile": "0.3.11"}
    monkeypatch.setattr(
        MODULE.importlib.metadata, "version", lambda name: versions[name]
    )

    assert MODULE._require_slither_runtime("0.10.0") == {
        "slither_analyzer": "0.10.0",
        "crytic_compile": "0.3.11",
        "required_slither_analyzer": "0.10.0",
    }


def test_exception_regression_selects_identity_bound_runtime() -> None:
    args = MODULE.argparse.Namespace(
        mode="regression",
        contract_id=[
            "caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9"
        ],
    )
    assert MODULE._required_generation_slither(args) == "0.11.5"


def test_regression_rejects_mixed_runtime_identities() -> None:
    args = MODULE.argparse.Namespace(
        mode="regression",
        contract_id=[
            "caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9",
            "a" * 64,
        ],
    )
    with pytest.raises(ValueError, match="cannot mix primary and exception"):
        MODULE._required_generation_slither(args)
