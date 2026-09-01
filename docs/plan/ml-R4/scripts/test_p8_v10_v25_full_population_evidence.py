"""Focused duplicate/multiplicity tests for full-population V10 evidence."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLECT = _load("p8_collect_v10_v25_full_population_write_evidence")
PROBE = _load("p8_probe_v10_v25_full_population")
GENERATE = _load("p8_generate_v10_v25_structural_repeat")


def _graph(types: list[str]):
    type_ids = {
        "CFG_NODE_OTHER": 12.0 / 13.0,
        "CFG_NODE_WRITE": 9.0 / 13.0,
    }
    return SimpleNamespace(
        x=torch.tensor([[type_ids[node_type], 0.25] for node_type in types]),
        node_metadata=[
            {
                "name": "EXPRESSION alias.value = 1",
                "type": node_type,
                "source_lines": [10],
            }
            for node_type in types
        ],
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([6, 6]),
    )


def test_target_derivation_preserves_duplicate_multiplicity(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    (reference_root / "dive").mkdir(parents=True)
    (candidate_root / "dive").mkdir(parents=True)
    contract_id = "a" * 64
    torch.save(_graph(["CFG_NODE_OTHER", "CFG_NODE_OTHER"]), reference_root / "dive" / f"{contract_id}.pt")
    torch.save(_graph(["CFG_NODE_WRITE", "CFG_NODE_WRITE"]), candidate_root / "dive" / f"{contract_id}.pt")

    targets, unexplained = COLLECT._derive_targets(
        reference_root=reference_root,
        candidate_root=candidate_root,
        identities=[f"dive/{contract_id}"],
    )

    assert unexplained == []
    assert targets[f"dive/{contract_id}"][0]["reference_multiplicity"] == 2
    assert targets[f"dive/{contract_id}"][0]["candidate_multiplicity"] == 2


def test_canonicalization_updates_every_duplicate_occurrence() -> None:
    graph = _graph(["CFG_NODE_OTHER", "CFG_NODE_OTHER"])
    target = {
        "name": "EXPRESSION alias.value = 1",
        "source_lines": [10],
        "coarse_type": "CFG_NODE",
        "reference_multiplicity": 2,
        "candidate_multiplicity": 2,
    }
    result = PROBE._canonicalize(graph, "dive/example", [target])
    assert [row["type"] for row in result.node_metadata] == [
        "CFG_NODE_WRITE",
        "CFG_NODE_WRITE",
    ]
    assert torch.allclose(result.x[:, 0], torch.tensor([9.0 / 13.0] * 2))
    assert [row["type"] for row in graph.node_metadata] == [
        "CFG_NODE_OTHER",
        "CFG_NODE_OTHER",
    ]


def test_canonicalization_fails_on_multiplicity_mismatch() -> None:
    target = {
        "name": "EXPRESSION alias.value = 1",
        "source_lines": [10],
        "coarse_type": "CFG_NODE",
        "reference_multiplicity": 2,
        "candidate_multiplicity": 2,
    }
    with pytest.raises(ValueError, match="resolves to 1 nodes"):
        PROBE._canonicalize(_graph(["CFG_NODE_OTHER"]), "dive/example", [target])


def test_exact_runtime_guard_rejects_wrong_environment(monkeypatch) -> None:
    versions = {"slither-analyzer": "0.11.5", "crytic-compile": "0.3.11"}
    monkeypatch.setattr(
        GENERATE.importlib.metadata, "version", lambda name: versions[name]
    )
    with pytest.raises(RuntimeError, match="exact slither-analyzer 0.10.0"):
        GENERATE._require_runtime()


def test_storage_resolution_accepts_slither_default_location() -> None:
    assert COLLECT._persistent_root(
        {
            "class": "LocalVariable",
            "location": "default",
            "is_storage": True,
        }
    )
