"""R4-D-010 generation guards and accepted-token reuse tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from sentinel_data.preprocessing.r4_versions import (
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.representation import r4_orchestrator as orchestrator
from sentinel_data.representation.r4_compatibility import (
    CompatibilityExtraction,
    FULL_ANALYSIS,
)


def _token_payload(contract_id: str, source: str) -> dict:
    return {
        "input_ids": torch.zeros((4, 512), dtype=torch.long),
        "attention_mask": torch.zeros((4, 512), dtype=torch.long),
        "sha256": contract_id,
        "source": source,
        "num_windows": 4,
        "stride": 256,
        "num_tokens": 0,
        "tokenizer_name": "fixture",
        "max_length": 512,
        "coverage_schema_version": "fixture-v1",
        "pre_subsampling_window_count": 4,
        "pre_subsampling_code_tokens": 0,
        "selected_window_indices": [0, 1, 2, 3],
        "selected_code_token_ranges": [],
        "retained_unique_code_tokens": 0,
        "retained_token_ratio": 0.0,
        "content_tokens_per_window": 510,
        "coverage_interpretation": "fixture",
    }


def test_v10_public_generation_refuses_wrong_root(tmp_path: Path) -> None:
    accepted_tokens = tmp_path / "tokens"
    accepted_tokens.mkdir()
    with pytest.raises(ValueError, match="v10 output must be under"):
        orchestrator.represent_repaired_source(
            "dive",
            tmp_path / "preprocessed",
            tmp_path / "wrong-root" / "dive",
            graph_schema_version="v10",
            extractor_version=V10_REPRESENTATION_EXTRACTOR_VERSION,
            accepted_tokens_dir=accepted_tokens,
        )


def test_v10_extract_copies_accepted_token_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_name = "dive"
    contract_id = "a" * 64
    source_path = tmp_path / f"{contract_id}.sol"
    source_path.write_text("contract CallKinds {}\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    token_dir = tmp_path / "accepted"
    output_dir.mkdir()
    token_dir.mkdir()

    token_path = token_dir / f"{contract_id}.tokens.pt"
    torch.save(_token_payload(contract_id, source_name), token_path)
    original_token_bytes = token_path.read_bytes()

    graph = Data(
        x=torch.zeros((1, 12)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0,), dtype=torch.long),
    )
    graph.node_metadata = [{"name": "CallKinds", "type": "CONTRACT", "source_lines": [1]}]
    graph.contract_name = "CallKinds"
    graph.num_nodes = 1
    graph.num_edges = 0
    graph.has_cei_path = 0
    graph.graph_schema_version = "v10"
    graph.representation_extractor_version = V10_REPRESENTATION_EXTRACTOR_VERSION
    graph.unclassified_call_ir = []
    graph.classified_call_ir_counts = {
        name: 0
        for name in (
            "HIGH_LEVEL_CALL",
            "LOW_LEVEL_CALL",
            "ETHER_TRANSFER",
            "ETHER_SEND",
            "LIBRARY_CALL",
            "CONTRACT_CREATION",
        )
    }
    graph.emitted_call_edge_counts = dict(graph.classified_call_ir_counts)
    graph.call_mapping_errors = []

    monkeypatch.setattr(orchestrator, "_select_targets", lambda *_: ("CallKinds",))
    monkeypatch.setattr(orchestrator, "_resolve_solc_binary", lambda *_: None)
    monkeypatch.setattr(
        orchestrator,
        "extract_components_with_compatibility",
        lambda *args, **kwargs: CompatibilityExtraction(
            graphs=(graph,),
            actual_targets=("CallKinds",),
            mode=FULL_ANALYSIS,
            fallback_errors=(),
        ),
    )

    orchestrator._extract_one(
        source_name,
        source_path,
        {
            "sha256": contract_id,
            "solc_version": "0.5.7",
            "meta_schema_version": "2",
        },
        output_dir,
        graph_schema_version="v10",
        extractor_version=V10_REPRESENTATION_EXTRACTOR_VERSION,
        accepted_tokens_dir=token_dir,
    )

    assert (output_dir / f"{contract_id}.tokens.pt").read_bytes() == original_token_bytes
    sidecar = json.loads(
        (output_dir / f"{contract_id}.rep.json").read_text(encoding="utf-8")
    )
    assert sidecar["schema_version"] == "v10"
    assert sidecar["extractor_version"] == V10_REPRESENTATION_EXTRACTOR_VERSION
    assert sidecar["token_lineage"] == "accepted_v9_byte_copy"
    assert sidecar["unclassified_call_ir_count"] == 0
    assert sidecar["classified_call_ir_counts"] == sidecar["emitted_call_edge_counts"]
