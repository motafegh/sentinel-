"""Focused D1-D3 tests for the R4-D-012 guarded-token candidate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.src.data_extraction.bounded_window_selector import (
    CONTROL_STRATEGY,
    GUARDED_STRATEGY,
    target_aware_greedy_indices,
    window_ranges,
)
from ml.src.data_extraction.guarded_window_tokenizer import (
    FALLBACK_NOT_STRICTLY_BETTER,
    GuardedWindowTokenizationError,
    tokenize_guarded_source,
)
from sentinel_data.preprocessing.r4_versions import (
    GUARDED_REPRESENTATION_ROOT_NAME,
    GUARDED_SELECTOR_POLICY_VERSION,
    GUARDED_TOKEN_LINEAGE_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.representation.r4_guarded_token_candidate import (
    PARENT_TOKEN_LINEAGE,
    build_guarded_candidate_identity,
    build_guarded_candidate_source,
)


class FakeTokenizer:
    pad_token_id = 0

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, code, **kwargs):
        total_tokens = len(code)
        if kwargs.get("add_special_tokens") is False:
            result = {"input_ids": list(range(total_tokens))}
            if kwargs.get("return_offsets_mapping"):
                result["offset_mapping"] = [
                    (index, index + 1) for index in range(total_tokens)
                ]
            return result

        ranges = window_ranges(
            total_tokens,
            content_capacity=510,
            stride=256,
        )
        rows = []
        masks = []
        for index, (start, end) in enumerate(ranges):
            row = torch.full((512,), index + 1, dtype=torch.long)
            mask = torch.zeros((512,), dtype=torch.long)
            mask[: min(512, (end - start) + 2)] = 1
            rows.append(row)
            masks.append(mask)
        return {
            "input_ids": torch.stack(rows),
            "attention_mask": torch.stack(masks),
        }


def test_guarded_over_cap_uses_strict_target_coverage_improvement():
    source = "x" * 2200
    result = tokenize_guarded_source(
        source,
        target_char_spans=[[250, 350], [1050, 1150]],
        tokenizer=FakeTokenizer(),
    )

    assert tuple(result["input_ids"].shape) == (4, 512)
    assert result["selector"]["requested_policy"] == GUARDED_STRATEGY
    assert result["selector"]["used_control_fallback"] is False
    assert result["selector"]["target_coverage_tokens"] > (
        result["selector"]["control_target_coverage_tokens"]
    )
    assert result["selected_window_indices"] == result["selector"]["candidate_indices"]


def test_guarded_under_cap_falls_back_to_historical_control_and_pads():
    result = tokenize_guarded_source(
        "x" * 100,
        target_char_spans=[[10, 20]],
        tokenizer=FakeTokenizer(),
    )

    assert tuple(result["input_ids"].shape) == (4, 512)
    assert result["selected_window_indices"] == [0]
    assert result["selector"]["effective_path"] == CONTROL_STRATEGY
    assert result["selector"]["used_control_fallback"] is True
    assert result["selector"]["fallback_reason"] == FALLBACK_NOT_STRICTLY_BETTER
    assert torch.count_nonzero(result["attention_mask"][1:]) == 0


def test_greedy_equal_gain_tie_breaks_to_lowest_window_index():
    assert target_aware_greedy_indices(
        [[0, 10], [0, 10], [20, 30]],
        [[0, 10]],
        count=1,
    ) == [0]


def test_guarded_requires_valid_target_evidence():
    with pytest.raises(
        GuardedWindowTokenizationError,
        match="requires at least one target character span",
    ):
        tokenize_guarded_source(
            "contract C {}",
            target_char_spans=[],
            tokenizer=FakeTokenizer(),
        )


def test_guarded_tokenization_is_deterministic():
    kwargs = {
        "source_text": "x" * 2200,
        "target_char_spans": [[450, 550], [1200, 1300]],
        "tokenizer": FakeTokenizer(),
    }
    first = tokenize_guarded_source(**kwargs)
    second = tokenize_guarded_source(**kwargs)

    assert first["selected_window_indices"] == second["selected_window_indices"]
    assert first["selector"] == second["selector"]
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["attention_mask"], second["attention_mask"])


def _write_parent_fixture(tmp_path: Path, *, requested_names=None):
    source = "fixture"
    contract_id = "a" * 64
    parent_root = tmp_path / "accepted-parent"
    preprocessed_root = tmp_path / "accepted-preprocessed"
    parent_source = parent_root / source
    preprocessed_source = preprocessed_root / source
    parent_source.mkdir(parents=True)
    preprocessed_source.mkdir(parents=True)

    graph_bytes = b"immutable-r4-d-011-graph-bytes"
    (parent_source / f"{contract_id}.pt").write_bytes(graph_bytes)
    (parent_source / f"{contract_id}.tokens.pt").write_bytes(
        b"immutable-r4-d-011-token-parent"
    )
    sidecar = {
        "sha256": contract_id,
        "source": source,
        "schema_version": "v10",
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "token_lineage": PARENT_TOKEN_LINEAGE,
        "requested_contract_names": (
            ["Target"] if requested_names is None else requested_names
        ),
        "graph_component_count": 1,
        "selected_window_indices": [0],
    }
    (parent_source / f"{contract_id}.rep.json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )
    (preprocessed_source / f"{contract_id}.sol").write_text(
        "contract Target { uint value; }\n",
        encoding="utf-8",
    )
    return source, contract_id, parent_root, preprocessed_root, graph_bytes


def test_candidate_identity_preserves_parent_graph_bytes_and_isolates_lineage(tmp_path):
    source, contract_id, parent_root, preprocessed_root, graph_bytes = (
        _write_parent_fixture(tmp_path)
    )
    output_source = tmp_path / "candidate" / source

    result = build_guarded_candidate_identity(
        source=source,
        contract_id=contract_id,
        parent_source_dir=parent_root / source,
        preprocessed_source_dir=preprocessed_root / source,
        output_source_dir=output_source,
        tokenizer=FakeTokenizer(),
        parent_binding_digest="d" * 64,
    )

    assert (output_source / f"{contract_id}.pt").read_bytes() == graph_bytes
    assert (parent_root / source / f"{contract_id}.pt").read_bytes() == graph_bytes
    assert result["parent_graph_sha256"] == result["candidate_graph_sha256"]

    sidecar = json.loads(
        (output_source / f"{contract_id}.rep.json").read_text(encoding="utf-8")
    )
    payload = torch.load(
        output_source / f"{contract_id}.tokens.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert sidecar["token_lineage"] == GUARDED_TOKEN_LINEAGE_VERSION
    assert sidecar["parent_token_lineage"] == PARENT_TOKEN_LINEAGE
    assert sidecar["selector_policy"] == GUARDED_SELECTOR_POLICY_VERSION
    assert sidecar["graph_bytes_reused_from_parent"] is True
    assert sidecar["physical_acceptance"] is False
    assert sidecar["training_authorized"] is False
    assert payload["token_lineage"] == GUARDED_TOKEN_LINEAGE_VERSION
    assert payload["selector_policy"] == GUARDED_SELECTOR_POLICY_VERSION
    assert tuple(payload["input_ids"].shape) == (4, 512)


def test_candidate_identity_rejects_missing_requested_target_without_output(tmp_path):
    source, contract_id, parent_root, preprocessed_root, _ = _write_parent_fixture(
        tmp_path,
        requested_names=[],
    )
    output_source = tmp_path / "candidate" / source

    with pytest.raises(ValueError, match="no requested_contract_names"):
        build_guarded_candidate_identity(
            source=source,
            contract_id=contract_id,
            parent_source_dir=parent_root / source,
            preprocessed_source_dir=preprocessed_root / source,
            output_source_dir=output_source,
            tokenizer=FakeTokenizer(),
            parent_binding_digest="d" * 64,
        )

    assert not (output_source / f"{contract_id}.pt").exists()
    assert not (output_source / f"{contract_id}.tokens.pt").exists()
    assert not (output_source / f"{contract_id}.rep.json").exists()


def test_source_builder_requires_bound_parent_and_writes_fresh_manifest(tmp_path):
    source, contract_id, parent_root, preprocessed_root, graph_bytes = (
        _write_parent_fixture(tmp_path)
    )
    output_root = tmp_path / GUARDED_REPRESENTATION_ROOT_NAME
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "decision_id": "R4-D-011",
                "physical_acceptance": True,
                "accepted_lineage": {
                    "contracts": 22540,
                    "graph_schema_version": "v10",
                    "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
                    "physical_root": str(parent_root),
                    "preprocessed_parent": str(preprocessed_root),
                    "binding_digest_sha256": "d" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    result = build_guarded_candidate_source(
        source,
        parent_root=parent_root,
        preprocessed_root=preprocessed_root,
        output_root=output_root,
        acceptance_path=acceptance_path,
        repo_root=tmp_path,
        tokenizer=FakeTokenizer(),
        limit=1,
    )

    assert result.contracts_seen == 1
    assert result.representations_written == 1
    assert result.representations_failed == 0
    assert (output_root / source / f"{contract_id}.pt").read_bytes() == graph_bytes
    manifest = json.loads(
        (output_root / source / "guarded_candidate_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["parent_physical_decision_id"] == "R4-D-011"
    assert manifest["parent_binding_digest_sha256"] == "d" * 64
    assert manifest["representation_root"] == GUARDED_REPRESENTATION_ROOT_NAME
    assert manifest["selector_policy"] == GUARDED_SELECTOR_POLICY_VERSION
    assert manifest["physical_acceptance"] is False
    assert manifest["training_authorized"] is False
