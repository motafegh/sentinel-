"""Repository-safe tests for the R4-D-012 guarded-token candidate builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from sentinel_data.preprocessing.r4_versions import (
    GUARDED_REPRESENTATION_ROOT_NAME,
    GUARDED_TOKEN_LINEAGE_VERSION,
    GUARDED_TOKEN_SELECTOR_VERSION,
    HISTORICAL_TOKEN_SELECTOR_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.representation import r4_guarded_token_candidate as guarded_candidate
from sentinel_data.representation.r4_guarded_token_candidate import (
    GuardedTokenCandidateError,
    TargetEvidenceError,
    build_guarded_token_candidate,
    _build_guarded_token_identity,
    load_accepted_v10_parent,
)


class CharTokenizer:
    """Tiny deterministic tokenizer with one raw token per source character."""

    pad_token_id = 0

    def num_special_tokens_to_add(self, pair=False):
        return 0

    def __call__(self, code, **kwargs):
        ids = [index + 1 for index in range(len(code))]
        if kwargs.get("add_special_tokens") is False:
            result = {"input_ids": ids}
            if kwargs.get("return_offsets_mapping"):
                result["offset_mapping"] = [
                    (index, index + 1) for index in range(len(code))
                ]
            return result

        max_length = int(kwargs["max_length"])
        stride = int(kwargs["stride"])
        step = max_length - stride
        windows = []
        masks = []
        start = 0
        while True:
            real = ids[start : start + max_length]
            mask = [1] * len(real)
            windows.append(real + [self.pad_token_id] * (max_length - len(real)))
            masks.append(mask + [0] * (max_length - len(mask)))
            if start + max_length >= len(ids):
                break
            start += step
        return {
            "input_ids": torch.tensor(windows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_with_target(*, long: bool) -> str:
    if not long:
        return "contract Target { uint value; function f() public {} }\n"

    prefix_body = " ".join(f"uint p{i};" for i in range(170))
    target_body = " ".join(f"uint t{i};" for i in range(35))
    suffix_body = " ".join(f"uint s{i};" for i in range(170))
    return (
        f"contract Prefix {{ {prefix_body} }}\n"
        f"contract Target {{ {target_body} }}\n"
        f"contract Suffix {{ {suffix_body} }}\n"
    )


def _fixture(
    tmp_path: Path,
    *,
    source_text: str,
    target_names: list[str] | None = None,
    actual_names: list[str] | None = None,
):
    source = "fixture"
    contract_id = "a" * 64
    repo_root = tmp_path / "repo"
    parent_root = (
        repo_root
        / "data_module/data/r4-d011-fixture"
        / "representations-r4-v3-candidate"
    )
    parent_dir = parent_root / source
    parent_dir.mkdir(parents=True)
    preprocessed_root = repo_root / "data_module/data/sentinel-preprocessed-r4-v2"
    source_dir = preprocessed_root / source
    source_dir.mkdir(parents=True)
    (source_dir / f"{contract_id}.sol").write_text(source_text, encoding="utf-8")

    graph_path = parent_dir / f"{contract_id}.pt"
    graph_path.write_bytes(b"immutable-r4-d011-graph")
    (parent_dir / f"{contract_id}.tokens.pt").write_bytes(b"historical-token-bytes")

    targets = ["Target"] if target_names is None else target_names
    actual = targets if actual_names is None else actual_names
    sidecar = {
        "sha256": contract_id,
        "source": source,
        "schema_version": "v10",
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "token_lineage": "accepted_v9_byte_copy",
        "requested_contract_names": targets,
        "actual_contract_names": actual,
        "graph_component_count": len(actual),
        "node_count": 1,
        "edge_count": 0,
    }
    (parent_dir / f"{contract_id}.rep.json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )

    acceptance_path = repo_root / "acceptance.json"
    relative_parent = parent_root.relative_to(repo_root).as_posix()
    acceptance_path.write_text(
        json.dumps(
            {
                "schema": "sentinel-r4-v10-v26-physical-acceptance-v1",
                "decision_id": "R4-D-011",
                "status": "PASS",
                "physical_acceptance": True,
                "accepted_lineage": {
                    "binding_digest_sha256": "b" * 64,
                    "contracts": 1,
                    "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
                    "graph_schema_version": "v10",
                    "physical_root": relative_parent,
                },
            }
        ),
        encoding="utf-8",
    )
    parent = load_accepted_v10_parent(
        acceptance_path=acceptance_path,
        repo_root=repo_root,
        parent_root=parent_root,
    )
    return {
        "repo_root": repo_root,
        "parent_root": parent_root,
        "preprocessed_root": preprocessed_root,
        "parent": parent,
        "source": source,
        "contract_id": contract_id,
        "parent_graph": graph_path,
    }


def _output_root(tmp_path: Path, label: str) -> Path:
    return tmp_path / label / GUARDED_REPRESENTATION_ROOT_NAME


def _build(tmp_path: Path, fixture: dict, label: str):
    output_root = _output_root(tmp_path, label)
    result = _build_guarded_token_identity(
        source=fixture["source"],
        contract_id=fixture["contract_id"],
        preprocessed_root=fixture["preprocessed_root"],
        parent_root=fixture["parent_root"],
        output_root=output_root,
        parent=fixture["parent"],
        tokenizer=CharTokenizer(),
    )
    source_dir = output_root / fixture["source"]
    token_path = source_dir / f"{fixture['contract_id']}.tokens.pt"
    sidecar_path = source_dir / f"{fixture['contract_id']}.rep.json"
    return result, output_root, token_path, sidecar_path


def test_under_cap_keeps_all_real_windows_and_pads_to_frozen_shape(tmp_path: Path):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=False))
    result, _, token_path, sidecar_path = _build(tmp_path, fixture, "under-cap")

    payload = torch.load(token_path, map_location="cpu", weights_only=True)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    decision = payload["selector_decision"]

    assert tuple(payload["input_ids"].shape) == (4, 512)
    assert tuple(payload["attention_mask"].shape) == (4, 512)
    assert decision["selected_indices"] == [0]
    assert decision["control_indices"] == [0]
    assert decision["candidate_indices"] == [0]
    assert decision["used_control_fallback"] is True
    assert decision["effective_selector"] == HISTORICAL_TOKEN_SELECTOR_VERSION
    assert decision["fallback_reason"] == (
        "candidate_target_coverage_not_strictly_greater_than_control"
    )
    assert result.selected_window_indices == (0,)
    assert sidecar["selector_decision"] == decision
    assert sidecar["selected_window_indices"] == [0]
    assert sidecar["token_lineage"] == GUARDED_TOKEN_LINEAGE_VERSION


def test_over_cap_uses_guarded_selector_only_for_strict_target_gain(tmp_path: Path):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=True))
    result, _, token_path, sidecar_path = _build(tmp_path, fixture, "over-cap")

    payload = torch.load(token_path, map_location="cpu", weights_only=True)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    decision = payload["selector_decision"]

    assert decision["pre_subsampling_window_count"] > 4
    assert decision["used_control_fallback"] is False
    assert decision["effective_selector"] == GUARDED_TOKEN_SELECTOR_VERSION
    assert decision["fallback_reason"] is None
    assert (
        decision["candidate_target_coverage_tokens"]
        > decision["control_target_coverage_tokens"]
    )
    assert decision["selected_indices"] == decision["candidate_indices"]
    assert len(decision["selected_indices"]) == 4
    assert decision["selected_indices"] == sorted(set(decision["selected_indices"]))
    assert result.effective_selector == GUARDED_TOKEN_SELECTOR_VERSION
    assert sidecar["selector_decision"] == decision


def test_guarded_build_is_deterministic_and_graph_bytes_are_parent_identical(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=True))
    first, first_root, first_tokens, first_sidecar = _build(tmp_path, fixture, "first")
    second, second_root, second_tokens, second_sidecar = _build(tmp_path, fixture, "second")

    first_payload = torch.load(first_tokens, map_location="cpu", weights_only=True)
    second_payload = torch.load(second_tokens, map_location="cpu", weights_only=True)
    first_meta = json.loads(first_sidecar.read_text(encoding="utf-8"))
    second_meta = json.loads(second_sidecar.read_text(encoding="utf-8"))

    assert torch.equal(first_payload["input_ids"], second_payload["input_ids"])
    assert torch.equal(
        first_payload["attention_mask"],
        second_payload["attention_mask"],
    )
    assert first_payload["selector_decision"] == second_payload["selector_decision"]
    assert first.selected_window_indices == second.selected_window_indices
    assert first_meta == second_meta

    first_graph = first_root / fixture["source"] / f"{fixture['contract_id']}.pt"
    second_graph = second_root / fixture["source"] / f"{fixture['contract_id']}.pt"
    assert first_graph.read_bytes() == fixture["parent_graph"].read_bytes()
    assert second_graph.read_bytes() == fixture["parent_graph"].read_bytes()
    assert first_meta["graph_parent"]["graph_sha256"] == _sha(fixture["parent_graph"])
    assert second_meta["graph_parent"]["graph_sha256"] == _sha(fixture["parent_graph"])


def test_missing_target_evidence_fails_closed_without_artifacts(tmp_path: Path):
    fixture = _fixture(
        tmp_path,
        source_text=_source_with_target(long=False),
        target_names=[],
        actual_names=[],
    )
    output_root = _output_root(tmp_path, "missing-target")

    with pytest.raises(TargetEvidenceError, match="requested_contract_names"):
        _build_guarded_token_identity(
            source=fixture["source"],
            contract_id=fixture["contract_id"],
            preprocessed_root=fixture["preprocessed_root"],
            parent_root=fixture["parent_root"],
            output_root=output_root,
            parent=fixture["parent"],
            tokenizer=CharTokenizer(),
        )

    assert not output_root.exists()


def test_ambiguous_or_missing_named_target_fails_closed(tmp_path: Path):
    fixture = _fixture(
        tmp_path,
        source_text=_source_with_target(long=False),
        target_names=["DoesNotExist"],
    )
    output_root = _output_root(tmp_path, "bad-target-name")

    with pytest.raises(TargetEvidenceError, match="target declaration count"):
        _build_guarded_token_identity(
            source=fixture["source"],
            contract_id=fixture["contract_id"],
            preprocessed_root=fixture["preprocessed_root"],
            parent_root=fixture["parent_root"],
            output_root=output_root,
            parent=fixture["parent"],
            tokenizer=CharTokenizer(),
        )


def test_parent_requested_actual_target_mismatch_is_rejected(tmp_path: Path):
    fixture = _fixture(
        tmp_path,
        source_text=_source_with_target(long=False),
        target_names=["Target"],
        actual_names=["Other"],
    )

    with pytest.raises(
        GuardedTokenCandidateError,
        match="requested/actual graph target mismatch",
    ):
        _build_guarded_token_identity(
            source=fixture["source"],
            contract_id=fixture["contract_id"],
            preprocessed_root=fixture["preprocessed_root"],
            parent_root=fixture["parent_root"],
            output_root=_output_root(tmp_path, "target-mismatch"),
            parent=fixture["parent"],
            tokenizer=CharTokenizer(),
        )


def test_builder_refuses_existing_artifact_overwrite(tmp_path: Path):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=False))
    _, output_root, _, _ = _build(tmp_path, fixture, "no-overwrite")

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        _build_guarded_token_identity(
            source=fixture["source"],
            contract_id=fixture["contract_id"],
            preprocessed_root=fixture["preprocessed_root"],
            parent_root=fixture["parent_root"],
            output_root=output_root,
            parent=fixture["parent"],
            tokenizer=CharTokenizer(),
        )


def test_parent_loader_rejects_noncanonical_parent_root(tmp_path: Path):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=False))
    wrong_root = tmp_path / "wrong" / "representations-r4-v3-candidate"
    wrong_root.mkdir(parents=True)

    acceptance = fixture["repo_root"] / "acceptance.json"
    with pytest.raises(GuardedTokenCandidateError, match="exact R4-D-011"):
        load_accepted_v10_parent(
            acceptance_path=acceptance,
            repo_root=fixture["repo_root"],
            parent_root=wrong_root,
        )


def test_bounded_candidate_manifest_binds_fresh_lineage_and_stop_lines(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=True))
    monkeypatch.setattr(
        guarded_candidate,
        "_source_commit",
        lambda _repo_root: "c" * 40,
    )
    monkeypatch.setattr(
        guarded_candidate,
        "_load_canonical_tokenizer",
        lambda: CharTokenizer(),
    )
    output_root = _output_root(tmp_path, "bounded-manifest")
    manifest = build_guarded_token_candidate(
        acceptance_path=fixture["repo_root"] / "acceptance.json",
        repo_root=fixture["repo_root"],
        preprocessed_root=fixture["preprocessed_root"],
        parent_root=fixture["parent_root"],
        output_root=output_root,
        identities=[(fixture["source"], fixture["contract_id"])],
    )

    assert manifest["status"] == "BOUNDED_GUARDED_TOKEN_CANDIDATE"
    assert manifest["physical_acceptance"] is False
    assert manifest["training_authorized"] is False
    assert manifest["source_commit"] == "c" * 40
    assert manifest["representation_lineage"] == GUARDED_TOKEN_LINEAGE_VERSION
    assert manifest["selector_policy"] == GUARDED_TOKEN_SELECTOR_VERSION
    assert manifest["control_selector"] == HISTORICAL_TOKEN_SELECTOR_VERSION
    assert manifest["full_population"] is False
    assert manifest["contracts_requested"] == 1
    assert manifest["contracts_written"] == 1
    assert manifest["parent"]["decision_id"] == "R4-D-011"
    assert manifest["parent"]["contracts"] == 1
    assert len(manifest["records"]) == 1

    persisted = json.loads(
        (output_root / "guarded_candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest


def test_batch_builder_rejects_misnamed_root_without_creating_it(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=False))
    wrong_root = tmp_path / "wrong-candidate-root"

    with pytest.raises(GuardedTokenCandidateError, match="root must be named"):
        build_guarded_token_candidate(
            acceptance_path=fixture["repo_root"] / "acceptance.json",
            repo_root=fixture["repo_root"],
            preprocessed_root=fixture["preprocessed_root"],
            parent_root=fixture["parent_root"],
            output_root=wrong_root,
            identities=[(fixture["source"], fixture["contract_id"])],
            )

    assert not wrong_root.exists()


def test_batch_builder_blocks_full_population_before_d4_acceptance(tmp_path: Path):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=False))
    output_root = _output_root(tmp_path, "blocked-full")

    with pytest.raises(GuardedTokenCandidateError, match="D5.*not authorized"):
        build_guarded_token_candidate(
            acceptance_path=fixture["repo_root"] / "acceptance.json",
            repo_root=fixture["repo_root"],
            preprocessed_root=fixture["preprocessed_root"],
            parent_root=fixture["parent_root"],
            output_root=output_root,
            identities=None,
            )

    assert not output_root.exists()


def test_builder_rejects_candidate_nested_inside_accepted_parent(tmp_path: Path):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=False))
    nested_root = (
        fixture["parent_root"]
        / "nested"
        / GUARDED_REPRESENTATION_ROOT_NAME
    )

    with pytest.raises(
        GuardedTokenCandidateError,
        match="outside the immutable R4-D-011 parent tree",
    ):
        build_guarded_token_candidate(
            acceptance_path=fixture["repo_root"] / "acceptance.json",
            repo_root=fixture["repo_root"],
            preprocessed_root=fixture["preprocessed_root"],
            parent_root=fixture["parent_root"],
            output_root=nested_root,
            identities=[(fixture["source"], fixture["contract_id"])],
            )

    assert not nested_root.exists()


def test_builder_rejects_candidate_nested_inside_preprocessed_parent(tmp_path: Path):
    fixture = _fixture(tmp_path, source_text=_source_with_target(long=False))
    nested_root = (
        fixture["preprocessed_root"]
        / "nested"
        / GUARDED_REPRESENTATION_ROOT_NAME
    )

    with pytest.raises(
        GuardedTokenCandidateError,
        match="outside the immutable repaired preprocessing parent tree",
    ):
        build_guarded_token_candidate(
            acceptance_path=fixture["repo_root"] / "acceptance.json",
            repo_root=fixture["repo_root"],
            preprocessed_root=fixture["preprocessed_root"],
            parent_root=fixture["parent_root"],
            output_root=nested_root,
            identities=[(fixture["source"], fixture["contract_id"])],
            )

    assert not nested_root.exists()
