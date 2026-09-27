from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import sentinel_data.representation.r4_guarded_candidate as candidate_module
from sentinel_data.preprocessing.r4_versions import (
    PREPROCESSING_ARTIFACT_VERSION,
    V10_PRIMARY_SLITHER_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.representation.r4_guarded_candidate import (
    FAILURE_LEDGER_NAME,
    MANIFEST_NAME,
    GuardedCandidateBuildError,
    assemble_guarded_candidate,
)
from sentinel_data.representation.r4_guarded_lineage import (
    CANDIDATE_ROOT_NAME,
    PARENT_ACCEPTANCE_SCHEMA,
    PARENT_BINDING_DIGEST_SHA256,
    PARENT_DECISION_ID,
    PARENT_EXTRACTOR_VERSION,
    PARENT_GRAPH_SCHEMA_VERSION,
    SELECTOR_CONFIG_SHA256,
    TOKEN_LINEAGE_ID,
    graph_parent_authority,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    source_name = "fixture"
    source_text = "contract Vault { uint x; }\n"
    contract_id = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    parent_root = tmp_path / "accepted-attempt" / V10_REPRESENTATION_ROOT_NAME
    parent_dir = parent_root / source_name
    parent_dir.mkdir(parents=True)
    graph_path = parent_dir / f"{contract_id}.pt"
    token_path = parent_dir / f"{contract_id}.tokens.pt"
    sidecar_path = parent_dir / f"{contract_id}.rep.json"
    graph_path.write_bytes(b"accepted-r4-d011-graph")
    token_path.write_bytes(b"accepted-r4-d011-token")

    sidecar = {
        "sha256": contract_id,
        "source": source_name,
        "schema_version": PARENT_GRAPH_SCHEMA_VERSION,
        "extractor_version": PARENT_EXTRACTOR_VERSION,
        "token_lineage": "accepted_v9_byte_copy",
        "requested_contract_names": ["Vault"],
        "actual_contract_names": ["Vault"],
        "graph_extraction_mode": "slither_full",
        "graph_analysis_degraded": False,
        "slither_runtime": {
            "slither_analyzer": V10_PRIMARY_SLITHER_VERSION,
            "crytic_compile": "0.3.11",
            "runtime_role": "primary",
            "required_for_physical_acceptance": V10_PRIMARY_SLITHER_VERSION,
        },
        "unclassified_call_ir": [],
        "unclassified_call_ir_count": 0,
        "classified_call_ir_counts": {"LOW_LEVEL_CALL": 1},
        "emitted_call_edge_counts": {"LOW_LEVEL_CALL": 1},
        "call_mapping_errors": [],
        "compute_time_ms": 12.5,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    preprocessed_root = tmp_path / PREPROCESSING_ARTIFACT_VERSION
    preprocessed_dir = preprocessed_root / source_name
    preprocessed_dir.mkdir(parents=True)
    (preprocessed_dir / f"{contract_id}.sol").write_text(
        source_text,
        encoding="utf-8",
    )

    acceptance_path = tmp_path / "acceptance.json"
    acceptance = {
        "schema": PARENT_ACCEPTANCE_SCHEMA,
        "decision": "ACCEPTED_IMMUTABLE_LOCAL_PHYSICAL_REPRESENTATION",
        "decision_id": PARENT_DECISION_ID,
        "physical_acceptance": True,
        "training_authorized": False,
        "selector_promoted": False,
        "accepted_lineage": {
            "binding_digest_sha256": PARENT_BINDING_DIGEST_SHA256,
            "contracts": 1,
            "files": 3,
            "extractor_version": PARENT_EXTRACTOR_VERSION,
            "graph_schema_version": PARENT_GRAPH_SCHEMA_VERSION,
            "physical_root": (
                "data_module/data/accepted-attempt/"
                + V10_REPRESENTATION_ROOT_NAME
            ),
            "preprocessed_parent": (
                "data_module/data/" + PREPROCESSING_ARTIFACT_VERSION
            ),
            "protected_local": True,
        },
        "runtime_distribution": [
            {
                "contracts": 1,
                "crytic_compile": "0.3.11",
                "runtime_role": "primary",
                "slither_analyzer": V10_PRIMARY_SLITHER_VERSION,
            }
        ],
    }
    acceptance_path.write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "source_name": source_name,
        "source_text": source_text,
        "contract_id": contract_id,
        "parent_root": parent_root,
        "parent_paths": (graph_path, token_path, sidecar_path),
        "preprocessed_root": preprocessed_root,
        "acceptance_path": acceptance_path,
    }


def _token_data() -> dict:
    input_ids = torch.arange(4 * 512, dtype=torch.long).reshape(4, 512)
    attention_mask = torch.zeros((4, 512), dtype=torch.long)
    attention_mask[0, :64] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "num_windows": 1,
        "stride": 256,
        "num_tokens": 64,
        "tokenizer_name": "microsoft/graphcodebert-base",
        "max_length": 512,
        "coverage_schema_version": "r4-token-coverage-v1",
        "pre_subsampling_window_count": 1,
        "pre_subsampling_code_tokens": 62,
        "selected_window_indices": [0],
        "selected_code_token_ranges": [[0, 62]],
        "retained_unique_code_tokens": 62,
        "retained_token_ratio": 1.0,
        "content_tokens_per_window": 510,
        "coverage_interpretation": "diagnostic_only_no_adequacy_threshold",
        "token_lineage": TOKEN_LINEAGE_ID,
        "token_lineage_parent_decision": PARENT_DECISION_ID,
        "token_lineage_parent_binding_digest_sha256": (
            PARENT_BINDING_DIGEST_SHA256
        ),
        "selector_config_sha256": SELECTOR_CONFIG_SHA256,
        "target_evidence": {
            "schema": "sentinel-r4-selector-target-evidence-v1",
            "sha256": "e" * 64,
        },
        "token_selector": {
            "schema": "sentinel-r4-token-selector-decision-v1",
            "effective_strategy": "historical_linspace_v1",
            "used_control_fallback": True,
        },
    }


def _candidate_root(tmp_path: Path, attempt: str = "candidate-attempt") -> Path:
    return tmp_path / attempt / CANDIDATE_ROOT_NAME


def _install_tokenizer_stub(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, str, tuple[str, ...]]] | None = None,
) -> None:
    def fake_tokenize(
        source_text: str,
        *,
        contract_id: str,
        requested_contract_names,
        tokenizer=None,
    ):
        if calls is not None:
            calls.append(
                (
                    source_text,
                    contract_id,
                    tuple(requested_contract_names),
                )
            )
        return _token_data()

    monkeypatch.setattr(
        candidate_module,
        "tokenize_repaired_source_guarded",
        fake_tokenize,
    )


def test_candidate_build_preserves_parent_and_writes_new_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    _install_tokenizer_stub(monkeypatch, calls)

    parent_paths = fixture["parent_paths"]
    before = tuple(_sha256(path) for path in parent_paths)
    candidate_root = _candidate_root(tmp_path)

    report = assemble_guarded_candidate(
        parent_root=fixture["parent_root"],
        preprocessed_root=fixture["preprocessed_root"],
        candidate_root=candidate_root,
        acceptance_path=fixture["acceptance_path"],
        tokenizer=object(),
    )

    assert report["passed"] is True
    assert report["complete_candidate_build"] is True
    assert report["physical_acceptance"] is False
    assert report["training_authorized"] is False
    assert report["representations_written"] == 1
    assert report["representations_failed"] == 0
    assert tuple(_sha256(path) for path in parent_paths) == before

    source_name = fixture["source_name"]
    contract_id = fixture["contract_id"]
    candidate_dir = candidate_root / source_name
    graph_path = candidate_dir / f"{contract_id}.pt"
    token_path = candidate_dir / f"{contract_id}.tokens.pt"
    sidecar_path = candidate_dir / f"{contract_id}.rep.json"

    assert graph_path.read_bytes() == parent_paths[0].read_bytes()
    token_payload = torch.load(token_path, map_location="cpu", weights_only=True)
    assert tuple(token_payload["input_ids"].shape) == (4, 512)
    assert token_payload["sha256"] == contract_id
    assert token_payload["source"] == source_name
    assert token_payload["token_lineage"] == TOKEN_LINEAGE_ID

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["token_lineage"] == TOKEN_LINEAGE_ID
    assert sidecar["graph_parent"] == graph_parent_authority()
    assert sidecar["selector_config_sha256"] == SELECTOR_CONFIG_SHA256
    assert sidecar["selected_window_indices"] == [0]
    assert sidecar["compute_time_ms"] == 12.5

    assert calls == [
        (
            fixture["source_text"],
            contract_id,
            ("Vault",),
        )
    ]
    assert str(tmp_path) not in (candidate_root / MANIFEST_NAME).read_text(
        encoding="utf-8"
    )


def test_candidate_build_records_invalid_parent_target_as_identity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    sidecar_path = fixture["parent_paths"][2]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["actual_contract_names"] = ["Other"]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    def must_not_tokenize(*args, **kwargs):
        raise AssertionError("tokenizer must not run for invalid parent targets")

    monkeypatch.setattr(
        candidate_module,
        "tokenize_repaired_source_guarded",
        must_not_tokenize,
    )
    before = tuple(_sha256(path) for path in fixture["parent_paths"])
    candidate_root = _candidate_root(tmp_path)

    report = assemble_guarded_candidate(
        parent_root=fixture["parent_root"],
        preprocessed_root=fixture["preprocessed_root"],
        candidate_root=candidate_root,
        acceptance_path=fixture["acceptance_path"],
    )

    assert report["passed"] is False
    assert report["representations_written"] == 0
    assert report["representations_failed"] == 1
    assert "requested/actual target identity differs" in report["failures"][0]["error"]
    assert (candidate_root / FAILURE_LEDGER_NAME).is_file()
    assert tuple(_sha256(path) for path in fixture["parent_paths"]) == before

    contract_id = fixture["contract_id"]
    candidate_dir = candidate_root / fixture["source_name"]
    for suffix in (".pt", ".tokens.pt", ".rep.json"):
        assert not (candidate_dir / f"{contract_id}{suffix}").exists()


def test_candidate_build_records_runtime_drift_before_tokenization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    sidecar_path = fixture["parent_paths"][2]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["slither_runtime"]["slither_analyzer"] = "0.11.5"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    def must_not_tokenize(*args, **kwargs):
        raise AssertionError("tokenizer must not run for runtime drift")

    monkeypatch.setattr(
        candidate_module,
        "tokenize_repaired_source_guarded",
        must_not_tokenize,
    )

    report = assemble_guarded_candidate(
        parent_root=fixture["parent_root"],
        preprocessed_root=fixture["preprocessed_root"],
        candidate_root=_candidate_root(tmp_path),
        acceptance_path=fixture["acceptance_path"],
    )

    assert report["passed"] is False
    assert report["representations_failed"] == 1
    assert "Slither version mismatch" in report["failures"][0]["error"]


def test_candidate_build_records_repaired_source_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_tokenizer_stub(monkeypatch)
    source_path = (
        fixture["preprocessed_root"]
        / fixture["source_name"]
        / f"{fixture['contract_id']}.sol"
    )
    source_path.write_text("contract Drift { uint y; }\n", encoding="utf-8")

    report = assemble_guarded_candidate(
        parent_root=fixture["parent_root"],
        preprocessed_root=fixture["preprocessed_root"],
        candidate_root=_candidate_root(tmp_path),
        acceptance_path=fixture["acceptance_path"],
    )

    assert report["passed"] is False
    assert "source bytes do not match" in report["failures"][0]["error"]


def test_candidate_root_must_be_fresh_and_outside_parent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    nonempty = _candidate_root(tmp_path)
    nonempty.mkdir(parents=True)
    (nonempty / "marker").write_text("occupied", encoding="utf-8")
    with pytest.raises(GuardedCandidateBuildError, match="fresh and empty"):
        assemble_guarded_candidate(
            parent_root=fixture["parent_root"],
            preprocessed_root=fixture["preprocessed_root"],
            candidate_root=nonempty,
            acceptance_path=fixture["acceptance_path"],
        )

    nested = fixture["parent_root"] / CANDIDATE_ROOT_NAME
    with pytest.raises(GuardedCandidateBuildError, match="must not be inside"):
        assemble_guarded_candidate(
            parent_root=fixture["parent_root"],
            preprocessed_root=fixture["preprocessed_root"],
            candidate_root=nested,
            acceptance_path=fixture["acceptance_path"],
        )


def test_candidate_build_rejects_acceptance_digest_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    acceptance_path = fixture["acceptance_path"]
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["accepted_lineage"]["binding_digest_sha256"] = "0" * 64
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

    candidate_root = _candidate_root(tmp_path)
    with pytest.raises(
        GuardedCandidateBuildError,
        match="binding_digest_sha256 mismatch",
    ):
        assemble_guarded_candidate(
            parent_root=fixture["parent_root"],
            preprocessed_root=fixture["preprocessed_root"],
            candidate_root=candidate_root,
            acceptance_path=acceptance_path,
        )
    assert not candidate_root.exists()


def test_repeat_candidate_builds_are_byte_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_tokenizer_stub(monkeypatch)

    first_root = _candidate_root(tmp_path, "attempt-one")
    second_root = _candidate_root(tmp_path, "attempt-two")

    first = assemble_guarded_candidate(
        parent_root=fixture["parent_root"],
        preprocessed_root=fixture["preprocessed_root"],
        candidate_root=first_root,
        acceptance_path=fixture["acceptance_path"],
    )
    second = assemble_guarded_candidate(
        parent_root=fixture["parent_root"],
        preprocessed_root=fixture["preprocessed_root"],
        candidate_root=second_root,
        acceptance_path=fixture["acceptance_path"],
    )

    assert first == second

    source = fixture["source_name"]
    contract_id = fixture["contract_id"]
    for suffix in (".pt", ".tokens.pt", ".rep.json"):
        first_path = first_root / source / f"{contract_id}{suffix}"
        second_path = second_root / source / f"{contract_id}{suffix}"
        assert _sha256(first_path) == _sha256(second_path)

    assert (first_root / MANIFEST_NAME).read_bytes() == (
        second_root / MANIFEST_NAME
    ).read_bytes()
