from __future__ import annotations

import copy

import pytest

from sentinel_data.representation.r4_guarded_lineage import (
    BINDING_REPORT_SCHEMA,
    CANDIDATE_ROOT_NAME,
    PARENT_BINDING_DIGEST_SHA256,
    SELECTOR_CONFIG,
    SELECTOR_CONFIG_SHA256,
    SELECTOR_DECISION_SCHEMA,
    TARGET_EVIDENCE_SCHEMA,
    TOKEN_LINEAGE_ID,
    build_selector_metadata,
    build_target_evidence,
    canonical_digest,
    graph_parent_authority,
    validate_graph_parent_authority,
    validate_selector_metadata,
    validate_target_evidence,
)
from sentinel_data.representation.r4_window_selector import select_guarded_windows


def _target_evidence() -> dict:
    return build_target_evidence(
        contract_id="a" * 64,
        requested_contract_names=["Vault"],
        target_char_spans=[(0, 120)],
        target_token_ranges=[(10, 40)],
        target_tokens=30,
    )


def _decision():
    return select_guarded_windows(
        [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]],
        [[12, 18], [31, 35]],
        max_windows=2,
    )


def test_versioned_lineage_identities_are_fresh_and_explicit() -> None:
    assert TOKEN_LINEAGE_ID == "r4-v10-v26-target-aware-guarded-v1"
    assert (
        CANDIDATE_ROOT_NAME
        == "representations-r4-v10-v26-target-aware-guarded-v1-candidate"
    )
    assert SELECTOR_DECISION_SCHEMA == "sentinel-r4-token-selector-decision-v1"
    assert TARGET_EVIDENCE_SCHEMA == "sentinel-r4-selector-target-evidence-v1"
    assert BINDING_REPORT_SCHEMA == "sentinel-r4-v10-guarded-candidate-binding-v1"


def test_selector_config_digest_is_canonical_and_stable() -> None:
    assert SELECTOR_CONFIG_SHA256 == canonical_digest(SELECTOR_CONFIG)
    assert (
        SELECTOR_CONFIG_SHA256
        == "7ce20027e124aef763b6e448bd70d0c562b6d33cb2156814a9ed19e04bb25151"
    )


def test_graph_parent_is_exact_r4_d_011_authority() -> None:
    parent = graph_parent_authority()
    assert parent == {
        "decision_id": "R4-D-011",
        "acceptance_schema": "sentinel-r4-v10-v26-physical-acceptance-v1",
        "binding_digest_sha256": PARENT_BINDING_DIGEST_SHA256,
        "graph_schema_version": "v10",
        "extractor_version": "v2.6-r4-call-semantics-deterministic-cfg-mutators",
    }
    validate_graph_parent_authority(parent)

    drift = dict(parent)
    drift["graph_schema_version"] = "v9"
    with pytest.raises(ValueError, match="immutable R4-D-011"):
        validate_graph_parent_authority(drift)


def test_target_evidence_round_trips_and_binds_its_own_digest() -> None:
    evidence = _target_evidence()
    assert evidence["sha256"] == canonical_digest(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    validate_target_evidence(evidence)


def test_target_evidence_rejects_tampering_and_inconsistent_counts() -> None:
    evidence = _target_evidence()
    tampered = copy.deepcopy(evidence)
    tampered["target_token_ranges"] = [[10, 41]]
    with pytest.raises(ValueError):
        validate_target_evidence(tampered)

    with pytest.raises(ValueError, match="target_tokens mismatch"):
        build_target_evidence(
            contract_id="a" * 64,
            requested_contract_names=["Vault"],
            target_char_spans=[(0, 120)],
            target_token_ranges=[(10, 40)],
            target_tokens=29,
        )


def test_target_evidence_requires_one_span_per_requested_target() -> None:
    with pytest.raises(ValueError, match="equal length"):
        build_target_evidence(
            contract_id="a" * 64,
            requested_contract_names=["Vault", "Other"],
            target_char_spans=[(0, 120)],
            target_token_ranges=[(10, 40)],
            target_tokens=30,
        )


def test_selector_metadata_round_trips_canonically() -> None:
    evidence = _target_evidence()
    metadata = build_selector_metadata(
        _decision(),
        target_evidence_sha256=evidence["sha256"],
    )
    validate_selector_metadata(metadata)
    assert metadata["selector_config_sha256"] == SELECTOR_CONFIG_SHA256
    assert metadata["target_evidence_sha256"] == evidence["sha256"]


def test_selector_metadata_rejects_drift() -> None:
    evidence = _target_evidence()
    metadata = build_selector_metadata(
        _decision(),
        target_evidence_sha256=evidence["sha256"],
    )

    tampered = copy.deepcopy(metadata)
    tampered["selected_indices"] = list(reversed(tampered["selected_indices"]))
    with pytest.raises(ValueError):
        validate_selector_metadata(tampered)

    extra = copy.deepcopy(metadata)
    extra["unbound_field"] = "not allowed"
    with pytest.raises(ValueError, match="not canonical"):
        validate_selector_metadata(extra)
