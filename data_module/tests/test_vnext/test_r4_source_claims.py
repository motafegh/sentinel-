"""Tests for repaired source-native claim reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel_data.vnext.r4_source_claims import claims_for_meta, load_dive_labels


@pytest.fixture
def policy():
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json").read_text())


def _meta(path: str, *, sha: str = "a" * 64):
    return {
        "sha256": sha,
        "source_records": [
            {
                "source_record_id": "record-1",
                "original_path": path,
                "raw_sha256": "r" * 64,
                "flattened_sha256": "f" * 64,
                "ingestion_entry": {"path": path},
            }
        ],
    }


def test_smartbugs_direct_time_manipulation_is_strong_timestamp(policy):
    rows = claims_for_meta(
        "smartbugs_curated",
        _meta("repo/time_manipulation/timestamp.sol"),
        policy,
    )
    assert len(rows) == 1
    assert rows[0]["native_category"] == "time_manipulation"
    assert rows[0]["mapped_class_name"] == "Timestamp"
    assert rows[0]["training_strength"] == "STRONG"
    assert rows[0]["target_value"] == 1
    assert rows[0]["outcome_state"] == "CONFIRMED_POSITIVE"


def test_smartbugs_bad_randomness_does_not_become_timestamp(policy):
    rows = claims_for_meta(
        "smartbugs_curated",
        _meta("repo/bad_randomness/random.sol"),
        policy,
    )
    assert rows[0]["native_category"] == "bad_randomness"
    assert rows[0]["mapped_class_name"] is None
    assert rows[0]["training_strength"] == "NONE"
    assert rows[0]["target_value"] is None


def test_solidifi_injection_category_is_strong_only_for_mapped_class(policy):
    rows = claims_for_meta(
        "solidifi",
        _meta("repo/buggy_contracts/Re-entrancy/reentrant.sol"),
        policy,
    )
    assert rows[0]["native_category"] == "Re-entrancy"
    assert rows[0]["mapped_class_name"] == "Reentrancy"
    assert rows[0]["training_strength"] == "STRONG"
    assert rows[0]["target_value"] == 1


def test_dive_front_running_is_weak_not_confirmed(policy):
    rows = claims_for_meta(
        "dive",
        _meta("repo/__source__/42.sol"),
        policy,
        dive_labels={"42": {"Front Running"}},
    )
    assert rows[0]["mapped_class_name"] == "TransactionOrderDependence"
    assert rows[0]["training_strength"] == "WEAK"
    assert rows[0]["target_value"] == 1
    assert rows[0]["outcome_state"] == "NOT_REVIEWED"


def test_dive_other_native_positive_stays_masked(policy):
    rows = claims_for_meta(
        "dive",
        _meta("repo/__source__/42.sol"),
        policy,
        dive_labels={"42": {"Arithmetic"}},
    )
    assert rows[0]["mapped_class_name"] == "IntegerUO"
    assert rows[0]["training_strength"] == "NONE"
    assert rows[0]["target_value"] is None


def test_no_claim_path_never_synthesizes_negative(policy):
    rows = claims_for_meta(
        "smartbugs_curated",
        _meta("repo/unknown_category/thing.sol"),
        policy,
    )
    assert rows[0]["source_claim_state"] == "NO_ASSERTION"
    assert rows[0]["target_value"] is None


def test_dive_labels_loader_is_fail_closed_for_nonbinary_cells(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("contractID,Front Running\n1,maybe\n")
    with pytest.raises(ValueError, match="must be binary"):
        load_dive_labels(path)
