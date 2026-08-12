from __future__ import annotations

import json
from pathlib import Path

from sentinel_data.vnext.policy import (
    crosswalk_action,
    effective_loss_mask,
    semantic_decision,
    source_claim_state,
)


ROOT = Path(__file__).resolve().parents[3]
POLICY = json.loads((ROOT / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json").read_text())


def row(source: str, class_name: str, class_index: int, target: int, *, crosswalk: str = "UNKNOWN") -> dict:
    return {
        "contract_id": "a" * 64,
        "primary_source": source,
        "class_name": class_name,
        "class_index": class_index,
        "historical_target": target,
        "source_native_state": "EXPLICIT_POSITIVE" if target else "ABSENT",
        "crosswalk_action": crosswalk,
    }


def test_solidifi_positive_is_strong_but_only_train_role_gets_loss() -> None:
    r = row("solidifi", "Reentrancy", 6, 1, crosswalk="DIRECT")
    d = semantic_decision(r, POLICY, "TRAIN_STRONG")
    assert d.outcome_state == "CONFIRMED_POSITIVE"
    assert d.target_value == 1
    assert d.training_strength == "STRONG"
    assert d.outcome_metric_eligible is False
    assert effective_loss_mask(d, "TRAIN_STRONG") is True
    assert effective_loss_mask(d, "MODEL_SELECTION") is False

    holdout = semantic_decision(r, POLICY, "MODEL_SELECTION")
    assert holdout.outcome_metric_eligible is True
    assert effective_loss_mask(holdout, "MODEL_SELECTION") is False


def test_smartbugs_approved_positive_is_strong_and_crosswalk_is_recovered() -> None:
    r = row("smartbugs_curated", "ExternalBug", 2, 1, crosswalk="UNKNOWN")
    d = semantic_decision(r, POLICY, "INTERNAL_AUDIT")
    assert d.outcome_state == "CONFIRMED_POSITIVE"
    assert d.training_strength == "STRONG"
    assert d.outcome_metric_eligible is True
    assert crosswalk_action(r) == ("DIRECT", "ExternalBug")


def test_smartbugs_timestamp_fails_closed_to_no_target() -> None:
    r = row("smartbugs_curated", "Timestamp", 7, 1, crosswalk="UNKNOWN")
    d = semantic_decision(r, POLICY, "TRAIN_STRONG")
    assert d.outcome_state == "NOT_REVIEWED"
    assert d.target_value is None
    assert d.training_strength == "NONE"
    assert d.source_policy_loss_eligible is False
    assert crosswalk_action(r) == ("LOSSY_NO_CANONICAL_TARGET", None)


def test_dive_tod_is_the_only_weak_positive_path() -> None:
    tod = row("dive", "TransactionOrderDependence", 8, 1, crosswalk="DIRECT")
    d = semantic_decision(tod, POLICY, "TRAIN_WEAK")
    assert d.outcome_state == "NOT_REVIEWED"
    assert d.target_value == 1
    assert d.training_strength == "WEAK"
    assert d.outcome_metric_eligible is False
    assert effective_loss_mask(d, "TRAIN_WEAK") is True
    assert effective_loss_mask(d, "TRAIN_STRONG") is False

    re = row("dive", "Reentrancy", 6, 1, crosswalk="DIRECT")
    masked = semantic_decision(re, POLICY, "TRAIN_UNLABELED")
    assert masked.target_value is None
    assert masked.training_strength == "NONE"
    assert masked.outcome_state == "NOT_REVIEWED"


def test_historical_zero_never_becomes_negative_target() -> None:
    for source in ("dive", "solidifi", "smartbugs_curated"):
        r = row(source, "Reentrancy", 6, 0)
        d = semantic_decision(r, POLICY, "TRAIN_UNLABELED")
        assert d.outcome_state == "UNKNOWN"
        assert d.target_value is None
        assert d.training_signal == "NONE"
        assert d.training_strength == "NONE"
        assert d.source_policy_loss_eligible is False


def test_disabled_classes_keep_position_but_have_no_supervision() -> None:
    for class_name, idx in (("GasException", 3), ("UnusedReturn", 9)):
        r = row("dive", class_name, idx, 1, crosswalk="DIRECT")
        d = semantic_decision(r, POLICY, "TRAIN_STRONG")
        assert d.target_value is None
        assert d.training_strength == "NONE"
        assert d.source_policy_loss_eligible is False
        assert d.outcome_metric_eligible is False


def test_historical_source_state_mapping_never_calls_absence_negative() -> None:
    r = row("solidifi", "Reentrancy", 6, 0)
    r["source_native_state"] = "ABSENT"
    assert source_claim_state(r) == "NO_ASSERTION"
    r["source_native_state"] = "MAPPED_NONVULNERABLE"
    assert source_claim_state(r) == "OUT_OF_TAXONOMY"
