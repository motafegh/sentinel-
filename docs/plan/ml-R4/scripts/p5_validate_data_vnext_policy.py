#!/usr/bin/env python3
"""Validate the complete R4 Phase-5 DATA vNext semantic contract."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("jsonschema is required: pip install jsonschema") from exc

ROOT = Path("docs/plan/ml-R4")
POLICY_PATH = ROOT / "specs/data_vnext_policy_v1.json"
ROW_SCHEMA_PATH = ROOT / "schemas/data_vnext_label_state_v1.schema.json"

EXPECTED_CLASSES = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]
EXPECTED_DISABLED = {"GasException", "UnusedReturn"}
EXPECTED_ADRS = [
    "ADR-R4-001-label-state-and-training-signal.md",
    "ADR-R4-002-source-class-authority-and-enablement.md",
    "ADR-R4-003-crosswalk-and-aggregation-semantics.md",
    "ADR-R4-004-export-and-ml-consumer-contract.md",
    "ADR-R4-005-lineage-versioning-and-rollback.md",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expect_valid(validator: Draft202012Validator, row: dict) -> None:
    errors = sorted(validator.iter_errors(row), key=lambda e: list(e.path))
    if errors:
        raise AssertionError("valid row rejected: " + "; ".join(e.message for e in errors))


def expect_invalid(validator: Draft202012Validator, row: dict) -> None:
    if not list(validator.iter_errors(row)):
        raise AssertionError("invalid row unexpectedly accepted")


def base_row() -> dict:
    return {
        "policy_version": "data-vnext-policy-v1",
        "contract_id": "a" * 64,
        "class_index": 6,
        "class_name": "Reentrancy",
        "historical_state": "HISTORICAL_POSITIVE",
        "source_claims": [{
            "source": "solidifi",
            "source_record_id": "a" * 64,
            "source_claim_state": "POSITIVE",
            "source_native_label": "Re-entrancy",
            "crosswalk_action": "DIRECT",
            "mapped_class_name": "Reentrancy",
            "evidence_ids": ["R4-PREV-OTHER-SOLIDIFI"],
            "limitations": []
        }],
        "outcome_state": "CONFIRMED_POSITIVE",
        "target_value": 1,
        "training_signal": "POSITIVE",
        "training_strength": "STRONG",
        "loss_eligible": True,
        "outcome_metric_eligible": True,
        "role_eligibility": ["TRAIN_STRONG", "INTERNAL_AUDIT"],
        "policy_decision_id": "R4-D-002",
        "evidence_ids": ["R4-PREV-OTHER-SOLIDIFI"],
        "limitations": []
    }


def validate_row_schema() -> None:
    schema = load_json(ROW_SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)

    strong = base_row()
    expect_valid(validator, strong)

    weak = base_row()
    weak.update({
        "class_index": 8,
        "class_name": "TransactionOrderDependence",
        "outcome_state": "UNKNOWN",
        "training_strength": "WEAK",
        "outcome_metric_eligible": False,
        "role_eligibility": ["TRAIN_WEAK", "EXCLUDE_OUTCOME_METRICS"],
        "policy_decision_id": "R4-D-002"
    })
    weak["source_claims"][0] = {
        "source": "dive",
        "source_record_id": "a" * 64,
        "source_claim_state": "POSITIVE",
        "source_native_label": "Front Running",
        "crosswalk_action": "DIRECT",
        "mapped_class_name": "TransactionOrderDependence",
        "evidence_ids": ["R4-GAP-002"],
        "limitations": ["weak source authority only"]
    }
    expect_valid(validator, weak)

    unlabeled = base_row()
    unlabeled.update({
        "outcome_state": "UNKNOWN",
        "target_value": None,
        "training_signal": "NONE",
        "training_strength": "NONE",
        "loss_eligible": False,
        "outcome_metric_eligible": False,
        "role_eligibility": ["TRAIN_UNLABELED", "EXCLUDE_OUTCOME_METRICS"],
        "policy_decision_id": "R4-D-001"
    })
    expect_valid(validator, unlabeled)

    confirmed_negative = base_row()
    confirmed_negative.update({
        "outcome_state": "CONFIRMED_NEGATIVE",
        "target_value": 0,
        "training_signal": "NEGATIVE",
        "training_strength": "STRONG",
        "loss_eligible": True,
        "outcome_metric_eligible": True,
        "policy_decision_id": "R4-D-FUTURE-NEGATIVE"
    })
    expect_valid(validator, confirmed_negative)

    unknown_as_zero = dict(unlabeled)
    unknown_as_zero.update({
        "target_value": 0,
        "training_signal": "NEGATIVE",
        "training_strength": "STRONG",
        "loss_eligible": True
    })
    expect_invalid(validator, unknown_as_zero)

    weak_metric = dict(weak)
    weak_metric["outcome_metric_eligible"] = True
    expect_invalid(validator, weak_metric)

    masked_with_target = dict(unlabeled)
    masked_with_target["target_value"] = 1
    expect_invalid(validator, masked_with_target)


def validate_policy() -> dict:
    p = load_json(POLICY_PATH)
    assert p["schema"] == "sentinel-data-vnext-policy-v1"
    assert p["policy_version"] == "data-vnext-policy-v1"
    assert p["class_vocabulary"]["classes"] == EXPECTED_CLASSES
    assert p["class_vocabulary"]["locked"] is True
    assert p["class_vocabulary"]["feature_schema_version"] == "v9"

    class_policy = p["class_supervision"]
    assert list(class_policy) == EXPECTED_CLASSES
    disabled = {name for name, cfg in class_policy.items() if cfg["status"] == "SUPERVISION_DISABLED_PENDING_EVIDENCE"}
    assert disabled == EXPECTED_DISABLED
    for name in EXPECTED_DISABLED:
        assert class_policy[name]["approved_strong_positive_sources"] == []
        assert class_policy[name]["approved_weak_positive_sources"] == []

    sources = p["sources"]
    assert sources["solidifi"]["negative_authority"] == "NONE"
    assert sources["smartbugs_curated"]["negative_authority"] == "NONE"
    assert sources["dive"]["negative_authority"] == "NONE"
    assert p["negative_authority"]["first_baseline_blanket_negative_sources"] == []
    assert p["negative_authority"]["weak_negative_training"] == "NOT_AUTHORIZED_IN_V1"

    dive = sources["dive"]
    expected_dive = {
        "Access Control": ("ExternalBug", "NONE", None),
        "Reentrancy": ("Reentrancy", "NONE", None),
        "DoS": ("DenialOfService", "NONE", None),
        "Arithmetic": ("IntegerUO", "NONE", None),
        "Time manipulation": ("Timestamp", "NONE", None),
        "Unchecked Return Values": ("UnusedReturn", "NONE", None),
        "Front Running": ("TransactionOrderDependence", "WEAK", 1),
    }
    assert set(dive["mapped_category_policy"]) == set(expected_dive)
    for native, expected in expected_dive.items():
        got = dive["mapped_category_policy"][native]
        assert (got["canonical_class"], got["training_strength"], got["target_value"]) == expected
    assert dive["no_target_categories"]["Bad Randomness"] == "OUT_OF_TAXONOMY_NO_CANONICAL_TARGET"
    assert set(dive["weak_tod_forbidden_roles"]) >= {"MODEL_SELECTION", "THRESHOLD_FIT", "CALIBRATION_FIT", "UNTOUCHED_ACCEPTANCE"}

    sb = sources["smartbugs_curated"]
    assert sb["no_target_categories"] == {
        "bad_randomness": "LOSSY_NO_CANONICAL_TARGET",
        "short_addresses": "OUT_OF_TAXONOMY_NO_CANONICAL_TARGET",
        "other": "OUT_OF_TAXONOMY_NO_CANONICAL_TARGET"
    }
    assert "NonVulnerable" not in set(sb["approved_mappings"].values())

    for excluded in ("web3bugs", "disl"):
        assert sources[excluded]["rows_allowed"] is False
        assert sources[excluded]["first_baseline_status"] == "EXCLUDED_UNAVAILABLE"
    for deferred in ("bccc", "defihacklabs"):
        assert sources[deferred]["rows_allowed"] is False
        assert sources[deferred]["first_baseline_status"] == "DEFERRED_NOT_IMPORTED"

    assert p["aggregation"]["positive_precedence_over_binary_zero"] == "REMOVED"
    assert p["aggregation"]["global_nonvulnerable_synthesis"] == "FORBIDDEN"
    assert p["export_contract"]["historical_export_schema"] == "v1_immutable"
    assert p["export_contract"]["vnext_export_schema"] == "v2"
    assert p["historical_compatibility"]["mutate_historical_artifacts"] is False
    assert p["historical_compatibility"]["historical_labels_parquet_sha256"] == "26e739b5d82ba512e5a1830817d09609216e2184b79cf4ca7ec2d62ef34e32b5"
    assert p["historical_compatibility"]["phase3_ledger_sha256"] == "3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7"

    for adr in EXPECTED_ADRS:
        text = (ROOT / "adrs" / adr).read_text(encoding="utf-8")
        assert "**Status:** Accepted" in text, adr

    spec = (ROOT / "findings/07_data_vnext_policy_and_design_specification.md").read_text(encoding="utf-8")
    assert "target `0`" in spec
    assert "GasException" in spec and "UnusedReturn" in spec
    assert "Phase 6" in spec and "Phase 8" in spec

    return {
        "passed": True,
        "policy_version": p["policy_version"],
        "classes": len(EXPECTED_CLASSES),
        "enabled_classes": len(EXPECTED_CLASSES) - len(EXPECTED_DISABLED),
        "disabled_classes": sorted(EXPECTED_DISABLED),
        "blanket_negative_sources": 0,
        "dive_weak_positive_strata": ["TransactionOrderDependence"],
        "export_schema": "v2"
    }


def main() -> int:
    validate_row_schema()
    result = validate_policy()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
