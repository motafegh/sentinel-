"""Deterministic bounded-pilot binding for the accepted R4 guarded lineage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ml.src.datasets.vnext_logical_v3_guarded_dataset import (
    R4_D011_BINDING_DIGEST,
    R4_D013_BINDING_DIGEST,
    R4_D013_DECISION_ID,
    validate_guarded_physical_acceptance,
    validate_logical_v3_acceptance,
)
from ml.src.training.vnext_binding import canonical_digest, runtime_binding_metadata, sha256_file
from ml.src.training.vnext_guarded_run_control import (
    R4_D014_DECISION_ID,
    R4_D014_EXECUTION_SCOPE,
    R4_D014_OBJECTIVE_ID,
)
from ml.src.training.vnext_phase8_config import (
    ARCHITECTURE,
    FROZEN_ARCHITECTURE,
    GRAPHCODEBERT_MODEL_NAME,
    GRAPHCODEBERT_REVISION,
    MODEL_VERSION,
)
from sentinel_data.preprocessing.r4_versions import (
    GUARDED_TOKEN_LINEAGE_VERSION,
    GUARDED_TOKEN_SELECTOR_VERSION,
    HISTORICAL_TOKEN_SELECTOR_VERSION,
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.vnext.policy import CLASS_NAMES
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)

R4_D014_DECISION_SCHEMA = "sentinel-r4-phase8-objective-evaluation-decision-v1"
R4_D014_DECISION = "ACCEPT_BOUNDED_POSITIVE_ONLY_CONTROL_HOLD_FULL_TRAINING"
GUARDED_RUN_BINDING_SCHEMA = "sentinel-r4-phase8-guarded-run-binding-v1"


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_r4_d014_decision(
    *,
    decision_path: Path,
    optimizer_config: Mapping[str, Any],
    weak_positive_weight: float,
) -> dict[str, Any]:
    decision = _load_json(decision_path)
    if decision.get("schema") != R4_D014_DECISION_SCHEMA:
        raise ValueError("R4-D-014 decision schema mismatch")
    if decision.get("decision_id") != R4_D014_DECISION_ID:
        raise ValueError("R4-D-014 decision ID mismatch")
    if decision.get("status") != "PASS" or decision.get("decision") != R4_D014_DECISION:
        raise ValueError("R4-D-014 objective/evaluation decision is not accepted")
    if decision.get("g8_passed") is not False or decision.get("full_training_authorized") is not False:
        raise ValueError("R4-D-014 must retain G8/full-training hold")

    physical = decision.get("accepted_physical_input") or {}
    if physical.get("logical_dataset_version") != DATASET_VERSION_V3:
        raise ValueError("R4-D-014 logical dataset mismatch")
    if physical.get("logical_partition_version") != ROLE_PARTITION_VERSION_V3:
        raise ValueError("R4-D-014 logical partition mismatch")
    if physical.get("graph_parent_decision_id") != "R4-D-011":
        raise ValueError("R4-D-014 graph-parent decision mismatch")
    if physical.get("graph_parent_binding_digest_sha256") != R4_D011_BINDING_DIGEST:
        raise ValueError("R4-D-014 graph-parent digest mismatch")
    if physical.get("token_decision_id") != R4_D013_DECISION_ID:
        raise ValueError("R4-D-014 token decision mismatch")
    if physical.get("token_lineage") != GUARDED_TOKEN_LINEAGE_VERSION:
        raise ValueError("R4-D-014 token lineage mismatch")
    if physical.get("token_binding_digest_sha256") != R4_D013_BINDING_DIGEST:
        raise ValueError("R4-D-014 token digest mismatch")
    if physical.get("selector_policy") != GUARDED_TOKEN_SELECTOR_VERSION:
        raise ValueError("R4-D-014 selector policy mismatch")

    objective = decision.get("objective") or {}
    if objective.get("objective_id") != R4_D014_OBJECTIVE_ID:
        raise ValueError("R4-D-014 objective identity mismatch")
    if objective.get("optimizer_authorized_roles") != ["TRAIN_STRONG", "TRAIN_WEAK"]:
        raise ValueError("R4-D-014 optimizer role authority mismatch")
    if objective.get("optimizer_authorized_target_values") != [1.0]:
        raise ValueError("R4-D-014 optimizer target authority mismatch")
    if objective.get("unknown_cells") != "MASKED_NO_OPTIMIZER_AUTHORITY":
        raise ValueError("R4-D-014 unknown-cell policy mismatch")
    if objective.get("confirmed_negative_optimizer_authority") is not False:
        raise ValueError("R4-D-014 unexpectedly authorizes negative optimizer cells")
    if objective.get("pu_optimizer_authority") is not False:
        raise ValueError("R4-D-014 unexpectedly authorizes PU")
    if float(objective.get("weak_positive_weight", -1.0)) != float(weak_positive_weight):
        raise ValueError("R4-D-014 weak-positive weight mismatch")

    expected_optimizer = {
        "objective": R4_D014_OBJECTIVE_ID,
        "objective_authority_decision_id": R4_D014_DECISION_ID,
        "execution_scope": R4_D014_EXECUTION_SCOPE,
        "model_selection_interpretation": "positive_fit_diagnostic_only",
        "full_training_authorized": False,
        "weak_positive_weight": float(objective["weak_positive_weight"]),
        "aux_loss_weight": float(objective["aux_loss_weight"]),
        "aux_loss_warmup_epochs": int(objective["aux_loss_warmup_epochs"]),
        "aux_phase2_loss_weight": float(objective["phase2_aux_loss_weight"]),
        "jk_entropy_reg_lambda": float(objective["jk_entropy_reg_lambda"]),
        "label_smoothing": float(objective["label_smoothing"]),
        "legacy_label_sampler": bool(objective["legacy_label_sampler"]),
        "threshold_tuning": False,
        "calibration_fit": False,
        "untouched_acceptance": False,
    }
    for key, expected in expected_optimizer.items():
        if optimizer_config.get(key) != expected:
            raise ValueError(f"R4-D-014 optimizer config mismatch for {key}: {optimizer_config.get(key)!r} != {expected!r}")

    if int(optimizer_config.get("epochs", 0)) <= 0 or int(optimizer_config.get("epochs", 0)) >= 100:
        raise ValueError("R4-D-014 binding requires a positive bounded horizon below 100 epochs")

    evaluation = decision.get("evaluation") or {}
    if evaluation.get("model_selection_role") != "MODEL_SELECTION":
        raise ValueError("R4-D-014 model-selection role mismatch")
    if evaluation.get("model_selection_interpretation") != "POSITIVE_FIT_DIAGNOSTIC_ONLY":
        raise ValueError("R4-D-014 model-selection interpretation mismatch")
    for field in (
        "discrimination_metrics_authorized",
        "threshold_fit_authorized",
        "calibration_fit_authorized",
        "untouched_acceptance_authorized",
        "production_promotion_authorized",
    ):
        if evaluation.get(field) is not False:
            raise ValueError(f"R4-D-014 unexpectedly enables {field}")

    pilot = decision.get("pilot_boundary") or {}
    if pilot.get("m3_lineage_integration_authorized") is not True:
        raise ValueError("R4-D-014 does not authorize M3 integration")
    if pilot.get("m4_bounded_pilot_authorized_after_preflight") is not True:
        raise ValueError("R4-D-014 does not authorize bounded M4 pilot")
    if pilot.get("full_training_authorized") is not False:
        raise ValueError("R4-D-014 pilot boundary unexpectedly authorizes full training")
    return decision


def build_guarded_run_binding(
    *,
    source_commit: str,
    repo_root: Path,
    manifest_path: Path,
    logical_acceptance_path: Path,
    physical_acceptance_path: Path,
    objective_decision_path: Path,
    representations_root: Path,
    seed: int,
    weak_positive_weight: float,
    optimizer_config: Mapping[str, Any],
    train_contracts: int,
    train_groups: int,
    selection_contracts: int,
    selection_groups: int,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    logical_acceptance_path = Path(logical_acceptance_path).resolve()
    physical_acceptance_path = Path(physical_acceptance_path).resolve()
    objective_decision_path = Path(objective_decision_path).resolve()
    representations_root = Path(representations_root).resolve()

    logical_manifest = _load_json(manifest_path)
    logical_acceptance = validate_logical_v3_acceptance(
        manifest_path=manifest_path,
        acceptance_path=logical_acceptance_path,
    )
    physical_acceptance, candidate_manifest, _ = validate_guarded_physical_acceptance(
        repo_root=repo_root,
        representations_root=representations_root,
        acceptance_path=physical_acceptance_path,
        expected_binding_digest=R4_D013_BINDING_DIGEST,
    )
    objective_decision = validate_r4_d014_decision(
        decision_path=objective_decision_path,
        optimizer_config=optimizer_config,
        weak_positive_weight=weak_positive_weight,
    )

    architecture_config = dict(FROZEN_ARCHITECTURE)
    architecture_config["graph_schema_version"] = V10_GRAPH_SCHEMA_VERSION
    artifacts = logical_manifest.get("artifacts") or {}
    logical_lineage = logical_acceptance.get("lineage") or {}
    physical_lineage = physical_acceptance.get("accepted_lineage") or {}

    payload: dict[str, Any] = {
        "schema": GUARDED_RUN_BINDING_SCHEMA,
        "scope": R4_D014_EXECUTION_SCOPE,
        "source_commit": str(source_commit),
        "architecture": ARCHITECTURE,
        "model_version": MODEL_VERSION,
        "architecture_config": architecture_config,
        "class_order": list(CLASS_NAMES),
        "data": {
            "logical_authority": {
                "decision_id": "R4-D-009",
                "dataset_version": DATASET_VERSION_V3,
                "grouping_version": GROUPING_VERSION_V3,
                "partition_version": ROLE_PARTITION_VERSION_V3,
                "manifest_sha256": sha256_file(manifest_path),
                "acceptance_record_sha256": sha256_file(logical_acceptance_path),
                "publication_manifest_sha256": logical_lineage.get("publication_manifest_sha256"),
                "policy_sha256": (artifacts.get("policy") or {}).get("sha256"),
                "grouping_sha256": (artifacts.get("grouping") or {}).get("sha256"),
                "claims_sha256": (artifacts.get("claims") or {}).get("sha256"),
                "ml_targets_sha256": (artifacts.get("ml_targets") or {}).get("sha256"),
            },
            "graph_parent": {
                "decision_id": "R4-D-011",
                "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
                "representation_extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
                "binding_digest_sha256": R4_D011_BINDING_DIGEST,
            },
            "guarded_tokens": {
                "decision_id": R4_D013_DECISION_ID,
                "representation_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
                "selector_policy": GUARDED_TOKEN_SELECTOR_VERSION,
                "control_selector": HISTORICAL_TOKEN_SELECTOR_VERSION,
                "binding_digest_sha256": R4_D013_BINDING_DIGEST,
                "physical_acceptance_record_sha256": sha256_file(physical_acceptance_path),
                "candidate_manifest_sha256": physical_lineage.get("candidate_manifest_sha256"),
                "candidate_source_commit": candidate_manifest.get("source_commit"),
                "physical_root": physical_lineage.get("physical_root"),
            },
        },
        "objective_evaluation": {
            "decision_id": R4_D014_DECISION_ID,
            "objective_id": R4_D014_OBJECTIVE_ID,
            "decision_record_sha256": sha256_file(objective_decision_path),
            "model_selection_interpretation": "positive_fit_diagnostic_only",
            "m4_bounded_pilot_authorized_after_preflight": bool((objective_decision.get("pilot_boundary") or {}).get("m4_bounded_pilot_authorized_after_preflight")),
            "full_training_authorized": False,
        },
        "roles": {
            "training": ["TRAIN_STRONG", "TRAIN_WEAK"],
            "model_selection": ["MODEL_SELECTION"],
            "internal_audit_for_selection": False,
            "train_contracts": int(train_contracts),
            "train_groups": int(train_groups),
            "selection_contracts": int(selection_contracts),
            "selection_groups": int(selection_groups),
        },
        "seed": int(seed),
        "weak_positive_weight": float(weak_positive_weight),
        "optimizer": dict(optimizer_config),
        "runtime": runtime_binding_metadata(),
        "pretrained_backbone": {
            "model_name": GRAPHCODEBERT_MODEL_NAME,
            "revision": GRAPHCODEBERT_REVISION,
        },
        "limits": {
            "confirmed_negative_cells": 0,
            "negative_optimizer_authority": False,
            "pu_optimizer_authority": False,
            "threshold_tuning": False,
            "calibration_fit": False,
            "untouched_acceptance": False,
            "production_promotion": False,
            "historical_checkpoint_reuse": False,
            "full_training_authorized": False,
            "g8_passed": False,
        },
    }
    payload["binding_digest_sha256"] = canonical_digest(payload)
    return payload


__all__ = [
    "GUARDED_RUN_BINDING_SCHEMA",
    "R4_D014_DECISION",
    "R4_D014_DECISION_SCHEMA",
    "build_guarded_run_binding",
    "validate_r4_d014_decision",
]
