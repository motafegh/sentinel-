"""Run-control helpers for the R4-D-013 / R4-D-014 bounded pilot seam."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ml.src.training.vnext_phase8_config import Phase8Settings
from ml.src.training.vnext_run_control import optimizer_binding_config

R4_D014_DECISION_ID = "R4-D-014"
R4_D014_OBJECTIVE_ID = "masked_positive_bce_control_v1"
R4_D014_EXECUTION_SCOPE = "bounded_pilot_only"
FULL_PHASE8_HORIZON_EPOCHS = Phase8Settings().epochs


def _load_logical_acceptance(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise ValueError("logical-v3 acceptance record is not passing")
    return value


def validate_guarded_phase8_populations(
    train_ds: Any,
    selection_ds: Any,
    *,
    logical_acceptance_path: Path,
) -> None:
    acceptance = _load_logical_acceptance(logical_acceptance_path)
    frozen = acceptance.get("role_contract_counts") or {}
    group_counts = acceptance.get("role_group_counts") or {}
    active = acceptance.get("active_supervision") or {}
    active_train = active.get("optimizer_contracts_by_role") or {}

    expected_train_frozen = {
        "TRAIN_STRONG": int(frozen.get("TRAIN_STRONG", -1)),
        "TRAIN_WEAK": int(frozen.get("TRAIN_WEAK", -1)),
    }
    expected_train_active = {
        "TRAIN_STRONG": int(active_train.get("TRAIN_STRONG", -1)),
        "TRAIN_WEAK": int(active_train.get("TRAIN_WEAK", -1)),
    }
    expected_train_groups = int(group_counts.get("TRAIN_STRONG", -1)) + int(group_counts.get("TRAIN_WEAK", -1))
    expected_train_contracts = int(active.get("optimizer_contracts", -1))
    expected_active_groups = int(active.get("optimizer_groups", -1))
    expected_train_skipped = {
        role: expected_train_frozen[role] - expected_train_active[role]
        for role in expected_train_frozen
        if expected_train_frozen[role] - expected_train_active[role] > 0
    }

    expected_selection_frozen = {"MODEL_SELECTION": int(frozen.get("MODEL_SELECTION", -1))}
    expected_selection_active = {"MODEL_SELECTION": int(active.get("model_selection_contracts", -1))}
    expected_selection_groups = int(active.get("model_selection_groups", -1))
    expected_selection_skipped = {
        "MODEL_SELECTION": expected_selection_frozen["MODEL_SELECTION"] - expected_selection_active["MODEL_SELECTION"]
    }
    if expected_selection_skipped["MODEL_SELECTION"] <= 0:
        expected_selection_skipped = {}

    checks = [
        (train_ds.frozen_role_counts == expected_train_frozen, f"unexpected logical-v3 frozen training roles: {train_ds.frozen_role_counts}"),
        (train_ds.role_counts == expected_train_active, f"unexpected logical-v3 active training roles: {train_ds.role_counts}"),
        (train_ds.frozen_group_count == expected_train_groups, f"unexpected logical-v3 frozen training groups: {train_ds.frozen_group_count}"),
        (len(train_ds) == expected_train_contracts, f"unexpected logical-v3 active training contracts: {len(train_ds)}"),
        (train_ds.group_count == expected_active_groups, f"unexpected logical-v3 active training groups: {train_ds.group_count}"),
        (train_ds.skipped_no_signal_counts == expected_train_skipped, f"unexpected logical-v3 training no-signal counts: {train_ds.skipped_no_signal_counts}"),
        (selection_ds.frozen_role_counts == expected_selection_frozen, f"unexpected logical-v3 frozen MODEL_SELECTION roles: {selection_ds.frozen_role_counts}"),
        (selection_ds.role_counts == expected_selection_active, f"unexpected logical-v3 active MODEL_SELECTION roles: {selection_ds.role_counts}"),
        (len(selection_ds) == expected_selection_active["MODEL_SELECTION"], f"unexpected logical-v3 active MODEL_SELECTION contracts: {len(selection_ds)}"),
        (selection_ds.frozen_group_count == int(group_counts.get("MODEL_SELECTION", -1)), f"unexpected logical-v3 frozen MODEL_SELECTION groups: {selection_ds.frozen_group_count}"),
        (selection_ds.group_count == expected_selection_groups, f"unexpected logical-v3 active MODEL_SELECTION groups: {selection_ds.group_count}"),
        (selection_ds.skipped_no_signal_counts == expected_selection_skipped, f"unexpected logical-v3 MODEL_SELECTION no-signal counts: {selection_ds.skipped_no_signal_counts}"),
    ]
    for passed, message in checks:
        if not passed:
            raise RuntimeError(message)


def guarded_optimizer_binding_config(
    *,
    settings: Phase8Settings,
    parameter_groups: list[dict[str, Any]],
    scheduler_metadata: Mapping[str, Any],
    num_workers: int,
    milestone_interval_epochs: int,
) -> dict[str, Any]:
    if int(settings.epochs) >= int(FULL_PHASE8_HORIZON_EPOCHS):
        raise ValueError(
            "R4-D-014 authorizes only a bounded pilot; configured epochs must be below the 100-epoch full horizon"
        )
    payload = optimizer_binding_config(
        settings=settings,
        parameter_groups=parameter_groups,
        scheduler_metadata=scheduler_metadata,
        num_workers=num_workers,
        milestone_interval_epochs=milestone_interval_epochs,
    )
    payload["objective"] = R4_D014_OBJECTIVE_ID
    payload["objective_authority_decision_id"] = R4_D014_DECISION_ID
    payload["execution_scope"] = R4_D014_EXECUTION_SCOPE
    payload["model_selection_interpretation"] = "positive_fit_diagnostic_only"
    payload["full_training_authorized"] = False
    return payload


__all__ = [
    "FULL_PHASE8_HORIZON_EPOCHS",
    "R4_D014_DECISION_ID",
    "R4_D014_EXECUTION_SCOPE",
    "R4_D014_OBJECTIVE_ID",
    "guarded_optimizer_binding_config",
    "validate_guarded_phase8_populations",
]
