"""Repository-safe tests for the repaired-v2 Phase-8 ML adapter."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from ml.src.datasets.vnext_repaired_dataset import RepairedVNextTrainingDataset
from sentinel_data.vnext.policy import CLASS_NAMES


def _row(contract_id: str, role: str, group_id: str, *, signal_class: int) -> dict:
    row = {
        "contract_id": contract_id,
        "source": "fixture",
        "group_id": group_id,
        "role": role,
        "representation_required": True,
    }
    for index in range(len(CLASS_NAMES)):
        active = index == signal_class
        is_train = role in {"TRAIN_STRONG", "TRAIN_WEAK"}
        is_selection = role == "MODEL_SELECTION"
        strength = (
            "STRONG"
            if active and role in {"TRAIN_STRONG", "MODEL_SELECTION"}
            else "WEAK"
            if active and role == "TRAIN_WEAK"
            else "NONE"
        )
        row[f"target_{index}"] = 1 if active else None
        row[f"strength_{index}"] = strength
        row[f"source_loss_eligible_{index}"] = bool(active and is_train)
        row[f"effective_loss_mask_{index}"] = bool(active and is_train)
        row[f"outcome_metric_mask_{index}"] = bool(active and is_selection)
        row[f"outcome_state_{index}"] = (
            "CONFIRMED_POSITIVE" if active and strength == "STRONG" else "NOT_REVIEWED"
        )
        row[f"policy_decision_id_{index}"] = "fixture"
    return row


def _publication(tmp_path):
    overlay = tmp_path / "overlay"
    reps = tmp_path / "reps"
    overlay.mkdir()
    (reps / "fixture").mkdir(parents=True)
    rows = [
        _row("a" * 64, "TRAIN_STRONG", "g1", signal_class=0),
        _row("b" * 64, "TRAIN_WEAK", "g2", signal_class=1),
        _row("c" * 64, "MODEL_SELECTION", "g3", signal_class=2),
    ]
    pq.write_table(pa.Table.from_pylist(rows), overlay / "ml_targets.parquet")
    digest = "d" * 64
    manifest = {
        "dataset_version": "sentinel-r4-vnext-v2",
        "export_schema_version": "v2",
        "partition_version": "r4-vnext-roles-v2",
        "status": "REPAIRED_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED",
        "confirmed_negative_rows": 0,
        "role_contract_counts": {
            "TRAIN_STRONG": 1,
            "TRAIN_WEAK": 1,
            "MODEL_SELECTION": 1,
        },
        "representation_binding_report": {
            "binding_digest_sha256": digest,
        },
    }
    (overlay / "manifest.json").write_text(json.dumps(manifest))
    (overlay / "representation_binding_report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "graph_schema_version": "v9",
                "binding_digest_sha256": digest,
            }
        )
    )
    return overlay, reps, digest


def test_repaired_adapter_uses_dynamic_training_counts(tmp_path):
    overlay, reps, digest = _publication(tmp_path)
    dataset = RepairedVNextTrainingDataset(
        overlay_dir=overlay,
        representations_root=reps,
        roles=("TRAIN_STRONG", "TRAIN_WEAK"),
        expected_binding_digest=digest,
    )
    assert dataset.frozen_role_counts == {"TRAIN_STRONG": 1, "TRAIN_WEAK": 1}
    assert dataset.role_counts == {"TRAIN_STRONG": 1, "TRAIN_WEAK": 1}
    assert dataset.frozen_group_count == 2
    assert dataset.group_count == 2
    assert len(dataset) == 2


def test_repaired_adapter_keeps_model_selection_separate(tmp_path):
    overlay, reps, digest = _publication(tmp_path)
    dataset = RepairedVNextTrainingDataset(
        overlay_dir=overlay,
        representations_root=reps,
        roles=("MODEL_SELECTION",),
        expected_binding_digest=digest,
    )
    assert dataset.frozen_role_counts == {"MODEL_SELECTION": 1}
    assert dataset.role_counts == {"MODEL_SELECTION": 1}
    assert dataset.group_count == 1
    assert len(dataset) == 1


def test_repaired_adapter_refuses_wrong_binding_digest(tmp_path):
    overlay, reps, _ = _publication(tmp_path)
    try:
        RepairedVNextTrainingDataset(
            overlay_dir=overlay,
            representations_root=reps,
            roles=("TRAIN_STRONG",),
            expected_binding_digest="x" * 64,
        )
    except ValueError as exc:
        assert "binding mismatch" in str(exc)
    else:
        raise AssertionError("repaired adapter must fail on binding mismatch")
