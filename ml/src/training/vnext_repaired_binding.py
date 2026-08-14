"""Run binding for the repaired-v2 bounded Phase-8 GPU smoke.

This is deliberately separate from :mod:`vnext_binding`, whose contract is the
historical G7-passed v1 baseline.  The repaired binding accepts only a physically
bound ``sentinel-r4-vnext-v2`` candidate and still binds the frozen model/runtime
identity.  It does not authorize the 100-epoch run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ml.src.training.vnext_binding import (
    canonical_digest,
    runtime_binding_metadata,
    sha256_file,
)
from ml.src.training.vnext_phase8_config import (
    ARCHITECTURE,
    FROZEN_ARCHITECTURE,
    MODEL_VERSION,
)
from sentinel_data.preprocessing.r4_versions import (
    REPAIRED_DATA_PUBLICATION_ID,
    REPAIRED_EVIDENCE_LEDGER_ID,
    REPAIRED_ROLE_PARTITION_ID,
)
from sentinel_data.vnext.policy import CLASS_NAMES


def build_repaired_smoke_binding(
    *,
    source_commit: str,
    manifest_path: Path,
    expected_representation_digest: str,
    seed: int,
    weak_positive_weight: float,
    optimizer_config: Mapping[str, Any],
    train_contracts: int,
    train_groups: int,
    selection_contracts: int,
    selection_groups: int,
) -> dict[str, Any]:
    """Build a deterministic identity for the bounded repaired-data GPU smoke."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != REPAIRED_DATA_PUBLICATION_ID:
        raise ValueError("repaired smoke requires sentinel-r4-vnext-v2")
    if manifest.get("export_schema_version") != "v2":
        raise ValueError("repaired smoke requires export schema v2")
    if manifest.get("ledger_version") != REPAIRED_EVIDENCE_LEDGER_ID:
        raise ValueError("repaired smoke evidence-ledger identity mismatch")
    if manifest.get("partition_version") != REPAIRED_ROLE_PARTITION_ID:
        raise ValueError("repaired smoke partition identity mismatch")
    if manifest.get("status") != "REPAIRED_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED":
        raise ValueError("repaired smoke requires a physically bound local publication")
    if manifest.get("confirmed_negative_rows") != 0:
        raise ValueError("repaired smoke refuses confirmed-negative rows")

    representation = manifest.get("representation_binding_report") or {}
    if representation.get("binding_digest_sha256") != expected_representation_digest:
        raise ValueError("repaired smoke representation binding mismatch")
    if not 0.0 < float(weak_positive_weight) <= 1.0:
        raise ValueError("weak_positive_weight must be in (0,1]")

    artifacts = manifest.get("artifacts") or {}
    payload: dict[str, Any] = {
        "schema": "sentinel-r4-phase8-repaired-smoke-binding-v1",
        "scope": "bounded_gpu_smoke_not_full_training_authorization",
        "source_commit": str(source_commit),
        "architecture": ARCHITECTURE,
        "model_version": MODEL_VERSION,
        "architecture_config": dict(FROZEN_ARCHITECTURE),
        "class_order": list(CLASS_NAMES),
        "data": {
            "dataset_version": manifest.get("dataset_version"),
            "export_schema_version": manifest.get("export_schema_version"),
            "ledger_version": manifest.get("ledger_version"),
            "partition_version": manifest.get("partition_version"),
            "manifest_sha256": sha256_file(manifest_path),
            "representation_binding_digest_sha256": expected_representation_digest,
            "policy_sha256": (artifacts.get("policy") or {}).get("sha256"),
            "grouping_sha256": (artifacts.get("grouping") or {}).get("sha256"),
            "claims_sha256": (artifacts.get("claims") or {}).get("sha256"),
            "ml_targets_sha256": (artifacts.get("ml_targets") or {}).get("sha256"),
        },
        "roles": {
            "training": ["TRAIN_STRONG", "TRAIN_WEAK"],
            "model_selection": ["MODEL_SELECTION"],
            "train_contracts": int(train_contracts),
            "train_groups": int(train_groups),
            "selection_contracts": int(selection_contracts),
            "selection_groups": int(selection_groups),
        },
        "seed": int(seed),
        "weak_positive_weight": float(weak_positive_weight),
        "optimizer": dict(optimizer_config),
        "runtime": runtime_binding_metadata(),
        "limits": {
            "confirmed_negative_cells": 0,
            "threshold_tuning": False,
            "calibration_fit": False,
            "untouched_acceptance": False,
            "full_training_authorized": False,
        },
    }
    payload["binding_digest_sha256"] = canonical_digest(payload)
    return payload


__all__ = ["build_repaired_smoke_binding"]
