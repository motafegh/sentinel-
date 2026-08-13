"""Deterministic run identity for the R4 Phase-8 baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from sentinel_data.vnext.policy import CLASS_NAMES
from ml.src.training.vnext_phase8_config import ARCHITECTURE, FROZEN_ARCHITECTURE, MODEL_VERSION


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_run_binding(
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
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "VALIDATED_G7_CANDIDATE":
        raise ValueError("Phase 8 requires a G7 candidate publication")
    if manifest.get("export_schema_version") != "v2":
        raise ValueError("Phase 8 requires export schema v2")
    if manifest.get("graph_schema_version") != "v9":
        raise ValueError("Phase 8 requires graph schema v9")
    if list(manifest.get("class_order") or []) != list(CLASS_NAMES):
        raise ValueError("Phase 8 class order mismatch")
    rep = manifest.get("representation_binding_report") or {}
    if rep.get("binding_digest_sha256") != expected_representation_digest:
        raise ValueError("Phase 8 representation binding mismatch")
    if not 0.0 < float(weak_positive_weight) <= 1.0:
        raise ValueError("weak_positive_weight must be in (0,1]")

    inputs = manifest.get("inputs") or {}
    payload: dict[str, Any] = {
        "schema": "sentinel-r4-phase8-run-binding-v1",
        "source_commit": str(source_commit),
        "architecture": ARCHITECTURE,
        "model_version": MODEL_VERSION,
        "architecture_config": dict(FROZEN_ARCHITECTURE),
        "class_order": list(CLASS_NAMES),
        "data": {
            "dataset_version": manifest.get("dataset_version"),
            "export_schema_version": manifest.get("export_schema_version"),
            "graph_schema_version": manifest.get("graph_schema_version"),
            "manifest_sha256": sha256_file(manifest_path),
            "representation_binding_digest_sha256": expected_representation_digest,
            "policy_sha256": (inputs.get("policy") or {}).get("sha256"),
            "partition_sha256": (inputs.get("partition_manifest") or {}).get("sha256"),
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
        "limits": {
            "confirmed_negative_cells": 0,
            "threshold_tuning": False,
            "calibration_fit": False,
            "untouched_acceptance": False,
        },
    }
    payload["binding_digest_sha256"] = canonical_digest(payload)
    return payload


__all__ = ["build_run_binding", "canonical_digest", "sha256_file"]
