"""Deterministic run identity for the R4 Phase-8 baseline."""
from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
from pathlib import Path
from typing import Any, Mapping

from sentinel_data.vnext.policy import CLASS_NAMES
from ml.src.training.vnext_phase8_config import (
    ARCHITECTURE,
    FROZEN_ARCHITECTURE,
    GRAPHCODEBERT_MODEL_NAME,
    GRAPHCODEBERT_REVISION,
    MODEL_VERSION,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Phase-8 runtime dependency is not installed: {name}"
        ) from exc


def runtime_binding_metadata() -> dict[str, Any]:
    """Resolve and fail-close the software/backbone runtime used by Phase 8.

    The generic TransformerEncoder intentionally remains unchanged. Phase 8
    instead proves that the mutable model name currently resolves to the exact
    accepted cached snapshot before any optimizer step can occur, then binds
    that snapshot and the effective ML software stack into the run digest.
    """
    import torch
    from transformers import AutoConfig

    resolved = AutoConfig.from_pretrained(
        GRAPHCODEBERT_MODEL_NAME,
        local_files_only=True,
    )
    resolved_name = str(getattr(resolved, "_name_or_path", "") or "")
    resolved_revision = str(getattr(resolved, "_commit_hash", "") or "")
    if resolved_name != GRAPHCODEBERT_MODEL_NAME:
        raise RuntimeError(
            "Phase-8 GraphCodeBERT model-name mismatch: "
            f"{resolved_name!r} != {GRAPHCODEBERT_MODEL_NAME!r}"
        )
    if resolved_revision != GRAPHCODEBERT_REVISION:
        raise RuntimeError(
            "Phase-8 GraphCodeBERT mutable-name resolution mismatch: "
            f"{resolved_revision!r} != {GRAPHCODEBERT_REVISION!r}"
        )

    pinned = AutoConfig.from_pretrained(
        GRAPHCODEBERT_MODEL_NAME,
        revision=GRAPHCODEBERT_REVISION,
        local_files_only=True,
    )
    pinned_revision = str(getattr(pinned, "_commit_hash", "") or "")
    if pinned_revision != GRAPHCODEBERT_REVISION:
        raise RuntimeError(
            "Phase-8 pinned GraphCodeBERT snapshot could not be resolved exactly: "
            f"{pinned_revision!r} != {GRAPHCODEBERT_REVISION!r}"
        )

    packages = {
        "numpy": _distribution_version("numpy"),
        "pandas": _distribution_version("pandas"),
        "peft": _distribution_version("peft"),
        "pyarrow": _distribution_version("pyarrow"),
        "torch-geometric": _distribution_version("torch-geometric"),
        "transformers": _distribution_version("transformers"),
    }
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_compiled_version": None
            if torch.version.cuda is None
            else str(torch.version.cuda),
            "cudnn_version": None
            if not hasattr(torch.backends, "cudnn")
            else torch.backends.cudnn.version(),
        },
        "packages": packages,
        "pretrained_backbone": {
            "model_name": GRAPHCODEBERT_MODEL_NAME,
            "revision": GRAPHCODEBERT_REVISION,
        },
    }


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
        "runtime": runtime_binding_metadata(),
        "limits": {
            "confirmed_negative_cells": 0,
            "threshold_tuning": False,
            "calibration_fit": False,
            "untouched_acceptance": False,
        },
    }
    payload["binding_digest_sha256"] = canonical_digest(payload)
    return payload


__all__ = [
    "build_run_binding",
    "canonical_digest",
    "runtime_binding_metadata",
    "sha256_file",
]
