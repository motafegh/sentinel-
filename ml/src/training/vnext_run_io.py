"""Deterministic durable run artifacts for R4 Phase 8."""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ml.src.training.vnext_checkpoint import (
    assert_checkpoint_binding,
    checkpoint_artifact_identity,
    load_checkpoint,
)

RUN_MANIFEST_SCHEMA = "sentinel-r4-phase8-run-manifest-v1"
CHECKPOINT_INDEX_SCHEMA = "sentinel-r4-phase8-checkpoint-index-v1"


def _json_text(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write canonical JSON atomically and fsync the file before promotion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def append_epoch_jsonl(path: Path, payload: Mapping[str, Any]) -> bool:
    """Append exactly one canonical event per epoch.

    Re-appending the identical final epoch is idempotent. A conflicting event,
    skipped epoch, or rewrite of historical epochs fails closed.
    """
    path = Path(path)
    event = dict(payload)
    if "epoch" not in event:
        raise ValueError("epoch JSONL event lacks epoch")
    epoch = int(event["epoch"])
    if epoch < 1:
        raise ValueError("epoch JSONL event epoch must be >= 1")
    line = _json_text(event)

    last_epoch = 0
    last_line: str | None = None
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                parsed = json.loads(raw)
                if not isinstance(parsed, dict) or "epoch" not in parsed:
                    raise ValueError(f"invalid epoch JSONL record in {path}")
                current = int(parsed["epoch"])
                if current != last_epoch + 1:
                    raise ValueError(
                        f"non-contiguous epoch JSONL history in {path}: "
                        f"{current} after {last_epoch}"
                    )
                last_epoch = current
                last_line = _json_text(parsed)

    if epoch == last_epoch:
        if last_line == line:
            return False
        raise ValueError(
            f"conflicting epoch {epoch} event already exists in {path}"
        )
    if epoch != last_epoch + 1:
        raise ValueError(
            f"cannot append epoch {epoch} to {path}; expected {last_epoch + 1}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return True


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    manifest: Path
    checkpoint_index: Path
    epoch_metrics: Path
    selection_records: Path

    @classmethod
    def from_root(cls, root: Path) -> "RunPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            manifest=root / "run_manifest.json",
            checkpoint_index=root / "checkpoint_index.json",
            epoch_metrics=root / "epoch_metrics.jsonl",
            selection_records=root / "model_selection_records.jsonl",
        )

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoints / "latest.pt"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints / "best_positive_nll.pt"

    @property
    def final_checkpoint(self) -> Path:
        return self.checkpoints / "final.pt"

    def milestone_checkpoint(self, epoch: int) -> Path:
        return self.checkpoints / f"epoch-{int(epoch):03d}.pt"


def population_payload(
    train_ds: Any,
    selection_ds: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = {
        "frozen_contracts": int(sum(train_ds.frozen_role_counts.values())),
        "active_contracts": int(len(train_ds)),
        "frozen_groups": int(train_ds.frozen_group_count),
        "active_groups": int(train_ds.group_count),
        "frozen_role_counts": dict(train_ds.frozen_role_counts),
        "active_role_counts": dict(train_ds.role_counts),
        "skipped_no_signal": dict(train_ds.skipped_no_signal_counts),
    }
    selection = {
        "frozen_contracts": int(sum(selection_ds.frozen_role_counts.values())),
        "active_contracts": int(len(selection_ds)),
        "frozen_groups": int(selection_ds.frozen_group_count),
        "active_groups": int(selection_ds.group_count),
        "frozen_role_counts": dict(selection_ds.frozen_role_counts),
        "active_role_counts": dict(selection_ds.role_counts),
        "skipped_no_signal": dict(selection_ds.skipped_no_signal_counts),
    }
    return train, selection


def relative_checkpoint_identity(
    identity: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    result = dict(identity)
    result["path"] = relative_artifact(Path(result["path"]), root)
    return result


def _verify_companion(
    path: Path,
    *,
    kind: str,
    epoch: int,
    run_binding: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    assert_checkpoint_binding(checkpoint, run_binding)
    if str(checkpoint.get("kind")) != kind:
        raise ValueError(
            f"Phase-8 checkpoint kind mismatch at {path}: "
            f"{checkpoint.get('kind')!r} != {kind!r}"
        )
    if int(checkpoint.get("epoch", -1)) != int(epoch):
        raise ValueError(
            f"Phase-8 checkpoint epoch mismatch at {path}: "
            f"{checkpoint.get('epoch')!r} != {epoch!r}"
        )
    del checkpoint
    return relative_checkpoint_identity(
        checkpoint_artifact_identity(path, kind=kind, epoch=epoch), root
    )


def reconcile_resume_index(
    *,
    index: dict[str, Any],
    paths: RunPaths,
    checkpoint: Mapping[str, Any],
    run_binding: Mapping[str, Any],
    total_epochs: int,
    milestone_interval_epochs: int,
) -> dict[str, Any]:
    """Repair metadata after a crash window following atomic latest.pt."""
    epoch = int(checkpoint["epoch"])
    index["latest"] = relative_checkpoint_identity(
        checkpoint_artifact_identity(
            paths.latest_checkpoint, kind="latest", epoch=epoch
        ),
        paths.root,
    )

    best_epoch = checkpoint.get("best_positive_nll_epoch")
    if best_epoch is not None:
        best_epoch = int(best_epoch)
        best_entry = index.get("best_positive_nll")
        if (
            not isinstance(best_entry, dict)
            or int(best_entry.get("epoch", -1)) != best_epoch
        ):
            if not paths.best_checkpoint.is_file():
                raise FileNotFoundError(
                    "latest checkpoint references best_positive_nll epoch "
                    f"{best_epoch}, but companion checkpoint is missing"
                )
            index["best_positive_nll"] = _verify_companion(
                paths.best_checkpoint,
                kind="best_positive_nll",
                epoch=best_epoch,
                run_binding=run_binding,
                root=paths.root,
            )

    if epoch % milestone_interval_epochs == 0 and epoch < total_epochs:
        existing = {
            int(item["epoch"]): item
            for item in index.get("milestones", [])
            if isinstance(item, dict) and "epoch" in item
        }
        if epoch not in existing:
            milestone_path = paths.milestone_checkpoint(epoch)
            if not milestone_path.is_file():
                raise FileNotFoundError(
                    f"latest epoch {epoch} requires milestone checkpoint "
                    f"{milestone_path}"
                )
            existing[epoch] = _verify_companion(
                milestone_path,
                kind="milestone",
                epoch=epoch,
                run_binding=run_binding,
                root=paths.root,
            )
        index["milestones"] = [existing[k] for k in sorted(existing)]

    if epoch == total_epochs:
        final_entry = index.get("final")
        if (
            not isinstance(final_entry, dict)
            or int(final_entry.get("epoch", -1)) != epoch
        ):
            if not paths.final_checkpoint.is_file():
                raise FileNotFoundError(
                    "latest checkpoint reached the fixed horizon but final.pt is missing"
                )
            index["final"] = _verify_companion(
                paths.final_checkpoint,
                kind="final",
                epoch=epoch,
                run_binding=run_binding,
                root=paths.root,
            )
    return index


def settings_payload(settings: Any) -> dict[str, Any]:
    if is_dataclass(settings):
        return asdict(settings)
    if isinstance(settings, Mapping):
        return dict(settings)
    raise TypeError("settings must be a dataclass or mapping")


def relative_artifact(path: Path, root: Path) -> str:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def runtime_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


def initial_checkpoint_index(run_binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_INDEX_SCHEMA,
        "run_binding_digest_sha256": str(
            run_binding["binding_digest_sha256"]
        ),
        "latest": None,
        "best_positive_nll": None,
        "final": None,
        "milestones": [],
    }


def validate_checkpoint_index(
    payload: Mapping[str, Any],
    run_binding: Mapping[str, Any],
) -> None:
    if payload.get("schema") != CHECKPOINT_INDEX_SCHEMA:
        raise ValueError("unsupported Phase-8 checkpoint-index schema")
    expected = str(run_binding["binding_digest_sha256"])
    actual = str(payload.get("run_binding_digest_sha256") or "")
    if actual != expected:
        raise ValueError(
            "Phase-8 checkpoint-index binding mismatch: "
            f"{actual!r} != {expected!r}"
        )


def build_run_manifest(
    *,
    state: str,
    run_binding: Mapping[str, Any],
    settings: Any,
    scheduler_metadata: Mapping[str, Any],
    output_root: Path,
    train_population: Mapping[str, Any],
    selection_population: Mapping[str, Any],
    started_from: str,
    completed_epoch: int,
    global_optimizer_step: int,
    best_positive_nll: float | None,
    best_positive_nll_epoch: int | None,
    checkpoint_index_path: Path,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the mutable status manifest around an immutable run binding."""
    if state not in {"RUNNING", "COMPLETE", "FAILED", "INTERRUPTED"}:
        raise ValueError(f"invalid Phase-8 run state: {state}")
    root = Path(output_root).resolve()
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "state": state,
        "run_binding_digest_sha256": str(
            run_binding["binding_digest_sha256"]
        ),
        "run_binding": dict(run_binding),
        "settings": settings_payload(settings),
        "scheduler": dict(scheduler_metadata),
        "output_root": str(root),
        "started_from": str(started_from),
        "progress": {
            "completed_epoch": int(completed_epoch),
            "global_optimizer_step": int(global_optimizer_step),
            "best_positive_nll": None
            if best_positive_nll is None
            else float(best_positive_nll),
            "best_positive_nll_epoch": None
            if best_positive_nll_epoch is None
            else int(best_positive_nll_epoch),
        },
        "populations": {
            "training": dict(train_population),
            "model_selection": dict(selection_population),
        },
        "model_selection_policy": {
            "support": "positive_only_limited",
            "checkpoint_diagnostic": "min_positive_nll",
            "early_stopping": False,
            "general_quality_claim": False,
            "threshold_tuning": False,
            "calibration_fit": False,
        },
        "completion_policy": {
            "primary_g8_checkpoint": "final",
            "fixed_horizon_epochs": int(settings_payload(settings)["epochs"]),
            "acceptance_access": False,
        },
        "checkpoint_index": relative_artifact(checkpoint_index_path, root),
        "logs": {
            "epoch_metrics": "epoch_metrics.jsonl",
            "model_selection_records": "model_selection_records.jsonl",
        },
        "runtime_environment": runtime_environment(),
        "error": None if error is None else dict(error),
    }


def validate_run_manifest(
    payload: Mapping[str, Any],
    run_binding: Mapping[str, Any],
) -> None:
    if payload.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported Phase-8 run-manifest schema")
    expected = str(run_binding["binding_digest_sha256"])
    actual = str(payload.get("run_binding_digest_sha256") or "")
    if actual != expected:
        raise ValueError(
            "Phase-8 run-manifest binding mismatch: "
            f"{actual!r} != {expected!r}"
        )
    if dict(payload.get("run_binding") or {}) != dict(run_binding):
        raise ValueError(
            "Phase-8 run-manifest payload differs despite digest match"
        )


__all__ = [
    "CHECKPOINT_INDEX_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "RunPaths",
    "append_epoch_jsonl",
    "atomic_write_json",
    "build_run_manifest",
    "initial_checkpoint_index",
    "population_payload",
    "read_json",
    "reconcile_resume_index",
    "relative_artifact",
    "relative_checkpoint_identity",
    "runtime_environment",
    "settings_payload",
    "validate_checkpoint_index",
    "validate_run_manifest",
]
