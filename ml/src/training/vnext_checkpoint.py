"""Durable checkpoint and resume state for R4 Phase 8."""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

CHECKPOINT_SCHEMA = "sentinel-r4-phase8-checkpoint-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_artifact_identity(
    path: Path,
    *,
    kind: str,
    epoch: int,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "epoch": int(epoch),
        "kind": str(kind),
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream Phase 8 controls."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG streams saved by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"checkpoint RNG state missing keys: {sorted(missing)}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    cuda_states = state["torch_cuda_all"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint CUDA RNG device count mismatch: "
                f"{len(cuda_states)} != {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _settings_payload(settings: Any) -> dict[str, Any]:
    if is_dataclass(settings):
        return asdict(settings)
    if isinstance(settings, Mapping):
        return dict(settings)
    raise TypeError("settings must be a dataclass or mapping")


def build_checkpoint_payload(
    *,
    kind: str,
    epoch: int,
    global_optimizer_step: int,
    run_binding: Mapping[str, Any],
    settings: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_positive_nll: float | None,
    best_positive_nll_epoch: int | None,
    epoch_event: Mapping[str, Any],
    selection_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one self-describing, resumable Phase-8 checkpoint."""
    digest = str(run_binding.get("binding_digest_sha256") or "")
    if not digest:
        raise ValueError("run binding lacks binding_digest_sha256")
    if int(epoch) < 1:
        raise ValueError("checkpoint epoch must be >= 1")
    if int(global_optimizer_step) < 1:
        raise ValueError("global_optimizer_step must be >= 1")
    if best_positive_nll is None and best_positive_nll_epoch is not None:
        raise ValueError("best_positive_nll_epoch requires best_positive_nll")
    if best_positive_nll is not None and best_positive_nll_epoch is None:
        raise ValueError("best_positive_nll requires best_positive_nll_epoch")

    return {
        "schema": CHECKPOINT_SCHEMA,
        "kind": str(kind),
        "epoch": int(epoch),
        "global_optimizer_step": int(global_optimizer_step),
        "run_binding_digest_sha256": digest,
        "run_binding": dict(run_binding),
        "settings": _settings_payload(settings),
        "best_positive_nll": None
        if best_positive_nll is None
        else float(best_positive_nll),
        "best_positive_nll_epoch": None
        if best_positive_nll_epoch is None
        else int(best_positive_nll_epoch),
        "epoch_event": dict(epoch_event),
        "selection_records": list(selection_records),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Atomically write a torch checkpoint and return its artifact identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

    return checkpoint_artifact_identity(
        path,
        kind=str(payload["kind"]),
        epoch=int(payload["epoch"]),
    )


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load and minimally validate a Phase-8 checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Phase-8 checkpoint payload is not a mapping")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "unsupported Phase-8 checkpoint schema: "
            f"{checkpoint.get('schema')!r}"
        )
    required = {
        "epoch",
        "global_optimizer_step",
        "run_binding_digest_sha256",
        "run_binding",
        "settings",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "epoch_event",
        "selection_records",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Phase-8 checkpoint missing keys: {sorted(missing)}")
    return checkpoint


def assert_checkpoint_binding(
    checkpoint: Mapping[str, Any],
    expected_run_binding: Mapping[str, Any],
) -> None:
    """Fail closed when resume state is not from the exact same run contract."""
    expected_digest = str(
        expected_run_binding.get("binding_digest_sha256") or ""
    )
    actual_digest = str(
        checkpoint.get("run_binding_digest_sha256") or ""
    )
    if not expected_digest:
        raise ValueError("expected run binding lacks digest")
    if actual_digest != expected_digest:
        raise ValueError(
            "Phase-8 resume binding mismatch: "
            f"{actual_digest!r} != {expected_digest!r}"
        )
    if dict(checkpoint.get("run_binding") or {}) != dict(expected_run_binding):
        raise ValueError(
            "Phase-8 resume run-binding payload differs despite digest match"
        )


def restore_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_run_binding: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    """Restore a full checkpoint; model-only/partial resume is intentionally absent."""
    assert_checkpoint_binding(checkpoint, expected_run_binding)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return {
        "completed_epoch": int(checkpoint["epoch"]),
        "next_epoch": int(checkpoint["epoch"]) + 1,
        "global_optimizer_step": int(checkpoint["global_optimizer_step"]),
        "best_positive_nll": checkpoint.get("best_positive_nll"),
        "best_positive_nll_epoch": checkpoint.get(
            "best_positive_nll_epoch"
        ),
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "assert_checkpoint_binding",
    "atomic_torch_save",
    "build_checkpoint_payload",
    "capture_rng_state",
    "checkpoint_artifact_identity",
    "load_checkpoint",
    "restore_checkpoint",
    "restore_rng_state",
    "sha256_file",
]
