#!/usr/bin/env python3
"""Run the deterministic R4 Phase-8 end-to-end GPU micro-smoke.

This is a runtime correctness probe, not a quality evaluation. It verifies the
exact G7 DATA vNext lineage can flow through the frozen Phase-8 architecture,
masked optimizer path, and positive-only MODEL_SELECTION path without loading
Run12 learned weights or writing a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

# Keep model loading deterministic/offline unless the caller explicitly overrides.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ml.src.datasets.vnext_dataset import (
    CANONICAL_G7_BINDING_DIGEST,
    VNextTrainingDataset,
    vnext_collate_fn,
)
from ml.src.training.group_sampler import DeterministicGroupSampler
from ml.src.training.vnext_binding import build_run_binding
from ml.src.training.vnext_epoch import (
    evaluate_positive_selection,
    train_masked_epoch,
)
from ml.src.training.vnext_model_factory import build_phase8_model
from ml.src.training.vnext_param_groups import build_parameter_groups
from ml.src.training.vnext_phase8_config import Phase8Settings

EXPECTED_TRAIN_FROZEN_ROLES = {
    "TRAIN_STRONG": 275,
    "TRAIN_WEAK": 773,
}
EXPECTED_TRAIN_ACTIVE_ROLES = {
    "TRAIN_STRONG": 275,
    "TRAIN_WEAK": 577,
}
EXPECTED_TRAIN_FROZEN_GROUPS = 703
EXPECTED_TRAIN_ACTIVE_CONTRACTS = 852
EXPECTED_SKIPPED_NO_SIGNAL = {"TRAIN_WEAK": 196}
EXPECTED_SELECTION_ROLES = {"MODEL_SELECTION": 56}
EXPECTED_SELECTION_GROUPS = 51


def _default_representations_root() -> Path:
    env = os.getenv("SENTINEL_REPRESENTATIONS_ROOT")
    if env:
        return Path(env).expanduser()

    local = REPO_ROOT / "data_module/data/representations"
    if local.is_dir():
        return local

    # Normal project checkout used alongside the detached Phase-8 worktree.
    return Path.home() / "projects/sentinel/data_module/data/representations"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _finite_number(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} is not numeric: {value!r}")
    if not np.isfinite(float(value)):
        raise RuntimeError(f"{name} is not finite: {value!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    overlay = args.overlay.resolve()
    representations = args.representations_root.expanduser().resolve()

    if not overlay.is_dir():
        raise FileNotFoundError(f"DATA vNext overlay not found: {overlay}")
    if not representations.is_dir():
        raise FileNotFoundError(
            "representation root not found: "
            f"{representations}. Pass --representations-root or set "
            "SENTINEL_REPRESENTATIONS_ROOT."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-8 micro-smoke requires CUDA")

    settings = Phase8Settings(
        epochs=1,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")

    train_ds = VNextTrainingDataset(
        overlay_dir=overlay,
        representations_root=representations,
        roles=("TRAIN_STRONG", "TRAIN_WEAK"),
    )
    selection_ds = VNextTrainingDataset(
        overlay_dir=overlay,
        representations_root=representations,
        roles=("MODEL_SELECTION",),
    )

    # Fail closed if the canonical Phase-6/7 population changes unexpectedly.
    if train_ds.frozen_role_counts != EXPECTED_TRAIN_FROZEN_ROLES:
        raise RuntimeError(
            f"unexpected frozen training roles: {train_ds.frozen_role_counts}"
        )
    if train_ds.frozen_group_count != EXPECTED_TRAIN_FROZEN_GROUPS:
        raise RuntimeError(
            f"unexpected frozen training groups: {train_ds.frozen_group_count}"
        )
    if train_ds.role_counts != EXPECTED_TRAIN_ACTIVE_ROLES:
        raise RuntimeError(f"unexpected active training roles: {train_ds.role_counts}")
    if len(train_ds) != EXPECTED_TRAIN_ACTIVE_CONTRACTS:
        raise RuntimeError(f"unexpected active training contracts: {len(train_ds)}")
    if train_ds.group_count != EXPECTED_TRAIN_FROZEN_GROUPS:
        raise RuntimeError(f"unexpected active training groups: {train_ds.group_count}")
    if train_ds.skipped_no_signal_counts != EXPECTED_SKIPPED_NO_SIGNAL:
        raise RuntimeError(
            f"unexpected no-signal siblings: {train_ds.skipped_no_signal_counts}"
        )

    if selection_ds.frozen_role_counts != EXPECTED_SELECTION_ROLES:
        raise RuntimeError(
            f"unexpected frozen MODEL_SELECTION roles: {selection_ds.frozen_role_counts}"
        )
    if selection_ds.role_counts != EXPECTED_SELECTION_ROLES:
        raise RuntimeError(
            f"unexpected active MODEL_SELECTION roles: {selection_ds.role_counts}"
        )
    if len(selection_ds) != 56 or selection_ds.group_count != EXPECTED_SELECTION_GROUPS:
        raise RuntimeError(
            "unexpected MODEL_SELECTION population: "
            f"contracts={len(selection_ds)} groups={selection_ds.group_count}"
        )

    source_commit = _git_head()
    amp_enabled = not args.no_amp

    binding = build_run_binding(
        source_commit=source_commit,
        manifest_path=overlay / "manifest.json",
        expected_representation_digest=CANONICAL_G7_BINDING_DIGEST,
        seed=settings.seed,
        weak_positive_weight=settings.weak_positive_weight,
        optimizer_config={
            "objective": "masked_positive_bce",
            "scheduler": "constant_micro_smoke_only",
            "label_smoothing": 0.0,
            "legacy_label_sampler": False,
            "threshold_tuning": False,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "mixed_precision": "bf16_autocast" if amp_enabled else "disabled",
        },
        train_contracts=len(train_ds),
        train_groups=train_ds.group_count,
        selection_contracts=len(selection_ds),
        selection_groups=selection_ds.group_count,
    )

    sampler = DeterministicGroupSampler(
        train_ds.group_to_indices,
        seed=settings.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=vnext_collate_fn,
    )
    selection_loader = DataLoader(
        selection_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=vnext_collate_fn,
    )

    model = build_phase8_model(device)
    param_groups, _ = build_parameter_groups(model, settings)
    optimizer = AdamW(param_groups, weight_decay=settings.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    train_metrics = train_masked_epoch(
        model=model,
        loader=train_loader,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        settings=settings,
        epoch=1,
        use_amp=amp_enabled,
        max_batches=args.train_batches,
    )
    selection_metrics, records = evaluate_positive_selection(
        model=model,
        loader=selection_loader,
        device=device,
        settings=settings,
        epoch=1,
        use_amp=amp_enabled,
        max_batches=args.selection_batches,
    )

    for key in ("loss", "main_loss", "aux_loss", "phase2_loss", "optimizer_steps"):
        _finite_number(train_metrics[key], f"train.{key}")
    if float(train_metrics["optimizer_steps"]) < 1.0:
        raise RuntimeError("micro-smoke produced no optimizer steps")
    for key in ("positive_nll", "mean_positive_probability"):
        _finite_number(selection_metrics[key], f"model_selection.{key}")

    result: dict[str, Any] = {
        "status": "PHASE8_END_TO_END_MICRO_SMOKE_PASS",
        "source_commit": source_commit,
        "binding_digest_sha256": binding["binding_digest_sha256"],
        "gpu": torch.cuda.get_device_name(0),
        "train_population": {
            "frozen_contracts": sum(train_ds.frozen_role_counts.values()),
            "active_contracts": len(train_ds),
            "frozen_groups": train_ds.frozen_group_count,
            "active_groups": train_ds.group_count,
            "active_role_counts": train_ds.role_counts,
            "skipped_no_signal": train_ds.skipped_no_signal_counts,
        },
        "selection_population": {
            "contracts": len(selection_ds),
            "groups": selection_ds.group_count,
        },
        "runtime_scope": {
            "train_batches": args.train_batches,
            "selection_batches": args.selection_batches,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "run12_weights_loaded": False,
            "checkpoint_written": False,
            "mixed_precision": "bf16_autocast" if amp_enabled else "disabled",
        },
        "train": train_metrics,
        "model_selection": selection_metrics,
        "selection_records": records,
        "cuda": {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
            "peak_allocated_mb": round(
                torch.cuda.max_memory_allocated() / 1024**2, 2
            ),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=REPO_ROOT / "data_module/data/exports/sentinel-r4-vnext-v1",
    )
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=_default_representations_root(),
    )
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--selection-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.train_batches < 1 or args.selection_batches < 1:
        parser.error("batch counts must be >= 1")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("batch size and gradient accumulation must be >= 1")

    print("=== R4 PHASE 8 MICRO-SMOKE ===")
    print(f"repo:                 {REPO_ROOT}")
    print(f"source commit:        {_git_head()}")
    print(f"overlay:              {args.overlay}")
    print(f"representations root: {args.representations_root}")
    print(f"GPU:                  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'UNAVAILABLE'}")
    print()

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")

    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"report written: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
