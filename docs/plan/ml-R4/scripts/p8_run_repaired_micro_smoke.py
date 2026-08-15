#!/usr/bin/env python3
"""Run the bounded Phase-8 GPU smoke against repaired DATA v2.

This is the first GPU action allowed after physical repaired-DATA acceptance.
It is intentionally *not* the 100-epoch launcher:

* at most a few explicitly requested train/selection batches run;
* Run12 weights are never loaded;
* no checkpoint is written;
* population counts are derived from repaired ``r4-vnext-roles-v2``;
* the exact repaired representation binding and runtime stack are bound;
* success still leaves full training unauthorized pending review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

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

from ml.src.datasets.vnext_dataset import vnext_collate_fn
from ml.src.datasets.vnext_repaired_dataset import RepairedVNextTrainingDataset
from ml.src.training.group_sampler import DeterministicGroupSampler
from ml.src.training.vnext_epoch import evaluate_positive_selection, train_masked_epoch
from ml.src.training.vnext_model_factory import build_phase8_model
from ml.src.training.vnext_param_groups import build_parameter_groups
from ml.src.training.vnext_phase8_config import Phase8Settings
from ml.src.training.vnext_repaired_binding import build_repaired_smoke_binding
from ml.src.training.vnext_run_control import git_source_commit

DEFAULT_OVERLAY = REPO_ROOT / "data_module/data/exports/sentinel-r4-vnext-v2"
DEFAULT_REPRESENTATIONS = REPO_ROOT / "data_module/data/representations-r4-v2"
DEFAULT_ACCEPTANCE = REPO_ROOT / "data_module/data/r4-v2-build/repaired_lineage_audit.json"


def _finite(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise RuntimeError(f"{name} is not finite numeric output: {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_acceptance(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_representation_digest: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing repaired lineage acceptance report: {path}. Run "
            "p8_audit_repaired_lineage.py and save its JSON output first."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("repository_data_acceptance_passed") is not True:
        raise ValueError("repaired lineage acceptance report is not passing")
    if report.get("training_authorized") is not False:
        raise ValueError(
            "repaired lineage report unexpectedly claims training authorization"
        )
    if report.get("publication_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("repaired lineage acceptance report is stale for this publication")
    if (
        report.get("representation_binding_digest_sha256")
        != expected_representation_digest
    ):
        raise ValueError("repaired lineage acceptance report is stale for this representation binding")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    overlay = args.overlay.resolve()
    representations = args.representations_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("bounded repaired-data Phase-8 smoke requires CUDA")

    manifest = json.loads((overlay / "manifest.json").read_text(encoding="utf-8"))
    rep_digest = str(
        (manifest.get("representation_binding_report") or {}).get(
            "binding_digest_sha256"
        )
        or ""
    )
    if not rep_digest:
        raise ValueError("repaired publication lacks representation binding digest")
    acceptance = _load_acceptance(
        args.acceptance_report.resolve(),
        expected_manifest_sha256=_sha256(overlay / "manifest.json"),
        expected_representation_digest=rep_digest,
    )

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
    train_ds = RepairedVNextTrainingDataset(
        overlay_dir=overlay,
        representations_root=representations,
        roles=("TRAIN_STRONG", "TRAIN_WEAK"),
        expected_binding_digest=rep_digest,
    )
    selection_ds = RepairedVNextTrainingDataset(
        overlay_dir=overlay,
        representations_root=representations,
        roles=("MODEL_SELECTION",),
        expected_binding_digest=rep_digest,
    )

    if "TRAIN_STRONG" not in train_ds.frozen_role_counts:
        raise RuntimeError("repaired smoke has no TRAIN_STRONG population")
    if "TRAIN_WEAK" not in train_ds.frozen_role_counts:
        raise RuntimeError("repaired smoke has no TRAIN_WEAK population")
    if selection_ds.frozen_role_counts.get("MODEL_SELECTION", 0) <= 0:
        raise RuntimeError("repaired smoke has no MODEL_SELECTION population")

    source_commit = git_source_commit(REPO_ROOT)
    use_amp = not args.no_amp
    binding = build_repaired_smoke_binding(
        source_commit=source_commit,
        manifest_path=overlay / "manifest.json",
        expected_representation_digest=rep_digest,
        seed=settings.seed,
        weak_positive_weight=settings.weak_positive_weight,
        optimizer_config={
            "objective": "masked_positive_bce",
            "scheduler": "constant_bounded_smoke_only",
            "label_smoothing": 0.0,
            "legacy_label_sampler": False,
            "threshold_tuning": False,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "mixed_precision": "bf16_autocast" if use_amp else "disabled",
            "full_training_authorized": False,
        },
        train_contracts=len(train_ds),
        train_groups=train_ds.group_count,
        selection_contracts=len(selection_ds),
        selection_groups=selection_ds.group_count,
    )

    sampler = DeterministicGroupSampler(train_ds.group_to_indices, seed=settings.seed)
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
        use_amp=use_amp,
        max_batches=args.train_batches,
    )
    selection_metrics, records = evaluate_positive_selection(
        model=model,
        loader=selection_loader,
        device=device,
        settings=settings,
        epoch=1,
        use_amp=use_amp,
        max_batches=args.selection_batches,
    )

    for key in ("loss", "main_loss", "aux_loss", "phase2_loss", "optimizer_steps"):
        _finite(train_metrics[key], f"train.{key}")
    if float(train_metrics["optimizer_steps"]) < 1.0:
        raise RuntimeError("bounded repaired-data smoke produced no optimizer step")
    for key in ("positive_nll", "mean_positive_probability"):
        _finite(selection_metrics[key], f"model_selection.{key}")

    return {
        "status": "PHASE8_REPAIRED_DATA_BOUNDED_GPU_SMOKE_PASS",
        "full_training_authorized": False,
        "source_commit": source_commit,
        "binding_digest_sha256": binding["binding_digest_sha256"],
        "representation_binding_digest_sha256": rep_digest,
        "acceptance_report_schema": acceptance.get("schema"),
        "gpu": torch.cuda.get_device_name(0),
        "train_population": {
            "frozen_role_counts": train_ds.frozen_role_counts,
            "active_role_counts": train_ds.role_counts,
            "frozen_groups": train_ds.frozen_group_count,
            "active_groups": train_ds.group_count,
            "active_contracts": len(train_ds),
            "skipped_no_signal": train_ds.skipped_no_signal_counts,
        },
        "selection_population": {
            "frozen_role_counts": selection_ds.frozen_role_counts,
            "active_role_counts": selection_ds.role_counts,
            "groups": selection_ds.group_count,
            "contracts": len(selection_ds),
            "skipped_no_signal": selection_ds.skipped_no_signal_counts,
        },
        "runtime_scope": {
            "train_batches": args.train_batches,
            "selection_batches": args.selection_batches,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "run12_weights_loaded": False,
            "checkpoint_written": False,
            "mixed_precision": "bf16_autocast" if use_amp else "disabled",
        },
        "train": train_metrics,
        "model_selection": selection_metrics,
        "selection_records": records,
        "cuda": {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        },
        "next_decision": (
            "Review repaired counts, token/window experiment, and this smoke. "
            "Only an explicit governance update may re-authorize the 100-epoch run."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--representations-root", type=Path, default=DEFAULT_REPRESENTATIONS
    )
    parser.add_argument("--acceptance-report", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--selection-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.train_batches,
        args.selection_batches,
        args.batch_size,
        args.gradient_accumulation_steps,
    ) < 1:
        parser.error("batch counts, batch size, and accumulation must be >= 1")

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
