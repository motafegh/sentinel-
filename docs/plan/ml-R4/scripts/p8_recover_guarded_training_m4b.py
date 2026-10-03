#!/usr/bin/env python3
"""Protected-local M4-B checkpoint/resume/recovery proof.

Runs the accepted guarded seam for a governed two-epoch recovery horizon:
epoch 1 is completed and deliberately paused after durable checkpoint
promotion, then the same run is resumed from latest.pt and completes epoch 2.

This is bounded recovery evidence only. It does not authorize M4-C or full
training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from ml.src.datasets.vnext_logical_v3_guarded_dataset import (  # noqa: E402
    LogicalV3GuardedTrainingDataset,
)
from ml.src.training.group_sampler import DeterministicGroupSampler  # noqa: E402
from ml.src.training.vnext_checkpoint import (  # noqa: E402
    assert_checkpoint_binding,
    load_checkpoint,
    sha256_file,
)
from ml.src.training.vnext_guarded_runner import (  # noqa: E402
    M4B_RECOVERY_EPOCHS,
    run_guarded_phase8_bounded,
)
from ml.src.training.vnext_phase8_config import Phase8Settings  # noqa: E402
from ml.src.training.vnext_run_io import read_json  # noqa: E402

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_OVERLAY = DATA_ROOT / "exports/sentinel-r4-vnext-v3"
DEFAULT_REPRESENTATIONS = (
    DATA_ROOT
    / "r4-guarded-d5-full-2026-09-29-a"
    / "representations-r4-v10-v26-guarded-v1-candidate"
)
DEFAULT_LOGICAL_ACCEPTANCE = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/"
    / "logical_v3_acceptance.json"
)
DEFAULT_PHYSICAL_ACCEPTANCE = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-09-29_guarded_token_physical_acceptance/"
    / "acceptance.json"
)
DEFAULT_OBJECTIVE_DECISION = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-10-01_phase8_objective_evaluation/"
    / "decision.json"
)
REPORT_SCHEMA = "sentinel-r4-phase8-m4b-guarded-recovery-v1"


def _write_fresh(path: Path, payload: dict) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"M4-B report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sequence_digest(values: list[int]) -> str:
    raw = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=DEFAULT_REPRESENTATIONS,
    )
    parser.add_argument(
        "--logical-acceptance",
        type=Path,
        default=DEFAULT_LOGICAL_ACCEPTANCE,
    )
    parser.add_argument(
        "--physical-acceptance",
        type=Path,
        default=DEFAULT_PHYSICAL_ACCEPTANCE,
    )
    parser.add_argument(
        "--objective-decision",
        type=Path,
        default=DEFAULT_OBJECTIVE_DECISION,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Fresh protected-local run directory for the two-epoch proof.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Fresh compact M4-B review report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"M4-B run root is not empty: {run_root}")
    if report_path.exists():
        raise FileExistsError(f"M4-B report already exists: {report_path}")

    settings = Phase8Settings(
        epochs=M4B_RECOVERY_EPOCHS,
        batch_size=8,
        gradient_accumulation_steps=8,
    )

    started = time.perf_counter()
    first = run_guarded_phase8_bounded(
        overlay_dir=args.overlay,
        representations_root=args.representations_root,
        logical_acceptance_path=args.logical_acceptance,
        physical_acceptance_path=args.physical_acceptance,
        objective_decision_path=args.objective_decision,
        settings=settings,
        output_dir=run_root,
        resume=None,
        num_workers=0,
        milestone_interval_epochs=1,
        stop_after_epoch=1,
    )
    if first.get("status") != "GUARDED_BOUNDED_CONTROLLED_PAUSE":
        raise RuntimeError(f"unexpected M4-B pause result: {first}")
    if int(first.get("epochs_completed", -1)) != 1:
        raise RuntimeError("M4-B did not pause after epoch 1")
    if int(first.get("next_epoch", -1)) != 2:
        raise RuntimeError("M4-B pause does not point to epoch 2")
    if int(first.get("global_optimizer_steps", -1)) != 15:
        raise RuntimeError("M4-B epoch-1 optimizer-step count is not 15")

    latest = run_root / "checkpoints/latest.pt"
    if not latest.is_file():
        raise FileNotFoundError(latest)
    checkpoint_sha_before_resume = sha256_file(latest)
    checkpoint = load_checkpoint(latest, map_location="cpu")
    original_binding = dict(checkpoint["run_binding"])
    if checkpoint.get("run_binding_digest_sha256") != first.get(
        "binding_digest_sha256"
    ):
        raise RuntimeError("M4-B checkpoint/run result binding mismatch")

    changed = copy.deepcopy(original_binding)
    changed["data"]["guarded_tokens"]["binding_digest_sha256"] = "0" * 64
    # Keep the top-level digest unchanged deliberately: this proves that a
    # changed lineage payload is rejected even if someone tries to preserve the
    # claimed digest string.
    try:
        assert_checkpoint_binding(checkpoint, changed)
    except ValueError as exc:
        if "payload differs despite digest match" not in str(exc):
            raise
        changed_binding_rejected = True
        changed_binding_error = str(exc)
    else:
        raise RuntimeError("M4-B changed guarded lineage was not rejected")

    train_ds = LogicalV3GuardedTrainingDataset(
        repo_root=REPO_ROOT,
        logical_acceptance_path=args.logical_acceptance,
        physical_acceptance_path=args.physical_acceptance,
        overlay_dir=args.overlay,
        representations_root=args.representations_root,
        roles=("TRAIN_STRONG", "TRAIN_WEAK"),
    )
    sampler_a = DeterministicGroupSampler(
        train_ds.group_to_indices,
        seed=settings.seed,
    )
    sampler_b = DeterministicGroupSampler(
        train_ds.group_to_indices,
        seed=settings.seed,
    )
    sampler_a.set_epoch(2)
    sampler_b.set_epoch(2)
    sequence_a = list(sampler_a)
    sequence_b = list(sampler_b)
    if sequence_a != sequence_b:
        raise RuntimeError("M4-B deterministic epoch-2 sampler reconstruction failed")
    sampler_epoch2_digest = _sequence_digest(sequence_a)

    resumed = run_guarded_phase8_bounded(
        overlay_dir=args.overlay,
        representations_root=args.representations_root,
        logical_acceptance_path=args.logical_acceptance,
        physical_acceptance_path=args.physical_acceptance,
        objective_decision_path=args.objective_decision,
        settings=settings,
        output_dir=run_root,
        resume=latest,
        num_workers=0,
        milestone_interval_epochs=1,
        stop_after_epoch=None,
    )
    if resumed.get("status") != "GUARDED_BOUNDED_TRAINING_COMPLETE":
        raise RuntimeError(f"unexpected M4-B completion result: {resumed}")
    if int(resumed.get("epochs_completed", -1)) != 2:
        raise RuntimeError("M4-B resume did not complete epoch 2")
    if int(resumed.get("global_optimizer_steps", -1)) != 30:
        raise RuntimeError("M4-B resumed optimizer-step count is not 30")
    if resumed.get("binding_digest_sha256") != first.get(
        "binding_digest_sha256"
    ):
        raise RuntimeError("M4-B binding changed across pause/resume")

    final_path = run_root / "checkpoints/final.pt"
    if not final_path.is_file():
        raise FileNotFoundError(final_path)
    final_checkpoint = load_checkpoint(final_path, map_location="cpu")
    assert_checkpoint_binding(final_checkpoint, original_binding)
    if int(final_checkpoint["epoch"]) != 2:
        raise RuntimeError("M4-B final checkpoint epoch mismatch")
    if int(final_checkpoint["global_optimizer_step"]) != 30:
        raise RuntimeError("M4-B final checkpoint optimizer-step mismatch")

    manifest = read_json(run_root / "run_manifest.json")
    index = read_json(run_root / "checkpoint_index.json")
    if manifest.get("state") != "COMPLETE":
        raise RuntimeError("M4-B run manifest is not COMPLETE")
    progress = manifest.get("progress") or {}
    if int(progress.get("completed_epoch", -1)) != 2:
        raise RuntimeError("M4-B run manifest epoch mismatch")
    if int(progress.get("global_optimizer_step", -1)) != 30:
        raise RuntimeError("M4-B run manifest optimizer-step mismatch")
    if manifest.get("execution_scope") != "bounded_pilot_only":
        raise RuntimeError("M4-B manifest lost bounded scope")
    if (manifest.get("completion_policy") or {}).get(
        "full_training_authorized"
    ) is not False:
        raise RuntimeError("M4-B manifest unexpectedly authorizes full training")
    if not isinstance(index.get("latest"), dict):
        raise RuntimeError("M4-B checkpoint index lacks latest")
    if not isinstance(index.get("final"), dict):
        raise RuntimeError("M4-B checkpoint index lacks final")

    epoch_lines = (
        run_root / "epoch_metrics.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    selection_lines = (
        run_root / "model_selection_records.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    if len(epoch_lines) != 2 or len(selection_lines) != 2:
        raise RuntimeError("M4-B durable logs do not contain exactly two epochs")

    elapsed = time.perf_counter() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS_M4B_GUARDED_RECOVERY_REVIEW_REQUIRED",
        "source_commit": resumed["source_commit"],
        "run_root": str(run_root.relative_to(REPO_ROOT)),
        "run_binding_digest_sha256": resumed["binding_digest_sha256"],
        "scope": "bounded_pilot_only",
        "recovery_horizon_epochs": M4B_RECOVERY_EPOCHS,
        "pause": {
            "epoch_completed": 1,
            "next_epoch": 2,
            "global_optimizer_steps": 15,
            "latest_checkpoint_sha256": checkpoint_sha_before_resume,
        },
        "resume": {
            "completed_epoch": 2,
            "global_optimizer_steps": 30,
            "binding_unchanged": True,
            "final_checkpoint_sha256": sha256_file(final_path),
            "latest_checkpoint_sha256_after_resume": sha256_file(latest),
            "epoch_metric_records": len(epoch_lines),
            "selection_records": len(selection_lines),
        },
        "binding_drift_probe": {
            "changed_guarded_lineage_rejected": changed_binding_rejected,
            "error": changed_binding_error,
        },
        "sampler_recovery": {
            "epoch": 2,
            "indices": len(sequence_a),
            "sequence_sha256": sampler_epoch2_digest,
            "repeat_identical": True,
        },
        "checkpoint_index": {
            "latest": index.get("latest"),
            "best_positive_nll": index.get("best_positive_nll"),
            "final": index.get("final"),
            "milestones": index.get("milestones"),
        },
        "runtime": {"elapsed_seconds": elapsed},
        "historical_checkpoint_loaded": False,
        "full_training_authorized": False,
        "g8_passed": False,
        "m4c_execution_authorized": False,
        "review_required": True,
        "decision_boundary": (
            "This report proves bounded guarded checkpoint/resume/recovery "
            "behavior only. It does not authorize M4-C or full training."
        ),
    }
    _write_fresh(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
