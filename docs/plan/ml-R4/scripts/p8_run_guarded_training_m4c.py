#!/usr/bin/env python3
"""Protected-local M4-C fixed eight-epoch guarded dynamics pilot.

Runs only the governed eight-epoch R4-D-014 bounded pilot, validates exact
step/checkpoint/log accounting, and emits a compact review report containing
positive-fit dynamics only. It does not authorize M5, G8, production inference,
threshold/calibration fitting, or full-horizon training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from ml.src.training.vnext_checkpoint import (  # noqa: E402
    assert_checkpoint_binding,
    load_checkpoint,
    sha256_file,
)
from ml.src.training.vnext_guarded_run_control import (  # noqa: E402
    M4C_PILOT_EPOCHS,
)
from ml.src.training.vnext_guarded_runner import (  # noqa: E402
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
REPORT_SCHEMA = "sentinel-r4-phase8-m4c-guarded-dynamics-v1"
EXPECTED_STEPS_PER_EPOCH = 15
EXPECTED_GLOBAL_STEPS = M4C_PILOT_EPOCHS * EXPECTED_STEPS_PER_EPOCH


def _write_fresh(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"M4-C report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _finite(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"M4-C {name} is not finite: {numeric}")
    return numeric


def _series_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty M4-C metric series")
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "delta_last_minus_first": float(values[-1] - values[0]),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "mean": float(statistics.fmean(values)),
    }


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
        help="Fresh protected-local directory for the fixed eight-epoch pilot.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Fresh compact M4-C review report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"M4-C run root is not empty: {run_root}")
    if report_path.exists():
        raise FileExistsError(f"M4-C report already exists: {report_path}")

    settings = Phase8Settings(
        epochs=M4C_PILOT_EPOCHS,
        batch_size=8,
        gradient_accumulation_steps=8,
    )

    started = time.perf_counter()
    result = run_guarded_phase8_bounded(
        overlay_dir=args.overlay,
        representations_root=args.representations_root,
        logical_acceptance_path=args.logical_acceptance,
        physical_acceptance_path=args.physical_acceptance,
        objective_decision_path=args.objective_decision,
        settings=settings,
        output_dir=run_root,
        resume=None,
        num_workers=0,
        milestone_interval_epochs=4,
        stop_after_epoch=None,
    )
    if result.get("status") != "GUARDED_BOUNDED_TRAINING_COMPLETE":
        raise RuntimeError(f"unexpected M4-C runner result: {result}")
    if int(result.get("epochs_completed", -1)) != M4C_PILOT_EPOCHS:
        raise RuntimeError("M4-C did not complete the fixed eight-epoch horizon")
    if int(result.get("optimizer_steps_per_epoch", -1)) != EXPECTED_STEPS_PER_EPOCH:
        raise RuntimeError("M4-C optimizer steps/epoch changed")
    if int(result.get("global_optimizer_steps", -1)) != EXPECTED_GLOBAL_STEPS:
        raise RuntimeError("M4-C global optimizer-step count changed")
    if int(result.get("planned_optimizer_steps", -1)) != EXPECTED_GLOBAL_STEPS:
        raise RuntimeError("M4-C planned optimizer-step count changed")
    if result.get("full_training_authorized") is not False:
        raise RuntimeError("M4-C runner unexpectedly authorizes full training")
    if result.get("g8_passed") is not False:
        raise RuntimeError("M4-C runner unexpectedly passes G8")

    manifest = read_json(run_root / "run_manifest.json")
    index = read_json(run_root / "checkpoint_index.json")
    epoch_rows = _load_jsonl(run_root / "epoch_metrics.jsonl")
    selection_rows = _load_jsonl(run_root / "model_selection_records.jsonl")

    if manifest.get("state") != "COMPLETE":
        raise RuntimeError("M4-C run manifest is not COMPLETE")
    if manifest.get("execution_scope") != "bounded_pilot_only":
        raise RuntimeError("M4-C manifest lost bounded-pilot scope")
    policy = manifest.get("completion_policy") or {}
    if int(policy.get("fixed_horizon_epochs", -1)) != M4C_PILOT_EPOCHS:
        raise RuntimeError("M4-C manifest horizon mismatch")
    if policy.get("full_training_authorized") is not False:
        raise RuntimeError("M4-C manifest unexpectedly authorizes full training")
    if policy.get("g8_passed") is not False:
        raise RuntimeError("M4-C manifest unexpectedly passes G8")

    if len(epoch_rows) != M4C_PILOT_EPOCHS:
        raise RuntimeError("M4-C epoch-metric record count is not 8")
    if len(selection_rows) != M4C_PILOT_EPOCHS:
        raise RuntimeError("M4-C selection record count is not 8")

    trajectory: list[dict[str, Any]] = []
    total_loss: list[float] = []
    main_loss: list[float] = []
    aux_loss: list[float] = []
    phase2_loss: list[float] = []
    positive_nll: list[float] = []
    mean_positive_probability: list[float] = []
    positive_recall: list[float] = []

    for expected_epoch, (epoch_row, selection_row) in enumerate(
        zip(epoch_rows, selection_rows), start=1
    ):
        if int(epoch_row.get("epoch", -1)) != expected_epoch:
            raise RuntimeError("M4-C epoch metrics are not contiguous")
        if int(selection_row.get("epoch", -1)) != expected_epoch:
            raise RuntimeError("M4-C selection logs are not contiguous")
        expected_step = expected_epoch * EXPECTED_STEPS_PER_EPOCH
        if int(epoch_row.get("global_optimizer_step", -1)) != expected_step:
            raise RuntimeError("M4-C global optimizer-step trajectory changed")

        train = epoch_row.get("train") or {}
        selection = epoch_row.get("model_selection") or {}
        losses = {
            "loss": _finite(train.get("loss"), f"epoch {expected_epoch} loss"),
            "main_loss": _finite(
                train.get("main_loss"), f"epoch {expected_epoch} main_loss"
            ),
            "aux_loss": _finite(
                train.get("aux_loss"), f"epoch {expected_epoch} aux_loss"
            ),
            "phase2_loss": _finite(
                train.get("phase2_loss"), f"epoch {expected_epoch} phase2_loss"
            ),
        }
        if int(float(train.get("optimizer_steps", -1))) != EXPECTED_STEPS_PER_EPOCH:
            raise RuntimeError("M4-C per-epoch optimizer-step count changed")

        selection_metrics = {
            "positive_nll": _finite(
                selection.get("positive_nll"),
                f"epoch {expected_epoch} positive_nll",
            ),
            "mean_positive_probability": _finite(
                selection.get("mean_positive_probability"),
                f"epoch {expected_epoch} mean_positive_probability",
            ),
            "positive_recall_at_fixed_threshold": _finite(
                selection.get("positive_recall_at_fixed_threshold"),
                f"epoch {expected_epoch} positive_recall_at_fixed_threshold",
            ),
            "metric_cells": int(selection.get("metric_cells", -1)),
        }
        if not 0.0 <= selection_metrics["mean_positive_probability"] <= 1.0:
            raise RuntimeError("M4-C mean positive probability is outside [0,1]")
        if not 0.0 <= selection_metrics["positive_recall_at_fixed_threshold"] <= 1.0:
            raise RuntimeError("M4-C positive recall is outside [0,1]")
        if selection_metrics["metric_cells"] <= 0:
            raise RuntimeError("M4-C selection diagnostics contain no metric cells")

        learning_rates = epoch_row.get("learning_rates") or {}
        if not learning_rates:
            raise RuntimeError("M4-C epoch lacks learning-rate telemetry")
        for name, value in learning_rates.items():
            _finite(value, f"epoch {expected_epoch} learning rate {name}")

        records = selection_row.get("records") or []
        if not isinstance(records, list) or not records:
            raise RuntimeError("M4-C selection epoch lacks positive-fit records")

        total_loss.append(losses["loss"])
        main_loss.append(losses["main_loss"])
        aux_loss.append(losses["aux_loss"])
        phase2_loss.append(losses["phase2_loss"])
        positive_nll.append(selection_metrics["positive_nll"])
        mean_positive_probability.append(
            selection_metrics["mean_positive_probability"]
        )
        positive_recall.append(
            selection_metrics["positive_recall_at_fixed_threshold"]
        )
        trajectory.append(
            {
                "epoch": expected_epoch,
                "global_optimizer_step": expected_step,
                "train": losses,
                "model_selection": selection_metrics,
                "selection_records": len(records),
                "learning_rates": learning_rates,
                "new_best_positive_nll": bool(
                    epoch_row.get("new_best_positive_nll")
                ),
            }
        )

    final_path = run_root / "checkpoints/final.pt"
    latest_path = run_root / "checkpoints/latest.pt"
    best_path = run_root / "checkpoints/best_positive_nll.pt"
    for checkpoint_path in (final_path, latest_path, best_path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

    final_checkpoint = load_checkpoint(final_path, map_location="cpu")
    run_binding = dict(final_checkpoint.get("run_binding") or {})
    assert_checkpoint_binding(final_checkpoint, run_binding)
    if int(final_checkpoint.get("epoch", -1)) != M4C_PILOT_EPOCHS:
        raise RuntimeError("M4-C final checkpoint epoch mismatch")
    if int(final_checkpoint.get("global_optimizer_step", -1)) != EXPECTED_GLOBAL_STEPS:
        raise RuntimeError("M4-C final checkpoint optimizer-step mismatch")
    if final_checkpoint.get("run_binding_digest_sha256") != result.get(
        "binding_digest_sha256"
    ):
        raise RuntimeError("M4-C final checkpoint/run-result binding mismatch")

    best_identity = index.get("best_positive_nll")
    final_identity = index.get("final")
    latest_identity = index.get("latest")
    if not all(isinstance(item, dict) for item in (best_identity, final_identity, latest_identity)):
        raise RuntimeError("M4-C checkpoint index is incomplete")

    elapsed = time.perf_counter() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS_M4C_GUARDED_DYNAMICS_REVIEW_REQUIRED",
        "source_commit": result["source_commit"],
        "run_root": str(run_root.relative_to(REPO_ROOT)),
        "run_binding_digest_sha256": result["binding_digest_sha256"],
        "scope": "bounded_pilot_only",
        "pilot": {
            "epochs": M4C_PILOT_EPOCHS,
            "optimizer_steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
            "planned_optimizer_steps": EXPECTED_GLOBAL_STEPS,
            "executed_optimizer_steps": int(result["global_optimizer_steps"]),
            "batch_size": settings.batch_size,
            "gradient_accumulation_steps": settings.gradient_accumulation_steps,
            "mixed_precision": "bf16_autocast",
            "crosses_aux_loss_warmup_boundary": True,
            "crosses_gnn_prefix_warmup_boundary": False,
        },
        "trajectory": trajectory,
        "summaries": {
            "loss": _series_summary(total_loss),
            "main_loss": _series_summary(main_loss),
            "aux_loss": _series_summary(aux_loss),
            "phase2_loss": _series_summary(phase2_loss),
            "positive_nll": _series_summary(positive_nll),
            "mean_positive_probability": _series_summary(
                mean_positive_probability
            ),
            "positive_recall_at_fixed_threshold": _series_summary(
                positive_recall
            ),
        },
        "best_positive_nll": {
            "value": _finite(
                result.get("best_positive_nll"), "best_positive_nll"
            ),
            "epoch": int(result.get("best_positive_nll_epoch", -1)),
            "scope": "positive_fit_diagnostic_only",
        },
        "checkpoints": {
            "best_positive_nll": best_identity,
            "final": final_identity,
            "latest": latest_identity,
            "milestones": index.get("milestones") or [],
            "best_positive_nll_sha256": sha256_file(best_path),
            "final_sha256": sha256_file(final_path),
            "latest_sha256": sha256_file(latest_path),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "cuda_peak_allocated_mb": _finite(
                result.get("cuda_peak_allocated_mb"),
                "cuda_peak_allocated_mb",
            ),
        },
        "known_architecture_limitations": {
            "fusion_max_nodes": 2048,
            "m4a_observed_graph_nodes_above_cap": 2759,
            "interpretation": (
                "Sparse GNN graph processing remains full-graph, but dense "
                "cross-attention fusion is capped at 2048 nodes. This pilot "
                "does not establish full-node fusion coverage."
            ),
        },
        "historical_checkpoint_loaded": False,
        "threshold_tuning": False,
        "calibration_fit": False,
        "untouched_acceptance": False,
        "production_promotion_authorized": False,
        "inference_promotion_authorized": False,
        "zkml_promotion_authorized": False,
        "full_training_authorized": False,
        "g8_passed": False,
        "m5_execution_authorized": False,
        "review_required": True,
        "decision_boundary": (
            "This report contains bounded positive-fit dynamics only. M4-D "
            "must review it before any launch/no-launch decision. It does not "
            "establish discrimination, false-positive control, calibration, "
            "threshold quality, production readiness, or full-training authority."
        ),
    }
    _write_fresh(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
