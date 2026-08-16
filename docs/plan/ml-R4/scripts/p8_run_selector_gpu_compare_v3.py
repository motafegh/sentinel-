#!/usr/bin/env python3
"""Run the bounded selector CUDA comparison against logical lineage V3.

This reuses the proven experiment mechanics from ``p8_run_selector_gpu_compare``
but swaps in the V3 dataset adapter and requires a lineage-bound V3 sensitivity
report so stale or mismatched worst-case probes cannot be consumed silently.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

import torch

from ml.src.datasets.vnext_logical_v3_dataset import LogicalV3TrainingDataset
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)

BASE_SCRIPT = Path(__file__).with_name("p8_run_selector_gpu_compare.py")
spec = importlib.util.spec_from_file_location("sentinel_p8_selector_gpu_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load selector GPU base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# The base helper functions resolve this global when they construct datasets.
# Replacing it here changes only the manifest/partition authority; model,
# optimizer, selector, sampler, and CUDA mechanics stay byte-for-byte shared.
base.RepairedVNextTrainingDataset = LogicalV3TrainingDataset

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_OVERLAY = DATA_ROOT / "exports/sentinel-r4-vnext-v3"
DEFAULT_REPRESENTATIONS = DATA_ROOT / "representations-r4-v2"
DEFAULT_PREPROCESSED = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_SENSITIVITY = DATA_ROOT / "r4-v3-logical-build/representation_sensitivity_v1.json"
DEFAULT_OUTPUT = DATA_ROOT / "r4-v3-logical-build/selector_gpu_compare_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--representations-root", type=Path, default=DEFAULT_REPRESENTATIONS)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED)
    parser.add_argument("--sensitivity-report", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--selection-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--worst-case-probes", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if min(
        args.train_batches,
        args.selection_batches,
        args.batch_size,
        args.gradient_accumulation_steps,
    ) < 1:
        parser.error("batch counts, batch size, and accumulation must be >= 1")
    if args.worst_case_probes < 0:
        parser.error("--worst-case-probes must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("logical-v3 selector GPU comparison requires CUDA")

    overlay = args.overlay.resolve()
    representations = args.representations_root.resolve()
    preprocessed = args.preprocessed_root.resolve()
    sensitivity_path = args.sensitivity_report.resolve()
    manifest_path = overlay / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != DATASET_VERSION_V3:
        raise ValueError("selector GPU comparison requires sentinel-r4-vnext-v3")
    if manifest.get("partition_version") != ROLE_PARTITION_VERSION_V3:
        raise ValueError("selector GPU comparison requires r4-vnext-roles-v3")
    if manifest.get("grouping_version") != GROUPING_VERSION_V3:
        raise ValueError("selector GPU comparison requires r4-leakage-groups-v3")
    if manifest.get("confirmed_negative_rows") != 0:
        raise ValueError("unexpected confirmed negatives in logical-v3")
    if (
        manifest.get("status")
        != "LOGICAL_V3_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED"
    ):
        raise ValueError("logical-v3 publication is not physically bound")

    manifest_sha = base._sha256_file(manifest_path)
    rep_digest = str(
        (manifest.get("representation_binding_report") or {}).get(
            "binding_digest_sha256"
        )
        or ""
    )
    if not rep_digest:
        raise ValueError("logical-v3 manifest lacks representation binding digest")

    current_source_commit = base._source_commit()
    expected_worst_case = 0
    sensitivity_sha = ""
    if args.worst_case_probes > 0:
        if not sensitivity_path.is_file():
            raise FileNotFoundError(
                f"worst-case probes requested but sensitivity report is missing: {sensitivity_path}"
            )
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        lineage = sensitivity.get("lineage") or {}
        expected_lineage = {
            "dataset_version": DATASET_VERSION_V3,
            "grouping_version": GROUPING_VERSION_V3,
            "partition_version": ROLE_PARTITION_VERSION_V3,
            "publication_manifest_sha256": manifest_sha,
            "representation_binding_digest_sha256": rep_digest,
            "source_commit": current_source_commit,
        }
        mismatches = {
            key: {"expected": expected, "observed": lineage.get(key)}
            for key, expected in expected_lineage.items()
            if lineage.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "sensitivity report lineage does not match current V3 publication/source: "
                f"{mismatches}"
            )
        wanted = (
            (sensitivity.get("comparison_sets") or {}).get("worst_case_gpu_contract_ids")
            or []
        )
        expected_worst_case = min(args.worst_case_probes, len(wanted))
        if expected_worst_case == 0:
            raise ValueError("sensitivity report contains no worst-case active contracts")
        sensitivity_sha = base._sha256_file(sensitivity_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base.TOKENIZER_MODEL,
        use_fast=True,
        local_files_only=True,
    )
    settings = base.Phase8Settings(
        epochs=1,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    use_amp = not args.no_amp
    device = torch.device("cuda")

    base._reset_rng(settings.seed)
    prototype = base.build_phase8_model(torch.device("cpu"))
    initial_state = base._clone_cpu_state(prototype)
    initial_digest = base._state_digest(initial_state)
    del prototype

    control = base._run_strategy(
        strategy=base.CONTROL_STRATEGY,
        initial_state=initial_state,
        initial_state_digest=initial_digest,
        overlay=overlay,
        representations=representations,
        preprocessed=preprocessed,
        rep_digest=rep_digest,
        tokenizer=tokenizer,
        settings=settings,
        train_batches=args.train_batches,
        selection_batches=args.selection_batches,
        use_amp=use_amp,
        device=device,
    )
    candidate = base._run_strategy(
        strategy=base.GUARDED_STRATEGY,
        initial_state=initial_state,
        initial_state_digest=initial_digest,
        overlay=overlay,
        representations=representations,
        preprocessed=preprocessed,
        rep_digest=rep_digest,
        tokenizer=tokenizer,
        settings=settings,
        train_batches=args.train_batches,
        selection_batches=args.selection_batches,
        use_amp=use_amp,
        device=device,
    )

    probability_delta = base._probability_delta(
        control["selection_records"], candidate["selection_records"]
    )
    worst_case = base._worst_case_forward_probes(
        sensitivity_report=sensitivity_path,
        initial_state=initial_state,
        overlay=overlay,
        representations=representations,
        preprocessed=preprocessed,
        rep_digest=rep_digest,
        tokenizer=tokenizer,
        settings=settings,
        use_amp=use_amp,
        device=device,
        limit=args.worst_case_probes,
    )
    if len(worst_case) != expected_worst_case:
        raise RuntimeError(
            "logical-v3 worst-case GPU probes were incomplete: "
            f"expected={expected_worst_case} observed={len(worst_case)}"
        )

    report = {
        "schema": "sentinel-r4-phase8-selector-gpu-compare-v3",
        "status": "LOGICAL_V3_BOUNDED_RESEARCH_COMPLETE",
        "source_commit": current_source_commit,
        "dataset_version": DATASET_VERSION_V3,
        "grouping_version": GROUPING_VERSION_V3,
        "partition_version": ROLE_PARTITION_VERSION_V3,
        "publication_manifest_sha256": manifest_sha,
        "representation_binding_digest_sha256": rep_digest,
        "sensitivity_report_sha256": sensitivity_sha,
        "gpu": torch.cuda.get_device_name(0),
        "seed": settings.seed,
        "initial_state_digest_sha256": initial_digest,
        "identical_initialization_verified": (
            control["initial_state_digest_sha256"]
            == candidate["initial_state_digest_sha256"]
            == initial_digest
        ),
        "runtime_scope": {
            "train_batches_per_strategy": args.train_batches,
            "selection_batches_per_strategy": args.selection_batches,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "mixed_precision": "bf16_autocast" if use_amp else "disabled",
            "run12_weights_loaded": False,
            "checkpoint_written": False,
            "worst_case_probes_required": expected_worst_case,
            "worst_case_probes_completed": len(worst_case),
        },
        "control": control,
        "candidate": candidate,
        "positive_selection_probability_delta": probability_delta,
        "worst_case_guarded_forward_probes": worst_case,
        "full_training_authorized": False,
        "selector_promotion_authorized": False,
        "decision_boundary": (
            "This bounded V3 comparison tests selector behavior and CUDA safety only. "
            "It cannot establish vulnerability discrimination because confirmed-negative "
            "evaluation evidence is still pending."
        ),
    }
    if not report["identical_initialization_verified"]:
        raise RuntimeError("logical-v3 selector experiment lost identical initialization")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
