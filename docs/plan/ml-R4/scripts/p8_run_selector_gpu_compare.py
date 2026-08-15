#!/usr/bin/env python3
"""Run identical-initialization bounded CUDA comparison of R4 window selectors.

This experiment compares the accepted historical linspace token selection with
the research-only guarded target-aware selector while holding the model's exact
initial state, seed, train group sampler, optimizer construction, batch limits,
and positive-only supervision fixed.

It does not write checkpoints, use Run12 weights, promote the selector, change
the accepted repaired-v2 artifacts, or authorize the 100-epoch run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
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
from torch.utils.data import DataLoader, Dataset

from ml.src.data_extraction.bounded_window_selector import (
    CONTROL_STRATEGY,
    GUARDED_STRATEGY,
    tokenize_with_selector,
)
from ml.src.data_extraction.windowed_tokenizer import TOKENIZER_MODEL
from ml.src.datasets.vnext_dataset import vnext_collate_fn
from ml.src.datasets.vnext_repaired_dataset import RepairedVNextTrainingDataset
from ml.src.training.group_sampler import DeterministicGroupSampler
from ml.src.training.vnext_epoch import evaluate_positive_selection, train_masked_epoch
from ml.src.training.vnext_model_factory import build_phase8_model
from ml.src.training.vnext_param_groups import build_parameter_groups
from ml.src.training.vnext_phase8_config import Phase8Settings
from sentinel_data.representation.r4_target_spans import target_contract_char_spans

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_OVERLAY = DATA_ROOT / "exports/sentinel-r4-vnext-v2"
DEFAULT_REPRESENTATIONS = DATA_ROOT / "representations-r4-v2"
DEFAULT_PREPROCESSED = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_SENSITIVITY = DATA_ROOT / "r4-v2-build/representation_sensitivity_v1.json"
DEFAULT_OUTPUT = DATA_ROOT / "r4-v2-build/selector_gpu_compare_v1.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SelectorDataset(Dataset):
    """Wrap repaired-v2 samples and replace only token windows for research."""

    def __init__(
        self,
        base: RepairedVNextTrainingDataset,
        *,
        strategy: str,
        overlay: Path,
        representations_root: Path,
        preprocessed_root: Path,
        tokenizer: Any,
        verify_control_identity: bool,
    ) -> None:
        self.base = base
        self.strategy = strategy
        self.overlay = overlay
        self.representations_root = representations_root
        self.preprocessed_root = preprocessed_root
        self.tokenizer = tokenizer
        self.verify_control_identity = verify_control_identity

        import pyarrow.parquet as pq

        rows = pq.read_table(overlay / "ml_targets.parquet").to_pylist()
        self.source_by_contract = {
            str(row["contract_id"]): str(row["source"]) for row in rows
        }
        self._telemetry: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.base)

    @property
    def group_to_indices(self) -> dict[str, tuple[int, ...]]:
        return self.base.group_to_indices

    @property
    def contract_ids(self) -> tuple[str, ...]:
        return self.base.contract_ids

    @property
    def telemetry(self) -> dict[str, dict[str, Any]]:
        return dict(self._telemetry)

    def __getitem__(self, index: int):
        graph, bound_tokens, supervision, contract_id, role, group_id = self.base[index]
        source = self.source_by_contract[contract_id]
        sol_path = self.preprocessed_root / source / f"{contract_id}.sol"
        sidecar_path = self.representations_root / source / f"{contract_id}.rep.json"
        source_text = sol_path.read_text(encoding="utf-8")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        targets = [
            str(value)
            for value in (sidecar.get("requested_contract_names") or ())
        ]
        if not targets:
            raise ValueError(
                f"{contract_id} representation sidecar has no requested targets"
            )
        char_spans = target_contract_char_spans(source_text, targets)
        dynamic = tokenize_with_selector(
            source_text,
            target_char_spans=char_spans,
            tokenizer=self.tokenizer,
            strategy=self.strategy,
        )
        dynamic_tokens = {
            "input_ids": dynamic["input_ids"],
            "attention_mask": dynamic["attention_mask"],
        }
        if self.strategy == CONTROL_STRATEGY and self.verify_control_identity:
            if not torch.equal(dynamic_tokens["input_ids"], bound_tokens["input_ids"]):
                raise RuntimeError(
                    f"dynamic historical control input_ids diverge from bound tokens: "
                    f"{contract_id}"
                )
            if not torch.equal(
                dynamic_tokens["attention_mask"], bound_tokens["attention_mask"]
            ):
                raise RuntimeError(
                    f"dynamic historical control attention_mask diverges from bound tokens: "
                    f"{contract_id}"
                )
        self._telemetry[contract_id] = {
            "strategy": self.strategy,
            "source": source,
            "role": role,
            "group_id": group_id,
            "target_coverage_ratio": dynamic["target_coverage_ratio"],
            "control_target_coverage_ratio": dynamic[
                "control_target_coverage_ratio"
            ],
            "retained_ratio": dynamic["retained_ratio"],
            "control_retained_ratio": dynamic["control_retained_ratio"],
            "selected_indices": dynamic["selector"]["selected_indices"],
            "control_indices": dynamic["selector"]["control_indices"],
            "used_control_fallback": dynamic["selector"]["used_control_fallback"],
            "total_windows": dynamic["total_windows"],
        }
        return (
            graph,
            dynamic_tokens,
            supervision,
            contract_id,
            role,
            group_id,
        )


def _build_base_datasets(
    *,
    overlay: Path,
    representations: Path,
    rep_digest: str,
) -> tuple[RepairedVNextTrainingDataset, RepairedVNextTrainingDataset]:
    train = RepairedVNextTrainingDataset(
        overlay_dir=overlay,
        representations_root=representations,
        roles=("TRAIN_STRONG", "TRAIN_WEAK"),
        expected_binding_digest=rep_digest,
    )
    selection = RepairedVNextTrainingDataset(
        overlay_dir=overlay,
        representations_root=representations,
        roles=("MODEL_SELECTION",),
        expected_binding_digest=rep_digest,
    )
    return train, selection


def _run_strategy(
    *,
    strategy: str,
    initial_state: dict[str, torch.Tensor],
    initial_state_digest: str,
    overlay: Path,
    representations: Path,
    preprocessed: Path,
    rep_digest: str,
    tokenizer: Any,
    settings: Phase8Settings,
    train_batches: int,
    selection_batches: int,
    use_amp: bool,
    device: torch.device,
) -> dict[str, Any]:
    _reset_rng(settings.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_base, selection_base = _build_base_datasets(
        overlay=overlay,
        representations=representations,
        rep_digest=rep_digest,
    )
    train_ds = SelectorDataset(
        train_base,
        strategy=strategy,
        overlay=overlay,
        representations_root=representations,
        preprocessed_root=preprocessed,
        tokenizer=tokenizer,
        verify_control_identity=True,
    )
    selection_ds = SelectorDataset(
        selection_base,
        strategy=strategy,
        overlay=overlay,
        representations_root=representations,
        preprocessed_root=preprocessed,
        tokenizer=tokenizer,
        verify_control_identity=True,
    )

    sampler = DeterministicGroupSampler(
        train_ds.group_to_indices,
        seed=settings.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=settings.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=vnext_collate_fn,
    )
    selection_loader = DataLoader(
        selection_ds,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=vnext_collate_fn,
    )

    model = build_phase8_model(device)
    model.load_state_dict(initial_state, strict=True)
    loaded_digest = _state_digest(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    if loaded_digest != initial_state_digest:
        raise RuntimeError(
            f"initial model state digest changed for {strategy}: "
            f"{loaded_digest} != {initial_state_digest}"
        )

    parameter_groups, parameter_summary = build_parameter_groups(model, settings)
    optimizer = AdamW(parameter_groups, weight_decay=settings.weight_decay)
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
        max_batches=train_batches,
    )
    selection_metrics, selection_records = evaluate_positive_selection(
        model=model,
        loader=selection_loader,
        device=device,
        settings=settings,
        epoch=1,
        use_amp=use_amp,
        max_batches=selection_batches,
    )

    result = {
        "strategy": strategy,
        "initial_state_digest_sha256": initial_state_digest,
        "train_population": {
            "active_contracts": len(train_base),
            "active_groups": train_base.group_count,
            "active_role_counts": train_base.role_counts,
        },
        "selection_population": {
            "active_contracts": len(selection_base),
            "active_groups": selection_base.group_count,
            "active_role_counts": selection_base.role_counts,
        },
        "parameter_group_summary": parameter_summary,
        "train": train_metrics,
        "model_selection": selection_metrics,
        "selection_records": selection_records,
        "selector_telemetry": {
            "train": train_ds.telemetry,
            "selection": selection_ds.telemetry,
        },
        "cuda": {
            "peak_allocated_mb": round(
                torch.cuda.max_memory_allocated() / 1024**2, 2
            ),
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        },
    }
    del optimizer, scheduler, model, train_loader, selection_loader
    torch.cuda.empty_cache()
    return result


def _probability_delta(
    control_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    def index(records: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
        return {
            (str(row["contract_id"]), int(row["class_index"])): float(
                row["probability"]
            )
            for row in records
        }

    control = index(control_records)
    candidate = index(candidate_records)
    if set(control) != set(candidate):
        raise RuntimeError(
            "selector comparison produced different positive-selection cell sets"
        )
    deltas = [candidate[key] - control[key] for key in sorted(control)]
    abs_deltas = [abs(value) for value in deltas]
    return {
        "cells": len(deltas),
        "mean_signed_probability_delta": (
            float(sum(deltas) / len(deltas)) if deltas else 0.0
        ),
        "mean_absolute_probability_delta": (
            float(sum(abs_deltas) / len(abs_deltas)) if abs_deltas else 0.0
        ),
        "max_absolute_probability_delta": max(abs_deltas) if abs_deltas else 0.0,
    }


@torch.no_grad()
def _worst_case_forward_probes(
    *,
    sensitivity_report: Path,
    initial_state: dict[str, torch.Tensor],
    overlay: Path,
    representations: Path,
    preprocessed: Path,
    rep_digest: str,
    tokenizer: Any,
    settings: Phase8Settings,
    use_amp: bool,
    device: torch.device,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not sensitivity_report.is_file():
        return []
    report = json.loads(sensitivity_report.read_text(encoding="utf-8"))
    wanted = [
        str(value)
        for value in (
            (report.get("comparison_sets") or {}).get(
                "worst_case_gpu_contract_ids"
            )
            or []
        )
    ][:limit]
    if not wanted:
        return []

    train_base, selection_base = _build_base_datasets(
        overlay=overlay,
        representations=representations,
        rep_digest=rep_digest,
    )
    bases = [train_base, selection_base]
    index_by_contract: dict[str, tuple[RepairedVNextTrainingDataset, int]] = {}
    for base in bases:
        for index, contract_id in enumerate(base.contract_ids):
            index_by_contract[contract_id] = (base, index)

    results: list[dict[str, Any]] = []
    for contract_id in wanted:
        resolved = index_by_contract.get(contract_id)
        if resolved is None:
            continue
        base, index = resolved
        wrapper = SelectorDataset(
            base,
            strategy=GUARDED_STRATEGY,
            overlay=overlay,
            representations_root=representations,
            preprocessed_root=preprocessed,
            tokenizer=tokenizer,
            verify_control_identity=False,
        )
        sample = wrapper[index]
        batch = vnext_collate_fn([sample])
        graphs, tokens, _, _, _, _ = batch
        graphs = graphs.to(device)
        tokens = {key: value.to(device) for key, value in tokens.items()}

        _reset_rng(settings.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_phase8_model(device)
        model.load_state_dict(initial_state, strict=True)
        model.eval()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits = model(
                graphs,
                tokens["input_ids"],
                tokens["attention_mask"],
            )
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"non-finite worst-case forward output for {contract_id}"
            )
        results.append(
            {
                "contract_id": contract_id,
                "peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated() / 1024**2, 2
                ),
                "logits_shape": list(logits.shape),
                "selector": wrapper.telemetry[contract_id],
            }
        )
        del model, graphs, tokens, logits
        torch.cuda.empty_cache()
    return results


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
        raise RuntimeError("selector GPU comparison requires CUDA")

    overlay = args.overlay.resolve()
    representations = args.representations_root.resolve()
    preprocessed = args.preprocessed_root.resolve()
    manifest_path = overlay / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rep_digest = str(
        (manifest.get("representation_binding_report") or {}).get(
            "binding_digest_sha256"
        )
        or ""
    )
    if manifest.get("dataset_version") != "sentinel-r4-vnext-v2":
        raise ValueError("selector GPU comparison requires repaired-v2")
    if not rep_digest:
        raise ValueError("repaired-v2 manifest lacks representation binding digest")
    if manifest.get("confirmed_negative_rows") != 0:
        raise ValueError("unexpected confirmed negatives in accepted repaired-v2")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        use_fast=True,
        local_files_only=True,
    )
    settings = Phase8Settings(
        epochs=1,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    use_amp = not args.no_amp
    device = torch.device("cuda")

    _reset_rng(settings.seed)
    prototype = build_phase8_model(torch.device("cpu"))
    initial_state = _clone_cpu_state(prototype)
    initial_digest = _state_digest(initial_state)
    del prototype

    control = _run_strategy(
        strategy=CONTROL_STRATEGY,
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
    candidate = _run_strategy(
        strategy=GUARDED_STRATEGY,
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

    probability_delta = _probability_delta(
        control["selection_records"],
        candidate["selection_records"],
    )
    worst_case = _worst_case_forward_probes(
        sensitivity_report=args.sensitivity_report.resolve(),
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

    report = {
        "schema": "sentinel-r4-phase8-selector-gpu-compare-v1",
        "status": "BOUNDED_RESEARCH_COMPLETE",
        "source_commit": _source_commit(),
        "publication_manifest_sha256": _sha256_file(manifest_path),
        "representation_binding_digest_sha256": rep_digest,
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
        },
        "control": control,
        "candidate": candidate,
        "positive_selection_probability_delta": probability_delta,
        "worst_case_guarded_forward_probes": worst_case,
        "full_training_authorized": False,
        "selector_promotion_authorized": False,
        "decision_boundary": (
            "This bounded identical-initialization comparison is evidence for "
            "selector review only. It cannot establish vulnerability discrimination "
            "because repaired-v2 still has no confirmed-negative evaluation population."
        ),
    }
    if not report["identical_initialization_verified"]:
        raise RuntimeError("identical initialization verification failed")
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
