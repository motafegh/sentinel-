"""
ml/scripts/tune_threshold.py

Per-class decision-threshold optimiser for SENTINEL Track 3 multi-label models.

Why this rewrite exists:
- Old version assumed binary output: probs [N]
- Old version swept one global threshold
- Track 3 outputs 10 logits per contract: probs [N, 10]
- Each class has different imbalance, so each class needs its own threshold

What this script does:
1. Load the validation split used during training
2. Load the trained checkpoint (supports config-aware architecture loading)
3. Run one forward pass to collect sigmoid probabilities for all 10 classes
4. Sweep thresholds per class on cached probabilities
5. Pick the threshold that maximises per-class F1
6. Report overall macro/micro F1, hamming loss, and exact-match accuracy
7. Save thresholds JSON next to the checkpoint

FIXES (2026-05-04):
    Fix #3 — load_model_from_checkpoint() now passes dropout (fusion_dropout),
               gnn_dropout, and lora_target_modules from the saved checkpoint
               config to SentinelModel(). Identical issue to predictor.py Fix #2.
               Missing args caused load_state_dict() to crash when the checkpoint
               was trained with non-default hyperparameters.
    Fix #5 — build_val_loader() DataLoader kwargs are now built conditionally.
               prefetch_factor, pin_memory, and persistent_workers are only added
               when num_workers > 0, eliminating the PyTorch 2.x UserWarning
               caused by passing prefetch_factor=None with num_workers=0.

Usage:
    python -m ml.scripts.tune_threshold
    python -m ml.scripts.tune_threshold \\
        --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch_geometric.loader import DataLoader

# Make project root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Keep output readable.
logger.remove()
logger.add(sys.stderr, level="INFO")

from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from ml.src.inference.predictor import _ARCH_TO_FUSION_DIM
from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.trainer import CLASS_NAMES, NUM_CLASSES, TrainConfig


@dataclass(frozen=True)
class SweepRow:
    """Metrics for one threshold on one class."""
    threshold: float
    f1: float
    precision: float
    recall: float
    predicted_positives: int


@dataclass(frozen=True)
class BestThreshold:
    """Best threshold summary for one class."""
    class_name: str
    threshold: float
    f1: float
    precision: float
    recall: float
    support: int
    predicted_positives: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Tune per-class decision thresholds for a multi-label SENTINEL checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/checkpoints/multilabel_crossattn_v2_best.pt",
        help="Path to the trained .pt checkpoint.",
    )
    parser.add_argument(
        "--label-csv",
        type=str,
        default="ml/data/processed/multilabel_index.csv",
        help="Path to the multi-label CSV index.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path for thresholds JSON. Defaults to <checkpoint_stem>_thresholds.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override validation batch size. Defaults to TrainConfig.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader worker count. Defaults to TrainConfig.num_workers.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.05,
        help="Threshold sweep start (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=0.95,
        help="Threshold sweep end (inclusive).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Threshold sweep step size.",
    )
    return parser.parse_args()


def build_threshold_grid(start: float, end: float, step: float) -> list[float]:
    """Build an inclusive threshold grid."""
    if not (0.0 < start <= 1.0):
        raise ValueError(f"--start must be in (0, 1], got {start}")
    if not (0.0 < end <= 1.0):
        raise ValueError(f"--end must be in (0, 1], got {end}")
    if start > end:
        raise ValueError(f"--start must be <= --end, got {start} > {end}")
    if step <= 0.0:
        raise ValueError(f"--step must be > 0, got {step}")

    count = int(round((end - start) / step)) + 1
    thresholds = np.linspace(start, end, count, dtype=np.float32)
    return [round(float(t), 4) for t in thresholds.tolist()]


def resolve_output_path(checkpoint_path: Path, explicit_output: str | None) -> Path:
    """Resolve JSON output path."""
    if explicit_output:
        return Path(explicit_output)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_thresholds.json")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> tuple[SentinelModel, dict[str, Any]]:
    """
    Load checkpoint with architecture-aware model construction.

    Supports both:
    - new dict checkpoints with {"model": ..., "config": ...}
    - older plain state_dict checkpoints

    Fix #3: passes dropout, gnn_dropout, and lora_target_modules from the
    saved checkpoint config so load_state_dict() never crashes due to LoRA
    key mismatches when non-default hyperparameters were used during training.
    """
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_config: dict[str, Any] = raw.get("config", {}) if isinstance(raw, dict) else {}
    architecture = ckpt_config.get("architecture", "cross_attention_lora")
    num_classes = int(ckpt_config.get("num_classes", NUM_CLASSES))
    fusion_output_dim = ckpt_config.get("fusion_output_dim", _ARCH_TO_FUSION_DIM.get(architecture, 128))

    model = SentinelModel(
        num_classes=num_classes,
        fusion_output_dim=fusion_output_dim,
        gnn_hidden_dim=ckpt_config.get("gnn_hidden_dim", 64),
        gnn_heads=ckpt_config.get("gnn_heads", 8),
        use_edge_attr=ckpt_config.get("use_edge_attr", True),
        gnn_edge_emb_dim=ckpt_config.get("gnn_edge_emb_dim", 16),
        lora_r=ckpt_config.get("lora_r", 8),
        lora_alpha=ckpt_config.get("lora_alpha", 16),
        lora_dropout=ckpt_config.get("lora_dropout", 0.1),
        dropout=ckpt_config.get("fusion_dropout", 0.3),
        gnn_dropout=ckpt_config.get("gnn_dropout", 0.2),
        lora_target_modules=ckpt_config.get(
            "lora_target_modules", ["query", "value"]
        ),
    ).to(device)

    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(
        "Loaded checkpoint | architecture={} | num_classes={} | fusion_output_dim={}",
        architecture,
        num_classes,
        fusion_output_dim,
    )
    return model, ckpt_config


def build_val_loader(
    config: TrainConfig,
    label_csv: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build the validation DataLoader using the same split as training.

    Fix #5: DataLoader kwargs are built conditionally so that prefetch_factor,
    pin_memory, and persistent_workers are only passed when num_workers > 0.
    Passing prefetch_factor=None (or any value) with num_workers=0 triggers a
    UserWarning in PyTorch 2.x because the option is meaningless without workers.
    """
    val_indices = np.load(Path(config.splits_dir) / "val_indices.npy")

    dataset = DualPathDataset(
        graphs_dir=config.graphs_dir,
        tokens_dir=config.tokens_dir,
        indices=val_indices.tolist(),
        label_csv=str(label_csv),
        cache_path=getattr(config, "cache_path", None),
    )

    # Fix #5 — only include worker-specific kwargs when workers are actually used.
    _loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": dual_path_collate_fn,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        _loader_kwargs["prefetch_factor"] = 2
        _loader_kwargs["pin_memory"] = device_is_cuda(config.device)
        _loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **_loader_kwargs)


def device_is_cuda(device: str) -> bool:
    """Return True if device string refers to CUDA."""
    return str(device).startswith("cuda")


def collect_probabilities(
    model: SentinelModel,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one full forward pass over the validation set.

    Returns:
        probs:  [N, C] float32 array in [0, 1]
        labels: [N, C] int64 array with values 0/1
    """
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            graphs, tokens, labels = batch

            graphs = graphs.to(device)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            if labels.ndim != 2:
                raise ValueError(
                    f"Expected multi-label labels of shape [B, C], got {tuple(labels.shape)}. "
                    "Did you forget to pass label_csv=multilabel_index.csv?"
                )

            labels = labels.to(device).float()
            logits = model(graphs, input_ids, attention_mask)

            if logits.ndim != 2:
                raise ValueError(
                    f"Expected multi-label logits of shape [B, C], got {tuple(logits.shape)}."
                )

            # Fix #8: .float() cast before sigmoid for BF16/FP16 safety
            probs = torch.sigmoid(logits.float())

            all_probs.append(probs.cpu().numpy().astype(np.float32))
            all_labels.append(labels.cpu().numpy().astype(np.int64))

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    if probs_np.shape != labels_np.shape:
        raise RuntimeError(
            f"Shape mismatch after collection: probs {probs_np.shape} vs labels {labels_np.shape}"
        )

    if probs_np.shape[1] != len(CLASS_NAMES):
        raise RuntimeError(
            f"Collected {probs_np.shape[1]} classes, but CLASS_NAMES has {len(CLASS_NAMES)} entries."
        )

    supports = labels_np.sum(axis=0).tolist()
    support_str = ", ".join(f"{name}={int(sup)}" for name, sup in zip(CLASS_NAMES, supports))
    logger.info("Collected predictions for {} validation samples", len(labels_np))
    logger.info("Per-class support | {}", support_str)

    return probs_np, labels_np


def sweep_one_class(
    class_name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> tuple[BestThreshold, list[SweepRow]]:
    """
    Sweep thresholds for one class and choose the best by per-class F1.

    Tie-breaks:
    1. Higher F1
    2. Higher recall
    3. Lower threshold (prefer catching rare positives when equally good)
    """
    rows: list[SweepRow] = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.int64)

        row = SweepRow(
            threshold=threshold,
            f1=float(f1_score(labels, preds, zero_division=0)),
            precision=float(precision_score(labels, preds, zero_division=0)),
            recall=float(recall_score(labels, preds, zero_division=0)),
            predicted_positives=int(preds.sum()),
        )
        rows.append(row)

    best_row = max(
        rows,
        key=lambda row: (round(row.f1, 8), round(row.recall, 8), -row.threshold),
    )

    best = BestThreshold(
        class_name=class_name,
        threshold=best_row.threshold,
        f1=best_row.f1,
        precision=best_row.precision,
        recall=best_row.recall,
        support=int(labels.sum()),
        predicted_positives=best_row.predicted_positives,
    )
    return best, rows


def apply_thresholds(
    probs: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Apply per-class thresholds to probability matrix."""
    threshold_vec = np.array([thresholds[name] for name in CLASS_NAMES], dtype=np.float32)
    return (probs >= threshold_vec[None, :]).astype(np.int64)


def evaluate_overall(
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float]:
    """Compute overall multi-label metrics."""
    exact_match = float(np.all(labels == preds, axis=1).mean())
    return {
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds)),
        "exact_match_accuracy": exact_match,
    }


def print_class_report(best: BestThreshold, rows: list[SweepRow]) -> None:
    """Print one class sweep table."""
    print(f"\n{'=' * 78}")
    print(f"{best.class_name}")
    print(f"{'=' * 78}")
    print(
        f"{'Threshold':>10} | {'F1':>8} | {'Precision':>10} | "
        f"{'Recall':>8} | {'Pred+':>8}"
    )
    print("-" * 78)

    for row in rows:
        marker = " ← best" if row.threshold == best.threshold else ""
        print(
            f"{row.threshold:>10.2f} | "
            f"{row.f1:>8.4f} | "
            f"{row.precision:>10.4f} | "
            f"{row.recall:>8.4f} | "
            f"{row.predicted_positives:>8d}{marker}"
        )

    print("-" * 78)
    print(
        f"Selected threshold={best.threshold:.2f} | "
        f"support={best.support} | "
        f"F1={best.f1:.4f} | "
        f"P={best.precision:.4f} | "
        f"R={best.recall:.4f}"
    )


def print_summary(
    best_thresholds: list[BestThreshold],
    overall_metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Print final summary."""
    print(f"\n{'#' * 78}")
    print("FINAL PER-CLASS THRESHOLDS")
    print(f"{'#' * 78}")
    print(
        f"{'Class':<28} | {'Threshold':>9} | {'F1':>8} | "
        f"{'Precision':>10} | {'Recall':>8} | {'Support':>7}"
    )
    print("-" * 78)

    for best in best_thresholds:
        print(
            f"{best.class_name:<28} | "
            f"{best.threshold:>9.2f} | "
            f"{best.f1:>8.4f} | "
            f"{best.precision:>10.4f} | "
            f"{best.recall:>8.4f} | "
            f"{best.support:>7d}"
        )

    print(f"\n{'#' * 78}")
    print("OVERALL METRICS WITH BEST PER-CLASS THRESHOLDS")
    print(f"{'#' * 78}")
    print(f"F1-macro            : {overall_metrics['f1_macro']:.4f}")
    print(f"F1-micro            : {overall_metrics['f1_micro']:.4f}")
    print(f"Hamming loss        : {overall_metrics['hamming_loss']:.4f}")
    print(f"Exact-match accuracy: {overall_metrics['exact_match_accuracy']:.4f}")
    print(f"Saved JSON          : {output_path}")


def save_results(
    output_path: Path,
    checkpoint_path: Path,
    thresholds: dict[str, float],
    best_thresholds: list[BestThreshold],
    overall_metrics: dict[str, float],
    ckpt_config: dict[str, Any],
    thresholds_grid: list[float],
) -> None:
    """Save threshold tuning results as JSON."""
    payload = {
        "checkpoint": str(checkpoint_path),
        "architecture": ckpt_config.get("architecture", "unknown"),
        "num_classes": int(ckpt_config.get("num_classes", len(CLASS_NAMES))),
        "class_names": CLASS_NAMES,
        "threshold_grid": thresholds_grid,
        "thresholds": thresholds,
        "per_class": [asdict(item) for item in best_thresholds],
        "overall_metrics": overall_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved threshold JSON to {}", output_path)


def main() -> None:
    """Script entry point."""
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    label_csv = Path(args.label_csv)
    output_path = resolve_output_path(checkpoint_path, args.output)
    thresholds_grid = build_threshold_grid(args.start, args.end, args.step)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    config = TrainConfig()
    device = config.device
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    num_workers = args.num_workers if args.num_workers is not None else getattr(config, "num_workers", 0)

    logger.info(
        "Threshold tuning start | device={} | batch_size={} | num_workers={} | grid={}",
        device,
        batch_size,
        num_workers,
        thresholds_grid,
    )

    model, ckpt_config = load_model_from_checkpoint(checkpoint_path, device)
    val_loader = build_val_loader(
        config=config,
        label_csv=label_csv,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    probs, labels = collect_probabilities(model, val_loader, device)

    best_thresholds: list[BestThreshold] = []
    threshold_dict: dict[str, float] = {}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_probs = probs[:, class_idx]
        class_labels = labels[:, class_idx]

        best, rows = sweep_one_class(
            class_name=class_name,
            probs=class_probs,
            labels=class_labels,
            thresholds=thresholds_grid,
        )
        best_thresholds.append(best)
        threshold_dict[class_name] = best.threshold
        print_class_report(best, rows)

    preds = apply_thresholds(probs, threshold_dict)
    overall_metrics = evaluate_overall(labels, preds)

    save_results(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        thresholds=threshold_dict,
        best_thresholds=best_thresholds,
        overall_metrics=overall_metrics,
        ckpt_config=ckpt_config,
        thresholds_grid=thresholds_grid,
    )
    print_summary(best_thresholds, overall_metrics, output_path)


if __name__ == "__main__":
    main()
