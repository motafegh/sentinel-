"""Train the SENTINEL ZKML student against an explicitly selected DATA export.

There is intentionally **no default export directory**. The old code silently
used ``sentinel-v2-baseline-2026-06-12`` even after the teacher/data lineage had
moved on. A future proxy retrain must name the promoted export explicitly and
that identity is embedded in the proxy checkpoint metadata.

The current distillation target is the teacher probability vector:
``sigmoid(teacher_logits)``. ``ProxyModel.forward()`` is fitted directly to that
vector; consumers must not apply another sigmoid.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.src.datasets.collate import sentinel_collate_fn
from ml.src.datasets.sentinel_dataset import SentinelDataset
from ml.src.models.sentinel_model import SentinelModel
from zkml.src.distillation.proxy_model import (
    CIRCUIT_VERSION,
    OUTPUT_SEMANTICS,
    ProxyModel,
)

TEACHER_CHECKPOINT = Path(
    "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"
)
PROXY_CHECKPOINT = Path("zkml/models/proxy_best.pt")
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
AGREEMENT_TARGET = 0.95
THRESHOLD = 0.50
RANDOM_SEED = 42


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_export(export_dir: Path) -> tuple[Path, str]:
    if not export_dir.exists() or not export_dir.is_dir():
        raise FileNotFoundError(f"explicit DATA export does not exist: {export_dir}")
    manifest = export_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"export manifest missing: {manifest}; refusing unbound proxy retraining"
        )
    return manifest, _sha256(manifest)


def _load_teacher(device: str) -> tuple[SentinelModel, dict]:
    if not TEACHER_CHECKPOINT.exists():
        raise FileNotFoundError(f"teacher checkpoint missing: {TEACHER_CHECKPOINT}")
    checkpoint = torch.load(
        TEACHER_CHECKPOINT, map_location=device, weights_only=False
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        config = checkpoint.get("config", {}) or {}
    else:
        state_dict = checkpoint
        config = {}

    teacher = SentinelModel(
        num_classes=config.get("num_classes", 10),
        fusion_output_dim=config.get("fusion_output_dim", 128),
        gnn_prefix_k=config.get("gnn_prefix_k", 0),
        gnn_prefix_warmup_epochs=config.get("gnn_prefix_warmup_epochs", 15),
        use_edge_attr=config.get("use_edge_attr", True),
        gnn_hidden_dim=config.get("gnn_hidden_dim", 256),
        gnn_num_layers=config.get("gnn_layers", 8),
        gnn_heads=config.get("gnn_heads", 8),
        gnn_dropout=config.get("gnn_dropout", 0.2),
        gnn_use_jk=config.get("gnn_use_jk", True),
        gnn_jk_mode=config.get("gnn_jk_mode", "attention"),
        fusion_max_nodes=config.get("fusion_max_nodes", 1024),
    ).to(device)
    teacher.load_state_dict(state_dict)
    teacher.float().eval()
    return teacher, config


@torch.no_grad()
def extract_features(
    teacher: SentinelModel,
    graphs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher.eval()
    logits, aux = teacher(
        graphs.to(device),
        input_ids.to(device),
        attention_mask.to(device),
        return_aux=True,
    )
    features = aux["fusion_embedding"]
    teacher_scores = torch.sigmoid(logits.float())
    if features.ndim != 2 or features.shape[1] != 128:
        raise RuntimeError(f"teacher fusion shape must be [B,128], got {tuple(features.shape)}")
    if teacher_scores.ndim != 2 or teacher_scores.shape[1] != 10:
        raise RuntimeError(
            f"teacher probability shape must be [B,10], got {tuple(teacher_scores.shape)}"
        )
    return features.cpu(), teacher_scores.cpu()


def compute_agreement(
    proxy_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    threshold: float = THRESHOLD,
) -> float:
    if proxy_scores.shape != teacher_scores.shape:
        raise ValueError(
            f"agreement shape mismatch: proxy={tuple(proxy_scores.shape)} "
            f"teacher={tuple(teacher_scores.shape)}"
        )
    proxy_labels = (proxy_scores >= threshold).long()
    teacher_labels = (teacher_scores >= threshold).long()
    return (proxy_labels == teacher_labels).float().mean().item()


def _extract_dataset_features(
    teacher: SentinelModel,
    loader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features_list: list[torch.Tensor] = []
    scores_list: list[torch.Tensor] = []
    for graphs, tokens, _y, _cids, _tiers in loader:
        features, scores = extract_features(
            teacher,
            graphs,
            tokens["input_ids"],
            tokens["attention_mask"],
            device,
        )
        features_list.append(features)
        scores_list.append(scores)
    if not features_list:
        raise RuntimeError("distillation dataset produced zero batches")
    return torch.cat(features_list), torch.cat(scores_list)


def train(
    export_dir: Path,
    *,
    output: Path = PROXY_CHECKPOINT,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Train a proxy and return the saved checkpoint metadata."""
    export_dir = export_dir.resolve()
    manifest_path, export_manifest_sha256 = _validate_export(export_dir)
    teacher_sha256 = _sha256(TEACHER_CHECKPOINT)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    logger.info(
        "Proxy distillation — teacher={} teacher_sha256={} export={} manifest_sha256={}",
        TEACHER_CHECKPOINT,
        teacher_sha256,
        export_dir,
        export_manifest_sha256,
    )

    teacher, teacher_config = _load_teacher(device)
    train_dataset = SentinelDataset(split="train", export_dir=str(export_dir))
    val_dataset = SentinelDataset(split="val", export_dir=str(export_dir))
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"distillation requires non-empty train/val populations; "
            f"got train={len(train_dataset)} val={len(val_dataset)}"
        )

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=sentinel_collate_fn,
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=sentinel_collate_fn,
    )

    train_features, train_targets = _extract_dataset_features(teacher, train_loader, device)
    val_features, val_targets = _extract_dataset_features(teacher, val_loader, device)

    proxy_train = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    proxy_val = DataLoader(
        TensorDataset(val_features, val_targets),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    proxy = ProxyModel().to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(proxy.parameters(), lr=LR)
    best_agreement = -1.0
    best_epoch = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    base_metadata = {
        "schema": "sentinel-zkml-proxy-checkpoint-v2",
        "circuit_version": CIRCUIT_VERSION,
        "output_semantics": OUTPUT_SEMANTICS,
        "teacher_target": "sigmoid_teacher_logits",
        "teacher_checkpoint": TEACHER_CHECKPOINT.as_posix(),
        "teacher_checkpoint_sha256": teacher_sha256,
        "teacher_config_num_classes": int(teacher_config.get("num_classes", 10)),
        "teacher_config_fusion_output_dim": int(
            teacher_config.get("fusion_output_dim", 128)
        ),
        "export_dir": export_dir.as_posix(),
        "export_manifest": manifest_path.as_posix(),
        "export_manifest_sha256": export_manifest_sha256,
        "random_seed": RANDOM_SEED,
        "agreement_threshold": THRESHOLD,
    }

    for epoch in range(1, EPOCHS + 1):
        proxy.train()
        total_loss = 0.0
        for features_batch, target_batch in proxy_train:
            features_batch = features_batch.to(device)
            target_batch = target_batch.to(device)
            optimiser.zero_grad()
            student_scores = proxy(features_batch)
            loss = criterion(student_scores, target_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        proxy.eval()
        proxy_scores: list[torch.Tensor] = []
        teacher_scores: list[torch.Tensor] = []
        with torch.no_grad():
            for features_batch, target_batch in proxy_val:
                student_scores = proxy(features_batch.to(device)).float().cpu()
                if not torch.isfinite(student_scores).all():
                    raise RuntimeError("proxy produced non-finite validation score(s)")
                proxy_scores.append(student_scores)
                teacher_scores.append(target_batch)

        agreement = compute_agreement(
            torch.cat(proxy_scores), torch.cat(teacher_scores)
        )
        mean_loss = total_loss / max(len(proxy_train), 1)
        logger.info(
            "Epoch {}/{} loss={:.6f} agreement={:.4f}",
            epoch,
            EPOCHS,
            mean_loss,
            agreement,
        )

        if agreement > best_agreement:
            best_agreement = agreement
            best_epoch = epoch
            checkpoint = {
                "model": proxy.state_dict(),
                "metadata": {
                    **base_metadata,
                    "best_agreement": best_agreement,
                    "best_epoch": best_epoch,
                },
            }
            torch.save(checkpoint, output)

        if agreement >= AGREEMENT_TARGET:
            break

    if best_epoch == 0:
        raise RuntimeError("proxy training completed without a valid checkpoint")

    metadata = {
        **base_metadata,
        "best_agreement": best_agreement,
        "best_epoch": best_epoch,
        "checkpoint_path": output.as_posix(),
        "checkpoint_sha256": _sha256(output),
    }
    logger.info("Distillation complete: {}", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Explicit promoted SENTINEL DATA export; no implicit historical default exists.",
    )
    parser.add_argument("--output", type=Path, default=PROXY_CHECKPOINT)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train(args.export_dir, output=args.output, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
