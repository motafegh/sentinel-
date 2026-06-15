"""calibrate_temperature_v3.py — V3-aware per-class temperature scaling for SENTINEL.

Replaces the legacy ml/scripts/calibrate_temperature.py which uses STALE v9/v10 paths
(ml/data/cached_dataset_v9.pkl, ml/data/processed/multilabel_index_deduped.csv,
ml/data/splits/deduped). The legacy script was incompatible with v3 exports.

This v3-aware version uses the same SentinelDataset + v3 export pattern as
tune_threshold.py (proven working, 2026-06-14).

Methodology (unchanged from legacy):
  Fit one scalar temperature T_c per class by minimising BCE NLL on the val set.
  Calibrated logit for class c = logit_c / T_c.
  Use LBFGS optimiser (200 max iter, lr=0.05, strong_wolfe line search).

Outputs:
  - temperatures_v<N>.json      : {class_name: T_c, ...}
  - temperatures_v<N>_stats.json: ECE before/after per class
  - temperatures_v<N>_ece_comparison.png: bar chart

Usage:
    ml/.venv/bin/python /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/calibrate_temperature_v3.py \
        --checkpoint ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt \
        --export-dir data_module/data/exports/sentinel-v3-smartbugs-2026-06-13 \
        --out ml/calibration/temperatures_run12.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.src.datasets import SentinelDataset, sentinel_collate_fn
from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.trainer import CLASS_NAMES, NUM_CLASSES, TrainConfig

N_BINS = 15


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Equal-width ECE for a single class."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += mask.mean() * abs(conf - acc)
    return float(ece)


def compute_all_ece(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.array([compute_ece(probs[:, c], labels[:, c]) for c in range(NUM_CLASSES)])


class PerClassTemperature(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(num_classes))

    @property
    def temperatures(self) -> torch.Tensor:
        return self.log_T.exp().clamp(min=0.05, max=20.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperatures


def fit_temperatures(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200, lr: float = 0.05) -> np.ndarray:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    model = PerClassTemperature(NUM_CLASSES)
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    bce = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = model(logits_t)
        loss = bce(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return model.temperatures.detach().numpy()


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> tuple[SentinelModel, dict]:
    """Same pattern as tune_threshold.py — reads config from checkpoint, builds model."""
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = raw.get("config", {}) if isinstance(raw, dict) else {}
    architecture = ckpt_config.get("architecture", "cross_attention_lora")
    num_classes = int(ckpt_config.get("num_classes", NUM_CLASSES))

    model = SentinelModel(
        num_classes=num_classes,
        fusion_output_dim=ckpt_config.get("fusion_output_dim", 128),
        gnn_hidden_dim=ckpt_config.get("gnn_hidden_dim", 64),
        gnn_num_layers=ckpt_config.get("gnn_layers", 4),
        gnn_heads=ckpt_config.get("gnn_heads", 8),
        use_edge_attr=ckpt_config.get("use_edge_attr", True),
        gnn_edge_emb_dim=ckpt_config.get("gnn_edge_emb_dim", 16),
        gnn_use_jk=ckpt_config.get("gnn_use_jk", True),
        lora_r=ckpt_config.get("lora_r", 8),
        lora_alpha=ckpt_config.get("lora_alpha", 16),
        lora_dropout=ckpt_config.get("lora_dropout", 0.1),
        dropout=ckpt_config.get("fusion_dropout", 0.3),
        gnn_dropout=ckpt_config.get("gnn_dropout", 0.2),
        lora_target_modules=ckpt_config.get("lora_target_modules", ["query", "value"]),
        gnn_prefix_k=ckpt_config.get("gnn_prefix_k", 0),
        gnn_prefix_warmup_epochs=ckpt_config.get("gnn_prefix_warmup_epochs", 15),
        gnn_phase2_edge_types=ckpt_config.get("gnn_phase2_edge_types", None),
        fusion_max_nodes=ckpt_config.get("fusion_max_nodes", 2048),
        drop_complexity_feature=bool(ckpt_config.get("drop_complexity_feature", False)),
        appnp_alpha=float(ckpt_config.get("appnp_alpha", 0.0)),
    ).to(device)

    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    # Resize edge_embedding if schema mismatch
    edge_emb_key = next((k for k in state_dict if "edge_embedding.weight" in k), None)
    if edge_emb_key and model.gnn.edge_embedding is not None:
        ckpt_n = state_dict[edge_emb_key].shape[0]
        curr_n = model.gnn.edge_embedding.num_embeddings
        if ckpt_n != curr_n:
            emb_dim = model.gnn.edge_embedding.embedding_dim
            model.gnn.edge_embedding = nn.Embedding(ckpt_n, emb_dim).to(device)
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    return model, ckpt_config


def collect_logits(model: SentinelModel, val_loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            graphs, tokens, labels, *_ = batch
            graphs = graphs.to(device)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            labels = labels.to(device).float()
            logits = model(graphs, input_ids, attention_mask)
            all_logits.append(logits.float().cpu().numpy().astype(np.float32))
            all_labels.append(labels.cpu().numpy().astype(np.int64))
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def plot_ece_comparison(ece_before: np.ndarray, ece_after: np.ndarray, out_path: Path) -> None:
    x = np.arange(NUM_CLASSES)
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars_b = ax.bar(x - width / 2, ece_before, width, label="Before", color="#e07070", alpha=0.85)
    bars_a = ax.bar(x + width / 2, ece_after, width, label="After", color="#70aae0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("ECE (lower = better)")
    ax.set_title("Per-class ECE before/after temperature scaling (v3 export)")
    ax.legend()
    ax.axhline(0.05, color="green", linestyle="--", linewidth=1, label="target 0.05")
    for bar in bars_a:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-dir", default="data_module/data/exports/sentinel-v3-smartbugs-2026-06-13")
    parser.add_argument("--out", default="ml/calibration/temperatures.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(export_dir=args.export_dir)
    device = config.device
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Export dir: {args.export_dir}")

    print("Loading model...")
    model, ckpt_config = load_model_from_checkpoint(Path(args.checkpoint), device)
    print(f"  architecture: {ckpt_config.get('architecture')}, num_classes: {ckpt_config.get('num_classes')}")

    print("Loading val split (SentinelDataset)...")
    val_dataset = SentinelDataset(split="val", export_dir=Path(args.export_dir))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=sentinel_collate_fn, num_workers=args.num_workers
    )
    print(f"  val contracts: {len(val_dataset)}")

    print("Collecting logits...")
    logits, labels = collect_logits(model, val_loader, device)
    print(f"  collected: logits={logits.shape}, labels={labels.shape}")

    probs = 1.0 / (1.0 + np.exp(-logits))
    ece_before = compute_all_ece(probs, labels)
    print("\nECE before calibration:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:30s} {ece_before[c]:.4f}")
    print(f"  Mean ECE: {ece_before.mean():.4f}")

    print(f"\nFitting per-class temperatures (LBFGS max_iter={args.max_iter})...")
    temperatures = fit_temperatures(logits, labels, max_iter=args.max_iter, lr=args.lr)
    print("Fitted temperatures:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:30s} T={temperatures[c]:.4f}")

    scaled_logits = logits / temperatures[np.newaxis, :]
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    ece_after = compute_all_ece(scaled_probs, labels)
    print("\nECE after calibration:")
    for c, name in enumerate(CLASS_NAMES):
        delta = ece_after[c] - ece_before[c]
        print(f"  {name:30s} {ece_after[c]:.4f}  (Δ={delta:+.4f})")
    print(f"  Mean ECE after: {ece_after.mean():.4f}  (Δ={ece_after.mean() - ece_before.mean():+.4f})")

    temps_dict = {name: float(temperatures[c]) for c, name in enumerate(CLASS_NAMES)}
    with open(out_path, "w") as f:
        json.dump(temps_dict, f, indent=2)
    print(f"\nTemperatures saved → {out_path}")

    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    stats = {
        "checkpoint": str(args.checkpoint),
        "export_dir": str(args.export_dir),
        "ece_before": {name: float(ece_before[c]) for c, name in enumerate(CLASS_NAMES)},
        "ece_after":  {name: float(ece_after[c]) for c, name in enumerate(CLASS_NAMES)},
        "mean_ece_before": float(ece_before.mean()),
        "mean_ece_after":  float(ece_after.mean()),
        "temperatures":    temps_dict,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved → {stats_path}")

    plot_path = out_path.with_name(out_path.stem + "_ece_comparison.png")
    plot_ece_comparison(ece_before, ece_after, plot_path)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()
