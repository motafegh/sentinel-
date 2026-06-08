"""
utils.py — Shared utilities for SENTINEL GNN interpretability scripts.

PURPOSE
───────
Centralises all common operations used across the interpretability experiment
suite: checkpoint loading (with torch.compile key stripping), dataset loading
from the v9 cache, node-type tensor extraction, batched inference collection,
and matplotlib heatmap plotting.

All interpretability scripts (exp_s1 through exp_a4) import from this module
to avoid duplicating boilerplate and to guarantee consistent checkpoint loading
behaviour across experiments.

USAGE (from any exp_*.py script)
──────────────────────────────────
    from ml.scripts.interpretability.utils import (
        load_model, load_val_split, get_node_type_tensor,
        collect_predictions, plot_class_heatmap, add_common_args,
    )

EXIT CODES
──────────
    These utilities raise RuntimeError on fatal errors; callers decide exit code.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for WSL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES, NUM_EDGE_TYPES

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]

NUM_CLASSES: int = 10

PHASE_NAMES: list[str] = [
    "Phase1 (struct+CONTAINS)",
    "Phase2 (CF/ICFG/DFG)",
    "Phase3 (rev-CONTAINS)",
]

_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))  # 13.0 in v9 schema


# ── Checkpoint loading ────────────────────────────────────────────────────────

def strip_orig_mod(state_dict: dict) -> dict:
    """Strip torch.compile ._orig_mod. infix from all state dict keys."""
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def load_model(
    checkpoint_path: Path,
    device: str,
    phase2_edge_types: Optional[list[int]] = None,
) -> "SentinelModel":
    """
    Load SentinelModel from a checkpoint file.

    Handles:
    - torch.compile ._orig_mod. key stripping
    - weights_only=False (LoRA peft objects in checkpoint)
    - BF16→float32 conversion
    - Edge embedding resize if checkpoint has different NUM_EDGE_TYPES
    - Architecture params read from checkpoint config dict

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device:          Target device string ("cuda", "cpu").
        phase2_edge_types: Optional override for GNN Phase 2 edge types.

    Returns:
        SentinelModel in eval mode, float32, on device.

    Raises:
        FileNotFoundError: if checkpoint_path does not exist.
        RuntimeError: if checkpoint format is unrecognised.
    """
    from ml.src.models.sentinel_model import SentinelModel

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Provide a valid --checkpoint path, e.g.:\n"
            "  ml/checkpoints/sentinel_best.pt"
        )

    log.info(f"Loading checkpoint: {checkpoint_path}")
    raw = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    ckpt_cfg   = raw.get("config", {}) if isinstance(raw, dict) else {}

    state_dict = strip_orig_mod(state_dict)

    def _ensure_list(v, default):
        if v is None:
            return default
        return list(v)

    model = SentinelModel(
        num_classes          = int(ckpt_cfg.get("num_classes", 10)),
        fusion_output_dim    = int(ckpt_cfg.get("fusion_output_dim", 128)),
        dropout              = float(ckpt_cfg.get("fusion_dropout", 0.3)),
        gnn_hidden_dim       = int(ckpt_cfg.get("gnn_hidden_dim", 256)),
        gnn_num_layers       = int(ckpt_cfg.get("gnn_layers", 8)),
        gnn_heads            = int(ckpt_cfg.get("gnn_heads", 8)),
        gnn_dropout          = float(ckpt_cfg.get("gnn_dropout", 0.2)),
        use_edge_attr        = bool(ckpt_cfg.get("use_edge_attr", True)),
        gnn_edge_emb_dim     = int(ckpt_cfg.get("gnn_edge_emb_dim", 64)),
        gnn_use_jk           = bool(ckpt_cfg.get("gnn_use_jk", True)),
        gnn_jk_mode          = str(ckpt_cfg.get("gnn_jk_mode", "attention")),
        gnn_phase2_edge_types= (
            phase2_edge_types if phase2_edge_types is not None
            else ckpt_cfg.get("gnn_phase2_edge_types")
        ),
        lora_r               = int(ckpt_cfg.get("lora_r", 16)),
        lora_alpha           = int(ckpt_cfg.get("lora_alpha", 32)),
        lora_dropout         = float(ckpt_cfg.get("lora_dropout", 0.1)),
        lora_target_modules  = _ensure_list(
            ckpt_cfg.get("lora_target_modules"), ["query", "value"]
        ),
        gnn_prefix_k             = int(ckpt_cfg.get("gnn_prefix_k", 48)),
        gnn_prefix_warmup_epochs = int(ckpt_cfg.get("gnn_prefix_warmup_epochs", 15)),
        fusion_max_nodes         = int(ckpt_cfg.get("fusion_max_nodes", 2048)),
        drop_complexity_feature  = bool(ckpt_cfg.get("drop_complexity_feature", False)),
        appnp_alpha              = float(ckpt_cfg.get("appnp_alpha", 0.0)),
    ).to(device)

    # Resize edge embedding if needed
    edge_emb_key = next((k for k in state_dict if "edge_embedding.weight" in k), None)
    if edge_emb_key and model.gnn.edge_embedding is not None:
        ckpt_n = state_dict[edge_emb_key].shape[0]
        curr_n = model.gnn.edge_embedding.num_embeddings
        if ckpt_n != curr_n:
            emb_dim = model.gnn.edge_embedding.embedding_dim
            model.gnn.edge_embedding = nn.Embedding(ckpt_n, emb_dim).to(device)
            log.info(f"Resized edge_embedding: {curr_n} -> {ckpt_n}")

    model.load_state_dict(state_dict, strict=False)
    model.float()
    model.eval()

    # Set epoch to large value so prefix is active
    model._current_epoch = 9999

    log.info(f"Model loaded successfully (device={device})")
    return model


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_val_split(
    cache_path: Path,
    label_csv: Path,
    splits_dir: Path,
    split: str = "val",
) -> tuple[list[str], pd.DataFrame, dict]:
    """
    Load the cache and return stems, label dataframe, and cache dict for a split.

    Args:
        cache_path:  Path to cached_dataset_v9.pkl (v9 schema, current)
        label_csv:   Path to multilabel_index_deduped.csv
        splits_dir:  Directory containing {split}_indices.npy
        split:       One of "train", "val", "test"

    Returns:
        (stems, df_split, cache)
        - stems:    list of md5_stem strings in split order
        - df_split: DataFrame with CLASS_NAMES columns and md5_stem
        - cache:    dict mapping stem -> (graph, token)

    Raises:
        FileNotFoundError: if any required file is missing.
    """
    cache_path  = Path(cache_path)
    label_csv   = Path(label_csv)
    splits_dir  = Path(splits_dir)
    npy_path    = splits_dir / f"{split}_indices.npy"

    for p, name in [
        (cache_path,  "cache"),
        (label_csv,   "label_csv"),
        (splits_dir,  "splits_dir"),
        (npy_path,    f"{split}_indices.npy"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"{name} not found: {p}\n"
                "Run with correct --cache / --label-csv / --splits-dir paths."
            )

    log.info(f"Loading cache: {cache_path} ({cache_path.stat().st_size / 1e9:.2f} GB)")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    log.info(f"Cache loaded: {len(cache):,} entries")

    df_full  = pd.read_csv(label_csv)
    indices  = np.load(npy_path)
    df_split = df_full.iloc[indices].reset_index(drop=True)
    stems    = df_split["md5_stem"].tolist()
    log.info(f"Split '{split}': {len(stems):,} samples")

    return stems, df_split, cache


# ── Node feature helpers ──────────────────────────────────────────────────────

def get_node_type_tensor(graph) -> torch.Tensor:
    """
    Recover integer node type IDs from graph.x[:, 0].

    graph.x[:, 0] is stored as float(type_id) / MAX_TYPE_ID (12.0).
    Multiply back and round to recover the integer.

    Returns:
        LongTensor of shape [N] with values in [0, 12].
    """
    return (graph.x[:, 0].float() * _MAX_TYPE_ID).round().long()


# ── Inference collection ──────────────────────────────────────────────────────

def collect_predictions(
    model,
    stems: list[str],
    df_split: pd.DataFrame,
    cache: dict,
    device: str,
    return_aux: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Run batched inference and collect predictions from all four heads.

    Args:
        model:       SentinelModel in eval mode.
        stems:       List of md5_stem strings.
        df_split:    DataFrame with CLASS_NAMES label columns.
        cache:       Cache dict mapping stem -> (graph, token).
        device:      Device string.
        return_aux:  If True, collect gnn/transformer/fused aux heads.
        max_samples: Subsample stems if provided.
        seed:        RNG seed for subsampling.

    Returns:
        dict with keys:
            "logits":      np.ndarray [N, 10]
            "gnn":         np.ndarray [N, 10]  (if return_aux)
            "transformer": np.ndarray [N, 10]  (if return_aux)
            "fused":       np.ndarray [N, 10]  (if return_aux)
            "labels":      np.ndarray [N, 10]  int binary labels
            "stems":       list[str]  stems in inference order
    """
    from torch_geometric.data import Batch

    if max_samples is not None and len(stems) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(stems), size=max_samples, replace=False)
        stems = [stems[i] for i in indices]

    stem_to_labels = {
        row["md5_stem"]: [int(row[c]) for c in CLASS_NAMES]
        for _, row in df_split.iterrows()
    }

    all_logits, all_gnn, all_tf, all_fused, all_labels, valid_stems = (
        [], [], [], [], [], []
    )

    model.eval()
    with torch.no_grad():
        for stem in stems:
            if stem not in cache:
                continue
            entry = cache[stem]
            if not isinstance(entry, tuple) or len(entry) < 2:
                continue

            graph, token = entry
            labels = stem_to_labels.get(stem)
            if labels is None:
                continue

            try:
                batch       = Batch.from_data_list([graph]).to(device)
                input_ids   = token["input_ids"].unsqueeze(0).to(device)
                attn_mask   = token["attention_mask"].unsqueeze(0).to(device)

                if return_aux:
                    logits, aux = model(batch, input_ids, attn_mask, return_aux=True)
                    all_gnn.append(aux["gnn"].cpu().numpy())
                    all_tf.append(aux["transformer"].cpu().numpy())
                    all_fused.append(aux["fused"].cpu().numpy())
                else:
                    logits = model(batch, input_ids, attn_mask, return_aux=False)

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels)
                valid_stems.append(stem)
            except Exception as exc:
                log.debug(f"Skipping {stem}: {exc}")
                continue

    if not all_logits:
        raise RuntimeError("No predictions collected — check cache/model compatibility.")

    result: dict = {
        "logits": np.vstack(all_logits),
        "labels": np.array(all_labels, dtype=np.int32),
        "stems":  valid_stems,
    }
    if return_aux and all_gnn:
        result["gnn"]         = np.vstack(all_gnn)
        result["transformer"] = np.vstack(all_tf)
        result["fused"]       = np.vstack(all_fused)

    log.info(f"Collected predictions for {len(valid_stems):,} samples")
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_class_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: Path,
    fmt: str = ".2f",
    cmap: str = "Blues",
    figsize: tuple = (14, 8),
) -> None:
    """
    Save a labelled heatmap as PNG.

    Args:
        matrix:      2D numpy array (rows=row_labels, cols=col_labels).
        row_labels:  Y-axis labels.
        col_labels:  X-axis labels.
        title:       Figure title.
        output_path: Where to save the PNG.
        fmt:         Cell annotation format string.
        cmap:        Matplotlib colormap.
        figsize:     Figure size tuple.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val > (matrix.max() * 0.7) else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=7, color=text_color)

    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Heatmap saved: {output_path}")


# ── Argument parser helpers ───────────────────────────────────────────────────

def add_common_args(parser: argparse.ArgumentParser, require_checkpoint: bool = True) -> None:
    """
    Add standard arguments used by all interpretability scripts.

    Args:
        parser:             ArgumentParser to augment.
        require_checkpoint: If False, --checkpoint is optional (for graph-only scripts).
    """
    parser.add_argument(
        "--checkpoint",
        default=None,
        required=require_checkpoint,
        help="Path to model checkpoint .pt (e.g. ml/checkpoints/sentinel_best.pt)",
    )
    parser.add_argument(
        "--cache",
        default="ml/data/cached_dataset_v9.pkl",
        help="Path to cached dataset pickle (v9 schema, current)",
    )
    parser.add_argument(
        "--label-csv",
        default="ml/data/processed/multilabel_index_deduped.csv",
        help="Path to multilabel_index_deduped.csv (41,576 rows, 10 classes)",
    )
    parser.add_argument(
        "--splits-dir",
        default="ml/data/splits/deduped",
        help="Directory containing {train,val,test}_indices.npy",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory or file path for JSON/PNG results",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--n-contracts",
        type=int,
        default=500,
        dest="n_contracts",
        help="Max contracts to sample from val split (default: 500)",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Which split to use (default: val)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible subsampling (default: 42)",
    )
    parser.add_argument(
        "--phase2-edge-types",
        type=int,
        nargs="+",
        default=None,
        dest="phase2_edge_types",
        help="Override Phase 2 edge types (e.g. 6 8 9). Must match training config.",
    )
