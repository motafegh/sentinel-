"""
generate_calibration.py — Extract calibration data for EZKL Step 2

RECALL — why calibration data is needed:
    EZKL Step 1 (gen_settings) makes an initial guess at the scale factor —
    the multiplier that converts floating point weights to integers for
    the ZK circuit. That guess is based on model structure alone.

    Step 2 (calibrate_settings) refines that guess using REAL data.
    It runs actual contract features through the model and observes
    the true range of values at each layer — catching overflow risks
    that the initial guess might miss.

    Without real calibration data:
      Scale too small → values overflow inside the circuit → wrong proofs
      Scale too large → circuit unnecessarily huge → slow proving

RECALL — what the calibration data actually is:
    A sample of real 128-dim CrossAttentionFusion outputs from the teacher model.
    The same kind of data the proxy sees at inference time (ADR-025, v2.0).
    EZKL needs this as a JSON file in its specific input format.

EZKL calibration data format:
    {
        "input_data": [[f1, f2, ..., f128], [f1, f2, ..., f128], ...]
    }
    A list of input samples — each sample is a list of 128 floats.
    EZKL runs these through the ONNX model and observes value ranges.

Output:
    zkml/ezkl/calibration.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from ml.src.models.sentinel_model import SentinelModel
from torch_geometric.loader import DataLoader

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

TEACHER_CHECKPOINT  = "ml/checkpoints/multilabel_crossattn_best.pt"
GRAPHS_DIR          = "ml/data/graphs"
TOKENS_DIR          = "ml/data/tokens"
SPLITS_DIR          = "ml/data/splits"
CALIBRATION_OUTPUT  = "zkml/ezkl/calibration.json"

# RECALL — how many samples to use:
#   Too few: EZKL misses edge cases in value distribution → overflow risk
#   Too many: calibration takes longer with diminishing returns
#   200 samples covers the distribution well for a proxy this small
N_CALIBRATION_SAMPLES = 200


@torch.no_grad()
def generate(
    n_samples: int = N_CALIBRATION_SAMPLES,
    output:    str = CALIBRATION_OUTPUT,
    device:    str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Extract real 128-dim CrossAttentionFusion features from the teacher model
    and save them in EZKL's calibration JSON format.

    Args:
        n_samples: Number of calibration samples to extract
        output:    Path to write calibration.json
        device:    cuda or cpu
    """
    logger.info(f"Generating {n_samples} calibration samples on: {device}")

    # ------------------------------------------------------------------
    # Load teacher — same pattern as train_proxy.py
    # ------------------------------------------------------------------
    checkpoint = torch.load(
        TEACHER_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        config     = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config     = {}

    num_classes       = config.get("num_classes", 10)
    fusion_output_dim = config.get("fusion_output_dim", 128)

    teacher = SentinelModel(
        num_classes=num_classes,
        fusion_output_dim=fusion_output_dim,
    ).to(device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    logger.info(
        f"Teacher loaded — num_classes={num_classes} fusion_output_dim={fusion_output_dim}"
    )

    # ------------------------------------------------------------------
    # Load val set — use val not train for calibration
    # ------------------------------------------------------------------
    # RECALL — why val set for calibration:
    #   Calibration data should represent the distribution of inputs
    #   the model sees in production — unseen contracts.
    #   Val set is held-out data, closer to production distribution
    #   than train set which the teacher already memorised.
    val_indices = np.load(f"{SPLITS_DIR}/val_indices.npy")

    # Only need enough indices to cover n_samples
    subset_indices = val_indices[:n_samples].tolist()

    dataset = DualPathDataset(
        graphs_dir=GRAPHS_DIR,
        tokens_dir=TOKENS_DIR,
        indices=subset_indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=dual_path_collate_fn,
    )

    logger.info(f"Val subset loaded — {len(dataset)} contracts")

    # ------------------------------------------------------------------
    # Extract CrossAttentionFusion features
    # ------------------------------------------------------------------
    # RECALL — we intercept at CrossAttentionFusion output, not final score.
    #   The proxy receives 128-dim features as input (ADR-025, v2.0).
    #   EZKL needs to see the actual distribution of those 128 values
    #   to calibrate the integer scale correctly.
    #   Final logits are [B, 10] — value range different from the fused
    #   embedding; calibrating on the embedding is more representative.
    all_features = []

    for graphs, tokens, _ in loader:
        graphs         = graphs.to(device)
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # GNN path: returns (node_embs [N, 64], batch [N])
        node_embs, batch = teacher.gnn(graphs.x, graphs.edge_index, graphs.batch)
        # Transformer path: returns [B, 512, 768]
        transformer_out = teacher.transformer(input_ids, attention_mask)
        # CrossAttentionFusion: (node_embs, batch, token_embs, attention_mask) → [B, 128]
        features = teacher.fusion(node_embs, batch, transformer_out, attention_mask)

        all_features.append(features.cpu())

        if sum(f.shape[0] for f in all_features) >= n_samples:
            break

    features_tensor = torch.cat(all_features)[:n_samples]  # [N, 128]
    logger.info(f"Features extracted — shape: {features_tensor.shape}")

    # ------------------------------------------------------------------
    # Format for EZKL
    # ------------------------------------------------------------------
    # EZKL calibration format:
    #   {"input_data": [[128 floats], [128 floats], ...]}
    #
    # Each inner list = one contract's feature vector.
    # Python floats required — not numpy floats, not torch tensors.
    # RECALL — EZKL reads this to observe real value ranges at each
    # layer and tune the scale factor accordingly.
    calibration_data = {
        "input_data": features_tensor.numpy().tolist()
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(calibration_data, f)

    logger.info(f"Calibration data saved — {n_samples} samples: {output}")

    # Quick sanity check on the value range
    flat = features_tensor.numpy().flatten()
    logger.info(
        f"Feature value range: [{flat.min():.4f}, {flat.max():.4f}] "
        f"— mean: {flat.mean():.4f}, std: {flat.std():.4f}"
    )
    logger.info("Ready for EZKL calibrate_settings — next: setup_circuit.py")


if __name__ == "__main__":
    generate()