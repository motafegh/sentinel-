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
    A sample of real 64-dim FusionLayer outputs from the teacher model.
    The same kind of data the proxy sees at inference time.
    EZKL needs this as a JSON file in its specific input format.

EZKL calibration data format:
    {
        "input_data": [[f1, f2, ..., f64], [f1, f2, ..., f64], ...]
    }
    A list of input samples — each sample is a list of 64 floats.
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

TEACHER_CHECKPOINT  = "ml/checkpoints/run-alpha-tune_best.pt"
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
    Extract real 64-dim FusionLayer features from the teacher model
    and save them in EZKL's calibration JSON format.

    Args:
        n_samples: Number of calibration samples to extract
        output:    Path to write calibration.json
        device:    cuda or cpu
    """
    logger.info(f"Generating {n_samples} calibration samples on: {device}")

    # ------------------------------------------------------------------
    # Load teacher — same as train_proxy.py
    # ------------------------------------------------------------------
    teacher = SentinelModel().to(device)
    state_dict = torch.load(
        TEACHER_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    teacher.load_state_dict(state_dict)
    teacher.eval()
    logger.info("Teacher loaded")

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
    # Extract FusionLayer features
    # ------------------------------------------------------------------
    # RECALL — we intercept at FusionLayer output, not final score.
    #   The proxy receives 64-dim features as input.
    #   EZKL needs to see the actual distribution of those 64 values
    #   to calibrate the integer scale correctly.
    #   Final score is a single scalar — too little information for
    #   calibrating the full circuit's value ranges.
    all_features = []

    for graphs, tokens, _ in loader:
        graphs         = graphs.to(device)
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Run teacher up to FusionLayer — same interception as train_proxy
        gnn_out         = teacher.gnn(graphs.x, graphs.edge_index, graphs.batch)
        transformer_out = teacher.transformer(input_ids, attention_mask)
        features        = teacher.fusion(gnn_out, transformer_out)  # [B, 64]

        all_features.append(features.cpu())

        if sum(f.shape[0] for f in all_features) >= n_samples:
            break

    features_tensor = torch.cat(all_features)[:n_samples]  # [N, 64]
    logger.info(f"Features extracted — shape: {features_tensor.shape}")

    # ------------------------------------------------------------------
    # Format for EZKL
    # ------------------------------------------------------------------
    # EZKL calibration format:
    #   {"input_data": [[64 floats], [64 floats], ...]}
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