"""
End-to-end test for SentinelModel.

Loads a real DataLoader batch and runs a full forward pass through
GNNEncoder → TransformerEncoder → FusionLayer → classifier.

Run with:
    poetry run python ml/scripts/test_sentinel_model.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np


import torch
from torch.utils.data import DataLoader
from loguru import logger

from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from ml.src.models.sentinel_model import SentinelModel
# Load train indices
train_indices = np.load("ml/data/splits/train_indices.npy")

def run_end_to_end_test() -> None:
    """
    Load one real batch and run full forward pass through SentinelModel.

    Verifies:
        - No shape errors through the full pipeline
        - Output shape is [B] — one score per contract
        - Scores are in [0, 1] range — sigmoid is working
        - Labels shape matches scores shape — ready for loss computation
    """
    logger.info("Starting end-to-end test...")

    # --- Dataset + DataLoader ---
    # Instantiate with correct arguments
    dataset = DualPathDataset(
        graphs_dir="ml/data/graphs",
        tokens_dir="ml/data/tokens",
        indices=train_indices,
        validate=False,  # skip validation — we just need one batch
    )
    loader = DataLoader(
        dataset,
        batch_size=4,           # small batch — just need one pass
        shuffle=False,
        collate_fn=dual_path_collate_fn,
    )

    # Grab exactly one batch — no need to iterate the full dataset
    batch = next(iter(loader))
    graphs, tokens, labels = batch
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    logger.info(f"Batch loaded — contracts in batch: {labels.shape[0]}")
    logger.info(f"  graphs.x:       {graphs.x.shape}")
    logger.info(f"  input_ids:      {input_ids.shape}")
    logger.info(f"  attention_mask: {attention_mask.shape}")
    logger.info(f"  labels:         {labels.shape}")

    # --- Model ---
    model = SentinelModel()
    model.eval()  # disable dropout for deterministic test output

    # --- Forward pass ---
    # torch.no_grad(): no gradients needed for testing — saves memory
    with torch.no_grad():
        scores = model(graphs, input_ids, attention_mask)

    # --- Verify output ---
    logger.info(f"Scores shape: {scores.shape}")
    logger.info(f"Scores values: {scores}")
    logger.info(f"Labels values: {labels}")

    # Shape check — one score per contract in batch
    assert scores.shape == labels.shape, (
        f"Shape mismatch: scores {scores.shape} vs labels {labels.shape}"
    )

    # Range check — sigmoid must constrain output to [0, 1]
    assert scores.min() >= 0.0 and scores.max() <= 1.0, (
        f"Scores out of [0,1] range: min={scores.min():.4f}, max={scores.max():.4f}"
    )

    logger.info("All assertions passed")
    logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    logger.info(f"Mean score: {scores.mean():.4f}")
    logger.success("End-to-end test PASSED ✅")


if __name__ == "__main__":
    run_end_to_end_test()