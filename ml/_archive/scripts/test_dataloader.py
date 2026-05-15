# ml/scripts/test_dataloader.py
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # adds ~/projects/sentinel
import torch
from torch.utils.data import DataLoader
from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn

# Load splits
train_indices = np.load("ml/data/splits/train_indices.npy")

# Build dataset (small subset — faster for testing)
dataset = DualPathDataset(
    graphs_dir="ml/data/graphs",
    tokens_dir="ml/data/tokens",
    indices=train_indices[:200],  # only first 200 samples
    validate=False                # skip validation, we're testing manually
)

# Build DataLoader with our custom collate
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=dual_path_collate_fn
)

# Grab exactly one batch and inspect shapes
graphs, tokens, labels = next(iter(loader))

print("=== BATCH SHAPE CHECK ===")
print(f"graphs.x shape:         {graphs.x.shape}")        # [total_nodes, 8]
print(f"graphs.edge_index:      {graphs.edge_index.shape}")# [2, total_edges]
print(f"graphs.batch shape:     {graphs.batch.shape}")     # [total_nodes]
print(f"input_ids shape:        {tokens['input_ids'].shape}")       # [4, 512]
print(f"attention_mask shape:   {tokens['attention_mask'].shape}")  # [4, 512]
print(f"labels shape:           {labels.shape}")           # [4]
print(f"labels values:          {labels}")                 # e.g. tensor([1,0,1,1])
print(f"unique batch ids:       {graphs.batch.unique()}")  # tensor([0,1,2,3])
