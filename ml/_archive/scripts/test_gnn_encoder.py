# ml/scripts/test_gnn_encoder.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from ml.src.models.gnn_encoder import GNNEncoder

# Real data — first 200 training samples
train_indices = np.load("ml/data/splits/train_indices.npy")
dataset = DualPathDataset(
    graphs_dir="ml/data/graphs",
    tokens_dir="ml/data/tokens",
    indices=train_indices[:200],
    validate=False
)

loader = DataLoader(dataset, batch_size=32,
                    shuffle=False, collate_fn=dual_path_collate_fn)

# One real batch
graphs, tokens, labels = next(iter(loader))

# Run through encoder
encoder = GNNEncoder(dropout=0.2)
encoder.eval()

with torch.no_grad():  # no gradients needed for shape testing
    graph_embeddings = encoder(graphs.x, graphs.edge_index, graphs.batch)

print("=== END-TO-END PIPELINE CHECK ===")
print(f"Batch size:             {labels.shape[0]}")        # 32
print(f"Graph embedding shape:  {graph_embeddings.shape}") # [32, 64]
print(f"Token input_ids shape:  {tokens['input_ids'].shape}") # [32, 512]
print(f"Labels shape:           {labels.shape}")           # [32]
print(f"Labels distribution:    {labels.sum().item()} vulnerable / "
      f"{(labels==0).sum().item()} safe")

