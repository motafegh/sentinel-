"""
Quick test script for DualPathDataset.
Run this to verify your data loads correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.src.datasets.dual_path_dataset import DualPathDataset
import torch
from torch_geometric.data import Batch

def test_basic_loading():
    """Test 1: Can we load samples?"""
    print("=" * 60)
    print("TEST 1: Basic Loading")
    print("=" * 60)
    
    dataset = DualPathDataset(
        graphs_dir="ml/data/graphs",
        tokens_dir="ml/data/tokens",
        validate=True  # Will check first sample
    )
    
    print(f"✓ Dataset created")
    print(f"✓ Total samples: {len(dataset)}")
    
    # Load first sample
    graph, tokens, label = dataset[0]
    
    print(f"✓ First sample loaded:")
    print(f"  - Graph nodes: {graph.x.shape[0]}")
    print(f"  - Graph edges: {graph.edge_index.shape[1]}")
    print(f"  - Token length: {tokens['input_ids'].shape[0]}")
    print(f"  - Label: {label.item()}")
    
    return dataset

def test_zero_edge_graphs(dataset):
    """Test 2: Can we handle zero-edge graphs?"""
    print("\n" + "=" * 60)
    print("TEST 2: Zero-Edge Graphs")
    print("=" * 60)
    
    zero_edge_count = 0
    
    # Check first 100 samples
    for i in range(min(100, len(dataset))):
        graph, _, _ = dataset[i]
        if graph.edge_index.shape[1] == 0:
            zero_edge_count += 1
    
    print(f"✓ Zero-edge graphs in first 100: {zero_edge_count}")
    print(f"  (Expected ~73 based on your 72.8% statistic)")

def test_batching(dataset):
    """Test 3: Can PyG batch our graphs?"""
    print("\n" + "=" * 60)
    print("TEST 3: Batching with PyG")
    print("=" * 60)
    
    # Load 4 samples manually
    samples = [dataset[i] for i in range(4)]
    
    # Separate graphs, tokens, labels
    graphs = [s[0] for s in samples]
    tokens = [s[1] for s in samples]
    labels = [s[2] for s in samples]
    
    # Batch graphs with PyG
    batched_graphs = Batch.from_data_list(graphs)
    
    # Batch tokens (stack into [B, 512])
    batched_input_ids = torch.stack([t['input_ids'] for t in tokens])
    batched_attention_mask = torch.stack([t['attention_mask'] for t in tokens])
    
    # Batch labels
    batched_labels = torch.stack(labels)
    
    print(f"✓ Batched 4 samples:")
    print(f"  - Graph batch nodes: {batched_graphs.x.shape}")
    print(f"  - Graph batch edges: {batched_graphs.edge_index.shape}")
    print(f"  - Token batch: {batched_input_ids.shape}")
    print(f"  - Labels: {batched_labels.shape}")

def test_label_distribution(dataset):
    """Test 4: Check label balance"""
    print("\n" + "=" * 60)
    print("TEST 4: Label Distribution")
    print("=" * 60)
    
    # Count labels in first 1000 samples
    sample_size = min(1000, len(dataset))
    labels = [dataset[i][2].item() for i in range(sample_size)]
    
    safe_count = sum(1 for l in labels if l == 0)
    vuln_count = sum(1 for l in labels if l == 1)
    
    print(f"✓ Label distribution (first {sample_size} samples):")
    print(f"  - Safe (0): {safe_count} ({safe_count/sample_size*100:.1f}%)")
    print(f"  - Vulnerable (1): {vuln_count} ({vuln_count/sample_size*100:.1f}%)")
    print(f"  Expected: ~35.7% safe / ~64.3% vulnerable")

def main():
    dataset = test_basic_loading()
    test_zero_edge_graphs(dataset)
    test_batching(dataset)
    test_label_distribution(dataset)
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create train/val/test split indices")
    print("2. Instantiate DualPathDataset with splits")
    print("3. Create DataLoader with custom collate_fn")

if __name__ == "__main__":
    main()
