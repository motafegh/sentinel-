"""
Scans all graph files once and extracts hash → label mapping.
Output: ml/data/processed/label_index.csv

Run this ONCE. Result is reused by create_splits.py.
"""

import sys
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_label_index(graphs_dir: str, output_path: str) -> None:
    graphs_dir = Path(graphs_dir)
    output_path = Path(output_path)

    graph_files = sorted(graphs_dir.glob("*.pt"))
    print(f"Found {len(graph_files)} graph files. Extracting labels...")

    records = []

    for graph_path in tqdm(graph_files, desc="Scanning graphs"):
        try:
            # Load graph - we need the label inside
            graph_data = torch.load(graph_path, weights_only=True)

            # Extract hash (filename stem)
            hash_id = graph_path.stem

            # Extract label
            if hasattr(graph_data, 'y'):
                label = int(graph_data.y.item())
            else:
                print(f"Warning: no label for {hash_id}, skipping")
                continue

            records.append({
                'hash': hash_id,
                'label': label
            })

        except Exception as e:
            print(f"Error on {graph_path.name}: {e}")
            continue

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nLabel index created: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Safe (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
    print(f"Vulnerable (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")


if __name__ == "__main__":
    create_label_index(
        graphs_dir="ml/data/graphs",
        output_path="ml/data/processed/label_index.csv"
    )
