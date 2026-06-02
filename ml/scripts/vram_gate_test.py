"""
vram_gate_test.py — Gate 5.3: VRAM worst-case test for max_nodes=2048 (IMP-D1)

Builds a full SENTINEL model and runs a realistic worst-case training step
(forward + backward + optimizer.step + AMP scaler) on synthetic graph data.

Must pass BEFORE raising fusion_max_nodes to 2048 in Run 5. Decision thresholds:
  - PASS  : peak < 7 500 MB → proceed with max_nodes=2048
  - WARN  : peak 7 500–8 000 MB → reduce batch_size to 8
  - FAIL  : peak > 8 000 MB → fall back to max_nodes=1536 or reduce batch_size

Usage:
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/vram_gate_test.py
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/vram_gate_test.py --max-nodes 1536 --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Data, Batch

from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.trainer import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 5.3 VRAM worst-case test")
    p.add_argument("--max-nodes",   type=int, default=2048,
                   help="fusion_max_nodes to test (default 2048 = IMP-D1 target)")
    p.add_argument("--batch-size",  type=int, default=16,
                   help="Synthetic batch size (test at 16 and 32)")
    p.add_argument("--nodes-per-graph", type=int, default=None,
                   help="Nodes per synthetic graph (default: max_nodes, worst case)")
    p.add_argument("--seq-len",     type=int, default=512,
                   help="Token sequence length per window (default 512)")
    p.add_argument("--windows",     type=int, default=4,
                   help="Token windows per contract (default 4)")
    p.add_argument("--num-steps",   type=int, default=3,
                   help="Number of steps to run (peak usually hit at step 1)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_synthetic_batch(
    batch_size: int,
    nodes_per_graph: int,
    node_feature_dim: int,
    num_edge_types: int,
    seq_len: int,
    windows: int,
    device: str,
) -> tuple:
    """Build a synthetic PyG Batch + token tensors for worst-case VRAM test."""
    graphs_list = []
    for _ in range(batch_size):
        n = nodes_per_graph
        # Sparse chain graph (minimum edges, max nodes — pure node VRAM cost)
        edge_index = torch.stack([
            torch.arange(n - 1),
            torch.arange(1, n),
        ], dim=0)
        edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.long)
        x = torch.randn(n, node_feature_dim)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs_list.append(g)

    batch = Batch.from_data_list(graphs_list).to(device)

    input_ids      = torch.randint(0, 50265, (batch_size, windows, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, windows, seq_len, dtype=torch.long).to(device)
    labels         = torch.zeros(batch_size, 10, dtype=torch.float).to(device)
    labels[:, 0]   = 1.0  # at least one positive per batch

    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    return batch, tokens, labels


def main() -> None:
    args = parse_args()
    device = args.device
    nodes_per_graph = args.nodes_per_graph or args.max_nodes

    print(f"\n{'='*60}")
    print(f"Gate 5.3 VRAM Test")
    print(f"  device          : {device}")
    print(f"  max_nodes       : {args.max_nodes}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  nodes_per_graph : {nodes_per_graph} (worst case = max_nodes)")
    print(f"  windows×seq_len : {args.windows}×{args.seq_len}")
    print(f"{'='*60}\n")

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    cfg = TrainConfig(
        fusion_max_nodes=args.max_nodes,
        use_amp=True,
        gnn_layers=8,
        gnn_hidden_dim=256,
        gnn_heads=8,
        gnn_edge_emb_dim=64,
        use_compile=False,
        lora_r=16,
        lora_alpha=32,
    )

    model = SentinelModel(
        num_classes=10,
        gnn_hidden_dim=cfg.gnn_hidden_dim,
        gnn_layers=cfg.gnn_layers,
        gnn_heads=cfg.gnn_heads,
        gnn_dropout=cfg.gnn_dropout,
        gnn_edge_emb_dim=cfg.gnn_edge_emb_dim,
        use_edge_attr=cfg.use_edge_attr,
        gnn_use_jk=cfg.gnn_use_jk,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        fusion_max_nodes=cfg.fusion_max_nodes,
    ).to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-2,
        fused=(device == "cuda"),
    )
    scaler = torch.amp.GradScaler(device, enabled=cfg.use_amp)
    loss_fn = nn.BCEWithLogitsLoss()

    from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES

    peak_mb_list = []
    for step in range(args.num_steps):
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        graphs, tokens, labels = make_synthetic_batch(
            batch_size=args.batch_size,
            nodes_per_graph=nodes_per_graph,
            node_feature_dim=NODE_FEATURE_DIM,
            num_edge_types=NUM_EDGE_TYPES,
            seq_len=args.seq_len,
            windows=args.windows,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=cfg.use_amp):
            logits = model(graphs, tokens["input_ids"], tokens["attention_mask"])
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if device.startswith("cuda"):
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            peak_mb = 0.0

        print(f"  step {step+1}: loss={loss.item():.4f}  peak_vram={peak_mb:.0f} MB")
        peak_mb_list.append(peak_mb)

    peak = max(peak_mb_list) if peak_mb_list else 0.0

    print(f"\n{'='*60}")
    print(f"RESULT  peak VRAM = {peak:.0f} MB  ({peak/1024:.2f} GB)")
    if peak == 0.0:
        print("STATUS  N/A (CPU run)")
    elif peak < 7500:
        print(f"STATUS  PASS — proceed with fusion_max_nodes={args.max_nodes}")
    elif peak < 8000:
        print(f"STATUS  WARN — peak near limit; reduce batch_size to 8 or max_nodes to 1536")
    else:
        print(f"STATUS  FAIL — exceeds 8 GB; use max_nodes=1536 or batch_size=8")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
