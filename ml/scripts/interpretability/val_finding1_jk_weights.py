"""
val_finding1_jk_weights.py — Validate JK weight phase ordering and entropy claim.

Checks:
1. Phase order in _live list: [phase1, phase2, phase3] → index 0=Ph1, 1=Ph2, 2=Ph3
2. last_weights buffer matches what we observe after a fresh forward pass
3. Entropy value is genuine (near log(3)=1.099) vs artefact
4. Check attn linear layer weights for bias towards any phase
"""

import torch
import sys
import os
import json
from pathlib import Path

os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.scripts.interpretability.utils import load_model, load_val_split, CLASS_NAMES
import numpy as np

device = torch.device('cpu')

print("=== FINDING 1 VALIDATION: JK Weight Phase Ordering ===\n")

# Load model
model = load_model(
    'ml/checkpoints/sentinel_best.pt',
    device='cpu'
)
model.eval()

# 1. Read the stored buffer (populated during last training forward)
buf = model.gnn.jk.last_weights.numpy()
buf_stds = model.gnn.jk.last_weight_stds.numpy()
print(f"[Buffer] last_weights (from checkpoint): {buf}")
print(f"[Buffer] last_weight_stds: {buf_stds}")
print(f"  → Phase1(idx0)={buf[0]:.4f}, Phase2(idx1)={buf[1]:.4f}, Phase3(idx2)={buf[2]:.4f}")
print(f"  → log(3) = {np.log(3):.4f}  max_entropy_weight = {1/3:.4f}")

# 2. Read attn linear weights (should be near-uniform if entropy is near-max)
attn_w = model.gnn.jk.attn.weight.data.numpy()
print(f"\n[attn.weight] shape={attn_w.shape}: {attn_w.flatten()[:10]} ...")

# 3. Fresh forward pass — do the weights update?
print("\n=== Running fresh forward pass on 5 val contracts ===")
stems, df_split, cache = load_val_split(
    'ml/data/cached_dataset_v9.pkl',
    'ml/data/processed/multilabel_index.csv',
    'ml/data/splits/v9_deduped'
)

from torch_geometric.data import Batch

fresh_weights = []
with torch.no_grad():
    for stem in stems[:10]:
        if stem not in cache:
            continue
        g, tok = cache[stem]
        batch = Batch.from_data_list([g]).to(device)
        input_ids = tok['input_ids'].unsqueeze(0).to(device)
        attn_mask = tok['attention_mask'].unsqueeze(0).to(device)
        _ = model(batch, input_ids, attn_mask, return_aux=True)
        w = model.gnn.jk.last_weights.numpy().copy()
        fresh_weights.append(w)
        if len(fresh_weights) >= 5:
            break

print("\nPer-contract fresh JK weights [Ph1, Ph2, Ph3]:")
for i, w in enumerate(fresh_weights):
    print(f"  Contract {i}: [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]")

fresh_arr = np.array(fresh_weights)
mean_fresh = fresh_arr.mean(0)
print(f"\nMean fresh weights: Ph1={mean_fresh[0]:.4f}, Ph2={mean_fresh[1]:.4f}, Ph3={mean_fresh[2]:.4f}")
print(f"Range of Phase 2 weights: [{fresh_arr[:,1].min():.4f}, {fresh_arr[:,1].max():.4f}]")

# 4. Check last_node_weights (per-node, populated in eval mode)
print("\n=== last_node_weights (per-node, populated in eval mode) ===")
if model.gnn.jk.last_node_weights is not None:
    lnw = model.gnn.jk.last_node_weights.numpy()  # [N, 3]
    print(f"Shape: {lnw.shape}")
    print(f"Mean per phase: {lnw.mean(0)}")
    print(f"Std per phase: {lnw.std(0)}")
    print(f"Min per phase: {lnw.min(0)}")
    print(f"Max per phase: {lnw.max(0)}")
else:
    print("last_node_weights is None (model was in training mode during checkpoint save?)")

# 5. Entropy check
if len(fresh_weights) > 0:
    # Compute entropy from mean fresh weights
    w = mean_fresh
    H = -(w * np.log(w + 1e-8)).sum()
    print(f"\nEntropy of mean fresh weights: H={H:.4f} (max=log(3)={np.log(3):.4f})")
    print(f"Difference from max entropy: {np.log(3) - H:.4f}")

# 6. Assess significance of Phase 2 being lowest
diff_p2_p3 = mean_fresh[2] - mean_fresh[1]
diff_p2_p1 = mean_fresh[0] - mean_fresh[1]
print(f"\n=== Significance Assessment ===")
print(f"Phase3 - Phase2 = {diff_p2_p3:.4f}")
print(f"Phase1 - Phase2 = {diff_p2_p1:.4f}")
print(f"All differences < 0.025? {'YES' if max(diff_p2_p3, diff_p2_p1) < 0.025 else 'NO'}")

# Summary
result = {
    "buffer_weights": buf.tolist(),
    "fresh_mean_weights": mean_fresh.tolist(),
    "phase_order": ["phase1_idx0", "phase2_idx1", "phase3_idx2"],
    "phase2_is_lowest_in_buffer": bool(buf[1] < buf[0] and buf[1] < buf[2]),
    "phase2_is_lowest_in_fresh": bool(mean_fresh[1] < mean_fresh[0] and mean_fresh[1] < mean_fresh[2]),
    "max_diff_from_uniform": float(max(abs(mean_fresh - 1/3))),
    "entropy_from_fresh": float(H) if len(fresh_weights) > 0 else None,
    "max_entropy": float(np.log(3)),
}
print("\n=== SUMMARY ===")
print(json.dumps(result, indent=2))

out_path = Path("ml/logs/interpretability/val_finding1_jk_weights.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {out_path}")
