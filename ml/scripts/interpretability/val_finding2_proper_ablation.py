"""
val_finding2_proper_ablation.py — Proper edge ablation by removing edges from edge_index.

The original exp_l2 zeroed the embedding vector but left edges structurally intact.
This script removes edges entirely, testing whether the structural signal (not just
embedding signal) from CFG/ICFG edges matters for predictions.
"""

import torch
import sys
import os
import json
import numpy as np
from pathlib import Path

os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch_geometric.data as pyg_data
from torch_geometric.data import Batch

from ml.scripts.interpretability.utils import load_model, load_val_split, CLASS_NAMES
from ml.src.preprocessing.graph_schema import EDGE_TYPES

device = torch.device('cpu')

print("=== FINDING 2 VALIDATION: Proper Edge Ablation (remove edges from edge_index) ===\n")

model = load_model('ml/checkpoints/sentinel_best.pt', device='cpu')
model.eval()

stems, df_split, cache = load_val_split(
    'ml/data/cached_dataset_v9.pkl',
    'ml/data/processed/multilabel_index.csv',
    'ml/data/splits/v9_deduped'
)

# Edge type constants
CONTROL_FLOW = EDGE_TYPES['CONTROL_FLOW']   # 6
CALL_ENTRY   = EDGE_TYPES['CALL_ENTRY']     # 8
RETURN_TO    = EDGE_TYPES['RETURN_TO']      # 9
DEF_USE      = EDGE_TYPES['DEF_USE']        # 10

print(f"Edge types: CONTROL_FLOW={CONTROL_FLOW}, CALL_ENTRY={CALL_ENTRY}, RETURN_TO={RETURN_TO}, DEF_USE={DEF_USE}")

# Get reentrancy-positive stems
reentrancy_idx = CLASS_NAMES.index('Reentrancy')
integer_uo_idx = CLASS_NAMES.index('IntegerUO')

stem_to_labels = {
    row['md5_stem']: row[CLASS_NAMES].values.tolist()
    for _, row in df_split.iterrows()
}

reentrancy_stems = [
    s for s in stems
    if s in cache and s in stem_to_labels and stem_to_labels[s][reentrancy_idx] == 1
][:30]

integer_uo_stems = [
    s for s in stems
    if s in cache and s in stem_to_labels and stem_to_labels[s][integer_uo_idx] == 1
][:30]

print(f"Reentrancy-positive stems: {len(reentrancy_stems)}")
print(f"IntegerUO-positive stems: {len(integer_uo_stems)}")

def get_probs(model, stem, cache, ablate_types, device):
    """Get sigmoid probabilities after removing specified edge types from edge_index."""
    g, tok = cache[stem]

    if ablate_types:
        keep = torch.ones(g.edge_attr.shape[0], dtype=torch.bool)
        for et in ablate_types:
            keep &= (g.edge_attr != et)
        g_abl = pyg_data.Data(
            x=g.x,
            edge_index=g.edge_index[:, keep],
            edge_attr=g.edge_attr[keep],
        )
    else:
        g_abl = g

    # Count edges before and after ablation
    n_orig = g.edge_attr.shape[0]
    n_kept = g_abl.edge_attr.shape[0] if ablate_types else n_orig

    batch = Batch.from_data_list([g_abl]).to(device)
    input_ids = tok['input_ids'].unsqueeze(0).to(device)
    attn_mask = tok['attention_mask'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(batch, input_ids, attn_mask)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.sigmoid(logits[0]).cpu().numpy()

    return probs, n_orig, n_kept


ablation_configs = [
    ('baseline',        []),
    ('no_cf',           [CONTROL_FLOW]),
    ('no_call_entry',   [CALL_ENTRY]),
    ('no_return_to',    [RETURN_TO]),
    ('no_icfg',         [CALL_ENTRY, RETURN_TO]),
    ('no_def_use',      [DEF_USE]),
    ('no_phase2_all',   [CONTROL_FLOW, CALL_ENTRY, RETURN_TO, DEF_USE]),
]

print("\n=== REENTRANCY POSITIVE CONTRACTS ===")
results_reentrant = {}
for ablate_name, ablate_types in ablation_configs:
    scores = []
    edge_counts = []
    for stem in reentrancy_stems:
        try:
            probs, n_orig, n_kept = get_probs(model, stem, cache, ablate_types, device)
            scores.append(probs[reentrancy_idx])
            edge_counts.append((n_orig, n_kept))
        except Exception as e:
            continue

    mean_score = np.mean(scores)
    results_reentrant[ablate_name] = mean_score
    n_orig_mean = np.mean([e[0] for e in edge_counts]) if edge_counts else 0
    n_kept_mean = np.mean([e[1] for e in edge_counts]) if edge_counts else 0
    print(f"  {ablate_name:<22}: Reentrancy prob = {mean_score:.4f}  "
          f"(edges: {n_orig_mean:.0f} → {n_kept_mean:.0f})")

print("\nDrops from baseline (REENTRANCY):")
for name, val in results_reentrant.items():
    if name != 'baseline':
        drop = results_reentrant['baseline'] - val
        print(f"  {name:<22}: drop = {drop:.4f} ({'MEANINGFUL (>0.03)' if drop > 0.03 else 'negligible'})")

print("\n=== INTEGERU POSITIVE CONTRACTS ===")
results_iu = {}
for ablate_name, ablate_types in ablation_configs:
    scores = []
    for stem in integer_uo_stems:
        try:
            probs, _, _ = get_probs(model, stem, cache, ablate_types, device)
            scores.append(probs[integer_uo_idx])
        except Exception as e:
            continue
    mean_score = np.mean(scores)
    results_iu[ablate_name] = mean_score
    print(f"  {ablate_name:<22}: IntegerUO prob = {mean_score:.4f}")

print("\nDrops from baseline (INTEGERUO):")
for name, val in results_iu.items():
    if name != 'baseline':
        drop = results_iu['baseline'] - val
        print(f"  {name:<22}: drop = {drop:.4f} ({'MEANINGFUL (>0.02)' if drop > 0.02 else 'negligible'})")

# Compare to original embedding-zero method
print("\n=== COMPARISON WITH ORIGINAL EMBEDDING-ZERO METHOD ===")
import json as _json
try:
    with open('ml/logs/interpretability/exp_l2_edge_ablation.json/exp_l2_ablation_delta.json') as f:
        orig = _json.load(f)
    print(f"Original embedding-zero results:")
    for chk in orig['checks']:
        if chk['result'] != 'INFO':
            print(f"  {chk['description']}: delta={chk['delta']:.8f}")
    print(f"\nProper edge-removal results:")
    print(f"  no_cf drop on Reentrancy:        {results_reentrant['baseline'] - results_reentrant['no_cf']:.8f}")
    print(f"  no_call_entry drop on Reentrancy: {results_reentrant['baseline'] - results_reentrant['no_call_entry']:.8f}")
    print(f"  no_def_use drop on IntegerUO:     {results_iu['baseline'] - results_iu['no_def_use']:.8f}")
except Exception as e:
    print(f"Could not load original results: {e}")

# Conclusion
print("\n=== VERDICT ===")
phase2_drop_r = results_reentrant['baseline'] - results_reentrant['no_phase2_all']
cf_drop_r = results_reentrant['baseline'] - results_reentrant['no_cf']
icfg_drop_r = results_reentrant['baseline'] - results_reentrant['no_icfg']

print(f"Phase2-all drop on Reentrancy:  {phase2_drop_r:.4f}")
print(f"CF drop on Reentrancy:           {cf_drop_r:.4f}")
print(f"ICFG drop on Reentrancy:         {icfg_drop_r:.4f}")

if max(abs(phase2_drop_r), abs(cf_drop_r), abs(icfg_drop_r)) < 0.005:
    print("\nVERDICT: CONFIRMED — Model genuinely ignores CFG/ICFG edges structurally.")
    print("The embedding-zero finding was NOT an artifact of the measurement method.")
elif max(abs(phase2_drop_r), abs(cf_drop_r), abs(icfg_drop_r)) < 0.03:
    print("\nVERDICT: PARTIALLY CONFIRMED — Structural CFG removal has small but real effect.")
    print("The embedding-zero finding understated the true effect (but effect is still small).")
else:
    print("\nVERDICT: ARTIFACT — Edge structure DOES matter. Embedding-zero method was flawed.")
    print("The original finding was wrong — CFG edges contribute structurally.")

out = {
    "method": "proper_edge_removal_from_edge_index",
    "n_reentrancy_stems": len(reentrancy_stems),
    "n_integeruo_stems": len(integer_uo_stems),
    "reentrancy_probs": {k: float(v) for k, v in results_reentrant.items()},
    "reentrancy_drops": {k: float(results_reentrant['baseline'] - v) for k, v in results_reentrant.items() if k != 'baseline'},
    "integeruo_probs": {k: float(v) for k, v in results_iu.items()},
    "integeruo_drops": {k: float(results_iu['baseline'] - v) for k, v in results_iu.items() if k != 'baseline'},
}

out_path = Path("ml/logs/interpretability/val_finding2_proper_ablation.json")
with open(out_path, 'w') as f:
    _json.dump(out, f, indent=2)
print(f"\nResults saved to {out_path}")
