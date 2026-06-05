"""
val_finding4_timestamp_size.py — Validate Timestamp size shortcut in TRAINING split.

If size difference exists in training data too, model may have learned size as proxy.
"""

import sys, os, json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

print("=== FINDING 4 VALIDATION: Timestamp Size Shortcut (Train vs Val) ===\n")

print("Loading cache...")
with open('ml/data/cached_dataset_v10.pkl', 'rb') as f:
    cache = pickle.load(f)
print(f"Cache loaded: {len(cache):,} entries")

df = pd.read_csv('ml/data/processed/multilabel_index.csv')
train_idx = np.load('ml/data/splits/v9_deduped/train_indices.npy')
val_idx = np.load('ml/data/splits/v9_deduped/val_indices.npy')

stems_all = list(cache.keys())
print(f"Total stems: {len(stems_all):,}")
print(f"Train indices: {len(train_idx):,}, Val indices: {len(val_idx):,}\n")

# Build fast lookup
stem_to_row = {row['md5_stem']: row for _, row in df.iterrows()}

results = {}
for split_name, idx in [('train', train_idx), ('val', val_idx)]:
    stems_split = [stems_all[i] for i in idx if i < len(stems_all)]
    pos_sizes, neg_sizes = [], []
    skipped = 0
    for s in stems_split[:5000]:
        if s not in cache:
            skipped += 1
            continue
        if s not in stem_to_row:
            skipped += 1
            continue
        g = cache[s][0]
        n = g.x.shape[0]
        row = stem_to_row[s]
        if row['Timestamp'] == 1:
            pos_sizes.append(n)
        else:
            neg_sizes.append(n)

    pos_mean = np.mean(pos_sizes) if pos_sizes else 0
    neg_mean = np.mean(neg_sizes) if neg_sizes else 0
    ratio = pos_mean / neg_mean if neg_mean > 0 else 0

    # Cohen's d
    pos_arr = np.array(pos_sizes)
    neg_arr = np.array(neg_sizes)
    pooled_std = np.sqrt((pos_arr.std()**2 + neg_arr.std()**2) / 2) if len(pos_sizes) > 1 else 1
    cohens_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0

    print(f"{split_name}:")
    print(f"  Timestamp+ : n={len(pos_sizes):,}, mean_nodes={pos_mean:.1f}, std={np.std(pos_sizes):.1f}")
    print(f"  Timestamp- : n={len(neg_sizes):,}, mean_nodes={neg_mean:.1f}, std={np.std(neg_sizes):.1f}")
    print(f"  Ratio (pos/neg): {ratio:.2f}x")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Skipped: {skipped}")
    print()

    results[split_name] = {
        "n_pos": len(pos_sizes),
        "n_neg": len(neg_sizes),
        "pos_mean_nodes": float(pos_mean),
        "neg_mean_nodes": float(neg_mean),
        "ratio": float(ratio),
        "cohens_d": float(cohens_d),
        "pos_median": float(np.median(pos_sizes)) if pos_sizes else 0,
        "neg_median": float(np.median(neg_sizes)) if neg_sizes else 0,
    }

# Also check for ALL classes to see if size is a general artifact
print("\n=== SIZE ANALYSIS FOR ALL CLASSES (val split, first 3000) ===")
val_stems = [stems_all[i] for i in val_idx if i < len(stems_all)][:3000]
class_names = ['CallToUnknown','DenialOfService','ExternalBug','GasException','IntegerUO',
               'MishandledException','Reentrancy','Timestamp','TransactionOrderDependence','UnusedReturn']

class_ratios = {}
for cls in class_names:
    if cls not in df.columns:
        continue
    pos_sizes, neg_sizes = [], []
    for s in val_stems:
        if s not in cache or s not in stem_to_row:
            continue
        n = cache[s][0].x.shape[0]
        if stem_to_row[s][cls] == 1:
            pos_sizes.append(n)
        else:
            neg_sizes.append(n)
    if pos_sizes and neg_sizes:
        ratio = np.mean(pos_sizes) / np.mean(neg_sizes)
        class_ratios[cls] = ratio
        print(f"  {cls:<28}: ratio={ratio:.2f}x (pos={np.mean(pos_sizes):.0f}, neg={np.mean(neg_sizes):.0f}, n_pos={len(pos_sizes)})")

# Verdict
ts_train_d = results.get('train', {}).get('cohens_d', 0)
ts_val_d = results.get('val', {}).get('cohens_d', 0)
print(f"\n=== VERDICT ===")
print(f"Timestamp Cohen's d: train={ts_train_d:.3f}, val={ts_val_d:.3f}")
if ts_train_d > 1.0:
    print("CONFIRMED: Large size difference exists in TRAINING data (Cohen's d > 1.0).")
    print("Model could have learned contract size as a proxy for Timestamp vulnerability.")
    print("This makes the shortcut REAL — model was exposed to this during training.")
elif ts_train_d > 0.5:
    print("PARTIALLY CONFIRMED: Medium size difference in training (Cohen's d > 0.5).")
    print("Some shortcut learning is plausible.")
else:
    print("INCONCLUSIVE: Small size difference in training data.")

out = {
    "split_results": results,
    "class_size_ratios_val": {k: float(v) for k, v in class_ratios.items()},
    "verdict": "CONFIRMED" if ts_train_d > 1.0 else ("PARTIALLY" if ts_train_d > 0.5 else "INCONCLUSIVE"),
}
out_path = Path("ml/logs/interpretability/val_finding4_timestamp_size.json")
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to {out_path}")
