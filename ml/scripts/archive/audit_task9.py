"""Task 9: Full Feature Value Range Audit"""
import sys
import os
import random
import math
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data

sys.path.insert(0, '/home/motafeq/projects/sentinel')

GRAPHS_DIR = Path('/home/motafeq/projects/sentinel/ml/data/graphs')
SAMPLE_N = 500

all_files = list(GRAPHS_DIR.glob('*.pt'))
random.seed(42)
sample_files = random.sample(all_files, min(SAMPLE_N, len(all_files)))
print(f"Sampled {len(sample_files)} graph files")

# Safe globals for torch.load
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data import Data
safe_globals = [Data, GlobalStorage]

all_features = []
decl_features = []  # type_id raw < 8
cfg_features = []   # type_id raw >= 8

errors = []
for fpath in sample_files:
    try:
        g = torch.load(fpath, weights_only=False)
        x = g.x  # [N, 12]
        if x is None or x.shape[1] != 12:
            errors.append(f"BAD_SHAPE: {fpath.name}")
            continue
        x_np = x.cpu().numpy()
        all_features.append(x_np)

        # Split by node type
        raw_type = np.round(x_np[:, 0] * 12).astype(int)
        decl_mask = raw_type < 8
        cfg_mask = raw_type >= 8

        if decl_mask.any():
            decl_features.append(x_np[decl_mask])
        if cfg_mask.any():
            cfg_features.append(x_np[cfg_mask])
    except Exception as e:
        errors.append(f"ERROR {fpath.name}: {e}")

print(f"Load errors: {len(errors)}")
for e in errors[:5]:
    print(f"  {e}")

all_mat = np.vstack(all_features) if all_features else np.zeros((0,12))
decl_mat = np.vstack(decl_features) if decl_features else np.zeros((0,12))
cfg_mat = np.vstack(cfg_features) if cfg_features else np.zeros((0,12))

print(f"\nTotal nodes: {len(all_mat)}")
print(f"Declaration nodes: {len(decl_mat)}")
print(f"CFG nodes: {len(cfg_mat)}")

FEATURE_NAMES = [
    'type_id/12', 'visibility', 'uses_block_globals', 'view', 'payable',
    'complexity', 'loc', 'return_ignored', 'call_target_typed', 'in_unchecked',
    'has_loop', 'ext_call_count'
]

def stats_table(mat, label):
    if len(mat) == 0:
        print(f"\n{label}: NO DATA")
        return
    print(f"\n{'='*120}")
    print(f"FEATURE STATS — {label} (N={len(mat)} nodes)")
    print(f"{'='*120}")
    header = f"{'Feature':<22} {'min':>8} {'max':>8} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8} {'>1':>8} {'<-1':>8} {'NaN':>6} {'Inf':>6}"
    print(header)
    print('-'*120)

    findings = []
    for i, name in enumerate(FEATURE_NAMES):
        col = mat[:, i]
        nan_count = int(np.isnan(col).sum())
        inf_count = int(np.isinf(col).sum())
        clean = col[~np.isnan(col) & ~np.isinf(col)]

        if len(clean) == 0:
            print(f"{name:<22} {'N/A':>8}")
            continue

        mn = float(np.min(clean))
        mx = float(np.max(clean))
        p5 = float(np.percentile(clean, 5))
        p25 = float(np.percentile(clean, 25))
        p50 = float(np.percentile(clean, 50))
        p75 = float(np.percentile(clean, 75))
        p95 = float(np.percentile(clean, 95))
        above1 = int((clean > 1.0).sum())
        below_neg1 = int((clean < -1.0).sum())

        print(f"{name:<22} {mn:>8.4f} {mx:>8.4f} {p5:>8.4f} {p25:>8.4f} {p50:>8.4f} {p75:>8.4f} {p95:>8.4f} {above1:>8} {below_neg1:>8} {nan_count:>6} {inf_count:>6}")

        # Check expected ranges
        if i == 6 and mx > 1.0:  # loc
            findings.append(f"BUG: loc[6] max={mx:.4f} > 1.0 in {label}")
        if i == 5 and mx > 100:  # complexity
            findings.append(f"NOTE: complexity[5] max={mx:.4f} (unbounded, known)")
        if i == 1 and (mx > 2.0 or mn < 0.0):  # visibility
            findings.append(f"BUG: visibility[1] range=[{mn},{mx}] out of {{0,1,2}}")
        if i == 7:  # return_ignored
            unique_vals = set(np.round(clean, 4))
            bad = [v for v in unique_vals if v not in {-1.0, 0.0, 1.0}]
            if bad:
                findings.append(f"BUG: return_ignored[7] unexpected values: {bad[:5]}")
        if i == 8:  # call_target_typed
            unique_vals = set(np.round(clean, 4))
            bad = [v for v in unique_vals if v not in {-1.0, 0.0, 1.0}]
            if bad:
                findings.append(f"BUG: call_target_typed[8] unexpected values: {bad[:5]}")
        if i not in {5, 6, 1, 7, 8} and (mn < -0.001 or mx > 1.001):
            findings.append(f"BUG: {name}[{i}] range=[{mn:.4f},{mx:.4f}] out of [0,1]")
        if nan_count > 0:
            findings.append(f"BUG: {name}[{i}] has {nan_count} NaN values in {label}")
        if inf_count > 0:
            findings.append(f"BUG: {name}[{i}] has {inf_count} Inf values in {label}")

    if findings:
        print(f"\n*** FINDINGS for {label} ***")
        for f in findings:
            print(f"  {f}")
    else:
        print(f"\n  No range violations found in {label}")

stats_table(all_mat, "ALL NODES")
stats_table(decl_mat, "DECLARATION NODES (type_id < 8)")
stats_table(cfg_mat, "CFG NODES (type_id >= 8)")
