"""Task 13: Graph Structural Integrity"""
import sys
import random
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('/home/motafeq/projects/sentinel')
GRAPHS_DIR = PROJECT_ROOT / 'ml/data/graphs'

all_files = list(GRAPHS_DIR.glob('*.pt'))
random.seed(77)
sample_files = random.sample(all_files, min(200, len(all_files)))
print(f"Sampled {len(sample_files)} graph files")

violations = {
    'x_shape_dtype': 0,       # check 1
    'edge_index_shape_dtype': 0,  # check 2
    'edge_attr_1d_dtype': 0,   # check 3
    'edge_index_oob': 0,       # check 4
    'edge_attr_bad_type': 0,   # check 5 (type 7 is runtime only, not in stored graphs)
    'x_nan_or_inf': 0,         # check 6
    'cf_self_loops': 0,        # check 7
    'no_edges': 0,             # check 8
    'contains_wrong_src': 0,   # check 9
}

cf_self_loop_examples = []
errors = []

for fpath in sample_files:
    try:
        g = torch.load(fpath, weights_only=False)

        x = g.x
        edge_index = g.edge_index
        edge_attr = getattr(g, 'edge_attr', None)
        num_nodes = g.num_nodes

        # Check 1: x.shape[1]==12, dtype float32
        if x is None or x.shape[1] != 12 or x.dtype != torch.float32:
            violations['x_shape_dtype'] += 1

        # Check 2: edge_index shape and dtype
        if edge_index is None or edge_index.shape[0] != 2 or edge_index.dtype != torch.int64:
            violations['edge_index_shape_dtype'] += 1

        if edge_index is None:
            violations['no_edges'] += 1
            continue

        # Check 3: edge_attr is 1-D int64
        if edge_attr is None:
            violations['edge_attr_1d_dtype'] += 1
        else:
            E = edge_index.shape[1]
            if edge_attr.shape != torch.Size([E]) or edge_attr.dtype != torch.int64:
                violations['edge_attr_1d_dtype'] += 1

        # Check 4: edge_index values in [0, num_nodes)
        if num_nodes is not None and num_nodes > 0:
            if edge_index.max().item() >= num_nodes or edge_index.min().item() < 0:
                violations['edge_index_oob'] += 1

        # Check 5: edge_attr values in {0,1,2,3,4,5,6} (7 is runtime-only)
        if edge_attr is not None:
            unique_types = edge_attr.unique().tolist()
            bad_types = [t for t in unique_types if t not in {0, 1, 2, 3, 4, 5, 6}]
            if bad_types:
                violations['edge_attr_bad_type'] += 1

        # Check 6: NaN or Inf in x
        if x is not None and (torch.isnan(x).any() or torch.isinf(x).any()):
            violations['x_nan_or_inf'] += 1

        # Check 7: No self-loops in CF edges (type 6)
        if edge_attr is not None:
            cf_mask = (edge_attr == 6)
            if cf_mask.any():
                cf_edges = edge_index[:, cf_mask]
                self_loops = (cf_edges[0] == cf_edges[1]).sum().item()
                if self_loops > 0:
                    violations['cf_self_loops'] += 1
                    cf_self_loop_examples.append(f"{fpath.name}: {self_loops} CF self-loops")

        # Check 8: At least 1 edge
        if edge_index.shape[1] == 0:
            violations['no_edges'] += 1

        # Check 9: CONTAINS edges (type 5): src should not be CFG node (raw_type < 8)
        if edge_attr is not None and x is not None:
            contains_mask = (edge_attr == 5)
            if contains_mask.any():
                src_nodes = edge_index[0, contains_mask]
                raw_src_types = (x[src_nodes, 0] * 12).round().long()
                cfg_src = (raw_src_types >= 8).sum().item()
                if cfg_src > 0:
                    violations['contains_wrong_src'] += 1

    except Exception as e:
        errors.append(f"{fpath.name}: {e}")

print(f"\nLoad errors: {len(errors)}")
for e in errors[:5]:
    print(f"  {e}")

print(f"\n{'='*70}")
print("GRAPH STRUCTURAL INTEGRITY (N=200)")
print(f"{'='*70}")
print(f"{'Check':<45} {'Violations':>12}")
print('-'*60)
check_labels = [
    ('x_shape_dtype', 'x.shape[1]==12 & float32'),
    ('edge_index_shape_dtype', 'edge_index shape[0]==2 & int64'),
    ('edge_attr_1d_dtype', 'edge_attr 1-D int64'),
    ('edge_index_oob', 'edge_index in [0, num_nodes)'),
    ('edge_attr_bad_type', 'edge_attr in {0..6} (no type-7)'),
    ('x_nan_or_inf', 'no NaN/Inf in x'),
    ('cf_self_loops', 'no self-loops in CF edges'),
    ('no_edges', 'at least 1 edge'),
    ('contains_wrong_src', 'CONTAINS src is not CFG node'),
]
all_ok = True
for key, label in check_labels:
    v = violations[key]
    status = "BUG" if v > 0 else "OK"
    if v > 0:
        all_ok = False
    print(f"{label:<45} {v:>12} {status}")

if cf_self_loop_examples:
    print(f"\nCF self-loop examples:")
    for e in cf_self_loop_examples[:5]:
        print(f"  {e}")

if all_ok:
    print("\nCONFIRMED: All structural checks passed.")
