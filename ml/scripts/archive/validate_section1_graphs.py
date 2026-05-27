"""
SECTION 1: Graph feature integrity (ALL 44,470 graphs)
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

GRAPHS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/graphs")

# Feature dimension expected ranges
FEATURE_RANGES = {
    0: (0.0, 1.0),      # type_id / 12.0
    1: (0.0, 1.0),      # visibility 0=public,1=internal (BUG-3: 2=private would be out of range)
    2: (0.0, 1.0),      # uses_block_globals
    3: (0.0, 1.0),      # view
    4: (0.0, 1.0),      # payable
    5: (0.0, 1.0),      # complexity
    6: (0.0, 1.0),      # loc
    7: (0.0, 1.0),      # return_ignored (spec says {-1,0,1} but range [0,1])
    8: (0.0, 1.0),      # call_target_typed (spec says {-1,0,1} but range [0,1])
    9: (0.0, 1.0),      # in_unchecked
    10: (0.0, 1.0),     # has_loop
    11: (0.0, 1.0),     # ext_call_count
}

FEATURE_NAMES = [
    "type_id/12", "visibility", "uses_block_globals", "view", "payable",
    "complexity", "loc", "return_ignored", "call_target_typed", "in_unchecked",
    "has_loop", "ext_call_count"
]

def main():
    graph_files = sorted(GRAPHS_DIR.glob("*.pt"))
    total = len(graph_files)
    print(f"Found {total} graph files")

    # Accumulators
    feat_mins = [float('inf')] * 12
    feat_maxs = [float('-inf')] * 12
    feat_sums = [0.0] * 12
    feat_oor_nodes = [0] * 12  # out-of-range node counts
    feat_total_nodes = 0

    nan_graphs = 0
    inf_graphs = 0
    oor_graphs = 0  # graphs with ANY out-of-range feature
    zero_edge_graphs = 0
    zero_node_graphs = 0

    edge_shape_bad = 0
    edge_val_oor = 0  # edge_attr values outside [0,7]
    edge_neg_index = 0

    node_counts = []
    edge_counts = []

    # Track visibility=2 specifically
    visibility_2_count = 0
    return_ignored_neg_count = 0
    call_target_neg_count = 0

    BATCH = 1000
    processed = 0

    for i, fpath in enumerate(graph_files):
        try:
            data = torch.load(fpath, weights_only=False)
        except Exception as e:
            print(f"ERROR loading {fpath.name}: {e}")
            continue

        processed += 1

        x = data.x  # [N, 12]
        edge_index = data.edge_index  # [2, E]
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        N = x.shape[0] if x is not None else 0
        E = edge_index.shape[1] if edge_index is not None else 0

        node_counts.append(N)
        edge_counts.append(E)

        if N == 0:
            zero_node_graphs += 1
        if E == 0:
            zero_edge_graphs += 1

        if x is not None and N > 0:
            feat_total_nodes += N
            x_np = x.float().numpy()

            # NaN/Inf check
            if np.isnan(x_np).any():
                nan_graphs += 1
            if np.isinf(x_np).any():
                inf_graphs += 1

            # Per-dim stats
            graph_oor = False
            for d in range(12):
                col = x_np[:, d]
                lo, hi = FEATURE_RANGES[d]
                dm = col.min()
                dM = col.max()
                feat_mins[d] = min(feat_mins[d], dm)
                feat_maxs[d] = max(feat_maxs[d], dM)
                feat_sums[d] += col.sum()
                oor = ((col < lo - 1e-6) | (col > hi + 1e-6)).sum()
                feat_oor_nodes[d] += int(oor)
                if oor > 0:
                    graph_oor = True

            if graph_oor:
                oor_graphs += 1

            # Specific checks
            vis_col = x_np[:, 1]
            visibility_2_count += int((vis_col > 1.0 + 1e-6).sum())

            ret_col = x_np[:, 7]
            return_ignored_neg_count += int((ret_col < -1e-6).sum())

            ct_col = x_np[:, 8]
            call_target_neg_count += int((ct_col < -1e-6).sum())

        # Edge attr check
        if edge_attr is not None:
            if edge_attr.dim() != 1:
                edge_shape_bad += 1
            else:
                oor = ((edge_attr < 0) | (edge_attr > 7)).sum().item()
                if oor > 0:
                    edge_val_oor += 1

        # Edge index check
        if edge_index is not None and E > 0:
            if (edge_index < 0).any():
                edge_neg_index += 1

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{total}...", flush=True)

    print(f"\nTotal processed: {processed}/{total}")
    print("=" * 70)
    print("SECTION 1: GRAPH FEATURE INTEGRITY")
    print("=" * 70)

    print(f"\n--- Feature dimension stats (over {feat_total_nodes:,} total nodes) ---")
    print(f"{'Dim':<4} {'Name':<22} {'Min':>8} {'Max':>8} {'Mean':>8} {'OOR_nodes':>10} {'OOR%':>8} {'Status'}")
    print("-" * 80)
    for d in range(12):
        lo, hi = FEATURE_RANGES[d]
        mean = feat_sums[d] / max(feat_total_nodes, 1)
        oor = feat_oor_nodes[d]
        oor_pct = 100.0 * oor / max(feat_total_nodes, 1)
        status = "PASS" if oor == 0 else "FAIL"
        print(f"{d:<4} {FEATURE_NAMES[d]:<22} {feat_mins[d]:>8.4f} {feat_maxs[d]:>8.4f} {mean:>8.4f} {oor:>10,} {oor_pct:>7.2f}% {status}")

    print(f"\n--- Specific value checks ---")
    print(f"  visibility > 1.0 nodes:       {visibility_2_count:,} ({'FAIL' if visibility_2_count > 0 else 'PASS'})")
    print(f"  return_ignored < 0 nodes:     {return_ignored_neg_count:,} ({'FAIL' if return_ignored_neg_count > 0 else 'PASS'})")
    print(f"  call_target_typed < 0 nodes:  {call_target_neg_count:,} ({'FAIL' if call_target_neg_count > 0 else 'PASS'})")

    print(f"\n--- Data quality checks ---")
    print(f"  Graphs with NaN features:     {nan_graphs:,} ({'FAIL' if nan_graphs > 0 else 'PASS'})")
    print(f"  Graphs with Inf features:     {inf_graphs:,} ({'FAIL' if inf_graphs > 0 else 'PASS'})")
    print(f"  Graphs with ANY OOR feature:  {oor_graphs:,} / {processed:,} ({100.*oor_graphs/processed:.1f}%) ({'FAIL' if oor_graphs > 0 else 'PASS'})")
    print(f"  Graphs with 0 nodes:          {zero_node_graphs:,} ({'FAIL' if zero_node_graphs > 0 else 'PASS'})")
    print(f"  Graphs with 0 edges:          {zero_edge_graphs:,} ({'WARN' if zero_edge_graphs > 0 else 'PASS'})")

    print(f"\n--- Edge integrity ---")
    print(f"  Graphs with edge_attr shape != 1D:  {edge_shape_bad:,} ({'FAIL' if edge_shape_bad > 0 else 'PASS'})")
    print(f"  Graphs with edge_attr OOR [0,7]:    {edge_val_oor:,} ({'FAIL' if edge_val_oor > 0 else 'PASS'})")
    print(f"  Graphs with negative edge indices:  {edge_neg_index:,} ({'FAIL' if edge_neg_index > 0 else 'PASS'})")

    nc = np.array(node_counts)
    ec = np.array(edge_counts)
    print(f"\n--- Graph size distribution ---")
    print(f"  Nodes — min:{nc.min()}, max:{nc.max()}, mean:{nc.mean():.1f}, p50:{np.percentile(nc,50):.0f}, p95:{np.percentile(nc,95):.0f}")
    print(f"  Edges — min:{ec.min()}, max:{ec.max()}, mean:{ec.mean():.1f}, p50:{np.percentile(ec,50):.0f}, p95:{np.percentile(ec,95):.0f}")

    print("\n--- Node type distribution (from dim[0]*12 rounded) ---")
    # Quick second pass for node types (we need to recompute — skip for efficiency, we have enough info)
    print("  (Node type breakdown omitted for speed — check dim[0] range above)")

if __name__ == "__main__":
    main()
