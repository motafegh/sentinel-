"""
Task 13: Graph Structural Integrity  (500 random samples)
Task 26: Stale v5.0 Graph Contamination Check (ALL graphs)
-----------------------------------------------------------
Run:
    python task_13_26_graph_integrity.py
"""
import sys
import numpy as np
import torch
from pathlib import Path
from common import get_dirs, load_csv, load_graph, random_pt_sample, DECL_THRESHOLD, print_header

N_SAMPLE = 500   # for Task 13

def main():
    print_header("13 + 26", "Graph Structural Integrity + Stale v5.0 Detection")
    _, _, graphs_dir, _, _, _ = get_dirs()

    all_pts = list(graphs_dir.glob("*.pt"))
    print(f"Total .pt files in graphs/: {len(all_pts):,}")

    # ════════════════════════════════════════════════════════════════════════
    # TASK 26 — Scan ALL graphs for stale 8-dim (v5.0) feature matrices
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n── Task 26: Stale v5.0 scan (all {len(all_pts):,} files) ──────────────")
    stale_stems   = []
    correct_count = 0
    other_dims    = {}
    failed_load   = []

    for p in all_pts:
        try:
            g = load_graph(p)
            dim = g.x.shape[1]
        except Exception as e:
            failed_load.append(p.stem)
            continue
        if dim == 12:
            correct_count += 1
        elif dim == 8:
            stale_stems.append(p.stem)
        else:
            other_dims[dim] = other_dims.get(dim, 0) + 1

    print(f"  v4 schema (12-dim): {correct_count:,}")
    print(f"  v5.0 stale (8-dim): {len(stale_stems):,}")
    print(f"  Other dimensions:   {dict(other_dims)}")
    print(f"  Failed to load:     {len(failed_load):,}")

    # Cross-reference stale with CSV
    if stale_stems:
        try:
            df = load_csv()
            csv_set = set(df["md5_stem"].astype(str).tolist())
            stale_in_csv = [s for s in stale_stems if s in csv_set]
            print(f"\n  Stale graphs in deduped CSV: {len(stale_in_csv)} / {len(stale_stems)}")
            if stale_in_csv:
                print("  [BUG] These stale graphs are in the training set!")
                for s in stale_in_csv[:20]:
                    print(f"    {s}")
            else:
                print("  [CONFIRMED] Stale graphs are NOT in the deduped CSV — safe to ignore.")
        except Exception as e:
            print(f"  [WARN] Could not cross-reference with CSV: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TASK 13 — Structural integrity on 500 random samples
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n── Task 13: Structural integrity (n={N_SAMPLE}) ─────────────────────")
    paths = random_pt_sample(graphs_dir, N_SAMPLE)

    checks = {
        "x_shape_12dim":       0,
        "x_dtype_float32":     0,
        "edge_index_shape_2x": 0,
        "edge_index_dtype_int64": 0,
        "edge_index_in_range": 0,
        "edge_attr_1d":        0,
        "edge_attr_dtype_int64": 0,
        "edge_attr_range_0_6": 0,
        "no_nan_in_x":         0,
        "no_inf_in_x":         0,
        "has_edges":           0,
        "no_cf_selfloops":     0,
        "contains_targets_cfg": 0,
        "cf_nodes_are_cfg":    0,
    }
    total = 0
    disconnected = []
    invalid_edge = []
    cf_selfloops = []
    wrong_contains = []

    for p in paths:
        try:
            g = load_graph(p)
        except Exception as e:
            print(f"  [ERROR] {p.name}: {e}")
            continue
        total += 1
        n = g.num_nodes
        x  = g.x
        ei = g.edge_index
        ea = g.edge_attr

        # 1. x shape
        if x.shape[1] == 12:
            checks["x_shape_12dim"] += 1

        # 2. x dtype
        if x.dtype == torch.float32:
            checks["x_dtype_float32"] += 1

        # 3. edge_index shape
        if ei.dim() == 2 and ei.shape[0] == 2:
            checks["edge_index_shape_2x"] += 1

        # 4. edge_index dtype
        if ei.dtype == torch.int64:
            checks["edge_index_dtype_int64"] += 1

        # 5. edge_index values in range
        if ei.numel() == 0 or (ei.max().item() < n and ei.min().item() >= 0):
            checks["edge_index_in_range"] += 1
        else:
            invalid_edge.append(p.stem)

        # 6. edge_attr is 1-D
        if ea.dim() == 1:
            checks["edge_attr_1d"] += 1

        # 7. edge_attr dtype
        if ea.dtype == torch.int64:
            checks["edge_attr_dtype_int64"] += 1

        # 8. edge_attr values 0-6
        if ea.numel() == 0 or (ea.max().item() <= 6 and ea.min().item() >= 0):
            checks["edge_attr_range_0_6"] += 1

        # 9. No NaN/Inf in x
        xnp = x.numpy()
        if not np.any(np.isnan(xnp)):
            checks["no_nan_in_x"] += 1
        if not np.any(np.isinf(xnp)):
            checks["no_inf_in_x"] += 1

        # 10. Has at least one edge
        if ei.shape[1] > 0:
            checks["has_edges"] += 1
        else:
            disconnected.append(p.stem)

        if ei.numel() == 0:
            checks["no_cf_selfloops"] += 1
            checks["contains_targets_cfg"] += 1
            checks["cf_nodes_are_cfg"] += 1
            continue

        ea_np = ea.numpy()
        src = ei[0].numpy()
        dst = ei[1].numpy()
        type_ids = xnp[:, 0]

        # 11. No self-loops in CF edges (type 6)
        cf_mask = ea_np == 6
        cf_selfloop = np.any(src[cf_mask] == dst[cf_mask])
        if not cf_selfloop:
            checks["no_cf_selfloops"] += 1
        else:
            cf_selfloops.append(p.stem)

        # 12. CONTAINS edges (type 5): dst should be CFG node
        c_mask = ea_np == 5
        if c_mask.any():
            c_dst_types = type_ids[dst[c_mask]]
            if np.all(c_dst_types >= DECL_THRESHOLD):
                checks["contains_targets_cfg"] += 1
            else:
                wrong_contains.append(p.stem)
        else:
            checks["contains_targets_cfg"] += 1

        # 13. CF edges: src and dst should be CFG nodes
        if cf_mask.any():
            cf_src_types = type_ids[src[cf_mask]]
            cf_dst_types = type_ids[dst[cf_mask]]
            if np.all(cf_src_types >= DECL_THRESHOLD) and np.all(cf_dst_types >= DECL_THRESHOLD):
                checks["cf_nodes_are_cfg"] += 1
        else:
            checks["cf_nodes_are_cfg"] += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nStructural checks (n={total}):")
    print(f"{'Check':<35} {'Pass':>6} / {'Total':>6}  {'Rate':>6}")
    print("-" * 58)
    for k, v in checks.items():
        rate = f"{v/total*100:.1f}%" if total else "N/A"
        print(f"  {k:<33} {v:>6} / {total:>6}  {rate:>6}")

    print(f"\n  Fully disconnected graphs: {len(disconnected)}")
    print(f"  Invalid edge_index values: {len(invalid_edge)}")
    print(f"  CF self-loops:             {len(cf_selfloops)}")
    print(f"  CONTAINS→non-CFG targets:  {len(wrong_contains)}")

    if disconnected:
        print(f"\n  [BUG] Disconnected graphs: {disconnected[:10]}")
    if invalid_edge:
        print(f"\n  [BUG] Out-of-range edge_index: {invalid_edge[:10]}")

    print("\n── Summary ─────────────────────────────────────────────────────────────")
    fail_list = [(k, checks[k]) for k in checks if checks[k] < total]
    if not fail_list:
        print("  [CONFIRMED] All structural checks passed.")
    else:
        for k, v in fail_list:
            print(f"  [BUG] {k}: {total - v} failures out of {total}")

if __name__ == "__main__":
    main()
