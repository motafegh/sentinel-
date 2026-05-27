#!/usr/bin/env python3
"""
task13_graph_integrity.py — Graph Integrity Audit for SENTINEL v6

Sample 500 graph .pt files. For each verify:
  - x.shape[1] == 12
  - x.dtype float32
  - edge_index shape [2, E]
  - edge_index dtype int64
  - all edge_index in [0, N)
  - edge_attr 1-D [E]
  - edge_attr in [0, 7)
  - no NaN/Inf in x
  - CONTAINS targets are CFG nodes
  - CF edges connect CFG nodes
  - at least 1 edge
  - no self-loops in CF

Also count graphs with x.shape[1] != 12 (stale v5).
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def main():
    print_header("Task 13: Graph Integrity Audit")

    # ── Collect graph files ────────────────────────────────────────────────
    all_graph_files = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_graph_files:
        print("ERROR: No graph .pt files found in", GRAPHS_DIR)
        return
    print(f"Found {len(all_graph_files)} graph files")

    sample_size = min(500, len(all_graph_files))
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(len(all_graph_files), size=sample_size, replace=False)
    sampled_files = [all_graph_files[i] for i in sampled_indices]
    print(f"Sampling {sample_size} files")

    # ── Verification checks ────────────────────────────────────────────────
    checks = {
        "x_shape_12": 0,           # x.shape[1] == 12
        "x_dtype_float32": 0,      # x.dtype == float32
        "edge_index_shape": 0,     # edge_index shape [2, E]
        "edge_index_dtype": 0,     # edge_index dtype int64
        "edge_index_in_range": 0,  # all edge_index values in [0, N)
        "edge_attr_1d": 0,         # edge_attr shape [E]
        "edge_attr_range": 0,      # edge_attr values in [0, 7)
        "no_nan_inf": 0,           # no NaN/Inf in x
        "contains_cfg_targets": 0, # CONTAINS targets are CFG nodes
        "cf_cfg_nodes": 0,         # CF edges connect CFG nodes
        "at_least_1_edge": 0,      # at least 1 edge
        "no_cf_self_loops": 0,     # no self-loops in CF edges
    }

    stale_v5_count = 0  # graphs with x.shape[1] != 12
    total = 0
    skipped = 0
    failures = []

    # Anomaly tracking
    edge_type_counts = Counter()
    node_type_dist = Counter()

    for i, fpath in enumerate(sampled_files):
        if (i + 1) % 100 == 0:
            print(f"  Checked {i + 1}/{sample_size} files...")

        try:
            data = load_graph(fpath)
        except Exception as e:
            skipped += 1
            failures.append({"stem": fpath.stem, "issue": f"load error: {e}"})
            continue

        total += 1
        stem = fpath.stem
        file_failures = []

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        N = x.shape[0] if x is not None else 0

        # Track node type distribution
        if x is not None and x.shape[1] >= 1:
            type_ids = x[:, 0].numpy() if hasattr(x, 'numpy') else np.array(x[:, 0])
            for tid in type_ids:
                # Map back to node type name
                tid_int = int(round(tid * 12))  # type_id = NODE_TYPES[kind] / 12.0
                node_type_dist[tid_int] += 1

        # 1. x.shape[1] == 12
        if x is not None:
            if x.dim() == 2 and x.shape[1] == 12:
                checks["x_shape_12"] += 1
            else:
                stale_v5_count += 1
                file_failures.append(f"x shape {tuple(x.shape)} (expected [N,12])")
        else:
            file_failures.append("x is None")
            stale_v5_count += 1

        # 2. x.dtype float32
        if x is not None:
            if x.dtype == torch.float32:
                checks["x_dtype_float32"] += 1
            else:
                file_failures.append(f"x dtype {x.dtype} != float32")

        # 3. edge_index shape [2, E]
        if edge_index is not None:
            if edge_index.dim() == 2 and edge_index.shape[0] == 2:
                checks["edge_index_shape"] += 1
            else:
                file_failures.append(f"edge_index shape {tuple(edge_index.shape)} (expected [2,E])")
        else:
            file_failures.append("edge_index is None")

        # 4. edge_index dtype int64
        if edge_index is not None:
            if edge_index.dtype == torch.int64:
                checks["edge_index_dtype"] += 1
            else:
                file_failures.append(f"edge_index dtype {edge_index.dtype} != int64")

        # 5. All edge_index in [0, N)
        if edge_index is not None and N > 0:
            ei_np = edge_index.numpy() if hasattr(edge_index, 'numpy') else np.array(edge_index)
            if np.all((ei_np >= 0) & (ei_np < N)):
                checks["edge_index_in_range"] += 1
            else:
                oob = int(ei_np.max())
                file_failures.append(f"edge_index out of range [0,{N}): max={oob}")

        # 6. edge_attr 1-D [E]
        if edge_attr is not None and edge_index is not None:
            E = edge_index.shape[1]
            if edge_attr.dim() == 1 and edge_attr.shape[0] == E:
                checks["edge_attr_1d"] += 1
            else:
                file_failures.append(f"edge_attr shape {tuple(edge_attr.shape)} (expected [{E}])")
        elif edge_attr is None:
            file_failures.append("edge_attr is None")

        # 7. edge_attr in [0, 7)
        if edge_attr is not None:
            ea_np = edge_attr.numpy() if hasattr(edge_attr, 'numpy') else np.array(edge_attr)
            if np.all((ea_np >= 0) & (ea_np < 7)):
                checks["edge_attr_range"] += 1
            else:
                oob_min = int(ea_np.min())
                oob_max = int(ea_np.max())
                file_failures.append(f"edge_attr out of range [{oob_min}, {oob_max}] (expected [0,7))")
            # Track edge type distribution
            for val in ea_np.flat:
                edge_type_counts[int(val)] += 1

        # 8. No NaN/Inf in x
        if x is not None:
            x_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)
            if not np.any(np.isnan(x_np)) and not np.any(np.isinf(x_np)):
                checks["no_nan_inf"] += 1
            else:
                nan_count = int(np.sum(np.isnan(x_np)))
                inf_count = int(np.sum(np.isinf(x_np)))
                file_failures.append(f"x contains NaN({nan_count}) or Inf({inf_count})")

        # 9. CONTAINS targets are CFG nodes (edge_type 5)
        if edge_attr is not None and edge_index is not None and x is not None and x.shape[1] == 12:
            ea_np = edge_attr.numpy() if hasattr(edge_attr, 'numpy') else np.array(edge_attr)
            ei_np = edge_index.numpy() if hasattr(edge_index, 'numpy') else np.array(edge_index)
            x_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)

            contains_mask = ea_np == 5  # CONTAINS
            if np.any(contains_mask):
                contains_targets = ei_np[1, contains_mask]  # target nodes
                target_type_ids = x_np[contains_targets, 0]
                # CFG nodes have type_id >= 8/12
                cfg_mask = target_type_ids >= (8.0 / 12.0)
                if np.all(cfg_mask):
                    checks["contains_cfg_targets"] += 1
                else:
                    n_non_cfg = int(np.sum(~cfg_mask))
                    file_failures.append(f"CONTAINS has {n_non_cfg} non-CFG target nodes")
            else:
                # No CONTAINS edges — pass trivially
                checks["contains_cfg_targets"] += 1

        # 10. CF edges connect CFG nodes (edge_type 6)
        if edge_attr is not None and edge_index is not None and x is not None and x.shape[1] == 12:
            ea_np = edge_attr.numpy() if hasattr(edge_attr, 'numpy') else np.array(edge_attr)
            ei_np = edge_index.numpy() if hasattr(edge_index, 'numpy') else np.array(edge_index)
            x_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)

            cf_mask = ea_np == 6  # CONTROL_FLOW
            if np.any(cf_mask):
                cf_src = ei_np[0, cf_mask]
                cf_dst = ei_np[1, cf_mask]
                cf_nodes = np.concatenate([cf_src, cf_dst])
                cf_type_ids = x_np[cf_nodes, 0]
                cfg_node_mask = cf_type_ids >= (8.0 / 12.0)
                if np.all(cfg_node_mask):
                    checks["cf_cfg_nodes"] += 1
                else:
                    n_non_cfg = int(np.sum(~cfg_node_mask))
                    file_failures.append(f"CONTROL_FLOW has {n_non_cfg} non-CFG endpoint nodes")
            else:
                # No CF edges — pass trivially
                checks["cf_cfg_nodes"] += 1

        # 11. At least 1 edge
        if edge_index is not None and edge_index.shape[1] > 0:
            checks["at_least_1_edge"] += 1
        else:
            file_failures.append("graph has 0 edges")

        # 12. No self-loops in CF edges
        if edge_attr is not None and edge_index is not None:
            ea_np = edge_attr.numpy() if hasattr(edge_attr, 'numpy') else np.array(edge_attr)
            ei_np = edge_index.numpy() if hasattr(edge_index, 'numpy') else np.array(edge_index)

            cf_mask = ea_np == 6  # CONTROL_FLOW
            if np.any(cf_mask):
                cf_src = ei_np[0, cf_mask]
                cf_dst = ei_np[1, cf_mask]
                if not np.any(cf_src == cf_dst):
                    checks["no_cf_self_loops"] += 1
                else:
                    n_loops = int(np.sum(cf_src == cf_dst))
                    file_failures.append(f"CONTROL_FLOW has {n_loops} self-loops")
            else:
                # No CF edges — pass trivially
                checks["no_cf_self_loops"] += 1

        # Collect failures
        if file_failures:
            failures.append({"stem": stem, "issues": file_failures})

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 13: Graph Integrity Audit\n")
    report_lines.append(f"**Sample size:** {sample_size}  \n")
    report_lines.append(f"**Successfully loaded:** {total}  \n")
    report_lines.append(f"**Skipped (load errors):** {skipped}  \n")
    report_lines.append(f"**Stale v5 (x.shape[1] != 12):** {stale_v5_count}\n")

    # Pass/fail summary
    report_lines.append("\n## Check Results\n")
    report_lines.append("| Check | Pass | Fail | Rate |\n")
    report_lines.append("|-------|------|------|------|\n")
    for check_name, pass_count in checks.items():
        fail_count = total - pass_count
        rate = f"{pass_count/total:.1%}" if total > 0 else "N/A"
        report_lines.append(f"| {check_name} | {pass_count} | {fail_count} | {rate} |\n")

    # Edge type distribution
    report_lines.append("\n## Edge Type Distribution\n")
    report_lines.append("| Edge Type ID | Name | Count |\n")
    report_lines.append("|--------------|------|-------|\n")
    for et_id in sorted(edge_type_counts.keys()):
        name = EDGE_TYPE_NAMES.get(et_id, f"UNKNOWN({et_id})")
        report_lines.append(f"| {et_id} | {name} | {edge_type_counts[et_id]} |\n")

    # Node type distribution
    NODE_TYPE_NAMES = {
        0: "STATE_VAR", 1: "FUNCTION", 2: "MODIFIER", 3: "EVENT",
        4: "FALLBACK", 5: "RECEIVE", 6: "CONSTRUCTOR", 7: "CONTRACT",
        8: "CFG_NODE_CALL", 9: "CFG_NODE_WRITE", 10: "CFG_NODE_READ",
        11: "CFG_NODE_CHECK", 12: "CFG_NODE_OTHER",
    }
    report_lines.append("\n## Node Type Distribution\n")
    report_lines.append("| Type ID | Name | Count |\n")
    report_lines.append("|---------|------|-------|\n")
    for nt_id in sorted(node_type_dist.keys()):
        name = NODE_TYPE_NAMES.get(nt_id, f"UNKNOWN({nt_id})")
        report_lines.append(f"| {nt_id} | {name} | {node_type_dist[nt_id]} |\n")

    # Stale v5 detail
    if stale_v5_count > 0:
        report_lines.append(f"\n## Stale v5 Graphs (x.shape[1] != 12)\n")
        report_lines.append(f"Found **{stale_v5_count}** graph(s) with wrong feature dimension. These need re-extraction.\n")

    # Failure details
    if failures:
        report_lines.append("\n## Failures Detail (first 30)\n")
        for f_entry in failures[:30]:
            issues_str = "; ".join(f_entry["issues"]) if isinstance(f_entry["issues"], list) else f_entry["issues"]
            report_lines.append(f"- **{f_entry['stem']}**: {issues_str}\n")
        if len(failures) > 30:
            report_lines.append(f"- ... and {len(failures) - 30} more files with failures\n")
    else:
        report_lines.append("\n✅ All checks passed on all sampled files.\n")

    report_content = "".join(report_lines)
    save_report("task13_graph_integrity", report_content)
    print_header("Task 13 Complete")


if __name__ == "__main__":
    main()
