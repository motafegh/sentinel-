#!/usr/bin/env python3
"""
task09_feature_range_audit.py — Feature Range Audit for SENTINEL v6

Loads 500 random graph .pt files, extracts node features [N,12], splits nodes
into declaration (type_id[0] < 8/12) vs CFG (type_id[0] >= 8/12), and computes
detailed per-feature statistics for ALL nodes, declaration-only, and CFG-only.

Flags any feature where declaration and CFG have different ranges, excluding
known bugs BUG-1 (CFG loc>1), BUG-2 (complexity>1), BUG-3 (visibility=2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *

import numpy as np

# ── Known bugs to suppress ────────────────────────────────────────────────────
KNOWN_BUGS = {
    ("loc", "CFG_range_exceeds_1"),        # BUG-1: CFG loc can exceed 1.0
    ("complexity", "CFG_range_exceeds_1"), # BUG-2: complexity can exceed 1.0
    ("visibility", "value_equals_2"),       # BUG-3: visibility can be 2
}

DECL_THRESHOLD = 8.0 / 12.0  # ≈0.5833


def compute_stats(values: np.ndarray) -> dict:
    """Compute statistics for a 1-D array of feature values."""
    n = len(values)
    if n == 0:
        return {
            "min": "N/A", "max": "N/A", "p5": "N/A", "p50": "N/A", "p95": "N/A",
            "count_gt1": "N/A", "count_lt_neg1": "N/A", "nan_count": "N/A", "inf_count": "N/A",
        }
    return {
        "min":           float(np.min(values)),
        "max":           float(np.max(values)),
        "p5":            float(np.percentile(values, 5)),
        "p50":           float(np.percentile(values, 50)),
        "p95":           float(np.percentile(values, 95)),
        "count_gt1":     int(np.sum(values > 1.0)),
        "count_lt_neg1": int(np.sum(values < -1.0)),
        "nan_count":     int(np.sum(np.isnan(values))),
        "inf_count":     int(np.sum(np.isinf(values))),
    }


def fmt_val(v) -> str:
    """Format a stat value for display."""
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return f"{v:.4f}"


def main():
    print_header("Task 09: Feature Range Audit")

    # ── Collect graph files ────────────────────────────────────────────────
    all_graph_files = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_graph_files:
        print("ERROR: No graph .pt files found in", GRAPHS_DIR)
        return
    print(f"Found {len(all_graph_files)} graph files")

    sample_size = min(500, len(all_graph_files))
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(all_graph_files), size=sample_size, replace=False)
    sampled_files = [all_graph_files[i] for i in sampled]
    print(f"Sampling {sample_size} files")

    # ── Accumulate features per category ───────────────────────────────────
    # For each of 12 features, accumulate all values, decl values, cfg values
    all_values  = [[] for _ in range(12)]
    decl_values = [[] for _ in range(12)]
    cfg_values  = [[] for _ in range(12)]

    total_nodes = 0
    decl_nodes = 0
    cfg_nodes = 0
    skipped = 0

    for i, fpath in enumerate(sampled_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{sample_size} files...")
        try:
            data = load_graph(fpath)
            x = data.x
            if x is None or x.dim() != 2 or x.shape[1] != 12:
                skipped += 1
                continue
            x_np = x.numpy().astype(np.float32)
            N = x_np.shape[0]
            total_nodes += N

            # Split by type_id (feature index 0)
            type_ids = x_np[:, 0]
            decl_mask = type_ids < DECL_THRESHOLD
            cfg_mask  = ~decl_mask

            decl_nodes += int(decl_mask.sum())
            cfg_nodes  += int(cfg_mask.sum())

            for feat_idx in range(12):
                col = x_np[:, feat_idx]
                all_values[feat_idx].append(col)
                decl_values[feat_idx].append(col[decl_mask])
                cfg_values[feat_idx].append(col[cfg_mask])

        except Exception as e:
            print(f"  WARNING: Failed to load {fpath.name}: {e}")
            skipped += 1

    print(f"  Loaded {sample_size - skipped} files, skipped {skipped}")
    print(f"  Total nodes: {total_nodes} (decl: {decl_nodes}, cfg: {cfg_nodes})")

    # ── Compute statistics ─────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 09: Feature Range Audit\n")
    report_lines.append(f"**Sample size:** {sample_size} graph files  ")
    report_lines.append(f"**Files loaded:** {sample_size - skipped}  ")
    report_lines.append(f"**Total nodes:** {total_nodes} (declaration: {decl_nodes}, CFG: {cfg_nodes})\n")

    flags = []

    stat_keys = ["min", "max", "p5", "p50", "p95", "count_gt1", "count_lt_neg1", "nan_count", "inf_count"]
    stat_headers = ["Feature", "Category"] + stat_keys

    all_rows = []

    for feat_idx in range(12):
        feat_name = FEATURE_NAMES[feat_idx]

        for category, values_list in [("ALL", all_values), ("DECL", decl_values), ("CFG", cfg_values)]:
            if not values_list[feat_idx]:
                cat_np = np.array([], dtype=np.float32)
            else:
                cat_np = np.concatenate(values_list[feat_idx])

            stats = compute_stats(cat_np)
            row = [feat_name, category] + [fmt_val(stats[k]) for k in stat_keys]
            all_rows.append(row)

        # ── Flag differences between DECL and CFG ──────────────────────────
        decl_np = np.concatenate(decl_values[feat_idx]) if decl_values[feat_idx] else np.array([], dtype=np.float32)
        cfg_np  = np.concatenate(cfg_values[feat_idx])   if cfg_values[feat_idx]   else np.array([], dtype=np.float32)

        if len(decl_np) > 0 and len(cfg_np) > 0:
            decl_stats = compute_stats(decl_np)
            cfg_stats  = compute_stats(cfg_np)

            # Check for range differences
            # CFG max > 1.0 for a feature that should be [0,1]
            if cfg_stats["max"] != "N/A" and cfg_stats["max"] > 1.0:
                flag_key = (feat_name, "CFG_range_exceeds_1")
                if flag_key not in KNOWN_BUGS:
                    flags.append(f"**NEW FLAG**: {feat_name} — CFG max = {cfg_stats['max']:.4f} > 1.0 (expected [0,1])")

            # CFG min < -1.0
            if cfg_stats["min"] != "N/A" and cfg_stats["min"] < -1.0:
                flag_key = (feat_name, "CFG_range_below_neg1")
                if flag_key not in KNOWN_BUGS:
                    flags.append(f"**NEW FLAG**: {feat_name} — CFG min = {cfg_stats['min']:.4f} < -1.0")

            # DECL max > 1.0 for a feature that should be [0,1]
            if decl_stats["max"] != "N/A" and decl_stats["max"] > 1.0:
                flag_key = (feat_name, "DECL_range_exceeds_1")
                if flag_key not in KNOWN_BUGS:
                    flags.append(f"**NEW FLAG**: {feat_name} — DECL max = {decl_stats['max']:.4f} > 1.0 (expected [0,1])")

            # DECL min < -1.0
            if decl_stats["min"] != "N/A" and decl_stats["min"] < -1.0:
                flag_key = (feat_name, "DECL_range_below_neg1")
                if flag_key not in KNOWN_BUGS:
                    flags.append(f"**NEW FLAG**: {feat_name} — DECL min = {decl_stats['min']:.4f} < -1.0")

            # visibility value == 2
            if feat_name == "visibility":
                if len(cfg_np) > 0 and np.any(cfg_np == 2.0):
                    flag_key = (feat_name, "value_equals_2")
                    if flag_key not in KNOWN_BUGS:
                        flags.append(f"**NEW FLAG**: visibility — CFG contains value 2.0 (private)")

            # NaN or Inf
            if cfg_stats["nan_count"] != "N/A" and cfg_stats["nan_count"] > 0:
                flags.append(f"**NEW FLAG**: {feat_name} — CFG has {cfg_stats['nan_count']} NaN values")
            if cfg_stats["inf_count"] != "N/A" and cfg_stats["inf_count"] > 0:
                flags.append(f"**NEW FLAG**: {feat_name} — CFG has {cfg_stats['inf_count']} Inf values")
            if decl_stats["nan_count"] != "N/A" and decl_stats["nan_count"] > 0:
                flags.append(f"**NEW FLAG**: {feat_name} — DECL has {decl_stats['nan_count']} NaN values")
            if decl_stats["inf_count"] != "N/A" and decl_stats["inf_count"] > 0:
                flags.append(f"**NEW FLAG**: {feat_name} — DECL has {decl_stats['inf_count']} Inf values")

            # Decl vs CFG significantly different ranges (min/max differ by >0.5)
            if decl_stats["min"] != "N/A" and cfg_stats["min"] != "N/A":
                if abs(decl_stats["min"] - cfg_stats["min"]) > 0.5:
                    flag_key = (feat_name, f"DECL_CFG_min_diff_{abs(decl_stats['min']-cfg_stats['min']):.2f}")
                    if flag_key not in KNOWN_BUGS:
                        flags.append(
                            f"**NEW FLAG**: {feat_name} — DECL min ({decl_stats['min']:.4f}) vs "
                            f"CFG min ({cfg_stats['min']:.4f}) differ by >0.5"
                        )
                if abs(decl_stats["max"] - cfg_stats["max"]) > 0.5:
                    flag_key = (feat_name, f"DECL_CFG_max_diff_{abs(decl_stats['max']-cfg_stats['max']):.2f}")
                    if flag_key not in KNOWN_BUGS:
                        flags.append(
                            f"**NEW FLAG**: {feat_name} — DECL max ({decl_stats['max']:.4f}) vs "
                            f"CFG max ({cfg_stats['max']:.4f}) differ by >0.5"
                        )

    # ── Build report ───────────────────────────────────────────────────────
    table = format_table(stat_headers, all_rows)
    report_lines.append("## Feature Statistics\n")
    report_lines.append("```\n" + table + "\n```\n")

    # Known bugs section
    report_lines.append("## Known Bugs (suppressed)\n")
    for bug_key, description in [
        (("loc", "CFG_range_exceeds_1"), "BUG-1: CFG loc can exceed 1.0 (raw line count on CFG nodes)"),
        (("complexity", "CFG_range_exceeds_1"), "BUG-2: complexity can exceed 1.0 (raw block count on CFG nodes)"),
        (("visibility", "value_equals_2"), "BUG-3: visibility=2 (private) is valid ordinal encoding"),
    ]:
        if bug_key in KNOWN_BUGS:
            report_lines.append(f"- {description}\n")

    # New flags section
    report_lines.append("## New Flags\n")
    if flags:
        for f in flags:
            report_lines.append(f"- {f}\n")
            print(f"  FLAG: {f}")
    else:
        report_lines.append("No new flags beyond known bugs.\n")
        print("  No new flags beyond known bugs.")

    report_content = "".join(report_lines)
    save_report("task09_feature_range_audit", report_content)
    print_header("Task 09 Complete")


if __name__ == "__main__":
    main()
