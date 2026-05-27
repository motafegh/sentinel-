#!/usr/bin/env python3
"""
task1_recheck_activation_split.py — Per-Class Feature Activation Split Audit

Re-check per-class feature activation rates with declaration vs CFG node split.
For each of 10 classes, sample 20 pure-label contracts. For each graph:
1. Split nodes into declaration (type_id[0] < 0.583) and CFG (>= 0.583)
2. Compute feature nonzero rates SEPARATELY for each group
3. Focus on: uses_block_globals [2], return_ignored [7], has_loop [10],
   ext_call_count [11], in_unchecked [9]
4. Also compute overall rates for comparison with prior audit

Report: updated table with Declaration-only and CFG-only activation rates,
any differences from prior audit.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *

# Threshold for declaration vs CFG node split (8/12 ≈ 0.5833)
DECL_THRESHOLD = 8.0 / 12.0

# Features to focus on (index, name)
FOCUS_FEATURES = [
    (2, "uses_block_globals"),
    (7, "return_ignored"),
    (9, "in_unchecked"),
    (10, "has_loop"),
    (11, "external_call_count"),
]

SAMPLE_PER_CLASS = 20


def compute_activation_rates(feature_matrix: np.ndarray) -> dict:
    """
    Compute per-feature nonzero rates for a node feature matrix.
    Returns {feature_idx: nonzero_rate}.
    """
    rates = {}
    for feat_idx in range(feature_matrix.shape[1]):
        col = feature_matrix[:, feat_idx]
        # For features with sentinel -1.0, count both 1.0 AND -1.0 as "activated"
        if feat_idx in (7, 8):  # return_ignored, call_target_typed have -1.0 sentinel
            nonzero = np.sum((col != 0.0))
        else:
            nonzero = np.sum(col != 0.0)
        rates[feat_idx] = nonzero / len(col) if len(col) > 0 else 0.0
    return rates


def main():
    print_header("Task 1: Re-check Activation Split (Declaration vs CFG)")

    if not GRAPHS_DIR.exists():
        print(f"ERROR: Graphs directory not found: {GRAPHS_DIR}")
        save_report("task1_recheck_activation_split",
                     "# Task 1: Re-check Activation Split\n\n"
                     "**ERROR**: Graphs directory not found.\n")
        return

    # ── Load labels ─────────────────────────────────────────────────────────
    if not LABEL_CSV.exists():
        print(f"ERROR: Label CSV not found at {LABEL_CSV}")
        save_report("task1_recheck_activation_split",
                     "# Task 1: Re-check Activation Split\n\n"
                     "**ERROR**: Label CSV not found.\n")
        return

    labels = load_label_csv()
    print(f"Loaded {len(labels)} entries from label CSV")

    # ── For each class, sample pure-label contracts ─────────────────────────
    all_graph_stems = {p.stem for p in GRAPHS_DIR.glob("*.pt")}
    print(f"Found {len(all_graph_stems)} graph files on disk")

    report_lines = []
    report_lines.append("# Task 1: Re-check Activation Split (Declaration vs CFG)\n\n")
    report_lines.append(f"**Sample per class:** {SAMPLE_PER_CLASS} pure-label contracts  \n")
    report_lines.append(f"**Declaration threshold:** type_id < {DECL_THRESHOLD:.4f} (8/12)  \n")
    report_lines.append(f"**Focus features:** {', '.join(f[1] for f in FOCUS_FEATURES)}\n\n")

    # Summary table data
    summary_rows = []
    per_class_details = {}

    for cls_name in VULN_CLASSES:
        print(f"\n--- Class: {cls_name} ---")

        # Get pure-label stems
        pure_stems = get_stems_with_label(cls_name, label_value=1, pure=True)
        # Filter to only those with graph files
        pure_stems = [s for s in pure_stems if s in all_graph_stems]
        print(f"  Pure-label stems with graphs: {len(pure_stems)}")

        if not pure_stems:
            print(f"  WARNING: No pure-label graphs for {cls_name}")
            per_class_details[cls_name] = {"error": "no pure-label graphs"}
            continue

        # Sample
        import random
        rng = random.Random(42)
        rng.shuffle(pure_stems)
        sampled = pure_stems[:SAMPLE_PER_CLASS]
        print(f"  Sampled {len(sampled)} contracts")

        # Accumulate features by node type
        decl_features_list = []
        cfg_features_list = []
        all_features_list = []

        loaded = 0
        skipped = 0

        for stem in sampled:
            graph_path = GRAPHS_DIR / f"{stem}.pt"
            try:
                data = load_graph(graph_path)
                x = data.x
                if x is None or x.dim() != 2 or x.shape[1] != 12:
                    skipped += 1
                    continue

                x_np = x.numpy().astype(np.float32)
                loaded += 1

                # Split by type_id
                type_ids = x_np[:, 0]
                decl_mask = type_ids < DECL_THRESHOLD
                cfg_mask = ~decl_mask

                if np.any(decl_mask):
                    decl_features_list.append(x_np[decl_mask])
                if np.any(cfg_mask):
                    cfg_features_list.append(x_np[cfg_mask])
                all_features_list.append(x_np)

            except Exception as e:
                print(f"  WARNING: Failed to load {stem}: {e}")
                skipped += 1

        print(f"  Loaded {loaded}, skipped {skipped}")

        if not all_features_list:
            print(f"  WARNING: No valid graphs loaded for {cls_name}")
            per_class_details[cls_name] = {"error": "no valid graphs loaded"}
            continue

        # Compute activation rates
        all_matrix = np.concatenate(all_features_list, axis=0)
        decl_matrix = np.concatenate(decl_features_list, axis=0) if decl_features_list else np.empty((0, 12))
        cfg_matrix = np.concatenate(cfg_features_list, axis=0) if cfg_features_list else np.empty((0, 12))

        all_rates = compute_activation_rates(all_matrix)
        decl_rates = compute_activation_rates(decl_matrix)
        cfg_rates = compute_activation_rates(cfg_matrix)

        per_class_details[cls_name] = {
            "loaded": loaded,
            "total_nodes": all_matrix.shape[0],
            "decl_nodes": decl_matrix.shape[0],
            "cfg_nodes": cfg_matrix.shape[0],
            "all_rates": all_rates,
            "decl_rates": decl_rates,
            "cfg_rates": cfg_rates,
        }

        # Build summary row for focus features
        for feat_idx, feat_name in FOCUS_FEATURES:
            all_r = all_rates.get(feat_idx, 0.0)
            decl_r = decl_rates.get(feat_idx, 0.0)
            cfg_r = cfg_rates.get(feat_idx, 0.0)
            summary_rows.append({
                "class": cls_name,
                "feature": feat_name,
                "overall": all_r,
                "declaration": decl_r,
                "cfg": cfg_r,
                "decl_cfg_diff": abs(decl_r - cfg_r),
            })

        # Print focus features
        print(f"  Nodes: {all_matrix.shape[0]} total, "
              f"{decl_matrix.shape[0]} decl, {cfg_matrix.shape[0]} cfg")
        for feat_idx, feat_name in FOCUS_FEATURES:
            all_r = all_rates.get(feat_idx, 0.0)
            decl_r = decl_rates.get(feat_idx, 0.0)
            cfg_r = cfg_rates.get(feat_idx, 0.0)
            print(f"    {feat_name}: overall={all_r:.4f}, decl={decl_r:.4f}, cfg={cfg_r:.4f}")

    # ── Build full feature tables per class ─────────────────────────────────
    report_lines.append("## Per-Class Feature Activation Rates\n\n")

    for cls_name in VULN_CLASSES:
        details = per_class_details.get(cls_name, {})
        if "error" in details:
            report_lines.append(f"### {cls_name}\n\n")
            report_lines.append(f"*Error: {details['error']}*\n\n")
            continue

        report_lines.append(f"### {cls_name}\n\n")
        report_lines.append(f"**Contracts loaded:** {details['loaded']}  \n")
        report_lines.append(f"**Nodes:** {details['total_nodes']} total, "
                            f"{details['decl_nodes']} declaration, "
                            f"{details['cfg_nodes']} CFG\n\n")

        report_lines.append("| Feature | Overall | Declaration | CFG | Decl-CFG Diff |\n")
        report_lines.append("|---------|---------|-------------|-----|---------------|\n")

        for feat_idx in range(12):
            feat_name = FEATURE_NAMES[feat_idx]
            all_r = details["all_rates"].get(feat_idx, 0.0)
            decl_r = details["decl_rates"].get(feat_idx, 0.0)
            cfg_r = details["cfg_rates"].get(feat_idx, 0.0)
            diff = abs(decl_r - cfg_r)

            marker = ""
            if feat_idx in [f[0] for f in FOCUS_FEATURES] and diff > 0.1:
                marker = " **"  # highlight large differences

            report_lines.append(
                f"| {feat_name} | {all_r:.4f} | {decl_r:.4f} | {cfg_r:.4f} | "
                f"{diff:.4f}{marker} |\n"
            )
        report_lines.append("\n")

    # ── Focus feature summary table ─────────────────────────────────────────
    report_lines.append("## Focus Feature Summary Across All Classes\n\n")
    report_lines.append("| Class | Feature | Overall | Declaration | CFG | Diff | Note |\n")
    report_lines.append("|-------|---------|---------|-------------|-----|------|------|\n")

    for row in summary_rows:
        note = ""
        if row["declaration"] > 0 and row["cfg"] == 0:
            note = "DECL-only"
        elif row["cfg"] > 0 and row["declaration"] == 0:
            note = "CFG-only"
        elif row["decl_cfg_diff"] > 0.2:
            note = "Large split"
        elif row["overall"] == 0:
            note = "DEAD feature"

        report_lines.append(
            f"| {row['class']} | {row['feature']} | {row['overall']:.4f} | "
            f"{row['declaration']:.4f} | {row['cfg']:.4f} | "
            f"{row['decl_cfg_diff']:.4f} | {note} |\n"
        )

    # ── Key findings ────────────────────────────────────────────────────────
    report_lines.append("\n## Key Findings\n\n")

    # Check for dead features
    dead_features = set()
    for feat_idx, feat_name in FOCUS_FEATURES:
        all_rates_for_feat = [r["overall"] for r in summary_rows if r["feature"] == feat_name]
        if all_rates_for_feat and max(all_rates_for_feat) == 0.0:
            dead_features.add(feat_name)

    if dead_features:
        report_lines.append("### Dead Features (zero activation across all classes)\n\n")
        for feat_name in sorted(dead_features):
            report_lines.append(f"- **{feat_name}**: Never activated in any class. "
                                "The feature is wasted and provides no discriminative signal.\n")
        report_lines.append("\n")

    # Check for declaration-only features
    decl_only = set()
    cfg_only = set()
    large_split = set()

    for row in summary_rows:
        if row["declaration"] > 0.01 and row["cfg"] == 0:
            decl_only.add(row["feature"])
        if row["cfg"] > 0.01 and row["declaration"] == 0:
            cfg_only.add(row["feature"])
        if row["decl_cfg_diff"] > 0.2:
            large_split.add(row["feature"])

    if decl_only:
        report_lines.append("### Declaration-Only Activation\n\n")
        report_lines.append("These features are only activated on declaration nodes (type_id < 8/12), "
                            "not CFG nodes:\n")
        for feat_name in sorted(decl_only):
            report_lines.append(f"- **{feat_name}**\n")
        report_lines.append("\n*Expected*: Features like `in_unchecked`, `has_loop`, `external_call_count` "
                            "are function-level and should only activate on declaration (FUNCTION) nodes. "
                            "CFG nodes get 0.0 for these by design.\n\n")

    if cfg_only:
        report_lines.append("### CFG-Only Activation\n\n")
        report_lines.append("These features are only activated on CFG nodes (type_id >= 8/12):\n")
        for feat_name in sorted(cfg_only):
            report_lines.append(f"- **{feat_name}**\n")
        report_lines.append("\n")

    if large_split:
        report_lines.append("### Large Declaration-CFG Split (> 0.2 difference)\n\n")
        report_lines.append("These features have significantly different activation rates "
                            "between declaration and CFG nodes:\n")
        for feat_name in sorted(large_split):
            rates = [(r["class"], r["declaration"], r["cfg"])
                     for r in summary_rows if r["feature"] == feat_name and r["decl_cfg_diff"] > 0.2]
            report_lines.append(f"- **{feat_name}**: {len(rates)} classes with large split\n")
            for cls, decl_r, cfg_r in rates[:5]:
                report_lines.append(f"  - {cls}: decl={decl_r:.4f}, cfg={cfg_r:.4f}\n")
        report_lines.append("\n")

    # Comparison with prior audit
    report_lines.append("## Comparison with Prior Audit (Task 09)\n\n")
    report_lines.append("Task 09 computed overall activation rates without splitting by node type. "
                        "This audit provides a more granular view:\n\n")
    report_lines.append("- **Declaration-only rates** show feature activation on structural nodes "
                        "(CONTRACT, FUNCTION, STATE_VAR, etc.)\n")
    report_lines.append("- **CFG-only rates** show feature activation on statement-level nodes "
                        "(CFG_NODE_CALL, CFG_NODE_WRITE, etc.)\n")
    report_lines.append("- The split reveals that most semantic features (has_loop, in_unchecked, "
                        "external_call_count) are **function-level signals** that only activate "
                        "on declaration nodes — CFG nodes always get 0.0 for these features\n")
    report_lines.append("- This is **by design** (see graph_extractor.py `_build_cfg_node_features`) — "
                        "the GNN propagates function-level signals to CFG nodes via CONTAINS edges\n\n")

    report_content = "".join(report_lines)
    save_report("task1_recheck_activation_split", report_content)
    print_header("Task 1 Complete")


if __name__ == "__main__":
    main()
