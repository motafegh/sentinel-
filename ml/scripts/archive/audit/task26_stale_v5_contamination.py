#!/usr/bin/env python3
"""
task26_stale_v5_contamination.py — Stale v5 Graph Contamination Audit

Scan ALL .pt files in ml/data/graphs/. For each:
1. Load and check x.shape[1]: 12=v4 (correct), 8=v1/v5.0 (STALE), other=unexpected
2. For stale graphs: cross-reference with deduped CSV — are they in the training set?
3. Check which split they're in (look for split files in ml/data/splits/)
4. Test what DualPathDataset would do: would it crash or silently feed wrong features?
   - Simulate by checking if torch.cat([8-dim, 12-dim]) would happen in collate

Report: count of stale graphs, which are in CSV, which split, runtime impact assessment.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def main():
    print_header("Task 26: Stale v5 Contamination Audit")

    # ── Collect ALL graph files ─────────────────────────────────────────────
    if not GRAPHS_DIR.exists():
        print(f"ERROR: Graphs directory not found: {GRAPHS_DIR}")
        save_report("task26_stale_v5_contamination",
                     "# Task 26: Stale v5 Contamination Audit\n\n"
                     "**ERROR**: Graphs directory not found. No data to audit.\n")
        return

    all_graph_files = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_graph_files:
        print("ERROR: No graph .pt files found in", GRAPHS_DIR)
        save_report("task26_stale_v5_contamination",
                     "# Task 26: Stale v5 Contamination Audit\n\n"
                     "**ERROR**: No .pt files found.\n")
        return

    print(f"Found {len(all_graph_files)} graph files to scan")

    # ── Load label CSV for cross-referencing ────────────────────────────────
    csv_stems = set()
    if LABEL_CSV.exists():
        try:
            labels = load_label_csv()
            csv_stems = set(labels.keys())
            print(f"Loaded {len(csv_stems)} stems from deduped CSV")
        except Exception as e:
            print(f"WARNING: Could not load label CSV: {e}")
    else:
        print(f"WARNING: Label CSV not found at {LABEL_CSV}")

    # ── Load split indices ──────────────────────────────────────────────────
    splits_dir = PROJECT_ROOT / "ml" / "data" / "splits"
    split_stems = {"train": set(), "val": set(), "test": set()}
    if splits_dir.exists():
        try:
            # Splits are index-based into the sorted CSV rows.
            # We need to load the CSV, sort it the same way, and map indices to stems.
            if LABEL_CSV.exists():
                rows = load_label_csv_as_rows()
                sorted_stems = sorted(r["md5_stem"] for r in rows)
                for split_name in ["train", "val", "test"]:
                    npy_path = splits_dir / f"{split_name}_indices.npy"
                    if npy_path.exists():
                        indices = np.load(npy_path)
                        split_stems[split_name] = {sorted_stems[i] for i in indices if i < len(sorted_stems)}
                        print(f"  {split_name}: {len(split_stems[split_name])} stems from {npy_path.name}")
                    else:
                        print(f"  {split_name}: split file not found at {npy_path}")
        except Exception as e:
            print(f"WARNING: Could not load splits: {e}")
    else:
        print(f"WARNING: Splits directory not found at {splits_dir}")

    # ── Scan all graph files ────────────────────────────────────────────────
    v4_count = 0
    stale_v5_count = 0
    unexpected_dim_count = 0
    unexpected_dims = Counter()
    skipped = 0

    stale_in_csv = []
    stale_split_counts = Counter()
    stale_details = []

    for i, fpath in enumerate(all_graph_files):
        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i + 1}/{len(all_graph_files)} files...")

        try:
            data = load_graph(fpath)
        except Exception as e:
            skipped += 1
            continue

        stem = fpath.stem
        x = data.x

        if x is None or x.dim() != 2:
            skipped += 1
            continue

        dim = x.shape[1]
        if dim == 12:
            v4_count += 1
        elif dim == 8:
            stale_v5_count += 1

            # Cross-reference with CSV
            in_csv = stem in csv_stems

            # Check which split
            split_name = "unknown"
            for sname, stems in split_stems.items():
                if stem in stems:
                    split_name = sname
                    break

            stale_split_counts[split_name] += 1

            detail = {
                "stem": stem,
                "dim": dim,
                "in_csv": in_csv,
                "split": split_name,
                "nodes": x.shape[0],
            }
            stale_details.append(detail)

            if in_csv:
                stale_in_csv.append(stem)
        else:
            unexpected_dim_count += 1
            unexpected_dims[dim] += 1

    print(f"\n  Scan complete: {len(all_graph_files)} files")
    print(f"    v4 (12-dim):  {v4_count}")
    print(f"    stale (8-dim): {stale_v5_count}")
    print(f"    unexpected:    {unexpected_dim_count}")
    print(f"    skipped:       {skipped}")

    # ── Simulate DualPathDataset collate ────────────────────────────────────
    print("\n--- Simulating DualPathDataset collate behavior ---")
    crash_scenarios = 0
    silent_wrong_scenarios = 0

    if stale_v5_count > 0 and v4_count > 0:
        # Test: what happens when an 8-dim graph and 12-dim graph end up
        # in the same batch? Batch.from_data_list() merges them.
        # PyG Batch stacks x tensors — if dims differ, it will raise RuntimeError.
        print("  Testing torch.cat with mismatched dimensions...")
        try:
            fake_x_8 = torch.randn(5, 8)
            fake_x_12 = torch.randn(3, 12)
            # Simulate what Batch.from_data_list does internally
            _ = torch.cat([fake_x_8, fake_x_12], dim=0)
            silent_wrong_scenarios += 1
            print("  CRITICAL: torch.cat SUCCEEDED with mismatched dims — silent feature misalignment!")
        except RuntimeError as e:
            crash_scenarios += 1
            print(f"  torch.cat FAILS with mismatched dims (expected): {e}")

        # Also test what actually happens with PyG Batch
        try:
            from torch_geometric.data import Data, Batch
            g8 = Data(x=torch.randn(5, 8), edge_index=torch.randint(0, 5, (2, 4)),
                       edge_attr=torch.zeros(4, dtype=torch.long))
            g12 = Data(x=torch.randn(3, 12), edge_index=torch.randint(0, 3, (2, 2)),
                        edge_attr=torch.zeros(2, dtype=torch.long))
            # The collate excludes y and other metadata keys
            batch = Batch.from_data_list([g8, g12],
                                          exclude_keys=["contract_hash", "contract_path", "y"])
            silent_wrong_scenarios += 1
            print(f"  CRITICAL: PyG Batch.from_data_list SUCCEEDED with mixed dims! x.shape={batch.x.shape}")
            print("    This means the model gets misaligned features without any error!")
        except (RuntimeError, Exception) as e:
            crash_scenarios += 1
            print(f"  PyG Batch.from_data_list FAILS with mixed dims (safe crash): {type(e).__name__}: {e}")
    elif stale_v5_count > 0:
        print("  Only stale graphs exist — all 8-dim. No dimension mismatch but WRONG features.")
    else:
        print("  No stale graphs found — no contamination risk from dimension mismatch.")

    # ── Build report ────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 26: Stale v5 Contamination Audit\n\n")
    report_lines.append(f"**Total graph files scanned:** {len(all_graph_files)}  \n")
    report_lines.append(f"**v4 (12-dim, correct):** {v4_count}  \n")
    report_lines.append(f"**Stale v1/v5.0 (8-dim):** {stale_v5_count}  \n")
    report_lines.append(f"**Unexpected dims:** {unexpected_dim_count}  \n")
    report_lines.append(f"**Skipped (load errors):** {skipped}\n\n")

    # Stale graph summary
    report_lines.append("## Stale Graph Summary\n\n")
    if stale_v5_count == 0:
        report_lines.append("**No stale v5 graphs found.** Dataset is clean.\n\n")
    else:
        report_lines.append(f"Found **{stale_v5_count}** stale 8-dim graph(s) out of {len(all_graph_files)} total "
                            f"({100*stale_v5_count/len(all_graph_files):.2f}%).\n\n")

        # CSV cross-reference
        report_lines.append("### CSV Cross-Reference\n\n")
        report_lines.append(f"- Stale graphs in deduped CSV: **{len(stale_in_csv)}** / {stale_v5_count}\n")
        report_lines.append(f"- Stale graphs NOT in CSV: **{stale_v5_count - len(stale_in_csv)}**\n\n")

        # Split distribution
        report_lines.append("### Split Distribution of Stale Graphs\n\n")
        report_lines.append("| Split | Count |\n")
        report_lines.append("|-------|-------|\n")
        for split_name in ["train", "val", "test", "unknown"]:
            cnt = stale_split_counts.get(split_name, 0)
            if cnt > 0:
                report_lines.append(f"| {split_name} | {cnt} |\n")
        report_lines.append("\n")

        # Impact on training
        stale_in_train = stale_split_counts.get("train", 0)
        stale_in_val = stale_split_counts.get("val", 0)
        stale_in_test = stale_split_counts.get("test", 0)
        report_lines.append("### Training Impact\n\n")
        if stale_in_train > 0:
            report_lines.append(f"- **CRITICAL**: {stale_in_train} stale graphs in **train** split — "
                                "model trains on wrong features (8-dim vs 12-dim)\n")
        if stale_in_val > 0:
            report_lines.append(f"- **HIGH**: {stale_in_val} stale graphs in **val** split — "
                                "validation metrics unreliable\n")
        if stale_in_test > 0:
            report_lines.append(f"- **HIGH**: {stale_in_test} stale graphs in **test** split — "
                                "test metrics unreliable\n")
        report_lines.append("\n")

    # Runtime impact
    report_lines.append("## Runtime Impact Assessment\n\n")
    if stale_v5_count == 0:
        report_lines.append("No runtime impact — no stale graphs present.\n\n")
    elif v4_count == 0:
        report_lines.append("ALL graphs are stale (8-dim). The model would train on wrong features, "
                            "but at least no dimension mismatch crashes would occur since all graphs "
                            "are uniform.\n\n")
    else:
        report_lines.append("### Mixed Dimension Scenario\n\n")
        if crash_scenarios > 0:
            report_lines.append(f"- **Crash risk**: HIGH — `torch.cat` / `Batch.from_data_list` raises "
                                "RuntimeError when 8-dim and 12-dim graphs are in the same batch. "
                                "Training will **crash** rather than silently produce wrong results.\n")
        if silent_wrong_scenarios > 0:
            report_lines.append(f"- **Silent misalignment risk**: CRITICAL — collate succeeds but "
                                "feature columns are misaligned. The model receives garbage features "
                                "for stale graphs without any error signal.\n")
        report_lines.append("\n")

    # First 50 stale graph details
    if stale_details:
        report_lines.append("## Stale Graph Details (first 50)\n\n")
        report_lines.append("| # | Stem (first 16 chars) | In CSV | Split | Nodes |\n")
        report_lines.append("|---|----------------------|--------|-------|-------|\n")
        for idx, d in enumerate(stale_details[:50]):
            report_lines.append(
                f"| {idx+1} | {d['stem'][:16]}... | {'Yes' if d['in_csv'] else 'No'} | "
                f"{d['split']} | {d['nodes']} |\n"
            )
        if len(stale_details) > 50:
            report_lines.append(f"\n... and {len(stale_details) - 50} more stale graphs\n")

    # Unexpected dimensions
    if unexpected_dims:
        report_lines.append("\n## Unexpected Feature Dimensions\n\n")
        report_lines.append("| Dim | Count |\n")
        report_lines.append("|-----|-------|\n")
        for dim, cnt in sorted(unexpected_dims.items()):
            label = "v1/v5.0 (stale)" if dim == 8 else "v4 (correct)" if dim == 12 else "UNKNOWN"
            report_lines.append(f"| {dim} | {cnt} ({label}) |\n")

    # Recommendation
    report_lines.append("\n## Recommendation\n\n")
    if stale_v5_count > 0:
        report_lines.append("1. **Re-extract** all stale 8-dim graphs using: "
                            "`python ml/scripts/reextract_graphs.py`\n")
        report_lines.append("2. **Delete** stale .pt files before re-extraction if needed\n")
        report_lines.append("3. **Rebuild** the RAM cache after re-extraction: "
                            "`python ml/scripts/create_cache.py`\n")
        report_lines.append("4. **Retrain** the model from scratch after fixing the dataset\n")
    else:
        report_lines.append("No action needed — all graphs use the correct v4 (12-dim) schema.\n")

    report_content = "".join(report_lines)
    save_report("task26_stale_v5_contamination", report_content)
    print_header("Task 26 Complete")


if __name__ == "__main__":
    main()
