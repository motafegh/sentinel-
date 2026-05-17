#!/usr/bin/env python3
"""
task20_dos_reentrancy_separability.py — DoS vs Reentrancy Separability Audit for SENTINEL v6

1. Find ALL DoS=1 contracts from CSV. Split into DoS+Reentrancy vs DoS-only
2. For DoS-only: read .sol source, load graphs, analyze what makes them DoS but NOT Reentrancy
3. For 10 DoS+Reentrancy: read .sol and check if both patterns exist
4. Compute per-contract feature aggregates for DoS-only vs DoS+Reentrancy
5. Check which split DoS-only contracts are in
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


# ── DoS and Reentrancy pattern detection ──────────────────────────────────────
_DOS_PATTERNS = {
    "loop_with_call":       re.compile(r'(for|while)\s*\([^)]*\)[^{]*\{[^}]*(?:\.call|\.transfer|\.send)\b', re.DOTALL),
    "loop_with_transfer":   re.compile(r'(for|while)\s*\([^)]*\)[^{]*\{[^}]*\.transfer\s*\(', re.DOTALL),
    "loop_with_send":       re.compile(r'(for|while)\s*\([^)]*\)[^{]*\{[^}]*\.send\s*\(', re.DOTALL),
    "unbounded_loop":       re.compile(r'(for|while)\s*\([^)]*\)[^{]*\{', re.DOTALL),
    "external_call":        re.compile(r'\.(call|transfer|send)\s*\('),
    "revert_in_loop":       re.compile(r'(for|while)\s*\([^)]*\)[^{]*\{[^}]*revert\b', re.DOTALL),
}

_REENTRANCY_PATTERNS = {
    "external_call_before_write": re.compile(
        r'\.(call|transfer|send)\s*\([^)]*\)[^;]*;[^}]*\w+\s*=',
        re.DOTALL
    ),
    "low_level_call":     re.compile(r'\.call\s*[\({]'),
    "delegate_call":      re.compile(r'\.delegatecall\s*[\({]'),
    "msg_sender_call":    re.compile(r'msg\.sender\.(call|transfer|send)'),
    "state_after_call":   re.compile(r'\.(call|delegatecall)\s*[\({][^;]*;[^}]*\w+\s*[\+\-]?='),
}


def detect_dos_patterns(source: str) -> list:
    """Return list of matched DoS pattern names."""
    matched = []
    for name, pattern in _DOS_PATTERNS.items():
        if pattern.search(source):
            matched.append(name)
    return matched


def detect_reentrancy_patterns(source: str) -> list:
    """Return list of matched Reentrancy pattern names."""
    matched = []
    for name, pattern in _REENTRANCY_PATTERNS.items():
        if pattern.search(source):
            matched.append(name)
    return matched


def compute_graph_features(g) -> dict:
    """
    Compute per-contract feature aggregates from a graph.

    Returns dict with: mean_has_loop, mean_ext_call_count, mean_complexity,
    graph_size, n_nodes, n_edges, n_cfg_nodes, n_function_nodes
    """
    try:
        x = g.x
        if x is None or x.dim() != 2 or x.shape[1] < 12:
            return {}

        x_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)

        n_nodes = x_np.shape[0]
        n_edges = g.edge_index.shape[1] if g.edge_index is not None else 0

        # Feature indices (v4 schema):
        # [0] type_id, [5] complexity, [10] has_loop, [11] external_call_count
        type_ids = x_np[:, 0]
        complexity = x_np[:, 5]
        has_loop = x_np[:, 10]
        ext_call_count = x_np[:, 11]

        # Count CFG nodes (type_id >= 8/12)
        cfg_mask = type_ids >= (8.0 / 12.0)
        n_cfg_nodes = int(np.sum(cfg_mask))

        # Count function nodes (type_id ≈ 1/12, 6/12 for constructor, 4/12 for fallback, 5/12 for receive)
        func_type_ids = {1.0/12.0, 4.0/12.0, 5.0/12.0, 6.0/12.0}
        n_function_nodes = sum(1 for t in type_ids if any(abs(t - ft) < 0.01 for ft in func_type_ids))

        # Mean features over function-like nodes only
        func_mask = np.zeros(n_nodes, dtype=bool)
        for i, t in enumerate(type_ids):
            if any(abs(t - ft) < 0.01 for ft in func_type_ids):
                func_mask[i] = True

        if np.any(func_mask):
            mean_has_loop = float(np.mean(has_loop[func_mask]))
            mean_ext_call = float(np.mean(ext_call_count[func_mask]))
            mean_complexity = float(np.mean(complexity[func_mask]))
        else:
            mean_has_loop = 0.0
            mean_ext_call = 0.0
            mean_complexity = 0.0

        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_cfg_nodes": n_cfg_nodes,
            "n_function_nodes": n_function_nodes,
            "mean_has_loop": mean_has_loop,
            "mean_ext_call_count": mean_ext_call,
            "mean_complexity": mean_complexity,
            "graph_size": n_nodes + n_edges,
        }
    except Exception:
        return {}


def check_split_membership(stems: list) -> dict:
    """Check which train/val/test split each stem belongs to."""
    processed_dir = PROJECT_ROOT / "ml" / "data" / "processed"
    split_map = {}  # stem → split_name

    for split_name in ["train", "val", "test"]:
        for suffix in [f"{split_name}_stems.txt", f"split_{split_name}.txt",
                       f"{split_name}.csv"]:
            p = processed_dir / suffix
            if p.exists():
                try:
                    if p.suffix == ".csv":
                        import csv
                        with open(p, newline="", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                stem = row.get("md5_stem", row.get("stem", ""))
                                if stem:
                                    split_map[stem] = split_name
                    else:
                        for line in p.read_text().splitlines():
                            stem = line.strip()
                            if stem:
                                split_map[stem] = split_name
                except Exception:
                    pass
                break

    result = {"train": [], "val": [], "test": [], "unknown": []}
    for stem in stems:
        split = split_map.get(stem, "unknown")
        result[split].append(stem)
    return result


def main():
    print_header("Task 20: DoS vs Reentrancy Separability Audit")

    # ── Load labels ───────────────────────────────────────────────────────
    try:
        labels = load_label_csv()
    except Exception as e:
        print(f"ERROR: Could not load label CSV: {e}")
        return

    # ── Find DoS=1 contracts ──────────────────────────────────────────────
    dos_stems = [
        stem for stem, lbls in labels.items()
        if lbls.get("DenialOfService", 0) == 1
    ]
    reentrancy_stems_set = {
        stem for stem, lbls in labels.items()
        if lbls.get("Reentrancy", 0) == 1
    }

    dos_and_re = [s for s in dos_stems if s in reentrancy_stems_set]
    dos_only = [s for s in dos_stems if s not in reentrancy_stems_set]

    print(f"Total DoS=1 contracts: {len(dos_stems)}")
    print(f"DoS + Reentrancy: {len(dos_and_re)}")
    print(f"DoS-only: {len(dos_only)}")

    # ── Build md5→path ────────────────────────────────────────────────────
    all_dos_md5s = set(dos_stems)
    print("Building md5→path mapping...")
    md5_to_path = build_md5_to_path(all_dos_md5s)
    print(f"Resolved {len(md5_to_path)}/{len(all_dos_md5s)} paths")

    # ── Analyze DoS-only contracts ────────────────────────────────────────
    print("\nAnalyzing DoS-only contracts...")
    dos_only_results = []
    dos_only_features = []

    for i, stem in enumerate(dos_only):
        if (i + 1) % 10 == 0:
            print(f"  DoS-only: {i + 1}/{len(dos_only)}")

        result = {"stem": stem, "source_patterns": [], "graph_features": {}}

        sol_path = md5_to_path.get(stem)
        if sol_path and sol_path.exists():
            try:
                source = sol_path.read_text(encoding="utf-8", errors="replace")
                result["source_patterns"] = detect_dos_patterns(source)
                result["reentrancy_patterns"] = detect_reentrancy_patterns(source)
                result["sol_name"] = sol_path.name
            except OSError:
                result["source_patterns"] = ["read_error"]
                result["reentrancy_patterns"] = []
        else:
            result["source_patterns"] = ["no_source"]
            result["reentrancy_patterns"] = []

        # Load graph
        graph_path = GRAPHS_DIR / f"{stem}.pt"
        if graph_path.exists():
            try:
                g = load_graph(graph_path)
                features = compute_graph_features(g)
                result["graph_features"] = features
                result["contract_name"] = getattr(g, "contract_name", "N/A")
                if features:
                    dos_only_features.append(features)
            except Exception as e:
                result["graph_features"] = {"error": str(e)}
        else:
            result["graph_features"] = {"no_graph": True}

        dos_only_results.append(result)

    # ── Analyze DoS+Reentrancy contracts (sample of 10) ──────────────────
    print("\nAnalyzing DoS+Reentrancy sample...")
    import random
    random.seed(42)
    dr_sample = random.sample(dos_and_re, min(10, len(dos_and_re)))
    dr_results = []
    dr_features = []

    for i, stem in enumerate(dr_sample):
        print(f"  DoS+Re: {i + 1}/{len(dr_sample)}")

        result = {"stem": stem, "source_patterns": [], "reentrancy_patterns": []}

        sol_path = md5_to_path.get(stem)
        if sol_path and sol_path.exists():
            try:
                source = sol_path.read_text(encoding="utf-8", errors="replace")
                result["source_patterns"] = detect_dos_patterns(source)
                result["reentrancy_patterns"] = detect_reentrancy_patterns(source)
                result["sol_name"] = sol_path.name
            except OSError:
                result["source_patterns"] = ["read_error"]
                result["reentrancy_patterns"] = []
        else:
            result["source_patterns"] = ["no_source"]
            result["reentrancy_patterns"] = []

        graph_path = GRAPHS_DIR / f"{stem}.pt"
        if graph_path.exists():
            try:
                g = load_graph(graph_path)
                features = compute_graph_features(g)
                result["graph_features"] = features
                result["contract_name"] = getattr(g, "contract_name", "N/A")
                if features:
                    dr_features.append(features)
            except Exception as e:
                result["graph_features"] = {"error": str(e)}
        else:
            result["graph_features"] = {"no_graph": True}

        dr_results.append(result)

    # ── Compute aggregate feature comparison ──────────────────────────────
    def feature_agg(features_list):
        """Compute mean and std for feature dicts."""
        if not features_list:
            return {}
        keys = features_list[0].keys()
        agg = {}
        for key in keys:
            vals = [f[key] for f in features_list if key in f and isinstance(f[key], (int, float))]
            if vals:
                agg[f"{key}_mean"] = np.mean(vals)
                agg[f"{key}_std"] = np.std(vals) if len(vals) > 1 else 0.0
        return agg

    dos_only_agg = feature_agg(dos_only_features)
    dr_agg = feature_agg(dr_features)

    # ── Check split membership ────────────────────────────────────────────
    print("\nChecking split membership for DoS-only contracts...")
    dos_only_splits = check_split_membership(dos_only)

    # ── Build report ──────────────────────────────────────────────────────
    report = []
    report.append("# Task 20: DoS vs Reentrancy Separability Audit\n")
    report.append(f"**Total DoS=1 contracts:** {len(dos_stems)}  ")
    report.append(f"**DoS + Reentrancy:** {len(dos_and_re)}  ")
    report.append(f"**DoS-only:** {len(dos_only)}\n")

    # Source pattern analysis for DoS-only
    report.append("\n## DoS-Only: Source Pattern Analysis\n")
    dos_pattern_counts = Counter()
    re_pattern_in_dos_only = Counter()
    for r in dos_only_results:
        for p in r.get("source_patterns", []):
            dos_pattern_counts[p] += 1
        for p in r.get("reentrancy_patterns", []):
            re_pattern_in_dos_only[p] += 1

    report.append("### DoS Patterns Found\n")
    report.append("| Pattern | Count | Percentage |")
    report.append("\n|---------|-------|------------|")
    for pattern_name, count in dos_pattern_counts.most_common():
        pct = f"{count/len(dos_only):.1%}" if len(dos_only) > 0 else "N/A"
        report.append(f"\n| {pattern_name} | {count} | {pct} |")

    report.append("\n\n### Reentrancy Patterns Also Present in DoS-Only (Source)\n")
    report.append("These DoS-only contracts have reentrancy-like patterns in source "
                  "but are NOT labelled Reentrancy=1:\n")
    report.append("| Pattern | Count |")
    report.append("\n|---------|-------|")
    for pattern_name, count in re_pattern_in_dos_only.most_common():
        report.append(f"\n| {pattern_name} | {count} |")

    if sum(re_pattern_in_dos_only.values()) > 0:
        report.append("\n\n⚠️ Some DoS-only contracts exhibit reentrancy-like patterns in source. "
                      "This suggests potential label inconsistency or that the reentrancy "
                      "pattern is present but not exploitable (e.g., state updates before call).")

    # DoS+Reentrancy sample analysis
    report.append("\n\n## DoS+Reentrancy: Sample Analysis (10 contracts)\n")
    for r in dr_results:
        stem_short = r["stem"][:12]
        dos_pats = ", ".join(r.get("source_patterns", [])) or "none detected"
        re_pats = ", ".join(r.get("reentrancy_patterns", [])) or "none detected"
        report.append(f"- **`{stem_short}...`**: DoS patterns=[{dos_pats}], "
                      f"Reentrancy patterns=[{re_pats}]")

    report.append("\n\n### Interpretation\n")
    both_present = sum(1 for r in dr_results
                       if r.get("source_patterns") and r.get("reentrancy_patterns"))
    report.append(f"- {both_present}/{len(dr_results)} DoS+Reentrancy contracts show both "
                  "pattern types in source (expected for dual-labelled).")

    # Feature comparison table
    report.append("\n\n## Feature Comparison: DoS-Only vs DoS+Reentrancy\n")
    feature_keys = ["n_nodes", "n_edges", "n_cfg_nodes", "n_function_nodes",
                    "mean_has_loop", "mean_ext_call_count", "mean_complexity", "graph_size"]

    report.append("| Feature | DoS-Only (mean±std) | DoS+Re (mean±std) | Difference |")
    report.append("\n|---------|--------------------|-------------------|------------|")

    for key in feature_keys:
        do_mean = dos_only_agg.get(f"{key}_mean", 0)
        do_std = dos_only_agg.get(f"{key}_std", 0)
        dr_mean = dr_agg.get(f"{key}_mean", 0)
        dr_std = dr_agg.get(f"{key}_std", 0)
        diff = do_mean - dr_mean
        report.append(f"\n| {key} | {do_mean:.2f}±{do_std:.2f} | {dr_mean:.2f}±{dr_std:.2f} | {diff:+.2f} |")

    # Statistical note
    report.append("\n\n*Note: DoS-only has limited sample size; statistics may not be robust.*")

    # Split membership
    report.append("\n\n## DoS-Only Contract Split Distribution\n")
    report.append("| Split | Count |")
    report.append("\n|-------|-------|")
    for split_name in ["train", "val", "test", "unknown"]:
        count = len(dos_only_splits.get(split_name, []))
        report.append(f"\n| {split_name} | {count} |")

    if dos_only_splits.get("train"):
        report.append(f"\n⚠️ {len(dos_only_splits['train'])} DoS-only contracts are in the "
                      "training set. If these are inherently ambiguous with Reentrancy, "
                      "they may confuse the model during training.")

    # Individual DoS-only contract details
    report.append("\n\n## DoS-Only Contract Details\n")
    report.append("| # | Stem | Contract | DoS Patterns | Re Patterns | n_nodes | mean_loop | mean_ext_call |")
    report.append("\n|---|------|----------|-------------|-------------|---------|-----------|---------------|")
    for i, r in enumerate(dos_only_results[:30]):
        stem_short = r["stem"][:12]
        contract = r.get("contract_name", "N/A")
        dos_pats = ", ".join(r.get("source_patterns", [])[:3]) or "-"
        re_pats = ", ".join(r.get("reentrancy_patterns", [])[:3]) or "-"
        gf = r.get("graph_features", {})
        n_nodes = gf.get("n_nodes", "N/A")
        mean_loop = f"{gf.get('mean_has_loop', 0):.2f}" if isinstance(gf.get('mean_has_loop'), float) else "N/A"
        mean_ext = f"{gf.get('mean_ext_call_count', 0):.2f}" if isinstance(gf.get('mean_ext_call_count'), float) else "N/A"
        report.append(f"\n| {i+1} | {stem_short} | {contract} | {dos_pats} | {re_pats} | {n_nodes} | {mean_loop} | {mean_ext} |")

    if len(dos_only_results) > 30:
        report.append(f"\n| ... | ... | ... | ... | ... | ... | ... | ... |")
        report.append(f"\n*({len(dos_only_results) - 30} more contracts omitted)*")

    # Recommendation
    report.append("\n\n## Recommendation\n")

    if len(dos_only) < 15:
        report.append(f"**DoS-only contracts are extremely rare** ({len(dos_only)} out of {len(dos_stems)} "
                      f"total DoS=1). This means:")
        report.append("\n1. **Keep separate**: The classes are largely overlapping — most DoS contracts "
                      "are also Reentrancy. Merging would dilute the DoS signal.")
        report.append("2. **Augment**: Consider generating synthetic DoS-only examples (e.g., "
                      "loop-with-transfer without reentrancy pattern) to give the model "
                      "more training signal for DoS-specific patterns.")
    else:
        report.append(f"DoS-only contracts represent {len(dos_only)/len(dos_stems):.0%} "
                      f"of all DoS=1 ({len(dos_only)}/{len(dos_stems)}).")

        # Check feature separation
        if dos_only_agg and dr_agg:
            loop_diff = abs(dos_only_agg.get("mean_has_loop_mean", 0) -
                           dr_agg.get("mean_has_loop_mean", 0))
            ext_diff = abs(dos_only_agg.get("mean_ext_call_count_mean", 0) -
                          dr_agg.get("mean_ext_call_count_mean", 0))

            if loop_diff > 0.1 or ext_diff > 0.1:
                report.append("\nThere is **some feature separation** between DoS-only and "
                              "DoS+Reentrancy groups (particularly in has_loop and external_call_count). "
                              "This suggests the model can potentially learn to distinguish them.")
                report.append("\n**Recommendation: Keep separate** — the model can benefit from "
                              "the distinction, and the separate labels capture different vulnerability semantics.")
            else:
                report.append("\nThere is **minimal feature separation** between DoS-only and "
                              "DoS+Reentrancy groups. The structural graph features are similar.")
                report.append("\n**Recommendation: Consider merging** DoS into Reentrancy as a sub-class, "
                              "or adding a joint DoS+Reentrancy label. The current separate labels "
                              "may be difficult for the model to distinguish without additional features.")

    save_report("task20_dos_reentrancy_separability", "".join(report))
    print_header("Task 20 Complete")


if __name__ == "__main__":
    main()
