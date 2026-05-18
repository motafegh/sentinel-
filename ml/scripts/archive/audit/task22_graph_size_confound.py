#!/usr/bin/env python3
"""
task22_graph_size_confound.py — Graph Size Confound Audit

Load 5000 graph .pt files (or all if feasible). For each: record num_nodes,
num_edges, num_cf_edges, num_functions (count type_id==1/12≈0.083), all 10 labels.
Compute:
1. Per-class stats: mean/median/p25/p75/p95 of num_nodes, num_edges, num_functions
2. Train a simple logistic regression per class using only [num_nodes, num_edges,
   num_functions] → AUC-ROC
3. Within each class, compare graph size between positive and negative samples
   (Mann-Whitney U test)
4. Compare label distributions between top-25% and bottom-25% graphs by size
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def _percentile(arr, q):
    """Compute percentile, handling empty arrays."""
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, q))


def main():
    print_header("Task 22: Graph Size Confound Audit")

    # ── Collect graph files ────────────────────────────────────────────────
    all_graph_files = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_graph_files:
        print("ERROR: No graph .pt files found in", GRAPHS_DIR)
        return
    print(f"Found {len(all_graph_files)} graph files")

    sample_size = min(5000, len(all_graph_files))
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(len(all_graph_files), size=sample_size, replace=False)
    sampled_files = [all_graph_files[i] for i in sampled_indices]
    print(f"Sampling {sample_size} files")

    # ── Load data ──────────────────────────────────────────────────────────
    print("  Loading labels...")
    labels = load_label_csv()

    records = []  # list of dicts
    skipped = 0

    for i, fpath in enumerate(sampled_files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{sample_size} files...")

        try:
            data = load_graph(fpath)
        except Exception as e:
            skipped += 1
            print(f"  WARNING: Could not load {fpath.stem}: {e}")
            continue

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if x is None or edge_index is None:
            skipped += 1
            continue

        stem = fpath.stem
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        # Count functions (type_id == 1/12 exactly; range filter avoids counting
        # CONTRACT nodes at 0/12 or other node types at 2/12+)
        x_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)
        if x_np.ndim == 2 and x_np.shape[1] >= 1:
            type_ids = x_np[:, 0]
            num_functions = int(np.sum((type_ids >= 1.0 / 12.0) & (type_ids < 2.0 / 12.0)))
        else:
            num_functions = 0

        # Count CF edges (edge_type 6)
        if edge_attr is not None:
            ea_np = edge_attr.numpy() if hasattr(edge_attr, 'numpy') else np.array(edge_attr)
            if ea_np.ndim > 1:
                ea_np = ea_np.squeeze()
            num_cf_edges = int(np.sum(ea_np == 6))
        else:
            num_cf_edges = 0

        # Get labels
        label_vals = {}
        if stem in labels:
            label_vals = labels[stem]
        else:
            skipped += 1
            continue

        records.append({
            "stem": stem,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_functions": num_functions,
            "num_cf_edges": num_cf_edges,
            **label_vals,
        })

    print(f"  Successfully processed: {len(records)}")
    print(f"  Skipped: {skipped}")

    if len(records) == 0:
        print("ERROR: No valid records.")
        return

    # ── Build arrays ───────────────────────────────────────────────────────
    num_nodes_arr = np.array([r["num_nodes"] for r in records])
    num_edges_arr = np.array([r["num_edges"] for r in records])
    num_funcs_arr = np.array([r["num_functions"] for r in records])
    num_cf_arr = np.array([r["num_cf_edges"] for r in records])

    size_metrics = {
        "num_nodes": num_nodes_arr,
        "num_edges": num_edges_arr,
        "num_functions": num_funcs_arr,
    }

    label_arrays = {}
    for cls in VULN_CLASSES:
        label_arrays[cls] = np.array([r[cls] for r in records], dtype=int)

    # ── 1. Per-class size stats ────────────────────────────────────────────
    print("  Computing per-class size statistics...")
    per_class_stats = {}
    for cls in VULN_CLASSES:
        pos_mask = label_arrays[cls] == 1
        neg_mask = label_arrays[cls] == 0
        per_class_stats[cls] = {
            "pos_count": int(pos_mask.sum()),
            "neg_count": int(neg_mask.sum()),
            "pos_num_nodes": num_nodes_arr[pos_mask],
            "neg_num_nodes": num_nodes_arr[neg_mask],
            "pos_num_edges": num_edges_arr[pos_mask],
            "neg_num_edges": num_edges_arr[neg_mask],
            "pos_num_functions": num_funcs_arr[pos_mask],
            "neg_num_functions": num_funcs_arr[neg_mask],
        }

    # ── 2. Logistic regression AUC per class ───────────────────────────────
    print("  Training logistic regression per class...")
    auc_results = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        X_size = np.column_stack([num_nodes_arr, num_edges_arr, num_funcs_arr])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_size)

        for cls in VULN_CLASSES:
            y = label_arrays[cls]
            if y.sum() < 5 or (1 - y).sum() < 5:
                auc_results[cls] = None
                continue
            try:
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_scaled, y)
                y_prob = lr.predict_proba(X_scaled)[:, 1]
                auc_val = roc_auc_score(y, y_prob)
                auc_results[cls] = auc_val
            except Exception as e:
                auc_results[cls] = None
                print(f"    WARNING: LR failed for {cls}: {e}")
    except ImportError:
        print("  WARNING: sklearn not available — AUC computation skipped")
    except Exception as e:
        print(f"  WARNING: AUC computation failed: {e}")

    # ── 3. Mann-Whitney U test ─────────────────────────────────────────────
    print("  Running Mann-Whitney U tests...")
    mw_results = {}
    try:
        from scipy.stats import mannwhitneyu

        for cls in VULN_CLASSES:
            stats = per_class_stats[cls]
            mw_results[cls] = {}
            for metric_name in ["num_nodes", "num_edges", "num_functions"]:
                pos_vals = stats[f"pos_{metric_name}"]
                neg_vals = stats[f"neg_{metric_name}"]
                if len(pos_vals) > 0 and len(neg_vals) > 0:
                    try:
                        u_stat, p_val = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
                        mw_results[cls][metric_name] = {
                            "u_stat": float(u_stat),
                            "p_value": float(p_val),
                            "pos_median": float(np.median(pos_vals)),
                            "neg_median": float(np.median(neg_vals)),
                            "pos_mean": float(np.mean(pos_vals)),
                            "neg_mean": float(np.mean(neg_vals)),
                        }
                    except Exception:
                        mw_results[cls][metric_name] = None
                else:
                    mw_results[cls][metric_name] = None
    except ImportError:
        print("  WARNING: scipy not available — Mann-Whitney U test skipped")
    except Exception as e:
        print(f"  WARNING: Mann-Whitney computation failed: {e}")

    # ── 4. Label distribution by size quartiles ────────────────────────────
    print("  Comparing label distributions by size quartiles...")
    total_size = num_nodes_arr + num_edges_arr  # composite size metric
    p25 = np.percentile(total_size, 25)
    p75 = np.percentile(total_size, 75)
    bottom_mask = total_size <= p25
    top_mask = total_size >= p75

    quartile_comparison = {}
    for cls in VULN_CLASSES:
        bottom_pos_rate = label_arrays[cls][bottom_mask].mean() if bottom_mask.sum() > 0 else 0.0
        top_pos_rate = label_arrays[cls][top_mask].mean() if top_mask.sum() > 0 else 0.0
        quartile_comparison[cls] = {
            "bottom_n": int(bottom_mask.sum()),
            "top_n": int(top_mask.sum()),
            "bottom_pos_rate": float(bottom_pos_rate),
            "top_pos_rate": float(top_pos_rate),
            "diff": float(top_pos_rate - bottom_pos_rate),
        }

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 22: Graph Size Confound Audit\n\n")
    report_lines.append(f"**Sample size:** {len(records)}  \n")
    report_lines.append(f"**Skipped:** {skipped}\n\n")

    # 1. Per-class size stats
    report_lines.append("## 1. Per-Class Graph Size Statistics\n\n")
    for metric_name in ["num_nodes", "num_edges", "num_functions"]:
        report_lines.append(f"### {metric_name}\n\n")
        report_lines.append(
            "| Class | Pos Count | Neg Count | "
            "Pos Mean | Pos Median | Pos P25 | Pos P75 | Pos P95 | "
            "Neg Mean | Neg Median | Neg P25 | Neg P75 | Neg P95 |\n"
        )
        report_lines.append(
            "|-------|-----------|-----------|"
            "----------|------------|---------|---------|---------|"
            "----------|------------|---------|---------|----------|\n"
        )
        for cls in VULN_CLASSES:
            s = per_class_stats[cls]
            pos_v = s[f"pos_{metric_name}"]
            neg_v = s[f"neg_{metric_name}"]
            if len(pos_v) > 0 and len(neg_v) > 0:
                report_lines.append(
                    f"| {cls} | {s['pos_count']} | {s['neg_count']} | "
                    f"{np.mean(pos_v):.1f} | {np.median(pos_v):.0f} | "
                    f"{_percentile(pos_v, 25):.0f} | {_percentile(pos_v, 75):.0f} | "
                    f"{_percentile(pos_v, 95):.0f} | "
                    f"{np.mean(neg_v):.1f} | {np.median(neg_v):.0f} | "
                    f"{_percentile(neg_v, 25):.0f} | {_percentile(neg_v, 75):.0f} | "
                    f"{_percentile(neg_v, 95):.0f} |\n"
                )
            elif len(pos_v) > 0:
                report_lines.append(
                    f"| {cls} | {s['pos_count']} | {s['neg_count']} | "
                    f"{np.mean(pos_v):.1f} | {np.median(pos_v):.0f} | "
                    f"{_percentile(pos_v, 25):.0f} | {_percentile(pos_v, 75):.0f} | "
                    f"{_percentile(pos_v, 95):.0f} | - | - | - | - | - |\n"
                )
            else:
                report_lines.append(
                    f"| {cls} | {s['pos_count']} | {s['neg_count']} | "
                    f"- | - | - | - | - | - | - | - | - | - |\n"
                )
        report_lines.append("\n")

    # 2. AUC-ROC per class
    report_lines.append("## 2. Logistic Regression AUC-ROC (Size Features Only)\n\n")
    report_lines.append("Using [num_nodes, num_edges, num_functions] → predict each class.\n\n")
    report_lines.append("| Class | AUC-ROC | Confounded? |\n")
    report_lines.append("|-------|---------|-------------|\n")
    confounded_classes = []
    for cls in VULN_CLASSES:
        auc_val = auc_results.get(cls)
        if auc_val is not None:
            confounded = "⚠️ YES" if auc_val > 0.65 else ("Borderline" if auc_val > 0.55 else "No")
            if auc_val > 0.65:
                confounded_classes.append(cls)
            report_lines.append(f"| {cls} | {auc_val:.4f} | {confounded} |\n")
        else:
            report_lines.append(f"| {cls} | N/A | Insufficient data |\n")
    report_lines.append("\n")

    # 3. Mann-Whitney U test
    report_lines.append("## 3. Mann-Whitney U Test: Size vs Label (Positive vs Negative)\n\n")
    report_lines.append("| Class | Metric | Pos Median | Neg Median | U-stat | p-value | Significant? |\n")
    report_lines.append("|-------|--------|------------|------------|--------|---------|---------------|\n")
    significant_pairs = []
    for cls in VULN_CLASSES:
        for metric_name in ["num_nodes", "num_edges", "num_functions"]:
            r = mw_results.get(cls, {}).get(metric_name)
            if r is not None:
                sig = "⚠️ YES" if r["p_value"] < 0.01 else ("Borderline" if r["p_value"] < 0.05 else "No")
                if r["p_value"] < 0.01:
                    significant_pairs.append((cls, metric_name, r["p_value"]))
                report_lines.append(
                    f"| {cls} | {metric_name} | {r['pos_median']:.0f} | "
                    f"{r['neg_median']:.0f} | {r['u_stat']:.0f} | "
                    f"{r['p_value']:.4f} | {sig} |\n"
                )
    report_lines.append("\n")

    # 4. Quartile comparison
    report_lines.append("## 4. Label Distribution: Top-25% vs Bottom-25% by Size\n\n")
    report_lines.append(f"Size metric: num_nodes + num_edges.  \n")
    report_lines.append(f"Bottom-25% threshold: ≤{p25:.0f} (n={int(bottom_mask.sum())})  \n")
    report_lines.append(f"Top-25% threshold: ≥{p75:.0f} (n={int(top_mask.sum())})\n\n")
    report_lines.append("| Class | Bottom-25% Pos Rate | Top-25% Pos Rate | Diff | Concern? |\n")
    report_lines.append("|-------|---------------------|-------------------|------|----------|\n")
    for cls in VULN_CLASSES:
        qc = quartile_comparison[cls]
        concern = "⚠️ YES" if abs(qc["diff"]) > 0.05 else "No"
        report_lines.append(
            f"| {cls} | {qc['bottom_pos_rate']:.4f} | "
            f"{qc['top_pos_rate']:.4f} | {qc['diff']:+.4f} | {concern} |\n"
        )
    report_lines.append("\n")

    # Summary & recommendations
    report_lines.append("## 5. Summary & Recommendations\n\n")
    report_lines.append(f"### Confounded Classes (AUC > 0.65 from size alone)\n\n")
    if confounded_classes:
        for cls in confounded_classes:
            report_lines.append(f"- **{cls}**: AUC = {auc_results[cls]:.4f}\n")
    else:
        report_lines.append("No classes strongly confounded by size.\n")
    report_lines.append("\n")

    report_lines.append(f"### Significant Size Differences (Mann-Whitney p < 0.01)\n\n")
    if significant_pairs:
        for cls, metric, pval in significant_pairs:
            report_lines.append(f"- **{cls}** — {metric} (p={pval:.4f})\n")
    else:
        report_lines.append("No significant size differences between positive/negative samples.\n")
    report_lines.append("\n")

    report_lines.append("### Recommendations\n\n")
    if confounded_classes:
        report_lines.append("1. **Add size normalization or stratification** for confounded classes.\n")
        report_lines.append("2. **Consider including graph size as a feature** explicitly so the model ")
        report_lines.append("can account for it rather than being silently biased.\n")
        report_lines.append("3. **Evaluate per-class metrics within size strata** to ensure performance ")
        report_lines.append("isn't driven solely by size differences.\n")
    else:
        report_lines.append("1. Graph size does not appear to be a major confound for most classes.\n")
        report_lines.append("2. Continue monitoring as new data is added.\n")
    if significant_pairs:
        report_lines.append("4. **For classes with significant Mann-Whitney results**, verify that ")
        report_lines.append("model predictions are not proxies for graph size.\n")

    report_content = "".join(report_lines)
    save_report("task22_graph_size_confound", report_content)
    print_header("Task 22 Complete")


if __name__ == "__main__":
    main()
