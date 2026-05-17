#!/usr/bin/env python3
"""
task21_feature_correlation.py — Feature Correlation & Redundancy Audit

Load 1000 graph .pt files. Flatten all node features into matrix [total_nodes, 12].
Compute:
1. Pearson correlation matrix [12,12]
2. Spearman rank correlation matrix [12,12]
3. Per contract: aggregate to contract-level (max for binary features [2,3,4,7,9,10],
   mean+max for continuous)
4. Mutual information between each contract-level feature and each of 10 labels
5. For each feature: unique info = 1 - R² from regressing on all other features
6. PCA on node-level matrix: how many components for 90%, 95%, 99% variance
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


# Binary feature indices (max for aggregation)
BINARY_INDICES = [2, 3, 4, 7, 9, 10]
# Continuous feature indices (mean + max for aggregation)
CONTINUOUS_INDICES = [0, 1, 5, 6, 8, 11]


def main():
    print_header("Task 21: Feature Correlation & Redundancy Audit")

    # ── Collect graph files ────────────────────────────────────────────────
    all_graph_files = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_graph_files:
        print("ERROR: No graph .pt files found in", GRAPHS_DIR)
        return
    print(f"Found {len(all_graph_files)} graph files")

    sample_size = min(1000, len(all_graph_files))
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(len(all_graph_files), size=sample_size, replace=False)
    sampled_files = [all_graph_files[i] for i in sampled_indices]
    print(f"Sampling {sample_size} files")

    # ── Load features ─────────────────────────────────────────────────────
    all_node_features = []  # list of [Ni, 12] arrays
    contract_features = []  # list of aggregated feature vectors
    contract_stems = []     # md5_stem for each contract
    skipped = 0

    for i, fpath in enumerate(sampled_files):
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i + 1}/{sample_size} files...")

        try:
            data = load_graph(fpath)
        except Exception as e:
            skipped += 1
            print(f"  WARNING: Could not load {fpath.stem}: {e}")
            continue

        x = data.x
        if x is None or x.dim() != 2 or x.shape[1] != 12:
            skipped += 1
            continue

        x_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)
        # Filter out NaN/Inf rows
        valid_mask = np.isfinite(x_np).all(axis=1)
        x_np = x_np[valid_mask]

        if x_np.shape[0] == 0:
            skipped += 1
            continue

        all_node_features.append(x_np)

        # Contract-level aggregation
        agg = np.zeros(12, dtype=np.float64)
        for j in range(12):
            if j in BINARY_INDICES:
                agg[j] = x_np[:, j].max()
            else:
                agg[j] = (x_np[:, j].mean() + x_np[:, j].max()) / 2.0

        contract_features.append(agg)
        contract_stems.append(fpath.stem)

    print(f"  Successfully loaded: {len(all_node_features)}")
    print(f"  Skipped: {skipped}")

    if len(all_node_features) == 0:
        print("ERROR: No valid graph files loaded.")
        return

    # ── 1. Build node-level feature matrix ─────────────────────────────────
    print("\n  Building node-level feature matrix...")
    node_matrix = np.vstack(all_node_features)  # [total_nodes, 12]
    print(f"  Node matrix shape: {node_matrix.shape}")

    # ── 2. Pearson correlation matrix ──────────────────────────────────────
    print("  Computing Pearson correlation matrix...")
    pearson_corr = np.corrcoef(node_matrix, rowvar=False)  # [12, 12]

    # ── 3. Spearman rank correlation matrix ────────────────────────────────
    print("  Computing Spearman rank correlation matrix...")
    from scipy.stats import spearmanr
    try:
        spearman_corr, spearman_p = spearmanr(node_matrix, axis=0)
        if spearman_corr.ndim == 0:
            # Edge case: only 1 feature
            spearman_corr = np.array([[float(spearman_corr)]])
        else:
            spearman_corr = np.array(spearman_corr)
    except Exception as e:
        print(f"  WARNING: Spearman correlation failed: {e}")
        spearman_corr = np.full((12, 12), np.nan)

    # ── 4. Contract-level feature matrix ───────────────────────────────────
    contract_matrix = np.vstack(contract_features)  # [num_contracts, 12]
    print(f"  Contract-level matrix shape: {contract_matrix.shape}")

    # Load labels
    print("  Loading labels...")
    labels = load_label_csv()

    # Build label matrix aligned with contract_stems
    label_matrix = np.zeros((len(contract_stems), 10), dtype=np.float32)
    valid_label_mask = np.ones(len(contract_stems), dtype=bool)
    for i, stem in enumerate(contract_stems):
        if stem in labels:
            for j, cls in enumerate(VULN_CLASSES):
                label_matrix[i, j] = labels[stem][cls]
        else:
            valid_label_mask[i] = False

    n_labeled = valid_label_mask.sum()
    print(f"  Contracts with labels: {n_labeled}/{len(contract_stems)}")

    # ── 5. Mutual information ──────────────────────────────────────────────
    print("  Computing mutual information (feature ↔ label)...")
    mi_results = np.zeros((12, 10))  # [feature, class]
    mi_available = False
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi_available = True
        if n_labeled > 50:
            X_labeled = contract_matrix[valid_label_mask]
            y_labeled = label_matrix[valid_label_mask]
            for j, cls in enumerate(VULN_CLASSES):
                y_cls = y_labeled[:, j].astype(int)
                if y_cls.sum() > 0 and y_cls.sum() < len(y_cls):
                    mi_results[:, j] = mutual_info_classif(
                        X_labeled, y_cls, random_state=42, n_neighbors=5
                    )
                else:
                    mi_results[:, j] = 0.0
        else:
            print(f"  WARNING: Only {n_labeled} labeled contracts — MI skipped")
    except ImportError:
        print("  WARNING: sklearn not available — MI computation skipped")
    except Exception as e:
        print(f"  WARNING: MI computation failed: {e}")

    # ── 6. Unique information per feature ──────────────────────────────────
    print("  Computing unique information per feature (1 - R²)...")
    unique_info = np.zeros(12)
    try:
        from sklearn.linear_model import LinearRegression
        for j in range(12):
            other_cols = [k for k in range(12) if k != j]
            X_other = contract_matrix[:, other_cols]
            y_feat = contract_matrix[:, j]
            lr = LinearRegression()
            lr.fit(X_other, y_feat)
            r2 = lr.score(X_other, y_feat)
            unique_info[j] = max(0.0, 1.0 - r2)
    except ImportError:
        print("  WARNING: sklearn not available — unique info skipped")
        unique_info[:] = np.nan
    except Exception as e:
        print(f"  WARNING: Unique info computation failed: {e}")
        unique_info[:] = np.nan

    # ── 7. PCA ─────────────────────────────────────────────────────────────
    print("  Computing PCA on node-level matrix...")
    pca_results = {}
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        # Subsample if very large
        if node_matrix.shape[0] > 50000:
            pca_sample_idx = rng.choice(node_matrix.shape[0], size=50000, replace=False)
            pca_data = node_matrix[pca_sample_idx]
        else:
            pca_data = node_matrix
        scaler = StandardScaler()
        pca_data_scaled = scaler.fit_transform(pca_data)
        pca = PCA()
        pca.fit(pca_data_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        for threshold in [0.90, 0.95, 0.99]:
            n_comp = int(np.searchsorted(cumvar, threshold) + 1)
            pca_results[threshold] = n_comp
            print(f"    {threshold:.0%} variance: {n_comp} components")
    except ImportError:
        print("  WARNING: sklearn not available — PCA skipped")
    except Exception as e:
        print(f"  WARNING: PCA computation failed: {e}")

    # ── Flag highly correlated pairs ───────────────────────────────────────
    high_corr_pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            p_val = abs(pearson_corr[i, j])
            s_val = abs(spearman_corr[i, j]) if not np.isnan(spearman_corr[i, j]) else 0.0
            if p_val > 0.5 or s_val > 0.5:
                high_corr_pairs.append({
                    "feat_i": FEATURE_NAMES[i],
                    "feat_j": FEATURE_NAMES[j],
                    "pearson": pearson_corr[i, j],
                    "spearman": spearman_corr[i, j] if not np.isnan(spearman_corr[i, j]) else None,
                })

    # ── Features with zero MI ──────────────────────────────────────────────
    zero_mi_features = []
    if mi_available:
        for j in range(12):
            total_mi = mi_results[j, :].sum()
            if total_mi < 1e-6:
                zero_mi_features.append(FEATURE_NAMES[j])

    # ── Recommend features to drop ─────────────────────────────────────────
    # Score each feature: low unique_info + high max correlation + low total MI
    drop_candidates = []
    for j in range(12):
        max_corr = max(abs(pearson_corr[j, k]) for k in range(12) if k != j)
        total_mi = mi_results[j, :].sum() if mi_available else 0.0
        uinfo = unique_info[j] if not np.isnan(unique_info[j]) else 0.5
        # Higher score = more droppable
        drop_score = max_corr * (1.0 - uinfo) / (1.0 + total_mi + 1e-8)
        drop_candidates.append({
            "feature": FEATURE_NAMES[j],
            "unique_info": uinfo,
            "max_abs_corr": max_corr,
            "total_mi": total_mi,
            "drop_score": drop_score,
        })
    drop_candidates.sort(key=lambda x: x["drop_score"], reverse=True)

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 21: Feature Correlation & Redundancy Audit\n\n")
    report_lines.append(f"**Sample size:** {len(all_node_features)}  \n")
    report_lines.append(f"**Total nodes:** {node_matrix.shape[0]:,}  \n")
    report_lines.append(f"**Skipped files:** {skipped}\n\n")

    # Pearson correlation table
    report_lines.append("## 1. Pearson Correlation Matrix\n\n")
    # Header row
    short_names = [n[:8] for n in FEATURE_NAMES]
    header = "         " + "".join(f"{s:>10}" for s in short_names)
    report_lines.append(f"```\n{header}\n")
    for i in range(12):
        row = f"{short_names[i]:>8} " + "".join(f"{pearson_corr[i, j]:>10.3f}" for j in range(12))
        report_lines.append(f"{row}\n")
    report_lines.append("```\n\n")

    # Spearman correlation table
    report_lines.append("## 2. Spearman Rank Correlation Matrix\n\n")
    report_lines.append(f"```\n{header}\n")
    for i in range(12):
        row_vals = []
        for j in range(12):
            v = spearman_corr[i, j]
            row_vals.append(f"{v:>10.3f}" if not np.isnan(v) else f"{'N/A':>10}")
        row = f"{short_names[i]:>8} " + "".join(row_vals)
        report_lines.append(f"{row}\n")
    report_lines.append("```\n\n")

    # Highly correlated pairs
    report_lines.append("## 3. Highly Correlated Feature Pairs (|r| > 0.5)\n\n")
    if high_corr_pairs:
        report_lines.append("| Feature A | Feature B | Pearson | Spearman |\n")
        report_lines.append("|-----------|-----------|---------|----------|\n")
        for p in high_corr_pairs:
            s_val = f"{p['spearman']:.3f}" if p['spearman'] is not None else "N/A"
            report_lines.append(
                f"| {p['feat_i']} | {p['feat_j']} | {p['pearson']:.3f} | {s_val} |\n"
            )
    else:
        report_lines.append("No feature pairs with |correlation| > 0.5 found.\n")
    report_lines.append("\n")

    # Mutual information table
    report_lines.append("## 4. Mutual Information: Feature ↔ Label\n\n")
    if mi_available:
        report_lines.append("| Feature | " + " | ".join(cls[:6] for cls in VULN_CLASSES) + " | Total |\n")
        report_lines.append("|---------|" + "|".join(["-------"] * 11) + "|\n")
        for j in range(12):
            vals = " | ".join(f"{mi_results[j, k]:.4f}" for k in range(10))
            total = mi_results[j, :].sum()
            report_lines.append(f"| {FEATURE_NAMES[j]} | {vals} | {total:.4f} |\n")
        report_lines.append("\n")
    else:
        report_lines.append("MI computation not available (sklearn missing).\n\n")

    # Zero MI features
    report_lines.append("## 5. Features with Zero MI (all labels)\n\n")
    if zero_mi_features:
        for f in zero_mi_features:
            report_lines.append(f"- **{f}**\n")
    else:
        report_lines.append("No features with zero total MI.\n")
    report_lines.append("\n")

    # Unique information
    report_lines.append("## 6. Unique Information per Feature (1 - R²)\n\n")
    report_lines.append("| Feature | Unique Info | Interpretation |\n")
    report_lines.append("|---------|-------------|----------------|\n")
    for j in range(12):
        uinfo = unique_info[j]
        if np.isnan(uinfo):
            interp = "N/A"
        elif uinfo < 0.05:
            interp = "⚠️ Nearly fully redundant"
        elif uinfo < 0.2:
            interp = "Low unique info"
        elif uinfo < 0.5:
            interp = "Moderate unique info"
        else:
            interp = "High unique info"
        report_lines.append(f"| {FEATURE_NAMES[j]} | {uinfo:.4f} | {interp} |\n")
    report_lines.append("\n")

    # PCA results
    report_lines.append("## 7. PCA Variance Explained\n\n")
    if pca_results:
        report_lines.append("| Variance Threshold | Components Needed |\n")
        report_lines.append("|--------------------|--------------------|\n")
        for threshold, n_comp in sorted(pca_results.items()):
            report_lines.append(f"| {threshold:.0%} | {n_comp} |\n")
        report_lines.append("\n")
        try:
            report_lines.append("### Individual Component Variance\n\n")
            report_lines.append("| Component | Variance Explained | Cumulative |\n")
            report_lines.append("|-----------|--------------------|-----------|\n")
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            for k in range(min(12, len(pca.explained_variance_ratio_))):
                report_lines.append(
                    f"| PC{k+1} | {pca.explained_variance_ratio_[k]:.4f} | {cumvar[k]:.4f} |\n"
                )
            report_lines.append("\n")
        except Exception:
            pass
    else:
        report_lines.append("PCA computation not available.\n\n")

    # Drop recommendation
    report_lines.append("## 8. Feature Drop Recommendation\n\n")
    report_lines.append("Top 3 candidates for removal (highest drop score):\n\n")
    report_lines.append("| Rank | Feature | Drop Score | Unique Info | Max |Corr| | Total MI |\n")
    report_lines.append("|------|---------|------------|-------------|--------------|----------|\n")
    for rank, c in enumerate(drop_candidates[:3], 1):
        report_lines.append(
            f"| {rank} | {c['feature']} | {c['drop_score']:.4f} | "
            f"{c['unique_info']:.4f} | {c['max_abs_corr']:.4f} | {c['total_mi']:.4f} |\n"
        )
    report_lines.append("\n")
    report_lines.append("**Recommendation:** Consider dropping the above features as they ")
    report_lines.append("contribute the least unique information and have the highest redundancy ")
    report_lines.append("with other features.\n\n")

    # Summary
    report_lines.append("## Summary\n\n")
    report_lines.append(f"- **Highly correlated pairs (|r|>0.5):** {len(high_corr_pairs)}\n")
    report_lines.append(f"- **Features with zero MI:** {len(zero_mi_features)}")
    if zero_mi_features:
        report_lines.append(f" ({', '.join(zero_mi_features)})")
    report_lines.append("\n")
    if pca_results:
        report_lines.append(f"- **PCA components for 95% variance:** {pca_results.get(0.95, 'N/A')}\n")
    report_lines.append(f"- **Recommended to drop:** {', '.join(c['feature'] for c in drop_candidates[:3])}\n")

    report_content = "".join(report_lines)
    save_report("task21_feature_correlation", report_content)
    print_header("Task 21 Complete")


if __name__ == "__main__":
    main()
