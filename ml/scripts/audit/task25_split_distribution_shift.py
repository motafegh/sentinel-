#!/usr/bin/env python3
"""
task25_split_distribution_shift.py — Split Distribution Shift Audit

Load split definitions from ml/data/splits/ (train_indices.npy, val_indices.npy,
test_indices.npy). For each split:
1. Solidity version distribution (sample 500 .sol per split)
2. Mean graph size (num_nodes, num_edges) — sample 200 per split
3. Per-class positive rate
4. Feature value distributions (mean loc, complexity, ext_call_count) with KS test
5. Token window distribution per split
6. Special: where are the 7 pure-DoS contracts? Are 0.8.x contracts evenly distributed?
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def _get_field(obj, key):
    """Get field from either PyG Data (attr) or dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def load_split_indices():
    """Load split indices from .npy files. Returns dict of {split_name: indices_array}.

    Looks for split .npy files in the following locations (in order):
      1. ml/data/splits/deduped/{name}_indices.npy   (indices into deduped CSV)
      2. ml/data/splits/{name}_indices.npy            (fallback)
    """
    candidate_dirs = [
        PROJECT_ROOT / "ml" / "data" / "splits" / "deduped",
        PROJECT_ROOT / "ml" / "data" / "splits",
    ]
    splits = {}

    for name in ["train", "val", "test"]:
        loaded = False
        for splits_dir in candidate_dirs:
            npy_path = splits_dir / f"{name}_indices.npy"
            if npy_path.exists():
                try:
                    splits[name] = np.load(npy_path)
                    print(f"  Loaded {name}: {len(splits[name])} indices from {npy_path}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"  WARNING: Could not load {npy_path}: {e}")

        if not loaded:
            # Fallback: try CSV/JSON split files matching *split*
            found_csv = False
            for splits_dir in candidate_dirs:
                for split_file in splits_dir.glob("*split*"):
                    if split_file.suffix in (".csv", ".json"):
                        print(f"  WARNING: Found split file {split_file} but only .npy format is supported")
                        found_csv = True
            if not found_csv:
                print(f"  WARNING: No split file found for '{name}' in any candidate directory")

    return splits


def main():
    print_header("Task 25: Split Distribution Shift Audit")

    # ── Load CSV ───────────────────────────────────────────────────────────
    print("  Loading label CSV...")
    labels = load_label_csv()
    csv_rows = load_label_csv_as_rows()
    print(f"  Total rows: {len(csv_rows)}")

    # ── Load splits ────────────────────────────────────────────────────────
    print("  Loading split indices...")
    splits = load_split_indices()

    if not splits:
        print("ERROR: No split files found. Cannot proceed.")
        print("  Expected at ml/data/splits/{{train,val,test}}_indices.npy")
        return

    # ── Map CSV indices to stems ───────────────────────────────────────────
    all_stems = [row["md5_stem"] for row in csv_rows]

    split_stems = {}
    for name, indices in splits.items():
        stems = []
        for idx in indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(all_stems):
                stems.append(all_stems[idx_int])
        split_stems[name] = stems
        print(f"  {name}: {len(stems)} stems")

    # ── Build md5_to_path ──────────────────────────────────────────────────
    all_needed = set()
    for stems in split_stems.values():
        all_needed.update(stems)
    print(f"\n  Building md5_to_path for {len(all_needed)} unique stems...")
    md5_to_path = build_md5_to_path(all_needed)
    print(f"  Resolved {len(md5_to_path)} stems to .sol files")

    # ── 1. Solidity version distribution ───────────────────────────────────
    print("\n  Computing Solidity version distributions...")
    version_dists = {}
    for name, stems in split_stems.items():
        sample_size_sol = min(500, len(stems))
        rng = np.random.default_rng(42)
        sampled = rng.choice(stems, size=sample_size_sol, replace=False).tolist() \
            if len(stems) > sample_size_sol else stems
        versions = Counter()
        resolved = 0
        for stem in sampled:
            sol_path = find_sol_for_stem(stem, md5_to_path)
            if sol_path is not None and sol_path.exists():
                ver = extract_pragma_version(sol_path)
                versions[ver if ver else "unknown"] += 1
                resolved += 1
        version_dists[name] = {
            "versions": versions,
            "sampled": len(sampled),
            "resolved": resolved,
        }
        print(f"    {name}: resolved {resolved}/{len(sampled)}, "
              f"versions: {dict(versions.most_common(5))}")

    # ── 2. Mean graph size per split ───────────────────────────────────────
    print("\n  Computing graph sizes per split...")
    graph_sizes = {}
    for name, stems in split_stems.items():
        sample_size_graph = min(200, len(stems))
        rng = np.random.default_rng(42)
        sampled = rng.choice(stems, size=sample_size_graph, replace=False).tolist() \
            if len(stems) > sample_size_graph else stems

        num_nodes_list = []
        num_edges_list = []
        loaded = 0

        for stem in sampled:
            graph_path = GRAPHS_DIR / f"{stem}.pt"
            if not graph_path.exists():
                continue
            try:
                data = load_graph(graph_path)
                if data.x is not None and data.edge_index is not None:
                    num_nodes_list.append(data.x.shape[0])
                    num_edges_list.append(data.edge_index.shape[1])
                    loaded += 1
            except Exception:
                continue

        graph_sizes[name] = {
            "num_nodes": np.array(num_nodes_list) if num_nodes_list else np.array([]),
            "num_edges": np.array(num_edges_list) if num_edges_list else np.array([]),
            "loaded": loaded,
            "sampled": len(sampled),
        }
        if num_nodes_list:
            print(f"    {name}: mean_nodes={np.mean(num_nodes_list):.1f}, "
                  f"mean_edges={np.mean(num_edges_list):.1f}")

    # ── 3. Per-class positive rate per split ───────────────────────────────
    print("\n  Computing per-class positive rates...")
    class_rates = {}
    for name, stems in split_stems.items():
        rates = {}
        for cls in VULN_CLASSES:
            pos = sum(1 for s in stems if s in labels and labels[s][cls] == 1)
            rates[cls] = {"positive": pos, "total": len(stems),
                          "rate": pos / max(len(stems), 1)}
        class_rates[name] = rates

    # ── 4. Feature value distributions + KS test ──────────────────────────
    print("\n  Computing feature distributions per split...")
    feature_dists = {}
    feature_indices = {
        "loc": 6,
        "complexity": 5,
        "external_call_count": 11,
    }

    for name, stems in split_stems.items():
        sample_size_feat = min(200, len(stems))
        rng = np.random.default_rng(42)
        sampled = rng.choice(stems, size=sample_size_feat, replace=False).tolist() \
            if len(stems) > sample_size_feat else stems

        feat_values = {fname: [] for fname in feature_indices}

        for stem in sampled:
            graph_path = GRAPHS_DIR / f"{stem}.pt"
            if not graph_path.exists():
                continue
            try:
                data = load_graph(graph_path)
                if data.x is not None and data.x.dim() == 2 and data.x.shape[1] == 12:
                    x_np = data.x.numpy() if hasattr(data.x, 'numpy') else np.array(data.x)
                    for fname, fidx in feature_indices.items():
                        feat_values[fname].append(float(x_np[:, fidx].mean()))
            except Exception:
                continue

        feature_dists[name] = {fname: np.array(vals) for fname, vals in feat_values.items()}

    # KS test between splits
    ks_results = {}
    split_names = list(splits.keys())
    try:
        from scipy.stats import ks_2samp
        ks_available = True
    except ImportError:
        print("  WARNING: scipy not available — KS test skipped")
        ks_available = False

    if ks_available:
        for fname in feature_indices:
            ks_results[fname] = {}
            for i, s1 in enumerate(split_names):
                for s2 in split_names[i+1:]:
                    vals1 = feature_dists.get(s1, {}).get(fname, np.array([]))
                    vals2 = feature_dists.get(s2, {}).get(fname, np.array([]))
                    if len(vals1) > 10 and len(vals2) > 10:
                        try:
                            stat, pval = ks_2samp(vals1, vals2)
                            ks_results[fname][f"{s1}_vs_{s2}"] = {
                                "statistic": float(stat),
                                "p_value": float(pval),
                                "significant": pval < 0.05,
                            }
                        except Exception:
                            ks_results[fname][f"{s1}_vs_{s2}"] = None
                    else:
                        ks_results[fname][f"{s1}_vs_{s2}"] = None

    # ── 5. Token window distribution per split ─────────────────────────────
    print("\n  Computing token window distributions per split...")
    token_window_dists = {}
    for name, stems in split_stems.items():
        sample_size_tok = min(200, len(stems))
        rng = np.random.default_rng(42)
        sampled = rng.choice(stems, size=sample_size_tok, replace=False).tolist() \
            if len(stems) > sample_size_tok else stems

        window_counts = Counter()
        loaded = 0
        for stem in sampled:
            token_path = TOKENS_WINDOWED_DIR / f"{stem}.pt"
            if not token_path.exists():
                continue
            try:
                t_data = load_token(token_path)
                nw = _get_field(t_data, "num_windows")
                if nw is not None:
                    window_counts[int(nw)] += 1
                else:
                    # Try to infer from shape
                    input_ids = _get_field(t_data, "input_ids")
                    if input_ids is not None and input_ids.dim() == 2:
                        window_counts[int(input_ids.shape[0])] += 1
                    else:
                        window_counts["unknown"] += 1
                loaded += 1
            except Exception:
                continue

        token_window_dists[name] = {
            "counts": window_counts,
            "loaded": loaded,
            "sampled": len(sampled),
        }

    # ── 6. Special: pure DoS contracts & 0.8.x distribution ───────────────
    print("\n  Locating pure DoS contracts...")
    pure_dos_stems = get_stems_with_label("DenialOfService", label_value=1, pure=True)
    print(f"  Pure DoS contracts: {len(pure_dos_stems)}")

    dos_split_location = {}
    for stem in pure_dos_stems:
        for name, stems in split_stems.items():
            if stem in stems:
                dos_split_location[stem] = name
                break
        else:
            dos_split_location[stem] = "not_in_split"

    # 0.8.x distribution across splits
    print("  Checking 0.8.x distribution across splits...")
    sol08x_per_split = {}
    for name, stems in split_stems.items():
        sample_08x = min(300, len(stems))
        rng = np.random.default_rng(42)
        sampled = rng.choice(stems, size=sample_08x, replace=False).tolist() \
            if len(stems) > sample_08x else stems
        v08_count = 0
        resolved_08x = 0
        for stem in sampled:
            sol_path = find_sol_for_stem(stem, md5_to_path)
            if sol_path is not None and sol_path.exists():
                ver = extract_pragma_version(sol_path)
                resolved_08x += 1
                if ver and ver.startswith("0.8"):
                    v08_count += 1
        sol08x_per_split[name] = {
            "0.8.x_count": v08_count,
            "resolved": resolved_08x,
            "rate": v08_count / max(resolved_08x, 1),
            "sampled": len(sampled),
        }

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 25: Split Distribution Shift Audit\n\n")
    report_lines.append(f"**CSV rows:** {len(csv_rows)}  \n")
    for name, indices in splits.items():
        report_lines.append(f"**{name} split:** {len(indices)} samples  \n")
    report_lines.append("\n")

    # 1. Solidity version distribution
    report_lines.append("## 1. Solidity Version Distribution per Split\n\n")
    # Collect all version keys
    all_versions = set()
    for vd in version_dists.values():
        all_versions.update(vd["versions"].keys())
    all_versions = sorted(all_versions)

    report_lines.append("| Version | " + " | ".join(split_names) + " |\n")
    report_lines.append("|---------|" + "|".join(["-------"] * len(split_names)) + "|\n")
    for ver in all_versions:
        row = f"| {ver} |"
        for name in split_names:
            vd = version_dists[name]
            count = vd["versions"].get(ver, 0)
            rate = count / max(vd["resolved"], 1)
            row += f" {count} ({rate:.1%}) |"
        report_lines.append(f"{row}\n")
    report_lines.append("\n")

    # 2. Graph size per split
    report_lines.append("## 2. Mean Graph Size per Split\n\n")
    report_lines.append("| Split | Loaded | Mean Nodes | Median Nodes | Mean Edges | Median Edges |\n")
    report_lines.append("|-------|--------|------------|--------------|------------|---------------|\n")
    for name in split_names:
        gs = graph_sizes.get(name, {})
        nn = gs.get("num_nodes", np.array([]))
        ne = gs.get("num_edges", np.array([]))
        loaded = gs.get("loaded", 0)
        if len(nn) > 0:
            report_lines.append(
                f"| {name} | {loaded} | {np.mean(nn):.1f} | {np.median(nn):.0f} | "
                f"{np.mean(ne):.1f} | {np.median(ne):.0f} |\n"
            )
        else:
            report_lines.append(f"| {name} | {loaded} | — | — | — | — |\n")
    report_lines.append("\n")

    # 3. Per-class positive rate
    report_lines.append("## 3. Per-Class Positive Rate per Split\n\n")
    report_lines.append("| Class | " + " | ".join(f"{s} Rate" for s in split_names) +
                        " | " + " | ".join(f"{s} Count" for s in split_names) + " |\n")
    report_lines.append("|-------|" + "|".join(["-------"] * len(split_names) * 2) + "|\n")
    for cls in VULN_CLASSES:
        rate_cols = []
        count_cols = []
        for name in split_names:
            cr = class_rates.get(name, {}).get(cls, {})
            rate_cols.append(f"{cr.get('rate', 0):.4f}")
            count_cols.append(f"{cr.get('positive', 0)}/{cr.get('total', 0)}")
        report_lines.append(f"| {cls} | " + " | ".join(rate_cols) + " | " +
                            " | ".join(count_cols) + " |\n")
    report_lines.append("\n")

    # 4. Feature distributions + KS test
    report_lines.append("## 4. Feature Distributions & KS Test\n\n")
    for fname in feature_indices:
        report_lines.append(f"### {fname}\n\n")
        report_lines.append("| Split | Mean | Std | Median | Min | Max |\n")
        report_lines.append("|-------|------|-----|--------|-----|-----|\n")
        for name in split_names:
            vals = feature_dists.get(name, {}).get(fname, np.array([]))
            if len(vals) > 0:
                report_lines.append(
                    f"| {name} | {np.mean(vals):.4f} | {np.std(vals):.4f} | "
                    f"{np.median(vals):.4f} | {np.min(vals):.4f} | {np.max(vals):.4f} |\n"
                )
            else:
                report_lines.append(f"| {name} | — | — | — | — | — |\n")
        report_lines.append("\n")

        # KS results
        if fname in ks_results and ks_results[fname]:
            report_lines.append("**KS Test Results:**\n\n")
            report_lines.append("| Comparison | Statistic | p-value | Significant? |\n")
            report_lines.append("|------------|-----------|---------|--------------|\n")
            for comp, result in ks_results[fname].items():
                if result is not None:
                    sig = "⚠️ YES" if result["significant"] else "No"
                    report_lines.append(
                        f"| {comp} | {result['statistic']:.4f} | "
                        f"{result['p_value']:.4f} | {sig} |\n"
                    )
                else:
                    report_lines.append(f"| {comp} | N/A | N/A | Insufficient data |\n")
            report_lines.append("\n")

    # 5. Token window distribution
    report_lines.append("## 5. Token Window Distribution per Split\n\n")
    all_window_keys = set()
    for twd in token_window_dists.values():
        all_window_keys.update(str(k) for k in twd["counts"].keys())
    all_window_keys = sorted(all_window_keys, key=lambda x: (x != "unknown", x))

    report_lines.append("| Windows | " + " | ".join(split_names) + " |\n")
    report_lines.append("|---------|" + "|".join(["-------"] * len(split_names)) + "|\n")
    for wk in all_window_keys:
        row = f"| {wk} |"
        for name in split_names:
            twd = token_window_dists.get(name, {})
            count = twd["counts"].get(int(wk) if wk != "unknown" else "unknown", 0)
            total = max(twd.get("loaded", 1), 1)
            rate = count / total
            row += f" {count} ({rate:.1%}) |"
        report_lines.append(f"{row}\n")
    report_lines.append("\n")

    # 6. Special: Pure DoS & 0.8.x
    report_lines.append("## 6. Special: Pure DoS Contracts & 0.8.x Distribution\n\n")
    report_lines.append(f"### Pure DenialOfService Contracts (n={len(pure_dos_stems)})\n\n")
    dos_by_split = Counter(dos_split_location.values())
    report_lines.append("| Split | Count |\n")
    report_lines.append("|-------|-------|\n")
    for name in split_names + ["not_in_split"]:
        count = dos_by_split.get(name, 0)
        report_lines.append(f"| {name} | {count} |\n")
    report_lines.append("\n")

    if len(pure_dos_stems) > 0 and len(pure_dos_stems) <= 20:
        report_lines.append("**Individual pure-DoS contracts:**\n\n")
        for stem, loc in sorted(dos_split_location.items()):
            report_lines.append(f"- `{stem}` → {loc}\n")
        report_lines.append("\n")

    # 0.8.x distribution
    report_lines.append("### Solidity 0.8.x Distribution\n\n")
    report_lines.append("| Split | 0.8.x Count | Resolved | 0.8.x Rate | Sampled |\n")
    report_lines.append("|-------|-------------|----------|------------|--------|\n")
    for name in split_names:
        d = sol08x_per_split.get(name, {})
        report_lines.append(
            f"| {name} | {d.get('0.8.x_count', 0)} | {d.get('resolved', 0)} | "
            f"{d.get('rate', 0):.1%} | {d.get('sampled', 0)} |\n"
        )
    report_lines.append("\n")

    # Summary
    report_lines.append("## 7. Summary & Distribution Shift Concerns\n\n")

    # Check for significant KS results
    significant_ks = []
    if ks_available:
        for fname, comparisons in ks_results.items():
            for comp, result in (comparisons or {}).items():
                if result is not None and result["significant"]:
                    significant_ks.append((fname, comp, result["p_value"]))

    if significant_ks:
        report_lines.append("### ⚠️ Significant Distribution Shifts Detected\n\n")
        for fname, comp, pval in significant_ks:
            report_lines.append(f"- **{fname}** — {comp} (p={pval:.4f})\n")
        report_lines.append("\n")
        report_lines.append("These shifts may indicate that splits are not representative ")
        report_lines.append("of each other, potentially leading to poor generalization.\n\n")
    else:
        report_lines.append("### No Significant Distribution Shifts\n\n")
        report_lines.append("Feature distributions appear consistent across splits.\n\n")

    # DoS balance check
    dos_in_train = dos_by_split.get("train", 0)
    dos_in_val = dos_by_split.get("val", 0)
    dos_in_test = dos_by_split.get("test", 0)
    if dos_in_val == 0 or dos_in_test == 0:
        report_lines.append("### ⚠️ Pure DoS Split Imbalance\n\n")
        report_lines.append(f"Pure DoS contracts: train={dos_in_train}, val={dos_in_val}, test={dos_in_test}.  \n")
        if dos_in_val == 0:
            report_lines.append("Zero pure DoS in validation set — DoS metrics will be unmeasurable.  \n")
        if dos_in_test == 0:
            report_lines.append("Zero pure DoS in test set — DoS generalization cannot be evaluated.  \n")
        report_lines.append("\n")

    # 0.8.x balance check
    rates_08x = [sol08x_per_split.get(n, {}).get("rate", 0) for n in split_names]
    if rates_08x and max(rates_08x) > 0:
        rate_range = max(rates_08x) - min(rates_08x)
        if rate_range > 0.05:
            report_lines.append("### ⚠️ 0.8.x Version Imbalance\n\n")
            report_lines.append(f"0.8.x rate range across splits: {min(rates_08x):.1%} – {max(rates_08x):.1%}  \n")
            report_lines.append("This could bias `in_unchecked` feature (0.8.x only) detection.  \n\n")
        else:
            report_lines.append("### 0.8.x Version Distribution: Balanced ✓\n\n")

    report_content = "".join(report_lines)
    save_report("task25_split_distribution_shift", report_content)
    print_header("Task 25 Complete")


if __name__ == "__main__":
    main()
