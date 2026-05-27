"""
Task 23: .send() Unchecked Return Prevalence
Task 25: Split Distribution Shift Analysis
------------------------------------------
Run:
    python task_23_25_send_splits.py [--task 23|25|both]
"""
import re
import sys
import random
import numpy as np
from pathlib import Path
from common import (
    get_dirs, load_csv, load_graph, sol_from_graph,
    LABEL_COLS, FEATURE_NAMES, print_header, random_pt_sample
)

# ════════════════════════════════════════════════════════════════════════════
# TASK 23 — .send() unchecked return
# ════════════════════════════════════════════════════════════════════════════

# .send() patterns
SEND_UNCHECKED = re.compile(r'(?:^|;|\{)\s*[\w\[\].]+\.send\s*\([^)]*\)\s*;', re.MULTILINE)
SEND_CHECKED   = re.compile(
    r'(?:if\s*\(|require\s*\(|bool\s+\w+\s*=)\s*[\w\[\].]+\.send\s*\(', re.MULTILINE
)

def analyse_send(source: str):
    """Returns (unchecked_count, checked_count)."""
    checked   = len(SEND_CHECKED.findall(source))
    all_sends = len(re.findall(r'\.send\s*\(', source))
    unchecked = max(0, all_sends - checked)
    return unchecked, checked

def task_23():
    print_header(23, ".send() Unchecked Return Prevalence")
    _, _, graphs_dir, _, _, bccc_dir = get_dirs()
    df = load_csv()

    mis_stems = df[df["MishandledException"] == 1]["md5_stem"].astype(str).tolist()
    dos_stems = df[df["DenialOfService"] == 1]["md5_stem"].astype(str).tolist()

    print(f"  MishandledException=1 pool: {len(mis_stems):,}")
    print(f"  DenialOfService=1 pool:     {len(dos_stems):,}")

    random.seed(42)
    mis_sample = random.sample(mis_stems, min(500, len(mis_stems)))

    results = {
        "has_send":           0,
        "unchecked_send":     0,
        "return_ignored_1":   0,  # graph feature fires for these
        "return_ignored_0":   0,  # graph feature misses these
        "no_sol":             0,
        "load_err":           0,
    }
    miss_examples = []

    for stem in mis_sample:
        gpath = graphs_dir / f"{stem}.pt"
        if not gpath.exists():
            results["load_err"] += 1
            continue
        try:
            g = load_graph(gpath)
        except Exception:
            results["load_err"] += 1
            continue
        sol = sol_from_graph(g, bccc_dir)
        if sol is None:
            results["no_sol"] += 1
            continue
        source = sol.read_text(errors="replace")
        unch, chk = analyse_send(source)
        if (unch + chk) > 0:
            results["has_send"] += 1
        if unch > 0:
            results["unchecked_send"] += 1
            # Check return_ignored [7] in graph (max across all nodes)
            x = g.x.numpy()
            ri_max = float(x[:, 7].max())
            if ri_max == 1.0:
                results["return_ignored_1"] += 1
            else:
                results["return_ignored_0"] += 1
                if len(miss_examples) < 10:
                    miss_examples.append((stem, getattr(g, "contract_path", "?")))

    analysed = len(mis_sample) - results["no_sol"] - results["load_err"]
    pct = lambda n: f"{n/analysed*100:.1f}%" if analysed else "N/A"

    print(f"\n.send() Analysis (n={analysed} resolved MishandledException contracts):")
    print(f"  Contracts with any .send():        {results['has_send']:>5}  ({pct(results['has_send'])})")
    print(f"  Contracts with UNCHECKED .send():  {results['unchecked_send']:>5}  ({pct(results['unchecked_send'])})")
    print(f"    Of those, return_ignored=1.0:    {results['return_ignored_1']:>5}  "
          f"(correctly detected)")
    print(f"    Of those, return_ignored=0:      {results['return_ignored_0']:>5}  "
          f"(MISSED by extractor — BUG-9)")

    if results['unchecked_send'] > 0:
        miss_rate = results['return_ignored_0'] / results['unchecked_send'] * 100
        print(f"\n  BUG-9 miss rate: {miss_rate:.1f}% of unchecked .send() contracts "
              "not caught by return_ignored")

    if miss_examples:
        print(f"\n  Example misses (return_ignored=0 despite unchecked .send()):")
        for stem, path in miss_examples:
            print(f"    {stem}  {path}")

    print("\n── Summary Task 23 ──────────────────────────────────────────────────────")
    if results['return_ignored_0'] > 0:
        print(f"  [CONFIRMED BUG-9] {results['return_ignored_0']} unchecked .send() missed "
              "by return_ignored feature.")
    else:
        print("  [FALSE ALARM] No unchecked .send() misses found in sample.")

# ════════════════════════════════════════════════════════════════════════════
# TASK 25 — Split distribution shift
# ════════════════════════════════════════════════════════════════════════════

PRAGMA_RE = re.compile(r'pragma\s+solidity\s+[^0-9]*(\d+)\.(\d+)', re.IGNORECASE)

def task_25():
    print_header(25, "Split Distribution Shift Analysis")
    _, ml_dir, graphs_dir, tokens_dir, _, bccc_dir = get_dirs()

    # Locate split files
    processed_dir = ml_dir / "data" / "processed"
    split_files = list(processed_dir.glob("*split*.csv")) + list(processed_dir.glob("*split*.json"))
    split_index_csv = processed_dir / "split_index.csv"  # common name
    if split_index_csv.exists():
        split_files.insert(0, split_index_csv)

    if not split_files:
        # Try to find any file that might contain split assignments
        for name in ["splits.csv", "data_splits.csv", "train_val_test.csv"]:
            f = processed_dir / name
            if f.exists():
                split_files.append(f)

    if not split_files:
        print("[WARN] No split files found in ml/data/processed/. "
              "Checking for split columns in the main CSV…")
        df = load_csv()
        if "split" not in df.columns:
            print("[ERROR] No 'split' column in CSV and no split files found. "
                  "Cannot run Task 25. Listing processed dir contents:")
            for f in sorted(processed_dir.iterdir()):
                print(f"  {f.name}")
            return
    else:
        import pandas as pd
        split_df = pd.read_csv(split_files[0])
        df = load_csv()
        df = df.merge(split_df[["md5_stem", "split"]], on="md5_stem", how="left") \
               if "split" not in df.columns else df
        print(f"  Split file: {split_files[0].name}")
        print(f"  Split distribution: {dict(df['split'].value_counts())}")

    if "split" not in df.columns:
        print("[ERROR] 'split' column not found after merge.")
        return

    splits = df["split"].dropna().unique()
    print(f"  Splits found: {list(splits)}\n")

    # Per-split: label distribution
    print("Per-class positive rate per split (%):")
    label_cols = [c for c in df.columns if c in LABEL_COLS]
    header = f"{'Class':<22}" + "".join(f" {str(s):>9}" for s in sorted(splits))
    print(header)
    print("-" * (22 + 10 * len(splits)))
    for cls in label_cols:
        row = f"  {cls:<20}"
        for sp in sorted(splits):
            sub = df[df["split"] == sp]
            pos = sub[cls].sum()
            row += f" {pos/len(sub)*100:>8.1f}%"
        print(row)

    # Per-split: graph size sample (200 per split)
    print("\nMean graph sizes per split (sample 200 per split):")
    print(f"  {'Split':<10} {'mean_nodes':>11} {'mean_edges':>11}")
    print("  " + "-" * 36)
    for sp in sorted(splits):
        stems = df[df["split"] == sp]["md5_stem"].astype(str).tolist()
        random.seed(42)
        sample = random.sample(stems, min(200, len(stems)))
        ns, es = [], []
        for stem in sample:
            gpath = graphs_dir / f"{stem}.pt"
            if not gpath.exists(): continue
            try:
                g = load_graph(gpath)
                ns.append(g.x.shape[0])
                es.append(g.edge_index.shape[1] if g.edge_index.numel() else 0)
            except Exception:
                continue
        if ns:
            print(f"  {str(sp):<10} {np.mean(ns):>11.1f} {np.mean(es):>11.1f}")

    # KS test: feature distributions train vs val, train vs test
    print("\nKolmogorov-Smirnov test on features (train vs other splits):")
    print("  (p < 0.05 = significant distribution shift)")
    try:
        from scipy import stats as scipy_stats

        split_feats = {}
        for sp in sorted(splits):
            stems = df[df["split"] == sp]["md5_stem"].astype(str).tolist()
            random.seed(42)
            sample = random.sample(stems, min(200, len(stems)))
            all_x = []
            for stem in sample:
                gpath = graphs_dir / f"{stem}.pt"
                if not gpath.exists(): continue
                try:
                    g = load_graph(gpath)
                    all_x.append(g.x.numpy())
                except Exception:
                    continue
            split_feats[sp] = np.concatenate(all_x, axis=0) if all_x else np.empty((0, 12))

        train_key = "train" if "train" in split_feats else sorted(splits)[0]
        for sp in sorted(splits):
            if sp == train_key: continue
            print(f"\n  {train_key} vs {sp}:")
            print(f"  {'Feature':<28} {'KS stat':>8} {'p-value':>9} {'Shift?':>7}")
            print("  " + "-" * 56)
            for fi, fname in enumerate(FEATURE_NAMES):
                a = split_feats[train_key][:, fi] if split_feats[train_key].size else np.array([0])
                b = split_feats[sp][:, fi]         if split_feats[sp].size         else np.array([0])
                ks, p = scipy_stats.ks_2samp(a, b)
                flag = "YES" if p < 0.05 else "no"
                print(f"  {fname:<28} {ks:>8.4f} {p:>9.4f} {flag:>7}")

    except ImportError:
        print("  [SKIP] scipy not available.")

    print("\n── Summary Task 25 ──────────────────────────────────────────────────────")
    print("  Check for classes where per-split positive rates differ significantly.")
    print("  Check for features with significant KS shift between train and val/test.")


def main():
    mode = "both"
    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        mode = sys.argv[idx + 1]
    if mode in ("23", "both"): task_23()
    if mode in ("25", "both"): task_25()

if __name__ == "__main__":
    main()
