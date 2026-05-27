"""
Task 20: DoS ↔ Reentrancy Class Separability Analysis
Task 21: Feature Correlation and Redundancy Analysis
-----------------------------------------------------
Task 20 analyses the 98% co-occurrence of DoS and Reentrancy.
Task 21 computes Pearson/Spearman correlations and PCA on node features.

Run:
    python task_20_21_correlation.py [--task 20|21|both]
"""
import sys
import random
import numpy as np
from pathlib import Path
from common import (
    get_dirs, load_csv, load_graph, sol_from_graph,
    FEATURE_NAMES, LABEL_COLS, print_header, random_pt_sample
)

# ════════════════════════════════════════════════════════════════════════════
# TASK 20
# ════════════════════════════════════════════════════════════════════════════

def task_20():
    print_header(20, "DoS ↔ Reentrancy Class Separability Analysis")
    _, _, graphs_dir, _, _, bccc_dir = get_dirs()
    df = load_csv()

    dos_all  = df[df["DenialOfService"] == 1]["md5_stem"].astype(str).tolist()
    dos_pure = df[(df["DenialOfService"] == 1) & (df["Reentrancy"] == 0)]["md5_stem"].astype(str).tolist()
    dos_ree  = df[(df["DenialOfService"] == 1) & (df["Reentrancy"] == 1)]["md5_stem"].astype(str).tolist()

    print(f"\n  DoS=1 total:              {len(dos_all):,}")
    print(f"  DoS=1 AND Reentrancy=1:   {len(dos_ree):,}  ({len(dos_ree)/len(dos_all)*100:.1f}%)")
    print(f"  DoS=1 AND Reentrancy=0:   {len(dos_pure):,}  (pure DoS)")

    # Feature comparison between the two groups
    feat_groups = {"DoS_pure": dos_pure, "DoS_plus_Ree": random.sample(dos_ree, min(30, len(dos_ree)))}
    agg = {}
    for gname, stems in feat_groups.items():
        means = []
        for stem in stems:
            gpath = graphs_dir / f"{stem}.pt"
            if not gpath.exists():
                continue
            try:
                g = load_graph(gpath)
            except Exception:
                continue
            x = g.x.numpy()
            # Per-contract aggregates
            row = [
                x.shape[0],                    # num_nodes
                g.edge_index.shape[1] if g.edge_index.numel() else 0,  # num_edges
                float((x[:, 10] > 0).any()),   # has_loop (any node)
                float(x[:, 11].max()),          # ext_call_count max
                float(x[:, 5].max()),           # complexity max
                float((g.edge_attr == 6).sum().item()) if g.edge_attr.numel() else 0,  # CF edges
            ]
            means.append(row)
        if means:
            agg[gname] = np.array(means).mean(axis=0)
        else:
            agg[gname] = np.zeros(6)

    cols = ["num_nodes", "num_edges", "has_loop", "ext_call_max", "complexity_max", "CF_edges"]
    print(f"\n{'Feature':<20} {'DoS pure':>12} {'DoS+Reentrancy':>14}")
    print("-" * 50)
    for i, c in enumerate(cols):
        v1 = agg.get("DoS_pure",      np.zeros(6))[i]
        v2 = agg.get("DoS_plus_Ree",  np.zeros(6))[i]
        print(f"  {c:<18} {v1:>12.2f} {v2:>14.2f}")

    # Source inspection for pure DoS
    if dos_pure and bccc_dir:
        print(f"\n  Pure DoS .sol source inspection (n={len(dos_pure)}):")
        for stem in dos_pure:
            gpath = graphs_dir / f"{stem}.pt"
            if not gpath.exists():
                continue
            try:
                g = load_graph(gpath)
            except Exception:
                continue
            sol = sol_from_graph(g, bccc_dir)
            if sol is None:
                print(f"    {stem}: [no .sol found]")
                continue
            source = sol.read_text(errors="replace")
            has_ext = ".call(" in source or ".send(" in source or ".transfer(" in source
            has_loop= "for " in source or "while " in source
            print(f"    {stem}: ext_call={has_ext}  loop={has_loop}  "
                  f"lines={len(source.splitlines())}  path={getattr(g, 'contract_path', '?')}")

    print("\n── Summary ─────────────────────────────────────────────────────────────")
    print(f"  [CONFIRMED] DoS↔Reentrancy co-occurrence: {len(dos_ree)/len(dos_all)*100:.1f}%")
    print(f"  [CONFIRMED] Only {len(dos_pure)} pure-DoS contracts — effectively untrainable.")
    if len(dos_pure) < 10:
        print("  Recommendation: Merge DoS into Reentrancy or apply extreme class weighting.")

# ════════════════════════════════════════════════════════════════════════════
# TASK 21
# ════════════════════════════════════════════════════════════════════════════

def task_21():
    print_header(21, "Feature Correlation and Redundancy Analysis")
    _, _, graphs_dir, _, _, _ = get_dirs()

    paths = random_pt_sample(graphs_dir, 1000)
    print(f"Sampling {len(paths)} graphs…")

    node_rows  = []  # flat node feature rows
    contract_rows = []  # per-contract aggregated features

    for p in paths:
        try:
            g = load_graph(p)
        except Exception:
            continue
        x = g.x.numpy().astype(np.float32)
        if x.shape[1] != 12:
            continue
        node_rows.append(x)
        # Binary features: max across nodes; continuous: mean and max
        binary   = [2, 3, 4, 7, 9, 10]
        contract = []
        for fi in range(12):
            col = x[:, fi]
            contract.append(col.mean())
            contract.append(col.max())
        contract_rows.append(contract)

    if not node_rows:
        print("[ERROR] No graphs loaded.")
        return

    all_nodes = np.concatenate(node_rows, axis=0)
    print(f"Total nodes: {len(all_nodes):,}\n")

    # ── Pearson correlation ───────────────────────────────────────────────────
    corr = np.corrcoef(all_nodes.T)  # [12, 12]
    print("Pearson correlation matrix (|r| > 0.5 flagged with *):")
    short_names = [n.split("]")[0]+"]" for n in FEATURE_NAMES]
    print(f"{'':>12}" + "".join(f"{n:>6}" for n in short_names))
    for i in range(12):
        row_str = f"  {short_names[i]:>10}"
        for j in range(12):
            v = corr[i, j]
            flag = "*" if abs(v) > 0.5 and i != j else " "
            row_str += f" {v:>4.2f}{flag}"
        print(row_str)

    # ── High-correlation pairs ────────────────────────────────────────────────
    print("\nFeature pairs with |r| > 0.5 (potentially redundant):")
    found = False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(corr[i, j]) > 0.5:
                print(f"  {FEATURE_NAMES[i]} ↔ {FEATURE_NAMES[j]}: r={corr[i,j]:.3f}")
                found = True
    if not found:
        print("  None found.")

    # ── Near-zero variance features ───────────────────────────────────────────
    print("\nFeature variance (near-zero = dead signal):")
    for i in range(12):
        var = float(np.var(all_nodes[:, i]))
        flag = " ← DEAD" if var < 1e-4 else ""
        print(f"  {FEATURE_NAMES[i]:<30} var={var:.6f}{flag}")

    # ── PCA ──────────────────────────────────────────────────────────────────
    print("\nPCA on node features:")
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=12)
        pca.fit(all_nodes)
        ev = pca.explained_variance_ratio_
        cumev = np.cumsum(ev)
        for thresh in [0.90, 0.95, 0.99]:
            n_comp = int(np.searchsorted(cumev, thresh)) + 1
            print(f"  Components for {thresh*100:.0f}% variance: {n_comp}")
        print("\n  Top-3 components — feature loadings:")
        for ci in range(min(3, 12)):
            top3 = np.argsort(np.abs(pca.components_[ci]))[::-1][:3]
            print(f"    PC{ci+1} (explains {ev[ci]*100:.1f}%): "
                  + ", ".join(f"{FEATURE_NAMES[j]}" for j in top3))
    except ImportError:
        print("  [SKIP] sklearn not available for PCA.")

    print("\n── Summary ─────────────────────────────────────────────────────────────")
    zero_var = [FEATURE_NAMES[i] for i in range(12) if np.var(all_nodes[:, i]) < 1e-4]
    if zero_var:
        print(f"  [CONFIRMED] Dead features (near-zero variance): {zero_var}")
    high_corr_pairs = [(FEATURE_NAMES[i], FEATURE_NAMES[j])
                       for i in range(12) for j in range(i+1, 12)
                       if abs(corr[i,j]) > 0.5]
    if high_corr_pairs:
        print(f"  [NEW FINDING] High-correlation pairs: {high_corr_pairs}")


def main():
    mode = "both"
    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        mode = sys.argv[idx + 1]

    if mode in ("20", "both"): task_20()
    if mode in ("21", "both"): task_21()

if __name__ == "__main__":
    main()
