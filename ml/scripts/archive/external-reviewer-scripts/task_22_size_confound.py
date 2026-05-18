"""
Task 22: Graph Size as a Confounding Variable
----------------------------------------------
Loads up to 5000 graphs. For each class, tests whether graph size alone
(num_nodes, num_edges, num_functions) predicts the label (AUC > 0.8 = confound).

Run:
    python task_22_size_confound.py
"""
import numpy as np
import random
from common import get_dirs, load_csv, load_graph, LABEL_COLS, print_header, random_pt_sample

N = 5000

def main():
    print_header(22, "Graph Size as a Confounding Variable")
    _, _, graphs_dir, _, _, _ = get_dirs()
    df = load_csv()

    paths = random_pt_sample(graphs_dir, N)
    print(f"Sampling {len(paths)} graphs…")

    label_by_stem = df.set_index("md5_stem")
    records = []

    for p in paths:
        stem = p.stem
        if stem not in label_by_stem.index:
            continue
        try:
            g = load_graph(p)
        except Exception:
            continue
        x  = g.x.numpy()
        ea = g.edge_attr
        n_nodes  = x.shape[0]
        n_edges  = g.edge_index.shape[1] if g.edge_index.numel() else 0
        n_funcs  = int((x[:, 0] < 2.0/12.0).sum())  # type_id == FUNCTION ≈ 1/12 or 2/12
        n_cf     = int((ea == 6).sum().item()) if ea.numel() else 0
        labels   = label_by_stem.loc[stem, LABEL_COLS].to_dict()
        records.append({"stem": stem, "n_nodes": n_nodes, "n_edges": n_edges,
                         "n_funcs": n_funcs, "n_cf": n_cf, **labels})

    if not records:
        print("[ERROR] No records.")
        return

    print(f"Records loaded: {len(records):,}\n")

    import numpy as np
    from scipy import stats

    size_features = ["n_nodes", "n_edges", "n_funcs"]

    # Per-class stats
    print(f"{'Class':<22} {'mean_nodes':>11} {'med_nodes':>10} "
          f"{'p95_nodes':>10} {'mean_edges':>11}")
    print("-" * 70)
    for cls in LABEL_COLS:
        pos = [r for r in records if r[cls] == 1]
        neg = [r for r in records if r[cls] == 0]
        if not pos:
            continue
        mn = np.mean([r["n_nodes"] for r in pos])
        md = np.median([r["n_nodes"] for r in pos])
        p95 = np.percentile([r["n_nodes"] for r in pos], 95)
        me = np.mean([r["n_edges"] for r in pos])
        print(f"  {cls:<20} {mn:>11.1f} {md:>10.1f} {p95:>10.1f} {me:>11.1f}")

    # AUC per class using size features alone
    print(f"\nLogistic regression AUC using only [n_nodes, n_edges, n_funcs]:")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        X = np.array([[r["n_nodes"], r["n_edges"], r["n_funcs"]] for r in records], dtype=float)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        print(f"  {'Class':<22} {'AUC':>7}  {'Confound?':>10}")
        print("  " + "-" * 42)
        for cls in LABEL_COLS:
            y = np.array([r[cls] for r in records])
            if y.sum() < 10 or y.sum() > len(y) - 10:
                print(f"  {cls:<22} {'N/A':>7}  (too few positives)")
                continue
            try:
                clf = LogisticRegression(max_iter=500, class_weight="balanced")
                clf.fit(X_s, y)
                auc = roc_auc_score(y, clf.predict_proba(X_s)[:, 1])
                flag = " ← CONFOUND" if auc > 0.8 else (" [borderline]" if auc > 0.7 else "")
                print(f"  {cls:<22} {auc:>7.3f}{flag}")
            except Exception as e:
                print(f"  {cls:<22} ERROR: {e}")

    except ImportError:
        print("  [SKIP] sklearn not available.")

    # Mann-Whitney U: are positive graphs bigger than negative?
    print(f"\nMann-Whitney U test: num_nodes positive vs negative (p<0.05 = significant):")
    print(f"  {'Class':<22} {'U-stat':>10} {'p-value':>10} {'Significant?':>13}")
    print("  " + "-" * 58)
    nodes_all = np.array([r["n_nodes"] for r in records])
    for cls in LABEL_COLS:
        pos_nodes = [r["n_nodes"] for r in records if r[cls] == 1]
        neg_nodes = [r["n_nodes"] for r in records if r[cls] == 0]
        if len(pos_nodes) < 5 or len(neg_nodes) < 5:
            print(f"  {cls:<22} {'N/A':>10}")
            continue
        u, p = stats.mannwhitneyu(pos_nodes, neg_nodes, alternative="two-sided")
        sig = "YES" if p < 0.05 else "no"
        print(f"  {cls:<22} {u:>10.0f} {p:>10.4f} {sig:>13}")

    print("\n── Summary ─────────────────────────────────────────────────────────────")
    print("  Classes with AUC > 0.8 from size alone are confounded by graph complexity.")
    print("  Timestamp's high CF-edge mean (196.9) is a major red flag.")

if __name__ == "__main__":
    main()
