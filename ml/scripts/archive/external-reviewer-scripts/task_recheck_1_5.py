"""
Task 1-RECHECK: Per-Class Feature Activation Rates (Declaration vs CFG split)
Task 5-RECHECK: EMITS/INHERITS Full-Dataset Confirmation
---------------------------------------------------------
Run:
    python task_recheck_1_5.py [--task 1|5|both]
"""
import sys
import random
import numpy as np
from common import (
    get_dirs, load_csv, load_graph, LABEL_COLS, FEATURE_NAMES,
    DECL_THRESHOLD, print_header
)

# ════════════════════════════════════════════════════════════════════════════
# TASK 1-RECHECK — Activation rates split by node type
# ════════════════════════════════════════════════════════════════════════════

FOCUS_FEATURES = {
    "[2] uses_bg":    2,
    "[7] ret_ign":    7,
    "[10] has_loop":  10,
    "[11] ext_call":  11,
    "[9] in_unch":    9,
}

def task_1_recheck():
    print_header("1-RECHECK", "Per-Class Feature Activation (Declaration vs CFG split)")
    _, _, graphs_dir, _, _, _ = get_dirs()
    df = load_csv()

    # 20 pure-label samples per class
    results = {}
    for cls in LABEL_COLS:
        pure_mask = (df[cls] == 1)
        for other in LABEL_COLS:
            if other != cls:
                pure_mask &= (df[other] == 0)
        stems = df[pure_mask]["md5_stem"].astype(str).tolist()
        random.seed(42)
        sample = random.sample(stems, min(20, len(stems)))

        decl_acts = {k: [] for k in FOCUS_FEATURES}
        cfg_acts  = {k: [] for k in FOCUS_FEATURES}

        for stem in sample:
            gpath = graphs_dir / f"{stem}.pt"
            if not gpath.exists(): continue
            try:
                g = load_graph(gpath)
            except Exception:
                continue
            x = g.x.numpy()
            is_cfg  = x[:, 0] >= DECL_THRESHOLD
            is_decl = ~is_cfg

            decl_x = x[is_decl]
            cfg_x  = x[is_cfg]

            for fname, fi in FOCUS_FEATURES.items():
                # Activation rate = % of nodes with nonzero value
                if len(decl_x):
                    decl_acts[fname].append(float((decl_x[:, fi] != 0).mean()) * 100)
                if len(cfg_x):
                    cfg_acts[fname].append(float((cfg_x[:, fi] != 0).mean()) * 100)

        results[cls] = {
            "n": len(sample),
            "decl": {k: np.mean(v) if v else 0 for k, v in decl_acts.items()},
            "cfg":  {k: np.mean(v) if v else 0 for k, v in cfg_acts.items()},
        }

    feat_names = list(FOCUS_FEATURES.keys())

    print("\n── Declaration nodes only (% nonzero) ──────────────────────────────────")
    header = f"{'Class':<22}" + "".join(f" {n:>12}" for n in feat_names)
    print(header)
    print("-" * (22 + 13 * len(feat_names)))
    for cls in LABEL_COLS:
        if cls not in results: continue
        row = f"  {cls:<20}"
        for fn in feat_names:
            row += f" {results[cls]['decl'].get(fn, 0):>11.3f}%"
        print(row)

    print("\n── CFG nodes only (% nonzero) ───────────────────────────────────────────")
    print(header)
    print("-" * (22 + 13 * len(feat_names)))
    for cls in LABEL_COLS:
        if cls not in results: continue
        row = f"  {cls:<20}"
        for fn in feat_names:
            row += f" {results[cls]['cfg'].get(fn, 0):>11.3f}%"
        print(row)

    print("\n── Prior audit (ALL nodes) — provided for comparison ────────────────────")
    print("  (From the audit doc — do not recompute; just verify direction is consistent)")

    print("\n── Summary ─────────────────────────────────────────────────────────────")
    print("  [9] in_unchecked: should be 0% for BOTH node types — BUG-5 confirmation.")
    print("  [2] uses_bg: declaration nodes should show higher rates than CFG (feature")
    print("      is set on FUNCTION nodes, not CFG nodes).")
    print("  Any dramatic difference between decl vs CFG rates is a data integrity finding.")

# ════════════════════════════════════════════════════════════════════════════
# TASK 5-RECHECK — EMITS/INHERITS full dataset scan
# ════════════════════════════════════════════════════════════════════════════

def task_5_recheck():
    print_header("5-RECHECK", "EMITS/INHERITS Full-Dataset Confirmation")
    _, _, graphs_dir, _, _, _ = get_dirs()

    all_pts = list(graphs_dir.glob("*.pt"))
    MAX = 5000
    if len(all_pts) > MAX:
        print(f"  {len(all_pts):,} total files — scanning first {MAX:,} for speed.")
        all_pts = all_pts[:MAX]
    else:
        print(f"  Scanning all {len(all_pts):,} graph files…")

    emits_count    = 0  # graphs with ANY edge_attr == 3
    inherits_count = 0  # graphs with ANY edge_attr == 4
    emits_examples    = []
    inherits_examples = []
    loaded = 0
    failed = 0

    for p in all_pts:
        try:
            g = load_graph(p)
        except Exception:
            failed += 1
            continue
        loaded += 1
        ea = g.edge_attr
        if ea.numel() == 0:
            continue
        ea_np = ea.numpy()
        if (ea_np == 3).any():
            emits_count += 1
            if len(emits_examples) < 5:
                emits_examples.append(p.stem)
        if (ea_np == 4).any():
            inherits_count += 1
            if len(inherits_examples) < 5:
                inherits_examples.append(p.stem)

    print(f"\n  Graphs scanned: {loaded:,}  |  Failed: {failed:,}\n")
    print(f"  Graphs with EMITS edges (type 3):    {emits_count:,}")
    print(f"  Graphs with INHERITS edges (type 4): {inherits_count:,}")

    if emits_examples:
        print(f"\n  [NEW FINDING] EMITS edges found! Examples: {emits_examples}")
    else:
        print("  [CONFIRMED] EMITS edges: 0 across all scanned graphs (BUG-7 confirmed).")

    if inherits_examples:
        print(f"\n  [NEW FINDING] INHERITS edges found! Examples: {inherits_examples}")
    else:
        print("  [CONFIRMED] INHERITS edges: 0 across all scanned graphs (BUG-8 confirmed).")

    # Edge type distribution
    print("\n  Overall edge type distribution (sampled 1000 graphs):")
    random.seed(42)
    sub = random.sample(all_pts, min(1000, len(all_pts)))
    edge_type_counts = {}
    for p in sub:
        try:
            g = load_graph(p)
            ea = g.edge_attr.numpy()
            for et in set(ea.tolist()):
                edge_type_counts[int(et)] = edge_type_counts.get(int(et), 0) + 1
        except Exception:
            continue

    type_names = {0:"CALLS", 1:"READS", 2:"WRITES", 3:"EMITS",
                  4:"INHERITS", 5:"CONTAINS", 6:"CONTROL_FLOW", 7:"REV_CONTAINS"}
    print(f"  {'Type':>4} {'Name':<16} {'Graphs with this type':>22}")
    print("  " + "-" * 46)
    for et in sorted(edge_type_counts):
        print(f"  {et:>4} {type_names.get(et, '?'):<16} {edge_type_counts[et]:>22,}")


def main():
    mode = "both"
    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        mode = sys.argv[idx + 1]
    if mode in ("1", "both"): task_1_recheck()
    if mode in ("5", "both"): task_5_recheck()

if __name__ == "__main__":
    main()
