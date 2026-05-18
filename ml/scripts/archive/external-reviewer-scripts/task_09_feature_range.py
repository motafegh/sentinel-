"""
Task 9: Full Feature Value Range Audit
--------------------------------------
Load 500 random graphs. For each of the 12 features compute min/max/percentiles
and count out-of-range values, split by declaration vs CFG node type.
Flags anything NEW beyond known BUG-1, BUG-2, BUG-3.

Run:
    python task_09_feature_range.py
"""
import sys
import numpy as np
from common import (
    get_dirs, load_graph, random_pt_sample,
    FEATURE_NAMES, DECL_THRESHOLD, print_header
)

N_GRAPHS = 500

def main():
    print_header(9, "Full Feature Value Range Audit")
    _, _, graphs_dir, _, _, _ = get_dirs()

    paths = random_pt_sample(graphs_dir, N_GRAPHS)
    print(f"Sampling {len(paths)} graphs from {graphs_dir}")

    all_decl = []  # list of [N_decl, 12] arrays
    all_cfg  = []  # list of [N_cfg, 12] arrays
    failed   = 0

    for p in paths:
        try:
            g = load_graph(p)
            x = g.x.numpy().astype(np.float32)  # [N, 12]
        except Exception as e:
            print(f"  [SKIP] {p.name}: {e}")
            failed += 1
            continue
        if x.shape[1] != 12:
            print(f"  [SKIP] {p.name}: x.shape={x.shape} (not 12-dim)")
            failed += 1
            continue
        is_cfg  = x[:, 0] >= DECL_THRESHOLD
        is_decl = ~is_cfg
        if is_decl.any():
            all_decl.append(x[is_decl])
        if is_cfg.any():
            all_cfg.append(x[is_cfg])

    if not all_decl and not all_cfg:
        print("[ERROR] No usable graphs loaded.")
        sys.exit(1)

    decl_mat = np.concatenate(all_decl, axis=0) if all_decl else np.empty((0, 12))
    cfg_mat  = np.concatenate(all_cfg,  axis=0) if all_cfg  else np.empty((0, 12))
    total    = len(decl_mat) + len(cfg_mat)

    print(f"\nLoaded {len(paths) - failed} graphs  |  "
          f"{len(decl_mat):,} declaration nodes  |  {len(cfg_mat):,} CFG nodes  |  "
          f"{total:,} total\n")

    # ── Per-feature table ────────────────────────────────────────────────────
    KNOWN_BUGS = {1, 5, 6}  # BUG-3, BUG-2, BUG-1 — do not re-flag as new

    header = (
        f"{'Feature':<28} | "
        f"{'Decl min':>8} {'Decl max':>8} {'Decl >1':>7} | "
        f"{'CFG min':>8} {'CFG max':>8} {'CFG >1':>7} | "
        f"{'NaN':>5} {'Inf':>5}"
    )
    print(header)
    print("-" * len(header))

    new_bugs = []
    for i, name in enumerate(FEATURE_NAMES):
        d = decl_mat[:, i] if len(decl_mat) else np.array([])
        c = cfg_mat[:, i]  if len(cfg_mat)  else np.array([])

        d_min  = float(np.min(d))   if len(d) else float("nan")
        d_max  = float(np.max(d))   if len(d) else float("nan")
        d_gt1  = int(np.sum(d > 1)) if len(d) else 0
        c_min  = float(np.min(c))   if len(c) else float("nan")
        c_max  = float(np.max(c))   if len(c) else float("nan")
        c_gt1  = int(np.sum(c > 1)) if len(c) else 0

        all_vals = np.concatenate([d, c]) if len(d) and len(c) else (d if len(d) else c)
        nan_ct = int(np.sum(np.isnan(all_vals)))
        inf_ct = int(np.sum(np.isinf(all_vals)))

        row = (
            f"{name:<28} | "
            f"{d_min:>8.3f} {d_max:>8.3f} {d_gt1:>7,} | "
            f"{c_min:>8.3f} {c_max:>8.3f} {c_gt1:>7,} | "
            f"{nan_ct:>5} {inf_ct:>5}"
        )
        print(row)

        # Flag NEW bugs (not in known set)
        if i not in KNOWN_BUGS:
            if d_gt1 > 0 or c_gt1 > 0:
                new_bugs.append((name, f"values > 1.0: decl={d_gt1}, cfg={c_gt1}"))
            if nan_ct > 0:
                new_bugs.append((name, f"NaN count={nan_ct}"))
            if inf_ct > 0:
                new_bugs.append((name, f"Inf count={inf_ct}"))

    # ── Percentile table ─────────────────────────────────────────────────────
    print("\nPercentile breakdown (ALL nodes combined):")
    print(f"{'Feature':<28} | {'p5':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7} {'p99':>7}")
    print("-" * 80)
    all_mat = np.concatenate([decl_mat, cfg_mat], axis=0) if total else np.empty((0, 12))
    for i, name in enumerate(FEATURE_NAMES):
        col = all_mat[:, i]
        ps = np.percentile(col, [5, 25, 50, 75, 95, 99])
        print(f"{name:<28} | {ps[0]:>7.3f} {ps[1]:>7.3f} {ps[2]:>7.3f} "
              f"{ps[3]:>7.3f} {ps[4]:>7.3f} {ps[5]:>7.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Graphs loaded:     {len(paths) - failed} / {len(paths)}")
    print(f"  Total nodes:       {total:,}")
    print(f"  Declaration nodes: {len(decl_mat):,}")
    print(f"  CFG nodes:         {len(cfg_mat):,}")
    print()
    print("  Known bugs confirmed (not re-flagged):")
    print("    BUG-1: CFG loc [6] raw — check CFG max above")
    print("    BUG-2: complexity [5] raw — check CFG max above")
    print("    BUG-3: visibility [1] ordinal 0/1/2 — max should be 2.0")
    print()
    if new_bugs:
        print("  [NEW FINDING] Unexpected out-of-range values:")
        for feat, msg in new_bugs:
            print(f"    {feat}: {msg}")
    else:
        print("  [CONFIRMED] No new out-of-range features beyond known bugs.")

if __name__ == "__main__":
    main()
