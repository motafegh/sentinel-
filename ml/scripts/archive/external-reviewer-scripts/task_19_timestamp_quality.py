"""
Task 19: Timestamp Label Quality Comprehensive Audit
-----------------------------------------------------
For ALL Timestamp=1 contracts (~2191), finds the .sol source,
greps for block global patterns, and checks if the graph feature fires.
Classifies into (a)/(b)/(c)/(d) categories.

Run:
    python task_19_timestamp_quality.py
"""
import re
from pathlib import Path
from common import get_dirs, load_csv, load_graph, sol_from_graph, print_header, FEATURE_NAMES

BLOCK_GLOBAL_PATS = [
    re.compile(r'\bblock\.timestamp\b'),
    re.compile(r'\bblock\.number\b'),
    re.compile(r'\bblock\.difficulty\b'),
    re.compile(r'\bblock\.basefee\b'),
    re.compile(r'\bblockhash\b'),
    re.compile(r'\bnow\b'),   # pre-0.7.0 alias for block.timestamp
]

def source_has_block_globals(source: str) -> bool:
    return any(p.search(source) for p in BLOCK_GLOBAL_PATS)

def graph_bg_fires(g) -> bool:
    """Check if any node has uses_block_globals [2] == 1.0."""
    x = g.x.numpy()
    return bool((x[:, 2] == 1.0).any())

def main():
    print_header(19, "Timestamp Label Quality Comprehensive Audit")
    _, _, graphs_dir, _, _, bccc_dir = get_dirs()

    df = load_csv()
    ts_stems = df[df["Timestamp"] == 1]["md5_stem"].astype(str).tolist()
    print(f"  Timestamp=1 contracts in CSV: {len(ts_stems):,}")

    if bccc_dir is None:
        print("[ERROR] BCCC directory not found.")
        return

    # Categories:
    # (a) source has BG AND feature fires → correct
    # (b) source has BG but feature does NOT fire → extraction failure / BUG-6
    # (c) source has NO BG AND feature doesn't fire → likely mislabelled
    # (d) source has NO BG but feature DOES fire → false positive feature
    cats = {"a": [], "b": [], "c": [], "d": [], "no_sol": [], "load_err": []}

    for i, stem in enumerate(ts_stems):
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(ts_stems)}…")
        gpath = graphs_dir / f"{stem}.pt"
        if not gpath.exists():
            cats["load_err"].append(stem)
            continue
        try:
            g = load_graph(gpath)
        except Exception:
            cats["load_err"].append(stem)
            continue
        sol = sol_from_graph(g, bccc_dir)
        if sol is None:
            cats["no_sol"].append(stem)
            continue
        source = sol.read_text(errors="replace")
        has_bg  = source_has_block_globals(source)
        fires   = graph_bg_fires(g)

        if   has_bg and fires:      cats["a"].append(stem)
        elif has_bg and not fires:  cats["b"].append(stem)
        elif not has_bg and not fires: cats["c"].append(stem)
        else:                       cats["d"].append(stem)  # no_bg but fires

    total = sum(len(v) for k, v in cats.items() if k not in ("no_sol", "load_err"))
    resolved = total

    print(f"\nTimestamp Label Quality Audit (n={resolved} resolved, "
          f"{len(cats['no_sol'])} no .sol, {len(cats['load_err'])} load errors)\n")

    def pct(lst): return f"{len(lst)/resolved*100:.1f}%" if resolved else "N/A"

    print(f"  (a) Source has BG AND feature fires:          {len(cats['a']):>5}  ({pct(cats['a'])})"
          "  [CONFIRMED correct]")
    print(f"  (b) Source has BG but feature does NOT fire:  {len(cats['b']):>5}  ({pct(cats['b'])})"
          "  [likely BUG-6 wrong contract]")
    print(f"  (c) Source has NO BG, feature doesn't fire:   {len(cats['c']):>5}  ({pct(cats['c'])})"
          "  [likely MISLABELLED in BCCC]")
    print(f"  (d) Source has NO BG but feature FIRES:       {len(cats['d']):>5}  ({pct(cats['d'])})"
          "  [false positive feature — investigate]")

    # Sample (c) contracts for investigation
    if cats["c"]:
        print(f"\n  Sample of category (c) — mislabelled Timestamp contracts:")
        for stem in cats["c"][:10]:
            try:
                g = load_graph(graphs_dir / f"{stem}.pt")
                cp = getattr(g, "contract_path", "?")
            except Exception:
                cp = "?"
            print(f"    {stem}  path={cp}")

    if cats["b"]:
        print(f"\n  Sample of category (b) — BG in source but feature doesn't fire:")
        for stem in cats["b"][:10]:
            try:
                g = load_graph(graphs_dir / f"{stem}.pt")
                cp = getattr(g, "contract_path", "?")
            except Exception:
                cp = "?"
            print(f"    {stem}  path={cp}")

    # Summary finding
    print("\n── Summary ─────────────────────────────────────────────────────────────")
    mislabelled_pct = len(cats["c"]) / resolved * 100 if resolved else 0
    if mislabelled_pct > 30:
        print(f"  [BUG] {mislabelled_pct:.1f}% of Timestamp contracts appear mislabelled "
              "(no block globals in source AND feature doesn't fire).")
        print("  Severity: HIGH — Timestamp class is mostly noise.")
    elif mislabelled_pct > 15:
        print(f"  [NEW FINDING] {mislabelled_pct:.1f}% mislabelled — significant noise in Timestamp class.")
    else:
        print(f"  [CONFIRMED] Mislabelling rate {mislabelled_pct:.1f}% — within acceptable range.")

    cat_b_pct = len(cats["b"]) / resolved * 100 if resolved else 0
    if cat_b_pct > 10:
        print(f"  [CONFIRMED BUG-6] {cat_b_pct:.1f}% have BG in source but feature absent — "
              "likely wrong contract selected.")

if __name__ == "__main__":
    main()
