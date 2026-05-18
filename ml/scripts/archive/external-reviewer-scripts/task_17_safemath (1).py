"""
Task 17: SafeMath Detection Viability Study
--------------------------------------------
Samples 200 IntegerUO=1 and 200 NonVulnerable contracts.
Greps source for SafeMath patterns. Checks if graph CALLS edges
detect SafeMath. Builds a confusion matrix: SafeMath vs IntegerUO label.

Run:
    python task_17_safemath.py
"""
import re
import random
import numpy as np
from pathlib import Path
from common import get_dirs, load_csv, load_graph, sol_from_graph, print_header

N_PER_GROUP = 200

SAFE_MATH_PATS = {
    "using_directive": re.compile(r'\busing\s+SafeMath\b'),
    "direct_call":     re.compile(r'\bSafeMath\.(mul|add|sub|div|mod)\b'),
    "inheritance":     re.compile(r'\bcontract\s+\w+\s+is\s+[^{]*\bSafeMath\b'),
    "any_ref":         re.compile(r'\bSafeMath\b'),
}

def detect_safemath(source: str) -> dict:
    return {k: bool(p.search(source)) for k, p in SAFE_MATH_PATS.items()}

def graph_has_safemath_edges(g) -> bool:
    """Check if any CALLS edge target node name contains 'SafeMath'."""
    metadata = getattr(g, "node_metadata", None)
    if metadata is None:
        return False
    for m in metadata:
        name = m.get("name", "") if isinstance(m, dict) else getattr(m, "name", "")
        if "safeMath" in name.lower() or "SafeMath" in name:
            return True
    return False

def main():
    print_header(17, "SafeMath Detection Viability Study")
    _, _, graphs_dir, _, _, bccc_dir = get_dirs()

    df = load_csv()
    int_uo = df[df["IntegerUO"] == 1]["md5_stem"].astype(str).tolist()
    nonvul = df[
        (df["IntegerUO"] == 0) &
        (df.drop(columns=["md5_stem", "IntegerUO"]).sum(axis=1) == 0)
    ]["md5_stem"].astype(str).tolist()

    print(f"  IntegerUO=1 pool: {len(int_uo):,}")
    print(f"  NonVulnerable pool: {len(nonvul):,}")

    random.seed(42)
    uo_sample   = random.sample(int_uo, min(N_PER_GROUP, len(int_uo)))
    safe_sample = random.sample(nonvul, min(N_PER_GROUP, len(nonvul)))

    rows = []  # (stem, group, safemath_in_source, sm_graph_edges)

    for group, stems in [("IntegerUO", uo_sample), ("NonVulnerable", safe_sample)]:
        for stem in stems:
            gpath = graphs_dir / f"{stem}.pt"
            if not gpath.exists():
                continue
            try:
                g = load_graph(gpath)
            except Exception:
                continue
            sol = sol_from_graph(g, bccc_dir)
            if sol is None:
                continue
            source = sol.read_text(errors="replace")
            sm = detect_safemath(source)
            sm_graph = graph_has_safemath_edges(g)
            rows.append({
                "stem": stem,
                "group": group,
                "any_ref": sm["any_ref"],
                "using": sm["using_directive"],
                "direct_call": sm["direct_call"],
                "inheritance": sm["inheritance"],
                "sm_graph": sm_graph,
            })

    if not rows:
        print("[ERROR] No data — check BCCC directory.")
        return

    uo_rows   = [r for r in rows if r["group"] == "IntegerUO"]
    safe_rows = [r for r in rows if r["group"] == "NonVulnerable"]

    def pct(num, den): return f"{num/den*100:.1f}%" if den else "N/A"

    print(f"\nAnalyzed: IntegerUO={len(uo_rows)}  NonVulnerable={len(safe_rows)}\n")

    print("SafeMath detection rates (% of contracts):")
    print(f"{'Pattern':<20} {'IntegerUO':>12} {'NonVulnerable':>14}")
    print("-" * 50)
    for key in ("any_ref", "using", "direct_call", "inheritance", "sm_graph"):
        uo_hit   = sum(1 for r in uo_rows   if r[key])
        safe_hit = sum(1 for r in safe_rows if r[key])
        print(f"  {key:<18} {pct(uo_hit, len(uo_rows)):>12} {pct(safe_hit, len(safe_rows)):>14}")

    # Confusion matrix: SafeMath present vs IntegerUO
    all_rows = rows
    sm_pres_uo    = sum(1 for r in all_rows if r["any_ref"] and r["group"] == "IntegerUO")
    sm_abs_uo     = sum(1 for r in all_rows if not r["any_ref"] and r["group"] == "IntegerUO")
    sm_pres_safe  = sum(1 for r in all_rows if r["any_ref"] and r["group"] == "NonVulnerable")
    sm_abs_safe   = sum(1 for r in all_rows if not r["any_ref"] and r["group"] == "NonVulnerable")

    print("\nConfusion matrix: SafeMath present vs IntegerUO label")
    print(f"{'':>20} {'SafeMath YES':>14} {'SafeMath NO':>12}")
    print(f"  {'IntegerUO=1':>18} {sm_pres_uo:>14} {sm_abs_uo:>12}")
    print(f"  {'NonVulnerable':>18} {sm_pres_safe:>14} {sm_abs_safe:>12}")

    # Precision / recall as feature
    tp = sm_abs_uo    # No SafeMath + IntegerUO → True Positive for vulnerability
    fn = sm_pres_uo   # SafeMath present but still IntegerUO → False Negative
    fp = sm_abs_safe  # No SafeMath + NonVulnerable → False Positive
    tn = sm_pres_safe # SafeMath + NonVulnerable → True Negative

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    print(f"\n  'No SafeMath' as IntegerUO predictor:")
    print(f"    Precision: {precision:.3f}  Recall: {recall:.3f}")

    print("\n── Summary ─────────────────────────────────────────────────────────────")
    if sm_abs_uo > 0.5 * len(uo_rows):
        print(f"  [CONFIRMED] Most IntegerUO contracts ({pct(sm_abs_uo, len(uo_rows))}) lack SafeMath — "
              "uses_safe_math is a viable replacement for in_unchecked.")
    else:
        print(f"  [NEW FINDING] Only {pct(sm_abs_uo, len(uo_rows))} of IntegerUO contracts lack SafeMath — "
              "feature may not be as discriminative as hoped.")

if __name__ == "__main__":
    main()
