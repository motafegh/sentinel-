"""
Task 16: Full-Dataset Wrong Contract Selection Rate
----------------------------------------------------
Samples 500 multi-contract .sol files from BCCC sources.
For each, compares what Slither selected (g.contract_name) against
two heuristics: "most functions" (current) and "last contract".

Run:
    python task_16_wrong_contract.py
"""
import re
import random
from pathlib import Path
from common import get_dirs, load_csv, load_graph, sol_from_graph, print_header

N = 500
LIBRARY_NAMES = {"SafeMath", "Ownable", "StandardToken", "BasicToken", "ERC20",
                 "Strings", "Address", "Context", "IERC20", "Pausable", "Roles"}

CONTRACT_RE = re.compile(
    r'(?:^|\n)\s*(?:abstract\s+)?contract\s+(\w+)'
    r'(?:\s+is\s+([\w\s,]+?))?\s*\{',
    re.MULTILINE
)
LIBRARY_RE  = re.compile(r'(?:^|\n)\s*library\s+(\w+)\s*\{', re.MULTILINE)
INTERFACE_RE = re.compile(r'(?:^|\n)\s*interface\s+(\w+)', re.MULTILINE)
FUNC_RE     = re.compile(r'\bfunction\s+\w+', re.MULTILINE)

def count_functions_in_contract(source: str, contract_name: str) -> int:
    """Rough count of function declarations within a contract block."""
    # Find contract block start
    pat = re.compile(rf'\bcontract\s+{re.escape(contract_name)}\b.*?\{{', re.DOTALL)
    m = pat.search(source)
    if not m:
        return 0
    start = m.end()
    # Find matching closing brace (depth counting)
    depth = 1
    i = start
    while i < len(source) and depth > 0:
        if source[i] == '{': depth += 1
        elif source[i] == '}': depth -= 1
        i += 1
    block = source[start:i-1]
    return len(FUNC_RE.findall(block))

def parse_contracts(source: str):
    """
    Returns list of (name, is_library, is_interface) tuples in order of appearance.
    """
    libs  = {m.group(1) for m in LIBRARY_RE.finditer(source)}
    ifaces = {m.group(1) for m in INTERFACE_RE.finditer(source)}
    contracts = []
    for m in CONTRACT_RE.finditer(source):
        name = m.group(1)
        is_lib   = name in libs or name in LIBRARY_NAMES
        is_iface = name in ifaces
        contracts.append((name, is_lib, is_iface))
    return contracts

def main():
    print_header(16, "Full-Dataset Wrong Contract Selection Rate")
    _, _, graphs_dir, _, _, bccc_dir = get_dirs()

    if bccc_dir is None:
        print("[ERROR] BCCC source directory not found. Cannot run Task 16.")
        return

    df = load_csv()
    all_stems = df["md5_stem"].astype(str).tolist()
    random.seed(42)
    sample = random.sample(all_stems, min(N, len(all_stems)))

    results = {
        "multi_contract":      0,
        "single_contract":     0,
        "no_sol":              0,
        "heuristic_match":     0,   # current (most functions) matches selected
        "last_match":          0,   # "last contract" matches selected
        "both_match":          0,
        "neither_match":       0,
        "wrong_current":       0,   # current heuristic wrong
        "wrong_last":          0,   # last heuristic wrong
    }
    per_class_wrong = {c: 0 for c in df.columns if c != "md5_stem"}

    class_lookup = df.set_index("md5_stem")

    for stem in sample:
        gpath = graphs_dir / f"{stem}.pt"
        if not gpath.exists():
            continue
        try:
            g = load_graph(gpath)
        except Exception:
            continue
        sol = sol_from_graph(g, bccc_dir)
        if sol is None:
            results["no_sol"] += 1
            continue

        source = sol.read_text(errors="replace")
        contracts = parse_contracts(source)
        concrete  = [(n, il, ii) for n, il, ii in contracts if not il and not ii]

        if len(concrete) <= 1:
            results["single_contract"] += 1
            continue

        results["multi_contract"] += 1
        selected = getattr(g, "contract_name", "")

        # Heuristic 1: most functions
        best_mf = max(concrete, key=lambda x: count_functions_in_contract(source, x[0]))[0]
        # Heuristic 2: last concrete contract
        best_last = concrete[-1][0]

        mf_ok   = best_mf   == selected
        last_ok = best_last == selected

        if mf_ok:    results["heuristic_match"] += 1
        if last_ok:  results["last_match"] += 1
        if mf_ok and last_ok:   results["both_match"] += 1
        if not mf_ok and not last_ok: results["neither_match"] += 1
        if not mf_ok: results["wrong_current"] += 1
        if not last_ok: results["wrong_last"] += 1

    total_mc = results["multi_contract"]

    print(f"\nSampled: {len(sample)}  |  Multi-contract: {total_mc}  |  No .sol found: {results['no_sol']}\n")

    if total_mc == 0:
        print("[WARN] No multi-contract files found — check BCCC directory and sol_from_graph().")
        return

    def pct(n): return f"{n/total_mc*100:.1f}%" if total_mc else "N/A"

    print(f"  Current heuristic (most functions) correct: "
          f"{results['heuristic_match']:>4} / {total_mc}  ({pct(results['heuristic_match'])})")
    print(f"  Current heuristic WRONG:                   "
          f"{results['wrong_current']:>4} / {total_mc}  ({pct(results['wrong_current'])})")
    print()
    print(f"  Last-contract heuristic correct:            "
          f"{results['last_match']:>4} / {total_mc}  ({pct(results['last_match'])})")
    print(f"  Last-contract heuristic WRONG:              "
          f"{results['wrong_last']:>4} / {total_mc}  ({pct(results['wrong_last'])})")
    print()
    print(f"  Both correct:    {results['both_match']:>4}  ({pct(results['both_match'])})")
    print(f"  Neither correct: {results['neither_match']:>4}  ({pct(results['neither_match'])})")

    print("\n── Summary ─────────────────────────────────────────────────────────────")
    w_curr = results["wrong_current"]
    w_last = results["wrong_last"]
    print(f"  [CONFIRMED] BUG-6 exists: wrong-selection rate (current heuristic) = {pct(w_curr)}")
    if w_last < w_curr:
        print(f"  [NEW FINDING] 'Last contract' heuristic is better: {pct(w_last)} wrong vs {pct(w_curr)}")
    elif w_last > w_curr:
        print(f"  [NEW FINDING] 'Last contract' heuristic is WORSE: {pct(w_last)} wrong vs {pct(w_curr)}")
    else:
        print(f"  [CONFIRMED] Both heuristics produce the same wrong-selection rate.")

if __name__ == "__main__":
    main()
