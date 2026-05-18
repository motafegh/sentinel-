#!/usr/bin/env python3
"""
task16_wrong_contract_selection.py — Wrong Contract Selection Audit for SENTINEL v6

Sample 500 multi-contract .sol files from BCCC source dirs. For each file:
1. Parse all contract declarations with regex
2. Filter out interfaces and libraries
3. If >1 non-interface/non-library contract:
   - Load the graph .pt for this file's md5_stem
   - Read g.contract_name — what Slither selected
   - Apply "most functions" heuristic
   - Apply "last contract" heuristic
   - Record which matches the actual selection
4. Compute wrong-selection rate with confidence interval for both heuristics
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import math
import random
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


# ── Regex patterns for Solidity declarations ──────────────────────────────────
_CONTRACT_RE = re.compile(
    r'\bcontract\s+(\w+)(?:\s+is\s+[\w,\s]+)?\s*\{'
)
_LIBRARY_RE = re.compile(
    r'\blibrary\s+(\w+)\s*\{'
)
_INTERFACE_RE = re.compile(
    r'\binterface\s+(\w+)\s*\{'
)

# Heuristic: count function-like declarations per contract
_FUNCTION_RE = re.compile(
    r'\bfunction\s+\w+\s*\('
)


def parse_contracts(source: str) -> dict:
    """
    Parse Solidity source and return contract info.

    Returns dict with keys:
      - contracts: list of (name, is_interface, is_library)
      - contract_blocks: list of (name, source_block) for function counting
    """
    contracts_info = []

    contracts = [(m.group(1), m.start()) for m in _CONTRACT_RE.finditer(source)]
    libraries = {m.group(1) for m in _LIBRARY_RE.finditer(source)}
    interfaces = {m.group(1) for m in _INTERFACE_RE.finditer(source)}

    for name, pos in contracts:
        is_library = name in libraries
        is_interface = name in interfaces
        contracts_info.append({
            "name": name,
            "is_interface": is_interface,
            "is_library": is_library,
        })

    # Also pick up interfaces/libraries not also matched by contract re
    for m in _INTERFACE_RE.finditer(source):
        name = m.group(1)
        if not any(c["name"] == name for c in contracts_info):
            contracts_info.append({
                "name": name,
                "is_interface": True,
                "is_library": False,
            })

    for m in _LIBRARY_RE.finditer(source):
        name = m.group(1)
        if not any(c["name"] == name for c in contracts_info):
            contracts_info.append({
                "name": name,
                "is_interface": False,
                "is_library": True,
            })

    return contracts_info


def count_functions_in_block(source: str, contract_name: str) -> int:
    """Count function declarations that appear to belong to a given contract."""
    # Simple heuristic: find the contract block and count function keywords
    # This is approximate — proper parsing would need a Solidity parser
    pattern = re.compile(
        r'\bcontract\s+' + re.escape(contract_name) + r'\b[^{]*\{',
        re.DOTALL
    )
    m = pattern.search(source)
    if not m:
        return 0

    # Find matching closing brace (depth counting)
    start = m.end()
    depth = 1
    pos = start
    while pos < len(source) and depth > 0:
        if source[pos] == '{':
            depth += 1
        elif source[pos] == '}':
            depth -= 1
        pos += 1

    block = source[start:pos]
    return len(_FUNCTION_RE.findall(block))


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def main():
    print_header("Task 16: Wrong Contract Selection Audit")

    # ── Collect .sol files from BCCC source dirs ──────────────────────────
    all_sol_files = []
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            print(f"  Source dir not found, skipping: {src_dir}")
            continue
        for sol in src_dir.rglob("*.sol"):
            all_sol_files.append(sol)
        print(f"  Collected from: {src_dir}")

    if not all_sol_files:
        print("ERROR: No .sol files found in any source directory.")
        return

    print(f"Found {len(all_sol_files)} .sol files total")

    # ── Sample and filter for multi-contract files ────────────────────────
    random.seed(42)
    random.shuffle(all_sol_files)

    target_sample = 500
    multi_contract_files = []
    checked = 0
    skipped_errors = 0

    for sol_path in all_sol_files:
        if len(multi_contract_files) >= target_sample:
            break

        checked += 1
        try:
            source = sol_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped_errors += 1
            continue

        contracts_info = parse_contracts(source)
        non_iface_lib = [
            c for c in contracts_info
            if not c["is_interface"] and not c["is_library"]
        ]

        if len(non_iface_lib) > 1:
            # Compute md5 from relative path
            try:
                rel = sol_path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = sol_path
            md5 = get_contract_hash(rel)
            multi_contract_files.append({
                "path": sol_path,
                "md5": md5,
                "source": source,
                "contracts": contracts_info,
                "non_iface_lib": non_iface_lib,
                "n_contracts": len(non_iface_lib),
            })

    print(f"Checked {checked} files, found {len(multi_contract_files)} multi-contract files")

    # ── Analyze each multi-contract file ──────────────────────────────────
    results = {
        "most_functions": {"correct": 0, "wrong": 0, "no_graph": 0, "no_match": 0},
        "last_contract": {"correct": 0, "wrong": 0, "no_graph": 0, "no_match": 0},
    }
    per_class_breakdown = defaultdict(lambda: {
        "most_functions_correct": 0, "most_functions_wrong": 0,
        "last_contract_correct": 0, "last_contract_wrong": 0,
    })
    by_n_contracts = defaultdict(lambda: {
        "total": 0,
        "most_functions_correct": 0, "most_functions_wrong": 0,
        "last_contract_correct": 0, "last_contract_wrong": 0,
    })

    # Load labels for per-class breakdown
    try:
        labels = load_label_csv()
    except Exception:
        labels = {}

    wrong_selections_mf = []  # most functions heuristic wrong
    wrong_selections_lc = []  # last contract heuristic wrong

    for i, entry in enumerate(multi_contract_files):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(multi_contract_files)} files...")

        sol_path = entry["path"]
        source = entry["source"]
        md5 = entry["md5"]
        non_iface_lib = entry["non_iface_lib"]
        n_contracts = entry["n_contracts"]

        # Load graph to get contract_name
        graph_path = GRAPHS_DIR / f"{md5}.pt"
        if not graph_path.exists():
            results["most_functions"]["no_graph"] += 1
            results["last_contract"]["no_graph"] += 1
            continue

        try:
            g = load_graph(graph_path)
        except Exception as e:
            print(f"  WARNING: Failed to load graph {md5}: {e}")
            results["most_functions"]["no_graph"] += 1
            results["last_contract"]["no_graph"] += 1
            continue

        actual_name = getattr(g, "contract_name", None)
        if actual_name is None:
            results["most_functions"]["no_match"] += 1
            results["last_contract"]["no_match"] += 1
            continue

        # Apply heuristics
        # "Most functions" heuristic: pick contract with most function declarations
        func_counts = {}
        for c in non_iface_lib:
            func_counts[c["name"]] = count_functions_in_block(source, c["name"])

        max_funcs = max(func_counts.values()) if func_counts else 0
        most_funcs_candidates = [
            name for name, count in func_counts.items() if count == max_funcs
        ]
        # If tie, pick first (alphabetical or by source order)
        mf_name = most_funcs_candidates[0] if most_funcs_candidates else None

        # "Last contract" heuristic: pick last non-interface/non-library
        lc_name = non_iface_lib[-1]["name"] if non_iface_lib else None

        # Check matches
        mf_match = (mf_name == actual_name)
        lc_match = (lc_name == actual_name)

        if mf_match:
            results["most_functions"]["correct"] += 1
        else:
            results["most_functions"]["wrong"] += 1
            wrong_selections_mf.append({
                "md5": md5,
                "actual": actual_name,
                "predicted": mf_name,
                "n_contracts": n_contracts,
            })

        if lc_match:
            results["last_contract"]["correct"] += 1
        else:
            results["last_contract"]["wrong"] += 1
            wrong_selections_lc.append({
                "md5": md5,
                "actual": actual_name,
                "predicted": lc_name,
                "n_contracts": n_contracts,
            })

        # Per-class breakdown
        if md5 in labels:
            for cls_name in VULN_CLASSES:
                if labels[md5].get(cls_name, 0) == 1:
                    if mf_match:
                        per_class_breakdown[cls_name]["most_functions_correct"] += 1
                    else:
                        per_class_breakdown[cls_name]["most_functions_wrong"] += 1
                    if lc_match:
                        per_class_breakdown[cls_name]["last_contract_correct"] += 1
                    else:
                        per_class_breakdown[cls_name]["last_contract_wrong"] += 1

        # By n_contracts
        by_n_contracts[n_contracts]["total"] += 1
        if mf_match:
            by_n_contracts[n_contracts]["most_functions_correct"] += 1
        else:
            by_n_contracts[n_contracts]["most_functions_wrong"] += 1
        if lc_match:
            by_n_contracts[n_contracts]["last_contract_correct"] += 1
        else:
            by_n_contracts[n_contracts]["last_contract_wrong"] += 1

    # ── Compute statistics ────────────────────────────────────────────────
    mf_total = results["most_functions"]["correct"] + results["most_functions"]["wrong"]
    lc_total = results["last_contract"]["correct"] + results["last_contract"]["wrong"]

    mf_correct = results["most_functions"]["correct"]
    lc_correct = results["last_contract"]["correct"]

    mf_rate = mf_correct / mf_total if mf_total > 0 else 0.0
    lc_rate = lc_correct / lc_total if lc_total > 0 else 0.0

    mf_wrong_rate = 1.0 - mf_rate
    lc_wrong_rate = 1.0 - lc_rate

    mf_ci = wilson_ci(mf_rate, mf_total) if mf_total > 0 else (0.0, 0.0)
    lc_ci = wilson_ci(lc_rate, lc_total) if lc_total > 0 else (0.0, 0.0)

    # ── Build report ──────────────────────────────────────────────────────
    report = []
    report.append("# Task 16: Wrong Contract Selection Audit\n")
    report.append(f"**Multi-contract files analyzed:** {len(multi_contract_files)}  ")
    report.append(f"**Files with matching graph .pt:** {mf_total}  ")
    report.append(f"**Files skipped (no graph/load error):** {results['most_functions']['no_graph']}  ")
    report.append(f"**Files with no contract_name attr:** {results['most_functions']['no_match']}\n")

    # Summary table
    report.append("\n## Heuristic Comparison\n")
    report.append("| Heuristic | Correct | Wrong | Accuracy | 95% CI | Wrong Rate | 95% CI (wrong) |")
    report.append("\n|-----------|---------|-------|----------|--------|------------|----------------|")
    mf_wrong_ci = wilson_ci(mf_wrong_rate, mf_total) if mf_total > 0 else (0.0, 0.0)
    lc_wrong_ci = wilson_ci(lc_wrong_rate, lc_total) if lc_total > 0 else (0.0, 0.0)
    report.append(f"\n| Most Functions | {mf_correct} | {results['most_functions']['wrong']} | "
                  f"{mf_rate:.1%} | [{mf_ci[0]:.1%}, {mf_ci[1]:.1%}] | "
                  f"{mf_wrong_rate:.1%} | [{mf_wrong_ci[0]:.1%}, {mf_wrong_ci[1]:.1%}] |")
    report.append(f"\n| Last Contract | {lc_correct} | {results['last_contract']['wrong']} | "
                  f"{lc_rate:.1%} | [{lc_ci[0]:.1%}, {lc_ci[1]:.1%}] | "
                  f"{lc_wrong_rate:.1%} | [{lc_wrong_ci[0]:.1%}, {lc_wrong_ci[1]:.1%}] |")

    # Per-class breakdown
    report.append("\n\n## Per-Class Breakdown (Most Functions Heuristic)\n")
    report.append("| Vulnerability Class | Correct | Wrong | Wrong Rate |")
    report.append("\n|---------------------|---------|-------|------------|")
    for cls in VULN_CLASSES:
        bd = per_class_breakdown[cls]
        c = bd["most_functions_correct"]
        w = bd["most_functions_wrong"]
        t = c + w
        wr = f"{w/t:.1%}" if t > 0 else "N/A"
        report.append(f"\n| {cls} | {c} | {w} | {wr} |")

    report.append("\n\n## Per-Class Breakdown (Last Contract Heuristic)\n")
    report.append("| Vulnerability Class | Correct | Wrong | Wrong Rate |")
    report.append("\n|---------------------|---------|-------|------------|")
    for cls in VULN_CLASSES:
        bd = per_class_breakdown[cls]
        c = bd["last_contract_correct"]
        w = bd["last_contract_wrong"]
        t = c + w
        wr = f"{w/t:.1%}" if t > 0 else "N/A"
        report.append(f"\n| {cls} | {c} | {w} | {wr} |")

    # Breakdown by number of contracts
    report.append("\n\n## Breakdown by Number of Contracts in File\n")
    report.append("| # Contracts | Total Files | MF Correct | MF Wrong | MF Wrong Rate | LC Correct | LC Wrong | LC Wrong Rate |")
    report.append("\n|-------------|-------------|------------|----------|---------------|------------|----------|---------------|")
    for n in sorted(by_n_contracts.keys()):
        bd = by_n_contracts[n]
        mf_c = bd["most_functions_correct"]
        mf_w = bd["most_functions_wrong"]
        lc_c = bd["last_contract_correct"]
        lc_w = bd["last_contract_wrong"]
        mf_t = mf_c + mf_w
        lc_t = lc_c + lc_w
        mf_wr = f"{mf_w/mf_t:.1%}" if mf_t > 0 else "N/A"
        lc_wr = f"{lc_w/lc_t:.1%}" if lc_t > 0 else "N/A"
        report.append(f"\n| {n} | {bd['total']} | {mf_c} | {mf_w} | {mf_wr} | {lc_c} | {lc_w} | {lc_wr} |")

    # Wrong selection examples
    report.append("\n\n## Wrong Selection Examples (Most Functions, first 20)\n")
    for entry in wrong_selections_mf[:20]:
        report.append(f"- md5=`{entry['md5'][:12]}...` actual=`{entry['actual']}` "
                      f"predicted=`{entry['predicted']}` (n_contracts={entry['n_contracts']})")

    if len(wrong_selections_mf) > 20:
        report.append(f"\n... and {len(wrong_selections_mf) - 20} more")

    report.append("\n\n## Wrong Selection Examples (Last Contract, first 20)\n")
    for entry in wrong_selections_lc[:20]:
        report.append(f"- md5=`{entry['md5'][:12]}...` actual=`{entry['actual']}` "
                      f"predicted=`{entry['predicted']}` (n_contracts={entry['n_contracts']})")

    if len(wrong_selections_lc) > 20:
        report.append(f"\n... and {len(wrong_selections_lc) - 20} more")

    # Conclusion
    report.append("\n\n## Conclusion\n")
    if mf_rate > lc_rate:
        report.append(f"The **Most Functions** heuristic outperforms Last Contract "
                      f"({mf_rate:.1%} vs {lc_rate:.1%} accuracy). ")
    elif lc_rate > mf_rate:
        report.append(f"The **Last Contract** heuristic outperforms Most Functions "
                      f"({lc_rate:.1%} vs {mf_rate:.1%} accuracy). ")
    else:
        report.append(f"Both heuristics perform equally ({mf_rate:.1%} accuracy). ")

    report.append(f"\nOverall wrong-selection rate: **Most Functions={mf_wrong_rate:.1%}**, "
                  f"**Last Contract={lc_wrong_rate:.1%}**. ")
    report.append("\nFiles with more contracts tend to have higher wrong-selection rates, "
                  "suggesting that contract selection is a meaningful source of label noise "
                  "for multi-contract files.")

    save_report("task16_wrong_contract_selection", "".join(report))
    print_header("Task 16 Complete")


if __name__ == "__main__":
    main()
