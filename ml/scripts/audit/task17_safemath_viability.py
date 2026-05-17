#!/usr/bin/env python3
"""
task17_safemath_viability.py — SafeMath Viability Audit for SENTINEL v6

Sample 200 IntegerUO=1 and 200 NonVulnerable contracts. For each:
1. Read .sol source
2. Grep for SafeMath patterns
3. Categorize SafeMath usage
4. For subsample, load graph and check CALLS edges for SafeMath nodes
5. Compute confusion matrix: SafeMath presence vs IntegerUO label
6. Check <0.8.0 contracts IntegerUO rate (pragma_version feature viability)
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import random
import re
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


# ── SafeMath detection patterns ───────────────────────────────────────────────
_SAFEMATH_PATTERNS = {
    "using_safemath_uint256": re.compile(r'using\s+SafeMath\s+for\s+uint256'),
    "using_safemath_general": re.compile(r'using\s+SafeMath\s+for'),
    "safemath_direct_call":   re.compile(r'SafeMath\.(mul|add|sub|div)\s*\('),
    "is_safemath":            re.compile(r'\bis\s+SafeMath\b'),
    "safemath_any":           re.compile(r'\bSafeMath\b'),
}


def categorize_safemath(source: str) -> str:
    """
    Categorize SafeMath usage in a Solidity source file.

    Returns one of:
      - 'using_for_uint256': using SafeMath for uint256
      - 'direct_calls': SafeMath.mul(...), SafeMath.add(...), etc.
      - 'inheritance': is SafeMath
      - 'none': No SafeMath references
    """
    if _SAFEMATH_PATTERNS["using_safemath_uint256"].search(source):
        return "using_for_uint256"
    if _SAFEMATH_PATTERNS["using_safemath_general"].search(source):
        return "using_for_uint256"  # Treat all "using SafeMath for" the same
    if _SAFEMATH_PATTERNS["safemath_direct_call"].search(source):
        return "direct_calls"
    if _SAFEMATH_PATTERNS["is_safemath"].search(source):
        return "inheritance"
    return "none"


def check_graph_safemath(g) -> bool:
    """
    Check if a graph has CALLS edges targeting nodes with SafeMath-related names.

    Returns True if any CALLS edge targets a node whose name contains
    'SafeMath', 'mul', 'add', 'sub', 'div' (in a SafeMath context).
    """
    try:
        node_metadata = getattr(g, "node_metadata", None)
        edge_attr = g.edge_attr
        edge_index = g.edge_index

        if node_metadata is None or edge_attr is None or edge_index is None:
            return False

        ea_np = edge_attr.numpy() if hasattr(edge_attr, 'numpy') else np.array(edge_attr)
        ei_np = edge_index.numpy() if hasattr(edge_index, 'numpy') else np.array(edge_index)

        # CALLS edges are type 0
        calls_mask = ea_np == 0
        if not np.any(calls_mask):
            return False

        calls_targets = ei_np[1, calls_mask]
        safemath_keywords = {"safemath", ".mul", ".add", ".sub", ".div"}

        for tidx in calls_targets:
            tidx = int(tidx)
            if tidx < len(node_metadata):
                name = node_metadata[tidx].get("name", "").lower()
                for kw in safemath_keywords:
                    if kw in name:
                        return True
        return False
    except Exception:
        return False


def main():
    print_header("Task 17: SafeMath Viability Audit")

    # ── Load labels ───────────────────────────────────────────────────────
    try:
        labels = load_label_csv()
    except Exception as e:
        print(f"ERROR: Could not load label CSV: {e}")
        return

    print(f"Loaded labels for {len(labels)} contracts")

    # ── Identify IntegerUO=1 and NonVulnerable contracts ──────────────────
    iuo_stems = [stem for stem, lbls in labels.items() if lbls.get("IntegerUO", 0) == 1]
    nonvuln_stems = [
        stem for stem, lbls in labels.items()
        if all(lbls.get(cls, 0) == 0 for cls in VULN_CLASSES)
    ]

    print(f"IntegerUO=1 contracts: {len(iuo_stems)}")
    print(f"NonVulnerable contracts: {len(nonvuln_stems)}")

    # ── Sample ────────────────────────────────────────────────────────────
    random.seed(42)
    iuo_sample = random.sample(iuo_stems, min(200, len(iuo_stems)))
    nonvuln_sample = random.sample(nonvuln_stems, min(200, len(nonvuln_stems)))

    print(f"Sampling {len(iuo_sample)} IntegerUO and {len(nonvuln_sample)} NonVulnerable")

    # ── Build md5_to_path ─────────────────────────────────────────────────
    all_sampled = set(iuo_sample + nonvuln_sample)
    print("Building md5→path mapping...")
    md5_to_path = build_md5_to_path(all_sampled)
    print(f"Resolved {len(md5_to_path)}/{len(all_sampled)} paths")

    # ── Analyze SafeMath patterns ─────────────────────────────────────────
    results = {
        "IntegerUO": Counter(),   # category → count
        "NonVulnerable": Counter(),
    }
    safemath_details = []  # For subsample graph analysis
    pragma_versions = {"IntegerUO": Counter(), "NonVulnerable": Counter()}

    for label_group, stems in [("IntegerUO", iuo_sample), ("NonVulnerable", nonvuln_sample)]:
        for i, stem in enumerate(stems):
            if (i + 1) % 50 == 0:
                print(f"  Processing {label_group}: {i + 1}/{len(stems)}")

            sol_path = md5_to_path.get(stem)
            if sol_path is None or not sol_path.exists():
                continue

            try:
                source = sol_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            category = categorize_safemath(source)
            results[label_group][category] += 1

            # Extract pragma version
            pragma = extract_pragma_version(sol_path)
            if pragma:
                major, minor = pragma.split(".")[:2]
                pragma_versions[label_group][f"{major}.{minor}"] += 1
            else:
                pragma_versions[label_group]["no_pragma"] += 1

            # Collect for graph analysis subsample
            safemath_details.append({
                "stem": stem,
                "label_group": label_group,
                "category": category,
                "sol_path": sol_path,
            })

    # ── Confusion matrix ──────────────────────────────────────────────────
    # SafeMath present (any category != 'none') vs IntegerUO label
    safemath_present_iuo = sum(v for k, v in results["IntegerUO"].items() if k != "none")
    safemath_absent_iuo = results["IntegerUO"].get("none", 0)
    safemath_present_nv = sum(v for k, v in results["NonVulnerable"].items() if k != "none")
    safemath_absent_nv = results["NonVulnerable"].get("none", 0)

    # ── Graph-based SafeMath verification (subsample) ─────────────────────
    print("\nVerifying SafeMath in graphs (subsample)...")
    graph_verification = {
        "source_yes_graph_yes": 0,
        "source_yes_graph_no": 0,
        "source_no_graph_yes": 0,
        "source_no_graph_no": 0,
    }

    # Pick up to 20 from each category
    subsample = []
    for cat in ["using_for_uint256", "direct_calls", "inheritance", "none"]:
        cat_entries = [d for d in safemath_details if d["category"] == cat]
        subsample.extend(random.sample(cat_entries, min(20, len(cat_entries))))

    for i, entry in enumerate(subsample):
        if (i + 1) % 20 == 0:
            print(f"  Graph check: {i + 1}/{len(subsample)}")

        graph_path = GRAPHS_DIR / f"{entry['stem']}.pt"
        if not graph_path.exists():
            continue

        try:
            g = load_graph(graph_path)
        except Exception:
            continue

        graph_has_sm = check_graph_safemath(g)
        source_has_sm = entry["category"] != "none"

        if source_has_sm and graph_has_sm:
            graph_verification["source_yes_graph_yes"] += 1
        elif source_has_sm and not graph_has_sm:
            graph_verification["source_yes_graph_no"] += 1
        elif not source_has_sm and graph_has_sm:
            graph_verification["source_no_graph_yes"] += 1
        else:
            graph_verification["source_no_graph_no"] += 1

    # ── <0.8.0 pragma analysis ────────────────────────────────────────────
    pre_08_iuo = 0
    pre_08_total = 0
    for stem in iuo_sample:
        sol_path = md5_to_path.get(stem)
        if sol_path is None:
            continue
        pragma = extract_pragma_version(sol_path)
        if pragma:
            major, minor = pragma.split(".")[:2]
            if int(major) == 0 and int(minor) < 8:
                pre_08_iuo += 1
                pre_08_total += 1
            elif int(major) == 0 and int(minor) < 8:
                pre_08_total += 1

    # Count all <0.8.0 contracts from full label set
    all_stems = list(labels.keys())
    # Sample a reasonable number for pragma check
    pragma_sample = random.sample(all_stems, min(1000, len(all_stems)))
    pragma_md5_to_path = build_md5_to_path(set(pragma_sample))

    pre08_iuo_count = 0
    pre08_non_iuo_count = 0
    ge08_iuo_count = 0
    ge08_non_iuo_count = 0

    for stem in pragma_sample:
        sol_path = pragma_md5_to_path.get(stem)
        if sol_path is None:
            continue
        pragma = extract_pragma_version(sol_path)
        is_iuo = labels[stem].get("IntegerUO", 0) == 1
        if pragma:
            major, minor = pragma.split(".")[:2]
            if int(major) == 0 and int(minor) < 8:
                if is_iuo:
                    pre08_iuo_count += 1
                else:
                    pre08_non_iuo_count += 1
            else:
                if is_iuo:
                    ge08_iuo_count += 1
                else:
                    ge08_non_iuo_count += 1

    # ── Build report ──────────────────────────────────────────────────────
    report = []
    report.append("# Task 17: SafeMath Viability Audit\n")
    report.append(f"**IntegerUO=1 sampled:** {len(iuo_sample)}  ")
    report.append(f"**NonVulnerable sampled:** {len(nonvuln_sample)}  ")
    report.append(f"**Paths resolved:** {len(md5_to_path)}  ")
    report.append(f"**Graph subsample for verification:** {len(subsample)}\n")

    # SafeMath category distribution
    report.append("\n## SafeMath Category Distribution\n")
    report.append("| Category | IntegerUO | NonVulnerable |")
    report.append("\n|----------|-----------|---------------|")
    for cat in ["using_for_uint256", "direct_calls", "inheritance", "none"]:
        iuo_c = results["IntegerUO"].get(cat, 0)
        nv_c = results["NonVulnerable"].get(cat, 0)
        report.append(f"\n| {cat} | {iuo_c} | {nv_c} |")

    # Confusion matrix
    report.append("\n\n## Confusion Matrix: SafeMath Presence vs IntegerUO Label\n")
    report.append("| | IntegerUO=1 | NonVulnerable |")
    report.append("\n|---|------------|---------------|")
    report.append(f"\n| SafeMath present | {safemath_present_iuo} | {safemath_present_nv} |")
    report.append(f"\n| SafeMath absent | {safemath_absent_iuo} | {safemath_absent_nv} |")

    total_checked = safemath_present_iuo + safemath_absent_iuo + safemath_present_nv + safemath_absent_nv
    if total_checked > 0:
        # Detection rates
        # Among IntegerUO=1: how many have no SafeMath (potentially detectable by absence)?
        iuo_no_safemath_rate = safemath_absent_iuo / (safemath_present_iuo + safemath_absent_iuo) \
            if (safemath_present_iuo + safemath_absent_iuo) > 0 else 0
        # Among NonVulnerable: how many have SafeMath?
        nv_safemath_rate = safemath_present_nv / (safemath_present_nv + safemath_absent_nv) \
            if (safemath_present_nv + safemath_absent_nv) > 0 else 0

        report.append(f"\n\n**IntegerUO without SafeMath:** {iuo_no_safemath_rate:.1%} "
                      f"({safemath_absent_iuo}/{safemath_present_iuo + safemath_absent_iuo})")
        report.append(f"\n**NonVulnerable with SafeMath:** {nv_safemath_rate:.1%} "
                      f"({safemath_present_nv}/{safemath_present_nv + safemath_absent_nv})")

    # Graph verification
    report.append("\n\n## Graph-Based SafeMath Verification (Subsample)\n")
    report.append("| | Graph has SafeMath | Graph no SafeMath |")
    report.append("\n|---|--------------------|-------------------|")
    report.append(f"\n| Source has SafeMath | {graph_verification['source_yes_graph_yes']} | "
                  f"{graph_verification['source_yes_graph_no']} |")
    report.append(f"\n| Source no SafeMath | {graph_verification['source_no_graph_yes']} | "
                  f"{graph_verification['source_no_graph_no']} |")

    gv_total = sum(graph_verification.values())
    if gv_total > 0:
        source_yes = graph_verification["source_yes_graph_yes"] + graph_verification["source_yes_graph_no"]
        if source_yes > 0:
            graph_recall = graph_verification["source_yes_graph_yes"] / source_yes
            report.append(f"\n\n**Graph SafeMath recall** (source→graph): {graph_recall:.1%} "
                          f"({graph_verification['source_yes_graph_yes']}/{source_yes})")

    # Pragma version analysis
    report.append("\n\n## Pragma Version Analysis (Feature Viability)\n")
    report.append("| Version | IntegerUO=1 | Not IntegerUO | Iuo Rate |")
    report.append("\n|---------|-------------|---------------|----------|")
    pre08_total_all = pre08_iuo_count + pre08_non_iuo_count
    ge08_total_all = ge08_iuo_count + ge08_non_iuo_count
    pre08_rate = f"{pre08_iuo_count/pre08_total_all:.1%}" if pre08_total_all > 0 else "N/A"
    ge08_rate = f"{ge08_iuo_count/ge08_total_all:.1%}" if ge08_total_all > 0 else "N/A"
    report.append(f"\n| <0.8.0 | {pre08_iuo_count} | {pre08_non_iuo_count} | {pre08_rate} |")
    report.append(f"\n| >=0.8.0 | {ge08_iuo_count} | {ge08_non_iuo_count} | {ge08_rate} |")

    # Pragma distribution by label group
    report.append("\n\n## Pragma Version Distribution by Label Group\n")
    all_versions = sorted(set(list(pragma_versions["IntegerUO"].keys()) +
                              list(pragma_versions["NonVulnerable"].keys())))
    report.append("| Version | IntegerUO | NonVulnerable |")
    report.append("\n|---------|-----------|---------------|")
    for v in all_versions:
        iuo_c = pragma_versions["IntegerUO"].get(v, 0)
        nv_c = pragma_versions["NonVulnerable"].get(v, 0)
        report.append(f"\n| {v} | {iuo_c} | {nv_c} |")

    # Recommendation
    report.append("\n\n## Recommendation\n")
    if iuo_no_safemath_rate > 0.5:
        report.append(f"**SafeMath absence is a moderately strong signal** for IntegerUO "
                      f"({iuo_no_safemath_rate:.1%} of IntegerUO contracts lack SafeMath). ")
        report.append("However, it is NOT sufficient alone — many NonVulnerable contracts also lack SafeMath. ")
    else:
        report.append(f"SafeMath absence is a **weak signal** for IntegerUO "
                      f"(only {iuo_no_safemath_rate:.1%} of IntegerUO contracts lack SafeMath). ")

    if pre08_total_all > 0 and ge08_total_all > 0:
        if pre08_iuo_count / pre08_total_all > ge08_iuo_count / ge08_total_all * 2:
            report.append(f"\n\n**Pragma version <0.8.0 is a strong feature** for IntegerUO: "
                          f"{pre08_rate} vs {ge08_rate} for >=0.8.0. "
                          "This aligns with Solidity 0.8.0's built-in overflow checks.")
        else:
            report.append(f"\n\nPragma version <0.8.0 shows **moderate** IntegerUO correlation: "
                          f"{pre08_rate} vs {ge08_rate} for >=0.8.0.")

    report.append("\n\n**Consider adding `pragma_version` as a graph-level feature** "
                  "(currently not in the 12-dim node feature vector). "
                  "SafeMath presence could also be encoded as a contract-level feature, "
                  "but the graph already captures it via CALLS edges to SafeMath functions "
                  f"(graph recall: {graph_verification['source_yes_graph_yes']}/{source_yes if source_yes > 0 else '?'}).")

    save_report("task17_safemath_viability", "".join(report))
    print_header("Task 17 Complete")


if __name__ == "__main__":
    main()
