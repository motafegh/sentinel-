#!/usr/bin/env python3
"""
task19_timestamp_label_quality.py — Timestamp Label Quality Audit for SENTINEL v6

Load ALL Timestamp=1 contracts from CSV (~2191). For each:
1. Find .sol source via md5_to_path
2. Grep for: block.timestamp, block.number, now, block.difficulty, blockhash
3. Load graph .pt: check if uses_block_globals [2] = 1.0 on ANY node
4. Classify: (a) signal in source AND feature fires, (b) signal in source but
   feature doesn't fire, (c) no signal AND feature doesn't fire (mislabelled),
   (d) no signal but feature fires
5. For (b): check if due to wrong contract selection
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


# ── Timestamp signal patterns ─────────────────────────────────────────────────
_TIMESTAMP_PATTERNS = {
    "block.timestamp":  re.compile(r'\bblock\.timestamp\b'),
    "block.number":     re.compile(r'\bblock\.number\b'),
    "now":              re.compile(r'\bnow\b'),
    "block.difficulty": re.compile(r'\bblock\.difficulty\b'),
    "blockhash":        re.compile(r'\bblockhash\s*\('),
}


def has_timestamp_signal(source: str) -> tuple:
    """
    Check if source code contains timestamp-related signals.

    Returns (has_signal: bool, matched_patterns: list[str])
    """
    matched = []
    for name, pattern in _TIMESTAMP_PATTERNS.items():
        if pattern.search(source):
            matched.append(name)
    return (len(matched) > 0, matched)


def graph_uses_block_globals(g) -> bool:
    """
    Check if ANY node in the graph has uses_block_globals [2] = 1.0.
    """
    try:
        x = g.x
        if x is None or x.dim() != 2 or x.shape[1] < 3:
            return False
        ubg = x[:, 2]
        np_ubg = ubg.numpy() if hasattr(ubg, 'numpy') else np.array(ubg)
        return bool(np.any(np_ubg > 0.5))
    except Exception:
        return False


def find_block_globals_contract(source: str) -> str:
    """
    Find which contract in a multi-contract file contains block.* references.

    Returns the contract name or empty string if not found.
    """
    # Find all contract blocks
    contract_re = re.compile(r'\bcontract\s+(\w+)(?:\s+is\s+[\w,\s]+)?\s*\{')
    matches = list(contract_re.finditer(source))

    for m in matches:
        name = m.group(1)
        start = m.end()
        # Find closing brace with depth counting
        depth = 1
        pos = start
        while pos < len(source) and depth > 0:
            if source[pos] == '{':
                depth += 1
            elif source[pos] == '}':
                depth -= 1
            pos += 1

        block = source[start:pos]
        # Check for timestamp signals in this block
        for pattern in _TIMESTAMP_PATTERNS.values():
            if pattern.search(block):
                return name

    return ""


def main():
    print_header("Task 19: Timestamp Label Quality Audit")

    # ── Load labels ───────────────────────────────────────────────────────
    try:
        labels = load_label_csv()
    except Exception as e:
        print(f"ERROR: Could not load label CSV: {e}")
        return

    # ── Get all Timestamp=1 contracts ─────────────────────────────────────
    timestamp_stems = [
        stem for stem, lbls in labels.items()
        if lbls.get("Timestamp", 0) == 1
    ]
    print(f"Found {len(timestamp_stems)} Timestamp=1 contracts")

    if not timestamp_stems:
        print("ERROR: No Timestamp=1 contracts found.")
        return

    # ── Build md5→path mapping ────────────────────────────────────────────
    print("Building md5→path mapping...")
    md5_to_path = build_md5_to_path(set(timestamp_stems))
    print(f"Resolved {len(md5_to_path)}/{len(timestamp_stems)} paths")

    # ── Analyze each contract ─────────────────────────────────────────────
    categories = {
        "a_signal_and_feature": 0,      # signal in source AND feature fires
        "b_signal_no_feature": 0,       # signal in source but feature doesn't fire
        "c_no_signal_no_feature": 0,    # no signal AND no feature (mislabelled)
        "d_no_signal_feature": 0,       # no signal but feature fires
        "no_source": 0,
        "no_graph": 0,
        "load_error": 0,
    }

    pattern_counts = Counter()  # which patterns matched in source
    category_b_wrong_contract = 0  # (b) cases due to wrong contract selection
    category_b_other = 0           # (b) cases not due to wrong contract
    category_b_details = []

    no_signal_contracts = []  # for (c) investigation

    processed = 0
    for stem in timestamp_stems:
        processed += 1
        if processed % 200 == 0:
            print(f"  Processed {processed}/{len(timestamp_stems)} contracts...")

        sol_path = md5_to_path.get(stem)

        # Read source
        source = None
        source_has_signal = False
        matched_patterns = []

        if sol_path and sol_path.exists():
            try:
                source = sol_path.read_text(encoding="utf-8", errors="replace")
                source_has_signal, matched_patterns = has_timestamp_signal(source)
                for p in matched_patterns:
                    pattern_counts[p] += 1
            except OSError:
                pass
        else:
            categories["no_source"] += 1
            continue

        # Load graph
        graph_path = GRAPHS_DIR / f"{stem}.pt"
        if not graph_path.exists():
            categories["no_graph"] += 1
            continue

        try:
            g = load_graph(graph_path)
        except Exception as e:
            categories["load_error"] += 1
            continue

        feature_fires = graph_uses_block_globals(g)

        # Classify
        if source_has_signal and feature_fires:
            categories["a_signal_and_feature"] += 1
        elif source_has_signal and not feature_fires:
            categories["b_signal_no_feature"] += 1

            # Check if due to wrong contract selection
            contract_name = getattr(g, "contract_name", None)
            if source and contract_name:
                bg_contract = find_block_globals_contract(source)
                if bg_contract and bg_contract != contract_name:
                    category_b_wrong_contract += 1
                    category_b_details.append({
                        "stem": stem,
                        "graph_contract": contract_name,
                        "block_globals_contract": bg_contract,
                    })
                else:
                    category_b_other += 1
                    if len(category_b_details) < 15:
                        category_b_details.append({
                            "stem": stem,
                            "graph_contract": contract_name,
                            "block_globals_contract": bg_contract,
                            "reason": "unknown (contract matches or no multi-contract)"
                        })
        elif not source_has_signal and not feature_fires:
            categories["c_no_signal_no_feature"] += 1
            if len(no_signal_contracts) < 30:
                no_signal_contracts.append({
                    "stem": stem,
                    "sol_path": str(sol_path.name) if sol_path else "N/A",
                    "contract_name": getattr(g, "contract_name", "N/A"),
                })
        elif not source_has_signal and feature_fires:
            categories["d_no_signal_feature"] += 1

    # ── Compute totals and percentages ────────────────────────────────────
    analyzed = (categories["a_signal_and_feature"] + categories["b_signal_no_feature"] +
                categories["c_no_signal_no_feature"] + categories["d_no_signal_feature"])
    total_with_data = analyzed + categories["no_source"] + categories["no_graph"] + categories["load_error"]

    def pct(val):
        return f"{val/analyzed:.1%}" if analyzed > 0 else "N/A"

    # ── Build report ──────────────────────────────────────────────────────
    report = []
    report.append("# Task 19: Timestamp Label Quality Audit\n")
    report.append(f"**Total Timestamp=1 contracts:** {len(timestamp_stems)}  ")
    report.append(f"**Successfully analyzed:** {analyzed}  ")
    report.append(f"**No source file:** {categories['no_source']}  ")
    report.append(f"**No graph file:** {categories['no_graph']}  ")
    report.append(f"**Load errors:** {categories['load_error']}\n")

    # Category breakdown
    report.append("\n## Signal vs Feature Classification\n")
    report.append("| Category | Description | Count | Percentage |")
    report.append("\n|----------|-------------|-------|------------|")
    report.append(f"\n| (a) | Signal in source AND feature fires | {categories['a_signal_and_feature']} | {pct(categories['a_signal_and_feature'])} |")
    report.append(f"\n| (b) | Signal in source but feature doesn't fire | {categories['b_signal_no_feature']} | {pct(categories['b_signal_no_feature'])} |")
    report.append(f"\n| (c) | No signal AND feature doesn't fire (mislabelled?) | {categories['c_no_signal_no_feature']} | {pct(categories['c_no_signal_no_feature'])} |")
    report.append(f"\n| (d) | No signal but feature fires | {categories['d_no_signal_feature']} | {pct(categories['d_no_signal_feature'])} |")

    # Mislabel rate assessment
    mislabel_pct = categories["c_no_signal_no_feature"] / analyzed if analyzed > 0 else 0
    report.append(f"\n\n### Potential Mislabel Rate: **{mislabel_pct:.1%}**")
    if mislabel_pct > 0.30:
        report.append(" ⚠️ **EXCEEDS 30% THRESHOLD** — significant label quality concern!")
    elif mislabel_pct > 0.15:
        report.append(" ⚠️ Moderate label quality concern.")
    else:
        report.append(" ✅ Within acceptable range.")

    # Pattern distribution in source
    report.append("\n\n## Timestamp Signal Patterns Found in Source\n")
    report.append("| Pattern | Count |")
    report.append("\n|---------|-------|")
    for pattern_name in ["block.timestamp", "block.number", "now", "block.difficulty", "blockhash"]:
        count = pattern_counts.get(pattern_name, 0)
        report.append(f"\n| {pattern_name} | {count} |")

    # Category (b) breakdown
    report.append("\n\n## Category (b) Analysis: Signal in Source but Feature Doesn't Fire\n")
    report.append(f"**Total (b) cases:** {categories['b_signal_no_feature']}  ")
    report.append(f"\n**Due to wrong contract selection:** {category_b_wrong_contract}  ")
    report.append(f"\n**Other reasons:** {category_b_other}\n")

    if category_b_wrong_contract > 0:
        report.append("\n### Wrong Contract Selection Cases\n")
        for detail in category_b_details[:20]:
            if "reason" in detail:
                report.append(f"- stem=`{detail['stem'][:12]}...` graph_contract=`{detail['graph_contract']}` "
                              f"block_globals_contract=`{detail['block_globals_contract']}` "
                              f"reason={detail.get('reason', 'N/A')}")
            else:
                report.append(f"- stem=`{detail['stem'][:12]}...` graph_contract=`{detail['graph_contract']}` "
                              f"block_globals_contract=`{detail['block_globals_contract']}`")

    # Possible other reasons for (b)
    report.append("\n\n### Possible Reasons for (b) Failures (Beyond Wrong Contract)\n")
    report.append("1. **Slither IR omission**: `block.timestamp` reads via `SolidityVariableComposed` "
                  "may not appear in IR for some code paths")
    report.append("2. **Inline assembly**: `block.timestamp` accessed via assembly blocks "
                  "where Slither doesn't track IR operations")
    report.append("3. **Indirect access**: timestamp read through a library or inherited contract "
                  "that the feature computation doesn't trace into")
    report.append("4. **Compilation failure**: If Slither fell back to partial parsing, "
                  "IR may be incomplete")

    # Category (c) examples
    if no_signal_contracts:
        report.append("\n\n## Category (c) Examples: No Signal in Source (Potential Mislabels)\n")
        report.append(f"**Count:** {categories['c_no_signal_no_feature']}\n")
        for entry in no_signal_contracts[:20]:
            report.append(f"- stem=`{entry['stem'][:12]}...` file=`{entry['sol_path']}` "
                          f"contract=`{entry['contract_name']}`")
        if len(no_signal_contracts) > 20:
            report.append(f"\n... and {len(no_signal_contracts) - 20} more")

    # Category (d) analysis
    report.append("\n\n## Category (d) Analysis: No Source Signal but Feature Fires\n")
    report.append(f"**Count:** {categories['d_no_signal_feature']}\n")
    report.append("These are likely **true positives** where:")
    report.append("- The timestamp signal is in an inherited contract not in the main file")
    report.append("- The source regex missed an obfuscated/indirect reference")
    report.append("- Slither detected block global access through IR analysis (more thorough than grep)")

    # Summary
    report.append("\n\n## Summary\n")
    feature_recall = categories["a_signal_and_feature"] / (
        categories["a_signal_and_feature"] + categories["b_signal_no_feature"]
    ) if (categories["a_signal_and_feature"] + categories["b_signal_no_feature"]) > 0 else 0

    report.append(f"- **Feature recall** (among source-verified Timestamp): {feature_recall:.1%}")
    report.append(f"- **Potential mislabel rate** (category c): {mislabel_pct:.1%}")
    if mislabel_pct > 0.30:
        report.append(f"\n- ⚠️ **CRITICAL**: Over 30% of Timestamp=1 labels may be incorrect. "
                      "Consider re-labelling or excluding these contracts from training.")
    report.append(f"\n- Wrong contract selection accounts for {category_b_wrong_contract} "
                  f"of {categories['b_signal_no_feature']} feature misses "
                  f"({category_b_wrong_contract/categories['b_signal_no_feature']:.0%})"
                  if categories['b_signal_no_feature'] > 0 else
                  "\n- No feature miss cases to analyze")

    save_report("task19_timestamp_label_quality", "".join(report))
    print_header("Task 19 Complete")


if __name__ == "__main__":
    main()
