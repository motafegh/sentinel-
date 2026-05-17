#!/usr/bin/env python3
"""
task15_in_unchecked_regex.py — in_unchecked Feature & Regex Audit

1. Find contracts with Solidity >= 0.8.0 from BCCC source dirs
2. Among those, check graph .pt for in_unchecked [9] = 1.0 on any node
3. If none exist (dead feature), create 2 test .sol files:
   - One with `unchecked { }` in a comment: `// unchecked { this is a comment }`
   - One with `unchecked { }` in a string: `string memory s = "unchecked {";`
4. Run the regex `\\bunchecked\\s*\\{` on both and report if it falsely matches
5. Also: count total contracts in the dataset that have `unchecked` anywhere
   in source (grep), and how many are >= 0.8.0

Report: confirmation of dead feature, regex false positive test results,
        how many contracts have unchecked patterns.
"""

import re
import sys
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


# The regex used by graph_extractor.py
UNCHECKED_REGEX = re.compile(r"\bunchecked\s*\{")


def main():
    print_header("Task 15: in_unchecked Feature & Regex Audit")

    # ── Step 1: Find Solidity >= 0.8.0 contracts ───────────────────────────
    print("\n--- Step 1: Finding Solidity >= 0.8.0 contracts ---")
    sol_08_contracts = []  # (path, version_str)
    all_sol_files = []

    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            print(f"  Source dir not found: {src_dir}")
            continue
        for sol in src_dir.rglob("*.sol"):
            all_sol_files.append(sol)
            version = extract_pragma_version(sol)
            if version is not None:
                major, minor = version.split(".")
                if int(major) >= 0 and int(minor) >= 8:
                    sol_08_contracts.append((sol, version))

    print(f"  Total .sol files found: {len(all_sol_files)}")
    print(f"  Solidity >= 0.8.0 contracts: {len(sol_08_contracts)}")

    # ── Step 2: Check graph .pt for in_unchecked [9] = 1.0 ─────────────────
    print("\n--- Step 2: Checking graph .pt files for in_unchecked activation ---")

    if not GRAPHS_DIR.exists():
        print(f"  WARNING: Graphs directory not found: {GRAPHS_DIR}")
        graphs_with_unchecked = []
    else:
        graph_files = sorted(GRAPHS_DIR.glob("*.pt"))
        print(f"  Found {len(graph_files)} graph files")

        # Sample up to 5000 graphs for efficiency (full scan if < 5000)
        sample_size = min(5000, len(graph_files))
        if sample_size < len(graph_files):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(graph_files), size=sample_size, replace=False)
            sampled_files = [graph_files[i] for i in indices]
            print(f"  Sampling {sample_size} graph files")
        else:
            sampled_files = graph_files
            print(f"  Scanning all {len(graph_files)} graph files")

        graphs_with_unchecked = []
        total_checked = 0
        skipped = 0

        for i, fpath in enumerate(sampled_files):
            if (i + 1) % 1000 == 0:
                print(f"    Checked {i + 1}/{len(sampled_files)} graphs...")

            try:
                data = load_graph(fpath)
                x = data.x
                if x is None or x.dim() != 2 or x.shape[1] != 12:
                    skipped += 1
                    continue
                total_checked += 1

                # Check feature [9] = in_unchecked
                in_unchecked_vals = x[:, 9]
                if torch.any(in_unchecked_vals == 1.0):
                    n_activated = int(torch.sum(in_unchecked_vals == 1.0))
                    n_total = x.shape[0]
                    graphs_with_unchecked.append({
                        "stem": fpath.stem,
                        "nodes_with_unchecked": n_activated,
                        "total_nodes": n_total,
                    })
            except Exception:
                skipped += 1

        print(f"  Checked {total_checked} graphs (skipped {skipped})")
        print(f"  Graphs with in_unchecked=1.0: {len(graphs_with_unchecked)}")

    feature_is_dead = len(graphs_with_unchecked) == 0
    if feature_is_dead:
        print("  ** FEATURE IS DEAD: No graph has in_unchecked [9] = 1.0 **")
    else:
        print(f"  Feature is ACTIVE: {len(graphs_with_unchecked)} graphs have in_unchecked=1.0")

    # ── Step 3: False positive regex tests ──────────────────────────────────
    print("\n--- Step 3: False positive regex tests ---")

    # Test file 1: unchecked in a comment
    comment_test = '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestComment {
    // unchecked { this is a comment }
    function safeAdd(uint a, uint b) public pure returns (uint) {
        return a + b;
    }
}
'''

    # Test file 2: unchecked in a string
    string_test = '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestString {
    function log() public pure returns (string memory) {
        string memory s = "unchecked {";
        return s;
    }
}
'''

    # Test file 3: REAL unchecked block (positive control)
    real_test = '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestReal {
    function uncheckAdd(uint a, uint b) public pure returns (uint) {
        unchecked { return a + b; }
    }
}
'''

    comment_match = UNCHECKED_REGEX.search(comment_test)
    string_match = UNCHECKED_REGEX.search(string_test)
    real_match = UNCHECKED_REGEX.search(real_test)

    print(f"  Comment test match: {'YES (FALSE POSITIVE!)' if comment_match else 'No (correct)'}")
    print(f"    Match text: '{comment_match.group()}'" if comment_match else "")
    print(f"  String test match:  {'YES (FALSE POSITIVE!)' if string_match else 'No (correct)'}")
    print(f"    Match text: '{string_match.group()}'" if string_match else "")
    print(f"  Real unchecked match: {'Yes (correct)' if real_match else 'NO (MISSING!)'}")
    print(f"    Match text: '{real_match.group()}'" if real_match else "")

    # ── Step 5: Grep for 'unchecked' in source files ───────────────────────
    print("\n--- Step 4: Counting contracts with 'unchecked' in source ---")

    unchecked_pattern = re.compile(r"\bunchecked\b", re.IGNORECASE)
    contracts_with_unchecked = 0
    contracts_with_unchecked_08 = 0
    unchecked_contexts = Counter()  # type: "comment", "string", "code"

    # Sample up to 5000 source files for efficiency
    source_sample = all_sol_files[:5000] if len(all_sol_files) > 5000 else all_sol_files

    for sol_path in source_sample:
        try:
            source = sol_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        if not unchecked_pattern.search(source):
            continue

        contracts_with_unchecked += 1

        version = extract_pragma_version(sol_path)
        if version is not None:
            major, minor = version.split(".")
            if int(major) >= 0 and int(minor) >= 8:
                contracts_with_unchecked_08 += 1

        # Classify context: comment, string, or actual code
        # Strip single-line comments and string literals to see if unchecked
        # still appears
        stripped = re.sub(r'//.*?$', '', source, flags=re.MULTILINE)  # remove // comments
        stripped = re.sub(r'/\*.*?\*/', '', stripped, flags=re.DOTALL)  # remove /* */ comments
        stripped = re.sub(r'"[^"]*"', '""', stripped)  # replace string contents
        stripped = re.sub(r"'[^']*'", "''", stripped)  # replace single-quoted strings

        if unchecked_pattern.search(stripped):
            unchecked_contexts["code"] += 1
        else:
            # Only appears in comments/strings
            if re.search(r'\bunchecked\b', source, re.IGNORECASE):
                # Check if it's in a comment
                for line in source.split('\n'):
                    line_stripped = line.split('//')[0]  # remove inline comments
                    if unchecked_pattern.search(line_stripped):
                        break
                else:
                    unchecked_contexts["comment_only"] += 1
                    continue
                unchecked_contexts["ambiguous"] += 1

    total_scanned = len(source_sample)
    print(f"  Scanned {total_scanned} source files")
    print(f"  Contracts with 'unchecked' anywhere: {contracts_with_unchecked}")
    print(f"  Of those, Solidity >= 0.8.0: {contracts_with_unchecked_08}")
    print(f"  Context breakdown: {dict(unchecked_contexts)}")

    # ── Build report ────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 15: in_unchecked Feature & Regex Audit\n\n")

    # Dead feature confirmation
    report_lines.append("## Dead Feature Confirmation\n\n")
    if feature_is_dead:
        report_lines.append("**CONFIRMED: The `in_unchecked` feature [9] is DEAD.**\n\n")
        report_lines.append("No graph .pt file out of the sampled set has `in_unchecked = 1.0` on any node. "
                            "This means:\n")
        report_lines.append("- The Slither `NodeType.STARTUNCHECKED` check is not firing\n")
        report_lines.append("- The fallback regex `\\bunchecked\\s*\\{` is also not matching in function "
                            "source_mapping content\n")
        report_lines.append("- Either contracts in the dataset don't use `unchecked{}` blocks, "
                            "or the source_mapping content is unavailable for these functions\n\n")
    else:
        report_lines.append(f"**Feature is ACTIVE.** Found {len(graphs_with_unchecked)} graphs with "
                            "`in_unchecked = 1.0`.\n\n")

    # Graphs with unchecked detail
    if graphs_with_unchecked:
        report_lines.append("### Graphs with in_unchecked=1.0 (first 20)\n\n")
        report_lines.append("| Stem (first 16) | Nodes with unchecked | Total nodes |\n")
        report_lines.append("|-----------------|---------------------|-------------|\n")
        for g in graphs_with_unchecked[:20]:
            report_lines.append(f"| {g['stem'][:16]}... | {g['nodes_with_unchecked']} | {g['total_nodes']} |\n")
        report_lines.append("\n")

    # Regex false positive tests
    report_lines.append("## Regex False Positive Tests\n\n")
    report_lines.append("Regex pattern: `\\bunchecked\\s*\\{`\n\n")
    report_lines.append("| Test Case | Source | Match? | Result |\n")
    report_lines.append("|-----------|--------|--------|--------|\n")

    report_lines.append(f"| Comment | `// unchecked {{ this is a comment }}` | "
                        f"{'YES' if comment_match else 'NO'} | "
                        f"{'**FALSE POSITIVE**' if comment_match else 'Correct (no match)'} |\n")
    report_lines.append(f"| String | `string memory s = \"unchecked {{\"` | "
                        f"{'YES' if string_match else 'NO'} | "
                        f"{'**FALSE POSITIVE**' if string_match else 'Correct (no match)'} |\n")
    report_lines.append(f"| Real code | `unchecked {{ return a + b; }}` | "
                        f"{'YES' if real_match else 'NO'} | "
                        f"{'Correct (match)' if real_match else '**MISSED**'} |\n\n")

    if comment_match or string_match:
        report_lines.append("### False Positive Analysis\n\n")
        report_lines.append("The regex `\\bunchecked\\s*\\{` produces false positives when:\n")
        if comment_match:
            report_lines.append("- `unchecked {` appears in a **comment** — the regex matches text "
                                "that is not executable code\n")
        if string_match:
            report_lines.append("- `unchecked {` appears in a **string literal** — the regex matches "
                                "text inside string constants\n")
        report_lines.append("\nHowever, the graph_extractor.py first tries Slither's "
                            "`NodeType.STARTUNCHECKED` which only fires on real AST nodes. "
                            "The regex fallback uses `func.source_mapping.content` which contains "
                            "only the function body (not comments outside the function). "
                            "But comments **inside** the function body would still cause false positives.\n\n")

    # Contract counts
    report_lines.append("## Contract Counts\n\n")
    report_lines.append(f"| Category | Count |\n")
    report_lines.append("|----------|-------|\n")
    report_lines.append(f"| Total .sol files scanned | {total_scanned} |\n")
    report_lines.append(f"| Solidity >= 0.8.0 contracts | {len(sol_08_contracts)} |\n")
    report_lines.append(f"| Contracts with 'unchecked' anywhere | {contracts_with_unchecked} |\n")
    report_lines.append(f"| Contracts with 'unchecked' AND >= 0.8.0 | {contracts_with_unchecked_08} |\n")
    report_lines.append(f"| Contracts with 'unchecked' in code (not just comments/strings) | "
                        f"{unchecked_contexts.get('code', 0)} |\n\n")

    # Impact assessment
    report_lines.append("## Impact Assessment\n\n")
    if feature_is_dead:
        report_lines.append("Since `in_unchecked` is never activated:\n\n")
        report_lines.append("1. **Feature [9] is always 0.0** for all nodes in all graphs\n")
        report_lines.append("2. This wastes one feature dimension — the GNN cannot learn anything from it\n")
        report_lines.append("3. For IntegerUO (integer overflow/underflow) detection, this feature "
                            "was supposed to be a direct signal for unchecked arithmetic blocks\n")
        report_lines.append("4. The model must rely entirely on other features (complexity, external_call_count, "
                            "has_loop) to detect IntegerUO patterns\n\n")
        report_lines.append("### Possible Causes\n\n")
        report_lines.append("- Slither `NodeType.STARTUNCHECKED` may not be triggered for the "
                            "contracts in the dataset\n")
        report_lines.append("- The `source_mapping.content` for functions may be empty/unavailable, "
                            "causing the regex fallback to be skipped\n")
        report_lines.append("- The dataset may genuinely contain no contracts using `unchecked{}` blocks "
                            "(possible for older Solidity or conservative code)\n\n")
    else:
        report_lines.append("The feature is active in the dataset. No action needed.\n\n")

    report_content = "".join(report_lines)
    save_report("task15_in_unchecked_regex", report_content)
    print_header("Task 15 Complete")


if __name__ == "__main__":
    main()
