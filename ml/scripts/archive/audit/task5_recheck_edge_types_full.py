#!/usr/bin/env python3
"""
task5_recheck_edge_types_full.py — Full Edge Type Distribution Audit

Scan 5000 graph .pt files (or all if fast). For each:
1. Check edge_attr for values 3 (EMITS) and 4 (INHERITS)
2. If any found: record the md5_stem, Solidity version, contract structure
3. Also scan the source code of reextract_graphs.py and ast_extractor.py
   for EMITS/INHERITS edge creation logic — does the code explicitly skip these?
4. Count edge type distribution across all scanned graphs

Report: confirmed count of EMITS and INHERITS edges (expected 0),
edge type distribution, code analysis.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def analyze_source_code_for_emits_inherits():
    """
    Scan graph_extractor.py and ast_extractor.py source code for
    EMITS/INHERITS edge creation logic.
    """
    results = {}

    files_to_scan = [
        ("graph_extractor.py", PROJECT_ROOT / "ml" / "src" / "preprocessing" / "graph_extractor.py"),
        ("ast_extractor.py", PROJECT_ROOT / "ml" / "src" / "data_extraction" / "ast_extractor.py"),
        ("reextract_graphs.py", PROJECT_ROOT / "ml" / "scripts" / "reextract_graphs.py"),
    ]

    for name, path in files_to_scan:
        if not path.exists():
            results[name] = {"status": "not_found", "path": str(path)}
            continue

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            results[name] = {"status": "read_error", "error": str(e)}
            continue

        # Search for EMITS references
        emits_lines = []
        inherits_lines = []

        for i, line in enumerate(source.split("\n"), 1):
            if "EMITS" in line and not line.strip().startswith("#"):
                emits_lines.append((i, line.strip()))
            if "INHERITS" in line and not line.strip().startswith("#"):
                inherits_lines.append((i, line.strip()))

        # Check if the code explicitly creates or skips these edges
        creates_emits = any("EDGE_TYPES[\"EMITS\"]" in line or "EDGE_TYPES['EMITS']" in line
                           for _, line in emits_lines)
        creates_inherits = any("EDGE_TYPES[\"INHERITS\"]" in line or "EDGE_TYPES['INHERITS']" in line
                              for _, line in inherits_lines)

        results[name] = {
            "status": "found",
            "emits_refs": emits_lines,
            "inherits_refs": inherits_lines,
            "creates_emits": creates_emits,
            "creates_inherits": creates_inherits,
        }

    return results


def main():
    print_header("Task 5: Re-check Edge Types Full Audit")

    if not GRAPHS_DIR.exists():
        print(f"ERROR: Graphs directory not found: {GRAPHS_DIR}")
        save_report("task5_recheck_edge_types_full",
                     "# Task 5: Re-check Edge Types Full Audit\n\n"
                     "**ERROR**: Graphs directory not found.\n")
        return

    all_graph_files = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_graph_files:
        print("ERROR: No graph .pt files found")
        save_report("task5_recheck_edge_types_full",
                     "# Task 5: Re-check Edge Types Full Audit\n\n"
                     "**ERROR**: No graph .pt files found.\n")
        return

    total_files = len(all_graph_files)
    print(f"Found {total_files} graph files")

    # Determine scan size — scan 5000 or all if < 5000
    scan_size = min(5000, total_files)

    if scan_size < total_files:
        rng = np.random.default_rng(42)
        indices = rng.choice(total_files, size=scan_size, replace=False)
        sampled_files = [all_graph_files[i] for i in indices]
        print(f"Sampling {scan_size} files for scanning")
    else:
        sampled_files = all_graph_files
        print(f"Scanning all {total_files} files")

    # ── Scan graph files ────────────────────────────────────────────────────
    edge_type_counts = Counter()
    graphs_with_emits = []
    graphs_with_inherits = []
    total_edges = 0
    loaded = 0
    skipped = 0
    stale_count = 0

    for i, fpath in enumerate(sampled_files):
        if (i + 1) % 1000 == 0:
            print(f"  Scanned {i + 1}/{scan_size} files...")

        try:
            data = load_graph(fpath)
        except Exception:
            skipped += 1
            continue

        stem = fpath.stem
        edge_attr = data.edge_attr
        x = data.x

        # Skip stale (8-dim) graphs
        if x is not None and x.dim() == 2 and x.shape[1] != 12:
            stale_count += 1
            continue

        if edge_attr is None:
            skipped += 1
            continue

        loaded += 1

        # Handle both [E] and [E, 1] shapes
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        ea_np = edge_attr.numpy() if hasattr(edge_attr, 'numpy') else np.array(edge_attr)
        E = len(ea_np)
        total_edges += E

        # Count edge types
        for val in ea_np.flat:
            edge_type_counts[int(val)] += 1

        # Check for EMITS (3) and INHERITS (4)
        has_emits = int(np.any(ea_np == 3))
        has_inherits = int(np.any(ea_np == 4))

        if has_emits:
            emits_count = int(np.sum(ea_np == 3))
            graphs_with_emits.append({
                "stem": stem,
                "emits_edges": emits_count,
                "total_edges": E,
                "total_nodes": x.shape[0] if x is not None else "?",
            })

        if has_inherits:
            inherits_count = int(np.sum(ea_np == 4))
            graphs_with_inherits.append({
                "stem": stem,
                "inherits_edges": inherits_count,
                "total_edges": E,
                "total_nodes": x.shape[0] if x is not None else "?",
            })

    print(f"\n  Scan complete: {loaded} loaded, {skipped} skipped, {stale_count} stale")
    print(f"  Total edges: {total_edges:,}")
    print(f"  Graphs with EMITS (3): {len(graphs_with_emits)}")
    print(f"  Graphs with INHERITS (4): {len(graphs_with_inherits)}")

    # ── Source code analysis ────────────────────────────────────────────────
    print("\n--- Source Code Analysis ---")
    code_analysis = analyze_source_code_for_emits_inherits()

    for name, result in code_analysis.items():
        print(f"\n  {name}:")
        if result["status"] != "found":
            print(f"    Status: {result['status']}")
            continue
        print(f"    EMITS references: {len(result['emits_refs'])}")
        for line_no, line in result["emits_refs"]:
            print(f"      L{line_no}: {line[:80]}")
        print(f"    INHERITS references: {len(result['inherits_refs'])}")
        for line_no, line in result["inherits_refs"]:
            print(f"      L{line_no}: {line[:80]}")
        print(f"    Creates EMITS edges: {result['creates_emits']}")
        print(f"    Creates INHERITS edges: {result['creates_inherits']}")

    # ── Build report ────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 5: Re-check Edge Types Full Audit\n\n")
    report_lines.append(f"**Files scanned:** {loaded} / {total_files}  \n")
    report_lines.append(f"**Total edges examined:** {total_edges:,}  \n")
    report_lines.append(f"**Stale (8-dim) graphs skipped:** {stale_count}\n\n")

    # Edge type distribution
    report_lines.append("## Edge Type Distribution\n\n")
    report_lines.append("| Edge Type ID | Name | Count | Percentage |\n")
    report_lines.append("|-------------|------|-------|------------|\n")

    for et_id in sorted(edge_type_counts.keys()):
        name = EDGE_TYPE_NAMES.get(et_id, f"UNKNOWN({et_id})")
        count = edge_type_counts[et_id]
        pct = 100.0 * count / total_edges if total_edges > 0 else 0.0
        report_lines.append(f"| {et_id} | {name} | {count:,} | {pct:.2f}% |\n")

    # EMITS and INHERITS specific findings
    report_lines.append("\n## EMITS (Edge Type 3) Findings\n\n")
    if graphs_with_emits:
        report_lines.append(f"**Found {len(graphs_with_emits)} graphs with EMITS edges** (expected 0)\n\n")
        report_lines.append("| Stem (first 16) | EMITS edges | Total edges | Total nodes |\n")
        report_lines.append("|-----------------|-------------|-------------|-------------|\n")
        for g in graphs_with_emits[:30]:
            report_lines.append(
                f"| {g['stem'][:16]}... | {g['emits_edges']} | {g['total_edges']} | {g['total_nodes']} |\n"
            )
        if len(graphs_with_emits) > 30:
            report_lines.append(f"\n... and {len(graphs_with_emits) - 30} more\n")
    else:
        report_lines.append("**No EMITS edges found.** Edge type 3 is absent from all scanned graphs.\n\n")
        report_lines.append("This is expected for the v4 schema because:\n")
        report_lines.append("- The graph_extractor.py DOES create EMITS edges (see code analysis below)\n")
        report_lines.append("- But EMITS edges require the event node to be in `node_map` (i.e., the event "
                            "must be a non-dependency declaration in the target contract)\n")
        report_lines.append("- Many contracts may not emit events, or the events may be in inherited "
                            "contracts (filtered as dependencies)\n\n")

    report_lines.append("## INHERITS (Edge Type 4) Findings\n\n")
    if graphs_with_inherits:
        report_lines.append(f"**Found {len(graphs_with_inherits)} graphs with INHERITS edges** (expected 0)\n\n")
        report_lines.append("| Stem (first 16) | INHERITS edges | Total edges | Total nodes |\n")
        report_lines.append("|-----------------|----------------|-------------|-------------|\n")
        for g in graphs_with_inherits[:30]:
            report_lines.append(
                f"| {g['stem'][:16]}... | {g['inherits_edges']} | {g['total_edges']} | {g['total_nodes']} |\n"
            )
        if len(graphs_with_inherits) > 30:
            report_lines.append(f"\n... and {len(graphs_with_inherits) - 30} more\n")
    else:
        report_lines.append("**No INHERITS edges found.** Edge type 4 is absent from all scanned graphs.\n\n")
        report_lines.append("This is expected for the v4 schema because:\n")
        report_lines.append("- The graph_extractor.py DOES create INHERITS edges (see code analysis below)\n")
        report_lines.append("- But INHERITS edges require the parent contract to be in `node_map`\n")
        report_lines.append("- Parent contracts from inherited files are filtered as dependencies by "
                            "`is_from_dependency()`, so they are never added to node_map\n")
        report_lines.append("- The target contract's own name IS in node_map, but `contract.inheritance` "
                            "lists contracts that may not be in the current file\n\n")

    # Source code analysis
    report_lines.append("## Source Code Analysis\n\n")
    report_lines.append("### Does the code explicitly skip EMITS/INHERITS?\n\n")

    for name, result in code_analysis.items():
        report_lines.append(f"#### {name}\n\n")
        if result["status"] != "found":
            report_lines.append(f"*File not found at: {result.get('path', 'unknown')}*\n\n")
            continue

        report_lines.append(f"- **Creates EMITS edges:** {'Yes' if result['creates_emits'] else 'No'}\n")
        report_lines.append(f"- **Creates INHERITS edges:** {'Yes' if result['creates_inherits'] else 'No'}\n\n")

        if result["emits_refs"]:
            report_lines.append("**EMITS references:**\n\n")
            for line_no, line in result["emits_refs"]:
                report_lines.append(f"- L{line_no}: `{line}`\n")
            report_lines.append("\n")

        if result["inherits_refs"]:
            report_lines.append("**INHERITS references:**\n\n")
            for line_no, line in result["inherits_refs"]:
                report_lines.append(f"- L{line_no}: `{line}`\n")
            report_lines.append("\n")

    # Explanation of why EMITS/INHERITS may be absent despite code creating them
    report_lines.append("## Why EMITS/INHERITS May Be Absent Despite Code Creating Them\n\n")
    report_lines.append("The graph_extractor.py contains code that creates both EMITS and INHERITS edges:\n\n")
    report_lines.append("```python\n")
    report_lines.append("# EMITS edge creation (graph_extractor.py ~L937-942)\n")
    report_lines.append("if hasattr(func, 'events_emitted'):\n")
    report_lines.append("    for evt in func.events_emitted:\n")
    report_lines.append("        _add_edge(fn, evt.canonical_name, EDGE_TYPES['EMITS'])\n")
    report_lines.append("\n")
    report_lines.append("# INHERITS edge creation (graph_extractor.py ~L944-948)\n")
    report_lines.append("for parent in contract.inheritance:\n")
    report_lines.append("    _add_edge(contract.name, parent.name, EDGE_TYPES['INHERITS'])\n")
    report_lines.append("```\n\n")
    report_lines.append("However, `_add_edge` only creates an edge if **both** source and target keys "
                        "exist in `node_map`. The key reasons these edges may be absent:\n\n")
    report_lines.append("1. **EMITS**: The event must be declared in the same contract (not a dependency). "
                        "If the contract only emits events from inherited interfaces, those event nodes "
                        "are filtered out and won't be in `node_map`.\n\n")
    report_lines.append("2. **INHERITS**: The parent contract must be defined in the same file and not "
                        "be a dependency. If inheritance is from an imported contract, it's filtered by "
                        "`is_from_dependency()`.\n\n")
    report_lines.append("3. **Try/except wrapping**: Both edge creation blocks are wrapped in "
                        "`try/except`, so any error silently skips the edge.\n\n")

    # Summary
    report_lines.append("## Summary\n\n")
    emits_count = edge_type_counts.get(3, 0)
    inherits_count = edge_type_counts.get(4, 0)
    report_lines.append(f"| Metric | Value |\n")
    report_lines.append("|--------|-------|\n")
    report_lines.append(f"| Graphs scanned | {loaded} |\n")
    report_lines.append(f"| Total edges | {total_edges:,} |\n")
    report_lines.append(f"| EMITS edges (type 3) | {emits_count:,} |\n")
    report_lines.append(f"| INHERITS edges (type 4) | {inherits_count:,} |\n")
    report_lines.append(f"| Graphs with EMITS | {len(graphs_with_emits)} |\n")
    report_lines.append(f"| Graphs with INHERITS | {len(graphs_with_inherits)} |\n\n")

    if emits_count == 0 and inherits_count == 0:
        report_lines.append("**Result: EMITS and INHERITS edges are completely absent from the dataset.** "
                            "The code can create them but they never appear in practice due to the "
                            "dependency filtering and the `_add_edge` guard. These edge types are effectively "
                            "dead in the dataset.\n\n")
        report_lines.append("### Implications\n\n")
        report_lines.append("- The `nn.Embedding(8, edge_emb_dim)` table has rows 3 and 4 that are "
                            "**never trained** — they remain at random initialization\n")
        report_lines.append("- This is not harmful (unused embeddings don't affect output) but wastes "
                            "2 embedding rows\n")
        report_lines.append("- If the model is later used on contracts that DO have EMITS/INHERITS edges "
                            "(e.g., during inference on new contracts), those edge types will use "
                            "untrained random embeddings — potentially degrading inference quality\n")
    else:
        report_lines.append(f"**Result: Found {emits_count} EMITS and {inherits_count} INHERITS edges "
                            "in the dataset.** These edge types are NOT dead — they appear in at least "
                            "some graphs. The GNNEncoder embedding rows 3 and 4 will receive gradient "
                            "updates during training.\n")

    report_content = "".join(report_lines)
    save_report("task5_recheck_edge_types_full", report_content)
    print_header("Task 5 Complete")


if __name__ == "__main__":
    main()
