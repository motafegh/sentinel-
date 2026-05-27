#!/usr/bin/env python3
"""
task11_file_triple_alignment.py — File Triple Alignment Audit for SENTINEL v6

Count CSV rows, graph .pt files, token_windowed .pt files.
Compute: CSV∩graphs, CSV∩tokens, graphs∩tokens, CSV∩graphs∩tokens.
List first 20 stems in CSV but not graphs, and first 20 in graphs but not tokens.
Check if retokenization checkpoint exists and is completed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def main():
    print_header("Task 11: File Triple Alignment Audit")

    # ── Gather stem sets ───────────────────────────────────────────────────
    print("Loading CSV...")
    labels = load_label_csv()
    csv_stems = set(labels.keys())
    print(f"  CSV stems: {len(csv_stems)}")

    print("Scanning graph files...")
    graph_stems = {p.stem for p in GRAPHS_DIR.glob("*.pt")}
    print(f"  Graph stems: {len(graph_stems)}")

    print("Scanning token_windowed files...")
    token_stems = {p.stem for p in TOKENS_WINDOWED_DIR.glob("*.pt")}
    print(f"  Token_windowed stems: {len(token_stems)}")

    # Also check legacy tokens
    tokens_legacy_stems = set()
    if TOKENS_DIR.exists():
        tokens_legacy_stems = {p.stem for p in TOKENS_DIR.glob("*.pt")}
        print(f"  Token (legacy) stems: {len(tokens_legacy_stems)}")

    # ── Compute intersections ──────────────────────────────────────────────
    csv_graph = csv_stems & graph_stems
    csv_token = csv_stems & token_stems
    graph_token = graph_stems & token_stems
    triple = csv_stems & graph_stems & token_stems

    print(f"\n  CSV ∩ Graphs: {len(csv_graph)}")
    print(f"  CSV ∩ Tokens: {len(csv_token)}")
    print(f"  Graphs ∩ Tokens: {len(graph_token)}")
    print(f"  CSV ∩ Graphs ∩ Tokens: {len(triple)}")

    # ── Missing sets ───────────────────────────────────────────────────────
    csv_not_graphs = sorted(csv_stems - graph_stems)
    graphs_not_tokens = sorted(graph_stems - token_stems)
    csv_not_tokens = sorted(csv_stems - token_stems)
    graphs_not_csv = sorted(graph_stems - csv_stems)
    tokens_not_csv = sorted(token_stems - csv_stems)

    # ── Check retokenization checkpoint ────────────────────────────────────
    checkpoint_path = TOKENS_WINDOWED_DIR / "checkpoint.json"
    checkpoint_exists = checkpoint_path.exists()
    checkpoint_completed = False
    checkpoint_info = {}

    if checkpoint_exists:
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            checkpoint_completed = ckpt.get("completed", False)
            checkpoint_info = {
                "total": ckpt.get("total", "N/A"),
                "completed": checkpoint_completed,
                "timestamp": ckpt.get("timestamp", "N/A"),
            }
            print(f"\n  Checkpoint: exists, completed={checkpoint_completed}, total={checkpoint_info['total']}")
        except Exception as e:
            print(f"\n  Checkpoint: exists but unreadable: {e}")
            checkpoint_info = {"error": str(e)}
    else:
        print("\n  Checkpoint: not found")

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 11: File Triple Alignment Audit\n")

    # Summary table
    report_lines.append("## Set Sizes\n")
    report_lines.append("| Set | Count |\n")
    report_lines.append("|-----|-------|\n")
    report_lines.append(f"| CSV (multilabel_index_deduped.csv) | {len(csv_stems)} |\n")
    report_lines.append(f"| Graph .pt files | {len(graph_stems)} |\n")
    report_lines.append(f"| Token_windowed .pt files | {len(token_stems)} |\n")
    if tokens_legacy_stems:
        report_lines.append(f"| Token (legacy) .pt files | {len(tokens_legacy_stems)} |\n")

    report_lines.append("\n## Intersections\n")
    report_lines.append("| Intersection | Count |\n")
    report_lines.append("|--------------|-------|\n")
    report_lines.append(f"| CSV ∩ Graphs | {len(csv_graph)} |\n")
    report_lines.append(f"| CSV ∩ Tokens_windowed | {len(csv_token)} |\n")
    report_lines.append(f"| Graphs ∩ Tokens_windowed | {len(graph_token)} |\n")
    report_lines.append(f"| CSV ∩ Graphs ∩ Tokens_windowed | {len(triple)} |\n")

    # Coverage rates
    report_lines.append("\n## Coverage Rates\n")
    report_lines.append("| Metric | Rate |\n")
    report_lines.append("|--------|------|\n")
    if csv_stems:
        report_lines.append(f"| CSV→Graphs coverage | {len(csv_graph)/len(csv_stems):.1%} |\n")
        report_lines.append(f"| CSV→Tokens coverage | {len(csv_token)/len(csv_stems):.1%} |\n")
        report_lines.append(f"| CSV→Triple coverage | {len(triple)/len(csv_stems):.1%} |\n")
    if graph_stems:
        report_lines.append(f"| Graphs→Tokens coverage | {len(graph_token)/len(graph_stems):.1%} |\n")

    # Missing files
    report_lines.append("\n## Stems in CSV but not in Graphs (first 20)\n")
    if csv_not_graphs:
        for s in csv_not_graphs[:20]:
            report_lines.append(f"- `{s}`\n")
        if len(csv_not_graphs) > 20:
            report_lines.append(f"- ... and {len(csv_not_graphs) - 20} more (total: {len(csv_not_graphs)})\n")
    else:
        report_lines.append("None — all CSV stems have graph files.\n")

    report_lines.append("\n## Stems in Graphs but not in Tokens_windowed (first 20)\n")
    if graphs_not_tokens:
        for s in graphs_not_tokens[:20]:
            report_lines.append(f"- `{s}`\n")
        if len(graphs_not_tokens) > 20:
            report_lines.append(f"- ... and {len(graphs_not_tokens) - 20} more (total: {len(graphs_not_tokens)})\n")
    else:
        report_lines.append("None — all graph stems have token files.\n")

    report_lines.append("\n## Stems in Graphs but not in CSV (first 20)\n")
    if graphs_not_csv:
        for s in graphs_not_csv[:20]:
            report_lines.append(f"- `{s}`\n")
        if len(graphs_not_csv) > 20:
            report_lines.append(f"- ... and {len(graphs_not_csv) - 20} more (total: {len(graphs_not_csv)})\n")
    else:
        report_lines.append("None — all graph stems are in CSV.\n")

    # Checkpoint status
    report_lines.append("\n## Retokenization Checkpoint\n")
    report_lines.append(f"- **Path:** `{checkpoint_path}`\n")
    report_lines.append(f"- **Exists:** {checkpoint_exists}\n")
    if checkpoint_info:
        for k, v in checkpoint_info.items():
            report_lines.append(f"- **{k}:** {v}\n")
    if checkpoint_completed:
        report_lines.append("\n✅ Retokenization appears **completed**.\n")
    elif checkpoint_exists:
        report_lines.append("\n⚠️ Retokenization checkpoint exists but is **not completed**.\n")
    else:
        report_lines.append("\n❌ No retokenization checkpoint found.\n")

    report_content = "".join(report_lines)
    save_report("task11_file_triple_alignment", report_content)
    print_header("Task 11 Complete")


if __name__ == "__main__":
    main()
