#!/usr/bin/env python3
"""
task23_send_unchecked_prevalence.py — .send() Unchecked Prevalence Audit

Sample 500 MishandledException=1 contracts. For each:
1. Grep .sol for `.send(` patterns
2. For each .send() call, classify: checked (if/require/bool assignment) or
   unchecked (bare statement)
3. Load graph .pt: check return_ignored [7] on function nodes
4. Cross-tabulate: .send() unchecked in source vs return_ignored in graph
Also check DoS=1 contracts for .send() in loops.
"""

import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *

# ── Regex patterns for .send() detection ───────────────────────────────────
_SEND_RE = re.compile(r'\.send\s*\(')
_CALL_RE = re.compile(r'\.call\s*[\({]')  # .call( or .call{

# Patterns to detect .send() result being CHECKED.
# Strategy: count ALL .send() occurrences, then identify which ones are
# wrapped in a checking construct.  The remainder are "unchecked".
# We use broader patterns with re.DOTALL so that multi-line wrapping
# (e.g. if-condition spanning several lines) is still detected.
_CHECKED_PATTERNS = [
    # if / require / assert wrapping .send() — allow any chars between
    re.compile(r'\b(if|require|assert)\s*\(.*?\.send\s*\(', re.DOTALL),
    # bool variable assignment: bool result = addr.send(...)
    re.compile(r'\bbool\s+\w+\s*=\s*.*?\.send\s*\(', re.DOTALL),
    # any variable assignment of .send() result: result = addr.send(...)
    re.compile(r'\w+\s*=\s*!?\s*\w+\.send\s*\('),
    # negation of .send(): !addr.send(...)
    re.compile(r'!\s*\w+\.send\s*\('),
    # ternary with .send(): addr.send(...) ? ... : ...
    re.compile(r'\.send\s*\([^)]*\)\s*\?'),
    # .send() in boolean expression: condition && addr.send(...)
    re.compile(r'(&&|\|\|)\s*.*?\.send\s*\(', re.DOTALL),
]

# Pattern for .send() inside loops
_LOOP_RE = re.compile(r'(for\s*\(|while\s*\()')


def classify_send_calls(sol_text: str) -> dict:
    """
    Analyze a .sol file for .send() call patterns.
    Returns dict with counts of checked/unchecked .send() calls.
    """
    lines = sol_text.split('\n')
    send_positions = []

    for line_num, line in enumerate(lines):
        if _SEND_RE.search(line):
            send_positions.append(line_num)

    if not send_positions:
        return {"total_send_calls": 0, "checked": 0, "unchecked": 0, "in_loop": 0}

    checked = 0
    unchecked = 0
    in_loop = 0

    for line_num in send_positions:
        line = lines[line_num]
        # Check surrounding context (5 lines before and 3 after — Solidity
        # if/require conditions often span multiple lines)
        start = max(0, line_num - 5)
        end = min(len(lines), line_num + 4)
        context = '\n'.join(lines[start:end])

        is_checked = False
        for pat in _CHECKED_PATTERNS:
            if pat.search(context):
                is_checked = True
                break

        if is_checked:
            checked += 1
        else:
            unchecked += 1

        # Check if in loop (wider context)
        wider_start = max(0, line_num - 10)
        wider_end = min(len(lines), line_num + 1)
        wider_context = '\n'.join(lines[wider_start:wider_end])
        if _LOOP_RE.search(wider_context):
            in_loop += 1

    return {
        "total_send_calls": len(send_positions),
        "checked": checked,
        "unchecked": unchecked,
        "in_loop": in_loop,
    }


def classify_call_calls(sol_text: str) -> dict:
    """Analyze a .sol file for .call() call patterns for comparison."""
    call_count = len(_CALL_RE.findall(sol_text))
    return {"total_call_calls": call_count}


def check_return_ignored_in_graph(graph_path: Path) -> dict:
    """
    Load graph .pt and check return_ignored [7] on function nodes.
    Returns stats about return_ignored values.
    """
    try:
        data = load_graph(graph_path)
    except Exception as e:
        return {"error": str(e)}

    x = data.x
    if x is None or x.dim() != 2 or x.shape[1] < 8:
        return {"error": "invalid feature matrix"}

    x_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)
    type_ids = x_np[:, 0]

    # Function nodes: type_id ≈ 1/12
    func_mask = np.abs(type_ids - (1.0 / 12.0)) < 0.01
    # Also CFG_NODE_CALL: type_id ≈ 8/12
    call_node_mask = np.abs(type_ids - (8.0 / 12.0)) < 0.01

    relevant_mask = func_mask | call_node_mask
    relevant_nodes = x_np[relevant_mask]

    if len(relevant_nodes) == 0:
        return {
            "num_func_nodes": int(func_mask.sum()),
            "num_call_nodes": int(call_node_mask.sum()),
            "return_ignored_1_count": 0,
            "return_ignored_any": 0,
        }

    return_ignored_vals = relevant_nodes[:, 7]
    return_ignored_1 = int(np.sum(return_ignored_vals > 0.5))
    return_ignored_any = int(np.sum(np.abs(return_ignored_vals) > 0.01))

    return {
        "num_func_nodes": int(func_mask.sum()),
        "num_call_nodes": int(call_node_mask.sum()),
        "return_ignored_1_count": return_ignored_1,
        "return_ignored_any": return_ignored_any,
    }


def main():
    print_header("Task 23: .send() Unchecked Prevalence Audit")

    # ── Load labels ────────────────────────────────────────────────────────
    print("  Loading labels...")
    labels = load_label_csv()
    print(f"  Total labeled contracts: {len(labels)}")

    # ── Sample MishandledException=1 contracts ─────────────────────────────
    me_stems = get_stems_with_label("MishandledException", label_value=1, pure=False)
    print(f"  MishandledException=1 contracts: {len(me_stems)}")

    sample_size_me = min(500, len(me_stems))
    rng = np.random.default_rng(42)
    me_sample = rng.choice(me_stems, size=sample_size_me, replace=False).tolist() \
        if len(me_stems) > sample_size_me else me_stems
    print(f"  Sampling {len(me_sample)} MishandledException contracts")

    # ── Sample DoS=1 contracts ─────────────────────────────────────────────
    dos_stems = get_stems_with_label("DenialOfService", label_value=1, pure=False)
    print(f"  DenialOfService=1 contracts: {len(dos_stems)}")

    sample_size_dos = min(200, len(dos_stems))
    dos_sample = rng.choice(dos_stems, size=sample_size_dos, replace=False).tolist() \
        if len(dos_stems) > sample_size_dos else dos_stems
    print(f"  Sampling {len(dos_sample)} DenialOfService contracts")

    # ── Build md5_to_path ──────────────────────────────────────────────────
    all_target_stems = set(me_sample) | set(dos_sample)
    print(f"  Building md5_to_path mapping for {len(all_target_stems)} stems...")
    md5_to_path = build_md5_to_path(all_target_stems)
    print(f"  Resolved {len(md5_to_path)} stems to .sol files")

    # ── Analyze MishandledException contracts ──────────────────────────────
    print("\n  Analyzing MishandledException contracts...")
    me_results = []
    me_send_count = 0
    me_send_unchecked_count = 0
    me_no_sol = 0
    me_no_graph = 0

    for i, stem in enumerate(me_sample):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(me_sample)} ME contracts...")

        sol_path = find_sol_for_stem(stem, md5_to_path)
        graph_path = GRAPHS_DIR / f"{stem}.pt"

        result = {"stem": stem, "has_sol": sol_path is not None, "has_graph": graph_path.exists()}

        # Source analysis
        if sol_path is not None and sol_path.exists():
            try:
                sol_text = sol_path.read_text(encoding="utf-8", errors="replace")
                send_info = classify_send_calls(sol_text)
                call_info = classify_call_calls(sol_text)
                result["send"] = send_info
                result["call"] = call_info
                if send_info["total_send_calls"] > 0:
                    me_send_count += 1
                    if send_info["unchecked"] > 0:
                        me_send_unchecked_count += 1
            except Exception as e:
                result["send_error"] = str(e)
                me_no_sol += 1
        else:
            me_no_sol += 1

        # Graph analysis
        if graph_path.exists():
            try:
                graph_info = check_return_ignored_in_graph(graph_path)
                result["graph"] = graph_info
            except Exception as e:
                result["graph_error"] = str(e)
                me_no_graph += 1
        else:
            me_no_graph += 1

        me_results.append(result)

    # ── Analyze DoS contracts ──────────────────────────────────────────────
    print("\n  Analyzing DenialOfService contracts...")
    dos_results = []
    dos_send_in_loop = 0
    dos_send_total = 0

    for i, stem in enumerate(dos_sample):
        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(dos_sample)} DoS contracts...")

        sol_path = find_sol_for_stem(stem, md5_to_path)
        result = {"stem": stem, "has_sol": sol_path is not None}

        if sol_path is not None and sol_path.exists():
            try:
                sol_text = sol_path.read_text(encoding="utf-8", errors="replace")
                send_info = classify_send_calls(sol_text)
                result["send"] = send_info
                if send_info["total_send_calls"] > 0:
                    dos_send_total += 1
                    if send_info["in_loop"] > 0:
                        dos_send_in_loop += 1
            except Exception as e:
                result["send_error"] = str(e)

        dos_results.append(result)

    # ── Cross-tabulation: .send() unchecked vs return_ignored ──────────────
    print("  Building cross-tabulation...")
    cross_tab = {
        "send_unchecked_and_return_ignored": 0,
        "send_unchecked_and_return_ok": 0,
        "send_checked_and_return_ignored": 0,
        "send_checked_and_return_ok": 0,
        "send_none_return_ignored": 0,
        "send_none_return_ok": 0,
        "incomplete": 0,
    }

    for r in me_results:
        send_info = r.get("send", {})
        graph_info = r.get("graph", {})

        if "error" in graph_info or "send_error" in r:
            cross_tab["incomplete"] += 1
            continue

        has_send = send_info.get("total_send_calls", 0) > 0
        has_unchecked = send_info.get("unchecked", 0) > 0
        has_ignored = graph_info.get("return_ignored_1_count", 0) > 0

        if not has_send:
            if has_ignored:
                cross_tab["send_none_return_ignored"] += 1
            else:
                cross_tab["send_none_return_ok"] += 1
        elif has_unchecked:
            if has_ignored:
                cross_tab["send_unchecked_and_return_ignored"] += 1
            else:
                cross_tab["send_unchecked_and_return_ok"] += 1
        else:
            # send checked
            if has_ignored:
                cross_tab["send_checked_and_return_ignored"] += 1
            else:
                cross_tab["send_checked_and_return_ok"] += 1

    # ── .call() detection rate comparison ───────────────────────────────────
    me_call_count = 0
    me_call_with_send = 0
    for r in me_results:
        call_info = r.get("call", {})
        send_info = r.get("send", {})
        if call_info.get("total_call_calls", 0) > 0:
            me_call_count += 1
            if send_info.get("total_send_calls", 0) > 0:
                me_call_with_send += 1

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 23: .send() Unchecked Prevalence Audit\n\n")
    report_lines.append(f"**MishandledException sample:** {len(me_sample)}  \n")
    report_lines.append(f"**DenialOfService sample:** {len(dos_sample)}  \n")
    report_lines.append(f"**.sol files resolved:** {len(md5_to_path)}/{len(all_target_stems)}\n\n")

    # ME .send() prevalence
    report_lines.append("## 1. .send() Prevalence in MishandledException Contracts\n\n")
    report_lines.append(f"| Metric | Count | Rate |\n")
    report_lines.append(f"|--------|-------|------|\n")
    me_with_send_rate = me_send_count / max(len(me_sample), 1)
    report_lines.append(f"| Contracts with .send() | {me_send_count} | {me_with_send_rate:.1%} |\n")
    me_unchecked_rate = me_send_unchecked_count / max(me_send_count, 1)
    report_lines.append(f"| Contracts with unchecked .send() | {me_send_unchecked_count} | {me_unchecked_rate:.1%} |\n")
    report_lines.append(f"| Contracts with no .sol file | {me_no_sol} | — |\n")
    report_lines.append(f"| Contracts with no graph file | {me_no_graph} | — |\n\n")

    # Aggregate .send() stats
    total_send_calls = sum(r.get("send", {}).get("total_send_calls", 0) for r in me_results)
    total_checked = sum(r.get("send", {}).get("checked", 0) for r in me_results)
    total_unchecked = sum(r.get("send", {}).get("unchecked", 0) for r in me_results)
    report_lines.append("### Aggregate .send() Statistics\n\n")
    report_lines.append(f"| Metric | Count |\n")
    report_lines.append(f"|--------|-------|\n")
    report_lines.append(f"| Total .send() calls found | {total_send_calls} |\n")
    report_lines.append(f"| Checked .send() calls | {total_checked} |\n")
    report_lines.append(f"| Unchecked .send() calls | {total_unchecked} |\n")
    if total_send_calls > 0:
        report_lines.append(f"| Unchecked rate | {total_unchecked/total_send_calls:.1%} |\n")
    report_lines.append("\n")

    # Cross-tabulation
    report_lines.append("## 2. Cross-Tabulation: .send() Unchecked vs return_ignored\n\n")
    report_lines.append("| Source \\ Graph | return_ignored=1 | return_ignored=0 | Total |\n")
    report_lines.append("|----------------|-------------------|-------------------|-------|\n")
    ct = cross_tab
    unchecked_total = ct["send_unchecked_and_return_ignored"] + ct["send_unchecked_and_return_ok"]
    checked_total = ct["send_checked_and_return_ignored"] + ct["send_checked_and_return_ok"]
    none_total = ct["send_none_return_ignored"] + ct["send_none_return_ok"]

    report_lines.append(
        f"| .send() unchecked | {ct['send_unchecked_and_return_ignored']} | "
        f"{ct['send_unchecked_and_return_ok']} | {unchecked_total} |\n"
    )
    report_lines.append(
        f"| .send() checked | {ct['send_checked_and_return_ignored']} | "
        f"{ct['send_checked_and_return_ok']} | {checked_total} |\n"
    )
    report_lines.append(
        f"| No .send() | {ct['send_none_return_ignored']} | "
        f"{ct['send_none_return_ok']} | {none_total} |\n"
    )
    report_lines.append(f"| Incomplete data | — | — | {ct['incomplete']} |\n\n")

    # Agreement rate
    agreement = ct["send_unchecked_and_return_ignored"] + ct["send_checked_and_return_ok"] + ct["send_none_return_ok"]
    total_crosstab = sum(v for k, v in ct.items() if k != "incomplete")
    agreement_rate = agreement / max(total_crosstab, 1)
    report_lines.append(f"**Agreement rate** (source and graph agree): {agreement_rate:.1%}\n\n")

    # Missed .send() analysis
    missed = ct["send_unchecked_and_return_ok"]  # source has unchecked .send() but graph doesn't flag
    report_lines.append(f"**Missed by graph:** {missed} contracts with unchecked .send() in source ")
    report_lines.append(f"but no return_ignored in graph\n\n")

    # DoS .send() in loops
    report_lines.append("## 3. .send() in Loops (DenialOfService Contracts)\n\n")
    report_lines.append(f"| Metric | Count | Rate |\n")
    report_lines.append(f"|--------|-------|------|\n")
    dos_send_rate = dos_send_total / max(len(dos_sample), 1)
    report_lines.append(f"| DoS contracts with .send() | {dos_send_total} | {dos_send_rate:.1%} |\n")
    dos_loop_rate = dos_send_in_loop / max(dos_send_total, 1)
    report_lines.append(f"| .send() in loops | {dos_send_in_loop} | {dos_loop_rate:.1%} |\n\n")

    # .call() detection comparison
    report_lines.append("## 4. .call() vs .send() Detection Comparison\n\n")
    report_lines.append(f"| Metric | Count | Rate |\n")
    report_lines.append(f"|--------|-------|------|\n")
    me_call_rate = me_call_count / max(len(me_sample), 1)
    report_lines.append(f"| ME contracts with .call() | {me_call_count} | {me_call_rate:.1%} |\n")
    report_lines.append(f"| ME contracts with both .call() and .send() | {me_call_with_send} | ")
    report_lines.append(f"{me_call_with_send/max(me_call_count,1):.1%} of .call() contracts |\n\n")

    # Conclusions
    report_lines.append("## 5. Conclusions & Recommendations\n\n")
    if me_send_count > 0:
        if me_unchecked_rate > 0.5:
            report_lines.append("- ⚠️ **High unchecked .send() rate** in MishandledException contracts. ")
            report_lines.append("The graph `return_ignored` feature should capture this but may be missing cases.\n")
        else:
            report_lines.append("- Most .send() calls in ME contracts are checked, suggesting the vulnerability ")
            report_lines.append("comes from other patterns (e.g., .call(), .transfer()).\n")
    else:
        report_lines.append("- Very few MishandledException contracts contain .send() calls. ")
        report_lines.append("The class likely captures broader exception mishandling patterns.\n")

    if missed > 0:
        report_lines.append(f"- **{missed} contracts** have unchecked .send() in source but no return_ignored ")
        report_lines.append("in the graph. Consider improving the AST extractor to catch these.\n")

    if dos_send_in_loop > 0:
        report_lines.append(f"- **{dos_send_in_loop} DoS contracts** have .send() inside loops — a classic DoS pattern. ")
        report_lines.append("Verify the graph captures loop + external_call correctly.\n")

    if me_call_count > me_send_count * 2:
        report_lines.append("- .call() is far more common than .send() in ME contracts. ")
        report_lines.append("The audit focus should extend beyond .send() to .call() and .transfer() patterns.\n")

    report_content = "".join(report_lines)
    save_report("task23_send_unchecked_prevalence", report_content)
    print_header("Task 23 Complete")


if __name__ == "__main__":
    main()
