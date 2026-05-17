#!/usr/bin/env python3
"""
task14_subsample_coverage.py — Sub-Sampling Coverage Audit

Find contracts where the .sol source is long enough to produce > 4 windows
(> ~2048 CodeBERT tokens). Sample 10 such contracts with known vulnerability
labels. For each:
1. Tokenize the FULL source with CodeBERT (return_overflowing_tokens=True) — get all W windows
2. Also load the sub-sampled token .pt from disk (4 windows)
3. Decode ALL windows back to text
4. Identify which windows contain vulnerability-relevant code
5. Check: are vulnerability-relevant windows included in the sub-sampled 4?
6. Compute: fraction of vulnerability code surviving sub-sampling

Report: per-contract analysis, survival rate, practical impact assessment.
"""

import sys
from pathlib import Path
from collections import Counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *

# Vulnerability-relevant patterns to search for in decoded windows
VULN_PATTERNS = [
    (r"\.call\s*[\({]", "low-level call (.call / .call{)"),
    (r"\.call\s*\.\s*value\s*\(", ".call.value() (old-style reentrancy)"),
    (r"block\.timestamp", "block.timestamp usage"),
    (r"block\.number", "block.number usage"),
    (r"unchecked", "unchecked block"),
    (r"\.send\s*\(", ".send() call"),
    (r"\.transfer\s*\(", ".transfer() call"),
    (r"require\s*\(", "require() check"),
    (r"assert\s*\(", "assert() check"),
    (r"delegatecall", "delegatecall"),
    (r"callcode", "callcode"),
    (r"selfdestruct", "selfdestruct"),
    (r"tx\.origin", "tx.origin usage"),
    (r"assembly", "inline assembly"),
    (r"revert\s*\(", "revert()"),
    (r"\.value\s*\(", ".value() call"),
]

import re

# Compile patterns
COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in VULN_PATTERNS]


def tokenize_full_source(sol_path: Path, tokenizer) -> dict:
    """
    Tokenize full source with CodeBERT using return_overflowing_tokens=True.
    Returns dict with all windows (not sub-sampled).
    """
    try:
        code = sol_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"error": str(e)}

    if len(code.strip()) == 0:
        return {"error": "empty file"}

    encoded = tokenizer(
        code,
        max_length=512,
        padding="max_length",
        truncation=True,
        stride=256,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"],        # [W, 512]
        "attention_mask": encoded["attention_mask"],  # [W, 512]
        "num_windows": encoded["input_ids"].shape[0],
    }


def load_windowed_token(stem: str) -> dict:
    """Load the sub-sampled token .pt from tokens_windowed/."""
    token_path = TOKENS_WINDOWED_DIR / f"{stem}.pt"
    if not token_path.exists():
        return {"error": f"token file not found: {token_path}"}
    try:
        data = load_token(token_path)
        return data
    except Exception as e:
        return {"error": str(e)}


def decode_window(tokenizer, input_ids_1d) -> str:
    """Decode a single window's input_ids back to text."""
    return tokenizer.decode(input_ids_1d, skip_special_tokens=True)


def find_vuln_patterns(text: str) -> list:
    """Search for vulnerability-relevant patterns in text. Returns list of (label, count)."""
    matches = []
    for pattern, label in COMPILED_PATTERNS:
        count = len(pattern.findall(text))
        if count > 0:
            matches.append((label, count))
    return matches


def main():
    print_header("Task 14: Sub-Sample Coverage Audit")

    # ── Load CodeBERT tokenizer ────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base",
            cache_dir=".cache/huggingface",
            use_fast=True,
        )
        print("CodeBERT tokenizer loaded successfully")
    except ImportError:
        print("ERROR: transformers not installed. Cannot run this audit.")
        save_report("task14_subsample_coverage",
                     "# Task 14: Sub-Sample Coverage Audit\n\n"
                     "**ERROR**: `transformers` library not installed.\n")
        return
    except Exception as e:
        print(f"ERROR: Could not load CodeBERT tokenizer: {e}")
        save_report("task14_subsample_coverage",
                     f"# Task 14: Sub-Sample Coverage Audit\n\n"
                     f"**ERROR**: Could not load tokenizer: {e}\n")
        return

    # ── Load labels and find candidates ─────────────────────────────────────
    if not LABEL_CSV.exists():
        print(f"ERROR: Label CSV not found at {LABEL_CSV}")
        save_report("task14_subsample_coverage",
                     "# Task 14: Sub-Sample Coverage Audit\n\n"
                     "**ERROR**: Label CSV not found.\n")
        return

    labels = load_label_csv()
    print(f"Loaded {len(labels)} entries from label CSV")

    # Build MD5→path mapping for all labeled contracts
    all_stems = set(labels.keys())
    print("Building MD5→path map (may take a moment)...")
    md5_to_path = build_md5_to_path(all_stems)
    print(f"  Found {len(md5_to_path)} paths for {len(all_stems)} stems")

    # ── Find contracts with > 4 windows ─────────────────────────────────────
    # A contract with > ~2048 CodeBERT tokens will produce > 4 windows.
    # We'll estimate by source file size and then verify with tokenization.
    print("\nFinding long contracts (>4 windows potential)...")

    # Find contracts with known vulnerability labels that also have
    # windowed token files AND source files
    if not TOKENS_WINDOWED_DIR.exists():
        print(f"ERROR: tokens_windowed directory not found at {TOKENS_WINDOWED_DIR}")
        save_report("task14_subsample_coverage",
                     "# Task 14: Sub-Sample Coverage Audit\n\n"
                     "**ERROR**: tokens_windowed directory not found.\n")
        return

    windowed_stems = {p.stem for p in TOKENS_WINDOWED_DIR.glob("*.pt")}
    print(f"  Found {len(windowed_stems)} windowed token files")

    # Find stems that exist in labels, have source, and have windowed tokens
    candidate_stems = []
    for stem, lbls in labels.items():
        # Must have at least one vulnerability label
        has_vuln = any(lbls[cls] == 1 for cls in VULN_CLASSES)
        if not has_vuln:
            continue
        # Must have source file
        if stem not in md5_to_path:
            continue
        # Must have windowed token file
        if stem not in windowed_stems:
            continue
        sol_path = md5_to_path[stem]
        # Estimate: file should be > ~6KB for > 4 windows
        try:
            file_size = sol_path.stat().st_size
            if file_size > 6000:  # rough threshold for >4 windows
                candidate_stems.append((stem, sol_path, file_size, lbls))
        except OSError:
            continue

    # Sort by file size descending (most likely to have many windows)
    candidate_stems.sort(key=lambda x: x[2], reverse=True)
    print(f"  Found {len(candidate_stems)} candidate contracts with vulnerability labels and size > 6KB")

    if not candidate_stems:
        print("WARNING: No candidate contracts found. Trying smaller threshold...")
        for stem, lbls in labels.items():
            has_vuln = any(lbls[cls] == 1 for cls in VULN_CLASSES)
            if not has_vuln:
                continue
            if stem not in md5_to_path:
                continue
            if stem not in windowed_stems:
                continue
            sol_path = md5_to_path[stem]
            try:
                file_size = sol_path.stat().st_size
                if file_size > 3000:
                    candidate_stems.append((stem, sol_path, file_size, lbls))
            except OSError:
                continue
        candidate_stems.sort(key=lambda x: x[2], reverse=True)
        print(f"  Found {len(candidate_stems)} candidates with lower threshold")

    # Sample 10 contracts
    sample_size = min(10, len(candidate_stems))
    sampled = candidate_stems[:sample_size]
    print(f"  Sampling {sample_size} contracts for analysis")

    # ── Analyze each contract ───────────────────────────────────────────────
    contract_reports = []

    for i, (stem, sol_path, file_size, lbls) in enumerate(sampled):
        print(f"\n  [{i+1}/{sample_size}] Analyzing {stem[:16]}... "
              f"({file_size} bytes, vuln: {[c for c in VULN_CLASSES if lbls[c]==1]})")

        # Step 1: Full tokenization (all windows)
        full_result = tokenize_full_source(sol_path, tokenizer)
        if "error" in full_result:
            print(f"    ERROR in full tokenization: {full_result['error']}")
            contract_reports.append({
                "stem": stem, "error": full_result["error"],
                "vuln_classes": [c for c in VULN_CLASSES if lbls[c] == 1],
            })
            continue

        W_full = full_result["num_windows"]
        print(f"    Full tokenization: {W_full} windows")

        if W_full <= 4:
            print(f"    Only {W_full} windows — no sub-sampling occurs. Skipping.")
            contract_reports.append({
                "stem": stem, "windows_full": W_full,
                "no_subsampling": True,
                "vuln_classes": [c for c in VULN_CLASSES if lbls[c] == 1],
            })
            continue

        # Step 2: Decode ALL full windows and find vuln-relevant ones
        full_windows_text = []
        vuln_windows_full = []  # (window_idx, patterns_found)

        for w_idx in range(W_full):
            ids = full_result["input_ids"][w_idx]
            text = decode_window(tokenizer, ids)
            full_windows_text.append(text)

            patterns = find_vuln_patterns(text)
            if patterns:
                vuln_windows_full.append((w_idx, patterns))

        print(f"    Vuln-relevant windows (full): {len(vuln_windows_full)}/{W_full}")

        # Step 3: Load sub-sampled token .pt
        sub_result = load_windowed_token(stem)
        if "error" in sub_result:
            print(f"    ERROR loading sub-sampled tokens: {sub_result['error']}")
            contract_reports.append({
                "stem": stem, "windows_full": W_full,
                "error_sub": sub_result["error"],
                "vuln_classes": [c for c in VULN_CLASSES if lbls[c] == 1],
            })
            continue

        sub_ids = sub_result["input_ids"]
        sub_num_windows = sub_result.get("num_windows", sub_ids.shape[0] if sub_ids.dim() == 2 else 1)
        print(f"    Sub-sampled: {sub_num_windows} windows")

        # Step 4: Decode sub-sampled windows
        sub_windows_text = []
        vuln_windows_sub = []

        if sub_ids.dim() == 2:
            for w_idx in range(sub_ids.shape[0]):
                text = decode_window(tokenizer, sub_ids[w_idx])
                sub_windows_text.append(text)
                patterns = find_vuln_patterns(text)
                if patterns:
                    vuln_windows_sub.append((w_idx, patterns))
        elif sub_ids.dim() == 1:
            text = decode_window(tokenizer, sub_ids)
            sub_windows_text.append(text)
            patterns = find_vuln_patterns(text)
            if patterns:
                vuln_windows_sub.append((0, patterns))

        print(f"    Vuln-relevant windows (sub): {len(vuln_windows_sub)}/{sub_num_windows}")

        # Step 5: Compute survival rate
        # Count unique vuln patterns in full windows vs sub-sampled
        full_pattern_labels = set()
        for _, patterns in vuln_windows_full:
            for label, _ in patterns:
                full_pattern_labels.add(label)

        sub_pattern_labels = set()
        for _, patterns in vuln_windows_sub:
            for label, _ in patterns:
                sub_pattern_labels.add(label)

        survived = full_pattern_labels & sub_pattern_labels
        lost = full_pattern_labels - sub_pattern_labels

        survival_rate = len(survived) / len(full_pattern_labels) if full_pattern_labels else 1.0
        print(f"    Survival rate: {survival_rate:.1%} ({len(survived)}/{len(full_pattern_labels)} patterns)")

        contract_reports.append({
            "stem": stem,
            "file_size": file_size,
            "windows_full": W_full,
            "windows_sub": sub_num_windows,
            "vuln_classes": [c for c in VULN_CLASSES if lbls[c] == 1],
            "vuln_windows_full": len(vuln_windows_full),
            "vuln_windows_sub": len(vuln_windows_sub),
            "full_patterns": sorted(full_pattern_labels),
            "sub_patterns": sorted(sub_pattern_labels),
            "survived_patterns": sorted(survived),
            "lost_patterns": sorted(lost),
            "survival_rate": survival_rate,
        })

    # ── Build report ────────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 14: Sub-Sample Coverage Audit\n\n")
    report_lines.append(f"**Contracts analyzed:** {len(contract_reports)}  \n")
    report_lines.append(f"**Max windows (sub-sampled):** 4  \n")
    report_lines.append(f"**Window size:** 512 tokens, stride 256\n\n")

    # Per-contract analysis
    report_lines.append("## Per-Contract Analysis\n\n")

    overall_survival_rates = []

    for i, cr in enumerate(contract_reports):
        report_lines.append(f"### Contract {i+1}: `{cr['stem'][:24]}...`\n\n")
        report_lines.append(f"- **Vulnerability classes:** {', '.join(cr.get('vuln_classes', []))}\n")

        if "error" in cr:
            report_lines.append(f"- **Error:** {cr['error']}\n\n")
            continue

        if cr.get("no_subsampling"):
            report_lines.append(f"- **Full windows:** {cr['windows_full']} (no sub-sampling needed)\n\n")
            continue

        if "error_sub" in cr:
            report_lines.append(f"- **Full windows:** {cr['windows_full']}\n")
            report_lines.append(f"- **Sub-sampled load error:** {cr['error_sub']}\n\n")
            continue

        report_lines.append(f"- **File size:** {cr['file_size']:,} bytes\n")
        report_lines.append(f"- **Full windows:** {cr['windows_full']}\n")
        report_lines.append(f"- **Sub-sampled windows:** {cr['windows_sub']}\n")
        report_lines.append(f"- **Vuln-relevant windows (full):** {cr['vuln_windows_full']}\n")
        report_lines.append(f"- **Vuln-relevant windows (sub):** {cr['vuln_windows_sub']}\n")
        report_lines.append(f"- **Survival rate:** {cr['survival_rate']:.1%}\n\n")

        if cr["lost_patterns"]:
            report_lines.append(f"- **Lost patterns:** {', '.join(cr['lost_patterns'])}\n")
        if cr["survived_patterns"]:
            report_lines.append(f"- **Survived patterns:** {', '.join(cr['survived_patterns'])}\n")
        report_lines.append("\n")

        overall_survival_rates.append(cr["survival_rate"])

    # Summary
    report_lines.append("## Summary\n\n")
    if overall_survival_rates:
        avg_survival = sum(overall_survival_rates) / len(overall_survival_rates)
        min_survival = min(overall_survival_rates)
        max_survival = max(overall_survival_rates)
        report_lines.append(f"- **Average survival rate:** {avg_survival:.1%}\n")
        report_lines.append(f"- **Min survival rate:** {min_survival:.1%}\n")
        report_lines.append(f"- **Max survival rate:** {max_survival:.1%}\n\n")
    else:
        report_lines.append("No survival rates computed (insufficient data).\n\n")

    # Practical impact
    report_lines.append("## Practical Impact Assessment\n\n")
    report_lines.append("### How Sub-Sampling Works\n\n")
    report_lines.append("The windowed tokenizer (`retokenize_windowed.py`) uses `linspace` sub-sampling:\n")
    report_lines.append("- Contracts with ≤ 4 windows: all windows retained (no loss)\n")
    report_lines.append("- Contracts with > 4 windows: 4 evenly-spaced windows selected via `np.linspace(0, W-1, 4)`\n")
    report_lines.append("  - This covers start, ~1/3, ~2/3, and end of the contract\n\n")
    report_lines.append("### Key Findings\n\n")
    if overall_survival_rates:
        if avg_survival >= 0.9:
            report_lines.append("- **LOW RISK**: Average survival rate ≥ 90%. Most vulnerability-relevant "
                                "code survives sub-sampling.\n")
        elif avg_survival >= 0.7:
            report_lines.append("- **MODERATE RISK**: Average survival rate 70-90%. Some vulnerability patterns "
                                "may be lost in long contracts.\n")
        else:
            report_lines.append("- **HIGH RISK**: Average survival rate < 70%. Significant vulnerability signal "
                                "is lost due to sub-sampling.\n")
        report_lines.append("- The `linspace` strategy biases toward start and end coverage, which may miss "
                            "vulnerability code in the middle of very long contracts.\n")
        report_lines.append("- Consider increasing `MAX_WINDOWS` from 4 to 6-8 if VRAM permits.\n")
    else:
        report_lines.append("- Insufficient data to assess impact. Ensure long contracts with vulnerability "
                            "labels exist in the dataset.\n")

    report_content = "".join(report_lines)
    save_report("task14_subsample_coverage", report_content)
    print_header("Task 14 Complete")


if __name__ == "__main__":
    main()
