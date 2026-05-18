#!/usr/bin/env python3
"""
task18_solidity_version_dist.py — Solidity Version Distribution Audit for SENTINEL v6

Scan 2000 .sol files from BCCC source dirs. For each:
1. Extract pragma version using extract_pragma_version()
2. Classify into buckets: 0.4.x, 0.5.x, 0.6.x, 0.7.x, 0.8.x, no_pragma
3. For each version bucket, compute count/percentage and per-class label distribution
4. For 0.8.x contracts: grep for unchecked{} and using SafeMath
5. Check train/val/test split version distribution (if split files exist)
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import random
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


# ── Version bucket classification ─────────────────────────────────────────────
def classify_version(version_str):
    """Classify a version string like '0.8' into a bucket."""
    if version_str is None:
        return "no_pragma"
    parts = version_str.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}.x"
    return "unknown"


# ── 0.8.x-specific patterns ──────────────────────────────────────────────────
_UNCHECKED_RE = re.compile(r'\bunchecked\s*\{')
_USING_SAFEMATH_RE = re.compile(r'\busing\s+SafeMath\b')


def main():
    print_header("Task 18: Solidity Version Distribution Audit")

    # ── Collect .sol files ────────────────────────────────────────────────
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

    # ── Sample 2000 ───────────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(all_sol_files)
    sample_size = min(2000, len(all_sol_files))
    sampled_files = all_sol_files[:sample_size]
    print(f"Sampling {sample_size} files")

    # ── Load labels ───────────────────────────────────────────────────────
    try:
        labels = load_label_csv()
    except Exception as e:
        print(f"WARNING: Could not load label CSV: {e}")
        labels = {}

    # ── Build reverse md5 mapping ─────────────────────────────────────────
    print("Building md5→path mapping for sampled files...")
    sampled_md5s = set()
    for sol_path in sampled_files:
        try:
            rel = sol_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = sol_path
        md5 = get_contract_hash(rel)
        sampled_md5s.add(md5)

    # ── Scan files ────────────────────────────────────────────────────────
    version_counts = Counter()           # bucket → count
    version_labels = defaultdict(Counter) # bucket → {class: count_of_positive}
    version_total_labeled = Counter()     # bucket → count with labels
    unchecked_count_08 = 0
    safemath_count_08 = 0
    total_08 = 0
    unchecked_examples = []
    safemath_examples_08 = []

    for i, sol_path in enumerate(sampled_files):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{sample_size} files...")

        try:
            rel = sol_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = sol_path
        md5 = get_contract_hash(rel)

        # Extract pragma version
        version_str = extract_pragma_version(sol_path)
        bucket = classify_version(version_str)
        version_counts[bucket] += 1

        # Match to labels
        if md5 in labels:
            version_total_labeled[bucket] += 1
            for cls in VULN_CLASSES:
                if labels[md5].get(cls, 0) == 1:
                    version_labels[bucket][cls] += 1

        # 0.8.x-specific analysis
        if bucket == "0.8.x":
            total_08 += 1
            try:
                source = sol_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if _UNCHECKED_RE.search(source):
                unchecked_count_08 += 1
                if len(unchecked_examples) < 10:
                    unchecked_examples.append(str(sol_path.name))

            if _USING_SAFEMATH_RE.search(source):
                safemath_count_08 += 1
                if len(safemath_examples_08) < 10:
                    safemath_examples_08.append(str(sol_path.name))

    # ── Check train/val/test split version distribution ───────────────────
    print("\nChecking for train/val/test split files...")
    processed_dir = PROJECT_ROOT / "ml" / "data" / "processed"
    split_dists = {}
    split_files = {
        "train": processed_dir / "train_stems.txt",
        "val": processed_dir / "val_stems.txt",
        "test": processed_dir / "test_stems.txt",
    }

    # Also check for CSV-based splits
    for suffix in ["train.csv", "val.csv", "test.csv",
                   "split_train.txt", "split_val.txt", "split_test.txt"]:
        p = processed_dir / suffix
        if p.exists():
            name = suffix.replace("split_", "").replace(".txt", "").replace(".csv", "")
            split_files[name] = p

    for split_name, split_path in split_files.items():
        if not split_path.exists():
            continue
        print(f"  Found split file: {split_path}")
        try:
            if split_path.suffix == ".csv":
                import csv
                with open(split_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    stems = [row.get("md5_stem", row.get("stem", "")) for row in reader]
            else:
                stems = [line.strip() for line in split_path.read_text().splitlines()
                         if line.strip()]

            split_version_counts = Counter()
            # Sample up to 500 stems for version checking
            check_stems = random.sample(stems, min(500, len(stems)))
            # Build md5→path for these
            check_md5s = set(check_stems)
            check_md5_to_path = build_md5_to_path(check_md5s)

            for stem in check_stems:
                sol_path = check_md5_to_path.get(stem)
                if sol_path and sol_path.exists():
                    v = extract_pragma_version(sol_path)
                    b = classify_version(v)
                    split_version_counts[b] += 1

            split_dists[split_name] = split_version_counts
        except Exception as e:
            print(f"  WARNING: Could not process split file {split_path}: {e}")

    # ── Build report ──────────────────────────────────────────────────────
    report = []
    report.append("# Task 18: Solidity Version Distribution Audit\n")
    report.append(f"**Files scanned:** {sample_size}  ")
    report.append(f"**Files with labels:** {sum(version_total_labeled.values())}  ")
    report.append(f"**Unique version buckets:** {len(version_counts)}\n")

    # Version distribution table
    report.append("\n## Version Distribution\n")
    report.append("| Version Bucket | Count | Percentage |")
    report.append("\n|----------------|-------|------------|")
    for bucket in sorted(version_counts.keys()):
        count = version_counts[bucket]
        pct = f"{count/sample_size:.1%}" if sample_size > 0 else "N/A"
        report.append(f"\n| {bucket} | {count} | {pct} |")

    # Per-class label distribution by version
    report.append("\n\n## Per-Class Label Distribution by Version Bucket\n")
    all_buckets = sorted(version_counts.keys())
    header = "| Version | " + " | ".join(VULN_CLASSES) + " | Total Labeled |"
    sep = "|---------|" + "|".join(["-------"] * len(VULN_CLASSES)) + "|---------------|"
    report.append(f"\n{header}")
    report.append(f"\n{sep}")

    for bucket in all_buckets:
        total_lbl = version_total_labeled[bucket]
        rates = []
        for cls in VULN_CLASSES:
            pos = version_labels[bucket].get(cls, 0)
            if total_lbl > 0:
                rates.append(f"{pos/total_lbl:.1%}")
            else:
                rates.append("N/A")
        rates_str = " | ".join(rates)
        report.append(f"\n| {bucket} | {rates_str} | {total_lbl} |")

    # 0.8.x-specific analysis
    report.append("\n\n## 0.8.x Analysis\n")
    report.append(f"**Total 0.8.x files:** {total_08}  ")
    report.append(f"**With `unchecked {{}}`:** {unchecked_count_08} "
                  f"({unchecked_count_08/total_08:.1%}) " if total_08 > 0 else "")
    report.append(f"\n**With `using SafeMath`:** {safemath_count_08} "
                  f"({safemath_count_08/total_08:.1%})" if total_08 > 0 else "")

    # Note: 0.8.x has built-in overflow protection, so:
    # - unchecked{} bypasses it → IntegerUO risk
    # - using SafeMath in 0.8.x is redundant (but may indicate migrated code)
    report.append("\n\n### Interpretation\n")
    report.append("- **`unchecked {}` in 0.8.x** signals potential IntegerUO: Solidity 0.8.0+ "
                  "has built-in overflow/underflow checks; `unchecked` blocks bypass them.")
    report.append("- **`using SafeMath` in 0.8.x** is typically redundant (SafeMath is "
                  "unnecessary with built-in checks) and may indicate code migrated from "
                  "older versions without cleanup.")
    report.append(f"\n- {unchecked_count_08} of {total_08} 0.8.x files ({unchecked_count_08/total_08:.1%} "
                  if total_08 > 0 else "- ")
    report.append("contain `unchecked {}` blocks — these are the primary IntegerUO candidates in 0.8.x.")

    if unchecked_examples:
        report.append("\n\n### Example files with `unchecked {}`\n")
        for ex in unchecked_examples[:10]:
            report.append(f"- `{ex}`")

    if safemath_examples_08:
        report.append("\n\n### Example 0.8.x files with `using SafeMath` (redundant)\n")
        for ex in safemath_examples_08[:10]:
            report.append(f"- `{ex}`")

    # Train/val/test split distribution
    if split_dists:
        report.append("\n\n## Train/Val/Test Split Version Distribution\n")
        all_buckets_split = sorted(set(b for sd in split_dists.values() for b in sd.keys()))
        header = "| Version | " + " | ".join(split_dists.keys()) + " |"
        sep = "|---------|" + "|".join(["-------"] * len(split_dists)) + "|"
        report.append(f"\n{header}")
        report.append(f"\n{sep}")

        for bucket in all_buckets_split:
            vals = []
            for split_name in split_dists:
                count = split_dists[split_name].get(bucket, 0)
                total_split = sum(split_dists[split_name].values())
                vals.append(f"{count} ({count/total_split:.0%})" if total_split > 0 else "0")
            vals_str = " | ".join(vals)
            report.append(f"\n| {bucket} | {vals_str} |")

        # Check for distribution shift
        report.append("\n\n### Distribution Shift Check\n")
        if len(split_dists) >= 2:
            split_names = list(split_dists.keys())
            for i, s1 in enumerate(split_names):
                for s2 in split_names[i+1:]:
                    t1 = sum(split_dists[s1].values())
                    t2 = sum(split_dists[s2].values())
                    if t1 == 0 or t2 == 0:
                        continue
                    max_diff = 0
                    diff_bucket = ""
                    for bucket in all_buckets_split:
                        r1 = split_dists[s1].get(bucket, 0) / t1
                        r2 = split_dists[s2].get(bucket, 0) / t2
                        diff = abs(r1 - r2)
                        if diff > max_diff:
                            max_diff = diff
                            diff_bucket = bucket
                    if max_diff > 0.05:
                        report.append(f"- **{s1} vs {s2}**: Max version distribution shift "
                                      f"= {max_diff:.1%} (bucket: {diff_bucket}) — potential concern")
                    else:
                        report.append(f"- **{s1} vs {s2}**: Version distributions are well-aligned "
                                      f"(max shift: {max_diff:.1%})")
    else:
        report.append("\n\n## Train/Val/Test Split Version Distribution\n")
        report.append("*No split files found in `ml/data/processed/`.*")

    # Key findings summary
    report.append("\n\n## Key Findings\n")

    # Find dominant version
    if version_counts:
        dominant = version_counts.most_common(1)[0]
        report.append(f"1. **Dominant version:** {dominant[0]} ({dominant[1]/sample_size:.1%} of files)")

    # Version class rate differences
    report.append("\n2. **Notable class rate differences across versions:**")
    for cls in VULN_CLASSES:
        rates_by_version = []
        for bucket in all_buckets:
            total_lbl = version_total_labeled[bucket]
            if total_lbl > 0:
                pos = version_labels[bucket].get(cls, 0)
                rate = pos / total_lbl
                rates_by_version.append((bucket, rate))
        if rates_by_version:
            rates_by_version.sort(key=lambda x: x[1], reverse=True)
            max_r = rates_by_version[0]
            min_r = rates_by_version[-1]
            if max_r[1] - min_r[1] > 0.05:
                report.append(f"   - **{cls}**: ranges from {min_r[1]:.1%} ({min_r[0]}) "
                              f"to {max_r[1]:.1%} ({max_r[0]})")

    save_report("task18_solidity_version_dist", "".join(report))
    print_header("Task 18 Complete")


if __name__ == "__main__":
    main()
