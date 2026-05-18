#!/usr/bin/env python3
"""
task12_token_integrity.py — Token Integrity Audit for SENTINEL v6

Sample 100 token .pt files from tokens_windowed/. For each verify:
  - input_ids shape [4,512]
  - attention_mask shape matches
  - padding windows have mask=0
  - real windows have mask!=0
  - input_ids in [0,50265]
  - num_tokens == mask.sum()
  - no NaN/negative in input_ids
  - feature_schema_version=="v4"

Also compute distribution of num_windows and num_tokens.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def _get_field(obj, key, default=None):
    """Get field from either dict or object with attributes."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def main():
    print_header("Task 12: Token Integrity Audit")

    # ── Collect token files ────────────────────────────────────────────────
    all_token_files = sorted(TOKENS_WINDOWED_DIR.glob("*.pt"))
    if not all_token_files:
        print("ERROR: No token .pt files found in", TOKENS_WINDOWED_DIR)
        return
    print(f"Found {len(all_token_files)} token files")

    # Exclude checkpoint files
    all_token_files = [p for p in all_token_files if p.suffix == ".pt" and not p.name.startswith("checkpoint")]
    print(f"  (after filtering: {len(all_token_files)} .pt files)")

    sample_size = min(100, len(all_token_files))
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(len(all_token_files), size=sample_size, replace=False)
    sampled_files = [all_token_files[i] for i in sampled_indices]
    print(f"Sampling {sample_size} files")

    # ── Verification checks ────────────────────────────────────────────────
    checks = {
        "input_ids_shape": 0,       # shape [4,512]
        "attention_mask_shape": 0,  # shape matches input_ids
        "padding_windows_mask0": 0, # padding windows have mask=0
        "real_windows_mask_non0": 0,# real windows have mask!=0
        "input_ids_range": 0,       # all values in [0,50265]
        "num_tokens_match": 0,      # num_tokens == mask.sum()
        "no_nan": 0,                # no NaN in input_ids
        "no_negative": 0,           # no negative values in input_ids
        "schema_v4": 0,             # feature_schema_version == "v4"
    }
    total = 0
    skipped = 0
    failures = []

    # Distribution tracking
    num_windows_dist = Counter()
    num_tokens_list = []

    for i, fpath in enumerate(sampled_files):
        if (i + 1) % 25 == 0:
            print(f"  Checked {i + 1}/{sample_size} files...")

        try:
            data = load_token(fpath)
        except Exception as e:
            skipped += 1
            failures.append({"stem": fpath.stem, "issue": f"load error: {e}"})
            continue

        total += 1
        stem = fpath.stem
        file_failures = []

        input_ids = _get_field(data, "input_ids")
        attention_mask = _get_field(data, "attention_mask")
        num_windows = _get_field(data, "num_windows")
        num_tokens = _get_field(data, "num_tokens")
        schema_ver = _get_field(data, "feature_schema_version")

        # 1. input_ids shape [4,512]
        if input_ids is not None:
            if input_ids.shape == (4, 512):
                checks["input_ids_shape"] += 1
            else:
                file_failures.append(f"input_ids shape {tuple(input_ids.shape)} != (4, 512)")
        else:
            file_failures.append("input_ids missing")

        # 2. attention_mask shape matches
        if attention_mask is not None and input_ids is not None:
            if attention_mask.shape == input_ids.shape:
                checks["attention_mask_shape"] += 1
            else:
                file_failures.append(f"attention_mask shape {tuple(attention_mask.shape)} != input_ids shape")
        else:
            file_failures.append("attention_mask missing or shape mismatch")

        # 3. Padding windows have mask=0
        if attention_mask is not None and num_windows is not None:
            mask_np = attention_mask.numpy() if hasattr(attention_mask, 'numpy') else np.array(attention_mask)
            nw = int(num_windows) if num_windows is not None else None
            if nw is not None and nw < 4:
                # Windows from nw to 4 should be all-zero
                padding_ok = True
                for w in range(nw, 4):
                    if w < mask_np.shape[0]:
                        if not np.all(mask_np[w] == 0):
                            padding_ok = False
                            break
                if padding_ok:
                    checks["padding_windows_mask0"] += 1
                else:
                    file_failures.append(f"padding window has non-zero mask (num_windows={nw})")
            elif nw is not None and nw == 4:
                # No padding windows — trivially pass
                checks["padding_windows_mask0"] += 1
            else:
                file_failures.append(f"unexpected num_windows={nw}")
        else:
            file_failures.append("cannot check padding windows")

        # 4. Real windows have mask!=0
        if attention_mask is not None and num_windows is not None:
            mask_np = attention_mask.numpy() if hasattr(attention_mask, 'numpy') else np.array(attention_mask)
            nw = int(num_windows)
            real_ok = True
            for w in range(min(nw, mask_np.shape[0])):
                if np.all(mask_np[w] == 0):
                    real_ok = False
                    break
            if real_ok:
                checks["real_windows_mask_non0"] += 1
            else:
                file_failures.append(f"real window {w} has all-zero mask")
        else:
            file_failures.append("cannot check real windows")

        # 5. input_ids in [0, 50265]
        if input_ids is not None:
            ids_np = input_ids.numpy() if hasattr(input_ids, 'numpy') else np.array(input_ids)
            if np.all((ids_np >= 0) & (ids_np <= 50265)):
                checks["input_ids_range"] += 1
            else:
                oob_min = int(ids_np.min())
                oob_max = int(ids_np.max())
                file_failures.append(f"input_ids out of range [{oob_min}, {oob_max}]")
        else:
            file_failures.append("input_ids missing for range check")

        # 6. num_tokens == mask.sum()
        if num_tokens is not None and attention_mask is not None:
            mask_np = attention_mask.numpy() if hasattr(attention_mask, 'numpy') else np.array(attention_mask)
            expected = int(mask_np.sum())
            reported = int(num_tokens)
            if expected == reported:
                checks["num_tokens_match"] += 1
            else:
                file_failures.append(f"num_tokens={reported} != mask.sum()={expected}")
        else:
            file_failures.append("cannot verify num_tokens")

        # 7. No NaN in input_ids
        if input_ids is not None:
            ids_np = input_ids.numpy() if hasattr(input_ids, 'numpy') else np.array(input_ids)
            if not np.any(np.isnan(ids_np.astype(np.float32))):
                checks["no_nan"] += 1
            else:
                file_failures.append("input_ids contains NaN")
        else:
            file_failures.append("input_ids missing for NaN check")

        # 8. No negative in input_ids
        if input_ids is not None:
            ids_np = input_ids.numpy() if hasattr(input_ids, 'numpy') else np.array(input_ids)
            if not np.any(ids_np < 0):
                checks["no_negative"] += 1
            else:
                file_failures.append("input_ids contains negative values")
        else:
            file_failures.append("input_ids missing for negative check")

        # 9. feature_schema_version == "v4"
        if schema_ver is not None:
            if str(schema_ver) == "v4":
                checks["schema_v4"] += 1
            else:
                file_failures.append(f"schema version '{schema_ver}' != 'v4'")
        else:
            file_failures.append("feature_schema_version missing")

        # Record distributions
        if num_windows is not None:
            num_windows_dist[int(num_windows)] += 1
        if num_tokens is not None:
            num_tokens_list.append(int(num_tokens))

        # Collect failures
        if file_failures:
            failures.append({"stem": stem, "issues": file_failures})

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 12: Token Integrity Audit\n")
    report_lines.append(f"**Sample size:** {sample_size}  \n")
    report_lines.append(f"**Successfully loaded:** {total}  \n")
    report_lines.append(f"**Skipped (load errors):** {skipped}\n")

    # Pass/fail summary
    report_lines.append("\n## Check Results\n")
    report_lines.append("| Check | Pass | Fail | Rate |\n")
    report_lines.append("|-------|------|------|------|\n")
    for check_name, pass_count in checks.items():
        fail_count = total - pass_count
        rate = f"{pass_count/total:.1%}" if total > 0 else "N/A"
        report_lines.append(f"| {check_name} | {pass_count} | {fail_count} | {rate} |\n")

    # Distribution of num_windows
    report_lines.append("\n## Distribution of num_windows\n")
    report_lines.append("| num_windows | Count |\n")
    report_lines.append("|-------------|-------|\n")
    for nw in sorted(num_windows_dist.keys()):
        report_lines.append(f"| {nw} | {num_windows_dist[nw]} |\n")

    # Distribution of num_tokens
    report_lines.append("\n## Distribution of num_tokens\n")
    if num_tokens_list:
        tokens_arr = np.array(num_tokens_list)
        report_lines.append(f"| Statistic | Value |\n")
        report_lines.append(f"|-----------|-------|\n")
        report_lines.append(f"| min | {tokens_arr.min()} |\n")
        report_lines.append(f"| max | {tokens_arr.max()} |\n")
        report_lines.append(f"| mean | {tokens_arr.mean():.1f} |\n")
        report_lines.append(f"| median | {np.median(tokens_arr):.1f} |\n")
        report_lines.append(f"| p5 | {np.percentile(tokens_arr, 5):.1f} |\n")
        report_lines.append(f"| p95 | {np.percentile(tokens_arr, 95):.1f} |\n")

        # Histogram buckets
        report_lines.append("\n### Token Count Histogram\n")
        buckets = [0, 256, 512, 768, 1024, 1536, 2048, 999999]
        bucket_labels = ["0-255", "256-511", "512-767", "768-1023", "1024-1535", "1536-2047", "2048+"]
        report_lines.append("| Range | Count |\n")
        report_lines.append("|-------|-------|\n")
        for j in range(len(bucket_labels)):
            lo, hi = buckets[j], buckets[j + 1]
            count = int(np.sum((tokens_arr >= lo) & (tokens_arr < hi)))
            report_lines.append(f"| {bucket_labels[j]} | {count} |\n")

    # Failure details
    if failures:
        report_lines.append("\n## Failures Detail (first 30)\n")
        for f_entry in failures[:30]:
            issues_str = "; ".join(f_entry["issues"]) if isinstance(f_entry["issues"], list) else f_entry["issues"]
            report_lines.append(f"- **{f_entry['stem']}**: {issues_str}\n")
        if len(failures) > 30:
            report_lines.append(f"- ... and {len(failures) - 30} more files with failures\n")
    else:
        report_lines.append("\n✅ All checks passed on all sampled files.\n")

    report_content = "".join(report_lines)
    save_report("task12_token_integrity", report_content)
    print_header("Task 12 Complete")


if __name__ == "__main__":
    main()
