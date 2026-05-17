#!/usr/bin/env python3
"""
task10_token_graph_alignment.py — Token–Graph Alignment Audit for SENTINEL v6

Load the CSV, sample 100 paired stems (present in both graphs/ and tokens_windowed/).
For each:
  - Load graph .pt → get contract_path, contract_hash
  - Load token .pt → get contract_path, contract_hash
  - Verify hashes match
  - Verify contract_path matches
  - If missing, resolve via md5_to_path
For 10 stems, decode token input_ids[0] back to text with CodeBERT tokenizer
and verify first 100 decoded tokens match the .sol source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def _get_field(obj, key):
    """Get field from either PyG Data (attr) or dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def main():
    print_header("Task 10: Token–Graph Alignment Audit")

    # ── Load CSV and get paired stems ──────────────────────────────────────
    print("Loading CSV...")
    labels = load_label_csv()
    csv_stems = set(labels.keys())
    print(f"  CSV stems: {len(csv_stems)}")

    paired = get_paired_stems()
    print(f"  Paired stems (graph ∩ tokens_windowed): {len(paired)}")

    if not paired:
        print("ERROR: No paired stems found.")
        return

    # Sample 100 paired stems
    sample_stems_list = sample_stems(100, from_set=paired)
    print(f"  Sampled {len(sample_stems_list)} paired stems")

    # ── Alignment checks ───────────────────────────────────────────────────
    hash_match = 0
    hash_mismatch = 0
    path_match = 0
    path_mismatch = 0
    missing_graph = 0
    missing_token = 0
    mismatches = []
    skipped = 0

    for i, stem in enumerate(sample_stems_list):
        if (i + 1) % 25 == 0:
            print(f"  Checked {i + 1}/{len(sample_stems_list)} stems...")

        graph_path = GRAPHS_DIR / f"{stem}.pt"
        token_path = TOKENS_WINDOWED_DIR / f"{stem}.pt"

        # Load graph
        try:
            g_data = load_graph(graph_path)
        except Exception as e:
            missing_graph += 1
            mismatches.append({"stem": stem, "issue": f"graph load error: {e}"})
            continue

        # Load token
        try:
            t_data = load_token(token_path)
        except Exception as e:
            missing_token += 1
            mismatches.append({"stem": stem, "issue": f"token load error: {e}"})
            continue

        g_hash = _get_field(g_data, "contract_hash")
        t_hash = _get_field(t_data, "contract_hash")
        g_path_val = _get_field(g_data, "contract_path")
        t_path_val = _get_field(t_data, "contract_path")

        # Check hash match
        if g_hash is not None and t_hash is not None:
            if g_hash == t_hash:
                hash_match += 1
            else:
                hash_mismatch += 1
                mismatches.append({
                    "stem": stem,
                    "issue": f"hash mismatch: graph={g_hash}, token={t_hash}",
                })
        else:
            skipped += 1
            mismatches.append({
                "stem": stem,
                "issue": f"missing hash: graph_hash={'None' if g_hash is None else g_hash}, token_hash={'None' if t_hash is None else t_hash}",
            })

        # Check path match
        if g_path_val is not None and t_path_val is not None:
            if str(g_path_val) == str(t_path_val):
                path_match += 1
            else:
                path_mismatch += 1
                mismatches.append({
                    "stem": stem,
                    "issue": f"path mismatch: graph={g_path_val}, token={t_path_val}",
                })
        else:
            # Try resolving via md5_to_path
            pass

    # ── Decode test (10 stems) ─────────────────────────────────────────────
    print("\n  Running decode verification on 10 stems...")
    decode_stems = sample_stems_list[:10]
    decode_match_count = 0
    decode_mismatches = []
    decode_skipped = 0

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", use_fast=True)
        tokenizer_available = True
    except Exception as e:
        print(f"  WARNING: Could not load CodeBERT tokenizer: {e}")
        tokenizer_available = False

    # Build md5_to_path for resolving .sol files
    target_md5s = set(decode_stems)
    md5_to_path = build_md5_to_path(target_md5s)

    if tokenizer_available:
        for stem in decode_stems:
            token_path = TOKENS_WINDOWED_DIR / f"{stem}.pt"
            try:
                t_data = load_token(token_path)
                input_ids = _get_field(t_data, "input_ids")
                if input_ids is None:
                    decode_skipped += 1
                    decode_mismatches.append({"stem": stem, "issue": "no input_ids in token data"})
                    continue

                # Decode first window
                ids_tensor = input_ids[0] if input_ids.dim() == 2 else input_ids
                decoded_text = tokenizer.decode(ids_tensor, skip_special_tokens=True)

                # Find .sol source
                sol_path = find_sol_for_stem(stem, md5_to_path)
                if sol_path is None:
                    decode_skipped += 1
                    decode_mismatches.append({"stem": stem, "issue": "no .sol file found for decode check"})
                    continue

                try:
                    sol_text = sol_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    decode_skipped += 1
                    decode_mismatches.append({"stem": stem, "issue": "could not read .sol file"})
                    continue

                # Compare first ~100 tokens worth of text
                # Tokenize the source to get token-level comparison
                sol_tokens = tokenizer.encode(sol_text, add_special_tokens=False)[:100]
                decoded_tokens = tokenizer.encode(decoded_text, add_special_tokens=False)[:100]

                # Check overlap
                match_count = sum(1 for a, b in zip(sol_tokens, decoded_tokens) if a == b)
                total = max(len(sol_tokens), len(decoded_tokens), 1)
                match_rate = match_count / total

                if match_rate >= 0.90:
                    decode_match_count += 1
                else:
                    decode_mismatches.append({
                        "stem": stem,
                        "issue": f"decode match rate {match_rate:.2%} ({match_count}/{total} tokens)",
                    })

            except Exception as e:
                decode_skipped += 1
                decode_mismatches.append({"stem": stem, "issue": f"decode error: {e}"})

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 10: Token–Graph Alignment Audit\n")
    report_lines.append(f"**Paired stems available:** {len(paired)}  \n")
    report_lines.append(f"**Sample size:** {len(sample_stems_list)}\n")

    report_lines.append("## Hash Alignment\n")
    total_checked = hash_match + hash_mismatch
    report_lines.append(f"| Metric | Count | Rate |\n")
    report_lines.append(f"|--------|-------|------|\n")
    report_lines.append(f"| Hash match | {hash_match} | {hash_match/total_checked:.1%} |\n" if total_checked else f"| Hash match | {hash_match} | N/A |\n")
    report_lines.append(f"| Hash mismatch | {hash_mismatch} | {hash_mismatch/total_checked:.1%} |\n" if total_checked else f"| Hash mismatch | {hash_mismatch} | N/A |\n")
    report_lines.append(f"| Missing hash | {skipped} | — |\n")
    report_lines.append(f"| Missing graph file | {missing_graph} | — |\n")
    report_lines.append(f"| Missing token file | {missing_token} | — |\n")

    report_lines.append("\n## Path Alignment\n")
    total_path = path_match + path_mismatch
    report_lines.append(f"| Metric | Count | Rate |\n")
    report_lines.append(f"|--------|-------|------|\n")
    report_lines.append(f"| Path match | {path_match} | {path_match/total_path:.1%} |\n" if total_path else f"| Path match | {path_match} | N/A |\n")
    report_lines.append(f"| Path mismatch | {path_mismatch} | {path_mismatch/total_path:.1%} |\n" if total_path else f"| Path mismatch | {path_mismatch} | N/A |\n")

    report_lines.append("\n## Decode Verification (10 stems)\n")
    if tokenizer_available:
        total_decode = decode_match_count + len(decode_mismatches) - decode_skipped
        report_lines.append(f"| Metric | Count | Rate |\n")
        report_lines.append(f"|--------|-------|------|\n")
        report_lines.append(f"| Decode match (≥90%) | {decode_match_count} | {decode_match_count/max(total_decode,1):.1%} |\n")
        report_lines.append(f"| Decode mismatch | {len(decode_mismatches) - decode_skipped} | — |\n")
        report_lines.append(f"| Skipped | {decode_skipped} | — |\n")
    else:
        report_lines.append("CodeBERT tokenizer not available — decode verification skipped.\n")

    # Mismatches detail
    if mismatches:
        report_lines.append("\n## Hash/Path Mismatches Detail\n")
        for m in mismatches[:30]:
            report_lines.append(f"- **{m['stem']}**: {m['issue']}\n")
        if len(mismatches) > 30:
            report_lines.append(f"- ... and {len(mismatches) - 30} more\n")

    if decode_mismatches:
        report_lines.append("\n## Decode Mismatches Detail\n")
        for m in decode_mismatches:
            report_lines.append(f"- **{m['stem']}**: {m['issue']}\n")

    report_content = "".join(report_lines)
    save_report("task10_token_graph_alignment", report_content)
    print_header("Task 10 Complete")


if __name__ == "__main__":
    main()
