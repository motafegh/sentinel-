#!/usr/bin/env python3
"""
task24_token_graph_source_alignment.py — Token-Graph-Source Alignment Audit

Similar to Task 10 but focuses on verifying the retokenization produced
correct pairings. Sample 100 stems with both graph and token files. For each:
1. Load graph .pt → contract_hash, contract_path
2. Load token .pt → contract_hash, contract_path
3. Verify hashes match
4. Read .sol file, compute its MD5, verify matches filename stem
5. If contract_path missing in token: resolve via md5_to_path and compare
6. For 10 stems: decode input_ids[0] with CodeBERT tokenizer, compare first
   200 decoded tokens with .sol source
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.common import *


def _get_field(obj, key):
    """Get field from either PyG Data (attr) or dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def compute_file_md5(path: Path) -> str:
    """Compute MD5 hash of a file's contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print_header("Task 24: Token-Graph-Source Alignment Audit")

    # ── Get paired stems ───────────────────────────────────────────────────
    print("  Finding paired stems...")
    paired = get_paired_stems()
    print(f"  Paired stems (graph ∩ tokens_windowed): {len(paired)}")

    if not paired:
        print("ERROR: No paired stems found.")
        return

    # Sample 100 paired stems
    sample_stems_list = sample_stems(100, from_set=paired)
    print(f"  Sampled {len(sample_stems_list)} paired stems")

    # ── Build md5_to_path for .sol resolution ──────────────────────────────
    print("  Building md5_to_path mapping...")
    target_md5s = set(sample_stems_list)
    md5_to_path = build_md5_to_path(target_md5s)
    print(f"  Resolved {len(md5_to_path)} stems to .sol files")

    # ── Alignment checks ───────────────────────────────────────────────────
    hash_match = 0
    hash_mismatch = 0
    hash_missing = 0
    path_match = 0
    path_mismatch = 0
    path_missing = 0
    file_md5_match = 0
    file_md5_mismatch = 0
    file_md5_unresolvable = 0
    resolved_path_match = 0
    resolved_path_mismatch = 0

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
            skipped += 1
            mismatches.append({"stem": stem, "issue": f"graph load error: {e}"})
            continue

        # Load token
        try:
            t_data = load_token(token_path)
        except Exception as e:
            skipped += 1
            mismatches.append({"stem": stem, "issue": f"token load error: {e}"})
            continue

        g_hash = _get_field(g_data, "contract_hash")
        t_hash = _get_field(t_data, "contract_hash")
        g_path_val = _get_field(g_data, "contract_path")
        t_path_val = _get_field(t_data, "contract_path")

        # Check 1: Hash match between graph and token
        if g_hash is not None and t_hash is not None:
            if str(g_hash) == str(t_hash):
                hash_match += 1
            else:
                hash_mismatch += 1
                mismatches.append({
                    "stem": stem,
                    "issue": f"hash mismatch: graph={g_hash}, token={t_hash}",
                })
        else:
            hash_missing += 1
            mismatches.append({
                "stem": stem,
                "issue": f"missing hash: graph_hash={g_hash}, token_hash={t_hash}",
            })

        # Check 2: Path match between graph and token
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
            path_missing += 1

            # Check 3: Resolve via md5_to_path and compare
            sol_path = find_sol_for_stem(stem, md5_to_path)
            if sol_path is not None:
                resolved_rel = str(sol_path.relative_to(PROJECT_ROOT)) \
                    if sol_path.is_relative_to(PROJECT_ROOT) else str(sol_path)
                g_match = str(g_path_val) == resolved_rel if g_path_val is not None else False
                t_match = str(t_path_val) == resolved_rel if t_path_val is not None else False
                if g_match or t_match:
                    resolved_path_match += 1
                else:
                    resolved_path_mismatch += 1
                    mismatches.append({
                        "stem": stem,
                        "issue": f"resolved path doesn't match: sol={resolved_rel}, "
                                 f"g_path={g_path_val}, t_path={t_path_val}",
                    })

        # Check 4: .sol file MD5 matches filename stem
        sol_path = find_sol_for_stem(stem, md5_to_path)
        if sol_path is not None and sol_path.exists():
            try:
                file_content_md5 = compute_file_md5(sol_path)
                # The filename stem is the path-based MD5 hash (not content MD5)
                # So we verify that get_contract_hash(relative_path) == stem
                rel_path = sol_path.relative_to(PROJECT_ROOT) \
                    if sol_path.is_relative_to(PROJECT_ROOT) else sol_path
                computed_path_hash = hashlib.md5(str(rel_path).encode("utf-8")).hexdigest()
                if computed_path_hash == stem:
                    file_md5_match += 1
                else:
                    file_md5_mismatch += 1
                    mismatches.append({
                        "stem": stem,
                        "issue": f"filename stem != path hash: stem={stem}, "
                                 f"computed={computed_path_hash}, file={rel_path}",
                    })
            except Exception as e:
                file_md5_unresolvable += 1
                mismatches.append({
                    "stem": stem,
                    "issue": f"MD5 computation error: {e}",
                })
        else:
            file_md5_unresolvable += 1

    # ── Decode verification (10 stems) ─────────────────────────────────────
    print("\n  Running decode verification on 10 stems...")
    decode_stems = sample_stems_list[:10]
    decode_results = []
    decode_skipped = 0

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", use_fast=True)
        tokenizer_available = True
    except Exception as e:
        print(f"  WARNING: Could not load CodeBERT tokenizer: {e}")
        tokenizer_available = False

    if tokenizer_available:
        for stem in decode_stems:
            token_path = TOKENS_WINDOWED_DIR / f"{stem}.pt"
            sol_path = find_sol_for_stem(stem, md5_to_path)

            if sol_path is None or not sol_path.exists():
                decode_skipped += 1
                decode_results.append({
                    "stem": stem, "status": "no_sol",
                    "detail": "no .sol file found"
                })
                continue

            try:
                t_data = load_token(token_path)
                input_ids = _get_field(t_data, "input_ids")
                if input_ids is None:
                    decode_skipped += 1
                    decode_results.append({
                        "stem": stem, "status": "no_input_ids",
                        "detail": "no input_ids in token data"
                    })
                    continue

                # Decode first window
                ids_tensor = input_ids[0] if input_ids.dim() == 2 else input_ids
                decoded_text = tokenizer.decode(ids_tensor, skip_special_tokens=True)

                # Read .sol source
                try:
                    sol_text = sol_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    decode_skipped += 1
                    decode_results.append({
                        "stem": stem, "status": "sol_read_error",
                        "detail": str(e)
                    })
                    continue

                # Compare first 200 tokens
                sol_tokens = tokenizer.encode(sol_text, add_special_tokens=False)[:200]
                decoded_tokens = tokenizer.encode(decoded_text, add_special_tokens=False)[:200]

                match_count = sum(1 for a, b in zip(sol_tokens, decoded_tokens) if a == b)
                total = max(len(sol_tokens), len(decoded_tokens), 1)
                match_rate = match_count / total

                # Also check text overlap
                sol_prefix = sol_text[:len(decoded_text) * 2][:500]
                text_overlap = 0
                if decoded_text and sol_prefix:
                    # Count matching characters from start
                    for ci in range(min(len(decoded_text), len(sol_prefix))):
                        if decoded_text[ci] == sol_prefix[ci]:
                            text_overlap += 1
                        else:
                            break
                    text_overlap_rate = text_overlap / max(len(decoded_text), 1)
                else:
                    text_overlap_rate = 0.0

                decode_results.append({
                    "stem": stem,
                    "status": "ok" if match_rate >= 0.90 else "mismatch",
                    "token_match_rate": match_rate,
                    "token_match_count": f"{match_count}/{total}",
                    "text_prefix_overlap": text_overlap_rate,
                    "decoded_len": len(decoded_text),
                    "sol_len": len(sol_text),
                })

            except Exception as e:
                decode_skipped += 1
                decode_results.append({
                    "stem": stem, "status": "error",
                    "detail": str(e)
                })

    # ── Build report ───────────────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Task 24: Token-Graph-Source Alignment Audit\n\n")
    report_lines.append(f"**Paired stems available:** {len(paired)}  \n")
    report_lines.append(f"**Sample size:** {len(sample_stems_list)}  \n")
    report_lines.append(f"**.sol files resolved:** {len(md5_to_path)}/{len(target_md5s)}\n\n")

    # Hash alignment
    report_lines.append("## 1. Hash Alignment (Graph ↔ Token)\n\n")
    total_hash = hash_match + hash_mismatch
    report_lines.append("| Metric | Count | Rate |\n")
    report_lines.append("|--------|-------|------|\n")
    report_lines.append(f"| Hash match | {hash_match} | "
                        f"{hash_match/max(total_hash,1):.1%} |\n")
    report_lines.append(f"| Hash mismatch | {hash_mismatch} | "
                        f"{hash_mismatch/max(total_hash,1):.1%} |\n")
    report_lines.append(f"| Hash missing | {hash_missing} | — |\n\n")

    # Path alignment
    report_lines.append("## 2. Path Alignment (Graph ↔ Token)\n\n")
    total_path = path_match + path_mismatch
    report_lines.append("| Metric | Count | Rate |\n")
    report_lines.append("|--------|-------|------|\n")
    report_lines.append(f"| Path match | {path_match} | "
                        f"{path_match/max(total_path,1):.1%} |\n")
    report_lines.append(f"| Path mismatch | {path_mismatch} | "
                        f"{path_mismatch/max(total_path,1):.1%} |\n")
    report_lines.append(f"| Path missing (one or both) | {path_missing} | — |\n")
    report_lines.append(f"| Resolved via md5_to_path (match) | {resolved_path_match} | — |\n")
    report_lines.append(f"| Resolved via md5_to_path (mismatch) | {resolved_path_mismatch} | — |\n\n")

    # File MD5 verification
    report_lines.append("## 3. Filename Stem ↔ Path Hash Verification\n\n")
    report_lines.append("| Metric | Count | Rate |\n")
    report_lines.append("|--------|-------|------|\n")
    total_md5 = file_md5_match + file_md5_mismatch
    report_lines.append(f"| Stem matches path hash | {file_md5_match} | "
                        f"{file_md5_match/max(total_md5,1):.1%} |\n")
    report_lines.append(f"| Stem mismatches path hash | {file_md5_mismatch} | "
                        f"{file_md5_mismatch/max(total_md5,1):.1%} |\n")
    report_lines.append(f"| Unresolvable | {file_md5_unresolvable} | — |\n\n")

    # Decode verification
    report_lines.append("## 4. Decode Verification (10 stems, first 200 tokens)\n\n")
    if tokenizer_available:
        report_lines.append("| Stem | Status | Token Match Rate | Token Match | Prefix Overlap | Decoded Len | Sol Len |\n")
        report_lines.append("|------|--------|-----------------|-------------|----------------|-------------|--------|\n")
        for dr in decode_results:
            if dr["status"] == "ok" or dr["status"] == "mismatch":
                report_lines.append(
                    f"| {dr['stem'][:12]}... | {dr['status']} | "
                    f"{dr['token_match_rate']:.2%} | {dr['token_match_count']} | "
                    f"{dr['text_prefix_overlap']:.2%} | {dr['decoded_len']} | {dr['sol_len']} |\n"
                )
            else:
                report_lines.append(
                    f"| {dr['stem'][:12]}... | {dr['status']} | — | — | — | — | — |\n"
                )
        report_lines.append("\n")

        ok_count = sum(1 for dr in decode_results if dr.get("status") == "ok")
        report_lines.append(f"**Decode pass rate (≥90% token match):** "
                            f"{ok_count}/{len(decode_results) - decode_skipped}\n\n")
    else:
        report_lines.append("CodeBERT tokenizer not available — decode verification skipped.\n\n")

    # Mismatches detail
    if mismatches:
        report_lines.append("## 5. Mismatches Detail (first 30)\n\n")
        for m in mismatches[:30]:
            report_lines.append(f"- **{m['stem']}**: {m['issue']}\n")
        if len(mismatches) > 30:
            report_lines.append(f"- ... and {len(mismatches) - 30} more\n")
        report_lines.append("\n")

    # Summary
    report_lines.append("## 6. Summary\n\n")
    if total_hash > 0:
        report_lines.append(f"- **Hash alignment rate:** {hash_match/total_hash:.1%}\n")
    if total_path > 0:
        report_lines.append(f"- **Path alignment rate:** {path_match/total_path:.1%}\n")
    if total_md5 > 0:
        report_lines.append(f"- **Stem↔hash verification rate:** {file_md5_match/total_md5:.1%}\n")
    if tokenizer_available and len(decode_results) > decode_skipped:
        ok_count = sum(1 for dr in decode_results if dr.get("status") == "ok")
        report_lines.append(f"- **Decode verification rate:** "
                            f"{ok_count}/{len(decode_results) - decode_skipped}\n")

    if hash_mismatch > 0:
        report_lines.append("\n⚠️ **Hash mismatches detected.** This indicates graph and token ")
        report_lines.append("files are out of sync — some contracts were re-extracted or ")
        report_lines.append("retokenized but not both. Run retokenize_windowed.py to fix.\n")
    if file_md5_mismatch > 0:
        report_lines.append("\n⚠️ **Filename stem mismatches.** Some .pt files may have been ")
        report_lines.append("renamed or moved since extraction. Verify source directory layout.\n")

    report_content = "".join(report_lines)
    save_report("task24_token_graph_source_alignment", report_content)
    print_header("Task 24 Complete")


if __name__ == "__main__":
    main()
