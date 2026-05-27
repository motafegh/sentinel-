"""
SECTION 2: Token integrity (sample 1000 random tokens)
Token files are dicts with 'input_ids' [4,512] and 'attention_mask' [4,512]
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import random
from collections import Counter

TOKENS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/tokens_windowed")
VOCAB_SIZE = 50265
# CodeBERT uses RoBERTa tokenizer: CLS=0, SEP=2, PAD=1
# But also check for BERT-style: CLS=101, SEP=102, PAD=0
CLS_ROBERTA = 0
SEP_ROBERTA = 2
PAD_ROBERTA = 1
CLS_BERT = 101
SEP_BERT = 102
PAD_BERT = 0

SAMPLE_N = 1000
SEED = 42

def main():
    token_files = sorted(TOKENS_DIR.glob("*.pt"))
    total = len(token_files)
    print(f"Found {total} token files")

    rng = random.Random(SEED)
    sample = rng.sample(token_files, min(SAMPLE_N, total))

    wrong_shape = 0
    wrong_dtype = 0
    oor_vocab = 0
    fully_padded_windows = 0
    total_windows = 0
    real_tokens_per_sample = []
    errors = 0
    shape_samples = []
    schema_versions = Counter()
    num_windows_dist = Counter()

    # CLS detection
    cls_at_pos0_roberta = 0
    cls_at_pos0_bert = 0
    missing_sep_roberta = 0
    missing_sep_bert = 0

    for fpath in sample:
        try:
            t = torch.load(fpath, weights_only=True)
        except Exception as e:
            print(f"ERROR loading {fpath.name}: {e}")
            errors += 1
            continue

        if isinstance(t, dict):
            schema_ver = t.get('feature_schema_version', 'unknown')
            schema_versions[schema_ver] += 1
            nw = t.get('num_windows', 4)
            num_windows_dist[nw] += 1
            input_ids = t.get('input_ids')
            if input_ids is None:
                errors += 1
                continue
        elif isinstance(t, torch.Tensor):
            input_ids = t
            schema_ver = 'tensor'
            schema_versions['raw_tensor'] += 1
        else:
            errors += 1
            continue

        # Shape check
        if input_ids.shape != torch.Size([4, 512]):
            wrong_shape += 1
            shape_samples.append((fpath.name, list(input_ids.shape)))

        # Dtype check
        if input_ids.dtype not in (torch.int64, torch.int32, torch.long):
            wrong_dtype += 1

        # Vocab range
        if (input_ids < 0).any() or (input_ids >= VOCAB_SIZE).any():
            oor_vocab += 1

        W = input_ids.shape[0]
        total_windows += W

        real_in_sample = 0
        for w in range(W):
            window = input_ids[w]  # [512]

            # CLS detection (RoBERTa vs BERT style)
            first_tok = window[0].item()
            if first_tok == CLS_ROBERTA:
                cls_at_pos0_roberta += 1
            elif first_tok == CLS_BERT:
                cls_at_pos0_bert += 1

            # SEP check
            if SEP_ROBERTA not in window and SEP_BERT not in window:
                # Neither SEP found — could be pad-only window
                pass
            if SEP_ROBERTA not in window:
                missing_sep_roberta += 1
            if SEP_BERT not in window:
                missing_sep_bert += 1

            # Fully padded window (PAD=1 for RoBERTa)
            # Check if window is all-pad after first token
            non_pad_roberta = (window != PAD_ROBERTA).sum().item()
            non_pad_bert = (window != PAD_BERT).sum().item()
            non_pad = min(non_pad_roberta, non_pad_bert)

            if non_pad <= 1:  # Only CLS or nothing
                fully_padded_windows += 1

            real_in_sample += max(non_pad_roberta, non_pad_bert)

        real_tokens_per_sample.append(real_in_sample)

    print(f"\nSampled: {len(sample)} files ({errors} errors)")
    print("=" * 70)
    print("SECTION 2: TOKEN INTEGRITY")
    print("=" * 70)

    total_checked = len(sample) - errors
    print(f"\n--- Shape and dtype ---")
    print(f"  Wrong shape (not [4,512]):     {wrong_shape:,}/{total_checked} ({'FAIL' if wrong_shape > 0 else 'PASS'})")
    if wrong_shape > 0 and shape_samples:
        for nm, sh in shape_samples[:5]:
            print(f"    {nm}: {sh}")
    print(f"  Wrong dtype (not int32/64):    {wrong_dtype:,}/{total_checked} ({'FAIL' if wrong_dtype > 0 else 'PASS'})")

    print(f"\n--- Schema version distribution ---")
    for sv, cnt in sorted(schema_versions.items()):
        status = "WARN" if sv != 'v5' else "PASS"
        print(f"  {sv}: {cnt} ({100.*cnt/total_checked:.1f}%) {status}")

    print(f"\n--- Num windows distribution ---")
    for nw in sorted(num_windows_dist.keys()):
        print(f"  {nw} windows: {num_windows_dist[nw]:,}")

    print(f"\n--- Vocab range [0, {VOCAB_SIZE}) ---")
    print(f"  Out-of-range token IDs:        {oor_vocab:,}/{total_checked} ({'FAIL' if oor_vocab > 0 else 'PASS'})")

    print(f"\n--- CLS token detection (RoBERTa=0, BERT=101) ---")
    print(f"  Windows with CLS=0 at pos 0:   {cls_at_pos0_roberta:,}/{total_windows:,}")
    print(f"  Windows with CLS=101 at pos 0: {cls_at_pos0_bert:,}/{total_windows:,}")
    neither_cls = total_windows - cls_at_pos0_roberta - cls_at_pos0_bert
    print(f"  Windows with neither CLS:      {neither_cls:,}/{total_windows:,} ({'WARN' if neither_cls > total_windows*0.01 else 'PASS'})")

    print(f"\n--- SEP token check ---")
    print(f"  Windows missing SEP=2(RoBERTa): {missing_sep_roberta:,}/{total_windows:,}")
    print(f"  Windows missing SEP=102(BERT):  {missing_sep_bert:,}/{total_windows:,}")

    print(f"\n--- Padding analysis ---")
    pad_pct = 100.0 * fully_padded_windows / max(total_windows, 1)
    print(f"  Fully padded windows:          {fully_padded_windows:,}/{total_windows:,} ({pad_pct:.1f}%) ({'WARN' if pad_pct > 40 else 'PASS'})")
    rtp = np.array(real_tokens_per_sample)
    print(f"  Real tokens/sample — min:{rtp.min()}, max:{rtp.max()}, mean:{rtp.mean():.1f}, p50:{np.percentile(rtp,50):.0f}")

if __name__ == "__main__":
    main()
