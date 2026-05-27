"""Task 12: Token Integrity Checks"""
import sys
import random
import torch
from pathlib import Path
import json

PROJECT_ROOT = Path('/home/motafeq/projects/sentinel')
TOKENS_DIR = PROJECT_ROOT / 'ml/data/tokens_windowed'

CODEBERT_VOCAB_SIZE = 50265
CODEBERT_PAD_ID = 1
MAX_WINDOWS = 4

# Sample 50 random token files
all_files = [p for p in TOKENS_DIR.glob('*.pt')
             if p.name not in {'checkpoint.json', 'failed_contracts.json'}]
random.seed(123)
sample_files = random.sample(all_files, min(50, len(all_files)))
print(f"Sampled {len(sample_files)} token files")

violations = {
    'input_ids_shape': 0,
    'attn_mask_shape': 0,
    'pad_window_nonzero_mask': 0,
    'real_window_no_leading_1': 0,
    'input_ids_vocab_range': 0,
    'num_tokens_mismatch': 0,
    'input_ids_nan': 0,
    'pad_window_non_pad_token': 0,
    'schema_version': 0,
}

details = []
errors = []

for fpath in sample_files:
    try:
        t = torch.load(fpath, weights_only=False)
        row = {'file': fpath.name, 'issues': []}

        input_ids = getattr(t, 'input_ids', None)
        attn_mask = getattr(t, 'attention_mask', None)
        num_windows = getattr(t, 'num_windows', None)
        num_tokens = getattr(t, 'num_tokens', None)
        schema_ver = getattr(t, 'feature_schema_version', None)

        # Check 1: input_ids shape
        if input_ids is None or input_ids.shape != torch.Size([MAX_WINDOWS, 512]):
            violations['input_ids_shape'] += 1
            row['issues'].append(f"input_ids_shape={input_ids.shape if input_ids is not None else None}")

        # Check 2: attention_mask shape
        if attn_mask is None or attn_mask.shape != torch.Size([MAX_WINDOWS, 512]):
            violations['attn_mask_shape'] += 1
            row['issues'].append(f"attn_mask_shape={attn_mask.shape if attn_mask is not None else None}")

        if input_ids is not None and attn_mask is not None and num_windows is not None:
            for w in range(MAX_WINDOWS):
                if w >= num_windows:
                    # Padding window
                    # Check 3: attn mask should be all 0
                    if attn_mask[w].sum().item() != 0:
                        violations['pad_window_nonzero_mask'] += 1
                        row['issues'].append(f"pad_window_{w}_nonzero_mask={attn_mask[w].sum().item()}")
                    # Check 8: input_ids should be all pad
                    non_pad = (input_ids[w] != CODEBERT_PAD_ID).sum().item()
                    if non_pad > 0:
                        violations['pad_window_non_pad_token'] += 1
                        row['issues'].append(f"pad_window_{w}_non_pad_tokens={non_pad}")
                else:
                    # Real window
                    # Check 4: attention_mask[w, 0] == 1
                    if attn_mask[w, 0].item() != 1:
                        violations['real_window_no_leading_1'] += 1
                        row['issues'].append(f"real_window_{w}_mask[0]={attn_mask[w,0].item()}")

            # Check 5: vocab range
            out_of_vocab = ((input_ids < 0) | (input_ids >= CODEBERT_VOCAB_SIZE)).sum().item()
            if out_of_vocab > 0:
                violations['input_ids_vocab_range'] += 1
                row['issues'].append(f"out_of_vocab={out_of_vocab}")

            # Check 6: num_tokens == attn_mask.sum()
            if num_tokens is not None:
                mask_sum = attn_mask.sum().item()
                if abs(num_tokens - mask_sum) > 1:  # allow off-by-one
                    violations['num_tokens_mismatch'] += 1
                    row['issues'].append(f"num_tokens={num_tokens} vs mask_sum={mask_sum}")

            # Check 7: NaN in input_ids
            if torch.isnan(input_ids.float()).any():
                violations['input_ids_nan'] += 1
                row['issues'].append("input_ids_has_nan")

        # Check 9: schema version
        if schema_ver != 'v4':
            violations['schema_version'] += 1
            row['issues'].append(f"schema_version={schema_ver}")

        if row['issues']:
            details.append(row)

    except Exception as e:
        errors.append(f"{fpath.name}: {e}")

print(f"\nLoad errors: {len(errors)}")
for e in errors[:5]:
    print(f"  {e}")

print(f"\n{'='*70}")
print("TOKEN INTEGRITY CHECK RESULTS")
print(f"{'='*70}")
print(f"{'Check':<40} {'Violations':>12} {'/ 50':>8}")
print('-'*60)
for k, v in violations.items():
    status = "BUG" if v > 0 else "OK"
    print(f"{k:<40} {v:>12} {status:>8}")

if details:
    print(f"\nFiles with violations ({len(details)}):")
    for d in details[:10]:
        print(f"  {d['file']}: {'; '.join(d['issues'])}")
    if len(details) > 10:
        print(f"  ... and {len(details)-10} more")
else:
    print("\nNo violations found in sampled files.")
