# ml/scripts/analyze_token_stats.py

import os
import torch
from pathlib import Path
from collections import Counter

# Point this at your actual tokens folder
TOKENS_DIR = Path("ml/data/tokens")
lengths = []  # will store one integer per token file

token_files = list(TOKENS_DIR.glob("*.pt"))
print(f"Found {len(token_files)} token files")

for i, fpath in enumerate(token_files):
    token_dict = torch.load(fpath, weights_only=True)
    
    # input_ids shape is [1, 512] — the 512 is padded/truncated
    # attention_mask tells us the REAL length (1=real token, 0=padding)
    real_length = token_dict["attention_mask"].sum().item()
    lengths.append(int(real_length))
    
    if i % 10000 == 0:
        print(f"  Scanned {i}/{len(token_files)}...")
import statistics

total = len(lengths)
truncated = sum(1 for l in lengths if l == 512)
not_truncated = total - truncated

print(f"\n=== TOKEN LENGTH STATS ===")
print(f"Total files scanned:   {total:,}")
print(f"Truncated (len==512):  {truncated:,}  ({100*truncated/total:.1f}%)")
print(f"Not truncated:         {not_truncated:,}  ({100*not_truncated/total:.1f}%)")
print(f"\nOf non-truncated contracts:")
short = [l for l in lengths if l < 512]
if short:
    print(f"  Median length: {statistics.median(short):.0f} tokens")
    print(f"  Mean length:   {statistics.mean(short):.0f} tokens")
    print(f"  Min length:    {min(short)} tokens")
