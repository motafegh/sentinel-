"""Task 14: Window Sub-sampling Vulnerability Coverage"""
import sys
import re
import torch
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path('/home/motafeq/projects/sentinel')
CSV_PATH = PROJECT_ROOT / 'ml/data/processed/multilabel_index_deduped.csv'
TOKENS_DIR = PROJECT_ROOT / 'ml/data/tokens_windowed'

# Load CSV for label lookup
df = pd.read_csv(CSV_PATH)
label_cols = ['CallToUnknown', 'DenialOfService', 'ExternalBug', 'GasException',
              'IntegerUO', 'MishandledException', 'Reentrancy', 'Timestamp', 'TOD', 'UnusedReturn']
# Only use label columns that exist
label_cols = [c for c in label_cols if c in df.columns]
md5_to_labels = {}
for _, row in df.iterrows():
    labels = [c for c in label_cols if row.get(c, 0) == 1]
    md5_to_labels[row['md5_stem']] = labels

PATTERNS = {
    'Reentrancy': re.compile(r'\.call\{?|\.call\.value', re.IGNORECASE),
    'DenialOfService': re.compile(r'for\s*\(', re.IGNORECASE),
    'Timestamp': re.compile(r'block\.timestamp|block\.number|\bnow\b'),
    'IntegerUO': re.compile(r'SafeMath|unchecked', re.IGNORECASE),
    'MishandledException': re.compile(r'\.call\(|\.send\(', re.IGNORECASE),
    'UnusedReturn': re.compile(r'\.call\(|\.send\(', re.IGNORECASE),
    'GasException': re.compile(r'gasleft\(\)|msg\.gas', re.IGNORECASE),
    'CallToUnknown': re.compile(r'\.call\(', re.IGNORECASE),
    'ExternalBug': re.compile(r'\.call\(|\.delegatecall\(|\.staticcall\(', re.IGNORECASE),
    'TOD': re.compile(r'msg\.sender\.transfer|\.transfer\(|msg\.value', re.IGNORECASE),
}

# Approximate: each token covers ~4 chars; stride=256 tokens; window=512 tokens
# chars_per_token ≈ 4
CHARS_PER_TOKEN = 4
WINDOW_SIZE_TOKENS = 512
STRIDE_TOKENS = 256
WINDOW_SIZE_CHARS = WINDOW_SIZE_TOKENS * CHARS_PER_TOKEN  # 2048
STRIDE_CHARS = STRIDE_TOKENS * CHARS_PER_TOKEN  # 1024

# Find contracts where num_windows==4 and were likely sub-sampled
# A contract was sub-sampled if raw_windows > 4, i.e. source > 4*2048 = 8192 chars approximately
# or if num_windows attribute == 4 and we can verify source is long

found = []
token_files = list(TOKENS_DIR.glob('*.pt'))

import random
random.seed(99)
random.shuffle(token_files)

for fpath in token_files:
    if len(found) >= 10:
        break
    try:
        t = torch.load(fpath, weights_only=False)
        num_windows = getattr(t, 'num_windows', None)
        if num_windows != 4:
            continue

        contract_path = getattr(t, 'contract_path', None)
        contract_hash = getattr(t, 'contract_hash', None)
        if not contract_path:
            continue

        sol_path = Path(contract_path)
        if not sol_path.exists():
            continue

        source = sol_path.read_text(encoding='utf-8', errors='replace')
        # Estimate raw windows needed: if source chars > 4 * STRIDE_CHARS + WINDOW_SIZE_CHARS
        # i.e., linspace would pick 4 start positions from a larger range
        # Rough: if total_tokens = len(source)/CHARS_PER_TOKEN > 4*STRIDE_TOKENS + WINDOW_SIZE_TOKENS
        est_tokens = len(source) / CHARS_PER_TOKEN
        # With linspace(0, total_tokens - window_size, num_windows_raw), if raw > 4, linspace subsamples
        # raw_windows = max(1, ceil((est_tokens - 512) / 256) + 1)
        import math
        raw_windows = max(1, math.ceil((est_tokens - WINDOW_SIZE_TOKENS) / STRIDE_TOKENS) + 1)
        if raw_windows <= 4:
            continue  # not sub-sampled

        found.append({
            'fpath': fpath,
            't': t,
            'sol_path': sol_path,
            'source': source,
            'contract_hash': contract_hash,
            'num_windows': num_windows,
            'raw_windows': raw_windows,
            'est_tokens': int(est_tokens),
        })
    except Exception:
        continue

print(f"Found {len(found)} sub-sampled contracts with num_windows=4")
print()

if not found:
    print("No sub-sampled contracts found. Trying just num_windows=4 with long source:")
    for fpath in token_files[:200]:
        if len(found) >= 10:
            break
        try:
            t = torch.load(fpath, weights_only=False)
            num_windows = getattr(t, 'num_windows', None)
            if num_windows != 4:
                continue
            contract_path = getattr(t, 'contract_path', None)
            if not contract_path:
                continue
            sol_path = Path(contract_path)
            if not sol_path.exists():
                continue
            source = sol_path.read_text(encoding='utf-8', errors='replace')
            contract_hash = getattr(t, 'contract_hash', None)
            found.append({
                'fpath': fpath,
                't': t,
                'sol_path': sol_path,
                'source': source,
                'contract_hash': contract_hash,
                'num_windows': num_windows,
                'raw_windows': 0,
                'est_tokens': int(len(source)/4),
            })
        except Exception:
            continue

print(f"{'='*100}")
print(f"{'Contract':<36} {'Class':<22} {'Pattern_Line':>12} {'Raw_W':>6} {'Covered?':>10}")
print('-'*100)

for item in found:
    contract_hash = item['contract_hash']
    labels = md5_to_labels.get(contract_hash, [])
    source = item['source']
    lines = source.split('\n')

    # Pick primary label for pattern matching
    primary = None
    pattern = None
    for lbl in labels:
        if lbl in PATTERNS:
            primary = lbl
            pattern = PATTERNS[lbl]
            break
    if primary is None and labels:
        primary = labels[0]
        pattern = None

    if pattern is None:
        print(f"{item['fpath'].stem:<36} {str(labels):<22} {'no_pattern':>12} {item['raw_windows']:>6} {'N/A':>10}")
        continue

    # Find line numbers where pattern appears
    matching_lines = []
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            matching_lines.append(i)

    if not matching_lines:
        print(f"{item['fpath'].stem:<36} {str(primary):<22} {'not_found':>12} {item['raw_windows']:>6} {'N/A':>10}")
        continue

    # Compute which char ranges the 4 windows cover
    # Window w covers chars: start_w to start_w + WINDOW_SIZE_CHARS
    # With linspace(0, raw_windows-1, 4) as the selected window indices in the original linspace
    # Actually: linspace(0, max_start_token, num_windows) start positions
    # max_start_token = est_tokens - WINDOW_SIZE_TOKENS
    est_tokens = item['est_tokens']
    max_start_token = max(0, est_tokens - WINDOW_SIZE_TOKENS)
    # 4 windows picked via linspace
    import numpy as np
    if item['raw_windows'] > 4:
        start_tokens = np.linspace(0, max_start_token, 4).astype(int)
    else:
        # No subsampling
        start_tokens = [w * STRIDE_TOKENS for w in range(4)]

    covered_ranges = [(int(st * CHARS_PER_TOKEN), int((st + WINDOW_SIZE_TOKENS) * CHARS_PER_TOKEN))
                      for st in start_tokens]

    # Find char offset of each matching line
    char_offsets = []
    offset = 0
    line_starts = {}
    for i, line in enumerate(lines, 1):
        line_starts[i] = offset
        offset += len(line) + 1  # +1 for newline

    for ln in matching_lines[:3]:
        char_start = line_starts.get(ln, -1)
        covered = any(r[0] <= char_start <= r[1] for r in covered_ranges)
        verdict = "COVERED" if covered else "MISSED"
        print(f"{item['fpath'].stem:<36} {str(primary):<22} {ln:>12} {item['raw_windows']:>6} {verdict:>10}")
        if verdict == "MISSED":
            print(f"  -> Char offset {char_start}, windows cover: {covered_ranges}")
