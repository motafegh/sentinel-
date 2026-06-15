"""Verify 2-tool agreement on the contracts we manually validated.
For each: show what slither AND aderyn each detected, compare to source."""
import json
from pathlib import Path
import re

BCCC_ME = Path('/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/MishandledException')
results = json.loads(Path('/tmp/bccc_me_full_results.json').read_text())

# 5 deep-verified from 60-sample audit
highconf_60sample = [
    '6eed6e45813fe3f6554d01f5aeabe1fa7f1b1a17b1c59c0cc84376e4fba50f3c',
    '61a5c2f7061ee8094712263d5be38a43389a48132c69f16aad1d9dfc6faf0b97',
    '3a43dd0a6320399fc8f01386e2648cbaf59c18261ea33299d09bdb6afcfb12bf',
    'df41b42bbb576a91e2c4616112d2ae88c2ecab8ebb22d0074d31e85266aeb1a5',
    'efdbdd42a393b133a88e5ec8b5e3a686359975bb1aac3bf23bdeee30ccafc3e9',
]

# Read the 29 real positives from the deep audit
real_positives_log = []
audit_log = Path('/tmp/bccc_me_aderyn_retry.log').read_text().splitlines()
for line in audit_log:
    if 'UNCHECKED lines' in line or 'unchecked lines' in line.lower():
        continue  # skip status lines
# Better: just use the deep_audit_139.py output via re-running
# Actually, let me reconstruct from the saved results

# Get all 139 STRONG-only contracts and find those with real unchecked patterns
ADERYN_ME_STRONG = {'unchecked-return', 'unchecked-send'}
WEAK = {'arbitrary-transfer-from', 'incorrect-erc20-interface', 'unsafe-erc20-operation'}
SLITHER_ME = {'unchecked-transfer', 'unchecked-send', 'unchecked-lowlevel',
              'arbitrary-send-erc20', 'arbitrary-send-eth', 'void-cst', 'incorrect-exp'}

strong_only = []
for bid, v in results.items():
    ar_dets = set(v.get('aderyn_detectors', []))
    if (ar_dets & ADERYN_ME_STRONG) and not (ar_dets & WEAK):
        sl_ok = v.get('slither_status') == 'OK'
        sl_hit = sl_ok and any(d in SLITHER_ME for d in v.get('slither_detectors', []))
        if not sl_hit:
            strong_only.append((bid, v))

# Re-scan for actual unchecked patterns
patterns = [
    (r'\.transfer\s*\(', 'transfer() call'),
    (r'\.send\s*\(', 'send() call'),
    (r'\.call\s*\{\s*value', '.call{value} call'),
    (r'\.call\s*\(', '.call() call'),
    (r'\.delegatecall\s*\(', 'delegatecall()'),
    (r'\.transferFrom\s*\(', 'transferFrom() call'),
]

def find_unchecked_lines(bid):
    src = BCCC_ME / f'{bid}.sol'
    if not src.exists():
        return []
    content = src.read_text(encoding='utf-8', errors='replace')
    lines = content.splitlines()
    unchecked = []
    for line_num, line in enumerate(lines, 1):
        for pat, name in patterns:
            if re.search(pat, line):
                stripped = line.strip()
                is_checked = ('require' in stripped or 'assert' in stripped or
                              re.search(r'\bif\s*\(', stripped) or
                              'bool success' in stripped or 'success =' in stripped or
                              '==' in stripped)
                if not is_checked and not stripped.startswith('//') and '///' not in stripped[:5]:
                    unchecked.append((line_num, name, stripped))
                break
    return unchecked

# Find the 29 real positives
real_positives_bids = []
for bid, v in strong_only:
    unchecked = find_unchecked_lines(bid)
    if unchecked:
        real_positives_bids.append(bid)

print(f'Real positives from Tier 3 audit: {len(real_positives_bids)}')
print()

# Now: 2-tool agreement analysis
def show_2tool(bid, label):
    v = results.get(bid, {})
    print(f'\n--- {label}: {bid[:16]}... ---')
    print(f'  Slither: status={v.get("slither_status")}, detectors={v.get("slither_detectors", [])}')
    print(f'  Aderyn:  status={v.get("aderyn_status")}, detectors={v.get("aderyn_detectors", [])}')
    # Show only ME-specific detectors
    sl_me = [d for d in v.get('slither_detectors', []) if d in SLITHER_ME]
    ar_me = [d for d in v.get('aderyn_detectors', []) if d in ADERYN_ME_STRONG or d in WEAK]
    print(f'  Slither ME detectors: {sl_me}')
    print(f'  Aderyn ME detectors:  {ar_me}')

    # Find the source lines
    unchecked = find_unchecked_lines(bid)
    if unchecked:
        print(f'  UNCHECKED LINES in source:')
        for ln, name, text in unchecked[:3]:
            print(f'    L{ln:4d}: {text[:90]}  ← {name}')
    return v

# Section 1: 5 deep-verified from 60-sample audit
print('='*80)
print('SECTION 1: 5 DEEP-VERIFIED contracts (60-sample audit)')
print('='*80)
for i, bid in enumerate(highconf_60sample, 1):
    show_2tool(bid, f'[{i}]')

# Section 2: 29 real positives from Tier 3 deep audit
print()
print('='*80)
print('SECTION 2: 29 REAL POSITIVES (Tier 3 deep audit, aderyn STRONG-only)')
print('='*80)
for i, bid in enumerate(real_positives_bids, 1):
    show_2tool(bid, f'[{i}]')

# Summary
print()
print('='*80)
print('SUMMARY: 2-tool agreement on the manually-validated contracts')
print('='*80)

def both_agree(bid):
    v = results.get(bid, {})
    sl_me = [d for d in v.get('slither_detectors', []) if d in SLITHER_ME]
    ar_me = [d for d in v.get('aderyn_detectors', []) if d in ADERYN_ME_STRONG or d in WEAK]
    return bool(sl_me) and bool(ar_me)

def slither_only(bid):
    v = results.get(bid, {})
    sl_me = [d for d in v.get('slither_detectors', []) if d in SLITHER_ME]
    return bool(sl_me)

def aderyn_only(bid):
    v = results.get(bid, {})
    ar_me = [d for d in v.get('aderyn_detectors', []) if d in ADERYN_ME_STRONG or d in WEAK]
    return bool(ar_me) and not slither_only(bid)

def neither_known(bid):
    v = results.get(bid, {})
    sl_me = [d for d in v.get('slither_detectors', []) if d in SLITHER_ME]
    ar_me = [d for d in v.get('aderyn_detectors', []) if d in ADERYN_ME_STRONG or d in WEAK]
    return not sl_me and not ar_me

both60 = sum(1 for b in highconf_60sample if both_agree(b))
sl60 = sum(1 for b in highconf_60sample if slither_only(b))
ar60 = sum(1 for b in highconf_60sample if aderyn_only(b))
ne60 = sum(1 for b in highconf_60sample if neither_known(b))

print()
print('5 deep-verified contracts (60-sample audit):')
print(f'  Both tools see ME: {both60}/5')
print(f'  Slither only:      {sl60}/5')
print(f'  Aderyn only:       {ar60}/5')
print(f'  Neither:           {ne60}/5')

bothRP = sum(1 for b in real_positives_bids if both_agree(b))
slRP = sum(1 for b in real_positives_bids if slither_only(b))
arRP = sum(1 for b in real_positives_bids if aderyn_only(b))
neRP = sum(1 for b in real_positives_bids if neither_known(b))

print()
print(f'29 real positives (Tier 3 deep audit):')
print(f'  Both tools see ME: {bothRP}/29')
print(f'  Slither only:      {slRP}/29')
print(f'  Aderyn only:       {arRP}/29')
print(f'  Neither:           {neRP}/29')

# What does the 29 mean in context?
print()
print('Interpretation:')
print(f'  Of the 5 deep-verified, {both60+sl60} are already in Tier 1+2 (slither-detected)')
print(f'  Of the 29 real positives, how many are also in Tier 1+2?')
# Check overlap
overlap = set(real_positives_bids) & set(json.loads(Path('/tmp/bccc_me_extraction_final.json').read_text()).get('slither_me_hits', []))
print(f'  Real positives also in slither-only set: {len(overlap)}/29')
print(f'  Real positives NOT in slither-only set: {29 - len(overlap)}/29 (truly aderyn-only)')
