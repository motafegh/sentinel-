"""Deep-dive helper - uses regex"""
import json
import re
from pathlib import Path

BCCC_ROOT = Path('/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes')

consensus = json.loads(Path('/tmp/bccc_2tool_consensus.json').read_text())

high_conf_me = []
for k, v in consensus.items():
    if k.startswith('MishandledException|') and v.get('both'):
        bid = k.split('|')[1]
        high_conf_me.append((bid, v))

print('='*80)
print('DEEP-DIVE: 5 HIGH-CONFIDENCE MishandledException contracts')
print('='*80)
print(f'Found {len(high_conf_me)} contracts to deep-dive')
print()

for i, (bid, c) in enumerate(high_conf_me, 1):
    src = BCCC_ROOT / 'MishandledException' / f'{bid}.sol'
    if not src.exists():
        print(f'  [{i}] {bid[:16]}... NOT FOUND')
        continue
    content = src.read_text(encoding='utf-8', errors='replace')
    lines = content.splitlines()
    print(f'>>> [{i}] Contract: {bid}')
    print(f'    Size: {len(content):,} bytes, {len(lines):,} lines')
    print(f'    Slither: {c["slither_detectors"]}')
    print(f'    Aderyn:  {c["aderyn_detectors"]}')
    print()

    # Look for unchecked patterns
    print(f'    Lines with potential unchecked calls:')
    patterns = [
        (r'\.transfer\(', 'transfer() call'),
        (r'\.send\(', 'send() call'),
        (r'\.call\{value:', '.call{value:} call'),
        (r'\.call\(', '.call() call'),
        (r'\.delegatecall\(', 'delegatecall'),
        (r'\.transferFrom\(', 'transferFrom() call'),
    ]
    found_unchecked = 0
    for line_num, line in enumerate(lines, 1):
        for pat, name in patterns:
            if re.search(pat, line):
                stripped = line.strip()
                is_checked = 'require' in stripped or 'assert' in stripped or 'if ' in stripped or '==' in stripped or '!= ' in stripped
                marker = '  (CHECKED)' if is_checked else '  *** UNCHECKED ***'
                print(f'    L{line_num:3d}: {stripped[:90]}{marker}')
                if not is_checked:
                    found_unchecked += 1
                break
    if found_unchecked == 0:
        print(f'    (no obvious unchecked calls in pattern list — slither/aderyn found other patterns)')

    # Show slither's exact finding (from raw output, but we have detectors list)
    # Read first 30 lines
    print()
    print(f'    Source preview (first 30 lines):')
    for line_num, line in enumerate(lines[:30], 1):
        print(f'    L{line_num:3d}: {line[:100]}')
    print()
    print('-'*80)
