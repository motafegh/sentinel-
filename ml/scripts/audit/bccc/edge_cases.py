"""Investigate the EDGE CASES for MishandledException:
- 1 aderyn-only case (slither missed it)
- 2 neither case (both tools missed it - false positive?)
- 4 failed cases (compilation failed)
Then build the final extraction list."""
import json
import re
from pathlib import Path

BCCC_ROOT = Path('/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes')

consensus = json.loads(Path('/tmp/bccc_2tool_consensus.json').read_text())
slither_results = json.loads(Path('/tmp/bccc_slither_results.json').read_text())
aderyn_results = json.loads(Path('/tmp/bccc_aderyn_per_file.json').read_text())

# Categorize the 12 MishandledException samples
print('='*80)
print('MishandledException: ALL 12 cases classified')
print('='*80)
print()

# Show each
for bid in json.loads(Path('/tmp/bccc_folder_ids.json').read_text())['MishandledException']:
    k = f'MishandledException|{bid}'
    c = consensus.get(k, {})
    sl = slither_results.get(k, {})
    ar = aderyn_results.get(k, {})
    src = BCCC_ROOT / 'MishandledException' / f'{bid}.sol'
    if not src.exists():
        continue
    content = src.read_text(encoding='utf-8', errors='replace')
    lines = content.splitlines()

    # Categorize
    if c.get('slither_failed') or c.get('aderyn_failed'):
        cat = '✗ FAILED'
    elif c.get('both'):
        cat = '✓✓ BOTH (HIGH CONF)'
    elif c.get('slither_hit'):
        cat = '✓ SLITHER-ONLY'
    elif c.get('aderyn_hit'):
        cat = '✓ ADERYN-ONLY'
    else:
        cat = '✗✗ NEITHER (likely FP)'

    print(f'>>> {cat}: {bid[:16]}... ({len(content):,} B)')
    print(f'    Slither status: {sl.get("status", "?")}, detectors: {sl.get("detectors", [])[:5]}')
    print(f'    Aderyn status: {ar.get("aderyn_status", "?")}, detectors: {ar.get("detectors", [])[:5]}')

    if cat == '✗ FAILED':
        err = sl.get('stderr', '')[:120] or ar.get('aderyn_error', '')[:120]
        print(f'    Error: {err}')

    if cat == '✗✗ NEITHER (likely FP)':
        # Look for ANY unchecked call to see if the contract has a real but different vuln
        patterns = [
            (r'\.transfer\(', 'transfer()'),
            (r'\.send\(', 'send()'),
            (r'\.call\{value:', '.call{value:}'),
            (r'\.call\(', '.call()'),
            (r'\.transferFrom\(', 'transferFrom()'),
        ]
        for line_num, line in enumerate(lines, 1):
            for pat, name in patterns:
                if re.search(pat, line):
                    stripped = line.strip()
                    is_checked = 'require' in stripped or 'assert' in stripped or 'if ' in stripped or '==' in stripped
                    if not is_checked:
                        print(f'    L{line_num}: {stripped[:90]}  ← UNCHECKED {name}')
                    break

    if cat == '✓ ADERYN-ONLY':
        # What did aderyn find that slither didn't?
        adet = ar.get('detectors', [])
        aclass = [d for d in adet if d in ['unchecked-return', 'unchecked-send', 'arbitrary-transfer-from', 'incorrect-erc20-interface', 'unsafe-erc20-operation']]
        print(f'    Aderyn ME detectors: {aclass}')

    print()

# Now: build the final extraction list
print('='*80)
print('FINAL EXTRACTION LIST')
print('='*80)
final_list = []
for k, c in consensus.items():
    if not k.startswith('MishandledException|'):
        continue
    if c.get('both'):
        bid = k.split('|')[1]
        final_list.append((bid, 'BOTH'))
    elif c.get('aderyn_hit'):
        bid = k.split('|')[1]
        final_list.append((bid, 'ADERYN-ONLY'))

print(f'Total in audit: 12 BCCC MishandledException contracts sampled')
print(f'Both tools agree: 5 (HIGH CONFIDENCE)')
print(f'Aderyn only: 1 (slither missed)')
print(f'Neither: 2 (likely false positives)')
print(f'Failed: 4 (compilation issues)')
print()
print(f'Estimated high-confidence extraction: ~2,147 contracts (42% of 5,154 BCCC ME folder)')
print(f'  → v3 ME goes from 39 → ~2,200 (56x increase)')
print()

# Save the high-confidence contract list
ids = [bid for bid, status in final_list if status == 'BOTH']
print(f'High-confidence contract IDs (5):')
for bid in ids:
    print(f'  {bid}')

# Save to file for potential extraction later
Path('/tmp/bccc_me_highconf.txt').write_text('\n'.join(ids))
print()
print('Saved to /tmp/bccc_me_highconf.txt')

# Also: how many contracts would be filterable from full BCCC ME folder?
# - 5,154 total in folder
# - 42% confirmed by both = ~2,164
# - 50% have aderyn at least = ~2,577
# - 33% (4/12) failed to compile = 1,701 unknown
# After strict filter, ~2,000 confirmed
print()
print('='*80)
print('RECOMMENDED EXTRACTION STRATEGY')
print('='*80)
print()
print('Step 1: Run slither + aderyn on ALL 5,154 BCCC ME contracts (1-2 days)')
print('Step 2: Filter to contracts where BOTH tools agree = ~2,000 confirmed')
print('Step 3: Cross-corpus dedup against v3 (graph-hash dedup, expected <5% overlap)')
print('Step 4: Compile probe with our solc versions (~80% expected to pass)')
print('Step 5: Inject ~1,500-2,000 contracts into v3')
print('Step 6: Re-export + re-split + re-train (Run 13)')
print('Step 7: Verify ME F1 is meaningful (expect 0.5-0.7, NOT 1.0 overfit)')
print()
print('Expected: v3 ME 39 → 1,500-2,000 contracts, F1 becomes reliable')
