"""Post-audit analysis: process the full ME audit results, compute consensus,
dedup against v3, build the injection list."""
import json
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict
import hashlib

# Inputs
PROGRESS_FILE = Path('/tmp/bccc_me_full_results.json')
V3_DIR = Path('/home/motafeq/projects/sentinel/data_module/data/exports/sentinel-v3-smartbugs-2026-06-13')
BCCC_ME = Path('/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/MishandledException')

# Outputs
SUMMARY_FILE = Path('/tmp/bccc_me_full_summary.json')
INJECTION_LIST = Path('/tmp/bccc_me_injection_candidates.json')

# Detector lists (must match the audit)
SLITHER_ME = [
    'unchecked-transfer', 'unchecked-send', 'unchecked-lowlevel',
    'arbitrary-send-erc20', 'arbitrary-send-eth',
    'void-cst', 'incorrect-exp',
]
ADERYN_ME = [
    'unchecked-return', 'unchecked-send', 'arbitrary-transfer-from',
    'incorrect-erc20-interface', 'unsafe-erc20-operation',
]
SLITHER_ME_SET = set(SLITHER_ME)
ADERYN_ME_SET = set(ADERYN_ME)

def main():
    if not PROGRESS_FILE.exists():
        print(f'ERROR: {PROGRESS_FILE} not found. Run the audit first.')
        return
    if not V3_DIR.exists():
        print(f'ERROR: {V3_DIR} not found.')
        return

    print('=== Loading full audit results ===')
    results = json.loads(PROGRESS_FILE.read_text())
    print(f'  total contracts analyzed: {len(results):,}')

    # Slither status
    sl_status = Counter(v['slither_status'] for v in results.values())
    print()
    print('Slither status:')
    for s, c in sl_status.most_common():
        print(f'  {s:20s}: {c:>5,} ({100*c/len(results):.1f}%)')

    # Aderyn status
    ar_status = Counter(v['aderyn_status'] for v in results.values())
    print()
    print('Aderyn status:')
    for s, c in ar_status.most_common():
        print(f'  {s:20s}: {c:>5,} ({100*c/len(results):.1f}%)')

    # Compute per-contract verdict
    print()
    print('=== Computing per-contract verdict ===')
    verdicts = {}
    for bid, v in results.items():
        sl_ok = v['slither_status'] == 'OK'
        ar_ok = v['aderyn_status'] == 'OK'
        sl_hit = sl_ok and any(d in SLITHER_ME_SET for d in v.get('slither_detectors', []))
        ar_hit = ar_ok and any(d in ADERYN_ME_SET for d in v.get('aderyn_detectors', []))
        if sl_ok and ar_ok and sl_hit and ar_hit:
            v_kind = 'BOTH_HIGH_CONF'
        elif sl_ok and sl_hit:
            v_kind = 'SLITHER_ONLY'
        elif ar_ok and ar_hit:
            v_kind = 'ADERYN_ONLY'
        elif sl_ok and ar_ok:
            v_kind = 'BOTH_OK_BUT_NO_ME'
        else:
            v_kind = 'FAILED'
        verdicts[bid] = v_kind

    # Summary
    vc = Counter(verdicts.values())
    print()
    print('Verdict distribution:')
    for k, c in vc.most_common():
        print(f'  {k:30s}: {c:>5,} ({100*c/len(verdicts):.1f}%)')

    # Build the injection list
    both_ids = [bid for bid, v in verdicts.items() if v == 'BOTH_HIGH_CONF']
    print()
    print(f'=== INJECTION CANDIDATES (both-tool agreement) ===')
    print(f'  Count: {len(both_ids):,}')
    print(f'  Folder: {BCCC_ME}')
    print(f'  Sample IDs: {both_ids[:5]}')

    # Cross-corpus dedup against v3
    print()
    print('=== Cross-corpus dedup against v3 ===')
    # Compute SHA256 of BCCC source for each candidate
    print('  Computing SHA256 of BCCC candidates...')
    bccc_sha = {}
    for bid in both_ids:
        src = BCCC_ME / f'{bid}.sol'
        if src.exists():
            content = src.read_text(encoding='utf-8', errors='replace')
            sha = hashlib.sha256(content.encode()).hexdigest()
            bccc_sha[bid] = sha
    print(f'  Computed SHA256 for {len(bccc_sha):,} candidates')

    # Get v3 SHA256s
    v3_shas = set()
    for split in ['train', 'val', 'test']:
        p = V3_DIR / f'{split}.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                d = json.loads(line)
                if 'sha256' in d:
                    v3_shas.add(d['sha256'])
    print(f'  v3 unique SHAs: {len(v3_shas):,}')

    # Find duplicates
    duplicates = [bid for bid, sha in bccc_sha.items() if sha in v3_shas]
    print(f'  BCCC candidates that are SHA256 duplicates of v3: {len(duplicates)}')
    unique_candidates = [bid for bid in both_ids if bid not in duplicates]
    print(f'  Unique (non-duplicate) BCCC candidates: {len(unique_candidates):,}')

    # Save
    summary = {
        'total_contracts': len(results),
        'verdict_distribution': dict(vc),
        'slither_status': dict(sl_status),
        'aderyn_status': dict(ar_status),
        'both_high_confidence': both_ids,
        'slither_only': [bid for bid, v in verdicts.items() if v == 'SLITHER_ONLY'],
        'aderyn_only': [bid for bid, v in verdicts.items() if v == 'ADERYN_ONLY'],
        'failed': [bid for bid, v in verdicts.items() if v == 'FAILED'],
        'v3_duplicates': duplicates,
        'unique_candidates': unique_candidates,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print()
    print(f'Saved: {SUMMARY_FILE}')

    # Also: write a simple list of unique candidates for injection
    INJECTION_LIST.write_text(json.dumps({
        'bccc_me_both_high_conf': both_ids,
        'bccc_me_unique_after_dedup': unique_candidates,
        'count_both_high_conf': len(both_ids),
        'count_unique_after_dedup': len(unique_candidates),
    }, indent=2))
    print(f'Saved: {INJECTION_LIST}')

    print()
    print('=== FINAL TALLY ===')
    print(f'  BCCC ME folder:                5,154')
    print(f'  Analyzed (slither+aderyn):     {len(results):,}')
    print(f'  Both tools OK + both detect ME: {vc.get("BOTH_HIGH_CONF", 0):,} HIGH CONFIDENCE')
    print(f'  Slither-only ME detect:        {vc.get("SLITHER_ONLY", 0):,}')
    print(f'  Aderyn-only ME detect:         {vc.get("ADERYN_ONLY", 0):,}')
    print(f'  v3 SHA256 duplicates:          {len(duplicates):,}')
    print(f'  → INJECTABLE (BOTH + unique):  {len(unique_candidates):,}')
    print(f'  v3 ME was: 39 → would be:      {39 + len(unique_candidates):,} ({(39+len(unique_candidates))/39:.1f}x increase)')


if __name__ == '__main__':
    main()
