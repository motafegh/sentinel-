"""Run BOTH slither AND aderyn on all 60 BCCC contracts.
Use CLASS-SPECIFIC detector lists for each tool.
Compute 2-tool consensus.
Then go deeper on confirmed matches."""
import json
import shutil
import subprocess
import tempfile
import re
from pathlib import Path
from collections import defaultdict, Counter

# Tools
SLITHER = '/home/motafeq/projects/sentinel/.venv/bin/slither'
ADERYN = '/home/motafeq/.cargo/bin/aderyn'
BCCC_ROOT = Path('/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes')

# Read the sample IDs
samples = json.loads(Path('/tmp/bccc_folder_ids.json').read_text())
CLASSES = ['MishandledException', 'GasException', 'CallToUnknown', 'DenialOfService', 'TransactionOrderDependence']

# =============================================================================
# CLASS-SPECIFIC DETECTOR LISTS (slither + aderyn)
# =============================================================================
SLITHER_DETECTORS = {
    'MishandledException': [
        'unchecked-transfer', 'unchecked-send', 'unchecked-lowlevel',
        'arbitrary-send-erc20', 'arbitrary-send-erc20-permit', 'arbitrary-send-eth',
        'void-cst', 'incorrect-exp',
    ],
    'GasException': [
        'calls-loop', 'msg-value-loop', 'locked-ether',
        'controlled-delegatecall', 'delegatecall-loop',
    ],
    'CallToUnknown': [
        'low-level-calls', 'unchecked-lowlevel', 'arbitrary-send-eth',
    ],
    'DenialOfService': [
        'calls-loop', 'msg-value-loop', 'locked-ether', 'controlled-delegatecall',
    ],
    'TransactionOrderDependence': [
        'timestamp', 'block-timestamp',
    ],
}

ADERYN_DETECTORS = {
    'MishandledException': [
        'unchecked-return', 'unchecked-send', 'arbitrary-transfer-from',
        'incorrect-erc20-interface', 'unsafe-erc20-operation',
    ],
    'GasException': [
        'costly-loop', 'msg-value-in-loop', 'return-bomb',
        'delegatecall-in-loop', 'require-revert-in-loop',
    ],
    'CallToUnknown': [
        'unchecked-low-level-call', 'delegate-call-unchecked-address',
        'eth-send-unchecked-address', 'arbitrary-send-eth',
    ],
    'DenialOfService': [
        'costly-loop', 'delegatecall-in-loop', 'require-revert-in-loop',
    ],
    'TransactionOrderDependence': [
        'block-timestamp-deadline', 'timestamp',
    ],
}

# =============================================================================
# STEP 1: Run slither per-file with class-specific detectors
# =============================================================================
print('='*80)
print('STEP 1: Running SLITHER per-file with class-specific detectors')
print('='*80)

slither_findings = {}
total = sum(len(samples[cls]) for cls in CLASSES)
done = 0
for cls in CLASSES:
    folder_src = BCCC_ROOT / cls
    detectors = ','.join(SLITHER_DETECTORS[cls])
    for bid in samples[cls]:
        done += 1
        src = folder_src / f'{bid}.sol'
        if not src.exists():
            slither_findings[(cls, bid)] = {'status': 'NOT_FOUND', 'detectors': []}
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            shutil.copy(src, work / src.name)
            # Slither: --detect to filter, --json - for JSON output to stdout
            result = subprocess.run(
                [SLITHER, str(work / src.name), '--solc-disable-warnings',
                 '--detect', detectors, '--json', '-'],
                capture_output=True, text=True, timeout=90,
                cwd=str(work),
                env={**__import__('os').environ, 'PATH': '/home/motafeq/projects/sentinel/.venv/bin:/usr/local/bin:/usr/bin:/bin'},
            )
            stdout = result.stdout or ''
            # Find the JSON in stdout (slither may emit non-JSON before)
            json_start = stdout.find('{')
            if json_start < 0:
                slither_findings[(cls, bid)] = {'status': 'NO_JSON', 'detectors': [], 'stderr': result.stderr[:200]}
                continue
            try:
                data = json.loads(stdout[json_start:])
            except Exception as e:
                slither_findings[(cls, bid)] = {'status': 'JSON_ERROR', 'detectors': [], 'stderr': str(e)[:200]}
                continue
            if not data.get('success', False):
                slither_findings[(cls, bid)] = {'status': 'COMPILE_ERROR', 'detectors': [], 'stderr': data.get('error', '')[:200]}
                continue
            # Extract detector check names
            results = data.get('results', {}).get('detectors', [])
            triggered = []
            for r in results:
                cn = r.get('check', '')
                if cn:
                    triggered.append(cn)
            slither_findings[(cls, bid)] = {'status': 'OK', 'detectors': triggered}
        if done % 5 == 0 or done == total:
            print(f'  [{done}/{total}] processed')

# =============================================================================
# STEP 2: Combine with aderyn (we already have those results)
# =============================================================================
print()
print('='*80)
print('STEP 2: Loading aderyn results (already done in previous run)')
print('='*80)
aderyn_results = json.loads(Path('/tmp/bccc_aderyn_per_file.json').read_text())
print(f'  loaded {len(aderyn_results)} aderyn findings')

# =============================================================================
# STEP 3: Compute 2-tool consensus per class
# =============================================================================
print()
print('='*80)
print('STEP 3: 2-tool consensus per class')
print('='*80)
print(f'{"class":<32} {"slither-only":>13} {"aderyn-only":>12} {"both":>6} {"neither":>9} {"failed":>8}')
print('-'*90)

consensus = {}  # (cls, bid) -> {slither, aderyn, both_match}
for cls in CLASSES:
    s_only = a_only = both = neither = failed = 0
    for bid in samples[cls]:
        sl = slither_findings.get((cls, bid), {'status': 'UNKNOWN', 'detectors': []})
        ar = aderyn_results.get(f'{cls}|{bid}', {'aderyn_status': 'UNKNOWN', 'detectors': []})
        # Did each tool fire any of its class-specific detectors?
        sl_hit = sl.get('status') == 'OK' and any(d in SLITHER_DETECTORS[cls] for d in sl.get('detectors', []))
        ar_hit = ar.get('aderyn_status') == 'OK' and any(d in ADERYN_DETECTORS[cls] for d in ar.get('detectors', []))
        sl_failed = sl.get('status') in ('COMPILE_ERROR', 'NO_JSON', 'JSON_ERROR', 'NOT_FOUND', 'UNKNOWN')
        ar_failed = ar.get('aderyn_status') not in ('OK',)
        if sl_failed or ar_failed:
            failed += 1
        elif sl_hit and ar_hit:
            both += 1
        elif sl_hit:
            s_only += 1
        elif ar_hit:
            a_only += 1
        else:
            neither += 1
        consensus[(cls, bid)] = {
            'slither_hit': sl_hit, 'aderyn_hit': ar_hit, 'both': sl_hit and ar_hit,
            'slither_failed': sl_failed, 'aderyn_failed': ar_failed,
            'slither_detectors': sl.get('detectors', []),
            'aderyn_detectors': ar.get('detectors', []),
        }
    total_c = len(samples[cls])
    print(f'  {cls:<32} {s_only:>13} {a_only:>12} {both:>6} {neither:>9} {failed:>8}')

# =============================================================================
# STEP 4: List the "both" cases (high-confidence positives) for each class
# =============================================================================
print()
print('='*80)
print('STEP 4: HIGH-CONFIDENCE positives (BOTH slither AND aderyn confirm the BCCC label)')
print('='*80)
for cls in CLASSES:
    matches = [(bid, consensus[(cls, bid)]) for bid in samples[cls] if consensus[(cls, bid)]['both']]
    if matches:
        print(f'\n--- {cls} ({len(matches)} HIGH-CONFIDENCE matches out of {len(samples[cls])}) ---')
        for bid, c in matches:
            sl_d = [d for d in c['slither_detectors'] if d in SLITHER_DETECTORS[cls]]
            ar_d = [d for d in c['aderyn_detectors'] if d in ADERYN_DETECTORS[cls]]
            print(f'  {bid[:16]}... slither: {sl_d}, aderyn: {ar_d}')

# Extrapolation
print()
print('='*80)
print('STEP 5: Extrapolate to full BCCC folders (assuming proportional rates)')
print('='*80)
folder_sizes = {
    'MishandledException': 5154,
    'GasException': 6879,
    'CallToUnknown': 11131,
    'DenialOfService': 12394,
    'TransactionOrderDependence': 3562,
}
for cls in CLASSES:
    s_only = a_only = both = neither = failed = 0
    for bid in samples[cls]:
        c = consensus[(cls, bid)]
        if c['slither_failed'] or c['aderyn_failed']:
            failed += 1
        elif c['both']:
            both += 1
        elif c['slither_hit']:
            s_only += 1
        elif c['aderyn_hit']:
            a_only += 1
        else:
            neither += 1
    total_c = len(samples[cls])
    fs = folder_sizes[cls]
    both_pct = both / total_c
    either_pct = (both + s_only + a_only) / total_c
    print(f'  {cls:<32} 2-tool consensus: {both}/{total_c} = {100*both_pct:.0f}% × {fs:,} = ~{int(fs*both_pct):,} HIGH-conf')
    print(f'  {chr(0x20)*32} 1-tool detection:  {either_pct:.0%} × {fs:,} = ~{int(fs*either_pct):,} ANY-conf')

# Save
out = Path('/tmp/bccc_2tool_consensus.json')
serializable = {f'{cls}|{bid}': v for (cls, bid), v in consensus.items()}
out.write_text(json.dumps(serializable, indent=2, default=str))
slither_out = Path('/tmp/bccc_slither_results.json')
slither_serializable = {f'{cls}|{bid}': v for (cls, bid), v in slither_findings.items()}
slither_out.write_text(json.dumps(slither_serializable, indent=2, default=str))
print()
print(f'Saved: {out}')
print(f'Saved: {slither_out}')
