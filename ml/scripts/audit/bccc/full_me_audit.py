"""Full BCCC MishandledException 2-tool audit.
Runs slither + aderyn on all 5,154 BCCC ME contracts.
Saves results incrementally so we can resume.
Parallelized with multiprocessing."""
import json
import shutil
import subprocess
import tempfile
import os
import sys
import time
import re
from pathlib import Path
from multiprocessing import Pool, Manager
from collections import defaultdict, Counter

# Tools and paths
SLITHER = '/home/motafeq/projects/sentinel/.venv/bin/slither'
ADERYN = '/home/motafeq/.cargo/bin/aderyn'
BCCC_ROOT = Path('/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes')
BCCC_ME = BCCC_ROOT / 'MishandledException'

# Output files
PROGRESS_FILE = Path('/tmp/bccc_me_full_progress.json')
FINAL_FILE = Path('/tmp/bccc_me_full_results.json')

# Detector lists
SLITHER_ME = [
    'unchecked-transfer', 'unchecked-send', 'unchecked-lowlevel',
    'arbitrary-send-erc20', 'arbitrary-send-eth',
    'void-cst', 'incorrect-exp',
]
ADERYN_ME = [
    'unchecked-return', 'unchecked-send', 'arbitrary-transfer-from',
    'incorrect-erc20-interface', 'unsafe-erc20-operation',
]
ADERYN_ME_SET = set(ADERYN_ME)

def process_contract(args):
    """Run slither + aderyn on one contract. Returns (bid, slither_status, slither_dets, aderyn_status, aderyn_dets)."""
    bid, workdir = args
    src = BCCC_ME / f'{bid}.sol'
    if not src.exists():
        return (bid, 'NOT_FOUND', [], 'NOT_FOUND', [])

    # Copy to workspace
    shutil.copy(src, Path(workdir) / src.name)

    # Run Aderyn
    aderyn_status = 'FAILED'
    aderyn_dets = []
    try:
        result = subprocess.run(
            [ADERYN, workdir, '-o', f'{workdir}/aderyn_report.json'],
            capture_output=True, text=True, timeout=30
        )
        report_path = Path(workdir) / 'aderyn_report.json'
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
                high = report.get('high_issues', {}).get('issues', [])
                low = report.get('low_issues', {}).get('issues', [])
                all_issues = high + low
                dets = []
                for issue in all_issues:
                    d = issue.get('detector_name', '')
                    if d and d not in dets:
                        dets.append(d)
                aderyn_dets = dets
                aderyn_status = 'OK'
            except Exception:
                aderyn_status = 'JSON_ERROR'
    except subprocess.TimeoutExpired:
        aderyn_status = 'TIMEOUT'
    except Exception:
        aderyn_status = 'ERROR'

    # Run Slither (in .venv with solc 0.4.24)
    slither_status = 'FAILED'
    slither_dets = []
    try:
        env = {**os.environ, 'PATH': '/home/motafeq/projects/sentinel/.venv/bin:/usr/local/bin:/usr/bin:/bin'}
        result = subprocess.run(
            [SLITHER, f'{workdir}/{src.name}', '--solc-disable-warnings',
             '--detect', ','.join(SLITHER_ME), '--json', '-'],
            capture_output=True, text=True, timeout=60,
            env=env,
        )
        stdout = result.stdout or ''
        json_start = stdout.find('{')
        if json_start >= 0:
            try:
                data = json.loads(stdout[json_start:])
                if data.get('success', False):
                    results_list = data.get('results', {}).get('detectors', [])
                    slither_dets = [r.get('check', '') for r in results_list if r.get('check')]
                    slither_status = 'OK'
                else:
                    slither_status = 'COMPILE_ERROR'
            except Exception:
                slither_status = 'JSON_ERROR'
    except subprocess.TimeoutExpired:
        slither_status = 'TIMEOUT'
    except Exception:
        slither_status = 'ERROR'

    # Clean up aderyn report file
    try:
        Path(f'{workdir}/aderyn_report.json').unlink(missing_ok=True)
    except Exception:
        pass

    return (bid, slither_status, slither_dets, aderyn_status, aderyn_dets)


def worker_init(workdir_base):
    """Initialize each worker with a workdir."""
    worker_init.workdir = workdir_base


def main():
    # Get all contract IDs
    print('=== Enumerating BCCC ME contracts ===')
    all_bids = sorted([f.stem for f in BCCC_ME.glob('*.sol')])
    print(f'  total: {len(all_bids):,} contracts')

    # Load progress (resumable)
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
            print(f'  loaded progress: {len(progress):,} contracts already done')
        except Exception:
            progress = {}

    # Filter to remaining
    remaining = [bid for bid in all_bids if bid not in progress]
    print(f'  remaining: {len(remaining):,} contracts')

    if not remaining:
        print('  All contracts processed!')
        return

    # Setup workdir
    workdir_base = tempfile.mkdtemp(prefix='bccc_me_')
    print(f'  workdir: {workdir_base}')

    # Per-worker subdirs
    n_workers = 10  # 12 CPUs available, leave 2 for system
    worker_dirs = [os.path.join(workdir_base, f'w{i}') for i in range(n_workers)]
    for d in worker_dirs:
        os.makedirs(d, exist_ok=True)

    # Build args list
    args_list = [(bid, worker_dirs[i % n_workers]) for i, bid in enumerate(remaining)]
    print(f'  launching {n_workers} workers on {len(args_list):,} contracts')

    start_time = time.time()
    completed = 0
    last_save = 0

    # Use multiprocessing.Pool
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_contract, args_list, chunksize=1):
            bid, sl_status, sl_dets, ar_status, ar_dets = result
            progress[bid] = {
                'slither_status': sl_status,
                'slither_detectors': sl_dets,
                'aderyn_status': ar_status,
                'aderyn_detectors': ar_dets,
            }
            completed += 1
            # Save every 50 contracts
            if completed - last_save >= 50:
                PROGRESS_FILE.write_text(json.dumps(progress))
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta_seconds = (len(remaining) - completed) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                print(f'  [{completed}/{len(remaining)}] {rate:.2f}/s, ETA {eta_minutes:.0f} min', flush=True)
                last_save = completed

    # Final save
    PROGRESS_FILE.write_text(json.dumps(progress))
    elapsed = time.time() - start_time
    print(f'\n=== DONE: {len(progress):,} contracts in {elapsed/60:.1f} min ({len(progress)/elapsed:.2f}/s) ===')

    # Move to final results
    FINAL_FILE.write_text(json.dumps(progress))
    print(f'Saved: {FINAL_FILE}')

    # Quick summary
    statuses = Counter(v['slither_status'] for v in progress.values())
    print()
    print('Slither status:')
    for s, c in statuses.most_common():
        print(f'  {s}: {c}')

    aderyn_statuses = Counter(v['aderyn_status'] for v in progress.values())
    print()
    print('Aderyn status:')
    for s, c in aderyn_statuses.most_common():
        print(f'  {s}: {c}')


if __name__ == '__main__':
    main()
