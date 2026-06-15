"""Re-run aderyn on contracts where it FAILED in the full audit.
Use FRESH dir per call to avoid the shared-dir issues.
PROPER LOGGING: every contract gets stdout, stderr, returncode, status, duration logged."""
import json
import subprocess
import tempfile
import os
import sys
import time
import re
from pathlib import Path
from multiprocessing import Pool
from collections import Counter, defaultdict

ADERYN = '/home/motafeq/.cargo/bin/aderyn'
PROGRESS_FILE = Path('/tmp/bccc_me_full_results.json')
LOG_FILE = Path('/tmp/bccc_me_aderyn_retry.log')
RETRY_PROGRESS = Path('/tmp/bccc_me_aderyn_retry_progress.json')

# Detector list for ME
ADERYN_ME = [
    'unchecked-return', 'unchecked-send', 'arbitrary-transfer-from',
    'incorrect-erc20-interface', 'unsafe-erc20-operation',
]
ADERYN_ME_SET = set(ADERYN_ME)

def retry_aderyn(bid):
    """Re-run aderyn on one contract with fresh temp dir.
    Returns (bid, status, detectors, duration_sec, stderr_first_300chars)."""
    src = Path('/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/MishandledException') / f'{bid}.sol'
    if not src.exists():
        return (bid, 'NOT_FOUND', [], 0.0, 'file not found')

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='aderyn_retry_') as workdir:
        # Copy contract to fresh dir
        import shutil
        shutil.copy(src, Path(workdir) / src.name)

        # Run aderyn
        try:
            result = subprocess.run(
                [ADERYN, workdir, '-o', f'{workdir}/report.json'],
                capture_output=True, text=True, timeout=30
            )
            duration = time.perf_counter() - start
            stderr_short = (result.stderr or '')[:300].replace('\n', ' / ')
            stdout_short = (result.stdout or '')[:300].replace('\n', ' / ')

            report_path = Path(workdir) / 'report.json'
            if not report_path.exists():
                return (bid, f'NO_REPORT_FILE rc={result.returncode}', [],
                        duration, f'stderr: {stderr_short} | stdout: {stdout_short}')

            try:
                report = json.loads(report_path.read_text())
            except Exception as e:
                return (bid, f'JSON_ERROR: {e}', [], duration,
                        f'stderr: {stderr_short} | stdout: {stdout_short}')

            high = report.get('high_issues', {}).get('issues', [])
            low = report.get('low_issues', {}).get('issues', [])
            all_issues = high + low
            dets = []
            for issue in all_issues:
                d = issue.get('detector_name', '')
                if d and d not in dets:
                    dets.append(d)
            return (bid, 'OK', dets, duration, '')

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start
            return (bid, 'TIMEOUT', [], duration, 'subprocess timeout after 30s')
        except Exception as e:
            duration = time.perf_counter() - start
            return (bid, f'ERROR: {e}', [], duration, '')


def log_line(line):
    """Append line to log file (with flush)."""
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
        f.flush()


def main():
    if not PROGRESS_FILE.exists():
        print(f'ERROR: {PROGRESS_FILE} not found. Run the full audit first.')
        return

    print('=== Loading full audit results ===')
    progress = json.loads(PROGRESS_FILE.read_text())
    print(f'  total contracts: {len(progress):,}')

    # Find contracts where aderyn FAILED (we want to retry these)
    failed_bids = []
    for bid, v in progress.items():
        if v.get('aderyn_status') != 'OK':
            failed_bids.append(bid)
    print(f'  aderyn failed contracts to retry: {len(failed_bids):,}')

    # Also include contracts where aderyn OK but we want to double-check
    # (skip — the original audit's "OK" is fine)

    # Load retry progress (resumable)
    retry_progress = {}
    if RETRY_PROGRESS.exists():
        try:
            retry_progress = json.loads(RETRY_PROGRESS.read_text())
            print(f'  already retried: {len(retry_progress):,}')
        except Exception:
            retry_progress = {}

    remaining = [bid for bid in failed_bids if bid not in retry_progress]
    print(f'  remaining: {len(remaining):,}')

    if not remaining:
        print('  All retries done!')
        # Skip to analysis
    else:
        # Open log file fresh
        with open(LOG_FILE, 'a') as f:
            f.write(f'\n=== RETRY STARTED at {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
            f.write(f'  total to retry: {len(remaining):,}\n')

        # Run with multiprocessing
        n_workers = 10
        print(f'  launching {n_workers} workers')
        completed = 0
        start_time = time.time()
        last_save = 0

        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(retry_aderyn, remaining, chunksize=1):
                bid, status, dets, duration, err_excerpt = result
                retry_progress[bid] = {
                    'aderyn_retry_status': status,
                    'aderyn_retry_detectors': dets,
                    'duration_sec': round(duration, 2),
                    'error_excerpt': err_excerpt,
                }
                # Log every contract
                status_short = status[:50]
                log_line(f'[{completed+1:5d}/{len(remaining)}] {bid[:16]}... {duration:5.1f}s {status_short:55s} dets={dets[:3]} err={err_excerpt[:100]}')
                completed += 1

                # Save every 25 contracts
                if completed - last_save >= 25:
                    RETRY_PROGRESS.write_text(json.dumps(retry_progress))
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta_sec = (len(remaining) - completed) / rate if rate > 0 else 0
                    print(f'  [{completed}/{len(remaining)}] {rate:.2f}/s, ETA {eta_sec/60:.0f} min', flush=True)
                    last_save = completed

        # Final save
        RETRY_PROGRESS.write_text(json.dumps(retry_progress))
        print(f'\n=== RETRY DONE: {len(retry_progress):,} contracts in {(time.time()-start_time)/60:.1f} min ===')

    # Now merge retry results back into the full progress
    print()
    print('=== Merging retry results into full progress ===')
    improved = 0
    still_failed = 0
    for bid, retry in retry_progress.items():
        if bid in progress:
            old_status = progress[bid].get('aderyn_status', '?')
            new_status = retry.get('aderyn_retry_status', '?')
            if old_status != 'OK' and new_status == 'OK':
                improved += 1
            elif old_status != 'OK' and new_status != 'OK':
                still_failed += 1
            # Update the progress with the retry result
            progress[bid]['aderyn_status'] = new_status
            progress[bid]['aderyn_detectors'] = retry.get('aderyn_retry_detectors', [])
            progress[bid]['aderyn_retry_error'] = retry.get('error_excerpt', '')

    PROGRESS_FILE.write_text(json.dumps(progress))
    print(f'  improved: {improved}')
    print(f'  still failed: {still_failed}')

    # Final summary
    print()
    print('='*80)
    print('FINAL Aderyn status after retry:')
    print('='*80)
    new_statuses = Counter(v.get('aderyn_status', '?') for v in progress.values())
    for s, c in new_statuses.most_common():
        print(f'  {s[:30]:30s}: {c:>5,} ({100*c/len(progress):.1f}%)')

    # Failure breakdown
    print()
    print('Failure breakdown (top 20):')
    fail_reasons = Counter()
    for v in progress.values():
        s = v.get('aderyn_status', 'OK')
        if s != 'OK':
            # Categorize
            err = v.get('aderyn_retry_error', '')
            if 'Compilation' in err or 'Undeclared' in err or 'pragma' in err:
                fail_reasons['compilation_error'] += 1
            elif 'serialize' in err.lower():
                fail_reasons['ast_serialize_error'] += 1
            elif 'timeout' in err.lower() or 'TIMEOUT' in s:
                fail_reasons['timeout'] += 1
            elif 'NOT_FOUND' in s:
                fail_reasons['not_found'] += 1
            elif 'NO_REPORT' in s:
                fail_reasons['no_report'] += 1
            elif 'JSON_ERROR' in s:
                fail_reasons['json_error'] += 1
            else:
                fail_reasons['other'] += 1
    for reason, c in fail_reasons.most_common(20):
        print(f'  {reason:30s}: {c:>5,}')


if __name__ == '__main__':
    main()
