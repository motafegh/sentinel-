"""
WS-I Stage 1: Build the 848-contract stratified sample for slither label validation,
test the slither harness on 5 contracts, and verify the full pipeline works end-to-end.

Sample design (from 03_phase3_plan.md WS-I):
  - 766 review_pending (NV+vuln contradictions)
  - 2 nine-folder "maxing" contracts
  - 50 multi-positive (n_pos >= 2), stratified across classes
  - 30 disagreement sample (selected AFTER the slither run in Stage 2)
  Total: 848 (when including 30 post-slither disagreements; Stage 1 saves 818 + 30 placeholder rows)

The 30 "disagreement" rows are PLACEHOLDERS at this stage. They will be filled in Stage 2
after we know which 30 contracts have the worst 2-way disagreement.

This script:
  1. Loads contracts_clean.csv
  2. Builds the 818-row sample (766 + 2 + 50)
  3. Saves to outputs/ws_i_sample_818.csv with columns: id, bccc_path, bccc_classes,
     n_pos, review_pending, sample_reason, slither_status (empty)
  4. Defines a slither_detect(path) -> dict function (the harness)
  5. Tests the harness on 5 contracts
  6. Prints a verification summary
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]  # scripts -> Phase3 -> BCCC-SCsVul-2024_Deep_Dive -> Data -> Deep_Dive -> sentinel
PHASE2_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/outputs"
PHASE3_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs"
PHASE3_OUT.mkdir(parents=True, exist_ok=True)

CONTRACTS_CSV = PHASE2_OUT / "contracts_clean.csv"
SAMPLE_OUT = PHASE3_OUT / "ws_i_sample_818.csv"
SAMPLE_TEST_OUT = PHASE3_OUT / "ws_i_harness_test.json"

# Per-class columns (in SENTINEL v9 order, NO Class05/Class07 per D-F1)
CLASS_COLS = [
    "Class01:ExternalBug", "Class02:GasException", "Class03:MishandledException",
    "Class04:Timestamp", "Class06:UnusedReturn", "Class08:CallToUnknown",
    "Class09:DenialOfService", "Class10:IntegerUO", "Class11:Reentrancy",
    "Class12:NonVulnerable",
]

# Slither detector -> BCCC class mapping (from 03_phase3_plan.md §2.1)
# Key: slither check name; Value: set of BCCC classes it implies
SLITHER_TO_BCCC = {
    # Class01: ExternalBug
    "arbitrary-send-eth": {"Class01:ExternalBug"},
    "arbitrary-send-erc20": {"Class01:ExternalBug"},
    "arbitrary-send-erc20-permit": {"Class01:ExternalBug"},
    "controlled-delegatecall": {"Class01:ExternalBug"},
    "delegatecall-loop": {"Class01:ExternalBug"},
    "msg-value-loop": {"Class01:ExternalBug"},
    "reentrancy-eth": {"Class01:ExternalBug", "Class11:Reentrancy"},
    "reentrancy-no-eth": {"Class01:ExternalBug", "Class11:Reentrancy"},
    "reentrancy-unlimited-gas": {"Class01:ExternalBug", "Class11:Reentrancy"},
    "reentrancy-benign": {"Class01:ExternalBug", "Class11:Reentrancy"},
    "reentrancy-events": {"Class01:ExternalBug", "Class11:Reentrancy"},
    # Class02: GasException
    "void-cst": {"Class02:GasException"},
    "constant-function-asm": {"Class02:GasException"},
    "constant-function-state": {"Class02:GasException"},
    "events-maths": {"Class02:GasException"},
    # Class03: MishandledException
    "locked-ether": {"Class02:GasException", "Class03:MishandledException"},
    "incorrect-return": {"Class03:MishandledException"},
    "uninitialized-state": {"Class03:MishandledException"},
    "uninitialized-storage": {"Class03:MishandledException"},
    "uninitialized-local": {"Class03:MishandledException"},
    "mapping-deletion": {"Class03:MishandledException"},
    "modifying-storage-array-by-value": {"Class03:MishandledException"},
    # Class04: Timestamp
    "timestamp": {"Class04:Timestamp"},
    "weak-prng": {"Class04:Timestamp"},
    "block-timestamp": {"Class04:Timestamp"},
    # Class06: UnusedReturn
    "unchecked-transfer": {"Class06:UnusedReturn"},
    "unchecked-send": {"Class06:UnusedReturn"},
    "unchecked-lowlevel": {"Class06:UnusedReturn"},
    "unused-return": {"Class06:UnusedReturn"},
    # Class08: CallToUnknown
    "missing-zero-check": {"Class08:CallToUnknown"},
    "uninitialized-fptr": {"Class08:CallToUnknown"},
    # Class09: DenialOfService
    "calls-loop": {"Class09:DenialOfService"},
    # Class10: IntegerUO
    "divide-before-multiply": {"Class10:IntegerUO"},
    "incorrect-exp": {"Class10:IntegerUO"},
    "tautological-compare": {"Class10:IntegerUO"},
    "incorrect-equality": {"Class10:IntegerUO"},
    "strict-equality": {"Class10:IntegerUO"},
    "out-of-bounds-array": {"Class10:IntegerUO"},
    "shift-parameter": {"Class10:IntegerUO"},
    # Class11: Reentrancy
    "tx-origin": {"Class11:Reentrancy"},  # proxy
    "suicidal": {"Class11:Reentrancy"},
    # Note: "suicidal" is actually Class07 (WeakAccessMod) but Class07 is dropped,
    # so we map it to nearest alive class. Will refine in Stage 2.
}


def fix_path(stored_path: str) -> str:
    """Phase 2 CSV stored 'Source Codes' (with space) but dir was renamed to 'SourceCodes'."""
    return stored_path.replace("Source Codes", "SourceCodes")


def build_sample() -> pd.DataFrame:
    """Build the 818-contract sample (766 + 2 + 50)."""
    print("Loading contracts_clean.csv ...")
    df = pd.read_csv(CONTRACTS_CSV)
    print(f"  Loaded {len(df):,} contracts, {df.shape[1]} cols")

    # Fix paths
    df["bccc_path_fixed"] = df["bccc_file_path"].apply(fix_path)

    samples = []

    # --- Bucket 1: All 766 review_pending ---
    rp = df[df["review_pending"] == 1].copy()
    rp["sample_reason"] = "review_pending"
    samples.append(rp)
    print(f"  Bucket 1 (review_pending): {len(rp):,}")

    # --- Bucket 2: Nine-folder "maxing" contracts ---
    # Per Phase 2 WS-N: contracts with the highest n_pos
    # The plan said 2 contracts; we find the actual top-2 by n_pos
    high_pos = df.nlargest(5, "n_pos")[["id", "bccc_path_fixed", "n_pos", "primary_class"] + CLASS_COLS]
    print(f"  Top-5 by n_pos: {high_pos[['id', 'n_pos', 'primary_class']].to_dict('records')}")
    # Take top 2 (these are the "nine-folder maxing" contracts)
    maxers = df.nlargest(2, "n_pos").copy()
    maxers["sample_reason"] = "nine_folder_maxing"
    samples.append(maxers)
    print(f"  Bucket 2 (maxing): {len(maxers)} (n_pos={maxers['n_pos'].tolist()})")

    # --- Bucket 3: 50 multi-positive, stratified ---
    # n_pos >= 2, balanced across primary_class. 5 per class for 10 classes = 50.
    multi = df[(df["n_pos"] >= 2) & (df["review_pending"] == 0)].copy()
    # If a class has < 5 multi-positive, take all of them
    multi_sample = (
        multi.groupby("primary_class", group_keys=False)
        .apply(lambda g: g.sample(n=min(5, len(g)), random_state=42))
    )
    multi_sample["sample_reason"] = "multi_positive"
    samples.append(multi_sample)
    print(f"  Bucket 3 (multi-positive, 5/class): {len(multi_sample)}")

    # --- Bucket 4: Placeholder for 30 post-slither disagreement (Stage 2) ---
    # Empty for now; will be filled in Stage 2.
    placeholder = pd.DataFrame(columns=df.columns.tolist() + ["sample_reason"])
    placeholder["sample_reason"] = "disagreement_post_slither"
    samples.append(placeholder)
    print(f"  Bucket 4 (disagreement, placeholder): {len(placeholder)}")

    combined = pd.concat(samples, ignore_index=True, sort=False)
    # Ensure sample_reason column exists for all rows
    if "sample_reason" not in combined.columns:
        combined["sample_reason"] = "unknown"

    # Add stage-2 placeholder columns
    combined["slither_status"] = ""   # PENDING / OK / COMPILE_ERROR / TIMEOUT / EXCEPTION
    combined["slither_hits"] = ""     # JSON list of detector names
    combined["agreement_per_class"] = ""  # JSON dict[class -> "agree" | "disagree" | "miss" | "false_pos"]
    combined["disagreement_score"] = -1  # 0=full agreement, 1=full disagreement (filled in Stage 2)

    # Save
    out_cols = ["id", "bccc_path_fixed", "sample_reason", "review_pending",
                "primary_class", "n_pos", "pragma"] + CLASS_COLS + [
                "slither_status", "slither_hits", "agreement_per_class", "disagreement_score"]
    # Ensure pragma column exists
    if "pragma" not in combined.columns:
        combined["pragma"] = ""
    combined[out_cols].to_csv(SAMPLE_OUT, index=False)
    print(f"\nSaved sample to {SAMPLE_OUT}")
    print(f"  Total rows: {len(combined)} (818 expected: 766+2+50+0)")

    return combined[out_cols]


# ---- Slither Harness ----

SLITHER_TIMEOUT = 30  # seconds per contract

# Solc versions to try (in priority order) when pragma is missing/unknown
DEFAULT_SOLC_VERSIONS = ["0.5.17", "0.4.26", "0.6.12", "0.7.6", "0.8.20"]


def pick_solc_version(pragma: str) -> str:
    """Pick the highest installed solc version that satisfies the pragma.

    Handles common pragma patterns:
      - ^0.5.0    -> 0.5.x (highest installed)
      - >=0.4.22  -> any 0.4.x or higher
      - >=0.4.22 <0.6.0 -> 0.4.x or 0.5.x
      - 0.4.24    -> 0.4.24
      - None/empty -> default 0.5.17

    Also verifies the chosen version is selectable (solc-select can switch to it).
    Versions like 0.8.35 might be on disk but not in solc-select's registry.
    """
    # Handle NaN, None, empty, and string "nan" (from pd.read_csv coercion)
    if pragma is None:
        return _verified_default_solc()
    if isinstance(pragma, float) and pd.isna(pragma):
        return _verified_default_solc()
    if not isinstance(pragma, str):
        return _verified_default_solc()
    s = pragma.strip()
    if s == "" or s == "False" or s.lower() == "nan":
        return _verified_default_solc()

    # Get all installed solc versions that solc-select can actually use
    solc_dir = Path.home() / ".solc-select" / "artifacts"
    if not solc_dir.exists():
        return _verified_default_solc()
    installed = []
    for p in solc_dir.iterdir():
        if p.name.startswith("solc-"):
            v = p.name[5:]  # strip "solc-"
            if _verify_solc_works(v):
                installed.append(v)

    def parse_ver(v):
        try:
            parts = v.split(".")
            return tuple(int(p) for p in parts[:3])
        except Exception:
            return (0, 0, 0)

    installed.sort(key=parse_ver, reverse=True)

    # Extract upper bound from pragma
    upper_match = re.search(r"<\s*(\d+\.\d+(?:\.\d+)?)", pragma)
    lower_match = re.search(r">=?\s*(\d+\.\d+(?:\.\d+)?)", pragma)
    caret_match = re.search(r"\^\s*(\d+\.\d+(?:\.\d+)?)", pragma)
    # Exact pragma (no operator, just X.Y.Z): treat as "X.Y.Z only"
    exact_match = re.fullmatch(r"\s*(\d+\.\d+(?:\.\d+)?)\s*", pragma)

    # For exact pragma (e.g. "0.4.24"), try the exact version first.
    if exact_match:
        exact_v = exact_match.group(1)
        if exact_v in installed:
            return exact_v
        # Exact version not installed; fall back to the highest installed in the same major.minor
        try:
            parts = exact_v.split(".")
            major_minor = (int(parts[0]), int(parts[1]))
            candidates = [v for v in installed if (parse_ver(v)[0], parse_ver(v)[1]) == major_minor]
            if candidates:
                candidates.sort(key=parse_ver, reverse=True)
                return candidates[0]
        except Exception:
            pass
        # If no version in same major.minor, fall back to general search
        # (i.e. just use lower bound)

    # For ^X.Y.Z, treat as >=X.Y.Z <X.(Y+1).0
    if caret_match:
        base = caret_match.group(1)
        try:
            parts = base.split(".")
            major, minor = int(parts[0]), int(parts[1])
            upper = f"{major}.{minor + 1}.0"
        except Exception:
            upper = None
        lower = base
    else:
        upper = upper_match.group(1) if upper_match else None
        lower = lower_match.group(1) if lower_match else None

    # Find best installed version satisfying [lower, upper)
    for v in installed:
        pv = parse_ver(v)
        if lower:
            pl = parse_ver(lower)
            if pv < pl:
                continue
        if upper:
            pu = parse_ver(upper)
            if pv >= pu:
                continue
        return v

    # Fallback: highest installed 0.5.x
    for v in installed:
        if v.startswith("0.5."):
            return v
    if installed:
        return installed[0]
    return _verified_default_solc()


def _verify_solc_works(version: str) -> bool:
    """Test that solc-select can switch to this version."""
    import subprocess
    try:
        r = subprocess.run(
            ["solc-select", "use", version],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _verified_default_solc() -> str:
    """Return the first of DEFAULT_SOLC_VERSIONS that's actually selectable."""
    for v in DEFAULT_SOLC_VERSIONS:
        if _verify_solc_works(v):
            return v
    return "0.5.17"


def _slither_worker(args):
    """Run slither on a single .sol file with timeout. Returns (status, hits, err).

    Uses subprocess + timeout (not signal.alarm) because slither spawns its own
    threads/processes and signal.alarm is unreliable in those cases.

    IMPORTANT: In slither 0.11+, detectors are NOT auto-registered.
    The driver script explicitly imports `slither.detectors.all_detectors` and
    registers each detector class with `slither.register_detector(cls)`.
    """
    contract_path, tmpdir, timeout, solc_version = args
    result = {
        "path": contract_path,
        "status": "PENDING",
        "hits": [],
        "hit_counts": {},
        "elapsed_sec": 0.0,
        "err": None,
        "solc_version": solc_version,
    }
    start = time.time()
    try:
        # Write a small driver script to tmpdir
        driver = Path(tmpdir) / "_run_slither.py"
        driver.write_text(f"""
import json, sys, os

# Switch solc version BEFORE importing slither
if {solc_version!r}:
    import subprocess
    subprocess.run(["solc-select", "use", {solc_version!r}], check=False, capture_output=True)

from slither import Slither
# CRITICAL: slither 0.11+ does NOT auto-register detectors.
# We must import the all_detectors module AND register each class.
from slither.detectors import all_detectors
import slither.detectors.all_detectors as _ad

# Collect all detector classes from the all_detectors module's namespace
_detector_classes = []
for _name in dir(_ad):
    _obj = getattr(_ad, _name)
    if isinstance(_obj, type):
        try:
            from slither.detectors.abstract_detector import AbstractDetector
            if issubclass(_obj, AbstractDetector) and _obj is not AbstractDetector:
                _detector_classes.append(_obj)
        except Exception:
            pass

slither = Slither({contract_path!r})
for _cls in _detector_classes:
    slither.register_detector(_cls)

detector_results = slither.run_detectors()
hits = []
hit_counts = {{}}
for finding in detector_results:
    check = finding.get("check") if isinstance(finding, dict) else None
    if check:
        hits.append(check)
        hit_counts[check] = hit_counts.get(check, 0) + 1
print(json.dumps({{"status": "OK", "hits": hits, "hit_counts": hit_counts, "n_detectors": len(_detector_classes)}}))
""")
        # Run with timeout
        import subprocess
        proc = subprocess.run(
            [sys.executable, str(driver)],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=str(Path(__file__).resolve().parents[5]),
        )
        result["elapsed_sec"] = time.time() - start
        if proc.returncode != 0:
            err_short = (proc.stderr or "")[:500]
            if "compilation" in err_short.lower() or "parse" in err_short.lower() or "syntax" in err_short.lower():
                result["status"] = "COMPILE_ERROR"
            else:
                result["status"] = "EXCEPTION"
            result["err"] = err_short
        else:
            # Parse last line of stdout as JSON
            try:
                last = proc.stdout.strip().splitlines()[-1]
                parsed = json.loads(last)
                result["status"] = parsed.get("status", "OK")
                result["hits"] = parsed.get("hits", [])
                result["hit_counts"] = parsed.get("hit_counts", {})
                result["n_detectors"] = parsed.get("n_detectors", 0)
            except Exception as e:
                result["status"] = "EXCEPTION"
                result["err"] = f"parse fail: {e}; stdout tail: {proc.stdout[-300:]}"
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["elapsed_sec"] = timeout
        result["err"] = f"timeout after {timeout}s"
    except Exception as e:
        result["status"] = "EXCEPTION"
        result["err"] = f"{type(e).__name__}: {str(e)[:200]}"
        result["elapsed_sec"] = time.time() - start
    return result


def slither_detect(contract_path: str, pragma: str = "", timeout: int = SLITHER_TIMEOUT) -> dict:
    """Public entry point. Returns dict with status, hits, hit_counts, elapsed_sec, err.

    Args:
        contract_path: Absolute or relative path to .sol file
        pragma: Solidity pragma string (e.g. "^0.5.0", ">=0.4.22 <0.6.0")
        timeout: Seconds before subprocess kill

    This is a SUBPROCESS-based wrapper so it works in any context (multiprocessing,
    sequential, etc.) and is robust to slither's internal threading.
    """
    import tempfile
    solc = pick_solc_version(pragma)
    with tempfile.TemporaryDirectory() as tmpdir:
        return _slither_worker((contract_path, tmpdir, timeout, solc))


def harness_test(sample: pd.DataFrame, n: int = 5):
    """Run slither on n sample contracts to verify the harness works."""
    print(f"\n=== Slither harness test on {n} contracts ===")
    # Pick a mix: 2 review_pending, 1 multi_positive, 1 from each class type
    rp_rows = sample[sample["sample_reason"] == "review_pending"].head(2)
    mp_rows = sample[sample["sample_reason"] == "multi_positive"].head(2)
    nf_rows = sample[sample["sample_reason"] == "nine_folder_maxing"].head(1)
    test_rows = pd.concat([rp_rows, mp_rows, nf_rows]).drop_duplicates("id").head(n)

    results = []
    for i, row in test_rows.reset_index(drop=True).iterrows():
        path = row["bccc_path_fixed"]
        if not os.path.exists(path):
            print(f"  [{i+1}/{n}] SKIP (path missing): {path}")
            results.append({"path": path, "status": "PATH_MISSING", "hits": [], "hit_counts": {}, "elapsed_sec": 0.0, "err": "path missing"})
            continue
        pragma = row.get("pragma", "") if "pragma" in row else ""
        print(f"  [{i+1}/{n}] {row['id'][:16]}... sample_reason={row['sample_reason']}, n_pos={row['n_pos']}, pragma={pragma}")
        r = slither_detect(path, pragma=pragma)
        print(f"        status={r['status']}, hits={len(r['hits'])}, detectors={r.get('n_detectors', 0)}, elapsed={r['elapsed_sec']:.1f}s")
        if r["hits"]:
            print(f"        first 5 detectors: {r['hits'][:5]}")
        if r["err"]:
            print(f"        err: {r['err'][:200]}")
        r["id"] = row["id"]
        r["sample_reason"] = row["sample_reason"]
        r["bccc_classes"] = [c for c in CLASS_COLS if row[c] == 1]
        results.append(r)

    # Save test output
    with open(SAMPLE_TEST_OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved harness test results to {SAMPLE_TEST_OUT}")

    # Summary
    status_counts = pd.Series([r["status"] for r in results]).value_counts().to_dict()
    print(f"\nStatus summary: {status_counts}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-test", action="store_true", help="Skip slither harness test")
    parser.add_argument("--test-n", type=int, default=5, help="How many contracts to test on")
    args = parser.parse_args()

    sample = build_sample()

    if not args.skip_test:
        harness_test(sample, n=args.test_n)

    print("\n=== Stage 1 done ===")
    print(f"Sample: {SAMPLE_OUT}")
    print(f"Test results: {SAMPLE_TEST_OUT}")
