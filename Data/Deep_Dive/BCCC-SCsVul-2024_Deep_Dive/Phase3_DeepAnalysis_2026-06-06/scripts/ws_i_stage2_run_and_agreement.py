"""
WS-I Stage 2: Run slither on all 808 sample contracts (parallelized), compute
per-class agreement metrics, identify the 30 worst-disagreement contracts.

Steps:
  1. Load ws_i_sample_818.csv
  2. Run slither in parallel (4 workers) with 30s timeout each
  3. Save raw results to ws_i_slither_results.csv
  4. Compute per-class agreement (BCCC label vs slither hit set)
  5. Identify 30 worst-disagreement contracts (highest |bccc_pos - slither_pos|)
  6. Save disagreement sample back to ws_i_sample_818.csv (appended to placeholder)
  7. Print summary + write ws_i_agreement_report.md

NOTE: Slither 0.11+ returns findings as a LIST of dicts (not a dict[check, list]).
Each finding has 'check' (detector name), 'impact', 'confidence', 'description', etc.
We deduplicate by 'check' to get the hit set.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]  # scripts -> Phase3 -> ... -> sentinel
PHASE3_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs"
PHASE3_OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_IN = PHASE3_OUT / "ws_i_sample_818.csv"
SLITHER_RESULTS_OUT = PHASE3_OUT / "ws_i_slither_results.csv"
SAMPLE_OUT = PHASE3_OUT / "ws_i_sample_848.csv"  # 818 + 30 disagreement
AGREEMENT_REPORT = PHASE3_OUT / "ws_i_agreement_report.md"

CLASS_COLS = [
    "Class01:ExternalBug", "Class02:GasException", "Class03:MishandledException",
    "Class04:Timestamp", "Class06:UnusedReturn", "Class08:CallToUnknown",
    "Class09:DenialOfService", "Class10:IntegerUO", "Class11:Reentrancy",
    "Class12:NonVulnerable",
]

# Slither detector -> set of BCCC classes it implies
SLITHER_TO_BCCC = {
    "arbitrary-send-eth": {"Class01:ExternalBug"},
    "arbitrary-send-erc20": {"Class01:ExternalBug"},
    "arbitrary-send-erc20-permit": {"Class01:ExternalBug"},
    "controlled-delegatecall": {"Class01:ExternalBug"},
    "delegatecall-loop": {"Class01:ExternalBug"},
    "msg-value-loop": {"Class01:ExternalBug", "Class09:DenialOfService"},
    "void-cst": {"Class02:GasException"},
    "constant-function-asm": {"Class02:GasException"},
    "constant-function-state": {"Class02:GasException"},
    "events-maths": {"Class02:GasException"},
    "locked-ether": {"Class02:GasException", "Class03:MishandledException"},
    "incorrect-return": {"Class03:MishandledException"},
    "uninitialized-state": {"Class03:MishandledException"},
    "uninitialized-storage": {"Class03:MishandledException"},
    "uninitialized-local": {"Class03:MishandledException"},
    "mapping-deletion": {"Class03:MishandledException"},
    "modifying-storage-array-by-value": {"Class03:MishandledException"},
    "timestamp": {"Class04:Timestamp"},
    "weak-prng": {"Class04:Timestamp"},
    "block-timestamp": {"Class04:Timestamp"},
    "unchecked-transfer": {"Class06:UnusedReturn"},
    "unchecked-send": {"Class06:UnusedReturn"},
    "unchecked-lowlevel": {"Class06:UnusedReturn"},
    "unused-return": {"Class06:UnusedReturn"},
    "missing-zero-check": {"Class08:CallToUnknown"},
    "uninitialized-fptr": {"Class08:CallToUnknown"},
    "calls-loop": {"Class09:DenialOfService"},
    "divide-before-multiply": {"Class10:IntegerUO"},
    "incorrect-exp": {"Class10:IntegerUO"},
    "tautological-compare": {"Class10:IntegerUO"},
    "incorrect-equality": {"Class10:IntegerUO"},
    "strict-equality": {"Class10:IntegerUO"},
    "out-of-bounds-array": {"Class10:IntegerUO"},
    "shift-parameter": {"Class10:IntegerUO"},
    "reentrancy-eth": {"Class11:Reentrancy"},
    "reentrancy-no-eth": {"Class11:Reentrancy"},
    "reentrancy-unlimited-gas": {"Class11:Reentrancy"},
    "reentrancy-benign": {"Class11:Reentrancy"},
    "reentrancy-events": {"Class11:Reentrancy"},
    "reentrancy-read-before-write": {"Class11:Reentrancy"},
    "tx-origin": {"Class11:Reentrancy"},
    "suicidal": {"Class11:Reentrancy"},
}

# Reverse: BCCC class -> set of slither detectors that map to it
BCCC_TO_SLITHER = defaultdict(set)
for det, classes in SLITHER_TO_BCCC.items():
    for c in classes:
        BCCC_TO_SLITHER[c].add(det)


# ---- Slither run (reuses logic from Stage 1) ----

SLITHER_TIMEOUT = 30
DEFAULT_SOLC_VERSIONS = ["0.5.17", "0.4.26", "0.6.12", "0.7.6", "0.8.20"]


def _verify_solc_works(version: str) -> bool:
    try:
        r = subprocess.run(
            ["solc-select", "use", version],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _verified_default_solc() -> str:
    for v in DEFAULT_SOLC_VERSIONS:
        if _verify_solc_works(v):
            return v
    return "0.5.17"


def pick_solc_version(pragma) -> str:
    if pragma is None:
        return _verified_default_solc()
    if isinstance(pragma, float) and pd.isna(pragma):
        return _verified_default_solc()
    if not isinstance(pragma, str):
        return _verified_default_solc()
    s = pragma.strip()
    if s == "" or s == "False" or s.lower() == "nan":
        return _verified_default_solc()

    solc_dir = Path.home() / ".solc-select" / "artifacts"
    if not solc_dir.exists():
        return _verified_default_solc()
    installed = []
    for p in solc_dir.iterdir():
        if p.name.startswith("solc-"):
            v = p.name[5:]
            if _verify_solc_works(v):
                installed.append(v)

    def parse_ver(v):
        try:
            return tuple(int(x) for x in v.split(".")[:3])
        except Exception:
            return (0, 0, 0)

    installed.sort(key=parse_ver, reverse=True)

    upper_match = re.search(r"<\s*(\d+\.\d+(?:\.\d+)?)", pragma)
    lower_match = re.search(r">=?\s*(\d+\.\d+(?:\.\d+)?)", pragma)
    caret_match = re.search(r"\^\s*(\d+\.\d+(?:\.\d+)?)", pragma)
    exact_match = re.fullmatch(r"\s*(\d+\.\d+(?:\.\d+)?)\s*", pragma)

    if exact_match:
        exact_v = exact_match.group(1)
        if exact_v in installed:
            return exact_v
        try:
            parts = exact_v.split(".")
            major_minor = (int(parts[0]), int(parts[1]))
            candidates = [v for v in installed if (parse_ver(v)[0], parse_ver(v)[1]) == major_minor]
            if candidates:
                candidates.sort(key=parse_ver, reverse=True)
                return candidates[0]
        except Exception:
            pass

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

    for v in installed:
        pv = parse_ver(v)
        if lower and pv < parse_ver(lower):
            continue
        if upper and pv >= parse_ver(upper):
            continue
        return v

    for v in installed:
        if v.startswith("0.5."):
            return v
    if installed:
        return installed[0]
    return _verified_default_solc()


def _run_one(contract_path: str, pragma, timeout: int = SLITHER_TIMEOUT) -> dict:
    """Run slither on a single contract. Returns dict with status, hits, hit_counts, etc.

    CRITICAL: Passes solc binary path directly to Slither() — does NOT use
    solc-select use (which is GLOBAL and breaks parallel execution where
    multiple workers would stomp on each other's solc version).
    """
    solc_version = pick_solc_version(pragma)
    # Build full path to the solc binary
    solc_path = str(Path.home() / ".solc-select" / "artifacts" / f"solc-{solc_version}" / f"solc-{solc_version}")
    if not Path(solc_path).exists():
        # Fallback: maybe the binary has a different layout
        alt = Path.home() / ".solc-select" / "artifacts" / f"solc-{solc_version}" / "solc"
        if alt.exists():
            solc_path = str(alt)
        else:
            return {
                "path": contract_path, "status": "EXCEPTION",
                "hits": [], "hit_counts": {}, "n_detectors": 0,
                "elapsed_sec": 0.0, "err": f"solc binary not found: {solc_path}",
                "solc_version": solc_version,
            }
    result = {
        "path": contract_path,
        "status": "PENDING",
        "hits": [],
        "hit_counts": {},
        "n_detectors": 0,
        "elapsed_sec": 0.0,
        "err": None,
        "solc_version": solc_version,
    }
    start = time.time()
    driver = None
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="slither_")
        driver = Path(tmpdir) / "_run.py"
        # Pass solc binary path directly to Slither() — no global switch
        driver.write_text(f"""
import json, sys
from slither import Slither
from slither.detectors import all_detectors as _ad
from slither.detectors.abstract_detector import AbstractDetector

_classes = []
for _n in dir(_ad):
    _o = getattr(_ad, _n)
    if isinstance(_o, type):
        try:
            if issubclass(_o, AbstractDetector) and _o is not AbstractDetector:
                _classes.append(_o)
        except Exception:
            pass

slither = Slither({contract_path!r}, solc={solc_path!r})
for _c in _classes:
    slither.register_detector(_c)
findings = slither.run_detectors()

# findings is a list (one entry per detector); each entry is a LIST of finding-dicts.
# Each finding-dict has 'check' (detector name slug), 'impact', 'confidence', 'description'.
hits = []
hit_counts = {{}}
for det_findings in findings:
    if not det_findings:
        continue
    for f in det_findings:
        if isinstance(f, dict):
            check = f.get("check")
        else:
            check = getattr(f, "check", None)
        if check:
            hits.append(check)
            hit_counts[check] = hit_counts.get(check, 0) + 1
print(json.dumps({{"status": "OK", "hits": hits, "hit_counts": hit_counts, "n_detectors": len(_classes)}}))
""")
        proc = subprocess.run(
            [sys.executable, str(driver)],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
        result["elapsed_sec"] = time.time() - start
        if proc.returncode != 0:
            err_short = (proc.stderr or "")[:500]
            low = err_short.lower()
            if any(k in low for k in ("compilation", "parse", "syntax", "import", "source file requires")):
                result["status"] = "COMPILE_ERROR"
            elif "timeout" in low:
                result["status"] = "TIMEOUT"
            else:
                result["status"] = "EXCEPTION"
            result["err"] = err_short
        else:
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
    finally:
        try:
            if driver and driver.exists():
                driver.unlink()
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
    return result


def _worker(args):
    row_idx, row = args
    path = row["bccc_path_fixed"]
    pragma = row.get("pragma", "")
    if not os.path.exists(path):
        return row_idx, {"path": path, "status": "PATH_MISSING", "hits": [], "hit_counts": {}, "n_detectors": 0, "elapsed_sec": 0.0, "err": "path missing", "solc_version": ""}
    r = _run_one(path, pragma)
    return row_idx, r


def run_parallel(sample: pd.DataFrame, n_workers: int = 4, save_every: int = 50) -> pd.DataFrame:
    """Run slither in parallel. Returns the sample df with slither results merged.

    Saves progress to SLITHER_RESULTS_OUT every `save_every` contracts so we don't
    lose progress on long runs / killed processes.
    """
    # Initialize result columns if not present
    for col, default in [
        ("slither_status", ""),
        ("slither_hits", ""),
        ("slither_elapsed_sec", 0.0),
        ("slither_solc", ""),
        ("slither_n_detectors", 0),
    ]:
        if col not in sample.columns:
            sample[col] = default

    args = list(sample.iterrows())
    results = [None] * len(args)
    completed = 0
    total = len(args)
    start = time.time()

    print(f"Running slither on {total} contracts with {n_workers} workers (timeout {SLITHER_TIMEOUT}s each)...")

    with mp.Pool(processes=n_workers) as pool:
        for row_idx, r in pool.imap_unordered(_worker, args, chunksize=1):
            results[row_idx] = r
            completed += 1
            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                status_counts = Counter(rr["status"] for rr in results if rr is not None)
                print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed, {rate:.2f}/s, ETA {eta:.0f}s | {dict(status_counts)}", flush=True)
            # Incremental save
            if save_every > 0 and completed % save_every == 0:
                _incremental_save(sample, results, SLITHER_RESULTS_OUT)

    # Merge results back into sample
    for i, r in enumerate(results):
        sample.at[i, "slither_status"] = r["status"]
        sample.at[i, "slither_hits"] = json.dumps(r["hits"])
        sample.at[i, "slither_elapsed_sec"] = round(r["elapsed_sec"], 2)
        sample.at[i, "slither_solc"] = r.get("solc_version", "")
        sample.at[i, "slither_n_detectors"] = r.get("n_detectors", 0)

    return sample


def _incremental_save(sample: pd.DataFrame, results: list, out_path: Path):
    """Save partial results to disk. Used to checkpoint progress."""
    out_cols = ["id", "bccc_path_fixed", "sample_reason", "primary_class", "n_pos",
                "slither_status", "slither_hits", "slither_elapsed_sec",
                "slither_solc", "slither_n_detectors"]
    # Build a copy with results merged
    save_df = sample.copy()
    for i, r in enumerate(results):
        if r is None:
            continue
        save_df.at[i, "slither_status"] = r["status"]
        save_df.at[i, "slither_hits"] = json.dumps(r["hits"])
        save_df.at[i, "slither_elapsed_sec"] = round(r["elapsed_sec"], 2)
        save_df.at[i, "slither_solc"] = r.get("solc_version", "")
        save_df.at[i, "slither_n_detectors"] = r.get("n_detectors", 0)
    save_df[out_cols].to_csv(out_path, index=False)


# ---- Agreement metrics ----

def compute_agreement(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """For each contract, compute per-class agreement between BCCC label and slither hits.

    Returns:
        per_class_df: rows = classes, cols = [n_bccc_pos, n_slither_pos, n_both,
                        n_bccc_only, n_slither_only, n_neither, precision, recall, f1]
        overall: dict with macro-F1, micro-F1, accuracy
    """
    # Filter to contracts where slither ran successfully
    ok = sample[sample["slither_status"] == "OK"].copy()
    print(f"\nAgreement metrics: {len(ok)}/{len(sample)} contracts with slither OK")

    # For each contract, build BCCC label set + slither-implied BCCC set
    def bccc_labels(row):
        return {c for c in CLASS_COLS if row[c] == 1}

    def slither_implied(row):
        hits = json.loads(row["slither_hits"]) if row["slither_hits"] else []
        implied = set()
        for h in hits:
            for c in SLITHER_TO_BCCC.get(h, set()):
                implied.add(c)
        return implied

    ok["bccc_set"] = ok.apply(bccc_labels, axis=1)
    ok["slither_set"] = ok.apply(slither_implied, axis=1)

    # Per-class: for each class, compute TP/FP/FN/TN
    rows = []
    for cls in CLASS_COLS:
        bccc_pos = ok["bccc_set"].apply(lambda s: cls in s)
        slither_pos = ok["slither_set"].apply(lambda s: cls in s)
        n_bccc = int(bccc_pos.sum())
        n_sli = int(slither_pos.sum())
        tp = int((bccc_pos & slither_pos).sum())
        fp = int((~bccc_pos & slither_pos).sum())
        fn = int((bccc_pos & ~slither_pos).sum())
        tn = int((~bccc_pos & ~slither_pos).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "class": cls,
            "n_bccc_pos": n_bccc,
            "n_slither_pos": n_sli,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })

    per_class = pd.DataFrame(rows)

    # Overall: macro-F1 (skip NonVulnerable as it's not a "vuln" class)
    vuln_classes = [c for c in CLASS_COLS if c != "Class12:NonVulnerable"]
    macro_f1 = per_class[per_class["class"].isin(vuln_classes)]["f1"].mean()
    micro_tp = per_class["TP"].sum()
    micro_fp = per_class["FP"].sum()
    micro_fn = per_class["FN"].sum()
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    overall = {
        "n_contracts_ok": len(ok),
        "n_contracts_total": len(sample),
        "compile_fail_rate": round((len(sample) - len(ok)) / len(sample), 4),
        "macro_f1_vuln_only": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
    }

    return per_class, overall


# ---- Disagreement score ----

def compute_disagreement_score(row) -> float:
    """Score = |bccc_pos_count - slither_implied_count| / (n_classes - 1) if bccc_set differs from slither_set.

    Range: 0.0 = full agreement, 1.0 = full disagreement.
    """
    bccc_set = {c for c in CLASS_COLS if row[c] == 1}
    hits = json.loads(row["slither_hits"]) if row["slither_hits"] else []
    slither_set = set()
    for h in hits:
        for c in SLITHER_TO_BCCC.get(h, set()):
            slither_set.add(c)

    # Symmetric difference
    sym_diff = bccc_set.symmetric_difference(slither_set)
    if not bccc_set and not slither_set:
        return 0.0
    return len(sym_diff) / (2 * (len(CLASS_COLS) - 1))  # normalize


# ---- Main ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N contracts (0=all)")
    parser.add_argument("--skip-run", action="store_true", help="Skip slither run, use existing results")
    parser.add_argument("--from-results", type=str, default="", help="Path to existing results CSV to load")
    args = parser.parse_args()

    # Load sample
    print(f"Loading sample from {SAMPLE_IN}")
    sample = pd.read_csv(SAMPLE_IN)
    print(f"  Loaded {len(sample)} contracts")

    if args.limit > 0:
        sample = sample.head(args.limit).copy()
        print(f"  Limited to {len(sample)} contracts")

    # Run slither
    if args.skip_run and args.from_results:
        print(f"\nLoading existing results from {args.from_results}")
        existing = pd.read_csv(args.from_results)
        # Merge by id
        for col in ["slither_status", "slither_hits", "slither_elapsed_sec", "slither_solc", "slither_n_detectors"]:
            if col in existing.columns:
                sample[col] = sample["id"].map(existing.set_index("id")[col].to_dict())
    else:
        sample = run_parallel(sample, n_workers=args.workers)

    # Save raw slither results
    out_cols_raw = ["id", "bccc_path_fixed", "sample_reason", "primary_class", "n_pos",
                    "slither_status", "slither_hits", "slither_elapsed_sec",
                    "slither_solc", "slither_n_detectors"]
    sample[out_cols_raw].to_csv(SLITHER_RESULTS_OUT, index=False)
    print(f"\nSaved raw slither results to {SLITHER_RESULTS_OUT}")

    # Compute agreement
    per_class, overall = compute_agreement(sample)
    print("\n=== Per-class agreement ===")
    print(per_class.to_string(index=False))
    print(f"\n=== Overall ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    # Compute disagreement scores
    sample["disagreement_score"] = sample.apply(compute_disagreement_score, axis=1)

    # Identify 30 worst-disagreement contracts (with slither OK)
    ok_sample = sample[sample["slither_status"] == "OK"].copy()
    worst = ok_sample.nlargest(30, "disagreement_score")
    print(f"\n=== 30 worst-disagreement contracts ===")
    for _, r in worst.iterrows():
        print(f"  {r['id'][:16]}... score={r['disagreement_score']:.3f} | bccc_set_size={int(sum(r[c]==1 for c in CLASS_COLS))} slither_hits={len(json.loads(r['slither_hits'])) if r['slither_hits'] else 0} | reason={r['sample_reason']}")

    # Add 30 worst as "disagreement" bucket to the sample
    worst_labeled = worst.copy()
    worst_labeled["sample_reason"] = "disagreement_post_slither"
    full = pd.concat([sample, worst_labeled], ignore_index=True)
    full.to_csv(SAMPLE_OUT, index=False)
    print(f"\nSaved full sample (808 + 30 = 838) to {SAMPLE_OUT}")

    # Write agreement report
    with open(AGREEMENT_REPORT, "w") as f:
        f.write(f"# WS-I Slither Label Validation — Agreement Report\n\n")
        f.write(f"**Date:** 2026-06-06\n")
        f.write(f"**Total contracts:** {overall['n_contracts_total']}\n")
        f.write(f"**Slither OK:** {overall['n_contracts_ok']} ({round(100*overall['n_contracts_ok']/overall['n_contracts_total'], 1)}%)\n")
        f.write(f"**Compile fail rate:** {round(100*overall['compile_fail_rate'], 1)}%\n\n")
        f.write(f"## Overall\n\n")
        f.write(f"- **Macro-F1 (vuln classes only):** {overall['macro_f1_vuln_only']}\n")
        f.write(f"- **Micro-F1:** {overall['micro_f1']}\n")
        f.write(f"- **Micro-precision:** {overall['micro_precision']}\n")
        f.write(f"- **Micro-recall:** {overall['micro_recall']}\n\n")
        f.write(f"## Per-class\n\n")
        f.write(per_class.to_markdown(index=False))
        f.write(f"\n\n## 30 worst-disagreement contracts\n\n")
        f.write(worst[["id", "sample_reason", "primary_class", "n_pos", "slither_status",
                      "slither_hits", "disagreement_score"]].to_markdown(index=False))
        f.write(f"\n\n## Interpretation\n\n")
        f.write(f"_Filled in after manual review._\n")
    print(f"\nWrote agreement report to {AGREEMENT_REPORT}")

    print("\n=== Stage 2 done ===")
    print("Next: manually review the 30 worst-disagreement contracts + 2 maxing contracts.")
