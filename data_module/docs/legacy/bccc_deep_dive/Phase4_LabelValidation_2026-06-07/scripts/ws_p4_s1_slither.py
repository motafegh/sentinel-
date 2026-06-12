"""
Stage 1 — Run Slither on the Stage 1 sample (~10,693 contracts).

V2 — Rewritten using ML pipeline patterns (ml/src/preprocessing/graph_extractor.py):
  - Version-grouped processing (batch all contracts of same solc version together)
  - Pinned solc binary per version group (.venv/.solc-select/artifacts/)
  - --allow-paths flag for solc 0.5+ to resolve imports
  - Better error handling (COMPILE_ERROR vs EXCEPTION)

Output: ws_p4_s1_slither_results.csv (one row per contract)
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
SAMPLE = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_sample.csv"
SRC = REPO / "BCCC-SCsVul-2024/SourceCodes"
OUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_slither_results.csv"
CHECKPOINT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_slither_checkpoint.jsonl"

SLITHER_TIMEOUT = 30

# ── Solc binary resolution (from ml/src/data_extraction/ast_extractor.py) ──────

def get_solc_binary(version: str):
    """Resolve pinned solc binary path from .venv/.solc-select/artifacts/."""
    if not version:
        return None
    venv_path = REPO / ".venv" / ".solc-select" / "artifacts" / f"solc-{version}"
    for p in [venv_path / f"solc-{version}", venv_path / "solc"]:
        if p.exists():
            return str(p)
    return None


def parse_solc_version(version: str):
    """Parse a solc version string into (major, minor, patch)."""
    try:
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
        if match:
            return tuple(int(x) for x in match.groups())
    except Exception:
        pass
    return (0, 0, 0)


def solc_supports_allow_paths(version: str) -> bool:
    """Return True if solc version supports --allow-paths (0.5.0+)."""
    major, minor, _ = parse_solc_version(version)
    return (major, minor) >= (0, 5)


def pick_solc_version(pragma: str) -> str:
    """Determine the best solc version for a given pragma string."""
    if not pragma or pragma == "False":
        return "0.8.28"
    s = str(pragma).strip()
    if not s or s.lower() == "nan":
        return "0.8.28"

    # Try each pattern
    caret_match = re.search(r"\^\s*(\d+\.\d+\.\d+)", s)
    tilde_match = re.search(r"~\s*(\d+\.\d+\.\d+)", s)
    exact_match = re.search(r"(\d+\.\d+\.\d+)", s)

    if caret_match:
        base = caret_match.group(1)
        parts = base.split(".")
        major, minor = int(parts[0]), int(parts[1])
        # Use highest available in this major.minor series
        version_map = {
            (0, 4): "0.4.26", (0, 5): "0.5.17", (0, 6): "0.6.12",
            (0, 7): "0.7.6", (0, 8): "0.8.28",
        }
        return version_map.get((major, minor), base)

    if tilde_match:
        return tilde_match.group(1)

    if exact_match:
        return exact_match.group(1)

    # Try >= ... < pattern
    range_match = re.search(r">=\s*(\d+\.\d+\.\d+)", s)
    if range_match:
        return range_match.group(1)

    # Fallback: any version-like string
    any_match = re.search(r"(\d+\.\d+(?:\.\d+)?)", s)
    if any_match:
        v = any_match.group(1)
        if v.count(".") == 1:
            v += ".0"
        return v

    return "0.8.28"


def fix_path(stored_path: str) -> str:
    if stored_path.startswith("BCCC-SCsVul-2024/Source Codes/"):
        return str(SRC / stored_path.replace("BCCC-SCsVul-2024/Source Codes/", ""))
    if stored_path.startswith("BCCC-SCsVul-2024/SourceCodes/"):
        return str(SRC / stored_path.replace("BCCC-SCsVul-2024/SourceCodes/", ""))
    return stored_path


# ── Worker (runs in subprocess) ───────────────────────────────────────────────

def _slither_worker(args):
    """Run slither on a single .sol file.

    Uses the ML pipeline approach:
      - Passes --allow-paths for solc 0.5+ to resolve imports
      - Uses pinned solc binary instead of solc-select switching
      - Better error classification (COMPILE_ERROR vs EXCEPTION)
    """
    contract_id, contract_path, solc_version, timeout = args
    result = {
        "id": contract_id,
        "status": "PENDING",
        "hits": [],
        "hit_counts": {},
        "n_detectors": 0,
        "elapsed_sec": 0.0,
        "err": "",
        "solc_version": solc_version,
    }
    start = time.time()
    try:
        if not Path(contract_path).exists():
            result["status"] = "PATH_MISSING"
            result["err"] = "file not found"
            return result

        venv_bin = str(Path(sys.executable).parent)
        solc_binary = get_solc_binary(solc_version)

        with tempfile.TemporaryDirectory() as tmpdir:
            driver = Path(tmpdir) / "_run_slither.py"
            result_file = Path(tmpdir) / "_result.json"

            # Build solc_args: --allow-paths for solc 0.5+
            solc_args_line = ""
            if solc_supports_allow_paths(solc_version) and solc_binary:
                solc_args_line = f'solc_args="--allow-paths .,{REPO}"'

            # Use pinned solc binary if available, otherwise fall back to solc-select
            if solc_binary:
                solc_init_line = f"# Using pinned solc binary: {solc_binary}"
            else:
                solc_init_line = (
                    f"import subprocess as _sp\n"
                    f'_sp.run(["solc-select", "use", {solc_version!r}], '
                    f"check=False, capture_output=True)"
                )

            driver.write_text(f"""
import json, sys, os

{solc_init_line}

from slither import Slither
from slither.detectors import all_detectors
import slither.detectors.all_detectors as _ad
from slither.detectors.abstract_detector import AbstractDetector

_dets = []
for _n in dir(_ad):
    _o = getattr(_ad, _n)
    if isinstance(_o, type):
        try:
            if issubclass(_o, AbstractDetector) and _o is not AbstractDetector:
                _dets.append(_o)
        except Exception:
            pass

try:
    slither = Slither(
        {contract_path!r},
        solc={solc_binary!r} if {solc_binary!r} else None,
        {solc_args_line}
    )
    for _c in _dets:
        slither.register_detector(_c)
    dr = slither.run_detectors()
    hits = []
    hc = {{}}
    for det_findings in dr:
        if not isinstance(det_findings, list):
            det_findings = [det_findings]
        for finding in det_findings:
            check = finding.get("check") if isinstance(finding, dict) else None
            if check:
                hits.append(check)
                hc[check] = hc.get(check, 0) + 1
    with open("{result_file}", "w") as f:
        json.dump({{"status": "OK", "hits": hits, "hit_counts": hc, "n_detectors": len(_dets)}}, f)
except Exception as e:
    err_type = type(e).__name__
    err_msg = str(e)[:500]
    err_lower = err_msg.lower()
    if any(kw in err_lower for kw in ["compil", "syntax", "invalid solidity", "parsing"]):
        status = "COMPILE_ERROR"
    else:
        status = "EXCEPTION"
    with open("{result_file}", "w") as f:
        json.dump({{"status": status, "hits": [], "hit_counts": {{}}, "n_detectors": 0, "err": f"{{err_type}}: {{err_msg}}"}}, f)
""")

            sub_env = os.environ.copy()
            sub_env["PATH"] = venv_bin + os.pathsep + sub_env.get("PATH", "")

            proc = subprocess.run(
                [sys.executable, str(driver)],
                capture_output=True, text=True, timeout=timeout,
                env=sub_env,
                cwd=str(REPO),
            )
            result["elapsed_sec"] = time.time() - start

            if result_file.exists():
                try:
                    parsed = json.loads(result_file.read_text())
                    result["status"] = parsed.get("status", "OK")
                    result["hits"] = parsed.get("hits", [])
                    result["hit_counts"] = parsed.get("hit_counts", {})
                    result["n_detectors"] = parsed.get("n_detectors", 0)
                    if "err" in parsed:
                        result["err"] = parsed["err"]
                except Exception as e:
                    result["status"] = "EXCEPTION"
                    result["err"] = f"result parse fail: {e}"
            else:
                err_short = (proc.stderr or proc.stdout or "")[:500]
                err_lower = err_short.lower()
                if any(kw in err_lower for kw in ["compilation", "parse", "syntax"]):
                    result["status"] = "COMPILE_ERROR"
                else:
                    result["status"] = "EXCEPTION"
                result["err"] = err_short
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["elapsed_sec"] = timeout
        result["err"] = f"timeout after {timeout}s"
    except Exception as e:
        result["status"] = "EXCEPTION"
        result["err"] = f"{type(e).__name__}: {str(e)[:200]}"
        result["elapsed_sec"] = time.time() - start
    return result


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def load_checkpoint():
    done = set()
    if CHECKPOINT.exists():
        with CHECKPOINT.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["id"])
                except Exception:
                    pass
    return done


def append_checkpoint(result):
    with CHECKPOINT.open("a") as f:
        f.write(json.dumps(result, default=str) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Stage 1 Slither (V2 with allow-paths)")
    ap.add_argument("--inp", type=Path, default=SAMPLE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=SLITHER_TIMEOUT)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N (for testing)")
    ap.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start over")
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input not found: {args.inp}", file=sys.stderr)
        return 1

    with args.inp.open() as f:
        rows = list(csv.DictReader(f))
    sample = [r for r in rows if r.get("in_stage1_sample") == "1"]
    if args.limit:
        sample = sample[:args.limit]
    print(f"Loaded {len(rows)} rows; {len(sample)} in Stage 1 sample"
          + (f" (limited to {args.limit})" if args.limit else ""))

    if args.fresh and args.checkpoint.exists():
        args.checkpoint.unlink()
        print("Cleared checkpoint (--fresh)")

    done = load_checkpoint()
    todo = [r for r in sample if r["id"] not in done]
    print(f"Already done: {len(done)}; to do: {len(todo)}")

    if not todo:
        print("All done. Use --fresh to redo.")
        return 0

    # Group by solc version for version-grouped processing
    for r in todo:
        r["_solc_version"] = pick_solc_version(r.get("pragma", ""))

    version_groups = {}
    for r in todo:
        sv = r["_solc_version"]
        if sv not in version_groups:
            version_groups[sv] = []
        version_groups[sv].append(r)

    print(f"\nVersion groups ({len(version_groups)}):")
    for sv in sorted(version_groups.keys()):
        vbin = get_solc_binary(sv)
        allow = "yes" if solc_supports_allow_paths(sv) else "no"
        print(f"  v{sv}: {len(version_groups[sv])} contracts, "
              f"binary={'found' if vbin else 'MISSING'}, --allow-paths={allow}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Process each version group
    t0 = time.time()
    completed = 0
    total = len(todo)

    for sv in sorted(version_groups.keys()):
        group = version_groups[sv]
        print(f"\n{'='*60}")
        print(f"Processing v{sv}: {len(group)} contracts")
        print(f"{'='*60}")

        work = [(r["id"], fix_path(r["bccc_file_path"]), sv, args.timeout) for r in group]

        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(_slither_worker, work, chunksize=4):
                append_checkpoint(result)
                completed += 1
                elapsed = time.time() - t0
                avg = elapsed / completed
                eta_min = (total - completed) * avg / 60 / args.workers
                print(f"  [{completed}/{total}] {result['id'][:16]}.. "
                      f"status={result['status']} hits={len(result['hits'])} "
                      f"elapsed={result['elapsed_sec']:.1f}s "
                      f"avg={avg:.1f}s ETA={eta_min:.0f}min", flush=True)

    print(f"\nDone {completed} contracts in {(time.time()-t0)/60:.1f} min")

    # Write final CSV
    all_results = []
    with args.checkpoint.open() as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except Exception:
                pass

    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "status", "n_hits", "n_detectors", "hits_json", "hit_counts_json",
                    "elapsed_sec", "solc_version", "err"])
        for r in all_results:
            w.writerow([r["id"], r["status"], len(r["hits"]), r.get("n_detectors", 0),
                        json.dumps(r["hits"]), json.dumps(r["hit_counts"]),
                        f"{r['elapsed_sec']:.2f}", r.get("solc_version", ""), r.get("err", "")[:200]])
    print(f"Wrote {args.out} ({len(all_results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
