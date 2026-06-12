"""Full-dataset Slither run — Phase 5 Gap Fix
Runs Slither on all 67,311 BCCC contracts. Skips the 10,693 already
analyzed in Phase 4. Checkpointed/resumable.

Run from repo root (use nohup or background):
    nohup python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/scripts/p5_gap_full_slither.py > /tmp/p5_slither.log 2>&1 &
    tail -f /tmp/p5_slither.log
"""
import csv, json, os, re, subprocess, sys, tempfile, time
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
SRC  = REPO / "BCCC-SCsVul-2024/SourceCodes"
BASE = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs"
P5   = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/outputs"

INPUT      = BASE / "ws_p4_s01b_d12_applied.csv"         # all 67,311
P4_DONE    = BASE / "ws_p4_s1_slither_results.csv"        # already done
OUT        = P5   / "p5_gap_full_slither_results.csv"
CHECKPOINT = P5   / "p5_gap_full_slither_checkpoint.jsonl"

SLITHER_TIMEOUT = 30
WORKERS = 6

# ─── Solc helpers (identical to Phase 4) ────────────────────────────────────

def get_solc_binary(version: str):
    if not version:
        return None
    venv_path = REPO / ".venv" / ".solc-select" / "artifacts" / f"solc-{version}"
    for p in [venv_path / f"solc-{version}", venv_path / "solc"]:
        if p.exists():
            return str(p)
    return None

def parse_solc_version(version: str):
    try:
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
        if m:
            return tuple(int(x) for x in m.groups())
    except Exception:
        pass
    return (0, 0, 0)

def solc_supports_allow_paths(version: str) -> bool:
    major, minor, _ = parse_solc_version(version)
    return (major, minor) >= (0, 5)

def pick_solc_version(pragma: str) -> str:
    if not pragma or str(pragma).strip().lower() in ("", "nan", "false"):
        return "0.8.28"
    s = str(pragma).strip()
    VERSION_MAP = {(0,4):"0.4.26",(0,5):"0.5.17",(0,6):"0.6.12",(0,7):"0.7.6",(0,8):"0.8.28"}
    for pattern, idx in [(r"\^\s*(\d+\.\d+\.\d+)", 1), (r"~\s*(\d+\.\d+\.\d+)", 1),
                         (r"(\d+\.\d+\.\d+)", 1)]:
        m = re.search(pattern, s)
        if m:
            base = m.group(1)
            parts = base.split(".")
            major, minor = int(parts[0]), int(parts[1])
            return VERSION_MAP.get((major, minor), base)
    m = re.search(r">=\s*(\d+\.\d+\.\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", s)
    if m:
        v = m.group(1)
        return v if v.count(".") == 2 else v + ".0"
    return "0.8.28"

def fix_path(stored_path: str) -> str:
    p = str(stored_path)
    for prefix in ("BCCC-SCsVul-2024/Source Codes/", "BCCC-SCsVul-2024/SourceCodes/"):
        if p.startswith(prefix):
            return str(SRC / p[len(prefix):])
    return p

# ─── Worker ─────────────────────────────────────────────────────────────────

def _slither_worker(args):
    contract_id, contract_path, solc_version, timeout = args
    result = {"id": contract_id, "status": "PENDING", "hits": [], "hit_counts": {},
              "n_detectors": 0, "elapsed_sec": 0.0, "err": "", "solc_version": solc_version}
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
            solc_args_line = ""
            if solc_supports_allow_paths(solc_version) and solc_binary:
                solc_args_line = f'solc_args="--allow-paths .,{REPO}"'
            solc_init_line = (f"# pinned: {solc_binary}" if solc_binary else
                              f"import subprocess as _sp\n_sp.run(['solc-select','use',{solc_version!r}],"
                              "check=False,capture_output=True)")
            driver.write_text(f"""
import json
{solc_init_line}
from slither import Slither
import slither.detectors.all_detectors as _ad
from slither.detectors.abstract_detector import AbstractDetector
_dets = [getattr(_ad,n) for n in dir(_ad)
         if isinstance(getattr(_ad,n),type) and
         issubclass(getattr(_ad,n),AbstractDetector) and
         getattr(_ad,n) is not AbstractDetector]
try:
    sl = Slither({contract_path!r},
                 solc={solc_binary!r} if {solc_binary!r} else None,
                 {solc_args_line})
    for c in _dets: sl.register_detector(c)
    dr = sl.run_detectors()
    hits=[]; hc={{}}
    for findings in dr:
        if not isinstance(findings,list): findings=[findings]
        for f in findings:
            ch = f.get("check") if isinstance(f,dict) else None
            if ch: hits.append(ch); hc[ch]=hc.get(ch,0)+1
    with open("{result_file}","w") as fp:
        json.dump({{"status":"OK","hits":hits,"hit_counts":hc,"n_detectors":len(_dets)}},fp)
except Exception as e:
    err_msg=str(e)[:500]
    status="COMPILE_ERROR" if any(k in err_msg.lower() for k in ["compil","syntax","invalid","parsing"]) else "EXCEPTION"
    with open("{result_file}","w") as fp:
        json.dump({{"status":status,"hits":[],"hit_counts":{{}},"n_detectors":0,"err":f"{{type(e).__name__}}:{{err_msg}}"}},fp)
""")
            sub_env = os.environ.copy()
            sub_env["PATH"] = venv_bin + os.pathsep + sub_env.get("PATH", "")
            proc = subprocess.run([sys.executable, str(driver)],
                                  capture_output=True, text=True, timeout=timeout,
                                  env=sub_env, cwd=str(REPO))
            result["elapsed_sec"] = time.time() - start
            if result_file.exists():
                parsed = json.loads(result_file.read_text())
                result.update({k: parsed.get(k, result[k]) for k in
                                ["status","hits","hit_counts","n_detectors","err"]})
            else:
                err_short = (proc.stderr or proc.stdout or "")[:500]
                result["status"] = ("COMPILE_ERROR" if any(k in err_short.lower()
                                    for k in ["compilation","parse","syntax"]) else "EXCEPTION")
                result["err"] = err_short
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"; result["elapsed_sec"] = timeout
        result["err"] = f"timeout after {timeout}s"
    except Exception as e:
        result["status"] = "EXCEPTION"; result["elapsed_sec"] = time.time() - start
        result["err"] = f"{type(e).__name__}: {str(e)[:200]}"
    return result

# ─── Checkpoint ─────────────────────────────────────────────────────────────

def load_checkpoint(path: Path):
    done = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line); done[r["id"]] = r
                except Exception: pass
    return done

def append_checkpoint(result, path: Path):
    with path.open("a") as f:
        f.write(json.dumps(result, default=str) + "\n")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--timeout", type=int, default=SLITHER_TIMEOUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    # Load all contracts
    with INPUT.open() as f:
        all_rows = list(csv.DictReader(f))
    print(f"Total contracts: {len(all_rows):,}")

    # Load Phase 4 already-done IDs
    p4_done = set()
    if P4_DONE.exists():
        with P4_DONE.open() as f:
            for row in csv.DictReader(f):
                p4_done.add(row["id"])
    print(f"Already done in Phase 4: {len(p4_done):,}")

    # Remaining contracts
    remaining = [r for r in all_rows if r["id"] not in p4_done]
    print(f"To process: {len(remaining):,}")

    if args.fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink(); print("Cleared checkpoint")

    checkpoint_done = load_checkpoint(CHECKPOINT)
    todo = [r for r in remaining if r["id"] not in checkpoint_done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Checkpoint done: {len(checkpoint_done):,}; remaining todo: {len(todo):,}")

    if not todo:
        print("Nothing to do — writing final CSV and exiting.")
    else:
        for r in todo:
            r["_solc_version"] = pick_solc_version(r.get("pragma", ""))

        version_groups = {}
        for r in todo:
            version_groups.setdefault(r["_solc_version"], []).append(r)

        print(f"\nVersion groups:")
        for sv in sorted(version_groups):
            vbin = get_solc_binary(sv)
            print(f"  v{sv}: {len(version_groups[sv]):,} contracts  "
                  f"binary={'found' if vbin else 'MISSING'}")

        P5.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        completed = 0
        total = len(todo)

        for sv in sorted(version_groups):
            group = version_groups[sv]
            print(f"\n{'='*60}\nv{sv}: {len(group):,} contracts\n{'='*60}")
            work = [(r["id"], fix_path(r["bccc_file_path"]), sv, args.timeout) for r in group]
            with Pool(args.workers) as pool:
                for result in pool.imap_unordered(_slither_worker, work, chunksize=4):
                    append_checkpoint(result, CHECKPOINT)
                    completed += 1
                    elapsed = time.time() - t0
                    avg = elapsed / completed
                    eta_min = (total - completed) * avg / 60 / args.workers
                    print(f"  [{completed}/{total}] {result['id'][:14]}.. "
                          f"status={result['status']} hits={len(result['hits'])} "
                          f"eta={eta_min:.0f}min", flush=True)

        print(f"\nDone {completed} in {(time.time()-t0)/60:.1f} min")

    # Write final CSV
    all_results = list(load_checkpoint(CHECKPOINT).values())
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","status","n_hits","n_detectors","hits_json",
                    "hit_counts_json","elapsed_sec","solc_version","err"])
        for r in all_results:
            w.writerow([r["id"], r["status"], len(r.get("hits",[])),
                        r.get("n_detectors",0), json.dumps(r.get("hits",[])),
                        json.dumps(r.get("hit_counts",{})),
                        f"{r.get('elapsed_sec',0):.2f}", r.get("solc_version",""),
                        str(r.get("err",""))[:200]])
    print(f"Wrote {OUT}  ({len(all_results):,} rows)")

if __name__ == "__main__":
    main()
