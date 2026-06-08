"""Stage 1 — Run Aderyn on the Stage 1 sample.

V2 fix: per-contract processing with parallel workers.
Root cause of V1 COMPILE_FAIL: aderyn compiles ALL files in a directory together,
and one broken contract fails the entire batch.
V2: each contract gets its own temp dir, processed in isolation.
"""
import csv, json, os, shutil, subprocess, sys, time, tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count

REPO = Path("/home/motafeq/projects/sentinel")
SAMPLE = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_sample.csv"
SRC = REPO / "BCCC-SCsVul-2024/SourceCodes"
OUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_aderyn_results.csv"
CHECKPOINT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_aderyn_checkpoint.jsonl"
ADERYN = Path.home() / ".cargo" / "bin" / "aderyn"
ADERYN_TIMEOUT = 60


def fix_path(stored_path: str) -> str:
    stored_path = stored_path.replace("BCCC-SCsVul-2024/Source Codes/", "")
    stored_path = stored_path.replace("BCCC-SCsVul-2024/SourceCodes/", "")
    return str(SRC / stored_path)


def process_one(row: tuple) -> dict:
    idx, cid, src_path = row
    try:
        with tempfile.TemporaryDirectory(prefix="aderyn_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            ext = ".sol"
            fname = f"{cid[:16]}{ext}"
            dest = tmpdir / fname
            try:
                shutil.copy2(src_path, dest)
            except Exception as e:
                return {"id": cid, "status": "PATH_MISSING", "n_hits": 0, "hits": [],
                        "elapsed_sec": 0.0, "err": f"copy failed: {e}"}

            start = time.time()
            report_path = tmpdir / "report.json"
            try:
                proc = subprocess.run(
                    [str(ADERYN), str(tmpdir), "-o", str(report_path)],
                    capture_output=True, text=True, timeout=ADERYN_TIMEOUT,
                )
                elapsed = time.time() - start
            except subprocess.TimeoutExpired:
                return {"id": cid, "status": "TIMEOUT", "n_hits": 0, "hits": [],
                        "elapsed_sec": ADERYN_TIMEOUT, "err": f"timeout {ADERYN_TIMEOUT}s"}
            except Exception as e:
                return {"id": cid, "status": "EXCEPTION", "n_hits": 0, "hits": [],
                        "elapsed_sec": time.time() - start, "err": f"{type(e).__name__}: {e}"}

            if not report_path.exists():
                err_msg = (proc.stdout + proc.stderr)[-500:]
                return {"id": cid, "status": "COMPILE_FAIL", "n_hits": 0, "hits": [],
                        "elapsed_sec": elapsed, "err": err_msg}

            try:
                data = json.loads(report_path.read_text())
            except Exception as e:
                return {"id": cid, "status": "PARSE_FAIL", "n_hits": 0, "hits": [],
                        "elapsed_sec": elapsed, "err": f"parse: {e}"}

            hits = []
            for severity in ["high_issues", "medium_issues", "low_issues"]:
                issues_dict = data.get(severity, {})
                if not isinstance(issues_dict, dict):
                    continue
                for issue in issues_dict.get("issues", []):
                    det = issue.get("detector_name", "")
                    for inst in issue.get("instances", []):
                        cp = inst.get("contract_path", "")
                        if fname in cp:
                            hits.append(det)

            return {"id": cid, "status": "OK", "n_hits": len(hits),
                    "hits": list(set(hits)),
                    "elapsed_sec": round(elapsed, 3), "err": ""}
    except Exception as e:
        return {"id": cid, "status": "EXCEPTION", "n_hits": 0, "hits": [],
                "elapsed_sec": 0.0, "err": f"{type(e).__name__}: {e}"}


def main():
    if not ADERYN.exists():
        print(f"ERROR: aderyn not found at {ADERYN}", file=sys.stderr)
        return 1

    with open(SAMPLE) as f:
        rows = list(csv.DictReader(f))
    sample = [r for r in rows if r.get("in_stage1_sample") == "1"]
    print(f"Loaded {len(rows)} rows; {len(sample)} in Stage 1 sample")

    done = set()
    if CHECKPOINT.exists():
        with CHECKPOINT.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["id"])
                except Exception:
                    pass
    print(f"Already done: {len(done)}")

    todo = [(i, r["id"], fix_path(r["bccc_file_path"]))
            for i, r in enumerate(sample) if r["id"] not in done
            and Path(fix_path(r["bccc_file_path"])).exists()]
    missing = [r["id"] for r in sample if r["id"] not in done
               and not Path(fix_path(r["bccc_file_path"])).exists()]
    if missing:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        for cid in missing:
            with CHECKPOINT.open("a") as f:
                f.write(json.dumps({"id": cid, "status": "PATH_MISSING", "n_hits": 0,
                                    "hits": [], "elapsed_sec": 0.0,
                                    "err": "source not found"}) + "\n")
        print(f"Missing source files: {len(missing)} (logged in checkpoint)")

    print(f"To do: {len(todo)}")

    if not todo:
        print("All done.")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, cpu_count() // 2)
    print(f"Processing with {n_workers} workers...")

    t0 = time.time()
    done_count = len(done) + len(missing)
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(process_one, todo, chunksize=5):
            with CHECKPOINT.open("a") as f:
                f.write(json.dumps(result) + "\n")
            done_count += 1
            elapsed = time.time() - t0
            rate = done_count / elapsed if elapsed > 0 else 0
            remaining = len(sample) - done_count
            eta = remaining / rate if rate > 0 else 0
            if done_count % 100 == 0:
                print(f"  {done_count}/{len(sample)}  rate={rate:.1f}/s  ETA={eta/60:.0f}min",
                      flush=True)

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f}min)")

    all_results = []
    with CHECKPOINT.open() as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except Exception:
                pass

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "status", "n_hits", "hits_json", "elapsed_sec", "err"])
        for r in all_results:
            w.writerow([r["id"], r["status"], r.get("n_hits", 0),
                        json.dumps(r.get("hits", [])),
                        r.get("elapsed_sec", 0), r.get("err", "")])
    print(f"Wrote {OUT} ({len(all_results)} rows)")

    statuses = {}
    for r in all_results:
        s = r["status"]
        statuses[s] = statuses.get(s, 0) + 1
    print(f"Status: {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
