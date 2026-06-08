"""Phase 5 — Compile Error Retry
Fixes two categories of recoverable compile failures in p5_gap_full_slither_results.csv:

  1. Spaced pragma: BCCC CSV has pragmas like '^ 0.4 .9' (spaces) which our pick_solc_version
     returns '0.4.0' for (no binary). Fix: normalize spaces → use correct 0.4.26 binary.
     Success rate: ~90% of 196 contracts.

  2. Exact version mismatch: pragma like '0.4.25' (no ^/~/>=) means ONLY that exact solc
     works. We mapped (0,4) → 0.4.26 which fails. Fix: use exact version from file pragma.
     We have ALL solc patch versions installed (0.4.0 through 0.8.31).
     Success rate: ~70-80% of ~3,922 contracts.

Note: SPDX failures in original run are actually missing-import failures (multi-file contracts
that reference files not present in BCCC) — not fixable.

Appends successful retries to the checkpoint file and regenerates the CSV.

Run from repo root:
    python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/scripts/p5_compile_retry.py
"""
import csv, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
SRC  = REPO / "BCCC-SCsVul-2024/SourceCodes"
P5   = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/outputs"
INPUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01b_d12_applied.csv"

CHECKPOINT = P5 / "p5_gap_full_slither_checkpoint.jsonl"
OUT        = P5 / "p5_gap_full_slither_results.csv"
SLITHER_TIMEOUT = 30

# Exact patch for each minor version
_LATEST_PATCH = {
    "0.4": "0.4.26", "0.5": "0.5.17", "0.6": "0.6.12",
    "0.7": "0.7.6",  "0.8": "0.8.31",
}
_SOLC_ARTIFACTS = REPO / ".venv" / ".solc-select" / "artifacts"
_PRAGMA_RE = re.compile(r'pragma\s+solidity\s+([^\n;]+)', re.S)
_VER_RE    = re.compile(r'(\d+\.\d+\.\d+)')


def _get_binary(version: str) -> str | None:
    if not version:
        return None
    for name in [f"solc-{version}", "solc"]:
        p = _SOLC_ARTIFACTS / f"solc-{version}" / name
        if p.exists():
            return str(p)
    return None


def _detect_version_from_file(sol_path: Path) -> str:
    """Read pragma from actual .sol file, normalize spaces, map to latest patch."""
    try:
        content = sol_path.read_text(errors="ignore")
    except Exception:
        return "0.8.31"
    m = _PRAGMA_RE.search(content)
    if not m:
        return "0.8.31"
    pragma = m.group(1).strip()
    # Normalize spaces: '0.4 .9' or '0 .4 .9' → '0.4.9'
    pragma = re.sub(r'(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)', r'\1.\2.\3', pragma)
    pragma = re.sub(r'(\d+)\s*\.\s*(\d+)(?!\.\d)', r'\1.\2', pragma)

    # Check for exact version (no ^, ~, >=) — use that exact binary if available
    stripped = pragma.strip()
    if _VER_RE.fullmatch(stripped):
        # Exact version like '0.4.25' — try exact binary first
        if _get_binary(stripped):
            return stripped
        # Fallback to latest patch
        minor = ".".join(stripped.split(".")[:2])
        return _LATEST_PATCH.get(minor, "0.8.31")

    # Pick from caret/tilde/gte
    m2 = _VER_RE.search(pragma)
    if m2:
        ver = m2.group(1)
        minor = ".".join(ver.split(".")[:2])
        return _LATEST_PATCH.get(minor, ver)

    m3 = re.search(r'(\d+\.\d+)(?:\.\d+)?', pragma)
    if m3:
        minor = m3.group(1)
        return _LATEST_PATCH.get(minor, "0.8.31")
    return "0.8.31"


def _run_slither(contract_path: Path, solc_version: str) -> dict:
    """Run Slither on contract_path with given solc version. Returns result dict."""
    result = {"status": "PENDING", "hits": [], "hit_counts": {}, "n_detectors": 0,
              "elapsed_sec": 0.0, "err": "", "solc_version": solc_version}
    solc_binary = _get_binary(solc_version)
    if not solc_binary:
        result["status"] = "COMPILE_ERROR"
        result["err"] = f"no binary for {solc_version}"
        return result

    start = time.time()
    try:
        venv_bin = str(Path(sys.executable).parent)
        solc_supports_allow = tuple(int(x) for x in solc_version.split(".")[:2]) >= (0, 5)
        solc_args_line = (f'solc_args="--allow-paths .,{REPO}"' if solc_supports_allow else "")
        target = str(contract_path)

        with tempfile.TemporaryDirectory() as tmpdir:

            driver = Path(tmpdir) / "_run.py"
            result_file = Path(tmpdir) / "_result.json"
            driver.write_text(f"""
import json
from slither import Slither
import slither.detectors.all_detectors as _ad
from slither.detectors.abstract_detector import AbstractDetector
_dets = [getattr(_ad,n) for n in dir(_ad)
         if isinstance(getattr(_ad,n),type) and
         issubclass(getattr(_ad,n),AbstractDetector) and
         getattr(_ad,n) is not AbstractDetector]
try:
    sl = Slither({target!r}, solc={solc_binary!r}, {solc_args_line})
    for c in _dets: sl.register_detector(c)
    dr = sl.run_detectors()
    hits=[]; hc={{}}
    for findings in dr:
        if not isinstance(findings, list): findings=[findings]
        for f in findings:
            ch = f.get("check") if isinstance(f, dict) else None
            if ch: hits.append(ch); hc[ch]=hc.get(ch,0)+1
    with open("{result_file}","w") as fp:
        json.dump({{"status":"OK","hits":hits,"hit_counts":hc,"n_detectors":len(_dets)}},fp)
except Exception as e:
    emsg=str(e)[:500]
    stat=("COMPILE_ERROR" if any(k in emsg.lower() for k in
          ["compil","syntax","invalid","parsing"]) else "EXCEPTION")
    with open("{result_file}","w") as fp:
        json.dump({{"status":stat,"hits":[],"hit_counts":{{}},"n_detectors":0,
                   "err":f"{{type(e).__name__}}:{{emsg}}"}},fp)
""")
            sub_env = os.environ.copy()
            sub_env["PATH"] = venv_bin + os.pathsep + sub_env.get("PATH", "")
            proc = subprocess.run(
                [sys.executable, str(driver)],
                capture_output=True, text=True,
                timeout=SLITHER_TIMEOUT, env=sub_env, cwd=str(REPO))
            result["elapsed_sec"] = time.time() - start
            if result_file.exists():
                parsed = json.loads(result_file.read_text())
                result.update({k: parsed.get(k, result[k]) for k in
                                ["status", "hits", "hit_counts", "n_detectors", "err"]})
            else:
                err_short = (proc.stderr or proc.stdout or "")[:500]
                result["status"] = ("COMPILE_ERROR" if any(
                    k in err_short.lower() for k in ["compilation", "parse", "syntax"])
                    else "EXCEPTION")
                result["err"] = err_short

    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["elapsed_sec"] = SLITHER_TIMEOUT
        result["err"] = f"timeout after {SLITHER_TIMEOUT}s"
    except Exception as e:
        result["status"] = "EXCEPTION"
        result["elapsed_sec"] = time.time() - start
        result["err"] = f"{type(e).__name__}: {str(e)[:200]}"
    return result


def fix_path(stored_path: str) -> Path:
    p = str(stored_path)
    for prefix in ("BCCC-SCsVul-2024/Source Codes/", "BCCC-SCsVul-2024/SourceCodes/"):
        if p.startswith(prefix):
            return SRC / p[len(prefix):]
    return Path(p)


def main():
    # Load existing checkpoint
    print("Loading checkpoint...")
    checkpoint = {}
    with CHECKPOINT.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                checkpoint[r["id"]] = r
            except Exception:
                pass
    print(f"  {len(checkpoint):,} entries ({sum(1 for r in checkpoint.values() if r.get('status')=='OK'):,} OK)")

    # Find compile-error candidates
    ce_rows = {cid: r for cid, r in checkpoint.items() if r.get("status") == "COMPILE_ERROR"}
    print(f"  {len(ce_rows):,} COMPILE_ERROR entries to retry")

    # Load source paths from input CSV
    id_to_path = {}
    id_to_csv_pragma = {}
    with INPUT.open() as f:
        for row in csv.DictReader(f):
            if row["id"] in ce_rows:
                id_to_path[row["id"]] = fix_path(row["bccc_file_path"])
                id_to_csv_pragma[row["id"]] = row.get("pragma", "")

    # Categorize candidates (two recoverable categories only)
    spaced_ids  = []  # spaced pragma in CSV → pick_solc_version returned wrong version
    version_ids = []  # exact pragma in .sol file → different from CSV-based version we used

    for cid, r in ce_rows.items():
        err = r.get("err", "")
        csv_p = id_to_csv_pragma.get(cid, "")

        # Skip SPDX-labeled failures — investigation shows these are actually
        # missing-import failures (multi-file contracts), not license issues.
        if "SPDX" in err or "spdx" in err.lower():
            continue

        # Category 1: spaced pragma in CSV (e.g. '^ 0.4 .9')
        if re.search(r'\d+\s+\.\d+|\d+\.\d+\s+\.\d+', csv_p):
            spaced_ids.append(cid)
            continue

        # Category 2: exact version in .sol file different from what we used
        sol_path = id_to_path.get(cid)
        if sol_path and sol_path.exists():
            file_ver = _detect_version_from_file(sol_path)
            used_ver = r.get("solc_version", "")
            if file_ver != used_ver and _get_binary(file_ver):
                version_ids.append((cid, file_ver))

    print(f"\nCategories to retry:")
    print(f"  Spaced pragma:    {len(spaced_ids):,}")
    print(f"  Exact version:    {len(version_ids):,}")

    # Build retry list: (cid, target_version)
    retry_list = [(cid, _detect_version_from_file(id_to_path[cid]))
                  for cid in spaced_ids if cid in id_to_path]
    retry_list += version_ids

    # Deduplicate by cid
    seen = set()
    all_retry = []
    for item in retry_list:
        cid = item[0] if isinstance(item, tuple) else item
        if cid not in seen:
            seen.add(cid)
            all_retry.append(item)

    print(f"  Total unique:     {len(all_retry):,}")
    print(f"  Total unique:     {len(all_retry):,}")

    if not all_retry:
        print("Nothing to retry.")
        return 0

    # Retry
    t0 = time.time()
    recovered = 0
    total = len(all_retry)

    for i, item in enumerate(all_retry, 1):
        cid, target_ver = item
        sol_path = id_to_path.get(cid)
        if not sol_path or not sol_path.exists():
            continue

        result = _run_slither(sol_path, target_ver)
        result["id"] = cid

        if result["status"] == "OK":
            recovered += 1
            # Update checkpoint in memory
            checkpoint[cid] = result
            # Append to checkpoint file
            with CHECKPOINT.open("a") as f:
                f.write(json.dumps(result, default=str) + "\n")

        elapsed = time.time() - t0
        eta = (total - i) * (elapsed / i) / 60
        print(f"  [{i}/{total}] {cid[:16]}.. ver={target_ver} "
              f"status={result['status']} hits={len(result.get('hits',[]))} "
              f"eta={eta:.0f}min", flush=True)

    print(f"\nRecovered {recovered}/{total} ({100*recovered/max(total,1):.1f}%) in "
          f"{(time.time()-t0)/60:.1f} min")

    # Re-read checkpoint (retries may have added entries after the last-written OK
    # for the same cid; use LAST occurrence per id)
    print("\nRegenerating p5_gap_full_slither_results.csv from checkpoint...")
    final = {}
    with CHECKPOINT.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                final[r["id"]] = r  # last write wins
            except Exception:
                pass

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "status", "n_hits", "n_detectors", "hits_json",
                    "hit_counts_json", "elapsed_sec", "solc_version", "err"])
        for r in final.values():
            w.writerow([r["id"], r["status"], len(r.get("hits", [])),
                        r.get("n_detectors", 0), json.dumps(r.get("hits", [])),
                        json.dumps(r.get("hit_counts", {})),
                        f"{r.get('elapsed_sec', 0):.2f}",
                        r.get("solc_version", ""),
                        str(r.get("err", ""))[:200]])

    ok_count = sum(1 for r in final.values() if r.get("status") == "OK")
    print(f"Written: {len(final):,} rows, {ok_count:,} OK")
    return recovered


if __name__ == "__main__":
    main()
