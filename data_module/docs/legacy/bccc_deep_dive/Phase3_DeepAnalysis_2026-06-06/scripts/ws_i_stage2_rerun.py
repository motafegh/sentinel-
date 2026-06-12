"""Re-run slither on all 808 contracts with the FIXED findings parser.

The previous run had a bug: findings is list-of-lists not list-of-dicts.
The 757 OK contracts were logged with empty hits, but they probably have real findings.
"""
import sys
sys.path.insert(0, "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/scripts")

# Force reload module
if "ws_i_stage2_run_and_agreement" in sys.modules:
    del sys.modules["ws_i_stage2_run_and_agreement"]
from ws_i_stage2_run_and_agreement import _run_one, SLITHER_RESULTS_OUT, _incremental_save
import pandas as pd
import multiprocessing as mp
import time
import json
from collections import Counter
from pathlib import Path

results_path = SLITHER_RESULTS_OUT

# Reset all slither_hits to empty (force re-parse)
df = pd.read_csv(results_path)
print(f"Loaded {len(df)} contracts")

# Force re-run all OK contracts too (the hits column is wrong)
# Skip only PATH_MISSING and clearly broken EXCEPTION
ok_mask = (df["slither_status"] == "OK")
ex_mask = (df["slither_status"] == "EXCEPTION")
print(f"  OK to re-run: {ok_mask.sum()}")
print(f"  EXCEPTION to re-run: {ex_mask.sum()}")

# Force re-run ALL contracts (the 0 hits on OK was a parser bug)
mask = ok_mask | ex_mask
todo = df[mask].copy()
print(f"  Total to re-run: {len(todo)}")

# Load pragma
sample_df = pd.read_csv("Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_sample_818.csv")
pragma_map = dict(zip(sample_df["id"], sample_df["pragma"]))
path_map = dict(zip(sample_df["id"], sample_df["bccc_path_fixed"]))

# Build work items: (idx_in_full_df, path, pragma)
work_items = []
for idx, row in todo.iterrows():
    pid = row["id"]
    work_items.append((idx, path_map.get(pid, row.get("bccc_path_fixed", "")), pragma_map.get(pid, "")))

# Reset all results for these rows
for idx in todo.index:
    df.at[idx, "slither_status"] = ""
    df.at[idx, "slither_hits"] = ""
    df.at[idx, "slither_elapsed_sec"] = 0.0
    df.at[idx, "slither_solc"] = ""
    df.at[idx, "slither_n_detectors"] = 0

def worker(args):
    idx, path, pragma = args
    r = _run_one(path, pragma, timeout=20)
    return idx, r

n_workers = 6
completed = 0
total = len(work_items)
results_dict = {}
start = time.time()
save_every = 50

print(f"\nRe-running {total} contracts with {n_workers} workers...")
with mp.Pool(processes=n_workers) as pool:
    for idx, r in pool.imap_unordered(worker, work_items, chunksize=1):
        results_dict[idx] = r
        completed += 1
        if completed % 25 == 0 or completed == total:
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            cur_statuses = Counter([r["status"] for r in results_dict.values()])
            # Show hit count stats
            total_hits = sum(len(r["hits"]) for r in results_dict.values())
            print(f"  [{completed}/{total}] {elapsed:.0f}s, {rate:.2f}/s, ETA {eta:.0f}s | {dict(cur_statuses)} | total_hits={total_hits}", flush=True)
        if save_every > 0 and completed % save_every == 0:
            for i, r in results_dict.items():
                df.at[i, "slither_status"] = r["status"]
                df.at[i, "slither_hits"] = json.dumps(r["hits"])
                df.at[i, "slither_elapsed_sec"] = round(r["elapsed_sec"], 2)
                df.at[i, "slither_solc"] = r.get("solc_version", "")
                df.at[i, "slither_n_detectors"] = r.get("n_detectors", 0)
            df.to_csv(results_path, index=False)
            print(f"    [saved {len(results_dict)} partial results]")

# Final save
for i, r in results_dict.items():
    df.at[i, "slither_status"] = r["status"]
    df.at[i, "slither_hits"] = json.dumps(r["hits"])
    df.at[i, "slither_elapsed_sec"] = round(r["elapsed_sec"], 2)
    df.at[i, "slither_solc"] = r.get("solc_version", "")
    df.at[i, "slither_n_detectors"] = r.get("n_detectors", 0)
df.to_csv(results_path, index=False)
print(f"\nFinal save done. {len(results_dict)} contracts re-run.")
print(f"Final status: {df['slither_status'].value_counts(dropna=False).to_dict()}")
total_hits = sum(len(json.loads(h)) for h in df["slither_hits"] if h and isinstance(h, str) and h.startswith("["))
print(f"Total findings across all contracts: {total_hits}")
