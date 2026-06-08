"""Resume the slither run on the 558 empty rows from the previous run.

Saves incrementally every 50 contracts.
"""
import sys
sys.path.insert(0, "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/scripts")
from ws_i_stage2_run_and_agreement import _run_one, SLITHER_RESULTS_OUT
import pandas as pd
import multiprocessing as mp
import time
import json
from collections import Counter
from pathlib import Path

results_path = Path("Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_slither_results.csv")
df = pd.read_csv(results_path)
print(f"Loaded {len(df)} contracts from existing results")

# Find rows with empty/NaN status
empty_mask = df["slither_status"].isna() | (df["slither_status"] == "")
empty_df = df[empty_mask].copy()
print(f"Empty rows to process: {len(empty_df)}")

if len(empty_df) == 0:
    print("Nothing to do!")
    sys.exit(0)

# Load original sample to get pragma info
sample_df = pd.read_csv("Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_sample_818.csv")
pragma_map = dict(zip(sample_df["id"], sample_df["pragma"]))
path_map = dict(zip(sample_df["id"], sample_df["bccc_path_fixed"]))

# Build list of (idx_in_full_df, path, pragma) for the empty rows
work_items = []
for idx, row in empty_df.iterrows():
    pid = row["id"]
    work_items.append((idx, path_map.get(pid, row.get("bccc_path_fixed", "")), pragma_map.get(pid, "")))

print(f"Work items: {len(work_items)}")

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

print(f"Starting with {n_workers} workers...")
with mp.Pool(processes=n_workers) as pool:
    for idx, r in pool.imap_unordered(worker, work_items, chunksize=1):
        results_dict[idx] = r
        completed += 1
        if completed % 25 == 0 or completed == total:
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            cur_statuses = Counter(df.iloc[list(results_dict.keys())]["slither_status"].fillna("EMPTY") if False else [r["status"] for r in results_dict.values()])
            print(f"  [{completed}/{total}] {elapsed:.0f}s, {rate:.2f}/s, ETA {eta:.0f}s | {dict(cur_statuses)}", flush=True)
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
print(f"\nFinal save done. {len(results_dict)} new results saved.")
print(f"Final status counts: {df['slither_status'].value_counts(dropna=False).to_dict()}")
