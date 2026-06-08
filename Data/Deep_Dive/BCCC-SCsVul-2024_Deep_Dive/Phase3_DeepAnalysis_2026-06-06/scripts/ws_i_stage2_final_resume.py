"""Final resume: re-run only the empty contracts that need to be filled."""
import sys
sys.path.insert(0, "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/scripts")
if "ws_i_stage2_run_and_agreement" in sys.modules:
    del sys.modules["ws_i_stage2_run_and_agreement"]
from ws_i_stage2_run_and_agreement import _run_one, SLITHER_RESULTS_OUT
import pandas as pd
import multiprocessing as mp
import time
import json
from collections import Counter

results_path = SLITHER_RESULTS_OUT
df = pd.read_csv(results_path)
print(f"Loaded {len(df)} contracts")
print(f"  Status counts: {df['slither_status'].fillna('EMPTY').value_counts().to_dict()}")

# Only process the EMPTY ones
empty_mask = (df["slither_status"].fillna("") == "") | (df["slither_status"].isna())
todo = df[empty_mask].copy()
print(f"  Empty: {len(todo)}")

if len(todo) == 0:
    print("Nothing to do!")
    sys.exit(0)

# Load pragma/path from sample
sample_df = pd.read_csv("Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_sample_818.csv")
pragma_map = dict(zip(sample_df["id"], sample_df["pragma"]))
path_map = dict(zip(sample_df["id"], sample_df["bccc_path_fixed"]))

work_items = []
for idx, row in todo.iterrows():
    pid = row["id"]
    work_items.append((idx, path_map.get(pid, row.get("bccc_path_fixed", "")), pragma_map.get(pid, "")))

def worker(args):
    idx, path, pragma = args
    r = _run_one(path, pragma, timeout=20)
    return idx, r

n_workers = 6
completed = 0
total = len(work_items)
results_dict = {}
start = time.time()

print(f"\nRe-running {total} empty contracts with {n_workers} workers...")
with mp.Pool(processes=n_workers) as pool:
    for idx, r in pool.imap_unordered(worker, work_items, chunksize=1):
        results_dict[idx] = r
        completed += 1
        if completed % 25 == 0 or completed == total:
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            cur_statuses = Counter([r["status"] for r in results_dict.values()])
            total_hits = sum(len(r["hits"]) for r in results_dict.values())
            print(f"  [{completed}/{total}] {elapsed:.0f}s, {rate:.2f}/s, ETA {eta:.0f}s | {dict(cur_statuses)} | new_hits={total_hits}", flush=True)
        # Save every 25 (more frequent)
        if completed % 25 == 0:
            for i, r in results_dict.items():
                df.at[i, "slither_status"] = r["status"]
                df.at[i, "slither_hits"] = json.dumps(r["hits"])
                df.at[i, "slither_elapsed_sec"] = round(r["elapsed_sec"], 2)
                df.at[i, "slither_solc"] = r.get("solc_version", "")
                df.at[i, "slither_n_detectors"] = r.get("n_detectors", 0)
            df.to_csv(results_path, index=False)

# Final save
for i, r in results_dict.items():
    df.at[i, "slither_status"] = r["status"]
    df.at[i, "slither_hits"] = json.dumps(r["hits"])
    df.at[i, "slither_elapsed_sec"] = round(r["elapsed_sec"], 2)
    df.at[i, "slither_solc"] = r.get("solc_version", "")
    df.at[i, "slither_n_detectors"] = r.get("n_detectors", 0)
df.to_csv(results_path, index=False)
print(f"\nFinal save done.")
print(f"Final status: {df['slither_status'].value_counts(dropna=False).to_dict()}")
total_hits = sum(len(json.loads(h)) for h in df["slither_hits"] if h and isinstance(h, str) and h.startswith("["))
print(f"Total findings across all 808 contracts: {total_hits}")
