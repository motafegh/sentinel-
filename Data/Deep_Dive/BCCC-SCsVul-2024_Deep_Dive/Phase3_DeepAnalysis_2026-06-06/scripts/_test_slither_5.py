"""Quick test of slither on first 10 sample contracts."""
import sys
sys.path.insert(0, "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/scripts")
from ws_i_stage2_run_and_agreement import _run_one, pick_solc_version
import pandas as pd

df = pd.read_csv("Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_sample_818.csv")
print(f"Testing first 5 of {len(df)} contracts...")

for i, row in df.head(5).iterrows():
    print(f"\n--- Contract {i}: id={row['id'][:16]} | pragma={row['pragma']} | reason={row['sample_reason']}")
    solc = pick_solc_version(row['pragma'])
    print(f"   solc picked: {solc}")
    r = _run_one(row['bccc_path_fixed'], row['pragma'], timeout=15)
    print(f"   status={r['status']}, elapsed={r['elapsed_sec']:.1f}s, detectors={r.get('n_detectors', 0)}")
    if r.get("err"):
        print(f"   err: {r['err'][:400]}")
    if r.get("hits"):
        print(f"   hits: {r['hits'][:5]}")
