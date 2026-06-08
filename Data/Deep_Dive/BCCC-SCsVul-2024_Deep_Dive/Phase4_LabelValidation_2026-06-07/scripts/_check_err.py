import json
from pathlib import Path

CKPT = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_slither_checkpoint.jsonl")
with open(CKPT) as f:
    results = [json.loads(line) for line in f]

# Find first EXCEPTION and show full error
for r in results:
    if r["status"] == "EXCEPTION":
        print(f"id: {r['id']}")
        print(f"elapsed: {r['elapsed_sec']:.1f}s")
        print(f"error:\n{r['err']}")
        print("---")
        break

# Show second exception
count = 0
for r in results:
    if r["status"] == "EXCEPTION":
        count += 1
        if count == 2:
            print(f"id: {r['id']}")
            print(f"elapsed: {r['elapsed_sec']:.1f}s")
            print(f"error:\n{r['err']}")
            break
