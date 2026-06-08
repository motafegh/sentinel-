import json
from pathlib import Path
from collections import Counter

CKPT = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_slither_checkpoint.jsonl")
with open(CKPT) as f:
    results = [json.loads(line) for line in f]

statuses = Counter(r["status"] for r in results)
print(f"Total: {len(results)}")
for s, c in statuses.most_common():
    print(f"  {s}: {c} ({100*c/len(results):.1f}%)")

oks = [r for r in results if r["status"] == "OK"]
if oks:
    print(f"\nOK hits distribution:")
    all_hits = []
    for r in oks:
        all_hits.extend(r["hits"])
    hits_c = Counter(all_hits)
    for h, c in hits_c.most_common(20):
        print(f"  {h}: {c}")
