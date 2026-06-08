"""Check slither exception patterns + preliminary agreement on completed contracts."""
import csv, json, sys
from pathlib import Path
from collections import Counter

BASE = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07")
CKPT = BASE / "outputs" / "ws_p4_s1_slither_checkpoint.jsonl"

if not CKPT.exists():
    print("No checkpoint yet")
    sys.exit(0)

with open(CKPT) as f:
    results = [json.loads(line) for line in f]

print(f"Total completed: {len(results)}")
statuses = Counter(r["status"] for r in results)
print(f"Status distribution: {dict(statuses)}")

# Show exception messages
exceptions = [r for r in results if r["status"] == "EXCEPTION"]
print(f"\nException samples (first 5):")
for r in exceptions[:5]:
    print(f"  id={r['id'][:16]}.. elapsed={r['elapsed_sec']:.1f}s err={r.get('err','')[:150]}")

# Show OK results
oks = [r for r in results if r["status"] == "OK"]
print(f"\nOK samples (first 5):")
for r in oks[:5]:
    print(f"  id={r['id'][:16]}.. hits={len(r['hits'])} elapsed={r['elapsed_sec']:.1f}s detectors={r['hits'][:5]}")
