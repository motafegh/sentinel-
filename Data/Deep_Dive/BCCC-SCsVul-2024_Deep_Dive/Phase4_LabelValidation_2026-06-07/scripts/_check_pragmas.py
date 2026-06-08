"""Check pragma values for failing vs passing contracts."""
import csv, json
from pathlib import Path
from collections import Counter

BASE = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07")
SAMPLE = BASE / "outputs" / "ws_p4_s1_sample.csv"
CKPT = BASE / "outputs" / "ws_p4_s1_slither_checkpoint.jsonl"

with open(SAMPLE) as f:
    sample = {r["id"]: r for r in csv.DictReader(f)}

with open(CKPT) as f:
    results = [json.loads(line) for line in f]

ok_ids = [r["id"] for r in results if r["status"] == "OK"]
ex_ids = [r["id"] for r in results if r["status"] == "EXCEPTION"]

ok_pragmas = [sample[cid]["pragma"] for cid in ok_ids if cid in sample]
ex_pragmas = [sample[cid]["pragma"] for cid in ex_ids if cid in sample]

print(f"OK contracts ({len(ok_ids)}):")
print(f"  Pragmas: {Counter(ok_pragmas).most_common(5)}")
print(f"\nEXCEPTION contracts ({len(ex_ids)}):")
print(f"  Pragmas: {Counter(ex_pragmas).most_common(5)}")

# Check error message more carefully
for r in results[:3]:
    if r["status"] == "EXCEPTION":
        err = r.get("err", "")
        # Find the actual error line
        lines = err.split("\n")
        for i, line in enumerate(lines):
            if "Error" in line or "error" in line or "Exception" in line:
                print(f"\n  id={r['id'][:16]} error line: {line.strip()}")
                break
        else:
            print(f"\n  id={r['id'][:16]} last 3 lines:")
            for line in lines[-3:]:
                print(f"    {line.strip()}")
