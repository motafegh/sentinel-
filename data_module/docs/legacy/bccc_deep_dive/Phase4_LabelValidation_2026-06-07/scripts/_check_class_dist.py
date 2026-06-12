"""Check per-class distribution of OK slither results."""
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

# BCCC class columns
class_cols = ["Reentrancy", "IntegerUO", "UncheckedReturn", "Selfdestruct", 
              "DelegateCall", "TxOrigin", "Etherleak", "DoS", "NonVulnerable"]

print("OK contracts per class:")
for cls in class_cols:
    ok_with_cls = [cid for cid in ok_ids if sample.get(cid, {}).get(cls, "0") == "1"]
    total_with_cls = sum(1 for cid in sample if sample[cid].get(cls, "0") == "1")
    print(f"  {cls:<16}: {len(ok_with_cls):>3} / {total_with_cls:>5} = {100*len(ok_with_cls)/max(total_with_cls,1):.1f}%")

print("\nEXCEPTION contracts per class:")
for cls in class_cols:
    ex_with_cls = [cid for cid in ex_ids if sample.get(cid, {}).get(cls, "0") == "1"]
    total_with_cls = sum(1 for cid in sample if sample[cid].get(cls, "0") == "1")
    print(f"  {cls:<16}: {len(ex_with_cls):>3} / {total_with_cls:>5} = {100*len(ex_with_cls)/max(total_with_cls,1):.1f}%")

# Check hits for BCCC classes
print("\nBCCC vulnerability hits in OK results:")
bccc_vulns = {
    "Reentrancy": ["reentrancy"],
    "IntegerUO": ["integer-overflow", "integer-underflow", "signed-arithmetic"],
    "UncheckedReturn": ["unchecked-return"],
    "Selfdestruct": ["selfdestruct"],
    "DelegateCall": ["delegatecall"],
    "TxOrigin": ["tx-origin"],
    "Etherleak": ["send-transfer", "ether-leak"],
    "DoS": ["dos", "denial-service"],
}
for cls, patterns in bccc_vulns.items():
    matches = []
    for r in results:
        if r["status"] == "OK":
            for hit in r["hits"]:
                if any(p in hit.lower() for p in patterns):
                    matches.append(hit)
    print(f"  {cls}: {len(matches)} hits ({Counter(matches).most_common(3) if matches else 'none'})")
