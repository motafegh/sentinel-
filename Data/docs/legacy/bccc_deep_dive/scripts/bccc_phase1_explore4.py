"""BCCC-SCsVul-2024 - CORRECTED unique-contract-level label analysis."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = Path("BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv")
CLASSES = [
    "Class01:ExternalBug", "Class02:GasException", "Class03:MishandledException",
    "Class04:Timestamp", "Class05:TransactionOrderDependence", "Class06:UnusedReturn",
    "Class07:WeakAccessMod", "Class08:CallToUnknown", "Class09:DenialOfService",
    "Class10:IntegerUO", "Class11:Reentrancy", "Class12:NonVulnerable",
]

# Aggregate per unique ID: collect set of positive classes across all rows
id_to_pos = defaultdict(set)
rows_per_id = Counter()
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        rows_per_id[row["ID"]] += 1
        for c in CLASSES:
            if row[c].strip() not in ("", "0", "0.0"):
                id_to_pos[row["ID"]].add(c)

print("=" * 70)
print("UNIQUE-CONTRACT-LEVEL LABEL DISTRIBUTION (n=68,433)")
print("=" * 70)
n = len(id_to_pos)
print(f"Total unique contracts: {n}")

# Multi-label distribution
ml_dist = Counter(len(s) for s in id_to_pos.values())
print("\nPositive classes per contract:")
for k in sorted(ml_dist):
    print(f"  {k} class(es): {ml_dist[k]:>7d}  ({100*ml_dist[k]/n:>5.1f}%)")

# Per-class counts (unique contracts with class positive)
pos = Counter()
for s in id_to_pos.values():
    for c in s: pos[c] += 1
print("\nPer-class (unique-contract count, n=68,433):")
for c in CLASSES:
    print(f"  {c:42s} {pos[c]:>7d}  ({100*pos[c]/n:>5.1f}%)")

# Class co-occurrence matrix
print("\nClass co-occurrence (top 20 pairs):")
pair_count = Counter()
for s in id_to_pos.values():
    s_list = sorted(s)
    for i in range(len(s_list)):
        for j in range(i+1, len(s_list)):
            pair_count[(s_list[i], s_list[j])] += 1
for (a, b), cnt in pair_count.most_common(20):
    short_a = a.split(":")[1]
    short_b = b.split(":")[1]
    print(f"  {short_a:25s} + {short_b:25s} {cnt:>5d}")

# How many contracts have NonVulnerable alongside vulns?
nv_with_vuln = 0
nv_alone = 0
vuln_alone = 0
vuln_no_nv = 0
for s in id_to_pos.values():
    has_nv = "Class12:NonVulnerable" in s
    has_vuln = any(c != "Class12:NonVulnerable" for c in s)
    if has_nv and has_vuln: nv_with_vuln += 1
    elif has_nv: nv_alone += 1
    elif has_vuln: vuln_no_nv += 1
    if has_vuln and not has_nv: vuln_alone += 1

print(f"\nNonVulnerable breakdown:")
print(f"  Has NonVulnerable AND vulnerabilities: {nv_with_vuln}")
print(f"  Has NonVulnerable only (truly clean):  {nv_alone}")
print(f"  Has vulnerabilities only (no NV flag): {vuln_no_nv}")
print(f"  Has vulnerabilities only (subset):     {vuln_alone}")

# Distribution: rows-per-contract
print(f"\nRows per unique contract:")
rd = Counter(rows_per_id.values())
for k in sorted(rd):
    print(f"  {k} row(s): {rd[k]:>7d}  ({100*rd[k]/n:>5.1f}%)")
