"""BCCC-SCsVul-2024 Phase 1 exploration.

Read-only. Produces structured findings printed to stdout.
"""
import csv
import os
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("BCCC-SCsVul-2024")
SRC = ROOT / "Source Codes"
CSV_PATH = ROOT / "BCCC-SCsVul-2024.csv"

CLASSES = [
    "Class01:ExternalBug", "Class02:GasException", "Class03:MishandledException",
    "Class04:Timestamp", "Class05:TransactionOrderDependence", "Class06:UnusedReturn",
    "Class07:WeakAccessMod", "Class08:CallToUnknown", "Class09:DenialOfService",
    "Class10:IntegerUO", "Class11:Reentrancy", "Class12:NonVulnerable",
]
CLASS_TO_FOLDER = {
    "Class01:ExternalBug": "ExternalBug",
    "Class02:GasException": "GasException",
    "Class03:MishandledException": "MishandledException",
    "Class04:Timestamp": "Timestamp",
    "Class05:TransactionOrderDependence": "TransactionOrderDependence",
    "Class06:UnusedReturn": "UnusedReturn",
    "Class07:WeakAccessMod": "WeakAccessMod",
    "Class08:CallToUnknown": "CallToUnknown",
    "Class09:DenialOfService": "DenialOfService",
    "Class10:IntegerUO": "IntegerUO",
    "Class11:Reentrancy": "Reentrancy",
    "Class12:NonVulnerable": "NonVulnerable",
}

# 1. PER-FOLDER FILE COUNTS
print("=" * 70)
print("1. PER-FOLDER FILE COUNTS")
print("=" * 70)
folder_counts = {}
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        n = sum(1 for _ in d.glob("*.sol"))
        folder_counts[d.name] = n
        print(f"  {d.name:32s} {n:>7d}")
total = sum(folder_counts.values())
print(f"  {'TOTAL':32s} {total:>7d}")

# 2. FILENAME PATTERN
print()
print("=" * 70)
print("2. FILENAME PATTERN (sample 5 per folder)")
print("=" * 70)
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        samples = sorted(d.glob("*.sol"))[:5]
        for s in samples:
            print(f"  {d.name:32s} {s.name}")
        break  # one folder is enough for pattern

# Check if filenames are SHA-256
print()
print("Hex-pattern check (all folders):")
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        sample = sorted(d.glob("*.sol"))[0]
        # CSV ID format = 64 hex chars
        is_hex = bool(re.match(r"^[0-9a-f]{64}\.sol$", sample.name))
        print(f"  {d.name:32s} {sample.name}  hex64?={is_hex}")

# 3. CSV ID <-> FOLDER FILE MAPPING
print()
print("=" * 70)
print("3. CSV ID UNIQUENESS + FOLDER MAPPING")
print("=" * 70)
print("Scanning CSV (one pass)...")
ids = []
id_to_classes = {}
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        ids.append(row["ID"])
        active = []
        for c in CLASSES:
            v = row[c].strip()
            if v and v not in ("0", "0.0", ""):
                active.append(c)
        id_to_classes[row["ID"]] = active

print(f"Total CSV rows: {len(ids)}")
print(f"Unique IDs: {len(set(ids))}")
dup_ids = [i for i, c in Counter(ids).items() if c > 1]
if dup_ids:
    print(f"  DUPLICATE IDs ({len(dup_ids)}): {dup_ids[:5]}")
else:
    print("  No duplicate IDs.")

# 4. CSV CLASS DISTRIBUTION
print()
print("=" * 70)
print("4. CSV CLASS DISTRIBUTION (n = %d contracts)" % len(ids))
print("=" * 70)
class_pos = Counter()
multi = 0
nv_with_vuln = 0
zero_active = 0
for i in ids:
    active = id_to_classes[i]
    for c in active:
        class_pos[c] += 1
    if len(active) > 1:
        multi += 1
    if "Class12:NonVulnerable" in active and len(active) > 1:
        nv_with_vuln += 1
    if len(active) == 0:
        zero_active += 1

for c in CLASSES:
    print(f"  {c:42s} {class_pos[c]:>7d}  ({100*class_pos[c]/len(ids):>5.1f}%)")
print(f"  Multi-label rows: {multi}  ({100*multi/len(ids):.1f}%)")
print(f"  NonVulnerable + another vuln: {nv_with_vuln}")
print(f"  Rows with NO active class: {zero_active}")

# 5. CSV ID EXISTS AS FOLDER FILE?
print()
print("=" * 70)
print("5. CSV ID -> FOLDER FILE RESOLUTION")
print("=" * 70)
folder_files = {}
for d in SRC.iterdir():
    if d.is_dir():
        for p in d.glob("*.sol"):
            folder_files[p.stem] = d.name

print(f"Total .sol files across all folders: {len(folder_files)}")

csv_ids = set(ids)
folder_ids = set(folder_files.keys())
only_in_csv = csv_ids - folder_ids
only_in_folder = folder_ids - csv_ids
in_both = csv_ids & folder_ids
print(f"  In CSV only:      {len(only_in_csv)}")
print(f"  In folders only:  {len(only_in_folder)}")
print(f"  In both:          {len(in_both)}")
if only_in_csv:
    print(f"  Sample CSV-only: {list(only_in_csv)[:3]}")
if only_in_folder:
    print(f"  Sample folder-only: {list(only_in_folder)[:3]}")

# 6. FOLDER <-> ACTIVE CLASS CONSISTENCY
print()
print("=" * 70)
print("6. FOLDER <-> ACTIVE CLASS CONSISTENCY")
print("=" * 70)
# For each file, check if its folder name matches the class(es) it's labeled with
mismatches = Counter()
match = 0
total_checked = 0
for fid in in_both:
    folder = folder_files[fid]
    active = id_to_classes[fid]
    total_checked += 1
    # File's folder should be one of the active vuln classes (or NonVulnerable folder for non-vuln)
    expected = set(CLASS_TO_FOLDER[c] for c in active if c != "Class12:NonVulnerable")
    if not expected:  # only NonVulnerable
        expected = {"NonVulnerable"}
    if folder in expected:
        match += 1
    else:
        # Record mismatch pattern
        for c in active:
            mismatches[(folder, c)] += 1
print(f"Files checked: {total_checked}")
print(f"Folder matches an active class: {match}  ({100*match/total_checked:.2f}%)")
print(f"Mismatches: {total_checked - match}")
if mismatches:
    print("Top 5 mismatch patterns (folder, csv_class) -> count:")
    for (folder, cls), n in mismatches.most_common(5):
        print(f"  {folder:32s} {cls:42s} {n}")
