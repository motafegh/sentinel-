"""BCCC-SCsVul-2024 Phase 1 exploration - second pass."""
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
CLASS_TO_FOLDER = {c.split(":")[0] + ":" + c.split(":")[1]: c.split(":")[1] for c in CLASSES}
# Fix keys
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

# 1. DUPLICATE ROWS: are CSV rows with same ID identical or different?
print("=" * 70)
print("1. DUPLICATE ROW ANALYSIS (same ID, multiple CSV rows)")
print("=" * 70)
print("Loading CSV (full)...")
rows_by_id = defaultdict(list)
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        rows_by_id[row["ID"]].append(row)
print(f"Unique IDs: {len(rows_by_id)}")

# Take first ID with multiple rows, dump 2 rows side by side for a few key columns
dups = {i: rows for i, rows in rows_by_id.items() if len(rows) > 1}
print(f"IDs with >1 row: {len(dups)}")
print(f"Distribution of row counts per duplicated ID:")
count_dist = Counter(len(rows) for rows in rows_by_id.values())
for k in sorted(count_dist):
    print(f"  {k} row(s): {count_dist[k]} IDs")

# Are duplicates truly identical (all 254 columns)?
print()
print("Are duplicate rows identical across all 254 columns?")
identical = 0
differing = 0
diff_samples = []
for i, rows in dups.items():
    first = rows[0]
    for other in rows[1:]:
        if all(first[c] == other[c] for c in CLASSES):
            pass  # not enough
        else:
            differing += 1
            diff_samples.append((i, first, other))
    # Check ALL columns
    if all(rows[0][c] == other[c] for other in rows[1:] for c in first):
        identical += 1
    else:
        # Check if just class columns differ
        class_diff = any(rows[0][c] != other[c] for other in rows[1:] for c in CLASSES)
        if class_diff:
            pass  # counted above
print(f"IDs where all rows identical: {identical}")
print(f"IDs where at least one column differs: {len(dups) - identical}")

# Show one example
sample_id = list(dups.keys())[0]
print(f"\nExample duplicated ID: {sample_id}")
print(f"Number of rows: {len(dups[sample_id])}")
for idx, row in enumerate(dups[sample_id][:3]):
    print(f"  Row {idx} active classes: {[c for c in CLASSES if row[c] not in ('','0','0.0')]}")

# 2. CONTENT HASH CHECK: does the CSV ID = SHA-256 of the file content?
print()
print("=" * 70)
print("2. IS ID = SHA-256 OF FILE CONTENT?")
print("=" * 70)
mismatches_hash = 0
checked = 0
samples_check = []
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        for p in sorted(d.glob("*.sol"))[:50]:  # first 50 per folder
            content = p.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            if sha != p.stem:
                mismatches_hash += 1
                if len(samples_check) < 5:
                    samples_check.append((p.name, sha, p.stem, d.name))
            checked += 1
print(f"Checked: {checked} files (50 per folder)")
print(f"ID != sha256(content): {mismatches_hash} ({100*mismatches_hash/checked:.1f}%)")
if samples_check:
    print("Sample mismatches (filename, sha256, expected_id, folder):")
    for s in samples_check:
        print(f"  {s[3]}/{s[0]}")
        print(f"    actual sha256:  {s[1]}")
        print(f"    filename stem:  {s[2]}")

# 3. SAMPLE .SOL FILES (one with multiple folder placements)
print()
print("=" * 70)
print("3. SAMPLE .SOL FILE (one in multiple folders)")
print("=" * 70)
target_id = "00001c839d754c4d89b3433aa51e4d6266226a9d907aff96dc019549e86f8289"
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        candidate = d / f"{target_id}.sol"
        if candidate.exists():
            content = candidate.read_text()
            sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            print(f"Folder: {d.name}")
            print(f"File: {candidate.name}")
            print(f"sha256(content): {sha}")
            print(f"=== First 50 lines ===")
            print("\n".join(content.split("\n")[:50]))
            print(f"=== Total lines: {len(content.split(chr(10)))} ===")
            print()

# 4. CONTRACT-LEVEL UNIQUE STATS
print("=" * 70)
print("4. UNIQUE-CONTRACT-LEVEL STATS")
print("=" * 70)
# Dedup the CSV by ID (keep first), recompute
unique_rows = {}
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        if row["ID"] not in unique_rows:
            unique_rows[row["ID"]] = row
print(f"Unique contracts: {len(unique_rows)}")
pos = Counter()
multi = 0
zero = 0
for row in unique_rows.values():
    active = [c for c in CLASSES if row[c].strip() not in ('','0','0.0')]
    for c in active: pos[c] += 1
    if len(active) == 0: zero += 1
    if len(active) > 1: multi += 1
print(f"  (after dedup) Multi-label: {multi}")
print(f"  (after dedup) Zero active: {zero}")
print()
print("Per-class (unique contracts, n=68,433):")
for c in CLASSES:
    print(f"  {c:42s} {pos[c]:>7d}  ({100*pos[c]/len(unique_rows):>5.1f}%)")

# 5. .md5 file content
print()
print("=" * 70)
print("5. .md5 FILE CONTENTS")
print("=" * 70)
for f in (ROOT / "BCCC-SCsVul-2024.md5", ROOT / "Sourcecodes.md5"):
    print(f"--- {f.name} ---")
    print(f.read_text())

# 6. PRAGMA / SPREAD PATTERNS in a sample
print("=" * 70)
print("6. PRAGMA + SPDX SAMPLE (200 contracts)")
print("=" * 70)
pragma_pattern = re.compile(r"pragma\s+solidity\s+([^;]+);")
spdx_pattern = re.compile(r"SPDX-License-Identifier\s*:\s*([^\n]+)")
pragma_vers = Counter()
spdx_seen = 0
sampled = 0
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        for p in sorted(d.glob("*.sol"))[:20]:
            try:
                text = p.read_text(errors="ignore")[:2000]
            except Exception:
                continue
            sampled += 1
            pm = pragma_pattern.search(text)
            if pm: pragma_vers[pm.group(1).strip()] += 1
            if spdx_pattern.search(text): spdx_seen += 1
print(f"Sampled {sampled} files (head 2KB each):")
for v, n in pragma_vers.most_common(10):
    print(f"  pragma {v:40s} {n}")
print(f"  SPDX header present: {spdx_seen} / {sampled} ({100*spdx_seen/sampled:.1f}%)")
