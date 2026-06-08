"""BCCC-SCsVul-2024 Phase 1 - final pass: content dedup, pragmas, .md5 contents."""
import csv
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
    "Class01:ExternalBug": "ExternalBug", "Class02:GasException": "GasException",
    "Class03:MishandledException": "MishandledException", "Class04:Timestamp": "Timestamp",
    "Class05:TransactionOrderDependence": "TransactionOrderDependence", "Class06:UnusedReturn": "UnusedReturn",
    "Class07:WeakAccessMod": "WeakAccessMod", "Class08:CallToUnknown": "CallToUnknown",
    "Class09:DenialOfService": "DenialOfService", "Class10:IntegerUO": "IntegerUO",
    "Class11:Reentrancy": "Reentrancy", "Class12:NonVulnerable": "NonVulnerable",
}

# 1. FULL .md5 FILE CONTENTS
print("=" * 70)
print("1. FULL .md5 FILE CONTENTS")
print("=" * 70)
for f in (ROOT / "BCCC-SCsVul-2024.md5", ROOT / "Sourcecodes.md5"):
    print(f"--- {f.name} ({f.stat().st_size} bytes) ---")
    print(repr(f.read_text()))
    print()

# 2. CONTENT-BASED DEDUP (per-folder and global)
print("=" * 70)
print("2. CONTENT-BASED DEDUP (file sha256)")
print("=" * 70)
print("Computing sha256 of all 111,897 files...")
content_hashes = defaultdict(list)  # sha -> [(folder, filename)]
total = 0
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        for p in d.glob("*.sol"):
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            content_hashes[sha].append((d.name, p.name))
            total += 1
print(f"Total files: {total}")
print(f"Unique content hashes: {len(content_hashes)}")
print(f"Files that are exact copies of another file: {total - len(content_hashes)}")

# Distribution: how many unique content hashes appear in N folders?
folder_count = Counter(len(set(f[0] for f in locs)) for locs in content_hashes.values())
print("Distribution of unique-content-hash appearing in N folders:")
for n, c in sorted(folder_count.items()):
    print(f"  in {n} folder(s): {c} unique contents")

# Top 10 most-copied content hashes
print()
print("Top 10 most-duplicated content hashes:")
ranked = sorted(content_hashes.values(), key=lambda locs: -len(locs))
for locs in ranked[:10]:
    folders = sorted(set(f[0] for f in locs))
    print(f"  {len(locs)} copies across {len(folders)} folders: {folders}")
    print(f"    example file: {locs[0][1]}")

# 3. UNIQUENESS ACROSS BOTH FOLDERS AND CSV (the real story)
print()
print("=" * 70)
print("3. CSV UNIQUENESS — what does ID count really mean?")
print("=" * 70)
ids_per_class = defaultdict(set)
total_rows = 0
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        total_rows += 1
        for c in CLASSES:
            if row[c].strip() not in ('', '0', '0.0'):
                ids_per_class[c].add(row["ID"])
print(f"CSV total rows: {total_rows}")
print(f"Unique IDs: {len(set().union(*ids_per_class.values()))}")
for c in CLASSES:
    print(f"  {c:42s} {len(ids_per_class[c]):>7d} unique IDs (positive class)")

# 4. PRAGMA SOLIDITY VERSION DISTRIBUTION (sample 1000 contracts)
print()
print("=" * 70)
print("4. PRAGMA SOLIDITY VERSION DISTRIBUTION (1000 contracts)")
print("=" * 70)
pragma_pattern = re.compile(r"pragma\s+solidity\s+([^;]+);")
pragma_vers = Counter()
sampled = 0
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        for p in sorted(d.glob("*.sol"))[:100]:
            try:
                text = p.read_text(errors="ignore")[:2000]
            except Exception:
                continue
            sampled += 1
            pm = pragma_pattern.search(text)
            if pm:
                pragma_vers[pm.group(1).strip()] += 1
            else:
                pragma_vers["(none found)"] += 1
print(f"Sampled {sampled} files:")
for v, n in pragma_vers.most_common():
    print(f"  {v:50s} {n}")

# 5. CONTRACT STYLE — function/contract keyword density
print()
print("=" * 70)
print("5. CONTRACT STYLE (1000 sampled files)")
print("=" * 70)
contract_kw = re.compile(r"\bcontract\s+(\w+)")
function_kw = re.compile(r"\bfunction\s+(\w+)\s*\(")
event_kw = re.compile(r"\bevent\s+(\w+)")
modifier_kw = re.compile(r"\bmodifier\s+(\w+)")
spdx_pattern = re.compile(r"SPDX-License-Identifier", re.IGNORECASE)
total_funcs = 0
total_contracts = 0
total_events = 0
total_modifiers = 0
spdx_count = 0
empty_files = 0
files_with_pragma = 0
sampled = 0
for d in sorted(SRC.iterdir()):
    if d.is_dir():
        for p in sorted(d.glob("*.sol"))[:100]:
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            sampled += 1
            if not text.strip():
                empty_files += 1
                continue
            if pragma_pattern.search(text):
                files_with_pragma += 1
            total_contracts += len(contract_kw.findall(text))
            total_funcs += len(function_kw.findall(text))
            total_events += len(event_kw.findall(text))
            total_modifiers += len(modifier_kw.findall(text))
            if spdx_pattern.search(text):
                spdx_count += 1
print(f"  Sampled: {sampled}")
print(f"  Empty/whitespace files: {empty_files}")
print(f"  With pragma: {files_with_pragma}")
print(f"  With SPDX header: {spdx_count}")
print(f"  Total `contract` declarations: {total_contracts} (mean per file: {total_contracts/sampled:.2f})")
print(f"  Total `function` declarations: {total_funcs} (mean per file: {total_funcs/sampled:.2f})")
print(f"  Total `event` declarations: {total_events} (mean per file: {total_events/sampled:.2f})")
print(f"  Total `modifier` declarations: {total_modifiers} (mean per file: {total_modifiers/sampled:.2f})")

# 6. SAMPLE 1 CONTRACT FROM EACH MINORITY CLASS
print()
print("=" * 70)
print("6. SAMPLE CONTRACTS — minority classes (ExternalBug, DenialOfService)")
print("=" * 70)
# Find an ID with ExternalBug active
extbug_id = None; dos_id = None
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        if extbug_id is None and row["Class01:ExternalBug"].strip() not in ('', '0', '0.0'):
            extbug_id = row["ID"]
        if dos_id is None and row["Class09:DenialOfService"].strip() not in ('', '0', '0.0'):
            dos_id = row["ID"]
        if extbug_id and dos_id: break
print(f"Sample ExternalBug contract: {extbug_id}")
print(f"Sample DenialOfService contract: {dos_id}")
for fid in (extbug_id, dos_id):
    if not fid: continue
    for d in SRC.iterdir():
        if d.is_dir():
            cand = d / f"{fid}.sol"
            if cand.exists():
                content = cand.read_text()
                print(f"\n--- {d.name}/{cand.name} ({len(content)} chars, {content.count(chr(10))+1} lines) ---")
                print(content[:1500])
                break
