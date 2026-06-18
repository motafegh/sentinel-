"""Method 8 — Parser Faithfulness: CSV → parser → v3 .labels.json comparison.

Provenance:
  Command: python3 scripts/parser_faithfulness.py
  RNG seed: N/A (full corpus, no sampling)
  Tool: Python 3.12
  Input CSV: data_module/data/raw_staging/dive_labels/DIVE_Labels.csv (raw, 22,330 rows)
  Input meta: data_module/data/preprocessed/dive/*.meta.json (v3 preprocessed, 22,073 files)
  Input labels: data_module/data/labels/dive/*.labels.json (v3 parsed, 22,073 files)
  Input crosswalk: data_module/sentinel_data/labeling/crosswalks/dive.yaml
  Frames: CSV = raw (22,330); meta+labels = v3 export (22,073)
  Output: stdout + writes to findings/08_parser_faithfulness.md section
  Timestamp: 2026-06-18
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path("/home/motafeq/projects/sentinel")
CSV_PATH = PROJECT / "data_module/data/raw_staging/dive_labels/DIVE_Labels.csv"
META_DIR = PROJECT / "data_module/data/preprocessed/dive"
LABELS_DIR = PROJECT / "data_module/data/labels/dive"
FINDINGS = PROJECT / "docs/plan/data_module/2026-06-18-dive-data-source-quality-investigation/findings/08_parser_faithfulness.md"

# Crosswalk (hardcoded from dive.yaml reading in Step 1)
CROSSWALK = {
    "Reentrancy": "Reentrancy",
    "DoS": "DenialOfService",
    "Arithmetic": "IntegerUO",
    "Time manipulation": "Timestamp",
    "Front Running": "TransactionOrderDependence",
    "Access Control": "ExternalBug",
    "Unchecked Return Values": "UnusedReturn",
    "Bad Randomness": None,  # DROPPED
}

# All DIVE column names that have a mapping (including dropped ones)
ALL_DIVE_COLS = list(CROSSWALK.keys())


def main() -> None:
    # --- Step 2: Build contractID → sha256 mapping from meta.json files ---
    print("Step 2: Building contractID → sha256 mapping...")
    meta_files = sorted(META_DIR.glob("*.meta.json"))
    cid_to_sha: dict[int, str] = {}
    sha_to_cid: dict[str, int] = {}
    cid_collisions = 0
    sha_collisions = 0
    parse_failures = 0

    for mp in meta_files:
        try:
            meta = json.loads(mp.read_text())
        except (json.JSONDecodeError, OSError):
            parse_failures += 1
            continue

        sha256 = meta.get("sha256", "")
        original_path = meta.get("original_path", "")

        if not sha256 or not original_path:
            parse_failures += 1
            continue

        # Extract contractID from original_path: "repo/__source__/<N>.sol" → N
        fname = Path(original_path).name
        try:
            cid = int(fname.replace(".sol", ""))
        except ValueError:
            parse_failures += 1
            continue

        # Check for collisions
        if cid in cid_to_sha:
            if cid_to_sha[cid] != sha256:
                cid_collisions += 1
        if sha256 in sha_to_cid:
            if sha_to_cid[sha256] != cid:
                sha_collisions += 1

        cid_to_sha[cid] = sha256
        sha_to_cid[sha256] = cid

    n_meta = len(meta_files)
    n_mapped = len(cid_to_sha)
    print(f"  meta.json files: {n_meta}")
    print(f"  successfully mapped: {n_mapped}")
    print(f"  parse failures: {parse_failures}")
    print(f"  cid collisions (different sha256 for same cid): {cid_collisions}")
    print(f"  sha collisions (different cid for same sha256): {sha_collisions}")

    # Verify 1:1
    if cid_collisions > 0 or sha_collisions > 0:
        print(f"  WARNING: mapping is NOT 1:1 — investigating...")
        # Dump collision details
        if cid_collisions > 0:
            cid_freq: dict[int, int] = defaultdict(int)
            for mp in meta_files:
                try:
                    meta = json.loads(mp.read_text())
                    cid = int(Path(meta["original_path"]).name.replace(".sol", ""))
                    cid_freq[cid] += 1
                except Exception:
                    pass
            collisions = {k: v for k, v in cid_freq.items() if v > 1}
            print(f"  Duplicate cids: {dict(list(collisions.items())[:10])}")

    # --- Step 2b: Verify that every meta sha256 has a corresponding labels.json ---
    print("\nStep 2b: Checking labels.json existence for each meta contract...")
    labels_exist = 0
    labels_missing = 0
    for sha in cid_to_sha.values():
        if (LABELS_DIR / f"{sha}.labels.json").exists():
            labels_exist += 1
        else:
            labels_missing += 1
    print(f"  labels.json exist: {labels_exist}")
    print(f"  labels.json MISSING: {labels_missing}")

    # --- Step 3: Read CSV ---
    print("\nStep 3: Reading CSV and comparing...")
    csv_labels: dict[int, set[str]] = {}  # contractID → set of DIVE column names where =1
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["contractID"])
            positive_cols = {col for col in ALL_DIVE_COLS if row.get(col, "0") == "1"}
            csv_labels[cid] = positive_cols

    n_csv = len(csv_labels)
    print(f"  CSV rows: {n_csv}")

    # --- Step 3: Build expected labels per contractID ---
    # For each contractID in CSV, apply crosswalk to get expected set of canonical classes
    expected: dict[int, set[str]] = {}
    for cid, dive_cols in csv_labels.items():
        canonicals = set()
        for col in dive_cols:
            mapped = CROSSWALK.get(col)
            if mapped is not None:
                canonicals.add(mapped)
        expected[cid] = canonicals

    # Sanity: Bad Randomness stats
    bad_random_count = sum(1 for cols in csv_labels.values() if "Bad Randomness" in cols)
    print(f"  Contracts with Bad Randomness in CSV: {bad_random_count}")

    # --- Step 3: Build actual labels per contractID (from .labels.json) ---
    actual: dict[int, set[str]] = {}
    labels_read = 0
    labels_parse_fail = 0

    for cid, sha in cid_to_sha.items():
        lp = LABELS_DIR / f"{sha}.labels.json"
        try:
            lbl = json.loads(lp.read_text())
        except (json.JSONDecodeError, OSError):
            labels_parse_fail += 1
            continue

        classes = lbl.get("classes", {})
        positive = {cls for cls, v in classes.items() if v.get("value", 0) == 1}
        actual[cid] = positive
        labels_read += 1

    print(f"  labels.json read: {labels_read}")
    print(f"  labels.json parse failures: {labels_parse_fail}")

    # --- Step 3: Per-class comparison ---
    # All canonical classes we care about
    all_canonical = list(CROSSWALK.values())
    all_canonical = sorted(set(c for c in all_canonical if c is not None))

    print("\n--- Per-class CSV-vs-parsed agreement ---")
    per_class_match = defaultdict(int)
    per_class_total = defaultdict(int)

    # Only compare contracts that appear in BOTH CSV and meta-derived map
    compared = 0
    only_csv = 0
    only_labels = 0
    mismatches: list[dict] = []

    for cid in set(list(csv_labels.keys()) + list(actual.keys())):
        exp = expected.get(cid, set())
        act = actual.get(cid, set())

        if cid not in csv_labels:
            only_labels += 1
            continue
        if cid not in actual:
            only_csv += 1
            continue

        compared += 1

        for cls in all_canonical:
            per_class_total[cls] += 1
            if (cls in exp) == (cls in act):
                per_class_match[cls] += 1
            else:
                mismatches.append({
                    "contractID": cid,
                    "sha256": cid_to_sha.get(cid, "UNKNOWN"),
                    "class": cls,
                    "in_csv_expected": cls in exp,
                    "in_labels_actual": cls in act,
                })

        # Also check for any classes in actual that are NOT in all_canonical
        extra_classes = act - set(all_canonical)
        if extra_classes:
            mismatches.append({
                "contractID": cid,
                "sha256": cid_to_sha.get(cid, "UNKNOWN"),
                "note": f"extra classes in labels not in canonical set: {extra_classes}",
            })

    print(f"  Contracts compared: {compared}")
    print(f"  Contracts in CSV only (no labels.json): {only_csv}")
    print(f"  Contracts in labels only (no CSV row): {only_labels}")

    print("\n  Per-class agreement:")
    for cls in all_canonical:
        match = per_class_match[cls]
        total = per_class_total[cls]
        pct = (match / total * 100) if total > 0 else 0.0
        print(f"    {cls}: {match}/{total} match ({pct:.2f}%)")

    print(f"\n  Total mismatches: {len(mismatches)}")
    if mismatches:
        print("  First 20 mismatches:")
        for m in mismatches[:20]:
            print(f"    {m}")

    # --- Step 4: Drop accounting ---
    print("\n--- Step 4: Raw → export drop accounting ---")
    n_csv_total = len(csv_labels)
    n_meta_total = len(meta_files)
    n_in_both = compared
    n_csv_only = only_csv
    n_labels_only = only_labels

    print(f"  Raw CSV contracts: {n_csv_total}")
    print(f"  Preprocessed meta.json: {n_meta_total}")
    print(f"  In both (compared): {n_in_both}")
    print(f"  CSV-only (have CSV row, no meta.json): {n_csv_only}")
    print(f"  Labels-only (have labels.json, no CSV row): {n_labels_only}")
    print(f"  Drop count: {n_csv_total - n_in_both}")

    # Summary of dropped: which contractIDs?
    dropped_cids = sorted(set(csv_labels.keys()) - set(actual.keys()))
    print(f"\n  Dropped contractIDs (CSV but no labels.json): {len(dropped_cids)}")
    if dropped_cids:
        print(f"  Range: {dropped_cids[0]} .. {dropped_cids[-1]}")
        print(f"  First 10: {dropped_cids[:10]}")
        # Check if these exist in __source__
        source_dir = PROJECT / "data_module/data/raw_staging/dive/__source__"
        # Quick check - just sample first 5
        for dc in dropped_cids[:5]:
            sol_file = source_dir / f"{dc}.sol"
            print(f"    {dc}.sol exists in __source__: {sol_file.exists()}")
            if not sol_file.exists():
                # Check if there's a symlink
                print(f"      (not a file — might be symlink?)")

    print("\nDone.")


if __name__ == "__main__":
    main()
