"""Method 2 — Verify DIVE folder symlinks match CSV per-contract.

Provenance:
  Command: python3 scripts/verify_folder_csv_agreement.py
  RNG seed: N/A (full corpus, 22,330 contracts × 8 classes)
  Tool: Python 3.12, stdlib only
  Input CSV: data_module/data/raw_staging/dive_labels/DIVE_Labels.csv (raw frame, 22,330 rows)
  Input folders: data_module/data/raw_staging/dive/{8 class folders} (raw frame, 54,919 symlinks)
  Timestamp: 2026-06-18
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

PROJECT = Path("/home/motafeq/projects/sentinel")
CSV_PATH = PROJECT / "data_module/data/raw_staging/dive_labels/DIVE_Labels.csv"
DIVE_DIR = PROJECT / "data_module/data/raw_staging/dive"

# DIVE folder names (matching CSV column names)
FOLDERS = [
    "Access Control",
    "Arithmetic",
    "Bad Randomness",
    "DoS",
    "Front Running",
    "Reentrancy",
    "Time manipulation",
    "Unchecked Return Values",
]

CSV_COL_TO_FOLDER = {
    "Access Control": "Access Control",
    "Arithmetic": "Arithmetic",
    "Bad Randomness": "Bad Randomness",
    "DoS": "DoS",
    "Front Running": "Front Running",
    "Reentrancy": "Reentrancy",
    "Time manipulation": "Time manipulation",
    "Unchecked Return Values": "Unchecked Return Values",
}


def main() -> None:
    # Step 1: Read CSV → {contractID: set(DIVE column names where =1)}
    print("Step 1: Reading CSV...")
    csv_labels: dict[int, set[str]] = {}
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["contractID"])
            positive = {col for col in CSV_COL_TO_FOLDER if row.get(col, "0") == "1"}
            csv_labels[cid] = positive
    print(f"  CSV rows: {len(csv_labels)}")

    # Step 2: Read folders → {contractID: set(folder names where symlink exists)}
    print("Step 2: Reading folder symlinks...")
    folder_labels: dict[int, set[str]] = defaultdict(set)
    folder_file_counts: dict[str, int] = {}
    broken_symlinks = 0

    for folder_name in FOLDERS:
        folder_path = DIVE_DIR / folder_name
        count = 0
        for sol_file in folder_path.glob("*.sol"):
            try:
                cid = int(sol_file.name.replace(".sol", ""))
            except ValueError:
                continue
            folder_labels[cid].add(folder_name)
            count += 1
        folder_file_counts[folder_name] = count
        print(f"  {folder_name}: {count} symlinks")

    # Step 3: Compare per-contract, per-class
    print("\nStep 3: Comparing...")
    per_class_match: dict[str, int] = defaultdict(int)
    per_class_miss_csv1_folder0: dict[str, int] = defaultdict(int)
    per_class_miss_csv0_folder1: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)
    mismatches: list[dict] = []

    all_cids = set(csv_labels.keys()) | set(folder_labels.keys())
    only_csv = all_cids - set(folder_labels.keys())
    only_folder = all_cids - set(csv_labels.keys())

    for cid in sorted(all_cids):
        csv_set = csv_labels.get(cid, set())
        folder_set = folder_labels.get(cid, set())

        for csv_col, folder_name in CSV_COL_TO_FOLDER.items():
            csv_val = 1 if csv_col in csv_set else 0
            folder_val = 1 if folder_name in folder_set else 0
            per_class_total[csv_col] += 1

            if csv_val == folder_val:
                per_class_match[csv_col] += 1
            else:
                if csv_val == 1 and folder_val == 0:
                    per_class_miss_csv1_folder0[csv_col] += 1
                else:
                    per_class_miss_csv0_folder1[csv_col] += 1
                mismatches.append({
                    "contractID": cid,
                    "class": csv_col,
                    "csv": csv_val,
                    "folder": folder_val,
                })

    # Results
    print(f"\n  Total cids compared: {len(all_cids)}")
    print(f"  Only in CSV (no folder entries): {len(only_csv)}")
    print(f"  Only in folders (no CSV row): {len(only_folder)}")
    print(f"  Total mismatches across 22,330 × 8 = {22330 * 8} pairs: {len(mismatches)}")

    if only_csv:
        print(f"  Only-CSV cids (first 10): {sorted(only_csv)[:10]}")

    print("\n--- Per-class agreement ---")
    for csv_col in CSV_COL_TO_FOLDER:
        match = per_class_match[csv_col]
        total = per_class_total[csv_col]
        csv1_f0 = per_class_miss_csv1_folder0[csv_col]
        csv0_f1 = per_class_miss_csv0_folder1[csv_col]
        pct = (match / total * 100) if total > 0 else 0.0
        issues = []
        if csv1_f0 > 0:
            issues.append(f"CSV=1/folder=0: {csv1_f0}")
        if csv0_f1 > 0:
            issues.append(f"CSV=0/folder=1: {csv0_f1}")
        issue_str = " | " + ", ".join(issues) if issues else ""
        print(f"  {csv_col}: {match}/{total} ({pct:.4f}%){issue_str}")

    if mismatches:
        print(f"\n  First 20 mismatches:")
        for m in mismatches[:20]:
            print(f"    cid={m['contractID']}, class={m['class']}, csv={m['csv']}, folder={m['folder']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
