"""
SECTION 3: CSV label integrity
"""
import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path("/home/motafeq/projects/sentinel/ml/data/processed/multilabel_index_deduped.csv")
CLASSES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn"
]

def main():
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"Loaded CSV: {total} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    print("\n" + "=" * 70)
    print("SECTION 3: CSV LABEL INTEGRITY")
    print("=" * 70)

    # Check which label columns exist
    present_classes = [c for c in CLASSES if c in df.columns]
    missing_classes = [c for c in CLASSES if c not in df.columns]
    print(f"\n--- Label columns ---")
    print(f"  Present: {present_classes}")
    if missing_classes:
        print(f"  MISSING: {missing_classes}  (FAIL)")

    # Check md5 column
    md5_col = None
    for c in ['md5_stem', 'md5', 'hash', 'contract_hash', 'id']:
        if c in df.columns:
            md5_col = c
            break
    print(f"  MD5/hash column: {md5_col}")

    # Duplicates
    if md5_col:
        dupes = df[md5_col].duplicated().sum()
        print(f"\n--- Duplicates ---")
        print(f"  Duplicate md5s: {dupes:,} ({'FAIL' if dupes > 0 else 'PASS'})")

    # Per-class counts
    print(f"\n--- Per-class label counts ---")
    print(f"{'Class':<30} {'Count':>8} {'%':>8} {'Status'}")
    print("-" * 55)
    for c in present_classes:
        cnt = df[c].sum()
        pct = 100. * cnt / total
        status = "WARN" if cnt < 500 else "PASS"
        print(f"  {c:<28} {cnt:>8,} {pct:>7.1f}% {status}")

    # Multi-label distribution
    label_df = df[present_classes]
    label_counts = label_df.sum(axis=1)

    print(f"\n--- Label density per row ---")
    for k in range(0, 6):
        cnt = (label_counts == k).sum()
        pct = 100. * cnt / total
        status = "WARN" if k == 0 and pct > 30 else "PASS"
        print(f"  {k} labels: {cnt:>8,} ({pct:.1f}%) {status}")
    cnt_6plus = (label_counts >= 6).sum()
    print(f"  6+ labels: {cnt_6plus:>8,} ({100.*cnt_6plus/total:.1f}%)")

    # Co-occurrence matrix
    print(f"\n--- Co-occurrence matrix (absolute counts) ---")
    co = label_df.T.dot(label_df)
    print(co.to_string())

    # DoS co-occurrence specifically
    print(f"\n--- DenialOfService co-occurrence ---")
    if "DenialOfService" in present_classes:
        dos_rows = df[df["DenialOfService"] == 1]
        dos_total = len(dos_rows)
        print(f"  Total DoS samples: {dos_total}")
        for c in present_classes:
            if c != "DenialOfService":
                overlap = dos_rows[c].sum()
                print(f"    DoS ∩ {c:<28}: {overlap:>5} ({100.*overlap/max(dos_total,1):.1f}%)")
    else:
        print("  DenialOfService column not found (FAIL)")

if __name__ == "__main__":
    main()
