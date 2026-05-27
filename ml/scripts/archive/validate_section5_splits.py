"""
SECTION 5: Split integrity
"""
import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path("/home/motafeq/projects/sentinel/ml/data/processed/multilabel_index_deduped.csv")
SPLITS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/splits/deduped")

CLASSES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn"
]

def main():
    df = pd.read_csv(CSV_PATH)
    total = len(df)

    train_idx = np.load(SPLITS_DIR / "train_indices.npy")
    val_idx = np.load(SPLITS_DIR / "val_indices.npy")
    test_idx = np.load(SPLITS_DIR / "test_indices.npy")

    print(f"CSV rows: {total:,}")
    print(f"Train indices: {len(train_idx):,}")
    print(f"Val indices:   {len(val_idx):,}")
    print(f"Test indices:  {len(test_idx):,}")

    print("\n" + "=" * 70)
    print("SECTION 5: SPLIT INTEGRITY")
    print("=" * 70)

    total_split = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"\n--- Count check ---")
    print(f"  train+val+test = {total_split:,} (expected {total:,}) {'PASS' if total_split == total else 'FAIL'}")

    # Overlap checks
    tr_set = set(train_idx.tolist())
    va_set = set(val_idx.tolist())
    te_set = set(test_idx.tolist())

    tr_va = tr_set & va_set
    tr_te = tr_set & te_set
    va_te = va_set & te_set

    print(f"\n--- Overlap checks ---")
    print(f"  train ∩ val overlap:   {len(tr_va):,} ({'FAIL' if tr_va else 'PASS'})")
    print(f"  train ∩ test overlap:  {len(tr_te):,} ({'FAIL' if tr_te else 'PASS'})")
    print(f"  val ∩ test overlap:    {len(va_te):,} ({'FAIL' if va_te else 'PASS'})")

    # Out-of-bounds indices
    oor_tr = ((train_idx < 0) | (train_idx >= total)).sum()
    oor_va = ((val_idx < 0) | (val_idx >= total)).sum()
    oor_te = ((test_idx < 0) | (test_idx >= total)).sum()
    print(f"\n--- Index range checks ---")
    print(f"  OOB train indices: {oor_tr} ({'FAIL' if oor_tr else 'PASS'})")
    print(f"  OOB val indices:   {oor_va} ({'FAIL' if oor_va else 'PASS'})")
    print(f"  OOB test indices:  {oor_te} ({'FAIL' if oor_te else 'PASS'})")

    # Per-class positive rate per split
    present_classes = [c for c in CLASSES if c in df.columns]
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    print(f"\n--- Per-class positive rate by split ---")
    print(f"{'Class':<30} {'Train%':>8} {'Val%':>8} {'Test%':>8} {'Status'}")
    print("-" * 62)
    for c in present_classes:
        tr_r = 100. * train_df[c].sum() / len(train_df)
        va_r = 100. * val_df[c].sum() / len(val_df)
        te_r = 100. * test_df[c].sum() / len(test_df)
        # Check if rates are within 5pp of each other (stratification quality)
        max_diff = max(abs(tr_r - va_r), abs(tr_r - te_r), abs(va_r - te_r))
        status = "PASS" if max_diff < 5.0 else "WARN"
        print(f"  {c:<28} {tr_r:>7.2f}% {va_r:>7.2f}% {te_r:>7.2f}% {status}")

    # Label density per split
    print(f"\n--- Label density (positives/row) per split ---")
    tr_density = train_df[present_classes].sum(axis=1).mean()
    va_density = val_df[present_classes].sum(axis=1).mean()
    te_density = test_df[present_classes].sum(axis=1).mean()
    print(f"  Train: {tr_density:.4f}")
    print(f"  Val:   {va_density:.4f}")
    print(f"  Test:  {te_density:.4f}")

    # Coverage — are all CSV indices covered?
    all_split = tr_set | va_set | te_set
    missing_from_splits = set(range(total)) - all_split
    extra_in_splits = all_split - set(range(total))
    print(f"\n--- Coverage ---")
    print(f"  Indices missing from any split: {len(missing_from_splits):,} ({'FAIL' if missing_from_splits else 'PASS'})")
    print(f"  Indices outside [0,{total}):     {len(extra_in_splits):,} ({'FAIL' if extra_in_splits else 'PASS'})")

if __name__ == "__main__":
    main()
