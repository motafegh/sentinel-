"""
Creates stratified train/val/test split indices.

Reads multilabel_index.csv and derives binary labels (any vuln class = 1).
Saves indices as .npy files in ml/data/splits/.

Split ratio: 70% train / 15% val / 15% test
Stratified: preserves label distribution in each split.

Note: label_index_path is IGNORED — binary labels are derived from
multilabel_index.csv (sum of class columns > 0) because ast_extractor.py
hardcodes graph.y=0 for all contracts, making label_index.csv unusable
for stratification.

--freeze-val-test mode (use when adding augmented data):
  Keeps existing val_indices.npy and test_indices.npy unchanged.
  Only rows NOT already covered by val/test are eligible for training.
  All newly added rows (indices beyond the original N) go to train only.
  This preserves comparability with previous experiments.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_splits(
    label_index_path: str,
    splits_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    multilabel_index_path: str = "ml/data/processed/multilabel_index.csv",
    freeze_val_test: bool = False,
) -> None:

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # === STEP 1: Derive binary labels from multilabel_index.csv ===
    # ast_extractor.py hardcodes graph.y=0 for all contracts, so label_index.csv
    # always shows 100% safe. Instead, use multilabel_index.csv: any contract
    # with at least one positive vulnerability class is labelled vulnerable (1).
    multilabel_path = Path(multilabel_index_path)
    if not multilabel_path.is_absolute():
        multilabel_path = project_root / multilabel_path
    df = pd.read_csv(multilabel_path)
    class_cols = [c for c in df.columns if c != "md5_stem"]
    print(f"Loaded multilabel index: {len(df)} samples, {len(class_cols)} classes")

    all_indices = np.arange(len(df))
    all_labels = (df[class_cols].sum(axis=1) > 0).astype(int).values

    splits_dir_path = Path(splits_dir)
    splits_dir_path.mkdir(parents=True, exist_ok=True)

    if freeze_val_test:
        # === FREEZE MODE: preserve existing val/test, assign new rows to train ===
        val_file  = splits_dir_path / "val_indices.npy"
        test_file = splits_dir_path / "test_indices.npy"
        train_file = splits_dir_path / "train_indices.npy"

        if not val_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                "--freeze-val-test requires existing val_indices.npy and "
                "test_indices.npy in the splits directory. Run without "
                "--freeze-val-test first to generate the initial splits."
            )

        existing_val  = np.load(val_file)
        existing_test = np.load(test_file)
        existing_train = np.load(train_file) if train_file.exists() else np.array([], dtype=np.int64)

        frozen_indices = set(existing_val.tolist()) | set(existing_test.tolist())
        original_n = len(existing_val) + len(existing_test) + len(existing_train)

        # All indices not in val/test go to train. This covers:
        #   - original train indices (unchanged)
        #   - any new indices (rows added to multilabel_index.csv beyond original_n)
        new_train_indices = np.array(
            [i for i in all_indices if i not in frozen_indices],
            dtype=np.int64,
        )

        new_count = max(0, len(df) - original_n)
        print(f"\nFREEZE MODE: val/test indices locked")
        print(f"  Existing val:  {len(existing_val):,}")
        print(f"  Existing test: {len(existing_test):,}")
        print(f"  New rows added to train: {new_count:,}")
        print(f"  Total train: {len(new_train_indices):,}")

        np.save(splits_dir_path / "train_indices.npy", new_train_indices)
        # val and test are not touched — they stay on disk unchanged

        train_labels = all_labels[new_train_indices]
        val_labels   = all_labels[existing_val]
        test_labels  = all_labels[existing_test]

        print("\nSplit Summary (after freeze):")
        print(f"{'Split':<10} {'Samples':>10} {'% Total':>10} {'% Vulnerable':>15}")
        print("-" * 50)
        for name, indices, labels in [
            ("Train", new_train_indices, train_labels),
            ("Val",   existing_val,      val_labels),
            ("Test",  existing_test,     test_labels),
        ]:
            vuln_pct  = labels.mean() * 100
            total_pct = len(indices) / len(df) * 100
            print(f"{name:<10} {len(indices):>10} {total_pct:>9.1f}% {vuln_pct:>14.1f}%")

        print(f"\nVal/test indices unchanged. New train_indices.npy saved to: {splits_dir_path}")
        return

    # === STEP 2: Full stratified split ===
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,
        random_state=random_seed
    )

    # === STEP 3: Split temp into val and test ===
    temp_labels = all_labels[temp_indices]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_seed
    )

    # === STEP 4: Save indices ===
    np.save(splits_dir_path / "train_indices.npy", train_indices)
    np.save(splits_dir_path / "val_indices.npy", val_indices)
    np.save(splits_dir_path / "test_indices.npy", test_indices)

    # === STEP 5: Verify ===
    print("\nSplit Summary:")
    print(f"{'Split':<10} {'Samples':>10} {'% Total':>10} {'% Vulnerable':>15}")
    print("-" * 50)

    for name, indices in [("Train", train_indices), ("Val", val_indices), ("Test", test_indices)]:
        split_labels = all_labels[indices]
        vuln_pct = split_labels.mean() * 100
        total_pct = len(indices) / len(df) * 100
        print(f"{name:<10} {len(indices):>10} {total_pct:>9.1f}% {vuln_pct:>14.1f}%")

    train_set = set(train_indices)
    val_set   = set(val_indices)
    test_set  = set(test_indices)

    assert len(train_set & val_set)  == 0, "Overlap between train and val!"
    assert len(train_set & test_set) == 0, "Overlap between train and test!"
    assert len(val_set   & test_set) == 0, "Overlap between val and test!"
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(df), \
        "Splits don't cover all samples!"

    print("\nVerification passed: no overlaps, full coverage")
    print(f"Indices saved to: {splits_dir_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits from multilabel_index.csv"
    )
    parser.add_argument(
        "--splits-dir",
        default="ml/data/splits",
        help="Directory to save/read split .npy files",
    )
    parser.add_argument(
        "--multilabel-csv",
        default="ml/data/processed/multilabel_index.csv",
        help="Path to multilabel_index.csv",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--freeze-val-test",
        action="store_true",
        help=(
            "Preserve existing val_indices.npy and test_indices.npy unchanged. "
            "Only assigns indices not already in val/test to train. "
            "Use this when adding augmented data to avoid contaminating the "
            "original validation set used for experiment comparability."
        ),
    )
    args = parser.parse_args()

    create_splits(
        label_index_path="ml/data/processed/label_index.csv",  # unused; kept for API compat
        splits_dir=args.splits_dir,
        multilabel_index_path=args.multilabel_csv,
        random_seed=args.seed,
        freeze_val_test=args.freeze_val_test,
    )
