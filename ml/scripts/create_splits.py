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

    # indices = positions in the sorted list (0 to N-1)
    all_indices = np.arange(len(df))
    all_labels = (df[class_cols].sum(axis=1) > 0).astype(int).values

    # === STEP 2: Split into train and temp (val+test) ===
    # stratify=all_labels tells sklearn: preserve label ratio in both splits
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,           # <-- this is the key parameter
        random_state=random_seed       # <-- reproducibility
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
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(splits_dir / "train_indices.npy", train_indices)
    np.save(splits_dir / "val_indices.npy", val_indices)
    np.save(splits_dir / "test_indices.npy", test_indices)

    # === STEP 5: Verify ===
    print("\nSplit Summary:")
    print(f"{'Split':<10} {'Samples':>10} {'% Total':>10} {'% Vulnerable':>15}")
    print("-" * 50)

    for name, indices in [("Train", train_indices), ("Val", val_indices), ("Test", test_indices)]:
        split_labels = all_labels[indices]
        vuln_pct = split_labels.mean() * 100
        total_pct = len(indices) / len(df) * 100
        print(f"{name:<10} {len(indices):>10} {total_pct:>9.1f}% {vuln_pct:>14.1f}%")

    # Verify no overlap between splits
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    assert len(train_set & val_set) == 0, "Overlap between train and val!"
    assert len(train_set & test_set) == 0, "Overlap between train and test!"
    assert len(val_set & test_set) == 0, "Overlap between val and test!"
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(df), \
        "Splits don't cover all samples!"

    print("\nVerification passed: no overlaps, full coverage")
    print(f"Indices saved to: {splits_dir}")


if __name__ == "__main__":
    create_splits(
        label_index_path="ml/data/processed/label_index.csv",  # unused; kept for API compat
        splits_dir="ml/data/splits",
        multilabel_index_path="ml/data/processed/multilabel_index.csv",
    )
