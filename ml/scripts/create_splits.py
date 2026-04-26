"""
Creates stratified train/val/test split indices.

Reads label_index.csv (lightweight, ~3MB).
Saves indices as .npy files in ml/data/splits/.

Split ratio: 70% train / 15% val / 15% test
Stratified: preserves label distribution in each split.
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
    random_seed: int = 42
) -> None:

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # === STEP 1: Load label index (instant — tiny CSV) ===
    df = pd.read_csv(label_index_path)
    print(f"Loaded label index: {len(df)} samples")

    # indices = positions in the sorted list (0 to 68554)
    all_indices = np.arange(len(df))
    all_labels = df['label'].values

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
        label_index_path="ml/data/processed/label_index.csv",
        splits_dir="ml/data/splits"
    )
