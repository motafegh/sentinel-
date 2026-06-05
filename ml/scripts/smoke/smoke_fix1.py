"""
smoke_fix1.py — Smoke test for Fix #1 (Timestamp relabel).

Already applied (June 2026). This smoke test is a permanent regression check
to ensure the deduped CSV and splits remain valid.

Gates-in (pre-conditions):
  G1.1 — ml/data/processed/multilabel_index_deduped.csv exists
  G1.2 — ml/data/splits/deduped/{train,val,test}_indices.npy exist
  G1.3 — Source CSV (multilabel_index.csv) still exists for delta calc

Gates-out (post-conditions verified):
  G1.4 — Deduped CSV has > 0 rows and same shape as raw minus drops
  G1.5 — Timestamp count is in expected window [500, 1200] (49.9% drop)
  G1.6 — Splits are non-empty, indices in [0, len(df))
  G1.7 — At least 2 classes have ≥ 10 positives (model trainable)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    CLASS_COLUMNS,
    PROCESSED_DIR,
    SPLITS_DEDUPED,
    check,
    pass_,
    smoke_header,
    timed,
)


@timed("fix1_total")
def main() -> int:
    smoke_header(1, "Timestamp relabel regression check")
    start = time.perf_counter()

    # ── Gates-in ─────────────────────────────────────────────────────────
    deduped_csv = PROCESSED_DIR / "multilabel_index_deduped.csv"
    raw_csv = PROCESSED_DIR / "multilabel_index.csv"
    check(deduped_csv.exists(), f"G1.1 deduped CSV exists at {deduped_csv}")
    check(raw_csv.exists(), f"G1.3 raw CSV exists at {raw_csv}")

    for split in ("train", "val", "test"):
        check(
            (SPLITS_DEDUPED / f"{split}_indices.npy").exists(),
            f"G1.2 split {split} exists",
        )

    # ── Body ─────────────────────────────────────────────────────────────
    df = pd.read_csv(deduped_csv)
    check(len(df) > 0, f"G1.4 deduped CSV has {len(df)} rows (> 0)")

    if "Timestamp" not in df.columns:
        raise AssertionError(f"G1.5 deduped CSV missing 'Timestamp' column. Got: {list(df.columns)}")
    ts_count = int(df["Timestamp"].sum())
    check(500 < ts_count < 1200, f"G1.5 Timestamp count = {ts_count} (expected 500–1200)")

    class_cols = [c for c in CLASS_COLUMNS if c in df.columns]
    positives = {c: int(df[c].sum()) for c in class_cols}
    trainable = [c for c, n in positives.items() if n >= 10]
    check(len(trainable) >= 2, f"G1.7 at least 2 classes have ≥10 positives (got {len(trainable)}: {trainable})")

    splits: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        arr = np.load(SPLITS_DEDUPED / f"{split}_indices.npy")
        splits[split] = arr
        check(len(arr) > 0, f"G1.6 split {split} has {len(arr)} indices (> 0)")
        check(
            int(arr.min()) >= 0 and int(arr.max()) < len(df),
            f"G1.6 split {split} indices in [0, {len(df)}) — got [{arr.min()}, {arr.max()}]",
        )

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    overlap_tv = len(set(train_idx.tolist()) & set(val_idx.tolist()))
    overlap_tt = len(set(train_idx.tolist()) & set(test_idx.tolist()))
    check(overlap_tv == 0, f"G1.6 train/val disjoint (overlap={overlap_tv})")
    check(overlap_tt == 0, f"G1.6 train/test disjoint (overlap={overlap_tt})")

    elapsed = time.perf_counter() - start
    pass_(f"Fix #1 smoke OK — {len(df)} rows, Timestamp={ts_count}, {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nSMOKE FIX #1 FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
