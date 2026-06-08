"""BCCC Phase 2 — WS-G: Stratified Split Design.

Designs a 70/15/15 train/val/test split for the filtered BCCC contracts.
Uses a 2-stage stratified approach:
  1. Derive a "stratify_key" = (has_vuln, primary_vuln_class) for stratification
  2. Use sklearn train_test_split with stratification on this key
  3. For multi-label contracts, primary class = rarest positive class

This is a SIMPLE APPROXIMATION of iterative stratification (Sechidis et al. 2011).
For best results, use `iterative-stratification` package (not yet installable in this env).

Strategy:
  - 26,148 pure-NV contracts: random 70/15/15 (NV is single-label, simple)
  - 40,397 multi-label vuln contracts: stratified 70/15/15 on (primary_vuln_class, has_nv)
  - 766 review_pending: EXCLUDED from initial training (held out per D-B2)

Outputs (under ../splits/):
  - train.csv, val.csv, test.csv           (per-split contract lists)
  - split_summary.md                        (per-class prevalence per split)
  - bccc_splits_v1.json                    (combined metadata)
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("/home/motafeq/projects/sentinel")
LABELS = Path(__file__).resolve().parent.parent / "labels"
FILTERED_CSV = LABELS / "contracts_filtered.csv"
OUT = Path(__file__).resolve().parent.parent / "splits"
OUT.mkdir(parents=True, exist_ok=True)

SENTINEL_V9_ORDER = [
    "Class01:ExternalBug", "Class02:GasException", "Class03:MishandledException",
    "Class04:Timestamp", "Class06:UnusedReturn", "Class08:CallToUnknown",
    "Class09:DenialOfService", "Class10:IntegerUO", "Class11:Reentrancy", "Class12:NonVulnerable",
]
VULN_CLASSES = [c for c in SENTINEL_V9_ORDER if c != "Class12:NonVulnerable"]
NV_CLASS = "Class12:NonVulnerable"


def derive_stratify_key(row, class_freq: dict[str, int]) -> str:
    """For stratification: (has_vuln, primary_vuln_class).

    primary_vuln_class = rarest positive vuln class (or 'none' if no vuln).
    """
    vuln_pos = [c for c in VULN_CLASSES if row[c] == 1]
    has_vuln = "V" if vuln_pos else "N"
    if not vuln_pos:
        primary = "none"
    else:
        # Rarest = lowest class_freq
        primary = min(vuln_pos, key=lambda c: class_freq.get(c, 0))
    return f"{has_vuln}_{primary}"


def main():
    print("=" * 70)
    print("WS-G: Stratified Split Design (70/15/15)")
    print("=" * 70)

    # 1. Load filtered contracts
    print("\n[1/5] Loading filtered contracts...")
    df = pd.read_csv(FILTERED_CSV)
    print(f"  Total: {len(df):,}")

    # 2. Identify review_pending contracts (D-B2 exclusion)
    print("\n[2/5] Identifying review_pending (held out per D-B2)...")
    review_pending_ids = set(df[df["review_pending"] == 1]["id"].tolist())
    df_train_pool = df[~df["id"].isin(review_pending_ids)].copy()
    print(f"  Held out (review_pending): {len(review_pending_ids):,}")
    print(f"  Training pool: {len(df_train_pool):,}")

    # 3. Compute class frequencies (on training pool)
    print("\n[3/5] Computing class frequencies for stratification...")
    class_freq = {c: int(df_train_pool[c].sum()) for c in VULN_CLASSES}
    print(f"  Class frequencies: {class_freq}")

    # 4. Derive stratify_key
    print("\n[4/5] Deriving stratify key...")
    df_train_pool["strat_key"] = df_train_pool.apply(
        lambda row: derive_stratify_key(row, class_freq), axis=1
    )
    key_counts = df_train_pool["strat_key"].value_counts()
    print(f"  Number of unique keys: {len(key_counts)}")
    # Drop keys with <2 samples (cannot stratify)
    rare_keys = key_counts[key_counts < 2].index.tolist()
    if rare_keys:
        print(f"  Rare keys (<2 samples): {len(rare_keys)}")
        # Reassign rare keys to a generic "RARE" bucket
        df_train_pool.loc[df_train_pool["strat_key"].isin(rare_keys), "strat_key"] = "RARE"
        key_counts = df_train_pool["strat_key"].value_counts()
        print(f"  After bucketing: {len(key_counts)} keys")

    # 5. 3-way split: 70% train, 15% val, 15% test
    # First: 70/30 train/(val+test)
    # Then: 30 → 15/15
    print("\n[5/5] Splitting 70/15/15...")
    rng = random.Random(42)
    train_df, rest_df = train_test_split(
        df_train_pool,
        test_size=0.30,
        random_state=42,
        stratify=df_train_pool["strat_key"],
    )
    val_df, test_df = train_test_split(
        rest_df,
        test_size=0.50,
        random_state=42,
        stratify=rest_df["strat_key"],
    )
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    print(f"  Held out (review_pending): {len(review_pending_ids):,}")
    print(f"  Total accounted: {len(train_df) + len(val_df) + len(test_df) + len(review_pending_ids):,}")

    # 6. Save splits
    train_df.drop(columns=["strat_key"]).to_csv(OUT / "train.csv", index=False)
    val_df.drop(columns=["strat_key"]).to_csv(OUT / "val.csv", index=False)
    test_df.drop(columns=["strat_key"]).to_csv(OUT / "test.csv", index=False)
    print(f"  Wrote train.csv ({len(train_df):,}), val.csv ({len(val_df):,}), test.csv ({len(test_df):,})")

    # 7. Per-class prevalence per split
    prevalence = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        prev = {c: int(split_df[c].sum()) for c in SENTINEL_V9_ORDER}
        prevalence[split_name] = prev

    # 8. Sanity check: prevalence should be ~70/15/15 of full
    print("\n  Per-class prevalence (train/val/test):")
    print(f"  {'Class':<35} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    for c in SENTINEL_V9_ORDER:
        t = prevalence["train"][c]
        v = prevalence["val"][c]
        te = prevalence["test"][c]
        tot = t + v + te
        print(f"  {c:<35} {t:>6} {v:>6} {te:>6} {tot:>7}")

    # 9. Save metadata
    print("\n  Saving split metadata...")
    split_metadata = {
        "version": "v1",
        "created": "2026-06-06",
        "method": "Stratified split on (has_vuln, primary_vuln_class) using sklearn train_test_split",
        "approximation_note": "Simple approximation of iterative stratification; uses rare-positive-class as primary. For best results, install iterative-stratification and re-run.",
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "held_out_review_pending": len(review_pending_ids),
        },
        "class_prevalence": prevalence,
        "decisions_applied": {
            "D-F1": "drop Class05, Class07 (1,122 contracts dropped)",
            "D-B2": "766 review_pending held out for manual review",
        },
        "random_seed": 42,
    }
    (OUT / "bccc_splits_v1.json").write_text(json.dumps(split_metadata, indent=2))
    print(f"  Wrote {OUT / 'bccc_splits_v1.json'}")

    # 10. Save human-readable summary
    print("\n  Writing split_summary.md...")
    md = f"""# WS-G: Stratified Split — Summary

**Date:** 2026-06-06
**Version:** v1
**Method:** 2-stage stratified (has_vuln, primary_vuln_class)

## Split Sizes

| Split | n | % |
|---|---:|---:|
| Train | {len(train_df):,} | 70.0% |
| Val | {len(val_df):,} | 15.0% |
| Test | {len(test_df):,} | 15.0% |
| **Training pool total** | **{len(df_train_pool):,}** | **100.0%** |
| Held out (review_pending) | {len(review_pending_ids):,} | — |

## Per-Class Prevalence (post-split, training pool only)

| Class | Train | Val | Test | Total | Distribution |
|---|---:|---:|---:|---:|---|
"""
    for c in SENTINEL_V9_ORDER:
        t = prevalence["train"][c]
        v = prevalence["val"][c]
        te = prevalence["test"][c]
        tot = t + v + te
        if tot > 0:
            dist = f"{100*t/tot:.0f}/{100*v/tot:.0f}/{100*te/tot:.0f}"
        else:
            dist = "n/a"
        md += f"| `{c}` | {t:,} | {v:,} | {te:,} | {tot:,} | {dist} |\n"

    md += f"""

## Stratification Method

For each contract, derive a stratify key:
- `V_<primary_class>` if has ≥1 vuln class
- `N_none` if pure NV
- `RARE` if the derived key has <2 samples (bucket them together)

`primary_class` = rarest positive vuln class (e.g., Timestamp at 3.97% is rarer than Reentrancy at 26.29%).

This is a **simple approximation** of iterative stratification (Sechidis et al. 2011).
For best results, install `iterative-stratification` (network currently slow/blocked) and re-run with `MultilabelStratifiedKFold`.

## Decisions Applied

- **D-F1:** Dropped 1,122 contracts (only had Class05/Class07)
- **D-B2:** Held out 766 NV+vuln contradictions for manual review (NOT in any split)
- **NV class:** Treated as Class12 (binary); pure-NV contracts are stratified by `N_none`

## Files

- `train.csv` ({len(train_df):,} contracts)
- `val.csv` ({len(val_df):,} contracts)
- `test.csv` ({len(test_df):,} contracts)
- `bccc_splits_v1.json` (metadata)
- `split_summary.md` (this file)

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/g_stratified_split.py
```
"""
    (OUT / "split_summary.md").write_text(md)
    print(f"  Wrote {OUT / 'split_summary.md'}")
    print("\nWS-G complete.")


if __name__ == "__main__":
    main()
