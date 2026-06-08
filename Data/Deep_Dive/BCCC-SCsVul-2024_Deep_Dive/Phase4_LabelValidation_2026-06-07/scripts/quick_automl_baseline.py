"""
Quick AutoML baseline on contracts_clean.csv using only the 7 complexity
features already present in the CSV (no K1 regex, no Slither, no graphs).
Gives a fast picture of what tabular models achieve vs SENTINEL Run7 F1=0.31.

Features used:
  loc, n_functions, n_events, n_modifiers, n_pos, is_pure_nv, has_pragma

Approach:
  - 10 independent binary XGBoost classifiers (one per class)
  - Use split_assignments.csv for train/val/test (same split as SENTINEL)
  - Exclude review_pending=1 contracts (same exclusion as SENTINEL training)
  - Report per-class F1 + macro-F1 on test set
  - Compare to SENTINEL Run7/Run9 numbers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parents[2]
CSV  = BASE / "Phase2_Validation_2026-06-06/outputs/contracts_clean.csv"
SPLIT = BASE / "Phase2_Validation_2026-06-06/outputs/split_assignments.csv"

LABEL_COLS = [
    "Class01:ExternalBug",
    "Class02:GasException",
    "Class03:MishandledException",
    "Class04:Timestamp",
    "Class06:UnusedReturn",
    "Class08:CallToUnknown",
    "Class09:DenialOfService",
    "Class10:IntegerUO",
    "Class11:Reentrancy",
    "Class12:NonVulnerable",
]

FEATURE_COLS = ["loc", "n_functions", "n_events", "n_modifiers",
                "n_pos", "is_pure_nv", "has_pragma"]

SENTINEL_RUN7  = {"Class11:Reentrancy": 0.307, "Class10:IntegerUO": 0.698,
                  "Class02:GasException": 0.376, "Class03:MishandledException": 0.324,
                  "Class06:UnusedReturn": 0.238, "Class09:DenialOfService": 0.164,
                  "Class04:Timestamp": 0.164, "__macro__": 0.307}

def main():
    print("Loading data...")
    df   = pd.read_csv(CSV)
    spl  = pd.read_csv(SPLIT)

    # merge split assignments
    df = df.merge(spl, on="id", how="left")
    print(f"  Total contracts: {len(df)}")
    print(f"  Split coverage: {df['split'].notna().sum()} have split assignment")

    # exclude review_pending
    df_use = df[df["review_pending"] == 0].copy()
    print(f"  After excluding review_pending=1: {len(df_use)}")

    # fill any NaN in features
    df_use[FEATURE_COLS] = df_use[FEATURE_COLS].fillna(0)

    train = df_use[df_use["split"] == "train"]
    val   = df_use[df_use["split"] == "val"]
    test  = df_use[df_use["split"] == "test"]
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    X_train = train[FEATURE_COLS].values
    X_val   = val[FEATURE_COLS].values
    X_test  = test[FEATURE_COLS].values

    results = {}
    val_results = {}

    print(f"\nTraining 10 XGBoost classifiers ({len(FEATURE_COLS)} features each)...\n")
    print(f"{'Class':<35} {'Val F1':>8} {'Test F1':>8} {'Support':>8} {'Run7 F1':>8}")
    print("-" * 75)

    macro_val_f1s  = []
    macro_test_f1s = []

    for col in LABEL_COLS:
        if col not in df_use.columns:
            continue

        y_train = train[col].values
        y_val   = val[col].values
        y_test  = test[col].values

        n_pos_train = y_train.sum()
        if n_pos_train < 10:
            print(f"{col:<35} {'SKIP (< 10 pos)':>25}")
            continue

        scale_pos = (y_train == 0).sum() / max(n_pos_train, 1)

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=4,
            verbosity=0,
        )

        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)

        val_pred  = clf.predict(X_val)
        test_pred = clf.predict(X_test)

        val_f1  = f1_score(y_val,  val_pred,  zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        support = int(y_test.sum())

        r7 = SENTINEL_RUN7.get(col, "-")
        r7_str = f"{r7:.3f}" if isinstance(r7, float) else r7

        print(f"{col:<35} {val_f1:>8.3f} {test_f1:>8.3f} {support:>8} {r7_str:>8}")

        results[col]     = test_f1
        val_results[col] = val_f1
        macro_val_f1s.append(val_f1)
        macro_test_f1s.append(test_f1)

    print("-" * 75)
    macro_val  = np.mean(macro_val_f1s)
    macro_test = np.mean(macro_test_f1s)
    print(f"{'MACRO (mean)':.<35} {macro_val:>8.3f} {macro_test:>8.3f} {'':>8} {'~0.307':>8}")

    print(f"\n{'='*75}")
    print(f"SENTINEL Run 7 macro-F1  (tuned, test): 0.3074")
    print(f"SENTINEL Run 9 macro-F1  (ep14 best):   0.2586")
    print(f"XGBoost complexity-only  (test):         {macro_test:.4f}")
    print(f"{'='*75}")

    gap = macro_test - 0.3074
    print(f"\nGap vs Run 7: {gap:+.4f}  ({'XGBoost WINS' if gap > 0 else 'SENTINEL wins by ' + str(abs(round(gap,3)))})")

    print(f"\nFeatures used ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print("Note: NO K1 regex, NO Slither, NO graph structure — pure CSV complexity counts.")


if __name__ == "__main__":
    main()
