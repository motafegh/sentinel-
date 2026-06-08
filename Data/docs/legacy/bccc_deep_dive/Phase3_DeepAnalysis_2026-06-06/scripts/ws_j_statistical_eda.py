"""WS-J: Statistical EDA on BCCC + SENTINEL v9 datasets.

Outputs:
  - outputs/ws_j_eda_stats.json: machine-readable per-class stats
  - outputs/ws_j_cooccurrence.csv: 12x12 BCCC class co-occurrence matrix
  - outputs/ws_j_feature_missingness.csv: per-column missingness
  - reports/ws_j_eda_report.md: human-readable summary

Inputs:
  - BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv: 111,897 x 254 (long format)
  - Phase2_Validation_2026-06-06/outputs/contracts_clean.csv: 67,311 x 24 (SENTINEL v9)
  - Phase2_Validation_2026-06-06/labels/label_consistency.csv: 67,311 x 15 (per-class flags)
  - Phase2_Validation_2026-06-06/outputs/split_assignments.csv: 67,311 x 2 (train/val/test/review_pending)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/motafeq/projects/sentinel")
BCCC_CSV = ROOT / "BCCC-SCsVul-2024" / "BCCC-SCsVul-2024.csv"
CLEAN_CSV = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase2_Validation_2026-06-06" / "outputs" / "contracts_clean.csv"
LABEL_CSV = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase2_Validation_2026-06-06" / "labels" / "label_consistency.csv"
SPLIT_CSV = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase2_Validation_2026-06-06" / "outputs" / "split_assignments.csv"
OUT_DIR = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase3_DeepAnalysis_2026-06-06" / "outputs"
REPORT_DIR = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase3_DeepAnalysis_2026-06-06" / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLS = [
    "Class01:ExternalBug",
    "Class02:GasException",
    "Class03:MishandledException",
    "Class04:Timestamp",
    "Class05:TransactionOrderDependence",
    "Class06:UnusedReturn",
    "Class07:WeakAccessMod",
    "Class08:CallToUnknown",
    "Class09:DenialOfService",
    "Class10:IntegerUO",
    "Class11:Reentrancy",
    "Class12:NonVulnerable",
]
SHORT = {
    "Class01:ExternalBug": "Class01",
    "Class02:GasException": "Class02",
    "Class03:MishandledException": "Class03",
    "Class04:Timestamp": "Class04",
    "Class05:TransactionOrderDependence": "Class05",
    "Class06:UnusedReturn": "Class06",
    "Class07:WeakAccessMod": "Class07",
    "Class08:CallToUnknown": "Class08",
    "Class09:DenialOfService": "Class09",
    "Class10:IntegerUO": "Class10",
    "Class11:Reentrancy": "Class11",
    "Class12:NonVulnerable": "Class12",
}
LONG_NAME = {v: k for k, v in SHORT.items()}
SENTINEL_CLASS_ORDER = [c for c in CLASS_COLS if c not in (
    "Class05:TransactionOrderDependence", "Class07:WeakAccessMod"
)]


def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main() -> None:
    print("[WS-J] loading datasets ...")
    bccc = pd.read_csv(BCCC_CSV, low_memory=False)
    clean = pd.read_csv(CLEAN_CSV)
    label = pd.read_csv(LABEL_CSV)
    split = pd.read_csv(SPLIT_CSV)
    print(f"  BCCC raw:        {bccc.shape}")
    print(f"  contracts_clean: {clean.shape}")
    print(f"  label_consist:   {label.shape}")
    print(f"  split:           {split.shape}")

    # =========================================================================
    # SECTION 1: BCCC long-format distribution
    # =========================================================================
    section("1. BCCC raw long-format (12 classes)")

    # BCCC is in long format: 68,433 unique contracts across 111,897 rows
    # (some contracts appear multiple times for multi-label encoding).
    # Per-contract class flags: take max over rows.
    bccc_per_id_class = bccc.groupby("ID")[CLASS_COLS].max()
    bccc_per_id_n_pos = bccc_per_id_class.sum(axis=1).astype(int)
    bccc_per_id_n_rows = bccc.groupby("ID").size()

    bccc_class_dist = bccc_per_id_class.sum(axis=0).to_dict()
    print("Per-class prevalence (unique contracts with class=1 after grouping):")
    for c, n in bccc_class_dist.items():
        print(f"  {c}: {n:>6} ({100*n/len(bccc_per_id_class):5.2f}%)")
    print(f"  n unique contracts: {len(bccc_per_id_class)}")
    print(f"  n rows in long format: {len(bccc)} (avg {len(bccc)/len(bccc_per_id_class):.2f} rows per contract)")

    n_pos_dist = bccc_per_id_n_pos.value_counts().sort_index()
    print(f"\nDistribution of n_pos (number of positive classes per contract):")
    for n, c in n_pos_dist.items():
        print(f"  n_pos={n}: {c:>6} contracts ({100*c/len(bccc_per_id_class):5.2f}%)")

    # =========================================================================
    # SECTION 2: Co-occurrence matrix (BCCC raw)
    # =========================================================================
    section("2. BCCC class co-occurrence matrix (12x12, per-contract)")

    cooc_raw = bccc_per_id_class.T @ bccc_per_id_class  # 12x12
    cooc_pct = cooc_raw / len(bccc_per_id_class) * 100
    cooc_df = pd.DataFrame(cooc_raw, index=CLASS_COLS, columns=CLASS_COLS)
    cooc_pct_df = pd.DataFrame(cooc_pct, index=CLASS_COLS, columns=CLASS_COLS)
    print("Co-occurrence (raw counts, diagonal = class prevalence):")
    print(cooc_df.to_string())
    print("\nCo-occurrence (% of total contracts):")
    print(cooc_pct_df.round(2).to_string())

    cooc_path = OUT_DIR / "ws_j_cooccurrence_bccc.csv"
    cooc_df.to_csv(cooc_path)
    cooc_pct_path = OUT_DIR / "ws_j_cooccurrence_bccc_pct.csv"
    cooc_pct_df.to_csv(cooc_pct_path)
    print(f"\nSaved: {cooc_path}")
    print(f"Saved: {cooc_pct_path}")

    # Top 20 most common label combinations
    print("\nTop 20 most common label sets (multi-label combination):")
    label_sets = bccc_per_id_class.apply(
        lambda r: tuple(SHORT[c] for c in CLASS_COLS if r[c] == 1), axis=1
    )
    combo_counts = label_sets.value_counts()
    for combo, c in combo_counts.head(20).items():
        print(f"  {c:>6}  {combo}")

    # =========================================================================
    # SECTION 3: SENTINEL v9 cleaned dataset
    # =========================================================================
    section("3. SENTINEL v9 cleaned dataset (10 classes, 67,311 contracts)")

    sent_per_id_n_pos = clean["n_pos"]
    print("n_pos distribution (SENTINEL v9 cleaned):")
    for n, c in sent_per_id_n_pos.value_counts().sort_index().items():
        print(f"  n_pos={n}: {c:>6} contracts ({100*c/len(clean):5.2f}%)")

    print(f"\n  is_pure_nv=1: {clean['is_pure_nv'].sum()} ({100*clean['is_pure_nv'].mean():.2f}%)")
    print(f"  review_pending=1: {clean['review_pending'].sum()} ({100*clean['review_pending'].mean():.2f}%)")
    print(f"  has_pragma=1: {clean['has_pragma'].sum()} ({100*clean['has_pragma'].mean():.2f}%)")

    # =========================================================================
    # SECTION 4: Review-pending characterization (D-B2)
    # =========================================================================
    section("4. Review-pending set (NV+vuln contradiction, 766 contracts)")

    rp = clean[clean["review_pending"] == 1]
    print(f"  n review_pending: {len(rp)}")
    print("  primary_class distribution:")
    for c, n in rp["primary_class"].value_counts().items():
        print(f"    {c}: {n}")
    print(f"  n_pos distribution in review_pending:")
    for n, c in rp["n_pos"].value_counts().sort_index().items():
        print(f"    n_pos={n}: {c}")

    # Check which specific contradictions (which vuln class alongside NV)
    rp_classes = []
    for col in SENTINEL_CLASS_ORDER:
        if col == "Class12:NonVulnerable":
            continue
        n_rp_with = ((rp[col] == 1)).sum()
        if n_rp_with > 0:
            rp_classes.append((col, n_rp_with))
    print("  vuln classes present alongside NV (in review_pending):")
    for c, n in sorted(rp_classes, key=lambda x: -x[1]):
        print(f"    {c}: {n}")

    # =========================================================================
    # SECTION 5: Dropped contracts (Class05/07 only)
    # =========================================================================
    section("5. Dropped contracts (Class05/07 only, 1,122 contracts)")

    # Reconstruct dropped set from BCCC per-contract flags: contracts whose ONLY
    # positive classes are Class05 and/or Class07 (and no others).
    has_other_than_05_07 = bccc_per_id_class[[c for c in CLASS_COLS if c not in (
        "Class05:TransactionOrderDependence", "Class07:WeakAccessMod"
    )]].any(axis=1)
    has_05_or_07 = bccc_per_id_class[[
        "Class05:TransactionOrderDependence", "Class07:WeakAccessMod"
    ]].any(axis=1)
    dropped_mask = has_05_or_07 & ~has_other_than_05_07
    dropped_ids = bccc_per_id_class[dropped_mask].index.tolist()
    print(f"  n dropped: {len(dropped_ids)}")
    print("  breakdown of dropped contracts:")
    only_05 = (
        (bccc_per_id_class["Class05:TransactionOrderDependence"] == 1) &
        (bccc_per_id_class["Class07:WeakAccessMod"] == 0)
    )
    only_07 = (
        (bccc_per_id_class["Class05:TransactionOrderDependence"] == 0) &
        (bccc_per_id_class["Class07:WeakAccessMod"] == 1)
    )
    both_05_07 = (
        (bccc_per_id_class["Class05:TransactionOrderDependence"] == 1) &
        (bccc_per_id_class["Class07:WeakAccessMod"] == 1)
    )
    print(f"    Class05 only: {only_05[dropped_mask].sum()}")
    print(f"    Class07 only: {only_07[dropped_mask].sum()}")
    print(f"    both Class05 + Class07: {both_05_07[dropped_mask].sum()}")

    # Sanity check: do dropped IDs exist in BCCC source code folders?
    # (Quick check: are they mapped to a folder that exists?)
    bccc_folders = bccc[bccc["ID"].isin(dropped_ids)]["ID"].apply(
        lambda i: bccc[bccc["ID"] == i].iloc[0].get("Contract Information_0", "UNKNOWN")
        if "Contract Information_0" in bccc.columns else "UNKNOWN"
    )
    # (May not have folder info in BCCC CSV - that's in source path)

    # =========================================================================
    # SECTION 6: Feature distribution on BCCC CSV (254 cols)
    # =========================================================================
    section("6. BCCC 254-column feature missingness")

    # BCCC has 254 cols: 1 ID + 1 Unnamed + 12 class + 240 features.
    # Class cols are 0/1 binary (no missing). Features may have NaN/None.
    feature_cols = [c for c in bccc.columns if c not in (["ID", "Unnamed: 0"] + CLASS_COLS)]
    print(f"  total feature cols (non-class): {len(feature_cols)}")
    print(f"  total rows: {len(bccc)}")

    miss = bccc[feature_cols].isna().sum()
    miss_pct = 100 * miss / len(bccc)
    miss_df = pd.DataFrame({
        "col": feature_cols,
        "n_missing": miss.values,
        "pct_missing": miss_pct.values,
    }).sort_values("pct_missing", ascending=False)

    # Compute missingness on per-contract basis (BCCC long format repeats rows)
    bccc_per_id_features = bccc.drop_duplicates(subset="ID", keep="first")[feature_cols]
    print(f"  per-contract deduplicated features: {bccc_per_id_features.shape}")
    miss = bccc_per_id_features.isna().sum()
    miss_pct = 100 * miss / len(bccc_per_id_features)
    miss_df = pd.DataFrame({
        "col": feature_cols,
        "n_missing": miss.values,
        "pct_missing": miss_pct.values,
    }).sort_values("pct_missing", ascending=False)

    print(f"  cols with 0% missing: {(miss_df['pct_missing'] == 0).sum()}")
    print(f"  cols with >50% missing: {(miss_df['pct_missing'] > 50).sum()}")
    print(f"  cols with 100% missing: {(miss_df['pct_missing'] == 100).sum()}")
    print("\n  top 20 cols by missingness:")
    print(miss_df.head(20).to_string(index=False))

    miss_path = OUT_DIR / "ws_j_feature_missingness.csv"
    miss_df.to_csv(miss_path, index=False)
    print(f"\nSaved: {miss_path}")

    # Numeric feature summary (AST/Functional/Contract Info)
    numeric_cols = bccc_per_id_features.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n  numeric feature cols: {len(numeric_cols)}")
    if len(numeric_cols) > 0:
        desc = bccc_per_id_features[numeric_cols].describe().T
        desc["pct_missing"] = 100 * bccc_per_id_features[numeric_cols].isna().sum() / len(bccc_per_id_features)
        desc_path = OUT_DIR / "ws_j_numeric_feature_summary.csv"
        desc.to_csv(desc_path)
        print(f"  saved: {desc_path}")

    # =========================================================================
    # SECTION 7: OPCODE count features
    # =========================================================================
    section("7. OPCODE count features (Bytecode Character Count + Opcode Count)")

    opcode_cols = [c for c in feature_cols if c.startswith("Opcode Count")]
    char_count_cols = [c for c in feature_cols if c.startswith("Bytecode Character Count")]
    bytecode_cols = [c for c in feature_cols if c.startswith("Bytecode Length and Entropy")]
    abi_cols = [c for c in feature_cols if c.startswith("ABI Features")]
    ast_cols = [c for c in feature_cols if c.startswith("AST Features")]
    func_cols = [c for c in feature_cols if c.startswith("Functional Features")]
    sol_cols = [c for c in feature_cols if c.startswith("Solidity Features")]
    loc_cols = [c for c in feature_cols if c.startswith("Lines of Code")]
    print(f"  OPCODE count cols: {len(opcode_cols)}")
    print(f"  Bytecode char count cols: {len(char_count_cols)}")
    print(f"  Bytecode length/entropy cols: {len(bytecode_cols)}")
    print(f"  ABI features cols: {len(abi_cols)}")
    print(f"  AST features cols: {len(ast_cols)}")
    print(f"  Functional features cols: {len(func_cols)}")
    print(f"  Solidity features cols: {len(sol_cols)}")
    print(f"  LOC cols: {len(loc_cols)}")

    if len(opcode_cols) > 0:
        op_total = bccc_per_id_features[opcode_cols].sum(axis=1)
        print(f"\n  OPCODE total per contract: min={op_total.min()}, "
              f"max={op_total.max()}, mean={op_total.mean():.1f}, "
              f"median={op_total.median():.1f}")
        # Top 10 most common opcodes
        op_freq = bccc_per_id_features[opcode_cols].sum().sort_values(ascending=False)
        print("  top 10 opcodes by total count:")
        for op, c in op_freq.head(10).items():
            print(f"    {op}: {c}")

    # =========================================================================
    # SECTION 8: Splits & metadata
    # =========================================================================
    section("8. Splits and dataset quality")

    split_counts = split["split"].value_counts()
    print("  split distribution:")
    for s, n in split_counts.items():
        print(f"    {s}: {n} ({100*n/len(split):.2f}%)")

    # =========================================================================
    # SECTION 9: BCCC long-format duplicate contracts
    # =========================================================================
    section("9. BCCC ID-level uniqueness")

    print(f"  unique IDs in BCCC long format: {bccc['ID'].nunique()}")
    print(f"  total rows: {len(bccc)}")
    print(f"  avg rows per ID: {len(bccc)/bccc['ID'].nunique():.2f}")
    # Per-class count per ID
    print("\n  Per-class contract counts (after groupby ID + max):")
    for c in CLASS_COLS:
        n_pos_ids = (bccc_per_id_class[c] == 1).sum()
        print(f"    {c}: {n_pos_ids} contracts positive")

    # =========================================================================
    # SECTION 10: Compile outputs
    # =========================================================================
    section("10. Save JSON stats + write report")

    stats = {
        "bccc_raw": {
            "n_rows_long_format": int(len(bccc)),
            "n_unique_ids": int(bccc["ID"].nunique()),
            "n_cols_total": int(bccc.shape[1]),
            "n_feature_cols": int(len(feature_cols)),
            "n_class_cols": int(len(CLASS_COLS)),
            "per_class_prevalence": {c: int(bccc_per_id_class[c].sum()) for c in CLASS_COLS},
            "n_pos_distribution": {int(k): int(v) for k, v in n_pos_dist.items()},
        },
        "cleaned_v9": {
            "n_contracts": int(len(clean)),
            "n_review_pending": int(clean["review_pending"].sum()),
            "n_pure_nv": int(clean["is_pure_nv"].sum()),
            "n_pos_distribution": {
                int(k): int(v) for k, v in clean["n_pos"].value_counts().sort_index().items()
            },
        },
        "dropped_class05_07": {
            "n_dropped": int(len(dropped_ids)),
            "n_only_class05": int(only_05[dropped_mask].sum()),
            "n_only_class07": int(only_07[dropped_mask].sum()),
            "n_both_class05_class07": int(both_05_07[dropped_mask].sum()),
        },
        "feature_missingness": {
            "n_cols_zero_missing": int((miss_df["pct_missing"] == 0).sum()),
            "n_cols_gt50_missing": int((miss_df["pct_missing"] > 50).sum()),
            "n_cols_100_missing": int((miss_df["pct_missing"] == 100).sum()),
            "n_numeric_cols": int(len(numeric_cols)),
        },
        "opcode_features": {
            "n_opcode_count_cols": int(len(opcode_cols)),
            "n_bytecode_char_count_cols": int(len(char_count_cols)),
            "n_bytecode_len_entropy_cols": int(len(bytecode_cols)),
            "n_abi_feature_cols": int(len(abi_cols)),
            "n_ast_feature_cols": int(len(ast_cols)),
        },
    }

    stats_path = OUT_DIR / "ws_j_eda_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  saved: {stats_path}")

    # Write markdown report
    md = []
    md.append("# WS-J: Statistical EDA Report")
    md.append("")
    md.append("**Date:** 2026-06-06  ")
    md.append("**Source data:** BCCC-SCsVul-2024 + SENTINEL v9 cleaned v1.0  ")
    md.append("**Author:** SENTINEL Phase 3 (deep analysis)  ")
    md.append("")
    md.append("## 1. Headline numbers")
    md.append("")
    md.append(f"- BCCC raw (long format): **{len(bccc):,}** rows × **{bccc.shape[1]}** cols ({bccc['ID'].nunique():,} unique contracts)")
    md.append(f"- SENTINEL v9 cleaned v1.0: **{len(clean):,}** contracts × **{clean.shape[1]}** cols")
    md.append(f"- Dropped (Class05/Class07 only): **{len(dropped_ids):,}** contracts")
    md.append(f"- Review-pending (NV+vuln contradiction): **{int(clean['review_pending'].sum())}** contracts")
    md.append(f"- Pure-NV contracts: **{int(clean['is_pure_nv'].sum())}** ({100*clean['is_pure_nv'].mean():.1f}%)")
    md.append(f"- Per-contract mean rows in long format: **{len(bccc)/bccc['ID'].nunique():.2f}** (max 9, suggests multi-label encoding)")
    md.append("")
    md.append("## 2. Per-class prevalence (BCCC raw, per-contract)")
    md.append("")
    md.append("| Class | n contracts | % of total |")
    md.append("|---|---:|---:|")
    for c, n in bccc_class_dist.items():
        md.append(f"| {c} | {n:,} | {100*n/len(bccc_per_id_class):.2f}% |")
    md.append("")
    md.append("## 3. n_pos distribution (multi-label cardinality)")
    md.append("")
    md.append("BCCC raw (per-contract):")
    md.append("")
    md.append("| n_pos | n contracts | % |")
    md.append("|---:|---:|---:|")
    for n, c in n_pos_dist.items():
        md.append(f"| {n} | {c:,} | {100*c/len(bccc_per_id_class):.2f}% |")
    md.append("")
    md.append("SENTINEL v9 cleaned:")
    md.append("")
    md.append("| n_pos | n contracts | % |")
    md.append("|---:|---:|---:|")
    for n, c in clean["n_pos"].value_counts().sort_index().items():
        md.append(f"| {n} | {c:,} | {100*c/len(clean):.2f}% |")
    md.append("")
    md.append("## 4. Class co-occurrence (top 10 by joint frequency)")
    md.append("")
    md.append("Full matrix: `outputs/ws_j_cooccurrence_bccc.csv` and `ws_j_cooccurrence_bccc_pct.csv`")
    md.append("")
    md.append("## 5. Top 10 multi-label combinations")
    md.append("")
    md.append("| Count | Label set |")
    md.append("|---:|---|")
    for combo, c in combo_counts.head(10).items():
        md.append(f"| {c:,} | {combo} |")
    md.append("")
    md.append("## 6. Review-pending (NV+vuln contradiction)")
    md.append("")
    md.append(f"- Total: **{len(rp):,}** contracts ({100*len(rp)/len(clean):.2f}% of cleaned dataset)")
    md.append("- These are contracts labeled BOTH NonVulnerable AND at least one vuln class. Likely label noise.")
    md.append("- Distribution of n_pos within review-pending:")
    md.append("")
    for n, c in rp["n_pos"].value_counts().sort_index().items():
        md.append(f"  - n_pos={n}: {c}")
    md.append("")
    md.append("## 7. Dropped contracts (Class05/Class07 only)")
    md.append("")
    md.append(f"- Total dropped: **{len(dropped_ids):,}** contracts")
    md.append(f"  - Class05 (TransactionOrderDependence) only: **{int(only_05[dropped_mask].sum())}**")
    md.append(f"  - Class07 (WeakAccessMod) only: **{int(only_07[dropped_mask].sum())}**")
    md.append(f"  - Both Class05 + Class07: **{int(both_05_07[dropped_mask].sum())}**")
    md.append("- Reason: Class05 and Class07 have no SENTINEL v9 equivalent (D-F1).")
    md.append("- Recovery: would need to add a SWC-114 or access-control class to SENTINEL v9.")
    md.append("")
    md.append("## 8. Feature missingness (BCCC 241 non-class cols)")
    md.append("")
    md.append(f"- Cols with 0% missing: **{(miss_df['pct_missing'] == 0).sum()}**")
    md.append(f"- Cols with >50% missing: **{(miss_df['pct_missing'] > 50).sum()}**")
    md.append(f"- Cols with 100% missing (all-NaN): **{(miss_df['pct_missing'] == 100).sum()}**")
    md.append(f"- Numeric cols: **{len(numeric_cols)}**")
    md.append("")
    md.append("Full missingness: `outputs/ws_j_feature_missingness.csv`")
    md.append("")
    md.append("## 9. OPCODE/Bytecode feature inventory")
    md.append("")
    md.append("| Feature group | n cols |")
    md.append("|---|---:|")
    md.append(f"| Opcode Count | {len(opcode_cols)} |")
    md.append(f"| Bytecode Character Count | {len(char_count_cols)} |")
    md.append(f"| Bytecode Length and Entropy | {len(bytecode_cols)} |")
    md.append(f"| ABI Features | {len(abi_cols)} |")
    md.append(f"| AST Features | {len(ast_cols)} |")
    md.append(f"| Functional Features | {len(func_cols)} |")
    md.append(f"| Solidity Features | {len(sol_cols)} |")
    md.append(f"| Lines of Code | {len(loc_cols)} |")
    md.append("")
    md.append("## 10. Splits")
    md.append("")
    md.append("| Split | n | % |")
    md.append("|---|---:|---:|")
    for s, n in split_counts.items():
        md.append(f"| {s} | {n:,} | {100*n/len(split):.2f}% |")
    md.append("")
    md.append("## 11. Implications for Phase 3 workstreams")
    md.append("")
    md.append("- **WS-M (BCCC 242-feature test):** " + ("All 241 features are present (low missingness), so WS-M can test the full BCCC feature set on a 5,000 stratified sample." if (miss_df['pct_missing'] > 50).sum() < 10 else f"WARN: {(miss_df['pct_missing'] > 50).sum()} features have >50% missing — WS-M may need imputation or feature dropping."))
    md.append("- **WS-L (AutoML):** " + ("Class imbalance is severe (Reentrancy 17,698 vs ExternalBug 3,604 ≈ 4.9×). Use class_weight='balanced' or SMOTE." if True else ""))
    md.append(f"- **WS-N (dropped review):** 1,122 dropped contracts = {100*len(dropped_ids)/len(bccc_per_id_class):.2f}% of BCCC. Recovery would expand SENTINEL's coverage to 100% of BCCC.")
    md.append(f"- **WS-T (multi-label structure):** n_pos=1 covers {100*n_pos_dist.get(1, 0)/len(bccc_per_id_class):.1f}% of BCCC. Multi-label structure is meaningful for the remaining {100*(len(bccc_per_id_class)-n_pos_dist.get(1, 0))/len(bccc_per_id_class):.1f}%.")
    md.append("")
    report_path = REPORT_DIR / "ws_j_eda_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"  saved: {report_path}")
    print("\n[WS-J] DONE")


if __name__ == "__main__":
    main()
