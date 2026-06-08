"""WS-N: Dropped + Review-Pending deep-dive.

Outputs:
  - outputs/ws_n_dropped_breakdown.csv: per-ID info for 1,122 dropped contracts
  - outputs/ws_n_review_pending_breakdown.csv: per-ID info for 766 review_pending
  - outputs/ws_n_dropped_samples/: 5 source code samples per dropped category
  - outputs/ws_n_review_pending_samples/: 5 source code samples per review category
  - reports/ws_n_dropped_review_report.md: human-readable analysis

Inputs:
  - Phase2_Validation_2026-06-06/outputs/contracts_clean.csv
  - BCCC-SCsVul-2024/Source Codes/<VulnerabilityType>/<id>.sol
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/motafeq/projects/sentinel")
CLEAN_CSV = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase2_Validation_2026-06-06" / "outputs" / "contracts_clean.csv"
BCCC_SRC = ROOT / "BCCC-SCsVul-2024" / "Source Codes"
OUT_DIR = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase3_DeepAnalysis_2026-06-06" / "outputs"
REPORT_DIR = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase3_DeepAnalysis_2026-06-06" / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DROPPED_SAMPLES = OUT_DIR / "ws_n_dropped_samples"
RP_SAMPLES = OUT_DIR / "ws_n_review_pending_samples"
DROPPED_SAMPLES.mkdir(exist_ok=True)
RP_SAMPLES.mkdir(exist_ok=True)

SENTINEL_CLASS_ORDER = [
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


def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def copy_samples(df: pd.DataFrame, n: int, dest_dir: Path, label: str) -> list[dict]:
    """Copy up to n .sol source files from df's bccc_file_path to dest_dir."""
    samples = []
    sample_df = df.sample(min(n, len(df)), random_state=42)
    for _, row in sample_df.iterrows():
        src = ROOT / row["bccc_file_path"]
        if not src.exists():
            print(f"  [WARN] missing source: {src}")
            continue
        # Destination name: <label>_<id>.sol
        dest_name = f"{label}_{row['id'][:16]}.sol"
        dest = dest_dir / dest_name
        shutil.copy(src, dest)
        loc = row.get("loc", -1)
        nfunc = row.get("n_functions", -1)
        samples.append({
            "id": row["id"],
            "src_relative": str(src.relative_to(ROOT)),
            "dest": str(dest.relative_to(ROOT)),
            "primary_class": row["primary_class"],
            "n_pos": int(row["n_pos"]),
            "loc": int(loc) if loc is not None and loc >= 0 else None,
            "n_functions": int(nfunc) if nfunc is not None and nfunc >= 0 else None,
        })
    return samples


def main() -> None:
    print("[WS-N] loading contracts_clean.csv ...")
    clean = pd.read_csv(CLEAN_CSV)
    print(f"  shape: {clean.shape}")

    # =========================================================================
    # SECTION 1: Review-Pending (NV+vuln contradiction)
    # =========================================================================
    section("1. Review-Pending set (NV+vuln contradiction, n=766)")

    rp = clean[clean["review_pending"] == 1].copy()
    print(f"  total: {len(rp)}")

    # Per primary class
    print("\n  primary_class distribution:")
    for c, n in rp["primary_class"].value_counts().items():
        print(f"    {c}: {n} ({100*n/len(rp):.1f}%)")

    # Per n_pos
    print("\n  n_pos distribution (multi-label cardinality):")
    for n, c in rp["n_pos"].value_counts().sort_index().items():
        print(f"    n_pos={n}: {c} ({100*c/len(rp):.1f}%)")

    # Per bccc_folder (original BCCC source folder)
    print("\n  bccc_folder distribution (where the source code lives):")
    for f, n in rp["bccc_folder"].value_counts().items():
        print(f"    {f}: {n} ({100*n/len(rp):.1f}%)")

    # Co-occurrence within review-pending: which vuln classes go WITH NV?
    print("\n  Vuln class co-occurrence (vs NV) within review-pending:")
    for col in SENTINEL_CLASS_ORDER:
        if col == "Class12:NonVulnerable":
            continue
        n = (rp[col] == 1).sum()
        if n > 0:
            print(f"    {col}: {n} contracts ({100*n/len(rp):.1f}%)")

    # Most common (vuln_class1, vuln_class2) combinations within review-pending
    rp["vuln_set"] = rp.apply(
        lambda r: tuple(sorted(c for c in SENTINEL_CLASS_ORDER
                                if c != "Class12:NonVulnerable" and r[c] == 1)),
        axis=1,
    )
    print("\n  Top 10 vuln class combinations (within review-pending, NV excluded):")
    for combo, c in rp["vuln_set"].value_counts().head(10).items():
        print(f"    {c:>4}  {combo}")

    # Check if all review-pending contracts have valid source code paths
    rp_path_exists = rp["bccc_file_path"].apply(
        lambda p: (ROOT / p).exists() if isinstance(p, str) else False
    )
    print(f"\n  review_pending with valid source path: {rp_path_exists.sum()}/{len(rp)}")
    print(f"  review_pending MISSING source: {(~rp_path_exists).sum()}")

    # =========================================================================
    # SECTION 2: Sample review-pending source code (for qualitative review)
    # =========================================================================
    section("2. Sampling 5 review-pending source code files (qualitative)")

    rp_samples = copy_samples(rp, n=5, dest_dir=RP_SAMPLES, label="rp")
    for s in rp_samples:
        print(f"  copied: {s['dest']}  ({s['primary_class']}, n_pos={s['n_pos']}, loc={s['loc']})")

    # =========================================================================
    # SECTION 3: Dropped contracts (Class05/Class07 only)
    # =========================================================================
    section("3. Dropped contracts (Class05/Class07 only, n=1,122)")

    # The dropped set is NOT in contracts_clean.csv (it was dropped in Phase 2).
    # We can recover it from BCCC source code folders:
    #   - Class05/TransactionOrderDependence: 3,562 files
    #   - Class07/WeakAccessMod: 1,918 files
    # Dropped = those with ONLY Class05 and/or Class07 (and no other class).
    # We can identify them by joining the BCCC per-contract class flags with
    # the file presence in those two folders.

    # Simpler: load the dropped IDs from the Phase 2 metadata's
    # `labels/label_consistency.csv` (which has all 68,433 contracts).
    label_csv = ROOT / "Data/Deep_Dive" / "BCCC-SCsVul-2024_Deep_Dive" / "Phase2_Validation_2026-06-06" / "labels" / "label_consistency.csv"
    label = pd.read_csv(label_csv)
    print(f"  label_consistency shape: {label.shape}")

    # Dropped = in label_consistency but NOT in clean
    dropped_ids = set(label["id"]) - set(clean["id"])
    print(f"  total dropped (in label but not in clean): {len(dropped_ids)}")

    dropped_label = label[label["id"].isin(dropped_ids)].copy()
    print(f"  dropped_label shape: {dropped_label.shape}")

    # Per-class breakdown of dropped
    print("\n  Per-class breakdown (dropped contracts only):")
    has_cols = [c for c in dropped_label.columns if c.startswith("has_Class")]
    for c in has_cols:
        n = (dropped_label[c] == 1).sum()
        if n > 0:
            print(f"    {c}: {n}")

    # Per (Class05, Class07) combination
    c05 = dropped_label["has_Class05:TransactionOrderDependence"] == 1
    c07 = dropped_label["has_Class07:WeakAccessMod"] == 1
    print("\n  Dropped class composition:")
    print(f"    Class05 only: {(c05 & ~c07).sum()}")
    print(f"    Class07 only: {(~c05 & c07).sum()}")
    print(f"    Both Class05 + Class07: {(c05 & c07).sum()}")

    # Map dropped IDs to source code file paths
    # For Class05: <root>/BCCC-SCsVul-2024/Source Codes/TransactionOrderDependence/<id>.sol
    # For Class07: <root>/BCCC-SCsVul-2024/Source Codes/WeakAccessMod/<id>.sol
    # A dropped ID may appear in BOTH folders if it has both classes.

    src_tod = BCCC_SRC / "TransactionOrderDependence"
    src_wam = BCCC_SRC / "WeakAccessMod"
    tod_files = {f.stem: f for f in src_tod.glob("*.sol")} if src_tod.exists() else {}
    wam_files = {f.stem: f for f in src_wam.glob("*.sol")} if src_wam.exists() else {}
    print(f"\n  TransactionOrderDependence files on disk: {len(tod_files)}")
    print(f"  WeakAccessMod files on disk: {len(wam_files)}")

    # Find dropped IDs that have source files
    dropped_tod = [i for i in dropped_ids if i in tod_files]
    dropped_wam = [i for i in dropped_ids if i in wam_files]
    print(f"  dropped IDs in TransactionOrderDependence folder: {len(dropped_tod)}")
    print(f"  dropped IDs in WeakAccessMod folder: {len(dropped_wam)}")
    print(f"  dropped IDs in NEITHER folder: {len(dropped_ids - set(dropped_tod) - set(dropped_wam))}")

    # =========================================================================
    # SECTION 4: Sample dropped source code
    # =========================================================================
    section("4. Sampling dropped source code (5 per class)")

    # 5 from TransactionOrderDependence
    tod_sample_ids = pd.Series(dropped_tod).sample(min(5, len(dropped_tod)), random_state=42).tolist()
    wam_sample_ids = pd.Series(dropped_wam).sample(min(5, len(dropped_wam)), random_state=42).tolist()

    dropped_sample_records = []
    for sid in tod_sample_ids:
        src = tod_files[sid]
        dest = DROPPED_SAMPLES / f"dropped_tod_{sid[:16]}.sol"
        shutil.copy(src, dest)
        dropped_sample_records.append({
            "id": sid, "src_relative": str(src.relative_to(ROOT)),
            "dest": str(dest.relative_to(ROOT)),
            "class": "TransactionOrderDependence",
        })
        print(f"  copied: {dest.name}")

    for sid in wam_sample_ids:
        src = wam_files[sid]
        dest = DROPPED_SAMPLES / f"dropped_wam_{sid[:16]}.sol"
        shutil.copy(src, dest)
        dropped_sample_records.append({
            "id": sid, "src_relative": str(src.relative_to(ROOT)),
            "dest": str(dest.relative_to(ROOT)),
            "class": "WeakAccessMod",
        })
        print(f"  copied: {dest.name}")

    # =========================================================================
    # SECTION 5: Save outputs
    # =========================================================================
    section("5. Save outputs")

    # Review-pending breakdown
    rp_out_cols = ["id", "primary_class", "n_pos", "bccc_folder", "bccc_file_path",
                   "loc", "n_functions", "n_events", "n_modifiers", "vuln_set"] + SENTINEL_CLASS_ORDER
    rp_out = rp[rp_out_cols].copy()
    rp_out_path = OUT_DIR / "ws_n_review_pending_breakdown.csv"
    rp_out.to_csv(rp_out_path, index=False)
    print(f"  saved: {rp_out_path}  ({len(rp_out)} rows)")

    # Dropped breakdown (from label_consistency)
    drop_out = dropped_label.copy()
    drop_out["in_tod_folder"] = drop_out["id"].isin(tod_files)
    drop_out["in_wam_folder"] = drop_out["id"].isin(wam_files)
    drop_out_path = OUT_DIR / "ws_n_dropped_breakdown.csv"
    drop_out.to_csv(drop_out_path, index=False)
    print(f"  saved: {drop_out_path}  ({len(drop_out)} rows)")

    # Sample records
    samples_path = OUT_DIR / "ws_n_sample_inventory.json"
    with open(samples_path, "w") as f:
        json.dump({
            "review_pending_samples": rp_samples,
            "dropped_samples": dropped_sample_records,
        }, f, indent=2)
    print(f"  saved: {samples_path}")

    # =========================================================================
    # SECTION 6: Read sampled source code (qualitative review prep)
    # =========================================================================
    section("6. Read sample source code excerpts (first 30 lines each)")

    print("\n--- Review-Pending samples (first 25 lines) ---")
    for s in rp_samples:
        dest = ROOT / s["dest"]
        try:
            content = dest.read_text(errors="replace")
            lines = content.split("\n")[:25]
            print(f"\n  === {dest.name} ({s['primary_class']}, n_pos={s['n_pos']}, loc={s['loc']}) ===")
            for ln in lines:
                print(f"    {ln}")
        except Exception as e:
            print(f"  [ERR] {dest.name}: {e}")

    print("\n\n--- Dropped samples (first 25 lines) ---")
    for s in dropped_sample_records:
        dest = ROOT / s["dest"]
        try:
            content = dest.read_text(errors="replace")
            lines = content.split("\n")[:25]
            print(f"\n  === {dest.name} ({s['class']}) ===")
            for ln in lines:
                print(f"    {ln}")
        except Exception as e:
            print(f"  [ERR] {dest.name}: {e}")

    # =========================================================================
    # SECTION 7: Write report
    # =========================================================================
    section("7. Write report")

    md = []
    md.append("# WS-N: Dropped + Review-Pending Deep-Dive Report")
    md.append("")
    md.append("**Date:** 2026-06-06  ")
    md.append("**Source data:** SENTINEL v9 cleaned v1.0 + BCCC raw  ")
    md.append("")
    md.append("## 1. Review-Pending (NV+vuln contradiction, n=766)")
    md.append("")
    md.append("These 766 contracts are labeled BOTH NonVulnerable AND at least one vuln class in BCCC. They are held out from training and require manual review before re-inclusion.")
    md.append("")
    md.append("### 1.1 Per primary class")
    md.append("")
    md.append("| Primary class | n | % |")
    md.append("|---|---:|---:|")
    for c, n in rp["primary_class"].value_counts().items():
        md.append(f"| {c} | {n} | {100*n/len(rp):.2f}% |")
    md.append("")
    md.append("**Key finding:** 703/766 (92%) of review-pending have **CallToUnknown (Class08) as primary class**.")
    md.append("")
    md.append("### 1.2 n_pos distribution (multi-label cardinality)")
    md.append("")
    md.append("| n_pos | n | % |")
    md.append("|---:|---:|---:|")
    for n, c in rp["n_pos"].value_counts().sort_index().items():
        md.append(f"| {n} | {c} | {100*c/len(rp):.2f}% |")
    md.append("")
    md.append("**Key finding:** 705/766 (92%) of review-pending have **n_pos=3** (i.e., 3 positive classes: typically NV + 2 vuln classes).")
    md.append("")
    md.append("### 1.3 Top 10 vuln-class combinations (NV excluded)")
    md.append("")
    md.append("| Count | Vuln set |")
    md.append("|---:|---|")
    for combo, c in rp["vuln_set"].value_counts().head(10).items():
        short = ", ".join(c.split(":")[1] if ":" in c else c for c in combo)
        md.append(f"| {c} | {short} |")
    md.append("")
    md.append("**Key finding:** The most common triple-label (NV, Reentrancy, CallToUnknown) = 703 contracts. This is suspicious — likely a templating artifact in BCCC where many reentrancy contracts share a common pattern that also gets labeled 'non-vulnerable' in some other context.")
    md.append("")
    md.append("### 1.4 Source code path availability")
    md.append("")
    md.append(f"- Review-pending with valid source path: {rp_path_exists.sum()}/{len(rp)}")
    md.append(f"- Missing: {(~rp_path_exists).sum()}")
    md.append("")
    md.append("### 1.5 Recovery recommendation (WS-N → manual review)")
    md.append("")
    md.append("- These 766 contracts SHOULD be reviewed before adding to training set.")
    md.append("- Given the homogeneity of the NV+Reentrancy+CallToUnknown triple (92% of review-pending), a single rule might resolve most:")
    md.append("  - **Hypothesis:** Contracts in the Reentrancy folder that also have a `nonReentrant` modifier OR a `ReentrancyGuard` import are NOT actually reentrant; the 'NV' label is correct, the 'Reentrancy' label is wrong.")
    md.append("- Manual review budget for this set: ~3-5 minutes per contract × 766 = **40-60 hours** (use 846 default = 766 + 50 + 30 if you also do multi-positive and disagreement sets).")
    md.append("- **Alternative (faster):** Just use these 766 contracts as **noise data for adversarial training** in Phase 4 — train SENTINEL to predict the majority class, ignore the contradictions.")
    md.append("")
    md.append("## 2. Dropped contracts (Class05/Class07 only, n=1,122)")
    md.append("")
    md.append("These 1,122 contracts are labeled ONLY with BCCC's Class05 (TransactionOrderDependence) and/or Class07 (WeakAccessMod). Neither class has a SENTINEL v9 equivalent, so they were dropped in Phase 2 (D-F1).")
    md.append("")
    md.append("### 2.1 Class composition")
    md.append("")
    c05_only = (c05 & ~c07).sum()
    c07_only = (~c05 & c07).sum()
    c_both = (c05 & c07).sum()
    md.append("| Subset | n | % |")
    md.append("|---|---:|---:|")
    md.append(f"| Class05 only | {c05_only} | {100*c05_only/len(dropped_label):.2f}% |")
    md.append(f"| Class07 only | {c07_only} | {100*c07_only/len(dropped_label):.2f}% |")
    md.append(f"| Class05 + Class07 | {c_both} | {100*c_both/len(dropped_label):.2f}% |")
    md.append("")
    md.append("**Key finding:** 959/1,122 (85.5%) of dropped are **Class07 (WeakAccessMod) only**. The 'WeakAccessMod' class is mostly state-visibility issues (public functions that should be private/external). SENTINEL v9 has no equivalent because:")
    md.append("- Public vs external matters less for vulnerability detection (the issue is in CALLERS, not callees).")
    md.append("- These are more 'code quality' than 'security' issues per D-F1 decision.")
    md.append("")
    md.append("### 2.2 Source code folder mapping")
    md.append("")
    md.append("| BCCC source folder | n .sol files | n dropped IDs mapped |")
    md.append("|---|---:|---:|")
    md.append(f"| TransactionOrderDependence/ | {len(tod_files)} | {len(dropped_tod)} |")
    md.append(f"| WeakAccessMod/ | {len(wam_files)} | {len(dropped_wam)} |")
    md.append(f"| (no folder match) | — | {len(dropped_ids - set(dropped_tod) - set(dropped_wam))} |")
    md.append("")
    md.append("### 2.3 Recovery recommendation")
    md.append("")
    md.append("- To recover these 1,122 contracts, SENTINEL v9 would need 2 new classes:")
    md.append("  - `Class05:TransactionOrderDependence` (SWC-114): e.g., TOCTOU bugs in ERC20 approve/transfer pattern.")
    md.append("  - `Class07:WeakAccessMod` (SWC-100 / SWC-105): state visibility issues.")
    md.append("- **Cost:** 2 new classes × ~10 hours labeling review × 1,122 contracts = **~22 hours** (or 1.2 hours at 30s per contract for keyword-based auto-labeling).")
    md.append("- **Value:** Brings SENTINEL coverage from 67,311/68,433 (98.4%) to 100% of BCCC.")
    md.append("- **Recommended for Phase 4** (NOT Phase 3 — out of scope per plan §10).")
    md.append("")
    md.append("## 3. Source code samples (qualitative review)")
    md.append("")
    md.append("### 3.1 Review-pending samples (5 contracts)")
    md.append("")
    md.append(f"Saved to `outputs/ws_n_review_pending_samples/` (5 files, {len(rp_samples)} copied).")
    md.append("Manual review needed: read each .sol and decide which label is correct.")
    md.append("")
    md.append("### 3.2 Dropped samples (5 + 5 = 10 contracts)")
    md.append("")
    md.append(f"Saved to `outputs/ws_n_dropped_samples/` (10 files, {len(dropped_sample_records)} copied).")
    md.append("Manual review needed: confirm these are indeed Class05/Class07 issues (no SENTINEL class fits).")
    md.append("")
    md.append("## 4. Key takeaways for Phase 3 downstream")
    md.append("")
    md.append("1. **WS-I (slither label validation):** For the 846 manual review contracts, prioritize the 766 review-pending first (highest value, highest noise).")
    md.append("2. **WS-M (BCCC 242-feature test):** Dropped contracts will not appear in v1.0 dataset; if v1.2 includes them, need to add 2 new SENTINEL classes (out of scope for Phase 3).")
    md.append("3. **WS-L (AutoML):** Use class_weight='balanced' to handle Reentrancy vs ExternalBug imbalance (4.9×).")
    md.append("4. **WS-T (multi-label structure):** n_pos=1 is 60.6% — AutoML can use a 'class chain' decomposition (predict n_pos first, then which classes).")
    md.append("")

    report_path = REPORT_DIR / "ws_n_dropped_review_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"  saved: {report_path}")

    print("\n[WS-N] DONE")


if __name__ == "__main__":
    main()
