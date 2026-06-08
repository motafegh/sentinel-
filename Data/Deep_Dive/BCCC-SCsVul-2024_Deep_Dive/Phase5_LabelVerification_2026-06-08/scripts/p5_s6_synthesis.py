"""Stage 5.6: Synthesis
Applies Stage 5.4 verdicts to the base dataset to produce contracts_clean_v1.3.csv.
Stage 5.5 (GraphCodeBERT propagation) is deferred — v1.3 is 'pre-5.5'.

Verified classes (labels kept unchanged):
  - Class03:MishandledException, Class06:UnusedReturn, Class10:IntegerUO
    (verified by Stage 5.1 manual path)

Applied verdict classes (Stage 5.4):
  - Class01:ExternalBug, Class02:GasException, Class04:Timestamp,
    Class08:CallToUnknown, Class09:DenialOfService, Class11:Reentrancy

All-labels-dropped rule: contracts with ALL class labels = 0 after applying
verdicts → set Class12:NonVulnerable = 1.

Cross-class consistency: D-I-11/D-I-12 already applied in Stage 4. Verify intact.

Run from repo root:
    python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/scripts/p5_s6_synthesis.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT   = Path(".")
P4_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs"
P5_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/outputs"

# -------------------------------------------------------------------------
# 1. Load base dataset (post-D-I-11/12)
# -------------------------------------------------------------------------
print("Loading base dataset (ws_p4_s01b_d12_applied.csv)...")
base = pd.read_csv(P4_OUT / "ws_p4_s01b_d12_applied.csv")
print(f"  {len(base):,} contracts × {len(base.columns)} columns")

# Identify class columns
CLASS_COLS = [c for c in base.columns if c.startswith("Class")]
print(f"  Class columns: {CLASS_COLS}")

# Baseline label counts
print("\nBaseline positive label counts (before Phase 5):")
for c in CLASS_COLS:
    print(f"  {c}: {int(base[c].sum()):,}")
total_before = base[CLASS_COLS].sum(axis=1)
print(f"  Contracts with ≥1 label: {(total_before > 0).sum():,}")
print(f"  NonVulnerable only: {((base.get('Class12:NonVulnerable',0)==1) & (total_before==1)).sum():,}")

# -------------------------------------------------------------------------
# 2. Load Stage 5.4 verdicts
# -------------------------------------------------------------------------
print("\nLoading Stage 5.4 final verdicts...")
s4 = pd.read_csv(P5_OUT / "p5_s4_final_verdict.csv")
print(f"  {len(s4):,} rows across {s4['class'].nunique()} classes")

# Map class name → column name (format in base CSV)
# s4 uses "Class11:Reentrancy" etc., base CSV same format
NOISY_CLASSES = [
    "Class01:ExternalBug",
    "Class02:GasException",
    "Class04:Timestamp",
    "Class08:CallToUnknown",
    "Class09:DenialOfService",
    "Class11:Reentrancy",
]

# -------------------------------------------------------------------------
# 3. Apply verdicts to class labels
# -------------------------------------------------------------------------
df = base.copy()

# Add verification metadata columns
for cls in NOISY_CLASSES:
    df[f"p5_verdict_{cls}"]     = "not_positive"
    df[f"p5_confidence_{cls}"]  = np.nan

dropped_total = 0
kept_total    = 0

for cls in NOISY_CLASSES:
    cls_verdicts = s4[s4["class"] == cls][["id", "verdict", "confidence"]].copy()
    cls_verdicts = cls_verdicts.set_index("id")

    n_positive = int(df[cls].sum())
    n_drop = 0
    n_keep = 0

    for idx, row in df.iterrows():
        if df.at[idx, cls] != 1:
            continue
        cid = df.at[idx, "id"]
        if cid in cls_verdicts.index:
            verdict = cls_verdicts.at[cid, "verdict"]
            conf    = cls_verdicts.at[cid, "confidence"]
            df.at[idx, f"p5_verdict_{cls}"]    = verdict
            df.at[idx, f"p5_confidence_{cls}"] = conf
            if verdict == "DROP":
                df.at[idx, cls] = 0
                n_drop += 1
            else:
                n_keep += 1
        else:
            # Contract is positive but has no verdict (shouldn't happen)
            df.at[idx, f"p5_verdict_{cls}"] = "no_verdict"

    dropped_total += n_drop
    kept_total    += n_keep
    print(f"  {cls}: {n_positive:,} positive → KEEP {n_keep:,} / DROP {n_drop:,}")

print(f"\n  Total labels DROPPED: {dropped_total:,}")
print(f"  Total labels KEPT:    {kept_total:,}")

# -------------------------------------------------------------------------
# 4. All-labels-dropped rule: if contract has 0 positive class labels → set NonVulnerable=1
# -------------------------------------------------------------------------
print("\nApplying all-labels-dropped rule...")
# Class12:NonVulnerable column
NV_COL = "Class12:NonVulnerable"
if NV_COL not in df.columns:
    df[NV_COL] = 0

# Check which contracts now have 0 positive labels across ALL active classes
active_label_cols = [c for c in CLASS_COLS if c != NV_COL]
all_zero_mask = (df[active_label_cols].sum(axis=1) == 0)
print(f"  Contracts with all active labels=0: {all_zero_mask.sum():,}")

# Among those, which previously had at least 1 label?
previously_had_label = (base[active_label_cols].sum(axis=1) > 0)
newly_nv = all_zero_mask & previously_had_label
print(f"  Newly reclassified to NonVulnerable: {newly_nv.sum():,}")

df.loc[newly_nv, NV_COL] = 1

# -------------------------------------------------------------------------
# 5. Cross-class consistency check (D-I-11/D-I-12 integrity)
# -------------------------------------------------------------------------
print("\nCross-class consistency check...")
# D-I-11: NV should not co-occur with {CallToUnknown, Reentrancy, GasException,
#          MishandledException, DenialOfService, Timestamp}
d_i_11_classes = ["Class08:CallToUnknown", "Class11:Reentrancy",
                   "Class02:GasException", "Class03:MishandledException",
                   "Class09:DenialOfService", "Class04:Timestamp"]
d_i_11_classes = [c for c in d_i_11_classes if c in df.columns]
d_i_11_violations = (df[NV_COL] == 1)
for c in d_i_11_classes:
    if c in df.columns:
        d_i_11_violations = d_i_11_violations & (df[c] == 1)
print(f"  D-I-11 violations (NV co-occurs with noisy class): {d_i_11_violations.sum()}")
if d_i_11_violations.sum() > 0:
    print("  ⚠️ Fixing D-I-11 violations (drop NV where still co-occurring after Phase 5)")
    df.loc[d_i_11_violations, NV_COL] = 0

# D-I-12: NV should not co-occur with IntegerUO
d_i_12_violations = (df[NV_COL] == 1) & (df.get("Class10:IntegerUO", 0) == 1)
print(f"  D-I-12 violations (NV + IntegerUO): {d_i_12_violations.sum()}")
if d_i_12_violations.sum() > 0:
    df.loc[d_i_12_violations, NV_COL] = 0

# -------------------------------------------------------------------------
# 6. Final statistics
# -------------------------------------------------------------------------
print("\n" + "="*65)
print("FINAL LABEL STATISTICS (contracts_clean_v1.3)")
print("="*65)

print(f"\nTotal contracts: {len(df):,}")
print("\nPositive label counts (post-Phase 5):")
for c in CLASS_COLS:
    before = int(base[c].sum()) if c in base.columns else 0
    after  = int(df[c].sum())
    delta  = after - before
    print(f"  {c:<35}: {after:6,}  (was {before:6,}, Δ{delta:+,})")

total_after = df[active_label_cols].sum(axis=1)
print(f"\nContracts with ≥1 active label: {(total_after > 0).sum():,}")
print(f"NonVulnerable contracts: {int(df[NV_COL].sum()):,}")
print(f"Contracts with ALL labels=0 (should be 0): {(df[CLASS_COLS].sum(axis=1)==0).sum():,}")

# Verification coverage
print("\nVerification coverage summary:")
for cls in NOISY_CLASSES:
    vc = df[f"p5_verdict_{cls}"].value_counts()
    print(f"  {cls}: {dict(vc)}")

# -------------------------------------------------------------------------
# 7. Save outputs
# -------------------------------------------------------------------------
out_path = P5_OUT / "contracts_clean_v1.3.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(df):,} × {len(df.columns)} cols)")

# Also save compact version (just id + class labels + verification stage)
compact_cols = ["id", "bccc_file_path"] + CLASS_COLS + \
               [c for c in df.columns if c.startswith("p5_verdict_") or
                c.startswith("p5_confidence_")]
compact = df[compact_cols]
compact.to_csv(P5_OUT / "contracts_clean_v1.3_compact.csv", index=False)
print(f"Saved: contracts_clean_v1.3_compact.csv  ({len(compact):,} × {len(compact.columns)} cols)")

# -------------------------------------------------------------------------
# 8. Class size comparison report
# -------------------------------------------------------------------------
print("\nGenerating class size comparison report...")
report_rows = []
for c in CLASS_COLS:
    b = int(base[c].sum()) if c in base.columns else 0
    a = int(df[c].sum())
    report_rows.append({
        "class": c,
        "before_phase5": b,
        "after_phase5": a,
        "dropped": b - a,
        "pct_retained": round(a / b * 100, 1) if b > 0 else 0.0,
        "verification_stage": (
            "Stage 5.1 manual (clean)" if c in ["Class03:MishandledException",
                                                 "Class06:UnusedReturn",
                                                 "Class10:IntegerUO"]
            else "Stage 5.4 automated"  if c in NOISY_CLASSES
            else "D-I-11/12 (no change)" if c == NV_COL
            else "unchanged"
        )
    })
report_df = pd.DataFrame(report_rows)
report_df.to_csv(P5_OUT / "p5_s6_class_size_comparison.csv", index=False)
print(report_df.to_string(index=False))

# -------------------------------------------------------------------------
# 9. Verification report
# -------------------------------------------------------------------------
report_md = []
report_md.append("# Phase 5 Verification Report\n")
report_md.append(f"**Generated:** 2026-06-08 (Session 2)\n")
report_md.append(f"**Stage 5.5 status:** DEFERRED (Run 9 training active — VRAM unavailable)\n")
report_md.append(f"**Dataset version:** v1.3 (pre-Stage 5.5)\n\n")
report_md.append("## Summary\n")
report_md.append(f"- Input: {len(base):,} contracts (v1.1+12, post D-I-11/12)\n")
report_md.append(f"- Output: {len(df):,} contracts with verified labels\n")
report_md.append(f"- Labels dropped: {dropped_total:,}\n")
report_md.append(f"- Labels kept: {kept_total:,}\n")
report_md.append(f"- Newly reclassified to NonVulnerable: {newly_nv.sum():,}\n\n")
report_md.append("## Per-Class Gate Results\n\n")

gate_df = pd.read_csv(P5_OUT / "p5_s4_gate_results.csv")
report_md.append(gate_df.to_csv(None, index=False))

report_md.append("\n## What was NOT verified (Stage 5.5 pending)\n")
report_md.append("- GraphCodeBERT embedding + HDBSCAN cluster-based propagation\n")
report_md.append("- This would improve confidence for Timestamp (52.6%) and DoS (64.5%) classes\n")
report_md.append("- PREREQUISITE: `ps aux | grep train.py` shows no active training process\n\n")
report_md.append("## Files\n")
report_md.append("- `contracts_clean_v1.3.csv` — full dataset with all metadata\n")
report_md.append("- `contracts_clean_v1.3_compact.csv` — id + labels + verdicts only\n")
report_md.append("- `p5_s4_final_verdict.csv` — per-contract automated verdicts\n")
report_md.append("- `p5_s4_gate_results.csv` — per-class gate summary\n")
report_md.append("- `review_batches/` — ~40 contracts per class for manual QA\n")

with open(P5_OUT / "p5_s6_verification_report.md", "w") as f:
    f.write("".join(report_md))
print("\nSaved: p5_s6_verification_report.md")
print("\nStage 5.6 complete.")
