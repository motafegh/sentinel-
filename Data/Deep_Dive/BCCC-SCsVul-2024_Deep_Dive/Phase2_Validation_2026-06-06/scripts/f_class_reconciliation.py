"""BCCC Phase 2 — WS-F: Class Reconciliation.

Applies the decisions:
  - D-F1: drop WeakAccessMod (Class07) and TransactionOrderDependence (Class05).
          Drop all contracts whose ONLY positive classes are Class05 and/or Class07.
  - D-B2: flag the 766 NV+vuln contradictions for manual review (mark review_pending).

Outputs (under ../labels/):
  - class_reconciliation_decision.md  (decision log)
  - contracts_filtered.csv            (one row per surviving contract, 10-class label vector)
  - review_pending_ids.csv            (766 IDs flagged for manual review)
  - dropped_contracts.csv             (5,480 contracts dropped per D-F1)
"""
import pandas as pd
from pathlib import Path

ROOT = Path("/home/motafeq/projects/sentinel")
BCCC_CSV = ROOT / "BCCC-SCsVul-2024" / "BCCC-SCsVul-2024.csv"
INTEG = Path(__file__).resolve().parent.parent / "integrity"
DEDUP_CSV = INTEG / "dedup_map.csv"
LABELS = Path(__file__).resolve().parent.parent / "labels"
OUT = LABELS

ORIGINAL_CLASSES = [
    "Class01:ExternalBug", "Class02:GasException", "Class03:MishandledException",
    "Class04:Timestamp", "Class05:TransactionOrderDependence", "Class06:UnusedReturn",
    "Class07:WeakAccessMod", "Class08:CallToUnknown", "Class09:DenialOfService",
    "Class10:IntegerUO", "Class11:Reentrancy", "Class12:NonVulnerable",
]

# D-F1: drop these 2 classes (no SENTINEL v9 equivalent)
DROPPED_CLASSES = [
    "Class05:TransactionOrderDependence",
    "Class07:WeakAccessMod",
]

# Surviving classes (10, matching SENTINEL v9 schema)
SENTINEL_V9_CLASSES = [c for c in ORIGINAL_CLASSES if c not in DROPPED_CLASSES]
assert len(SENTINEL_V9_CLASSES) == 10
NV_CLASS = "Class12:NonVulnerable"

# SENTINEL class name mapping (BCCC class → SENTINEL index in v9 schema)
# v9 schema order (per ADR-0001, locked): ExternalBug, GasException, MishandledException,
#   Timestamp, UnusedReturn, CallToUnknown, DenialOfService, IntegerUO, Reentrancy, NonVulnerable
SENTINEL_V9_ORDER = [
    "Class01:ExternalBug", "Class02:GasException", "Class03:MishandledException",
    "Class04:Timestamp", "Class06:UnusedReturn", "Class08:CallToUnknown",
    "Class09:DenialOfService", "Class10:IntegerUO", "Class11:Reentrancy", "Class12:NonVulnerable",
]


def main():
    print("=" * 70)
    print("WS-F: Class Reconciliation")
    print("=" * 70)

    # 1. Load CSV and collapse long format
    print("\n[1/5] Loading BCCC CSV (long format)...")
    df = pd.read_csv(BCCC_CSV)
    grouped = df.groupby("ID")[ORIGINAL_CLASSES].max()
    print(f"  Collapsed: {grouped.shape[0]:,} unique contracts")
    print(f"  Original 12 classes: {ORIGINAL_CLASSES}")

    # 2. Apply D-F1: drop contracts whose ONLY positive classes are dropped ones
    print("\n[2/5] Applying D-F1 (drop Class05, Class07)...")
    has_dropped_only = (
        (grouped["Class05:TransactionOrderDependence"] == 1) |
        (grouped["Class07:WeakAccessMod"] == 1)
    ) & (
        grouped[[c for c in ORIGINAL_CLASSES if c not in DROPPED_CLASSES + [NV_CLASS]]].sum(axis=1) == 0
    )
    # Note: the above keeps contracts that have Class05 or Class07 AND any non-dropped vuln class.
    # We drop only contracts that have ONLY dropped classes (and possibly NV).
    # But wait — if a contract has Class05=1 AND Class10=1, we keep it (with Class05 stripped to 0).
    # If a contract has Class05=1 AND nothing else, we drop it entirely.

    # Count: contracts that have any non-dropped class (other than NV) — these are kept, with dropped classes stripped
    has_non_dropped_vuln = grouped[
        [c for c in SENTINEL_V9_ORDER if c != NV_CLASS]
    ].sum(axis=1) > 0
    has_only_dropped = has_dropped_only & ~has_non_dropped_vuln
    # Wait, this is wrong. Let me re-derive.
    # For each contract, compute: "has any non-dropped vuln class?"
    non_dropped_vuln_classes = [c for c in SENTINEL_V9_ORDER if c != NV_CLASS]
    has_any_non_dropped_vuln = grouped[non_dropped_vuln_classes].sum(axis=1) > 0
    # A contract is "dropped" if it has only dropped classes (no non-dropped vuln classes), and is NOT pure-NV.
    # But pure-NV contracts (Class12=1 only) should be KEPT.
    is_pure_nv = (grouped[NV_CLASS] == 1) & (grouped[[c for c in ORIGINAL_CLASSES if c != NV_CLASS]].sum(axis=1) == 0)
    has_dropped_class = (grouped["Class05:TransactionOrderDependence"] == 1) | (grouped["Class07:WeakAccessMod"] == 1)
    drop_these = has_dropped_class & ~has_any_non_dropped_vuln & ~is_pure_nv
    n_dropped_due_to_f1 = drop_these.sum()
    print(f"  Contracts dropped per D-F1 (only have Class05/Class07 + maybe NV, no other vulns): {n_dropped_due_to_f1:,}")
    # Hmm — but these contracts might also have NV. Let's break it down.
    breakdown = pd.DataFrame({
        "has_dropped_class": has_dropped_class,
        "is_pure_nv": is_pure_nv,
        "has_any_non_dropped_vuln": has_any_non_dropped_vuln,
        "drop": drop_these,
    })
    print(f"  Breakdown of contracts with Class05/Class07:")
    print(f"    Total with Class05 or Class07: {has_dropped_class.sum():,}")
    print(f"    Of those, also has non-dropped vuln: {(has_dropped_class & has_any_non_dropped_vuln).sum():,}")
    print(f"    Of those, pure NV: {(has_dropped_class & is_pure_nv).sum():,}")
    print(f"    Of those, dropped: {n_dropped_due_to_f1:,}")

    # Strip Class05 and Class07 from kept contracts
    grouped_kept = grouped[~drop_these].copy()
    grouped_kept["Class05:TransactionOrderDependence"] = 0
    grouped_kept["Class07:WeakAccessMod"] = 0
    # Now rebuild as 10-class SENTINEL v9 vector
    filtered = grouped_kept[SENTINEL_V9_ORDER].copy()
    print(f"  Surviving contracts after D-F1: {len(filtered):,}")

    # 3. Apply D-B2: flag 766 NV+vuln contradictions for manual review
    print("\n[3/5] Applying D-B2 (flag 766 NV+vuln contradictions)...")
    nv = filtered[NV_CLASS] == 1
    any_vuln = filtered[[c for c in SENTINEL_V9_ORDER if c != NV_CLASS]].sum(axis=1) > 0
    contra = nv & any_vuln
    n_contra = contra.sum()
    print(f"  NV+vuln contradictions: {n_contra:,}")
    # For these, set review_pending = 1; keep all labels as-is (don't modify Class12 yet)
    filtered["review_pending"] = 0
    filtered.loc[contra, "review_pending"] = 1
    print(f"  Marked review_pending=1: {filtered['review_pending'].sum():,}")

    # 4. Recompute per-class prevalence
    print("\n[4/5] Final per-class prevalence (10-class, post-D-F1)...")
    for c in SENTINEL_V9_ORDER:
        n = (filtered[c] == 1).sum()
        print(f"  {c:35s}: {n:>6,} ({100*n/len(filtered):.2f}%)")

    n_pos = filtered[SENTINEL_V9_ORDER].sum(axis=1)
    print(f"  Per-contract n_pos: max={n_pos.max()}, mean={n_pos.mean():.3f}")
    print(f"  Pure NV contracts (no vuln, Class12=1): {((filtered[NV_CLASS]==1) & (filtered[[c for c in SENTINEL_V9_ORDER if c != NV_CLASS]].sum(axis=1)==0)).sum():,}")

    # 5. Save outputs
    print("\n[5/5] Saving outputs...")
    filtered.insert(0, "id", filtered.index)
    filtered.to_csv(OUT / "contracts_filtered.csv", index=False)
    print(f"  Wrote {OUT / 'contracts_filtered.csv'} ({len(filtered):,} contracts × 12 cols)")

    # Dropped contracts (per D-F1)
    dropped_df = pd.DataFrame({
        "id": grouped.index[drop_these],
        "n_dropped_classes": grouped.loc[drop_these, ["Class05:TransactionOrderDependence", "Class07:WeakAccessMod"]].sum(axis=1).values,
        "had_class05": grouped.loc[drop_these, "Class05:TransactionOrderDependence"].values,
        "had_class07": grouped.loc[drop_these, "Class07:WeakAccessMod"].values,
        "had_nv": grouped.loc[drop_these, NV_CLASS].values,
    })
    dropped_df.to_csv(OUT / "dropped_contracts.csv", index=False)
    print(f"  Wrote {OUT / 'dropped_contracts.csv'} ({len(dropped_df):,} dropped contracts)")

    # Review pending (per D-B2)
    review_df = pd.DataFrame({
        "id": filtered.index[contra],
        "n_pos": filtered.loc[contra, SENTINEL_V9_ORDER].sum(axis=1).values,
    })
    for c in SENTINEL_V9_ORDER:
        review_df[f"has_{c}"] = filtered.loc[contra, c].values
    review_df.to_csv(OUT / "review_pending_ids.csv", index=False)
    print(f"  Wrote {OUT / 'review_pending_ids.csv'} ({len(review_df):,} review-pending contracts)")

    # 6. Write decision log
    print("\n[decision] Writing class_reconciliation_decision.md...")
    report = f"""# WS-F: Class Reconciliation — Decision Log

**Date:** 2026-06-06
**Status:** Complete (D-F1 applied; D-B2 marked for review)

## Decisions Applied

### D-F1: Drop 2 BCCC classes (WeakAccessMod + TransactionOrderDependence)

**Chosen option:** A — Drop 2 BCCC classes
- WeakAccessMod (Class07): 1,918 contracts (2.80%)
- TransactionOrderDependence (Class05): 3,562 contracts (5.21%)

**Result:**
- **{n_dropped_due_to_f1:,} contracts dropped** (had ONLY Class05/Class07, no other vulns, and not pure-NV).
- Contracts with Class05/Class07 AND any other non-dropped vuln class: {(has_dropped_class & has_any_non_dropped_vuln).sum():,} (kept, with Class05/Class07 stripped).
- Surviving contracts: {len(filtered):,}

### D-B2: NV+vuln contradictions → manual review

**Chosen option:** D — Manual review 766
- 766 contracts have Class12=1 AND ≥1 vuln class.
- These are flagged `review_pending=1` in `contracts_filtered.csv`.
- They will be **excluded from initial training** (WS-G) but can be re-added after review.
- See `review_pending_ids.csv` for the full list.

## Final 10-Class Schema (SENTINEL v9-aligned)

| Order | Class |
|---:|---|
"""
    for i, c in enumerate(SENTINEL_V9_ORDER, 1):
        report += f"| {i} | `{c}` |\n"

    report += f"""

This matches SENTINEL's locked v9 schema (per ADR-0001).

## Final Per-Class Prevalence (n={len(filtered):,})

| Class | n | % |
|---|---:|---:|
"""
    for c in SENTINEL_V9_ORDER:
        n = (filtered[c] == 1).sum()
        report += f"| `{c}` | {n:,} | {100*n/len(filtered):.2f}% |\n"

    report += f"""

## Per-Contract n_pos Distribution (filtered)

| n_pos | contracts | % |
|---:|---:|---:|
"""
    pos_dist = n_pos.value_counts().sort_index()
    for k, v in pos_dist.items():
        if k <= 8:
            report += f"| {k} | {v:,} | {100*v/len(filtered):.2f}% |\n"

    n_pure_nv = ((filtered[NV_CLASS]==1) & (filtered[[c for c in SENTINEL_V9_ORDER if c != NV_CLASS]].sum(axis=1)==0)).sum()
    n_with_vuln = (filtered[[c for c in SENTINEL_V9_ORDER if c != NV_CLASS]].sum(axis=1) > 0).sum()
    n_review = filtered["review_pending"].sum()

    report += f"""

## Summary Stats

- **Surviving contracts:** {len(filtered):,}
- **Pure NV contracts (no vuln):** {n_pure_nv:,} ({100*n_pure_nv/len(filtered):.1f}%)
- **Contracts with ≥1 vuln:** {n_with_vuln:,} ({100*n_with_vuln/len(filtered):.1f}%)
- **Review-pending (NV+vuln contradictions):** {n_review:,} ({100*n_review/len(filtered):.2f}%)

## What This Means for Training

- **Vuln training set:** {n_with_vuln - n_review:,} contracts (excluding review-pending)
- **Clean NV test set:** {n_pure_nv:,} contracts
- **Review-pending set:** {n_review:,} contracts (held out, manual review needed)

## Files

- `contracts_filtered.csv` — {len(filtered):,} contracts × 12 cols (10 class labels + id + review_pending)
- `dropped_contracts.csv` — {n_dropped_due_to_f1:,} contracts dropped per D-F1
- `review_pending_ids.csv` — {n_review:,} contracts flagged for manual review
- `class_reconciliation_decision.md` — this file

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/f_class_reconciliation.py
```
"""
    (OUT / "class_reconciliation_decision.md").write_text(report)
    print(f"  Wrote {OUT / 'class_reconciliation_decision.md'}")
    print("\nWS-F complete.")


if __name__ == "__main__":
    main()
