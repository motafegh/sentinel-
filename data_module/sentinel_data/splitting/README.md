# `sentinel_data.splitting` — Preparing Data for Training

## What This Module Does

The splitting module is Stage 6 of the SENTINEL data pipeline. It takes verified, labeled contracts and produces **deterministic, leak-free, stratified train/val/test splits** ready for model training.

The module implements 4 splitting strategies, a deduplication enforcer, and a leakage auditor. The output is versioned parquet files with a manifest that records every decision made during the split.

## Why This Matters

The BCCC dataset had a **34.9% cross-split leakage rate** in v6 — contracts that appeared in both train and test, inflating Run 9's F1 by an estimated ~0.05. The splitting module prevents this by:

1. **Enforcing deduplication BEFORE splitting** — near-duplicate groups are kept in one split
2. **Auditing for leakage AFTER splitting** — an independent safety net catches what the enforcer misses
3. **Stratifying by source** — prevents one dominant source from skewing the splits
4. **Capping NonVulnerable at 3:1** — prevents the "predict negative and win 99%+" failure mode

## Architecture Overview

```
Verified labels (.labels.json)
        │
        ▼
┌─────────────────────────────────────────┐
│      4 Splitting Strategies             │
│  (random, stratified, project, temporal)│
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Dedup Enforcer                  │
│  Reassigns near-dup groups that         │
│  straddle split boundaries              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Leakage Auditor                 │
│  Independent post-split check           │
│  (shingled text similarity)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Split Manifest                  │
│  Versioned JSON with full audit trail   │
└─────────────────────────────────────────┘
```

## The 4 Splitting Strategies

| Strategy | When to Use | Key Property |
|----------|-------------|--------------|
| **Random** | Default for simple cases | Contracts randomly assigned to splits |
| **Stratified** | Tool-derived datasets (Slither-Audited, SmartBugs Wild) | Preserves per-class distribution across splits |
| **Project-Level** | Audit datasets (Bastet, ScaBench, Web3Bugs, DeFiHackLabs) | Entire project stays in one split |
| **Temporal** | Time-sensitive analyses | Contracts ordered by deployment date |

The strategy is **per-source** in `config.yaml`:
```yaml
sources:
  bastet:
    split:
      strategy: project_level
  slither_audited:
    split:
      strategy: stratified
```

## The Dedup Enforcer (The BCCC Fix)

The dedup enforcer is the structural fix for the BCCC 38.8% duplication rate:

```python
def enforce_dedup(splits, dedup_groups):
    """Reassign any near-dup group that straddles a split boundary.
    
    Rule: the group goes to the split where the majority of its members are.
    Ties go to train.
    """
    for group in dedup_groups:
        split_assignments = [contract.split for contract in group]
        if len(set(split_assignments)) > 1:
            # Group straddles boundary — reassign to majority split
            majority_split = Counter(split_assignments).most_common(1)[0][0]
            for contract in group:
                contract.split = majority_split
```

The enforcer records all reassignments in the split manifest for auditing.

## The Leakage Auditor (The Safety Net)

After the two-pass split, the leakage auditor does an **independent** similarity check using a different algorithm (shingled text similarity instead of AST similarity):

```python
def audit_leakage(splits, threshold=0.5):
    """Check for near-duplicate pairs across split boundaries.
    
    Uses shingled text similarity (different from the dedup enforcer's
    AST similarity) as an independent safety net.
    """
    for pair in cross_split_pairs:
        similarity = shingled_similarity(pair.contract_a, pair.contract_b)
        if similarity > threshold:
            report.add_leak(pair, similarity)
    return report
```

The two methods (AST similarity in the enforcer, text similarity in the auditor) can disagree — and that's the point. The auditor catches what the enforcer misses.

## The NonVulnerable 3:1 Cap

DISL provides 514,506 unlabeled contracts as NonVulnerable examples. With ~1,200 positives from the 5 critical-path sources, the default ratio would be 428:1 — the same BCCC failure pattern at larger scale.

The cap enforces: **NonVulnerable count ≤ 3 × total positive count across all 10 classes**.

```python
def apply_nonvulnerable_cap(splits, total_positive_count, cap=3.0):
    """Subsample NonVulnerable to at most cap * total_positive_count.
    
    Stratified by source to preserve the per-source distribution.
    """
    max_nonvuln = int(cap * total_positive_count)
    for split_name in ['train', 'val', 'test']:
        # Stratified subsample by source
        ...
```

The default 3:1 ratio is the empirical sweet spot — high enough to provide sufficient negative examples, low enough to prevent the model from defaulting to "predict negative."

## Split Manifest

Every split produces a versioned JSON manifest:

```json
{
  "version": "v1",
  "seed": 42,
  "strategy": {"bastet": "project_level", "scabench": "project_level"},
  "contract_counts": {"train": 3500, "val": 750, "test": 750},
  "class_distribution": {
    "train": {"Reentrancy": 450, "CallToUnknown": 320, ...},
    "val": {"Reentrancy": 95, "CallToUnknown": 68, ...},
    "test": {"Reentrancy": 95, "CallToUnknown": 68, ...}
  },
  "dedup_groups_resolved": 142,
  "reassignments": [...],
  "leakage_auditor": {"leaks_found": 0, "max_similarity": 0.23},
  "nonvulnerable_cap": {"original": 5000, "capped": 3600, "ratio": 3.0},
  "generated_at": "2026-07-28T10:00:00Z"
}
```

## How to Use

```bash
# Split all verified data
sentinel-data split

# Split with a specific config
sentinel-data split --config split-config.yaml

# Dry-run
sentinel-data split --dry-run
```

## Pipeline Position

```
Stage 5: Verification (are these labels correct?)
    ↓
Stage 6: Splitting ← YOU ARE HERE (train/val/test splits)
    ↓
Stage 7: Registry (catalog + lineage)
```

## Design Decisions

1. **Two-pass split** — stratified split first, then dedup enforcer (prevents BCCC-style leakage)
2. **Project-level for audit datasets** — a project is entirely in one split (no cross-contamination)
3. **Leakage auditor as safety net** — independent algorithm catches what the enforcer misses
4. **NonVulnerable 3:1 cap** — prevents the "predict negative and win 99%+" failure mode
5. **Stratified subsampling** — preserves per-source distribution when capping NonVulnerable
