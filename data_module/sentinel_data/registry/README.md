# `sentinel_data.registry` — The Versioned Artifact Catalog

## What This Module Does

The registry module is Stage 7 of the SENTINEL data pipeline. It maintains a **SQLite + YAML catalog** of every artifact produced by the pipeline — sources, representations, splits, exports, and dataset versions.

Think of it as the "library card catalog" for your data. Every time you run the pipeline, the registry records what was produced, what version it is, and what it depends on. Six months from now, when someone asks "what dataset did Run 11 train on?", the answer is in the registry.

## Why This Matters

Without a registry:
- You can't answer "what exact version of the dataset did this model use?"
- You can't verify that an export file hasn't been tampered with
- You can't compare two dataset versions to see what changed
- You can't trace an artifact's lineage back to its source commits

The BCCC dataset had none of this — the v1.4 labels, v8 graphs, v9 graphs, and v10 graphs were scattered across directories with no version tracking. The registry prevents that class of failure.

## Architecture Overview

```
All pipeline artifacts
        │
        ▼
┌─────────────────────────────────────────┐
│         SQLite Catalog                   │
│  (fast lookup, indexed)                  │
│                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ sources  │ │ artifacts│ │ splits   ││
│  └──────────┘ └──────────┘ └──────────┘│
│  ┌──────────────────┐ ┌────────────────┐│
│  │ dataset_versions │ │schema_migrations││
│  └──────────────────┘ └────────────────┘│
│  ┌──────────────────────────────────────┐│
│  │ dataset_version_retirements          ││
│  └──────────────────────────────────────┘│
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│         YAML Mirror                     │
│  (human-readable, version-controlled)   │
└─────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `catalog.py` | SQLite catalog with YAML mirror + migrations + retirement chain |
| `lineage_tracker.py` | Records the DAG of transformations for every artifact |
| `artifact_hasher.py` | Computes SHA-256 of every exported file; verifies on load |
| `dataset_diff.py` | Compares two dataset versions with per-class metric projection |
| `changelog.md` | Updated with every dataset version registration |

## The 6 Database Tables

| Table | Purpose |
|-------|---------|
| `sources` | Per-source pin + last-fetched timestamp |
| `artifacts` | Per-exported-artifact hash + lineage |
| `splits` | Per-split-version seed + strategy |
| `dataset_versions` | Named composite: source set + preprocessing config + split version |
| `schema_migrations` | Tracks every schema change (append-only) |
| `dataset_version_retirements` | Old versions marked as "superseded" (not deleted) |

## How to Use

```bash
# Register a completed dataset version
sentinel-data register --name sentinel-v2-dryrun-2026-08

# Load an artifact
sentinel_data.registry.load_artifact("sentinel-v2-dryrun-2026-08")

# Verify an artifact's hash
sentinel_data.registry.verify_artifact_hash("path/to/export.pt")

# Compare two dataset versions
sentinel_data.registry.dataset_diff("v1", "v2")
```

## The Retirement Chain

Old dataset versions are never deleted — they're marked as "superseded" with a pointer to their replacement:

```
v1.4 BCCC labels → superseded_by: v8 BCCC graphs
v8 BCCC graphs → superseded_by: v9 graphs
v9 graphs → superseded_by: v10 deduped
v10 deduped → superseded_by: v2 gold 2026-08
```

This preserves the audit trail — you can always reconstruct what the v1.4 labels looked like, even though they've been superseded.

## Hash Verification (The Load-Time Gate)

The ML module's `SentinelDataset.__init__` calls `verify_artifact_hash()` before loading:

```python
def verify_artifact_hash(export_path):
    """Verify that the export file hasn't been tampered with.
    
    Uses the same SHA-256 algorithm as ml/src/inference/cache.py
    (shared via sentinel_data.registry.compute_hash).
    """
    registered_hash = catalog.get_hash(export_path)
    actual_hash = compute_hash(export_path)
    return registered_hash == actual_hash
```

If the hash doesn't match, the load fails. This prevents "I edited the export file by hand and the model trained on the wrong data."

## Pipeline Position

```
Stage 6: Splitting (train/val/test splits)
    ↓
Stage 7: Registry ← YOU ARE HERE (catalog + lineage + hash verification)
    ↓
Stage 8: Export (sharded output for ML module)
```

## Design Decisions

1. **SQLite + YAML mirror** — fast lookup + human-readable version control
2. **Append-only retirement chain** — audit trail is permanent
3. **Hash verification at load time** — prevents tampering
4. **Schema migrations table** — tracks every catalog schema change
5. **Per-class metric projection in diffs** — predicts how Run 11's F1 will compare to Run 9
