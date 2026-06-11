# `sentinel_data.export` — Producing the Consumable Artifact

## What This Module Does

The export module is Stage 9 of the SENTINEL data pipeline. It takes all the artifacts produced by Stages 1–7 and produces the **sharded export** that the ML module's `SentinelDataset` reads during training.

The export produces 4 file types per shard:
1. **Graphs** (PyG `Batch` objects) — the GNN input
2. **Tokens** (torch tensors) — the GraphCodeBERT input
3. **Labels** (parquet) — the multi-label tensor with confidence tiers
4. **Metadata** (parquet) — provenance, source, solc version, etc.

## Why This Matters

The export is the **seam** between the data module and the ML module. It's the only point where the two packages touch. The format must be:
- **Byte-identical** — the same input produces the same output, every time
- **Versioned** — schema v1 is the contract; future bumps are new files
- **Hash-verified** — the ML module verifies the hash on load

The export is also where the **seam swap** happens — the old `dual_path_dataset.py` is replaced by a thin `sentinel_dataset.py` that reads from the v2 export.

## Architecture Overview

```
Representations (.pt) + Labels (.json) + Splits + Metadata
        │
        ▼
┌─────────────────────────────────────────┐
│         4 Shard Writers                 │
│  ┌──────────┐ ┌──────────┐             │
│  │ graph_   │ │ token_   │             │
│  │ writer   │ │ writer   │             │
│  └──────────┘ └──────────┘             │
│  ┌──────────┐ ┌──────────┐             │
│  │ label_   │ │ metadata_│             │
│  │ writer   │ │ writer   │             │
│  └──────────┘ └──────────┘             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Chunker                        │
│  Orchestrates 4 writers per shard      │
│  Manages shard index + manifest        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         SentinelDatasetExport           │
│  Consumer-facing API for ML module     │
└─────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `format_schema/v1.yaml` | The export format contract |
| `graph_writer.py` | Writes sharded PyG `Batch` objects |
| `token_writer.py` | Writes sharded token tensors |
| `label_writer.py` | Writes `labels.parquet` with per-class columns |
| `metadata_writer.py` | Writes `metadata.parquet` with provenance |
| `chunker.py` | Orchestrates the 4 writers per shard |
| `export.py` | `SentinelDatasetExport` consumer-facing API |

## The Format Schema

The export format is defined in `format_schema/v1.yaml`:

```yaml
shard_size: 5000
file_types:
  graphs_shard:
    pattern: "graphs-{shard:05d}.pt"
    content: "PyG Batch object"
  tokens_shard:
    pattern: "tokens-{shard:05d}.pt"
    content: "torch.Tensor [N, 4, 512]"
  labels_parquet:
    filename: "labels.parquet"
    columns:
      - contract_id: string
      - source: string
      - class_0: int8  # CallToUnknown
      - class_1: int8  # DenialOfService
      # ... (10 classes total)
      - confidence_0: float32
      # ...
      - split: string
  metadata_parquet:
    filename: "metadata.parquet"
    columns:
      - contract_id: string
      - source: string
      - solc_version: string
      - version_bucket: string
      - loc: int
      - n_functions: int
      - n_pos: int
      - primary_class: string
      - confidence_tier: string
```

## How to Use

```bash
# Export a registered dataset version
sentinel-data export --dataset-version sentinel-v2-dryrun-2026-08

# Export with default (latest registered version)
sentinel-data export

# Dry-run
sentinel-data export --dry-run
```

## Pipeline Position

```
Stage 8: Analysis (6 exploratory tools)
    ↓
Stage 9: Export ← YOU ARE HERE (sharded output for ML)
    ↓
Stage 10: Seam Swap (replace old ML loader)
```

## Design Decisions

1. **Sharded export** — 5,000 contracts per shard for efficient lazy loading
2. **Schema versioned** — v1 is the contract; future bumps are new files
3. **Hash-verified** — ML module verifies hash on load (prevents tampering)
4. **Separate label + metadata parquet** — labels are training input; metadata is analysis input
5. **Confidence tier as a field** — enables per-class loss weighting in Run 11
