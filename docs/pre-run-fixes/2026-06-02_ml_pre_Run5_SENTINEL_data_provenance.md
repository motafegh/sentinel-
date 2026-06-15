# SENTINEL Run 5 — Data Preparation Provenance

**Date:** 2026-06-02  
**Purpose:** Record every data preparation step executed for Run 5, including source,
script, output, and compatibility notes. This document is the authoritative reference
for "what data does Run 5 train on and where did it come from."

---

## Overview

Run 5 trains on **v9 data**: v9 graphs (re-extracted with all Phase 2 fixes) paired with
**v9 tokens** (re-tokenized with `microsoft/graphcodebert-base`, same tokenizer used by
the TransformerEncoder at training time).

| Artifact | Version | Count | Size | Source script |
|---|---|---|---|---|
| `ml/data/graphs/*.pt` | v9 | 41,576 | ~0.9 GB | `reextract_graphs.py` |
| `ml/data/tokens_windowed/*.pt` | v9 | ~44,530 | ~1.5 GB | `retokenize_windowed.py` |
| `ml/data/cached_dataset_v9.pkl` | v9 | 41,576 pairs | ~2.2 GB | `create_cache.py` |
| `ml/data/processed/multilabel_index.csv` | v9 | 41,576 rows | ~2.2 MB | `build_multilabel_index.py` |
| `ml/data/splits/v9_deduped/` | v9 | train/val/test | — | `create_splits.py` |

---

## Step 1 — Archive v8 Data

**Script:** `ml/scripts/archive_v8_data.py`  
**Command:**
```bash
echo "yes" | PYTHONPATH=. python ml/scripts/archive_v8_data.py
```
**What it moved to `ml/data/archive/`:**

| Source | Archive name | Size |
|---|---|---|
| `ml/data/graphs/` | `graphs_v8_pre_run5/` | 0.90 GB (41,576 files) |
| `ml/data/cached_dataset_v8.pkl` | `cached_dataset_v8.pkl` | 2.22 GB |
| `ml/data/tokens_windowed/` | `tokens_windowed_codebert_base/` | 1.45 GB (44,530 files) |
| `ml/data/processed/multilabel_index_cleaned.csv` | `multilabel_index_cleaned_v8.csv` | 2.3 MB |
| `ml/data/processed/multilabel_index.csv` | `multilabel_index_v8.csv` | 2.2 MB |
| `ml/data/splits/deduped/` | `splits_v8_deduped/` | — |
| `ml/checkpoints/` | `checkpoints_pre_run5/` | 12.85 GB (32 files) |
| `ml/logs/` | `logs_pre_run5/` | 0.22 GB |

**Manifest:** `ml/data/archive/v8_archive_manifest.txt`

> **Note on archived tokens:** The v8 tokens in `tokens_windowed_codebert_base/` were
> tokenized using `microsoft/codebert-base` (2026-05-18). Although `codebert-base` and
> `graphcodebert-base` share an **identical tokenizer** (same 50,265-token BPE vocabulary,
> byte-for-byte identical encoding — verified 2026-06-02), v9 tokens were re-generated
> with `graphcodebert-base` for correctness and to eliminate any ambiguity.

---

## Step 2 — Graph Re-extraction (v9)

**Script:** `ml/scripts/reextract_graphs.py`  
**Command:**
```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/reextract_graphs.py --workers 11
```
**Source:** 44,524 `.sol` contracts from `multilabel_index_deduped.csv` located in:
- `BCCC-SCsVul-2024/SourceCodes/`
- `ml/data/SolidiFI-processed/`
- `ml/data/SolidiFI/`
- `ml/data/smartbugs-curated/`
- `ml/data/smartbugs-wild/`
- `ml/data/augmented/`

**Extractor:** `ml/src/preprocessing/graph_extractor.py` (v8 schema, all Phase 2 fixes applied: A4–A18, NF-1/7/10/11)  
**Output:** `ml/data/graphs/` — 41,576 `.pt` files  
**Checkpoint:** `ml/data/graphs/reextract_checkpoint.json` (44,524 entries — all processed)

**Results:**

| Outcome | Count | Notes |
|---|---|---|
| OK (nodes > 3) | ~41,576 | Written to `ml/data/graphs/*.pt` |
| Ghost (≤3 nodes) | ~78 | Interface-only contracts; not written |
| Skip (Slither/solc fail) | ~2,870 | Compilation errors; not written |
| Fail | 0 | No unexpected failures |

**Schema:** v8 (NODE_FEATURE_DIM=11, NUM_EDGE_TYPES=11 incl. runtime-only REVERSE_CONTAINS=7)  
**New in v9 graphs:** `graph.has_cei_path` attribute (BFS-based CEI detection, see Phase 7 / `_compute_has_cei_path()`)

**Gate 7.1 validation:**
```bash
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py \
  --check-contains-edges --check-control-flow --check-block-globals
```
Result: 41,456 PASS / 120 advisory (interface-only contracts with no CFG — expected). 0 schema violations.

---

## Step 3 — Token Re-extraction (v9)

**Script:** `ml/scripts/retokenize_windowed.py`  
**Command:**
```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/retokenize_windowed.py --workers 11
```
**Source:** Same 44,524 `.sol` contracts as graph extraction (MD5-keyed).  
**Tokenizer:** `microsoft/graphcodebert-base` — the same model used by `TransformerEncoder` at training time.

**Parameters:**
- Window size: 512 tokens (GraphCodeBERT max sequence length)
- Stride: 256 tokens (50% overlap between consecutive windows)
- Max windows per contract: 4 (very long contracts sub-sampled via linspace)
- Output shape per contract: `[W, 512]` where W ∈ {1, 2, 3, 4}

**Output:** `ml/data/tokens_windowed/` — ~44,530 `.pt` files  
**Schema per file:**
```python
{
  "input_ids":      Tensor[W, 512],  # int64
  "attention_mask": Tensor[W, 512],  # int64
  "num_windows":    int,
  "stride":         int,             # 256
  "tokenizer_name": "microsoft/graphcodebert-base",
}
```

> **Why re-tokenized (not reused from v8):**  
> The v8 tokens were created with `microsoft/codebert-base`. Although that tokenizer is
> provably identical to `microsoft/graphcodebert-base` (same 50,265-token vocabulary,
> byte-for-byte identical IDs verified 2026-06-02), re-tokenizing with the correct model
> eliminates any ambiguity and ensures the `tokenizer_name` metadata in each `.pt` file
> matches the model used at training time.

---

## Step 4 — Cache Build (v9)

**Script:** `ml/scripts/create_cache.py`  
**Command:**
```bash
PYTHONPATH=. python ml/scripts/create_cache.py --output ml/data/cached_dataset_v9.pkl
```
**Input:** `ml/data/graphs/` (41,576 files) + `ml/data/tokens_windowed/` (~44,530 files)  
**Output:** `ml/data/cached_dataset_v9.pkl` — 41,576 `(graph, tokens)` pairs, ~2.2 GB  
**Skipped:** 2,948 MD5s with tokens but no graph (Slither failures from Step 2)

> **Note:** First run used v8 (codebert-base) tokens and was rebuilt after Step 3
> completes with v9 (graphcodebert-base) tokens.

---

## Step 5 — Label Index (v9)

**Script:** `ml/scripts/build_multilabel_index.py`  
**Command:**
```bash
PYTHONPATH=. python ml/scripts/build_multilabel_index.py
```
**Source:** BCCC-SCsVul-2024 ground-truth CSV (`BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv`, 111,897 rows, 68,433 unique SHA256s)  
**Graph input:** `ml/data/graphs/` (each `.pt` has `.contract_path` attribute for SHA256 lookup)  
**Output:** `ml/data/processed/multilabel_index.csv` — 41,576 rows, 10 label columns  
**Classes (10, locked):** CallToUnknown, DenialOfService, ExternalBug, GasException, IntegerUO, MishandledException, Reentrancy, Timestamp, TransactionOrderDependence, UnusedReturn

---

## Step 6 — Splits (v9)

**Script:** `ml/scripts/create_splits.py`  
**Command:**
```bash
PYTHONPATH=. python ml/scripts/create_splits.py \
  --splits-dir ml/data/splits/v9_deduped \
  --multilabel-csv ml/data/processed/multilabel_index.csv
```
**Output:** `ml/data/splits/v9_deduped/` — `train_indices.npy`, `val_indices.npy`, `test_indices.npy`  
**Ratio:** 70/15/15 stratified split

---

## Compatibility Matrix

| Component | Uses | Verified |
|---|---|---|
| GNNEncoder | `graph.x [N,11]`, `graph.edge_attr [E]` IDs 0–10 (excl. 7 runtime) | Gate 7.1 ✓ |
| TransformerEncoder | `input_ids [W,512]`, `attention_mask [W,512]` from `graphcodebert-base` tokenizer | Step 3 ✓ |
| CrossAttentionFusion | `fusion_max_nodes=1024`; 227 contracts >1024 nodes truncated (0.55%) | Gate 5.3 ✓ |
| DualPathDataset | reads `cached_dataset_v9.pkl`; MD5 keys match between graphs and tokens | Step 4 ✓ |
| Label CSV | `multilabel_index.csv` with 10 classes, `md5_stem` key matching graph filenames | Step 5 ✓ |
| Splits | `ml/data/splits/v9_deduped/` — numpy `.npy` index arrays | Step 6 |

---

## v8 → v9 Delta

What changed in v9 vs v8:

| Item | v8 | v9 |
|---|---|---|
| Graph extraction | Original extractor (A4–A18 bugs unfixed) | All Phase 2 fixes applied |
| CEI path label | Absent | `graph.has_cei_path` on every graph |
| Tokenizer model string | `microsoft/codebert-base` | `microsoft/graphcodebert-base` |
| Token content | Functionally identical (same vocab) | Identical byte-for-byte |
| Token metadata `tokenizer_name` | `"microsoft/codebert-base"` | `"microsoft/graphcodebert-base"` |
| Cache file | `cached_dataset_v8.pkl` (2.32 GB) | `cached_dataset_v9.pkl` (~2.2 GB) |
| Splits dir | `ml/data/splits/deduped/` | `ml/data/splits/v9_deduped/` |
