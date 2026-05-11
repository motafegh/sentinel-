# scripts — Runnable Entry Points

All commands must be run from the **project root** (`~/projects/sentinel`), not from `ml/scripts/`.

```bash
poetry run python ml/scripts/<script>.py [args]
```

Every script inserts the project root into `sys.path` so `ml.src.*` imports work regardless of working directory.

---

## Script Index

### Data Pipeline (run once, in order)

| Script | Purpose |
|---|---|
| `fix_labels_from_csv.py` | Patches `graph.y` in all `.pt` files from `contract_labels_correct.csv` |
| `create_label_index.py` | Scans all graph files → `ml/data/processed/label_index.csv` |
| `create_splits.py` | Reads `label_index.csv` → stratified train/val/test `.npy` index files |

### Preprocessing (run once to generate data)

| Script | Purpose |
|---|---|
| `ast_extractor_v4_production.py` | Slither-based AST extraction → `ml/data/graphs/` (68,556 `.pt` files) |
| `tokenizer_v1_production.py` | CodeBERT tokenization → `ml/data/tokens/` (68,570 `.pt` files) |

### Training

| Script | Purpose |
|---|---|
| `run_overnight_experiments.py` | Sequential launcher for 4 hyperparameter experiments |

### Post-Training

| Script | Purpose |
|---|---|
| `tune_threshold.py` | Sweeps decision thresholds on val set → optimal inference cutoff |

### Validation & Analysis

| Script | Purpose |
|---|---|
| `comprehensive_data_validation.py` | Full integrity check on graphs, tokens, labels, and splits |
| `analyze_token_stats.py` | Token length distribution across all 68,570 token files |

### Smoke Tests (not pytest — direct execution)

| Script | Purpose |
|---|---|
| `test_dataset.py` | Sanity check on `DualPathDataset` |
| `test_dataloader.py` | Sanity check on `DataLoader` with `dual_path_collate_fn` |
| `test_gnn_encoder.py` | Forward pass test for `GNNEncoder` |
| `test_fusion_layer.py` | Forward pass test for `FusionLayer` |
| `test_sentinel_model.py` | End-to-end forward pass test for `SentinelModel` |

---

## Detailed Reference

### fix_labels_from_csv.py

Reads `ml/data/processed/contract_labels_correct.csv` (columns: `file_hash`, `binary_label`, `class_label`).
Rewrites `graph.y` on every graph `.pt` file to match the CSV ground truth.

**Run when:** label corrections were applied to the CSV after graphs were generated.

```bash
poetry run python ml/scripts/fix_labels_from_csv.py
```

---

### create_label_index.py

Scans all `ml/data/graphs/*.pt` files, extracts `graph.y` from each, and writes:

```
ml/data/processed/label_index.csv
   columns: hash, label
   rows:    one per graph file
```

Input to `create_splits.py`. Run once — output is stable as long as graphs don't change.

```bash
poetry run python ml/scripts/create_label_index.py
```

---

### create_splits.py

Reads `label_index.csv`, performs stratified 70/15/15 split (seed=42), writes:

```
ml/data/splits/train_indices.npy   # 47,988 positions
ml/data/splits/val_indices.npy     # 10,283 positions
ml/data/splits/test_indices.npy    # 10,284 positions
```

Indices are integer positions into the sorted paired-hash list.
Class ratio (64.3%/35.7%) is preserved in all three splits.

```bash
poetry run python ml/scripts/create_splits.py
```

---

### ast_extractor_v4_production.py (V4.2)

Uses Slither to parse each `.sol` file into an AST/CFG, then extracts per-node features
(type, visibility, mutability, state-variable flags, etc.).
Saves each contract as `{md5_hash}.pt` in `ml/data/graphs/`.

**Node features (8-dimensional):**
Encodes structural contract properties — exact feature set defined in `ASTExtractor`.

Features:
- Multiprocessing (11 workers)
- Checkpoint/resume system (skips already-processed files)
- Handles older Slither APIs and solc versions ≥ 0.5.0

```bash
poetry run python ml/scripts/ast_extractor_v4_production.py
```

---

### tokenizer_v1_production.py

Tokenizes each `.sol` file with `microsoft/codebert-base` tokenizer.
Saves `{input_ids: [1,512], attention_mask: [1,512], contract_hash: str}` as `{md5_hash}.pt` in `ml/data/tokens/`.

Settings: `max_length=512`, `truncation=True`, `padding="max_length"`.
MD5 hash naming matches graph files — same hash = same contract.

Features:
- Multiprocessing (11 workers)
- Checkpoint/resume system
- Batch processing

```bash
poetry run python ml/scripts/tokenizer_v1_production.py
```

---

### run_overnight_experiments.py

Sequential launcher for 4 training experiments. Runs them one after another on the GPU.
Each experiment creates a separate MLflow run and checkpoint.

**Experiment matrix:**

| Run name | Change vs baseline | Hypothesis |
|---|---|---|
| `run-alpha-tune` | `focal_alpha=0.35` | Softer weight gap — test alpha sensitivity |
| `run-more-epochs` | `epochs=40` | Find real plateau (model still climbing at ep16) |
| `run-lr-lower` | `lr=3e-5, epochs=30` | Reduce F1 oscillation (~0.15 range) |
| `run-combined` | `focal_alpha=0.35, lr=3e-5, epochs=30` | Both nudges combined |

Baseline defaults not listed in a row are unchanged: `epochs=20`, `lr=1e-4`, `focal_alpha=0.25`, `batch_size=32`.

Each failed experiment is caught and logged — the launcher continues to the next run rather than aborting.

**First run:**
```bash
nohup poetry run python ml/scripts/run_overnight_experiments.py \
    > ml/logs/overnight.log 2>&1 &
echo "PID: $!"
```

**Resume after crash (e.g. from experiment 3):**
```bash
nohup poetry run python ml/scripts/run_overnight_experiments.py \
    --start-from 3 > ml/logs/overnight_resume.log 2>&1 &
```

**Morning check:**
```bash
tail -50 ml/logs/overnight.log
```

Checkpoints saved to: `ml/checkpoints/{run_name}_best.pt`

---

### tune_threshold.py

Sweeps decision thresholds from 0.30 to 0.70 (step 0.05) on the **val set**.
Prints F1-vuln, Precision, Recall, and F1-macro at each threshold.
Reports the single threshold that maximises F1-vuln.

```bash
poetry run python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/sentinel_best.pt
```

**Output:**
```
 Threshold |  F1-vuln |  Precision |   Recall |  F1-macro
------------------------------------------------------------
      0.30 |   0.7841 |     0.7102 |   0.8742 |   0.6934
      0.35 |   0.7923 |     0.7289 |   0.8695 |   0.7012
      ...
      0.50 |   0.7133 |     0.8442 |   0.6153 |   0.6515  ← best
      ...
────────────────────────────────────────────────────────────
  Best threshold : 0.50
  → Set INFERENCE_THRESHOLD = 0.50 in predictor.py
```

**Design:** runs one forward pass over the val set, caches the probabilities as numpy arrays, then applies all thresholds to the cache — no repeated GPU forward passes.

Takes `--checkpoint` (default: `ml/checkpoints/sentinel_best.pt`) — pass the overnight experiment checkpoints to compare:
```bash
poetry run python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/run-combined_best.pt
```

---

### comprehensive_data_validation.py

Full integrity check before training. Validates:
- Graph files: node feature shape `[N, 8]`, valid `edge_index`, label `0` or `1`
- Token files: `input_ids` and `attention_mask` shape `[1, 512]` (or `[512]`)
- Label CSV: coverage against graph hash set
- A random sample of 1,000 paired contracts end-to-end

```bash
poetry run python ml/scripts/comprehensive_data_validation.py
```

---

### analyze_token_stats.py

Scans all 68,570 token files and reports:
- Real token length distribution (via `attention_mask.sum()`)
- Count and % of truncated contracts (real length == 512)
- Mean, median, percentile statistics

```bash
poetry run python ml/scripts/analyze_token_stats.py
```
