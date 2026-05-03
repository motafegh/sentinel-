# M1 — ML Core

Dual-path vulnerability detector for Solidity smart contracts. A Graph Attention Network encodes the contract's structure (including typed edge relations); a LoRA-fine-tuned CodeBERT encodes its source text. A bidirectional CrossAttentionFusion merges the two paths before a 10-class multi-label classifier produces per-vulnerability probabilities.

---

## Model Architecture

```
Solidity source string
        │
        ├── ContractPreprocessor
        │     ├── Slither  →  PyG Data(x [N,8], edge_index [2,E], edge_attr [E])
        │     └── CodeBERT tokenizer  →  input_ids [1,512], attention_mask [1,512]
        │      └── InferenceCache (T1-A)  ← content_md5_FEATURE_SCHEMA_VERSION key
        │
        ▼
SentinelModel
  ├── GNNEncoder  (3-layer GAT + edge-type embeddings)
  │     edge_emb: Embedding(5, 16)  →  edge vectors [E, 16]
  │     conv1: in=8,  out=8,  heads=8, edge_dim=16  → [N, 64]
  │     conv2: in=64, out=8,  heads=8, edge_dim=16  → [N, 64]
  │     conv3: in=64, out=64, heads=1, edge_dim=16  → [N, 64]
  │     Returns (node_embs [N,64], batch [N])  — NO pooling yet
  │     Graceful degradation: edge_attr=None → zero-vectors (old .pt files still run)
  │
  ├── TransformerEncoder  (CodeBERT + LoRA)
  │     Backbone: microsoft/codebert-base  (124 M frozen params)
  │     LoRA r=8, lora_alpha=16 on query+value of all 12 layers  (~295 K trainable)
  │     Returns last_hidden_state [B, 512, 768]  — all token positions
  │
  ├── CrossAttentionFusion  (bidirectional)
  │     Project: nodes [N,64]→[N,256], tokens [B,512,768]→[B,512,256]
  │     Node→Token attention: each node queries which tokens are relevant
  │     Token→Node attention: each token queries which nodes are relevant
  │     Masked mean pool AFTER enrichment  →  pooled_nodes [B,256], pooled_tokens [B,256]
  │     Concat [B,512]  →  Linear → ReLU → Dropout  →  [B,128]
  │     output_dim = 128  (LOCKED — ZKML proxy depends on this)
  │
  └── Classifier
        nn.Linear(128, 10)  →  raw logits [B,10]  (no Sigmoid inside model)

Predictor._score():
  probs = sigmoid(logits)          [1, 10]
  apply per-class threshold JSON
  vulnerabilities = [{vulnerability_class, probability} for prob ≥ threshold[cls]]
  label = "vulnerable" if vulnerabilities else "safe"
```

### Long-contract path (T1-C — sliding window)

Contracts exceeding 512 CodeBERT tokens are split into overlapping 512-token windows (stride 256, max 8 windows). The GNN graph is built once from the full AST. `predictor.predict_source()` automatically routes to the windowed path and aggregates per-class probabilities via `max` across windows. The response includes `windows_used: int`.

### Node Feature Vector (8-dim, locked order)

| Index | Feature | Encoding |
|-------|---------|---------|
| 0 | `type_id` | CONTRACT=7 STATE_VAR=0 FUNCTION=1 MODIFIER=2 EVENT=3 FALLBACK=4 RECEIVE=5 CONSTRUCTOR=6 |
| 1 | `visibility` | public/external=0 internal=1 private=2 |
| 2 | `pure` | 0/1 |
| 3 | `view` | 0/1 |
| 4 | `payable` | 0/1 |
| 5 | `reentrant` | 0/1 |
| 6 | `complexity` | float (CFG nodes) |
| 7 | `loc` | float (lines of source) |

Node insertion order: `CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs`
Edge types: `CALLS=0`, `READS=1`, `WRITES=2`, `EMITS=3`, `INHERITS=4` — stored in `graph.edge_attr [E]`

### Output Classes

Defined in `src/training/trainer.py` as `CLASS_NAMES` — the single source of truth.

| Index | Class |
|-------|-------|
| 0 | CallToUnknown |
| 1 | DenialOfService |
| 2 | ExternalBug |
| 3 | GasException |
| 4 | IntegerUO |
| 5 | MishandledException |
| 6 | Reentrancy |
| 7 | Timestamp |
| 8 | TransactionOrderDependence |
| 9 | UnusedReturn |

**Never insert into the middle.** Append new classes at index 10+; indices 0–9 must remain stable.

---

## Dataset

| Item | Value |
|------|-------|
| Source | BCCC-SCsVul-2024 |
| Graph `.pt` files | 68 523 (MD5 stem, in `ml/data/graphs/`) — re-extracted 2026-05-03 for `edge_attr [E]` shape |
| Token `.pt` files | 68 568 (MD5 stem, in `ml/data/tokens/`) |
| Splits | train 47 966 / val 10 278 / test 10 279 — 64.3% vulnerable (stratified from multilabel_index.csv) |
| Label CSV | `ml/data/processed/multilabel_index.csv` (68 523 rows × 10 classes) |

**Two hash systems — never mix:**
- SHA256 = hash of `.sol` file content → BCCC filename, CSV column 2
- MD5    = hash of `.sol` file path    → `.pt` filename

The bridge: `graph.contract_path` inside each `.pt` → `Path(...).stem` = SHA256.

### Graph dataset validation

Before retraining, confirm all `.pt` files have the correct `edge_attr`:

```bash
python ml/scripts/validate_graph_dataset.py [--graphs-dir ml/data/graphs]
# Checks: edge_attr present, shape [E] (not [E,1]), values in [0, 5)
# Exit 0 = safe to retrain. Exit 1 = regenerate with ast_extractor.py first.
```

---

## Active Checkpoint

```
── Baseline (still active) ──────────────────────────────────────────────────
File:         ml/checkpoints/multilabel_crossattn_best.pt
Thresholds:   ml/checkpoints/multilabel_crossattn_best_thresholds.json
Val F1-macro: 0.4679  (epoch 34)
Architecture: "cross_attention_lora"  (trained WITHOUT edge_attr, pre-P0-B)

── Retrain in progress ───────────────────────────────────────────────────────
File:         ml/checkpoints/multilabel_crossattn_v2_best.pt   ← will be written
Run name:     multilabel-v2-edge-attr
Experiment:   sentinel-retrain-v2
Epochs:       40 (running as of 2026-05-03)
edge_attr:    True  (P0-B active — Embedding(5,16) for CALLS/READS/WRITES/EMITS/INHERITS)
```

The thresholds companion file must travel with the checkpoint. Run `tune_threshold.py`
after the retrain completes to produce `multilabel_crossattn_v2_best_thresholds.json`.

### Retrain Evaluation Protocol

Before launching a retrain, all of the following must be confirmed:

| Gate | Requirement |
|------|-------------|
| Graph dataset | `validate_graph_dataset.py` exits 0 (edge_attr present + valid) |
| Held-out split | Use `ml/data/splits/val_indices.npy` with the same seed — do NOT regenerate |
| Success threshold | val F1-macro > **0.4679** on the same held-out split |
| Per-class floor | No class drops > 0.05 F1 from pre-retrain value |
| Rollback rule | If F1 < 0.4679 after 40 epochs: revert checkpoint; investigate `edge_emb_dim` |
| MLflow experiment | `sentinel-retrain-v2`; compare against `sentinel-multilabel` baseline run |

Load pattern:
```python
raw = torch.load(path, weights_only=False)   # weights_only=True breaks LoRA
state_dict = raw["model"] if "model" in raw else raw
```

---

## Inference API

Port `8001`. Start with:

```bash
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel_crossattn_best.pt \
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001
```

Optional env vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SENTINEL_CHECKPOINT` | `ml/checkpoints/multilabel_crossattn_v2_best.pt` | Checkpoint path |
| `SENTINEL_PREDICT_TIMEOUT` | `60` | Inference timeout (seconds) |
| `SENTINEL_DRIFT_BASELINE` | `ml/data/drift_baseline.json` | Drift detection baseline |
| `SENTINEL_DRIFT_CHECK_INTERVAL` | `50` | KS check every N requests |

### `POST /predict`

```json
// Request
{ "source_code": "<solidity source string>" }

// Response
{
  "label": "vulnerable",
  "vulnerabilities": [
    { "vulnerability_class": "Reentrancy", "probability": 0.8943 },
    { "vulnerability_class": "IntegerUO",  "probability": 0.7102 }
  ],
  "threshold": 0.5,
  "truncated": false,
  "windows_used": 1,
  "num_nodes": 12,
  "num_edges": 18
}
```

`truncated: true` means a single window reached 512 tokens and was cut. `windows_used > 1` means the contract was long enough to trigger the sliding-window path (T1-C).

### `GET /health`

```json
{
  "status": "ok",
  "predictor_loaded": true,
  "checkpoint": "ml/checkpoints/multilabel_crossattn_best.pt",
  "architecture": "cross_attention_lora",
  "thresholds_loaded": true
}
```

### `GET /metrics`

Prometheus metrics endpoint (added by `prometheus-fastapi-instrumentator`). Custom gauges:

| Metric | Type | Description |
|--------|------|-------------|
| `sentinel_model_loaded` | Gauge | 1 when predictor is loaded, 0 on shutdown |
| `sentinel_gpu_memory_bytes` | Gauge | GPU memory allocated (bytes), updated per request |
| `sentinel_drift_alerts_total{stat}` | Counter | KS drift alerts fired, labelled by stat name |

### HTTP status codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid / empty Solidity input |
| 413 | Source too large (> 1 MB) or GPU out-of-memory |
| 503 | Predictor not loaded yet |
| 504 | Inference timeout |

---

## Inference Cache (T1-A)

`InferenceCache` (`src/inference/cache.py`) is a disk-backed content-addressed cache that eliminates the 3–5 s Slither cost for repeated contracts.

```python
from ml.src.inference.cache import InferenceCache
from ml.src.inference.preprocess import ContractPreprocessor

cache = InferenceCache(cache_dir="~/.cache/sentinel/preprocess", ttl_seconds=86400)
preprocessor = ContractPreprocessor(cache=cache)
```

Cache key: `"{content_md5}_{FEATURE_SCHEMA_VERSION}"` — bumping `FEATURE_SCHEMA_VERSION` in `graph_schema.py` automatically invalidates all stale entries.

---

## Drift Detection (T2-B)

`DriftDetector` (`src/inference/drift_detector.py`) runs a KS test per feature stat against a pre-built baseline. It fires `sentinel_drift_alerts_total` (Prometheus counter) when p < 0.05.

```bash
# After collecting ≥ 500 real audit requests via the warm-up phase:
python ml/scripts/compute_drift_baseline.py \
    --source warmup \
    --warmup-log ml/data/warmup_stats.jsonl \
    --output ml/data/drift_baseline.json
```

**Do not use `--source training`** — the BCCC-2024 corpus will cause false alerts on modern 2026 contracts.

---

## Training

```bash
cd ml
TRANSFORMERS_OFFLINE=1 \
poetry run python scripts/train.py \
  --label-csv data/processed/multilabel_index.csv \
  --epochs 50 \
  --batch-size 16 \
  --lr 2e-4
```

Key `TrainConfig` fields:

| Field | Default | Notes |
|-------|---------|-------|
| `num_classes` | 10 | Must match `len(CLASS_NAMES)` |
| `architecture` | `"cross_attention_lora"` | Written into checkpoint config |
| `batch_size` | 16 | Safe on RTX 3070 8 GB |
| `grad_clip` | 1.0 | Prevents LoRA gradient spikes |
| `lora_r` | 8 | LoRA rank — try 16 or 32 for next retrain |
| `lora_alpha` | 16 | LoRA scaling factor |
| `use_edge_attr` | True | Edge relation type embeddings (P0-B) |
| `gnn_edge_emb_dim` | 16 | Edge embedding dimension |
| `fusion_output_dim` | 128 | Fused representation size (LOCKED) |

Speed optimisations active: AMP/BF16, TF32 matmuls, persistent DataLoader workers, `zero_grad(set_to_none=True)`.

### Resume from checkpoint

```bash
poetry run python scripts/train.py \
  --resume-from ml/checkpoints/multilabel_crossattn_v2_best.pt
# Validates: num_classes and architecture must match current TrainConfig
```

### Per-class threshold tuning

Run after retraining completes. Sweeps thresholds 0.05–0.95 per class, picks the
value that maximises per-class F1, and writes a companion JSON.

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python scripts/tune_threshold.py \
  --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt
# Writes: ml/checkpoints/multilabel_crossattn_v2_best_thresholds.json
```

---

## MLflow & Model Registry

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

Experiments: `sentinel-multilabel` (baseline, epoch 34, F1 0.4679) and `sentinel-retrain-v2` (v2 retrain, in progress).
Tracked per run: all `TrainConfig` fields, `val_f1_macro`, `val_f1_micro`, `val_hamming`, `val_f1_{class}` × 10.

### Promoting a checkpoint to the registry

```bash
python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt \
    --stage Staging \
    --val-f1-macro <actual_f1> \
    --note "v2 retrain: edge_attr embeddings (P0-B) active"

# --dry-run to preview without writing to MLflow
# --stage Production to promote (archives previous Production version)
```

---

## Data Extraction (Offline Batch)

Convert raw `.sol` files to `.pt` graph + token files:

```bash
# Requires: Docker (for Slither isolation) or local solc + slither
cd ml
docker build -f docker/Dockerfile.slither -t sentinel-slither .

# Step 1 — extract graphs (edge_attr shape [E] required)
poetry run python data_extraction/ast_extractor.py \
  --input ml/data/processed/_cache/contracts_metadata.parquet \
  --output ml/data/graphs/

# Step 2 — tokenise
poetry run python data_extraction/tokenizer.py \
  --input ml/data/processed/_cache/contracts_metadata.parquet \
  --output ml/data/tokens/

# Step 3 — build multi-label index (BCCC SHA256 → multi-hot labels)
poetry run python scripts/build_multilabel_index.py

# Step 4 — create stratified splits (reads multilabel_index.csv for binary labels)
poetry run python scripts/create_splits.py
```

**Note on label_index.csv:** `create_label_index.py` is obsolete — `ast_extractor.py`
hardcodes `graph.y=0` for all contracts. Binary labels for stratification are derived
from `multilabel_index.csv` by `create_splits.py` (`sum(class_cols) > 0`).

---

## DVC

Large files (graphs, tokens, splits, checkpoints) are DVC-tracked, not stored in git.

```bash
dvc pull          # download current data version
dvc push          # push new artifacts after retraining
```

---

## Testing

```bash
cd ml
poetry run pytest tests/ -v
```

10 test modules, all using synthetic data (no real contracts or checkpoints required):

| Test file | What it covers |
|-----------|---------------|
| `test_model.py` | SentinelModel forward pass, output shape |
| `test_gnn_encoder.py` | GNNEncoder: edge_attr embedding, graceful degradation, head divisibility |
| `test_fusion_layer.py` | CrossAttentionFusion: output shape, masked pooling, device mismatch |
| `test_preprocessing.py` | ContractPreprocessor and graph_extractor |
| `test_dataset.py` | DualPathDataset: pairing, splits, collation |
| `test_trainer.py` | FocalLoss, trainer utilities |
| `test_api.py` | /predict and /health endpoint contracts |
| `test_cache.py` | InferenceCache: miss/hit/TTL/schema-version |
| `test_drift_detector.py` | DriftDetector: warm-up suppression, KS fires on drift, buffer rolling |
| `test_promote_model.py` | promote_model.py: stage validation, dry-run, MLflow tags |

---

## File Reference

```
ml/src/models/
  sentinel_model.py         SentinelModel — orchestrates all sub-modules
  gnn_encoder.py            GNNEncoder — 3-layer GAT + edge-type embeddings [N,64]
  transformer_encoder.py    TransformerEncoder — CodeBERT + LoRA r=8
  fusion_layer.py           CrossAttentionFusion — output_dim=128

ml/src/preprocessing/
  graph_schema.py           Single source of truth: NODE_TYPES, EDGE_TYPES, FEATURE_NAMES,
                            FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM, NUM_EDGE_TYPES
  graph_extractor.py        extract_contract_graph() — Slither → PyG Data; typed exceptions

ml/src/inference/
  api.py                    FastAPI app — lifespan, /predict, /health, /metrics
  predictor.py              Predictor — checkpoint loading, sigmoid, per-class thresholds
  preprocess.py             ContractPreprocessor — Slither graph + tokenisation + cache
  cache.py                  InferenceCache — disk-backed content-addressed cache (T1-A)
  drift_detector.py         DriftDetector — KS-based feature drift monitoring (T2-B)

ml/src/training/
  trainer.py                Trainer, TrainConfig, CLASS_NAMES, NUM_CLASSES
  focalloss.py              FocalLoss — gamma=2.0, FP32 cast; used when loss_fn="focal"

ml/src/datasets/
  dual_path_dataset.py      DualPathDataset, dual_path_collate_fn

ml/src/utils/
  hash_utils.py             get_contract_hash(), get_contract_hash_from_content()

ml/data_extraction/
  ast_extractor.py          Offline batch Slither → PyG .pt conversion
  tokenizer.py              Offline CodeBERT tokenisation

ml/scripts/
  train.py                  Main training entry point
  tune_threshold.py         Per-class threshold sweep
  analyse_truncation.py     Measure token truncation across the dataset
  build_multilabel_index.py Build multilabel_index.csv from BCCC labels
  create_splits.py          Fixed train/val/test split indices
  validate_graph_dataset.py Validate edge_attr presence + shape + value range in .pt files
  promote_model.py          MLflow model registry CLI (Staging / Production)
  compute_drift_baseline.py Build drift_baseline.json from warmup logs or training graphs

ml/checkpoints/             Not in git — managed by DVC
  multilabel_crossattn_best.pt                        ← baseline (pre-P0-B, F1 0.4679)
  multilabel_crossattn_best_thresholds.json
  multilabel_crossattn_v2_best.pt                     ← v2 retrain (edge_attr active)
  multilabel_crossattn_v2_best_thresholds.json        ← written by tune_threshold.py post-retrain
```

---

## Critical Constraints

| Constraint | Value | Break condition |
|-----------|-------|----------------|
| `GNNEncoder in_channels` | 8 | Rebuild 68 K graph files + retrain |
| CodeBERT model | `microsoft/codebert-base` | Rebuild token files + retrain |
| `MAX_TOKEN_LENGTH` | 512 | Rebuild token files + retrain |
| Node feature order | fixed 8-dim | Rebuild graph files + retrain |
| `CrossAttentionFusion output_dim` | 128 | Rebuild ZKML circuit + redeploy |
| `FEATURE_SCHEMA_VERSION` | `"v1"` | Bump only alongside graph rebuild; invalidates inference cache |
| `CLASS_NAMES` order | indices 0–9 stable | Silent wrong-class mapping |
| `weights_only=False` on checkpoint load | required | LoRA state dict is not a plain dict |
| `TRANSFORMERS_OFFLINE` | set at shell level | Cannot be set inside Python |
| `NUM_EDGE_TYPES` | 5 | Changing requires edge_emb layer rebuild + retrain |

---

## Known Limitations

1. **Multi-contract files** — only the first non-dependency contract per `.sol` file is analysed. `GraphExtractionConfig.multi_contract_policy` scaffold exists (`"first"`, `"by_name"`); `"all"` policy is not yet implemented. See ROADMAP Move 9.
2. **DoS / CallToUnknown** — data-limited classes; per-class thresholds are the recommended tuning lever. Do not retrain solely for these two classes.
3. **Retrain in progress** — the v2 retrain (`multilabel-v2-edge-attr`, 40 epochs) is running as of 2026-05-03 with edge_attr active (P0-B). The baseline checkpoint (F1 0.4679) remains active until the retrain completes and clears the success gate (val F1-macro > 0.4679).
