# M1 — ML Core

Dual-path vulnerability detector for Solidity smart contracts. A Graph Attention Network encodes the contract's AST structure (including typed edge relations); a LoRA-fine-tuned CodeBERT encodes its source text. A bidirectional CrossAttentionFusion merges the two representations before a 10-class multi-label sigmoid classifier produces per-vulnerability probabilities.

---

## Table of Contents

- [Model Architecture](#model-architecture)
- [Output Classes](#output-classes)
- [Dataset](#dataset)
- [Active Checkpoint](#active-checkpoint)
- [Inference API](#inference-api)
- [Inference Cache](#inference-cache-t1-a)
- [Drift Detection](#drift-detection-t2-b)
- [Training](#training)
- [MLflow and Model Registry](#mlflow-and-model-registry)
- [Data Extraction (Offline Batch)](#data-extraction-offline-batch)
- [DVC](#dvc)
- [Testing](#testing)
- [File Reference](#file-reference)
- [Critical Constraints](#critical-constraints)
- [Known Limitations](#known-limitations)

---

## Model Architecture

```
Solidity source string
        │
        ├── ContractPreprocessor
        │     ├── InferenceCache lookup  ←  key: "{content_md5}_{FEATURE_SCHEMA_VERSION}"
        │     ├── Slither  →  PyG Data(x [N,8], edge_index [2,E], edge_attr [E])
        │     └── CodeBERT tokenizer  →  input_ids [1,512], attention_mask [1,512]
        │
        ▼
SentinelModel
  ├── GNNEncoder  (3-layer GAT + edge-type embeddings)
  │     edge_emb: Embedding(5, 16)  →  edge vectors [E, 16]
  │     conv1: in=8,  out=8,  heads=8, edge_dim=16  → [N, 64]
  │     conv2: in=64, out=8,  heads=8, edge_dim=16  → [N, 64]
  │     conv3: in=64, out=64, heads=1, edge_dim=16  → [N, 64]
  │     Returns (node_embs [N,64], batch [N])  — no graph-level pooling yet
  │     Graceful degradation: edge_attr=None → zero-vectors (legacy .pt files still run)
  │
  ├── TransformerEncoder  (CodeBERT + LoRA)
  │     Backbone: microsoft/codebert-base  (124 M frozen params)
  │     LoRA r=8, lora_alpha=16 on query+value of all 12 layers  (~295 K trainable)
  │     Returns last_hidden_state [B, 512, 768]  — all token positions
  │
  ├── CrossAttentionFusion  (bidirectional)
  │     Project: nodes [N,64]→[N,256], tokens [B,512,768]→[B,512,256]
  │     Node→Token: each node attends over tokens to pick up relevant text context
  │     Token→Node: each token attends over nodes to pick up structural context
  │     Masked mean pool after enrichment  →  pooled_nodes [B,256], pooled_tokens [B,256]
  │     Concat [B,512]  →  Linear → ReLU → Dropout  →  [B,128]
  │     output_dim = 128  (LOCKED — ZKML proxy input_dim depends on this)
  │
  └── Classifier
        nn.Linear(128, 10)  →  raw logits [B,10]  (no Sigmoid inside model)

Predictor._score():
  probs = sigmoid(logits)             [1, 10]
  apply per-class threshold JSON
  vulnerabilities = [{vulnerability_class, probability} for prob ≥ threshold[cls]]
  label = "vulnerable" if vulnerabilities else "safe"
```

### Long-contract path — sliding window (T1-C)

Contracts exceeding 512 CodeBERT tokens are automatically split into overlapping 512-token windows (stride 256, max 8 windows). The GNN graph is built once from the full AST. Per-class probabilities are aggregated via `max` across all windows. The API response includes `windows_used: int`.

### Node feature vector (8-dim, locked order)

| Index | Feature | Encoding |
|-------|---------|---------|
| 0 | `type_id` | CONTRACT=7, STATE_VAR=0, FUNCTION=1, MODIFIER=2, EVENT=3, FALLBACK=4, RECEIVE=5, CONSTRUCTOR=6 |
| 1 | `visibility` | public/external=0, internal=1, private=2 |
| 2 | `pure` | 0/1 |
| 3 | `view` | 0/1 |
| 4 | `payable` | 0/1 |
| 5 | `reentrant` | 0/1 |
| 6 | `complexity` | float (CFG node count) |
| 7 | `loc` | float (lines of source) |

Node insertion order: `CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs`

Edge types (stored in `graph.edge_attr [E]`): `CALLS=0`, `READS=1`, `WRITES=2`, `EMITS=3`, `INHERITS=4`

---

## Output Classes

Defined in `src/training/trainer.py` as `CLASS_NAMES` — the single source of truth for index order.

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

Never insert into the middle. Append new classes at index 10+; indices 0–9 must remain stable across all checkpoints.

---

## Dataset

| Item | Value |
|------|-------|
| Source | BCCC-SCsVul-2024 |
| Graph `.pt` files | 68,523 (MD5 stem, in `ml/data/graphs/`) — re-extracted 2026-05-03 for `edge_attr [E]` shape |
| Token `.pt` files | 68,568 (MD5 stem, in `ml/data/tokens/`) |
| Splits | train 47,966 / val 10,278 / test 10,279 — 64.3% vulnerable (stratified from multilabel_index.csv) |
| Label CSV | `ml/data/processed/multilabel_index.csv` (68,523 rows × 10 classes) |

**Two hash systems — never mix:**
- SHA256 = hash of `.sol` file content → BCCC filename, CSV column 2
- MD5    = hash of `.sol` file path    → `.pt` filename

The bridge: `graph.contract_path` inside each `.pt` → `Path(...).stem` = SHA256.

### Graph dataset validation

Before retraining, confirm all `.pt` files pass:

```bash
python ml/scripts/validate_graph_dataset.py [--graphs-dir ml/data/graphs]
# Checks: edge_attr present, shape [E] (not [E,1]), values in [0, 5)
# Exit 0 = safe to retrain.  Exit 1 = regenerate with ast_extractor.py first.
```

---

## Active Checkpoint

```
── v3 (current best) ────────────────────────────────────────────────────────
File:        ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
Thresholds:  ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json
Run:         multilabel-v3-fresh-60ep  (sentinel-retrain-v3)
Completed:   2026-05-05  |  60 epochs  |  batch_size=32
Best epoch:  ~52–53
Raw F1-macro:    0.4715
Tuned F1-macro:  0.5069  ✅ (gate was > 0.4884)
Architecture:    cross_attention_lora  (LoRA r=8 α=16, edge_attr active P0-B)

── v2 (paused — superseded) ─────────────────────────────────────────────────
File:        ml/checkpoints/multilabel_crossattn_v2_best.pt
Status:      Stopped at epoch 43, batch-size mismatch. Superseded by v3.
Best raw F1: 0.4629 (epoch 37)

── baseline (pre-edge_attr) ─────────────────────────────────────────────────
File:        ml/checkpoints/multilabel_crossattn_best.pt
Val F1-macro: 0.4679  (epoch 34)
Architecture: cross_attention_lora  (trained WITHOUT edge_attr, pre-P0-B)
```

The thresholds companion JSON must travel with the checkpoint. Always run `tune_threshold.py` after completing a new training run.

Load pattern:

```python
raw = torch.load(path, weights_only=False)   # weights_only=True breaks LoRA state dict
state_dict = raw["model"] if "model" in raw else raw
```

### Per-class thresholds and F1 (v3)

| Class | Threshold | F1 | Precision | Recall | Support |
|-------|-----------|----|-----------|--------|---------|
| CallToUnknown | 0.70 | 0.394 | 0.322 | 0.507 | 1,266 |
| DenialOfService | 0.95 | 0.400 | 0.318 | 0.540 | 137 |
| ExternalBug | 0.65 | 0.435 | 0.312 | 0.715 | 1,622 |
| GasException | 0.55 | 0.550 | 0.403 | 0.867 | 2,589 |
| IntegerUO | 0.50 | 0.821 | 0.759 | 0.896 | 5,343 |
| MishandledException | 0.60 | 0.492 | 0.365 | 0.754 | 2,207 |
| Reentrancy | 0.65 | 0.536 | 0.449 | 0.665 | 2,501 |
| Timestamp | 0.75 | 0.479 | 0.403 | 0.591 | 1,077 |
| TransactionOrderDependence | 0.60 | 0.477 | 0.342 | 0.787 | 1,800 |
| UnusedReturn | 0.70 | 0.486 | 0.395 | 0.631 | 1,716 |

### Retrain evaluation protocol

| Gate | Requirement |
|------|-------------|
| Graph dataset | `validate_graph_dataset.py` exits 0 |
| Held-out split | Use `ml/data/splits/val_indices.npy` with the same seed — do NOT regenerate |
| **v4 success gate** | Tuned val F1-macro > **0.5069** on the same held-out split |
| Per-class floor | No class drops > 0.05 F1 from v3 tuned values above |
| Rollback rule | If tuned F1 < 0.5069: revert to v3 checkpoint; adjust hyperparameters |
| MLflow experiment | `sentinel-retrain-v4` |

---

## Inference API

Port `8001`. Start with:

```bash
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
SENTINEL_THRESHOLDS=ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json \
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001
```

Optional environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SENTINEL_CHECKPOINT` | `ml/checkpoints/multilabel-v3-fresh-60ep_best.pt` | Checkpoint path |
| `SENTINEL_THRESHOLDS` | auto-detected (same stem + `_thresholds.json`) | Per-class threshold JSON |
| `SENTINEL_PREDICT_TIMEOUT` | `60` | Inference timeout (seconds) |
| `SENTINEL_DRIFT_BASELINE` | `ml/data/drift_baseline.json` | Drift detection baseline |
| `SENTINEL_DRIFT_CHECK_INTERVAL` | `50` | KS test every N requests |

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
  "thresholds": [0.70, 0.95, 0.65, 0.55, 0.50, 0.60, 0.65, 0.75, 0.60, 0.70],
  "truncated": false,
  "windows_used": 1,
  "num_nodes": 12,
  "num_edges": 18
}
```

`thresholds` is a list of 10 per-class values in `CLASS_NAMES` index order.
`truncated: true` means the single window was cut at 512 tokens.
`windows_used > 1` means the sliding-window path (T1-C) was taken.

### `GET /health`

```json
{
  "status": "ok",
  "predictor_loaded": true,
  "checkpoint": "ml/checkpoints/multilabel-v3-fresh-60ep_best.pt",
  "architecture": "cross_attention_lora",
  "thresholds_loaded": true
}
```

### `GET /metrics`

Prometheus metrics endpoint (added by `prometheus-fastapi-instrumentator`).

| Metric | Type | Description |
|--------|------|-------------|
| `sentinel_model_loaded` | Gauge | 1 when predictor is loaded, 0 on shutdown |
| `sentinel_gpu_memory_bytes` | Gauge | GPU memory allocated (bytes), updated per request |
| `sentinel_drift_alerts_total{stat}` | Counter | KS drift alerts, labelled by stat name |

### HTTP status codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid or empty Solidity input |
| 413 | Source too large (> 1 MB) or GPU out-of-memory |
| 503 | Predictor not yet loaded |
| 504 | Inference timeout |

---

## Inference Cache (T1-A)

`InferenceCache` (`src/inference/cache.py`) is a disk-backed content-addressed cache that eliminates the 3–5 s Slither cost on repeated contracts.

Cache key: `"{content_md5}_{FEATURE_SCHEMA_VERSION}"` — bumping `FEATURE_SCHEMA_VERSION` in `graph_schema.py` automatically invalidates all stale entries.

```python
from ml.src.inference.cache import InferenceCache
from ml.src.inference.preprocess import ContractPreprocessor

cache = InferenceCache(cache_dir="~/.cache/sentinel/preprocess", ttl_seconds=86400)
preprocessor = ContractPreprocessor(cache=cache)
```

---

## Drift Detection (T2-B)

`DriftDetector` (`src/inference/drift_detector.py`) runs a Kolmogorov-Smirnov test per feature statistic against a pre-built baseline. It fires the `sentinel_drift_alerts_total` Prometheus counter when p < 0.05.

**Important:** do not build the baseline from training data. The BCCC-2024 corpus will produce false alerts on modern 2026 contracts.

```bash
# After collecting ≥ 500 real audit requests via the warm-up phase:
python ml/scripts/compute_drift_baseline.py \
    --source warmup \
    --warmup-log ml/data/warmup_stats.jsonl \
    --output ml/data/drift_baseline.json
```

Correct strategy: suppress all KS alerts during the first 500 requests (warm-up mode), then write the baseline and enable alerts.

---

## Training

```bash
cd ml
TRANSFORMERS_OFFLINE=1 poetry run python scripts/train.py \
  --run-name multilabel-v3-fresh-60ep \
  --experiment sentinel-retrain-v3 \
  --label-csv data/processed/multilabel_index.csv \
  --epochs 60 \
  --batch-size 32 \
  --patience 10
```

Key `TrainConfig` fields:

| Field | v3 value | Notes |
|-------|----------|-------|
| `architecture` | `"cross_attention_lora"` | Written into checkpoint config |
| `batch_size` | 32 | Safe on RTX 3070 8 GB with AMP |
| `lora_r` | 8 | LoRA rank — v4 plan: try **16** |
| `lora_alpha` | 16 | LoRA scaling factor |
| `loss_fn` | `"bce"` | v4 plan: switch to `"focal"` |
| `use_edge_attr` | True | Edge relation type embeddings (P0-B) |
| `gnn_edge_emb_dim` | 16 | Edge embedding dimension |
| `fusion_output_dim` | 128 | Fused representation size (LOCKED) |
| `grad_clip` | 1.0 | Prevents LoRA gradient spikes |
| `patience` | 10 | Early-stop on val F1-macro |

Speed optimisations active: AMP/BF16, TF32 matmuls, persistent DataLoader workers, `zero_grad(set_to_none=True)`.

### Resume from checkpoint

```bash
poetry run python scripts/train.py \
  --resume-from ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
# Validates: num_classes and architecture must match current TrainConfig
# Use --no-resume-model-only for full resume (model + optimizer + scheduler + patience counter)
# Use --resume-reset-optimizer to keep model weights but reset optimizer state
```

### Per-class threshold tuning

Run after every training completion. Sweeps thresholds 0.05–0.95 per class on the held-out validation split and writes a companion JSON.

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python scripts/tune_threshold.py \
  --checkpoint ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
# Writes: ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json
```

### Recommended v4 configuration

v3 plateaued at raw F1-macro 0.4715 from epoch ~54. The plateau signals a capacity ceiling under BCE loss with LoRA r=8. Recommended changes:

```bash
TRANSFORMERS_OFFLINE=1 poetry run python scripts/train.py \
  --run-name multilabel-v4-focal-lora16 \
  --experiment sentinel-retrain-v4 \
  --epochs 60 \
  --batch-size 32 \
  --patience 10 \
  --loss-fn focal \
  --focal-gamma 2.0 \
  --lora-r 16 \
  --lora-alpha 32
```

Key changes and rationale:
- `--loss-fn focal --focal-gamma 2.0` — down-weights easy negatives, forces attention on DenialOfService (137 support) and CallToUnknown
- `--lora-r 16` — doubles trainable params (~589K vs 295K), addressing the capacity ceiling
- Weighted sampler for DenialOfService recommended in addition (39× underrepresented vs IntegerUO)

---

## MLflow and Model Registry

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

Experiments:
- `sentinel-multilabel` — baseline, epoch 34, tuned F1 0.4679
- `sentinel-retrain-v2` — paused at epoch 43 (batch-size mismatch)
- `sentinel-retrain-v3` — complete, best raw F1 0.4715, tuned F1 **0.5069**

Tracked per run: all `TrainConfig` fields, `val_f1_macro`, `val_f1_micro`, `val_hamming`, `val_exact_match`, `focal_gamma`, `focal_alpha`, and `val_f1_{class}` × 10.

### Promoting a checkpoint to the registry

```bash
python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
    --stage Staging \
    --val-f1-macro 0.5069 \
    --note "v3: edge_attr active; tuned F1-macro 0.5069"

# --dry-run to preview without writing to MLflow
# --stage Production to promote (archives previous Production version)
```

---

## Data Extraction (Offline Batch)

Convert raw `.sol` files to `.pt` graph and token files. Only needed when the dataset changes or `FEATURE_SCHEMA_VERSION` is bumped.

```bash
cd ml

# Step 1 — extract graphs (requires local solc + slither, or Docker)
docker build -f docker/Dockerfile.slither -t sentinel-slither .
poetry run python data_extraction/ast_extractor.py \
  --input ml/data/processed/_cache/contracts_metadata.parquet \
  --output ml/data/graphs/

# Step 2 — tokenise
poetry run python data_extraction/tokenizer.py \
  --input ml/data/processed/_cache/contracts_metadata.parquet \
  --output ml/data/tokens/

# Step 3 — build multi-label index
poetry run python scripts/build_multilabel_index.py

# Step 4 — create stratified splits (do NOT regenerate if splits already exist)
poetry run python scripts/create_splits.py

# Step 5 — validate graph dataset before retraining
poetry run python scripts/validate_graph_dataset.py
```

> `create_label_index.py` is obsolete — `ast_extractor.py` sets `graph.y=0`. Binary labels for stratification are derived from `multilabel_index.csv` by `create_splits.py` (`sum(class_cols) > 0`).

---

## DVC

Large files (graphs, tokens, splits, checkpoints) are DVC-tracked, not stored in git.

```bash
dvc pull   # download current data version
dvc push   # push new artifacts after retraining
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
| `test_model.py` | SentinelModel forward pass, output shape, class count |
| `test_gnn_encoder.py` | GNNEncoder: edge_attr embedding, graceful degradation on None, head divisibility |
| `test_fusion_layer.py` | CrossAttentionFusion: output shape, masked pooling, device parity |
| `test_preprocessing.py` | ContractPreprocessor, graph_extractor typed exceptions |
| `test_dataset.py` | DualPathDataset: pairing logic, split loading, collation |
| `test_trainer.py` | FocalLoss forward, trainer utilities |
| `test_api.py` | /predict and /health endpoint contracts, error codes |
| `test_cache.py` | InferenceCache: miss writes, hit returns same object, TTL expiry, schema version invalidation |
| `test_drift_detector.py` | DriftDetector: warm-up suppression, KS fires on p < 0.05, buffer rolling |
| `test_promote_model.py` | promote_model.py: stage validation, dry-run no-op, MLflow tag writes |

---

## File Reference

```
ml/src/models/
  sentinel_model.py         SentinelModel — top-level orchestrator
  gnn_encoder.py            GNNEncoder — 3-layer GAT + edge-type embeddings → [N, 64]
  transformer_encoder.py    TransformerEncoder — CodeBERT + LoRA (r=8 default)
  fusion_layer.py           CrossAttentionFusion — output_dim=128 (LOCKED)

ml/src/preprocessing/
  graph_schema.py           NODE_TYPES, EDGE_TYPES, FEATURE_NAMES,
                            FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM, NUM_EDGE_TYPES
  graph_extractor.py        extract_contract_graph() — Slither → PyG Data; typed exceptions

ml/src/inference/
  api.py                    FastAPI app — lifespan, /predict, /health, /metrics
  predictor.py              Predictor — checkpoint loading, sigmoid, per-class thresholds list
  preprocess.py             ContractPreprocessor — Slither + tokenisation + sliding window + cache
  cache.py                  InferenceCache — disk-backed content-addressed cache (T1-A)
  drift_detector.py         DriftDetector — KS-based feature drift monitoring (T2-B)

ml/src/training/
  trainer.py                Trainer, TrainConfig, CLASS_NAMES, NUM_CLASSES
  focalloss.py              FocalLoss — gamma=2.0 default, FP32 cast; opt-in via loss_fn="focal"

ml/src/datasets/
  dual_path_dataset.py      DualPathDataset, dual_path_collate_fn

ml/src/utils/
  hash_utils.py             get_contract_hash(), get_contract_hash_from_content()

ml/data_extraction/
  ast_extractor.py          Offline batch Slither → PyG .pt conversion (V4.3)
  tokenizer.py              Offline CodeBERT tokenisation with schema version metadata

ml/scripts/
  train.py                  Main training entry point (full-resume, reset-optimizer flags)
  tune_threshold.py         Per-class threshold sweep (0.05–0.95 grid)
  create_splits.py          Fixed stratified train/val/test split indices
  build_multilabel_index.py Build multilabel_index.csv from BCCC labels
  validate_graph_dataset.py Validate edge_attr presence + shape [E] + value range in .pt files
  analyse_truncation.py     Measure token truncation stats across dataset
  promote_model.py          MLflow model registry CLI — Staging / Production promotion
  compute_drift_baseline.py Build drift_baseline.json from warmup logs

ml/checkpoints/             Not in git — managed by DVC
  multilabel-v3-fresh-60ep_best.pt           ← active (v3 complete, tuned F1 0.5069)
  multilabel-v3-fresh-60ep_best_thresholds.json
  multilabel_crossattn_v2_best.pt            ← paused v2 (superseded)
  multilabel_crossattn_best.pt               ← original baseline (pre-edge_attr)
  multilabel_crossattn_best_thresholds.json
```

---

## Critical Constraints

| Constraint | Value | Consequence of change |
|-----------|-------|-----------------------|
| `GNNEncoder in_channels` | **8** | Rebuild all 68K graph files + retrain |
| CodeBERT model | `microsoft/codebert-base` | Rebuild token files + retrain |
| `MAX_TOKEN_LENGTH` | **512** | Rebuild token files + retrain |
| Node feature order | fixed 8-dim | Rebuild graph files + retrain |
| `CrossAttentionFusion output_dim` | **128** | Rebuild ZKML circuit + redeploy verifier |
| `FEATURE_SCHEMA_VERSION` | **`"v1"`** | Bump only alongside graph rebuild — invalidates inference cache |
| `CLASS_NAMES` order | indices **0–9 stable** | Silent wrong-class mapping across all consumers |
| `NUM_EDGE_TYPES` | **5** | Rebuild edge_emb layer + retrain |
| `weights_only=False` on checkpoint load | required | LoRA state dict is not a plain tensor dict |
| `TRANSFORMERS_OFFLINE` | must be set at shell level | Cannot be set inside Python after import |

---

## Known Limitations

1. **Multi-contract files** — only the first non-dependency contract per `.sol` file is analysed. `GraphExtractionConfig.multi_contract_policy` scaffold exists (`"first"`, `"by_name"`); the `"all"` policy is not yet implemented. See ROADMAP Move 9.

2. **DenialOfService class** — 137 training samples (39× fewer than IntegerUO). Even with threshold tuning to 0.95, F1 is 0.40. Weighted sampling and focal loss are the planned remediation for v4.

3. **Drift baseline not yet collected** — `DriftDetector` is code-complete but cannot be activated in production until the warm-up phase (first 500 real audit requests) has been run to generate `drift_baseline.json`. Use `compute_drift_baseline.py --source warmup` after warm-up completes.
