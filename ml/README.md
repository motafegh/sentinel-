# M1 — ML Core

Dual-path vulnerability detector for Solidity smart contracts. A Graph Attention Network encodes the contract's structure; a LoRA-fine-tuned CodeBERT encodes its source text. A bidirectional CrossAttentionFusion merges the two paths before a 10-class multi-label classifier produces per-vulnerability probabilities.

---

## Model Architecture

```
Solidity source string
        │
        ├── ContractPreprocessor
        │     ├── Slither  →  PyG Data(x [N,8], edge_index [2,E])
        │     └── CodeBERT tokenizer  →  input_ids [1,512], attention_mask [1,512]
        │
        ▼
SentinelModel
  ├── GNNEncoder  (3-layer GAT)
  │     conv1: in=8,  out=8,  heads=8  → [N, 64]
  │     conv2: in=64, out=8,  heads=8  → [N, 64]
  │     conv3: in=64, out=64, heads=1  → [N, 64]
  │     Returns (node_embs [N,64], batch [N])  — NO pooling yet
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
Edge types: `CALLS`, `READS`, `WRITES`, `EMITS`, `INHERITS`

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
| Graph `.pt` files | 68 555 (MD5 stem, in `ml/data/graphs/`) |
| Token `.pt` files | 68 570 (MD5 stem, in `ml/data/tokens/`) |
| Splits | train 47 988 / val 10 283 / test 10 284 |
| Label CSV | `ml/data/processed/multilabel_index.csv` (68 555 rows × 10 classes) |

**Two hash systems — never mix:**
- SHA256 = hash of `.sol` file content → BCCC filename, CSV column 2
- MD5    = hash of `.sol` file path    → `.pt` filename

The bridge: `graph.contract_path` inside each `.pt` → `Path(...).stem` = SHA256.

---

## Active Checkpoint

```
File:       ml/checkpoints/multilabel_crossattn_best.pt
Thresholds: ml/checkpoints/multilabel_crossattn_best_thresholds.json
Val F1-macro: 0.4679  (epoch 34)
Architecture: "cross_attention_lora"
```

The thresholds companion file must travel with the checkpoint. Values are sweep-derived per class.

**DO NOT RETRAIN** without opening a new ML milestone:
- 8/10 classes are healthy
- DoS and CallToUnknown are data-limited; use per-class thresholds as the lever

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
  "num_nodes": 12,
  "num_edges": 18
}
```

`truncated: true` means the contract exceeded 512 CodeBERT tokens and was silently cut.
There is **no** top-level `confidence` field (removed in Track 3).

### `GET /health`

```json
{ "status": "ok", "architecture": "cross_attention_lora", "thresholds_loaded": true }
```

### HTTP status codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid / empty Solidity input |
| 413 | GPU out-of-memory |
| 503 | Predictor not loaded yet |
| 504 | Inference timeout |

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
| `batch_size` | 16 | Safe on RTX 3070 8 GB; set 32 only on >8 GB VRAM |
| `architecture` | `"cross_attention_lora"` | Written into checkpoint config |
| `grad_clip` | 1.0 | Prevents LoRA gradient spikes |
| `warmup_pct` | 0.1 | OneCycleLR warm-up fraction |

Speed optimisations active: AMP/BF16, TF32 matmuls, persistent DataLoader workers, `zero_grad(set_to_none=True)`.

### Resume from checkpoint

```bash
poetry run python scripts/train.py \
  --resume-from ml/checkpoints/multilabel_crossattn_best.pt
```

### Per-class threshold tuning

```bash
poetry run python scripts/tune_threshold.py \
  --checkpoint ml/checkpoints/multilabel_crossattn_best.pt
# Writes: ml/checkpoints/multilabel_crossattn_best_thresholds.json
```

---

## MLflow

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

Experiment: `sentinel-multilabel`.
Tracked per run: all `TrainConfig` fields, `val_f1_macro`, `val_f1_micro`, `val_hamming`, `val_f1_{class}` × 10.

---

## Data Extraction (Offline Batch)

Convert raw `.sol` files to `.pt` graph + token files:

```bash
# Requires: Docker (for Slither isolation) or local solc + slither
cd ml
docker build -f docker/Dockerfile.slither -t sentinel-slither .
poetry run python data_extraction/ast_extractor.py \
  --input-dir /path/to/contracts \
  --output-dir data/graphs/
poetry run python scripts/create_splits.py
poetry run python scripts/build_multilabel_index.py
```

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

Key tests: `tests/test_api.py` covers `/predict` and `/health` endpoint contracts.

---

## File Reference

```
ml/src/models/
  sentinel_model.py         SentinelModel — orchestrates all sub-modules
  gnn_encoder.py            GNNEncoder — 3-layer GAT, returns node_embs [N,64]
  transformer_encoder.py    TransformerEncoder — CodeBERT + LoRA r=8
  fusion_layer.py           CrossAttentionFusion — output_dim=128

ml/src/inference/
  api.py                    FastAPI app — lifespan, /predict, /health
  predictor.py              Predictor — checkpoint loading, sigmoid, thresholds
  preprocess.py             ContractPreprocessor — Slither graph + tokenisation

ml/src/training/
  trainer.py                Trainer, TrainConfig, CLASS_NAMES, NUM_CLASSES
  focalloss.py              FocalLoss (currently unused; BCEWithLogitsLoss active)

ml/src/datasets/
  dual_path_dataset.py      DualPathDataset, dual_path_collate_fn

ml/data_extraction/
  ast_extractor.py          Offline batch Slither → PyG .pt conversion
  tokenizer.py              Offline CodeBERT tokenisation

ml/checkpoints/             Not in git — managed by DVC
  multilabel_crossattn_best.pt
  multilabel_crossattn_best_thresholds.json
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
| `CLASS_NAMES` order | indices 0–9 stable | Silent wrong-class mapping |
| `weights_only` on `torch.load` | `False` | LoRA state dict is not a plain dict |
| `TRANSFORMERS_OFFLINE` | set at shell level | Cannot be set inside Python |

---

## Known Limitations

1. **512-token ceiling** — contracts longer than ~2 KB of source are truncated. `truncated: true` in the response signals this. A sliding-window or long-context model approach requires retraining. See `SENTINEL-SPEC.md §5.11`.
2. **Single-contract scope** — only the first non-dependency contract in a multi-contract file is analysed. Multi-file protocol audits are not yet supported.
3. **DoS / CallToUnknown** — data-limited classes; their per-class thresholds may be looser. Do not retrain; tune thresholds instead.
