# ml — SENTINEL Machine Learning Core

> **Status:** Current — v9 schema, four-eye v8.1 architecture, Run 12 in Staging

Dual-path smart contract vulnerability detector. An 8-layer **Graph Attention Network** encodes AST/CFG structure with typed edge relations across three phases; a **LoRA-adapted GraphCodeBERT** encodes source text across sliding windows with optional GNN-prefix injection. A **four-eye classifier** (GNN eye, Transformer eye, Fused eye, CFG eye) produces per-class probabilities across **10 vulnerability classes**.

**Architecture tag:** `"four_eye_v8"` | **Schema:** `FEATURE_SCHEMA_VERSION = "v9"` | **Node features:** `NODE_FEATURE_DIM = 12` | **Backbone:** `microsoft/graphcodebert-base`

---

## Table of Contents

- [Setup](#setup)
- [System Overview](#system-overview)
- [Model Architecture](#model-architecture)
  - [GNNEncoder](#gnnencoder)
  - [TransformerEncoder](#transformerencoder)
  - [GNN Prefix Injection](#gnn-prefix-injection)
  - [CrossAttentionFusion](#crossattentionfusion)
  - [Four-Eye Classifier](#four-eye-classifier)
  - [Node Feature Vector (v9, 12-dim)](#node-feature-vector)
  - [Edge Types (12)](#edge-types)
- [Output Classes](#output-classes)
- [Training](#training)
- [Inference](#inference)
- [Repository Layout](#repository-layout)
- [Key Invariants](#key-invariants)

---

## Setup

```bash
cd ~/projects/sentinel
poetry install
source ml/.venv/bin/activate
export TRANSFORMERS_OFFLINE=1          # required — prevents HuggingFace network calls
export TRITON_CACHE_DIR=/tmp/triton_cache  # required on WSL2 — avoids p9io crash
```

**Key dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | `^2.5.0` | Training, inference |
| `torch-geometric` | `^2.6.0` | GNN layers, graph batching |
| `transformers` | `^4.45.0` | GraphCodeBERT base model |
| `peft` | `>=0.13.0,<0.16.0` | LoRA adapters on GraphCodeBERT |
| `fastapi` | `^0.115.0` | Inference API server |
| `scipy` | `^1.13.0` | KS drift detection |
| `sentinel-data` | local | Shared data module (graph schema, export) |

---

## System Overview

```
Input: Solidity contract (.sol)
        |
        +-- graph_extractor.py ----> PyG Data (NODE_FEATURE_DIM=12, 12 edge types)
        |
        +-- retokenize_windowed.py -> tokens [W, 512] (W <= 4 windows, stride=256)
                |
        SentinelDataset (v2 export artifacts)
                |
        +-------+--------------------------------------------+
        |                                                   |
   GNNEncoder                                       TransformerEncoder
   8-layer GAT, 3 phases (2+3+3)                   GraphCodeBERT + LoRA r=16
   JK attention, hidden_dim=256                     12 layers, Q+V adapters
   type_embedding(14,16), input_proj skip           frozen base, BF16
   IMP-G1 edge subsets, IMP-G3 bidir CONTAINS      Flash Attention 2
        |                                           |
        |  [B, K, 256] prefix nodes (K=48)               |
        |  --gnn_to_bert_proj(256->768)-->               |
        |  prepended as inputs_embeds prefix              |
        |  (suppressed during warmup epochs 0-14)         |
        |                                                 |
        +---------------- CrossAttentionFusion -----------+
                         bidirectional cross-attention
                         node<->token, attn_dim=256
                         token_norm LayerNorm (BUG-C2)
                         need_weights=False (Fix #26)
                         output_dim=128 LOCKED
                |
         Four-Eye Classifier
         GNN eye [B,128] + TF eye [B,128] + Fused eye [B,128] + CFG eye [B,128]
         -> [B,512] -> Linear(512,256) -> Linear(256,10)
                |
         [B, 10] logits -> sigmoid -> thresholded predictions
```

---

## Model Architecture

### GNNEncoder

**File:** `ml/src/models/gnn_encoder.py` (667 lines)

8-layer Graph Attention Network, three phases (2+3+3), Jumping Knowledge (JK) attention aggregation. Fixed at `SENTINEL_GNN_NUM_LAYERS = 8` — any other value raises `ValueError`.

**Parameters:**
- `hidden_dim=256`, `heads=8` (Phase 1), `dropout=0.2`
- `use_edge_attr=True`, `edge_emb_dim=64`
- `use_jk=True`, `jk_mode='attention'`
- `drop_complexity=False`, `appnp_alpha=0.0`

```
Type Embedding (BUG-R7-2)
  Embedding(14, 16) -> [N, 16]
  Concatenated with raw features -> [N, 28]

Phase 1 -- Structural + CONTAINS (layers 1+2)
  GAT over edge types 0-5, 8 heads, add_self_loops=True
  IMP-G2: input_proj skip connection (Linear(28,256)) added before relu in Layer 1
  LayerNorm after phase

Phase 2 -- CFG + ICFG directed (layers 3+4+5)
  IMP-G1: each layer processes a DISTINCT edge subset
  conv3:  CONTROL_FLOW(6) only -- intra-function execution ordering
  conv3b: CALL_ENTRY(8) + RETURN_TO(9) only -- cross-function call structure
  conv3c: CF(6)+CALL_ENTRY(8)+RETURN_TO(9)+DEF_USE(10)+EXTERNAL_CALL(11) joint
  add_self_loops=False (CRITICAL -- self-loops cancel directional signal)
  heads=4 (IMP-R7-1), concat=True, out=64/head -> total 256
  LayerNorm after phase

Phase 3 -- Bidirectional CONTAINS (layers 6+7+8)
  conv4:  REVERSE_CONTAINS up -- CFG->FUNCTION (Phase 2 signal rises)
  conv4b: REVERSE_CONTAINS up -- second hop (multi-function patterns)
  conv4c: CONTAINS down (IMP-G3) -- FUNCTION->CFG, distributes enriched context
  1 head, concat=False
  LayerNorm after phase

JK attention aggregation over all 3 phase outputs -> hidden_dim=256
Edge type embedding: Embedding(12, 64) concatenated per message
```

**Forward returns:**
- `(node_embeddings, batch, jk_entropy)` — default
- `(node_embeddings, batch, jk_entropy, intermediates)` — when `return_intermediates=True` (diagnostic, detached)
- `(node_embeddings, batch, jk_entropy, phase2_x)` — when `return_phase2_embs=True` (CEI aux loss)

### TransformerEncoder

**File:** `ml/src/models/transformer_encoder.py` (388 lines)

`microsoft/graphcodebert-base` (124M params) + LoRA:
- Base model frozen; LoRA r=16, alpha=32 on Q+V of all 12 layers (~590K trainable)
- Flash Attention 2 support (falls back to SDPA if unavailable)
- `peft` library is a hard requirement — missing raises `RuntimeError` at import

**Input:**
- Single-window: `[B, L]` where L=512
- Multi-window: `[B, W, L]` where W windows of L=512 tokens each

**Output:**
- Single-window: `[B, L, 768]` — all token embeddings
- Multi-window: `[B, W*L, 768]` — all windows concatenated along seq dim

**GNN prefix path:** when `gnn_prefix_nodes` is not None:
- Uses `inputs_embeds` instead of `input_ids`
- Prefix occupies positions 0..K-1, code occupies K..K+code_budget-1
- `WindowAttentionPooler` CLS extraction: `i * window_size + prefix_k`

### GNN Prefix Injection

**Files:** `sentinel_model.py` (select_prefix_nodes), `transformer_encoder.py`

Declaration-level GNN node embeddings are projected into BERT space and prepended as soft prefix tokens:

```
select_prefix_nodes()
  Priority: CONSTRUCTOR > FALLBACK > RECEIVE > MODIFIER > FUNCTION
  Secondary sort: FUNCTION nodes by external_call_count descending (IMP-M1)
  Selects top-K=48 declaration nodes per contract

gnn_to_bert_proj: Linear(256, 768) -- projects GNN hidden_dim to BERT embedding_dim
prefix_type_embedding: Embedding(5, 768) -- type-specific bias per prefix type

Position IDs:
  Prefix tokens: position_id = 1  (RoBERTa padding slot)
  Code tokens:   position_ids = 3..3+code_budget-1

Warmup suppression (epochs 0..gnn_prefix_warmup_epochs-1):
  gnn_prefix_nodes = None passed to TransformerEncoder
  Projection starts from random init at epoch gnn_prefix_warmup_epochs (default 15)

IMP-M3: actual node count masking
  gnn_prefix_counts [B] tracks real (non-padded) nodes per graph
  Zero-padded prefix positions are masked in attention
```

**Inference:** `predictor.py` sets `model._current_epoch = 9999` after load so the prefix is always active.

### CrossAttentionFusion

**File:** `ml/src/models/fusion_layer.py` (282 lines)

Bidirectional cross-attention fusing graph and text modalities:

```
1. Project nodes [N,256] -> [N,256]
2. token_norm LayerNorm(768) + project tokens [B,512,768] -> [B,512,256] (BUG-C2 fix)
3. _scatter_to_dense -> static [B,2048,256]  (compile-safe; max_nodes=2048)
4. Node->Token cross-attention -> enriched_nodes [B,2048,256]
5. Token->Node cross-attention -> enriched_tokens [B,512,256]
6. Masked mean pooling of real nodes  -> [B,256]
7. Masked mean pooling of real tokens -> [B,256]
8. Concat [B,512] -> Linear + ReLU -> [B,128]
```

**output_dim=128 is LOCKED** -- the ZKML proxy MLP depends on this shape.

`_scatter_to_dense` replaces PyG's `to_dense_batch` to eliminate the `GuardOnDataDependentSymNode` compile graph break.

### Four-Eye Classifier

**File:** `ml/src/models/sentinel_model.py` (670 lines)

```
GNN eye:    max_pool + mean_pool over FUNCTION+MODIFIER+FALLBACK+RECEIVE+CONSTRUCTOR nodes
            -> [B,512] -> Linear -> [B,128]

TF eye:     WindowAttentionPooler -> [B,768] -> Linear -> [B,128]

Fused eye:  CrossAttentionFusion -> [B,128]

CFG eye:    max_pool + mean_pool over CFG_NODE types [8-12] (raw Phase 2 output)
            -> [B,512] -> Linear -> [B,128]

Concat [B,512] -> Linear(512,256) -> ReLU -> Dropout -> Linear(256,10) -> logits

Auxiliary heads (training only):
  aux_gnn         = Linear(128, 10)(gnn_eye)
  aux_transformer = Linear(128, 10)(transformer_eye)
  aux_fused       = Linear(128, 10)(fused_eye)
  aux_phase2      = MLP(256->128->10) pooled over CEI nodes (CALL+WRITE+CHECK)
```

### Node Feature Vector

**v9 schema, 12 dimensions:**

| Dim | Feature | Notes |
|-----|---------|-------|
| [0] | `type_id / 13.0` | NodeType enum value (0-13 -> 0.0-1.0) |
| [1] | `visibility` | 0.0=public/external, 0.5=internal, 1.0=private |
| [2] | `uses_block_globals` | 1.0 if reads block.timestamp/number/difficulty/etc. |
| [3] | `view` | 0/1 |
| [4] | `payable` | 0/1 |
| [5] | `complexity` | log1p(CFG_block_count) / log1p(100) |
| [6] | `loc` | log1p(lines) / log1p(1000) |
| [7] | `return_ignored` | 1.0 if call return value unused downstream |
| [8] | `call_target_typed` | 0=dynamic/unknown target, 1=typed interface |
| [9] | `has_loop` | 0/1 |
| [10] | `external_call_count` | log1p(count) / log1p(20) |
| [11] | `in_unchecked_block` | 1.0 if inside an unchecked block |

CFG nodes inherit dims [1,3,4,5,9] from their parent FUNCTION node.

Schema constants are defined in `sentinel_data.representation.graph_schema` and re-exported via `ml/src/preprocessing/graph_schema.py`.

### Edge Types

| Type | Name | Description |
|------|------|-------------|
| 0 | CALLS | function -> called function (internal) |
| 1 | READS | function -> state variable |
| 2 | WRITES | function -> state variable |
| 3 | EMITS | function -> event |
| 4 | INHERITS | contract -> parent contract |
| 5 | CONTAINS | contract/function -> child node |
| 6 | CONTROL_FLOW | CFG block -> CFG block |
| 7 | REVERSE_CONTAINS | flip of type 5, generated at runtime only |
| 8 | CALL_ENTRY | call site -> function entry CFG block |
| 9 | RETURN_TO | function exit CFG block -> call-site continuation |
| 10 | DEF_USE | definition -> use (data-flow) |
| 11 | EXTERNAL_CALL | CFG call site -> external contract target |

`NUM_EDGE_TYPES=12` is locked. `edge_attr` is 1-D int64 of shape `[E]`.

---

## Output Classes

10 vulnerability classes (fixed, append-only -- changing order breaks all checkpoints):

| Index | Class | Description |
|-------|-------|-------------|
| 0 | CallToUnknown | Low-level calls to unknown addresses |
| 1 | DenialOfService | Gas griefing / unbounded loops |
| 2 | ExternalBug | External contract dependency bugs |
| 3 | GasException | Unchecked send/transfer failures |
| 4 | IntegerUO | Integer overflow/underflow (Solidity <0.8) |
| 5 | MishandledException | Unchecked return values from external calls |
| 6 | Reentrancy | Cross-function/cross-contract reentrancy |
| 7 | Timestamp | Block timestamp manipulation |
| 8 | TransactionOrderDependence | Front-running / TOD |
| 9 | UnusedReturn | Return value of internal functions ignored |

`CLASS_NAMES` list is defined in `ml/src/training/trainer.py:105-116`.

---

## Training

### Launch

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. nohup \
    python ml/scripts/train.py \
    --run-name run12-$(date +%Y%m%d) \
    --experiment-name sentinel-retrain-v2 \
    --epochs 100 \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    > ml/logs/run12-$(date +%Y%m%d).log 2>&1 &
```

### TrainConfig Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | `8` | Fits 8 GB GPU with MAX_WINDOWS=4 |
| `gradient_accumulation_steps` | `8` | effective batch = 64 |
| `gnn_hidden_dim` | `256` | |
| `gnn_layers` | `8` | 2+3+3 three-phase |
| `gnn_heads` | `8` | Phase 1; Phase 2 uses 4 |
| `lora_r` | `16` | LoRA rank on Q+V |
| `lora_alpha` | `32` | effective scale = alpha/r = 2.0 |
| `epochs` | `100` | |
| `lr` | `2e-4` | Base learning rate |
| `loss_fn` | `"asl"` | AsymmetricLoss(gamma_neg=2.0, gamma_pos=1.0, clip=0.01) |
| `eval_threshold` | `0.35` | Training-time eval (lower than inference to reduce noise) |
| `early_stop_patience` | `30` | |
| `aux_loss_weight` | `0.3` | Main auxiliary loss weight |
| `aux_phase2_loss_weight` | `0.20` | CEI Phase 2 aux head weight |
| `jk_entropy_reg_lambda` | `0.005` | JK entropy regularizer |
| `gnn_prefix_k` | `0` | 0 = disabled; 48 for prefix injection |
| `gnn_prefix_warmup_epochs` | `15` | Epochs before prefix activates |
| `gnn_lr_multiplier` | `2.5` | GNN effective LR = lr * 2.5 |
| `lora_lr_multiplier` | `0.3` | LoRA effective LR = lr * 0.3 |
| `fusion_lr_multiplier` | `0.5` | Fusion effective LR = lr * 0.5 |
| `dos_loss_weight` | `0.5` | 50% DoS gradient |
| `drop_complexity_feature` | `True` | Zeros feat[5] at GNN input |
| `use_compile` | `True` | Submodule-level; transformer excluded |
| `use_amp` | `True` | BF16; no GradScaler |
| `num_workers` | `4` | DataLoader workers |
| `use_weighted_sampler` | `"timestamp-size"` | Oversamples large Timestamp+ contracts |

### Key Training Features

- **AMP (BF16):** Automatic mixed precision with `torch.amp.autocast`
- **Gradient accumulation:** `gradient_accumulation_steps=8` for effective batch of 64
- **Early stopping:** 30-epoch patience on `f1_macro_tuned`
- **Per-class label smoothing:** Calibrated to noise rates (e.g., Reentrancy=0.14, Timestamp=0.05)
- **Weighted sampling:** `timestamp-size` mode oversamples large Timestamp+ contracts 4x
- **NaN guard (A38):** Checks loss finiteness BEFORE backward to prevent Adam state corruption
- **Post-clip guard (A38):** Skips optimizer.step on non-finite gradients
- **GNN collapse detection:** Warns when GNN gradient share < 10% for 3 consecutive intervals

### Structured Logging (Phase 4.6)

Three JSONL streams in `ml/logs/<run_name>/`:
- `step_metrics.jsonl` -- per-step loss, lr, grad_norm, vram
- `epoch_summary.jsonl` -- 37-field epoch summary (Spec section 8)
- `alerts.jsonl` -- WARN/KILL alerts with timestamps

Alert tiers: KILL (raise TrainingAbortError), WARN_SKIP (skip batch), WARN (log and continue).

### Post-Training Workflow

```bash
# Threshold tuning (per-class, 19 candidates each)
python ml/scripts/tune_threshold.py --checkpoint ml/checkpoints/<run>_best.pt

# Temperature scaling calibration
python ml/scripts/calibrate_temperature.py --checkpoint ml/checkpoints/<run>_best.pt

# Promote to MLflow staging
python ml/scripts/promote_model.py --checkpoint ml/checkpoints/<run>_best.pt
```

---

## Inference

### API Server

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. uvicorn ml.src.inference.api:app --port 8001
```

**Configuration:** `ml/mlops_config.json` (env vars override):
```json
{
  "checkpoint": "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt",
  "thresholds": "ml/checkpoints/..._thresholds.json",
  "num_classes": 10,
  "drift_baseline": "ml/data/drift_baseline_run12.json",
  "drift_check_interval": 50,
  "predict_timeout": 60
}
```

**Endpoints:**
- `POST /predict` -- Score a Solidity contract. Returns three-tier output (confirmed/suspicious/safe), full probability vector, per-class thresholds, graph stats.
- `POST /hotspots` -- GNN attention hotspots + ML prediction in one round-trip. Returns top-20 functions by GNN embedding norm.
- `GET /health` -- Liveness check with model metadata.
- `GET /metrics` -- Prometheus metrics (auto-instrumented).

**Request validation:** Rejects source_code without `pragma` or `contract` keywords. Rejects files > 1 MB.

### Three-Tier Suspicion Output

| Tier | Threshold | Action |
|------|-----------|--------|
| CONFIRMED | `prob >= per_class_threshold` (default 0.55) | Hard-flag; ZK proof candidate |
| SUSPICIOUS | `0.25 <= prob < per_class_threshold` | Send to RAG + static analysis |
| NOTEWORTHY | `prob < 0.25` | Included in probabilities dict only |

### Predictor

**File:** `ml/src/inference/predictor.py`

- Loads checkpoint with architecture-aware model construction
- Per-class thresholds from `{checkpoint_stem}_thresholds.json`
- Warmup forward pass at startup (3-node graph with FUNCTION node to exercise prefix path)
- `_current_epoch = 9999` ensures prefix is always active at inference
- Handles legacy binary checkpoints (num_classes=1)
- Strips `._orig_mod.` infix from torch.compile state dicts

### Drift Detection

**File:** `ml/src/inference/drift_detector.py`

Kolmogorov-Smirnov (KS) test comparing rolling request stats against baseline:
- Stats monitored: `num_nodes`, `num_edges`, `confirmed_count`, `suspicious_count`
- Baseline: `ml/data/drift_baseline_run12.json` (real synthetic warmup data)
- `MIN_SAMPLES_FOR_KS = 30`, `KS_ALPHA = 0.05`
- Prometheus counter: `sentinel_drift_alerts_total{stat=<name>}`

### Inference Cache

**File:** `ml/src/inference/cache.py`

Content-addressed disk cache keyed on `{content_md5}_{FEATURE_SCHEMA_VERSION}`:
- Default TTL: 86400s (24h)
- Atomic writes via tmp file + rename
- Schema validation on load (catches stale cached graphs)
- Default dir: `~/.cache/sentinel/preprocess/`

---

## Repository Layout

```
ml/
|-- README.md                          this file
|-- CLAUDE.md                          Claude Code instructions for this module
|-- DIAGRAMS.md                        Mermaid architecture diagrams
|-- pyproject.toml                     Poetry config, dependencies
|-- mlops_config.json                  Active MLOps config (checkpoint, drift, timeout)
|-- src/
|   |-- models/
|   |   |-- sentinel_model.py          SentinelModel v8.1 (four-eye + GNN prefix)
|   |   |-- gnn_encoder.py             8-layer three-phase GAT (2+3+3), type_embedding
|   |   |-- transformer_encoder.py     GraphCodeBERT + LoRA + Flash Attention 2 + prefix
|   |   |-- fusion_layer.py            CrossAttentionFusion, compile-safe, token_norm
|   |   |-- README.md
|   |-- training/
|   |   |-- trainer.py                 TrainConfig, train(), evaluate(), BF16, gradient accum
|   |   |-- training_logger.py         StructuredLogger (3 streams), MLflow integration
|   |   |-- losses.py                  AsymmetricLoss (ASL)
|   |   |-- focalloss.py               FocalLoss, MultiLabelFocalLoss
|   |   |-- README.md
|   |-- inference/
|   |   |-- api.py                     FastAPI app (:8001), /predict, /hotspots, /health
|   |   |-- predictor.py               Checkpoint loading, warmup, hotspot extraction
|   |   |-- drift_detector.py          KS-based drift detection with Prometheus
|   |   |-- preprocess.py              ContractPreprocessor (Slither + CodeBERT)
|   |   |-- cache.py                   InferenceCache (content-addressed, TTL, atomic)
|   |   |-- README.md
|   |-- preprocessing/
|   |   |-- graph_schema.py            Thin re-export shim from sentinel_data
|   |   |-- graph_extractor.py         Slither -> PyG graph extraction
|   |   |-- README.md
|   |-- datasets/
|   |   |-- sentinel_dataset.py        SentinelDataset (v2 export artifacts)
|   |   |-- collate.py                 sentinel_collate_fn
|   |   |-- README.md
|   |-- data_extraction/
|   |   |-- windowed_tokenizer.py      Windowed tokenization (graphcodebert-base)
|   |   |-- README.md
|   |-- utils/
|   |   |-- hash_utils.py              MD5 contract identification
|   |   |-- README.md
|   |-- data/
|   |   |-- graphs/                    .pt graph files (v9, 12-dim)
|   |   |-- README.md
|-- scripts/
|   |-- train.py                       Training entry point (CLI)
|   |-- tune_threshold.py              Per-class threshold tuning
|   |-- calibrate_temperature.py       Post-training temperature scaling
|   |-- promote_model.py               MLflow staging promotion
|   |-- build_warmup_baseline.py       Drift baseline from synthetic warmup
|   |-- set_active_checkpoint.py       Atomic mlops_config.json update
|   |-- smoke/                         18 smoke test scripts
|   |-- eval/                          6 evaluation scripts
|   |-- audit/                         Contamination analysis, OOD analysis
|   |-- interpretability/              34 experiment scripts
|   |-- util/                          Shell scripts for eval monitoring
|   |-- README.md
|-- tests/
|   |-- test_api.py                    API endpoint tests (18 tests)
|   |-- test_model.py                  Forward pass shapes, aux output
|   |-- test_trainer.py                TrainConfig, loss, gradient flow
|   |-- test_drift_detector.py         Drift detector unit tests
|   |-- test_sentinel_dataset.py       Dataset loading, collation
|   |-- test_preprocessing.py          Schema, feature builders
|   |-- test_cache.py                  Cache key, TTL, atomic write
|   |-- test_fusion_layer.py           CrossAttentionFusion shapes
|   |-- test_gnn_encoder.py            GNNEncoder shapes, phase routing
|   |-- test_predictor.py              Checkpoint loading, warmup
|   |-- test_framework_gates.py        Testing framework gate tests
|   |-- test_api_config.py             API config loading
|   |-- test_promote_model.py          Model promotion tests
|   |-- conftest.py                    Shared fixtures
|   |-- README.md
|-- testing_specs/
|   |-- framework/                     Testing framework (CLI, gates, reporters)
|   |-- 00_rules.md                    Spec rules
|   |-- A-L spec files                 Validation procedures (A-L)
|   |-- README.md
|-- calibration/
|   |-- temperatures_run12.json        Run 12 temperature scaling params
|   |-- temperatures_run12_stats.json  Run 12 calibration statistics
|   |-- README.md
|-- deploy/
|   |-- Dockerfile.inference           Multi-layer Docker image
|   |-- docker-compose.yml             Inference + Prometheus stack
|   |-- prometheus.yml                 Scrape config (15s interval)
|   |-- .env.example                   Environment variables
|   |-- README.md
|-- checkpoints/                       Model checkpoints (not committed)
|-- data/                              Training data (not committed)
|-- logs/                              Training logs (not committed)
|-- interpretability_results/          Experiment outputs
|-- training_snapshots/                Training state snapshots
|-- audit_docs/                        Audit findings and plans
```

---

## Key Invariants

| Invariant | Value | Break condition |
|-----------|-------|----------------|
| `NODE_FEATURE_DIM` | **12** | Rebuild all graph `.pt` files + retrain |
| `FEATURE_SCHEMA_VERSION` | **`"v9"`** | Bump on any schema change; invalidates inference cache |
| `NUM_CLASSES` | **10** | Locked -- ZKML circuit and CLASS_NAMES order both depend on this |
| `NUM_EDGE_TYPES` | **12** | GNNEncoder Embedding(12,64) + retrain |
| `fusion_output_dim` | **128** | ZKML proxy MLP depends on this; never change |
| `gnn_num_layers` | **8** | 2+3+3 phases (IMP-G3 added conv4c) |
| Classifier input | **512** (4 x 128) | Four-eye architecture |
| Classifier hidden | **256** | Linear(512,256) -> Linear(256,10) |
| `add_self_loops` in Phase 2 | `False` | Self-loops cancel directional CF signal |
| `add_self_loops` in Phase 3 | `False` | Bidirectional CONTAINS requires directional edges |
| JK tensors in GNNEncoder | Collected **without** `.detach()` | Zero gradients to JK attention weights |
| Backbone model | `microsoft/graphcodebert-base` | Token files + retrain if changed |
| Prefix position IDs | prefix=1, code=3.. | RoBERTa uses 0=BOS, 1=padding, 2=EOS slots |
| `token_norm` in CrossAttentionFusion | Always active | BUG-C2 fix; prevents CodeBERT norm dominance |
| `need_weights=False` in MHA | Always set | Fix #26; saves ~12.6 MB VRAM per forward pass |
| `type_embedding` in GNNEncoder | Embedding(14, 16) | Concatenated with features -> [N, 28] input |
| Phase 2 attention heads | **4** | IMP-R7-1; was 1 before Run 7 |
| `weights_only` for checkpoints | `False` | LoRA PEFT objects not safe-tensors serialisable |
| `TRANSFORMERS_OFFLINE` | Set at **shell level** | HuggingFace reads this at import time |
