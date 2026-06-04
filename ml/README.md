# ml — SENTINEL Machine Learning Core

Dual-path smart contract vulnerability detector. An 8-layer **Graph Attention Network** encodes AST/CFG structure with typed edge relations across three phases; a **LoRA-adapted GraphCodeBERT** encodes source text across sliding windows with optional GNN-prefix injection. A **four-eye CrossAttentionFusion** (GNN eye, Transformer eye, Fused eye, CFG eye) produces per-class probabilities across **10 vulnerability classes**.

**Current architecture: v8.1** — `FEATURE_SCHEMA_VERSION = "v8"`, `NODE_FEATURE_DIM = 11`, backbone: `microsoft/graphcodebert-base`, architecture tag: `"four_eye_v8"`

---

## Table of Contents

- [Setup](#setup)
- [System Overview](#system-overview)
- [Data Pipeline](#data-pipeline)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [GNN Encoder (8-layer, three-phase GAT)](#gnn-encoder)
  - [Transformer Encoder (GraphCodeBERT + LoRA + GNN Prefix)](#transformer-encoder)
  - [GNN Prefix Injection](#gnn-prefix-injection)
  - [CrossAttentionFusion](#crossattentionfusion)
  - [Four-Eye Classifier](#four-eye-classifier)
  - [Node Feature Vector (v8 Schema, 11-dim)](#node-feature-vector)
  - [Edge Types](#edge-types)
- [Output Classes](#output-classes)
- [Training](#training)
- [Inference](#inference)
- [Interpretability Suite](#interpretability-suite)
- [Tests](#tests)
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
| `torch` | `2.5.1+cu124` | Training, inference |
| `torch-geometric` | `2.7.0` | GNN layers, graph batching |
| `torch-scatter` | `2.1.2+pt25cu124` | GPU scatter kernels for pooling |
| `transformers` | `4.x` | GraphCodeBERT base model |
| `peft` | `0.x` | LoRA adapters on GraphCodeBERT |
| `slither-analyzer` | `>=0.9.3` | Graph extraction from Solidity |

---

## System Overview

```
Input: Solidity contract (.sol)
        │
        ├─ graph_extractor.py ──► .pt graph (NODE_FEATURE_DIM=11, 11 edge types)
        │
        └─ retokenize_windowed.py ► .pt tokens ([4, 512] windows, stride=256)
                │
        DualPathDataset (cached_dataset_v10.pkl)
                │
        ┌───────┴────────────────────────────────────────────┐
        │                                                    │
   GNNEncoder                                    TransformerEncoder
   8-layer GAT, 3 phases (2+3+3)              GraphCodeBERT + LoRA r=16
   JK attention, hidden_dim=256                  12 layers, Q+V adapters
   type_embedding(13,16), input_proj skip        frozen base, BF16
   IMP-G1 edge subsets, IMP-G3 bidir CONTAINS   Flash Attention 2
        │                                        │
        │  [B, K, 256] prefix nodes (K=48)            │
        │  ──gnn_to_bert_proj(256→768)──►              │
        │  prepended as inputs_embeds prefix            │
        │  (suppressed during warmup epochs 0–14)       │
        │  IMP-M3: actual node count masking            │
        │                                               │
        └─────────── CrossAttentionFusion ──────────────┘
                     bidirectional cross-attention
                     node↔token, attn_dim=256
                     token_norm LayerNorm (BUG-C2)
                     need_weights=False (Fix #26)
                     output_dim=128 LOCKED
                │
         Four-Eye Classifier
         GNN eye [B,128] + TF eye [B,128] + Fused eye [B,128] + CFG eye [B,128]
         → [B,512] → Linear(512,256) → Linear(256,10)
                │
         [B, 10] logits → sigmoid → thresholded predictions
```

---

## Data Pipeline

Run scripts in this order for a full re-extraction:

```bash
# 1. Re-extract graphs from BCCC corpus
poetry run python ml/scripts/reextract_graphs.py

# 2. Tokenize all contracts (windowed, [4,512], stride=256)
poetry run python ml/scripts/retokenize_windowed.py

# 3. Build label CSV
poetry run python ml/scripts/build_multilabel_index.py

# 4. Generate splits
poetry run python ml/scripts/create_splits.py

# 5. Build paired cache
poetry run python ml/scripts/create_cache.py

# 6. Validate dataset integrity
poetry run python ml/scripts/validate_graph_dataset.py
```

**Current data state (v10):**

| File | Count / Size | Contents |
|------|-------------|----------|
| `ml/data/graphs/` | ~41K .pt | v8 graphs, 11-dim, FEATURE_SCHEMA_VERSION="v8" |
| `ml/data/tokens_windowed/` | ~41K .pt | windowed tokens [4,512], stride=256 |
| `ml/data/cached_dataset_v10.pkl` | paired | (graph, tokens) pairs |
| `ml/data/processed/multilabel_index.csv` | 68,523 rows | 10 classes |
| `ml/data/splits/v10_deduped/` | 3 .npy | train / val / test indices |

**Note on retokenization for K=48:** With stride=256 and code_budget=464 (K=48), overlap per window = 464−256 = 208 tokens. Since stride < code_budget there are no gaps — retokenization is not required unless K > 256.

---

## Dataset

`DualPathDataset` (`ml/src/datasets/dual_path_dataset.py`) loads pairs from the pre-built `.pkl` cache. `dual_path_collate_fn` batches graph data via PyG `Batch.from_data_list` and stacks token tensors.

**Label distribution (training targets):**

| Class | Total | Train |
|-------|-------|-------|
| IntegerUO | 13,797 | 9,613 |
| GasException | 4,957 | ~3,500 |
| Reentrancy | 3,886 | 2,775 |
| ExternalBug | 3,009 | ~2,100 |
| TOD | 3,028 | ~2,100 |
| MishandledException | 2,442 | ~1,700 |
| CallToUnknown | 2,873 | ~2,000 |
| UnusedReturn | 1,051 | ~730 |
| Timestamp | 538 | ~375 |
| DenialOfService | 372 | ~260 |

59.3% of rows are all-zero. Handled by `WeightedRandomSampler` (3× weight for any-vuln rows) and `AsymmetricLoss` (γ⁻=2.0).

---

## Model Architecture

### GNN Encoder

**File:** `ml/src/models/gnn_encoder.py`

8-layer Graph Attention Network, three phases (2+3+3), Jumping Knowledge (JK) attention aggregation:

```
Type Embedding (BUG-R7-2)
  Embedding(13, 16) → [N, 16]
  Concatenated with raw features → [N, 27]

Phase 1 — Structural + CONTAINS (layers 1+2)
  GAT over edge types 0–5, 8 heads, add_self_loops=True
  IMP-G2: input_proj skip connection (Linear(27,256)) added before relu in Layer 1
         Prevents raw feature loss when GAT attention weights start near-uniform
  LayerNorm after phase

Phase 2 — CFG + ICFG directed (layers 3+4+5)
  IMP-G1: each layer processes a DISTINCT edge subset (vs same cfg_mask before)
  conv3:  CONTROL_FLOW(6) only — intra-function execution ordering
  conv3b: CALL_ENTRY(8) + RETURN_TO(9) only — cross-function call structure
  conv3c: CF(6)+CALL_ENTRY(8)+RETURN_TO(9)+DEF_USE(10) joint — integration layer
  add_self_loops=False (CRITICAL — self-loops cancel directional signal)
  heads=4 (IMP-R7-1), concat=False
  LayerNorm after phase

Phase 3 — Bidirectional CONTAINS (layers 6+7+8)
  conv4:  REVERSE_CONTAINS up — CFG→FUNCTION (Phase 2 signal rises)
  conv4b: REVERSE_CONTAINS up — second hop (multi-function patterns)
  conv4c: CONTAINS down (IMP-G3) — FUNCTION→CFG, distributes enriched
          FUNCTION context back to CFG children. All nodes carry Phase 3 depth after this.
  1 head
  LayerNorm after phase

JK attention aggregation over all 8 layer outputs → hidden_dim=256
Edge type embedding: Embedding(11, 64) concatenated per message
```

**Node types** are defined in `graph_schema.py` (13 types, IDs 0–12). Always use named constants, never raw integers.

### Transformer Encoder

**File:** `ml/src/models/transformer_encoder.py`

`microsoft/graphcodebert-base` (124M params) + LoRA:
- Base model frozen; LoRA r=16, α=32 on Q+V of all 12 layers
- Flash Attention 2 support (falls back to SDPA if unavailable)
- Input: `[B, 4, 512]` — 4 sliding windows of 512 tokens each (stride=256)
- Each window processed independently via `WindowAttentionPooler` → `[B, 768]`
- **GNN prefix path:** when `gnn_prefix_nodes` is not None, uses `inputs_embeds` instead of `input_ids`; prefix occupies positions 0..K−1, code occupies positions K..K+code_budget−1
- `WindowAttentionPooler` CLS extraction: `i * window_size + prefix_k` (accounts for prefix offset)
- BF16 precision; not compiled (HuggingFace control flow isolates cleanly)

```python
# Signature
def forward(self, input_ids, attention_mask, gnn_prefix_nodes=None, gnn_prefix_counts=None, output_attentions=False):
    # gnn_prefix_nodes: [B, K, 768] or None
    # gnn_prefix_counts: [B] real node counts (IMP-M3)
    # output_attentions: returns prefix_attn_mean when True (IMP-M2)
```

### GNN Prefix Injection

**Files:** `ml/src/models/sentinel_model.py`, `ml/src/models/transformer_encoder.py`

Declaration-level GNN node embeddings are projected into BERT space and prepended as soft prefix tokens:

```
select_prefix_nodes()
  Priority: CONSTRUCTOR > FALLBACK > RECEIVE > MODIFIER > FUNCTION
  Secondary sort: FUNCTION nodes by external_call_count descending (IMP-M1)
  Selects top-K=48 declaration nodes per contract
  Audit (P95 decl count=47): K=48 covers 95.5% of contracts without truncation

gnn_to_bert_proj: Linear(256, 768) — projects GNN hidden_dim to BERT embedding_dim
prefix_type_embedding: Embedding(5, 768) — type-specific bias per STRUCTURAL_PREFIX_TYPES

Position IDs:
  Prefix tokens: position_id = 1  (RoBERTa padding slot — avoids colliding with 0/2)
  Code tokens:   position_ids = 3..3+code_budget-1

Warmup suppression (epochs 0..gnn_prefix_warmup_epochs-1):
  gnn_prefix_nodes = None passed to TransformerEncoder
  gnn_to_bert_proj receives zero gradient during warmup
  Projection starts from random init at epoch gnn_prefix_warmup_epochs (default 15)

IMP-M3: actual node count masking
  gnn_prefix_counts [B] tracks real (non-padded) nodes per graph
  Zero-padded prefix positions are masked in attention (95.5% of contracts fill all K slots)
  Prevents transformer from attending to meaningless zero vectors
```

**Inference:** `predictor.py` sets `model._current_epoch = 9999` after load so the prefix is always active regardless of the trained warmup value.

### CrossAttentionFusion

**File:** `ml/src/models/fusion_layer.py`

Bidirectional cross-attention fusing graph and text modalities:

```
1. Project nodes [N,256] → [N,256]
2. token_norm LayerNorm(768) + project tokens [B,512,768] → [B,512,256] (BUG-C2 fix)
3. _scatter_to_dense → static [B,1024,256]  (compile-safe; max_nodes=1024)
4. Node→Token cross-attention → enriched_nodes [B,1024,256]
5. Token→Node cross-attention → enriched_tokens [B,512,256]
6. Masked mean pooling of real nodes  → [B,256]
7. Masked mean pooling of real tokens → [B,256]
8. Concat [B,512] → Linear + ReLU → [B,128]
```

**output_dim=128 is LOCKED** — the ZKML proxy MLP depends on this shape.

`_scatter_to_dense` replaces PyG's `to_dense_batch` to eliminate the `GuardOnDataDependentSymNode` compile graph break (zero graph breaks confirmed in production).

**Key improvements:**
- BUG-C2: `token_norm` LayerNorm before token projection prevents CodeBERT embeddings (L2 norm ~10-15) from dominating cross-attention dot products
- Fix #26: `need_weights=False` on both MHA calls saves ~12.6 MB VRAM per forward pass by skipping attention weight matrix materialization

### Four-Eye Classifier

**File:** `ml/src/models/sentinel_model.py`

```
GNN eye:    max_pool + mean_pool over FUNCTION+MODIFIER+FALLBACK+RECEIVE+CONSTRUCTOR nodes
            → [B,512] → Linear → [B,128]

TF eye:     pooled token embedding → Linear(768,128) → [B,128]

Fused eye:  CrossAttentionFusion → [B,128]

CFG eye:    max_pool + mean_pool over CFG_NODE types [8-12]
            (raw Phase 2 output, NOT JK-aggregated)
            → [B,512] → Linear → [B,128]

Concat [B,512] → Linear(512,256) → GELU → Linear(256,10) → logits
Aux heads: one Linear(128,10) per eye + aux_phase2 MLP(256→128→10) for auxiliary loss
```

### Node Feature Vector

**v8 schema, 11 dimensions:**

| Dim | Feature | Notes |
|-----|---------|-------|
| [0] | `type_id / 12.0` | NodeType enum value (0–12 → 0.0–1.0) |
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

CFG nodes inherit dims [1,3,4,5,9] from their parent FUNCTION node.

### Edge Types

| Type | Name | Description |
|------|------|-------------|
| 0 | CALLS | function → called function (internal) |
| 1 | READS | function → state variable |
| 2 | WRITES | function → state variable |
| 3 | EMITS | function → event |
| 4 | INHERITS | contract → parent contract |
| 5 | CONTAINS | contract/function → child node |
| 6 | CONTROL_FLOW | CFG block → CFG block |
| 7 | REVERSE_CONTAINS | flip of type 5, generated at runtime only |
| 8 | CALL_ENTRY | call site → function entry CFG block |
| 9 | RETURN_TO | function exit CFG block → call-site continuation |
| 10 | DEF_USE | definition → use (data-flow) |

`NUM_EDGE_TYPES=11` is locked. `edge_attr` is 1-D int64 of shape `[E]`.

---

## Output Classes

10 vulnerability classes (fixed, append-only — changing order breaks all checkpoints):

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

---

## Training

### Launch (v10 training)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. nohup \
    python ml/scripts/train.py \
    --run-name v10-$(date +%Y%m%d) \
    --experiment-name sentinel-v10 \
    --epochs 100 \
    --gradient-accumulation-steps 8 \
    --gnn-prefix-k 48 \
    --gnn-prefix-warmup-epochs 15 \
    --gnn-prefix-proj-lr-mult 5.0 \
    --phase2-edge-types 6 8 9 10 \
    --weighted-sampler positive \
    --cache-path ml/data/cached_dataset_v10.pkl \
    > ml/logs/v10-$(date +%Y%m%d).log 2>&1 &

# Monitor:
tail -f ml/logs/v10-$(date +%Y%m%d).log
```

### Key Training Milestones

| Epoch | Event |
|-------|-------|
| 0–14 | Warmup: `gnn_prefix_nodes=None`; transformer learns code representations without prefix |
| 15 | Prefix fires for first time; expect brief loss spike at ep15–16 |
| 20 | Check: GNN share trend and `prefix_proj_weight_norm` growth |
| 40+ | Expected convergence region based on prior ablation runs |

### TrainConfig Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `batch_size` | `8` | 6.9 / 8.0 GB VRAM |
| `gradient_accumulation_steps` | `8` | effective batch = 64 |
| `gnn_hidden_dim` | `256` | |
| `gnn_layers` | `8` | 2+3+3 three-phase (IMP-G3 added conv4c) |
| `lora_r` | `16` | LoRA rank on Q+V |
| `lora_alpha` | `32` | |
| `epochs` | `100` | |
| `loss_fn` | `"asl"` | AsymmetricLoss(γ⁻=2.0, γ⁺=1.0, clip=0.01) |
| `use_weighted_sampler` | `"positive"` | 3× for any-vuln rows |
| `lora_lr_mult` | `0.3` | LoRA adapter LR multiplier |
| `gnn_lr_mult` | `2.5` | GNN LR multiplier |
| `fusion_lr_multiplier` | `0.5` | Fusion LR multiplier |
| `dos_loss_weight` | `0.5` | 50% DoS gradient (half-weight) |
| `aux_loss_weight` | `0.3` | Main auxiliary loss weight |
| `aux_phase2_loss_weight` | `0.20` | Phase 2 aux head loss weight |
| `jk_entropy_reg_lambda` | `0.005` | JK entropy regularization |
| `gnn_prefix_k` | `48` | Prefix token count (0 = disabled) |
| `gnn_prefix_warmup_epochs` | `15` | Epochs before prefix activates |
| `gnn_prefix_proj_lr_mult` | `5.0` | `gnn_to_bert_proj` LR multiplier |
| `phase2_edge_types` | `[6, 8, 9, 10]` | CF + CALL_ENTRY + RETURN_TO + DEF_USE |
| `use_compile` | `True` | Submodule-level; transformer excluded |
| `use_amp` | `True` | BF16; no GradScaler |

### Per-epoch Prefix Logging

The trainer emits to both logger and MLflow each epoch when `gnn_prefix_k > 0`:

```
GNN prefix K=48: WARMUP (starts ep15)          ← epochs 0–14
GNN prefix K=48: ACTIVE                         ← epochs 15+
gnn_to_bert_proj weight norm: 16.0000           ← constant during warmup (zero gradient)
gnn_to_bert_proj weight norm: 16.xxxx           ← drifts after ep15 (projection learning)
```

MLflow metrics: `prefix_active` (0/1), `prefix_proj_weight_norm`.

### Post-Training Workflow

```bash
# Threshold tuning
poetry run python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/v10-<date>_best.pt

# Behavioral gate (≥80% detection, ≥80% specificity required)
poetry run python ml/scripts/manual_test.py \
    --checkpoint ml/checkpoints/v10-<date>_best.pt

# Promote to production
poetry run python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/v10-<date>_best.pt
```

---

## Inference

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. uvicorn ml.src.inference.api:app --port 8001
```

**Endpoints:**
- `POST /predict` — `{"contract_source": "..."}` → per-class probabilities + detections
- `GET /health` — liveness
- `GET /metrics` — Prometheus

`predictor.py` reads `gnn_prefix_k` and `gnn_prefix_warmup_epochs` from the checkpoint's saved config and sets `model._current_epoch = 9999` so the prefix is always active at inference time. Architecture detection via `_ARCH_TO_FUSION_DIM`: `"four_eye_v8"` → fusion_dim=128, node_dim=11.

---

## Interpretability Suite

**Directory:** `ml/scripts/interpretability/`

21 experiment scripts for model behavior analysis. Run against any checkpoint.

| Script | Experiment | What it measures |
|--------|-----------|------------------|
| `exp_l2_edge_ablation.py` | Edge type ablation | Impact of removing each edge type on per-class F1 |
| `exp_a4_aux_contribution.py` | Auxiliary heads | Contribution of each eye to classification |
| `exp_l4_gradient_saliency.py` | Gradient saliency | Per-feature importance via gradient × input |
| `exp_l8_permutation_importance.py` | Permutation importance | Per-feature importance via shuffled prediction delta |
| `exp_e1_receptive_field.py` | GNN receptive field | Effective hop distance for each Phase 2 layer |
| `exp_l3_attention_visualization.py` | Cross-attention | Token-to-node attention weight patterns |
| `exp_b1_phase2_gradient_norm.py` | Phase 2 gradients | Gradient magnitude through conv3/3b/3c |
| `exp_l5_probing_classifiers.py` | Probing classifiers | Linear probes on intermediate GNN embeddings |
| `exp_l10_training_ablation.py` | Training ablation | Impact of individual training components |

See `ml/scripts/interpretability/README.md` for the full experiment catalog and interpretation guide.

---

## Tests

```bash
cd ml && poetry run pytest tests/ -v
```

| File | Coverage |
|------|---------|
| `test_preprocessing.py` | Schema (NODE_FEATURE_DIM=11, 13 types), feature builders, CFG inheritance |
| `test_model.py` | Forward pass shapes, aux output, [B,10] output, prefix path |
| `test_training.py` | TrainConfig, ASL loss, gradient flow, prefix warmup suppression |
| `test_cache.py` | Cache key, schema-version invalidation, atomic write |
| `test_dataset.py` | DualPathDataset loading, collate, batch shapes |

---

## Repository Layout

```
ml/
├── README.md                       ← this file
├── DIAGRAMS.md                     ← Mermaid architecture diagrams
├── pyproject.toml
├── scripts/
│   ├── train.py                    ← training entry point
│   ├── reextract_graphs.py         ← graph extraction
│   ├── retokenize_windowed.py      ← tokenization
│   ├── build_multilabel_index.py   ← label CSV builder
│   ├── create_cache.py             ← paired cache builder
│   ├── create_splits.py            ← stratified splits
│   ├── validate_graph_dataset.py   ← dataset integrity check
│   ├── tune_threshold.py           ← per-class threshold tuning
│   ├── calibrate_temperature.py    ← post-training temperature scaling
│   ├── promote_model.py            ← MLflow staging promotion
│   ├── vram_gate_test.py           ← VRAM usage validation
│   ├── compile_smoke_test.py       ← torch.compile smoke test
│   ├── archive_v8_data.py          ← v8 data archival
│   ├── archive/                    ← completed one-off scripts
│   ├── interpretability/           ← 21 experiment scripts
│   └── test_contracts/             ← behavioral smoke test contracts
├── src/
│   ├── models/
│   │   ├── sentinel_model.py       ← SentinelModel v8.1, four-eye + GNN prefix
│   │   ├── gnn_encoder.py          ← 8-layer three-phase GAT (2+3+3), type_embedding
│   │   ├── transformer_encoder.py  ← GraphCodeBERT + LoRA + Flash Attention 2 + prefix
│   │   └── fusion_layer.py         ← CrossAttentionFusion, compile-safe, token_norm
│   ├── preprocessing/
│   │   ├── graph_schema.py         ← NODE_FEATURE_DIM=11, FEATURE_SCHEMA_VERSION="v8"
│   │   └── graph_extractor.py      ← Slither → v8 graph .pt files
│   ├── datasets/
│   │   └── dual_path_dataset.py    ← DualPathDataset + collate
│   ├── training/
│   │   ├── trainer.py              ← TrainConfig, train(), BF16, submodule compile
│   │   ├── training_logger.py      ← StructuredLogger, MLflow integration
│   │   ├── losses.py               ← AsymmetricLoss
│   │   └── focalloss.py            ← FocalLoss (alternative)
│   ├── inference/
│   │   ├── api.py                  ← FastAPI :8001
│   │   ├── predictor.py            ← reads gnn_prefix_k; _current_epoch=9999
│   │   ├── cache.py                ← InferenceCache (content-addressed, TTL)
│   │   ├── drift_detector.py       ← KS test drift detection
│   │   └── preprocess.py           ← ContractPreprocessor
│   ├── data/                       ← data loading helpers
│   ├── tools/                      ← utility tools
│   ├── utils/                      ← shared utilities
│   └── validation/                 ← validation helpers
├── data/                           ← NOT committed (.gitignore)
│   ├── graphs/                     ← .pt graph files (v8, 11-dim)
│   ├── tokens_windowed/            ← .pt token files ([4,512], stride=256)
│   ├── processed/                  ← CSV label files
│   ├── splits/v10_deduped/         ← train/val/test .npy indices
│   ├── augmented/                  ← DoS and CEI augmented .sol files
│   └── cached_dataset_v10.pkl      ← paired cache
├── checkpoints/                    ← NOT committed
└── logs/                           ← NOT committed
```

---

## Key Invariants

| Invariant | Value | Break condition |
|-----------|-------|----------------|
| `NODE_FEATURE_DIM` | **11** | Rebuild all graph `.pt` files + retrain |
| `FEATURE_SCHEMA_VERSION` | **`"v8"`** | Bump on any schema change; invalidates inference cache |
| `NUM_CLASSES` | **10** | Locked — ZKML circuit and CLASS_NAMES order both depend on this |
| `NUM_EDGE_TYPES` | **11** | GNNEncoder Embedding(11,64) + retrain |
| `fusion_output_dim` | **128** | ZKML proxy MLP (M2) depends on this; never change |
| `gnn_num_layers` | **8** | 2+3+3 phases (IMP-G3 added conv4c for downward CONTAINS) |
| Classifier input | **512** (4 × 128) | Four-eye architecture; changing eye count breaks this |
| Classifier hidden | **256** | Linear(512,256) → Linear(256,10) |
| `gnn_to_bert_proj` at inference | always active | `predictor.py` sets `_current_epoch=9999` |
| `weights_only` for graph `.pt` | `False` | PyG 2.7 metadata not safe-tensors serialisable |
| `weights_only` for checkpoint `.pt` | `False` | LoRA PEFT objects not safe-tensors serialisable |
| Checkpoint state dict keys | Strip `._orig_mod.` infix | `torch.compile` adds this prefix; strip at save time |
| Checkpoint dtype | BF16 → call `.float()` | For diagnostic inference outside training loop |
| `TRANSFORMERS_OFFLINE` | Set at **shell level** | HuggingFace reads this at `transformers` import time |
| `add_self_loops` in Phase 2 | `False` | Self-loops cancel directional CF signal |
| `add_self_loops` in Phase 3 | `False` | Bidirectional CONTAINS requires directional edges |
| JK tensors in GNNEncoder | Collected **without** `.detach()` | Zero gradients to JK attention weights |
| Backbone model | `microsoft/graphcodebert-base` | Token files + retrain if changed |
| Prefix position IDs | prefix=1, code=3..466 | RoBERTa uses 0=BOS, 1=padding, 2=EOS slots |
| `token_norm` in CrossAttentionFusion | Always active | BUG-C2 fix; prevents CodeBERT norm dominance |
| `need_weights=False` in MHA | Always set | Fix #26; saves ~12.6 MB VRAM per forward pass |
| `type_embedding` in GNNEncoder | Embedding(13, 16) | Concatenated with features → [N, 27] input |
| Phase 2 attention heads | **4** | IMP-R7-1; was 1 before Run 7 |
