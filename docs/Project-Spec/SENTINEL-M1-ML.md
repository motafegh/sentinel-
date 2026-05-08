
# SENTINEL — Module 1: ML Core

Load for: ML inference / prediction work (preprocess.py, predictor.py, api.py) and ML training / retrain (trainer.py, focalloss.py, datasets).
Always load alongside: **SENTINEL-CONSTRAINTS.md**

---

## Tech Stack

| Tool | Role |
|------|------|
| PyTorch ^2.5.0 | Model training and inference |
| PyTorch Geometric ^2.6.0 | GNN layers, PyG Data, Batch, DataLoader |
| HuggingFace Transformers ^4.45.0 | CodeBERT tokenizer and model |
| peft ≥0.13.0,<0.16.0 | LoRA fine-tuning of CodeBERT |
| scikit-learn ^1.4 | f1_score, hamming_loss, train_test_split |
| MLflow ^2.17.0 | Experiment tracking |
| DVC ^3.49 | Dataset and model versioning |
| Slither 0.10.x | Solidity AST extraction |
| FastAPI ^0.115.0 | Inference HTTP API |
| loguru ^0.7 | Structured logging |
| prometheus-fastapi-instrumentator | /metrics endpoint + custom gauges (T2-A) |
| scipy ^1.13.0 | KS drift detection (T2-B) |
| httpx ^0.27 | Async HTTP (declared in pyproject.toml as of 2026-05-03) |
| uvicorn[standard] | ASGI server (declared in pyproject.toml as of 2026-05-03) |

---

## Model Architecture

```

Input: Solidity source string
│
├── ContractPreprocessor
│     ├── Slither AST extraction  (InferenceCache check first — T1-A)
│     │     Node features [N, 8]:
│     │       [0] type_id      STATE_VAR=0 FUNCTION=1 MODIFIER=2
│     │                        EVENT=3 FALLBACK=4 RECEIVE=5
│     │                        CONSTRUCTOR=6 CONTRACT=7
│     │       [1] visibility   public/external=0 internal=1 private=2
│     │       [2] pure         0/1
│     │       [3] view         0/1
│     │       [4] payable      0/1
│     │       [5] reentrant    0/1
│     │       [6] complexity   float (CFG nodes)
│     │       [7] loc          float (lines of source)
│     │     Edge attributes: graph.edge_attr [E] int64 (1-D)
│     │       CALLS=0, READS=1, WRITES=2, EMITS=3, INHERITS=4
│     │     Node insertion order: CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs
│     │     Output: PyG Data — graph.x [N, 8], graph.edge_index [2, E], graph.edge_attr [E]
│     │
│     └── CodeBERT tokenizer
│           model: microsoft/codebert-base
│           Short contracts (≤512 tokens): single window
│           Long contracts (>512 tokens): sliding-window T1-C
│             stride=256, max_windows=8; returns list[dict]
│           Output: input_ids [1, 512], attention_mask [1, 512]
│
▼
SentinelModel.forward(graphs, input_ids, attention_mask)
│
├── GNNEncoder                              (P0-B: edge-type embeddings, active in v3 checkpoint)
│     edge_emb: nn.Embedding(5, 16)        ← CALLS/READS/WRITES/EMITS/INHERITS
│     Graceful degradation: edge_attr=None → zero-vectors (old .pt files run; lose edge signal)
│     3-layer GATConv:
│       conv1: in=8,  out=8,  heads=8, edge_dim=16, concat=True  → [N, 64]
│       conv2: in=64, out=8,  heads=8, edge_dim=16, concat=True  → [N, 64]
│       conv3: in=64, out=64, heads=1, edge_dim=16, concat=False → [N, 64]
│     Dropout p=0.2 on attention coefficients AND on node activations
│     NO global_mean_pool — pooling DEFERRED to CrossAttentionFusion
│     Returns: (node_embeddings [N, 64], batch [N])
│     N = total nodes across all B contracts in the batch
│
├── TransformerEncoder
│     CodeBERT with LoRA: r=8, lora_alpha=16, lora_dropout=0.1  (configurable via TrainConfig)
│     LoRA injected into query+value of all 12 layers (~295K trainable)
│     Backbone frozen: 124,705,536 params
│     Returns ALL token embeddings: last_hidden_state → [B, 512, 768]
│     NOT just CLS — cross-attention needs all 512 positions
│
├── CrossAttentionFusion (bidirectional)
│     Step 1: Project to common attention space
│       node_proj:  [N, 64]       → [N, 256]
│       token_proj: [B, 512, 768] → [B, 512, 256]
│     Step 2: Pad nodes → [B, max_nodes, 256]
│       node_padding_mask [B, max_nodes]: True = padding position
│     Step 3: Node → Token cross-attention
│       Q=nodes [B,max_n,256], K=V=tokens [B,512,256]
│       Output: enriched_nodes [B, max_nodes, 256]
│     Step 4: Token → Node cross-attention
│       Q=tokens [B,512,256], K=V=nodes [B,max_n,256]
│       key_padding_mask=node_padding_mask (ignores padded positions)
│       Output: enriched_tokens [B, 512, 256]
│     Step 5: Pool AFTER enrichment (not before)
│       pooled_nodes:  masked mean of enriched_nodes → [B, 256]
│       pooled_tokens: mean of enriched_tokens       → [B, 256]
│     Step 6: Concat + project
│       cat([B,256], [B,256]) → [B,512] → Linear → ReLU → Dropout → [B,128]
│     Output: [B, 128]  — output_dim=128 LOCKED (ZKML proxy depends on this)
│
└── Classifier
nn.Linear(128, 10)  — NO Sigmoid — raw logits
Output: [B, 10]

Parameter counts:
GNNEncoder:            ~100K trainable (including edge_emb Embedding(5,16))
TransformerEncoder:    ~295K trainable (LoRA only) + 124,705,536 frozen
CrossAttentionFusion:  ~530K trainable (projections + 2× MHA + output MLP)
Classifier:            128×10 + 10 = 1,290 trainable
Total trainable:       ~925K
Total frozen:          124,705,536

Predictor._score():
logits = model.forward()           → [1, 10]
probs  = torch.sigmoid(logits.float())  → [1, 10]  (FP32 cast before sigmoid — BF16 guard)
Per-class thresholds from multilabel-v3-fresh-60ep_best_thresholds.json
vulnerabilities = [{vulnerability_class, probability} for prob >= threshold[class]]
label = "vulnerable" if vulnerabilities else "safe"

Long-contract path (T1-C — sliding window):
Contracts exceeding 512 CodeBERT tokens are split into overlapping 512-token windows
(stride=256, max_windows=8). GNN graph is built once from the full AST.
predictor.predict_source() automatically routes to the windowed path and aggregates
per-class probabilities via max() across windows. Response includes windows_used: int.

Predictor backward compatibility:
architecture = ckpt["config"].get("architecture", "legacy")
"cross_attention_lora" → fusion_output_dim prefers saved_cfg.get("fusion_output_dim") first,
falls back to _ARCH_TO_FUSION_DIM (128) for legacy checkpoints
"legacy"               → fusion_output_dim = 64  (old binary concat+MLP checkpoints)
Prevents silent Linear(64→10) vs Linear(128→10) shape mismatch.
weights_only=False required for checkpoint load — LoRA state dict contains peft-specific classes.

Predictor startup warmup (_warmup()):
2-node 1-undirected-edge synthetic graph (so GATConv.propagate() actually runs)
dummy_x [2,8], dummy_edge_index [[0,1],[1,0]]
When use_edge_attr=True: dummy_edge_attr = torch.zeros(2, dtype=torch.long)  ← Fix #4
(1-node 0-edge graph would skip GATConv.propagate() entirely — shape bugs invisible)

```

---

## Active Checkpoint

```

File:       ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
Thresholds: ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json
Run:        multilabel-v3-fresh-60ep
Experiment: sentinel-retrain-v3
Completed:  2026-05-05
Batch size: 32
Epochs:     60/60 (no early stop; patience counter=6 at end)
Best epoch: ~52–53
Best raw F1-macro: 0.4715
Tuned F1-macro:    0.5069 ✅ (gate: > 0.4884)
Architecture key: "cross_attention_lora"
edge_attr:  True (P0-B active — this checkpoint WAS trained with edge_attr)

Per-class tuned thresholds (saved in thresholds JSON):
CallToUnknown:              0.70  F1=0.3936
DenialOfService:            0.95  F1=0.4000
ExternalBug:                0.65  F1=0.4345
GasException:               0.55  F1=0.5501
IntegerUO:                  0.50  F1=0.8214
MishandledException:        0.60  F1=0.4916
Reentrancy:                 0.65  F1=0.5362
Timestamp:                  0.75  F1=0.4789
TransactionOrderDependence: 0.60  F1=0.4770
UnusedReturn:               0.70  F1=0.4860

Load pattern:
raw = torch.load(path, weights_only=False)   # weights_only=True breaks LoRA
state_dict = raw["model"] if "model" in raw else raw

Historical checkpoints (superseded):
ml/checkpoints/multilabel_crossattn_best.pt        pre-edge_attr baseline (epoch 34, F1=0.4679)
ml/checkpoints/multilabel_crossattn_best_thresholds.json  Threshold companion for legacy baseline
ml/checkpoints/multilabel_crossattn_v2_best.pt     v2 paused (epoch 37, F1=0.4629, batch mismatch)

```

---

## TrainConfig Key Fields

```

num_classes:          10
architecture:         "cross_attention_lora"   # written into checkpoint config
# module constant ARCHITECTURE = "cross_attention_lora"
batch_size:           32                       # v3 safe on RTX 3070 8GB (was 16 in v1/v2)
grad_clip:            1.0                       # clips trainable params only (not frozen)
lora_r:               8                         # v3 — try 16 for v4 (capacity ceiling suspected)
lora_alpha:           16
lora_dropout:         0.1
lora_target_modules:  ["query", "value"]        # LoRA injection points
use_edge_attr:        True                      # P0-B edge relation embeddings (active in v3)
gnn_edge_emb_dim:     16                        # Embedding(5, 16)
gnn_hidden_dim:       64                        # configurable (P0-C)
gnn_heads:            8
gnn_dropout:          0.2
fusion_output_dim:    128                       # LOCKED — ZKML proxy depends on this
loss_fn:              "bce"                     # v3 used BCE; v4 will use "focal"

loss_fn validation: must be one of {"bce", "focal"} — unknown value raises ValueError immediately
--focal-gamma / --focal-alpha: CLI args wired end-to-end to TrainConfig (2026-05-04)
focal_gamma and focal_alpha: always logged to MLflow regardless of loss_fn

OneCycleLR on resume: uses remaining_epochs = config.epochs - start_epoch + 1
(not config.epochs — otherwise LR never reaches cosine minimum on resumed runs)
guard: skip scheduler state when total_steps mismatches (Fix #10, Fix #25)

Patience sidecar: {checkpoint}.state.json written after every epoch
— restores real patience_counter on resume (Fix #23)
— overrides checkpoint's saved counter (which resets to 0 on each new best)

batch-size guard on full resume: warning emitted when batch size differs (Fix #12)
--resume-reset-optimizer flag: discards optimizer/scheduler while preserving model weights

DataLoader kwargs built conditionally (Fix #5):
prefetch_factor, pin_memory, persistent_workers only when num_workers > 0

```

---

## Start ML API

```

TRANSFORMERS_OFFLINE=1 
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel-v3-fresh-60ep_best.pt 
SENTINEL_THRESHOLDS=ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json 
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001

Optional env vars:
SENTINEL_THRESHOLDS           (default: auto-detected as {checkpoint_stem}_thresholds.json)
SENTINEL_PREDICT_TIMEOUT      (default 60s)
SENTINEL_DRIFT_BASELINE       (default ml/data/drift_baseline.json)
SENTINEL_DRIFT_CHECK_INTERVAL (default 50 requests)

```

Health check response:

```json
{
  "status": "ok",
  "predictor_loaded": true,
  "checkpoint": "ml/checkpoints/multilabel-v3-fresh-60ep_best.pt",
  "architecture": "cross_attention_lora",
  "thresholds_loaded": true
}
```

---

API Contract

```
POST /predict
  Request:  {"source_code": str}   ← field name is source_code, not contract_code
  Response: {
    "label": "vulnerable" | "safe",
    "vulnerabilities": [{"vulnerability_class": str, "probability": float}],
    "thresholds": [float, ...],    ← per-class threshold list (Fix #6, BREAKING CHANGE)
                                     ⚠️ was "threshold": float in older versions
                                     Downstream consumers must use "thresholds" (list)
    "truncated": bool,             ← True if single window reached 512-token limit
    "windows_used": int,           ← 1 for short contracts; >1 when sliding-window triggered
    "num_nodes": int,
    "num_edges": int
  }
  NOTE: NO top-level "confidence" field. Removed in Track 3. Any code using
  ml_result["confidence"] is stale and will KeyError or produce silent wrong routing.
  Max source size: 1 MB (1 * 1024 * 1024 bytes) — enforced before preprocessing

GET /health → see health check response above
GET /metrics → Prometheus text format (T2-A)
  Custom gauges: sentinel_model_loaded, sentinel_gpu_memory_bytes
  Counter: sentinel_drift_alerts_total{stat} (T2-B)
HTTP errors: 400 bad input | 413 GPU OOM or source > 1MB | 422 Pydantic | 503 not loaded | 504 timeout

⚠️ MAX_SOURCE_BYTES imported from ContractPreprocessor.MAX_SOURCE_BYTES (not duplicated in api.py)
```

---

10 Output Classes

Defined in ml/src/training/trainer.py as CLASS_NAMES — single source of truth.

```
CLASS_NAMES = [
    "CallToUnknown",               # index 0
    "DenialOfService",             # index 1
    "ExternalBug",                 # index 2
    "GasException",                # index 3
    "IntegerUO",                   # index 4
    "MishandledException",         # index 5
    "Reentrancy",                  # index 6
    "Timestamp",                   # index 7
    "TransactionOrderDependence",  # index 8
    "UnusedReturn",                # index 9
    # WeakAccessMod excluded: zero .pt files extracted (Slither failures)
    # Append as index 10 if re-extracted; existing indices 0–9 must remain stable
]
NUM_CLASSES = 10
```

---

Dataset Facts

```
Source:           BCCC-SCsVul-2024
Graph .pt files:  68,523  (ml/data/graphs/, MD5 stem, re-extracted 2026-05-03 with edge_attr=[E])
Token .pt files:  68,568  (ml/data/tokens/, MD5 stem, regenerated 2026-05-03)
Splits:           train/val/test_indices.npy (47,966/10,278/10,279)
Label CSV:        ml/data/processed/multilabel_index.csv

Contracts metadata: ml/data/processed/_cache/contracts_metadata.parquet
  (Note: ast_extractor.py CLI default is ml/data/processed/contracts_metadata.parquet
   but actual file is in _cache/ — always pass --input explicitly)

Two hash systems — never mix:
  SHA256 = hash of .sol file content → BCCC filename, CSV col 2
  MD5    = hash of .sol file path    → .pt filename
  Bridge: graph.contract_path inside .pt → Path(...).stem = SHA256

Class distribution (training split pos_weights — for reference; v3 used training-split pos_weights):
  CallToUnknown               pos=8,028   pw=7.53
  DenialOfService             pos=995     pw=67.75  ← only 137 in val split; severely underrepresented
  ExternalBug                 pos=11,069  pw=5.16
  GasException                pos=17,319  pw=2.96
  IntegerUO                   pos=35,724  pw=0.92
  MishandledException         pos=15,148  pw=3.52
  Reentrancy                  pos=16,666  pw=3.09
  Timestamp                   pos=7,304   pw=8.41
  TransactionOrderDependence  pos=11,783  pw=4.82
  UnusedReturn                pos=11,325  pw=5.05
```

---

File Inventory

```
ml/src/models/
  sentinel_model.py          SentinelModel — orchestrates all sub-modules; accepts arch config params (P0-C)
  gnn_encoder.py             GNNEncoder — 3-layer GAT + edge-type embeddings (P0-B); configurable hidden_dim/heads (P0-C)
  transformer_encoder.py     TransformerEncoder — CodeBERT + LoRA; configurable r/alpha/dropout (P0-A)
  fusion_layer.py            CrossAttentionFusion — output_dim=128

ml/src/preprocessing/                 ← single source of truth for graph feature engineering
  __init__.py                re-exports all public symbols
  graph_schema.py            NODE_TYPES, VISIBILITY_MAP, EDGE_TYPES, FEATURE_NAMES,
                             FEATURE_SCHEMA_VERSION="v1", NODE_FEATURE_DIM=8, NUM_EDGE_TYPES=5
                             compile-time assert: len(FEATURE_NAMES) == NODE_FEATURE_DIM
  graph_extractor.py         extract_contract_graph(sol_path, config) → Data
                             GraphExtractionConfig dataclass:
                               multi_contract_policy: "first"|"by_name" (scaffold; "all" not implemented — Move 9)
                               target_contract_name: str | None
                               include_edge_attr: bool = True
                               solc_binary, solc_version, allow_paths
                             edge_attr shape: [E] 1-D int64 (validated at extraction boundary)
                             GraphExtractionError / SolcCompilationError /
                             SlitherParseError / EmptyGraphError exception hierarchy
                             Never returns None — always raises on failure

ml/src/inference/
  preprocess.py              ContractPreprocessor — Slither graph + tokenization; optional InferenceCache;
                             process_source() single-window; process_source_windowed() sliding-window (T1-C)
                             _extract_graph() is thin exception translator (SolcCompilationError→ValueError HTTP 400)
                             cache key: "{content_md5}_{FEATURE_SCHEMA_VERSION}"
                             MAX_SOURCE_BYTES: class constant imported by api.py (no duplication)
                             Partial Audit #9 fix (Move 8): atexit handler unlinks _active_temp_files on
                             SIGTERM/normal exit; _purge_orphaned_sentinel_temps() at startup cleans
                             sentinel_prep_*.sol files left by a prior SIGKILL. Files orphaned by
                             SIGKILL in the *current* process remain until next restart (low-priority).
  predictor.py               Predictor — sigmoid (FP32 cast) + per-class thresholds + result dict
                             routes long contracts to windowed path; aggregates via max() across windows
                             _warmup(): 2-node 1-edge graph with edge_attr when use_edge_attr=True (Fix #4)
                             _format_result(): returns "thresholds" list (Fix #6)
                             fusion_output_dim: prefers saved_cfg first, falls back to _ARCH_TO_FUSION_DIM (Fix #7)
                             All arch fields forwarded from saved checkpoint config (Fix #2)
  api.py                     FastAPI — lifespan, Pydantic schemas, /predict, /health, /metrics
                             MAX_SOURCE_BYTES imported from ContractPreprocessor (Fix: no duplication)
  cache.py                   InferenceCache — disk-backed content-addressed cache (T1-A)
                             key: "{content_md5}_{FEATURE_SCHEMA_VERSION}"; TTL via file mtime (default 24h)
  drift_detector.py          DriftDetector — KS-based feature drift monitoring (T2-B)
                             rolling buffer 200 requests; warm-up suppression until N≥500
                             baseline from warmup data (NOT training data — BCCC-2024 causes false alerts)

ml/src/datasets/
  dual_path_dataset.py       DualPathDataset + dual_path_collate_fn
                             graph .pt loading: weights_only=True with add_safe_globals
                             squeeze(-1) guard in __getitem__ normalises legacy [E,1] files (Fix #1)

ml/src/training/
  trainer.py                 Training loop, TrainConfig (loss_fn: "bce"|"focal"), CLASS_NAMES, NUM_CLASSES
                             ARCHITECTURE = "cross_attention_lora"  (module constant; Fix #9 training)
                             _VALID_LOSS_FNS frozenset — unknown value raises ValueError immediately
                             _FocalFromLogits wrapper applies sigmoid before forwarding to FocalLoss.forward()
                             OneCycleLR uses remaining_epochs on resume (not config.epochs)
                             scheduler total_steps guard with explicit existence check (Fix #25)
                             clip_grad_norm_ clips trainable params only (not frozen CodeBERT)
                             patience_counter saved in checkpoint + JSON sidecar (Fix #11, Fix #23)
                             --resume-reset-optimizer flag; warning on missing optimizer key (Fix #24)
                             --focal-gamma / --focal-alpha CLI args wired end-to-end (2026-05-04)
                             focal_gamma/focal_alpha always logged to MLflow (Fix #9 MLflow)
  focalloss.py               FocalLoss(gamma, alpha) — FP32 cast at top of forward() (BF16 underflow guard)
                             activatable via TrainConfig.loss_fn="focal"; expects post-sigmoid probs
                             (_FocalFromLogits in trainer.py applies sigmoid before calling forward())

ml/src/utils/
  hash_utils.py              get_contract_hash(), get_contract_hash_from_content() (MD5-based)

ml/data/
  graphs/                    68,523 .pt PyG Data files (MD5 stem, edge_attr=[E] 1-D)
  tokens/                    68,568 .pt dicts (MD5 stem)
  splits/                    train/val/test_indices.npy (47,966/10,278/10,279)
  processed/
    multilabel_index.csv     label CSV (SHA256 col 2)
    _cache/
      contracts_metadata.parquet  source metadata (pass via --input to ast_extractor.py)
  cached_dataset.pkl         RAM cache — loaded into DualPathDataset at startup

ml/tests/                    10 test modules (all synthetic data; no real contracts or checkpoints)
  test_model.py              SentinelModel forward shapes; _StubTransformer avoids 500MB load
  test_gnn_encoder.py        GNNEncoder: edge_attr embedding, graceful degradation, head divisibility
  test_fusion_layer.py       CrossAttentionFusion: output shape, masked pooling, device mismatch
  test_preprocessing.py      ContractPreprocessor — error types, shapes, hash consistency
  test_dataset.py            DualPathDataset — length, shapes, collation, binary vs multi-label
  test_trainer.py            TrainConfig, FocalLoss, evaluate(), 3-epoch loss decrease
  test_api.py                /predict and /health endpoint contracts; windows_used field
  test_cache.py              InferenceCache: miss/hit/TTL/schema-version invalidation
  test_drift_detector.py     DriftDetector: warm-up suppression, KS fires on drift, buffer rolling
  test_promote_model.py      promote_model.py: stage validation, dry-run, MLflow tags

ml/checkpoints/              .pt files — not in git; managed by DVC
  multilabel-v3-fresh-60ep_best.pt               ACTIVE checkpoint (60 epochs, P0-B, batch=32)
  multilabel-v3-fresh-60ep_best_thresholds.json  Per-class tuned thresholds (v4 gate = 0.5069)
  multilabel_crossattn_best.pt                   Legacy baseline checkpoint (epoch 34, F1=0.4679, pre-P0-B)
  multilabel_crossattn_best_thresholds.json      Threshold companion for legacy baseline
  multilabel_crossattn_v2_best.pt                Paused v2 run (epoch 37, F1=0.4629, superseded)

ml/data_extraction/
  ast_extractor.py           ASTExtractorV4.3 — thin offline wrapper; parquet loading,
                             solc version resolution (get_solc_binary), mp.Pool,
                             checkpoint/resume, writes <md5>.pt files to ml/data/graphs/
                             imports graph extraction from ml/src/preprocessing/
                             contract_to_pyg() catches GraphExtractionError → None (skip-and-log)
  tokenizer.py               Offline CodeBERT tokenisation; stores feature_schema_version
                             in output .pt files (P0-D)

ml/scripts/
  train.py                   Main training entry point (--no-resume-model-only, --resume-reset-optimizer)
  tune_threshold.py          Per-class threshold sweep; full arch args + fusion_dim from config (Fix #3, #5)
  analyse_truncation.py      Measure token truncation across dataset
  build_multilabel_index.py  Build multilabel_index.csv from BCCC labels
  create_splits.py           Fixed train/val/test split indices (stratification fix 2026-05-03)
  validate_graph_dataset.py  Validate edge_attr presence + shape [E] + value range [0,5) in .pt files
                             MUST exit 0 before retraining; imports NUM_EDGE_TYPES from graph_schema.py
  compute_drift_baseline.py  Build drift_baseline.json; --source warmup|training
                             WARNING: --source training causes false alerts on modern contracts
  promote_model.py           MLflow model registry CLI (Staging/Production); --dry-run available (T2-C)
  run_overnight_experiments.py  ⚠️ LEGACY / UNMAINTAINED — 4-experiment sequential launcher from binary
                             model era; does NOT set loss_fn="focal", ignores focal_alpha, uses outdated
                             defaults; incompatible with current 10-class multi-label training.
                             Do not use for v4 or any future multi-label run.
  create_label_index.py      ⚠️ OBSOLETE — binary label_index.csv from graph.y; superseded by
                             build_multilabel_index.py (full 10-class multi-label index)
  auto_experiment.py         ← Never implemented; replaced in intent by run_overnight_experiments.py
                             (which is itself unmaintained). No working hyperparameter-optimisation
                             harness exists yet. Proper search (Optuna/grid) is deferred.

ml/docker/
  docker/Dockerfile.slither  Isolated Slither extraction environment

ml/_archive/                 Legacy files (git mv — history preserved)

ml/DIAGRAMS.md               Mermaid visual diagrams: system lifecycle, model architecture,
                             dataset loading flow (GitHub-rendered)
```


