

## How to Use This File

This file contains timeless project facts: architecture, data contracts, locked constraints,
ADRs, file inventories, and module technical specifications.

Current state (what is built, what is broken, what is next) lives in session handovers, not here.
Do not treat any section of this file as a status update. Verify against actual files when in doubt.

---

## 1. What SENTINEL Is

A decentralised AI security oracle. Smart contracts are analysed by a dual-path
GNN + CodeBERT model with CrossAttention fusion and LoRA fine-tuning. Results are proved
correct via a ZK circuit (EZKL / Groth16) and stored on-chain via AuditRegistry.
LangGraph orchestrates five specialised agents. The RAG knowledge base grounds findings
in historical DeFi exploits.

**GitHub:** https://github.com/motafegh/-sentinel
**Environment:** WSL2 Ubuntu, RTX 3070 8GB VRAM, Python 3.12, Poetry

---

## 2. System Data Flow

```
User uploads .sol contract
        │
        ▼
[M6 — API Gateway]  POST /v1/audit → job_id
        │  Celery task queue (Redis)
        ▼
[M5 — LangGraph Orchestration]
  agents/src/orchestration/graph.py
  ├── ml_assessment   → inference MCP (port 8010)
  │       │ POST /predict {"source_code": "..."}
  │       ▼
  │   [M1 — FastAPI port 8001]
  │       │ ContractPreprocessor → SentinelModel → per-class thresholds
  │       ▼
  │   {label, vulnerabilities[{vulnerability_class, probability}],
  │    threshold, truncated, num_nodes, num_edges}
  │       ← NO top-level confidence field. confidence was binary-era. Removed.
  │
  ├── rag_research    → RAG MCP (port 8011)
  │       │ search(query, k, filters)
  │       ▼
  │   [M4 RAG — FAISS+BM25+RRF over 752 DeFiHackLabs chunks]
  │
  ├── audit_check     → audit MCP (port 8012)
  │       │ get_audit_history(contract_address)
  │       ▼
  │   [AuditRegistry on Sepolia]
  │
  ├── static_analysis → Slither direct call (not via MCP)
  │
  └── synthesizer     → AuditReport
        {overall_label, risk_probability, top_vulnerability,
         vulnerabilities[], rag_evidence[], audit_history[]}
        │
        ▼
[M2 — ZKML Proof Generation]
  proxy model (input_dim=128) → EZKL → proof π + publicSignals[10 class scores]
        │
        ▼
[M5 — Blockchain]  AuditRegistry.submitAudit(proof, signals)
  ZKMLVerifier.verify() on-chain → true/false
  Emit AuditSubmitted(contractAddress, proofHash, agent, score)
```

**Routing logic (LangGraph conditional):**

```python
def _is_high_risk(ml_result: dict) -> bool:
    vulns = ml_result.get("vulnerabilities", [])
    if not vulns:
        return False
    return max(v["probability"] for v in vulns) >= 0.70

# True  → deep path: rag_research → static_analysis → synthesizer
# False → fast path: synthesizer directly
```

---

## 3. Module Dependency Map

```
M1 (ML) — no upstream deps
  → M2 needs trained checkpoint for knowledge distillation (proxy input_dim=128)
  → M4/M5 agents call M1 /predict via inference MCP

M2 (ZKML) — depends on M1 trained checkpoint
  → generates ZKMLVerifier.sol → deployed in M5
  → MUST rebuild when fusion output_dim changes (currently 128)

M3 (MLOps) — depends on M1 (tracks training runs)
  → runs parallel to M1 training

M4 (Agents/MCP) — depends on M1 inference API + RAG index
  → feeds M5 LangGraph

M5 (LangGraph) — depends on M4 MCP servers + M5 Solidity contracts
  → feeds M6

M5 (Solidity/Contracts) — depends on M2 ZKMLVerifier address
  → feeds M6

M6 (Integration) — depends on all modules
```

---

## 4. Port Map

| Port | Service | Notes |
|---|---|---|
| 8000 | M6 API gateway | FastAPI + Celery |
| 8001 | M1 FastAPI inference | uvicorn, CUDA startup ~6s |
| 8010 | sentinel-inference MCP | SSE transport |
| 8011 | sentinel-rag MCP | SSE transport |
| 8012 | sentinel-audit MCP | SSE transport |
| 1234 | LM Studio | Windows host — WSL2 gateway IP changes on reboot |
| 3000 | Dagster UI | On demand |
| 5000 | MLflow UI | On demand |

---

## 5. Environment Variables

As it might be changed thorough the progress ask me whatever you need and i will give you 

## 6. Critical Constraints

Violating any of these without the corresponding rebuild/retrain will produce silent failures.

```
GNNEncoder in_channels = 8
  — locked to 68K .pt training graph files
  — change requires: full graph dataset rebuild + retrain

CodeBERT model = "microsoft/codebert-base"
  — must match offline tokenizer used to build ml/data/tokens/

MAX_TOKEN_LENGTH = 512
  — matches training data; change requires token .pt rebuild + retrain

Node feature vector — 8-dim ordinal, fixed order:
  [type_id, visibility, pure, view, payable, reentrant, complexity, loc]
  — any change: rebuild all 68K graph .pt files + retrain

Node insertion order in graph builder:
  CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs
  — edge_index values are positional; reordering breaks edges

Fusion architecture = CrossAttentionFusion
  — output_dim = 128 (changed from 64 when cross-attention replaced concat+MLP)
  — ZKML proxy input_dim depends on this — change requires ZKML rebuild + redeploy

CLASS_NAMES order in trainer.py — source of truth for all downstream code
  — never INSERT into the middle; only APPEND new classes at the end
  — indices 0–9 must remain stable across any future additions

weights_only = False for all torch.load() calls
  — LoRA state dict is not a plain state_dict; weights_only=True rejects it silently

TRANSFORMERS_OFFLINE = 1 — must be set at shell level
  — cannot be set inside Python; if set inside Python it has no effect

Per-class thresholds companion file:
  ml/checkpoints/multilabel_crossattn_best_thresholds.json
  — must travel with the checkpoint; threshold values are sweep-derived per class

DO NOT RETRAIN — locked decision
  — 8/10 classes are healthy; DoS + CallToUnknown are data-limited
  — use per-class thresholds as the lever, not retraining
  — open a new explicit ML milestone before any retrain

ONNX opset version = 11
  — EZKL compatibility requirement; do not change

Files never to commit:
  zkml/ezkl/proving_key.pk   (~10MB, gitignored)
  zkml/ezkl/srs.params       (gitignored)

solc version:
  ZKMLVerifier.sol: ≤0.8.17 (EZKL-generated assembly uses deprecated opcodes)
  All other contracts: 0.8.20

RAG chunk configuration:
  chunk_size = 1536, chunk_overlap = 128
  — index was rebuilt at these values; changing requires full RAG rebuild
  — rebuild command: cd agents && poetry run python -m src.rag.build_index

BCCC SHA256 vs internal MD5 — never mix:
  SHA256 = hash of .sol file content → BCCC filename, CSV col 2
  MD5    = hash of .sol file path    → .pt filename in ml/data/graphs/
  bridge: graph.contract_path inside .pt → Path(...).stem = SHA256
```

---

## 7. Architecture Decision Records

| # | Decision | Chosen | Rejected | Reason | Revisit if |
|---|---|---|---|---|---|
| 001 | GNN architecture | 3-layer GAT | GCN, GraphSAGE | GAT learns per-edge attention weights; GCN averages all neighbours equally | Dataset grows to protocol-scale graphs (1000s of nodes) |
| 002 | CodeBERT training | LoRA fine-tuning (r=8, peft) | Frozen feature extractor / full fine-tune | Frozen gave weak rare-class F1; full fine-tune risks catastrophic forgetting on 68K samples; LoRA trains ~500K params safely | Dataset > 500K samples; switch to full fine-tune |
| 003 | Fusion method | CrossAttentionFusion (bidirectional, 128-dim) | Concat+MLP / GMU | Cross-attention: nodes and tokens enrich each other BEFORE pooling; withdraw() node attends to "call.value" directly. Concat+MLP pooled before fusing — node/token detail already gone. GMU requires equal-dim projections. | Val F1 plateau after retrain |
| 004 | Classifier loss | BCEWithLogitsLoss + pos_weight | FocalLoss + external sigmoid | BCEWithLogitsLoss numerically stable; external sigmoid collapses to float32 zero at logit > ±38 | — |
| 005 | Sigmoid location | Removed from model; applied in predictor._score() | nn.Sigmoid inside model | BCEWithLogitsLoss requires raw logits; inference side sigmoid keeps training/inference consistent | — |
| 006 | Label source for retrain | External CSV (multilabel_index.csv) keyed by MD5 stem | Patch graph.y in 68K .pt files | CSV is auditable, updatable; patching 68K binary files is slow and irreversible | — |
| 007 | Output classes | 10 (WeakAccessMod excluded) | 11 | Zero .pt graph files for WeakAccessMod (Slither failures); zero positives → pos_weight=inf → NaN | Re-extract WeakAccessMod contracts |
| 008 | pos_weight scope | Training split only | Full dataset | Full dataset leaks val/test label distribution into loss hyperparameter | — |
| 009 | Node features | 8-dim ordinal | 17-dim one-hot | Ordinal encodes security-relevant ordering; GNNEncoder in_channels=8 locked to 68K training graphs | Full retrain with richer features (B-06) |
| 010 | Graph scope | First non-dependency contract only | All user contracts merged | Works for single-file contracts; simpler | Multi-contract protocol audits (C-01) |
| 011 | MCP transport | SSE (HTTP) | stdio (subprocess) | Production standard; survives Docker network hops; multi-client capable | Tools only run on same machine |
| 012 | MCP schema validation | inputSchema + handler defensive cap | Handler-only | mcp 1.27.0 enforces inputSchema at protocol level | — |
| 013 | HybridRetriever instantiation | Module-level at import time | Per-request | 400ms + 5MB per request if per-call; single RAM instance | Multiple retriever configs per session |
| 014 | Batch inference loop | Sequential in _handle_batch_predict | asyncio.gather concurrent | GPU-bound — concurrent serialises on GPU anyway | M1 adds native batched CUDA forward pass |
| 015 | ZKML proxy architecture | ~6K param MLP (128→64→32→num_classes) | Full SentinelModel in ZK circuit | Full model ~125M params — far too large for EZKL; proxy via knowledge distillation; input_dim=128 matches CrossAttentionFusion output (was 64 with old FusionLayer) | EZKL supports larger circuits |
| 016 | On-chain proof storage | keccak256(zkProof) hash only | Full proof bytes (~2KB) | Gas cost at scale; hash sufficient for verification reference | — |
| 017 | Solidity proxy pattern | UUPS | Transparent proxy | Gas efficient; upgrade logic in implementation; OpenZeppelin v5 compatible | — |
| 018 | Checkpoint format | Dict {model, optimizer, scheduler, epoch, best_f1, config{architecture}} | Plain state_dict | config.architecture field enables predictor to auto-detect fusion_output_dim without code change | — |
| 019 | assert → RuntimeError in ZKML | RuntimeError with message | Python assert | python -O strips assert silently; EZKL cascade means silent failure corrupts circuit | — |
| 020 | DualPathDataset CSV loading | Vectorised numpy | iterrows() per-row | iterrows() on 68K rows took 45s; vectorised takes <2s | — |
| 021 | RAG chunk size | 1536 chars, overlap 128 | 512 chars (original) | 512 too small for Solidity exploit context; 1536 preserves full function bodies | Embedding model changes |
| 022 | GNNEncoder pooling location | Deferred to CrossAttentionFusion | Pooled inside GNNEncoder | Early pooling destroys node-level detail before fusion; cross-attention needs per-node embeddings so withdraw() can query "call.value" tokens individually | — |
| 023 | TransformerEncoder output | All token embeddings [B,512,768] | CLS only [B,768] | CLS is a blurry contract-level summary; cross-attention requires all 512 positions so each GNN node can identify its specific relevant tokens | — |
| 024 | LoRA target modules | query + value projections only | All attention / FFN | Q+V control what CodeBERT attends to and extracts; 295K trainable params with maximum security-pattern adaptation | — |
| 025 | CrossAttentionFusion output dim | 128 | 64 (old FusionLayer) | Both enriched modalities (256-dim each after pooling) concatenated → 512 → 128; wider justified because both paths contribute semantically enriched representations | Retrain shows diminishing returns; reduce if VRAM constrained |
| 026 | batch_size for cross-attention | 16 | 32 (old) | CrossAttentionFusion padding [B,max_nodes,256] + LoRA gradients increase VRAM vs old frozen concat+MLP; 16 safe on RTX 3070 8GB | GPU with >8GB VRAM |
| 027 | RAG ingestion scheduling | Dagster single asset `rag_index` | Multi-asset fake chain | Old 3-asset chain had no data flow, re-fetched source 3×; single asset encapsulates full pipeline with honest lineage | True multi-asset lineage with IO managers |
| 028 | Deduplication strategy | JSON file of seen document hashes | SQLite | Simple, human-readable, git-trackable at current scale (~1K docs); upgradable to SQLite without API change | Document count >100K |
| 029 | RAG index write safety | FileLock + atomic rename (.tmp → real) | In-place overwrites | Concurrent cron/Dagster/feedback cycles could corrupt index; lock serialises writes; atomic rename prevents partial reads on crash | — |
| 030 | Feedback loop back-pressure | Exponential backoff on RPC error (capped 5 min) | Fixed 30s retry | Protects public RPC from hammering; respects provider rate limits | — |

## 8. Module Technical Facts

### 8.1 Module 1 — ML Core

**Path:** `ml/`

#### Tech Stack

| Tool                           | Role                                     |
| ------------------------------ | ---------------------------------------- |
| PyTorch ^2.2                   | Model training and inference             |
| PyTorch Geometric ^2.5         | GNN layers, PyG Data, Batch, DataLoader  |
| HuggingFace Transformers ^4.40 | CodeBERT tokenizer and model             |
| peft                           | LoRA fine-tuning of CodeBERT             |
| scikit-learn ^1.4              | f1_score, hamming_loss, train_test_split |
| MLflow ^2.12                   | Experiment tracking                      |
| DVC ^3.49                      | Dataset and model versioning             |
| Slither 0.10.x                 | Solidity AST extraction                  |
| FastAPI ^0.111                 | Inference HTTP API                       |
| loguru ^0.7                    | Structured logging                       |

#### Model Architecture

```
Input: Solidity source string
        │
        ├── ContractPreprocessor
        │     ├── Slither AST extraction
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
        │     │     Node insertion order: CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs
        │     │     Edge types: CALLS, READS, WRITES, EMITS, INHERITS
        │     │     edge_index [2, E] int64
        │     │     Output: PyG Data — graph.x [N, 8], graph.edge_index [2, E]
        │     │
        │     └── CodeBERT tokenizer
        │           model: microsoft/codebert-base
        │           max_length=512, truncation=True, padding="max_length"
        │           Output: input_ids [1, 512], attention_mask [1, 512]
        │
        ▼
SentinelModel.forward(graphs, input_ids, attention_mask)
  │
  ├── GNNEncoder
  │     3-layer GATConv:
  │       conv1: in=8,  out=8,  heads=8, concat=True  → [N, 64]
  │       conv2: in=64, out=8,  heads=8, concat=True  → [N, 64]
  │       conv3: in=64, out=64, heads=1, concat=False → [N, 64]
  │     Dropout p=0.2 on attention coefficients AND on node activations
  │     NO global_mean_pool — pooling DEFERRED to CrossAttentionFusion
  │     Returns: (node_embeddings [N, 64], batch [N])
  │     N = total nodes across all B contracts in the batch
  │
  ├── TransformerEncoder
  │     CodeBERT with LoRA: r=8, lora_alpha=16, lora_dropout=0.1
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
  │       Each node attends to which tokens are relevant to it
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
  GNNEncoder:            ~100K trainable
  TransformerEncoder:    ~295K trainable (LoRA only) + 124,705,536 frozen
  CrossAttentionFusion:  ~530K trainable (projections + 2× MHA + output MLP)
  Classifier:            128×10 + 10 = 1,290 trainable
  Total trainable:       ~925K
  Total frozen:          124,705,536

Predictor._score():
  logits = model.forward()           → [1, 10]
  probs  = torch.sigmoid(logits)     → [1, 10]
  Per-class thresholds from multilabel_crossattn_best_thresholds.json
  vulnerabilities = [{vulnerability_class, probability} for prob >= threshold[class]]
  label = "vulnerable" if vulnerabilities else "safe"

Predictor backward compatibility:
  architecture = ckpt["config"].get("architecture", "legacy")
  "cross_attention_lora" → fusion_output_dim = 128
  "legacy"               → fusion_output_dim = 64  (old binary concat+MLP checkpoints)
  Prevents silent Linear(64→10) vs Linear(128→10) shape mismatch.
  weights_only=False required everywhere — LoRA state dict contains peft-specific classes.
```

#### Active Checkpoint

```
File:       ml/checkpoints/multilabel_crossattn_best.pt
Thresholds: ml/checkpoints/multilabel_crossattn_best_thresholds.json
Best val F1-macro: 0.4679 (epoch 34)
Architecture key: "cross_attention_lora"

Load pattern:
  raw = torch.load(path, weights_only=False)   # weights_only=True breaks LoRA
  state_dict = raw["model"] if "model" in raw else raw
```

#### Start ML API

```bash
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel_crossattn_best.pt \
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001
```

Health check response must include:
```json
{"architecture": "cross_attention_lora", "thresholds_loaded": true}
```

#### API Contract

```
POST /predict
  Request:  {"source_code": str}   ← field name is source_code, not contract_code
  Response: {
    "label": "vulnerable" | "safe",
    "vulnerabilities": [{"vulnerability_class": str, "probability": float}],
    "threshold": float,
    "truncated": bool,
    "num_nodes": int,
    "num_edges": int
  }
  NOTE: NO top-level "confidence" field. Removed in Track 3. Any code using
  ml_result["confidence"] is stale and will KeyError or produce silent wrong routing.

GET /health → {"status": "ok"|"degraded", "architecture": str, "thresholds_loaded": bool}
HTTP errors: 400 bad input | 413 GPU OOM | 422 Pydantic | 503 not loaded | 504 timeout
```

#### 10 Output Classes

Defined in `ml/src/training/trainer.py` as `CLASS_NAMES` — single source of truth.

```python
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

#### Dataset Facts

```
Source:           BCCC-SCsVul-2024
Graph .pt files:  68,555 (ml/data/graphs/, MD5 stem)
Token .pt files:  68,570 (ml/data/tokens/, MD5 stem)
Splits:           train/val/test_indices.npy (47988/10283/10284)
Label CSV:        ml/data/processed/multilabel_index.csv (68,555 rows × 10 classes)

Two hash systems — never mix:
  SHA256 = hash of .sol file content → BCCC filename, CSV col 2
  MD5    = hash of .sol file path    → .pt filename
  Bridge: graph.contract_path inside .pt → Path(...).stem = SHA256

Class distribution (training split pos_weights):
  CallToUnknown               pos=8,028   pw=7.53
  DenialOfService             pos=995     pw=67.75
  ExternalBug                 pos=11,069  pw=5.16
  GasException                pos=17,319  pw=2.96
  IntegerUO                   pos=35,724  pw=0.92
  MishandledException         pos=15,148  pw=3.52
  Reentrancy                  pos=16,666  pw=3.09
  Timestamp                   pos=7,304   pw=8.41
  TransactionOrderDependence  pos=11,783  pw=4.82
  UnusedReturn                pos=11,325  pw=5.05
```

#### File Inventory

```
ml/src/models/
  sentinel_model.py          SentinelModel — orchestrates all sub-modules
  gnn_encoder.py             GNNEncoder — 3-layer GAT
  transformer_encoder.py     TransformerEncoder — CodeBERT + LoRA
  fusion_layer.py            CrossAttentionFusion — output_dim=128
  classifier.py              nn.Linear(128, 10), no sigmoid

ml/src/inference/
  preprocess.py              ContractPreprocessor — Slither + tokenizer
  predictor.py               Predictor — sigmoid + per-class thresholds + result dict
  api.py                     FastAPI — lifespan, Pydantic schemas, /predict, /health

ml/src/datasets/
  dual_path_dataset.py       DualPathDataset + dual_path_collate_fn

ml/src/training/
  trainer.py                 Training loop, TrainConfig, CLASS_NAMES, NUM_CLASSES

ml/data/
  graphs/                    68,555 .pt PyG Data files (MD5 stem)
  tokens/                    68,570 .pt dicts (MD5 stem)
  splits/                    train/val/test_indices.npy (47988/10283/10284)
  processed/
    multilabel_index.csv     68,555 rows × 10-class multi-hot labels
    contract_labels_correct.csv  44,442 rows binary+class labels (historical)
    label_index.csv          68,555 rows binary labels (historical)
  cached_dataset.pkl         RAM cache — loaded into DualPathDataset at startup

ml/checkpoints/              .pt files — not in git
  multilabel_crossattn_best.pt               Active checkpoint (489MB)
  multilabel_crossattn_best_thresholds.json  Per-class sweep-derived thresholds
  run-alpha-tune_best.pt                     Legacy binary checkpoint (477MB)

ml/_archive/                 Legacy files (git mv — history preserved)
```

---

### 8.2 Module 2 — ZKML

**Path:** `zkml/`

#### Tech Stack

| Tool | Role |
|---|---|
| EZKL 23.0.5 | ZK circuit generation, proving, verification |
| PyTorch | Proxy model definition and distillation |
| ONNX opset 11 | Intermediate representation for EZKL |
| Foundry | ZKMLVerifier.sol deployment and testing |
| web3.py ^7.15 | Proof submission to AuditRegistry |

#### Proxy Model Architecture

```
Full SentinelModel: ~315K trainable params → too large for EZKL (~10K limit)

Proxy MLP:
  Input:  128-dim fused embedding (CrossAttentionFusion output — BEFORE classifier)
  Layers: Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→10)
  Params: ~8K
  Target: proxy agrees with full model ≥95% per class
  Loss:   MSE(proxy_output, full_model_output)

ONNX export:
  opset_version=11        — EZKL requirement
  dynamic_axes: batch     — required for variable batch sizes
  dummy_input shape: (1, 128)   — must match CrossAttentionFusion output exactly
```

#### EZKL Pipeline

```
Step 1: gen_settings(model.onnx, settings.json)
Step 2: calibrate_settings(model.onnx, calibration_data.json, settings.json)
Step 3: compile_circuit(model.onnx, settings.json, model.compiled)
Step 4: setup()  ← ONE TIME, expensive, generates proving_key.pk + verification_key.vk
Step 5: prove()  ← PER AUDIT, ~30-60s
  outputs: proof π + publicSignals[10 class scores]
Step 6: verify()  ← on-chain or off-chain
  gas: ~250K on-chain

CRITICAL: use RuntimeError not assert in setup_circuit.py
  python -O strips assert silently; EZKL cascade means silent failure corrupts circuit

EZKL 23.0.5 function names:
  compile_model → compile_circuit
  calibrate     → calibrate_settings
  get_srs requires asyncio.run() + await (PyO3 Rust future wrapping)
  all other EZKL functions are synchronous

EZKL instance value encoding:
  Values stored as little-endian 32-byte hex strings
  Correct decode: int.from_bytes(bytes.fromhex(x), byteorder='little')
  int(x, 16) treats as big-endian — produces garbage values
```

#### File Inventory

```
zkml/src/ezkl/
  setup_circuit.py         EZKL pipeline steps 1–4
  run_proof.py             Proof generation per audit (step 5)
  extract_calldata.py      Format proof for Solidity calldata

zkml/src/distillation/
  proxy_model.py           Proxy MLP definition (input_dim=128)
  train_proxy.py           Knowledge distillation training
  export_onnx.py           ONNX export

zkml/ezkl/
  proof.json               Most recent proof artifact
  proving_key.pk           PRIVATE — gitignored
  srs.params               PRIVATE — gitignored
  settings.json            Circuit settings
  model.compiled           Compiled circuit
```

---

### 8.3 Module 3 — MLOps

**Path:** `mlops/`

#### Tech Stack

| Tool | Role |
|---|---|
| MLflow ^2.12 | Experiment tracking, metric logging |
| DVC ^3.49 | Dataset and model versioning |
| Dagster 1.12.22 | Asset orchestration (RAG ingestion pipeline) |

#### MLflow

```
Tracking URI: sqlite:///mlruns.db (project root)
Experiments:
  "sentinel-training"     — binary model (historical)
  "sentinel-multilabel"   — Track 3 multi-label runs

Per training run params logged:
  num_classes, epochs, batch_size=16, lr, weight_decay, threshold,
  grad_clip, warmup_pct, num_workers, device,
  architecture="cross_attention_lora",
  label_csv, resume_from, resume_model_only,
  pos_weight_{classname} × 10

Metrics/epoch:
  train_loss, val_f1_macro, val_f1_micro, val_hamming,
  val_f1_{classname} × 10

Start UI: mlflow ui --port 5000
```

#### Dagster (RAG ingestion)

```
Asset:    rag_index — full ingestion pipeline
          (DeFiHackLabs → chunk → embed → FAISS+BM25 index)
Schedule: daily_ingestion_schedule (cron: 0 2 * * *)
Home:     agents/.dagster (DAGSTER_HOME env var)

Start UI:
  cd ~/projects/sentinel/agents
  poetry run dagster dev -f src/ingestion/scheduler_dagster.py
  → http://localhost:3000
```

---

### 8.4 Module 4 — Agents / MCP / RAG

**Path:** `agents/`

#### Tech Stack

| Tool | Version | Role |
|---|---|---|
| LangChain ^0.3 | Agent framework base |
| LangGraph ^0.2 | State machine orchestration |
| langchain-openai ^0.2 | OpenAI-compatible → LM Studio |
| mcp 1.27.0 | MCP protocol SDK — enforces inputSchema at protocol level |
| httpx ^0.27 | Async HTTP for ML API calls |
| faiss-cpu ^1.8 | Vector similarity search |
| rank-bm25 ^0.2 | BM25Okapi keyword search |
| pydantic ^2.0 | Structured outputs |
| filelock ^3.13 | Single-writer lock on index writes |
| web3 ^7.15 | Ethereum RPC (feedback loop + audit MCP) |
| dagster 1.12.22 | RAG ingestion scheduling |
| loguru ^0.7 | Structured logging |
| python-dotenv ^1.0 | .env loading |

#### File Inventory

```
agents/src/
  ingestion/
    pipeline.py              DeFiHackLabsFetcher → Chunker → Embedder → index write
    deduplicator.py          SHA256 seen_hashes.json deduplication
    feedback_loop.py         Polls AuditRegistry, 1999-block batch chunks, exp backoff
    scheduler_cron.py        Cron-based scheduling
    scheduler_dagster.py     Dagster asset + daily schedule

  llm/
    client.py                LM Studio client, AGENT_MODEL_MAP, model routing

  mcp/servers/
    inference_server.py      Port 8010 — wraps ML API /predict
    rag_server.py            Port 8011 — wraps HybridRetriever
    audit_server.py          Port 8012 — reads AuditRegistry on Sepolia

  orchestration/
    state.py                 AuditState TypedDict
    nodes.py                 Node functions — ml_assessment, rag_research,
                             audit_check, static_analysis, synthesizer
    graph.py                 StateGraph definition + conditional routing

  rag/
    build_index.py           Full rebuild: lock + atomic writes + rollback snapshot
    chunker.py               chunk_size=1536, overlap=128
    embedder.py              nomic-embed-text-v1.5 via direct OpenAI client
    retriever.py             HybridRetriever: FAISS + BM25 + RRF (k=60)
    fetchers/
      base_fetcher.py        Abstract base
      github_fetcher.py      DeFiHackLabsFetcher — scans src/test/ AND past/

agents/scripts/
  smoke_inference_mcp.py
  smoke_rag_mcp.py
  smoke_audit_mcp.py
  smoke_langgraph.py
  test_k_cap.py

agents/tests/
  test_inference_server.py
  test_rag_server.py (pending verify — may not exist yet)
  test_audit_server.py
  test_chunker.py
  test_deduplicator.py
  test_github_fetcher.py
  test_retriever_filters.py
  test_graph_routing.py

agents/data/
  index/
    faiss.index              FAISS IndexFlatL2, 752 vectors × 768-dim
    bm25.pkl                 BM25Okapi over 752 chunks
    chunks.pkl               752 Chunk dataclass instances
    index_metadata.json      Build ID, config hash, artifact SHA256s
    seen_hashes.json         726 source file hashes for deduplication
    .index.lock              FileLock — prevents concurrent writes
    backups/<build_id>/      Rollback snapshots
  feedback_state.json        Last processed Sepolia block number
```

#### RAG Pipeline Technical Facts

```
Source:              DeFiHackLabs GitHub repo
Data:                726 .sol files (src/test/ AND past/ directories)
Index:               752 chunks × 768-dim float32
Chunk size:          1536 chars, overlap 128
Embedding model:     text-embedding-nomic-embed-text-v1.5
Index schema ver:    rag_index_v2
Build ID:            20260424T185700Z-6ca204aa
Config hash:         106f0a5ba55b2a0c8b49a943f3fcfbc383142d339b2eee93ba6545e9b182c0f5

Retrieval algorithm:
  FAISS: similarity_search with k=20 candidates
  BM25:  BM25Okapi.get_scores, top-20 candidates
  RRF:   score[doc] = 1/(k+rank_faiss) + 1/(k+rank_bm25), k=60
  Post-filter → top-k returned

Artifact hashes: recorded in agents/data/index/index_metadata.json — verify there.

HybridRetriever: instantiated at module level (import time), not per-request
  — startup validation: checks FAISS vector count matches chunks length; fails fast if corrupt
  — single RAM instance shared across all tool calls
  — ~400ms load, ~5MB

Ingestion pipeline safety:
  FileLock(timeout=300) on .index.lock
  Atomic writes: write to .tmp then os.replace() (POSIX rename)
  Rollback: snapshot to backups/<build_id>/ before writing

Rebuild command:
  cd ~/projects/sentinel/agents && poetry run python -m src.rag.build_index
```

#### MCP Server Patterns

```
All three servers share the same wiring:
  SSE transport (not stdio — production/Docker compatible)
  load_dotenv(override=True) at top of file
  Module-level client/retriever init (fail-fast, not per-request)
  Broad except Exception in handlers (unhandled exception closes SSE session)
  Full traceback logged via logger.exception()

inference_server.py (port 8010):
  Tool: predict(contract_code, contract_address?)
    → POST localhost:8001/predict {"source_code": contract_code}
    → TimeoutException → fall back to mock (transient)
    → HTTPStatusError  → re-raise (4xx = payload bug, not transient)
    → RequestError     → fall back to mock (M1 not running)
  Tool: batch_predict(contracts, max 20)
    → sequential loop (GPU-bound)
    → partial failure: HTTPStatusError per contract recorded, batch continues

rag_server.py (port 8011):
  Tool: search(query, k=5, filters={})
  Filters (additionalProperties: False):
    vuln_type, date_gte, loss_gte, source, has_summary
  Returns: {query, k_requested, k_returned, filters_applied, results[]}
  Result shape: {chunk_id, content, doc_id, chunk_index, total_chunks, metadata{}, score}
  Known issue: score not currently returned (retriever returns plain Chunk, not scored)

audit_server.py (port 8012):
  AsyncWeb3 client (one client reused)
  Mock mode: auto-enabled when SEPOLIA_RPC_URL is missing (AUDIT_MOCK=true also forces it)
  Score decoding: score = field_element / 8192  (EZKL scale factor)
  Tools:
    get_latest_audit(contract_address) → {score, label, proof_hash, timestamp, agent}
    get_audit_history(contract_address, limit=10) → list of AuditResult, newest first
    check_audit_exists(contract_address) → {exists: bool, count: int}
    submit_audit — deferred (requires valid ZK proof + MIN_STAKE tokens staked)
  Known issue: ABI loaded at import even in mock mode
    — crashes if contracts/out/AuditRegistry.sol/AuditRegistry.json is missing
    — fix: lazy load ABI only in real mode (A-P2)
```

#### LangGraph Orchestration

```
Files:
  agents/src/orchestration/state.py    AuditState TypedDict
  agents/src/orchestration/nodes.py    Node functions
  agents/src/orchestration/graph.py    StateGraph + conditional routing

AuditState TypedDict:
  contract_code:    str
  contract_address: str
  ml_result:        dict | None
  rag_results:      list | None
  audit_history:    list | None
  static_findings:  dict | None
  final_report:     dict | None
  error:            str | None

Graph nodes:
  ml_assessment    → calls predict tool via MultiServerMCPClient
  rag_research     → calls search (query built from top vulnerability class)
  audit_check      → calls get_audit_history
  static_analysis  → Slither direct call (not via MCP)
  synthesizer      → assembles final_report

Conditional routing:
  After ml_assessment:
    _is_high_risk() == True  → rag_research → static_analysis → synthesizer
    _is_high_risk() == False → synthesizer (fast path)

  _is_high_risk(ml_result):
    vulns = ml_result.get("vulnerabilities", [])
    return max(v["probability"] for v in vulns) >= 0.70 if vulns else False

Final report schema:
  overall_label, risk_probability, top_vulnerability,
  vulnerabilities[], rag_evidence[], audit_history[]
  — NO confidence field anywhere in this schema

MultiServerMCPClient config:
  "sentinel-inference": {"url": "http://localhost:8010/sse", "timeout": 120.0}
  "sentinel-rag":       {"url": "http://localhost:8011/sse", "timeout": 30.0}
  "sentinel-audit":     {"url": "http://localhost:8012/sse", "timeout": 30.0}

Test patterns (non-obvious):
  Loguru capture: add temporary sink (list) — capsys/caplog don't capture loguru
  MagicMock pickling: use plain dict for Chunk objects — MagicMock raises PicklingError
  Module-level retriever: mock HybridRetriever before import to avoid index load
  pytest config: agents/pyproject.toml [tool.pytest.ini_options] addopts=""
                 overrides root sentinel/pyproject.toml --cov flags
```

---

### 8.5 Module 5 — Solidity Contracts

**Path:** `contracts/`

#### Tech Stack

| Tool | Role |
|---|---|
| Foundry (forge, cast) | Build, test, deploy |
| OpenZeppelin v5 | ERC-20, UUPS, Ownable |
| solc-select | Compiler version management |

#### Contract Architecture

```
SentinelToken.sol (ERC-20 + staking)
  stake(amount)         → lock SENTINEL, become audit-eligible agent
  unstake(amount)       → unlock after lock period
  slash(agent, amount)  → penalise dishonest agent
  stakedBalance(agent)  → view staked amount
  MIN_STAKE = 1000 SENTINEL tokens

AuditRegistry.sol (UUPS upgradeable proxy)
  submitAudit(contractAddress, score, zkProof, publicSignals)
    requires: stakedBalance(msg.sender) >= MIN_STAKE
    requires: ZKMLVerifier.verify(proof, signals) == true
    stores:   keccak256(zkProof) on-chain
    stores:   AuditResult{score, proofHash, timestamp, agent, verified}
    emits:    AuditSubmitted(contractAddress, proofHash, agent, score)
  getLatestAudit(contractAddress) → AuditResult
  getAuditHistory(contractAddress) → AuditResult[]
  ABI path: contracts/out/AuditRegistry.sol/AuditRegistry.json

ZKMLVerifier.sol (auto-generated by EZKL from verification_key.vk)
  verify(bytes proof, uint256[] publicSignals) → bool
  BN254 curve parameters embedded as constants
  Reference via IZKMLVerifier interface in AuditRegistry (swappable)
  Must redeploy when fusion_dim changes (proxy input_dim changed 64→128)

solc versions:
  ZKMLVerifier.sol: ≤0.8.17 (EZKL-generated assembly uses deprecated opcodes)
  All other contracts: 0.8.20
  solc-select use 0.8.17 before compiling ZKMLVerifier standalone

Key patterns:
  UUPS over transparent proxy — upgrade logic in implementation, gas efficient
  keccak256(proof) not full bytes — gas cost at scale
  IZKMLVerifier interface — AuditRegistry references verifier via interface, swappable
  _disableInitializers() in constructor — standard UUPS re-init protection
```

### 8.6 Module 6 — Integration (Planned Architecture)

**Path:** `root/` (not built)

```
API Gateway (FastAPI + Celery + Redis):
  POST /v1/audit        → {job_id}           (immediate, spawns Celery task)
  GET  /v1/audit/{id}   → {status, result?}  (poll for completion)
  GET  /v1/proof/{id}   → {proof, signals, on_chain_tx?}
  GET  /health          → {status, services: {ml, agents, db}}

Docker Compose services:
  Port 8000  api             FastAPI gateway
  Port 8001  ml-server       Module 1 inference
  Port 8002  agents          LangGraph orchestration
  Port 1234  lm-studio       Windows host (external)
  Port 5432  postgres        Audit record persistence
  Port 6379  redis           Celery task queue
  Port 5000  mlflow          Experiment tracking
  Port 9090  prometheus      Metrics
  Port 3000  grafana         Dashboards

CI/CD (GitHub Actions):
  On PR to main:
    test-ml:        poetry install, pytest, coverage ≥80%
    test-contracts: forge test -vvv --gas-report, slither
  On merge to main:
    deploy:         forge script Deploy.s.sol --rpc-url $SEPOLIA_RPC --broadcast --verify
  Secrets:          DEPLOYER_PRIVATE_KEY, ETHERSCAN_API_KEY, SEPOLIA_RPC_URL
```

---

## 12. Improvement Backlog

Ask me if needed to send you the current version and then update it and give me the final version it is dynamic 