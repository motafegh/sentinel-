# SENTINEL ML — Learning Roadmap with Claude

> **Goal:** Master the SENTINEL ML codebase deeply enough to ace ML, AI, MLOps, and Blockchain/Solidity interviews.
> Each session is hands-on: you read code, understand patterns, and build mental models you can explain out loud.

---

## What is SENTINEL?

SENTINEL is a **smart contract vulnerability detector** — it reads Solidity code and predicts which of 10 security vulnerabilities are present. It is a **dual-path deep learning system**:

- **Path 1 — Graph Neural Network (GNN):** Converts Solidity code into a graph (nodes = functions/variables/statements, edges = calls/reads/writes/control-flow). A 8-layer Graph Attention Network learns structural vulnerability patterns.
- **Path 2 — Transformer:** Uses GraphCodeBERT (a BERT variant trained on code) with LoRA adapters to read the raw source text.
- **Fusion:** A CrossAttention layer merges both views. A 3-eye classifier outputs 10 vulnerability probabilities.

This system spans **ML, AI, MLOps, and Blockchain** — making it perfect for interview prep across all four domains.

---

## Architecture at a Glance

```
Solidity Contract (.sol)
        │
        ├── graph_extractor.py ──► PyG graph (.pt)
        │   └── NODE_FEATURE_DIM=11, 11 edge types, 13 node types
        │
        └── tokenizer.py ──► token windows (.pt)
                │         [4 windows × 512 tokens, stride=256]
                │
        DualPathDataset (cached_dataset_v8.pkl)
                │
    ┌───────────┴───────────────────────────────────┐
    │                                               │
GNNEncoder (8 layers, 3 phases)        TransformerEncoder
  Phase 1: Structural (layers 1-2)       GraphCodeBERT + LoRA r=16
  Phase 2: CFG/ICFG (layers 3-5)         Flash Attention 2
  Phase 3: Bidirectional (layers 6-8)    GNN Prefix Injection (K=48)
  JK Attention aggregation               BF16 precision
    │                                               │
    └──────────── CrossAttentionFusion ─────────────┘
                      ↓
             Three-Eye Classifier
         (GNN eye + TF eye + Fused eye)
                      ↓
         [B, 10] logits → 10 vulnerability classes
```

---

## The 10 Vulnerability Classes (output targets)

| Index | Class | What it is |
|-------|-------|-----------|
| 0 | CallToUnknown | Low-level `.call()` to unknown address |
| 1 | DenialOfService | Gas griefing / unbounded loop |
| 2 | ExternalBug | External contract dependency bugs |
| 3 | GasException | Unchecked send/transfer failures |
| 4 | IntegerUO | Integer overflow/underflow (Solidity <0.8) |
| 5 | MishandledException | Unchecked return values |
| 6 | Reentrancy | Classic reentrancy attack |
| 7 | Timestamp | Block timestamp manipulation |
| 8 | TransactionOrderDependence | Front-running / TOD |
| 9 | UnusedReturn | Return value of internal functions ignored |

---

## Learning Journey — Module by Module

### Module 1: Preprocessing `ml/src/preprocessing/` ← **Start Here**
The foundation. Everything in the system depends on this.

| File | What you'll learn |
|------|------------------|
| `graph_schema.py` | Graph vocabulary: node types, edge types, feature schema, version history |
| `graph_extractor.py` | Core logic: Solidity → graph, feature engineering, CFG extraction |
| `__init__.py` | Python package design, single source of truth pattern |

**Chunks:**
- `01_big_picture_and_context.md` — System overview + Solidity/Blockchain primer
- `02_graph_schema_node_types_edges.md` — Node types, edge types, schema constants
- `03_feature_engineering_deep_dive.md` — 11-dim feature vector, each feature explained
- `04_cfg_extraction_and_graph_building.md` — How CFG nodes are built, ICFG edges, DEF_USE
- `05_contract_selection_and_main_pipeline.md` — Contract heuristics, `extract_contract_graph()`, exception hierarchy
- `06_batch_pipeline_ast_extractor.md` — Multiprocessing, checkpoint/resume, solc binary management

---

### Module 2: Data Extraction `ml/src/data_extraction/`

| File | What you'll learn |
|------|------------------|
| `tokenizer.py` | Sliding window tokenization, GraphCodeBERT tokenizer, stride |
| `ast_extractor.py` | (Cross-file — already covered in Module 1 Chunk 6) |

**Chunks:**
- `01_tokenizer_windowed_approach.md`

---

### Module 3: Datasets `ml/src/datasets/`

| File | What you'll learn |
|------|------------------|
| `dual_path_dataset.py` | PyTorch Dataset, DataLoader, collate_fn, caching |

**Chunks:**
- `01_dual_path_dataset_and_dataloader.md`

---

### Module 4: Models `ml/src/models/`
The heart of the AI system.

| File | What you'll learn |
|------|------------------|
| `gnn_encoder.py` | Graph Attention Networks, message passing, 3-phase design, JK aggregation |
| `transformer_encoder.py` | BERT internals, LoRA fine-tuning, Flash Attention, prefix injection |
| `fusion_layer.py` | Cross-attention, multi-modal fusion, compile-safe design |
| `sentinel_model.py` | Full model assembly, three-eye classifier, prefix warmup |

**Chunks:**
- `01_gnn_fundamentals_and_gat.md`
- `02_gnn_encoder_three_phases.md`
- `03_transformer_and_lora.md`
- `04_cross_attention_fusion.md`
- `05_full_sentinel_model_assembly.md`

---

### Module 5: Training `ml/src/training/`

| File | What you'll learn |
|------|------------------|
| `trainer.py` | Training loop, gradient accumulation, BF16, WeightedSampler, MLflow |
| `losses.py` | AsymmetricLoss for imbalanced multi-label classification |
| `focalloss.py` | Focal loss mechanics |

**Chunks:**
- `01_training_loop_and_config.md`
- `02_loss_functions_imbalanced_learning.md`

---

### Module 6: Inference `ml/src/inference/`
MLOps-heavy module — production deployment.

| File | What you'll learn |
|------|------------------|
| `api.py` | FastAPI, REST endpoints, production patterns |
| `predictor.py` | Model loading, threshold tuning, checkpoint management |
| `preprocess.py` | Online preprocessing, schema version checks |
| `drift_detector.py` | Feature drift detection, statistical monitoring |
| `cache.py` | Content-addressed caching, schema invalidation |

**Chunks:**
- `01_fastapi_inference_service.md`
- `02_predictor_and_threshold_tuning.md`
- `03_drift_detection_and_monitoring.md`
- `04_cache_and_schema_versioning.md`

---

### Module 7: Utilities `ml/src/utils/`

| File | What you'll learn |
|------|------------------|
| `hash_utils.py` | Content-addressed file naming, MD5 hashing |

---

### Module 8: Scripts (Selected) `ml/scripts/`
High-signal scripts for interview prep.

| Script | What you'll learn |
|--------|------------------|
| `train.py` | End-to-end training CLI, argparse, MLflow integration |
| `tune_threshold.py` | Post-training threshold optimization for imbalanced classes |
| `promote_model.py` | Model versioning, promotion gates, MLOps patterns |
| `label_cleaner.py` | Data quality, structural precondition heuristics |
| `compare_pipelines.py` | A/B experiment comparison, statistical significance |

---

## Interview Coverage Map

| Topic | Where you learn it |
|-------|--------------------|
| **Graph Neural Networks (GNN/GAT)** | Module 4 — models/gnn_encoder.py |
| **Transformer + BERT + LoRA** | Module 4 — models/transformer_encoder.py |
| **Multi-label imbalanced classification** | Module 5 — training/losses.py |
| **Multi-modal fusion (text + graph)** | Module 4 — models/fusion_layer.py |
| **Feature engineering** | Module 1 — preprocessing/graph_extractor.py |
| **Data pipelines + multiprocessing** | Module 1 — data_extraction/ast_extractor.py |
| **PyTorch Dataset / DataLoader** | Module 3 — datasets/dual_path_dataset.py |
| **MLOps: REST API serving** | Module 6 — inference/api.py |
| **MLOps: Drift detection** | Module 6 — inference/drift_detector.py |
| **MLOps: Caching / schema versioning** | Module 6 — inference/cache.py |
| **MLOps: Threshold tuning** | Module 8 — scripts/tune_threshold.py |
| **Blockchain: Solidity vulnerabilities** | Module 1 (context) + all modules |
| **Blockchain: Smart contract analysis** | Module 1 — preprocessing/ |
| **Solidity: AST / CFG** | Module 1 — graph_extractor.py |
| **Slither static analysis tool** | Module 1 — graph_extractor.py |
| **torch.compile, BF16, Flash Attention** | Module 4+5 — advanced topics |
| **Content-addressed storage** | Module 7 — utils/hash_utils.py |

---

## How to Use This Roadmap

1. **Study one chunk at a time** — don't rush. Each chunk has ~20–30 min of reading.
2. **After each chunk, close it and explain it out loud** (rubber duck technique).
3. **Chunks marked with 🎯 INTERVIEW FOCUS** have explicit interview questions — answer them before reading the answers.
4. **Chunks with ⚡ FAST FORWARD** cover things you should know exist but don't need to memorize.
5. **When done with a module**, revisit the interview coverage map above and test yourself.

> Start with: `Preprocessing/01_big_picture_and_context.md`
