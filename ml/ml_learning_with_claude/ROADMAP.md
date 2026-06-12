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

| File | Lines | What you'll learn |
|------|-------|------------------|
| `gnn_encoder.py` | 581 | Graph Attention Networks, message passing, 3-phase design, JK aggregation |
| `transformer_encoder.py` | 350 | BERT internals, LoRA fine-tuning, Flash Attention, prefix injection |
| `fusion_layer.py` | 281 | Cross-attention, multi-modal fusion, compile-safe design |
| `sentinel_model.py` | 562 | Full model assembly, three-eye classifier, prefix warmup |

**Chunks — `gnn_encoder.py` (581 lines):**

- ✅ `01_gnn_fundamentals_and_gat.md` — *Done*
  - Message passing concept
  - GATConv attention formula: `e_uv = LeakyReLU(a^T · [Wh_u || Wh_v || W_e·edge_type])`
  - Multi-head attention: 8 heads Phase 1 vs 1 head Phases 2+3
  - `add_self_loops=False` in Phase 2: why direction matters
  - Residual connections + IMP-G2 input skip (concept)
  - LayerNorm after each phase: why before JK
  - JK connections: over-smoothing problem, attention aggregation (concept)
  - Edge type embeddings: 11 types × 64-dim
  - Input guards overview

- ⬜ `02_gnn_encoder_forward_pass.md` — *TODO* — **lines 338–581**
  - Guards in detail (lines 364–393): schema version, edge_attr=None, OOB node index
  - Edge embedding + OOB clamping (lines 394–414): Fix C1/H9 — why clamp not crash
  - Edge mask construction (lines 416–498):
    - `struct_mask` (types 0–5), `cfg_mask`, `contains_mask`
    - `phase2_edge_types` ablation parameter (lines 428–431)
    - IMP-G1: `cf_only_ei` / `icfg_only_ei` — Layer 3/4/5 process distinct edge subsets (lines 461–476)
    - REVERSE_CONTAINS runtime synthesis: `.flip(0)` + type-7 embeddings (lines 481–497) — not stored on disk
  - `_live` vs `_intermediates` (lines 499–504): gradient-attached vs detached
  - Phase 1 execution (lines 506–526): IMP-G2 skip in code, Layer 1+2 residual
  - Phase 2 execution (lines 528–547): IMP-G1 in code — 3 layers × 3 edge subsets
  - Phase 3 execution (lines 549–568): 2 upward hops (rev_contains) + IMP-G3 downward (fwd_contains)
  - JK aggregation + `return_intermediates` (lines 570–581)

- ⬜ `03_jk_attention_internals.md` — *TODO* — **lines 76–131**
  - `register_buffer` vs Python attributes: survives `.to(device)`, `state_dict`, DDP
  - `last_weights` buffer: mean per-phase weight across batch — trainer monitoring
  - `last_weight_stds` buffer: std of per-node weights — is JK routing or constant?
  - `last_node_weights`: eval-only, per-node, can't be buffer (N varies per batch)
  - `jk_entropy` (C-3): gradient-attached entropy regularizer
    - H ≈ 0: one phase dominates (collapsed JK — bad)
    - H ≈ log(3): uniform weights (JK not routing — also bad)
    - H in between: healthy diversity
  - training vs eval mode behavior difference
  - `use_jk=False` fallback: returns Phase 3 only, jk_entropy=0

---

**Chunks — `transformer_encoder.py` (351 lines, 2 classes):**

- ✅ `04_transformer_init_lora_flash_attention.md` — *Done* — **lines 1–165**
  - P5: file role — why LoRA, why not full fine-tune, what this file's two classes do
  - Hard requirement check (lines 62–72): why `RuntimeError` not a warning — the silent failure mode
  - `LoraConfig` construction (lines 118–125): `r`, `alpha`, `target_modules`, `bias="none"`, `task_type`
  - Flash Attention 2 with BF16 (lines 134–149): try/finally dtype pollution guard
  - `get_peft_model()` — exactly 3 things it does (lines 157–165)
  - AUDIT: `lora_target_modules` str→list guard (line 116) — why MLflow breaks this
  - P7: full fine-tune vs frozen vs LoRA — comparison with trade-offs

- ✅ `05_transformer_forward_and_window_pooler.md` — *Done* — **lines 167–351**
  - `_word_embeddings` property (lines 167–170): what it accesses and why a property
  - Standard path: single-window (lines 211–215), multi-window flatten/unflatten (lines 217–222)
  - Prefix path single-window (lines 224–264):
    - `code_budget = L - K`, `inputs_embeds` bypass, `_word_embeddings` call
    - IMP-M3 Python loop for prefix mask (lines 239–244)
    - Position IDs: pos=1 for prefix, pos=3+ for code, why those values (lines 247–250)
    - `output_attentions` diagnostic: attention slice `[K:, :K]` (lines 258–263)
  - Prefix path multi-window (lines 266–306): prefix expansion via `unsqueeze(1).expand`
  - `WindowAttentionPooler` (lines 309–351): CLS index formula, single-window fast path, learned attention
  - AUDIT: Python `for b in range(B)` loop at lines 241/285 — vectorizable; performance impact at B=64
  - P7: sliding windows vs Longformer sparse attention — trade-offs

---

**Chunks — `fusion_layer.py` (281 lines):**

- ❌ *(was combined in old 05 — gets its own chunk)*

- ✅ `06_cross_attention_fusion.md` — *Done* — **lines 1–281**
  - P5: what replaced concat+MLP, why fine-grained interaction before pooling matters
  - `_scatter_to_dense` (lines 68–117): `torch.compile` graph break, BUG-C2 valid-before-clamp
  - `CrossAttentionFusion.__init__` (lines 120–195): `node_proj`, `token_proj`, BUG-C2 `token_norm`, two MHA modules
  - `forward` step by step (lines 197–281):
    - Fix #4 device assertion
    - Project + normalize (BUG-C2)
    - `_scatter_to_dense` padding + mask inversion
    - Node→Token attention: Fix #26 `need_weights=False`, Fix #8 zero-out padded positions
    - Token→Node attention: Fix #26
    - Masked mean pooling for both: Fix #6 token masking
    - Concatenate `[B,512]` → project `[B,128]`
  - AUDIT: Fix #8 necessity — pooling already excludes pads via mask, but Fix #8 is a structural invariant guarantee

---

**Chunks — `sentinel_model.py` (562 lines):**

- ❌ *(was combined in old 05 — split into 2 chunks)*

- ⬜ `07_sentinel_model_architecture.md` — *TODO* — **lines 1–259**
  - P5: three-eye concept overview — what problem three eyes solve vs single classifier
  - Module-level constants (lines 70–113): `_MAX_TYPE_ID`, `_FUNC_TYPE_IDS`, `_PREFIX_NODE_PRIORITY`, `_PREFIX_TYPE_IDX` — why defined here, not in graph_schema
  - `__init__` (lines 116–258): GNN/TF/Fusion sub-modules, eye projections, prefix modules, auxiliary heads, classifier
  - `parameter_summary` (lines 532–562): how to audit trainable vs frozen params

- ⬜ `08_sentinel_model_forward_and_prefix.md` — *TODO* — **lines 260–531**
  - `select_prefix_nodes` (lines 260–332): priority sort, IMP-M1 secondary sort by ext_call_count, type embedding bias
  - `forward` (lines 334–488): flat_mask, GNN path, GNN eye (func-only pool, max+mean, BUG-H2 ghost graph), prefix warmup guard, transformer path, three eyes, classifier, auxiliary heads
  - `compute_prefix_attention_mean` (lines 490–530): IMP-M2 diagnostic — what it measures, when to call it
  - AUDIT: empty batch guard (lines 393–407) — correctness analysis

---

### Module 5: Training `ml/src/training/`

| File | What you'll learn |
|------|------------------|
| `trainer.py` | Training loop, gradient accumulation, BF16, WeightedSampler, MLflow |
| `losses.py` | AsymmetricLoss for imbalanced multi-label classification |
| `focalloss.py` | Focal loss mechanics |

**Chunks:**
- ✅ `01_focal_loss_and_imbalance.md` — *Done* — `focalloss.py` (FocalLoss, MultiLabelFocalLoss, alpha_t audit fix, BF16 guard)
- ✅ `02_asymmetric_loss.md` — *Done* — `losses.py` (AsymmetricLoss, clip mechanism, gamma_neg≠gamma_pos, BUG-M3 per-class tensors, BCE→Focal→ASL evolution)
- ✅ `03_trainer_config_and_setup.md` — *Done* — `TrainConfig` dataclass, hyperparameter groups, `__post_init__` validation, `compute_pos_weight` (sqrt scaling, caps), VRAM helpers, `_parse_version`
- ✅ `04_trainer_eval_and_train_one_epoch.md` — *Done* — `evaluate()` (F1/Hamming, BUG-M8 threshold sweep), `train_one_epoch()` (label smoothing, DoS gradient scaling, loss combination, gradient accumulation, Fix #28 grad norm timing), `_grad_norm`, `_build_weighted_sampler`
- ✅ `05_trainer_train_setup.md` — *Done* — `train()` lines 752–1270: shared cache pattern, fork workers, model init, C-1 dtype check, checkpoint resume (strict=False, version gate, optimizer restore), loss construction, 5 param groups + fused AdamW, `torch.compile` submodule strategy, OneCycleLR + Fix #32
- ✅ `06_trainer_epoch_loop_and_mlops.md` — *Done* — `train()` lines 1270–1645: MLflow params+metrics, aux loss warmup ramp (Fix #33), NC-1 Adam state reset, JK attention monitoring, prefix attention diagnostic, BUG-M10 guardrails, atomic checkpoint write + `._orig_mod.` stripping, `.state.json` sidecar, early stopping

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
