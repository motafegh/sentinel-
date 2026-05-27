
---

# 🔍 SENTINEL ML Module — Adversarial Audit Grouping Plan

I've read all 6 documentation files and explored every source file in the ML module. Here's my understanding and proposed grouping.

## Context Summary

The project is a **Solidity smart contract vulnerability detector** using a dual-path GNN + CodeBERT (LoRA) architecture. The current state:

- **v5.0** cleared validation gates (tuned F1=0.5828) but **failed behavioral tests catastrophically** (15% detection, 0% specificity)
- **v5.1 Phase 0** code fixes applied (interface selection, function-level pooling, aux loss weight, CFG failure counter)  
- **Dataset deduplication** discovered 34.9% cross-split leakage — original 68K→44K clean rows
- Re-extraction completed (44,140/44,420 canonical = 99.4% coverage)
- v5.1 retraining not yet done (pending Phase 2 data additions + Phase 3 retrain)

---

## Proposed Audit Groups

I'm grouping by **attack surface and failure domain** — what can go wrong, how it propagates, and who depends on it. This is not a directory-based grouping (which would miss cross-cutting vulnerabilities), but a **threat-model grouping**.

### **Group 1: Schema & Feature Contracts** 🎯 *The Foundation*
**Files:** `graph_schema.py`, `hash_utils.py`  
**Why:** These are the **single source of truth** that every other module depends on. If `NODE_FEATURE_DIM`, `NUM_EDGE_TYPES`, `NODE_TYPES`, or the hash system is wrong, everything downstream is silently broken. The v5.0 `type_id` normalisation bug (raw 0-12 vs /12.0) is exactly this class of failure. The dual-hash system (path-MD5 vs content-MD5) was the root cause of the 34.9% cross-split leakage.
**Adversarial lens:** Are constants truly locked? Are there implicit contracts not enforced by assertions? Can hash collisions occur? Are sentinel values (-1.0) consistently used across all consumers?

### **Group 2: Graph Extraction Engine** 🎯 *The Data Pipeline Gate*
**Files:** `graph_extractor.py`, `ast_extractor.py`  
**Why:** This is the **largest and most complex module** (~890 lines for graph_extractor alone). The v5.0 interface selection bug (ghost graphs for ~10% of training data) and the CFG_NODE_WRITE mapping bug (ReferenceVariable vs StateVariable) both lived here. It directly produces the training data — if it's wrong, the model learns wrong things silently.
**Adversarial lens:** Are edge cases handled (empty functions, interfaces, abstract contracts, multi-contract files, solc version mismatches)? Is the deterministic sort actually deterministic? Can `_build_control_flow_edges` silently produce broken graphs? Is the `_select_contract` fix complete?

### **Group 3: GNN Architecture & Signal Propagation** 🎯 *The Core Reasoner*
**Files:** `gnn_encoder.py`, `sentinel_model.py`  
**Why:** The GNN is **why v5 exists** — the whole point was to encode execution order via CFG subgraphs. The three-phase architecture (Phase 1: structural, Phase 2: CONTROL_FLOW, Phase 3: reverse-CONTAINS) is the most critical design decision. The v5.0 GNN eye gradient collapse to 6.7% by epoch 43 shows the architecture has failure modes. The Phase 3 edge embedding symmetry (same type-5 for forward and reverse CONTAINS) is a known limitation.
**Adversarial lens:** Is the three-phase signal path actually correct? Can Phase 3 reverse-CONTAINS produce wrong signals? Is function-level pooling correctly implemented? Are residual connections correct? What happens on graphs with zero CFG nodes?

### **Group 4: Transformer, Fusion & Classifier** 🎯 *The Token Path & Decision*
**Files:** `transformer_encoder.py`, `fusion_layer.py`  
**Why:** The transformer eye dominates gradient share (40%+ by epoch 44) and the fused eye takes 52% — together they provide 92% of the learning signal. The cross-attention fusion has a documented NaN issue on all-PAD inputs. The fusion layer's `to_dense_batch` padding + masking has had 8 documented bug fixes. If the fusion is leaking or short-cutting, the GNN's structural signal is irrelevant.
**Adversarial lens:** Is LoRA actually training the right layers? Can the fusion layer shortcut around the GNN eye? Are padding masks correct in cross-attention? Is the classifier's 384-dim input well-conditioned?

### **Group 5: Training Pipeline & Loss Functions** 🎯 *The Optimisation*
**Files:** `trainer.py`, `focalloss.py`, `train.py` (CLI)  
**Why:** The trainer has 25+ documented audit fixes. The pos_weight sqrt scaling, aux loss wiring, resume logic (patience counter, optimizer state, scheduler), gradient clipping, and early stopping are all places where subtle bugs can silently degrade training. The v5.0 aux_loss_weight=0.1 was too low and caused GNN collapse.
**Adversarial lens:** Is the aux loss correctly applied per-eye? Is the pos_weight correctly computed from the deduped dataset? Can resume corrupt training state? Is the weighted sampler actually fixing class imbalance or just adding noise?

### **Group 6: Dataset & Data Loading** 🎯 *The Training Data Integrity*
**Files:** `dual_path_dataset.py`, tokenizer scripts (`tokenizer.py`), `build_multilabel_index.py`, `create_splits.py`, `dedup_multilabel_index.py`, `verify_splits.py`, `create_cache.py`  
**Why:** The 34.9% cross-split leakage was a data integrity failure, not a model failure. The dual-hash system, the OR-merge deduplication, the stratified splitting, and the RAM cache are all places where data corruption can silently inflate metrics. The `edge_attr` shape guard ([E,1] → [E] squeeze) is a known source of silent bugs.
**Adversarial lens:** Is the deduplication correct? Are splits truly leak-free? Can the cache go stale? Is the hash pairing (graph ↔ token) reliable? What happens when a graph exists but no token file, or vice versa?

### **Group 7: Inference & Production API** 🎯 *The Deployment Surface*
**Files:** `api.py`, `predictor.py`, `preprocess.py`, `drift_detector.py`, `cache.py`  
**Why:** This is the actual attack surface — the HTTP API, the checkpoint loading, the sliding-window tokenization, the drift detection. The predictor's architecture-aware model construction has had 7 documented bugs. The preprocess module's temp file management (SIGKILL-safe cleanup) is security-sensitive. The cache can serve stale results if schema version changes.
**Adversarial lens:** Can the API be crashed with malformed input? Can stale cache produce wrong predictions? Is the drift detector's KS test correctly implemented? Are there race conditions in concurrent requests?

### **Group 8: Operational Scripts & Augmentation** 🎯 *The Tooling*
**Files:** `generate_safe_variants.py`, `extract_augmented.py`, `reextract_graphs.py`, `auto_experiment.py`, `tune_threshold.py`, `promote_model.py`, `validate_graph_dataset.py`, `compute_drift_baseline.py`, `analyse_truncation.py`, `manual_test.py`, `run_augmentation.sh`, `run_overnight_experiments.py`  
**Why:** These scripts produce and validate training data, tune thresholds, and manage model lifecycle. A bug in `generate_safe_variants.py` can produce contracts that are NOT actually safe (the two-step verification gate is critical). A bug in `tune_threshold.py` can produce over-optimistic thresholds. A bug in `validate_graph_dataset.py` can miss corrupted graphs.
**Adversarial lens:** Is the safe-variant verification gate actually verifying? Can threshold tuning overfit? Is the re-extraction script handling solc version fallbacks correctly?

---

## Execution Order & Rationale

| Order | Group | Rationale |
|-------|-------|-----------|
| **1** | Schema & Feature Contracts | Everything depends on this. Must be verified first. |
| **2** | Graph Extraction Engine | Produces training data. If wrong, everything downstream is wrong. |
| **3** | GNN Architecture | The core innovation. Most complex code. Most likely to have subtle bugs. |
| **4** | Transformer, Fusion & Classifier | Completes the model picture. |
| **5** | Training Pipeline & Loss | How the model is optimised. Depends on Groups 1-4 being correct. |
| **6** | Dataset & Data Loading | Data integrity — the 34.9% leak lesson makes this critical. |
| **7** | Inference & Production API | Deployment surface — separate concern from training. |
| **8** | Operational Scripts | Tooling — can be audited last since it doesn't affect core correctness. |

---
