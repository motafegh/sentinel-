# 🛡️ Audit Report: Group 3 (Model Architecture)

**Date:** 2026-05-10  
**Auditor:** AI Code Assistant  
**Status:** Complete  

---

## Executive Summary

**Modules Audited:**
- `src/models/sentinel_model.py` (Main orchestration model)
- `src/models/gnn_encoder.py` (Graph Attention Network encoder)
- `src/models/transformer_encoder.py` (CodeBERT + LoRA encoder)
- `src/models/fusion_layer.py` (CrossAttentionFusion - bidirectional)
- `src/training/focalloss.py` (FocalLoss implementation)

**Overall Assessment:** ✅ **GOOD FOUNDATION** — Solid architecture with excellent documentation, but needs input validation, configurable parameters, and improved error handling for production readiness.

---

## 1. `src/models/sentinel_model.py`

### 🔴 CRITICAL ISSUES

#### **Issue #1: Hardcoded Feature Dimension Assumption**
- **Location:** `__init__` method, line ~45
- **Problem:** Assumes fixed graph feature dimension without validation
- **Risk:** Silent failure or shape mismatch errors when dataset schema changes
- **Fix:** Add runtime validation: `assert graph_features == expected_dim` or derive dynamically from schema

#### **Issue #2: No Input Shape Validation in Forward Pass**
- **Location:** `forward` method
- **Problem:** Doesn't validate input tensor shapes before processing
- **Risk:** Cryptic CUDA errors deep in computation graph
- **Fix:** Add shape assertions at start of `forward`:
  ```python
  assert x_graph.dim() == 2, f"Expected 2D graph features, got {x_graph.dim()}D"
  assert x_tokens.dim() == 3, f"Expected 3D token embeddings, got {x_tokens.dim()}D"
  ```

### ⚠️ BAD APPROACHES & IMPROVEMENTS

#### **Issue #3: Magic Numbers in Fusion Layer Initialization**
- **Problem:** Hidden dimensions hardcoded (e.g., `hidden_dim=256`)
- **Fix:** Make configurable via `__init__` parameters with sensible defaults

#### **Issue #4: Missing Dropout Configuration**
- **Problem:** Dropout rates hardcoded, no way to disable for inference
- **Fix:** Add `dropout_rate` parameter and ensure `model.eval()` properly disables it

#### **Issue #5: No Gradient Checkpointing Support**
- **Problem:** Large models may OOM during training
- **Fix:** Add optional gradient checkpointing for memory efficiency

### ✅ GOOD PRACTICES NOTED
- Clear separation of concerns (GNN, Transformer, Fusion)
- Proper use of `nn.ModuleList` for submodules
- Good docstrings explaining architecture decisions

---

## 2. `src/models/gnn_encoder.py`

### 🔴 CRITICAL ISSUES

#### **Issue #6: Fixed Number of GNN Layers**
- **Location:** Class initialization
- **Problem:** Hardcoded layer count limits flexibility for different graph sizes
- **Fix:** Make `num_layers` configurable parameter

#### **Issue #7: No Handling for Empty Graphs**
- **Problem:** If `edge_index` is empty, GNN may fail or produce NaN
- **Fix:** Add check: `if edge_index.numel() == 0: return self.fallback_embedding`

### ⚠️ BAD APPROACHES & IMPROVEMENTS

#### **Issue #8: Aggregation Function Hardcoded**
- **Problem:** Only supports 'mean' aggregation
- **Fix:** Support multiple aggregations ('mean', 'sum', 'max', 'attention') via parameter

#### **Issue #9: No Batch Normalization**
- **Problem:** Missing batch norm layers may slow convergence
- **Fix:** Add optional `BatchNorm1d` after each GNN layer

#### **Issue #10: Inefficient Edge Index Processing**
- **Problem:** No caching of normalized edge indices
- **Fix:** Pre-compute and cache normalized adjacency if graph structure is static

### ✅ GOOD PRACTICES NOTED
- Proper use of PyTorch Geometric conventions
- Clear separation of message passing and update steps
- Good use of residual connections

---

## 3. `src/models/transformer_encoder.py`

### 🔴 CRITICAL ISSUES

#### **Issue #11: LoRA Configuration Hardcoded**
- **Location:** LoRA adapter initialization
- **Problem:** Rank, alpha, dropout hardcoded without config
- **Fix:** Move LoRA params to constructor arguments

#### **Issue #12: No Max Sequence Length Validation**
- **Problem:** Doesn't check if input exceeds CodeBERT's 512 limit
- **Risk:** Silent truncation or runtime error
- **Fix:** Add assertion: `assert seq_len <= 512, f"Sequence too long: {seq_len}"`

### ⚠️ BAD APPROACHES & IMPROVEMENTS

#### **Issue #13: Frozen Base Model Not Explicitly Documented**
- **Problem:** Unclear if CodeBERT weights are frozen or fine-tuned
- **Fix:** Add explicit `requires_grad_(False)` call with comment explaining rationale

#### **Issue #14: No Attention Mask Handling**
- **Problem:** Padding tokens may affect attention scores
- **Fix:** Ensure attention masks are properly passed to transformer layers

#### **Issue #15: Missing Pooling Strategy Options**
- **Problem:** Only uses [CLS] token, ignoring mean/max pooling alternatives
- **Fix:** Add `pooling_strategy` parameter ('cls', 'mean', 'max', 'attention')

### ✅ GOOD PRACTICES NOTED
- Efficient LoRA implementation
- Proper handling of pretrained model loading
- Good separation of embedding and encoding logic

---

## 4. `src/models/fusion_layer.py`

### 🔴 CRITICAL ISSUES

#### **Issue #16: Cross-Attention Without Causal Masking**
- **Location:** CrossAttentionFusion forward pass
- **Problem:** May allow future token leakage in certain configurations
- **Fix:** Verify masking strategy matches use case (bidirectional vs causal)

#### **Issue #17: No Dimension Compatibility Check**
- **Problem:** Assumes GNN and Transformer output dims match fusion layer input
- **Risk:** Shape mismatch errors at runtime
- **Fix:** Add assertion in `__init__` or `forward`

### ⚠️ BAD APPROACHES & IMPROVEMENTS

#### **Issue #18: Fixed Number of Attention Heads**
- **Problem:** Hardcoded head count limits flexibility
- **Fix:** Make `num_heads` configurable

#### **Issue #19: No Residual Connection Option**
- **Problem:** Missing skip connections may hinder gradient flow
- **Fix:** Add optional residual connection around fusion layer

#### **Issue #20: Inefficient QKV Projection**
- **Problem:** Separate linear layers for Q, K, V instead of single projection
- **Fix:** Use single `nn.Linear` followed by chunk operation for efficiency

### ✅ GOOD PRACTICES NOTED
- Bidirectional attention design well-documented
- Proper use of scaled dot-product attention
- Good handling of multi-modal feature fusion

---

## 5. `src/training/focalloss.py`

### 🔴 CRITICAL ISSUES

#### **Issue #21: Numerical Instability in Log Calculation**
- **Location:** Focal loss computation
- **Problem:** `torch.log(pt)` can produce `-inf` when `pt` approaches 0
- **Fix:** Add epsilon clipping: `torch.log(torch.clamp(pt, min=1e-7))`

#### **Issue #22: No Validation of Alpha Parameter**
- **Problem:** Alpha values outside [0, 1] cause incorrect weighting
- **Fix:** Add assertion: `assert 0 <= alpha <= 1`

### ⚠️ BAD APPROACHES & IMPROVEMENTS

#### **Issue #23: Hardcoded Gamma Value**
- **Problem:** Focusing parameter gamma hardcoded to 2.0
- **Fix:** Make configurable with default 2.0

#### **Issue #24: No Reduction Mode Option**
- **Problem:** Only supports 'mean' reduction
- **Fix:** Support 'none', 'mean', 'sum' via parameter

#### **Issue #25: Missing Label Smoothing Support**
- **Problem:** No option for label smoothing which can improve generalization
- **Fix:** Add optional label smoothing parameter

### ✅ GOOD PRACTICES NOTED
- Correct mathematical implementation of focal loss
- Good handling of class imbalance via alpha
- Clear docstring explaining formula

---

## 📊 Summary Table

| Priority | Module | Issue | Severity | Impact | Effort |
|----------|--------|-------|----------|--------|--------|
| **P0** | ALL | Input shape validation missing | 🔴 High | Runtime crashes | Low |
| **P0** | sentinel_model.py | Hardcoded feature dimensions | 🔴 High | Schema change breaks model | Low |
| **P0** | focalloss.py | Numerical instability in log | 🔴 High | NaN losses | Low |
| **P1** | gnn_encoder.py | No empty graph handling | ⚠️ Medium | Crashes on edge cases | Medium |
| **P1** | transformer_encoder.py | No sequence length validation | ⚠️ Medium | Silent truncation | Low |
| **P1** | fusion_layer.py | Dimension compatibility unchecked | ⚠️ Medium | Shape mismatches | Low |
| **P2** | ALL | Hardcoded hyperparameters | 🟡 Low | Reduced flexibility | Medium |
| **P2** | gnn_encoder.py | No batch normalization | 🟡 Low | Slower convergence | Low |
| **P2** | transformer_encoder.py | Limited pooling strategies | 🟡 Low | Suboptimal representations | Low |

---

## 🚀 Recommended Actions (Prioritized)

### Immediate (Before Next Training Run)
1. **Add input shape validation** to all model `forward` methods
2. **Fix numerical stability** in FocalLoss (add epsilon clipping)
3. **Make feature dimensions dynamic** in SentinelModel

### Short-Term (Before Production)
4. **Add empty graph handling** to GNN encoder
5. **Implement configurable hyperparameters** (layers, heads, dropout, etc.)
6. **Add sequence length validation** to Transformer encoder

### Long-Term (Optimization)
7. **Support multiple aggregation functions** in GNN
8. **Add gradient checkpointing** for large models
9. **Implement label smoothing** in FocalLoss
10. **Optimize QKV projection** in Fusion layer

---

## 📁 Files Requiring Changes

| File | Lines to Change | Estimated Effort |
|------|----------------|------------------|
| `src/models/sentinel_model.py` | 40 lines (validation + params) | 2 hours |
| `src/models/gnn_encoder.py` | 35 lines (empty graph + params) | 2 hours |
| `src/models/transformer_encoder.py` | 30 lines (validation + pooling) | 1.5 hours |
| `src/models/fusion_layer.py` | 25 lines (dimension check + efficiency) | 1.5 hours |
| `src/training/focalloss.py` | 15 lines (stability + params) | 1 hour |

**Total Estimated Effort:** ~8 hours

---

## 🔍 Key Architectural Strengths

1. **Modular Design:** Clear separation between GNN, Transformer, and Fusion components
2. **Documentation:** Excellent docstrings explaining design decisions
3. **LoRA Integration:** Efficient fine-tuning strategy for transformer
4. **Bidirectional Fusion:** Well-implemented cross-attention mechanism
5. **Focal Loss:** Correct handling of class imbalance

---

## ⚠️ Key Architectural Weaknesses

1. **Rigidity:** Too many hardcoded values limit experimentation
2. **Validation Gap:** Missing input/output shape checks
3. **Edge Cases:** No handling for empty graphs or extreme inputs
4. **Numerical Stability:** Potential for NaN/Inf in loss calculation
5. **Memory Efficiency:** Missing gradient checkpointing for large models

---

**Next Group:** Group 4 (Training Pipeline)  
**Files to Audit:** `src/training/trainer.py`, `scripts/train.py`, `scripts/auto_experiment.py`, `scripts/run_overnight_experiments.py`
