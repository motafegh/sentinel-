# SENTINEL Proposal v2.0 — Audit Report

**Date:** 2026-06-16  
**Module:** `ml`  
**Phase:** Run 13 preparation  
**Auditor:** Ali  
**Target:** `docs/proposal/ml/2026-06-15_ml_Run13_prep_proposal_next_gen_ml_methods.md` (v2.0, 1129 lines)

---

## 1. Executive Summary

Audited the merged proposal v2.0 against actual source code. Found **12 factual discrepancies** — mostly stale docstrings propagating outdated schema values into the proposal's baseline table. None are fatal to the proposal's recommendations, but the baseline numbers in §2 are wrong and will mislead anyone using them as a reference.

**Severity breakdown:**
- **Critical (3):** GNN parameter count overstated 4×, total trainable overstated 2×, schema constants wrong
- **Moderate (5):** Stale docstrings in sentinel_model.py and gnn_encoder.py, paper title mismatches
- **Low (4):** Minor approximation errors, missing edge type count

---

## 2. Critical Findings

### 2.1 GNN Parameter Count Overstated 4×

**Proposal §2 claims:** GNN Encoder = "~2.5M"  
**Actual (verified via `GNNEncoder()` instantiation):** 615,136 params (~0.6M)

```
GNN total params:     615,136
GNN trainable params: 615,136
```

The proposal's "~2.5M" figure is **4.1× the actual count**. This propagates to:
- §2 baseline table (line 76)
- §5.2 "SENTINEL's GNN (2.5M params) is small enough that pre-training converges fast" (line 289)
- §5.3 "Small fast model (GNN-only, no transformer, ~2.5M params)" (line 332)
- §6 ranking (implicit in all VRAM/complexity estimates)

**Root cause:** The docstring at `gnn_encoder.py:42` says "~2.5M GNN" — this is stale. The actual GNN has never been 2.5M; the 8-layer GAT with 256 hidden dim and 8 heads produces ~615K params.

**Impact on recommendations:** None — the GNN is actually *smaller* than claimed, making contrastive pre-training even more feasible. But all cost/VRAM estimates in the proposal are based on wrong numbers.

**Fix:** Update §2 table row "GNN Encoder" from "~2.5M" to "~615K". Update all references.

---

### 2.2 Total Trainable Parameters Overstated 2×

**Proposal §2 claims:** "~5M trainable / ~129M total"  
**Actual (verified via `SentinelModel()` instantiation):**

```
Total params:     127,109,266  (~127M, not ~129M)
Trainable params:   2,463,634  (~2.5M, not ~5M)
Frozen params:    124,645,632  (~125M)
```

The proposal claims ~5M trainable — the actual is ~2.5M. A 2× overstatement.

**Breakdown of actual trainable:**
| Component | Trainable | Proposal claim |
|-----------|-----------|----------------|
| GNN Encoder | 615,136 | ~2.5M |
| Transformer LoRA | 589,824 | ~590K |
| CrossAttentionFusion | ~660K | ~1.5M |
| Eyes + Classifier + Aux | ~600K | ~330K |
| **Total** | **2,463,634** | **~5M** |

**Impact:** The "5M trainable" claim inflates the apparent model capacity. The model is actually quite parameter-efficient — 2.5M trainable out of 127M total (1.9% active). This is *good news* for the proposal's goals (more room for LoRA ensemble, MoE, etc.) but should be stated accurately.

---

### 2.3 Schema Constants Wrong in Proposal §2

**Proposal §2 claims (line 12 of sentinel_model.py docstring):**
- `NODE_FEATURE_DIM=11`
- `_GNN_IN_DIM=27`
- `NUM_NODE_TYPES=13` (implied by "type_embedding nn.Embedding(13,16)")
- `NUM_EDGE_TYPES=11` (line 13)

**Actual (verified):**
- `NODE_FEATURE_DIM=12` (v9 schema)
- `_GNN_IN_DIM=28` (12 + 16 type embedding)
- `NUM_NODE_TYPES=14` (IDs 0–13)
- `NUM_EDGE_TYPES=12`

All four constants are wrong. The sentinel_model.py docstring at line 12 (`NODE_FEATURE_DIM=11`) and gnn_encoder.py at lines 236, 243 (`_GNN_IN_DIM (27)`) are stale — they reference v8 schema values.

**Impact:** Anyone reading the docstrings to understand the architecture will get wrong dimensions. The code itself is correct (it imports from `graph_schema.py` which has the right values).

---

## 3. Moderate Findings

### 3.1 Stale Docstrings in sentinel_model.py

`sentinel_model.py:12` states `NODE_FEATURE_DIM=11 (on disk), _GNN_IN_DIM=27 (model-internal after type emb)`. Both are wrong for v9 schema. The code dynamically computes `_GNN_IN_DIM` from `graph_schema.NODE_FEATURE_DIM + _TYPE_EMB_DIM`, so the runtime values are correct — only the docstring is stale.

---

### 3.2 Stale Docstrings in gnn_encoder.py

`gnn_encoder.py:42` states "~2.5M GNN" — actual is 615K.  
`gnn_encoder.py:236` states "_GNN_IN_DIM (27) × 256 = 6,912 parameters" — should be 28 × 256 = 7,168.  
`gnn_encoder.py:243` states `in_channels=_GNN_IN_DIM,  # 27 (11 features + 16 type-embedding)` — should be 28 (12 features + 16 type-embedding).

---

### 3.3 SigGate-GT Paper Title Mismatch

**Proposal §5.9 (line 616) references:** "SigGate-GT" method  
**Proposal §10 (line 1029) cites:** `arXiv:2604.17324` — "SigGate-GT: Taming Over-Smoothing"

**Actual paper title:** "Capacity-Controlled Global Attention for Graph Transformers" (`arXiv:2604.17324`). The name "SigGate-GT" appears in the abstract but is not the paper title. The paper is about sigmoid-gated global attention, not specifically about "taming over-smoothing" (that's a different paper).

**Impact:** Minor — the method description is accurate, but the paper title is wrong. If someone looks up the paper, they won't find "SigGate-GT: Taming Over-Smoothing".

---

### 3.4 ContractShield Architecture Mismatch

**Proposal §4.5 (line 202) claims:** ContractShield uses "xLSTM for opcodes, GATv2 for CFG"  
**Actual (per `arXiv:2604.02771`):** ContractShield uses **CodeBERT** (not GraphCodeBERT), xLSTM for opcodes, GATv2 for CFG. The proposal's description is correct about xLSTM and GATv2 but omits the CodeBERT component and implies GraphCodeBERT.

**Impact:** Low — the method description is accurate for the components mentioned. The full architecture includes CodeBERT which isn't mentioned.

---

### 3.5 Proposal §5 Methods Not Clearly Distinguished as Novel vs Established

The proposal introduces several methods as if they are novel SENTINEL contributions:
- **CP-MIAD** and **CP-MIAR** (§5.5 area) — referenced in context summary but not clearly defined in the proposal text
- **ETN** (§4.6) — "Evidential Transformation Network" described as "post-hoc module" without clear reference
- **F-EDL** (§5.7) — described as "Flexible EDL" from NeurIPS 2025 without clear reference

These appear to be established methods from the literature, not novel SENTINEL contributions. The proposal should clarify which methods are existing techniques being applied to SENTINEL vs which are novel contributions.

---

## 4. Low Findings

### 4.1 NUM_EDGE_TYPES Not in Proposal

The proposal doesn't mention NUM_EDGE_TYPES anywhere. Actual value is 12 (not 11 as in stale docstrings). The gnn_encoder.py docstring at line 218 says "11 edge types including REVERSE_CONTAINS(7)" but the schema has 12 edge types.

---

### 4.2 Transformer Frozen Count Approximation

**Proposal claims:** "125M frozen"  
**Actual:** 124,645,632 (~124.6M)  
**Difference:** ~354K params — negligible rounding.

---

### 4.3 Total Params Approximation

**Proposal claims:** "~129M total"  
**Actual:** 127,109,266 (~127M)  
**Difference:** ~2M — within acceptable rounding for a summary table.

---

### 4.4 Proposal §9 "Don't add more than 1–2GB VRAM" Context

The proposal says "8GB RTX 3070 is already near capacity with the full 4-eye model (~5.5GB during training)". This is not verifiable from source code alone — it depends on batch size, sequence length, and graph size. The claim is plausible but unverified.

---

## 5. Verification Summary

### What the Proposal Gets Right

| Claim | Verified? |
|-------|-----------|
| Transformer: GraphCodeBERT (125M frozen) + LoRA (r=16, Q+V) | ✅ |
| LoRA trainable: ~590K (actual: 589,824) | ✅ |
| Frozen: ~125M (actual: 124,645,632) | ✅ |
| ASL loss (γ_neg=2.0, γ_pos=1.0, clip=0.01) | Need to verify |
| AdamW lr=2e-4, weight_decay=1e-2 | Need to verify |
| AMP BF16 | ✅ (line 2: "SDPA active") |
| Batch 8, grad accum ×8 | Need to verify |
| 3-tier thresholds (0.55/0.25) | Need to verify |
| JK attention aggregation | ✅ |
| 8-layer GAT (2+3+3 phases) | ✅ |
| CrossAttentionFusion (node↔token MHA) | ✅ |
| 4 auxiliary heads (training only) | ✅ |
| sigmoid applied externally | ✅ |

### What the Proposal Gets Wrong

| Claim | Actual | Severity |
|-------|--------|----------|
| GNN ~2.5M params | 615,136 | Critical |
| Total trainable ~5M | 2,463,634 | Critical |
| NODE_FEATURE_DIM=11 | 12 | Critical |
| _GNN_IN_DIM=27 | 28 | Critical |
| NUM_NODE_TYPES=13 | 14 | Moderate |
| NUM_EDGE_TYPES=11 | 12 | Low |
| Total ~129M | ~127M | Low |
| SigGate-GT paper title | Wrong title | Moderate |
| ContractShield uses GraphCodeBERT | Uses CodeBERT | Moderate |

---

## 6. Recommendations

### Immediate Fixes (Before Run 13)

1. **Update §2 baseline table** with verified numbers:
   - GNN: 615K (not ~2.5M)
   - Total trainable: 2.5M (not ~5M)
   - NODE_FEATURE_DIM: 12 (not 11)
   - _GNN_IN_DIM: 28 (not 27)
   - NUM_NODE_TYPES: 14 (not 13)

2. **Fix stale docstrings** in sentinel_model.py and gnn_encoder.py (these affect code readability for anyone new to the project).

3. **Correct paper references** — SigGate-GT paper title, ContractShield architecture.

### Before Run 14

4. **Re-estimate VRAM and cost** based on actual 2.5M trainable params — the model is smaller than claimed, so some estimates may be optimistic.

5. **Clarify which methods are novel vs applied** — important for documentation and for distinguishing SENTINEL contributions from literature survey.

---

## 7. Appendix: Actual Model Architecture (Verified)

```
SentinelModel v8.1
├── GNNEncoder (615,136 params)
│   ├── type_embedding: Embedding(14, 16)
│   ├── edge_embedding: Embedding(12, 64)
│   ├── input_proj: Linear(28, 256, bias=False)
│   ├── conv1: GATConv(28→32, heads=8, self-loops, edge_dim=64)
│   ├── conv2: GATConv(256→32, heads=8, self-loops, edge_dim=64)
│   ├── conv3: GATConv(256→64, heads=4, no self-loops, edge_dim=64)
│   ├── conv3b: GATConv(256→64, heads=4, no self-loops, edge_dim=64)
│   ├── conv3c: GATConv(256→64, heads=4, no self-loops, edge_dim=64)
│   ├── conv4: GATConv(256→256, heads=1, no self-loops, edge_dim=64)
│   ├── conv4b: GATConv(256→256, heads=1, no self-loops, edge_dim=64)
│   ├── conv4c: GATConv(256→256, heads=1, no self-loops, edge_dim=64)
│   ├── jk_attn: Linear(256, 1)
│   └── phase LayerNorms (3×)
├── TransformerEncoder (589,824 trainable / 124,645,632 frozen)
│   ├── RobertaModel (microsoft/graphcodebert-base)
│   └── LoRA adapters on query + value (r=16, alpha=32)
├── CrossAttentionFusion (~660K params)
│   ├── node_proj: Linear(256, 256)
│   ├── token_proj: Linear(768, 256)
│   └── MHA: 8 heads, 256 dim
├── Eye projections (3× Linear(256→128))
├── Classifier: Linear(512→256) → ReLU → Dropout → Linear(256→10)
├── Aux heads: 4× Linear(128→10)
└── Total: 127,109,266 params (2,463,634 trainable)
```

Schema constants (v9):
- NODE_FEATURE_DIM=12
- _GNN_IN_DIM=28
- NUM_NODE_TYPES=14
- NUM_EDGE_TYPES=12
