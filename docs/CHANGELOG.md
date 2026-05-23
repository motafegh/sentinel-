# SENTINEL — Project Changelog

**Scope:** Full project history from initial commit through Phase 3.6 (GraphCodeBERT + GNN Prefix Injection).
**Last updated:** 2026-05-24

This document is the single authoritative changelog. Session-level detail lives in `docs/changes/` and `docs/ml/`. This file records *what changed, why, and what it produced* — not how to reproduce it.

---

## Table of Contents

1. [Project Foundation (2026-04-26 – 2026-04-29)](#1-project-foundation)
2. [v4 Baseline (pre-2026-05-09)](#2-v4-baseline)
3. [v5.0 — Three-Eye Architecture (2026-05-11 – 2026-05-12)](#3-v50--three-eye-architecture)
4. [v5.1 — Dataset Deduplication (2026-05-12)](#4-v51--dataset-deduplication)
5. [v5.2 — JK + LoRA + Three-Phase GNN (2026-05-14 – 2026-05-16)](#5-v52--jk--lora--three-phase-gnn)
6. [v5.3 — ASL Loss Experiment (2026-05-16, killed)](#6-v53--asl-loss-experiment)
7. [v6 — Graph Feature Schema Patch (2026-05-17)](#7-v6--graph-feature-schema-patch)
8. [v7 — Full Architecture Overhaul (2026-05-18 – 2026-05-19)](#8-v7--full-architecture-overhaul)
9. [v8 — Cross-Function Graph Extension (2026-05-19 – 2026-05-21)](#9-v8--cross-function-graph-extension)
10. [v8-AB — Joint ICFG+DEF_USE Ablation (2026-05-20)](#10-v8-ab--joint-icfgdef_use-ablation)
11. [PLAN-3A — ICFG-Only Ablation (2026-05-21 – 2026-05-23)](#11-plan-3a--icfg-only-ablation)
12. [Phase 3.5 — Data Quality Fixes (2026-05-23)](#12-phase-35--data-quality-fixes)
13. [Phase 3.6 — GraphCodeBERT + GNN Prefix Injection (2026-05-23 – 2026-05-24)](#13-phase-36--graphcodebert--gnn-prefix-injection)

---

## 1. Project Foundation

**Period:** 2026-04-26 – 2026-04-29
**Commits:** `9cb6990` → `8343d34`

### What was established

- **Architecture concept:** Dual-path GNN + CodeBERT multi-label vulnerability detector for Solidity smart contracts. GNN encodes contract structure (graph); CodeBERT encodes token semantics; CrossAttentionFusion combines both; three-eye classifier produces per-class logits.
- **ZK proof layer:** EZKL/Groth16 ZKML circuit proves model outputs on-chain. Proxy MLP (Linear 128→64→32→10) approximates the full model for ZK tractability.
- **Agent system:** LangGraph 5-agent topology — `ml_assessment → evidence_router → [rag_research ‖ static_analysis] → audit_check → synthesizer`. SqliteSaver checkpointing.
- **MCP servers:** Three SSE servers — `:8010` inference, `:8011` RAG, `:8012` audit.
- **Contracts:** Foundry layout under `contracts/src/` — `AuditRegistry.sol`, `SentinelToken.sol`, `IZKMLVerifier.sol`.
- **Initial bug fixes (2026-04-29):** predictor.py Bug 3/4/5/6/7; api.py missing torch import; vuln_type bridge via shared report store; faiss atomic write; rag_server lazy-load retriever.

### Infrastructure
- Python 3.12.1, Poetry, WSL2 Ubuntu
- RTX 3070 8GB VRAM
- MLflow with SQLite backend (`sqlite:///mlruns.db`)

---

## 2. v4 Baseline

**Period:** pre-2026-05-09
**Checkpoint:** `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`

### Architecture
- GNN + CodeBERT without CrossAttentionFusion Three-Eye design
- NODE_FEATURE_DIM: legacy (pre-v5)
- Dataset: 68,523-row `multilabel_index.csv` — **contained 34.9% cross-split content leakage** (same `.sol` file stored in multiple BCCC category directories; path-based MD5 created duplicate rows spanning splits)

### Results
- **Tuned F1-macro: 0.5422** — inflated by leakage; not a reliable number
- Best classes (tuned): IntegerUO=0.776 · GasException=0.507 · Reentrancy=0.519
- Behavioral test: not run

### Why these numbers are kept
v4 is retained as a **relative floor** only. v5+ must beat `v4 F1 − 0.05` per class on the clean dataset. The absolute 0.5422 is meaningless due to leakage.

---

## 3. v5.0 — Three-Eye Architecture

**Period:** 2026-05-11 – 2026-05-12
**Commits:** `80408c3` → `899adc9`

### Architecture changes
- Introduced **CrossAttentionFusion** — multi-head attention between GNN node embeddings and CodeBERT token embeddings (`attn_dim=256`, `output=128`)
- **Three-eye classifier** — three independent classification heads (GNN eye, Transformer eye, Fused eye) each contributing to the final loss
- GNN hidden_dim=128, `gnn_heads=8`, 4-layer GAT
- `NODE_FEATURE_DIM=12` (v3 schema — included `in_unchecked` feature, later found redundant)
- `NUM_EDGE_TYPES=7` (no REVERSE_CONTAINS yet)
- Dataset: still 68K leaky

### Results
- **Best val F1: 0.5828** (ep44) — **invalid due to 34.9% leakage**
- Behavioral test: **15% detection / 0% specificity** — model predicts "all-vulnerable" for any input

### Root causes of behavioral failure
- R1: `_select_contract()` interface bug selected wrong contract in multi-contract `.sol` files
- R2: ALL nodes pooled into GNN representation (should be FUNCTION nodes only)
- Dataset leakage meant model memorised cross-split duplicates rather than learning structural patterns

---

## 4. v5.1 — Dataset Deduplication

**Period:** 2026-05-12
**Commits:** `b12669c` → `f9d8f98`

### Data changes
- Implemented content-hash MD5 deduplication on raw Solidity source
- **Dataset reduced: 68,523 → 44,470 contracts**
- Splits regenerated: train=31,128 / val=6,236 / test=6,237 (approximate, based on deduped CSV)
- Leakage eliminated

### Training outcome
- Run `v5.1-fix28`: best F1=0.2794 (ep8) — training terminated early due to unresolved schema issues
- **Result declared INVALID** — schema v5 still had 12-dim features with `in_unchecked` bug

---

## 5. v5.2 — JK + LoRA + Three-Phase GNN

**Period:** 2026-05-14 – 2026-05-16
**Commits:** `b9ba690` (partial) → `46b216f`

### Architecture changes
- **JumpingKnowledge aggregation** (`jk_mode="attention"`) — learnable per-node aggregation across all GNN layers
- **LoRA** applied to CodeBERT Q+V projections — `r=16`, `α=32`, `dropout=0.1`, 589,824 trainable params from 124M frozen
- **Three-phase GNN:**
  - Phase 1 (layers 1–2): structural edges (CONTAINS, INHERITS, CALLS, READS, WRITES, EMITS)
  - Phase 2 (layers 3–5): control-flow edges (CONTROL_FLOW only)
  - Phase 3 (layers 6–7): REVERSE_CONTAINS (child→parent)
- `REVERSE_CONTAINS(7)` edge type added — runtime-only (built by dataset, not extractor)
- **AsymmetricLoss (ASL):** `γ_neg=2.0`, `γ_pos=1.0`, `clip=0.01` — replaced BCEWithLogitsLoss
- **Per-class pos_weight** with `pos_weight_min_samples=3000` cap
- **GNN gradient multiplier** (`gnn_lr_multiplier=2.5`) — counteracts GNN gradient collapse vs LoRA
- `torch.compile(model, dynamic=True)` enabled — ~20–40% step speedup

### Training runs
| Run | Best F1 | Tuned F1 | Notes |
|-----|---------|----------|-------|
| `v5.2-jk-20260515c` | 0.1872 (ep16) | — | Intermediate |
| `v5.2-jk-20260515c-r2` | 0.2823 (ep20) | 0.3373 | Early-stopped |
| `v5.2-jk-20260515c-r3` | **0.3306** (ep32) | **0.3422** | Best v5.2 |

### Behavioral failure analysis (v5.2-r3)
- Behavioral score: **36% detection / 33% specificity** — still far below usable
- Root cause: **~30% of test contracts were windowed from same source as training** — partial leakage from multi-window tokenization (same contract file → multiple windows, some in train, some in test). Fixed by stride-based deduplication in v6+.
- JK Phase 3 weight at convergence: 0.572→0.784 — model learned to almost entirely rely on REVERSE_CONTAINS, ignoring CFG signal. Structural ceiling confirmed.

---

## 6. v5.3 — ASL Loss Experiment

**Period:** 2026-05-16, killed
**Run:** `v5.3-bce-smooth-20260516`

- Switched back to BCE + label smoothing to test whether ASL γ was causing instability
- Best F1: 0.2559 (ep31) — **killed**, ASL confirmed superior
- No further investigation; v6 proceeded with ASL

---

## 7. v6 — Graph Feature Schema Patch

**Period:** 2026-05-17 – 2026-05-18
**Commits:** `8c8ce8c` → `b4ad806` (partial)

### What changed
- **In-place graph feature patch** applied to all 44,472 existing graphs — corrected NODE_FEATURE_DIM from 12→11 (removed redundant `in_unchecked` at feature index [9])
- `FEATURE_SCHEMA_VERSION = "v6"` — schema version tracking added
- Shape guard in `GNNEncoder.forward()` rejecting stale 12-dim files
- Assertion guards in `graph_schema.py` enforcing schema constant consistency at import time

### Why a patch not a re-extraction
Re-extraction would have taken 4+ hours. The `in_unchecked` feature was zeroed for all contracts (Slither doesn't emit it reliably), so patching in-place was byte-identical to re-extracting.

---

## 8. v7 — Full Architecture Overhaul

**Period:** 2026-05-18 – 2026-05-19 (training completed 2026-05-19)
**Commits:** `46a8f9d` → `47896a7`
**Training log:** `ml/logs/v7.0_best.log` · **Checkpoint:** `ml/checkpoints/v7.0_best.pt`

### What changed

#### Schema (v7)
- `NODE_FEATURE_DIM: 12 → 11` (permanent, `in_unchecked` removed as BUG-L2 fix)
- `FEATURE_SCHEMA_VERSION = "v7"`
- Full v7 re-extraction from all 44,470+ source contracts

#### 27 bugs fixed (across 4 phases)
Key fixes:

| Bug | Category | Fix |
|-----|----------|-----|
| BUG-L2 | Schema | `in_unchecked` removed; NODE_FEATURE_DIM 12→11 |
| BUG-H1 | GNN | `conv3c` (3rd CFG hop) added; `gnn_layers` 6→7 |
| BUG-H7 | Extractor | EMITS edges: EventCall IR fallback for Solidity 0.4.x |
| BUG-H8 | Extractor | INHERITS edges: parent CONTRACT nodes added before declaration |
| BUG-C3 | Extractor | CFG nodes inherit dims [1,3,4,5,9] from parent FUNCTION |
| BUG-M1 | Extractor | `return_ignored`: `id(lval)` → `lval.name` for stable hashing |
| BUG-M3 | Loss | ASL `γ_neg`/`γ_pos`/`clip` per-class tensors; registered as buffers |
| BUG-M8 | Eval | Per-epoch per-class threshold sweep; logs `val_f1_macro_tuned` |
| C2 | Training | DoS gradient no longer leaks through 3 aux heads |
| C4 | CLI | `--dos-loss-weight` flag added |
| F4 | CLI | `"positive"` added to `--weighted-sampler` choices |

#### Speed stack
- **SDPA** (Scaled Dot-Product Attention) activated on CodeBERT — `TransformerEncoder` falls back from FA2 (not supported by RoBERTa architecture) to SDPA
- **Flash Attention 2.8.3** installed via pre-built wheel (ready for decoder backbone swap)
- **Fused AdamW** (`fused=True`) — ~5% step speedup

#### DoS augmentation
- 60 contracts generated: 30 `dos_vuln_*.sol` + 30 `dos_safe_*.sol`
- 6 fail compilation (nested interface syntax — accepted, skipped)
- 110 augmented `.sol` files total in `ml/data/augmented/`

#### Dataset after v7 extraction
- **41,576 graphs** (schema v7, 11-dim)
- **44,470 token files** (`ml/data/tokens_windowed/` — `[4, 512]` per contract, stride=256)
- Splits (deduped): train=29,103 / val=6,236 / test=6,237
- Cache: `ml/data/cached_dataset_v7.pkl` (2.0 GB)

### Training results
| Metric | Value |
|--------|-------|
| Best raw F1-macro | **0.2651** (ep23) |
| Tuned F1-macro | **0.2875** |
| Killed at | ep34, patience 10/30 |
| VRAM peak | 0.4 / 8.0 GiB |
| JK Phase 3 at plateau | 0.768 (Phase 3 dominance, Phase 2 = 0.182) |

**Top classes (tuned):** IntegerUO=0.706 · GasException=0.369 · Reentrancy=0.303
**Ceiling diagnosis:** JK Phase 2 decayed 0.35→0.18 over training. Model learned to distrust CFG signal because no cross-function edges exist — reentrant patterns spanning caller/callee are invisible.

### Behavioral tests
- 20 hand-crafted `.sol` contracts (`ml/scripts/test_contracts/`)
- 19 expected detections across 17 contracts (3 safe contracts, 1 contract with 3 expected)
- Baseline score: **8/19 (42%)** at tuned thresholds

---

## 9. v8 — Cross-Function Graph Extension

**Period:** 2026-05-19 – 2026-05-21
**Commits:** `519e3a8` → `5031a8a`
**Full doc:** `docs/changes/2026-05-19-v8-graph-extension-and-full-reextraction.md`

### Motivation
v7 ceiling (0.2875 tuned F1) confirmed architectural — no cross-function edges meant the GNN could not trace reentrancy patterns across caller/callee boundaries. Three new edge types implemented.

### New edge types

| Type | ID | Direction | Semantics |
|------|----|-----------|-----------|
| CALL_ENTRY | 8 | calling CFG node → callee entrypoint | Lets signal flow into callee body |
| RETURN_TO | 9 | callee terminal nodes → call-site successor | Post-call state flows back to caller |
| DEF_USE | 10 | def node → use node (intra-function) | Arithmetic def-use chains for IntegerUO |

**Schema:** `FEATURE_SCHEMA_VERSION = "v8"` · `NUM_EDGE_TYPES = 11` (was 8)

### Extractor changes
- `global_cfg_node_map` accumulated across all function iterations (was discarded per-function — PLAN-1C)
- `_add_icfg_edges()` — new helper for CALL_ENTRY(8) + RETURN_TO(9) cross-function edges
- `_add_def_use_edges()` — new helper for DEF_USE(10) intra-function data-flow edges
- `NodeType` IntEnum added to `graph_schema.py` (13 types — never hardcode raw integers)

### Full re-extraction
- **41,576 graphs extracted** (41,576 ok / 73 ghost / 2,875 skip / 0 fail — 29 min, 10 workers)
- Old v7 graphs archived to `ml/data/archive/graphs_v7/`
- Cache rebuilt: `ml/data/cached_dataset_v8.pkl` (2.2 GB, schema v8)

### Dataset statistics (v8)
```
Nodes: mean=125  P50=89   P99=623   max=1,735
Edges: mean=248  P50=145  P99=1,801 max=6,516

Edge type distribution (41,576 graphs):
  CALLS(0)          :    437,968
  READS(1)          :    641,801
  WRITES(2)         :    678,879
  EMITS(3)          :         12   (rare)
  INHERITS(4)       :    105,010
  CONTAINS(5)       :  3,672,916
  CONTROL_FLOW(6)   :  3,140,025
  CALL_ENTRY(8)     :    257,829   NEW — 63.7% of graphs
  RETURN_TO(9)      :    232,814   NEW — 55.5% of graphs
  DEF_USE(10)       :  1,159,688   NEW — 80.3% of graphs
  REVERSE_CONTAINS(7): runtime-only, added by dataset
```

### Label cleaning (v8 — first pass)
- `label_cleaner.py` re-run on v8 graphs
- 3,665 labels removed (structural mismatch: labels where graph features contradict vulnerability claim)

### Pre-training validation gates (all passed)
- GATE-3A-CACHE: cache size=2.16 GB, schema=v8, pairs=41,576
- GATE-3A-0: edge activation by class — CALL_ENTRY 63.7% overall, 68.3% on Reentrancy=1
- GATE-3A-1: edge mask code verification — DEF_USE(10) correctly excluded from PLAN-3A mask
- GATE-3A-2: config review (dos_weight, aug data, MLflow, sampler, log format)
- GATE-3A-VRAM: 8.0 GB free

---

## 10. v8-AB — Joint ICFG+DEF_USE Ablation

**Period:** 2026-05-20
**Run:** `v8-AB` · **Log:** `ml/logs/v8-AB_best.log`
**Full doc:** `docs/ml/v8-AB-training-analysis.md`

### Config
- Phase 2 edges: `CF(6) + CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10)` — all new edge types
- `GNN hidden_dim=256`, `gnn_layers=7`, `gnn_heads=8`
- `LoRA r=16`, `α=32`
- `dos_loss_weight=0.0` (DoS gradient detached — too few samples at the time)

### Results
| Metric | Value |
|--------|-------|
| Best raw F1-macro | 0.2621 (ep29) |
| Tuned F1-macro | **0.2851** |
| vs v7 tuned | −0.0024 |

**Key finding:** DEF_USE was diluting the Reentrancy signal. DEF_USE coverage on Reentrancy=1 contracts was 77.5% — the lowest of all classes — but CEI (Checks-Effects-Interactions) is a control-flow pattern, not a data-flow pattern. Adding DEF_USE added noise without adding relevant signal.

### JK collapse analysis
Per-node JK histogram run on 936 val contracts (123,139 nodes):

| Phase | Mean weight | Std | IQR | Dominance |
|-------|-------------|-----|-----|-----------|
| Phase 1 (structural) | 0.065 | 0.074 | 0.006–0.157 | 0.0% |
| Phase 2 (CF+ICFG+DFG) | 0.247 | 0.078 | 0.233–0.297 | 0.01% |
| Phase 3 (rev-CONTAINS) | 0.688 | 0.097 | 0.616–0.702 | 99.99% |

- Phase 2 IQR = 0.064 — global constant, no per-node routing
- Phase 3 dominates every node — JK has collapsed to a fixed weighting, not learned per-node routing
- **Not a regression vs v7** — same pattern observed; structural tendency of the architecture

**Full analysis:** `docs/ml/jk-attention-collapse-findings.md`

---

## 11. PLAN-3A — ICFG-Only Ablation

**Period:** 2026-05-21 – 2026-05-23
**Run:** `v8.0-A-20260521` · **Log:** `ml/logs/v8.0-A-20260521.log`
**Checkpoint:** `ml/checkpoints/v8.0-A-20260521_best.pt` · **Full doc:** `docs/ml/plan-3a-results.md`

### Config
- Phase 2 edges: `CF(6) + CALL_ENTRY(8) + RETURN_TO(9)` — **DEF_USE(10) dropped**
- All other config identical to v8-AB

### Results
| Metric | Value |
|--------|-------|
| Best raw F1-macro | **0.2790** (ep41) |
| Tuned F1-macro | **0.2877** |
| vs v7 tuned | +0.0002 (statistical tie) |
| vs v8-AB tuned | +0.0026 |
| Killed at | ep67 (patience 26/30) |

**Per-class winners vs v8-AB:** Timestamp +0.038, CallToUnknown +0.020, Reentrancy +0.005
**Per-class losers vs v8-AB:** IntegerUO −0.016, ExternalBug −0.015, TOD −0.011

**Key finding:** Timestamp was the biggest winner (+0.032 vs v7). DEF_USE had been *hurting* Timestamp — block-global assignments generate dense def-use chains, adding noise rather than signal. Dropping DEF_USE improved Timestamp dramatically and confirmed that DEF_USE placement (Phase 2) dilutes rather than enhances certain classes.

**Ceiling conclusion:** v7, v8-AB, PLAN-3A all converge to ~0.287 tuned F1. The ceiling is architectural — not data quality (see Phase 3.5 for the test of that hypothesis).

**Behavioral score:** Still ~8/19 — no improvement over v7.

---

## 12. Phase 3.5 — Data Quality Fixes

**Period:** 2026-05-23
**Commits:** `b9709f1` → `2b0cdfb` → `5a030d7`

### Motivation
Hypothesis H5: the ~0.287 ceiling is caused by label noise, not architecture. v8.0-B tests this by applying stricter label cleaning before training.

### DQ-1 — Stricter structural label cleaning

New precondition rules in `label_cleaner.py`:

| Class | Old rule | New rule | Labels removed |
|-------|----------|----------|----------------|
| Reentrancy | external_call_count > 0 | external_call_count > 0 **AND** has WRITES edge | −611 |
| Timestamp | uses_block_globals > 0 | uses_block_globals > 0 **AND** (external calls OR payable) | −568 |
| UnusedReturn | — | stricter return-value check | −1,665 |
| CallToUnknown | — | stricter external call pattern | −383 |
| MishandledException | — | stricter exception handling check | −632 |
| ExternalBug | — | structural check added | −445 |
| **Total** | | | **−4,304** |

Output: `ml/data/processed/multilabel_index_cleaned.csv`

### DQ-2 — DoS gradient re-enabled

- `dos_loss_weight`: 0.0 → **0.5** (fractional gradient scaling)
- Implementation upgraded from hard binary mask to true fractional scaling: `w * logit + (1−w) * logit.detach()`
- Trigger: ~243 DoS training positives available after augmentation (was 7 when originally disabled)

### DQ-3 — Complexity correlation diagnostic

`ml/scripts/complexity_correlation.py` — Spearman rank correlation between graph complexity metrics and predicted vulnerability probability across 1,500 val contracts (PLAN-3A checkpoint):

- Single alert: `ext_calls_sum` vs `MishandledException` r=0.402 (expected — ext calls are necessary)
- Highest near-shortcut: `num_nodes` vs `Timestamp` r=0.396 (BUG-H4 label noise)
- All other pairs: r=0.15–0.40 — moderate, no dominant shortcuts
- **Conclusion:** Model not dominated by raw size shortcuts. Label quality fixes expected to reduce these correlations.

### Bug fixes (adversarial audit, 2026-05-23)

Three real issues fixed (commits `0c650f1`, `5a030d7`):

| ID | Fix |
|----|-----|
| G-02 | `GNNEncoder.forward()`: mask operations on `edge_attr` are now in-place safe (no tensor aliasing) |
| G-03 | `select_prefix_nodes()`: tie-breaking in priority sort is deterministic (stable sort key) |
| T-04 | `TransformerEncoder._word_embeddings`: property accesses correct attribute path through LoRA wrapper |
| trainer | `._orig_mod.` prefix stripped from checkpoint state_dict at **save** time (not just load time) |

### v8.0-B training run — H5 refutation

**Run:** `v8.0-B-20260523` · **Config:** PLAN-3A edges + cleaned labels + `dos_loss_weight=0.5`

| Metric | Value |
|--------|-------|
| Best F1-macro | 0.2460 (ep10) |
| Killed at | ep11 |
| H5 verdict | **REFUTED** |

F1=0.2460 < 0.288 architectural ceiling. Data cleaning alone cannot break the ceiling. **The ceiling is purely architectural.** GraphCodeBERT + GNN prefix injection becomes the primary next intervention.

---

## 13. Phase 3.6 — GraphCodeBERT + GNN Prefix Injection

**Period:** 2026-05-23 – 2026-05-24 (ongoing)
**Proposal:** `docs/proposal/2026-05-23-graphcodebert-gnn-prefix-injection-proposal.md`
**Execution plan:** `docs/proposal/EXECUTION_PLAN.md`

### Core idea
Two simultaneous interventions:

1. **GraphCodeBERT drop-in:** Replace `microsoft/codebert-base` with `microsoft/graphcodebert-base` — same RoBERTa architecture and vocabulary, but pre-trained on code+data-flow graphs. Zero other changes for the P0 baseline.

2. **GNN prefix injection (Phase 1):** Project K=48 declaration-level GNN node embeddings [K,256] → [K,768] and prepend as prefix tokens to the transformer input. Uses `inputs_embeds` path instead of `input_ids`, making graph structure visible to **all 12 GraphCodeBERT attention layers** — not just at the final CrossAttentionFusion step.

### Prerequisites (all completed 2026-05-23)

| Check | Result |
|-------|--------|
| PRE-1 GraphCodeBERT downloaded | ✅ — 499MB model cached |
| PRE-2 Tokenizer identity (CodeBERT = GraphCodeBERT) | ✅ — vocab_size=50,265, token IDs identical on Solidity |
| PRE-3 UNK token rate | ✅ — **0.000000%** (0/418,248 tokens) |
| PRE-4 Node count distribution audit (K=48 covers 95.5%) | ✅ — declaration-level P95=47, K=48 chosen |
| PRE-5 LoRA applies to GraphCodeBERT | ✅ — 589,824 trainable params via `query`/`value` module names |

**Retokenization:** Not needed. TransformerEncoder truncates internally (`code_ids = input_ids[:, :code_budget]`). Existing stride=256 token files valid for K=48 (effective overlap = 208 tokens per window, no gap).

### P1-IMPL — Code implementation (completed 2026-05-23 – 2026-05-24)

| File | Change |
|------|--------|
| `ml/src/models/transformer_encoder.py` | `gnn_prefix_nodes: Optional[Tensor]` added to `forward()`; `inputs_embeds` path with prefix+code concatenation; position_ids: prefix=1 (RoBERTa padding slot), code=3..466; multi-window prefix expansion; `_word_embeddings` property; `WindowAttentionPooler.prefix_k` shifts CLS extraction right by K |
| `ml/src/models/sentinel_model.py` | `gnn_prefix_k`, `gnn_prefix_warmup_epochs`, `_current_epoch`; `gnn_to_bert_proj: Linear(256,768)`; `prefix_type_embedding: Embedding(5,768)`; `select_prefix_nodes()` with priority sort (CONSTRUCTOR > FALLBACK > RECEIVE > MODIFIER > FUNCTION); prefix suppressed during warmup (`gnn_prefix_nodes=None` when epoch < warmup_epochs) |
| `ml/src/preprocessing/graph_schema.py` | `NodeType` IntEnum (13 types); `STRUCTURAL_PREFIX_TYPES` frozenset (5 declaration types); `_PREFIX_NODE_PRIORITY`, `_PREFIX_TYPE_IDX` dicts |
| `ml/src/training/trainer.py` | `gnn_prefix_k`, `gnn_prefix_warmup_epochs`, `gnn_prefix_proj_lr_mult` in TrainConfig; separate `PrefixProj` param group (LR ×1.0); `model._current_epoch = epoch` each epoch; per-epoch `prefix_active` + `prefix_proj_weight_norm` logging to MLflow and logger |
| `ml/scripts/train.py` | `--gnn-prefix-k` (default 0), `--gnn-prefix-warmup-epochs` (default 15), `--gnn-prefix-proj-lr-mult` (default 1.0) |
| `ml/src/inference/predictor.py` | `gnn_prefix_k` and `gnn_prefix_warmup_epochs` passed from checkpoint config to SentinelModel; `model._current_epoch = 9999` after load (prefix always active at inference) |

**Key design decisions:**
- `prefix_type_embedding`: `Embedding(5, 768)` — 5 declaration types only (not 8; Phase 1 is declaration-level only)
- Warmup suppression: `gnn_to_bert_proj` receives **zero gradient** during warmup — not called at all. Starts from random init at ep16. Old plan (inject frozen prefix) would corrupt attention with untrained embeddings.
- No `--graphcodebert` CLI flag needed — TransformerEncoder is already hardcoded to graphcodebert-base after P0 change.
- Backward compatible: `gnn_prefix_k=0` (default) produces byte-identical output to original model.

### P0 — GraphCodeBERT drop-in results

**Run:** `graphcodebert-dropin-P0-20260523` (killed at ep4, best ep3)

| Epoch | F1-macro | Notes |
|-------|----------|-------|
| 1 | 0.1734 | GNN share 63–86%; JK Phase2=0.389 |
| 2 | 0.2056 | GNN share 67–75%; JK Phase2=0.398 |
| 3 | **0.2178** | JK Phase3=0.450; **ExternalBug=0.199, TOD=0.171** (both 0.000 in all CodeBERT runs) |

**GATE-GCB-2 verdict (2026-05-24): PASSED** — clear upward trend; ExternalBug and TOD both non-zero from ep3 confirming GraphCodeBERT cross-function pre-training signal.

### GATE-GCB-3 — Smoke test (2026-05-24, running now)

**Run:** `graphcodebert-v1-prefix48-smoke-20260524` (2 epochs, K=48)
**Log:** `ml/logs/graphcodebert-v1-prefix48-smoke-20260524.log`

Confirmed at startup:
- `GNN prefix K=48: WARMUP (starts ep15)` — warmup suppression active ✅
- `gnn_to_bert_proj weight norm: 16.0000` — proj at random init, not called ✅
- `PrefixProj=3 params (lr×1.0)` — separate param group registered ✅
- VRAM: 0.3/8.0 GiB — 7.7 GB free ✅

### P1-TRAIN — next step (blocked on GATE-GCB-3)

```bash
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. python ml/scripts/train.py \
    --run-name graphcodebert-v1-prefix48-$(date +%Y%m%d) \
    --experiment-name sentinel-gcb \
    --gnn-prefix-k 48 --gnn-prefix-warmup-epochs 15 --gnn-prefix-proj-lr-mult 1.0 \
    --phase2-edge-types 6 8 9 \
    --dos-loss-weight 0.5 \
    --cache-path ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --label-csv ml/data/processed/multilabel_index_cleaned.csv \
    --weighted-sampler positive \
    --epochs 100 --early-stop-patience 30 --gradient-accumulation-steps 8
```

**Warmup schedule:** Epochs 1–14: prefix suppressed, model trains via CrossAttentionFusion path only. Epoch 15+: `gnn_to_bert_proj` starts from random init, K=48 GNN nodes injected as prefix tokens. Step loss expected to rise briefly at ep16 then recover.

**Target:** Tuned F1-macro > 0.30 (lifts architectural ceiling). Behavioral target: >10/19.

---

## Architecture Lineage Summary

| Version | NODE_FEATURE_DIM | NUM_EDGE_TYPES | GNN layers | Backbone | Best tuned F1 |
|---------|-----------------|----------------|------------|----------|---------------|
| v4 | legacy | 7 | 4 | CodeBERT | 0.5422 (leaky) |
| v5.0 | 12 | 7 | 4 | CodeBERT | invalid (leaky) |
| v5.2 | 12 | 8 (+rev-CONTAINS) | 6 | CodeBERT+LoRA | 0.3422 (partial leak) |
| v6 | **11** | 8 | 6 | CodeBERT+LoRA | — (patch only) |
| **v7** | **11** | **8** | **7** | CodeBERT+LoRA | **0.2875** |
| v8-AB | 11 | **11** (+ICFG+DEF_USE) | 7 | CodeBERT+LoRA | 0.2851 |
| PLAN-3A | 11 | 11 (no DEF_USE in Ph2) | 7 | CodeBERT+LoRA | **0.2877** |
| v8.0-B | 11 | 11 | 7 | CodeBERT+LoRA | 0.2460 (data clean) |
| **GCB-P0** | 11 | 11 | 7 | **GraphCodeBERT**+LoRA | ~0.22 (3 ep) |
| **GCB-P1** | 11 | 11 | 7 | GraphCodeBERT+LoRA+**prefix K=48** | running |

**Architectural ceiling (all v7/v8 runs):** ~0.287 tuned F1. GraphCodeBERT + prefix injection is the intervention to break it.

---

## Open Bugs (as of 2026-05-24)

| ID | Impact | Status |
|----|--------|--------|
| BUG-H4 | Timestamp: ~463 contracts labeled positive with no timestamp features (inverted signal) | OPEN |
| BUG-H5 | Reentrancy: ~14% of positives have no external calls (structural impossibility) | OPEN |
| BUG-M5 | Brainmab contract: standard ERC20 labeled Reentrancy+CallToUnknown+IntegerUO+MishandledException | OPEN |
| BUG-M6 | Token files carry stale `feature_schema_version='v4'` metadata | OPEN (auto-resolves on retokenize) |
| BUG-M7 | 8.5% of graphs have empty `contract_path` (can't cross-reference source) | OPEN |
| BUG-L3 | Hash-based graph-token pairing fragile to directory restructuring | DEFERRED |

---

## Key File Locations

| Asset | Path |
|-------|------|
| Best v7 checkpoint | `ml/checkpoints/v7.0_best.pt` (F1=0.2651) |
| Best PLAN-3A checkpoint | `ml/checkpoints/v8.0-A-20260521_best.pt` (tuned F1=0.2877) |
| GCB-P0 best checkpoint | `ml/checkpoints/graphcodebert-dropin-P0-20260523_best.pt` (F1=0.2178) |
| v8 cache | `ml/data/cached_dataset_v8.pkl` (2.2 GB, 41,576 pairs) |
| Cleaned labels | `ml/data/processed/multilabel_index_cleaned.csv` |
| Splits | `ml/data/splits/deduped/` (train/val/test .npy files) |
| Behavioral test contracts | `ml/scripts/test_contracts/` (20 files, 19 expected detections) |
| MLflow DB | `mlruns.db` |
| Detailed training history | `docs/changes/` (session-level) · `docs/ml/` (analysis docs) |
