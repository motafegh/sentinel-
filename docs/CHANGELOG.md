# SENTINEL — Project Changelog

**Scope:** Full project history from initial commit through Phase 3.6 (GraphCodeBERT + GNN Prefix Injection, IMP-* architectural fixes), agent Step E (cross_validator + graph topology), Phase 1 A1–A5 (hotspots, graph_inspector Phase 2, quick_screen, Aderyn deep-path, end-to-end smoke test), and pre-Run-5 implementation (interpretability fixes, label cleaning scripts, CEI aux loss, temperature scaling).
**Last updated:** 2026-05-31

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
14. [IMP-* Architectural Fixes + P1-TRAIN Run 2 (2026-05-24)](#14-imp--architectural-fixes--p1-train-run-2)
15. [P1-TRAIN Runs 3 and 4 (2026-05-25 – 2026-05-26)](#15-p1-train-runs-3-and-4)
16. [Three-Tier ML Output (2026-05-27)](#16-three-tier-ml-output)
17. [MLOps — Model Registry + Drift Detector (2026-05-27)](#17-mlops--model-registry--drift-detector)
18. [Agent Layer — Three-Tier Schema Integration (2026-05-27 – 2026-05-28)](#18-agent-layer--three-tier-schema-integration)
19. [Agent Layer — Step D: Graph Inspector (2026-05-29)](#19-agent-layer--step-d-graph-inspector)
20. [Agent Layer — Step E: cross_validator + Graph Topology (2026-05-29)](#20-agent-layer--step-e-cross_validator--graph-topology)
21. [Agent Layer — Phase 1 A1/A2/A3: Hotspots + GNN Attention + quick_screen (2026-05-30)](#21-agent-layer--phase-1-a1a2a3-hotspots--gnn-attention--quick_screen)
22. [Agent Layer — Phase 1 A4/A5: Aderyn deep-path + End-to-End Smoke Test (2026-05-30)](#22-agent-layer--phase-1-a4a5-aderyn-deep-path--end-to-end-smoke-test)

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
| GCB-P1-Run1 | 11 | 11 | 7 | GraphCodeBERT+LoRA+prefix K=48 | 0.2628 (ep27, killed ep28) |
| **GCB-P1-Run2** | 11 | 11 | **8** | GraphCodeBERT+LoRA+prefix K=48+**IMP-G1/G2/G3** | running |

**Architectural ceiling (all v7/v8 runs):** ~0.287 tuned F1. GraphCodeBERT + prefix injection is the intervention to break it.

---

## Open Bugs (as of 2026-05-24)

| ID | Impact | Status |
|----|--------|--------|
| BUG-H4 | Timestamp: ~463 contracts labeled positive with no timestamp features (inverted signal) | **DONE (2026-05-23)** — `label_cleaner.py` −568 Timestamp labels |
| BUG-H5 | Reentrancy: ~14% of positives have no external calls (structural impossibility) | **DONE (2026-05-23)** — `label_cleaner.py` −611 Reentrancy labels |
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
| GCB-P1-Run1 best checkpoint | `ml/checkpoints/graphcodebert-v1-prefix48-20260524_best.pt` (F1=0.2628, ep27) |
| GCB-P1-Run2 log | `ml/logs/graphcodebert-p1-run2-20260524.log` (PID 80610, running) |
| v8 cache | `ml/data/cached_dataset_v8.pkl` (2.2 GB, 41,576 pairs) |
| Cleaned labels | `ml/data/processed/multilabel_index_cleaned.csv` |
| Splits | `ml/data/splits/deduped/` (train/val/test .npy files) |
| Behavioral test contracts | `ml/scripts/test_contracts/` (20 files, 19 expected detections) |
| MLflow DB | `mlruns.db` |
| Detailed training history | `docs/changes/` (session-level) · `docs/ml/` (analysis docs) |

---

## 14. IMP-* Architectural Fixes + P1-TRAIN Run 2

**Period:** 2026-05-24
**Session changelog:** `docs/changes/2026-05-24-imp-all-fixes-and-p1-train-run2.md`
**Analysis doc:** `docs/ml/gcb-p1-run1-analysis-and-imp-all.md`

### Motivation: P1-TRAIN Run 1 autopsy

Run 1 (7-layer GNN, K=48, killed ep28, best ep27 F1=0.2628) was killed after JK weight analysis revealed three root causes preventing the architectural ceiling from being broken:

| JK signal | Trajectory | Root cause |
|-----------|-----------|------------|
| Phase 1 (0.058–0.063) | **Flat throughout** | 11→256 dim change in conv1 discards all raw feature information without a skip connection |
| Phase 2 (0.387→0.234) | **Collapsing** | conv3/conv3b/conv3c shared identical `cfg_mask` edge set → identical inputs → JK correctly downweights them |
| Phase 3 (0.550→0.707) | **Growing unchecked** | REVERSE_CONTAINS became sole dominant signal; CFG nodes had no Phase 3 context → representation gap in CrossAttentionFusion |

Additional finding: `gnn_to_bert_proj` weight norm stagnated at 2 BF16 ULPs (16.0000→16.2500) across 13 post-warmup epochs — quantization floor at norm≈16 prevents fine-grained gradient accumulation.

### IMP-G1 — Phase 2 Layer-Specific Edge Subsets

**File:** `ml/src/models/gnn_encoder.py`

Phase 2 layers previously shared the full `cfg_mask` edge set (CF ∪ CALL_ENTRY ∪ RETURN_TO). Identical input + identical edge set = identical representation — JK attention correctly collapses them.

Fix: construct three distinct edge subsets before Phase 2:
- `cf_only_ei/ea` — `edge_attr == 6` (CONTROL_FLOW only)
- `icfg_only_ei/ea` — `edge_attr ∈ {8, 9}` (CALL_ENTRY + RETURN_TO, cross-function only)
- `cfg_ei/ea` — CF ∪ ICFG joint (existing mask)

Phase 2 routing: `conv3(cf_only)` → `conv3b(icfg_only)` → `conv3c(cfg joint)`

### IMP-G2 — Phase 1 Input Projection Skip

**File:** `ml/src/models/gnn_encoder.py`

New `__init__` parameter:
```python
self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim, bias=False)  # 2,816 params
```

Forward block (before conv1):
```python
x_init = x
_proj_dtype = next(self.input_proj.parameters()).dtype
x_skip = self.input_proj(x_init.to(_proj_dtype)).to(x.dtype)
x = self.conv1(x_init, struct_ei, struct_ea)
x = self.relu(x + x_skip)
```

Also added dtype normalisation at `forward()` entry — required because GNN is always float32 while BERT load was polluting torch default dtype to BF16 (see DTYPE FIX below):
```python
_param_dtype = next(self.parameters()).dtype
if x.dtype != _param_dtype:
    x = x.to(_param_dtype)
```

### IMP-G3 — Phase 3 Bidirectional Context Pass

**File:** `ml/src/models/gnn_encoder.py`

Existing Phase 3 uses REVERSE_CONTAINS (CFG→FUNCTION upward). CFG nodes therefore received no Phase 3 context — only FUNCTION nodes were enriched by Phase 3. This created a representation gap that CrossAttentionFusion had to bridge across.

New downward CONTAINS pass (FUNCTION→CFG direction) after existing upward passes:
```python
self.conv4c = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                      add_self_loops=False, edge_dim=_edge_dim)
# ...in forward():
x4c = self.conv4c(x, fwd_contains_ei, fwd_contains_ea)
x   = x + self.dropout(x4c)
x   = self.phase_norm[2](x)
```

Architecture is now **8 layers** (2+3+3). `gnn_num_layers` default updated 7→8 in `GNNEncoder.__init__`, `TrainConfig.gnn_layers`, and `trainer.py` warning threshold.

### IMP-M1 — FUNCTION Node Secondary Sort

**File:** `ml/src/models/sentinel_model.py`

`select_prefix_nodes()` now sorts FUNCTION nodes by `external_call_count` (feature[10]) descending when K truncation occurs. Sort key: `(priority, -ext_call_count, local_idx)`.

### IMP-M2 Tier 2 — prefix_attention_mean Diagnostic

**Files:** `ml/src/models/transformer_encoder.py`, `ml/src/training/trainer.py`

`TransformerEncoder.forward()` gains `gnn_prefix_counts` and `output_attentions` parameters. When `output_attentions=True`, extracts mean attention weight from code positions → prefix positions: `attn[:, :, :, K:, :K].mean()`.

New `SentinelModel.compute_prefix_attention_mean()` decorated `@torch.no_grad()`. Trainer logs `prefix_attention_mean` to MLflow each epoch post-warmup; warns if < 0.002.

### IMP-M3 — Zero-Padded Prefix Attention Mask Fix

**Files:** `ml/src/models/sentinel_model.py`, `ml/src/models/transformer_encoder.py`

`select_prefix_nodes()` return type changed to `tuple[Tensor, Tensor]`:
- `prefix` — `[B, K, 768]` projected node embeddings (zero-padded for graphs with < K nodes)
- `node_counts` — `[B]` int tensor, real node count per graph

`TransformerEncoder` constructs a count-based prefix attention mask so padded positions get `attention_mask=0` instead of attending to meaningless zero vectors.

### IMP-D1 — return_ignored Temporal Ordering Fix

**File:** `ml/src/preprocessing/graph_extractor.py`

`_compute_return_ignored()` rewritten. Old implementation built `all_read_names` as a global set across the entire function — false negative when a TemporaryVariable name collided with an unrelated read elsewhere.

New implementation: iterate `func.nodes` in CFG topological order, building `all_ops_ordered = [(node, op) for node in nodes for op in (node.irs or [])]`. For each call op at `call_idx`, check if `lval_name` appears in any `later_op.read` at `all_ops_ordered[call_idx + 1:]`. Uses direct `func.nodes` access (not `getattr`) so `AttributeError` propagates to the sentinel return.

Re-extraction of all 41K graphs pending (separate run, not blocking training).

### BF16 Global Dtype Side-Effect Fix

**File:** `ml/src/models/transformer_encoder.py`

`AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)` calls `torch.set_default_dtype(bfloat16)` as a side effect, causing all `nn.Linear` created after BERT initialisation to have BF16 weights. Fixed:

```python
_prev_default_dtype = torch.get_default_dtype()
try:
    self.bert = AutoModel.from_pretrained(...)
finally:
    torch.set_default_dtype(_prev_default_dtype)
```

### Test suite (134/134 pass)

All 134 tests pass after the IMP-* changes. Key test fixes:

| Test file | Root cause | Fix |
|-----------|-----------|-----|
| `test_model.py` | `_StubTransformer` missing new params; `classifier` is now `Sequential` | Updated stub signature; `classifier[0].in_features`; shape `(N,128)→(N,256)` |
| `test_preprocessing.py` (schema) | v8 schema: `NODE_FEATURE_DIM=11`, `NUM_EDGE_TYPES=11`, `in_unchecked` removed | Updated all schema constant assertions |
| `test_preprocessing.py` (ReturnIgnored) | IMP-D1 API change: `func.slithir_operations` → `func.nodes[i].irs` | Tests now build `_make_mock_slither_node(irs=[...])` |
| `test_preprocessing.py` (integration) | `graph.x[:, 0]` stores `type_id / 12.0`; `.int()` always 0; raw integer comparisons invalid | Added `_type_ids()` / `_type_mask()` helpers using `(x * 12).round().long()` |
| `test_preprocessing.py` (integration) | Slither inserts intermediate CFG_NODE_OTHER between WRITE and CALL | Relaxed edge check to BFS reachability via `cf_adj` |
| `test_trainer.py` | Stale `scaler=scaler` kwarg removed from function signature | Removed from both `train_one_epoch()` calls |

### P1-TRAIN Run 2

**PID:** 80610  
**Log:** `ml/logs/graphcodebert-p1-run2-20260524.log`  
**Startup confirmed:** layers=8, gnn_prefix_k=48, warmup=15, VRAM 0.3/8.0 GiB, proj_norm=15.9853

Fresh start from random init (architecture changed — not resumed from Run 1). Target: tuned F1-macro > 0.30.

Monitor schedule:
- ep15: warmup ends
- ep16: prefix activates — expect brief loss spike then recovery
- ep17–20: `prefix_attention_mean` in MLflow; target > 0.005 by ep20
- ep17–20: Phase 2 JK weight — expect > 0.10 (vs 0.234 declining in Run 1)

---

## 15. P1-TRAIN Runs 3 and 4

**Period:** 2026-05-25 – 2026-05-26
**Commits:** `e022018`, `f1e228b`, `f2d0965`, `a6d4323`, `de4fe10`
**Log (Run 4):** `ml/logs/graphcodebert-p1-run4-20260525.log`
**Best checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt`

### Run 2 outcome (killed 2026-05-25, ep4)

Run 2 died from JK Phase 3 weight spiking to 86.6% (structural collapse) — a different manifestation of the same IMP-G1 routing ambiguity. Autopsy confirmed ASL `pos_weight` passed through was amplifying DoS gradient ×20,000 combined with ASL gamma (double-amp, named NC-4).

### Run 3 — 8L + entropy reg λ=0.01 (killed ep16)

**Config:** 8-layer GNN, `jk_entropy_reg_lambda=0.01`, ASL pos_weight still passed (NC-4 not yet identified as root cause).

| Metric | Value |
|--------|-------|
| Best F1 | 0.17 (stuck) |
| Killed at | ep16 |
| Kill reason | NC-4 double-amp + JK uniform collapse (λ=0.01 forced 33/33/33 routing, no per-node differentiation) |

### Run 4 pre-flight fixes (commits `e022018`, `f1e228b`)

| Fix ID | Description |
|--------|-------------|
| NC-4 REVERTED | `pos_weight` NOT passed to ASL — double-amp was Run 3 primary root cause (DoS 10× pos_weight × ASL gamma ≈ 20,000× amplification) |
| C-3 λ 0.01→0.005 | Halved JK entropy regularisation — λ=0.01 forced perfect uniform 33/33/33 (no per-node routing); λ=0.005 allows differentiation |
| NC-1 logging fix | Reports total params vs state-initialised; clears all params regardless of count |
| C2 scatter collision fix | `_scatter_to_dense`: `valid = local_idx < max_nodes` evaluated BEFORE clamp — excess nodes truly dropped, not collided at slot 1023 |
| C1 gnn_enc_norm | `_grad_norm(model.gnn)` logged every step alongside `gnn_eye_proj` — full backbone visibility |
| H5 aux_fused compile | `aux_fused` added to `torch.compile` submodule list |

Speed logging commit `f2d0965` added per-epoch step timing to log output (applied at ep3, Run 4 restarted from ep3 checkpoint).

### Run 4 — 8L + λ=0.005 + no ASL pos_weight (best ep32, F1=0.3362)

**Run 4 PID:** 82628  
**Launched:** 2026-05-25 14:52 local, restarted 16:55 at ep3.  
**Killed:** 2026-05-26 20:12 at ep44 — F1 locked 0.31–0.34 for 12 epochs, capacity ceiling reached.

| Epoch | F1 | Notes |
|-------|-----|-------|
| ep6 | 0.2555 | First strong epoch |
| ep9 | 0.2670 | Run 1 beaten |
| ep10 | 0.2787 | New best |
| ep13 | 0.3153 | 0.30 barrier broken; DoS+Timestamp first detections |
| ep21 | 0.3224 | Plateau resolved; prefix working |
| ep25 | **0.3272** | New best at time |
| ep32 | **0.3362** | Final best checkpoint |
| ep44 | — | Killed — 12-epoch F1 ceiling 0.31–0.34 |

**ep30 gate (11:23 2026-05-26):** CONTINUE — best F1=0.3272 (+0.0395 above PLAN-3A tuned peak 0.2877); loss still declining; patience 5/30.

**Top classes (ep32):** IntegerUO=0.647, Timestamp=0.329, GasException=0.317  
**Hard classes (ep32):** CallToUnknown≈0.247, ExternalBug≈0.246, TOD≈0.235

**Pending fixes committed but requiring restart to take effect:**
- JK STD threshold 0.05→0.015 (commit `a6d4323`) — prevents false-alarm warnings
- `prefix_attention_mean` unpack bug: 3-value vs 4-value (commit `de4fe10`)

### Training history summary

| Run | Config | Best ep | F1 | Kill reason |
|-----|--------|---------|-----|-------------|
| GCB-P0 | GraphCodeBERT drop-in | ep3 | 0.2178 | GATE-GCB-2 passed |
| GCB-P1-Run1 | GCB+K=48 prefix (7L GNN) | ep27 | 0.2628 | IMP fixes needed |
| GCB-P1-Run2 | 8L GNN, all IMP | ep4 | — | JK Phase3=86.6% structural collapse |
| GCB-P1-Run3 | 8L+entropy reg λ=0.01 | ep16 | 0.17 (stuck) | NC-4 double-amp + JK uniform collapse |
| **GCB-P1-Run4** | **8L+λ=0.005+no ASL pw** | **ep32** | **0.3362** | Killed ep44 — capacity ceiling |

---

## 16. Three-Tier ML Output

**Period:** 2026-05-27

### `ml/src/inference/predictor.py`

Added three-tier threshold logic:
- `TIER_CONFIRMED_THRESHOLD = 0.55` — probability above which a class is "confirmed_vulnerable"
- `TIER_SUSPICIOUS_THRESHOLD = 0.25` — probability above which a class is "suspicious"
- New `_format_result()` method producing structured output with keys: `confirmed`, `suspicious`, `probabilities`, `tier_thresholds`

### `ml/src/inference/api.py`

- `VulnerabilityResult` gains `tier` field (`"safe"` | `"suspicious"` | `"confirmed_vulnerable"`)
- `PredictResponse` gains full three-tier schema: `confirmed`, `suspicious`, `probabilities` dict
- `/health` endpoint returns `tier_thresholds`, `model_epoch`, `model_f1_val`

### Label values

| Value | Meaning |
|-------|---------|
| `"safe"` | All class probabilities < `TIER_SUSPICIOUS_THRESHOLD` |
| `"suspicious"` | At least one class in `[TIER_SUSPICIOUS_THRESHOLD, TIER_CONFIRMED_THRESHOLD)` |
| `"confirmed_vulnerable"` | At least one class ≥ `TIER_CONFIRMED_THRESHOLD` |

Legacy label `"vulnerable"` still accepted by consumers for backward compatibility.

---

## 17. MLOps — Model Registry + Drift Detector

**Period:** 2026-05-27

### `ml/scripts/promote_model.py`

- Production F1 gate: new model must beat current production F1 before promotion
- Baseline check: rejects promotion if source stage is `"warmup"` (unless `--require-baseline` bypassed)
- Threshold JSON artifact logged to MLflow run at promotion time
- `--require-baseline` CLI arg: enforce warmup-stage source check

### `ml/scripts/exercise_drift_detector.py`

Four-phase smoke test script:
1. Warm-up: 500 requests to establish service baseline
2. Baseline build: `drift_detector.record_baseline()` called on in-distribution contracts
3. In-distribution check: detector should NOT fire on clean contracts
4. Shift detection: distributional shift injected; detector expected to fire

Uses `np.random.default_rng(42)` for reproducibility. Exits 0 on pass, 1 on failure.

---

## 18. Agent Layer — Three-Tier Schema Integration

**Period:** 2026-05-27 – 2026-05-28
**Commits:** `4aca6ff`

### Routing (`agents/src/orchestration/routing.py`)

Added `_iter_class_probs()` backward-compatibility helper. Old `ml_result` format used a flat list of `(class, prob)` tuples; new format uses a `probabilities` dict. Helper handles both transparently.

### Nodes (`agents/src/orchestration/nodes.py`)

Updated nodes to consume three-tier schema:

| Node | Change |
|------|--------|
| `evidence_router` | Reads `ml_result["probabilities"]` dict; falls back via `_iter_class_probs()` |
| `rag_research` | Builds query from confirmed + suspicious class names |
| `static_analysis` | Scopes Slither detectors to confirmed + suspicious classes |
| `synthesizer` | Reads `confirmed`/`suspicious` keys; formats tier into final report |

### State (`agents/src/orchestration/state.py`)

- `external_call_summary` field added to `AuditState` TypedDict
- Comment on `ml_result` updated to document three-tier schema keys

### ExternalBug structural gap fix

`_extract_external_call_summary()` function added using `fn.high_level_calls` from Slither. `rag_research` uses the summary to build a targeted query. `synthesizer` includes the call graph summary in the LLM prompt context.

### Tests

- `agents/tests/test_graph_routing.py` completely rewritten — prior version referenced non-existent fields from the old flat schema
- `agents/src/mcp/servers/inference_server.py`: `_mock_prediction()` rewritten to return full three-tier schema (`confirmed`, `suspicious`, `probabilities`, `tier_thresholds`)

---

## 19. Agent Layer — Step D: Graph Inspector

**Period:** 2026-05-29

### New MCP server (`agents/src/mcp/servers/graph_inspector_server.py`)

Port 8013, SSE transport. Phase 1 implementation (true GNN attention weights deferred to Phase 2).

**Tool:** `get_graph_hotspots(contract_code: str, flagged_classes: list[str]) → list[HotspotResult]`

**Scoring engine (`_analyze_hotspots()`):**

| Signal | Weight |
|--------|--------|
| Slither detector hit for class | ×3.0 |
| Structural signal match for class | ×1.0 |
| External-facing function | ×0.5 |
| Complexity (cyclomatic proxy) | ×0.2 |

**Key constants:**
- `_CLASS_STRUCTURAL_SIGNALS` dict: 10 vulnerability classes → list of structural feature names to check
- `_DETECTOR_CLASS_MAP` dict: Slither detector name → vulnerability class

Returns top-20 hotspot nodes ranked by score. Each result includes: `node_id`, `function_name`, `score`, `signals` (list of matched signals), `detector_hits` (list of matched Slither detectors).

### New `graph_explain` node (`agents/src/orchestration/nodes.py`)

Calls `sentinel-graph-inspector:get_graph_hotspots` via MCP. Always runs alongside `rag_research` + `static_analysis` in the deep path (fan-out of three).

State written:
- `ml_hotspots`: flat list of hotspot dicts (top-20 nodes, score, signals)
- `graph_explanations`: per-class breakdown dict + graph statistics (`node_count`, `edge_count`, `hotspot_count`)

---

## 20. Agent Layer — Step E: cross_validator + Graph Topology

**Period:** 2026-05-29

### New `cross_validator` node (`agents/src/orchestration/nodes.py`)

LLM-adjudicated per-class verdicts. Runs after `audit_check`, before `synthesizer` (deep path only).

**Verdict values:**
| Verdict | Meaning |
|---------|---------|
| `CONFIRMED` | All evidence streams agree — vulnerable |
| `LIKELY` | ML + at least one tool agree |
| `DISPUTED` | ML and static analysis contradict each other |
| `WATCH` | Suspicious ML signal, no static confirmation |
| `SAFE` | No evidence across all streams |

**Evidence context built per class:**
- ML tier and probability from `ml_result`
- Slither findings filtered to class from `static_analysis` output
- RAG topics mentioning class from `rag_research` output
- Prior audit count for class from `audit_check` output

Falls back to empty dict on LLM failure — `synthesizer` uses rule-based fallback in that case.

### `synthesizer` update

When `cross_validator` verdicts are present in state, `synthesizer` uses them directly rather than rebuilding evidence assessment. Falls back to rule-based logic when verdicts dict is absent or empty.

### Graph topology update (`agents/src/orchestration/graph.py`)

New topology:

```
START → ml_assessment → evidence_router
         ↓ (deep path only)
  [rag_research ‖ static_analysis ‖ graph_explain]
         ↓
  audit_check → cross_validator → synthesizer → END
         ↓ (shallow path)
  synthesizer → END
```

`evidence_router` fans out to three parallel nodes on the deep path. `cross_validator` is inserted after `audit_check`.

### Test suite

23 new tests added:
- `graph_explain` node: hotspot retrieval, MCP error handling, empty result handling
- `cross_validator` node: all 5 verdict types, LLM failure fallback, partial class coverage
- Routing: fan-out-of-three verification, shallow path unchanged
- `build_graph`: topology structure assertions, node/edge count checks

**Total test count: 187 passing.**

### Architecture lineage update

| Version | NODE_FEATURE_DIM | NUM_EDGE_TYPES | GNN layers | Backbone | Best F1 |
|---------|-----------------|----------------|------------|----------|---------|
| v7 | 11 | 8 | 7 | CodeBERT+LoRA | 0.2875 tuned |
| v8-AB | 11 | 11 | 7 | CodeBERT+LoRA | 0.2851 tuned |
| PLAN-3A | 11 | 11 | 7 | CodeBERT+LoRA | 0.2877 tuned |
| GCB-P0 | 11 | 11 | 7 | GraphCodeBERT+LoRA | 0.2178 (3 ep) |
| GCB-P1-Run1 | 11 | 11 | 7 | GraphCodeBERT+LoRA+prefix K=48 | 0.2628 (ep27) |

---

## 21. Agent Layer — Phase 1 A1/A2/A3: Hotspots + GNN Attention + quick_screen

**Period:** 2026-05-30
**Commits:** `4bf5ba5` (A1) → `9ede400` (A2) → `94ca2c5` (A3)

### Motivation

Three gaps identified in the existing agent layer needed addressing before Run 5 data quality work:

1. **GNN attention was fake** — graph_inspector_server used Slither as a proxy for GNN attention hotspots. The real signal was in the model's GNN node embeddings, unused.
2. **ML blind spot** — contracts where all class probabilities fell below DEEP_THRESHOLDS were routed directly to the synthesizer with zero static tool evidence. A contract consistently scoring below 0.25 on all classes (plausible for obfuscated or atypical contracts) would receive a "safe" report based on ML alone.
3. **graph_inspector_server Phase 2** needed A1 to land first.

### A1 — `/hotspots` ML Inference Endpoint

**Files:** `ml/src/inference/predictor.py`, `ml/src/inference/api.py`, `ml/tests/test_api.py`

**What was added:**

`predict_with_hotspots(source_code: str) -> dict` method in `Predictor`:
- Runs the same windowed preprocessing as `predict_source`
- Calls `GNNEncoder.forward()` to get per-node embeddings `x` of shape `[N, 256]`
- Filters to function-type nodes (`_FUNC_TYPE_IDS_SET = {1,2,4,5,6}` — FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR)
- Computes L2 norm per node → normalized `[0,1]` score
- Maps node index → function name via `graph.node_metadata[i]["name"]`
- Returns full ML result + `"hotspots"` list + `"hotspot_stats"` dict

`POST /hotspots` FastAPI endpoint:
- Request: `{"source_code": str}`  (same Pydantic validator as `/predict`)
- Response: `HotspotsResponse` — `{hotspots, hotspot_stats, label, probabilities, confirmed, suspicious}`
- Top-20 functions sorted by embedding-norm score descending
- One round-trip: full ML result + hotspots together

9 new tests in `TestHotspotsEndpoint` — all 18 API tests pass.

### A2 — graph_inspector_server Phase 2 (Real GNN Attention)

**File:** `agents/src/mcp/servers/graph_inspector_server.py`

**What changed:**

Phase 2 fallback chain: `_analyze_hotspots_gnn()` → `_analyze_hotspots_slither()` → `_mock_hotspots()`

`_analyze_hotspots_gnn()`: calls `POST {_ML_API_URL}/hotspots` via httpx, transforms response into the inspector's hotspot format. Returns `None` on any failure (timeout, HTTP error, malformed response) so the fallback chain activates.

Health endpoint now reports `phase: "2"` and lists active backends.

Config: `_ML_API_URL = os.getenv("SENTINEL_ML_API_URL", "http://localhost:8000")`, `_HOTSPOTS_TIMEOUT = 60`.

### A3 — quick_screen Tier-0 Node

**Files:** `agents/src/orchestration/state.py`, `agents/src/orchestration/nodes.py`, `agents/src/orchestration/graph.py`, `agents/tests/test_graph_routing.py`

**What changed:**

New `quick_screen` LangGraph node inserted between `ml_assessment` and `evidence_router`:

```
ml_assessment → quick_screen → evidence_router → [fan-out or fast path]
```

The node runs on **every** contract (Tier 0), regardless of ML score. It:
- Runs Slither with `_SCREEN_SLITHER_DETECTORS` (15 High-impact detectors)
- Runs Aderyn subprocess; parses JSON output for H-*/C-* rule IDs
- Both tools are non-fatal — ImportError/FileNotFoundError/subprocess failure all yield empty list
- Returns `{"quick_screen_hits": {"slither": [...], "aderyn": [...]}}`

`_route_from_evidence_router` updated with two-signal gate:
- Fast path: ML safe AND screen clean → synthesizer
- Screen-escalated deep path: ML safe BUT screen fired → `["static_analysis", "graph_explain"]`
- Normal deep path: ML flagged → existing fan-out unchanged

`evidence_router` node updated to log quick_screen signal in `routing_decisions`.

`AuditState` updated: `quick_screen_hits: dict[str, list[str]]` field added.

**Tests:** 16 new tests — `TestQuickScreenNode` (7), routing escalation cases (5), graph node set check updated. 59/59 passing in `test_graph_routing.py`, 212/212 across all agent tests.

### Why the design choices

- **In-process** (not MCP): Slither and Aderyn are installed in the process. MCP hop for Tier-0 screening would add latency with no benefit — same rationale as `static_analysis` (established pattern in existing code).
- **High/Critical only**: Running all 90+ Slither detectors in quick_screen would produce too many informational hits and escalate clean contracts. Only High-impact detectors (reentrancy-eth, arbitrary-send-eth, etc.) justify escalation.
- **Non-fatal design**: quick_screen is a screening gate, not a hard dependency. Slither unavailable → skip silently. Aderyn not installed → skip silently. Node always returns both keys regardless.
- **Separate node not merged with static_analysis**: quick_screen is scoped differently (broader detector set, both tools) and runs before routing. `static_analysis` runs after routing (deep path only) with class-scoped detectors. Merging them would couple the Tier-0 screen to the deep-path tool.

| Version | NODE_FEATURE_DIM | NUM_EDGE_TYPES | GNN layers | Backbone | Best F1 |
|---------|-----------------|----------------|------------|----------|---------|
| GCB-P1-Run4 | 11 | 11 | **8** | GraphCodeBERT+LoRA+prefix K=48+IMP | **0.3362** (ep32) |

---

## 22. Agent Layer — Phase 1 A4/A5: Aderyn deep-path + End-to-End Smoke Test

**Period:** 2026-05-30
**Commit:** `330e68e`

### A4 — Aderyn in deep-path static_analysis

**File:** `agents/src/orchestration/nodes.py`

**Why:** quick_screen runs Aderyn at Tier 0 (High/Critical only, Tier-0 screen). The deep path should also benefit from Aderyn's full detector set, giving cross_validator and synthesizer a second independent signal alongside Slither.

**Note on scope correction:** The original A4 spec said "add Aderyn to audit_server MCP (:8012)". During implementation we clarified that audit_server is the on-chain AuditRegistry server, not a static analysis tool server. The correct home for Aderyn is in-process in `static_analysis`, consistent with how Slither already runs. Result is equivalent: deep path now runs both tools.

**What was added:**

`_run_aderyn_on_file(tmp_path)` helper function:
- Runs `aderyn --output json <file>` as a subprocess (90s timeout)
- Parses JSON output: iterates `high`, `medium`, `low` buckets
- Extracts `id/title`, `description`, `instances` (line numbers + function names)
- Returns findings with `tool="aderyn"`, `impact` derived from bucket name, `confidence="Medium"`
- Non-fatal: `FileNotFoundError`, `TimeoutExpired`, any other exception → returns `[]`

Integration into `static_analysis`:
- Called on the same `tmp_path` after Slither finishes (file still exists before `finally` cleanup)
- `findings.extend(aderyn_findings)` — combined into the single `static_findings` list
- Log line updated: `slither=N aderyn=M external_calls=K`

**Impact on cross_validator:** the `static_findings` list now has both `tool="slither"` and `tool="aderyn"` entries. The existing `compute_verdict()` in routing.py looks for Slither detector names — it ignores Aderyn findings (they use rule IDs like "H-1" not Slither detector names). This is intentional: Aderyn findings are available to the synthesizer LLM but don't auto-confirm ML verdicts without a routing.py mapping. Adding ADERYN_TO_CLASSES mapping is a Phase 2 item (B1 in the plan).

### A5 — End-to-End Smoke Test

**File:** `agents/tests/test_smoke_e2e.py` (7 tests, all pass)

**Coverage:**

| Test | What it verifies |
|------|-----------------|
| `test_deep_path_vault_produces_final_report` | Full Phase 1 field set in final_report; path_taken=="deep" |
| `test_quick_screen_hits_in_final_state` | quick_screen_hits present with slither+aderyn keys |
| `test_deep_path_graph_explanations_present` | graph_explanations non-empty after deep path |
| `test_routing_decisions_logged` | routing_decisions populated with class names |
| `test_fast_path_safe_contract` | ML safe + screen clean → path_taken=="fast", empty rag_evidence |
| `test_screen_escalated_path_when_ml_safe_but_screen_fires` | ML safe but Slither fires → NOT fast path |
| `test_ml_failure_still_produces_report` | Inference server down → still produces report with ML failure note |

All MCP calls are mocked. Slither runs in-process on the Vault contract for the screen-escalated test (using `patch.object(slither_module, "Slither", ...)` to inject a mock instance). Aderyn subprocess is stubbed with `FileNotFoundError` to avoid CI dependency.

---

## 23. GNN Interpretability Suite (2026-05-30)

**Period:** 2026-05-30
**Commits:** pending
**Checkpoint evaluated:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (ep32, F1=0.3362)

### Overview

A structured GNN interpretability and validation suite was designed, implemented, run, and documented against the Run 4 best checkpoint. The study spans three analysis layers: (1) structural data quality, (2) architectural expressivity, and (3) learned model behaviour. 21 experiments were designed; 22 scripts were written; 12 Solidity test contracts were written. Experiments running against the checkpoint were executed; those requiring `solc` were blocked by a toolchain gap.

### Scripts and Experiments

| Location | Count | Description |
|----------|-------|-------------|
| `ml/scripts/interpretability/` | 22 scripts | One per experiment plus `utils.py` and `__init__.py` |
| `ml/scripts/interpretability/test_contracts/` | 12 files | Solidity contracts for EXP-L6 counterfactual testing |
| `docs/interpretability/` | 24 files | One markdown report per experiment + master report + index |

Experiments by layer:

| Layer | Experiments | Scripts |
|-------|-------------|---------|
| 1 — Structural | EXP-S1, S2, S3, S4, A1, A2 | exp_s1 through exp_s4, exp_a1, exp_a2 |
| 2 — Expressivity | EXP-E1, E2, E3, E4 | exp_e1 through exp_e4 |
| 3 — Learning | EXP-A3, A4, L1–L10 | exp_a3, exp_a4, exp_l1 through exp_l10 |

Additionally: 3 validation scripts (`val_finding1_jk_weights.py`, `val_finding2_proper_ablation.py`, `val_finding4_timestamp_size.py`) and a training ablation shell script (`run_training_ablation.sh`).

### Key Findings (validated)

| Finding | Experiment | Key Number | Status |
|---------|-----------|-----------|--------|
| Phase 3 (REVERSE_CONTAINS) marginally dominant in JK weights | EXP-L1, EXP-A3 | Phase 3 = 0.346–0.381, entropy 99.98% of max | Validated: all phases nearly equally weighted; regulariser prevents collapse but not Phase 2 suppression |
| CFG edge ablation has near-zero effect | EXP-L2 | CONTROL_FLOW+CALL_ENTRY combined Δ = 1.08×10⁻⁶ | Validated: five orders below the 0.03 threshold |
| GNN eye useful for only 3/10 classes | EXP-A4 | CallToUnknown, IntegerUO, Timestamp only | Validated: GNN F1=0 for 7 classes including Reentrancy |
| All 4 tested classes WL-distinguishable | EXP-E2 | Timestamp 0% collision, Reentrancy 11.1% | Validated: expressivity is not the bottleneck |
| All classes severely miscalibrated | EXP-L7 | ECE range 0.205–0.310, mean ≈ 0.252 | Validated: calibration required before deployment |
| Timestamp size shortcut confirmed at prediction level | EXP-L7 | F1=1.0 (small/medium) vs F1=0.364 (large); gap=0.636 | Validated: Cohen's d=1.657 (EXP-S3) now confirmed in predictions |
| `type_id_norm` dominates permutation importance by 3× | EXP-L8 | Mean importance 0.0786 vs next 0.0262 | Validated: `uses_block_globals` rank 10 (not last — stale feature names caused wrong earlier claim) |
| CEI paths present in data but GNN doesn't traverse them | EXP-S4 + EXP-L2 | 76% of Reentrancy-pos have CALL_ENTRY; ablation effect = 5.3×10⁻⁷ | Validated: data gap is not the cause |
| EMITS edges near-absent (12 total across 41K contracts) | EXP-S2 | 12 edges, baseline 0.051% | Validated: not extractable as signal |
| CEI reachability only 37.7% at k=8 | EXP-E1 | 37.7% Reentrancy-positive vs 26.7% negative | Validated: GNN hop depth insufficient for most CEI paths |

### Script Bugs Fixed

| Script | Bug | Fix |
|--------|-----|-----|
| `exp_l8_permutation_importance.py` | Stale `FEATURE_NAMES` list from pre-v8 schema (11 entries in wrong order) | Now imports `FEATURE_NAMES` from `graph_schema.py` |
| `exp_l1_jk_weight_analysis.py` | `mean_entropy` field stored mean JK weight (0.333) instead of Shannon entropy (1.099) | Fixed field extraction to compute entropy from per-phase weights |
| `ml/src/models/gnn_encoder.py` | Docstring said forward returns 3-tuple; `return_intermediates=True` returns 4-tuple | Docstring updated to document both return shapes |

### Output Files

| Path | Description |
|------|-------------|
| `docs/interpretability/INTERPRETABILITY_MASTER_REPORT.md` | Full 970-line report with per-class analysis, root cause analysis, capacity ceiling analysis, and Run 5 recommendations |
| `docs/interpretability/EXPERIMENT_INDEX.md` | Experiment table (22 rows), root cause analysis for failures, argparse bugs fixed |
| `docs/interpretability/exp_*.md` | 22 individual experiment reports with method, results, pass/fail, and implications |
| `docs/proposal/GNN_INTERPRETABILITY_FIXES_PROPOSAL.md` | Concrete fix proposals for Run 5 based on validated findings |


---

## 23. Pre-Run-5 Implementation — Label Cleaning + CEI Aux Loss + Calibration (2026-05-31)

**Period:** 2026-05-31
**What changed:** All code changes required before Run 5 training implemented.

### Interpretability Script Fixes (Phase A)

| Script | Fix |
|--------|-----|
| `exp_l4_gradient_saliency.py` | Import FEATURE_NAMES from graph_schema (stale pre-v8 list replaced) |
| `exp_a2_cfg_inheritance.py` | `_CONTAINS_EDGE=0` → `EDGE_TYPES["CONTAINS"]` (was matching CALLS not CONTAINS) |
| `exp_e1_receptive_field.py` | Analysis 2/3 redesigned: REVERSE_CONTAINS is runtime-only; redesigned as CONTAINS + CALLS checks |
| `exp_l3_attention_visualization.py` | Extended to hook conv3b (CALL_ENTRY+RETURN_TO) alongside conv3 (CF) |

### New Measurement Scripts (Phase B)

| Script | Purpose |
|--------|---------|
| `exp_b1_phase2_gradient_norm.py` | Grad norm at each phase LayerNorm — measures Phase 2 gradient flow |
| `exp_b2_per_eye_ece.py` | Per-eye (GNN/TF/Fused) ECE — prerequisite for temperature scaling |
| `exp_b3_jk_weight_distribution.py` | JK weight distribution per class — tracks Phase 2 contribution |
| `exp_b4_unusedreturn_saliency.py` | Gradient saliency for UnusedReturn top-scored contracts |

### Calibration (Phase C)

- `ml/scripts/calibrate_temperature.py` — fits per-class temperatures via LBFGS minimising BCE NLL on val set. Outputs `ml/calibration/temperatures.json`. Run after B2.

### Data Quality Scripts (Phase D)

Scripts written but require manual validation before running. Reliability varies:

| Script | Sol ID | Reliability | Notes |
|--------|--------|-------------|-------|
| `ml/scripts/clean_integeruo_labels.py` | Sol-2 | **High** | Version detection is deterministic; skips contracts without source path |
| `ml/scripts/gate_timestamp_labels.py` | Sol-3 | **Medium** | Catches direct timestamp-in-branch patterns; misses indirect propagation through state variables |
| `ml/scripts/clean_reentrancy_labels.py` | Sol-1 | **Low** | CFG BFS misses cross-function reentrancy and is only as complete as the extractor's `CFG_NODE_WRITE` coverage. Run `--dry-run` and cross-check a sample with `slither --detect reentrancy-eth` before committing. Alternative: skip and rely on E1 CEI aux loss instead. |
| `ml/scripts/inject_openzeppelin_negatives.py` | IMP-D2 | **High** | OZ contracts are audited clean; adds genuine zero-label negatives |

**Execution results (2026-05-31 dry-runs):** All three label scripts produced results that make running them unsafe or redundant:

| Script | Dry-run result | Decision |
|--------|---------------|---------|
| Sol-2 IntegerUO | 0 removals — BCCC dataset is 100% pre-0.8.0 Solidity (0.4.x/0.5.x) | SKIP — labels correct |
| Sol-3 Timestamp | 380/380 (100%) removal rate — `uses_block_globals` (x[:,2]) is function-level only, never set on CFG_NODE_CHECK child nodes → false detection | SKIP — feature architecture incompatible |
| Sol-1 Reentrancy | 2,606/3,887 (67%) removal rate — too aggressive for LOW-reliability BFS heuristic | SKIP — rely on E1 CEI aux loss instead |

**Label CSV unchanged:** `ml/data/processed/multilabel_index_cleaned.csv` is the Run 5 input.

### Run 5 Training Changes (Phase E)

**E1 — CEI auxiliary loss on Phase 2 embeddings:**
- `gnn_encoder.py`: added `return_phase2_embs=True` mode returning gradient-attached Phase 2 tensor
- `sentinel_model.py`: added `aux_phase2` head (256→128→10, GELU+Dropout)
- `trainer.py`: `TrainConfig.aux_phase2_loss_weight=0.10`; CEI classes (ExternalBug/Reentrancy/TOD) get 3× weight

**E2 — Timestamp size normalisation:**
- `graph_extractor.py`: complexity (dim 5) normalised by contract total node count (E2 Interp-3 fix)
- `trainer.py`: `use_weighted_sampler="timestamp-size"` mode — large Timestamp+ contracts get 4× weight

**IMP-D1 readiness:**
- `sentinel_model.py`: `fusion_max_nodes` parameter wired through to CrossAttentionFusion
- `trainer.py`: `TrainConfig.fusion_max_nodes=1024`; raise to 2048 after re-extraction

---

## 24. Interpretability Suite — Audit Fixes + Phase B Measurements (2026-06-01)

**Period:** 2026-06-01  
**Checkpoint evaluated:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (ep32, F1=0.3362)  
**All 21 interpretability experiments + B1–B4 resolved. No pending experiments.**

### Audit Fixes Applied (2026-06-01)

Four corrections applied after re-examination of measurement methodology and source code cross-referencing:

#### EXP-B1 — Gradient Method Corrected (BCEWithLogitsLoss)

**Script:** `ml/scripts/interpretability/exp_b1_phase2_gradient_norm.py`

Original run backpropagated through raw logit `logits[0, class_idx]`. This is non-standard — training uses `BCEWithLogitsLoss` which applies sigmoid before gradient. Fixed to use `F.binary_cross_entropy_with_logits(logits[0, class_idx].unsqueeze(0), target=1)`.

Corrected P2/P1 ratios: **72–91%** (was 75–86% with raw logit). Ordering unchanged. Timestamp remains highest (91.3%), DenialOfService lowest (72.2%).

| Class | Phase1 | Phase2 | Phase3 | P2/P1 |
|-------|--------|--------|--------|-------|
| CallToUnknown | 0.051893 | 0.041190 | 0.035657 | 79.4% |
| DenialOfService | 0.034882 | 0.025193 | 0.020473 | 72.2% |
| ExternalBug | 0.044038 | 0.033292 | 0.028976 | 75.6% |
| GasException | 0.041313 | 0.032616 | 0.025441 | 78.9% |
| IntegerUO | 0.039092 | 0.028423 | 0.022379 | 72.7% |
| MishandledException | 0.045841 | 0.036230 | 0.031026 | 79.0% |
| Reentrancy | 0.050614 | 0.037378 | 0.035040 | 73.8% |
| Timestamp | 0.094683 | 0.086430 | 0.073734 | 91.3% |
| TOD | 0.033661 | 0.025190 | 0.020505 | 74.8% |
| UnusedReturn | 0.080223 | 0.061586 | 0.052428 | 76.8% |

#### EXP-L2 — Structural Ablation Added

**Script:** `ml/scripts/interpretability/exp_l2_edge_ablation.py`  
**Output:** `ml/interpretability_results/exp_l2/exp_l2_ablation_delta.json`

Prior report only measured embedding ablation (zeroing edge embeddings). Structural ablation (removing edges entirely from the graph) was missing, inflating the denominator of the embedding/structural ratio. Structural ablation rerun and added.

Results:

| Metric | Value |
|--------|-------|
| cfg_combined_drop_embedding | 1.11×10⁻⁶ |
| cfg_combined_drop_structural | 0.0121 |
| Ratio (structural / embedding) | **10,944×** (corrected from earlier claimed 450×) |

Key sign-reversal finding: removing Phase 2 edges (CONTROL_FLOW, CALL_ENTRY, RETURN_TO) **increases** Reentrancy prediction scores:

| Edge type | Reentrancy structural delta | Timestamp structural delta |
|-----------|---------------------------|--------------------------|
| CONTROL_FLOW | +0.020328 | +0.162891 |
| CALL_ENTRY | +0.010085 | +0.019450 |
| RETURN_TO | +0.018186 | ≈0 |

Phase 2 is actively suppressing Reentrancy predictions — a shortcut learned from the training distribution (dense CFG → large well-engineered → not Reentrancy).

#### EXP-L3 — Reclassified ARCHITECTURAL N/A

**Script:** `ml/scripts/interpretability/exp_l3_attention_visualization.py`

Prior report claimed PASS (100% CF fraction in top attention edges). Retracted: 100% CF fraction is architecturally guaranteed because conv3 is wired exclusively to the CF-only subgraph. This is not a learned finding.

Real finding from the corrected run (conv3b also hooked): **all GAT attention weights = 1.0 (uniform)**. No selective attention learned within CFG or ICFG edges. Phase 2 attention is a weighted average with weight 1.0 — equivalent to a simple sum.

Status: **ARCHITECTURAL N/A** (not PASS, not FAIL — criterion was ill-posed).

#### EXP-L4 — Rerun with Correct Feature Names

**Script:** `ml/scripts/interpretability/exp_l4_gradient_saliency.py`  
**Output:** `ml/interpretability_results/exp_l4/`

FEATURE_NAMES was a stale hardcoded pre-v8 list. Fixed to import from `graph_schema.py`. Rerun 2026-06-01. Corrected results confirmed original finding structure but with accurate feature labels.

Per-class top-3 (correct feature names):

| Class | Rank 1 | Rank 2 | Rank 3 |
|-------|--------|--------|--------|
| All 10 classes | `external_call_count` (21–24%) | `complexity` (10–11%) | varies |

Pass criteria:
- Timestamp `uses_block_globals` ≥ 20%: actual 10.0% → **FAIL**
- Reentrancy CFG_NODE_CALL + has_state_write ≥ 20%: actual 8.9% → **FAIL**
- Global sensitivity artifact confirmed: `external_call_count` dominates gradient for ALL classes regardless of class semantics.

### Phase B — New Measurement Scripts (All Complete)

| Script | Status | Key Finding |
|--------|--------|-------------|
| `exp_b1_phase2_gradient_norm.py` | COMPLETE | Phase 1 > Phase 2 > Phase 3; P2/P1 = 72–91% |
| `exp_b2_per_eye_ece.py` | COMPLETE | GNN/TF/Fused ECE 0.057–0.065 (good); main head ECE 0.249 (severe) |
| `exp_b3_jk_weight_distribution.py` | COMPLETE | Universal Phase3 > Phase1 > Phase2; no class upweights Phase 2 |
| `exp_b4_unusedreturn_saliency.py` | COMPLETE | external_call_count + complexity dominate; return_ignored rank 4 (2.3% diff only) |

### Temperature Calibration

Fitted `ml/calibration/temperatures_run4.json` via `ml/scripts/calibrate_temperature.py` (LBFGS on val set).

| Before | After |
|--------|-------|
| ECE 0.249 (main head, severely miscalibrated) | ECE 0.028 (post temperature scaling) |

### Phase G — Interpretability Completeness Gaps (All Resolved)

Six completeness gaps identified from the INTERPRETABILITY_AUDIT_AND_COMPLETENESS.md document, all resolved by 2026-06-01:

| Gap | Fix |
|-----|-----|
| P1 — EXP-S3 "dead feature" finding | Retracted — was CFG-node artifact; FUNCTION-node mean computed correctly |
| P2 — EXP-E1 DEF_USE missing from Phase 2 | DEF_USE(10) added; Phase 2 k=8 reachability now 38.2% (was 37.7%) |
| P3 — EXP-L5 wrong pooling | max+mean [512] pooling fixed; IntegerUO Phase1 F1 corrected 0.114→0.419 |
| P4 — EXP-E4 only CF tested | All 4 Phase 2 edge types tested (CF/DEF_USE/CALL_ENTRY/RETURN_TO); all 0.0% directional diff |
| P5 — EXP-L9 non-discriminative criterion | Relative-rank criterion; FAIL confirmed (safe CW > vuln CW, delta=−0.00654) |
| P6 — Documentation stale | EXPERIMENT_INDEX + MASTER_REPORT fully updated; 4 B-experiment docs created |

### Pre-Run-5 Root Cause Analysis

Full Phase 2 root cause analysis documented in `docs/pre-run-fixes/phase2_root_cause_analysis.md`. Seven confirmed root causes identified:

1. FUNCTION nodes get identity transform from Phase 2 (CF edges don't reach FUNCTION nodes)
2. `aux_phase2_loss_weight` = 0.0 throughout all of Run 4 (feature not yet added to TrainConfig)
3. Phase 2 has 8× lower attention head capacity than Phase 1 (heads=1 vs heads=8)
4. JK entropy regularizer pushes Phase 2 weight below 1/3 by default
5. DEF_USE edges get only 1 hop (Layer 5 only)
6. Phase 3 does Phase 2's job via REVERSE_CONTAINS lift
7. Suppression encoded in learned weights: dense CFG → not Reentrancy (inversion)

### Current State After This Section

- All 21 interpretability experiments resolved (+ B1–B4)
- Temperature calibration fitted
- Run 5 config committed (aux_phase2_loss_weight=0.10, timestamp-size sampler)
- Pre-run-fixes analysis docs in `docs/pre-run-fixes/`
- **Next action: Launch Run 5**

**Last updated: 2026-06-01**
