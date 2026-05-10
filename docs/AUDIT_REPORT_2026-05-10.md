# SENTINEL ML Audit Report — 2026-05-10

**Scope:** Model training pipeline, inference system, v4 experiment 1, manual behavioral testing  
**Checkpoint audited:** `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` (epoch 26/30)  
**Author:** AI-assisted audit (session 2026-05-09/10)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Training History and Progression](#3-training-history-and-progression)
4. [v4 Experiment 1 — Results](#4-v4-experiment-1--results)
5. [Manual Behavioral Testing — Full Results](#5-manual-behavioral-testing--full-results)
6. [Per-Class Behavioral Analysis](#6-per-class-behavioral-analysis)
7. [Identified Failure Modes](#7-identified-failure-modes)
8. [Root Cause Analysis](#8-root-cause-analysis)
9. [Code Fixes Applied (v4 Sprint)](#9-code-fixes-applied-v4-sprint)
10. [Assumptions and Known Limitations](#10-assumptions-and-known-limitations)
11. [Recommendations and Priority Order](#11-recommendations-and-priority-order)
12. [Appendix — Test Contracts](#12-appendix--test-contracts)

---

## 1. Executive Summary

The SENTINEL v4 experiment 1 checkpoint achieves a **tuned val F1-macro of 0.5422**, clearing the 0.5069 gate by +0.035. On quantitative validation metrics the model represents a meaningful improvement over v3 across all 10 classes.

However, manual behavioral testing on 20 hand-crafted contracts reveals **fundamental limitations that metrics alone do not capture**:

- **Detection rate: 3/19 expected vulnerabilities (15%)** on contracts designed to exhibit clear, isolated vulnerability patterns
- **Specificity: 1/3 safe contracts correctly classified** — the model flags clean contracts with proper `call()` usage as vulnerable
- The model operates primarily as a **call-pattern detector**, not a semantic vulnerability detector — it associates the presence of external calls with vulnerability rather than the semantic context (order, return handling, trust assumptions)
- **5 of 10 classes** (DoS, IntegerUO, GasException, TOD, CallToUnknown) show near-zero or misdirected signal on canonical test cases
- **4 classes** (Reentrancy, CallToUnknown, MishandledException, UnusedReturn) exhibit high false-positive rates on each other's patterns

The v4 model is **not production-ready for standalone deployment** but is a meaningful research baseline. The next priority is targeted data collection, not further hyperparameter tuning.

---

## 2. System Architecture

### 2.1 Model Architecture (LOCKED)

```
Input:  Solidity source (.sol)
           │
    ┌──────┴──────────────────────────────────┐
    │  graph_extractor.py (Slither + AST)     │  → graph .pt  [N nodes, E edges]
    │  tokenizer.py (CodeBERT tokenizer)      │  → token .pt  [512 tokens]
    └──────────────────────────────────────────┘
           │
    ┌──────┴──────────────────────────────────┐
    │  GNNEncoder (3-layer GAT)               │
    │    in_channels=8 (LOCKED)               │
    │    Embedding(5,16) for edge types       │
    │    Output: node_embs [N, 64]            │
    └──────────────────────────────────────────┘
           │
    ┌──────┴──────────────────────────────────┐
    │  TransformerEncoder (CodeBERT 124M)     │
    │    LoRA r=8 α=16 on Q+V, all 12 layers  │
    │    ~295K trainable / 124M frozen        │
    │    Output: last_hidden_state [B,512,768] │
    └──────────────────────────────────────────┘
           │                    │
    ┌──────┴────────────────────┴──────────────┐
    │  CrossAttentionFusion (bidirectional)    │
    │    Node→Token MHA + Token→Node MHA       │
    │    Output: [B, 128] (LOCKED)             │
    └──────────────────────────────────────────┘
           │
    ┌──────┴──────────────────────────────────┐
    │  Classifier: Linear(128, 10)            │
    │    NO sigmoid inside — BCEWithLogitsLoss │
    │    Sigmoid applied at inference only     │
    └──────────────────────────────────────────┘
           │
    Output: 10 independent binary predictions
```

### 2.2 Output Classes

| Index | Class | Training Positive Rate |
|-------|-------|----------------------|
| 0 | CallToUnknown | ~11.7% |
| 1 | DenialOfService | ~1.5% (137 train samples) |
| 2 | ExternalBug | ~16.1% |
| 3 | GasException | ~25.3% |
| 4 | IntegerUO | ~52.1% (largest class) |
| 5 | MishandledException | ~22.1% |
| 6 | Reentrancy | ~24.3% |
| 7 | Timestamp | ~10.0% |
| 8 | TransactionOrderDependence | ~17.1% |
| 9 | UnusedReturn | ~14.3% |

### 2.3 Node Feature Vector [8-dim, LOCKED]

| Idx | Feature | Encoding |
|-----|---------|---------|
| 0 | node type | STATE_VAR=0 FUNCTION=1 MODIFIER=2 EVENT=3 FALLBACK=4 RECEIVE=5 CONSTRUCTOR=6 CONTRACT=7 |
| 1 | visibility | public/external=0 internal=1 private=2 |
| 2 | pure | 0/1 |
| 3 | view | 0/1 |
| 4 | payable | 0/1 |
| 5 | reentrant | 0/1 |
| 6 | complexity | float (CFG nodes) |
| 7 | loc | float (lines of code) |

### 2.4 Training Configuration (v4 exp1)

```
loss_fn:          BCEWithLogitsLoss + per-class pos_weight
optimizer:        AdamW (lr=1e-4, weight_decay=1e-2)
scheduler:        OneCycleLR (30 epochs, pct_start=0.3)
batch_size:       16
epochs:           30
lora_r:           8
lora_alpha:       16
early_stop:       patience=7
AMP:              BF16 on CUDA
resume_from:      v3 checkpoint (epoch 54, model-only mode)
```

### 2.5 Per-Class pos_weight (training split)

| Class | pos_weight | Implication |
|-------|-----------|------------|
| CallToUnknown | 7.59 | Missing a positive costs 7.59× more than a false positive |
| DenialOfService | 68.02 | Extreme — model should aggressively predict DoS |
| ExternalBug | 5.22 | |
| GasException | 2.95 | |
| IntegerUO | 0.92 | Near-balanced — no imbalance correction needed |
| MishandledException | 3.53 | |
| Reentrancy | 3.12 | |
| Timestamp | 8.40 | |
| TransactionOrderDependence | 4.85 | |
| UnusedReturn | 5.09 | |

---

## 3. Training History and Progression

### 3.1 Baseline and Gate History

| Version | Config | Raw F1 | Tuned F1 | Gate |
|---------|--------|--------|---------|------|
| v3 (60 ep, full retrain) | lr=3e-4, batch=32, lora_r=8 | 0.4715 | **0.5069** | 0.4884 ✅ |
| v4 exp1 (30 ep, fine-tune) | lr=1e-4, batch=16, lora_r=8 | 0.5064 | **0.5422** | 0.5069 ✅ |

### 3.2 v3 Plateau Analysis

The v3 plateau at epoch 60 was **incorrectly diagnosed as a capacity ceiling** in earlier sessions. Inspection of actual MLflow curves (sqlite:///mlruns.db, run d2ee23a) revealed:

- Training loss was **still decreasing** at epoch 60 (not converged)
- The plateau was **OneCycleLR exhaustion** — the learning rate had decayed to near-zero
- The `OneCycleLR` schedule completes in exactly `epochs` steps; after epoch 60, the LR is ≈ 0 and gradients are functionally zero

**Confirmation:** v4 exp1 epoch 1 alone (lr=1e-4 fresh cycle, warm weights) immediately exceeded v3's 60-epoch best raw F1 (0.4754 > 0.4715). A 1-epoch improvement over 60 epochs confirms LR exhaustion was the bottleneck, not capacity.

### 3.3 v4 exp1 Training Curve

| Epoch | Raw F1-macro | Notes |
|-------|-------------|-------|
| 1 | 0.4754 | First epoch beats v3 60-epoch best |
| 7 | 0.4829 | Steady climb |
| 12 | 0.4848 | |
| 14 | 0.4927 | |
| 16 | 0.4931 | |
| 18 | 0.5000 | Crosses 0.50 milestone |
| 20 | 0.5022 | |
| 22 | 0.5053 | |
| 26 | **0.5064** | Best checkpoint |
| 30 | 0.5064 | patience=4/7 — still learning, not early-stopped |

**Key observation:** The model was still improving at epoch 26/30. Patience counter=4/7 at end — the run was bounded by the epoch limit, not by a genuine plateau. This means Experiment 2 (more epochs at lr=5e-5) has a realistic chance of further improvement.

---

## 4. v4 Experiment 1 — Results

### 4.1 Per-Class Comparison (v3 → v4 exp1)

| Class | v3 Threshold | v3 Tuned F1 | v4 Threshold | v4 Tuned F1 | Δ | Floor | Pass |
|-------|-------------|------------|-------------|------------|---|-------|------|
| CallToUnknown | 0.70 | 0.394 | 0.70 | 0.4474 | +0.053 | 0.344 | ✅ |
| DenialOfService | 0.95 | 0.400 | 0.95 | 0.4343 | +0.034 | 0.350 | ✅ |
| ExternalBug | 0.65 | 0.435 | 0.70 | 0.4838 | +0.049 | 0.385 | ✅ |
| GasException | 0.55 | 0.550 | 0.55 | 0.5568 | +0.007 | 0.500 | ✅ |
| IntegerUO | 0.50 | 0.821 | 0.50 | 0.8259 | +0.005 | 0.771 | ✅ |
| MishandledException | 0.60 | 0.492 | 0.55 | 0.5094 | +0.017 | 0.442 | ✅ |
| Reentrancy | 0.65 | 0.536 | 0.65 | 0.5687 | +0.033 | 0.486 | ✅ |
| Timestamp | 0.75 | 0.479 | 0.80 | 0.5283 | +0.049 | 0.429 | ✅ |
| TransactionOrderDependence | 0.60 | 0.477 | 0.65 | 0.5220 | +0.045 | 0.427 | ✅ |
| UnusedReturn | 0.70 | 0.486 | 0.70 | 0.5452 | +0.059 | 0.436 | ✅ |
| **Macro average** | | **0.507** | | **0.5422** | **+0.035** | | ✅ |

All 10 classes improved. No class dropped below its floor.

### 4.2 Threshold Observations

- **DoS threshold=0.95:** The model is severely underconfident on DoS — it never exceeds 0.50 probability on DoS samples, so the only way to capture any recall is to lower the threshold all the way to 0.95. This means almost any contract with some DoS signal (even 0.38 probability) is classified as DoS.
- **Timestamp threshold=0.80, CallToUnknown threshold=0.70:** High thresholds are needed because the model over-predicts these classes — setting the bar high reduces false positives at the cost of some recall.
- **IntegerUO threshold=0.50:** The model is well-calibrated on this class — the threshold is near-neutral, meaning the raw probability is already meaningful.

### 4.3 What the Val Set Metrics Don't Capture

The val F1 improvement from 0.507 to 0.542 is real — on the held-out split from the same dataset as training. However, the manual tests below reveal that **in-distribution performance does not generalize to out-of-distribution test contracts**.

The val set was drawn from the same BCCC dataset as training, with the same label distribution, the same contract lengths, the same coding styles. Manual tests use contracts written specifically to isolate single vulnerability patterns. The gap between 0.54 F1 (val) and 15% detection rate (manual) quantifies the distribution mismatch.

---

## 5. Manual Behavioral Testing — Full Results

### 5.1 Test Setup

- **Checkpoint:** `multilabel-v4-finetune-lr1e4_best.pt`
- **Thresholds:** per-class from `multilabel-v4-finetune-lr1e4_best_thresholds.json`
- **Infrastructure:** Slither (graph extraction) + CodeBERT tokenizer + v4 model on CUDA
- **Contracts:** 20 hand-crafted contracts — 7 single-class, 3 multi-class, 3 safe, 7 minimal/variant
- **Annotation:** Each contract's first line specifies expected class(es) via `// expect: Class1,Class2`

### 5.2 Full Results Table

```
Contract                        Expected                    Detected (≥threshold)            Grade
─────────────────────────────────────────────────────────────────────────────────────────────────
01_reentrancy_classic           Reentrancy                  Reentrancy                        ✓ HIT
02_reentrancy_tricky            Reentrancy                  (none)                            ✗ MISS
03_integer_overflow             IntegerUO                   (none)                            ✗ MISS
04_timestamp_dependence         Timestamp                   GasEx,IntegerUO,Misha,Reentr,     ~ PARTIAL
                                                            Timestamp,TOD,UnusedReturn
05_denial_of_service            DenialOfService             (none)                            ✗ MISS
06_mishandled_exception         MishandledException         CallToUnknown, Reentrancy         ✗ WRONG CLASS
07_tx_order_dependence          TransactionOrderDependence  (none)                            ✗ MISS
08_unused_return                UnusedReturn                CallToUnknown                     ✗ WRONG CLASS
09_call_to_unknown              CallToUnknown               GasEx,IntegerUO,Misha,Reentr,     ✗ WRONG CLASSES
                                                            UnusedReturn
10_gas_exception                GasException                (none)                            ✗ MISS
11_external_bug                 ExternalBug                 CallToUnknown, GasEx, IntegerUO   ✗ WRONG CLASSES
12_safe_contract                (safe)                      CallToUnknown, Reentrancy         ✗ FALSE POS
13_multilabel_complex           Reentrancy,Timestamp,UnusedRet IntegerUO                     ✗ WRONG CLASS
14_reentrancy_minimal           Reentrancy                  (none)                            ✗ MISS
15_tod_minimal                  TransactionOrderDependence  (none)                            ✗ MISS
16_gas_minimal                  GasException                CallToUnknown, Reentrancy         ✗ WRONG CLASSES
17_integer_simple               IntegerUO                   (none)                            ✗ MISS
18_safe_no_calls                (safe)                      (none)                            ✓ CORRECT
19_safe_with_transfer           (safe)                      Reentrancy                        ✗ FALSE POS
20_unused_return_minimal        UnusedReturn                Exter,IntegerUO,Reentr,TOD,UnusedRet ~ PARTIAL
```

**Summary:** 3/19 expected classes detected (15%). 1/3 safe contracts correctly clean.

### 5.3 Full Probability Matrix

```
Contract                   CallT  Denia  Exter  GasEx  Integ  Misha  Reent  Times  Trans  Unuse
──────────────────────────────────────────────────────────────────────────────────────────────────
Thresholds →               0.70   0.95   0.70   0.55   0.50   0.55   0.65   0.80   0.65   0.70
──────────────────────────────────────────────────────────────────────────────────────────────────
01_reentrancy_classic      0.313  0.000  0.000  0.045  0.036  0.000  0.672* 0.002  0.000  0.000
02_reentrancy_tricky       0.027  0.000  0.000  0.000  0.001  0.000  0.018* 0.000  0.000  0.000
03_integer_overflow        0.019  0.002  0.001  0.004  0.011* 0.002  0.025  0.000  0.000  0.000
04_timestamp_dependence    0.000  0.000  0.046  0.962  0.967  0.640  0.955  0.944* 0.812  0.970
05_denial_of_service       0.000  0.000* 0.000  0.004  0.004  0.000  0.002  0.000  0.000  0.000
06_mishandled_exception    0.992  0.000  0.000  0.006  0.011  0.000* 0.958  0.000  0.000  0.000
07_tx_order_dependence     0.006  0.000  0.002  0.019  0.073  0.007  0.005  0.009  0.001* 0.000
08_unused_return           0.894  0.000  0.669  0.142  0.435  0.115  0.503  0.000  0.003  0.316*
09_call_to_unknown         0.092* 0.002  0.699  0.653  0.719  0.619  0.736  0.029  0.484  0.856
10_gas_exception           0.001  0.000  0.000  0.024* 0.095  0.015  0.001  0.000  0.000  0.000
11_external_bug            0.720  0.000  0.201* 0.592  0.755  0.547  0.464  0.076  0.172  0.183
12_safe_contract [SAFE]    0.899  0.094  0.001  0.326  0.225  0.020  0.897  0.000  0.047  0.015
13_multilabel_complex      0.123  0.000  0.456  0.491  0.592* 0.498  0.182* 0.000  0.528  0.393*
14_reentrancy_minimal      0.029  0.000  0.000  0.000  0.001  0.000  0.056* 0.000  0.000  0.000
15_tod_minimal             0.160  0.007  0.000  0.018  0.032  0.003  0.622  0.000  0.000* 0.000
16_gas_minimal             0.983  0.170  0.000  0.033* 0.008  0.000  0.993  0.000  0.001  0.000
17_integer_simple          0.039  0.000  0.003  0.044  0.109* 0.013  0.010  0.001  0.012  0.000
18_safe_no_calls [SAFE]    0.395  0.019  0.014  0.346  0.349  0.066  0.432  0.000  0.042  0.003
19_safe_with_transfer [SAFE] 0.013 0.030 0.036  0.179  0.075  0.022  0.829  0.000  0.002  0.090
20_unused_return_minimal   0.400  0.000  0.931  0.427  0.618  0.464  0.805  0.000  0.659  0.878*
──────────────────────────────────────────────────────────────────────────────────────────────────
* = expected class for that contract
```

---

## 6. Per-Class Behavioral Analysis

### 6.1 Reentrancy — Partial (detects classic, misses variants)

- **Classic pattern detected** (01): `withdraw()` with `call{value:}` before balance zeroed → 0.672 ✓
- **Minimal pattern missed** (14): same logic in 4 lines → 0.056 ✗
- **Indirect reentrancy missed** (02): attack via internal helper function → 0.018 ✗
- **False positive on safe contract** (12, 19): correct CEI pattern still triggers 0.897/0.829

**Interpretation:** The model detected Reentrancy by recognizing the *overall contract structure* (deposit + withdraw functions, balance mapping, external call) — not the dangerous ordering. When the same ordering appears in fewer lines (14) or through indirection (02), the structural cues are absent and the model fails. The model cannot reason about execution order.

### 6.2 DenialOfService — Non-functional (data starvation)

- **All DoS contracts**: probability ≈ 0.000–0.004 across all 20 tests
- **DoS pos_weight=68**: even with heavy up-weighting, the 137 training samples provide insufficient diverse patterns
- **Loop-based DoS (05, 16)**: fires Reentrancy and CallToUnknown instead — the model sees "loop + call" → Reentrancy pattern

**Interpretation:** The DoS class head has essentially learned nothing generalizable. 4 batches per epoch × 30 epochs = 120 gradient updates on 137 diverse samples. This is data starvation. No hyperparameter change fixes this.

### 6.3 ExternalBug — Confused with other call patterns

- **ExternalBug (11)**: target prob 0.201 (below 0.70 threshold); CallToUnknown fires at 0.720, GasEx at 0.592
- **UnusedReturn minimal (20)**: ExternalBug fires at 0.931 — the external interface call (`token.transfer`) maps to ExternalBug not UnusedReturn in the model's feature space

**Interpretation:** ExternalBug, CallToUnknown, MishandledException, and UnusedReturn are four classes that all involve external calls. The model has difficulty distinguishing them. They form a confusion cluster.

### 6.4 GasException — Near-zero signal

- **Classic gas patterns (10)**: max prob 0.095 (IntegerUO, not GasException)
- **Loop+transfer pattern (16)**: GasException prob 0.033; Reentrancy fires at 0.993 instead
- The model sees "loop with transfer" as Reentrancy, not GasException

**Interpretation:** Gas-related vulnerabilities require understanding of EVM gas accounting — a semantic concept not directly expressible in the 8-dimensional node feature vector. The model has no "gas" feature and cannot learn this class without semantic enrichment.

### 6.5 IntegerUO — Training distribution mismatch

- **Unchecked block (03, 17)**: max probs 0.011 and 0.109
- **Val set F1: 0.8259** — the best-performing class on the val set by far
- **Interpretation of the contradiction:** The val set contains pre-0.8 Solidity contracts where overflow occurs via bare arithmetic (`a += b` without `unchecked`). Manual tests use 0.8.x `unchecked {}` blocks. The graph structure differs (the `unchecked` scope is represented differently in the AST). The model learned the pre-0.8 pattern perfectly but doesn't recognize the 0.8+ equivalent.

**Key assumption exposed:** The training dataset (BCCC) was assembled when Solidity 0.8.x was new or not yet adopted. Most contracts are likely 0.5.x–0.7.x, where overflow happens silently. Modern contracts (0.8+) with `unchecked` are underrepresented.

### 6.6 MishandledException — Confused with Reentrancy and CallToUnknown

- **Mishandled exception (06)**: target prob 0.000; CallToUnknown=0.992, Reentrancy=0.958
- The contract calls `r.call{value: amount}("")` in a loop — the model sees this as Reentrancy/CTU pattern

**Interpretation:** The distinguishing feature of MishandledException is *ignoring* the return value of a call. The model's graph features do not include "return value used/ignored" — this information exists in the token sequence but the token representation of `r.call{value: amount}("")` vs `(bool ok,) = r.call{value: amount}("")` is lexically similar enough that the model cannot reliably distinguish them.

### 6.7 Timestamp — Detected, but over-triggers the whole model

- **Timestamp (04)**: Timestamp prob 0.944 ✓ — correctly detected
- **But 6/10 other classes also fire** at 0.81–0.97, including IntegerUO (0.967), Reentrancy (0.955), UnusedReturn (0.970)
- **Timestamp minimal (in contract 13)**: Timestamp prob 0.000 — completely missed in the multi-vulnerability contract

**Interpretation:** The Timestamp class may be primarily detected via the contract-level features (payable + `block.timestamp` + `transfer`) which happen to co-occur with other vulnerability patterns in training data. When `block.timestamp` appears in isolation without the full contract structure, it is missed.

### 6.8 TransactionOrderDependence — Essentially invisible

- **TOD classic (07)**: max prob 0.073 (IntegerUO), TOD prob 0.001
- **TOD minimal (15)**: Reentrancy 0.622 fires (the `payable(msg.sender).transfer(reward)` triggers it); TOD prob 0.000

**Interpretation:** TOD is a semantic vulnerability about transaction ordering — it requires understanding that two transactions interact through shared state in a dangerous way. This is not encodable in the current 8-dim node feature vector or the static AST graph. The model has never learned a reliable TOD signal.

### 6.9 CallToUnknown — High false-positive rate, actual class missed

- **CallToUnknown contract (09)**: target prob 0.092 (below 0.70 threshold); instead GasEx (0.653), IntegerUO (0.719), Reentrancy (0.736), UnusedReturn (0.856) all fire
- **False positive on mishandled (06)**: CallToUnknown 0.992 — the `r.call{value:}("")` pattern without return value check fires CTU
- **False positive on safe contract (12)**: 0.899

**Interpretation:** The model fires CallToUnknown on any low-level `.call()` usage, regardless of whether the target address is "known" (typed interface) or "unknown" (raw address). The distinguishing feature (typed vs raw call) is not in the graph features.

### 6.10 UnusedReturn — Partially works for minimal cases

- **Minimal single-line (20)**: UnusedReturn 0.878 ✓ — but 4 other classes also fire above threshold
- **Complex multi-function (08)**: UnusedReturn 0.316 (below 0.70 threshold); CallToUnknown 0.894 fires instead
- **Safe contract (12)**: UnusedReturn 0.015 — correctly low

**Interpretation:** The model recognizes the minimal pattern (bare `token.transfer(to, amt)` without assignment) when it's the dominant pattern in the contract. When it competes with other patterns (interfaces, multiple functions), the signal is diluted.

---

## 7. Identified Failure Modes

### FM-1: Call Pattern Over-Generalization

**Symptom:** Any contract containing `.call{value:}("")`, `delegatecall`, or similar low-level calls triggers Reentrancy and/or CallToUnknown regardless of correct or incorrect usage.

**Evidence:**
- Safe CEI contract (12): Reentrancy=0.897, CallToUnknown=0.899
- Correct pull-payment contract (19): Reentrancy=0.829
- Mishandled exception (06): fires Reentrancy=0.958, CTU=0.992 instead of MishandledException

**Affected classes:** Reentrancy, CallToUnknown (false positives); MishandledException, UnusedReturn (false negatives via substitution)

### FM-2: Graph Cannot Encode Execution Order

**Symptom:** The model cannot distinguish `call()` before state update (dangerous) from `call()` after state update (safe).

**Evidence:** Both `12_safe_contract` (correct CEI) and `01_reentrancy_classic` (vulnerable) contain the same node types and edge connections. The GNN sees identical graph topology; only the token sequence encodes the ordering. But the token sequence of `balances[msg.sender] = 0; (bool ok,) = msg.sender.call{value: amount}("");` vs the reverse looks similar enough that CrossAttention apparently does not fully capture it.

**Root cause:** The AST graph does not include control-flow edges that would encode order. Only data-flow edges (READS/WRITES) are present. Without a CFG layer, ordering is invisible to the GNN.

### FM-3: Data Starvation for Rare Classes

**Symptom:** DoS has effectively zero predictive power. TOD and GasException are near-zero.

**Evidence:**
- DoS: 137 training samples (~4 batches/epoch); val F1=0.4343 only achieved at threshold=0.95
- TOD: ~1,800 val samples but near-zero detection on isolated contracts
- Manual DoS tests: 0.000–0.004 probability on all DoS contracts

**The specific DoS problem:** With pos_weight=68, a false negative on a DoS positive example costs 68× more than a false positive. Despite this, the model cannot predict DoS. The signal-to-noise ratio is too low with 137 training samples — the model averages across noisy representations.

### FM-4: Solidity Version Distribution Mismatch

**Symptom:** IntegerUO performs excellently on val set (0.8259) but fails completely on `unchecked {}` syntax contracts (0.011).

**Evidence:** The BCCC training dataset was assembled primarily from pre-0.8 Solidity contracts. Overflow in pre-0.8 is syntactically invisible (`a += b`). In 0.8+, developers must explicitly opt out with `unchecked { a += b; }`. These two are syntactically distinct in the AST, producing different graph structures. The model learned the pre-0.8 graph pattern but has not seen the 0.8+ equivalent.

### FM-5: The "External Call Confusion Cluster"

**Symptom:** Four classes (ExternalBug, CallToUnknown, MishandledException, UnusedReturn) are systematically confused with each other.

**Evidence:**
- `09_call_to_unknown`: fires ExternalBug(0.699), GasEx(0.653), IntegerUO(0.719), MishandledException(0.619), Reentrancy(0.736), UnusedReturn(0.856) — everything fires except CallToUnknown itself
- `20_unused_return_minimal`: fires ExternalBug(0.931), Reentrancy(0.805), TOD(0.659), UnusedReturn(0.878) — 4 classes simultaneously

All four classes share the presence of an external call. Their differences are:
- ExternalBug: trusting untrusted external contract's result
- CallToUnknown: calling address without known ABI
- MishandledException: ignoring call failure
- UnusedReturn: ignoring non-bool return value

These distinctions require understanding of type information and data flow — neither fully present in the current feature set.

### FM-6: Poor Specificity on Contract-Level Features

**Symptom:** The model learns some contract-level "fingerprints" that fire broadly.

**Evidence:** `04_timestamp_dependence` triggers 7/10 classes at 0.81–0.97. The combination of `payable + block.timestamp + transfer + loop` seems to activate a broad "this contract type is dangerous" signal rather than specific class signals.

**Implication:** Some classes may have been learned as contract-type indicators rather than specific vulnerability indicators. A "lottery contract with ETH" might be labeled across many classes in the training data, and the model learned the contract type rather than the individual vulnerabilities.

---

## 8. Root Cause Analysis

### 8.1 The 8-Dimensional Node Feature Ceiling

The node feature vector encodes only static structural properties:

```
[type_id, visibility, pure, view, payable, reentrant, complexity, loc]
```

**What it cannot encode:**
- Whether a return value is used (FM-5: MishandledException vs UnusedReturn)
- Whether the call target has a known ABI (FM-5: CallToUnknown vs others)
- The order of operations within a function (FM-2: Reentrancy vs safe CEI)
- Gas cost of operations (FM-4-adjacent: GasException)
- The semantic intent of a storage write (FM-1)

The GNN receives these 8 values per node and learns to aggregate them over the graph structure. For classes that require information not in these 8 dimensions, the GNN cannot learn a reliable signal regardless of training data quantity.

### 8.2 Edge Types Do Not Include Control Flow

The 5 edge types are: CALLS, READS, WRITES, EMITS, INHERITS.

These are data-flow and call-graph edges. There are no control-flow edges (sequential execution order, branch targets, loop back-edges). Without CFG edges:

- The graph for `withdraw()` where `call` precedes `balance = 0` is **identical** to one where `balance = 0` precedes `call`
- Loop detection is only implicit via complexity and loc node features

### 8.3 BCE pos_weight Pushes Toward Recall, Not Precision

For a class with pos_weight=w:
- Missing a positive: gradient = w × BCE_gradient
- False positive: gradient = 1 × BCE_gradient

With w=68 for DoS, the model should heavily predict DoS to avoid the penalty. But with only 137 training samples, the model cannot learn what DoS looks like — it simply learns that the gradient is high and may pattern-match to the most common features in those 137 contracts.

For classes like Reentrancy (w=3.12) and CallToUnknown (w=7.59), the pos_weight creates pressure to over-predict on any contract that resembles a "positive" contract from training — which includes contracts with external calls.

### 8.4 The Threshold Compensation Problem

Threshold tuning compensates for poor calibration but does not fix underlying signal quality:

- DoS threshold=0.95: the model is so underconfident about DoS that we need to accept nearly any probability ≥ 0.38 (the lowest prob in the 0.95-threshold range). This has near-zero precision.
- CallToUnknown threshold=0.70: the model fires this on almost every contract with `.call()`, so we need a high threshold to control false positives, at the cost of recall.

Threshold tuning optimized F1 on the val set but the val set is in-distribution. Out of distribution (new contracts), thresholds calibrated to the val set may perform very differently.

### 8.5 Label Co-occurrence in Training Data

A contract with Reentrancy vulnerability often *also* has MishandledException or UnusedReturn (because the developer who writes vulnerable reentrancy code may also ignore call return values). If these labels co-occur in training, the model learns to associate the graph patterns with multiple labels simultaneously — which produces the "fires everything" behavior seen in contracts 04 and 20.

---

## 9. Code Fixes Applied (v4 Sprint)

All fixes are committed and present in the current codebase.

### Fix #25 — start_epoch Bug on Model-Only Resume (CRITICAL)

**File:** `ml/src/training/trainer.py`  
**Commit:** `6f3b7d1`

**Bug:** When resuming with `resume_model_only=True` (fine-tune from base checkpoint), the code loaded `start_epoch = ckpt["epoch"] + 1`. For the v3 checkpoint at epoch 54, this set `start_epoch=55`. With `epochs=30`, the remaining epochs = 30-55+1 = -24 → the training loop returned immediately without running any epochs.

**Fix:** The resume block now distinguishes two modes:
- `resume_model_only=True` (fine-tune): load weights only, reset `start_epoch=1`, `best_f1=0.0`, `patience_counter=0`
- `resume_model_only=False` (full resume): restore epoch counter, patience, and best_f1 from checkpoint

### Fix — strict=False for lora_r Mismatch

**File:** `ml/src/training/trainer.py`  
**Commit:** `6f3b7d1`

**Bug:** `model.load_state_dict(ckpt["model"])` with `strict=True` (default) would crash if the checkpoint used `lora_r=8` and the new config used `lora_r=16` (different weight shapes for LoRA A/B matrices).

**Fix:** Changed to `strict=False` with explicit validation:
- LoRA key mismatches → `logger.warning()` (expected when changing lora_r)
- Non-LoRA key mismatches → `raise RuntimeError()` (unexpected, always a bug)

This allows fine-tuning with a different `lora_r` — the GNN, fusion, and classifier weights load from checkpoint; LoRA adapters re-initialize fresh.

### Fix — Slither/crytic_compile solc=None Bug (discovered during audit)

**File:** `ml/src/preprocessing/graph_extractor.py`

**Bug:** The `Slither()` call passed `solc=None` explicitly when `config.solc_binary` was not set. Passing `solc=None` to crytic_compile overrides the default `"solc"` string, making `cmd = [None, "--version"]` which crashes with `TypeError`.

**Fix:** Only pass the `solc` kwarg when `config.solc_binary` is explicitly set:
```python
slither_kwargs: dict = {"solc_args": solc_args, "detectors_to_run": []}
if config.solc_binary:
    slither_kwargs["solc"] = str(config.solc_binary)
sl = Slither(str(sol_path), **slither_kwargs)
```

### Auto_experiment.py — MIN_VRAM_GB Fix

**File:** `ml/scripts/auto_experiment.py`

**Bug:** `MIN_VRAM_GB = 7.0` caused the script to report "insufficient VRAM" on the RTX 3070 Laptop (8GB total), which holds ~1.6GB for display at all times, leaving ~6.4GB free.

**Fix:** Lowered to `MIN_VRAM_GB = 5.5`.

### Previously Applied Fixes (v3 Sprint, Documented Elsewhere)

| Fix | Module | Description |
|-----|--------|------------|
| #1 | dual_path_dataset.py | edge_attr shape [E,1]→[E] squeeze guard |
| #2 | predictor.py | Missing SentinelModel args on load |
| #3 | tune_threshold.py | Missing SentinelModel args + fusion_dim |
| #4 | predictor.py | Warmup dummy graph missing edge_attr |
| #5 | tune_threshold.py | prefetch_factor PyTorch 2.x warning |
| #6 | predictor.py | API "threshold"→"thresholds" key rename (BREAKING) |
| #7 | predictor.py | fusion_output_dim fallback order |
| #9 | trainer.py | focal_gamma/focal_alpha now logged to MLflow |
| #11 | trainer.py | patience_counter persistence on resume |
| #12 | trainer.py | batch-size guard on full resume |
| #13 | trainer.py | pos_weight consistency warning |
| #23 | trainer.py | Patience sidecar JSON for per-epoch persistence |
| #24 | trainer.py | Warning on missing optimizer key |

---

## 10. Assumptions and Known Limitations

### 10.1 Assumptions in the Training Pipeline

| Assumption | Status | Risk if Wrong |
|-----------|--------|--------------|
| BCCC dataset labels are accurate | **Unverified** | Model trained on noisy labels; val F1 reflects noise |
| 8-dim node features are sufficient to represent vulnerability-relevant structure | **Questionable** — see FM-1 through FM-3 | Hard ceiling on model performance without feature extension |
| Val set distribution = deployment distribution | **Likely False** — BCCC is historical data | Tuned F1 does not predict real-world performance |
| `unchecked {}` (Solidity 0.8) = pre-0.8 arithmetic overflow in training | **False** — confirmed by testing | IntegerUO on modern contracts likely under-detected |
| DoS class is learnable at 137 training samples | **False** — confirmed by testing | DoS predictions are essentially random |

### 10.2 Known Limitations

**L1 — Single contract scope:** Only the first non-dependency contract per file is analysed. Multi-contract files silently ignore contracts 2+. This is tracked as Move 9 with scaffold in `GraphExtractionConfig.multi_contract_policy`.

**L2 — No control-flow edges in graph:** The GNN cannot reason about execution ordering within functions. This fundamentally limits Reentrancy, MishandledException, and others.

**L3 — MAX_TOKEN_LENGTH=512:** Contracts longer than 512 tokens use sliding-window max-aggregation. Vulnerabilities in the latter half of a long contract may be detected in isolation but with degraded context (the GNN still sees the full graph, but each window provides independent token context).

**L4 — Solidity version bias:** The training data is heavily skewed toward pre-0.8 contracts. Modern contracts using 0.8+ safety features (checked arithmetic by default, `unchecked` blocks) have different AST structures that may not match training distribution.

**L5 — Label co-occurrence produces false correlations:** Classes that frequently co-occur in training (e.g., Reentrancy + MishandledException) are learned as correlated, causing one to fire when the other is present.

**L6 — Threshold tuning is in-distribution:** Per-class thresholds were tuned on the BCCC val set. On out-of-distribution contracts, these thresholds may be poorly calibrated.

**L7 — No confidence calibration:** The model's raw probabilities are not calibrated (Platt scaling or temperature scaling not applied). A probability of 0.85 does not mean "85% confident" in a statistically meaningful sense.

**L8 — Slither dependency:** Graph extraction requires Slither + appropriate solc version on PATH. Inference fails for contracts that Slither cannot parse (unusual pragma versions, compiler extensions, file system imports).

---

## 11. Recommendations and Priority Order

### Tier 1 — Data (Highest Leverage)

**R1.1 — DoS data collection (target: 300+ contracts)**
- Current: 137 train samples
- Sources: SmartBugs Curated, SWC-113/SWC-128 registry examples, LLM-generated (qwen2.5-coder-7b-instruct), GitHub search for unbounded-loop patterns
- Pipeline: new .sol → graph_extractor → tokenizer → append to multilabel_index.csv → add to train split only (keep val_indices.npy fixed)
- Expected impact: DoS F1 from ~0.43 (near-random) to potentially 0.60+ with 3× more data

**R1.2 — Solidity 0.8+ IntegerUO examples**
- Current training data: primarily pre-0.8 contracts; `unchecked {}` patterns underrepresented
- Add 200–300 contracts using `unchecked` blocks with verifiable overflow vulnerabilities
- This directly addresses the IntegerUO out-of-distribution gap

**R1.3 — TransactionOrderDependence examples**
- The class has 1,800 val samples but manual testing shows near-zero detection
- Current val F1=0.522 is likely achieved on simple patterns; real TOD (front-running, approve-race) is not detected
- Add curated TOD contracts from known front-running exploits

**R1.4 — Safe counterexamples with call() patterns**
- Current safe class representation is likely dominated by contracts with no external calls
- Add 500+ contracts with correct CEI patterns, pull payments, and typed interface calls — labeled as safe
- This directly addresses FM-1 (call over-generalization) and FM-2 (safe contracts flagged as vulnerable)

### Tier 2 — Training (After Tier 1 Data)

**R2.1 — Experiment 2: More epochs at lr=5e-5 from exp1 best**
- Model was still learning at epoch 30 (patience=4/7)
- Command: fine-tune from `multilabel-v4-finetune-lr1e4_best.pt`, lr=5e-5, 30 epochs
- Gate: tuned F1-macro > 0.5422
- NOTE: Do this before or concurrently with Tier 1 data, but do not expect it to fix FM-1 through FM-5

**R2.2 — Precision-focused loss (only after Tier 1 data collected)**
- Current BCE pos_weight aggressively pushes recall at the cost of precision
- After adding safe counterexamples (R1.4), consider reducing pos_weight or using focal loss with α>0.5
- Do NOT use focal loss with α=0.25 (reduces DoS gradient ~200× vs BCE pos_weight=68)

### Tier 3 — Architecture (Long-Term, Only if Tier 1+2 Fail)

**R3.1 — Add CFG edges to the graph**
- Control-flow edges (sequential, branch, loop) would allow the GNN to reason about execution order
- This would directly enable Reentrancy to distinguish `call-then-update` from `update-then-call`
- Breaking change: requires rebuilding all 68K graph files and retraining from scratch

**R3.2 — Extend node features: return-value used / call-target type**
- Add boolean features: `return_value_consumed` (boolean), `call_target_typed` (boolean)
- Would directly address the External Call Confusion Cluster (FM-5)
- Breaking change: in_channels=8 is LOCKED; changing requires full rebuild

**R3.3 — Separate DoS head with dedicated architecture**
- After exhausting data collection, if DoS F1 remains below 0.50
- A binary DoS classifier on the 128-dim fusion output, trained separately with augmented data
- Does not require retraining the base model

### Tier 4 — Calibration and Monitoring

**R4.1 — Apply temperature scaling after training**
- A single scalar T applied to logits: `sigmoid(logit / T)` makes probabilities statistically calibrated
- Tune T on the val set using negative log-likelihood
- Enables meaningful probability thresholds instead of per-class sweep

**R4.2 — Track precision separately from F1 in val metrics**
- Current val metric is F1 (harmonic mean of precision and recall)
- Add precision and recall as separate logged metrics to MLflow
- Precision dropping while recall rises is a warning sign of pos_weight over-tuning

---

## 12. Appendix — Test Contracts

Test contracts are in `ml/scripts/test_contracts/`. Each has a `// expect:` annotation. Test runner: `ml/scripts/manual_test.py`.

| File | Class | Pattern Tested | Result |
|------|-------|---------------|--------|
| 01_reentrancy_classic.sol | Reentrancy | Standard deposit/withdraw, call before update | ✓ DETECTED |
| 02_reentrancy_tricky.sol | Reentrancy | Reentrancy hidden in internal helper | ✗ MISSED |
| 03_integer_overflow.sol | IntegerUO | unchecked {} blocks (Solidity 0.8+) | ✗ MISSED |
| 04_timestamp_dependence.sol | Timestamp | block.timestamp for lottery + unlock | ~ DETECTED + 6 false positives |
| 05_denial_of_service.sol | DenialOfService | Unbounded loop over growing array | ✗ MISSED |
| 06_mishandled_exception.sol | MishandledException | call() return ignored in loop | ✗ WRONG CLASS |
| 07_tx_order_dependence.sol | TransactionOrderDependence | Approve-race + price oracle front-run | ✗ MISSED |
| 08_unused_return.sol | UnusedReturn | ERC20 transfer() + approve() return ignored | ✗ WRONG CLASS |
| 09_call_to_unknown.sol | CallToUnknown | delegatecall + raw call to state variable | ✗ WRONG CLASSES |
| 10_gas_exception.sol | GasException | Large struct copy + unbounded view + transfer() stipend | ✗ MISSED |
| 11_external_bug.sol | ExternalBug | Flash-loan oracle + untrusted callback | ✗ WRONG CLASSES |
| 12_safe_contract.sol | (safe) | Correct CEI, no timestamp, 0.8.0 | ✗ FALSE POSITIVE |
| 13_multilabel_complex.sol | Reentrancy, Timestamp, UnusedReturn | All three combined | ✗ WRONG CLASS |
| 14_reentrancy_minimal.sol | Reentrancy | Same pattern as 01, 4 lines | ✗ MISSED |
| 15_tod_minimal.sol | TransactionOrderDependence | Minimal front-run: solve() race | ✗ MISSED |
| 16_gas_minimal.sol | GasException | Loop + transfer() stipend DoS | ✗ WRONG CLASSES |
| 17_integer_simple.sol | IntegerUO | 3 unchecked operations, minimal | ✗ MISSED |
| 18_safe_no_calls.sol | (safe) | No external calls, pure state | ✓ CLEAN |
| 19_safe_with_transfer.sol | (safe) | Correct CEI with call{value:} | ✗ FALSE POSITIVE |
| 20_unused_return_minimal.sol | UnusedReturn | Single ERC20 transfer() ignored | ~ DETECTED + 4 false positives |

---

*End of audit report. For questions, see `docs/changes/` for per-session changelogs and `docs/ML_TRAINING_GUIDE.md` for architecture and training concepts reference.*
