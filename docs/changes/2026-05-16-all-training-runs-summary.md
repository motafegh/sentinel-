# SENTINEL — Complete Training Runs History
**Date:** 2026-05-16
**Author:** Internal audit session

This document is the canonical record of every training run executed for SENTINEL from v4 through v5.3.
Each section covers configuration, dataset, best results, failure modes, and lessons learned.

---

## Comparison Table

| Run | Dataset | Best Val F1 | Tuned F1 | Behavioral | Status |
|-----|---------|-------------|----------|------------|--------|
| v4 finetune lr1e4 | 68K leaky | — | 0.5422 | Not run | Active fallback |
| v5.0 full-60ep | 68K leaky | 0.5828 (ep44) | 0.5828 | 15% det / 0% spec | FAILED |
| v5.1-fix28 | 44K deduped | 0.2794 (ep8) | invalid | Never run | INVALID |
| v5.2-jk-20260515c | 44,470 | 0.1872 (ep16) | — | — | Intermediate |
| v5.2-jk-20260515c-r2 | 44,470 | 0.2823 (ep20) | 0.3373 | — | Early-stopped |
| v5.2-jk-20260515c-r3 | 44,470 | 0.3306 (ep32) | 0.3422 | 36% det / 33% spec | FAILED |
| v5.3-bce-smooth-20260516 | 44,470 | 0.2559 (ep31) | — | Not run | KILLED |

---

## v4 Baseline — `multilabel-v4-finetune-lr1e4`

### Configuration
- **Checkpoint:** `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`
- **Architecture:** v4 (GNN + CodeBERT without CrossAttentionFusion Three-Eye design)
- **Dataset:** 68,523-row original `multilabel_index.csv` (68K leaky)
- **Splits:** `ml/data/splits/` — train=47,966 / val=10,278 / test=10,279
- **Learning rate:** 1e-4 finetune

### Results
- **Tuned F1-macro:** 0.5422

### Per-Class Tuned F1 (v4 reference)
| Class | F1 | v5 floor (v4 − 0.05) |
|-------|----|----------------------|
| CallToUnknown | 0.397 | 0.347 |
| DenialOfService | 0.384 | 0.334 |
| ExternalBug | 0.434 | 0.384 |
| GasException | 0.507 | 0.457 |
| IntegerUO | 0.776 | 0.726 |
| MishandledException | 0.459 | 0.409 |
| Reentrancy | 0.519 | 0.469 |
| Timestamp | 0.478 | 0.428 |
| TOD | 0.472 | 0.422 |
| UnusedReturn | 0.495 | 0.445 |

### Critical Caveat
All v4 metrics are inflated by 34.9% cross-split content leakage. The 68K dataset was built
from BCCC, which stores the same .sol file in multiple category directories. Path-based MD5
hashing created separate rows for each copy, resulting in 7,630 content groups spanning multiple
splits. The real-world performance of v4 is likely substantially lower than 0.5422.
These numbers are used only as a relative floor for v5 comparison (v5 must beat v4 − 0.05 per class).

---

## v5.0 — `v5-full-60ep`

### Configuration
- **Checkpoint:** `ml/checkpoints/v5-full-60ep_best.pt`
- **Architecture:** v5.0 (new Three-Eye + CrossAttentionFusion, GNN hidden_dim=128)
- **Dataset:** 68K leaky (same as v4)
- **NODE_FEATURE_DIM:** 12 (v3 schema, introduced 2026-05-12)
- **NUM_EDGE_TYPES:** 7 (REVERSE_CONTAINS not yet added)
- **aux_loss_weight:** 0.1
- **GNN pooling:** ALL nodes (bug — not yet fixed to function-only)
- **Epochs scheduled:** 60; **best epoch:** 44

### Results
- **Best Val F1:** 0.5828 (evaluated at threshold=0.5)
- **Tuned F1-macro:** 0.5828 (appears strong but is invalid due to leakage)
- **Behavioral test:** 15% detection rate, 0% safe-contract specificity
  - 85% of known-vulnerable contracts not detected
  - 100% of safe contracts flagged as vulnerable
  - Model degenerated to predicting all-vulnerable for any input

### Root Causes of Behavioral Failure
- **R1 (Interface selection):** `_select_contract()` interface bug selected wrong contract in multi-contract
  .sol files, sending the wrong node features into the GNN for many contracts.
- **R2 (CFG_RETURN pool flood):** GNN used all-node mean pooling for the GNN eye. CFG_NODE_RETURN
  nodes make up ~77% of all nodes in typical contracts. These nodes carry structural-only information
  (they mark function exit points) and completely dominate the pooled embedding, washing out function-level
  vulnerability signals. The GNN eye learned to average over noise.
- **R3 (aux_loss_weight too low):** aux_loss_weight=0.1 meant the per-eye auxiliary heads contributed
  almost nothing to the gradient. The model could not develop independent representations in each eye.
  The GNN eye and TF eye converged to nearly identical embeddings.
- **R4 (No JK connections):** Without Jumping Knowledge, the GNN was a simple 4-layer stacked GAT.
  Gradients did not flow back to early layers effectively. The model effectively collapsed to a
  shallow GNN using only the last-layer representation.

### Lessons
- Dataset leakage produces misleadingly high val F1 that does not correspond to real-world detection.
- Behavioral tests (manual_test.py) are the only honest evaluation metric.
- Pool size matters enormously: dominating node type must not be CFG_RETURN.

---

## v5.1-fix28 — INVALID RUN

### Configuration
- **Architecture:** v5.1 (with R1, R2, R3 root causes fixed)
  - _select_contract() fixed
  - GNN pool restricted to FUNCTION/FALLBACK/RECEIVE/CONSTRUCTOR/MODIFIER nodes
  - aux_loss_weight raised to 0.3, warmup over 3 epochs
- **Dataset:** 44,420 deduped (`multilabel_index_deduped.csv`)
- **Splits:** `ml/data/splits/deduped/` — train=31,092 / val=6,661 / test=6,667

### History
This run suffered multiple crash/resume cycles throughout training. Each resume restarted the
scheduler from scratch, causing total_steps mismatches (20,941 vs 16,558 expected) that corrupted
the cosine annealing decay curve. The effective LR schedule was a piecewise cosine function with
re-warming at each resume — not what was intended.

- **Multiple resume crashes:** Epochs 1-8 ran cleanly
- **GNN gradient collapse at epoch 8:** GNN encoder gradients dropped to ~0, only LoRA was training
- **Scheduler skip on last resume:** total_steps computed from remaining epochs only, not full schedule
- **Best checkpoint F1:** 0.2794 — achieved only at epoch 8 before GNN collapse

### Why INVALID
The GNN collapse at epoch 8 means any result from epoch 9 onward reflects a model where the GNN
encoder is essentially frozen. The best F1=0.2794 was achieved with a partially-trained model.
No behavioral test was run. These results cannot be used to evaluate v5.1 architecture changes.

### Root Causes That Were NOT YET Fixed (carried into v5.2)
- **R4 (GNN gradient):** JK connections not yet implemented at v5.1 time
- **Residual dropout order:** `dropout(x2 + x)` instead of correct `x + dropout(x2)`
- **No REVERSE_CONTAINS edge type:** NUM_EDGE_TYPES=7, edge_emb=Embedding(7,32)
- **No per-phase LayerNorm:** gradient flow through 4 layers unstable without LN

---

## v5.2 Series — Three Sub-runs

All three sub-runs share these properties:
- Architecture: v5.2 with all 27 audit fixes applied (JK, residual dropout order, REVERSE_CONTAINS,
  per-phase LayerNorm, function-node pool, etc.)
- Dataset: 44,470 rows (44,420 deduped + 50 CEI augmentation pairs)
- Splits: `ml/data/splits/deduped/` train=31,142 / val=6,661 / test=6,667
- NUM_EDGE_TYPES=8 (REVERSE_CONTAINS=7 added, runtime-only)
- NODE_FEATURE_DIM=12, FEATURE_SCHEMA_VERSION="v3"
- GNN hidden_dim=128, 3-phase 4-layer GAT, JK attention aggregation
- CrossAttentionFusion attn_dim=256, output=128
- Three-Eye classifier: concat [384] → Linear(384,10)
- aux_loss_weight=0.3

### Sub-run 1: v5.2-jk-20260515c

**Purpose:** First clean run with all 27 fixes. Validation of basic training stability.

**Configuration:**
- PID: 84037
- eval_threshold: 0.5 (bug — not yet using 0.35)
- gradient_accumulation_steps: 4 (effective batch = 32)
- Early stop patience: 20
- Loss: BCE with per-class pos_weight (Reentrancy pos_weight = 2.82)
- epochs: ran to epoch 16

**Results:**
- Best val F1 at epoch 16: 0.1872 (eval_threshold=0.5)
- Ep1 gates: PASSED (GNN share within expected range)
- Run purpose completed: confirmed training was stable and not crashing

**Issues:**
- eval_threshold=0.5 causing massive noise: minority classes cluster in [0.35, 0.50] probability range
- Reentrancy pos_weight=2.82 causing Reentrancy overfit (not yet identified as a problem)
- This sub-run was deliberately stopped to continue as r2 with same checkpoint

### Sub-run 2: v5.2-jk-20260515c-r2

**Purpose:** Continue from r1 checkpoint, run longer.

**Configuration:**
- Resumed from: v5.2-jk-20260515c_best.pt (epoch 16)
- eval_threshold: 0.5 (still buggy — not yet fixed)
- Early stop patience: 20
- All other settings same as r1

**Training trajectory (threshold=0.5, noisy):**
- ep16 (start): 0.1872
- ep20: 0.2823 (BEST)
- ep30: early-stopped (patience=10 since ep20)

**After threshold tuning (post-hoc analysis):**
The r2 checkpoint achieved **tuned F1=0.3373** with per-class thresholds:
| Class | F1 | Threshold |
|-------|----|-----------|
| CallToUnknown | 0.284 | 0.40 |
| DenialOfService | 0.329 | 0.95 |
| ExternalBug | 0.262 | 0.40 |
| GasException | 0.407 | 0.50 |
| IntegerUO | 0.732 | 0.55 |
| MishandledException | 0.342 | 0.45 |
| Reentrancy | 0.322 | 0.40 |
| Timestamp | 0.174 | 0.45 |
| TOD | 0.283 | 0.40 |
| UnusedReturn | 0.238 | 0.35 |

**Key observations from tuned results:**
- DoS threshold=0.95: model almost never fires DoS. Maximum probability on positive DoS examples is
  around 0.3–0.4 in most cases. The tuner set threshold=0.95 because this maximized F1 by preventing
  false positives, not because the model is confident on DoS. Practically useless.
- Timestamp F1=0.174: worst class. block.timestamp is a SolidityVariableComposed, not in
  state_variables_read → zero READS edges → completely invisible in the graph. CodeBERT truncated
  to 512 tokens often misses it too.
- UnusedReturn F1=0.238 and MishandledException F1=0.342: both crippled by the return_ignored
  feature bug (always returning 0.0 due to incorrect Slither lvalue checking).

**Why early-stopped was wrong:**
The threshold=0.5 evaluation was comparing against a noisy baseline. The true metric at epoch 20
was 0.3373 after tuning, not 0.2823. The model was improving (0.1872 → 0.3373) but the noisy
metric showed apparent stagnation from epoch 20 onward, triggering early stop.

**Lesson learned:** eval_threshold must be ≤ inference threshold for minority classes. Fixed in r3.

### Sub-run 3: v5.2-jk-20260515c-r3 (FINAL v5.2 run)

**Purpose:** Continue from r2, with eval_threshold=0.35 fix, run to full convergence.

**Configuration:**
- Resumed from: v5.2-jk-20260515c-r2_best.pt (epoch 20)
- eval_threshold: **0.35** (fixed)
- Early stop patience: 20
- epochs: scheduled 60; best epoch 32; early-stopped epoch 52

**Training trajectory:**
- ep21: 0.3130
- ep22: 0.3202
- ep24: 0.3203
- ep27: 0.3282
- ep28: 0.3290
- ep32: 0.3306 **(BEST)**
- ep52: early-stopped (patience=20 since ep32)

**Post-training tuning:**
- Tuned macro F1: **0.3422**
- Thresholds saved: `ml/checkpoints/v5.2-jk-20260515c-r2_best_thresholds.json`

**Behavioral test results (FAILED):**
- 19 test contracts (known vulnerable + safe)
- Detection rate: **7/19 = 36%**
- Safe-contract specificity: **33%** (fires on safe contracts)
- Both gates FAILED (required: ≥ 80% detection, ≥ 80% specificity)

**Root causes of behavioral failure (post-mortem analysis):**
- **RC1 — Fusion gradient dominance:** CrossAttentionFusion was trained at the full base LR.
  Fusion has direct access to all embeddings and can easily overfit to shortcut patterns
  (e.g., "any external call present → Reentrancy"). With the GNN and LoRA both at lower effective
  LRs, fusion dominated the gradient signal and learned to fire Reentrancy on structural patterns
  (external calls present) rather than CEI ordering patterns.
- **RC2 — Reentrancy pos_weight=2.82:** The 2.82× positive weight pushed the model to aggressively
  fire Reentrancy on any contract with an external call. Combined with RC1, this created a model
  that fires Reentrancy on virtually every contract with an external call, including safe ones.
- **RC3 — No label smoothing:** Hard BCE targets (0 or 1) with high pos_weight pushed logits to
  extremes. Reentrancy probability on safe contracts reached 0.97 on some examples.
- **RC4 — DoS/Timestamp data starvation:** DenialOfService has only 257 training positives
  (~1 per batch of 16 at effective batch=32 with grad accum 4). Timestamp has only 1,493.
  These classes cannot converge reliably from so few examples.
- **RC5 — return_ignored always 0.0 (code bug):** The feature `return_ignored` in graph_extractor.py
  checked `op.lvalue is None` but in Slither, `op.lvalue` is never None for return-value ops — it
  is always set to a temporary variable. The correct check is whether that lvalue ID appears in
  any subsequent `op.read` set. MishandledException and UnusedReturn both depend on this feature
  and were effectively blind to actual return-value mishandling.
- **RC6 — block.timestamp invisible:** SolidityVariableComposed variables (timestamp, block.number,
  difficulty) are not in `state_variables_read` and therefore generate no READS edges in the graph.
  The Timestamp class has essentially zero graph signal. CodeBERT at 512-token truncation misses
  most timestamp usage that appears deeper in the contract.

---

## v5.3 — `v5.3-bce-smooth-20260516` — KILLED

### Configuration
- **PID:** 144823
- **Started:** 2026-05-16
- **Checkpoint:** `ml/checkpoints/v5.3-bce-smooth-20260516_best.pt`
- **Log:** `ml/logs/v5.3-bce-smooth-20260516-stdout.log`
- **MLflow experiment:** `sentinel-v5.3`
- **Dataset:** 44,470 rows (same as v5.2)

**Fixes applied (relative to v5.2-r3):**
| Fix | Setting | v5.2 value | v5.3 value |
|-----|---------|-----------|-----------|
| RC1 | fusion_lr mult | 1.0× (full LR) | 0.5× |
| RC3 | label_smooth | 0.0 (none) | 0.05 |
| RC-B | pos_weight_min_samples | — (no floor) | 3000 |
| RC-C | CEI pairs | 50 (present but not carefully verified) | 50 (same) |

**RC2 NOT fixed:** Reentrancy pos_weight was addressed indirectly by pos_weight_min_samples=3000
capping it to 1.0 (257 Reentrancy positives → well below 3000 floor → capped).

**pos_weight_min_samples=3000 effect on all classes:**
| Class | Positives | Capped to 1.0? |
|-------|-----------|----------------|
| Reentrancy | ~5,000 | Yes (capped) |
| GasException | ~5,597 | Yes (capped) |
| IntegerUO | ~15,529 | Yes (capped) |
| MishandledException | ~4,709 | Yes (capped) |
| DenialOfService | ~257 | No (kept 10.96) |
| Timestamp | ~1,493 | No (kept 4.46) |
| CallToUnknown | ~3,610 | No (raw 3.38) |
| ExternalBug | ~3,404 | No (raw 3.44) |
| TOD | ~3,391 | No (raw 3.49) |
| UnusedReturn | ~3,037 | No (raw 3.70) |

5 of the 10 classes had their pos_weight capped to 1.0. This effectively removed the class imbalance
correction for the majority classes, expecting the model to learn from balanced examples for those.

### Training Trajectory
| Epoch | Val F1 | Notes |
|-------|--------|-------|
| 1 | — | loss=0.84 |
| 23 | 0.2428 | Early progress |
| 27 | 0.2531 | Plateau beginning |
| 29 | 0.2544 | Minor improvement |
| 31 | **0.2559** | **BEST** |
| 32–46 | 0.25xx | Flat / minor variance |
| 47 | 0.2512 | Slight decline |

Loss trajectory: 0.84 (epoch 1) → 0.807 (epoch 47). Extremely slow decay — model was not
converging at normal speed. Expected loss at epoch 47 with typical convergence: ~0.65.

### Kill Decision
**Killed at epoch 47.** Decision rationale:
1. F1=0.2559 plateau had persisted for 16+ epochs with no improvement trend.
2. Best F1 (0.2559) is LOWER than v5.2-r3 best (0.3306), despite having supposedly fixed RC1 and RC3.
3. Loss barely moved (0.84 → 0.807 over 47 epochs vs typical 0.84 → 0.60 at epoch 47).
4. This suggests the problem is not hyperparameter tuning but fundamental feature schema issues.

### Root Cause of v5.3 Failure
**pos_weight_min_samples=3000 over-corrected.** By capping 5 classes to pos_weight=1.0, we removed
discriminative gradient signal for those classes. With IntegerUO (10,886 positives, 20,206 negatives),
the model still sees imbalanced data but no longer gets the pos_weight signal to weight positives
appropriately. The 5 capped classes effectively train with BCEWithLogitsLoss at equal weighting,
which for multi-label training on imbalanced data produces degraded minority-within-class gradients.

More fundamentally: the feature schema problems (return_ignored=0 always, block.timestamp invisible,
Transfer/Send invisible in CFG, loc not normalized) mean the model cannot encode the correct
discriminative features even with perfect loss weighting. Fixing hyperparameters on a broken
feature schema is the wrong approach.

**Decision:** Do not tune further. Fix root causes in v6 (see v6 plan doc).

---

## Key Lessons Across All Runs

1. **Dataset leakage inflates metrics by ~20–30%:** v4's 0.5422 and v5.0's 0.5828 are not honest
   baselines. The deduped 44K dataset reveals the true difficulty (v5.2 achieves 0.3422 without leakage).

2. **Behavioral tests expose what val F1 hides:** v5.0 had F1=0.5828 but 0% safe specificity.
   v5.2 had F1=0.3422 but only 36% detection. The model can "learn" to exploit leakage or simple
   heuristics (any external call = Reentrancy) that score well on val but fail on real contracts.

3. **Feature schema bugs compound:** return_ignored=0 affected two classes. Transfer/Send invisibility
   affects DoS and Reentrancy. block.timestamp invisibility kills Timestamp. These are not hyperparameter
   problems — they require code changes in graph_extractor.py.

4. **Token truncation is catastrophic:** With median contract length 2,469 tokens and max_length=512,
   96.2% of contracts are truncated. The CodeBERT path sees only ~21% of the median contract.
   This fundamentally limits what the text encoder can learn.

5. **eval_threshold must match probability distribution:** Using threshold=0.5 during training
   evaluation when minority classes cluster around 0.35–0.45 creates metric noise that causes
   premature early stopping. Always use eval_threshold ≤ expected inference threshold.

6. **pos_weight must be proportional, not capped by sample count:** Capping to 1.0 for classes
   above 3000 samples removes necessary gradient signal for those classes. Use actual frequency-based
   pos_weight with only a floor (≥1.0), not an upper bound.

7. **Co-occurrence in BCCC corrupts multi-label training:** 99% DoS→Reentrancy co-occurrence means
   the model cannot distinguish between "contract has DoS" and "contract has Reentrancy". It learns
   proxies (any external call pattern) rather than specific vulnerability patterns. Augmentation
   with clean single-label examples is required.
