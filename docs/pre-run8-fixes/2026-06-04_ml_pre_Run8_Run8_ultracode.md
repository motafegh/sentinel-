# Run 8 Ultracode — SENTINEL
**Created:** 2026-06-04 | **Updated:** 2026-06-05 (all pre-run changes complete, ready to launch)
**Based on:** Run 7 analysis + Phase 2 interpretability (all Tier 1+2 experiments complete)
**Code investigation:** `docs/pre-run8-fixes/FINDINGS.md` — 15 findings across 5 source files
**Run 7 best:** ep39, F1=0.3074 fixed / 0.3329 tuned
**Goal:** Break the complexity-proxy ceiling and force class-specific structural learning

> **STATUS 2026-06-05:** All code changes IMPLEMENTED. All pre-launch steps COMPLETE.
> MLflow ghost run KILLED. Thresholds extracted to `ml/calibration/temperatures_run7.json`.
> **Ready to launch.** Copy the command from Part 4.

---

## Part 1 — Why Run 7 Got Nothing

After 24 hours and 40 epochs, the effective F1 gain over Run 4 was ~zero (0.3329 tuned ≈ 0.3362 Run4). Five experiments explain exactly why.

### 1.1 The Core Failure: Complexity Proxy (L4)

The model learned one feature — `complexity` (feat[5], log1p CFG block count) — as a universal proxy for ALL 10 vulnerability classes.

| Class | complexity share | return_ignored | external_call_count | uses_block_globals |
|-------|-----------------|----------------|--------------------|--------------------|
| UnusedReturn | 34.6% | 7.7% | 7.8% | 9.9% |
| Reentrancy | 34.4% | 8.0% | 7.5% | 9.7% |
| Timestamp | 34.9% | 7.9% | 7.4% | **10.7%** |
| *every other class* | 33–36% | ~7.8% | ~7.5% | ~9.7% |

The class-specific features designed into the v8 schema show **zero discriminative elevation**. `return_ignored` (built for UnusedReturn) has the exact same gradient share whether the prediction is "yes UnusedReturn" or "no UnusedReturn". The model cannot tell these classes apart by structure — it just ranks contracts by how complex they are.

**Why this happened:** `complexity` correlates with *all* vulnerability types — complex functions have more code paths and are statistically more vulnerable to everything. The model found this shortcut early (by ep10 the feature ranking was already locked in) and training pressure never dislodged it, because the shortcut actually works reasonably well (achieves ~0.28 F1) and the entropy regularizer λ=0.005 only targets JK weight uniformity, not feature saliency.

### 1.2 The GNN Ignores Graph Topology (L2)

Remove any single edge type. Maximum prediction change across all 10 classes: **0.013** — less than the DoS sawtooth noise floor (±0.008 per prediction). The model could be presented with a graph that has only CONTAINS edges (just "function X is in contract Y") and produce essentially identical output.

**Why:** A model relying on node-level complexity statistics doesn't need relational edges. The GNN architecture is present but the GNN is functioning as a node-level MLP, not a graph reasoner.

### 1.3 JK Routing Is Near-Uniform (L1)

JK attention entropy = **1.094 / 1.099 maximum** (99.5%). All 10 classes use Phase3 at 0.367–0.377 with Phase1≈Phase2≈0.313. No per-class phase specialization. The 3-phase GNN that was supposed to route structural vs CFG vs hierarchy information differently depending on the vulnerability type… routes everything the same.

**Why:** Phase3 drifted to 0.395 by ep40 because it encodes a useful prior ("which contract family this function belongs to"). But the drift stopped before collapse and was global, not class-specific. The real problem is that without class-specific feature gradients (point 1.1), there's no training signal to differentiate JK routing by class.

### 1.4 The DoS Sawtooth Masked the Real Trend

The macro F1 "improved" from 0.2780 (ep10) to 0.3074 (ep39). But ~0.017 of that 0.030 gain is DoS, which has 65 positive val samples — each correct prediction changes F1 by 0.008. The 9-class non-DoS trend was **flat since ep20** at ~0.287–0.295. The best checkpoint just happened to catch DoS on a good epoch.

### 1.5 BUG-SL-1 Blind for 40 Epochs

`ml/src/training/training_logger.py:305` — `head[-1]` on an `OptimizedModule` raised a silently-caught exception every epoch. All structured epoch-level data was empty: AUC-ROC curves, Brier scores, ECE calibration metrics, aux head weight norms, probability distributions. MLflow F1/JK metrics survived (logged before the error), but the diagnostic visibility that would have caught the complexity proxy problem earlier was gone.

### 1.6 gnn_prefix_k Was Disabled

`--gnn-prefix-k 0` (not passed to train.py). The GNN-to-transformer prefix injection — k=48 tokens prepended to the transformer input from the GNN's Phase 2 embedding — was completely off. This was the mechanism designed to let structural patterns prime the transformer's attention before it reads the source code. 40 epochs trained without it.

### Summary Table

| Problem | Severity | Evidence | Fix for Run 8 |
|---------|----------|----------|---------------|
| Complexity proxy dominates all classes | **Critical** | L4: 34-36% all classes, zero class-specific elevation | `--drop-complexity-feature` |
| GNN ignores edge topology | **Critical** | L2: max delta 0.013 across all edge types | Remove complexity forces structural learning; `--appnp-alpha 0.2` preserves CEI signal |
| JK near-uniform, no class routing | High | L1: entropy 99.5% max | λ=0.0075; optional JK routing loss |
| DoS noise masks 9-class plateau | Medium | Run 7 training analysis §5 | Stratified sampler; DoS-aware patience |
| BUG-SL-1 (silent structured logger failure) | High | analysis doc §10 | 1-line fix in training_logger.py |
| gnn_prefix_k disabled | Medium | Run 7 config: gnn_prefix_k=0 | `--gnn-prefix-k 48` |
| fusion_lr_multiplier too high | Low | Recurring spikes 0.09–0.165 at step 100–200 | 0.5 → 0.3 |
| CEI signal dilution in Phase 2 (~1.5 effective CF hops) | High | PHASE2-RECEPTIVE-FIELD-ANALYSIS.md §2 | `--appnp-alpha 0.2` teleport |
| aux_phase2 pools over READ+OTHER (CEI noise) | Medium | audit §2.2 A-8 | CEI-only pooling (CALL+WRITE+CHECK) |
| compute_pos_weight wasted with ASL | Low | audit BUG-I2 | Guarded behind `loss_fn != "asl"` |
| Shared cache bypasses integrity validation | Medium | audit BUG-P4 | Cache loaded via DualPathDataset init |
| 227 graphs (0.55%) truncated at 1024 nodes | Low | audit BUG-C4 | `--fusion-max-nodes 2048` (default) |

---

## Part 2 — Pre-Launch Checklist (ALL COMPLETE ✅)

### Step 1: ✅ BUG-SL-1 FIXED (2026-06-04)

`ml/src/training/training_logger.py:305` — `head = getattr(head, "_orig_mod", head)` unwraps torch.compile OptimizedModule before `head[-1]`. Structured epoch data (AUC/Brier/ECE) live in Run 8.

### Step 2: ✅ Core code changes IMPLEMENTED (2026-06-05)

See Part 7 for the full list. Key: `--drop-complexity-feature`, `--appnp-alpha`, CEI pooling, BUG-P4 fix.

### Step 3: ✅ BUG-C4 QUANTIFIED (2026-06-05)

- >1024 nodes: 227 graphs (0.55%) — max=1,735
- >2048 nodes: 0 graphs (0.00%)
- Decision: `--fusion-max-nodes 2048` (now the default in TrainConfig and train.py)

### Step 4: ✅ Run 7 Thresholds Extracted (2026-06-05)

Best checkpoint (ep39) did not have `tuned_thresholds` cached (ep39 is not a multiple of `threshold_tune_interval=10`). Computed via `ml/scripts/tune_threshold.py` on the full validation set (6,236 samples).

Output: `ml/calibration/temperatures_run7.json`

**Per-class tuned thresholds (F1-macro tuned = 0.3423):**

| Class | Threshold | F1 | Precision | Recall |
|-------|-----------|-----|-----------|--------|
| CallToUnknown | 0.35 | 0.266 | 0.164 | 0.713 |
| **DenialOfService** | **0.45** | **0.457** | **0.600** | **0.369** |
| ExternalBug | 0.35 | 0.270 | 0.164 | 0.757 |
| GasException | 0.40 | 0.392 | 0.271 | 0.708 |
| **IntegerUO** | **0.50** | **0.731** | **0.706** | **0.757** |
| MishandledException | 0.35 | 0.324 | 0.206 | 0.757 |
| Reentrancy | 0.40 | 0.322 | 0.228 | 0.548 |
| Timestamp | 0.40 | 0.166 | 0.139 | 0.205 |
| TransactionOrderDependence | 0.35 | 0.257 | 0.158 | 0.692 |
| UnusedReturn | 0.35 | 0.239 | 0.147 | 0.624 |

**Key observations:** DoS needs 0.45 (high-precision mode for rare class), IntegerUO needs 0.50 (well-calibrated). Most structural-ceiling classes sit at 0.35 with high recall / low precision — they fire broadly but hit correctly only ~16-20% of the time. Timestamp at F1=0.166 confirms the structural ceiling.

### Step 5: ✅ MLflow Ghost Run Closed (2026-06-05)

Run `541345bab6864f738e484794122607bc` terminated with status KILLED via `mlflow.tracking.MlflowClient().set_terminated()`.

---

## Part 3 — Run 8 Core Changes

### 3.1 `--drop-complexity-feature` ✅

Zeroes feat[5] at GNN input. Implemented in `GNNEncoder.forward()` after dtype normalization:
```python
if self.drop_complexity:
    x = x.clone()   # clone is mandatory — .to() no-op if dtype matches
    x[:, 5] = 0.0
```
`.clone()` is critical: without it, in-place zeroing corrupts the cached graph tensor for all batches sharing the same sample.

**What to expect:** F1 drops for ep1–15 (shortcut gone). Recovery begins ep15–25 as `return_ignored`, `external_call_count`, `uses_block_globals` become discriminative.

### 3.2 `--appnp-alpha 0.2` ✅

APPNP-style Phase 1 teleport applied at each of the 3 Phase 2 layers:
```python
x = 0.2 * phase1_output.detach() + 0.8 * x
```
Prevents CEI signal dilution. Without teleport, CHECK signal reaching a WRITE node after 2 CF hops = <4% of original magnitude (diluted by avg_degree^k ≈ 5^2). Teleport keeps Phase 1 structural signal ≥20% at every Phase 2 layer.

`detach()` prevents gradient shortcut back to Phase 1 — Phase 1 gradients still flow only through JK aggregation.

### 3.3 CEI-Only aux_phase2 Pooling ✅

`aux_phase2` now pools over CFG_NODE_CALL + CFG_NODE_WRITE + CFG_NODE_CHECK only (types 8, 9, 11) instead of all 5 CFG types. READ and OTHER nodes dilute the CEI signal in the mean pool. The `cfg_eye` (4th classifier eye) still uses all 5 CFG types.

New constant: `_CEI_IDS_CPU` in `sentinel_model.py`.

### 3.4 Option B (fallback, not active): Normalize complexity

If `--drop-complexity-feature` causes catastrophic IntegerUO regression past ep25 with no recovery, batch-normalize feat[5]: `x[:, 5] = x[:, 5] - x[:, 5].mean()`. Preserves relative complexity but breaks the global-proxy shortcut. Use only as fallback.

---

## Part 4 — Run 8 Training Command

```bash
source ml/.venv/bin/activate

TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --run-name    GCB-P1-Run8-v10-20260605 \
  --experiment-name sentinel-multilabel \
  --splits-dir  ml/data/splits/v10_deduped \
  --cache-path  ml/data/cached_dataset_v10.pkl \
  --epochs      100 \
  --batch-size  8 \
  --gradient-accumulation-steps 8 \
  --lr          1e-4 \
  --gnn-lr-multiplier     2.5 \
  --fusion-lr-multiplier  0.3 \
  --gnn-prefix-k         48 \
  --gnn-prefix-warmup-epochs 5 \
  --jk-entropy-reg-lambda  0.0075 \
  --aux-loss-weight       0.30 \
  --aux-phase2-loss-weight 0.20 \
  --threshold-tune-interval 10 \
  --early-stop-patience 30 \
  --drop-complexity-feature \
  --appnp-alpha 0.2 \
  2>&1 | tee /tmp/run8_v10.log
```

> All argument names verified against `train.py` argparse (2026-06-05). `--fusion-max-nodes` defaults to 2048 — no need to pass explicitly.

### Parameter delta from Run 7

| Parameter | Run 7 | Run 8 | Reason |
|-----------|-------|-------|--------|
| `--drop-complexity-feature` | absent | **present** | L4/B4: complexity proxy suppression — the core change |
| `--appnp-alpha` | absent | **0.2** | CEI signal dilution fix: ~1.5 effective CF hops → guaranteed ≥20% Phase 1 anchor at every Phase 2 layer |
| `--gnn-prefix-k` | 0 (disabled) | **48** | L4/L2: structural priming was off; enables GNN→transformer early communication |
| `--fusion-lr-multiplier` | 0.5 | **0.3** | Recurring early-epoch fusion spikes (0.09–0.165) |
| `--jk-entropy-reg-lambda` | 0.005 | **0.0075** | A3/L1: Phase3 drifted to 0.395; stronger entropy reg |
| `--fusion-max-nodes` | 1024 | **2048 (default)** | BUG-C4: 227 graphs (0.55%) truncated |
| CEI-only aux_phase2 pooling | all 5 CFG types | **CALL+WRITE+CHECK** | Reduces dilution by READ+OTHER in mean pool |

---

## Part 5 — Monitoring Checklist for Run 8

After BUG-SL-1 fix, structured epoch data is live again. Watch these signals:

### Green indicators
- `ph2_ph1_grad_ratio` stays 0.6–0.9 (B1 confirmed healthy in Run 7; should persist)
- `jk_phase3_weight` < 0.40 throughout (λ=0.0075 should prevent hitting 0.395 again)
- `val_f1_macro_tuned` > `val_f1_macro` gap narrowing after ep10
- Per-class ECE < 0.05 for individual eyes (B2 baseline was 0.040–0.046)
- `return_ignored` gradient saliency rising above 0.10 for UnusedReturn by ep15 (run L4-style probe)
- Reentrancy F1 > 0.33 by ep20 (was still improving in Run 7)

### Red indicators (stop and investigate)
- `jk_phase3_weight` > 0.42 for two consecutive epochs + `jk_phase1_weight` < 0.26 → JK collapse risk; raise λ further
- Any class F1 below Run 7 ep10 baseline for >5 consecutive epochs after ep15 → investigate feature saliency
- L4-style probe at ep5: if `complexity` still shows 30%+ saliency, the `x.clone()` + zero-out path is not executing — check `drop_complexity=True` in model
- `gnn_grad_share` < 15% after ep10 → prefix injection destabilising transformer; reduce `--gnn-prefix-k` to 24
- `fusion_grad_norm` spikes > 0.12 still appearing → reduce `--fusion-lr-multiplier` to 0.2

### Epoch milestones

| Milestone | Epoch | What to check |
|-----------|-------|---------------|
| Complexity masking live | ep1 | L4-style 1-batch gradient probe: `complexity` saliency ≈ 0% |
| Gradient share settling | ep5 | GNN share should be 80%+ early (LoRA cold), dropping by ep10 |
| Feature learning begins | ep10 | `return_ignored` and `external_call_count` saliency visibly higher than Run 7 |
| Threshold tune 1 | ep10 | `val_f1_macro_tuned` gap narrowing vs Run 7 baseline |
| JK entropy check | ep20 | Phase3 should be < 0.37 (vs Run 7's 0.379 at ep20) |
| Break-even with Run 7 | ep25 | `val_f1_macro` should match Run 7 ep20 (0.287) or better |
| First ceiling check | ep30 | B4-style UnusedReturn probe: is `return_ignored` rising in rank? |
| APPNP effectiveness | ep30 | E1-style CEI reachability probe: is Reentrancy F1 > 0.35? |

---

## Part 6 — Optional Additions (Not Required for Run 8 to Start)

### 6.1 Auxiliary JK Routing Loss

**Motivation:** L1 showed JK is near-uniform. Explicitly penalising uniform JK weights would force per-class phase specialization.

**Implementation sketch:**
```python
# In trainer.py, after the main loss is computed:
jk_weights = model.gnn.jk_agg.last_weights  # [B, 3]
jk_routing_loss = -jk_weights.var(dim=0).mean()
loss = loss + jk_routing_lambda * jk_routing_loss
```

**Status:** Optional. Implement only if Run 8 ep20 still shows JK entropy > 1.09.

### 6.2 DEF_USE Edges (RC5)

**Motivation:** After removing complexity, the model will need structural paths to discriminate classes. DEF_USE edges are designed specifically for UnusedReturn (return value consumption trace) and Reentrancy (state mutations across call boundaries).

**Status:** Deferred. RC5 requires graph re-extraction. Start Run 8 without it. If Run 8 ep30 L2 shows model finally using edge topology, add DEF_USE for Run 9.

### 6.3 Stratified DoS Sampler

**Motivation:** DoS has 65 val positives. The ±0.008/epoch F1 noise floor makes detecting improvements in early stopping unreliable.

**Status:** Optional. 30-epoch patience handles it. Add only if early stopping triggers prematurely on DoS crashes.

---

## Part 7 — Code Change Summary

### Core (Required) — ALL DONE ✅

| File | Change | Status |
|------|--------|--------|
| `ml/src/training/training_logger.py:304` | BUG-SL-1: `head = getattr(head, "_orig_mod", head)` | ✅ 2026-06-04 |
| `ml/src/models/gnn_encoder.py` | `drop_complexity` param + `x[:, 5]=0.0` in `forward()` | ✅ 2026-06-05 |
| `ml/src/models/gnn_encoder.py` | `appnp_alpha` param + Phase 2 teleport at each of 3 layers | ✅ 2026-06-05 |
| `ml/src/models/sentinel_model.py` | `drop_complexity_feature` + `appnp_alpha` → GNNEncoder | ✅ 2026-06-05 |
| `ml/src/models/sentinel_model.py` | `_CEI_IDS_CPU` constant; `aux_phase2` pools CALL+WRITE+CHECK only | ✅ 2026-06-05 |
| `ml/src/training/trainer.py` | `TrainConfig.drop_complexity_feature` + `appnp_alpha` + `fusion_max_nodes=2048` | ✅ 2026-06-05 |
| `ml/src/training/trainer.py` | `compute_pos_weight` guarded behind `loss_fn != "asl"` (BUG-I2) | ✅ 2026-06-05 |
| `ml/src/training/trainer.py` | Cache loaded via `DualPathDataset(cache_path=...)` — fixes BUG-P4 bypass | ✅ 2026-06-05 |
| `ml/scripts/train.py` | `--drop-complexity-feature` + `--appnp-alpha` flags; `fusion_max_nodes` default=2048 | ✅ 2026-06-05 |
| `ml/src/inference/predictor.py` | Reads `drop_complexity_feature` + `appnp_alpha` + `gnn_phase2_edge_types` from saved_cfg | ✅ 2026-06-05 |
| `ml/scripts/tune_threshold.py` | `gnn_phase2_edge_types`, `fusion_max_nodes`, `drop_complexity_feature`, `appnp_alpha` in model load | ✅ 2026-06-05 |

### Secondary Fixes — DONE ✅

| File | Change | Finding |
|------|--------|---------|
| `ml/src/training/trainer.py` | `use_weighted_sampler` default: `"positive"` → `"timestamp-size"` | TrainConfig/CLI mismatch |
| `ml/src/training/trainer.py` | `import math` moved to module level | F15 |
| `ml/scripts/train.py` | `--aux-cei-loss-weight` help text updated (non-functional placeholder) | F3 |

### Optional (Not blocking Run 8)

| File | Change | Condition |
|------|--------|-----------|
| `ml/src/training/trainer.py` | JK routing loss term | Only if ep20 JK still near-uniform |
| `ml/src/training/trainer.py` | `gnn_to_bert_proj`+`prefix_type_embedding` in torch.compile list | Performance only |

---

## Part 8 — Expectations for Run 8

### What should improve
- **UnusedReturn**: 0.234 → 0.28–0.32. `return_ignored` should become discriminative after complexity removal.
- **Reentrancy**: Was still improving at +0.036/30ep. With APPNP teleport (CEI signal preserved) + gnn_prefix_k=48 should accelerate.
- **GasException**: Structurally detectable; benefits from stable Phase 2 gradient.
- **Timestamp**: Modest (+0.02–0.04) if `uses_block_globals` rises in saliency; still bounded without per-CFG-node inheritance (Run 9).

### What won't change
- **TransactionOrderDependence, ExternalBug**: Cross-contract reasoning ceiling. Same 0.245–0.250 expected.
- **DoS noise floor**: 65 val positives, same prevalence. F1 oscillates ±0.008/epoch regardless.

### What could regress (temporarily)
- **IntegerUO (ep1–15)**: Uses complexity legitimately for arithmetic-heavy functions. Drops initially, recovers as `has_loop` + `external_call_count` substitute. If still below Run 7 ep10 at ep25, complexity removal may be net negative for this class.
- **Macro F1 ep1–15**: Lower than Run 7 at same epochs. The complexity proxy gave fast early F1; the model now starts from scratch structurally.

### Target
- F1-macro (tuned) > 0.36 at ep30, > 0.38 at ep50
- Run 7 tuned was 0.3329 at ep40. With complexity removed and class-specific features active, 10–15% lift from structural classes is realistic.
- If F1-macro (fixed) < 0.25 at ep20 → abort, add back complexity as normalised (Option B)

---

*Based on Run 7 analysis (`docs/training/GCB-P1-Run7-analysis-2026-06-04.md`) and Phase 2 interpretability (`docs/interpretability/SENTINEL-Understanding-Run7.md`). Experiment references (L1, L2, L4, B1–B4, A3) correspond to `docs/interpretability/EXPERIMENT_INDEX.md`. Audit findings (A-2, A-3, A-8, D-5, D-6, BUG-P4 etc.) from `docs/pre-run8-fixes/OFFLINE-PIPELINE-AUDIT-VERIFIED.md`.*
