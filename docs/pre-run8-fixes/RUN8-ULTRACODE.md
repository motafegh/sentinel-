# Run 8 Ultracode — SENTINEL
**Date:** 2026-06-04  
**Based on:** Run 7 analysis + Phase 2 interpretability (all Tier 1+2 experiments complete)  
**Run 7 best:** ep39, F1=0.3074 fixed / 0.3329 tuned  
**Goal:** Break the complexity-proxy ceiling and force class-specific structural learning

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
| Complexity proxy dominates all classes | **Critical** | L4: 34-36% all classes, zero class-specific elevation | Remove or normalize `complexity` feature |
| GNN ignores edge topology | **Critical** | L2: max delta 0.013 across all edge types | Enable gnn_prefix_k=48; remove complexity forces the model to find structure |
| JK near-uniform, no class routing | High | L1: entropy 99.5% max | λ=0.0075; optional JK routing loss |
| DoS noise masks 9-class plateau | Medium | Run 7 training analysis §5 | Stratified sampler; DoS-aware patience |
| BUG-SL-1 (silent structured logger failure) | High | analysis doc §10 | 1-line fix in training_logger.py |
| gnn_prefix_k disabled | Medium | Run 7 config: gnn_prefix_k=0 | --gnn-prefix-k 48 |
| fusion_lr_multiplier too high | Low | Recurring spikes 0.09–0.165 at step 100–200 | 0.5 → 0.3 |

---

## Part 2 — Immediate Pre-Run 8 Actions (ordered)

### Step 1: Fix BUG-SL-1 (5 minutes)

**File:** `ml/src/training/training_logger.py`, line 305

**Current code:**
```python
head = getattr(model, "aux_phase2", None)
if head is None:
    return result
# aux_phase2 is nn.Sequential — inspect the final Linear layer
final_linear = head[-1]
```

**Fixed code:**
```python
head = getattr(model, "aux_phase2", None)
if head is None:
    return result
head = getattr(head, "_orig_mod", head)   # unwrap torch.compile wrapper
final_linear = head[-1]
```

**Verify:** Run `python -c "import torch; seq = torch.nn.Sequential(torch.nn.Linear(10,5)); wrapped = torch._dynamo.eval_frame.OptimizedModule(seq, None); inner = getattr(wrapped, '_orig_mod', wrapped); print(inner[-1])"` — should print the Linear layer without error.

### Step 2: Run Threshold Calibration

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/calibration/calibrate_thresholds.py \
  --checkpoint ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt \
  --out ml/calibration/temperatures_run7.json \
  --split val
```

Expected output: per-class optimal thresholds, should show DoS threshold ~0.20–0.24 (not 0.35), consistent with the +0.032 tuned/fixed F1 gap.

### Step 3: Close MLflow Ghost Run

```bash
source ml/.venv/bin/activate
mlflow runs set-terminated \
  --run-id 541345bab6864f738e484794122607bc \
  --status KILLED
```

### Step 4: Quantify BUG-C4 (node truncation scope)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python - <<'EOF'
import torch
from pathlib import Path

graph_dir = Path("ml/data/graphs")
splits = {
    "train": open("ml/data/splits/v10_deduped/train.txt").read().splitlines(),
    "val":   open("ml/data/splits/v10_deduped/val.txt").read().splitlines(),
}
threshold = 1024

for split_name, ids in splits.items():
    over = 0
    for cid in ids:
        pt = graph_dir / f"{cid}.pt"
        if not pt.exists():
            continue
        g = torch.load(pt, map_location="cpu", weights_only=True)
        n = g.x.shape[0] if hasattr(g, "x") else 0
        if n > threshold:
            over += 1
    print(f"{split_name}: {over}/{len(ids)} graphs > {threshold} nodes ({over/len(ids)*100:.2f}%)")
EOF
```

**Decision rule:** If >1% of train graphs exceed 1024 nodes, increase `fusion_max_nodes=2048` for Run 8.

---

## Part 3 — Run 8 Core Change: Remove `complexity` Feature

This is the most impactful change. The model cannot learn class-specific structural patterns while `complexity` provides a good-enough universal proxy.

### Option A (Recommended): Zero-out complexity at model input

**Rationale:** Preserves the v8 schema on disk (no re-extraction), backward-compatible with all existing .pt graph files. Implemented in `GNNEncoder.forward()` by masking `feat[5]=0` before embedding.

**File:** `ml/src/models/gnn_encoder.py`, in the `forward()` method, just before the type embedding is concatenated.

Locate the section that builds the input tensor (around the `forward` method where `x` is the raw feature matrix). Add:

```python
# Run 8: zero-out complexity (feat[5]) — universal proxy suppression (L4/B4)
# complexity dominated all 10 class gradients at 34-36%; removing it forces
# the model to use class-specific features (return_ignored, external_call_count, etc.)
x = x.clone()
x[:, 5] = 0.0
```

Place this **before** the skip-connection Linear and type embedding concatenation, at the top of `forward()` after input validation.

**Add a train.py flag to enable/disable:**

```bash
--drop-complexity-feature   # bool flag, default False for backward compat
```

Pass to GNNEncoder constructor as `drop_complexity: bool = False` and implement the masking conditionally.

### Option B: Normalize complexity relative to batch

Subtract the batch mean: `x[:, 5] = x[:, 5] - x[:, 5].mean()`. This removes the absolute scale information while preserving relative complexity within a batch. Less aggressive than full removal but also less likely to force the model to find alternative features.

**Recommendation:** Use Option A (zero-out) for Run 8. If it causes regression on IntegerUO (which legitimately uses complexity as one signal), fall back to Option B.

### What to expect after removing complexity

- First 5–10 epochs: F1 will drop below Run 7 ep1 levels. The model's easy shortcut is gone.
- Epochs 10–20: The model should start learning `return_ignored`, `external_call_count`, `uses_block_globals` as discriminative signals — these are the features that were being suppressed.
- If the model doesn't recover by ep25: the remaining 10 features are insufficient and we need to revisit the feature set, not add complexity back.
- IntegerUO may temporarily regress (it was the class most appropriately using complexity-like signals for arithmetic overflow patterns).

---

## Part 4 — Run 8 Training Command

```bash
source ml/.venv/bin/activate

TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --run-name    GCB-P1-Run8-v10-20260605 \
  --experiment-name sentinel-multilabel \
  --data-dir    ml/data \
  --split-dir   ml/data/splits/v10_deduped \
  --cache-file  ml/data/cached_dataset_v10.pkl \
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
  --aux-loss-warmup-epochs 8 \
  --aux-phase2-weight     0.20 \
  --threshold-tune-interval 10 \
  --patience 30 \
  --drop-complexity-feature \
  2>&1 | tee /tmp/run8_v10.log
```

### Parameter delta from Run 7

| Parameter | Run 7 | Run 8 | Reason |
|-----------|-------|-------|--------|
| `--gnn-prefix-k` | 0 (disabled) | **48** | L4/L2: structural priming was off; enables GNN→transformer early communication |
| `--fusion-lr-multiplier` | 0.5 | **0.3** | Recurring early-epoch fusion spikes (0.09–0.165); 4-eye arch routes more loss through fusion |
| `--jk-entropy-reg-lambda` | 0.005 | **0.0075** | A3/L1: Phase3 drifted to 0.395; stronger entropy reg to hold diversity |
| `--drop-complexity-feature` | absent | **present** | L4/B4: complexity proxy suppression — the core change |

Everything else unchanged (same data, same splits, same batch size, same aux warmup schedule).

---

## Part 5 — Monitoring Checklist for Run 8

After BUG-SL-1 is fixed, structured epoch data will be live again. Watch for:

### Green indicators
- `ph2_ph1_grad_ratio` stays in 0.6–0.9 range (B1 confirmed healthy in Run 7; should stay)
- `jk_phase3_weight` < 0.40 throughout (stronger λ=0.0075 should prevent hitting 0.395 again)
- `val_f1_macro_tuned` > `val_f1_macro` gap begins closing after ep10 (calibration improving)
- Per-class ECE in structured logger < 0.05 for individual eyes (B2 baseline was 0.040–0.046)
- `return_ignored` gradient saliency rising above 0.10 for UnusedReturn (run L4 at ep15, ep30)
- Reentrancy F1 > 0.33 by ep20 (was still improving in Run 7, this should continue)

### Red indicators (stop and investigate)
- `jk_phase3_weight` > 0.42 for two consecutive epochs + `jk_phase1_weight` < 0.26 → JK collapse risk; raise λ further or add JK routing loss
- Any class F1 drops below its Run 7 ep10 baseline for more than 5 consecutive epochs after ep15 → complexity removal may have broken that class; investigate feature saliency
- `complexity` masking verification: run L4-style gradient saliency at ep5. If complexity still shows 30%+ saliency, the masking isn't working (check the x.clone() + zero-out code path is actually executing)
- `gnn_grad_share` < 15% after ep10 → prefix injection may be destabilising the transformer; reduce `--gnn-prefix-k` to 24
- `fusion_grad_norm` spikes > 0.12 still appearing despite multiplier=0.3 → reduce further to 0.2

### Epoch milestones

| Milestone | When | What to check |
|-----------|------|---------------|
| Complexity masking live | ep1 | Verify via L4-style 1-batch gradient probe: `complexity` saliency should be ~0% |
| Gradient share settling | ep5 | GNN share should be 80%+ early (LoRA cold), dropping by ep10 |
| Feature learning begins | ep10 | `return_ignored` and `external_call_count` saliency should be visibly higher than in Run 7 |
| Threshold tune 1 | ep10 | `val_f1_macro_tuned` gap should narrow vs Run 7 (better calibration from the start) |
| JK entropy check | ep20 | Phase3 should be < 0.37 (vs Run 7's 0.379 at ep20) |
| Break-even with Run 7 | ep25 | `val_f1_macro` should match Run 7 ep20 (0.287) or better |
| First ceiling check | ep30 | Run B4-style UnusedReturn probe: is `return_ignored` rising in rank? |

---

## Part 6 — Optional Additions (Not Required for Run 8 to Start)

These would improve Run 8 further but carry implementation risk. Add only if the core changes are implemented cleanly.

### 6.1 Auxiliary JK Routing Loss

**Motivation:** L1 showed JK is near-uniform. Explicitly penalising uniform JK weights would force per-class phase specialization.

**Implementation sketch:**
```python
# In trainer.py, after the main loss is computed:
jk_weights = model.gnn.jk_agg.last_weights  # [B, 3]
# Target: maximize variance of JK weights across samples in the batch
# Loss = -var(jk_weights, dim=0).mean()  → penalise uniform weights
jk_routing_loss = -jk_weights.var(dim=0).mean()
loss = loss + jk_routing_lambda * jk_routing_loss
```

**Risk:** Could destabilise early training if λ is too high. Start with `jk_routing_lambda=0.001` and monitor `jk_phase1/2/3_std` in MLflow.

**Status:** Optional. Implement only if Run 8 ep20 still shows JK entropy > 1.09.

### 6.2 DEF_USE Edges (RC5)

**Motivation:** L2 showed even removing DEF_USE barely hurts predictions (delta 0.010 for IntegerUO) — but that's because the current model doesn't USE them. After removing complexity, the model will need structural paths to discriminate classes. DEF_USE edges are the ones designed specifically for UnusedReturn (trace return value consumption) and Reentrancy (track state mutations across call boundaries).

**Status:** Deferred. RC5 requires graph re-extraction (new graph builder, new cache). Start Run 8 without it. If Run 8 shows the model finally using other edge types (L2 run at ep30), add DEF_USE for Run 9.

### 6.3 Stratified DoS Sampler

**Motivation:** DoS has 65 val positives and ~216 train positives (estimated). The DoS F1 noise floor (±0.008/epoch) makes it impossible to detect improvements in early stopping.

**Implementation:** Weight DoS-positive samples by `total_samples / (10 * dos_count)` in the training sampler. Already available infrastructure via `timestamp_sampler` in Run 5 — apply same pattern to DoS.

**Status:** Optional. The 30-epoch patience handles the noise correctly. Add if Run 8's early stopping triggers prematurely on DoS crashes.

### 6.4 fusion_max_nodes=2048

**Motivation:** BUG-C4. Quantify first (Step 4 above). If >1% of graphs exceed 1024 nodes, the CrossAttentionFusion is silently dropping nodes from large contracts.

**Cost:** Doubles VRAM for the fusion layer (~400MB on RTX 3070). Will it fit? Current peak VRAM in Run 7 was ~7.2GB on an 8GB card. 2048 fusion adds ~300–400MB. May require `--batch-size 6` or `--gradient-accumulation-steps 10` to compensate.

**Status:** Decide after quantifying BUG-C4 scope.

---

## Part 7 — Code Change Summary

### Required (must do before Run 8)

| File | Change | Lines |
|------|--------|-------|
| `ml/src/training/training_logger.py` | BUG-SL-1: add `head = getattr(head, "_orig_mod", head)` | ~305 |
| `ml/src/models/gnn_encoder.py` | Add `drop_complexity: bool` constructor arg and masking in `forward()` | ~160, ~395 |
| `ml/scripts/train.py` | Add `--drop-complexity-feature` flag, wire to GNNEncoder | ~217 |

### Optional

| File | Change | Condition |
|------|--------|-----------|
| `ml/src/training/trainer.py` | JK routing loss term | Only if ep20 JK still near-uniform |
| `ml/scripts/train.py` | `--jk-routing-lambda` arg | Same |
| `ml/src/models/sentinel_model.py` | `fusion_max_nodes=2048` | Only if >1% graphs exceed 1024 nodes |

---

## Part 8 — Expectations for Run 8

### What should improve
- **UnusedReturn**: Should rise from 0.234 toward 0.28–0.32 if `return_ignored` becomes discriminative after complexity removal
- **Reentrancy**: Was still improving at +0.036 per 30 epochs; with better CFG priming (gnn_prefix_k=48) should accelerate
- **GasException**: Structurally detectable class; should benefit from Phase2 having more stable gradient
- **Timestamp**: Modest improvement possible if `uses_block_globals` rises in saliency; still bounded without data-flow provenance

### What won't change
- **TransactionOrderDependence, ExternalBug**: Cross-contract reasoning ceiling. Same 0.245–0.250 expected.
- **DoS noise floor**: 65 val positives, same prevalence. F1 will still oscillate ±0.008/epoch.

### What could regress
- **IntegerUO (first 10 epochs)**: Uses complexity legitimately for arithmetic-heavy functions. Will drop initially, should recover as the model finds alternative signals in `has_loop` + `external_call_count`.
- **Macro F1 epochs 1–15**: Lower than Run 7 at the same epochs. The complexity proxy gave fast early F1; losing it means slower start. Don't panic until ep20.

### Target
F1-macro (tuned) > 0.36 at ep30, > 0.38 at ep50. Run 7 tuned was 0.3329 at ep40. With complexity removed and class-specific features active, a 10–15% lift from the structural classes is realistic.

---

*Generated from Run 7 analysis (`docs/training/GCB-P1-Run7-analysis-2026-06-04.md`) and Phase 2 interpretability (`docs/interpretability/SENTINEL-Understanding-Run7.md`). All experiment references (L1, L2, L4, B1, B2, B3, B4) correspond to `docs/interpretability/EXPERIMENT_INDEX.md`.*
