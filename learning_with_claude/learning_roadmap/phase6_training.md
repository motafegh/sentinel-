# Phase 6 — `training/` Teaching Roadmap

**Files:** `focalloss.py`, `losses.py`, `trainer.py`
**Total lines:** 1,902
**Sessions planned:** 14–18 (5 sessions, 5 chunks)
**Status:** Not started

---

## The Big Picture

The training module answers three questions:
1. **What signal do we optimize?** (`focalloss.py`, `losses.py`) — how to survive 44K×10 label matrix with 85%+ negatives
2. **How do we run one step?** (`trainer.py:train_one_epoch`) — gradient accumulation, AMP, DoS gradient scaling, aux losses, JK entropy reg
3. **How do we run the full job?** (`trainer.py:train`) — dataset, optimizer groups, scheduler, checkpoint resume, MLflow, guardrails

```
TrainConfig (dataclass) — all hyperparameters + validation
        │
compute_pos_weight()  ←  label_csv + train_indices
        │
SentinelModel.to(device)  ←  Phase 5
        │
Optimizer (AdamW with per-group LR):
    GNN  × 2.5   (fix gradient share collapse)
    LoRA × 0.3   (prevent CodeBERT forgetting)
    Fusion × 0.5 (prevent cross-attn Reentrancy bias)
    PrefixProj × 5.0 (cold-start needs faster LR)
        │
OneCycleLR scheduler (full epochs, Fix #32)
        │
for epoch in range(start, config.epochs+1):
    model._current_epoch = epoch              ← prefix gate
    aux_loss_weight = ramp(epoch) or config   ← Fix #33 warmup
    NC-1: reset Adam state for prefix_proj at warmup epoch
    │
    train_one_epoch():
        for batch:
            AMP autocast BF16
            forward(return_aux=True) → logits + aux
            DoS gradient scaling (partial detach)
            loss = main + λ*aux + μ*jk_entropy_reg
            loss / actual_window_size → backward
            clip_grad_norm → optimizer.step / scheduler.step / zero_grad
        │
    evaluate() → F1-macro + per-class + threshold sweep
        │
    guardrails: all-zeros collapse, class death, GNN collapse
    checkpoint → atomic save (.tmp → replace) + .state.json sidecar
    early stopping (patience counter)
```

---

## File Summary

| File | Lines | Class(es)/Functions | Role |
|------|-------|---------------------|------|
| `focalloss.py` | 143 | `FocalLoss`, `MultiLabelFocalLoss` | Focal loss variants: post-sigmoid scalar alpha vs per-class logit alpha |
| `losses.py` | 126 | `AsymmetricLoss` | ASL: asymmetric gamma for pos/neg + probability clip for easy negatives |
| `trainer.py` | 1633 | `TrainConfig`, `train_one_epoch`, `train` + helpers | Full training orchestration |

---

## Session Plan

### Session 14 — `focalloss.py` + `losses.py`
**Lines:** focalloss.py 1–143, losses.py 1–126
**Covers:**
- Class imbalance in multi-label detection: 44K×10 = 440K cells, 85%+ negative; why BCE fails
- Standard BCE review — equal weight on all cells, dominated by easy negatives
- `FocalLoss`: `(1-pt)^gamma` mechanism — down-weight easy examples; `alpha` for pos/neg balance;
  BF16 guard (Audit fix #6); post-sigmoid contract ("expects probabilities, not logits")
- `_FocalFromLogits` wrapper — inline class inside train() for sigmoid→focal chaining
- `MultiLabelFocalLoss`: per-class alpha (List[float]), raw logits input, `register_buffer` for alpha;
  alpha_t bug fix (alpha was applied to both pos AND neg, inverting rare-class weighting)
- `AsymmetricLoss` (ASL, Ridnik et al. ICCV 2021): gamma_neg > gamma_pos asymmetry;
  probability clip: `prob_neg = (prob - clip).clamp(min=0)` zero-gradients easy negatives;
  per-class gamma/clip via tensor buffers; `pos_weight` optional additive amplification;
  why pos_weight is NOT passed to ASL in trainer.py (double-amplification Run 3 collapse)

**New concepts:** Focal Loss mechanism, class imbalance in multi-label, ASL asymmetric gamma, probability clipping

---

### Session 15 — `trainer.py` Chunk 1
**Lines:** 1–492 (module constants + `TrainConfig` + helpers)
**Covers:**
- `CLASS_NAMES` list — 10 vulnerability classes, order-sensitive (DoS at index 1 for BUG-H6 masking)
- VRAM helpers: `_vram_pct()` uses `memory_reserved` not `memory_allocated` — why reserved matters
- `_parse_version()` — version-aware resume warning for pre-v5.2 checkpoints
- `TrainConfig` dataclass: key hyperparameter groups and rationale
  - `eval_threshold=0.35` vs `threshold=0.5` — why lower training threshold prevents patience noise
  - `gradient_accumulation_steps=8` → effective batch 64
  - Per-group LR multipliers (gnn × 2.5, lora × 0.3, fusion × 0.5, prefix_proj × 5.0)
  - `asl_gamma_neg=2.0` (not 4.0) — BUG-C4: γ⁻=4 caused all-zeros collapse with 60% zero-label rows
  - `asl_clip=0.01` (not 0.05) — BUG-M2: hard boundary at 0.05 caused oscillation
  - Per-class label smoothing dict — calibrated noise rates per vulnerability class
  - `dos_loss_weight=0.5` — BUG-H6: DoS had 3 samples originally; now 0.5 with ~243 positives
- `__post_init__` validation: gnn_layers guard, accumulation guard, class_name guard
- `compute_pos_weight()`: sqrt-scaled ratio `√((N-pos)/pos)`, `pos_weight_min_samples` cap,
  `pos_weight_cap` ceiling — why sqrt not linear (linear at 120× for DoS = gradient spikes)
- `evaluate()`: `model.eval()`, `torch.no_grad()`, `sigmoid(logits.float())` post-AMP cast,
  F1-macro/micro + Hamming loss, `tune_thresholds` sweep (19 candidates per class, BUG-M8)

**New concepts:** Hamming loss, threshold tuning, pos_weight scaling, dataclass `__post_init__` validation

---

### Session 16 — `trainer.py` Chunk 2
**Lines:** 498–698 (`train_one_epoch` + `_grad_norm`)
**Covers:**
- Function signature: 10 parameters including AMP, aux_loss_weight, gradient_accumulation_steps, class_eps, dos_loss_weight, jk_entropy_reg_lambda
- Per-class label smoothing application: `labels = labels * (1 - class_eps) + 0.5 * class_eps`
  — symmetric around 0.5, not standard 1/C smoothing
- AMP autocast block: `torch.amp.autocast(device, dtype=torch.bfloat16, enabled=use_amp)`
- DoS gradient scaling (BUG-H6): `logits_for_loss[:, dos_idx] = w * logit + (1-w) * logit.detach()`
  — partial gradient via blend; predictions unaffected (uses original logits for inference)
- Loss composition: `main_loss + aux_loss_weight * (gnn + tf + fused) / actual_window_size`
- `actual_window_size` vs fixed `accum_steps` — tail window fix; why wrong division underscales last gradient
- JK entropy regularizer (C-3): `lambda * (log(K) - jk_entropy)` — penalizes collapse to one phase
- Gradient flow: `.backward()` → `clip_grad_norm_` → read norms (Fix #28: after clip, before zero_grad) → `optimizer.step()` → `scheduler.step()` → `zero_grad(set_to_none=True)`
- Accumulation condition: `(batch_idx + 1) % accum_steps == 0 or is_last_batch`
- GNN collapse detection: 3-consecutive log_interval streak < 10% gradient share
- NaN loss handling: count nan batches, skip from total_loss, warn if > 5%
- `_grad_norm()`: L2 norm of `.grad.detach().float()` — float() needed for BF16 training

**New concepts:** Gradient accumulation mechanics, partial gradient detach pattern, JK entropy regularization, NaN batch policy

---

### Session 17 — `trainer.py` Chunk 3
**Lines:** 744–1200 (`train()` setup: dataset → optimizer → compile → scheduler → resume)
**Covers:**
- Environment + backend setup: `TRANSFORMERS_OFFLINE=1`, TF32, `cudnn.benchmark`
- Per-run file log: loguru `logger.add()` with append mode, not stdlib logging.FileHandler
- Shared cache: `_shared_cache` dict loaded once, `.cached_data` assigned to both datasets
  — halves RAM (2.28 GB → once); fork workers inherit via copy-on-write
- DataLoader kwargs: `pin_memory`, `persistent_workers`, `prefetch_factor=4`,
  `multiprocessing_context="fork"` (CUDA-safe: workers never call CUDA)
- `WeightedRandomSampler` vs shuffle: mutual exclusion, 3× weight for any-vuln rows (BUG-H10)
- `compute_pos_weight` call + why NOT passed to ASL (double-amplification)
- C-1 GNN dtype check: `next(model.gnn.conv1.parameters()).dtype == float32` — catches BF16 pollution regression
- Per-group optimizer params: partition by `pname.startswith()` / `"lora_"` — `_seen_param_ids` prevents double-counting; `weight_decay=0` for LoRA (standard PEFT practice)
- `AdamW(..., fused=True)` — fused kernel on CUDA, single kernel for weight update (faster)
- `torch.compile` submodule strategy: `"gnn", "fusion", "classifier"` compiled, `"transformer"` skipped
  — why transformer causes graph breaks (HuggingFace Python-level control flow)
- `dynamic=True` compile flag: handles variable node/edge counts without recompilation
- OneCycleLR: `max_lr=_max_lrs` (list matches param group order), Fix #32 (full epochs always)
- Checkpoint resume: `strict=False` with LoRA key split; optimizer param_group count guard; scheduler total_steps match guard

**New concepts:** `fused=True` AdamW, `torch.compile` submodule strategy, OneCycleLR max_lr list, `strict=False` load with key inspection, multiprocessing fork + CoW

---

### Session 18 — `trainer.py` Chunk 4
**Lines:** 1200–1633 (epoch loop + MLflow + guardrails + checkpoint save + early stopping)
**Covers:**
- `model._current_epoch = epoch` — prefix gate hook (connects to Session 12 select_prefix_nodes)
- Fix #33 aux loss warmup: `warmup_frac = (epoch - 1) / warmup_epochs` — starts at 0 on epoch 1
- NC-1: `optimizer.state[p] = {}` for prefix_proj params at warmup transition — clears stale Adam m1/m2
- `train_one_epoch` call + `evaluate` call order
- Fix #27: `gc.collect()` + `torch.cuda.empty_cache()` BETWEEN epochs (not mid-epoch)
- JK attention weight logging (Phase 2-C1): reads `model.gnn.jk.last_weights` cache;
  JK STD collapse alert (all stds < 0.05); phase dominance alert (> 80%)
- Prefix attention diagnostic call: `next(iter(val_loader))` creates fresh iterator (does NOT skip validation)
- Training guardrails (BUG-M10): all-zeros collapse (Hamming > 0.85 × 3 epochs), class death (F1=0 × 5 epochs), GNN collapse (gnn_share < 10% × 5 epochs)
- Checkpoint atomic save: `.tmp` path, `_tmp_path.replace(checkpoint_path)` — POSIX atomic rename
- `._orig_mod.` key stripping — torch.compile state_dict key normalization
- `.state.json` sidecar: `{"epoch", "patience_counter", "best_f1"}` — resilient patience tracking
- MLflow `log_artifact` OUTSIDE epoch loop (Fix #6)
- Early stopping break + final return dict

**New concepts:** Atomic file rename on POSIX, Adam state reset pattern, sidecar file pattern for crash resilience, `gc.collect()` + CUDA cache between epochs

---

## New Concepts This Phase

| Concept | First appears | Why it matters |
|---------|--------------|----------------|
| Focal Loss | Session 14 | Core technique for imbalanced classification; ubiquitous in object detection and multi-label tasks |
| Asymmetric Loss (ASL) | Session 14 | State-of-the-art for multi-label imbalance; independent gamma for pos/neg is the key insight |
| Probability clipping | Session 14 | Easy negative removal — zeros gradient for confident negatives; avoids hard boundary with small clip |
| Gradient accumulation | Session 16 | Effective batch size without VRAM cost; tail window division is a subtle correctness point |
| Partial gradient detach | Session 16 | DoS gradient scaling — blends gradient without modifying predictions; general technique for per-class gradient control |
| JK entropy regularization | Session 16 | Penalty for JK attention collapse; connects to `_JKAttention.jk_entropy` from Session 8 |
| Per-group LR | Session 17 | Different parameter families need different learning rates; GNN/LoRA/Fusion at 2.5×/0.3×/0.5× fixes gradient share collapse |
| `fused=True` AdamW | Session 17 | Single CUDA kernel for weight update; faster than element-wise PyTorch ops |
| torch.compile submodule | Session 17 | Isolates HuggingFace graph breaks while still compiling GNN/fusion |
| OneCycleLR fix | Session 17 | Scheduler total_steps must match original run to resume correctly |
| Atomic checkpoint save | Session 18 | `.tmp` → rename prevents corrupted checkpoint on crash; POSIX rename is atomic |
| Adam state reset | Session 18 | NC-1: stale m1/m2 from warmup suppression must be cleared before prefix fires |

---

## Anticipated Audit Flags

| ID | File | Issue |
|----|------|-------|
| A35 | `trainer.py:~1003` | `_FocalFromLogits` is an inline class inside `train()` — cannot be pickled for multiprocessing; isinstance checks fail |
| A36 | `trainer.py:~1486` | `next(iter(val_loader))` for prefix diagnostic creates a separate iterator — safe, but wasteful; diagnostic runs on the SAME first batch every epoch, not a representative sample |
| A37 | `trainer.py:~1613` | `.state.json` sidecar written AFTER checkpoint; crash between the two leaves stale sidecar from previous epoch |

---

## Cross-File Dependencies

```
focalloss.py    ──► trainer.py    (FocalLoss → _FocalFromLogits wrapper)
losses.py       ──► trainer.py    (AsymmetricLoss as default loss_fn="asl")

trainer.py      ──► sentinel_model.py    (SentinelModel instantiation, _current_epoch, return_aux)
trainer.py      ──► gnn_encoder.py       (model.gnn.jk.last_weights for JK diagnostics)
trainer.py      ──► dual_path_dataset.py (DualPathDataset, dual_path_collate_fn — Phase 4, not yet taught)
trainer.py      ──► fusion_layer.py      (CrossAttentionFusion — compiled as submodule)
```
