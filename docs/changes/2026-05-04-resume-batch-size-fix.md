# Resume Fixes: Batch-Size Guard, patience_counter Persistence, pos_weight Warning

Date: 2026-05-04  
Author: motafegh  
Fixes: #11, #12, #13

---

## Background

After the v2 checkpoint at epoch 37 (best_f1=0.4629) was resumed with
`--no-resume-model-only` and `--batch-size 32` (the checkpoint was trained at
`batch_size=16`), the resumed run showed declining F1 (0.4552 at epoch 41,
0.4427 at epoch 42) and loss spikes to ~1.86 and ~1.53. The training was
stopped after epoch 43 started.

Root-cause analysis identified three distinct bugs in `trainer.py` and one
missing CLI guard in `train.py`.

---

## Fix #11 — `patience_counter` not saved/restored on checkpoint resume

### Problem

`patience_counter` was declared in `train()` as `patience_counter = 0` and
never written into the checkpoint dict:

```python
# BEFORE — checkpoint dict did NOT include patience_counter
torch.save({
    "model":     model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch":     epoch,
    "best_f1":   best_f1,
    "config":    {...},
}, checkpoint_path)
```

On resume the counter was reset to zero:

```python
# BEFORE — always reset, history lost
start_epoch      = 1
best_f1          = 0.0
patience_counter = 0   # ← never restored
```

### Effect

If the model had already accumulated N consecutive non-improving epochs before
the checkpoint was saved, those N epochs of patience budget were silently
forgiven on resume. A model could train `early_stop_patience × N_resumes`
epochs without improvement instead of `early_stop_patience` epochs total.
With `patience=7` and two resumes this could double or triple the allowed
stagnation budget.

### Fix

1. **Save** `patience_counter` in the checkpoint dict alongside `best_f1`.
2. **Restore** it with `ckpt.get("patience_counter", 0)` (defaults to 0 for
   backward compatibility with old checkpoints that pre-date this fix).
3. **Log** the restored counter at resume time so it is visible in the
   training log.

```python
# AFTER — restored from checkpoint
patience_counter = ckpt.get("patience_counter", 0)
logger.info(
    f"Resumed from epoch {start_epoch-1} | "
    f"best_f1={best_f1:.4f} | "
    f"patience_counter={patience_counter}/{config.early_stop_patience}"
)
```

---

## Fix #12 — No guard when `batch_size` changes on full resume

### Problem

When `--no-resume-model-only` is used, `trainer.py` restores the full
optimizer state (Adam `m` and `v` moment vectors) from the checkpoint. These
moment vectors are exponential moving averages of gradients and squared
gradients accumulated over all training steps. Their scale is calibrated to
the gradient noise level of the **original** batch size.

When `batch_size` changes (e.g. 16 → 32), `steps_per_epoch` changes
proportionally (2998 → 1499 in the affected run). The gradient variance at
each step is different because each batch is drawn from a different number
of samples. Loading the old `m` and `v` into the new training loop causes:

- **Adam learning-rate inflation**: the `v` (second moment) underestimates
  gradient variance at the new batch size, inflating the effective per-step
  LR for the first several hundred steps.
- **Loss spikes**: visible in the logs at epochs 41 (batch 1300: loss=0.8633,
  batch 1400: loss=0.9620) and epoch 42 (batch 200: loss=1.8588, batch 1200:
  loss=1.5339).
- **Declining F1**: DenialOfService collapsed from 0.267 → 0.154 between
  epochs 41 and 42 as the rarest class was destabilised by the overshot
  gradient updates.
- **Recovery takes 5–10 epochs** due to `β2=0.999` — the second moment
  decays slowly, so the stale calibration persists for hundreds of steps.

### Fix

1. **Detection**: compare `ckpt_cfg.get("batch_size")` to `config.batch_size`
   at resume time.

2. **Warning** (always): log a prominent multi-line warning when a mismatch
   is detected on full resume, explaining the risk and providing three
   recommended actions.

3. **New flag `force_optimizer_reset`**: added to both `TrainConfig` and
   the CLI as `--resume-reset-optimizer`. When set alongside
   `--no-resume-model-only`, the optimizer and scheduler state are
   discarded despite being present in the checkpoint. Model weights and
   `patience_counter` (Fix #11) are still restored. This gives the correct
   epoch counter while starting AdamW fresh.

4. **Resume guide added to `train.py` docstring**: documents the four
   cases and which flags to use.

### Recommended usage after this fix

| Situation | Correct flags |
|---|---|
| Same config, epochs extended | `--no-resume-model-only` |
| batch_size changed (cleanest) | _(no flags — model-only resume)_ |
| batch_size changed, keep epoch counter | `--no-resume-model-only --resume-reset-optimizer` |
| Any hyperparameter change | _(no flags — model-only is always safe)_ |

---

## Fix #13 — `pos_weight` recomputed fresh but optimizer state restored from checkpoint

### Problem

`compute_pos_weight()` always runs from scratch at training start, using the
current training split. When a full resume is performed, the Adam moment
vectors were accumulated under the checkpoint's `pos_weight` values, which
came from the same training split (because splits are fixed at creation with
a seeded shuffle). In the normal case this is a no-op difference.

However, if splits were ever regenerated between runs, the effective loss
scale changes while the optimizer moments reflect the old scale — a silent
inconsistency with no log evidence.

### Fix

Added a `logger.warning()` when `loss_fn="bce"` AND a full resume is active,
explaining the consistency assumption. The warning explicitly states when
this is safe (splits not regenerated) and when it is not (splits changed).
No behaviour change — purely diagnostic.

---

## Files Changed

| File | Changes |
|---|---|
| `ml/src/training/trainer.py` | Fix #11 (patience_counter save/restore), Fix #12 (batch-size guard + force_optimizer_reset field), Fix #13 (pos_weight warning), updated module docstring |
| `ml/scripts/train.py` | `--resume-reset-optimizer` flag added, `--no-resume-model-only` help updated with batch-size warning, resume guide added to module docstring |
| `docs/changes/2026-05-04-resume-batch-size-fix.md` | This file |

---

## How to Resume the Stopped Run Correctly

The run `multilabel-v2-edge-attr-ext` was stopped at epoch 43 batch 903.
The last saved checkpoint is `ml/checkpoints/multilabel_crossattn_v2_best.pt`
(best_f1 from the v2 run, epoch before stale-momentum degradation).

**Recommended command (model-only resume, cleanest):**

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python -m ml.scripts.train \\
    --run-name multilabel-v2-edge-attr-ext-clean \\
    --epochs 60 \\
    --batch-size 32 \\
    --resume ml/checkpoints/multilabel_crossattn_v2_best.pt
    # NOTE: do NOT use --no-resume-model-only
    # Fresh AdamW + fresh OneCycleLR calibrated to 1499 steps/epoch @ batch=32
    # Model weights from best epoch of v2 run are preserved
```

**Alternative (full resume, epoch counter preserved, optimizer reset):**

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python -m ml.scripts.train \\
    --run-name multilabel-v2-edge-attr-ext-clean \\
    --epochs 60 \\
    --batch-size 32 \\
    --resume ml/checkpoints/multilabel_crossattn_v2_best.pt \\
    --no-resume-model-only \\
    --resume-reset-optimizer
    # Epoch counter and patience_counter restored from checkpoint
    # Optimizer/scheduler discarded — fresh AdamW for batch_size=32
```

**Do NOT use:**
```bash
# This combination will reproduce the stale-moment bug
--batch-size 32 --no-resume-model-only   # without --resume-reset-optimizer
```
