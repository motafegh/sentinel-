# 2026-05-04 — Resume Fixes, Full-Resume CLI, and Autoresearch Strategy

Date: 2026-05-04  
Session: overnight retrain resume + autoresearch exploration

---

## Summary

Two bugs were found and fixed that prevented the 40-epoch checkpoint from being
correctly resumed. The retrain was then extended from 40 → 60 epochs using a
proper full-resume (optimizer + scheduler state restored). An autoresearch
strategy document was also drafted describing how the Karpathy autoresearch
pattern can be applied to Sentinel's ML module.

---

## Changes Made

### Fix 1 — `ml/scripts/train.py`: expose `--no-resume-model-only` CLI flag

**Commit:** b56eec8cec26bca94ac0ed1ddc36441e36b7f834

**Problem:**  
`TrainConfig` has a `resume_model_only: bool = True` field, and `trainer.py`
already had full-resume logic — restoring optimizer and scheduler state — gated
behind `if not config.resume_model_only`. But `train.py`'s CLI never exposed
this field. There was no way to set it to `False` from the command line without
editing source code.

As a result, every resume was `resume_model_only=True` by default, meaning:
- Adam momentum buffers (m/v vectors from 37 epochs of training) were thrown away
- `OneCycleLR` scheduler started a brand-new cosine cycle from `max_lr=3e-4`
- The LR spiked back to its original maximum on already-converged weights
- This destabilises training for the first 3–5 resumed epochs

**Fix:**  
Added `--no-resume-model-only` argument to `parse_args()` and wired it to
`resume_model_only=args.resume_model_only` in `TrainConfig()`.

```python
p.add_argument(
    "--no-resume-model-only",
    dest="resume_model_only",
    action="store_false",
    default=True,
    help=(
        "When set, restores optimizer AND scheduler state from checkpoint "
        "(full resume). Default is model-weights-only resume."
    ),
)
```

**Backward compatibility:** Default is still `True` — all existing training
commands that don't pass this flag continue to work identically.

---

### Fix 2 — `ml/src/training/trainer.py`: Fix #9 — AttributeError on resume

**Commit:** 5a6715e7736fe455900991b56130d5dde61810df

**Problem:**  
Line 520 of the resume cross-check read:

```python
if ckpt_arch is not None and ckpt_arch != config.architecture:
```

`TrainConfig` has no `architecture` field — it was never declared as a dataclass
field. The architecture is always `"cross_attention_lora"` and was only used as a
hardcoded string literal in two places (checkpoint save dict and MLflow params).

This raised `AttributeError: 'TrainConfig' object has no attribute 'architecture'`
on the first resume attempt ever made in this project.

**Why it never fired before:**  
The entire block is inside `if config.resume_from:`. During the original 40-epoch
training run, `config.resume_from` was `None` — so Python never entered this block.
The buggy line sat dormant until the first resume attempt tonight.

This is a classic dead-code-path bug: syntactically valid, passes all imports,
runs fine in the common case, but explodes the first time a specific branch is hit.

**Fix:**  
Extracted `ARCHITECTURE = "cross_attention_lora"` as a module-level constant
(single source of truth). Replaced `config.architecture` with `ARCHITECTURE` in
the resume check. Also replaced the two scattered string literals in the checkpoint
save dict and MLflow params with the same constant.

```python
# New module-level constant
ARCHITECTURE = "cross_attention_lora"

# Resume check — was: config.architecture
if ckpt_arch is not None and ckpt_arch != ARCHITECTURE:
    raise ValueError(...)

# Checkpoint save dict — was: "cross_attention_lora" literal
"architecture": ARCHITECTURE,

# MLflow params — was: "cross_attention_lora" literal
"architecture": ARCHITECTURE,
```

**Docstring updated:** Fix #9 added to the `trainer.py` module docstring audit log.

---

## Retrain Extended: 40 → 60 Epochs

After both fixes the retrain was resumed correctly:

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/train.py \
    --run-name multilabel-v2-edge-attr-60ep \
    --experiment-name sentinel-retrain-v2 \
    --epochs 60 \
    --batch-size 16 \
    --lr 3e-4 \
    --resume ml/checkpoints/multilabel_crossattn_v2_best.pt \
    --checkpoint-name multilabel_crossattn_v2_best.pt \
    --no-resume-model-only
```

**Confirmed from logs:**
- `Resumed from epoch 37 | best_f1=0.4629`
- `Optimizer state restored.`
- `Scheduler state restored.`
- `Epoch 38/60` — correct starting epoch
- Training speed: ~3.78 batch/s (normal)

**What full-resume means in practice:**  
The `OneCycleLR` scheduler state was saved mid-curve at epoch 37. Restoring it
means the LR continues from where the cosine curve actually was at epoch 37, not
from the peak `max_lr=3e-4`. The optimizer's Adam m/v accumulators from 37 epochs
of gradients are also preserved, so the optimizer "knows" the gradient history
and doesn't take large exploratory steps on already-fine-tuned parameters.

**Remaining:** 23 epochs (38–60). Early stopping patience=7 still active.
Estimated runtime: ~5 hours overnight.

**Success gate:** val F1-macro (post threshold tuning) > 0.4884
(0.4884 = best tuned F1 from the interrupted 40-epoch run at epoch 37).

---

## Post-Training Protocol (Morning)

When training completes:

**1. Check result:**  
Look for `✅ Training complete. Best val F1-macro: X.XXXX` in logs.
If any epoch above 37 improved: checkpoint was updated in place.

**2. Tune thresholds:**
```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python -m ml.scripts.tune_threshold \
    --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt \
    --start 0.05 --end 0.95 --step 0.05
```

**3. Promote if tuned F1 > 0.4884:**
```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt \
    --stage Staging \
    --val-f1-macro <TUNED_F1> \
    --note "60ep full-resume: tuned F1=X.XXXX"
```

**4. If no improvement over epoch 37 (best_f1 stayed 0.4629):**  
Early stopping likely fired. Investigate per-class F1 distribution in MLflow.
Consider next steps from autoresearch plan below.

---

## Autoresearch Strategy Note

### What It Is

Karpathy's autoresearch pattern (github.com/karpathy/autoresearch) is a methodology
for iterative ML improvement: fix data and metric; give an agent a single training
entrypoint it can edit; run a loop of edit → train (fixed budget) → parse scalar
score → keep or revert.

### How It Maps to Sentinel

Sentinel's ML module already matches this pattern closely:

| Autoresearch concept | Sentinel equivalent |
|----------------------|---------------------|
| `prepare.py` (fixed) | `DualPathDataset`, split files, `multilabel_index.csv` |
| `train.py` (agent edits) | `ml/scripts/auto_experiment.py` (to be created) |
| scalar metric (`val_bpb`) | `val_f1_macro` printed by trainer |
| `program.md` | `ml/autoresearch/program.md` (to be created) |
| Fixed time budget | Epoch budget (e.g. 15 epochs) or wall-clock `max_minutes` |

The existing `run_overnight_experiments.py` is already a hand-written autoresearch
loop over a static list of `TrainConfig` variants.

### What Needs Building (When Ready)

**1. `ml/scripts/auto_experiment.py`**  
Thin CLI wrapper around `trainer.train()` that:
- Accepts hyperparams via `--lr`, `--batch-size`, `--loss-fn`, `--gnn-hidden-dim`,
  `--lora-r`, etc.
- Builds `TrainConfig` from args
- Calls `train(config)`
- Prints a single parseable score line at the end:
  ```
  SENTINEL_SCORE: val_f1_macro=0.6842
  ```

**2. `ml/autoresearch/program.md`**  
Instructions for the agent:
- Goal: maximise `val_f1_macro`; don't reduce recall on Reentrancy/IntegerUO
- Fixed (don't touch): data directories, split files, `CLASS_NAMES`
- Allowed knobs: `lr`, `batch_size`, `loss_fn`, GNN dims/heads/dropout, LoRA r/alpha
- Constraints: VRAM budget, ≤ 60 min per experiment on RTX 3070
- Stability rules: keep both GNN and text paths; keep `num_classes=10`

**3. Architecture search (Phase 2)**  
Once config-only search is working, extend to allow agent to edit `TrainConfig`
architecture fields and `SentinelModel.__init__()` branches, keeping the public API
`SentinelModel(graphs, input_ids, attention_mask) -> logits` unchanged.

**4. Objective shaping**  
Instead of raw F1-macro, define a scalar that weights critical vulnerability classes
(Reentrancy, IntegerUO) more heavily. Compute from per-class F1s already logged
by `evaluate()`.

**5. Threshold-in-loop**  
Incorporate `tune_threshold.py` into the experiment score: train → threshold tune
→ report tuned F1. This makes the scalar representative of real deployment performance.

### Current Status
Not yet started. Unblocked after the 60-epoch retrain completes and produces a
new best checkpoint. First step: implement `auto_experiment.py`.
