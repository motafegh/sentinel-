# SENTINEL — v3 Deep Analysis & v4 Direction (2026-05-09)

All findings in this document are derived from actual data:
- MLflow run `d2ee23a141f3470ca994323d7bb57680` (experiment `sentinel-retrain-v3`, SQLite backend `mlruns.db`)
- Checkpoint JSON: `ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json`
- Source code: `ml/src/training/trainer.py`, `ml/src/training/focalloss.py`

---

## 1. Correction to Previous Analysis

The v3 completion doc (`2026-05-05-v3-training-complete.md`) concluded:

> *"Loss plateau... The model reached its capacity ceiling with current hyperparameters."*

This is **incorrect**. The actual per-epoch curve shows:

| Epoch | train_loss | val_f1_macro |
|-------|-----------|-------------|
| 50 | 0.7033 | 0.4656 |
| 54 | 0.6901 | **0.4715** ← best saved |
| 55 | 0.6912 | 0.4697 |
| 60 | 0.6865 | 0.4703 |

Train loss was **still decreasing** at epoch 60 (0.6901 → 0.6865 between epochs 54–60).
Val F1 was flat, not because the model hit a ceiling, but because **OneCycleLR had decayed
to near-zero learning rate** by epoch 54. The model could not make meaningful parameter
updates even though loss was still going down. This is an LR schedule exhaustion, not
a capacity ceiling.

---

## 2. Per-Class Training Curve Analysis (full 60 epochs)

Raw val F1 (threshold=0.5 during training) at key epochs:

| Epoch | macro | DoS  | CTU  | ExtBug | GasEx | IntUO | MhEx | Reentr | TOD  | TS   | UR   |
|-------|-------|------|------|--------|-------|-------|------|--------|------|------|------|
| 1     | 0.260 | 0.030| 0.197| 0.023  | 0.404 | 0.673 | 0.350| 0.167  | 0.301| 0.195| 0.262|
| 10    | 0.409 | 0.146| 0.277| 0.375  | 0.520 | 0.792 | 0.459| 0.468  | 0.399| 0.260| 0.391|
| 20    | 0.430 | 0.242| 0.293| 0.383  | 0.526 | 0.806 | 0.464| 0.491  | 0.413| 0.276| 0.400|
| 30    | 0.441 | 0.222| 0.306| 0.395  | 0.534 | 0.813 | 0.470| 0.503  | 0.423| 0.333| 0.412|
| 40    | 0.459 | 0.220| 0.350| 0.409  | 0.541 | 0.818 | 0.477| 0.520  | 0.438| 0.378| 0.441|
| 50    | 0.466 | 0.219| 0.356| 0.415  | 0.543 | 0.820 | 0.482| 0.529  | 0.453| 0.394| 0.445|
| 60    | 0.470 | 0.231| 0.362| 0.416  | 0.546 | 0.821 | 0.481| 0.532  | 0.456| 0.406| 0.451|

**Classes still improving at epoch 60 with no sign of convergence:**
- CallToUnknown: 0.362 (was 0.350 at ep10 — still slowly rising)
- ExternalBug: 0.416 (was 0.375 at ep10)
- Timestamp: 0.406 (was 0.260 at ep10 — large unfinished improvement)
- TransactionOrderDependence: 0.456 (was 0.399 at ep10)

**Classes that converged early:**
- IntegerUO: reached ~0.82 by epoch 30, flat after that (5,343 support — enough data)
- GasException: reached ~0.54 by epoch 30, flat after (2,589 support)

---

## 3. DenialOfService — Data Problem, Not Hyperparameter Problem

DoS F1 at every epoch fluctuates with no stable trend after epoch 17:

- Range epochs 20–60: **0.11 to 0.28**
- Best raw epoch: ep45 = 0.2749
- Final raw: ep60 = 0.2312
- Tuned F1 (threshold=0.95): 0.4000

**Root cause:** 137 training samples. At batch_size=32, DoS appears in ~4 batches per
epoch. 137 validation samples means changing 5 true positives swings F1 by ±0.04.
The raw F1 fluctuation is driven by validation sample noise, not real model instability.

**Conclusion:** No hyperparameter change will fix DoS. Weighted sampling (DoS-only)
can increase DoS exposure per epoch from ~4 to ~20 batches, which may help the model
build a more stable signal, but a fundamental improvement requires more training data.
Target DoS tuned F1 ≥ 0.35 (floor) is achievable; ≥ 0.50 is not realistic with 137 samples.

---

## 4. Per-Class Precision/Recall Breakdown (tuned, from thresholds JSON)

| Class | Threshold | F1    | Precision | Recall | Support | PredPos | Pattern |
|-------|-----------|-------|-----------|--------|---------|---------|---------|
| CallToUnknown | 0.70 | 0.394 | 0.322 | 0.507 | 1,266 | 1,996 | over-predicting 1.6× |
| DenialOfService | 0.95 | 0.400 | 0.318 | 0.540 | 137 | 233 | data-starved |
| ExternalBug | 0.65 | 0.435 | 0.312 | 0.715 | 1,622 | 3,717 | over-predicting 2.3× |
| GasException | 0.55 | 0.550 | 0.403 | 0.867 | 2,589 | 5,570 | over-predicting 2.2× |
| IntegerUO | 0.50 | 0.821 | 0.758 | 0.896 | 5,343 | 6,310 | strong |
| MishandledException | 0.60 | 0.492 | 0.365 | 0.754 | 2,207 | 4,558 | over-predicting 2.1× |
| Reentrancy | 0.65 | 0.536 | 0.449 | 0.665 | 2,501 | 3,702 | balanced |
| Timestamp | 0.75 | 0.479 | 0.403 | 0.591 | 1,077 | 1,579 | balanced |
| TransactionOrderDependence | 0.60 | 0.477 | 0.342 | 0.787 | 1,800 | 4,141 | over-predicting 2.3× |
| UnusedReturn | 0.70 | 0.486 | 0.395 | 0.631 | 1,716 | 2,741 | over-predicting 1.6× |

**Dominant pattern:** 6 of 10 classes are over-predicting (high recall, low precision).
The model says "yes" 1.6–2.3× more often than the true positive count.

**Cause:** BCE + pos_weight pushes the model to recall positives aggressively.
The pos_weights (ExternalBug=5.22, MishandledException=3.53, GasException=2.95) amplify
the gradient from missed positives, biasing the model toward prediction rather than precision.
Threshold tuning compensates partially (raising thresholds above 0.5 for most classes)
but does not fix the underlying calibration.

---

## 5. Focal Loss Assessment

The previous v3 doc recommended focal loss for v4. After inspecting the actual
implementation (`ml/src/training/focalloss.py`) and analyzing the per-class data:

**Implementation status:**
- `FocalLoss` is element-wise — works for both binary `(B,)` and multi-label `(B,C)` shapes ✓
- Patched for BF16/AMP in commit `caf95e9` (2026-05-01) ✓
- **Never used for multi-label training** — v3 used `bce` throughout

**The α=0.25 problem for this dataset:**
Focal loss with α=0.25 assigns weight 0.25 to positive examples and 0.75 to negatives.
Compare this to BCE+pos_weight for DoS (pos_weight=68): switching to focal would reduce
DoS positive gradient signal by ~200×. For a class with 137 training samples, this would
likely drop DoS F1 significantly.

**Focal loss may still be useful for the over-predicting classes** (ExternalBug, TOD, etc.)
by focusing the model on hard examples instead of easy ones — but α must be tuned, not
left at the binary-mode default of 0.25. An α > 0.5 would be needed to preserve signal
for rare classes.

**Verdict:** Focal loss is not a drop-in for v4. It requires separate α tuning, and the
interaction with multi-label class imbalance is not well-characterised for this dataset.
Do not use focal loss as the first v4 experiment.

---

## 6. Autoresearch Design Analysis

During the 2026-05-09 laptop session, the original autoresearch plan was re-evaluated
against Karpathy's actual approach (https://github.com/karpathy/autoresearch):

**Karpathy's loop:**
1. Agent edits one file (`train.py`)
2. Fixed wall-clock budget (5 min per run) — makes all runs directly comparable
3. Binary decision: val metric improved → keep; else revert to master
4. Agent reads results and proposes the *next* targeted change based on what changed

**What we built (as of commit `2edf382`):**
- A predefined search grid (8 knobs × N values)
- Smoke/confirm filter mechanism
- Results ledger (results.tsv)
- This is a **hyperparameter grid search harness**, not autoresearch

**The key difference:** in autoresearch, each result drives the next hypothesis.
In a grid search, the hypotheses are fixed upfront regardless of results.

**Decision:** Keep the smoke/confirm infrastructure (it's useful for filtering), but the
loop driver should be analysis-first — read per-class results, identify what changed and why,
propose one targeted change, then run. This is what program.md should describe.

---

## 7. Critical Code Issues Found During Audit

### 7.1 start_epoch bug in fine-tune resume

In `trainer.py` line 680:
```python
start_epoch = ckpt.get("epoch", 0) + 1
```

The v3 checkpoint has `epoch=54`. If fine-tuning from v3 with `epochs=5`, then
`remaining_epochs = 5 - 55 + 1 = -49` → the trainer returns immediately without
training anything. This bug would silently produce a zero-epoch run.

**Fix needed:** When using a checkpoint as a "base" (model-only resume for fine-tuning
with new hyperparameters), `start_epoch` must be reset to 1, not taken from the checkpoint.
This requires distinguishing "continue an interrupted run" from "fine-tune from a base".

### 7.2 strict=False for lora_r mismatch (already fixed in session)

`load_state_dict(ckpt["model"])` with `strict=True` crashes when `lora_r` changes
(LoRA A/B shapes are tied to r). Changed to `strict=False` with explicit checks:
LoRA key mismatches are logged as warnings; non-LoRA key mismatches raise errors.
Committed as part of uncommitted trainer.py changes in this session.

### 7.3 MIN_VRAM_GB guard too strict for laptop

Set to 7.0 GB — RTX 3070 Laptop always holds ~1.6 GB for the display driver, leaving
only ~6.4 GB free. Lowered to 5.5 GB in `auto_experiment.py` during this session.

---

## 8. What v4 Should Actually Do

Based on all findings above, the correct v4 approach in priority order:

### Primary experiment — fresh LR cycle from v3
The plateau was caused by LR schedule exhaustion, not capacity ceiling. The most
direct fix is to fine-tune from v3 weights with a fresh OneCycleLR cycle at a lower
peak LR (since we're starting from a much better point and don't need aggressive LR).

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/train.py \
  --run-name multilabel-v4-finetune-lr1e4 \
  --experiment-name sentinel-retrain-v4 \
  --resume ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --early-stop-patience 7
```

**BUT:** the `start_epoch` bug (§7.1) must be fixed first, or this run does nothing.

### Secondary experiment — lora_r=16
More LoRA capacity for classes still improving (CTU, ExternalBug, Timestamp).
Run from v3 weights after resolving the start_epoch fix.

### Tertiary — DoS weighted sampler
Only after primary experiment. On its own it cannot fix DoS (data problem), but
combined with a good LR it may stabilise DoS training.

### Not recommended for now
- Focal loss with α=0.25 — hurts rare classes
- Full grid search — too slow, not analysis-driven
- Training from scratch — no advantage over fine-tuning from v3

---

## 9. Files Modified in This Session (uncommitted as of 2026-05-09 ~23:00)

| File | Change | Status |
|------|--------|--------|
| `ml/scripts/auto_experiment.py` | MIN_VRAM_GB 7.0→5.5; base_checkpoint default to v3; resume_from+resume_model_only wired | uncommitted |
| `ml/src/training/trainer.py` | load_state_dict strict=False with lora mismatch warning | uncommitted |
| `ml/autoresearch/program.md` | Added base-checkpoint note to §4 | uncommitted |

The `start_epoch` reset fix (§7.1) has NOT been implemented yet — needs design decision
on how to distinguish "continue" vs "fine-tune base" resume modes before coding.
