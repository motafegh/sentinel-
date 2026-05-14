# SENTINEL ML Training Guide — Practitioner's Reference

> **Audience**: Engineers who need to run training, interpret logs, diagnose failures, and decide
> whether to abort or continue. This guide assumes you have already read the architecture docs.
> All commands are run from the project root: `/home/motafeq/projects/sentinel`.

---

## TL;DR — Full Workflow in 5 Steps

- **Verify** environment and data are ready using the pre-training checklist before touching the GPU.
- **Smoke run first** (2 epochs, 10% data): if GNN share falls below 15% or JK phases are dead, abort and diagnose before wasting 10 hours.
- **Full 60-epoch run** only after all smoke gates pass; background it with `nohup` and monitor with `tail -f nohup.out`.
- **Tune thresholds** on the validation set after training ends — raw probabilities alone are not enough for inference.
- **Behavioral tests are the real gate**: F1 on validation is necessary but not sufficient; ≥ 60% behavioral pass rate is the success criterion.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Pre-Training Checklist](#2-pre-training-checklist)
3. [Step 1 — Smoke Run](#3-step-1--smoke-run)
4. [Step 2 — Full 60-Epoch Run](#4-step-2--full-60-epoch-run)
5. [Reading Training Logs](#5-reading-training-logs)
6. [Abort Criteria](#6-abort-criteria)
7. [After Training — Threshold Tuning](#7-after-training--threshold-tuning)
8. [Behavioral Testing](#8-behavioral-testing)
9. [Resuming a Checkpoint](#9-resuming-a-checkpoint)
10. [Running the Ablation (JK Disabled)](#10-running-the-ablation-jk-disabled)
11. [MLflow Monitoring](#11-mlflow-monitoring)
12. [Common Failure Modes and Fixes](#12-common-failure-modes-and-fixes)

---

## 1. Environment Setup

### Shell Environment

Every training session starts with these two steps. Skipping either will cause subtle,
hard-to-diagnose failures.

```bash
source ml/.venv/bin/activate
export TRANSFORMERS_OFFLINE=1
```

`TRANSFORMERS_OFFLINE=1` must be set at the shell level — not inside a script — before
importing any HuggingFace library. Without it, CodeBERT will attempt a network round-trip
on every import and either hang or crash if the network is unavailable.

### Hardware Context

- GPU: RTX 3070, 8 GB VRAM
- Expected training speed: roughly 8–12 hours for a 60-epoch full run
- If VRAM OOM occurs, see [Common Failure Modes](#12-common-failure-modes-and-fixes) for the
  batch-size / gradient-accumulation adjustment

---

## 2. Pre-Training Checklist

Run every item below before starting any training run — smoke or full. A misconfigured
environment caught here saves hours of wasted compute.

| # | What to verify | Command | Expected result |
|---|----------------|---------|-----------------|
| 1 | Cache file exists and is non-trivially sized | `ls -lh ml/data/cached_dataset_deduped.pkl` | Several GB; file present |
| 2 | Graph count on disk | `ls ml/data/graphs/ \| wc -l` | 44,470 |
| 3 | Token count on disk | `ls ml/data/tokens/ \| wc -l` | 44,470 |
| 4 | CSV row count (with header) | `wc -l ml/data/processed/multilabel_index_deduped.csv` | 44,471 |
| 5 | Deduped splits present | `ls ml/data/splits/deduped/` | `train.npy  val.npy  test.npy` |
| 6 | Model version constant | `grep MODEL_VERSION ml/src/training/trainer.py` | `"v5.2"` |
| 7 | GNN encoder unit tests | `PYTHONPATH=. python -m pytest ml/tests/test_gnn_encoder.py -v` | **11/11 pass** — non-negotiable |

### Why 11/11 Tests Are Non-Negotiable

The `test_jk_gradient_flow` test specifically verifies that gradients flow back through all
three JK phases to the GNN encoder parameters. This is the exact failure mode that caused
the GNN to collapse in `v5.1-fix28` (epoch 8, GNN share dropped to near-zero). If this test
fails, the full run will exhibit the same collapse — guaranteed.

---

## 3. Step 1 — Smoke Run

### Purpose

The smoke run uses 10% of training data for 2 epochs. Its job is to catch shape errors, NaN
explosions, and unhealthy gradient distributions before committing to a 10-hour run. Running
the full 60 epochs with a misconfigured model wastes GPU time and produces a useless checkpoint.

### Command

```bash
source ml/.venv/bin/activate
export TRANSFORMERS_OFFLINE=1
PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-smoke \
    --experiment-name sentinel-v5.2 \
    --epochs 2 \
    --smoke-subsample-fraction 0.1 \
    --gradient-accumulation-steps 4
```

### What to Watch During the Smoke Run

Around step 100, the console will emit a line like:

```
[step 100] gnn_norm=X.XXX tf_norm=X.XXX fused_norm=X.XXX
GNN share: XX.X%
```

This is the most important line in the smoke run output. GNN share is computed as:

```
gnn_share = gnn_norm / (gnn_norm + tf_norm + fused_norm)
```

A GNN share in the healthy range (15–65%) tells you the GNN encoder is actually learning,
not just riding along on the transformer's gradient.

### Smoke Run Gates — ALL Must Pass Before Proceeding

| Gate | Metric | Threshold | Where to check |
|------|--------|-----------|----------------|
| 1 | GNN share at step 100 | ≥ 15% | Console output |
| 2 | JK phase weights after epoch 1 | All three > 0.05 | MLflow: `jk_phase1_weight`, `jk_phase2_weight`, `jk_phase3_weight` |
| 3 | NaN batch count in epoch 2 | 0 | MLflow: `nan_batch_count` |
| 4 | Loss trend | Epoch 2 loss < epoch 1 loss | Console / MLflow: `train/loss` |

**If any gate fails, stop.** Do not proceed to the full run. Check the
[Failure Modes](#12-common-failure-modes-and-fixes) section for diagnosis.

A note on gate 3: the first 50 steps can occasionally produce a NaN during the warmup
phase when learning rates are ramping up. This is marginal but not automatically fatal.
What matters is that `nan_batch_count` reaches 0 by epoch 2.

---

## 4. Step 2 — Full 60-Epoch Run

Only run after all four smoke gates pass.

### Interactive (for initial testing)

```bash
PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4
```

### Background (recommended for overnight runs)

```bash
nohup TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4 > nohup.out 2>&1 &
echo $! > train.pid
```

Monitor progress:

```bash
tail -f nohup.out
```

Check if still running:

```bash
cat train.pid | xargs ps -p
```

### Expected Duration

8–12 hours on RTX 3070 8 GB. Actual time depends on the fraction of data that hits the
RAM cache vs. disk. If it is running slower than ~15 min/epoch, see the RAM cache failure
mode in section 12.

### Early Stopping

Training stops automatically after 10 consecutive epochs with no improvement in
`val/f1_macro`. The best checkpoint is saved whenever a new best is reached.

---

## 5. Reading Training Logs

### 5.1 GNN Share

```
[step N] gnn_norm=X.XXX tf_norm=X.XXX fused_norm=X.XXX
GNN share: XX.X%
```

| Range | Interpretation |
|-------|---------------|
| 15–65% | Healthy — GNN is contributing |
| 10–15% | Watch closely |
| < 10% for 3 consecutive intervals | Streak alert — potential collapse |
| < 10% for > 5 epochs | Abort criterion — see section 6 |

The streak alert logic is built into the trainer. When it fires, a warning line appears in
the console. This is the early warning before full collapse. If you see it, watch the next
few log lines carefully. If GNN share does not recover, stop training.

**Historical context**: In `v5.1-fix28`, the GNN collapsed at epoch 8 — GNN share dropped
to near-zero and stayed there for the rest of the run. The resulting checkpoint was
entirely driven by the transformer, which is why behavioral tests showed no improvement
over v4. The JK attention connections and separate LR groups in v5.2 were specifically
designed to prevent recurrence. If collapse happens again despite those fixes, there is an
undiscovered root cause that must be investigated before running again.

### 5.2 JK Attention Weights

Logged to MLflow once per epoch as `jk_phase1_weight`, `jk_phase2_weight`,
`jk_phase3_weight`.

| State | Interpretation |
|-------|---------------|
| All three > 5% after epoch 1 | Healthy |
| Any single phase > 80% | Phase dominance — check per-phase LayerNorm |
| Any phase at or near 0% | That phase has been effectively disabled |

The model learns these weights during training — the distribution will shift. Any
distribution where all phases stay above 5% is acceptable. The model knows what to weight.
What you are guarding against is one phase being completely zeroed out, which means the
information from that phase of the GNN is not reaching the classifier.

### 5.3 NaN Batch Count

Logged to MLflow per epoch as `nan_batch_count`.

| Value | Action |
|-------|--------|
| 0 | Perfect |
| 1–2 in epochs 1–5 | Marginal, monitor |
| Any NaN after epoch 10 | Investigate immediately |
| > 5 in any single epoch after epoch 5 | Abort criterion |

NaN loss after the warmup phase usually means one of three things: gradient explosion
(check gradient norm trends), a zero-node graph slipping through the dataset, or a
learning rate that is too high. See section 12 for specific diagnosis steps.

### 5.4 Val F1-Macro — The Primary Checkpoint Metric

Logged per epoch as `val/f1_macro`. The checkpoint is saved whenever a new best is
reached.

| Threshold | Meaning |
|-----------|---------|
| > 0.5828 | Exceeds v5.0 best |
| > 0.5422 | Exceeds v4 fallback (minimum acceptable floor) |

Per-class F1 floors from v4 (v5.2 must exceed `floor = v4_F1 − 0.05`):

| Class | v4 F1 | Floor |
|-------|-------|-------|
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

DenialOfService has only 377 samples (train ≈ 257). Weak F1 here is expected — do not
treat it as a signal of general model failure.

IntegerUO has the most samples (15,529) and should consistently produce the strongest
per-class F1.

### 5.5 Loss Curves

Logged per epoch as `train/loss` and `val/loss`.

| Pattern | Interpretation |
|---------|---------------|
| Both decreasing | Healthy learning |
| val rising, train still falling | Overfitting — early stopping will engage at 10 epochs |
| Both flat | Learning rate exhausted or too low |
| val loss > 2× initial val loss after epoch 10 | Abort criterion |

---

## 6. Abort Criteria

Stop training immediately if any of the following occur. Do not wait for the epoch to
finish.

1. **GNN collapse**: Streak alert fires AND GNN share stays below 10% for more than
   5 consecutive epochs.
2. **NaN explosion**: `nan_batch_count` > 5 in any single epoch after epoch 5.
3. **Val loss explosion**: `val/loss` exceeds 2× its initial value after epoch 10.
4. **CUDA OOM**: `RuntimeError: CUDA out of memory`. Reduce batch size and restart —
   see section 12.

To stop a backgrounded run:

```bash
cat train.pid | xargs kill
```

---

## 7. After Training — Threshold Tuning

The model's output is raw probabilities (after sigmoid). These are not directly comparable
across classes — IntegerUO and DenialOfService have very different base rates. Per-class
decision thresholds must be tuned on the validation set before inference is meaningful.

```bash
PYTHONPATH=. python ml/scripts/tune_thresholds.py \
    --checkpoint ml/checkpoints/v5.2-jk_best.pt \
    --label-csv ml/data/processed/multilabel_index_deduped.csv \
    --splits-dir ml/data/splits/deduped
```

This produces `ml/checkpoints/v5.2-jk_best_thresholds.json`. This file is required for
both inference (`ml/src/inference/api.py`) and behavioral testing. Do not run behavioral
tests without it.

---

## 8. Behavioral Testing

### Why F1 Alone Is Not Enough

v5.0 achieved a tuned `val/f1_macro` of 0.5828 — better than v4. Yet behavioral tests
showed only 15% pass rate (same as v4). The model had learned statistical correlations in
the validation set without actually learning to detect vulnerability patterns. Validation
F1 is a necessary gate, not a sufficient one.

The behavioral tests use real contracts with known ground truth in
`ml/scripts/test_contracts/`, including:

- Known-reentrancy contracts — must predict Reentrancy = 1
- CEI-pattern safe contracts — must predict Reentrancy = 0
- Integer overflow examples — must predict IntegerUO = 1

### Running Behavioral Tests

```bash
PYTHONPATH=. python ml/scripts/manual_test.py \
    --checkpoint ml/checkpoints/v5.2-jk_best.pt \
    --thresholds ml/checkpoints/v5.2-jk_best_thresholds.json
```

### Success Gate

**≥ 60% behavioral test pass rate.**

v5.0 was 15%. v4 was 15%. This gate has never been passed. A result above 60% would be
a meaningful first success.

If behavioral tests fail despite a good validation F1, the model is overfitting to
distributional artifacts in the dataset, not learning code semantics. The most likely root
cause is training data quality (duplicate contracts that inflate certain patterns without
structural diversity).

---

## 9. Resuming a Checkpoint

If a run is interrupted or you want to continue from a saved checkpoint:

```bash
PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk-resumed \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --resume ml/checkpoints/v5.2-jk_best.pt
```

### Resume Caveats

**Architecture version mismatch**: If the checkpoint was saved with a different
`model_version` than `"v5.2"`, a WARNING is logged. JK attention weights and per-phase
LayerNorm parameters will be randomly initialized for those layers. This is expected when
resuming from a pre-v5.2 checkpoint — the new structural components simply start fresh.

**Optimizer group mismatch**: v5.2 uses 3 optimizer param groups (GNN, LoRA, rest). If
the checkpoint has 1 group (from a pre-v5.2 run), the optimizer state is NOT restored
and a WARNING is logged. Training starts with fresh AdamW momentum buffers. This is
usually the correct behavior.

**Resuming from a collapsed run**: If you are resuming from a checkpoint where GNN
collapsed (e.g., `v5.1-fix28`), use `--resume-reset-optimizer` to force fresh momentum
buffers even if group counts match. Stale momentum buffers from a collapsed run can bias
the first few optimizer steps in the wrong direction.

```bash
PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk-resumed \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --resume ml/checkpoints/v5.2-jk_best.pt \
    --resume-reset-optimizer
```

---

## 10. Running the Ablation (JK Disabled)

To establish a controlled comparison between the JK-enabled and JK-disabled variants,
run the no-JK ablation under the same experiment name so results appear side-by-side in
MLflow.

```bash
PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-no-jk \
    --no-jk \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4
```

The `--no-jk` flag disables the JK attention aggregation — the GNN returns only the
final phase output instead of the attention-weighted combination of all three phase
outputs. If `v5.2-no-jk` collapses in the same way as `v5.1-fix28` but `v5.2-jk` does
not, that confirms the JK connections are doing the work they were designed to do.

---

## 11. MLflow Monitoring

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Open `http://localhost:5000` in a browser.

### Recommended Column Set

Add these columns to the run comparison table:

| Column | Purpose |
|--------|---------|
| `val/f1_macro` | Primary success metric |
| `nan_batch_count` | Training stability |
| `jk_phase1_weight` | JK health (phase 1) |
| `jk_phase2_weight` | JK health (phase 2) |
| `jk_phase3_weight` | JK health (phase 3) |
| `gnn_grad_share` | GNN contribution (if logged) |

### Experiment Naming Convention

- Smoke runs: `sentinel-v5.2`, run name `v5.2-smoke`
- Full runs: `sentinel-v5.2`, run name `v5.2-jk`
- Ablation: `sentinel-v5.2`, run name `v5.2-no-jk`
- Resumed runs: `sentinel-v5.2`, run name `v5.2-jk-resumed`

Keeping all variants under `sentinel-v5.2` allows direct comparison in the MLflow UI.
Previous experiments (`sentinel-v5`, `sentinel-v5.1`) are corrupt or invalid baselines
and should not be used for comparison.

---

## 12. Common Failure Modes and Fixes

### VRAM Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Fix**: Reduce batch size and compensate with more gradient accumulation steps to preserve
the effective batch size.

```bash
PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --batch-size 8 \
    --gradient-accumulation-steps 8
```

This keeps effective batch size at 64 (8 × 8 = 64, same as 16 × 4).

---

### GNN Collapse Before Epoch 10

**Symptom**: GNN share drops below 10% and stays there. Streak alert fires.

**First check**: Is `--gnn-lr-multiplier` at the default of 2.5? If it was accidentally
lowered, the GNN learning rate is not sufficient to compete with the transformer's gradient
magnitude.

**Fix**: Try increasing to 3.0.

```bash
PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk-lr3 \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4 \
    --gnn-lr-multiplier 3.0
```

If collapse persists even at higher LR, re-run the GNN encoder tests. A newly failing
`test_jk_gradient_flow` means the gradient path was broken by a recent code change.

---

### All-Zero JK Weight for One Phase

**Symptom**: One of `jk_phase1_weight`, `jk_phase2_weight`, `jk_phase3_weight` stays at
or near 0.0 after epoch 1.

**Likely cause**: The per-phase LayerNorm for that phase has died (extremely unlikely
but possible if the phase receives zero-variance inputs).

**Diagnosis**: Check whether the `phase_norm` parameters for that phase have a nonzero
gradient. If their gradient is zero, no data is flowing through that phase.

**Fix**: Investigate the graph data for that phase's edge types. CONTROL_FLOW (phase 2)
and REVERSE_CONTAINS (phase 3) are sparse edge types — if a batch has no edges of those
types, the phase output is undefined.

---

### NaN Loss from the First Step

**Symptom**: Loss is NaN at step 1 or 2.

**Likely cause**: A graph with zero nodes slipped through the dataset. The GNN pooling
operation on an empty graph returns NaN.

**Diagnosis**: Run a quick scan for empty graphs.

```bash
PYTHONPATH=. python -c "
import torch, glob
bad = []
for f in glob.glob('ml/data/graphs/*.pt'):
    g = torch.load(f, weights_only=False)
    if g.x.shape[0] == 0:
        bad.append(f)
print(bad)
"
```

Any files printed are the culprits. Remove them and rebuild the cache.

---

### Slow Training (> 15 min/epoch)

**Symptom**: Each epoch takes much longer than expected. GPU utilization is high but
training is crawling.

**Likely cause**: The RAM cache is not being used. Every sample is being loaded from disk
and re-processed.

**Fix**: Verify the cache path in `TrainConfig` points to the existing
`ml/data/cached_dataset_deduped.pkl` file, and that the file is actually non-empty.

```bash
ls -lh ml/data/cached_dataset_deduped.pkl
```

If the file is missing or the path is wrong, rebuild it:

```bash
PYTHONPATH=. python ml/scripts/create_cache.py
```

---

### Checkpoint Load Fails with Key Mismatch

**Symptom**: `RuntimeError: Error(s) in loading state_dict` with missing or unexpected
keys.

**Likely cause**: Checkpoint was saved with JK enabled but you are loading with `--no-jk`
(or vice versa). The `jk_attn` and `phase_norm` parameters are present in one state dict
but not the other.

**Fix**: Use the flag that matches how the checkpoint was saved. If you are not sure,
check the `model_version` field in the checkpoint:

```bash
PYTHONPATH=. python -c "
import torch
ckpt = torch.load('ml/checkpoints/v5.2-jk_best.pt', weights_only=False)
print(ckpt.get('model_version'), ckpt.get('config', {}).get('use_jk'))
"
```

---

*Last updated: 2026-05-14. Applies to model version `v5.2` and training run `v5.2-jk`.*
