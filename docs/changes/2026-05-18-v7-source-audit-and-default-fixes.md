# v7 Full Source Audit — Config/Default Alignment Fixes
**Date:** 2026-05-18  
**Status:** All fixes applied. v7.0 training command correct with no override flags needed.

---

## Summary

Full audit of all training and model source files for alignment, correctness, and
optimization against the v7 spec. Found 12 config/default misalignments that would
have silently broken or degraded v7.0 training — none were caught by tests because
they only affect runtime behavior (wrong path → empty dataset, wrong default → BCE
instead of ASL, etc.).

---

## Fixes Applied

### trainer.py

| Item | Before | After | Impact |
|------|--------|-------|--------|
| `ARCHITECTURE` | `"three_eye_v5"` | `"three_eye_v7"` | Saved into every checkpoint; checked on resume — stale value triggers architecture mismatch warning |
| `MODEL_VERSION` | `"v6.0"` | `"v7.0"` | Same — version gate in resume path |
| `tokens_dir` default | `"ml/data/tokens"` | `"ml/data/tokens_windowed"` | **CRITICAL** — wrong directory has no .pt files; DualPathDataset would find 0 paired samples |
| `label_csv` default | `multilabel_index_deduped.csv` | `multilabel_index_cleaned.csv` | **HIGH** — deduped CSV has 17,722 noisy labels that label_cleaner removed |
| `batch_size` default | `16` | `8` | **HIGH** — 16 saturates 8 GB VRAM with MAX_WINDOWS=4; OOM after epoch 1; comment already said "was 16, reduced" but value was never updated |
| `gradient_accumulation_steps` default | `1` | `8` | **HIGH** — effective batch = 8 (not 64); gradient signal too noisy for minority classes |
| Resume `weights_only` | `True` | `False` | **HIGH** — LoRA peft objects not in torch safe globals; resume would crash loading any v5+ checkpoint |
| Module docstring | "Cross-Attention + LoRA Upgrade" | "v7 — Three-Eye GNN+CodeBERT+LoRA" | Cosmetic |
| Resume warn message | References "v5.2 / JK / REVERSE_CONTAINS" | References "v7 / conv3c / 11-dim schema" | Cosmetic |

### train.py (CLI defaults that override TrainConfig at runtime)

| Flag | Before | After | Impact |
|------|--------|-------|--------|
| `--tokens-dir` | `"ml/data/tokens"` | `"ml/data/tokens_windowed"` | **CRITICAL** — same path bug; CLI always overrides TrainConfig |
| `--loss-fn` | `"bce"` | `"asl"` | **HIGH** — BCE ignores class imbalance; ASL with γ⁻=2 is the v7 loss |
| `--batch-size` | `16` | `8` | **HIGH** — OOM |
| `--gradient-accumulation-steps` | `1` | `8` | **HIGH** — effective batch 8 vs 64 |
| `--asl-gamma-neg` | `4.0` | `2.0` | **MEDIUM** — 4.0 caused all-zeros collapse (BUG-C4); 2.0 is the validated v7 value |
| `--asl-clip` | `0.05` | `0.01` | **MEDIUM** — 0.05 caused oscillation at p≈0.03–0.06 (BUG-M2) |
| `--label-smoothing` | `0.05` | `0.0` | **LOW** — uniform smoothing replaced by per-class smoothing in TrainConfig.class_label_smoothing |
| `--weighted-sampler` | `"none"` | `"positive"` | **MEDIUM** — without 3× vuln-row weight, 60% zero-label rows dominate training |
| Module docstring | v5.2 examples throughout | v7.0 examples | Cosmetic |

### sentinel_model.py

| Item | Before | After |
|------|--------|-------|
| Module docstring | "v5 architecture" / "V5 CHANGES FROM V4" | "v7 architecture" / v7 description |

### fusion_layer.py

| Item | Before | After |
|------|--------|-------|
| Module docstring | "Node embeddings [N,64]" | "[N,256]" |
| `forward()` inline comment | `# [N, 64] all nodes` | `# [N, gnn_hidden_dim] (hidden_dim=256)` |
| Architecture description | "Project GNN nodes [N,64] → [N,256]" | "[N,256] → [N,256] (identity-sized; hidden_dim=256)" |

---

## Root Cause

TrainConfig dataclass defaults (trainer.py) and train.py CLI defaults were maintained
separately and drifted. Every time a v7 decision was made (ASL, batch=8, tokens_windowed,
positive sampler) it was recorded in the v7 spec/MEMORY but only some were applied to
both files. Since train.py always passes args explicitly to TrainConfig, the TrainConfig
defaults only matter for direct `TrainConfig()` instantiation — but in practice the
CLI defaults are what a `python train.py ...` invocation uses.

The critical bug was `--tokens-dir default="ml/data/tokens"` — if training had been
launched with only the STATUS.md command (`--run-name v7.0 --experiment-name sentinel-v7
--epochs 100 --gradient-accumulation-steps 8 --compile --num-workers 4`), the dataset
would have silently loaded 0 paired samples from the non-existent `tokens/` directory
(all windowed tokens are in `tokens_windowed/`) and training would have crashed on the
first batch with an index error.

---

## Corrected Training Command

After these fixes, the STATUS.md command no longer needs `--gradient-accumulation-steps 8`
(now the default) but still needs `--compile` and `--num-workers 4` since those are
opt-in / non-default:

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v7.0 --experiment-name sentinel-v7 \
    --epochs 100 --compile --num-workers 4
```

Expected: batch=8, grad_accum=8, effective_batch=64, loss=ASL(γ⁻=2.0, γ⁺=1.0, clip=0.01),
sampler=positive, tokens_dir=tokens_windowed, label_csv=multilabel_index_cleaned.csv.

---

## Files Changed

| File | Lines changed |
|------|--------------|
| `ml/src/training/trainer.py` | ARCHITECTURE, MODEL_VERSION, tokens_dir, label_csv, batch_size, grad_accum, resume weights_only, docstrings |
| `ml/scripts/train.py` | 8 CLI defaults + module docstring |
| `ml/src/models/sentinel_model.py` | Module docstring |
| `ml/src/models/fusion_layer.py` | 3 [N,64]→[N,256] docstring fixes |
