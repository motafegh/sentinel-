# 2026-05-10 — v4 Experiment 1 Complete

## Summary

v4 experiment 1 (`multilabel-v4-finetune-lr1e4`) completed 30 epochs. Gate cleared.

## Results

| Metric | v3 | v4 exp1 | Delta |
|--------|-----|---------|-------|
| Raw F1-macro (best epoch) | 0.4715 (ep 52-53) | 0.5064 (ep 26) | +0.0349 |
| **Tuned F1-macro** | **0.5069** | **0.5422** | **+0.0353** |
| Hamming loss | — | 0.2043 | — |
| Exact-match accuracy | — | 0.3070 | — |

Gate condition: tuned F1-macro > 0.5069 ✅ PASSED

## Per-Class Results

| Class | v3 Threshold | v3 F1 | v4 Threshold | v4 F1 | Delta | Floor | Pass |
|-------|-------------|-------|-------------|-------|-------|-------|------|
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

## Training Curve (best epochs only)

| Epoch | Raw F1-macro |
|-------|-------------|
| 1 | 0.4754 |
| 7 | 0.4829 |
| 12 | 0.4848 |
| 14 | 0.4927 |
| 16 | 0.4931 |
| 18 | 0.5000 |
| 20 | 0.5022 |
| 22 | 0.5053 |
| 26 | **0.5064** ← best |
| 30 | patience=4/7 |

Model was still improving at epoch 26/30. Patience counter=4/7 at end — early stopping did not trigger. This suggests more epochs could yield further gains.

## Configuration

```
run-name:          multilabel-v4-finetune-lr1e4
experiment-name:   sentinel-retrain-v4
resume:            ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
resume-mode:       model-only (fine-tune — Fix #25 resets epoch/patience/best_f1)
epochs:            30
batch-size:        16
lr:                1e-4
weight-decay:      1e-2
early-stop-patience: 7
loss-fn:           bce
lora_r:            8 (same as v3)
lora_alpha:        16
```

## Files

- Checkpoint: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`
- Thresholds: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best_thresholds.json`
- State sidecar: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.state.json`
- Training log: `ml/autoresearch/runs/v4-finetune-lr1e4.log`

## Interpretation

**LR exhaustion hypothesis confirmed.** Epoch 1 of the fine-tune already exceeded v3's 60-epoch best raw F1 (0.4754 > 0.4715). The fresh OneCycleLR cycle at lr=1e-4 (vs v3's lr=3e-4 exhausted after 60 epochs) gave the model enough gradient signal to continue improving.

**Remaining weak classes:** CTU (0.4474), DoS (0.4343), ExternalBug (0.4838) all below 0.50. DoS remains a data problem (137 training samples). CTU and ExternalBug may benefit from more capacity (lora_r=16) or more epochs.

## Next Experiment Options

- **Option A:** Continue fine-tuning from exp 1 best, 30 more epochs at lr=5e-5 (lower to avoid overshoot)
- **Option B:** lora_r=16 fine-tune from exp 1 best (strict=False; LoRA re-init, GNN+fusion+classifier preserved)
- **DoS sampler:** Separate experiment — DoS-only weighted sampler — after architecture is settled
