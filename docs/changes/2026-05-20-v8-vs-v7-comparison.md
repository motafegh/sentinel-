# 2026-05-20 — v8.0-AB vs v7.0 Comparison

## Summary

Full threshold tuning and behavioral comparison of the v7.0 and v8.0-AB checkpoints after v8-AB was killed at epoch 37 (patience 8/30). Five hypotheses tested. Three infrastructure bugs fixed. PLAN-3A confirmed as next step.

---

## Checkpoint details at comparison time

- v7.0: `ml/checkpoints/v7.0_best.pt` — epoch 23, F1=0.2651
- v8.0-AB: `ml/checkpoints/v8.0-AB-20260520_best.pt` — epoch 29, F1=0.2621

---

## Infrastructure bugs fixed

Three bugs prevented loading cross-version checkpoints. All fixed in `tune_threshold.py` and `ml/src/inference/predictor.py`:

1. **`_orig_mod.` prefix**: `torch.compile` wraps submodule keys with `._orig_mod.` in the state dict. `load_state_dict()` fails because the plain-named parameters don't exist. Fix: strip `._orig_mod.` from all keys before loading.

2. **Edge embedding size mismatch**: v7 checkpoint has `edge_embedding.weight [8,64]` (8 edge types) but current code builds the model with 11 (v8 schema). Fix: detect embedding size from state dict key shape; rebuild `nn.Embedding(ckpt_n, emb_dim)` before `load_state_dict()`.

3. **BF16 mixed-dtype checkpoint**: BF16 AMP training stores 230/329 tensors as BF16 in the checkpoint. After `load_state_dict()` these stay BF16, causing a dtype mismatch in Linear forward. Fix: `model.float()` after loading.

Additional fix in `ml/src/models/gnn_encoder.py`: OOB edge_attr clamp now uses `self.edge_embedding.num_embeddings - 1` instead of the global `NUM_EDGE_TYPES` constant, so resized embeddings (v7 on v8 data) correctly clamp types 8/9/10 to 7.

---

## Threshold tuning results

| | Default (0.5) | Tuned | Gain |
|---|---|---|---|
| v7 | 0.2651 | **0.2875** | +0.022 |
| v8 | 0.2621 | **0.2851** | +0.023 |
| Gap | 0.003 v7 lead | 0.0024 v7 lead | closed 20% |

H4 (calibration shift) partially confirmed — both models benefit equally from tuning. Gap persists.

---

## Per-class winners (tuned thresholds)

v8 wins: IntegerUO (+0.009), ExternalBug (+0.013), TOD (+0.005)  
v7 wins: Reentrancy (−0.017 for v8), GasException (−0.009), CallToUnknown (−0.014), Timestamp (−0.006), UnusedReturn (−0.006)

---

## Behavioral test results

v7: 7/19 correct, 0/3 safe clean  
v8: 8/19 correct, 1/3 safe clean

v8 uniquely detects ExternalBug (contract 11). v8 has false positive explosion on complex contracts (6–8 classes fired simultaneously). v7 has DoS noise everywhere (threshold=0.05 fires on every contract).

---

## Hypothesis verdicts

| H | Hypothesis | Verdict |
|---|-----------|---------|
| H1 | Phase 2 dilution hurts Reentrancy CEI pattern | CONFIRMED |
| H2 | DEF_USE intra-function limits cross-function reach | PARTIALLY CONFIRMED |
| H3 | Label ceiling limits both models equally | CONFIRMED |
| H4 | Threshold calibration explains the gap | PARTIALLY CONFIRMED (minor effect) |
| H5 | Class-specific tradeoff, net negative | CONFIRMED |
| H6 | 4-type Phase 2 spread causes focus loss | PROBABLE |

---

## Files changed

| File | Change |
|------|--------|
| `ml/scripts/tune_threshold.py` | `--cache` arg; `_orig_mod` strip; edge embedding resize; `model.float()` |
| `ml/src/inference/predictor.py` | `_orig_mod` strip; edge embedding resize; `model.float()` |
| `ml/src/models/gnn_encoder.py` | OOB clamp uses `self.edge_embedding.num_embeddings` not global constant |
| `ml/checkpoints/v7.0_best_thresholds.json` | v7 per-class tuned thresholds |
| `ml/checkpoints/v8.0-AB-20260520_best_thresholds.json` | v8 per-class tuned thresholds |
| `docs/ml/v8-AB-training-analysis.md` | Added ep35–37 rows; updated to COMPLETE |
| `docs/ml/v8-vs-v7-comparison-results.md` | NEW — full comparison findings |
| `docs/STATUS.md` | Updated: v8-AB COMPLETE, comparison COMPLETE, PLAN-3A ready |
| `docs/changes/2026-05-20-v8-vs-v7-comparison.md` | This file |
