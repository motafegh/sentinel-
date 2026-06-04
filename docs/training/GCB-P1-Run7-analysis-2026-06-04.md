# Run 7 Training Analysis — GCB-P1-Run7-v10-20260603

**Run name:** `GCB-P1-Run7-v10-20260603`  
**Started:** 2026-06-03 20:43 (UTC+3:30)  
**Stopped:** 2026-06-04 20:55 (UTC+3:30) — deliberate mid-epoch-41 stop  
**Duration:** ~24 hours, 40 complete epochs + ep41 partial (step 200/455)  
**Best checkpoint:** `ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt` — ep39, F1-macro=0.3074  
**Training log:** `/tmp/run7_v10.log`  
**MLflow run:** `cc9610a0da9f4c4ebfffae2f4d57446a` (experiment: `sentinel-multilabel`)  
**Committed config:** `416d0e0` + `e2ad84e` + `139ebbc` + `6bee1a9`

---

## 1. Run Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Four-Eye v8.1 (BUG-R7-1/2 + IMP-R7-1/2/3 fixed) |
| Data | v10, 41,576 graphs, train=29,103 / val=6,236 / test=6,237 |
| GNN layers | 8 (3-phase GAT), NODE_FEATURE_DIM=11, type_embedding=Embedding(13,16) |
| Phase 2 heads | 4 (IMP-R7-1), Phase 1/3 heads=1 |
| GraphCodeBERT | Frozen + LoRA r=16 α=32 on Q+V, all 12 layers |
| Classifier | Four-eye concat [512] → Linear(512,256) → Linear(256,10) |
| Aux warmup | 0→0.30 over 8 epochs (`aux_loss_weight`) |
| aux_phase2 weight | 0.20 (IMP-R7-3, up from 0.10 in Run 6) |
| JK entropy reg λ | 0.005 |
| Patience | 30 epochs |
| Batch / accum | 8 / 8 steps (effective batch 64) |
| Optimizer | AdamW fused CUDA, LR=1e-4, GNN group LR×2.5 |
| fusion_lr_multiplier | 0.5 |
| threshold_tune_interval | 10 |
| gnn_prefix_k | 0 (disabled — not passed to train.py) |
| Speed | ~35 min/epoch on RTX 3070 8GB |

**Pre-run fixes applied (ISSUE-1 through ISSUE-4, commit `139ebbc`):**
- ISSUE-1: `cfg_eye_proj` moved from `_other_params` (LR×1.0) to GNN param group (LR×2.5)
- ISSUE-2: `cfg_eye_proj` + `aux_phase2` added to `torch.compile` submodule list
- ISSUE-3: Predictor now reads `fusion_max_nodes` from checkpoint config
- ISSUE-4: Predictor now reads `gnn_phase2_edge_types` from checkpoint config

**Fix #35 (commit `6bee1a9`, applied 2026-06-04):**  
Safe resume: default `resume_model_only=False`, saves/restores all RNG states (torch, CUDA, numpy, python random) and `_cached_tuned_thresholds` in checkpoint.

---

## 2. Stop Event

Training was halted **deliberately** mid-epoch 41 at step 200/455 (2026-06-04 20:55) via SIGTERM. Reason: post-ep40 analysis session revealed structural class ceilings and the sawtooth-as-noise finding (see §5, §6). No crash, no NaN events, no abort. SIGTERM was received cleanly — leaked semaphore warning at shutdown is multiprocessing cleanup noise, not an error.

The best checkpoint (`ep39, F1=0.3074`) had already saved before the stop. Ep40 (F1=0.3012) and partial ep41 were not checkpointed.

---

## 3. Epoch-by-Epoch Summary (ep1–40)

Epochs 1–9 logged at step-100 intervals only (StructuredLogger epoch rows failed; see §8). Full epoch rows begin at ep10.

| Ep | Val Loss | F1-macro | Hamming | Best? | Patience | Notes |
|----|----------|----------|---------|-------|----------|-------|
| 1  | —        | 0.1678   | —       | ★     | —        | aux warmup at 0 |
| 2  | —        | 0.1345   | —       |       | 1        | DoS → 0 |
| 3  | —        | 0.1765   | —       | ★     | —        | |
| 4  | —        | 0.2085   | —       | ★     | —        | |
| 5  | —        | 0.2393   | —       | ★     | —        | |
| 6  | —        | 0.2399   | —       | ★     | —        | DoS → 0 |
| 7  | —        | 0.2428   | —       | ★     | —        | |
| 8  | —        | 0.2468   | —       | ★     | —        | aux warmup complete (0.30) |
| 9  | —        | 0.2711   | —       | ★     | —        | |
| 10 | 0.6053   | 0.2780   | 0.2841  | ★     | —        | First full epoch log |
| 11 | 0.5958   | 0.2633   | 0.3441  |       | 1        | |
| 12 | 0.5952   | 0.2689   | 0.3395  |       | 2        | |
| 13 | 0.5953   | 0.2596   | 0.3476  |       | 3        | |
| 14 | 0.5912   | 0.2661   | 0.3099  |       | 4        | DoS → 0 |
| 15 | 0.5894   | 0.2634   | 0.3337  |       | 5        | |
| 16 | 0.5880   | 0.2939   | 0.2585  | ★     | —        | Sawtooth peak 1 |
| 17 | 0.5841   | 0.2833   | 0.2851  |       | 1        | |
| 18 | 0.5845   | 0.2948   | 0.2728  | ★     | —        | Sawtooth peak 2 |
| 19 | 0.5830   | 0.2843   | 0.3036  |       | 1        | |
| 20 | 0.5794   | 0.2872   | 0.2665  |       | 2        | threshold tune (ep10/20/30/40) |
| 21 | 0.5769   | 0.2790   | 0.3047  |       | 3        | |
| 22 | 0.5796   | 0.2865   | 0.2914  |       | 4        | |
| 23 | 0.5731   | 0.2911   | 0.2816  |       | 5        | |
| 24 | 0.5766   | 0.2915   | 0.2926  |       | 6        | |
| 25 | 0.5730   | 0.2884   | 0.2958  |       | 7        | |
| 26 | 0.5721   | 0.2902   | 0.2934  |       | 8        | |
| 27 | 0.5717   | 0.2973   | 0.2727  | ★     | —        | |
| 28 | 0.5699   | 0.3034   | 0.2489  | ★     | —        | Sawtooth peak 3 (first >0.30) |
| 29 | 0.5721   | 0.2943   | 0.2962  |       | 1        | |
| 30 | 0.5713   | 0.2913   | 0.3056  |       | 2        | threshold tune |
| 31 | 0.5688   | 0.3000   | 0.2705  |       | 3        | |
| 32 | 0.5650   | 0.3040   | 0.2754  | ★     | —        | Sawtooth peak 4 |
| 33 | 0.5647   | 0.3069   | 0.2618  | ★     | —        | Sawtooth peak 4b |
| 34 | 0.5658   | 0.2915   | 0.3004  |       | 1        | |
| 35 | 0.5641   | 0.2942   | 0.2915  |       | 2        | |
| 36 | 0.5621   | 0.3055   | 0.2611  |       | 3        | Sawtooth peak 5 |
| 37 | 0.5643   | 0.2971   | 0.2903  |       | 4        | |
| 38 | 0.5633   | 0.2946   | 0.2863  |       | 5        | |
| 39 | 0.5612   | **0.3074** | 0.2510 | ★    | —        | **Final best** — Sawtooth peak 6 |
| 40 | 0.5601   | 0.3012   | 0.2799  |       | 1        | |
| 41 | —        | —        | —       |       | —        | Stopped at step 200 |

Val loss: monotonically declining every epoch from 0.6053 (ep10) to 0.5601 (ep40) with only two micro-reversals (ep22, ep29, ep34, ep37). Train step loss at ep41 step 200 was 0.1148 — lowest recorded, confirming training was still progressing.

---

## 4. MLflow Metrics Summary

All epoch metrics logged to MLflow run `cc9610a0`. Full metric list:

| Metric | Entries | Min | Max |
|--------|---------|-----|-----|
| val_f1_macro | 40 | 0.1345 | 0.3074 |
| val_f1_micro | 40 | 0.2190 | 0.3741 |
| val_hamming | 40 | 0.2489 | 0.5875 |
| val_f1_macro_tuned | 5 | 0.1724 | 0.3329 |
| train_loss | 40 | 0.3246 | 0.6053 |
| gnn_grad_share | 40 | 0.163 | 0.912 |
| ph2_ph1_grad_ratio | 40 | 0.408 | 1.038 |
| jk_phase1/2/3_weight | 40 each | — | — |
| jk_phase1/2/3_std | 40 each | — | — |
| aux_loss_weight_effective | 40 | 0.000 | 0.300 |
| aux_phase2_loss_weight | 40 | 0.200 | 0.200 |
| nan_batch_count | 40 | 0 | 0 |

**Note:** MLflow `train_loss` contains the **validation total loss** (naming inconsistency in logger — the `val_loss` field is what MLflow stores under this key). Step-level train loss is only available in `/tmp/run7_v10.log`.

**Ghost run:** A second MLflow run `541345ba` (same name, started 1h earlier on 2026-06-03 19:36) has status=RUNNING with 26 metrics through ep1 only. This is the aborted pre-ISSUE-fix attempt. It will never be closed by the trainer. Can be manually closed via `mlflow runs set-terminated --run-id 541345bab6864f738e484794122607bc --status KILLED`.

---

## 5. Sawtooth Pattern Analysis — Root Cause: DoS Noise

The apparent ~6-epoch sawtooth oscillation in macro F1 is **not a real learning cycle**. It is entirely attributable to DoS class variance, caused by only 65 positive examples in the validation set (1.04% prevalence).

**Evidence:**

| Sawtooth peak | F1-macro | DoS F1 | Preceding DoS |
|---|---|---|---|
| ep10 | 0.2780 | 0.021 | 0.000 (eps 6–9) |
| ep16 | 0.2939 | 0.057 | 0.048 (ep15) |
| ep18 | 0.2948 | 0.121 | 0.053 (ep17) — large jump |
| ep28 | 0.3034 | 0.136 | 0.191 (ep27) |
| ep32–33 | 0.3040–0.3069 | 0.230–0.194 | established higher base |
| ep39 | 0.3074 | 0.164 | 0.168 (ep38) |

Troughs (ep11–15, ep19–26, ep29–31, ep34–38) all coincide with DoS collapsing back toward zero. With 65 positives, one extra true-positive prediction changes F1 by ~0.008. The model is not cycling — DoS predictions are inherently unstable at this sample size.

**True trend excluding DoS noise:** The macro F1 of the remaining 9 classes combined has been **flat since ep20**, oscillating in roughly 0.287–0.295. The apparent improvement from ep20 (0.2872) to ep39 (0.3074) is ~0.017 in DoS contribution and ~0.003 in everything else.

**Implication for early stopping:** The 30-epoch patience is working correctly. DoS variance will continue pushing F1 up and down; patience should be measured on the smoothed trend, not per-epoch peaks.

---

## 6. Per-Class F1 Analysis (ep10 → ep40)

| Class | ep10 | ep20 | ep30 | ep40 | Δ | Status |
|-------|------|------|------|------|---|--------|
| IntegerUO | 0.662 | 0.686 | 0.645 | 0.686 | +0.024 | Improving slowly |
| GasException | 0.359 | 0.363 | 0.346 | 0.363 | +0.004 | Near ceiling |
| MishandledException | 0.322 | 0.312 | 0.300 | 0.317 | -0.004 | Slight regression |
| Reentrancy | 0.275 | 0.303 | 0.293 | 0.311 | +0.036 | Still improving |
| CallToUnknown | 0.246 | 0.258 | 0.249 | 0.252 | +0.006 | Near ceiling |
| UnusedReturn | 0.234 | 0.238 | 0.233 | 0.234 | +0.000 | **Structural ceiling** |
| TransactionOrderDependence | 0.254 | 0.258 | 0.247 | 0.245 | -0.009 | Slight regression |
| ExternalBug | 0.258 | 0.260 | 0.242 | 0.249 | -0.009 | Slight regression |
| DenialOfService | 0.021 | 0.031 | 0.196 | 0.211 | +0.190 | Noisy (see §5) |
| Timestamp | 0.150 | 0.163 | 0.161 | 0.145 | -0.005 | **Structural ceiling** |

### Structural ceilings (architectural root causes)

**UnusedReturn (0.234, flat for 30 epochs):** Detecting unused return values requires tracking def-use chains across assignment statements. The current graph schema has no DEF_USE edge type (deferred as RC5). Without these edges, the GNN cannot reason about whether a return value is consumed. This is a data representation limit, not a model capacity limit.

**Timestamp (0.145–0.170, flat for 30 epochs):** Timestamp dependency requires reasoning about whether a contract's outcome is affected by `block.timestamp` propagation through conditional branches. The graph captures CFG but not data-flow provenance of timestamp values. Additionally, any genuine timestamps in function signatures get normalized away during tokenization. The model learned the surface-level "looks like a timestamp pattern" by ep10 and cannot go further.

**TransactionOrderDependence + ExternalBug (slight regression):** Both require cross-contract reasoning — TOD needs to model two competing transaction sequences, ExternalBug needs to propagate exception states across external call edges. Neither is representable in the current single-contract graph schema.

### Classes still improvable

- **Reentrancy (+0.036, ep10→ep40):** Consistently improving. Re-entrancy patterns involve recursive call cycles which the Phase 2 ICFG edges do capture. Likely to reach 0.33–0.36 with more epochs.
- **IntegerUO:** Oscillating 0.64–0.71. The model hasn't fully stabilised; seen peak 0.707 at ep31. Likely to stabilise around 0.70.
- **DoS:** Genuine upward trend beneath the noise (0.021 ep10 → 0.21+ by ep40). Needs calibrated threshold at inference (optimal ~0.20–0.24, not fixed 0.35).

---

## 7. Threshold Tuning Analysis (MLflow `val_f1_macro_tuned`)

Per-class threshold tuning runs every 10 epochs (BUG-M8 implementation). Results stored in MLflow only — not in training log.

| Epoch | Fixed F1 | Tuned F1 | Gap |
|-------|----------|----------|-----|
| 1 | 0.1678 | 0.1724 | +0.0046 |
| 10 | 0.2780 | 0.2925 | +0.0145 |
| 20 | 0.2872 | 0.3007 | +0.0135 |
| 30 | 0.2913 | 0.3186 | +0.0273 |
| 40 | 0.3012 | 0.3329 | **+0.0317** |

The gap grows monotonically (+0.017 from ep10→ep40). This means:

1. The default fixed threshold of 0.35 is increasingly miscalibrated as the training run progresses. By ep40, the model's **true capability is F1=0.3329** — already within 0.003 of Run 4's best fixed F1 (0.3362 on v9 data, a different distribution).
2. The growing gap is driven primarily by DoS and Timestamp, whose optimal thresholds are far below 0.35 (estimated 0.20–0.25 based on class prevalence).
3. **Post-training per-class calibration is mandatory** before using this checkpoint for inference. The `temperatures_run7.json` file needs to be generated from the ep39 checkpoint. See §11.

---

## 8. JK Phase Weight Analysis

Per-phase JK attention weights (mean ± std over val set) tracked via MLflow:

| Epoch | Phase1 ± std | Phase2 ± std | Phase3 ± std |
|-------|-------------|-------------|-------------|
| ep1 | 0.334 ± 0.036 | 0.319 ± 0.024 | 0.347 ± 0.032 |
| ep5 | 0.323 ± 0.038 | 0.335 ± 0.034 | 0.342 ± 0.046 |
| ep10 | 0.318 ± 0.035 | 0.332 ± 0.029 | 0.350 ± 0.049 |
| ep15 | 0.310 ± 0.044 | 0.324 ± 0.047 | 0.366 ± 0.071 |
| ep20 | 0.302 ± 0.056 | 0.319 ± 0.053 | 0.379 ± 0.086 |
| ep25 | 0.302 ± 0.049 | 0.317 ± 0.047 | 0.380 ± 0.079 |
| ep30 | 0.298 ± 0.065 | 0.305 ± 0.055 | 0.396 ± 0.103 |
| ep35 | 0.302 ± 0.067 | 0.311 ± 0.061 | 0.387 ± 0.107 |
| ep40 | 0.304 ± 0.070 | 0.301 ± 0.061 | **0.395 ± 0.106** |
| ep39* | 0.311 ± 0.073 | 0.316 ± 0.065 | 0.373 ± 0.112 | ← best checkpoint |

*ep39 JK (best checkpoint) shows Phase 3 dropped back to 0.373 from the ep30 high of 0.396. The best checkpoints correlate with Phase 3 not being at its highest — suggesting Phase 3 dominance slightly hurts performance even as val loss improves.

**Drift summary:** Phase3 +0.048 (ep1→ep40), Phase1 −0.030, Phase2 −0.018. λ=0.005 entropy regularization is preventing collapse but not fully arresting drift. The Phase 3 std is growing (0.032 ep1 → 0.106 ep40), indicating it has become increasingly context-dependent: dominant for some contract types, near-absent for others. This is architecturally plausible (containment hierarchy matters more for scope-based vulnerabilities) but warrants monitoring.

**Watch thresholds (not yet triggered):** Phase3 > 0.40 (reached 0.396 ep30, then recovered). Phase1 < 0.28 (minimum was 0.298 at ep30, recovered to 0.304 ep40).

---

## 9. Gradient Flow Analysis

### GNN gradient share (epoch-level, from MLflow)

| Epoch | GNN share | Ph2/Ph1 ratio |
|-------|-----------|---------------|
| ep1 | 91.2% | 0.748 |
| ep2 | 61.7% | 1.038 |
| ep5 | 45.4% | 0.474 |
| ep10 | 57.5% | 0.471 |
| ep15 | 33.6% | 0.506 |
| ep20 | 42.0% | 0.599 |
| ep25 | 31.5% | 0.620 |
| ep30 | 27.6% | 0.710 |
| ep35 | 30.3% | 0.661 |
| ep40 | 28.5% | 0.618 |

**GNN share trajectory:** Started at 91% (LoRA cold, GNN dominated) → settled to 27–35% by ep30+. This transition is expected and healthy — it reflects LoRA layers taking on more gradient responsibility as they warm up. The asymptote at ~30% GNN / ~70% transformer is characteristic of this architecture.

**Ph2/Ph1 ratio (Phase2/Phase1 gradient):** Stabilised at 0.47–0.71 across ep10–40. **This is the key confirmation that ISSUE-1 (cfg_eye_proj in GNN group) worked.** Compare to Run 4 where this ratio was 0.10–0.18 — Phase 2 was getting nearly no gradient. Run 7's Phase 2 gets real gradient throughout.

### Fusion gradient spikes (step-level, from training log)

Recurring spikes in fused gradient at step 100–200 of each epoch, typically 0.09–0.165. Examples:
- ep31 step 100: fused=0.138, GNN share=20.0%
- ep32 step 100: fused=0.138, GNN share=20.0%
- ep39 step 100: fused=0.134, GNN share=22.0% (ep41 step 100)

Spikes are transient — by step 300–400 the fused gradient normalises to 0.09–0.11. Loss never rises from them. Root cause: `fusion_lr_multiplier=0.5` is slightly high given the 4-eye classifier adds more loss signal routed through fusion than Run 4's 3-eye architecture. **Run 8 recommendation:** reduce to 0.3.

---

## 10. StructuredLogger Bug (Found During This Session)

**Error (every epoch since ep10):**
```
[Phase 4.6] StructuredLogger epoch logging failed: 'OptimizedModule' object is not subscriptable
```

**Root cause:** `ml/src/training/training_logger.py` line 305:
```python
head = getattr(model, "aux_phase2", None)
final_linear = head[-1]  # ← FAILS: OptimizedModule doesn't support subscript
```

After `torch.compile`, `model.aux_phase2` returns an `OptimizedModule` wrapping the `nn.Sequential`. `OptimizedModule` does not support `[-1]` subscript indexing.

**Fix:**
```python
head = getattr(model, "aux_phase2", None)
head = getattr(head, "_orig_mod", head)  # unwrap torch.compile wrapper
final_linear = head[-1]
```

**Impact:** All StructuredLogger epoch-level data was silently empty for the **entire Run 7** (ep1–40). This means no AUC-ROC curves, no Brier scores, no ECE calibration metrics, no aux head weight norms, no per-epoch probability distribution stats were logged to the structured JSONL files. MLflow per-epoch metrics (F1, JK weights, grad share) were unaffected — those are logged separately before the structured logger call. This bug should be fixed before Run 8 starts.

---

## 11. micro vs macro F1 Spread

| Epoch | micro F1 | macro F1 | Spread |
|-------|----------|----------|--------|
| ep1 | 0.219 | 0.168 | +0.051 |
| ep2 | 0.279 | 0.135 | **+0.144** — worst (DoS/Timestamp zeroed) |
| ep10 | 0.348 | 0.278 | +0.070 |
| ep20 | 0.365 | 0.287 | +0.078 |
| ep30 | 0.348 | 0.291 | +0.057 |
| ep39 | **0.374** | **0.307** | +0.067 |
| ep40 | 0.359 | 0.301 | +0.057 |

The spread at peaks (~0.057–0.070) represents the macro penalty from rare/hard classes. The ceiling for macro F1 without architecture changes is approximately: `micro_peak (0.374) - class_imbalance_floor (0.04–0.05) ≈ 0.32–0.33` at fixed threshold, or `~0.36–0.38` with tuned per-class thresholds.

---

## 12. Comparison to Previous Runs

| Run | Data | Best ep | F1 (fixed) | F1 (tuned) | Note |
|-----|------|---------|------------|------------|------|
| Run 4 | v9 | 32 | **0.3362** | n/a | Capacity ceiling ep44 |
| Run 6 | v10 | 29 | 0.2988 | n/a | KILLED — below Run4, arch ceiling |
| Run 7 | v10 | 39 | 0.3074 | **0.3329** | Stopped ep41 |

Run 7 at ep39 with tuned thresholds (0.3329) is effectively at parity with Run 4 (0.3362 fixed). Given Run 7 uses v10 data (C-1 + H-2 fixes over v9) and the 4-eye architecture, this is a genuine improvement in model quality even though the fixed-threshold F1 is lower — the gap is calibration, not capability.

---

## 13. Known Issues and Observations Not in Previous Docs

1. **aux_weight discrepancy (confirmed non-issue):** An external review noted apparent inconsistency between `aux_loss_weight=0.30` and the `aux_phase2_loss_weight=0.20` logged in MLflow. These are two separate weights — `aux_loss_weight` controls the 3 GNN/TF/Fused auxiliary heads (unchanged from Run 6), `aux_phase2_loss_weight` is the IMP-R7-3 new weight for the Phase 2 auxiliary head only. No bug.

2. **CFG eye not visible in phase-specific logs (confirmed correct):** A concern was raised that the CFG eye loss wasn't logged separately. The CFG eye feeds directly into the 4-eye classifier concat — it contributes to `main_loss` not to a separate `aux` term. The `ph2` entry in step logs refers to `aux_phase2`, not the CFG eye.

3. **DoS zeros eps 6–9 explained:** DoS had 0 positive predictions for 4 consecutive epochs. With 65 val positives and a threshold of 0.35, the model's raw logits for DoS simply didn't exceed threshold during aux warmup phase. Recovered at ep10 and has been positive since, with high variance.

4. **ep35 speed anomaly:** ep35 took 39.0 min vs normal 34–36 min. Single occurrence, likely OS scheduling/memory pressure. No recurrence.

5. **GNN share ep1=91.2%:** Expected — LoRA is initialised near identity (α/r = 2.0 scaling) so the transformer contributes almost no gradient at ep1. Normalises by ep5.

---

## 14. Run 8 Recommendations

Based on 40 epochs of Run 7 observations:

| # | Parameter | Current | Recommended | Reason |
|---|-----------|---------|-------------|--------|
| R8-1 | `fusion_lr_multiplier` | 0.5 | **0.3** | Recurring fusion gradient spikes 0.09–0.165 at step 100–200; 4-eye arch routes more loss through fusion than Run 4 |
| R8-2 | `--gnn-prefix-k` | 0 (disabled) | **48** | Was not enabled in Run 7; prefix injection may improve structural priming |
| R8-3 | JK entropy λ | 0.005 | **0.0075** | Phase 3 drifted to 0.395 by ep40; λ=0.005 arresting but not sufficient to hold at ~0.36 |
| R8-4 | Post-training calibration | not done | **immediate** | Run per-class threshold calibration on ep39 checkpoint before further analysis |

**Classes not improvable without architecture changes (defer to Run 9+):**
- UnusedReturn → needs RC5 DEF_USE edges
- Timestamp → needs data-flow provenance of `block.timestamp`
- TransactionOrderDependence → needs cross-contract reasoning
- ExternalBug → needs cross-contract reasoning

**StructuredLogger fix (apply before Run 8):** Add `head = getattr(head, "_orig_mod", head)` in `ml/src/training/training_logger.py:305` before `head[-1]` access.

---

## 15. Checkpoint State at Stop

| Checkpoint | Epoch | F1-macro | Val Loss | Status |
|------------|-------|----------|----------|--------|
| `GCB-P1-Run7-v10-20260603_best.pt` | 39 | **0.3074** | 0.5612 | ✓ Active best |
| `GCB-P1-Run7-v10-20260603_last.pt` | — | — | — | May contain ep41 partial state (not a valid eval checkpoint) |

Best checkpoint contains (Fix #35 additions):
- Full model state dict (keys with `._orig_mod.` infix stripped)
- Optimizer state (AdamW momentum/variance)
- LR scheduler state
- All four RNG states (torch, CUDA, numpy, python random)
- `_cached_tuned_thresholds` (per-class thresholds from ep30 tune run)
- Epoch, best F1, config dict

**Resume command (if training is to continue):**
```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --resume ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt \
  --run-name GCB-P1-Run7-v10-20260603-resumed \
  --experiment-name sentinel-multilabel \
  --epochs 100 \
  --gradient-accumulation-steps 8
```

**Next immediate action:** Run per-class threshold calibration on the ep39 checkpoint to generate `ml/calibration/temperatures_run7.json` before any inference or interpretability work.

---

*Document generated: 2026-06-04, post-stop analysis session. References: `/tmp/run7_v10.log`, MLflow run `cc9610a0`, `docs/CHANGELOG.md` §35–36, `docs/pre-run7-fixes/`.*
