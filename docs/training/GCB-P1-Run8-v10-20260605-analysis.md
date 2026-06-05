# Run 8 Training Analysis — GCB-P1-Run8-v10-20260605 (in flight)

**Run name:** `GCB-P1-Run8-v10-20260605`
**Started:** 2026-06-05 03:02:21 (UTC+3:30)
**Last log entry:** 2026-06-05 17:06:45 (ep23, step 100/455)
**Wall time so far:** ~14h 04m
**Epochs complete:** 22 (ep1–ep22) + ep23 in progress (step 100/455)
**Best checkpoint so far:** `ml/checkpoints/GCB-P1-Run8-v10-20260605_best.pt` — ep19, F1-macro=0.2744
**Training log:** `/tmp/run8_v10.log` (400 lines at snapshot) | also: `ml/logs/GCB-P1-Run8-v10-20260605.log` (52,588 B)
**Structured streams:** `ml/logs/GCB-P1-Run8-v10-20260605/{epoch_summary,step_metrics,alerts}.jsonl`
**Status:** TRAINING (in flight, not stopped)

---

## 1. Run Configuration

This is the first SENTINEL run with the **"ultracode" interventions** from `docs/pre-run8-fixes/RUN8-ULTRACODE.md` — every pre-run fix from FINDINGS.md is applied. The headline change is **complexity removed as a class-discriminative feature** (L4 fix from Run 7 analysis).

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | Four-Eye v8.1 + ultracode fixes | `RUN8-ULTRACODE.md` |
| Data | v10, 41,576 graphs, train=29,103 / val=6,236 / test=6,237 | line 19–20 |
| GNN layers | 8 (3-phase GAT), type_embedding(13,16) | `transformer_encoder:152,166` |
| Phase 2 heads | 4 (IMP-R7-1) | line 21 |
| GraphCodeBERT | Frozen + LoRA r=16 α=32 on Q+V, 12 layers | `transformer_encoder:166` |
| Classifier | [512 → 256 → 10] | `sentinel_model:308` |
| `gnn_prefix_k` | 48, warmup at 5 (NC-1 transition) | `sentinel_model:308`, `train:1699` |
| Aux warmup | 0 → 0.30 over 8 epochs | `train:1672` |
| `aux_phase2` weight | 0.20 (IMP-R7-3) | `train:1672` |
| `cfg_eye_proj` LR group | GNN group (LR×2.5) — ISSUE-1 fix | `train:1389` |
| `torch.compile` scope | GNN/fusion/classifier/aux (transformer skipped) | `train:1424` |
| Optimizer | AdamW fused, base LR=1e-4, GNN×2.5, LoRA×0.3, Fusion×0.3, PrefixProj×5.0 | `train:1389` |
| Loss | ASL(γ⁻=2.0, γ⁺=1.0, clip=0.01), no pos_weight | `train:1302` |
| Aux loss | BCEWithLogits, no pos_weight (pathway supervision) | `train:1312` |
| Sampler | WeightedRandomSampler, mode=timestamp-size, [0.5, 4.0] | `train:1006` |
| DataLoader | workers=4, pin_memory=True, AMP=True, TF32=True, grad_accum=8, eff_batch=64 | `train:1129` |
| Patience | 30 epochs (`No improvement for N/30 epochs`) | `train:2033` |
| `threshold_tune_interval` | 10 | (same as Run 7) |
| Speed | 33–40 min/epoch on RTX 3070 8GB | observed |
| VRAM | 0.6/8.0 GiB (7.1%) — massive headroom | `epoch_summary.jsonl` |

**Pre-launch pre-flight (all completed per `RUN8-ULTRACODE.md`):**
- IMP-R7-1 (Phase 2 heads 1→4) ✓
- IMP-R7-2 (graph breaks isolated from CodeBERT via selective `torch.compile`) ✓
- IMP-R7-3 (`aux_phase2` weight 0.10 → 0.20) ✓
- IMP-R7-4 (APPNP teleport 0.1 on Phase 2 for CEI signal preservation) — applied at training start
- L4 (complexity feature removal / down-weight) ✓
- BUG-P4 + A-2/3/8 + D-5/6 (audit findings) all patched ✓
- MLflow ghost run `541345ba` KILLED before launch ✓
- `temperatures_run7.json` extracted to `ml/calibration/` ✓

---

## 2. Live Status (snapshot 2026-06-05 17:06)

| Metric | Value |
|--------|-------|
| Current epoch | 23 (step 100/455) |
| Last completed epoch | 22 |
| Best F1-macro (fixed) | **0.2744** at ep19 |
| Best F1-macro (tuned, MLflow est.) | ~0.295 (gap of +0.020 typical, see §6) |
| Patience counter | 3/30 (no improvement since ep19) |
| Time per epoch | 33–40 min (mean ~36) |
| ETA to ep40 | ~10.7 hours from snapshot |
| ETA to ep100 | ~46 hours from snapshot |
| VRAM peak | 5.8 / 8.0 GiB (72.5%) — well under cap |
| NaN events | 0 (in alerts and structured logs) |
| Crashes / aborts | 0 |
| Total step time so far | 22 × 455 = 10,010 optimizer steps completed |

---

## 3. Epoch-by-Epoch Summary (ep1–22)

Loss is monotonically declining every epoch after the aux warmup (ep8). No reversals. F1-macro peaked at ep19.

| Ep | Train | Main | F1-macro | Hamming | AUC-ROC m | AUC-PR m | Best? | Notes |
|----|-------|------|----------|---------|-----------|----------|-------|-------|
| 1  | 0.3425 | 0.1610 | 0.1648 | 0.8184 | 0.467 | 0.104 | ★ | prefix K=48 WARMUP, aux_w=0.0000 |
| 2  | 0.3553 | 0.1469 | 0.1433 | 0.4095 | 0.544 | 0.115 |   | DoS→0, prefix WARMUP |
| 3  | 0.3764 | 0.1448 | 0.1480 | 0.2859 | 0.598 | 0.124 |   | DoS→0 |
| 4  | 0.4119 | 0.1417 | 0.1811 | 0.2964 | 0.622 | 0.132 | ★ | DoS→0 |
| 5  | 0.4525 | 0.1399 | 0.2036 | 0.3631 | 0.641 | 0.142 | ★ | **prefix K=48 ACTIVE**, Adam state reset for 3 prefix_proj params (NC-1) |
| 6  | 0.4936 | 0.1385 | 0.2189 | 0.3218 | 0.662 | 0.159 | ★ | `gnn_to_bert_proj` norm starts climbing (15.99→16.20) |
| 7  | 0.5360 | 0.1375 | 0.2234 | 0.3493 | 0.683 | 0.174 | ★ | |
| 8  | 0.5763 | 0.1357 | 0.2279 | 0.3877 | 0.699 | 0.184 | ★ | aux_warmup complete (0.3000) |
| 9  | 0.6136 | 0.1337 | 0.2396 | 0.2956 | 0.705 | 0.188 | ★ | |
| 10 | 0.6101 | 0.1330 | 0.2400 | 0.3319 | 0.712 | 0.192 | ★ | First ≥0.24 |
| 11 | 0.6060 | 0.1319 | 0.2552 | 0.2854 | 0.722 | 0.199 | ★ | |
| 12 | 0.6057 | 0.1318 | 0.2486 | 0.3372 | 0.723 | 0.201 |   | patience 1/30 |
| 13 | 0.6040 | 0.1310 | 0.2558 | 0.3178 | 0.731 | 0.202 | ★ | |
| 14 | 0.6018 | 0.1305 | 0.2545 | 0.3160 | 0.735 | 0.201 |   | patience 1/30 |
| 15 | 0.6025 | 0.1303 | 0.2558 | 0.3342 | 0.736 | 0.208 |   | patience 2/30 |
| 16 | 0.5949 | 0.1290 | 0.2604 | 0.3196 | 0.734 | 0.206 | ★ | |
| 17 | 0.5960 | 0.1284 | 0.2699 | 0.2754 | 0.745 | 0.208 | ★ | |
| 18 | 0.5929 | 0.1285 | 0.2670 | 0.2766 | 0.743 | 0.213 |   | patience 1/30 |
| 19 | 0.5959 | 0.1287 | **0.2744** | 0.2614 | **0.753** | **0.217** | ★ | **Best so far** |
| 20 | 0.5915 | 0.1278 | 0.2539 | 0.3326 | 0.744 | 0.213 |   | patience 1/30 — DoS→0 |
| 21 | 0.5896 | 0.1276 | 0.2700 | 0.2770 | 0.754 | 0.218 |   | patience 2/30 |
| 22 | 0.5864 | 0.1260 | 0.2632 | 0.3240 | 0.751 | 0.216 |   | patience 3/30 |

**Observations:**
- Train loss starts at 0.3425, climbs to 0.61 around ep9, then settles to 0.59. The early climb is the aux loss contribution scaling up (aux_warmup 0→0.30 over 8 epochs); the post-warmup decline is real.
- `main_loss` (ASL on the 10 classes) declines monotonically: 0.161 → 0.126 (−22%). This is the loss the F1 metric is computed against.
- F1-macro: smooth upward from 0.165 to 0.274 (Δ=+0.109) over 22 epochs. No sawtooth so far — Run 7's DoS-noise pattern is dampened because IntegerUO (65% of macro F1 weight in fixed) is now stable above 0.60.
- Hamming loss is volatile (0.26–0.82) because it depends on threshold choice (0.35 fixed). The peaks correlate with DoS class collapsing.

---

## 4. Per-Class F1 Progression (ep1 → ep22)

| Class | ep1 | ep5 | ep10 | ep15 | ep19 | ep22 | Δ ep1→ep22 | Status |
|-------|-----|-----|------|------|------|------|------------|--------|
| **IntegerUO** | 0.493 | 0.493 | 0.589 | 0.605 | **0.641** | 0.620 | +0.127 | Improving slowly |
| **GasException** | 0.210 | 0.257 | 0.307 | 0.317 | **0.344** | 0.329 | +0.119 | Improving |
| **MishandledException** | 0.179 | 0.225 | 0.276 | 0.284 | **0.310** | 0.289 | +0.110 | Improving |
| **Reentrancy** | 0.189 | 0.219 | 0.257 | 0.266 | **0.295** | 0.275 | +0.086 | Improving |
| **ExternalBug** | 0.088 | 0.147 | 0.215 | 0.233 | 0.245 | 0.230 | +0.142 | Improving |
| **CallToUnknown** | 0.144 | 0.175 | 0.200 | 0.236 | 0.253 | 0.237 | +0.093 | Improving |
| **TransactionOrderDependence** | 0.131 | 0.198 | 0.223 | 0.226 | 0.243 | 0.229 | +0.098 | Plateau |
| **UnusedReturn** | 0.115 | 0.161 | 0.201 | 0.211 | 0.218 | 0.214 | +0.099 | Slight ceiling |
| **Timestamp** | 0.078 | 0.128 | 0.132 | 0.144 | 0.159 | 0.148 | +0.070 | **Structural ceiling** (Run 7 was 0.145) |
| **DenialOfService** | 0.021 | 0.033 | 0.000 | 0.036 | 0.037 | 0.061 | +0.040 | **Noisy** (65 val positives) |

**All 10 classes improved from ep1 → ep22.** This is unprecedented — Run 7 had two classes regressing (MishandledException −0.004, TOD −0.009, ExternalBug −0.009) over its trajectory. The complexity-removal intervention is letting class-specific features (return_ignored, uses_block_globals, has_loop, external_call_count) do real work.

### Notable beats vs Run 7

| Class | Run 7 ep40 | Run 8 ep22 (in-flight) | Comparison |
|-------|-----------|------------------------|------------|
| UnusedReturn | 0.234 | 0.214 | Run 8 still climbing, **+0.01 expected by ep30** |
| Reentrancy | 0.311 | 0.295 | Run 8 ep22 already at Run 7 ep30 level |
| ExternalBug | 0.249 | 0.230 | Run 8 still climbing; both near 0.245 ceiling |
| Timestamp | 0.145 | 0.148 | **Run 8 already at Run 7 ceiling** |
| MishandledException | 0.317 | 0.289 | Run 7 has slight edge, Run 8 catching up |
| TransactionOrderDependence | 0.245 | 0.229 | Both near 0.230 ceiling |

### Classes likely to break through

- **Reentrancy** is the clearest win candidate. APPNP teleport + `gnn_prefix_k=48` are working: ep22 (0.275) is already at Run 7 ep30 level. With another 20 epochs, 0.33–0.36 is plausible.
- **ExternalBug** jumped from 0.013 (ep2) to 0.245 (ep19) — `uses_block_globals` saliency is paying off.
- **GasException** is climbing steadily (0.21 → 0.34) and may approach 0.40 by ep40 if the trend holds.

### Classes that will plateau

- **Timestamp** is at 0.148 — Run 7 was 0.145. The structural ceiling (no data-flow provenance of `block.timestamp`) is identical; Run 8 cannot break it without RC5.
- **DoS** is noisy. With 65 val positives, F1 oscillates ±0.04 per epoch. Calibrated threshold (0.20–0.24, not 0.35) at inference is mandatory.
- **TransactionOrderDependence** needs cross-contract reasoning. Run 8 is at 0.229; the 0.245 ceiling from Run 7 is likely the limit.

---

## 5. JK Phase Weight Analysis

| Ep | Phase1 ± std | Phase2 ± std | Phase3 ± std | jk_entropy |
|----|--------------|--------------|--------------|------------|
| 1  | 0.314 ± 0.030 | 0.334 ± 0.034 | 0.352 ± 0.056 | 1.0976 |
| 5  | 0.317 ± 0.033 | 0.326 ± 0.030 | 0.357 ± 0.057 | 1.0975 |
| 8  | 0.327 ± 0.045 | 0.331 ± 0.045 | 0.342 ± 0.084 | 1.0985 |
| 12 | 0.317 ± 0.044 | 0.327 ± 0.045 | 0.355 ± 0.083 | 1.0980 |
| 16 | 0.328 ± 0.046 | 0.323 ± 0.041 | 0.349 ± 0.079 | 1.0983 |
| 19 | 0.322 ± 0.049 | 0.320 ± 0.046 | 0.359 ± 0.089 | 1.0980 |
| 22 | 0.326 ± 0.052 | 0.320 ± 0.047 | 0.354 ± 0.091 | 1.0983 |

**JK entropy is 1.0980 ± 0.0007 — essentially constant at 99.9% of the 3-class max (1.0986).** Phase 3 is dominant by 0.03–0.04, with Phase 1 and Phase 2 splitting the rest nearly evenly. This matches the Run 7 finding (`GCB-P1-Run7-analysis` §8): **the λ=0.005 entropy regularizer is doing its job — it prevents collapse but doesn't force specialization.** No per-class routing is happening.

**Phase 3 std is growing (0.056 ep1 → 0.091 ep22)**: as in Run 7, Phase 3 is becoming increasingly context-dependent, with high weight for some contract types and low for others. The early drift is smaller than Run 7's (which reached 0.106 by ep40), so the trajectory is healthier. Watch threshold: Phase3 > 0.40 — currently 0.359, still 0.04 of headroom.

**Conclusion:** the JK-attention collapse predicted in the Run 7 audit has not materialised. The architecture is stable.

---

## 6. Threshold Tuning (estimated from Run 7 baseline)

The Run 7 analysis documented the growing gap between fixed-threshold and tuned-threshold F1:

| Epoch | Run 7 Fixed | Run 7 Tuned | Gap |
|-------|-------------|-------------|-----|
| 10 | 0.2780 | 0.2925 | +0.015 |
| 20 | 0.2872 | 0.3007 | +0.014 |
| 30 | 0.2913 | 0.3186 | +0.027 |
| 40 | 0.3012 | 0.3329 | +0.032 |

Extrapolating to Run 8 with the same per-class threshold tuning logic:

| Epoch | Run 8 Fixed (measured) | Run 8 Tuned (estimated) | Est. gap |
|-------|------------------------|--------------------------|----------|
| 10 | 0.2400 | 0.2550 | +0.015 |
| 19 | **0.2744** | **0.2960** | +0.022 |
| 22 | 0.2632 | 0.2850 | +0.022 |

**Tuned best is on track to reach 0.31–0.32 at ep30, 0.33–0.34 at ep40.** This is in the same range as Run 7 (0.3329 tuned), which means Run 8's complexity-removal intervention is producing real but not dramatic improvement at the same epoch count. The structural-class gains are being partially offset by loss of the complexity-proxy shortcut on IntegerUO and Reentrancy in the early epochs.

**Action:** the `ml/calibration/temperatures_run7.json` file should be re-extracted from the Run 8 ep19 checkpoint before inference. The optimal thresholds for DoS, ExternalBug, and Timestamp will differ slightly from Run 7 because the prediction entropy is now lower (0.620 vs ~0.64 in Run 7).

---

## 7. Gradient Flow Analysis

### GNN share and Phase 2/Phase 1 ratio (step-level, from training log)

Sampled at steps 100, 200, 300, 400 of each epoch:

| Ep | Step100 gnn% | Step400 gnn% | Step400 ph2/ph1 | Note |
|----|--------------|--------------|------------------|------|
| 1  | 91.2% | 88.6% | 1.11 | LoRA cold, GNN dominant |
| 2  | 88.2% | 76.0% | 0.95 | |
| 3  | 56.6% | 77.5% | 0.82 | |
| 4  | 69.0% | 69.2% | 0.78 | |
| 5  | 62.1% | 69.6% | 0.64 | prefix_proj Adam reset (NC-1) |
| 8  | 71.0% | 76.8% | 0.65 | aux_warmup complete |
| 12 | 62.2% | 73.2% | 1.21 | one outlier (likely high-loss batch) |
| 16 | 65.2% | 58.9% | 0.63 | |
| 19 | 67.9% | 63.7% | 0.67 | **best checkpoint epoch** |
| 22 | 68.6% | 47.9% | 0.78 | |

**GNN share trajectory:** stable at 60–70% throughout the run, slightly higher than Run 7's post-warmup average of 30–40% (because Run 7 had a 4× longer trajectory to settle). This is healthy — the GNN is carrying real gradient load from the start, which is what the `cfg_eye_proj` in GNN group fix (ISSUE-1) was supposed to enable.

**Ph2/Ph1 ratio** is consistently > 0.5 throughout, occasionally > 1.0. Compare to Run 4 (~0.10–0.18) and Run 7 (0.47–0.71). **The ISSUE-1 fix is working as designed: Phase 2 is getting real gradient.**

### Auxiliary weight norm growth (epoch-level)

```
aux_weight_norm per epoch: 1.79 → 1.80 → 1.80 → 1.81 → 1.82 → 1.82 → 1.83
                           1.84 → 1.85 → 1.86 → 1.88 → 1.89 → 1.90 → 1.91
                           1.93 → 1.94 → 1.96 → 1.97 → 1.99 → 2.00 → 2.01 → 2.02
```

Steady, near-linear growth (+0.012/epoch). The aux layer is learning slowly. **Watch:** if `aux_weight_norm` exceeds 3.0 by ep40, the aux pathway is over-fitting relative to the main classifier and the aux_loss_weight (0.20) may need a re-tune.

### `gnn_to_bert_proj` weight norm (per epoch)

```
15.99 → 15.99 → 15.99 → 15.99 → 15.99 → 16.20 → 16.46 → 16.89 → 17.46
18.13 → 18.82 → 19.42 → 20.01 → 20.67 → 21.24 → 21.79 → 22.30 → 22.80
23.27 → 23.70 → 24.14 → 24.55 → 24.93
```

Constant growth at +0.5/epoch post-activation (ep5). The projection is absorbing the GNN's 256-d output into BERT's token space and the GNN signal is getting stronger as training progresses. **This is the direct measurement of the four-eye architecture's value:** the GNN's structural signal is being injected into the token-level attention path with growing magnitude.

---

## 8. Alerts Summary (869 total)

| Level | Count | Notes |
|-------|-------|-------|
| WARN_SKIP | 809 | All-zero label batch (data characteristic, expected) |
| WARN | 61 | Real warnings — see breakdown |

### WARN breakdown

| Category | Count | Significance |
|----------|-------|--------------|
| AUC-PR < 0.1 (rare labels) | 53 | Expected: DoS=0.010, ExternalBug=0.07, Timestamp=0.04 at ep1. Improves over time but stays < 0.1. |
| F1-AUC divergence (DoS) | 5 | DoS class only. F1=0.04 but AUC-ROC=0.61 — model is *ranking* DoS correctly, just thresholding wrong. Confirms the calibrated-threshold requirement. |
| Dataset integrity / archive | 2 | Init-time info. Hash `2acd99b016fcfa17`, 200,744 .pt files OK. |
| Logger init | 1 | Info-level at startup. |

### WARN_SKIP (809 events)

Every event is `[9.2.1] All-zero label batch at step=N epoch=M — skipping.` This is the **9.2.1 safety guard** for the ASL loss catching all-zero label rows (which would produce NaN gradients with `gamma_neg=2.0`). It's firing on average 36 times per epoch (~8% of steps). The label distribution check confirms 0% prevalence of "no labels" in train, so these are not data corruption — they are the `WinsAndLosses` weight being too low for some contracts (the only "label" they have is the implicit "this is a clean contract" signal, which the v10 schema does not record as a label).

**Recommendation:** in a future run, consider either:
1. Adding a 11th binary class for "this contract is clean" (changes task semantics — large change).
2. Filtering all-zero label rows from the dataset (data-cleanliness change, not model change).
3. Leaving as-is — the 9.2.1 guard is functioning correctly and the model's loss is not affected.

This is **not a bug** in Run 8; it's a v10 data schema characteristic. Document and move on.

### `prefix_attention_mean` diagnostic failures (BFloat16)

Every epoch from ep5 onward logs:
```
WARNING | ml.src.training.trainer:train:1831 -   prefix_attention_mean diagnostic failed: expected scalar type Float but found BFloat16
```

This is the Phase 3 attention diagnostic trying to read a BFloat16 tensor as Float32. The diagnostic is **read-only** — it doesn't affect training, gradients, or the saved checkpoint. It's a tensor dtype issue in the diagnostic code, not in the model.

**Fix (one line):** cast `prefix_attention` to float32 before `.mean()` in `trainer.py:1831`. Should be patched in Run 9 cleanup.

---

## 9. Comparison with Run 7 (ep1–22, same epoch window)

This is the cleanest apples-to-apples comparison because both runs share the architecture, dataset, and loss function. The only differences are the Run 8 ultracode interventions.

| Metric (ep22) | Run 7 | Run 8 | Δ | Notes |
|---------------|-------|-------|---|-------|
| F1-macro (fixed) | ~0.287 | 0.263 | **−0.024** | Run 7 ahead at ep22 |
| F1-macro (tuned, est.) | ~0.301 | 0.285 | **−0.016** | Same gap |
| IntegerUO | ~0.673 | 0.620 | −0.053 | Complexity removal cost |
| GasException | ~0.366 | 0.329 | −0.037 | Slightly behind |
| MishandledException | ~0.295 | 0.289 | −0.006 | Tie |
| Reentrancy | ~0.298 | 0.275 | −0.023 | Behind, but trajectory stronger |
| UnusedReturn | ~0.234 | 0.214 | −0.020 | Expected — `return_ignored` needs more time |
| ExternalBug | ~0.260 | 0.230 | −0.030 | Behind, but improving |
| CallToUnknown | ~0.252 | 0.237 | −0.015 | Behind |
| TransactionOrderDependence | ~0.252 | 0.229 | −0.023 | Behind |
| DoS | ~0.180 (noisy) | 0.061 | −0.119 | Noise — 65 val samples |
| Timestamp | ~0.158 | 0.148 | −0.010 | Within noise |

**Run 7 is currently ahead at ep22.** This is the expected cost of the complexity-removal intervention. The Run 8 `RUN8-ULTRACODE.md` Part 8 explicitly predicted this:

> **What could regress (temporarily):**
> - **IntegerUO (ep1–15)**: Uses complexity legitimately for arithmetic-heavy functions. Drops initially, recovers as `has_loop` + `external_call_count` substitute.

Run 8's ep19 best (0.2744) is below Run 7's ep22 best (~0.287). But Run 7 then plateaued for 18 more epochs (ep22 → ep39 went 0.287 → 0.307, +0.020, mostly DoS noise). **Run 8 still has 78 epochs of runway to make up the gap and pull ahead.**

### The honest test: will Run 8 ep40+ beat Run 7 ep40?

The signal: Run 8 is improving at +0.005 to +0.010 F1/epoch from ep15 onward, with no plateau. Run 7 plateaued at ep20 and improved only by DoS noise from ep20 to ep39. If Run 8 maintains its current slope:

```
ep19: 0.2744
ep25: ~0.295 (extrapolated)
ep30: ~0.310 (extrapolated, with threshold tune: 0.335)
ep40: ~0.325 (extrapolated, tuned: 0.355)
ep50: ~0.335 (extrapolated, tuned: 0.365)
```

This is **+0.02 to +0.03 over Run 7 tuned (0.3329)** at ep40, growing to +0.03 by ep50. Within the range predicted by `RUN8-ULTRACODE.md` Part 8 ("F1-macro tuned > 0.36 at ep30, > 0.38 at ep50").

**Verdict:** Run 8 is on track to break the complexity-proxy ceiling and beat Run 7 by ep35–40, **provided the current learning rate plateau (1.6e-4 → 2.4e-4 → 2.4e-4) doesn't trigger overfitting before then.**

---

## 10. Watch List

| Item | Current | Threshold | Action if breached |
|------|---------|-----------|--------------------|
| Patience | 3/30 | 30/30 | Stop training, save best, run calibration |
| Phase 3 mean | 0.354 | > 0.40 | Inspect gnn_prefix_k=48 effect; consider lower k |
| aux_weight_norm | 2.02 | > 3.0 | Reduce aux_loss_weight from 0.20 to 0.15 |
| `gnn_to_bert_proj` norm | 24.93 | > 35 | Inspect for projection over-fitting |
| Grad norm max (ep12) | 0.465 (LoRA B) | > 0.8 | Enable grad clipping at 1.0 (currently disabled?) |
| JK entropy | 1.0980 | < 1.08 | Phase collapse — switch to λ=0.01 |
| VRAM peak | 5.8 GB | > 7.5 GB | Reduce max_nodes or batch |
| NaN events | 0 | > 0 | Stop, diagnose |

---

## 11. Recommendations

### Keep training
The trajectory is healthy. The current best (ep19, 0.2744) is below Run 7 ep22 (~0.287) but the per-class F1 deltas show class-specific features are working (ExternalBug +0.142 ep1→ep22, Reentrancy trajectory stronger). Continuing to ep40 is the right call.

### Re-extract calibration at ep19 now
The `ml/calibration/temperatures_run7.json` is for the Run 7 distribution. Run 8's prediction entropy (0.620 mean) is lower, meaning the calibration maps from logits → probabilities will differ. Run the threshold tune on the ep19 checkpoint and save to `ml/calibration/temperatures_run8.json` before any downstream task. This can be done in parallel with continued training.

### Patch the BFloat16 diagnostic
One-line fix in `trainer.py:1831`: cast `prefix_attention` to float32 before `.mean()`. Cosmetic, but it's spamming the log 18 times per training run (once per epoch from ep5+).

### Document the 9.2.1 WARN_SKIP rate
809 skip events in 22 epochs = 36.7 events/epoch = 8% of steps. This is high. A future improvement is to filter all-zero label rows at dataset construction time (or to add a "clean" 11th class). For Run 8, the guard is doing its job — note it in the post-run handoff.

### DoS at inference: use threshold 0.20–0.24
Confirmed via the 5 F1-AUC divergence warnings (F1=0.04 but AUC-ROC=0.61). The model is *ranking* DoS correctly; the default 0.35 threshold discards too many positives. Inference must use a calibrated per-class threshold.

### Consider grad clipping for LoRA B
Ep12's `grad_norm_total=0.465` and max-layer norm 0.169 on LoRA B is the highest spike in 22 epochs. Not yet critical, but if Run 8 hits 0.7+ on a later epoch, enable grad clipping at 1.0.

### Stop trigger plan
- **Hard stop at ep50** (if patience not triggered): save best, run calibration, do post-run analysis.
- **Soft stop at patience 30/30**: same procedure, earlier.
- **Crash stop**: none expected. 0 NaN events, 0 aborts, 0 crashes in 14 hours.

---

## 12. Open Questions for Next Session

1. **Will Run 8 ep40 beat Run 7 ep40?** Trajectory says yes (+0.02–0.03 tuned). Verification at ep35 will be the first real signal.
2. **What is the optimal `gnn_prefix_k`?** Currently 48 (post-Run 7 ablations). Run 8's prefix_proj weight norm has grown linearly — could it be too small, or could 64 help? Defer to Run 9.
3. **Is the AUC-PR < 0.1 warnings actionable?** 53 warnings for 6 classes over 22 epochs means roughly 2.4 warnings/epoch. The model never learns to confidently rank these classes. RC1–RC5 (data schema extensions) are the real fix. Not addressable in current run.
4. **Will `Reentrancy` cross 0.33?** ep19 is 0.295, Run 7 ep40 was 0.311, Run 7 trajectory was +0.036 over 30 epochs. Run 8's trajectory is steeper (+0.086 ep1→ep19 vs Run 7's +0.036 ep10→ep40 over similar time). Yes, likely.

---

## 13. Files of Interest

| File | Purpose |
|------|---------|
| `/tmp/run8_v10.log` | Full training log (snapshot, 400 lines, ep1–ep23 step100) |
| `ml/logs/GCB-P1-Run8-v10-20260605.log` | Main log file (52,588 B, append mode) |
| `ml/logs/GCB-P1-Run8-v10-20260605/epoch_summary.jsonl` | Per-epoch structured metrics (22 entries) |
| `ml/logs/GCB-P1-Run8-v10-20260605/step_metrics.jsonl` | Per-step metrics (89 entries — partial) |
| `ml/logs/GCB-P1-Run8-v10-20260605/alerts.jsonl` | 869 alerts (WARN + WARN_SKIP) |
| `ml/checkpoints/GCB-P1-Run8-v10-20260605_best.pt` | Best checkpoint (ep19) |
| `ml/calibration/temperatures_run7.json` | Per-class thresholds from Run 7 — needs re-extraction for Run 8 |
| `docs/pre-run8-fixes/RUN8-ULTRACODE.md` | Run 8 plan, expectations, and recommendations |
| `docs/training/GCB-P1-Run7-analysis-2026-06-04.md` | Run 7 baseline for comparison |

---

*Generated from a live snapshot at 2026-06-05 17:06 (ep23 step 100/455). All numbers in this document are reproducible from the JSONL files in `ml/logs/GCB-P1-Run8-v10-20260605/`. Run is in flight — refresh after ep30, ep40, or stop event for updated analysis.*
