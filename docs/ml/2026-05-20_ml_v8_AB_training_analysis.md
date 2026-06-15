# v8.0-AB Training Analysis
**Run name:** `sentinel-v8` (`v8.0-AB`)  
**Log:** `ml/logs/v8.0-AB-20260520.log`  
**Last updated:** 2026-05-20 (COMPLETE — killed ep37, patience 8/30)  
**Best checkpoint:** `ml/checkpoints/v8.0-AB-20260520_best.pt` — Epoch 29, F1-macro=0.2621  
**Patience:** 8/30 at kill (epoch 37)  
**Kill condition:** EXECUTED — killed at epoch 37 (patience 8/30, 3 epochs before the 10/30 trigger), sufficient evidence accumulated

---

## Log extraction note

The log was launched with `2>&1 | tee`, merging stderr (tqdm) and stdout (loguru). tqdm uses `\r` between updates, so each loguru line is physically on the same byte-line as the preceding tqdm bar. Normal `grep` returns fused lines; tqdm text floods the display.

**To extract clean data from this log or any similarly polluted future log:**
```bash
tr '\r' '\n' < ml/logs/v8.0-AB-20260520.log | grep "eyes:"
tr '\r' '\n' < ml/logs/v8.0-AB-20260520.log | grep "JK attention"
tr '\r' '\n' < ml/logs/v8.0-AB-20260520.log | grep "Epoch.*F1-macro"
```

**Fix applied (2026-05-20):** `trainer.py` lines 411 and 504 now pass `disable=not sys.stdout.isatty()` to tqdm. Future runs via `nohup ... | tee` will produce clean logs.

---

## Complete per-epoch table — epochs 1–37

*Step columns = values from Step 400/455 proxy (last logged step before epoch end). Fsd = fused.*

| Ep | F1 | ★ | Ep Loss | Hamming | Step loss | GNN-eye | TF-eye | Fsd-eye | GNN-g | TF-g | Fsd-g | GNN% | JK-P1 | JK-P2 | JK-P3 |
|----|----|---|---------|---------|-----------|---------|--------|---------|-------|------|-------|------|-------|-------|-------|
| 1 | 0.1502 | ★ | 0.179 | 0.913 | 0.172 | 0.747 | 0.689 | 0.700 | 0.045 | 0.044 | 0.012 | 70.0 | 0.185 | 0.329 | 0.486 |
| 2 | 0.1479 | | 0.240 | 0.727 | 0.165 | 0.553 | 0.647 | 0.672 | 0.031 | 0.029 | 0.013 | 69.5 | 0.098 | 0.358 | 0.544 |
| 3 | 0.1449 | | 0.282 | 0.611 | 0.164 | 0.514 | 0.508 | 0.501 | 0.029 | 0.023 | 0.028 | 62.1 | 0.084 | 0.362 | 0.554 |
| 4 | 0.1550 | ★ | 0.334 | 0.614 | 0.163 | 0.499 | 0.520 | 0.492 | 0.031 | 0.022 | 0.027 | 66.4 | 0.083 | 0.356 | 0.560 |
| 5 | 0.1678 | ★ | 0.388 | 0.594 | 0.160 | 0.484 | 0.538 | 0.478 | 0.041 | 0.027 | 0.028 | 72.1 | 0.081 | 0.352 | 0.567 |
| 6 | 0.2134 | ★ | 0.437 | 0.537 | 0.158 | 0.490 | 0.528 | 0.462 | 0.053 | 0.041 | **0.067** | 56.2 | 0.077 | 0.347 | 0.576 |
| 7 | 0.2095 | | 0.487 | 0.605 | 0.157 | 0.485 | 0.529 | 0.461 | 0.042 | 0.030 | 0.036 | 67.0 | 0.079 | 0.345 | 0.575 |
| 8 | 0.2194 | ★ | 0.535 | 0.541 | 0.152 | 0.477 | 0.530 | 0.456 | 0.042 | 0.027 | 0.036 | 68.1 | 0.081 | 0.346 | 0.573 |
| 9 | 0.2272 | ★ | 0.583 | 0.507 | 0.149 | 0.463 | 0.521 | 0.449 | 0.041 | 0.032 | 0.048 | 57.9 | 0.086 | 0.340 | 0.575 |
| 10 | 0.2395 | ★ | 0.575 | 0.461 | 0.148 | 0.467 | 0.516 | 0.442 | 0.055 | 0.040 | 0.057 | 62.1 | 0.097 | 0.330 | 0.573 |
| 11 | 0.2285 | | 0.574 | 0.515 | 0.148 | 0.460 | 0.517 | 0.442 | 0.040 | 0.031 | 0.038 | 63.1 | 0.121 | 0.323 | 0.555 |
| 12 | 0.2377 | | 0.572 | 0.478 | 0.146 | 0.458 | 0.518 | 0.438 | 0.044 | 0.032 | 0.042 | 64.0 | 0.115 | 0.327 | 0.558 |
| 13 | 0.2362 | | 0.571 | 0.485 | 0.147 | 0.460 | 0.519 | 0.441 | 0.042 | 0.032 | 0.042 | 62.3 | **0.138** | 0.317 | 0.545 |
| 14 | 0.2405 | ★ | 0.568 | 0.470 | 0.145 | 0.459 | 0.515 | 0.435 | 0.053 | 0.046 | **0.070** | 53.6 | 0.099 | 0.316 | 0.585 |
| 15 | 0.2416 | ★ | 0.566 | 0.466 | 0.145 | 0.452 | 0.515 | 0.437 | 0.034 | 0.029 | 0.041 | 56.3 | 0.095 | 0.316 | 0.589 |
| 16 | 0.2398 | | 0.568 | 0.471 | 0.145 | 0.462 | 0.514 | 0.436 | 0.057 | 0.045 | **0.070** | 57.0 | 0.112 | 0.302 | 0.586 |
| 17 | 0.2403 | | 0.564 | 0.474 | 0.144 | 0.453 | 0.511 | 0.435 | 0.053 | 0.038 | 0.051 | 64.2 | 0.075 | 0.315 | 0.610 |
| 18 | 0.2493 | ★ | 0.563 | 0.443 | 0.144 | 0.455 | 0.503 | 0.433 | 0.049 | 0.035 | 0.038 | 69.2 | 0.079 | 0.314 | 0.607 |
| 19 | 0.2468 | | 0.563 | 0.453 | 0.144 | 0.453 | 0.505 | 0.436 | 0.056 | 0.037 | 0.051 | 66.7 | 0.080 | 0.307 | 0.614 |
| 20 | 0.2581 | ★ | 0.561 | 0.425 | 0.143 | 0.455 | 0.503 | 0.432 | 0.055 | 0.043 | **0.073** | 54.2 | 0.082 | 0.310 | 0.607 |
| 21 | 0.2449 | | 0.561 | 0.465 | 0.144 | 0.457 | 0.504 | 0.434 | 0.044 | 0.031 | 0.041 | 65.3 | 0.072 | 0.294 | 0.634 |
| 22 | 0.2593 | ★ | 0.560 | 0.418 | 0.143 | 0.455 | 0.504 | 0.432 | 0.041 | 0.039 | 0.052 | 53.4 | 0.075 | 0.290 | 0.635 |
| 23 | 0.2381 | | 0.556 | 0.489 | 0.142 | 0.443 | 0.499 | 0.429 | 0.049 | 0.033 | 0.047 | 64.4 | 0.072 | 0.289 | 0.639 |
| 24 | 0.2453 | | 0.558 | 0.459 | 0.144 | 0.451 | 0.502 | 0.433 | 0.053 | 0.033 | 0.053 | 65.2 | 0.071 | 0.284 | 0.645 |
| 25 | 0.2588 | | 0.557 | 0.424 | 0.143 | 0.452 | 0.497 | 0.429 | 0.040 | 0.035 | 0.052 | 53.6 | 0.074 | 0.272 | 0.654 |
| 26 | 0.2567 | | 0.556 | 0.431 | 0.142 | 0.449 | 0.497 | 0.429 | 0.041 | 0.035 | 0.052 | 55.0 | 0.072 | 0.263 | 0.665 |
| 27 | 0.2544 | | 0.557 | 0.438 | 0.142 | 0.445 | 0.500 | 0.426 | 0.042 | 0.042 | 0.049 | 54.1 | 0.071 | 0.264 | 0.666 |
| 28 | 0.2449 | | 0.556 | 0.468 | 0.142 | 0.449 | 0.501 | 0.429 | 0.037 | 0.033 | 0.043 | 56.9 | 0.067 | 0.246 | 0.687 |
| **29** | **0.2621** | **★** | **0.554** | **0.420** | **0.142** | **0.450** | **0.500** | **0.429** | **0.051** | **0.041** | **0.052** | **61.2** | **0.063** | **0.243** | **0.694** |
| 30 | 0.2580 | | 0.553 | 0.429 | 0.141 | 0.442 | 0.498 | 0.424 | 0.043 | 0.038 | 0.055 | 54.4 | 0.067 | 0.245 | 0.689 |
| 31 | 0.2567 | | 0.556 | 0.429 | 0.141 | 0.447 | 0.499 | 0.425 | 0.036 | 0.041 | 0.061 | **43.6** | 0.064 | 0.244 | 0.692 |
| 32 | 0.2509 | | 0.553 | 0.447 | 0.142 | 0.444 | 0.498 | 0.425 | 0.043 | 0.035 | 0.043 | 61.9 | 0.069 | 0.246 | 0.685 |
| 33 | 0.2564 | | 0.554 | 0.429 | 0.142 | 0.448 | 0.499 | 0.425 | 0.049 | 0.043 | 0.056 | 57.6 | 0.060 | 0.230 | **0.711** |
| 34 | 0.2605 | | 0.554 | 0.418 | 0.142 | 0.449 | 0.500 | 0.427 | 0.042 | 0.041 | 0.056 | 52.2 | 0.059 | 0.233 | 0.708 |
| 35 | 0.2607 | | 0.5515 | 0.4130 | 0.1419 | 0.4488 | 0.4988 | 0.4258 | 0.039 | 0.034 | 0.042 | 58.3 | 0.059 | 0.228 | 0.713 |
| 36 | 0.2588 | | 0.5518 | 0.4225 | 0.1402 | 0.4435 | 0.4946 | 0.4210 | 0.042 | 0.035 | 0.050 | 56.9 | 0.056 | 0.217 | 0.728 |
| 37 | 0.2536 | | 0.5510 | 0.4412 | 0.1418 | 0.4459 | 0.4971 | 0.4271 | 0.046 | 0.034 | 0.047 | 61.7 | 0.052 | 0.204 | 0.744 |

---

## Training phases

### Phase 1 — Aux warmup, ep1–8 | F1: 0.150→0.219

Epoch loss rises 0.179→0.535 as aux loss weight ramps 0→0.3× linearly. Step criterion loss drops 0.172→0.152 — the model is genuinely learning, but the increasing aux weight makes the epoch total rise. This is expected and healthy.

Hamming crashes 0.913→0.541: the model abandons the all-zeros prediction strategy. At ep1 Hamming=0.913 is trivially achieved by predicting nothing (base negative rate ~90%); by ep8 the model fully commits to positive predictions.

**The ep6 fusion breakthrough (+0.047 F1 jump):** Fused grad spikes to 0.067 — the largest single-step gradient the fusion layer has produced. GNN share drops to 56% as the fusion layer temporarily dominates. The jump from 0.168→0.213 is the biggest single-epoch F1 gain of the entire run. This pattern — fused grad spike, GNN share dip, F1 jump — recurs throughout.

JK weights settle by ep8: P1 ~0.081, P2 ~0.346, P3 ~0.573. The model has already learned to down-weight Phase 1 (structural features) and distribute attention between Phase 2 (CFG/ICFG) and Phase 3 (containment hierarchy).

---

### Phase 2 — Post-warmup rapid gain, ep9–10 | F1: 0.227→0.240

Aux weight fully baked in. Two consecutive new bests. Step loss 0.149→0.148, barely moving — all the improvement is in how the model uses the gradient, not in the gradient magnitude. GNN share 57–62%. This is the cleanest, most productive two-epoch window of the run.

---

### Phase 3 — First plateau, ep11–17 | F1: 0.228–0.242

F1 oscillates without a clear upward trend. Eye losses freeze: GNN 0.458–0.467, TF 0.514–0.519, fused 0.435–0.442. Nothing is improving in isolation — the model is searching the loss landscape without finding a consistent direction.

**The P1 spike at ep13 (0.138):** Phase 1 weight briefly nearly doubles (from 0.075–0.097 to 0.138), the model trying to escape the plateau by re-weighting structural/CONTAINS features. It produces a modest new best at ep14 (+0.003) via a fused grad spike (0.070). By ep16 P1 is back at 0.112, then collapses to 0.075 at ep17 — the structural escape route is exhausted.

**Fused grad spikes at ep14 (0.070) and ep16 (0.070)** both correspond to new bests or near-bests. The fusion layer is the only component that can produce breakthroughs; the individual eyes are frozen.

---

### Phase 4 — Second climb, ep18–22 | F1: 0.249→0.259

Five-epoch productive window with a new best in three of them. JK P2 slides 0.314→0.290 while P3 grows 0.607→0.635 — the model is consolidating trust in the containment hierarchy. This consolidation is what enables the climb: once the model settles on a stable weighting, the fusion layer can refine its combination rather than searching.

The fused grad spike at ep20 (0.073 — the highest of the entire run to this point) drives the F1 jump from 0.249→0.258. GNN share dips to 54.2% on the same epoch — the transformer+fused pair is driving most of the update.

---

### Phase 5 — Second plateau, ep23–28 | F1: 0.238–0.259

Same ceiling, longer dip. F1 oscillates ~0.02 amplitude. JK Phase 2 declines faster here (0.289→0.246), Phase 3 climbing toward 0.690. The step loss is almost completely flat at 0.142. The model is searching on a very shallow loss surface.

The ep23 dip to 0.238 (largest single-epoch drop of the run, −0.021) suggests the landscape at the new plateau is noisier — more val set variance — despite the model not having changed structurally.

---

### Phase 6 — Second breakthrough, ep29 | F1: 0.2621 ★ new all-time best

F1 jumps +0.017 from ep28 (0.245) to ep29 (0.2621). The largest single-epoch gain since ep6 (+0.047). What made it happen:

- Step loss: 0.142 — identical to surrounding epochs. No loss improvement.
- Eye losses: 0.450/0.500/0.429 — no dramatic change.
- JK: P1=0.063, P2=0.243, P3=0.694 — continuation of the slow trend.
- GNN share: 61.2% — slightly higher than surrounding epochs.
- Fused grad: 0.052 — moderate, no spike.

There is no clear mechanistic cause in the step-400 proxy. The breakthrough appears to be a val-set alignment event — the model's current weighting happens to match the val distribution better on this epoch. This is consistent with the observed oscillation pattern: the model circles the true ceiling and occasionally pierces it.

---

### Phase 7 — Current plateau, ep30–34 | F1: 0.251–0.261 | Patience=5/30

Post-breakthrough settling. F1 hovers 0.251–0.261, unable to exceed 0.2621.

**Critical new development: JK Phase 3 hits 0.711 at ep33** — the highest Phase 3 weight of the entire run. Phase 2 simultaneously hits 0.230 — the lowest. This is an acceleration of the Phase 3 consolidation trend:

```
JK Phase 3 trajectory:
ep22: 0.635 → ep25: 0.654 → ep29: 0.694 → ep33: 0.711
Rate: +0.019/3ep → +0.017/3ep → +0.040/4ep ← accelerating
```

The model is increasingly deciding that the REVERSE_CONTAINS hierarchy (which contract/function a node belongs to) is the dominant signal — and deprioritising the ICFG/DEF_USE edges in Phase 2. By ep33 Phase 2 is almost as low as v7's final value was (v7 final: 0.182; v8 ep33: 0.230).

**GNN share hits 43.6% at ep31** — lowest of the entire run. The transformer+fused pair was doing 56% of gradient work. This is the most extreme transfer-of-gradient from GNN to fusion yet. Fused grad at ep31 was 0.061 (elevated) — the fusion layer is actively searching. Ep34 GNN share down to 52.2%, fused grad still 0.056. This pattern suggests the fusion layer is in an extended learning phase.

**Ep34 F1=0.2605** — the second-highest value ever (after ep29's 0.2621). The model is back in the near-ceiling zone. If the oscillation holds, the next spike above 0.2621 is plausible within the next 5–8 epochs.

---

## Signal trajectories summary

### Step criterion loss
```
ep1: 0.172 → ep10: 0.148 → ep20: 0.143 → ep29: 0.142 → ep34: 0.142
```
Dropped 17% total. Now effectively flat — at the bottom of the learning rate curve.

### Eye losses (GNN / TF / Fused)
```
ep1:  0.747 / 0.689 / 0.700   ← start
ep10: 0.467 / 0.516 / 0.442   ← post-aux-warmup plateau onset
ep22: 0.455 / 0.504 / 0.432   ← best F1 at that time
ep34: 0.449 / 0.500 / 0.427   ← current
```
All three have plateaued since ep10 and are only ticking down fractionally. Fused leads (lowest loss), GNN second, TF trails. The gap between fused and TF (0.427 vs 0.500) tells you the fusion is contributing real cross-modal signal — it's not just amplifying what either individual eye sees.

### JK attention weights
```
           P1      P2      P3
ep1:      0.185   0.329   0.486   ← starts near-uniform
ep3:      0.084   0.362   0.554   ← P2 peaks (new ICFG edges attract weight)
ep13:     0.138   0.317   0.545   ← P1 anomaly (plateau escape attempt)
ep22:     0.075   0.290   0.635   ← second climb consolidation
ep29:     0.063   0.243   0.694   ← best F1 epoch
ep33:     0.060   0.230   0.711   ← P3 peak, P2 at run low
ep34:     0.059   0.233   0.708   ← slight P2 recovery
```

**The P2 story:** Starts at 0.329 (highest, because the model initially finds the new ICFG/DEF_USE edges novel and informative). Slowly declines as the model realises the intra-function DEF_USE has limited reach and CALL_ENTRY/RETURN_TO are sparse in most contracts. Currently at 0.233 — approaching v7's final value of 0.182. The gap remaining (~0.05) represents the residual ICFG/DEF_USE contribution above the v7 baseline.

**The P3 story:** 0.486→0.711. The REVERSE_CONTAINS hierarchy (node→function→contract) is becoming the dominant signal. The model has learned that "which contract this node belongs to" predicts vulnerability class more reliably than any intra-function or cross-function edge pattern currently available.

### Fused grad norms and F1 jumps
Every significant F1 gain correlates with an elevated fused grad:
```
ep6:  fused=0.067 → F1 +0.047 (0.168→0.213) ← biggest jump
ep14: fused=0.070 → F1 +0.003 (0.240→0.241) ← modest
ep16: fused=0.070 → F1 flat  (minor)
ep20: fused=0.073 → F1 +0.009 (0.249→0.258) ← second largest
ep30: fused=0.055 → minor
ep31: fused=0.061 → potential precursor
ep33: fused=0.056 → potential precursor
ep34: fused=0.056 → potential precursor
```
The fusion layer is elevated (0.055–0.061) in ep30–34 without yet spiking. This suggests it is building toward a breakout but has not found the direction yet.

### GNN share
```
Range: 43.6% (ep31, run minimum) — 72.1% (ep5)
Recent: ep29=61%, ep30=54%, ep31=44%, ep32=62%, ep33=58%, ep34=52%
```
GNN share below 55% consistently since ep25 (with ep32 exception). The transformer+fused combination is increasingly doing more of the gradient work. This is not necessarily bad — it may reflect the GNN having stabilised its representation while fusion continues learning.

---

## Comparison with v7.0 (reference)

| Metric | v7.0 final (ep33) | v8-AB ep34 | Delta |
|--------|------------------|------------|-------|
| Best F1 | 0.2651 (ep23) | **0.2621 (ep29)** | −0.003 |
| Step loss at similar epoch | ~0.135 | 0.142 | +0.007 |
| JK Phase 2 | 0.182 | 0.233 | **+0.051** |
| JK Phase 3 | 0.768 | 0.708 | −0.060 |
| Eye gap (GNN−Fused) | ~0.017 | 0.022 | +0.005 |
| Plateau onset | ep10 | ep10 | identical |
| Oscillation amplitude | ~0.02 | ~0.02 | identical |

v8 is 0.003 below v7's best with 34 epochs trained vs v7's 33-epoch kill point. The gap is within the oscillation noise band — both models have the same structural ceiling. v8's Phase 2 carrying 0.051 more JK weight than v7 is the confirmed ICFG contribution, but it is not yet translating to F1 gains beyond the oscillation.

---

## Final state at kill

**Killed at epoch 37 (patience 8/30), 3 epochs before the planned patience=10 trigger.**

**Rationale for early kill:** JK Phase 2 declined to 0.204 (approaching v7 final 0.182), Phase 3 accelerated to 0.744 (strongest of the entire run), fused grad subdued (0.042–0.050) across eps 35–37 with no spike forming, and 8 consecutive epochs without improvement. Sufficient evidence accumulated to conclude the run had converged.

**Final best:** F1=0.2621 at epoch 29. v7 best was 0.2651 at epoch 23 — gap 0.0030, within the oscillation noise band.

**JK trajectory at kill:** P1=0.052, P2=0.204, P3=0.744 — Phase 2 approaching v7's final 0.182, model structurally converging to the same basin as v7. The ICFG/DEF_USE edges contributed a measurable P2 elevation throughout but did not translate to F1 gains beyond the structural ceiling shared with v7.

---

## Next steps (in order)

1. **Comparison analysis — COMPLETE.** See `docs/ml/v8-vs-v7-comparison-results.md`.
2. **PLAN-3A** — `--phase2-edge-types 6 8 9` (ICFG-only, drop DEF_USE) — the ablation that will tell us whether DEF_USE is helping or hurting Phase 2
3. **PLAN-3B** — `--phase2-edge-types 6 10` (DFG-only, drop CALL_ENTRY/RETURN_TO)

```bash
# PLAN-3A launch command (after comparison is done):
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. nohup python ml/scripts/train.py \
    --run-name v8.0-A-$(date +%Y%m%d) \
    --experiment-name sentinel-v8 \
    --phase2-edge-types 6 8 9 \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --epochs 100 --gradient-accumulation-steps 8 \
    > ml/logs/v8.0-A-$(date +%Y%m%d).log 2>&1 &
```
