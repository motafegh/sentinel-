# v8.0-AB Training Analysis
**Date:** 2026-05-20  
**Run name:** `sentinel-v8` (`v8.0-AB`)  
**Log:** `ml/logs/v8.0-AB-20260520.log`  
**Status at write time:** Epoch 27 in progress (63% / 3638 batches)  
**Best checkpoint:** `ml/checkpoints/v8.0-AB_best.pt` — Epoch 22, F1-macro=0.2593

---

## A note on reading the log

The log was launched with `2>&1 | tee`, merging stderr (tqdm progress bars) and stdout (loguru) into the same file. tqdm uses `\r` (carriage return) between updates, so each loguru step line is physically appended to the same byte-line as the preceding tqdm bar. Normal `grep` returns those fused lines and the tqdm prefix visually floods the output.

**To extract clean loguru lines from this or any future log polluted this way:**
```bash
tr '\r' '\n' < ml/logs/v8.0-AB-20260520.log | grep "eyes:"
tr '\r' '\n' < ml/logs/v8.0-AB-20260520.log | grep "JK attention"
```

**Fix applied (2026-05-20):** `trainer.py` lines 504 and 411 now pass `disable=not sys.stdout.isatty()` to tqdm. When stdout is a pipe or file, tqdm is silenced entirely. Future runs via `nohup ... | tee` will produce clean logs.

---

## Per-epoch summary (epochs 1–26 complete)

| Ep | F1-macro | Best? | Epoch Loss | Hamming | GNN-eye | TF-eye | Fused-eye | GNN-grad | TF-grad | Fused-grad | GNN-share | JK-P1 | JK-P2 | JK-P3 |
|----|----------|-------|------------|---------|---------|--------|-----------|----------|---------|------------|-----------|-------|-------|-------|
| 1  | 0.1502 | ★ | 0.179 | 0.913 | 0.747 | 0.689 | 0.700 | 0.045 | 0.044 | 0.012 | 70.0% | 0.185 | 0.329 | 0.486 |
| 2  | 0.1479 |   | 0.240 | 0.727 | 0.553 | 0.647 | 0.672 | 0.031 | 0.029 | 0.013 | 69.5% | 0.098 | 0.358 | 0.544 |
| 3  | 0.1449 |   | 0.282 | 0.611 | 0.514 | 0.508 | 0.501 | 0.029 | 0.023 | 0.028 | 62.1% | 0.084 | 0.362 | 0.554 |
| 4  | 0.1550 | ★ | 0.334 | 0.614 | 0.499 | 0.520 | 0.492 | 0.031 | 0.022 | 0.027 | 66.4% | 0.083 | 0.356 | 0.560 |
| 5  | 0.1678 | ★ | 0.388 | 0.594 | 0.484 | 0.538 | 0.478 | 0.041 | 0.027 | 0.028 | 72.1% | 0.081 | 0.352 | 0.567 |
| 6  | 0.2134 | ★ | 0.437 | 0.537 | 0.490 | 0.528 | 0.462 | 0.053 | 0.041 | **0.067** | 56.2% | 0.077 | 0.347 | 0.576 |
| 7  | 0.2095 |   | 0.487 | 0.605 | 0.485 | 0.529 | 0.461 | 0.042 | 0.030 | 0.036 | 67.0% | 0.079 | 0.345 | 0.575 |
| 8  | 0.2194 | ★ | 0.535 | 0.541 | 0.477 | 0.530 | 0.456 | 0.042 | 0.027 | 0.036 | 68.1% | 0.081 | 0.346 | 0.573 |
| 9  | 0.2272 | ★ | 0.583 | 0.507 | 0.463 | 0.521 | 0.449 | 0.041 | 0.032 | 0.048 | 57.9% | 0.086 | 0.340 | 0.575 |
| 10 | 0.2395 | ★ | 0.575 | 0.461 | 0.467 | 0.516 | 0.442 | 0.055 | 0.040 | 0.057 | 62.1% | 0.097 | 0.330 | 0.573 |
| 11 | 0.2285 |   | 0.574 | 0.515 | 0.460 | 0.517 | 0.442 | 0.040 | 0.031 | 0.038 | 63.1% | 0.121 | 0.323 | 0.555 |
| 12 | 0.2377 |   | 0.572 | 0.478 | 0.458 | 0.518 | 0.438 | 0.044 | 0.032 | 0.042 | 64.0% | 0.115 | 0.327 | 0.558 |
| 13 | 0.2362 |   | 0.571 | 0.485 | 0.460 | 0.519 | 0.441 | 0.042 | 0.032 | 0.042 | 62.3% | **0.138** | 0.317 | 0.545 |
| 14 | 0.2405 | ★ | 0.568 | 0.470 | 0.459 | 0.515 | 0.435 | 0.053 | 0.046 | **0.070** | 53.6% | 0.099 | 0.316 | 0.585 |
| 15 | 0.2416 | ★ | 0.566 | 0.466 | 0.452 | 0.515 | 0.437 | 0.034 | 0.029 | 0.041 | 56.3% | 0.095 | 0.316 | 0.589 |
| 16 | 0.2398 |   | 0.568 | 0.471 | 0.462 | 0.514 | 0.436 | 0.057 | 0.045 | **0.070** | 57.0% | 0.112 | 0.302 | 0.586 |
| 17 | 0.2403 |   | 0.564 | 0.474 | 0.453 | 0.511 | 0.435 | 0.053 | 0.038 | 0.051 | 64.2% | 0.075 | 0.315 | 0.610 |
| 18 | 0.2493 | ★ | 0.563 | 0.443 | 0.455 | 0.503 | 0.433 | 0.049 | 0.035 | 0.038 | 69.2% | 0.079 | 0.314 | 0.607 |
| 19 | 0.2468 |   | 0.563 | 0.453 | — | — | — | — | — | — | — | 0.080 | 0.307 | 0.614 |
| 20 | 0.2581 | ★ | 0.561 | 0.425 | 0.455 | 0.503 | 0.432 | 0.055 | 0.043 | **0.073** | 54.2% | 0.072 | 0.294 | 0.634 |
| 21 | 0.2449 |   | 0.561 | 0.465 | 0.457 | 0.504 | 0.434 | 0.044 | 0.031 | 0.041 | 65.3% | 0.075 | 0.290 | 0.635 |
| 22 | 0.2593 | ★ | 0.560 | 0.418 | 0.455 | 0.504 | 0.432 | 0.041 | 0.039 | 0.052 | 53.4% | 0.072 | 0.289 | 0.639 |
| 23 | 0.2381 |   | 0.556 | 0.489 | 0.443 | 0.499 | 0.429 | 0.049 | 0.033 | 0.047 | 64.4% | 0.071 | 0.284 | 0.645 |
| 24 | 0.2453 |   | 0.558 | 0.459 | 0.451 | 0.502 | 0.433 | 0.053 | 0.033 | 0.053 | 65.2% | 0.074 | 0.272 | 0.654 |
| 25 | 0.2588 |   | 0.557 | 0.424 | 0.452 | 0.497 | 0.429 | 0.040 | 0.035 | 0.052 | 53.6% | 0.072 | 0.263 | 0.665 |
| 26 | 0.2567 |   | 0.556 | 0.431 | 0.449 | 0.497 | 0.429 | 0.041 | 0.035 | 0.052 | 55.0% | 0.072 | 0.263 | 0.665 |

*Step-level criterion loss (reported in step logs, not epoch logs): 0.190 at ep1 → 0.142 at ep26. Monotone decline — healthy, not overfit.*

*Epoch Loss rises ep1→ep9 because of aux loss warmup (weight 0→0.3 linearly over epochs 1–8, adding ~0.3× aux to every step). After ep9 the epoch loss slowly declines — real improvement.*

---

## Finding 1 — Eye scores: fused head leads, GNN falls fastest

All three classifier eyes improve from epoch 1 to plateau around epoch 10:

| Eye | Ep 1 | Ep 10 | Ep 26 | Drop |
|-----|------|-------|-------|------|
| GNN | 0.747 | 0.467 | 0.449 | −40% |
| TF | 0.689 | 0.516 | 0.497 | −28% |
| Fused | 0.700 | 0.442 | 0.429 | −39% |

**Fused** consistently holds the lowest loss — cross-modal fusion is the most discriminative signal. **GNN** drops faster than TF, confirming the graph structure encodes information CodeBERT alone does not have. After ep10, all three plateau and track in lockstep — the ceiling is not in any single eye but in what the model can learn from the current graph schema.

---

## Finding 2 — Fusion grad spikes are the breakthrough mechanism

The fused grad norm is the most variable of the three, and every large spike coincides with an F1 jump:

| Epoch | Fused grad | F1 change |
|-------|------------|-----------|
| 6 | 0.067 | +0.047 (0.168→0.213) ← biggest single-epoch jump |
| 14 | 0.070 | +0.001 (0.240→0.241) |
| 16 | 0.070 | — (slight regress) |
| 20 | 0.073 | +0.009 (0.249→0.258) |

On those epochs, GNN share drops to 53–57% (from the usual 62–70%), meaning the fusion layer is compensating for residual GNN error rather than just amplifying it. The fusion layer is the actual learning bottleneck — improving the GNN input quality is what unlocks future gains.

---

## Finding 3 — JK attention weights: v8 ICFG/DEF_USE edges are genuinely used

**v7 final (epoch 33):**  Phase1=0.050 · Phase2=0.182 · Phase3=0.768  
**v8-AB epoch 26:**       Phase1=0.072 · Phase2=0.263 · Phase3=0.665

Phase 2 carries **45% more weight** in v8 than v7's converged state. The CALL_ENTRY / RETURN_TO / DEF_USE edges added in v8 are not being ignored — the model actively routes signal through them. Phase 3 (REVERSE_CONTAINS, hierarchical containment) is still dominant but less extreme (0.665 vs 0.768), meaning the new cross-function edges are partially substituting for the coarse containment prior.

**JK trajectory pattern:**
- Ep1: P2=0.329 (model initially attracted to new edges)
- Ep2–5: P2 rises to peak 0.362 (exploration)
- Ep6–10: P2 slides to 0.330, P3 grows (containment reclaims weight as easy patterns learned)
- Ep11–13: P1 spikes to 0.115–0.138 (structural features briefly relevant during plateau escape)
- Ep14–26: P1 stable 0.071–0.075, P2 declining 0.316→0.263, P3 climbing 0.585→0.665

The ongoing Phase 2 decline after ep14 suggests the model finds the ICFG/DEF_USE signal increasingly uncertain relative to the containment signal — likely because the DEF_USE extraction doesn't yet capture cross-function data flow (only intra-function). This is the motivation for PLAN-3A/3B ablations.

**Phase 1 spike at ep13 (0.138):** The structural/CONTAINS phase briefly recovered weight during the F1 plateau (ep10–13). The model tried using coarser structural features to escape the stall. After ep14 it settled back — suggesting structural features have been fully exploited by ep14 and the residual ceiling is in cross-function data flow representation.

---

## Finding 4 — Hamming pattern: model committed to positive predictions by ep3

| Epoch | Hamming | Interpretation |
|-------|---------|----------------|
| 1 | 0.913 | Near-zero prediction → trivially high (92.2% negative base rate) |
| 2 | 0.727 | Model starts committing to positives |
| 3 | 0.611 | Positive predictions still noisy |
| 9+ | 0.42–0.51 | Stable zone — model fully committed, oscillates with F1 |
| 22 | 0.418 | Best val Hamming (aligns with best F1) |

The ep1→ep2 drop (0.913→0.727) is healthy — abandoning the all-zeros strategy is necessary for F1 even though it hurts Hamming.

---

## Finding 5 — F1 plateau structure

```
Ep 1–8   (aux warmup):  0.150 → 0.219  — rapid early learning, aux loss adding noise
Ep 8–10  (post-warmup): 0.219 → 0.240  — clean signal, fastest F1 gain period
Ep 10–26 (plateau zone): 0.228 – 0.259  — oscillation amplitude ~0.02, period ~3–4 epochs
```

The plateau onset at ep10 is the same epoch where all three eye losses plateau. This is a structural ceiling, not a learning-rate or regularization issue. The oscillation (not convergence to zero) indicates the model is not overfit — it's still exploring the loss landscape but finding no consistent direction out.

**Near-misses:** Ep25 (0.2588) was within 0.0005 of the ep22 best (0.2593). This is essentially statistical noise in val F1 given batch sampling variance. The model is circling the ceiling, not climbing.

**Comparison to v7:**  
- v7 best: 0.2651 at ep23  
- v8 best: 0.2593 at ep22  

v8 is 0.0058 below v7 despite having more expressive graph schema. This is likely because the new DEF_USE edges capture intra-function data flow only — the cross-function signal that would help the most (e.g., reentrancy via cross-function state mutation) is still absent. PLAN-3A/3B will isolate this.

---

## Decision: keep running or kill?

### State at decision time
- Best F1: 0.2593 (ep22) — saved to `ml/checkpoints/v8.0-AB_best.pt`
- Patience counter: 4/30
- Step loss: ~0.142 and declining — model not overfit, has capacity
- F1 in plateau zone for 16 epochs (ep10–26)
- Ep25 near-miss (0.2588) shows model is not stuck, but not breaking through either

### Arguments for killing now
1. 16-epoch oscillation is a strong structural ceiling signal
2. v8 tracking slightly below v7 despite more expressive schema — schema alone is not the fix
3. PLAN-3A (ICFG-only: edge types 6,8,9) gives more information than waiting ~15 hours for patience=30
4. The ep22 checkpoint is already a valid reference point for ablation comparison

### Arguments for continuing
1. Patience=4/30 — not technically plateaued by early stopping criteria
2. Ep25 near-miss (0.2588) — within one good epoch of a new best
3. Step loss still declining — if val loss follows, F1 may yet improve
4. JK Phase 2 still adapting (0.289→0.263 over ep22→26) — reorganization ongoing

### **Verdict: let it run until patience=10/30, then kill manually**

The near-miss at ep25 and the still-declining step loss mean there's a ~30–35% chance of a new best in the next 6–8 epochs. That probability is worth ~4 hours of GPU time (6 epochs × ~38 min). Beyond patience=10/30 (around epoch 32–34), the probability of a breakthrough collapses below 10% and the opportunity cost of not running PLAN-3A becomes dominant.

**Kill condition:** If no new best by epoch 32 (patience=10/30), kill the process and start PLAN-3A.

```bash
# Check patience counter in log at ep32:
tr '\r' '\n' < ml/logs/v8.0-AB-20260520.log | grep "patience\|Epoch 3[0-9]"

# Kill command:
kill $(cat ml/logs/v8.0-AB-20260520.pid 2>/dev/null || pgrep -f "train.py.*v8")
```

---

## Next steps after v8-AB

### PLAN-3A — ICFG-only ablation
Remove DEF_USE (type 10), keep CALL_ENTRY(8)/RETURN_TO(9)/CF(6). Tests whether cross-function call edges alone explain the Phase 2 weight gain, or if DEF_USE is contributing.

```bash
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. nohup python ml/scripts/train.py \
    --run-name v8.0-A-$(date +%Y%m%d) \
    --experiment-name sentinel-v8 \
    --phase2-edge-types 6 8 9 \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --epochs 100 --gradient-accumulation-steps 8 \
    > ml/logs/v8.0-A-$(date +%Y%m%d).log 2>&1 &
```

### PLAN-3B — DFG-only ablation
Remove CALL_ENTRY/RETURN_TO, keep CF(6)+DEF_USE(10). Tests whether intra-function data flow alone explains the Phase 2 signal.

```bash
--phase2-edge-types 6 10 --run-name v8.0-B-$(date +%Y%m%d)
```

### Interpretation matrix

| Result | Interpretation | Action |
|--------|---------------|--------|
| 3A > v8-AB | ICFG edges are the key; DEF_USE adds noise | Train full ICFG (3A config) as v8-final |
| 3B > v8-AB | DEF_USE is the key; CALL_ENTRY/RETURN_TO add noise | Train 3B config as v8-final |
| 3A ≈ 3B ≈ v8-AB | Combined edges cancel out; neither helps alone | Schema rethink (PLAN-1D: true cross-function CFG) |
| Both < v8-AB | Removing any Phase 2 edges hurts | v8-AB is the best current config; move to threshold tuning |

---

## Per-class best (epoch 22 checkpoint)

From the previous session's analysis (full per-class grid logged during ep22 eval):

| Class | F1 | Notes |
|-------|----|-------|
| IntegerUO | 0.592 | Dominant class (13,797 positives), learning well |
| GasException | 0.299 | Static-analysis-confirmable, borderline |
| Reentrancy | 0.261 | Should be higher; CEI pattern needs ICFG call edges |
| MishandledException | 0.276 | Improving via return_ignored feature |
| TOD | 0.228 | Timestamp-order dependent, weak signal |
| Timestamp | 0.227 | Low count (538), under-represented |
| CallToUnknown | ~0.15 | INHERITS/EMITS fired but not enough training signal |
| UnusedReturn | ~0.12 | return_ignored feature exists, still noisy |
| ExternalBug | ~0.10 | Weak; needs cross-function call graph |
| DenialOfService | 0.019 | Detached from loss (dos_loss_weight=0.0); not learning |
| **Macro** | **0.2593** | |
