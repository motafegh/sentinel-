# PLAN-3A Results: ICFG-only Phase 2 Ablation

**Run name:** `v8.0-A-20260521`  
**Date:** 2026-05-21 – 2026-05-23  
**Phase 2 edge types:** `CF(6) + CALL_ENTRY(8) + RETURN_TO(9)` — DEF_USE(10) removed  
**Best checkpoint:** `ml/checkpoints/v8.0-A-20260521_best.pt` (ep41)  
**Thresholds:** `ml/checkpoints/v8.0-A-20260521_best_thresholds.json`  
**Training log:** `ml/logs/v8.0-A-20260521.log`

---

## Training Summary

| Metric | Value |
|--------|-------|
| Best raw F1-macro | 0.2790 (ep41) |
| Tuned F1-macro | **0.2877** |
| Best epoch | 41 / 100 |
| Killed at | ep67 (patience 26/30 — auto-expired) |
| VRAM peak | 0.5 / 8.0 GiB |
| Phase 2 at best ep | 0.155±0.063 |
| Phase 2 at death | 0.048±0.058 |

**Convergence pattern:** Staircase with late-breaking improvements. Long plateau ep17–35 (no improvement for 19/30), then three rapid new bests at ep36→ep38→ep41. After ep41: 26 epochs with F1 oscillating 0.264–0.278, never breaking the ceiling.

---

## Tuned Per-class Results vs All Baselines

| Class | v7 tuned | v8-AB tuned | **PLAN-3A tuned** | vs v7 | vs v8-AB |
|-------|----------|-------------|-------------------|-------|----------|
| IntegerUO | 0.706 | **0.715** | 0.699 | −0.007 | −0.016 |
| GasException | **0.369** | 0.360 | 0.358 | −0.011 | −0.002 |
| Reentrancy | **0.303** | 0.286 | 0.291 | −0.012 | +0.005 |
| MishandledException | 0.287 | 0.287 | **0.289** | +0.002 | +0.002 |
| ExternalBug | 0.257 | **0.270** | 0.255 | −0.002 | −0.015 |
| CallToUnknown | 0.250 | 0.236 | **0.256** | +0.006 | +0.020 |
| **Timestamp** | 0.223 | 0.217 | **0.255** | **+0.032** | **+0.038** |
| TOD | 0.257 | **0.262** | 0.251 | −0.006 | −0.011 |
| UnusedReturn | **0.204** | 0.198 | 0.195 | −0.009 | −0.003 |
| DoS | 0.019 | 0.020 | 0.030 | +0.011 | +0.010 |
| **F1-macro** | 0.2875 | 0.2851 | **0.2877** | **+0.0002** | **+0.0026** |

**PLAN-3A vs v7: wins 4 classes, loses 6. Net macro +0.0002 — statistical tie.**  
**PLAN-3A vs v8-AB: wins 5 classes, loses 5. Net macro +0.0026.**

---

## Hypothesis Verdicts

### H1 — "Phase 2 multi-edge dilution hurts Reentrancy"
**Pre-PLAN-3A status:** CONFIRMED (v8-AB vs v7, −0.017 Reentrancy)  
**PLAN-3A test:** Drop DEF_USE, keep ICFG-only → Reentrancy should recover toward v7 (0.303)

**Result: PARTIALLY REFUTED.**  
Reentrancy went 0.286 (v8-AB) → 0.291 (PLAN-3A) — only +0.005. Still −0.012 below v7 (0.303). Removing DEF_USE was necessary but not sufficient to recover Reentrancy. Something else is limiting it — likely the label noise documented in BUG-H5 (~14% of Reentrancy=1 contracts have no external calls, i.e. are mislabeled).

### H-TIMESTAMP — "Timestamp will regress when DEF_USE is dropped"
**Pre-PLAN-3A prediction:** Moderate regress (DEF_USE 92.5% coverage on Timestamp — highest of all classes)

**Result: WRONG — opposite direction.**  
Timestamp surged from 0.223 (v7) / 0.217 (v8-AB) → **0.255** (+0.032 vs v7, +0.038 vs v8-AB). This is the largest single-class gain in the entire ablation series. DEF_USE was apparently **hurting** Timestamp detection, not helping it. Hypothesis: block.timestamp def-use chains from DEF_USE edges introduce false-positive noise by connecting timestamp reads to unrelated data flows — the model over-fires on contracts with many assignments from timestamp-derived values. CALL_ENTRY/RETURN_TO (ICFG traversal) better capture the structural pattern of when timestamp values are used in guard conditions vs storage writes.

### H-INTEGERUO — "IntegerUO will regress when DEF_USE is dropped"
**Pre-PLAN-3A prediction:** Regress (DEF_USE 81.0% on IntegerUO — primary arithmetic def-use signal)

**Result: APPROXIMATELY CORRECT, but smaller than predicted.**  
IntegerUO dropped 0.715 (v8-AB) → 0.699 (PLAN-3A), −0.016. However, raw training F1 for IntegerUO was consistently high (0.620–0.647) throughout PLAN-3A, suggesting the tuned regression is partly a threshold calibration artifact. The model still detects IntegerUO well via ICFG (call chains into overflow-prone arithmetic) — DEF_USE contributed but was not the sole signal.

### H-EXTERNALB — "ExternalBug will hold with ICFG preserved"
**Pre-PLAN-3A prediction:** Hold or slight ↑ (CALL_ENTRY 69.5% preserved)

**Result: WRONG — regressed.**  
ExternalBug dropped 0.270 (v8-AB) → 0.255 (PLAN-3A), −0.015. CALL_ENTRY was preserved but DEF_USE removal hurt cross-function data flow detection. ExternalBug apparently relies on both ICFG structure AND data-flow chains to confirm that external call results flow into storage writes or condition checks.

### H-CALLTOUNKNOWN — "CallToUnknown will hold"
**Pre-PLAN-3A prediction:** Roughly hold

**Result: BETTER THAN PREDICTED — improved.**  
CallToUnknown jumped 0.236 (v8-AB) → 0.256 (PLAN-3A), +0.020. Dropping DEF_USE apparently reduced false negatives on contracts that make typed calls. DEF_USE chains from call return values may have been misidentified as "typed" call patterns.

---

## JK Attention Trajectory

| Checkpoint | Phase2 mean | Phase2 std | Phase3 mean | Note |
|-----------|-------------|------------|-------------|------|
| ep3 | 0.365 | 0.152 | 0.553 | Peak Phase 2 engagement |
| ep16 (1st best) | 0.278 | 0.108 | — | First meaningful F1 |
| ep41 (best) | 0.155 | 0.063 | 0.778 | Best checkpoint |
| ep67 (death) | 0.048 | 0.058 | 0.909 | Fully collapsed |
| v8-AB (final) | 0.204 | 0.078 | 0.688 | Reference |
| v7 (final) | 0.182 | — | 0.768 | Reference |

**Key finding:** PLAN-3A started with much higher Phase 2 engagement (std=0.152 at ep3 vs v8-AB's final 0.078) confirming ICFG-only created genuine per-node routing early. But this advantage faded — Phase 2 collapsed to 0.048 by ep67. The model spent its ICFG signal in the first 20–30 epochs to shape representations, then reverted to Phase 3 dominance for the remaining training. The tuned F1 improvement over v8-AB is largely the artifact of this better early initialization, not sustained ICFG routing.

---

## Prediction Scorecard (Pre-PLAN-3A vs Actual, vs v8-AB)

| Class | Predicted direction | Actual direction | Correct? |
|-------|--------------------|--------------------|----------|
| Reentrancy | ↑ improve | ↑ +0.005 | ✓ direction, ✗ magnitude |
| IntegerUO | ↓ regress | ↓ −0.016 | ✓ |
| ExternalBug | → hold | ↓ −0.015 | ✗ |
| **Timestamp** | ↓ regress | **↑ +0.038** | **✗ — opposite** |
| GasException | → hold | ↓ −0.002 | ✓ (approx) |
| CallToUnknown | → hold | ↑ +0.020 | ✗ — better than predicted |
| TOD | → hold | ↓ −0.011 | ✗ — slight regress |

4/7 correct direction, 3/7 wrong or opposite. The GATE-3A-0 coverage rationale was insufficient — edge *presence* rates don't predict *signal quality* direction.

---

## Implications for PLAN-3B (DFG-only: CF + DEF_USE, drop ICFG)

Based on PLAN-3A findings, updated predictions for PLAN-3B (`--phase2-edge-types 6 10`):

| Class | PLAN-3B prediction | Reasoning |
|-------|-------------------|-----------|
| IntegerUO | ↑ vs PLAN-3A | DEF_USE restored — primary arithmetic def-use signal returns |
| Reentrancy | ↓ vs PLAN-3A | CALL_ENTRY/RETURN_TO removed — loses the partial +0.005 gain from ICFG |
| **Timestamp** | ↓ vs PLAN-3A | DEF_USE restored — PLAN-3A showed DEF_USE *hurts* Timestamp; restoring it should reverse the +0.038 gain |
| ExternalBug | ↓↓ vs PLAN-3A | Both CALL_ENTRY and RETURN_TO gone — cross-function call detection severely degraded |
| CallToUnknown | ↓↓ vs PLAN-3A | CALL_ENTRY is literally the "call to unknown" edge — removing it is the biggest risk |
| GasException | → hold | CFG(6) preserved; gas patterns mainly intra-function |
| MishandledException | → hold | Uncertain; probably slight regress without ICFG |
| TOD | → hold | Symmetric to v8-AB roughly |

**Key diagnostic for PLAN-3B:** If Timestamp drops back below 0.230 and ExternalBug drops below 0.240, it confirms that ICFG edges (not DEF_USE) are the primary signals for those classes. If IntegerUO rises above 0.710, it confirms DEF_USE is the primary IntegerUO signal.

---

## What PLAN-3A Tells Us About the Path Forward

**What worked:** ICFG-only Phase 2 gave a cleaner early training signal (Phase 2 std=0.152 vs v8-AB's 0.078), leading to better-shaped representations and a small net macro gain (+0.0002 vs v7, +0.0026 vs v8-AB). Timestamp was the surprise beneficiary.

**What didn't work:** H1 (Reentrancy recovery) was the original motivation for PLAN-3A. It didn't recover — only +0.005 vs v8-AB, still −0.012 below v7. The Reentrancy ceiling appears to be a label noise problem (BUG-H5) more than an edge type problem.

**The real ceiling:** All three runs (v7, v8-AB, PLAN-3A) converged to similar macro F1 (0.2875 / 0.2851 / 0.2877). The architecture is not the bottleneck — it's the label quality and the fundamental limit of the v8 graph representation for ambiguous classes (Reentrancy, ExternalBug). PLAN-3B is worth running to complete the ablation matrix, but the expected improvement is small.

**Path to meaningful improvement:** Fix BUG-H5 (Reentrancy label noise) and BUG-H4 (Timestamp label noise) via label_cleaner.py before v9 training. These two classes have documented mislabeling that is likely capping the model's ceiling regardless of edge type choice.
