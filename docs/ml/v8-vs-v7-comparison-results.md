# v8.0-AB vs v7.0 — Comparison Results
**Date:** 2026-05-20  
**Checkpoints compared:**
- v7.0: `ml/checkpoints/v7.0_best.pt` — epoch 23, F1=0.2651 (default threshold 0.5)
- v8.0-AB: `ml/checkpoints/v8.0-AB-20260520_best.pt` — epoch 29, F1=0.2621 (default threshold 0.5)

**Tests run:** threshold tuning (both models, val split), manual behavioral test (20 contracts)  
**Comparison plan:** `docs/ml/v8-vs-v7-comparison-plan.md`

---

## Infrastructure bugs discovered during comparison

Three bugs in `tune_threshold.py` and `ml/src/inference/predictor.py` prevented any cross-version loading. All fixed before results were obtained.

| Bug | Symptom | Fix | Files |
|-----|---------|-----|-------|
| `_orig_mod.` prefix in state dict | `RuntimeError: Missing key(s)` on all checkpoints saved with `use_compile=True` | Strip `._orig_mod.` from all state dict keys before `load_state_dict()` | `tune_threshold.py`, `predictor.py` |
| Edge embedding size mismatch | `size mismatch for gnn.edge_embedding.weight: [8,64] vs [11,64]` — v7 was trained with 8 edge types, current schema has 11 | Detect size from state dict key shape; rebuild `nn.Embedding(ckpt_num_edge_types, emb_dim)` before loading | `tune_threshold.py`, `predictor.py` |
| BF16 mixed-dtype checkpoint | `mat1 Float and mat2 BFloat16` during forward pass — BF16 AMP training leaves 230/329 tensors as BF16 in the checkpoint | `model.float()` after `load_state_dict()` normalises everything to float32 | `tune_threshold.py`, `predictor.py` |
| OOB clamp used global constant | v7 model (8-type embedding) running on v8 graphs (types 0–10) would pass edge types 8/9/10 through the `clamp(0, NUM_EDGE_TYPES-1)` guard since NUM_EDGE_TYPES=11 globally | Change OOB clamp to use `self.edge_embedding.num_embeddings - 1` instead of the global constant | `ml/src/models/gnn_encoder.py` |
| v7 cache schema version rejected | `RuntimeError: RAM cache schema mismatch: cache has version='v7' but FEATURE_SCHEMA_VERSION='v8'` — can't use the v7 cache file at all | Use v8 cache for both models; v7 model uses OOB clamping to handle unknown edge types 8/9/10 | (no code change — use correct cache) |

---

## Test 1 — Threshold tuning (H4: calibration shift)

**Hypothesis:** Default threshold 0.5 may be suboptimal for v8 due to architecture changes. If v8's tuned F1 meets or beats v7's tuned F1, the apparent gap is purely a calibration artefact.

### Results

| | Default threshold (0.5) | Tuned thresholds | Gain from tuning |
|---|---|---|---|
| **v7.0** | 0.2651 | **0.2875** | +0.022 |
| **v8.0-AB** | 0.2621 | **0.2851** | +0.023 |
| Gap (v7 lead) | 0.0030 | **0.0024** | closed by 0.0006 |

### Per-class tuned thresholds and F1 (val split)

| Class | v7 threshold | v7 F1 | v8 threshold | v8 F1 | Winner | Δ |
|-------|-------------|--------|-------------|--------|--------|---|
| IntegerUO | 0.55 | 0.706 | 0.55 | **0.715** | v8 | +0.009 |
| GasException | **0.45** | **0.369** | 0.45 | 0.360 | v7 | −0.009 |
| Reentrancy | **0.40** | **0.303** | 0.40 | 0.286 | v7 | −0.017 |
| MishandledException | 0.40 | 0.287 | 0.40 | 0.287 | Tie | 0.000 |
| ExternalBug | 0.40 | 0.257 | 0.40 | **0.270** | v8 | +0.013 |
| CallToUnknown | **0.40** | **0.250** | 0.40 | 0.236 | v7 | −0.014 |
| TransactionOrderDep | 0.40 | 0.257 | 0.40 | **0.262** | v8 | +0.005 |
| Timestamp | 0.45 | **0.223** | 0.30 | 0.217 | v7 | −0.006 |
| UnusedReturn | **0.35** | **0.204** | 0.35 | 0.198 | v7 | −0.006 |
| DenialOfService | 0.05 | 0.019 | 0.45 | 0.020 | Tie | +0.001 |

**v8 wins 3 classes, v7 wins 5, 2 ties.**

### H4 verdict: PARTIALLY CONFIRMED — not the main cause

Threshold miscalibration affects both models roughly equally (+0.022 each). After tuning, v7 still leads by 0.0024. The gap narrowed by 0.0006 (20% closure). Calibration is a real issue but explains only a small fraction of the overall performance gap.

Notable: v8's Timestamp threshold drops to 0.30 (vs v7's 0.45). v8's probability scores for Timestamp are systematically lower, suggesting the ICFG/DEF_USE edges dilute the signal that was cleanly detecting Timestamp patterns in v7.

---

## Test 2 — Manual behavioral test (H1, H2, H5)

20 hand-crafted contracts covering all 10 vulnerability classes plus 3 safe contracts. Both models run with their tuned per-class thresholds.

### Results per contract

| # | Contract | Expected | v7 result | v8 result |
|---|----------|----------|-----------|-----------|
| 01 | reentrancy_classic | Reentrancy | ✓ (+CallToUnk, DoS FP) | ✓ (+CallToUnk FP) |
| 02 | reentrancy_tricky | Reentrancy | ✓ (+CallToUnk, DoS FP) | ✓ (+CallToUnk FP) |
| 03 | integer_overflow | IntegerUO | ✗ (DoS FP only) | ✗ (nothing) |
| 04 | timestamp_dependence | Timestamp | ✓ (+DoS FP) | ✓ (clean) |
| 05 | denial_of_service | DoS | ✓ | **✗** (nothing) |
| 06 | mishandled_exception | MishandledException | ✗ (Reent FP) | ✗ (Reent FP) |
| 07 | tx_order_dependence | TOD | ✗ (DoS FP only) | ✗ (nothing) |
| 08 | unused_return | UnusedReturn | ✗ (CallToUnk+Reent+DoS FP) | ✗ (CallToUnk+Reent+DoS FP) |
| 09 | call_to_unknown | CallToUnknown | ✓ (+3 FP) | ✓ (+6 FP) |
| 10 | gas_exception | GasException | ✗ (DoS FP only) | ✗ (Reent FP only) |
| 11 | external_bug | ExternalBug | **✗** (DoS FP only) | ✓ (+7 FP) |
| 12 | safe_contract | (safe) | FP (CallToUnk+Reent+DoS) | FP (CallToUnk+Reent) |
| 13 | multilabel_complex | Reent+TS+UR | ✗ all (DoS FP) | partial: Timestamp only |
| 14 | reentrancy_minimal | Reentrancy | ✓ (+CallToUnk, DoS FP) | ✓ (clean) |
| 15 | tod_minimal | TOD | ✗ (DoS FP) | ✗ (multiple FP) |
| 16 | gas_minimal | GasException | ✗ (DoS FP) | ✗ (nothing) |
| 17 | integer_simple | IntegerUO | ✗ (DoS FP) | ✗ (nothing) |
| 18 | safe_no_calls | (safe) | **FP (DoS)** | ✓ clean |
| 19 | safe_with_transfer | (safe) | FP (CallToUnk+Reent+DoS) | FP (CallToUnk+Reent) |
| 20 | unused_return_minimal | UnusedReturn | ✓ (+DoS FP) | ✓ (+7 FP) |

**v7: 7/19 correct (37%) · 0/3 safe contracts clean**  
**v8: 8/19 correct (42%) · 1/3 safe contracts clean**

### Key patterns

**v8 uniquely correct (not v7):**
- Contract 11 (ExternalBug): v8 detects it, v7 misses entirely. The CALL_ENTRY/RETURN_TO edges create a cross-function call path that the model reads as an external call bug pattern. This is H2 partially confirmed — cross-function edges do help for ExternalBug.
- Contract 13 (multilabel): v8 detects Timestamp; v7 gets nothing. Partial credit.
- Contract 18 (safe): v8 is clean; v7 fires DoS because its DoS threshold is 0.05 (fires on everything).

**v7 uniquely correct (not v8):**
- Contract 05 (DoS): v8 misses it. v8's DoS threshold is 0.45; the DoS probability is 0.392 — just below. v7's threshold is 0.05 so it trivially fires everywhere (this is a calibration artifact, not real DoS detection capability).

**Both miss completely:** IntegerUO (03, 17), GasException (10, 16), TOD (07, 15), MishandledException (06). These classes need structural improvements that neither model has.

**v8 false positive explosion:** On contracts 09, 11, 20 v8 fires 6–8 classes simultaneously. The ICFG/DEF_USE edges create correlated activations across classes — when the model sees a complex call graph, it over-predicts broadly rather than pointing precisely. v7 also over-predicts but with a narrower, more consistent pattern (Reentrancy + CallToUnknown + DoS).

**v7's DoS noise:** The 0.05 threshold means DoS is effectively always predicted, appearing in 15/20 contracts' prediction lists. It creates constant noise but doesn't fail for the actual DoS contract. This is a pure calibration failure — the model has no real DoS discrimination.

---

## Hypothesis resolution

| ID | Hypothesis | Evidence | Verdict |
|----|-----------|---------|---------|
| **H1** | Phase 2 conv hops with 4 edge types (CF+CE+RT+DU) dilute the CEI pattern, hurting Reentrancy | Reentrancy F1: v7=0.303, v8=0.286 (−0.017). Manual test: Reentrancy detection equal on clean contracts, but v8 has more false positives. JK Phase 2 peak of 0.362 at ep3 declining to 0.204 at kill — model initially found the new edges interesting, then deprioritised them. | **CONFIRMED** — the multi-type Phase 2 hops hurt Reentrancy |
| **H2** | DEF_USE is intra-function only; cross-function reentrancy patterns remain invisible | ExternalBug: v8=0.270, v7=0.257 (+0.013). v8 uniquely detects contract 11 (ExternalBug). CALL_ENTRY/RETURN_TO edges do help for cross-function patterns. DEF_USE contribution unclear without PLAN-3A/3B ablation. | **PARTIALLY CONFIRMED** — ICFG edges (not DEF_USE specifically) provide cross-function signal; they help ExternalBug and TOD, not Reentrancy |
| **H3** | Label ceiling limits both models equally around F1≈0.26 | Both models plateau at same onset (ep10), same amplitude (~0.02), same ceiling (~0.26–0.29 tuned). Both miss the same 4 classes on hand-crafted contracts. | **CONFIRMED** — data/label quality is the structural ceiling for both; PLAN-3A will still hit this ceiling |
| **H4** | Default 0.5 threshold is suboptimal for v8 | Both models gain +0.022 from tuning; gap narrows only 0.0006. | **PARTIALLY CONFIRMED** — real effect but small; not the main cause |
| **H5** | v8 helps some classes, hurts others; net is negative | v8 wins IntegerUO/ExternalBug/TOD, loses Reentrancy/GasException/CallToUnknown. Manual test: v8 +1 correct, −1 (DoS, thin margin). | **CONFIRMED** — class-specific tradeoff is the main explanation |
| **H6** | Phase 2 edge type spread (4 types) makes convolutions less focused | Consistent with H1 results; v8 false positive explosion on complex contracts suggests edge type noise. Cannot isolate from H1 without PLAN-3A. | **PROBABLE** — consistent with all evidence, directly testable by PLAN-3A |

---

## Expectations vs reality

### What we expected

When adding ICFG-Lite (CALL_ENTRY, RETURN_TO) and DEF_USE edges to v7's schema:

1. **Expected:** Reentrancy detection improves — call entry/return edges should capture the CEI pattern (external call → state change) more directly than the existing CF(6) edges.
2. **Expected:** IntegerUO improves slightly — DEF_USE edges trace data flow for overflow variables.
3. **Expected:** ExternalBug improves — CALL_ENTRY/RETURN_TO directly model cross-function call boundaries where external call bugs appear.
4. **Expected:** Overall F1 improves by 0.005–0.015 over v7.

### What actually happened

1. **Reentrancy degraded (−0.017 tuned F1).** The Phase 2 hops now process 4 edge types simultaneously. The GATConv attention mechanism spreads across CONTROL_FLOW, CALL_ENTRY, RETURN_TO, and DEF_USE in the same 3-hop convolution. The CEI pattern (external call → temp write → storage write) that v7 learned cleanly through CF-only hops is now being mixed with data-flow and call-boundary edges. The model can no longer isolate the CEI ordering signal.

2. **IntegerUO improved (+0.009 tuned F1).** DEF_USE edges tracing variable definitions and uses do help with overflow detection. This is the expected mechanism: seeing how a variable flows from an unchecked arithmetic operation to a storage write lets the model identify overflow patterns.

3. **ExternalBug improved (+0.013 tuned F1, +1 manual test detection).** Confirmed. CALL_ENTRY/RETURN_TO edges provide the cross-function call boundary context that v7 had no way to see.

4. **Net F1 is negative (−0.003 default, −0.0024 tuned).** The Reentrancy regression (the largest class by impact) outweighs the gains on IntegerUO, ExternalBug, and TOD.

### Why this diverged from expectations

The CEI pattern for Reentrancy is a *sequential ordering* signal: external call happens first, then state is written. In v7, the Phase 2 CF-only hops learned this ordering on control flow edges. In v8, those same hops also process CALL_ENTRY and RETURN_TO edges — which create a very similar graph topology to the CEI pattern (call boundary → cross-function → return). The model cannot easily distinguish "CEI vulnerability" from "normal cross-function call" when both look like call-boundary → function-boundary → return patterns at the edge-type level.

DEF_USE edges add a different problem: they connect variable definitions to uses across CFG nodes. This is genuinely useful for data-flow classes (IntegerUO) but creates noise for control-flow classes (Reentrancy, TOD) where the bug is about execution ordering, not data flow.

---

## Probability score analysis

### v7 probability pattern (from manual test)

Most probabilities cluster tightly between 0.22–0.55 regardless of the actual vulnerability. The model has poor separation — it sees "something suspicious" but can't cleanly distinguish class types. Exceptions:
- Reentrancy and CallToUnknown: consistently high (0.50–0.70) on contracts with external calls
- DoS: uniformly high (0.44–0.54) on almost everything — no discrimination at all

### v8 probability pattern (from manual test)

Probabilities are more spread but the spread is not always in the right direction. On complex contracts (09, 11, 20), many classes spike simultaneously (0.40–0.67 range for 6–8 classes). This correlated multi-class activation is the v8 false positive explosion — the ICFG edges create a "high connectivity" signal that the model interprets as suspicious across multiple vulnerability types simultaneously.

IntegerUO probabilities are higher in v8 on integer-related contracts (0.50–0.68 vs v7's 0.41–0.53) — consistent with DEF_USE edges helping data-flow detection.

---

## What neither model can detect

These classes have consistent detection failure across both models, manual test, and val set:

| Class | Why both models fail |
|-------|---------------------|
| **GasException** | Requires understanding gas usage within loops and external calls — a cross-function, cross-loop signal that neither CF(6) nor ICFG edges adequately capture. Needs explicit loop gas modeling. |
| **TransactionOrderDependence** | Requires detecting that the same state variable is read *and* written across different transaction contexts — a global/temporal pattern invisible within a single transaction's CFG. |
| **MishandledException** | In hand-crafted test: model confuses it with Reentrancy (both involve external calls). The distinguishing feature (return value unchecked) is in `return_ignored` node feature [7], but the model hasn't learned to isolate it from the call pattern. |
| **IntegerUO (hand-crafted)** | The simple hand-crafted overflow contracts (03, 17) are too minimal — just an `+` operation without the full call/state context that appears in training data. The model learned from real-world patterns which are more complex. |

---

## Key metrics summary

| Metric | v7.0 | v8.0-AB | Winner |
|--------|------|---------|--------|
| Default threshold F1 (val) | 0.2651 | 0.2621 | v7 |
| Tuned threshold F1 (val) | **0.2875** | 0.2851 | v7 |
| Tuned F1 hamming loss | 0.2821 | **0.2517** | v8 |
| Tuned F1 micro | — | 0.3304 | — |
| Manual test correct (19 vuln) | 7/19 (37%) | **8/19 (42%)** | v8 |
| Manual test safe clean (3) | 0/3 | **1/3** | v8 |
| Reentrancy F1 (tuned) | **0.303** | 0.286 | v7 |
| IntegerUO F1 (tuned) | 0.706 | **0.715** | v8 |
| ExternalBug F1 (tuned) | 0.257 | **0.270** | v8 |
| JK Phase 2 at kill | 0.182 | 0.204 | — |

---

## Decision for PLAN-3A

**Confirmed by this comparison:** v8's regression on Reentrancy is caused by Phase 2 multi-edge-type dilution (H1 confirmed, H6 probable). The ICFG edges (CALL_ENTRY/RETURN_TO) contribute to ExternalBug and IntegerUO, but DEF_USE is likely adding noise for control-flow classes.

**PLAN-3A** drops DEF_USE, keeps ICFG only (`--phase2-edge-types 6 8 9`). Predictions:
- Reentrancy should recover toward v7's 0.303 (CEI pattern less diluted with one fewer edge type in Phase 2)
- ExternalBug should stay at v8 level or improve (CALL_ENTRY/RETURN_TO preserved)
- IntegerUO may regress slightly (loses DEF_USE data-flow edges) — the key question is how much
- Net F1 should exceed both v7 and v8-AB if Reentrancy recovers sufficiently

**PLAN-3B** (`--phase2-edge-types 6 10`, DFG-only) will tell us the DEF_USE contribution in isolation — if IntegerUO drops when DEF_USE is removed, that confirms it was genuinely helping.

### PLAN-3A launch command

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

---

## Files produced

| File | Content |
|------|---------|
| `ml/checkpoints/v7.0_best_thresholds.json` | v7 per-class tuned thresholds (val split) |
| `ml/checkpoints/v8.0-AB-20260520_best_thresholds.json` | v8 per-class tuned thresholds (val split) |
| `ml/logs/tune_threshold_v7.log` | Full v7 threshold sweep output |
| `ml/logs/tune_threshold_v8.log` | Full v8 threshold sweep output |
| `ml/logs/manual_test_v7.log` | v7 behavioral test: 20-contract per-contract table + probability matrix |
| `ml/logs/manual_test_v8.log` | v8 behavioral test: 20-contract per-contract table + probability matrix |
