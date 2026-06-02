# Interpretability Validation Summary

**Date:** 2026-05-30  
**Why validated:** A prior run incorrectly reported "0 CFG nodes in cache" (REVERSE_CONTAINS was absent by design, not a data gap). That precedent established that sub-agent measurement errors are possible and must be independently verified before acting on findings.

---

## Validation Table

| Finding | Original claim | Validation method | Corrected finding | Status |
|---|---|---|---|---|
| **JK Phase weights** (EXP-L1) | "Phase 2 meaningfully underused" (weight 0.322 vs 0.346) | Read checkpoint buffer; run 5 fresh forward passes; check entropy value | Phase ranking confirmed real. But entropy = 1.0984/1.0986 = **99.98% of max** — all phases nearly equal. Gap is statistically real but practically tiny. | **PARTIALLY CONFIRMED** |
| **Edge ablation near-zero** (EXP-L2) | CFG ablation Δ = 1.08e-6 — "essentially zero" | Remove edges from edge_index (structural ablation) instead of zeroing embedding | Original method was flawed (embedding-zero ≠ structural removal). Corrected: CF removal → Δ = 0.0048 (450× larger). Combined Phase 2 Δ = 0.014 — still far below 0.03 threshold. | **PARTIALLY CONFIRMED** (conclusion correct, original magnitude wrong method) |
| **`uses_block_globals` ranks last** (EXP-L8) | "uses_block_globals ranks 10th/11th — semantics ignored" | Check FEATURE_NAMES against graph_schema.py | Script had **stale hardcoded FEATURE_NAMES** from pre-v8 schema (9/11 features mislabelled). Actual ranking: `uses_block_globals` **2nd for Timestamp** (importance 0.0113). Still 10th globally (0.0055). | **ARTIFACT — original claim wrong** |
| **Timestamp Cohen's d = 1.657** (EXP-S3) | "Timestamp d=1.657, 2.75× larger" | Recompute d on full val split and training split separately | Val d = **0.643** (not 1.657). Training split: 2.34× size ratio (confirms model had exposure). Shortcut is real but less extreme. | **PARTIALLY CONFIRMED** |
| **Phase 2 probe zero** (EXP-L5) | Phase 2 adds 0pp F1 over Phase 1 | Check probe implementation: gnn returns 4-tuple not 3-tuple with return_intermediates=True | Implementation correct. New nuance: Phase 2 AUROC **0.618 > Phase 1 0.612** for Reentrancy. Phase 3 AUROC = 0.526 (worst) despite highest JK weight. | **CONFIRMED** (with new nuance) |

---

## What Was Fixed

| Script / File | Bug | Fix |
|---|---|---|
| `exp_l8_permutation_importance.py` | Stale hardcoded `FEATURE_NAMES` from pre-v8 schema — 9 of 11 features mislabelled | Replaced with `from ml.src.preprocessing.graph_schema import FEATURE_NAMES` |
| `exp_l1_jk_weight_analysis.py` | `mean_entropy` field stored mean weight (0.333) not Shannon entropy (1.099) | Fixed to compute `H = -sum(p * log(p))` correctly |
| `ml/src/models/gnn_encoder.py` | Docstring said 3-tuple return for `return_intermediates=True` | Corrected to 4-tuple: `(x, batch, jk_entropy, intermediates)` |
| `exp_l4_gradient_saliency.py` | Same stale FEATURE_NAMES as L8 — dims 3–9 all wrong labels | Fixed to import from graph_schema (2026-05-31) |
| `exp_a2_cfg_inheritance.py` | `_CONTAINS_EDGE = 0` hardcoded — was matching CALLS (type 0) not CONTAINS (type 5) | Fixed to `EDGE_TYPES["CONTAINS"]` + import (2026-05-31) |
| `exp_e1_receptive_field.py` | Analysis 2 used REVERSE_CONTAINS (runtime-only, never stored) — always 0% | Rewritten to check FUNCTION→CFG via CONTAINS only (2026-05-31) |
| `exp_e1_receptive_field.py` | Analysis 3 checked CONTRACT→FUNCTION — no such edge type in v8 schema | Rewritten to check FUNCTION→FUNCTION CALLS connectivity (2026-05-31) |
| `exp_l3_attention_visualization.py` | Only hooked conv3 (CF edges) — missed conv3b (CALL_ENTRY+RETURN_TO) | Extended to hook both conv3 and conv3b in single forward pass (2026-05-31) |

---

## Corrected Unified Conclusion

The core conclusion from the interpretability suite stands, but is softened:

> The GNN has not fully exploited Phase 2 control-flow structure. All three JK phases contribute nearly equally (entropy at 99.98% of maximum), but Phase 2 contributes the least. Structural edge ablation confirms Phase 2 dropout reduces Reentrancy prediction by only 0.014 — real but small. The primary GNN signal comes from structural hierarchy (Phase 3 REVERSE_CONTAINS) and node type routing (`type_id_norm` is 3× more important than the next feature globally). GraphCodeBERT token semantics carry most class-specific prediction. Timestamp classification has a confirmed but moderate size shortcut (d=0.643 in val). The `uses_block_globals` feature IS used for Timestamp (ranks 2nd) — semantic features are not fully ignored.

---

## What Changes for the Proposal

| Original proposal item | After validation |
|---|---|
| "Phase 2 essentially unused — add CEI auxiliary loss" | Still valid — 0.014 drop is real but small; CEI aux loss will force larger signal |
| "Timestamp is entirely a size shortcut" | Partially true — size IS a proxy but `uses_block_globals` also contributes; Sol-3 gating still recommended |
| "GNN ignores semantic features" | Partially true — type_id_norm dominates globally but per-class relevant features are used (uses_block_globals 2nd for Timestamp, return_ignored 6th globally) |
| Calibration fix | Unchanged — ECE 0.205–0.310 confirmed across all classes |
| solc-select fix | Unchanged — unblocks EXP-L6 |
| max_nodes increase | Unchanged — 1414 nodes observed, truncation at 1024 confirmed |
