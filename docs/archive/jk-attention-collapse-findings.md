# JK Attention Weight Collapse — Findings & Implications

**Date:** 2026-05-21  
**Diagnostic script:** `ml/scripts/jk_weight_hist.py`  
**Checkpoint analysed:** `ml/checkpoints/v8.0-AB-20260520_best.pt` (ep29, F1=0.2621)  
**Dataset:** val split, 936 contracts, 123,139 nodes  
**Report:** `ml/logs/jk_hist_v8AB.json`

---

## What We Measured

`_JKAttention` in `ml/src/models/gnn_encoder.py` combines three phase outputs via learned softmax attention weights. The training log reports the mean per-phase weight across all nodes in the last training batch — one number per phase per epoch. The diagnostic script ran all 936 val contracts through the model in eval mode and collected the full `[N, 3]` per-node weight tensor, giving the distribution that the mean obscures.

---

## Results

| Phase | Mean | Std | p5 | p25 | p50 | p75 | p95 | Dominant% |
|-------|------|-----|----|-----|-----|-----|-----|-----------|
| Phase 1 (struct+CONTAINS) | 0.065 | 0.074 | 0.002 | 0.006 | 0.018 | 0.157 | 0.188 | 0.0% |
| Phase 2 (CF/ICFG/DFG) | 0.247 | 0.078 | 0.035 | 0.233 | 0.269 | 0.297 | 0.315 | 0.01% |
| Phase 3 (rev-CONTAINS) | 0.688 | 0.097 | 0.569 | 0.616 | 0.688 | 0.702 | 0.938 | 99.99% |

**Normalised entropy:** 0.650 (scale 0–1, where 1 = perfectly uniform per node)

---

## Diagnosis: Global Constant, Not Per-Node Routing

**Phase 3 dominates 100% of nodes.** The per-node routing mechanism is not routing — it learned a fixed global weighting where every node gets approximately the same Phase 3 priority regardless of its type (CFG node, AST node, function node).

**Phase 2 is a global constant.** The IQR is 0.064 (p25=0.233, p75=0.297). Every node gets almost exactly the same ~0.27 Phase 2 weight. There is no differentiation between, say, CFG nodes (which have control-flow edges) and function nodes.

**Phase 1 is bimodal but suppressed.** The p50 is 0.018 but p75 jumps to 0.157 — two populations likely corresponding to CONTAINS-node types (function/contract nodes, which have CONTAINS edges) and leaf nodes (which don't). Even the higher-weight population averages ~0.16, still much less than Phase 3.

**This was also true in v7.** v7 final training weights: Phase1=0.050, Phase2=0.182, Phase3=0.768. Collapse is not a v8-specific regression. The JK attention mechanism has consistently learned to defer to reverse-CONTAINS as the primary readout in both models.

---

## Why Phase 3 Dominates

Reverse-CONTAINS (type 7) flows information from child nodes up to parent contract/function nodes. Contract-level nodes are the final classification points — the classifier reads the embedding of the root contract node. Phase 3 is essentially the only mechanism that aggregates leaf-level information (individual operations, expressions) back to the root. From the model's perspective, always up-weighting Phase 3 is a rational choice: it's the aggregation path closest to the output head.

This also explains why the entropy is moderate (0.65) rather than at maximum collapse (~0.30): Phase 2 still contributes a non-trivial ~0.25 to every node, and Phase 1 contributes ~0.07 on average. The model hasn't zeroed them out; it just doesn't vary them per node.

---

## Does This Matter?

**For v8-AB vs v7 comparison: no.** Both models collapsed the same way and still learned useful representations. The tiny F1 gap (0.003) is not explained by JK collapse — it's explained by edge content (DEF_USE vs ICFG-only).

**For PLAN-3A: no blocker.** PLAN-3A's hypothesis is about which edge types feed Phase 2, not about how JK combines phases. Running PLAN-3A as designed (drop DEF_USE) is the right next step. Fixing JK collapse simultaneously would make the experiment uninterpretable.

**For long-term ceiling: potentially yes.** If the model can't route different node types to different phases, it's leaving representational capacity on the table. A CFG node with heavy control-flow edges should weight Phase 2 more. A leaf AST node with no edges should weight Phase 1 (structural position) or Phase 3 (back to parent). The current collapse means the 3-phase architecture is, in practice, a weighted average with fixed global weights — equivalent to a much simpler design.

---

## What To Watch In PLAN-3A

The trainer now logs `±std` for each phase (added 2026-05-21):
```
JK attention weights — Phase1=0.211±0.074 Phase2=0.363±0.078 Phase3=0.426±0.097
```
And MLflow metrics: `jk_phase1_std`, `jk_phase2_std`, `jk_phase3_std`.

**Key signal:** If PLAN-3A Phase 2 std rises above 0.10 by ep10 (v8-AB was 0.078), the narrower ICFG-only signal is creating genuine per-node differentiation. If it stays below 0.08, the collapse is structural — edge content doesn't affect the routing.

A std collapse alert fires if all phase stds stay below 0.05 after ep3. That would indicate complete per-node routing collapse.

---

## If PLAN-3A Still Shows Collapse: PLAN-3D

If PLAN-3A confirms the same collapse pattern (Phase 3 dominant, Phase 2 IQR < 0.07), the follow-on experiment is **PLAN-3D: JK regularization or mode switch**.

Two concrete options:

**Option A — Entropy regularization:**  
Add a per-batch term `λ * (H_max - H_mean)` to the loss, penalising low per-node entropy. Forces the attention to spread across phases. Risk: fights the gradient signal if Phase 3 genuinely is best for most nodes.

**Option B — Switch `gnn_jk_mode` from `"attention"` to `"cat"`:**  
Concatenation doesn't collapse — it lets the downstream linear layers decide what to extract from each phase independently per class. Doubles the hidden dim of the JK output (3×256=768), so the fusion head needs resizing. More parameters, but eliminates the routing-collapse problem entirely. Likely the cleaner fix.

Run PLAN-3A and PLAN-3B first. Decide on PLAN-3D only after both ablations converge and the per-phase std diagnostic confirms the collapse pattern holds across configs.

---

## Infrastructure Added (2026-05-21)

| Change | File | Purpose |
|--------|------|---------|
| `last_node_weights` attribute | `ml/src/models/gnn_encoder.py:_JKAttention` | Stores full `[N, K]` per-node weights in eval mode for diagnostic scripts |
| `last_weight_stds` buffer | `ml/src/models/gnn_encoder.py:_JKAttention` | Per-phase std across nodes, updated every forward pass |
| `forward()` std computation | `ml/src/models/gnn_encoder.py:_JKAttention.forward` | `self.last_weight_stds.copy_(w_nk.std(0).detach())` |
| JK std logging + MLflow metrics | `ml/src/training/trainer.py:~1315` | Logs `jk_phase{i}_std`; fires collapse warning if all stds < 0.05 after ep3 |
| `jk_weight_hist.py` diagnostic | `ml/scripts/jk_weight_hist.py` | Full per-node weight distribution report; run on any checkpoint post-training |
