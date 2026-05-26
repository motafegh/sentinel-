# GCB-P1 Run 4 — Final Analysis

**Checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt`
**Best epoch:** 32  |  **Best F1:** 0.3362
**Killed:** 2026-05-26 20:12 at epoch 44 — F1 plateau for 12 epochs (capacity ceiling)
**Log:** `ml/logs/graphcodebert-p1-run4-20260525.log`

---

## Run History

| Run | Root cause of death | Best ep | Best F1 |
|-----|---------------------|---------|---------|
| Run 1 (GCB-P0 ep3 warmstart) | Killed to apply IMP-* fixes | ep27 | 0.2628 |
| Run 2 (8-layer GNN, all IMP) | JK Phase3 structural collapse — CONTAINS downward pass made Phase3 deepest | ep4 | — |
| Run 3 (λ=0.01, entropy reg) | NC-4 double-amp (DoS pos_weight × ASL gamma ≈ 20,000×) + JK uniform collapse | ep16 | 0.17 |
| **Run 4** (λ=0.005, no ASL pw) | Capacity ceiling — F1 locked 0.31–0.34 for 12 eps | **ep32** | **0.3362 ★** |

Run 4 is the highest F1 in SENTINEL history: +0.0485 over PLAN-3A tuned (0.2877), +0.0734 over Run 1.

---

## Run 4 Fixes vs Run 3

| Fix | What | Why |
|-----|------|-----|
| NC-4 REVERTED | `pos_weight` NOT passed to ASL | Double-amp: DoS 10× pos_weight × ASL asymmetric gamma ≈ 20,000× amplification. 243 DoS positives vs 28,860 negatives — completely overrode all other gradients |
| C-3 λ 0.01→0.005 | Halved JK entropy regularizer | λ=0.01 forced perfect 33/33/33 — no per-node routing; all three phases carried identical signal |
| NC-1 logging fix | Now logs total params vs state-initialized; clears all params regardless | Previous version logged "0 params reset" — prefix_proj Adam state never initialized during warmup (zero grad = never touched by Adam) |
| C2 collision fix | `_scatter_to_dense`: `valid = local_idx < max_nodes` BEFORE clamp | Excess nodes (>1024) were clamped to slot 1023 and wrote there — collision overwrote valid node |
| C1 gnn_enc_norm | Full GNN backbone grad norm logged every step | Visibility into backbone vs eye proj learning rates |
| H5 aux_fused compile | Added to torch.compile submodule list | Missing compile caused recompile on every call |

---

## Epoch-by-Epoch History

| Epoch | JK (Ph1/Ph2/Ph3) ±STD | GNN% | F1 | Notes |
|-------|----------------------|------|-----|-------|
| ep1 | 0.334±0.020 / 0.329±0.018 / 0.337±0.035 | 71–79% | 0.2059 | Warm start from Run 1 ep3 |
| ep6 | 0.322±0.021 / 0.328±0.022 / 0.350±0.040 | 32–58% | 0.2555 | New best |
| ep9 | 0.350±0.053 / 0.325±0.031 / 0.325±0.065 | 49–50% | 0.2670 | Run 1 beaten |
| ep10 | 0.356±0.033 / 0.335±0.026 / 0.309±0.053 | 42–52% | 0.2787 | New best |
| ep13 | 0.343±0.043 / 0.336±0.032 / 0.320±0.060 | 37–57% | 0.3153 | **0.30 barrier broken** |
| ep15 | 0.326±0.030 / 0.340±0.036 / 0.334±0.054 | 31–49% | 0.2946 | NC-1 fired; prefix ACTIVE |
| ep21 | — | 38–43% | 0.3224 | Plateau broken; prefix working |
| ep25 | — | — | **0.3272** | New best; proj_norm ~26 |
| ep30 | 0.319±0.030 / 0.320±0.026 / 0.361±0.047 | 21–35% | 0.3166 | proj_norm 30.50 |
| ep32 | — | — | **0.3362** | **ALL-TIME BEST** |
| ep33–44 | — | — | 0.31–0.34 | Plateau — killed ep44 |

**Prefix trajectory:** proj_norm grew 16.0 → ~32 over 15 post-warmup epochs, stabilizing at ~+0.5/ep. GNN share fell from 71% to 21% as the TF eye gained signal from the now-active prefix — consistent with the prefix hypothesis: prefix carries structural context into token attention, reducing the need for pure GNN classification.

---

## What the Prefix Actually Did

Before prefix activation (ep1–14, warmup): GNN share 71–79%. The GNN carried almost all classification signal because BERT had no structural context.

After prefix activation (ep15+): GNN share fell to 21–43%. The TF eye gained relative weight because the prefix injected structural context (FUNCTION/CONSTRUCTOR/etc. embeddings) into BERT's attention. CrossAttentionFusion could then rely more on TF path for function-level semantics.

**Key evidence:** F1 ep14=0.3117 → dropped to ep15=0.2946 (prefix activation disruption) → recovered and exceeded by ep21=0.3224. This dip-then-exceed is the canonical prefix injection signal — the warmup boundary causes a brief disruption as the model re-optimizes with the new prefix input.

**proj_norm growth (16→32):** The projection layer `gnn_to_bert_proj: Linear(256, 768)` grew steadily during post-warmup training. Growth rate slowing (+0.5/ep at ep30 vs +1.5/ep at ep20) suggests the projection was converging toward a stable embedding that aligns GNN node features with BERT's token embedding space.

---

## Capacity Ceiling Analysis

**Why F1 plateaued at 0.31–0.34:**

1. **Data quality ceiling (primary):** PLAN-3A, v8-AB, and Run 4 all converge to the same region. The 41K training corpus has:
   - DoS/Reentrancy 98.6% co-occurrence → model cannot separate them
   - UnusedReturn: structural signal is semantic (return value ignored) → near-zero GNN signal
   - CEI ordering noise in Reentrancy labels
   - IntegerUO: `unchecked{}` is a code pattern invisible in CFG structure

2. **Class imbalance despite ASL:** DoS (243 positives), ExternalBug (~600), MishandledException (~800) are severely underrepresented. ASL without pos_weight does help (Run 4 >> Run 3) but doesn't fully solve imbalance at this scale.

3. **Minimal contract generalization gap:** Training corpus skews toward 100–300 line multi-function contracts. Single-function minimal contracts (5–15 nodes) produce near-uniform probability vectors because the model has no structural feature to differentiate them.

**What would break the ceiling:**
- Sol-1–Sol-8 data quality fixes (CEI ordering, OpenZeppelin negatives, SmartBugs augmentation)
- IMP-D1 re-extraction: `reextract_graphs.py` rebuilds 41K graphs with `CONTRACT→FUNCTION CONTAINS` edges (Phase-3 REVERSE_CONTAINS would then traverse up from CFG nodes through functions to contract level)
- Three-tier output (does not improve the model but properly uses its signal at inference time)

---

## Per-Class Performance at ep32

From `tune_threshold.py` results and test contract evaluation:

| Class | Tuned threshold | Val F1 (approx) | Test contracts |
|-------|----------------|-----------------|----------------|
| Timestamp | 0.30 | ~0.47 | ✓ 2/2 correct |
| Reentrancy | 0.40 | ~0.38 | Partial (co-fires CU) |
| IntegerUO | 0.50 | ~0.42 | Partial (FP on safe arithmetic) |
| GasException | 0.40 | ~0.32 | Missed on minimal contracts |
| CallToUnknown | 0.40 | ~0.25 | High FP rate |
| MishandledException | 0.40 | ~0.12 | Mostly missed |
| ExternalBug | 0.35 | ~0.10 | Near-zero (structural gap) |
| DenialOfService | 0.45 | ~0.00 | Always missed |
| TransactionOrderDependence | 0.35 | ~0.23 | Missed on minimal |
| UnusedReturn | 0.35 | ~0.00 | Always missed (semantic gap) |

---

## Next Steps

Run 4 has reached its capacity ceiling. Options in priority order:

1. **Three-tier inference output** (immediate, no retraining) — expose full probability vector
   to agent module; lower `SUSPICIOUS` tier to 0.25; agents do final reasoning.
   See: `docs/proposal/2026-05-27-three-tier-inference-output.md`

2. **IMP-D1 re-extraction** — rebuild 41K graphs with `CONTRACT→FUNCTION CONTAINS` edges.
   Phase-3 REVERSE_CONTAINS currently cannot traverse from CFG nodes up through functions to
   contract level. This is the highest-leverage structural fix remaining.

3. **Sol-1–Sol-8 data quality** — CEI ordering fix, OpenZeppelin negatives, SmartBugs augmentation.
   Primary bottleneck for DoS, UnusedReturn, ExternalBug.

4. **Phase 2 (P2)** — shared DFG edges (EXECUTION_PLAN.md). After data quality stabilizes,
   add def-use edges scoped to function-level rather than per-CFG-node.
