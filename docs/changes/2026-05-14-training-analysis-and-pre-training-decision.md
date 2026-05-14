# v5.1 Training Analysis & Pre-Training Decision — 2026-05-14

## Context

This session reviewed the complete v5.1-fix28 training run (using manually saved early-epoch
logs for epochs 1-26 and the resumed-run log for epochs 27-54), audited the external GNN
Enhancement Proposal v2.0 against the current source code, and decided on a comprehensive
pre-training improvement plan before any future training is launched.

---

## v5.1-fix28 Run: Full Analysis

### Timeline

| Phase | Epochs | Notes |
|---|---|---|
| Fresh start | 1–10 | System crash mid-ep 10 batch 685/1947 |
| Resume 1 | 10–11 | Crash again at ep 11 batch 483/1947 |
| Resume 2 | 11–20 | System crash during ep 20 |
| Resume 3 | 20–26 | Crash mid-ep 26 batch 1013/1947 |
| Resume 4 | 27–54 | Completed. Scheduler SKIPPED (total_steps mismatch 20941 vs 16558) |

Final best checkpoint: epoch 53, F1-macro=0.2794, patience_counter=8/10.

### GNN Gradient Collapse — Confirmed

The GNN eye collapsed by epoch 8. Gradient share timeline from optimizer-step logs:

| Epoch | gnn_eye | tf_eye | GNN share (approx) |
|---|---|---|---|
| 1 | 0.601 | 0.170 | ~65% |
| 2 | 0.658 | 0.054 | ~75% |
| 4 | 0.148 | 0.151 | **~45% — crossing over** |
| 7 | 0.125 | 0.222 | **~30% — TF dominant** |
| 8 | 0.104 | 0.219 | ~25% |
| 19 | 0.064 | 0.141 | **~10% — collapsed** |
| 24 | 0.034 | 0.119 | **~10% — dead** |
| 27+ | 0.086 | 0.115 | ~10% through end |

By epoch 10 the GNN eye was functionally passive. The entire resumed portion (ep 27–54)
trained with a dead GNN at frozen LR (scheduler skipped).

### F1 Trajectory and Class Learning

| Epoch | F1-macro | Classes learning (F1 > 0.1) |
|---|---|---|
| 1 | 0.0012 | IntegerUO=0.012 only |
| 4 | 0.1340 | IntegerUO, GasException |
| 9 | 0.1694 | IntegerUO, GasException, MishandledException |
| 18 | 0.2345 | IntegerUO, GasException, MishandledException |
| 25 | 0.2478 | As above + partial TOD/Timestamp/DoS |
| 53 | 0.2794 | IntegerUO (0.72), GasException (0.40), MishandledException (0.30), Reentrancy (0.19) |

Six classes (Timestamp, TOD, ExternalBug, CallToUnknown, DoS, UnusedReturn) never exceeded 0.13
in any epoch. They require GNN structural signal to differentiate — once the GNN collapsed, the
transformer alone could not learn them reliably.

### Confirmed Root Causes of This Run

1. **GNN gradient collapse (L3):** The structural root cause — only h4 available for pooling,
   over-smoothed, Phase 2 execution-order signal buried. Leads to TF eye dominance by epoch 4-8.

2. **Shared CONTAINS embedding (L2):** Phase 3 uses same type-5 embedding for reversed CONTAINS,
   reducing the quality of Phase 3 aggregation and contributing to GNN weakness.

3. **Single CONTROL_FLOW hop (L1):** Diameter-4+ CFGs (complex reentrancy patterns) can't be
   distinguished from safe patterns. GNN provides weak structural signal even when active.

4. **Scheduler skipped on resume:** `total_steps mismatch (20941 vs 16558)` — LR frozen at
   mid-schedule value for the entire 28-epoch resumed portion. Prevented any meaningful late-
   training learning dynamics.

5. **Multiple crash/resume cycles:** Different code versions and pos_weight recomputations across
   restarts created an inconsistent optimization trajectory. The run is not reproducible.

### Decision on Previous Metrics

**No metrics from any previous training run (v4, v5.0, v5.1-fix28) are valid baselines.**

- v4 and v5.0 trained on leaky 68K dataset (34.9% cross-split contamination)
- v5.1-fix28 trained with a collapsed GNN and broken scheduler
- F1=0.5828 (v5.0) was inflated by leakage; F1=0.2794 (v5.1-fix28) reflects a broken run

The behavioral test suite (`ml/scripts/manual_test.py`) is the real judge.

---

## External Proposal Audit: GNN Enhancement v2.0

### Confirmed Accurate Against Source Code

| Claim | Code Location | Status |
|---|---|---|
| L4: `gnn_layers != 4` hard raise | `trainer.py:233-237` | ✅ Exact match |
| L2: reversed CONTAINS reuses type-5 embedding | `gnn_encoder.py:280` | ✅ Exact match |
| L1: single CONTROL_FLOW hop (conv3 only) | `gnn_encoder.py:296-301` | ✅ Exact match |
| detach bug in `return_intermediates` | `gnn_encoder.py:294,303,313` | ✅ Critical, confirmed |
| `test_gnn_encoder.py` uses `randn(n_nodes, 8)` | `ml/tests/test_gnn_encoder.py:23` | ✅ Bug confirmed |
| 3 intermediate representations (not 4) | dict keys in `gnn_encoder.py` | ✅ after_phase1/2/3 |
| PyG ≥ 2.4 for JumpingKnowledge | installed 2.7.0 | ✅ Satisfied |

### One Discrepancy

- Section 2.4 states ~295K trainable LoRA params; actual is ~589,824 (r=16, Q+V, 12 layers:
  4 × 16 × 768 × 12 × 2 projections = 1,179,648 / 2 ≈ 589,824). The proposal appears to
  have used r=8 for the estimate. Not a correctness issue — the code is right.

### Accepted Recommendations

- JK Connections (attention mode) ✅
- Per-phase LayerNorm ✅
- REVERSE_CONTAINS embedding ✅ — but re-extraction NOT required (see proposal)
- TrainConfig constraint relaxation ✅
- Live intermediates (not detached) for JK ✅
- Tuple version comparison for checkpoints ✅
- Non-negotiable gradient flow test ✅

---

## Decision

A comprehensive pre-training improvement plan has been written covering all architecture,
pipeline, data, monitoring, and test improvements before the next training run.

**No training will be launched until all pre-training gates are cleared.**

See: `docs/proposals/SENTINEL_PRE_TRAINING_IMPROVEMENT_PLAN_v1.md`

---

## Files Changed

None — documentation session only. All implementations deferred to the pre-training plan.
