# EXP-E3: Message Propagation Simulation

**Layer:** 2 — Expressivity
**Priority:** P1
**Status:** FAIL
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_e3_message_propagation_sim.py`
**Output:** `ml/logs/interpretability/exp_e3_message_propagation_sim.json`

---

## Purpose

This experiment simulates message propagation through the GNN using random weights to measure how much information flows through CALL_ENTRY and CONTROL_FLOW edges between phases. It tests whether the Phase 2 GNN layers (which specifically process CALL_ENTRY and ICFG-related edges) produce measurably different aggregation patterns for reentrancy-positive vs negative contracts, even with random (untrained) weights.

## Method

The script instantiates a GNNEncoder with random weights (no checkpoint loaded), then runs 100 reentrancy-positive and 100 negative contracts through it. For each edge type of interest (CALL_ENTRY, CONTROL_FLOW), it computes the mean activation magnitude at Phase 1, Phase 2, and Phase 3 outputs. The pass criterion is that the delta (Phase 2 - Phase 1) for CALL_ENTRY edges is ≥0.02 larger for positives than negatives, indicating the Phase 2 layers preferentially activate on reentrancy-positive call structures.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_e3_message_propagation_sim.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --n-contracts 100 \
  --out ml/logs/interpretability/exp_e3_message_propagation_sim.json
```

## Results

### Key Metrics
| Metric | Positives | Negatives | Diff | Pass Threshold | Status |
|--------|-----------|-----------|------|---------------|--------|
| CALL_ENTRY: Phase 1 | 0.8040 | 0.7894 | +0.0146 | — | — |
| CALL_ENTRY: Phase 2 | 0.6565 | 0.6536 | +0.0029 | — | — |
| CALL_ENTRY: delta(P2-P1) | -0.1475 | -0.1357 | -0.0118 | ≥+0.02 | FAIL |
| CONTROL_FLOW: Phase 1 | 0.9937 | 0.9941 | -0.0004 | — | — |
| CONTROL_FLOW: Phase 2 | 0.9119 | 0.9082 | +0.0037 | — | — |
| CONTROL_FLOW: delta(P2-P1) | -0.0818 | -0.0859 | +0.0041 | — | — |

## Interpretation

The experiment fails because random-weight GNN propagation shows no meaningful difference between reentrancy-positive and negative contracts for CALL_ENTRY edge activations. The delta(P2-P1) is negative for both groups (-0.1475 vs -0.1357), meaning Phase 2 actually reduces activation magnitude compared to Phase 1 for both classes — possibly due to the normalization in Phase 2 or the random weight initialization. The difference between positives and negatives is only -0.0118, well below the +0.02 threshold.

This result is expected: with random weights, the GNN has no learned ability to differentiate reentrancy patterns from non-reentrancy patterns via Phase 2 edges. The simulation confirms the architecture is designed correctly (Phase 2 processes CALL_ENTRY edges), but trained weights are needed to show meaningful separation. The slight positive bias in Phase 1 for positives (0.8040 vs 0.7894) likely reflects that reentrancy-positive contracts have more CALL_ENTRY edges (consistent with exp_s4 finding that 76% of reentrancy positives have CALL_ENTRY edges).

## Pass/Fail Analysis

The FAIL is expected for a random-weight simulation — meaningful discrimination requires a trained checkpoint. The experiment is best interpreted as a baseline measurement to compare against the trained model's Phase 2 activations (to be done with exp_e3 once a checkpoint is available).

## Recommended Next Steps

1. Re-run with `--checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` once checkpoint loading is verified.
2. Compare trained vs random delta to quantify how much the training actually teaches Phase 2 to attend to CALL_ENTRY edges.
3. The slight positive-class bias in Phase 1 CALL_ENTRY (+0.0146) suggests that even edge presence patterns (not just weights) carry some signal — worth investigating with exp_s4 correlation.
