# EXP-L10: Training Ablation Command Generator

**Layer:** 3 — Learning
**Priority:** P2
**Status:** PASS
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_l10_training_ablation.py`
**Output:** `ml/logs/interpretability/exp_l10_training_ablation.json/`

---

## Purpose

This experiment generates ready-to-run training ablation commands for systematically removing each edge type from the graph and measuring the resulting F1 drop. It identifies which edge types are structurally critical to model performance versus which are inert. This is a planning and command-generation tool — it does not train models itself.

## Method

The script enumerates all 11 edge types defined in the SENTINEL schema (UNKNOWN, AST_PARENT_OF, CONTAINS, INHERITS, USES, CALL, CONTROL_FLOW, RETURN_TO, CALL_ENTRY, DATA_FLOW_DEP_OF, REACHES) and generates one training command per edge type using the Run 4 hyperparameter configuration. It also creates a CSV tracking template for recording results. Note: `--ablate-edge-type` flag is not yet implemented in `train.py`; the script documents the required one-line addition.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_l10_training_ablation.py \
  --out ml/logs/interpretability/exp_l10_training_ablation.json
```

## Results

Generated 12 ablation run commands (1 baseline + 11 edge-type ablations).

### Key Metrics
| Metric | Value | Pass Threshold | Status |
|--------|-------|---------------|--------|
| Commands generated | 12 | 12 | PASS |
| Output files | 3 (JSON, txt, CSV) | > 0 | PASS |

**Estimated impact by edge type (from script annotations):**
| Edge Type | ID | Estimated F1 Impact |
|-----------|----|---------------------|
| CONTAINS | 2 | -0.03 to -0.06 |
| CONTROL_FLOW | 6 | -0.05 to -0.10 |
| CALL | 5 | -0.03 to -0.07 |
| CALL_ENTRY | 8 | -0.02 to -0.04 |
| RETURN_TO | 7 | -0.02 to -0.04 |
| DATA_FLOW_DEP_OF | 9 | -0.02 to -0.05 |
| REACHES | 10 | -0.01 to -0.03 |
| INHERITS | 3 | -0.01 to -0.03 |
| UNKNOWN | 0 | Neutral |
| AST_PARENT_OF | 1 | Neutral |
| USES | 4 | Neutral |

## Interpretation

This command generator provides the full training ablation study plan for Run 4's architecture. The script correctly notes that inference ablation (exp_l2) is a faster proxy for these results without requiring full retraining. The annotations suggest CONTROL_FLOW and CALL edges will have the highest impact on vulnerability detection, consistent with the fact that reentrancy, DoS, and timestamp manipulation all fundamentally depend on control-flow paths. The CONTAINS edge is critical for contract-level hierarchy propagation in Phase 3.

## Pass/Fail Analysis

All criteria passed:
- 12 well-formed commands generated with correct Run 4 hyperparameters.
- CSV tracking template created for result collection.
- `--ablate-edge-type` flag clearly documented as needing implementation before execution.

## Recommended Next Steps

1. Implement `--ablate-edge-type` in `train.py` (one-line trainer patch documented in script).
2. Run exp_l2 (inference ablation, no retraining) first as a fast proxy.
3. After exp_l2 results, prioritize full training ablation for the 2-3 edge types with largest inference-time impact.
4. Use the generated CSV template to track results across ablation runs.
