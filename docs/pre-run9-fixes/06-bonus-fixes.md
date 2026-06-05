# Bonus Fixes #6, #7, #8 — Cleanup (not blocking Run 9)

**Effort:** 30 min + 1 hr + 0 hr
**Impact:** Eval methodology and display
**Risk:** Very low
**Order:** Do these anytime.

---

## Fix #6 — Predictor tier-threshold bug

**Effort:** 30 minutes
**Impact:** Manual eval display only (no model change)

### Problem (Finding H from audit)

`ml/src/inference/predictor.py:_format_result()` uses hardcoded `TIER_CONFIRMED_THRESHOLD = 0.55`
to display "confirmed" vs "suspect" tiers. The per-class tuned thresholds from
`{ckpt}_thresholds.json` are loaded into `self.thresholds` but never consulted in tier logic.

**Effect:** `manual_test.py` "Detected" column hides true positives where prob is 0.30-0.55
but above the per-class tuned threshold. Run 8 manual test showed 14/19 hits but display said
"0/19" because everything fell below the 0.55 hardcoded tier cutoff.

### Source Code References

- `ml/src/inference/predictor.py:150-151` — `TIER_CONFIRMED_THRESHOLD = 0.55` (hardcoded)
- `ml/src/inference/predictor.py:712-715` — tier dispatch logic that ignores `self.thresholds`
- `ml/checkpoints/*_thresholds.json` — per-class tuned thresholds (already saved by trainer)

### Fix

```python
# ml/src/inference/predictor.py
# Replace the hardcoded tier with per-class threshold comparison:

def _format_result(self, contract_name, logits):
    probs = torch.sigmoid(logits).squeeze().tolist()
    detections = []
    for cls_idx, cls_prob in enumerate(probs):
        tuned_thresh = self.thresholds.get(CLASS_NAMES[cls_idx], 0.5)
        if cls_prob >= tuned_thresh:
            detections.append({
                "class": CLASS_NAMES[cls_idx],
                "prob": cls_prob,
                "threshold": tuned_thresh,
                "tier": "confirmed" if cls_prob >= max(tuned_thresh + 0.2, 0.55) else "suspect",
            })
    return {"contract": contract_name, "detections": detections}
```

### Validation

```bash
# Run manual_test.py on 12_safe_contract.sol and verify no detections above per-class threshold
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/manual_test.py \
  --checkpoint ml/checkpoints/GCB-P1-Run8-v10-20260605_best.pt \
  --contract ml/scripts/test_contracts/12_safe_contract.sol
# Expected: 0 detections (was: 5+ classes > 0.55)
```

---

## Fix #7 — Add manual_test_smartbugs.py benchmark

**Effort:** 1 hour
**Impact:** Eval methodology — replaces OOD synthetic contracts with in-distribution real contracts

### Problem (Finding G from audit)

The 20 synthetic test contracts in `ml/scripts/test_contracts/` are massively OOD:
- Median 20 nodes / 40 edges (training: 90 / 258 — bottom 7th percentile)
- All use Solidity 0.8+ with `unchecked{}` blocks (training: 87.9% pre-0.8)
- 5 contracts (reentrancy_min, tod_min, integer_simple, etc.) are in the bottom 1st percentile

**They exaggerate brokenness.** The model appears to fail because the test contracts don't
match the training distribution, not because the model is broken.

### Source Code References

- `ml/scripts/test_contracts/20_*.sol` — synthetic benchmarks
- `ml/data/smartbugs-curated/` — 143 real contracts, hand-labeled, pre-0.8 Solidity
- `ml/scripts/manual_test.py` — current evaluator

### Fix

Create `ml/scripts/manual_test_smartbugs.py` that mirrors `manual_test.py` but iterates over
`ml/data/smartbugs-curated/`:

```python
# ml/scripts/manual_test_smartbugs.py
"""
Run model on SmartBugs Curated dataset (143 real contracts, 41 LOC median).
Reports per-category accuracy and macro-F1.
"""
from pathlib import Path
import pandas as pd

SMARTBUGS_ROOT = Path("ml/data/smartbugs-curated")
CATEGORIES = ["reentrancy", "arithmetic", "denial_of_service", "time_manipulation",
              "unchecked_low_level_calls", "front_running", "access_control",
              "bad_randomness", "short_addresses", "other"]

# Map SmartBugs category → BCCC class
SB_CAT_TO_BCCC = {
    "reentrancy": "Reentrancy",
    "arithmetic": "IntegerUO",
    "denial_of_service": "DenialOfService",
    "time_manipulation": "Timestamp",
    "unchecked_low_level_calls": "CallToUnknown",
    "front_running": "TransactionOrderDependence",
    "access_control": None,  # not in BCCC
    "bad_randomness": None,
    "short_addresses": None,
    "other": None,
}

# For each contract, run model, record (true_class, top_pred, top_prob, all_probs)
# Compute per-category precision/recall/F1
# Report confusion matrix
```

### Validation

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/manual_test_smartbugs.py \
  --checkpoint ml/checkpoints/GCB-P1-Run8-v10-20260605_best.pt
# Expected output: per-category accuracy table
# Run 8 prediction (current best):
#   reentrancy → 4/5 correct top-1
#   arithmetic → 0/2 correct (predicted Reentrancy)
#   time_manipulation → 0/1 correct (predicted IntegerUO)
#   denial_of_service → 0/1 correct (model has DoS bias)
```

---

## Fix #8 — Document complexity-proxy bias fix

**Effort:** 0 hours (already applied in Run 8)
**Impact:** None (already done) — pure documentation

### Context

Run 7 audit (L4 finding) showed the model uses `complexity` (feat[5]) as a complexity proxy —
large contracts fire higher probabilities across all classes regardless of true signal.

### Already Applied

`ml/scripts/train.py:91-92` (current state) — Run 8 used:
```python
--drop-complexity-feature   # in train.py / TrainConfig (need to verify exact flag name)
```

This drops feat[5] from the GNN input during training. Run 8 also used `--appnp-alpha 0.2`
to add APPNP-style smoothing.

### Documentation TODO

Add a one-paragraph note to `docs/architecture-decisions.md` explaining:
1. Why complexity was dropped
2. The hypothesis (feat[5] correlates with node count, model was using it as size proxy)
3. What to do if Run 9 still shows size bias (try `gnn_jk_entropy_reg_lambda` increase or
   additional size normalization in graph_extractor)

---

## Summary — bonus fixes don't block Run 9

These are quality-of-life improvements. Run 9 can launch after Fix #1-#5 are applied without
touching any of #6-#8. They're listed here so they don't get lost.
