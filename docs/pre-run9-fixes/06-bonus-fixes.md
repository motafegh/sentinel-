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

- `ml/src/inference/predictor.py:150` — `TIER_CONFIRMED_THRESHOLD: float = 0.55` (hardcoded)
- `ml/src/inference/predictor.py:151` — `TIER_SUSPICIOUS_THRESHOLD: float = 0.25` (hardcoded)
- `ml/src/inference/predictor.py:660-757` — `_format_result()` implementation
- `ml/src/inference/predictor.py:710-715` — tier dispatch logic that ignores `self.thresholds`
  (per-class tuned values from `_thresholds.json`)
- `ml/checkpoints/*_thresholds.json` — per-class tuned thresholds (already saved by trainer)

The actual system is THREE-tier, not two-tier. CONFIRMED (>= 0.55) | SUSPICIOUS (0.25 <= p < 0.55)
| NOTEWORTHY (p < 0.25). The current `_format_result` only consults the two class-level tier
thresholds, never the per-class tuned `self.thresholds`.

### Fix

The fix must preserve the three-tier output structure. Replace per-class-aware tier comparison:

```python
# ml/src/inference/predictor.py:_format_result
# Replace hardcoded tier thresholds with per-class tuned thresholds + 3-tier output.

def _format_result(self, graph, probs, tokens, windows_used):
    probs_cpu = probs.cpu().tolist()
    confirmed:  list[dict] = []
    suspicious: list[dict] = []
    for cls_name, prob in zip(self._class_names, probs_cpu):
        tuned = self.thresholds.get(cls_name, self.DEFAULT_THRESHOLD)
        # Per-class tier offset: classes with a higher tuned threshold get a
        # proportionally higher CONFIRMED cutoff.
        confirmed_cutoff  = max(self.tier_confirmed_threshold,  tuned + 0.20)
        suspicious_cutoff = max(self.tier_suspicious_threshold, tuned)
        if prob >= confirmed_cutoff:
            confirmed.append({"vulnerability_class": cls_name, "probability": prob, "tier": "CONFIRMED"})
        elif prob >= suspicious_cutoff:
            suspicious.append({"vulnerability_class": cls_name, "probability": prob, "tier": "SUSPICIOUS"})
    # ... rest of method unchanged
```

### Validation

```bash
# Run manual_test.py on 12_safe_contract.sol and verify no detections above per-class threshold
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/archive/manual_test.py \
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

- `ml/scripts/test_contracts/` — 20 synthetic benchmarks (verified: `01_reentrancy_classic.sol`
  through `20_unused_return_minimal.sol`)
- `ml/data/smartbugs-curated/dataset/` — 143 real contracts, hand-labeled
  (verified: 143 .sol files across 10 category subdirs)
- `ml/scripts/archive/manual_test.py` — current evaluator (file does NOT exist at
  `ml/scripts/manual_test.py` — only in `archive/`)

### Fix

Create `ml/scripts/manual_test_smartbugs.py` that mirrors `archive/manual_test.py` but iterates
over `ml/data/smartbugs-curated/`:

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

`ml/scripts/train.py:219-220` (CLI flag definition):
```python
p.add_argument("--drop-complexity-feature", action="store_true", default=False,
               dest="drop_complexity_feature")
```

`ml/src/models/gnn_encoder.py:168` (model flag):
```python
drop_complexity:          bool           = False,  # Run 8: zero feat[5] to break complexity-proxy
```

`ml/src/models/gnn_encoder.py:435-437` (forward pass application):
```python
if self.drop_complexity:
    x = x.clone()
    x[:, 5] = 0.0
```

This drops feat[5] from the GNN input during training. Run 8 also used `--appnp-alpha`
(train.py:227-228, default 0.0) to add APPNP-style smoothing — the actual Run 8 value is not
recorded in any visible file, would need to grep the Run 8 launcher script.

### Documentation TODO

Add a one-paragraph note to a future architecture-decisions doc (does NOT exist yet at
`docs/architecture-decisions.md`) explaining:
1. Why complexity was dropped
2. The hypothesis (feat[5] correlates with node count, model was using it as size proxy)
3. What to do if Run 9 still shows size bias (try increasing `--appnp-alpha` or adding
   size normalization in graph_extractor)

Note: a previously proposed `gnn_jk_entropy_reg_lambda` flag does NOT exist in train.py,
sentinel_model.py, or gnn_encoder.py. Do not reference it.

---

## Summary — bonus fixes don't block Run 9

These are quality-of-life improvements. Run 9 can launch after Fix #1-#5 are applied without
touching any of #6-#8. They're listed here so they don't get lost.
