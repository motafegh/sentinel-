# Fix #5 — Re-derive labels from Slither detectors

**Effort:** 1 day (no re-extract needed; partial re-extract for some classes)
**Impact:** All 10 classes
**Risk:** Medium — replaces noisy BCCC labels with Slither's own detector outputs
**Order:** Do this AFTER all schema fixes (Fix #2, #3, #4) so the new features can re-trigger
detectors with better context.

---

## Problem (Generalisation of Finding A + B)

BCCC-SCsVul-2024's labels are noisy:
- 46% of Timestamp=1 contracts have no block.timestamp read (Finding A/B)
- 45% of IntegerUO=1 contracts may be false positives (audit estimate)
- DoS/Reentrancy co-occurrence is BCCC's own multi-labeling — but with no provenance
- CallToUnknown, ExternalBug have no on-disk structural validation

**Root cause:** BCCC's labels are derived from per-category folder placement, not from any
structural or detector-based analysis. A contract placed in `Timestamp/` gets Timestamp=1
regardless of whether it actually reads block.timestamp.

**Slither already has detectors for many of these classes** — they live in
`ml/data/slither_results/` (per the audit). Re-deriving labels from Slither's outputs gives:
- Provenance: "this contract was flagged by Slither's `timestamp` detector for THIS specific reason"
- Tighter precision: Slither detectors are less noisy than BCCC folder placement
- Cross-validation: a contract that Slither flags AND BCCC labels is high-confidence positive

---

## Source Code References

### Slither detector output location

`ml/data/slither_results/` — pre-existing per-contract detector outputs (per the audit).
Format: one JSON or text file per contract, containing detector names + findings.

### Mapping BCCC classes to Slither detectors

| BCCC Class | Slither Detector | Detector Slug |
|---|---|---|
| CallToUnknown | `low-level-calls` (when dest is `address(this)` or unknown) | `slither.detectors.reentrancy.event_access` not relevant; use `low-level-calls` |
| DenialOfService | `dos-uniswap`, `dos-gas-limit` | `slither.detectors.dos` |
| ExternalBug | not directly mapped; use HighLevelCall to non-this + missing source | manual heuristic |
| GasException | `unchecked-send`, `unchecked-transfer` | `slither.detectors.functions.void` not relevant; use `unchecked-send` |
| IntegerUO | `integer-overflow` (for pre-0.8); for 0.8+ need `unchecked-block` heuristic | `slither.detectors.arithmetic` |
| MishandledException | `unchecked-return` (a.k.a. `unchecked-send`) | `slither.detectors.functions.void` or `unchecked-send` |
| Reentrancy | `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-benign`, `reentrancy-events` | `slither.detectors.reentrancy.*` |
| Timestamp | `timestamp` | `slither.detectors.timestamp` |
| TransactionOrderDependence | not directly mapped; use block.number as proxy | manual heuristic (Slither has no TOD detector) |
| UnusedReturn | `unchecked-lowlevel` (for `.call()`/`.delegatecall()`/`.send()`) | `slither.detectors.functions.void` not relevant; use `unchecked-lowlevel` |

### Slither detector invocation

`ml/src/preprocessing/graph_extractor.py:1071-1080` — already runs Slither with
`detectors_to_run=[]` (no detectors). Need a separate script to run detectors and capture
findings.

### Current label-loading path

`ml/scripts/build_multilabel_index.py:25-35` — currently uses BCCC CSV (`BCCC-SCsVul-2024.csv`).
The new script would use a Slither-derived CSV.

---

## Fix — Re-derivation Strategy

### Phase 1: Run Slither detectors on all training contracts

```python
# ml/scripts/derive_slither_labels.py
"""
For each .sol in BCCC source dirs, run `slither --json -` and parse the detector findings.
Map detector slugs to BCCC class names. Output a new CSV with provenance metadata.
"""
import json
import subprocess
from pathlib import Path

# Detector slug → BCCC class name (10-class output)
DETECTOR_TO_CLASS = {
    "timestamp":                          "Timestamp",
    "integer-overflow":                   "IntegerUO",
    "reentrancy-eth":                     "Reentrancy",
    "reentrancy-no-eth":                  "Reentrancy",
    "reentrancy-benign":                  "Reentrancy",
    "reentrancy-events":                  "Reentrancy",
    "unchecked-send":                     "GasException",  # OR MishandledException (ambiguous)
    "unchecked-transfer":                 "GasException",
    "unchecked-lowlevel":                 "UnusedReturn",
    "unchecked-return":                   "MishandledException",
    "dos-uniswap":                        "DenialOfService",
    "dos-gas-limit":                      "DenialOfService",
    "low-level-calls":                    "CallToUnknown",  # only if dest is raw address
    "arbitrary-send-eth":                 "ExternalBug",
    "controlled-delegatecall":            "ExternalBug",
    "delegatecall-loop":                  "ExternalBug",
    # TOD has no Slither detector — derive manually from block.number pattern
}

# For each contract, run:
#   slither <path> --json - 2>/dev/null | jq '.results.detectors'
# Then for each finding, look up the detector check name in DETECTOR_TO_CLASS
# and set the corresponding class to 1.
```

### Phase 2: Merge with BCCC labels (intersection for high-confidence)

For each contract, compute:
- `slither_label[vuln] = 1` if any Slither detector fired for this class
- `bccc_label[vuln] = 1` if BCCC folder placement was for this class
- `final_label[vuln] = slither_label[vuln] OR (bccc_label[vuln] AND contract_passes_strong_filter)`

The strong filter (e.g., for Timestamp: source must contain `block.timestamp` or `now`) keeps
BCCC's high-confidence positives. Slither's labels add any contract the detector flags even if
BCCC missed it.

### Phase 3: Output CSV

```python
# Output: ml/data/processed/multilabel_index_slither.csv
# Columns: md5_stem, CallToUnknown, DenialOfService, ..., UnusedReturn, provenance_json
# provenance_json = {
#   "Timestamp": ["timestamp"],  # list of detector slugs that fired
#   "Reentrancy": ["reentrancy-eth", "reentrancy-no-eth"],
#   ...
# }
```

### Phase 4: Manual class mapping decisions

| BCCC Class | Decision |
|---|---|
| **CallToUnknown** | Use Slither `low-level-calls` filtered to raw-address targets. Drop any BCCC labels that lack raw-address calls. |
| **DenialOfService** | Slither `dos-uniswap`, `dos-gas-limit`. Keep BCCC labels ONLY if the contract has `has_loop=1` AND `external_call_count > 0` in the graph. |
| **ExternalBug** | Manual: HighLevelCall to address other than `this`, AND the target contract is not declared in the file. |
| **GasException** | Slither `unchecked-send`, `unchecked-transfer`. Drop BCCC labels without `.send()` or `.transfer()`. |
| **IntegerUO** | Slither `integer-overflow` for pre-0.8; for 0.8+ use the new `in_unchecked_block` feature (from Fix #4). |
| **MishandledException** | Slither `unchecked-return` for external calls. Drop BCCC labels without external calls. |
| **Reentrancy** | Slither `reentrancy-*` (any of 4). BCCC labels with no external call are dropped. |
| **Timestamp** | Slither `timestamp` detector (catches `now` and library wrappers). BCCC labels require source grep match. |
| **TOD** | Manual heuristic: function reads `block.number` AND has external call after. No Slither detector. |
| **UnusedReturn** | Slither `unchecked-lowlevel`. BCCC labels require `return_ignored=1` in graph. |

---

## Validation Steps

```bash
# 1. Run detector re-derivation
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/derive_slither_labels.py --workers 8
# Expected: ~30-60 min on 8 workers, processes 41,576 contracts

# 2. Compare label distributions
python -c "
import pandas as pd
old = pd.read_csv('ml/data/processed/multilabel_index.csv')
new = pd.read_csv('ml/data/processed/multilabel_index_slither.csv')
for cls in ['CallToUnknown','DenialOfService','ExternalBug','GasException',
            'IntegerUO','MishandledException','Reentrancy','Timestamp',
            'TransactionOrderDependence','UnusedReturn']:
    o, n = int(old[cls].sum()), int(new[cls].sum())
    print(f'{cls:28}  BCCC={o:>5}  Slither={n:>5}  Δ={n-o:+5d}')
"

# 3. Spot-check 10 contracts per class: do the new labels match Slither's actual findings?
# (Manual review — see docs/run8-audit.md for the audit methodology.)
```

---

## Expected Impact

| Class | BCCC (current) | Slither-derived (expected) | Notes |
|---|---|---|---|
| Timestamp | 1,901 | 700-900 | Drop 46-63% noise |
| IntegerUO | 13,559 | 8,000-10,000 | Drop 25-40% noise |
| Reentrancy | 3,886 | 2,500-3,000 | Drop 25-35% noise |
| DoS | 372 | 150-250 | Drop 33-60% noise |
| Others | varies | varies | Generally tighter, fewer false positives |

**The model trained on cleaner labels should achieve higher precision (fewer false positives)
at the cost of slightly lower recall. The Run 8 audit showed precision was the bottleneck —
9/10 classes degenerate on test. This fix targets precision directly.**

---

## Risk Assessment

**MEDIUM.** This is a label-space change, not a feature-space change:
1. All v8 cached training runs are invalidated
2. Manual class mapping decisions need validation (especially GasException vs
   MishandledException which both use `unchecked-return` in some Slither versions)
3. TOD has no Slither detector — manual heuristic may have its own biases

**Mitigation:** keep both CSVs in `ml/data/processed/`:
- `multilabel_index_bccc.csv` — original BCCC labels (frozen for reference)
- `multilabel_index_slither.csv` — new Slither-derived labels (use for Run 9)

If Run 9 regresses, fall back to BCCC + Fix #1 cleaning.

---

## Files Changed / Created

| File | Change |
|---|---|
| `ml/scripts/derive_slither_labels.py` | NEW: run Slither detectors + map to BCCC classes |
| `ml/data/processed/multilabel_index_slither.csv` | NEW: Slither-derived labels |
| `ml/data/processed/multilabel_index_bccc.csv` | NEW: snapshot of BCCC labels (frozen) |
| `ml/data/splits/slither_*/` | NEW: stratified splits on new labels |
| `docs/label-provenance.md` | NEW: documents the per-class mapping decisions |

---

## Open Question: How to Handle Multi-Label Conflicts?

The current `build_multilabel_index.py` groups by SHA256 and ORs the per-folder class labels.
This gives multi-label contracts (e.g., a contract that is both DoS and Reentrancy). With
Slither-derived labels, the same SHA256 may be flagged by `dos-uniswap` AND `reentrancy-eth`
because Slither runs all detectors on each contract — same OR semantics apply, no change needed.
