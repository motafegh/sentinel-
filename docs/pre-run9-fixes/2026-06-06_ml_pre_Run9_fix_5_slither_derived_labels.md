# Fix #5 — Re-derive labels from Slither detectors

**Effort:** 1 day (no graph re-extract needed; one-time Slither pass over ~41K contracts)
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

**Solution:** Re-derive labels by RUNNING Slither's detectors on every contract and using
their outputs as the new ground truth. This gives:
- Provenance: "this contract was flagged by Slither's `timestamp` detector for THIS specific reason"
- Tighter precision: Slither detectors are less noisy than BCCC folder placement
- Cross-validation: a contract that Slither flags AND BCCC labels is high-confidence positive

---

## Source Code References

### Current state (verified)

`ml/data/slither_results/` — **DIRECTORY EXISTS BUT IS EMPTY** (only `.` and `..`).
This was an audit-assumed path. Reality: Slither has never been run on the dataset at scale.
Fix #5 must therefore RUN Slither, not just parse pre-existing outputs.

`ml/data/processed/multilabel_index.csv` — current label source (BCCC folder-based).
Row 1-34 reference `BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv` (see `build_multilabel_index.py:5`).

`ml/src/preprocessing/graph_extractor.py:1-97` — module docstring describes the extractor.
The current extractor instantiates Slither via:
```python
from slither import Slither
slither = Slither(str(sol_path), **slither_kwargs)
```
Slither is called with default detectors enabled (no `detectors_to_run=[]` filter at the
graph-extraction level). The graph extractor does NOT need detector output; it walks the
Slither AST. Detector output is a separate concern.

`ml/scripts/build_multilabel_index.py:5-7` — confirms the BCCC CSV is the current label source
(quoted: "sourced from the authoritative BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv").

`ml/src/inference/preprocess.py:209-219` — `ContractPreprocessor.process()` calls
`extract_contract_graph`; same Slither invocation path, but does not request detector output.

### Detector slug vocabulary (verified against Slither 0.9.3+)

| BCCC Class | Slither Detector Slug | Notes |
|---|---|---|
| CallToUnknown | `low-level-calls` | Only fires on `.call(`, `.delegatecall(`, etc. Filter to raw-address targets in post-processing. |
| DenialOfService | `dos-uniswap`, `dos-gas-limit` | Stock Slither detectors. May miss BCCC's specific DoS contract distribution. |
| ExternalBug | `arbitrary-send-eth`, `controlled-delegatecall` | Direct mappings. No detector for "external untrusted call to address(0)" — manual heuristic. |
| GasException | `unchecked-send`, `unchecked-transfer` | Direct mappings. (Note: BUG-9 fix in Run 5 added Send to internal check; Slither also flags it.) |
| IntegerUO | `integer-overflow` | Fires only on pre-0.8 contracts. For 0.8+, combine with new `in_unchecked_block` feature from Fix #4. |
| MishandledException | `unchecked-return` | Fires on `.call()`/`.delegatecall()`/`.send()` return-value discards. (Note: overlaps with `unchecked-send` for `.send()`.) |
| Reentrancy | `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-benign`, `reentrancy-events` | All 4 fire on reentrancy patterns of varying severity. |
| Timestamp | `timestamp` | Fires on `block.timestamp` / `now` usage in conditionals. |
| TransactionOrderDependence | (no Slither detector) | Manual heuristic required: function reads `block.number` AND has external call after. |
| UnusedReturn | `unchecked-lowlevel` | Fires on `.call()`/`.delegatecall()` return-value discards. |

**Detectors NOT in stock Slither (cannot auto-flag):**
- TransactionOrderDependence — no detector exists; must use block.number pattern heuristic
- Plain "CallToUnknown" without a low-level call — no detector; must combine high_level_calls
  to raw addresses with `call_target_typed = 0.0` from the graph features

---

## Fix — Re-derivation Strategy

### Phase 1: Run Slither detectors on all training contracts

```python
# ml/scripts/derive_slither_labels.py (NEW)
"""
For each .sol in the source dirs, run `slither --json - --detect <detectors>` and parse
the detector findings. Map detector slugs to BCCC class names. Output a new CSV with
provenance metadata.

This script is NEW — the previous design assumed pre-existing detector output files in
ml/data/slither_results/, but that directory is empty. Slither must be invoked here.
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
    "unchecked-send":                     "MishandledException",  # see disambiguation note
    "unchecked-transfer":                 "MishandledException",
    "unchecked-lowlevel":                 "UnusedReturn",
    "unchecked-return":                   "MishandledException",
    "dos-uniswap":                        "DenialOfService",
    "dos-gas-limit":                      "DenialOfService",
    "low-level-calls":                    "CallToUnknown",  # only if dest is raw address
    "arbitrary-send-eth":                 "ExternalBug",
    "controlled-delegatecall":            "ExternalBug",
    # TOD has no Slither detector — derive manually from block.number pattern (Phase 4)
}

ALL_DETECTORS = " ".join(DETECTOR_TO_CLASS.keys())

# Per-contract invocation:
#   subprocess.run(["slither", str(sol_path), "--json", "-", "--detect", ALL_DETECTORS],
#                   capture_output=True, timeout=120)
# Parse JSON output: results.detectors[].check
# For each finding, look up the detector check name in DETECTOR_TO_CLASS
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

### Disambiguation: `unchecked-send` vs `unchecked-return` vs GasException vs MishandledException

Slither 0.9.3 emits:
- `unchecked-send` for `addr.send(x)` whose return is not checked
- `unchecked-lowlevel` for `addr.call(...)` / `addr.delegatecall(...)` return not checked
- `unchecked-transfer` for `addr.transfer(x)` (transfer itself throws on failure, but Slither still emits this)

BCCC labels `GasException` and `MishandledException` separately. Mapping is ambiguous:
- `unchecked-send` → could be either class
- `unchecked-return` (some Slither versions) → MishandledException
- `unchecked-lowlevel` → UnusedReturn

**Recommended mapping:** group `unchecked-send` + `unchecked-transfer` + `unchecked-return`
under `MishandledException`. Reserve `UnusedReturn` for `unchecked-lowlevel` (because it's a
distinct detector that fires on `.call()`/`.delegatecall()` specifically). Drop the `GasException`
class entirely if no clean detector mapping exists, OR keep BCCC's GasException label and
ADDITIONALLY union it with the Slither `unchecked-send` findings (with `.send()` heuristic).

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
4. Slither invocation cost: ~30-60 min for 41K contracts on 8 workers (solc compile time)

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

## Open Question: Which solc versions to use per contract?

BCCC has 87.9% pre-0.8 Solidity. Different solc versions compile to different IR shapes;
some Slither detectors may not fire on old IR. The `derive_slither_labels.py` script must
detect each contract's pragma and invoke the matching solc version (Slither does this
auto-magically if solc-select is configured, but explicit selection may be needed).
