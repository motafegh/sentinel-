# Why Tests Missed the ExternalBug Failure — Complete Root Cause Analysis

**Date:** 2026-06-17
**Author:** Claude (Plan Onboarding, DEEP DIVE triggered by user question)
**Trigger:** E2E test of agents module on `safe_storage.sol` showed ExternalBug=0.82 false positive. User asked: "we tested the model but don't see this problem — why?"

---

## TL;DR

The validation pipeline **DID** identify the problem during Run 12 promotion:
- **FIND-R12-01:** "ExternalBug threshold-gaming signal" (C attestation)
- **FIND-R12-B01:** "65% S_only rate is NOT over-prediction for Timestamp/Reentrancy; IS class-definition mismatch for ExternalBug" (A attestation)
- **C.2.2 FP probe:** 9-contract manual inspection found "ExternalBug high-conf FP identified"

These findings were **logged, classified, and deferred to Run 13**. The model was promoted to Staging with these known issues unresolved. The clean benchmark (v0.1) that would have caught this was **skipped** because the clean test set was only 66 contracts and the C.2.1 SmartBugs Curated was 95.8% contaminated.

**The root cause of the FP is a label-mapping issue in the DIVE crosswalk:**

```yaml
# /home/motafeq/projects/sentinel/data_module/sentinel_data/labeling/crosswalks/dive.yaml
# Access Control → ExternalBug
"Access Control":         ExternalBug
```

DIVE's "Access Control" folder contains contracts with **owner patterns** (`address public owner` + `msg.sender == owner` check), which is **standard Solidity**, not a vulnerability. The crosswalk maps ALL of them to ExternalBug-positive. The model correctly learned: "owner pattern = ExternalBug" because 75% of DIVE training data is in this folder.

**This is a 3-layer problem:**
1. **DIVE labeling** (external) — Access Control is too broad a folder
2. **Crosswalk** (our code) — maps Access Control to ExternalBug (correct for SENTINEL schema, but bad input)
3. **Training/Validation** — no clean test to catch the issue

---

## Layer 1: The DIVE crosswalk maps the wrong way

**File:** `data_module/sentinel_data/labeling/crosswalks/dive.yaml` (line 38-44)

```yaml
class_map:
  Reentrancy:               Reentrancy
  DoS:                      DenialOfService
  Arithmetic:               IntegerUO
  "Time manipulation":      Timestamp
  "Front Running":          TransactionOrderDependence

  # Access Control → ExternalBug
  # DIVE's "Access Control" covers contracts where ownership, role checks,
  # or privilege enforcement are missing or bypassable by external callers.
  # This maps to ExternalBug (id=2), which is the canonical class for
  # access control violations and external interaction flaws.
  "Access Control":         ExternalBug

  "Unchecked Return Values": UnusedReturn
```

**The intent** of the mapping: any contract where ownership/role enforcement is missing should be labeled ExternalBug (access control vulnerability).

**The reality:** DIVE's "Access Control" folder is too broad. It includes:
- Contracts with **benign** `owner` patterns (just storing the deployer)
- Contracts with `require(msg.sender == owner)` checks
- Contracts with simple role-based access control
- AND actual privilege violations

When a contract like `safe_storage.sol` (which has `address public owner` + `require(msg.sender == owner, "Not owner")`) is in the "Access Control" folder, it gets labeled ExternalBug-positive **even though it's safe**.

**Effect on training data:**
- 22,073 contracts come from DIVE (98.1% of training data)
- 16,582 (75.1%) of DIVE contracts are ExternalBug-positive
- The "ExternalBug" class effectively becomes: "DIVE said this has owner/access control issues"

**The model learned this correctly.** It gives ExternalBug=0.85 to `safe_storage.sol` (owner pattern) and ExternalBug=0.00 to a contract with `to.call(data)` (no owner pattern, but a real risk).

---

## Layer 2: The validation set has the same label noise

**File:** `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/`

The validation set is a **random split of the same v3 export**. It has the same noisy labels as training:

```python
# From Run 12 epoch_summary.jsonl (ep50, the FINAL epoch)
label_dist_val: {
  "CallToUnknown": 10,
  "DenialOfService": 109,
  "ExternalBug": 1372,   # 1372/1983 = 69.2% positive
  "GasException": 0,
  "IntegerUO": 777,
  "MishandledException": 5,
  "Reentrancy": 846,
  "Timestamp": 593,
  "TransactionOrderDependence": 81,
  "UnusedReturn": 477
}
```

**The validation set is 69.2% ExternalBug-positive.** The model achieves `per_class_f1: {ExternalBug: 0.884}` because:
- Model predicts ExternalBug=1 for most contracts
- Labels say ExternalBug=1 for 69% of contracts
- They agree → F1=0.88

**The F1 metric is misleading because both the predictions and labels are biased the same way.**

The headline metric `f1_macro_tuned: 0.7004` averages across all 10 classes. The 0.88 for ExternalBug (noisy data) is offset by 0.0 for GasException (0 positives in val) and 0.30 for DenialOfService. The macro average masks the broken class.

**Additionally, `val_loss: null` is never computed during training.** Only per-class F1 and AUC metrics. The training log has no signal that would tell us "the ExternalBug class is over-confident and the labels are wrong".

---

## Layer 3: The clean benchmark was skipped

**File:** `ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/attestations/C_diagnostic_checks_attestation.md`

```
C.2.1 Smoke inference: UNVERIFIED
  → Set is 95.8% contaminated. Wild eval (47K) used as functional substitute.
```

The C.2.1 smoke test is supposed to run the model on the **clean** SmartBugs Curated benchmark. But that set is 95.8% contaminated (already in v3 training data), so the test was skipped. The "function" was filled by:
- **Wild 47K eval** — measures OOD performance, not behavior
- **v0.1 honest benchmark** (66 contracts) — DID find the FP, but was run as a separate analysis, not as a gating test

**The only clean test that ran was the v0.1 honest benchmark (66 contracts, F1=0.8743).** This is where the ExternalBug FP was actually identified — by manual inspection of 9 contracts:

> "Manual inspection: 9 contracts, 4 TP / 3 FP / 2 borderline. **ExternalBug high-conf FP identified.**"

**The FP was found but the model was already promoted to Staging.** The fix was deferred to Run 13.

---

## The unit tests don't catch model behavior

**File:** `ml/tests/`

The unit tests cover:
- `test_model.py` — model architecture (shapes, dimensions, layer connectivity)
- `test_trainer.py` — loss function, training utilities
- `test_api.py` — API integration
- `test_drift_detector.py` — drift detection
- `test_fusion_layer.py` — fusion layer shapes

**None of these probe model behavior on synthetic inputs.** There is no test like "given a contract with no external calls, does the model correctly output low ExternalBug probability?"

If such a test existed, it would have caught the bug at unit-test time. The 10 contract variations I ran in this investigation (just_owner=0.85, to_call=0.00) is exactly this kind of test.

**The smoke tests (smoke_fix1-8.py) check data infrastructure, not model behavior:**
- G1.5: Timestamp count is in expected range
- G1.6: Splits are non-empty
- G1.7: At least 2 classes have ≥10 positives

---

## What the testing pipeline DID catch

To be fair, the testing infrastructure DID identify the problem:
1. **C.1.6 Per-class F1 convergence:** "ExternalBug threshold-gaming signal" (FIND-R12-01)
2. **C.2.2 FP probe:** "ExternalBug high-conf FP identified"
3. **A attestation:** "FIND-R12-B01: 65% S_only rate is NOT over-prediction for Timestamp/Reentrancy; IS class-definition mismatch for ExternalBug"

These findings were:
- Logged in `ml/audit_docs/ISSUES.md`
- Filed in `attestations/C_diagnostic_checks_attestation.md`
- Documented in the manual inspection report
- Recorded in the L.5 session handoff
- Added to MEMORY.md

**The findings were correct. The fix was deferred.**

---

## Why the fix was deferred

The Run 13 plan (`docs/plans/2026-06-14_Run13_4_fixes_preparation.md`) includes:
1. Drop GasException (NUM_CLASSES 10→9)
2. Extend L4 to drop `loc` feature
3. Strip Solidifi `bug_*` prefix
4. Inject 658 BCCC ME contracts

**The plan does NOT include relabeling ExternalBug or fixing the DIVE crosswalk.** The fix is treated as a "Run 14" or "Phase A training" concern, not a "Run 13 critical fix" concern.

This is a **process gap**: the FP was identified, the fix was scheduled, but the urgency was underestimated. The model was promoted to Staging with a known broken class.

---

## Per-source breakdown of ExternalBug labels

| Source | Contracts | ExtBug positive | Rate |
|---|---|---|---|
| **DIVE** | 22,073 | **16,582** | **75.1%** |
| Solidifi | 283 | 39 | 13.8% |
| SmartBugs Curated | 137 | 17 | 12.4% |
| **TOTAL** | 22,493 | 16,638 | 74.0% |

**DIVE is the sole driver of the label noise.** Solidifi and SmartBugs Curated have reasonable positive rates (12-14%).

The Solidifi 13.8% rate is also explained by the same problem: Solidifi contracts are labeled for **multiple classes simultaneously** (each contract gets all 6 labels). Looking at the data:
- CallToUnknown: 39 (13.8%)
- ExternalBug: 39 (13.8%)
- IntegerUO: 49 (17.3%)
- MishandledException: 39 (13.8%)
- Reentrancy: 39 (13.8%)
- Timestamp: 39 (13.8%)
- TransactionOrderDependence: 39 (13.8%)

**All exactly 39.** This is multi-label co-occurrence — the same 39 contracts get all 6 labels. Same pattern as DIVE.

---

## What should change

### 1. Relabel DIVE ExternalBug (most important)

**Option A — Re-define ExternalBug in DIVE crosswalk** (preferred):
- Map DIVE's "Access Control" folder to a NEW class, not ExternalBug
- OR: Add a filter in the parser that requires actual external calls for ExternalBug label
- OR: Map DIVE's "Access Control" to a separate class like "AccessControl" and only label as ExternalBug when there's an actual external call

**Option B — Audit each "Access Control" contract** (most thorough):
- For each of the 16,582 contracts, check if it has:
  - `interface.method()` on typed variables
  - `address.call()` / `.delegatecall()` to non-constant addresses
  - User-supplied calldata forwarding
- Only keep ExternalBug label for contracts with these patterns

**Option C — Drop ExternalBug from DIVE entirely** (safest):
- DIVE's "Access Control" is too noisy to be useful
- Get ExternalBug labels from SmartBugs Curated (12.4% rate, more reasonable) + a new clean source
- Reduces training data but improves label quality

### 2. Add a clean test to the validation pipeline

Create a new "C.2.4 Synthetic Probes" step that:
- Tests the model on a fixed set of synthetic contracts (like the 10 I ran)
- Verifies ExternalBug probability is LOW for "just owner" and HIGH for actual external calls
- Runs as a gate before promotion, not as an optional analysis

### 3. Add label quality checks before training

- Detect over-labeled classes (positive rate > 50% is suspicious)
- Detect F1-AUC divergence (high F1 but low AUC = gaming the threshold)
- Compare label distribution across sources (DIVE vs SmartBugs Curated vs Solidifi)

---

## Timeline of discovery

| Date | Event | Reference |
|---|---|---|
| 2026-04 | DIVE crosswalk written | `data_module/sentinel_data/labeling/crosswalks/dive.yaml` |
| 2026-05 | v3 export generated with 75.1% DIVE ExternalBug | `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/` |
| 2026-06-13 | Run 12 training complete (f1_macro_tuned=0.7004) | `ml/logs/GCB-P1-Run12-v3dospatched-20260613/` |
| 2026-06-14 | Model promoted to Staging | `ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/` |
| 2026-06-15 | C.2.2 FP probe finds ExternalBug FP | `attestations/C_diagnostic_checks_attestation.md` |
| 2026-06-15 | FIND-R12-01 + FIND-R12-B01 filed | `ISSUES.md` |
| 2026-06-15 | Run 13 plan prepared, ExternalBug fix deferred | `docs/plans/2026-06-14_Run13_4_fixes_preparation.md` |
| 2026-06-17 | E2E test of agents module finds the FP in production | This investigation |
| 2026-06-17 | Root cause traced to DIVE crosswalk | This document |

---

## Summary — the user's question

> "we tested the model but don't see this problem — why?"

**The testing infrastructure identified the problem, but:**
1. The C.2.1 clean benchmark was UNVERIFIED (set was 95.8% contaminated)
2. The C.2.2 manual FP probe DID find the ExternalBug FP, but was a one-time analysis, not a gating test
3. The validation set has the same 75% ExternalBug label noise as training, so F1=0.88 looks good
4. The unit tests (test_model.py, test_trainer.py) test architecture, not behavior
5. **There is no synthetic probe test** that would have caught the broken class at unit-test time

**The fix is in the training data:** the DIVE crosswalk maps "Access Control" → "ExternalBug", which is too broad. The "Access Control" folder contains contracts with benign owner patterns, which then get labeled ExternalBug-positive and teach the model the wrong feature.

**Three actionable fixes:**
1. **DIVE crosswalk** — re-define ExternalBug mapping (don't include all "Access Control" contracts)
2. **Add synthetic probe test** — verify model gives low ExternalBug for owner-only contracts
3. **Label quality checks** — detect over-labeling before training

---

**Status:** Audit complete. No code changes made. Awaiting decision on which fix to prioritize.
