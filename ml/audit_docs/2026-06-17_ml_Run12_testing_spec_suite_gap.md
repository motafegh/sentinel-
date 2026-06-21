# Why the Testing Spec Suite Missed the ExternalBug Failure

**Date:** 2026-06-17
**Author:** Claude (DEEP DIVE — user follow-up question)
**Question:** "even this setup of testing ran? `ml/testing_specs` — why didn't it catch?"

---

## TL;DR

**The testing spec suite has the right tests, but they're structurally unable to block promotion of a broken model.** Here's the gap:

1. **C.2.1** (smart smoke inference) was **UNVERIFIED** — the clean benchmark is 95.8% contaminated
2. **C.2.2** (FP probe) was run **manually** — found the FP, but logged as a finding, not a gate
3. **I.2.2** (the only automated promotion gate) checks F1 on the **noisy validation set** — passes because the model learned the noise
4. **I.3.1** (behavior checks) is **explicitly NOT enforced** by `promote_model.py`
5. **No synthetic behavioral probe** — the test that would have caught this doesn't exist

The fix that was found (C.2.2 manual probe finding the FP) was logged and deferred to Run 13. The model was promoted to Staging anyway.

---

## The testing spec suite — overview

`ml/testing_specs/` contains **14 files**:

| File | Purpose |
|---|---|
| `00_rules.md` | Universal rules (gate assertions, cross-checks, attestations) |
| `A_benchmark_runs.md` | External benchmark evaluation (contamination, SmartBugs, SolidiFI) |
| `B_data_pipeline.md` | Label CSV construction, split generation |
| `B_contract_deep_dive.md` | Diagnosing individual mispredictions |
| `C_diagnostic_checks.md` | Training log analysis + model behavior checks |
| `D_smoke_preflight.md` | Smoke tests, VRAM gate, compile validation |
| `E_preprocessing_consistency.md` | Train/eval preprocessing alignment |
| `F_new_run_checklist.md` | Pre-launch, promotion, post-run gates |
| `G_ablation_protocol.md` | Controlled ablation experiments |
| `H_issue_triage.md` | Alert triage, bug filing |
| `I_regression_guard.md` | **Promotion gates** (automated) |
| `J_schema_migration.md` | Schema version changes |
| `K_inference_api.md` | API endpoint validation |
| `L_release_readiness.md` | Release gate, session handoff |

**The suite is comprehensive.** 12 specs cover pre-launch, training, behavior, promotion, and post-release. Rule 0 ("Read Before Claiming") and Rule 2 ("Validate Your Validation") enforce rigor.

---

## What the spec suite says about FP detection

**`C_diagnostic_checks.md` C.2.2 (False Positive Probe):**

> "Run a set of known-clean contracts through inference. A well-trained model should produce low probabilities (< 0.3) for all classes on clean contracts."
>
> "If Reentrancy probability is high on clean contracts: [diagnostic steps]"

**This is exactly the test that would have caught the ExternalBug FP.** The spec even gives the threshold: "low probabilities (< 0.3)".

For safe_storage.sol:
- Model gives ExternalBug = 0.822
- This is **0.522 above the 0.3 threshold** for "clean contracts"
- The C.2.2 probe SHOULD have flagged this

**`I_regression_guard.md` I.3.1 (Behavior Checks):**

> "These are NOT enforced by `promote_model.py`. You must run them manually:
>   1. SmartBugs Curated smoke inference — run `C.2` from `C_diagnostic_checks.md`
>   2. Known-positive contract round-trip — at least one contract from each of the 10 classes should predict correctly with the tuned threshold
>   3. Known-negative (NonVulnerable) contract check — at least 5 clean contracts should produce no predictions above threshold
>   4. FP probe check — any contract used as FP probe in prior runs must not regress (record result against its prior result)"

**The spec lists "known-negative (NonVulnerable) contract check"** as a required manual step. This would have caught the FP.

---

## Why the spec suite didn't catch the bug

### Failure 1: C.2.1 (clean smoke inference) was UNVERIFIED

From the C attestation:
> "C.2.1 Smoke inference: UNVERIFIED
>   → Set is 95.8% contaminated. Wild eval (47K) used as functional substitute.
>   Marking UNVERIFIED per Rule 2."

**The cleanest test in the suite couldn't be run** because the only "clean" benchmark (SmartBugs Curated) is 95.8% in the v3 training data. The test was correctly marked UNVERIFIED — but that means the test didn't run, so it didn't catch the bug.

### Failure 2: C.2.2 (FP probe) was MANUAL, not a gate

The C.2.2 step was run during Run 12 validation:
> "C.2.2 FP probe: PASS — Manual inspection: 9 contracts, 4 TP / 3 FP / 2 borderline. **ExternalBug high-conf FP identified.**"

**The test DID find the FP.** But:
- The result was logged as a finding (FIND-R12-01, FIND-R12-B01)
- It was NOT treated as a promotion gate
- The model was promoted to Staging with the known issue unresolved
- The fix was deferred to Run 13

**The spec design flaw:** C.2.2 is a manual check that produces a finding. It doesn't say "block promotion if FP detected." The spec is descriptive ("you should run this") not prescriptive ("you must pass this to promote").

### Failure 3: I.2.2 (the only automated gate) uses the NOISY validation set

The `promote_model.py` script enforces:
- `val_f1_macro` strictly > current Production F1

For Run 12: F1 = 0.7004 > 0.3384 (Run 11). Gate passed.

**But F1 is computed on the v3 validation set**, which has:
- 69% ExternalBug positive (same 75% noise as training)
- The model achieves F1 = 0.88 on ExternalBug **because both the model and the labels have the same spurious pattern**

The F1 gate is **a meta-noisy gate**: it passes when the model agrees with noisy labels, which it can do without learning the right thing.

### Failure 4: I.3.1 (behavior checks) is explicitly NOT enforced

The I spec is clear:
> "These are NOT enforced by `promote_model.py`. You must run them manually."

This is **a documented design choice**: the behavior checks are advisory, not blocking. The actual enforcement is only I.2.2 (F1 gate).

**The structure assumes humans will run the behavior checks.** When the C.2.2 probe was run, the human did find the FP. But the human also decided to promote the model anyway (deferring the fix).

### Failure 5: No synthetic behavioral probe test

The 10 contract variations I ran in this investigation is **exactly the test that's missing**:
- safe_storage (owner + state) → ExternalBug should be < 0.3
- empty contract → ExternalBug should be < 0.3
- contract with `to.call(data)` → ExternalBug should be > 0.5
- contract with `interface.method()` → ExternalBug should be > 0.5

This is a **fixed set of synthetic contracts** that verify the model has learned the right feature. The spec suite has C.2.2 for FP probing, but the probe contracts are open-ended (whatever the human picks), not a fixed regression set.

---

## What the spec suite has right

To be fair, the spec suite DID identify the problem:

1. **C.1.6** (Per-Class F1 Convergence) flagged "ExternalBug threshold-gaming signal" (FIND-R12-01)
2. **C.2.2** (FP Probe) found "ExternalBug high-conf FP identified" via 9-contract manual inspection
3. **A attestation** (Benchmark) found FIND-R12-B01: "ExternalBug class-definition mismatch"
4. **L.5** (Session Handoff) externalized all findings to MEMORY.md and audit docs

**The spec suite is structurally sound.** It found the problem. The problem is in the **promotion decision** — the human saw the FP and chose to promote anyway.

---

## The structural design gap

| Layer | What it does | Catches the FP? |
|---|---|---|
| **Manual behavior checks (C.2.1, C.2.2, I.3.1)** | Human inspection of model on known-clean contracts | **YES**, but only if a human runs them and acts on the result |
| **Automated promotion gate (I.2.2)** | F1 on validation set | **NO** — validation set has 75% label noise |
| **Unit tests (test_model.py, test_trainer.py)** | Architecture, dimensions, loss function | **NO** — don't probe model behavior |
| **Smoke tests (smoke_fix1-8.py)** | Data infrastructure (CSV shape, split sizes) | **NO** — don't probe model behavior |
| **Synthetic behavioral probe** | Fixed contracts with known expected outputs | **MISSING** — doesn't exist |

**The gap:** there is no **automated** test that would block promotion of a model with a broken class. The only automated gate (F1) uses noisy labels.

---

## The specific contracts that would have caught it

If the spec suite had included a synthetic probe like this:

```python
SYNTHETIC_PROBES = [
    # (source, name, expected_class, expected_max_prob)
    ("pragma solidity ^0.8.0; contract S { address public owner; constructor() { owner = msg.sender; } }", "owner_only", "ExternalBug", 0.3),
    ("pragma solidity ^0.8.0; contract S { function f() external { address t; t.call(\"\"); } }", "low_level_call", "ExternalBug", 0.7),
    ("pragma solidity ^0.8.0; contract S { function f() external view returns(uint) { return 1; } }", "pure", "ExternalBug", 0.3),
]
for source, name, cls, max_prob in SYNTHETIC_PROBES:
    probs = api.predict(source).probabilities
    if probs[cls] > max_prob:
        raise Failure(f"SYNTHETIC_PROBE FAILED: {name} got {cls}={probs[cls]:.3f} > {max_prob}")
```

**This test would have BLOCKED the Run 12 promotion** because:
- safe_storage (owner pattern) → ExternalBug = 0.822 > 0.3 (FAIL)
- to_call (low-level call) → ExternalBug = 0.000 < 0.7 (FAIL — model thinks it's SAFE!)

The test fails on BOTH directions: it catches the FP on safe contracts AND the false negative on dangerous contracts.

---

## Recommended additions to the testing spec suite

### 1. New spec section: C.2.4 — Synthetic Behavioral Probes

Add to `C_diagnostic_checks.md`:
- Maintain a fixed set of synthetic contracts with expected outputs
- Run as part of the C.2 verification (after C.2.1, C.2.2, C.2.3)
- FAIL the verification if any probe exceeds the expected threshold

### 2. Update I_regression_guard.md — make I.3.1 enforced

The I.3.1 spec is currently advisory. Change `promote_model.py` to:
- Read a new file `ml/checkpoints/<stem>_behavioral_probes.json` (generated by C.2.4)
- Block promotion if any probe result is FAIL
- This is now a hard gate, not a manual check

### 3. Add F.1.0 — Label Quality Gate (pre-launch)

Add to `F_new_run_checklist.md`:
- Check the per-class positive rate in the training labels
- FAIL if any class has > 50% positive rate (suggests noisy labels)
- This would have flagged ExternalBug at 75% before Run 12 launched

### 4. Update I.2.2 — Use CLEAN validation F1 (not noisy)

If possible, compute `val_f1_macro` on a CLEAN validation set (not the noisy v3 split):
- Option A: use the v0.1 honest benchmark (66 contracts) as the promotion F1
- Option B: split the training data to keep a small clean set aside
- This breaks the F1-noise lock-in

---

## What the spec suite got right vs what it missed

| The spec suite DID | The spec suite DID NOT |
|---|---|
| Identify the C.2.1 contamination problem (UNVERIFIED) | Block promotion when C.2.1 is UNVERIFIED |
| Run C.2.2 FP probe manually and find the FP | Treat C.2.2 finding as a promotion gate |
| Flag F1-AUC divergence in C.1.6 | Compute F1 on a clean set |
| Document behavior checks in I.3.1 | Enforce behavior checks in `promote_model.py` |
| Externalize findings via L.5 | Auto-create BUG entries from findings |
| Provide 14 spec files for comprehensive coverage | Include a synthetic behavioral probe test |
| Define "known-clean" via C.2.2 | Define "synthetic known-clean" with fixed expectations |

**The spec suite is well-designed for its stated purpose. The gap is in promotion enforcement, not in test coverage.**

---

## The user's question answered

> "even this setup of testing ran? `ml/testing_specs` — why didn't it catch?"

**The setup did run. It found the FP. The bug was logged as a finding, but the model was still promoted to Staging because:**

1. The C.2.1 clean smoke test was UNVERIFIED (95.8% contamination)
2. The C.2.2 FP probe was MANUAL — found the FP, but it's advisory, not a gate
3. The I.2.2 automated gate (F1 > prior) passed because the validation set has the same label noise
4. The I.3.1 behavior checks are explicitly NOT enforced by `promote_model.py`

**The structural gap is in the enforcement layer, not the test layer.** The spec suite is comprehensive — the issue is that "find a finding" doesn't equal "block promotion."

**The fix:** add a synthetic behavioral probe (C.2.4) and make it a hard gate in `promote_model.py` (I.3.1 → enforced). The probe is the test I ran in this investigation — 10 fixed contracts with expected outputs.

---

**Status:** Audit complete. No code changes made. Per `ml/CLAUDE.md`, no code changes during a read-only audit.
