# SENTINEL ML Audit Issues

Next BUG-ID to assign: BUG-7

---

## How to File a New Issue

Follow `H_issue_triage.md` H.5:
1. Assign the next sequential ID from "Next BUG-ID" above, then increment it
2. Record: symptom, trigger condition, epoch first observed, config values at the time,
   alert code (if any), and resolution or investigation path
3. Write the entry before the session closes (Rule 3 — no floating findings)

---

## Open Issues

### BUG-6 — Pragma-only source crashes the inference preprocessor (MEDIUM — infra robustness) — **CLOSED 2026-06-18**

**Discovered:** 2026-06-17 during adversarial probe expansion in
`synthetic_probes.py` (edge_pragma_only probe)

**Symptom:** The synthetic probe suite includes an adversarial probe for
pragma-only source (`pragma solidity ^0.8.0;` with no contract). When the model
pipeline tries to extract a graph from this source, the preprocessor raises
`EmptyGraphError: No non-dependency contracts found`.

**Resolution (2026-06-18):** Modified `ml/src/inference/predictor.py:565`
(`predict_source()`) to catch `ValueError` (the wrapper for `EmptyGraphError`
raised by `preprocess.py:466`) and return a structured `no_contracts_found`
response. The schema matches `_format_result()` exactly, so downstream
callers don't need to wrap in try/except.

The structured response has:
- `label: "no_contracts_found"`
- `probabilities: {class: 0.0 for all}`
- `num_nodes: 0, num_edges: 0, windows_used: 0`
- `warning: "no_contracts_found: <reason>"`

**Verification:**
- `pragma solidity ^0.8.0;` → `label: "no_contracts_found"`, num_nodes: 0
- Empty contract `contract Empty {}` → works as before
- Real contract → works as before (ExternalBug=0.86 for owner pattern)
- `synthetic_probes.py` `edge_pragma_only` no longer INFRA_ERROR

**Effect on probes:** 26/40 PASS (was 25/40), 0 INFRA_ERROR (was 1).

**Cross-references:** BUG-4 (synthetic probe expansion), C.2.4.

---

### BUG-5 — Testing framework overhaul (CLOSED 2026-06-17)

**Discovered:** 2026-06-17 (per `2026-06-17_testing_suite_overhaul_plan.md`)

**Resolution:** The testing spec suite was overhauled to add:
- `synthetic_probes.py` — 30+ fixed synthetic contracts (C.2.4)
- `label_quality.py` — pre-launch label audit (F.1.0)
- `framework/` package — project-agnostic reusable infrastructure
- `promote_model.py` updated — wires in behavioral + label gates
- All spec .md files updated (C, F, I, L, README)
- `QUICKSTART.md`, `MIGRATION.md` written
- `framework/templates/sentinel_v2.yaml` — SENTINEL template

**Verification on Run 12 (2026-06-17):**
- `synthetic_probes.py`: **19/30 pass, 11 FAIL** — most notably all 3 ExternalBug
  probes (the FP issue)
- `label_quality.py`: **13 FAILs** — most notably ExternalBug at 74% positive
  (the root cause of the model issue)
- `framework/cli.py run --exit-on-fail`: **Exit 1** (correctly blocks promotion)

**Result:** Run 12 promotion is now correctly blocked by the new gates.
The broken ExternalBug class can no longer slip through.

**Cross-references:** BUG-2 (model issue, now caught), BUG-3 (test process gap, now
closed), BUG-4 (test design gap, now closed). All 3 BUGs are addressed.

### BUG-4 — Testing spec suite has no synthetic behavioral probe (HIGH — design gap)

**Discovered:** 2026-06-17 during user follow-up question: "even this setup of
testing ran? `ml/testing_specs` — why didn't it catch?"

**Symptom:** The 14-file `ml/testing_specs/` suite is comprehensive but
**structurally unable to block promotion of a broken model class**:
1. C.2.1 (clean smoke) is UNVERIFIED (95.8% contamination)
2. C.2.2 (FP probe) is MANUAL — found the FP, logged as finding, not a gate
3. I.2.2 (only automated gate) checks F1 on NOISY validation set
4. I.3.1 (behavior checks) is explicitly NOT enforced by `promote_model.py`

**Root cause:** Missing synthetic behavioral probe test. There is no
automated test that verifies "given contract X, model output Y" with fixed
expectations. The C.2.2 probe is open-ended (whatever the human picks).

**Trigger conditions:** Any model trained on noisy labels passes the I.2.2 F1
gate by gaming the noise. A synthetic probe (10 fixed contracts with expected
outputs) would have BLOCKED Run 12 promotion because safe_storage gives
ExternalBug=0.822 (expected < 0.3) and to.call gives 0.000 (expected > 0.7).

**Root cause analysis:** `ml/audit_docs/2026-06-17_ml_Run12_testing_spec_suite_gap.md`

**Recommended fixes (in priority order):**
1. **Add C.2.4 — Synthetic Behavioral Probes** to `C_diagnostic_checks.md`
   - 10 fixed contracts with expected (max/min) probability thresholds
   - Run as part of C.2 verification
2. **Update I.3.1 → I.2.2 enforced** in `I_regression_guard.md`
   - `promote_model.py` reads `behavioral_probes.json` and blocks promotion
3. **Add F.1.0 — Label Quality Gate** to `F_new_run_checklist.md`
   - FAIL if any class has > 50% positive rate
4. **Add adversarial probes** for each class (not just ExternalBug)
   - e.g., "just owner" for ExternalBug, "pure function" for Reentrancy
   - Catches any future class with a broken feature

**Workaround:** Manual 10-contract probe (as in this investigation). Run before
any model promotion.

**Cross-references:** BUG-2, BUG-3, FIND-R12-01, FIND-R12-B01.

### BUG-3 — Testing pipeline missed the ExternalBug FP (HIGH — process gap)

**Discovered:** 2026-06-17 during user follow-up question: "we tested the model
but don't see this problem — why?"

**Symptom:** The ExternalBug FP was identified during Run 12 validation
(FIND-R12-01, FIND-R12-B01, C.2.2 FP probe), but the model was promoted to
Staging with the issue unresolved. The fix was deferred to Run 13/Phase A.

**Root cause:** Multi-layer testing gap:
1. **C.2.1 clean benchmark UNVERIFIED** — SmartBugs Curated is 95.8% contaminated
2. **Validation set has same 75% ExternalBug label noise** as training — F1=0.88
   looks good but is gaming the noisy labels
3. **No synthetic probe test** — unit tests cover architecture, not behavior
4. **Fix deferred, not skipped** — findings were logged but the model was promoted

**Trigger conditions:** Any model trained on the v3 export (22,493 contracts,
75.1% ExternalBug positive in DIVE source)

**Epoch/model:** Run 12 FINAL (promoted to Staging 2026-06-14)
**Config at time:** Standard Run 12 config
**Alert code:** FIND-R12-01, FIND-R12-B01 (filed, not actioned)

**Root cause analysis:** `ml/audit_docs/2026-06-17_ml_Run12_testing_gap_root_cause.md`

**Recommended fixes (in priority order):**
1. **Add synthetic probe test (C.2.4):** Verify model gives LOW ExternalBug for
   "just owner" contracts and HIGH for actual external calls. Gate before promotion.
2. **Add label quality check (C.0):** Detect over-labeled classes before training.
   Flag any class with >50% positive rate as suspicious.
3. **Relabel DIVE crosswalk:** Map "Access Control" folder more narrowly
   (currently maps ALL of it to ExternalBug-positive).
4. **Don't defer "model behavior" findings** — file as a blocking issue, not
   a future-improvement.

**Workaround:** Manual inspection of any model promoted in 2026-06.

**Cross-references:** BUG-2, FIND-R12-01, FIND-R12-B01, C.2.2 FP probe.

### BUG-2 — Run 12 ExternalBug class is fundamentally broken (CRITICAL)

**Discovered:** 2026-06-17 during E2E test of agents module on `safe_storage.sol`
**Symptom:** Model gives ExternalBug=0.82 (CONFIRMED) for a SAFE contract that
has zero external calls. Also gives ExternalBug=0.00 for a contract with the
textbook dangerous pattern `to.call(data)`. The model's ExternalBug head is
**inverted** — it fires on "owner + msg.sender" patterns (safe) and not on
actual external calls (risky).

**Root cause:** Training data label quality. The DIVE crosswalk
(`sentinel_data/labeling/crosswalks/dive.yaml`) maps the "Access Control" folder
to ExternalBug. The Access Control folder contains 16,582 contracts (75% of
DIVE), most of which have benign owner patterns. The model correctly learned
"owner pattern = ExternalBug" because 75% of training data is in this folder.

**Trigger conditions:** Any contract with `address public owner` + `msg.sender`
in constructor or function (very common in Solidity).

**Epoch/model:** Run 12 FINAL (268.5 MB, md5=f1a04c12bda6)
**Config at time:** Standard Run 12 config (no special changes)
**Alert code:** None at training time — FP was found post-training via C.2.2

**Root cause analysis:** `ml/audit_docs/2026-06-17_ml_Run12_externalbug_false_positive_root_cause.md`

**Recommended fix (in priority order):**
1. **Relabel the DIVE crosswalk** — re-define ExternalBug mapping
2. **Add a rule-based inference filter** in agents module
3. **Retrain** with tighter ExternalBug definition (target 5-15% positive rate)

**Workaround:** Don't trust ExternalBug verdicts from Run 12.

**Cross-reference:** This was discovered via `docs/plan/agents/2026-06-17-agents-real-e2e-test/`.

---

## Resolved Issues

### BUG-1 — False-positive claim: API `tier_thresholds` schema mismatch (CLOSED 2026-06-17)

**Status:** CLOSED — false positive. No code change required.
**Filed:** 2026-06-17
**Closed:** 2026-06-17 (same session)
**Reporter:** Self-flag (Ali asked to investigate the P0 from MEMORY.md)

**Symptom (claimed):**
`/predict` and `/hotspots` return HTTP 500 when serving Run 12 with per-class
thresholds loaded. `ml/tests/test_api.py` reported 4 failures.

**Investigation:**
- `ml/src/inference/api.py:209` declares `tier_thresholds: dict[str, float | list[float]]`
  (Pydantic v2 union — accepts both scalar and list).
- `ml/src/inference/predictor.py:750-754` returns `{"confirmed": list, "suspicious": float, "noteworthy": float}`.
- `pytest ml/tests/test_api.py` → **18/18 PASS** (re-run 2026-06-17).

**Root cause of false positive:**
- The original claim (in MEMORY.md and the 2026-06-16 MLOps state check report) was based
  on reading the type as `dict[str, float]` — likely from a stale mental model of Pydantic v1
  syntax, or from skimming the type without recognising the `|` as a union.
- The "4 fail" claim in the original report was not reproducible on re-run.

**Resolution:**
- No code change required — the API was always correct.
- Updated docs: MEMORY.md, MLOps state check report, K_inference_api.md spec.
- This BUG-1 entry exists so the next session sees the resolution and doesn't redo the work.

**Residual minor inconsistency (not a bug, not part of BUG-1):**
- `/health` returns scalar `tier_thresholds.confirmed`; `/predict` returns per-class list.
- Not a 500. Not user-blocking. Flagged for future cleanup.

**Cross-references:**
- `docs/learning_sentinel/2026-06-17_step1_api_bug_investigation.md`
- `docs/reports/2026-06-16_ml_Run12_mlops_full_state_check/2026-06-16_ml_Run12_INDEX_mlops_full_state_check.md` (§REVISION)
- `docs/changes/2026-06-17-ml-api-schema-claim-correction.md`
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`

