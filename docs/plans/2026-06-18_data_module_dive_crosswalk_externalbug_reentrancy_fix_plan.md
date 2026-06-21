# Plan — Fix DIVE Crosswalk Over-Labeling (ExternalBug + Reentrancy)

**Date:** 2026-06-18
**Module:** data_module (labeling pipeline)
**Status:** IN PROGRESS — Step 1 complete, Step 2 corroboration pass running
**Owner:** Ali

---

## 1. Problem

Run 12's model has a confirmed-broken `ExternalBug` head (BUG-2, `ml/audit_docs/ISSUES.md`):
fires 0.82 on a contract with zero external calls, 0.00 on a contract with the textbook
`to.call(data)` pattern. Root-cause digging (2026-06-18, `~/.claude/scratch/externalbug_datamodule_rootcause_20260618.md`)
traced this to the v3 export's label source, not the model architecture.

### Root cause (verified against source code, not docs)

`data_module/sentinel_data/labeling/parsers/dive.py` + `crosswalks/dive.yaml`:
DIVE stores every contract once in `__source__/`, and again inside per-vulnerability
folders (`Reentrancy/`, `Access Control/`, etc). A contract is labeled positive for
class X purely by **filename appearing in the folder mapped to X** — no opcode check,
no static-analysis corroboration, nothing dynamic (`_build_folder_index`, dive.py:54-69).

`"Access Control"` was deliberately crosswalked to `ExternalBug` (taxonomy.yaml:50-60
defines ExternalBug as a DASP-2 + DASP-10 catch-all covering access control, flash
loans, oracle manipulation, delegatecall injection). That mapping decision was
intentional. **The bug is that nobody checked the resulting base rate before
accepting it into the export.**

DIVE's own `"Access Control"` folder turns out to be its single **largest** folder
(16,723 of 22,330 files — larger than `Reentrancy/`). Mapped wholesale, this produced
ExternalBug = 74.0% positive in v3 (16,638 / 22,493 contracts).

### This is systemic, not isolated to ExternalBug

Verified via `ml/checkpoints/v3_label_quality.json` (13 FAIL checks already on record,
2026-06-17/18 testing-suite overhaul):

| Class | Positive rate | DIVE source dominance | Gate status |
|---|---|---|---|
| ExternalBug | 74.0% | 99.7% | FAIL (>50% over-labeled) |
| Reentrancy | 50.7% | 99.4% | FAIL (>50% over-labeled) |
| IntegerUO | 42.0% | 99.3% | pass (not yet flagged, high) |
| Timestamp | 28.1% | 99.2% | pass |
| UnusedReturn | 26.0% | 100.0% | pass |
| CallToUnknown | 0.4% | — | FAIL (<1%, unlearnable) |
| GasException | 0.0% | — | FAIL (<1%, unlearnable, zero positives) |
| MishandledException | 0.2% | — | FAIL (<1%, unlearnable) |

7 of 10 classes get >90% of their positive labels from DIVE alone — for most
classes the model is learning "did this filename sit in one of DIVE's folders,"
not "is this vulnerability present."

### Decision (made 2026-06-18)

- Fix **ExternalBug + Reentrancy together** (same crosswalk file, same gate, same
  root cause — fixing one and shipping the other known-bad would be inconsistent).
- Fix approach: **sample-validate the crosswalk, then add a permanent base-rate
  sanity gate** to the pipeline (not a taxonomy split — that's a larger redesign,
  deferred).
- The 3 unlearnable classes (CallToUnknown, GasException, MishandledException) are
  the **opposite failure mode** (too few positives, not too many) and are explicitly
  **out of scope** for this plan — tracked as a follow-up, not forgotten.
- Model retraining (Run 13) is **out of scope** for this plan. This plan stops at
  "clean export + passing `label_quality.py` gate." Run 13 consumes the result.

---

## 2. Plan

### Step 1 — Sample-validate the two broken mappings

For each of `Access Control/` (→ExternalBug) and `Reentrancy/` (→Reentrancy) DIVE
folders: pull ~75 random contracts, read the actual `.sol` source, and judge
whether the contract genuinely exhibits the target vulnerability or was filed
there by DIVE's own loose/broad tagging.

**Output:** an empirical true-positive rate per folder, with examples of false
inclusions, written to a dated audit doc (`data_module/audit/2026-06-18_dive_crosswalk_sample_validation.md`).

### Step 2 — Decide per-class remediation based on Step 1's findings

- If a folder's true-positive rate is reasonably high (rule of thumb: >60%) →
  keep the mapping but fix the base rate: downsample the positive set, and/or
  require a cheap secondary corroboration signal (e.g. a related Slither tag)
  before accepting a DIVE-folder label as positive.
- If true-positive rate is low (DIVE's tagging really is "any access-control-
  adjacent mention, regardless of severity") → drop the wholesale folder mapping.
  Keep only contracts independently corroborated by a second source/tier
  (SmartBugs curated, explicit Slither finding, etc.) — same treatment
  `"Bad Randomness"` already received (dropped wholesale; contracts remain
  represented/graphed but unlabeled for that class).

### Step 3 — Add a permanent crosswalk sanity gate to the pipeline

New automated check (extends the existing `ml/testing_specs/label_quality.py`
pattern into the data module's own labeling stage, not just a post-hoc ML-side
audit): before any crosswalk YAML is accepted into a release export, compute
per-class positive rate and per-source dominance, and **fail hard** if:
- positive rate > 50% or < 1%, OR
- single-source dominance > 90%

...unless the crosswalk YAML carries an explicit `override: { reason: "..." }`
block justifying the exception (e.g. legitimately rare classes like GasException
may need an override rather than a fix). This formalizes the check
`label_quality.py` already discovered ad hoc, baking it into Stage 3 of the data
module pipeline so future crosswalk additions (new sources, new folders) can't
silently reintroduce this failure mode.

### Step 4 — Re-export

Re-run Stage 3 (labeling) + Stage 5 (export) for the `dive` source only — other
sources (SmartBugs curated, SolidiFI, BCCC) are untouched. Produces a new export
(v3.1 or v4, naming TBD at execution time) containing corrected ExternalBug and
Reentrancy labels.

### Step 5 — Verify

Re-run `ml/testing_specs/label_quality.py` (or its data-module-side equivalent
from Step 3) against the new export. Confirm ExternalBug and Reentrancy both pass
the over-labeling gate. Update `ml/audit_docs/ISSUES.md` (BUG-2) with the
resolution.

### Step 6 (explicitly deferred, separate follow-up)

The 3 unlearnable classes (CallToUnknown 0.4%, GasException 0.0%,
MishandledException 0.2%) need their own fix — either sourcing more positives
(e.g. the already-audited 658 BCCC MishandledException contracts, per
`2026-06-14_project_bccc_2tool_audit.md`) or an explicit override + documented
limitation. Not part of this plan; raise as its own dated plan when picked up.

---

## 3. Out of scope

- Taxonomy redesign (splitting ExternalBug into narrower DASP-2/DASP-10 sub-classes).
- Model retraining / Run 13 execution.
- Agents-module quarantine of ExternalBug verdicts (separate, smaller plan —
  not yet written, can be done independently and in parallel since it touches
  `agents/src/orchestration/routing.py`, not the data module).
- The 3 unlearnable classes (see Step 6).

---

## 4. Execution log (appended incrementally as steps complete)

### Step 1 — COMPLETE (2026-06-18)

Sample-validated 75 contracts each from `Access Control/` (→ExternalBug) and
`Reentrancy/` (→Reentrancy) DIVE folders (seed=42, reproducible).

**Result: both TP rates far below the 60% keep-threshold.**
- Access Control → ExternalBug: **4/75 TP = 5.3%**
- Reentrancy → Reentrancy: **3/72 decidable = ~4.2%** (3 UNCLEAR excluded)

DIVE filed contracts by superficial pattern resemblance (any `onlyOwner`-style
code → "Access Control"; any `.call{value}`/`.transfer()` → "Reentrancy"), not
verified exploitability. Full detail + per-contract verdicts:
`data_module/audit/2026-06-18_dive_crosswalk_sample_validation.md`.

**Decision point surfaced:** dropping the DIVE folder labels wholesale (the
plan's original "option b") would leave only ~56 ExternalBug and ~69 Reentrancy
positives project-wide (from solidifi + smartbugs_curated only) — i.e. it trades
"over-labeled" for "unlearnable," the same disease as GasException/CallToUnknown/
MishandledException. Reentrancy is too important a class to gut to 69 positives
without a recovery plan.

**Ali's decision (2026-06-18):** run a corroboration pass — use Slither detectors
across the FULL folders (not just the 75-sample) and keep only DIVE-folder
positives that an independent tool also flags, rather than trusting folder
membership alone or dropping it outright.

### Step 2 — CORROBORATION PASS COMPLETE (2026-06-18)

**Sub-step 2a — full Slither corroboration pass (15.5 min) — DONE.**

Reused existing infra (`data_module/sentinel_data/verification/tool_validator.py`
+ `slither_runner.py` `CLASS_TO_DETECTORS` map). Detector mapping used:
- ExternalBug → `arbitrary-send-eth`, `low-level-calls`, `tx-origin`, `controlled-delegatecall`
- Reentrancy → `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-benign`, `reentrancy-events`

Ran as a monitored background job (`/tmp/run_corroboration_monitored.py`) with
checkpointed JSON output every batch — full results:
- **ExternalBug: 15,920 checked, 6,804 AGREE (42.7%), 9,116 DISAGREE, 662 errored**
- **Reentrancy:  11,018 checked, 8,258 AGREE (75.0%), 2,760 DISAGREE, 312 errored**

JSON: `data_module/audit/2026-06-18_dive_externalbug_reentrancy_slither_corroboration.json`

**Sub-step 2b — manual sample-validation of the AGREE subset — DONE.**

The 42.7% (EB) and 75.0% (RE) agreement rates are suspiciously high against the
5.3% (EB) and 4.2% (RE) raw DIVE-folder TP rates from Step 1 — Slither's
detectors are themselves syntactic proxies that may co-fire on the same false
positives DIVE already mislabeled. Manually re-validated a sample of the agreed
subset (seed=7, reproducible) to check whether the intersection actually
improves precision.

| Class | n reviewed | TP | FP | Empirical TP | 95% CI | vs raw DIVE |
|---|---|---|---|---|---|---|
| ExternalBug | 100 | 4 | 96 | **4.0%** | [1.1%, 9.9%] | -1.3pp (CI overlap) |
| Reentrancy  |  75 | 2 | 73 | **2.7%** | [0.3%, 9.3%] | -1.5pp (CI overlap) |

Full per-contract verdicts + per-class conclusions: `data_module/audit/2026-06-18_dive_slither_agreed_subset_validation.md`.

**Verdict: the DIVE+Slither agreement is NOT a precision filter for either class.
It trends marginally *worse* than raw DIVE (though CIs overlap). The "agreement"
is illusory — Slither and DIVE are co-firing on the same superficial patterns
(`sendValue` library function, `approveAndCall`, fee-swap mutex, etc). The 6
true positives we found (4 EB + 2 RE) are seed points worth keeping, but the
~15,000+ "agreed" contracts that are not manually validated are noise.**

**Sub-step 2c — Aderyn (Cyfrin) cross-tool validation — DONE (2026-06-18).**

After 2b, ran Aderyn 0.6.8 (Rust, installed at `~/.cargo/bin/aderyn`, used
previously in BCCC 2-tool audit 2026-06-14) as a second independent corroboration
source. Built `data_module/sentinel_data/verification/aderyn_runner.py` with
per-class detector mapping (EB: `tx-origin-used-for-auth`,
`eth-send-unchecked-address`, `delegate-call-unchecked-address`,
`arbitrary-transfer-from`, `state-no-address-check`, `incorrect-erc20-interface`,
`constant-function-changes-state`; RE: `reentrancy-state-change`,
`non-reentrant-not-first`, `unchecked-send`).

| Signal | EB TP | RE TP | n | Notes |
|---|---|---|---|---|
| Raw DIVE folder | 5.3% | 4.2% | 75/75 | Step 1 |
| DIVE ∩ Slither | 4.0% | 2.7% | 100/75 | Step 2b |
| DIVE ∩ Slither ∩ Aderyn (3-way) | **3.0%** | **1.7%** | 66/59 | **Worse than Slither alone** |
| Aderyn-only on Slither-disagreed (added-signal test) | 0/30 (0%) | n/a | 30 | Aderyn's added signal = noise |

3-way precision (DIVE ∩ Slither ∩ Aderyn) is **3.0% (EB) / 1.7% (RE)** —
strictly *worse* than Slither-only. Aderyn's detectors are mostly a
superset of Slither's syntactic signal (same `sendValue`/`approveAndCall`/
fee-swap-mutex false positives, plus additional patterns like
`state-no-address-check` that catch low-severity `address` state var
assignments without zero-checks — not ExternalBug exploits).

To verify Aderyn isn't catching TPs that Slither misses, ran Aderyn on 400
DIVE-positive / Slither-DISAGREED contracts (200 EB + 200 RE). Aderyn agreed
on 73/200 EB (36.5%) and 25/200 RE (12.5%) — Aderyn is more lenient. Then
manually reviewed 30 EB Aderyn-only-positive contracts (seed=13): **0/30 TPs**.
All 30 are standard ERC20s with proper onlyOwner gating, OZ libraries, or
low-severity `state-no-address-check` findings.

**Verdict: Aderyn does NOT add precision on top of Slither for either class.
None of the 3 "corroboration" tools — DIVE folder, Slither, Aderyn — provides
an independent precision signal for ExternalBug or Reentrancy.**

Per-contract results: `data_module/audit/2026-06-18_dive_aderyn_per_contract_v1.json`
+ `data_module/audit/2026-06-18_dive_aderyn_on_slither_disagreed_v1.json`.
Full audit doc (updated with Phase 3): `data_module/audit/2026-06-18_dive_slither_agreed_subset_validation.md`.

**Sub-step 2d — DECISION (2026-06-18, awaiting Ali's confirmation):**

With Option A (use agreed set) and Option C (hybrid) empirically falsified
by 2 independent corroboration tools (Slither + Aderyn, on 175 + 30 + 30
contracts), **the only honest path is Option B**: drop the DIVE folder
labels for ExternalBug and Reentrancy entirely. Replace with the few
independent positives from SolidiFi + SmartBugs Curated + the 6
manually-validated seeds (4 EB + 2 RE).

Resulting v3.1 label counts (vs v3):
- **ExternalBug: 60 positives** (39 solidifi + 17 smartbugs_curated + 4 seeds) — vs 16,638 in v3 (**99.6% reduction**)
- **Reentrancy:  71 positives** (39 solidifi + 30 smartbugs_curated + 2 seeds) — vs 11,399 in v3 (**99.4% reduction**)

Both classes will then be in the "rare positive" regime like the other 3
unlearnable classes (CallToUnknown 0.4%, GasException 0.0%, MishandledException
0.2%). The model will not be able to learn either class from these label sets
alone — Run 13 must be paired with either synthetic adversarial positives
or sourced real positives (BCCC, CGT, audit-firms, Kaggle). The label_quality
gate will catch the <1% rate at the next launch and force the conversation.

**This will be paired with explicit `override: { reason: "..." }` blocks in
the crosswalk YAML** explaining the deviation from the >1% positive-rate gate,
so future maintainers (and future me) don't repeat the same dig.

**Open follow-ups (deferred to separate plans):**
1. The 3 unlearnable classes (CallToUnknown / GasException / MishandledException)
   are a separate-but-related problem (too few positives, not too many) — own
   dated plan needed.
2. `data_module/sentinel_data/verification/aderyn_runner.py` is now in place
   and can be used for future cross-tool validations on other classes
   (e.g. when the 3 unlearnable classes get re-sourced, the Aderyn vs
   Slither agreement test can be re-run cheaply).

## 5. Cross-references

- `ml/audit_docs/ISSUES.md` — BUG-2
- `ml/audit_docs/2026-06-17_ml_Run12_externalbug_false_positive_root_cause.md` — original (partially superseded) root-cause doc
- `~/.claude/scratch/externalbug_datamodule_rootcause_20260618.md` — full verification trail for this plan's root-cause section
- `docs/plans/2026-06-14_Run13_4_fixes_preparation.md` — Run 13 fix #5 (ExternalBug only; this plan supersedes/extends that scope to include Reentrancy)
- `ml/checkpoints/v3_label_quality.json` — gate evidence
- `data_module/sentinel_data/labeling/crosswalks/dive.yaml` — file to be edited in Step 2
- `data_module/sentinel_data/labeling/parsers/dive.py` — labeling logic, may need updates for Step 2/3
