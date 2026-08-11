# R4 Phase 4 — Gap Authorization and Targeted Adjudication Plan

**Phase:** 4 — Targeted Evidence-Gap Adjudication  
**Gate:** G4  
**Branch:** `r4/phase4-targeted-gap-adjudication`  
**Entry condition:** G3 PASS on canonical `main` (`e7bbae5a...` and later)  
**Current execution state:** `R4-GAP-002` APPROVED; WP1 population/sample freeze in progress

## 1. Objective

Fill only the evidence gaps that are necessary to make the first DATA vNext source/class/role decisions. Do not turn Phase 4 into a general benchmark, tooling, fuzzing, or model-improvement program.

## 2. Governing constraints

- Reuse recovered Phase-1/2/3 evidence before any new review.
- No contract review may start unless the corresponding gap is approved.
- The human owner delegated routine technical/governance approvals to the AI assistant on 2026-08-11; approvals must still be explicit in repository records and evidence-based.
- The historical DATA export, split, representations, checkpoints, thresholds, and calibration artifacts remain immutable.
- The existing model architecture remains frozen.
- Tool agreement is evidence, not ground truth.
- Historical zero is not a negative by default.
- Unsupported or unresolved populations may be masked/excluded instead of forcing a review.
- Review size is adaptive; there is no universal sample target.

## 3. Phase-4 entry triage

### Approved critical gap: R4-GAP-002 — DIVE mapped, unreviewed classes

The active DIVE source has recovered manual precision evidence for Access Control→ExternalBug and Reentrancy only. Five other DIVE native classes currently map to canonical supervised positives without class-specific semantic precision evidence:

| DIVE native class | Canonical class | Current source role | Why Phase 5 is blocked |
|---|---|---|---|
| `Arithmetic` | `IntegerUO` | T2 positive assertion | Cannot decide strong/weak/masked/excluded role from source assertion alone. |
| `DoS` | `DenialOfService` | T2 positive assertion | No recovered class-specific semantic precision evidence. |
| `Unchecked Return Values` | `UnusedReturn` | T2 positive assertion | No recovered class-specific semantic precision evidence. |
| `Time manipulation` | `Timestamp` | T2 positive assertion | No recovered class-specific semantic precision evidence. |
| `Front Running` | `TransactionOrderDependence` | T2 positive assertion | No recovered class-specific semantic precision evidence. |

`Bad Randomness` is not part of the critical review population for the first baseline: the current DIVE crosswalk drops it rather than mapping it to a canonical positive. Its rows can remain unknown/masked unless a later policy decision proposes a new mapping.

**Decision blocked:** for each of the five mapped DIVE strata, choose whether DIVE positives may be retained as stronger evidence, weak training signal, unknown/masked, or excluded in DATA vNext.

**Prior evidence reused:** recovered DIVE source labels/crosswalk; EB/RE manual reviews; Slither/Aderyn corroboration and dependence analysis; Phase-2 source-authority/zero-semantics reconstruction; Phase-3 evidence ledger.

**Approval:** granted 2026-08-11 under owner-delegated technical/governance authority. See `docs/plan/ml-R4/authorizations/2026-08-11_R4-GAP-002_authorization.md`.

**Initial review:** deterministic leakage/project-group-aware DIVE-positive screening sample, 20 contracts per stratum / 100 total, from TRAIN-only groups that do not touch val/test. Review source semantics blind to historical/model/tool conclusions where possible, reconcile with recovered evidence afterward, and expand only strata whose DATA-role decision remains ambiguous or strong enough to justify higher-authority review.

**Second review:** required only if a stratum is proposed for a high-authority role (for example `TRAIN_STRONG` or acceptance-like evidence). A result supporting only masking/exclusion does not justify broad second-review expansion.

### R4-GAP-001 — Web3Bugs/all classes

**Recommendation:** do not acquire/review in Phase 4 for the first DATA vNext baseline. Phase 2 proved zero executable historical contribution. Phase 5 can explicitly exclude/disable the absent source unless later source expansion is intentionally approved.

### R4-GAP-003 — BCCC provisional CTU/ExternalBug/GasException

**Recommendation:** do not run deferred GraphCodeBERT/HDBSCAN propagation in Phase 4. BCCC v1.4 is not an active source in the historical export. The first baseline can leave these BCCC strata deferred/masked; importing BCCC is a Phase-5 policy choice.

### R4-GAP-004 — BCCC 2-tool consensus

**Recommendation:** not critical for G4. The planned consensus is incomplete and has detector-coverage gaps (including DoS/GasException). Tool intersection is not ground truth and does not need to block DATA vNext.

### R4-GAP-005 — Echidna/fuzzing precision estimates

**Recommendation:** defer. Useful complementary evidence for later evaluation/case studies, but not required to decide the current active source/class roles.

### R4-GAP-006 — exploit-reproduction PoCs

**Recommendation:** defer to evidence-qualified evaluation / case-study work. A broad 5–10 PoCs per class is not the smallest evidence needed to resolve the current DATA vNext source decisions.

## 4. Authorized work packages

1. **P4-WP1 — Authorization and population freeze** — IN PROGRESS
   - approval recorded;
   - five native→canonical class definitions frozen;
   - enumerate DIVE-positive contract IDs and dedup/project groups from the committed Phase-3 ledger;
   - construct deterministic initial sample manifest.

2. **P4-WP2 — Blind semantic review** — WAITING FOR WP1
   - bind sampled contract IDs to source material;
   - review contract source against the frozen class definition;
   - record `SUPPORTS_POSITIVE`, `DOES_NOT_SUPPORT_POSITIVE`, `UNCLEAR_INSUFFICIENT`, or `CLASS_BOUNDARY_CONFLICT` with rationale;
   - do not use model prediction as adjudication truth.

3. **P4-WP3 — Evidence reconciliation** — WAITING
   - reveal historical DIVE assertion and available Slither/Aderyn/tool evidence;
   - preserve conflicts and independence groups;
   - expand only ambiguous strata.

4. **P4-WP4 — Role recommendation and uncertainty** — WAITING
   - recommend stronger/weak/masked/excluded status per DIVE stratum;
   - second-review only strata proposed for high-authority roles;
   - quantify unresolved/unclear populations.

5. **P4-WP5 — Gap/G4 closeout** — WAITING
   - close or mask/exclude each authorized stratum;
   - document why the other proposed gaps do not block first-baseline DATA vNext;
   - assess G4.

## 5. Stop conditions

Stop rather than improvise if:

- the committed Phase-3 ledger identity changes;
- the five class definitions cannot remain frozen without a new taxonomy/policy decision;
- deterministic source population/sample identity cannot be established;
- a proposed strong role lacks sufficient independent semantic evidence;
- new review starts drifting into model architecture, threshold, calibration, Web3Bugs acquisition, BCCC expansion, or general tooling work.

## 6. Current next action

Run and validate `p4_freeze_gap002_sample.py` against the committed Phase-3 ledger. Publish the population/sample identities. Then bind those 100 sampled contract IDs to local DIVE source material before any semantic verdict is recorded.
