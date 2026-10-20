# R4 Claim Status Matrix

This matrix describes the **current evidence-qualified claim boundary**, not the historical binary-label surface. R4-D-008 accepts repaired-v2 physical DATA as immutable reproducibility evidence; R4-D-009 accepts corrected logical V3 grouping/roles; R4-D-010 prohibits v9 for the new full training lineage; R4-D-011 accepts the exact V10 V2.6 graph/control root; R4-D-013 accepts the guarded-token successor; R4-D-014 limits the executable ML regime to bounded positive-only control/positive-fit diagnostics; R4-D-015 accepts the completed M4 bounded pilot while explicitly keeping M5/full training on HOLD; and R4-D-016 resolves `CallToUnknown` semantic scope without changing class identity/order. None of these facts establishes general model discrimination, calibration, thresholds, or production acceptance.

| Index | Class | DATA vNext supervision | Current positive authority | Confirmed-negative support | Discrimination | Calibration / threshold | Current claim status | Key limitation |
|---:|---|---|---|---|---|---|---|---|
| 0 | CallToUnknown | ENABLED | STRONG: SolidiFI Unchecked-Send; SmartBugs unchecked_low_level_calls | NONE — candidate #1 historical rationale is definition-stale; candidate #2 remains UNKNOWN/PENDING_REVIEW/target `None`; admissible v2 independent review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | R4-D-016 resolves the class to raw/dynamic low-level unknown-target interactions, but no confirmed-negative evaluation support exists yet |
| 1 | DenialOfService | ENABLED | STRONG: SmartBugs denial_of_service | NONE — V3 pilot queue available | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE DoS is masked; no confirmed negatives yet |
| 2 | ExternalBug | ENABLED | STRONG: SolidiFI tx.origin; SmartBugs access_control | NONE — V3 pilot queue available | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Access Control is masked; class scope remains broader than any one source category |
| 3 | GasException | SUPERVISION_DISABLED_PENDING_EVIDENCE | none | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | DISABLED_PENDING_EVIDENCE | No active approved class-specific positive authority |
| 4 | IntegerUO | ENABLED | STRONG: SolidiFI Overflow-Underflow; SmartBugs arithmetic | NONE — V3 pilot queue available | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Arithmetic is masked; no confirmed negatives yet |
| 5 | MishandledException | ENABLED | STRONG: SolidiFI Unhandled-Exceptions | NONE — V3 pilot queue available | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | Positive evidence only; no discrimination-grade negative population |
| 6 | Reentrancy | ENABLED | STRONG: SolidiFI Re-entrancy; SmartBugs reentrancy | NONE — V3 pilot queue available | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Reentrancy is masked; no confirmed negatives yet |
| 7 | Timestamp | ENABLED | STRONG: SolidiFI Timestamp-Dependency; SmartBugs time_manipulation | NONE — V3 pilot queue available | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | SmartBugs bad_randomness→Timestamp is superseded/no-target; DIVE time manipulation is masked |
| 8 | TransactionOrderDependence | ENABLED | STRONG: SolidiFI TOD; SmartBugs front_running; WEAK training-only: DIVE Front Running | NONE — V3 pilot queue available | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE contribution is weak and barred from outcome metrics/model selection; no confirmed negatives yet |
| 9 | UnusedReturn | SUPERVISION_DISABLED_PENDING_EVIDENCE | none | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | DISABLED_PENDING_EVIDENCE | DIVE Unchecked Return Values failed the Phase-4 authority threshold and no other active source directly supports the class |

## Global current claim boundary

Physical DATA:

- repaired-v2 physical DATA: **ACCEPTED_IMMUTABLE_REPRODUCIBILITY_EVIDENCE** under R4-D-008 / ADR-R4-008;
- contracts / contract×class rows / physical files: 22,540 / 225,400 / 67,620;
- physical binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- graph schema v9 is **NOT ELIGIBLE FOR A NEW FULL TRAINING RUN** under R4-D-010;
- V10 `v2.3-r4-call-semantics` is preserved as the frozen structural-reference diagnostic lineage;
- the protected V2.4 compatibility candidate completed the former 26-contract parse-only remediation and binds all 22,540 identities with exact accepted-V9 token bytes, zero parse-only artifacts, zero unclassified call IR, and the required 22,539 primary + 1 identity-bound runtime split; it remains diagnostic history, not physical acceptance;
- the historical V2.5 candidate and its Stage-E blocker remain preserved diagnostic evidence;
- the accepted extractor is `v2.6-r4-call-semantics-deterministic-cfg-mutators`;
- protected-local V2.6 Stages A-D pass for 22,540 identities, exact accepted-V9 token bytes, zero missing/extra/invalid artifacts, and the exact 22,539 ordinary primary + one declared runtime-exception split; binding digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- three fresh 355-identity generations and three byte-identical semantic-evidence reports support V4's 349 persistent-storage WRITE corrections and 6 exact index-equivalent graphs;
- R4-D-011 physically accepts this exact V10 V2.6 graph/control root as immutable local evidence;
- R4-D-013 physically accepts the exact guarded-token successor `r4-v10-v26-guarded-tokens-v1` with digest `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`, 14,751 guarded selections and 7,789 exact historical fallbacks.

Logical authority:

- logical V3 grouping/roles/publication: **ACCEPTED** under R4-D-009 / ADR-R4-009;
- V3 groups: 22,394; max group size: 7; address-authority edges: 0;
- target/strength semantic counts unchanged from repaired-v2.

Current V3 supervised/evaluation surface:

- effective loss cells: **932**, every authorized target `1`;
- active `MODEL_SELECTION`: **71 contracts / 71 groups**;
- active `INTERNAL_AUDIT`: **72 contracts / 71 groups**;
- combined outcome-metric population: **143 contracts / 142 unique groups**;
- total outcome-metric cells recorded by V3: **143**; this must not be mislabeled as MODEL_SELECTION-only evidence;
- confirmed-negative targets: **0**;
- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`;
- no repaired/V3 full training checkpoint exists;
- no class is validated for general vulnerability discrimination, specificity/FPR, calibration, or production outcome claims.

## M4 bounded-model evidence boundary

R4-D-015 accepts the completed M4 program as **bounded mechanics/recovery/dynamics evidence only**:

- M4-A proved guarded V10/guarded-token CUDA forward/backward mechanics;
- M4-B proved exact-binding checkpoint/recovery, RNG restoration, deterministic sampler reconstruction and changed-lineage rejection;
- M4-C completed 8 epochs / 120 optimizer steps with finite execution and strong positive fit;
- positive NLL reached approximately `0.00109` and mean positive probability approximately `0.9989`;
- this does not establish negative discrimination because confirmed-negative evidence remains zero;
- `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD` is the accepted M4-D outcome;
- M5/full training, G8, threshold/calibration and promotion remain unauthorized.

The strong positive-fit result makes negative-side observability more important, not less: unknown/non-target behavior is still unconstrained by accepted negative truth.

## R4-D-016 `CallToUnknown` semantic boundary

R4-D-016 is accepted semantic authority for class index 0.

`CallToUnknown` is the raw/dynamic low-level unknown-target interaction class:

- included family: `call`, `delegatecall`, legacy `callcode`, applicable `staticcall`, and `send`;
- a checked raw low-level call can still qualify because target/callee uncertainty is class-defining;
- unchecked return/failure handling is additional positive evidence, not the sole defining condition;
- `transfer`, ordinary typed high-level interface calls, same-file/in-scope library calls, contract creation, and selfdestruct/suicide destinations do **not** qualify solely because the destination is external/caller-controlled/unverified.

The decision aligns canonical prose with original crosswalk/pattern semantics and R4-D-010/V10 consumers. It does not change class index/order, model output shape, source role mappings, or taxonomy schema version.

Authority:

- `adrs/ADR-R4-016-call-to-unknown-semantic-clarification.md`;
- `evidence/2026-10-20_call_to_unknown_semantic_alignment/decision.json`.

## Hardened V3 negative-evidence pilot state

R4-GAP-007 remains in progress and confirmed-negative support remains `NONE` for every enabled class.

Candidate #1:

- historical primary `NOT_CONFIRMED` rationale depended on the broad old taxonomy phrase;
- R4-D-016 makes that rationale definition-stale;
- authoritative queue state remains `UNKNOWN / PENDING_REVIEW / target None`;
- if reused, candidate #1 needs a new primary review under R4-D-016.

Candidate #2:

- existing source-first primary review is semantically compatible with R4-D-016;
- authoritative queue truth remains:

```text
UNKNOWN / PENDING_REVIEW / target None
```

- admissible independent agreeing verification is still missing.

Three external review attempts performed after M4 closure are diagnostic only and must not be vote-counted:

- one saw the prior primary-review direction through repository governance;
- one correctly stopped because the v1 blind bundle lacked sufficient neutral class/policy material;
- one disclosed candidate-specific prior context and produced a literal-old-taxonomy `NOT_CONFIRMED` result.

Blind-review v1 is historical only and retired for future adjudication. The next evidence artifact is the self-contained v2 bundle built by:

`docs/plan/ml-R4/scripts/p8_build_gap007_candidate2_independent_review_v2.py`

The generated v2 archive must contain complete source, manifest/hashes, review task, the neutral R4-D-016 definition/exclusions, and generic class-specific confirmed-negative policy, with no primary verdict/history leakage. The reviewer must have no candidate-specific prior conclusion in context.

Until admissible independent evidence returns:

- do not change candidate #2 target/state;
- do not infer target `0`;
- do not authorize M5/full training;
- do not infer that one/few future confirmed negatives support stable FPR, threshold, or calibration estimation.

The committed hardened pilot queue still contains:

- 200 candidate cells;
- 25 candidates for each of the eight enabled classes;
- 200 globally unique reserved V3 leakage groups;
- `group_uniqueness_scope=GLOBAL_ACROSS_ENABLED_CLASSES`;
- all candidates `PENDING_REVIEW` at queue creation;
- all current targets `None` unless separately adjudicated;
- source roles `TRAIN_UNLABELED` at queue creation;
- `negative_truth_claim=false`;
- manifest/source lineage validated by the final coherence snapshot.

Any accepted negative is initially `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY`.

## Selector/model-execution evidence boundary

`target_aware_guarded_v1` completed the corrected-V3 CPU coverage comparison and bounded identical-initialization CUDA comparison, then R4-D-012 promoted it for a new physical lineage and R4-D-013 accepted that lineage.

Durable CPU result:

- 1,018 records analyzed;
- 737 over-cap;
- 476 guarded improvements;
- 261 equal;
- 0 regressions;
- 0 failures.

Durable CUDA result:

- identical initialization true;
- 4/4 required worst-case probes completed;
- no Run12 weights;
- no checkpoint;
- historical comparison itself did not authorize promotion/training.

The full-population control-equivalence prerequisite passes: dynamic `historical_linspace_v1` reconstruction matches all 22,540 R4-D-011 `input_ids`, `attention_mask`, and selected-window-index payloads with zero failures. R4-D-013 subsequently accepts the guarded successor after D6 independently reproduced all 22,540 identities, the exact digest, selector counts, parent bindings, runtime split and deterministic probes.

The positive-only model evidence cannot establish false-positive discrimination.

## Durable current evidence

Final coherent V3 snapshot:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Current accepted V10 physical boundary:

`runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`.

Current M4 decision boundary:

- `runs/2026-10-17_PHASE8_m4_bounded_pilot_review_and_full_training_hold.md`;
- `adrs/ADR-R4-015-bounded-pilot-pass-full-training-hold.md`;
- `evidence/2026-10-17_phase8_m4_bounded_pilot/decision.json`.

Current `CallToUnknown` semantic authority:

- `adrs/ADR-R4-016-call-to-unknown-semantic-clarification.md`;
- `evidence/2026-10-20_call_to_unknown_semantic_alignment/decision.json`.

Candidate #2 primary-review record remains primary evidence only:

`runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md`.

The accepted pre-pilot logical baseline remains `runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`.

## Status vocabulary

- `VALIDATED_FOR_DEFINED_USE`
- `PROVISIONAL`
- `TRAINING_ONLY`
- `UNSUPPORTED_FOR_OUTCOME_CLAIMS`
- `DISABLED_PENDING_EVIDENCE`
