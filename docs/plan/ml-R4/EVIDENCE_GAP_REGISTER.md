# Evidence Gap Register

No new contract review is permitted without an approved gap entry.

| Gap ID | Source/class/population | Missing decision evidence | Prior evidence searched | Why reuse is insufficient | Smallest required review | Intended role decision | Status | Approval |
|---|---|---|---|---|---|---|---|---|
| R4-GAP-001 | Web3Bugs/all classes | Entire source corpus, crosswalk, parser, connector (~3,500 contest-verified bugs with PoE) | Full repo search across data/ crosswalks/ config/ parsers/ | Declared Tier-1 Gold `enabled: true` but zero artifacts exist; cannot inform any current class decision | Acquire/configure only if a later baseline explicitly proposes adding Web3Bugs; first DATA vNext baseline excludes absent source | Tier-1 Gold or explicitly excluded | MASK_OR_EXCLUDE | 2026-08-11 — first-baseline exclusion selected |
| R4-GAP-002 | DIVE/Arithmetic,DoS,UncheckedReturn,TimeManip,FrontRunning | Source-specific semantic precision for five mapped DIVE strata | Recovered DIVE reviews; Slither/Aderyn corroboration; DIVE labels/crosswalk; Phase-2/3 evidence | Five mapped strata lacked source-specific semantic precision | Deterministic leakage-aware sample + checksum-bound blind semantic review | Strong/weak/masked/excluded role per stratum | RESOLVED | 2026-08-11 — 100-contract review completed; DoS/IntegerUO/Timestamp/UnusedReturn masked, TOD TRAIN_WEAK |
| R4-GAP-003 | BCCC/CallToUnknown,ExternalBug,GasException | ML-propagated verification deferred in Phase 5 | BCCC v1.4 verified labels provisional | Stage 5.5 never ran; BCCC remains deferred | Reopen only if importing these provisional BCCC strata | PROVISIONAL→VERIFIED or deferred | MASK_OR_EXCLUDE | 2026-08-11 — deferred for first baseline |
| R4-GAP-004 | BCCC/all classes | 2-tool consensus baseline | skeleton exists; results absent | Tool intersection is not ground truth | Execute only if later benchmark decision requires it | Tier-C benchmark baseline | PROPOSED | Non-blocking |
| R4-GAP-005 | All classes | Echidna/fuzzing precision estimates | no retained fuzzing evidence | complementary, not required for current DATA source roles | targeted future fuzzing only | Complementary fuzz evidence | PROPOSED | Non-blocking |
| R4-GAP-006 | All classes | Exploit-reproduction PoC tests | no retained PoC corpus | useful for outcome validation but not smallest current evidence | targeted future PoCs only | Outcome validation/case study | PROPOSED | Non-blocking |
| R4-GAP-007 | V3 `TRAIN_UNLABELED` groups / eight enabled classes | Confirmed-negative class-specific evaluation evidence sufficient to observe false-positive behavior and support later evaluation design | Historical zeros, source silence, unlabeled cells, V2/V3 positive evidence, tools, R4-GAP-002 review, pilot-queue mechanics | None establishes target `0`; historical/source absence is explicitly non-negative | Use the **committed hardened V3 queue**: 25 UNKNOWN/PENDING_REVIEW candidates per enabled class with one globally unique leakage group per queued cell. Any `CONFIRMED_NEGATIVE` requires complete class-specific primary review plus independent agreeing verification. Use observed pilot yield before expanding; 59 confirmed negatives/class at 5% max FPR and 95% confidence remains planning-only | Confirmed-negative **evaluation-only** evidence; optimizer/training authority requires separate later policy/ADR | IN_PROGRESS | 2026-08-21 — candidate #1 is `NOT_CONFIRMED`. Candidate #2 primary review supports a negative but awaits genuinely independent verification from the blind bundle; it remains UNKNOWN/target `None`. Confirmed negatives remain zero. |
| R4-GAP-008 | repaired-v2 graph schema v9 / `EXTERNAL_CALL` edge / CallToUnknown, ExternalBug, Reentrancy consumers | Population impact and remediation decision for library calls emitted as external-call edges while Solidity `Transfer`/`Send` IR is not emitted by the same edge builder | Candidate #2 full source/graph inspection; graph extractor source/tests; CallToUnknown pattern/library exclusion; historical BCCC false-positive analysis | Candidate #2 has 30 type-11 edges and all 30 are same-file `SafeMath` library calls, while its real Ether `transfer` has no type-11 edge. Existing tests document library counting as known behavior, but the class pattern explicitly excludes library calls. One candidate cannot establish population impact or safe remediation lineage | Run a read-only full-population v9 edge-origin audit plus queue-focused impact analysis; preserve v9 artifacts; then decide whether a versioned extractor/schema candidate and regeneration are required before G8 | Representation semantic-integrity decision; no label, selector, or training authority | RESOLVED | 2026-08-22 — v10 full 22,540-identity generation/binding/transition mechanics pass, but physical acceptance is rejected for now because 26 parse-only contracts lack complete call IR. R4-B008 remains open. |

## R4-GAP-002 closure evidence

- population manifest: `manifests/p4_gap002_population_manifest.json`
- frozen sample: `manifests/p4_gap002_initial_sample.jsonl`
- frozen sample SHA-256: `2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9`
- blind source bundle SHA-256: `2b1ce12fdd96819c89bbb9fe1dfb2d9aa992ec0a05ce32f651c6b834b97ddf38`
- machine-readable semantic review: `reviews/R4-GAP-002/p4_gap002_blind_semantic_review_v1.jsonl`
- review rows SHA-256: `7d7f0cce287c847df2376ac0f580abf6be05b46f6a2c90b5f00f9b34f8fc5473`
- review report: `findings/06_gap002_blind_semantic_review_report.json`
- interpretation/role decision: `findings/06_gap002_blind_semantic_review.md`

No `DOES_NOT_SUPPORT_POSITIVE` review verdict is a confirmed negative. The Phase-4 result controls source-assertion role only.

## R4-GAP-007 current stop line

Current logical authority is R4-D-009 / `sentinel-r4-vnext-v3`.

The hardened queue is now durably captured at:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/confirmed_negative_review_queue_v1.json`

It was regenerated under source commit `83bd566b9c4f4f653e530c2c0f5c990858dd759d`, passed snapshot coherence, and was committed in snapshot commit `44fbb9c1d2033be8002fe404d650cf09f08b0f29`.

Current durable queue invariants:

- 200 queued cells;
- 25 candidates per enabled class;
- 200 reserved groups;
- `group_uniqueness_scope=GLOBAL_ACROSS_ENABLED_CLASSES`;
- every candidate `PENDING_REVIEW`;
- every current target `None`;
- every role at queue creation `TRAIN_UNLABELED`;
- `negative_truth_claim=false`;
- queue bound to the accepted V3 publication manifest and hardened source commit.

The old/pre-hardening queue is obsolete for review. **R4-GAP-007 pilot adjudication may proceed only from the committed hardened queue.**

Queue membership is review reservation only. Do not convert source absence, unlabeled state, static-tool silence, or failed/ambiguous review into target `0`. Any `CONFIRMED_NEGATIVE` requires class-specific primary review plus an independent reviewer that agrees from sufficient evidence.

Accepted confirmed negatives remain `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY` unless a later versioned decision explicitly grants optimizer authority.

### Pilot progress — 2026-08-21

The pilot has started with the first deterministic `CallToUnknown` queue candidate:

```text
candidate_id = r4neg-f6a71e420a116cb4b9a334ba961ba1b6
contract_id  = defe4690028dc863df4611176a4c35f0ffd0bbc90f61db2bd4f25f5ad7f2a384
group_id     = r4grp-91091daa51a561493045bd21a5d321fa
source       = dive
state        = UNKNOWN / PENDING_REVIEW
target       = None
```

Complete whole-source and bound-representation primary review identified a typed callback `spender.receiveApproval(...)` to a caller-supplied address and a legacy Solidity `msg.sender.transfer(...)` value transfer. Solidity 0.4.18 error semantics and targeted Slither analysis corroborate that neither is an unchecked raw low-level call. However, the governing taxonomy also includes calls to unverified external addresses, so the dynamic typed callback is contradictory or ambiguous positive evidence and prevents class-specific negative confirmation.

Candidate #1 was therefore adjudicated `NOT_CONFIRMED`. The explicit-path validator passed with one `NOT_CONFIRMED` decision, zero accepted cells, no errors, and all training/threshold/calibration authorizations false. The queue remains immutable; candidate #1 remains `UNKNOWN` / `PENDING_REVIEW`, target `None`, and confirmed negatives remain zero. A non-confirmed decision does not require independent verification.

Primary review closeout:

`runs/2026-08-21_PHASE8_gap007_candidate1_primary_review.md`

Machine-readable adjudication and evaluation:

- `reviews/R4-GAP-007/candidate1_primary_adjudication_v1.jsonl`
- `reviews/R4-GAP-007/candidate1_evaluation_v1.json`

Prior partial-review handoff:

`runs/2026-08-21_PHASE8_gap007_candidate1_local_handoff.md`

Candidate #2 (`r4neg-bfe90ef82e33a324d612256a5d4053c6`) then completed source-first primary review. The primary result supports `CONFIRMED_NEGATIVE` for `CallToUnknown`, but the candidate remains UNKNOWN / PENDING_REVIEW / target `None` because genuinely independent verification is still pending. The primary reviewer must not self-verify.

Candidate #2 primary record:

`runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md`

Blind independent-review archive:

`review_bundles/r4_gap007_candidate2_independent_review_v1.zip`

SHA-256: `2e7f48c9648097624406d167266a42a31055f222a0f468a0453b2f353b343f1a`.

The 2026-08-16 hardened evidence-snapshot closeout remains the accepted pre-pilot baseline and must still be read before the current primary-review closeout.

## R4-GAP-008 resolution and current stop line

Candidate #2 primary review exposed a representation-semantic contradiction that must be quantified before treating v9 graph signals as trustworthy evaluation or future-training inputs:

- all 30 candidate #2 `EXTERNAL_CALL` edges point to same-file `SafeMath` library-call nodes;
- the actual `_customerAddress.transfer(...)` node has no `EXTERNAL_CALL` edge;
- `verification/patterns/CallToUnknown.yaml` explicitly excludes library calls;
- `graph_extractor.py` intentionally treats `LibraryCall` as high-level/external in the type-11 edge path and does not add `Transfer`/`Send` there;
- the semantic checker treats any type-11 edge as a positive `CallToUnknown` and `ExternalBug` signal.

The full-population audit resolved the evidence question and R4-D-010 accepted the remediation boundary. Until the versioned v10 candidate passes local acceptance:

- do not mutate accepted v9/repaired-v2 artifacts;
- do not reinterpret type-11 presence or absence as class truth;
- do not authorize G8/full training;
- candidate review may continue from whole source, but any representation-based conclusion must carry the v9 limitation;
- do not change extractor behavior under the same schema/extractor identity.

Current audit record:

`runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md`

Governing decision:

`adrs/ADR-R4-010-versioned-external-call-representation-correction.md`

Repository/local implementation handoff:

`runs/2026-08-21_PHASE8_v10_external_call_implementation_handoff.md`

Repository implementation and bounded source-reviewed regression evidence:

`runs/2026-08-21_PHASE8_v10_implementation_and_local_regression.md`

The implementation now emits exact v10 call kinds, records unclassified IR and
IR-to-CFG mapping failures, and fails population binding unless classified,
emitted, and observed call counts agree. Candidate #1 reproduces one typed
high-level callback plus one transfer; candidate #2 reproduces 30 library calls
plus one transfer. Full generation/binding/transition mechanics pass for all
22,540 identities with digest `6087dc6d...b0260`. Physical acceptance remains
rejected for now under R4-B008 because 26 parse-only contracts cannot establish
complete call IR.

## Approval delegation note

On 2026-08-11, the human owner delegated routine technical and project-governance approvals to the AI assistant, with the expectation that the assistant choose the technically strongest, scope-minimal option. This delegation does not authorize fabricated evidence or bypassing safety, integrity, irreversible-external-action, or value-dependent decision boundaries.

## Allowed gap reasons

- `MISSING_RAW_EVIDENCE`
- `CONTRADICTORY_PRIOR_CONCLUSIONS`
- `UNREVIEWED_SOURCE_CLASS_STRATUM`
- `INSUFFICIENT_CONFIRMED_NEGATIVES`
- `INSUFFICIENT_ACCEPTANCE_SUPPORT`
- `AMBIGUOUS_CROSSWALK_EFFECT`
- `UNRESOLVED_DUPLICATE_OR_PROJECT_FAMILY`
- `DEFINITION_CHANGED_WITH_MATERIAL_EFFECT`
- `ARTIFACT_INTEGRITY_FAILURE`

## Status

- `PROPOSED`
- `APPROVED`
- `IN_PROGRESS`
- `RESOLVED`
- `REJECTED`
- `MASK_OR_EXCLUDE`
