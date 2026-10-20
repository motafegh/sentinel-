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
| R4-GAP-007 | V3 `TRAIN_UNLABELED` groups / eight enabled classes | Confirmed-negative class-specific evaluation evidence sufficient to observe false-positive behavior and support later evaluation design; for candidate #2, the neutral `CallToUnknown` semantic boundary must first be unambiguous | Historical zeros, source silence, unlabeled cells, V2/V3 positive evidence, tools, R4-GAP-002 review, pilot-queue mechanics, candidate #1 review, candidate #2 primary review, three non-authoritative independent-review attempts, canonical taxonomy and source crosswalks | None establishes target `0`. Candidate #2 independent adjudication is additionally blocked because the locked taxonomy includes the broad phrase “calls to unverified external addresses” while DIVE/SolidiFI operational mappings define `CallToUnknown` around raw/unchecked low-level call/send/delegatecall behavior. Reviewers can legitimately diverge until that neutral class boundary is resolved. | First resolve the `CallToUnknown` definition independently of candidate outcome. Then build a self-contained v2 blind bundle with the exact neutral class definition + generic negative-evidence rule and obtain genuinely isolated independent verification. Continue the hardened V3 queue only after that. | Confirmed-negative **evaluation-only** evidence; optimizer/training authority requires separate later policy/ADR | IN_PROGRESS | 2026-10-20 — candidate #1 remains `NOT_CONFIRMED`; candidate #2 remains UNKNOWN/PENDING_REVIEW/target `None`; three recent external attempts are diagnostic only and not admissible independent verification. Semantic-definition alignment is the active blocker. |
| R4-GAP-008 | repaired-v2 graph schema v9 / external-call semantics / CallToUnknown, ExternalBug, Reentrancy consumers | Population impact and remediation decision for v9 call-kind distortion | Candidate #2 source/graph review; full-population v9 edge-origin audit; V10 call-kind implementation; parse-only compatibility repair; bounded V2.5 analysis; fresh V2.6 Stages A-D; three-run 355-identity evidence; V4 audit; R4-D-011 acceptance review | Reuse is now sufficient only for the exact accepted V2.6 root and digest. Historical V2.3/V2.4/V2.5 roots remain diagnostic evidence and are not interchangeable. | Complete: V4 re-proved 349 persistent-storage WRITE corrections plus 6 exact index-equivalent graphs with zero unexplained drift; R4-D-011 accepted digest `d9f925...`. | Exact V10 V2.6 physical representation authority only; no label, selector, objective, threshold/calibration, or training authority | RESOLVED | 2026-09-02 — R4-D-011 / ADR-R4-011 accepts the exact 22,540-identity V2.6 root after refreshed binder and current-commit V4 review. |

## R4-GAP-002 closure evidence

- population manifest: `manifests/p4_gap002_population_manifest.json`
- frozen sample: `manifests/p4_gap002_initial_sample.jsonl`
- frozen sample SHA-256: `2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9`
- blind source bundle SHA-256: `2b1ce12fdd96819c89bbb9fe1dfb2d9aa992ec0a05ce32f651c6b834b97ddf38`
- machine-readable semantic review: `reviews/R4-GAP-002/p4_gap002_blind_semantic_review_v1.jsonl`
- review rows SHA-256: `7d7f0cce287f847df2376ac0f580abf6be05b46f6a2c90b5f00f9b34f8fc5473`
- review report: `findings/06_gap002_blind_semantic_review_report.json`
- interpretation/role decision: `findings/06_gap002_blind_semantic_review.md`

No `DOES_NOT_SUPPORT_POSITIVE` review verdict is a confirmed negative. The Phase-4 result controls source-assertion role only.

## R4-GAP-007 current stop line

Current logical authority is R4-D-009 / `sentinel-r4-vnext-v3`.

The hardened queue is durably captured at:

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

Queue membership is review reservation only. Do not convert source absence, unlabeled state, static-tool silence, or failed/ambiguous review into target `0`. Any `CONFIRMED_NEGATIVE` requires class-specific primary review plus an independent reviewer that agrees from sufficient evidence under an unambiguous governing class definition.

Accepted confirmed negatives remain `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY` unless a later versioned decision explicitly grants optimizer authority.

### Pilot progress — candidate #1

The first deterministic `CallToUnknown` candidate was completely reviewed. Whole-source and bound-representation review found a typed callback to a caller-supplied address plus a legacy `msg.sender.transfer(...)` value transfer. The governing taxonomy’s broad external-address clause prevented class-specific negative confirmation.

Candidate #1 is durably `NOT_CONFIRMED`; queue state remains UNKNOWN/PENDING_REVIEW/target `None`. A non-confirmed decision does not require independent verification.

Primary review closeout:

`runs/2026-08-21_PHASE8_gap007_candidate1_primary_review.md`

Machine-readable adjudication and evaluation:

- `reviews/R4-GAP-007/candidate1_primary_adjudication_v1.jsonl`
- `reviews/R4-GAP-007/candidate1_evaluation_v1.json`

### Candidate #2 authoritative state

Candidate #2 (`r4neg-bfe90ef82e33a324d612256a5d4053c6`) completed source-first primary review. The primary result supports `CONFIRMED_NEGATIVE` for `CallToUnknown`, but the candidate remains UNKNOWN / PENDING_REVIEW / target `None` because admissible independent verification has not completed.

Candidate #2 primary record:

`runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md`

Blind independent-review archive v1:

`review_bundles/r4_gap007_candidate2_independent_review_v1.zip`

SHA-256: `2e7f48c9648097624406d167266a42a31055f222a0f468a0453b2f353b343f1a`.

### 2026-10-20 independent-review diagnostic attempts

Three external attempts are **not adjudication evidence**:

1. a repository-aware reviewer saw the prior candidate #2 direction in `CLAUDE.md` before returning a substantive negative conclusion;
2. a strict bundle-only reviewer correctly stopped because v1 lacks the governing class definition and complete generic negative-evidence rule, and also disclosed candidate-specific prior context;
3. a ZIP + neutral-material reviewer disclosed candidate-specific prior context and returned `NOT_CONFIRMED` under a literal reading of the broad taxonomy phrase “calls to unverified external addresses.”

Do not vote-count or reconcile these results as candidate truth. They instead exposed a neutral semantic inconsistency:

- `data_module/sentinel_data/labeling/schema/taxonomy.yaml` broadly includes unchecked low-level calls **or calls to unverified external addresses**;
- `data_module/sentinel_data/labeling/crosswalks/dive.yaml` says `CallToUnknown` is reserved for raw low-level `call` / `send` / `delegatecall` patterns where the callee is unverified;
- `data_module/sentinel_data/labeling/crosswalks/solidifi.yaml` likewise explains the class through unchecked low-level `send/call/delegatecall` behavior.

This ambiguity can materially change whether an address-level `transfer` is considered potentially in-scope. Therefore **no further candidate #2 review is permitted until the class-definition audit resolves the neutral boundary**.

Active working record:

`docs/plan/system-finalization/technical-completion/working/2026-10-20_ml_gap007_call_to_unknown_semantic_alignment.md`

After the definition is resolved, build a self-contained v2 blind review package containing the exact neutral class definition, generic negative-evidence rule, complete source/task and hashes, with no prior-candidate conclusion. The independent reviewer must also have no candidate-specific prior context.

The 2026-08-16 hardened evidence-snapshot closeout remains the accepted pre-pilot baseline.

## R4-GAP-008 resolution and current physical stop line

R4-GAP-008 began with the v9 representation-semantic contradiction exposed by candidate #2. The full-population audit and R4-D-010 resolved the policy question: accepted v9 remains immutable reproduction evidence but is ineligible for the new full training lineage, which must use a separately versioned V10 representation.

Subsequent execution closed both bounded technical sub-blockers:

1. **Parse-only remediation complete.** The protected V2.4 diagnostic candidate contains all 22,540 identities, exact accepted-V9 token bytes, zero parse-only outputs, zero unclassified call IR, and the exact required runtime split of 22,539 primary Slither-0.10 artifacts plus one identity-bound Slither-0.11.5 exception.
2. **Unexpected structural drift complete.** Under extractor `v2.5-r4-call-semantics-deterministic-cfg`, three fresh bounded generations of the exact 20 identities from transition audit v2 resolved all 20 with zero unexplained drift: 8 exact node-index-invariant labelled graph-equivalence identities and 12 persistent-storage WRITE corrections backed by exact semantic evidence.

The final bounded evidence is recorded at:

- `reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`;
- `reviews/R4-GAP-008/v10_v25_reproducibility_probe_v2.json` (protected-local/generated evidence when present);
- `runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`.

The full-candidate execution then supplied concrete evidence that reopens this gap only for population-wide structural proof. Stages A-D pass for all 22,540 identities. Stage E passes base mechanics but finds 311 raw non-parse-only drifts; 298 are outside the old bounded evidence classes. A diagnostic probe classifies 298 feature/metadata drifts, 12 node-order cases, and one semantic structure drift, while duplicate semantic identities prevent unique node matching in 128 contracts.

R4-B008 is closed for the exact R4-D-011 root and digest. Preserve the accepted candidate, all Stage A-D reports, the three-run 355-identity evidence, the V4 audit, and the acceptance record as immutable evidence. Any later graph semantic or selector change requires a new versioned root and acceptance chain.

Current construction protocol:

`runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`

Current structural-evidence protocol:

`runs/2026-08-30_PHASE8_v10_v25_full_population_structural_evidence_plan.md`

Governing decision:

`adrs/ADR-R4-010-versioned-external-call-representation-correction.md`

Do not mutate accepted v9/repaired-v2 artifacts, the frozen V2.3 structural reference, protected V2.4/V2.5 diagnostics, or the R4-D-011 accepted V2.6 root. Physical acceptance does not authorize G8/full training.

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
