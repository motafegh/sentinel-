# Evidence Gap Register

No new contract review is permitted without an approved gap entry.

| Gap ID | Source/class/population | Missing decision evidence | Prior evidence searched | Why reuse is insufficient | Smallest required review | Intended role decision | Status | Approval |
|---|---|---|---|---|---|---|---|---|
| R4-GAP-001 | Web3Bugs/all classes | Entire source corpus, crosswalk, parser, connector (~3,500 contest-verified bugs with PoE) | Full repo search across data/ crosswalks/ config/ parsers/ | Declared Tier-1 Gold `enabled: true` but zero artifacts exist; cannot inform any current class decision | Acquire/configure only if a later baseline explicitly proposes adding Web3Bugs; the first DATA vNext baseline excludes the absent source | Tier-1 Gold or explicitly excluded | MASK_OR_EXCLUDE | 2026-08-11 — first-baseline exclusion selected; no source acquisition is required for G4 |
| R4-GAP-002 | DIVE/Arithmetic,DoS,UncheckedReturn,TimeManip,FrontRunning | Source-specific semantic precision evidence for five mapped DIVE strata not covered by prior EB/RE reviews | Recovered DIVE EB/RE manual reviews; Slither/Aderyn corroboration; DIVE labels/crosswalk; Phase-2 source authority and zero-semantics reconstruction; Phase-3 ledger | These five DIVE folder assertions actively map to `IntegerUO`, `DenialOfService`, `UnusedReturn`, `Timestamp`, and `TransactionOrderDependence`, but source-specific semantic precision was not established before Phase 4 | Deterministic leakage/project-group-aware DIVE-positive sample; checksum-bound source-only blind semantic review; evidence reconciliation; adaptive stop once the first-baseline role was bounded | Strong/weak/masked/excluded role per mapped DIVE stratum | RESOLVED | 2026-08-11 — delegated approval; 100-contract blind review completed. DoS/IntegerUO/Timestamp/UnusedReturn -> mask/exclude source assertion; TOD -> TRAIN_WEAK only, metrics/selection/calibration/acceptance masked |
| R4-GAP-003 | BCCC/CallToUnknown,ExternalBug,GasException | ML-propagated verification (GraphCodeBERT+HDBSCAN) deferred in Phase 5 Stage 5.5 | BCCC v1.4 verified labels are PROVISIONAL for these 3 classes | Stage 5.5 never ran; BCCC remains deferred rather than an active first-baseline source | Reopen only if Phase 5 proposes importing these provisional BCCC strata; otherwise keep them deferred/masked | PROVISIONAL->VERIFIED or deferred/excluded | MASK_OR_EXCLUDE | 2026-08-11 — provisional BCCC strata remain deferred for the first baseline; no Stage 5.5 run required for G4 |
| R4-GAP-004 | BCCC/all classes | 2-tool consensus baseline (Slither+Aderyn intersection) | `consensus.py` skeleton exists but patterns/results are empty; historical BCCC verification evidence recovered | Tool intersection is not ground truth and detector coverage is absent for some classes, including DoS/GasException | Execute only if a later benchmark-role decision specifically requires this baseline | Tier-C benchmark baseline | PROPOSED | Non-blocking future evidence opportunity; not required for first-baseline G4 |
| R4-GAP-005 | All classes | Echidna/fuzzing-based precision estimates | Full repo search — no Echidna cache or results recovered | No fuzzing evidence, but fuzzing is complementary rather than required to decide current active source roles | Define targeted fuzzing only for a later evaluation/case-study gap | Complementary fuzz evidence | PROPOSED | Non-blocking future evidence opportunity; not required for first-baseline G4 |
| R4-GAP-006 | All classes | Exploit-reproduction PoC tests | No retained Foundry/Hardhat vulnerability PoC corpus recovered | Functional exploit evidence would strengthen outcome validation but is not the smallest evidence needed for current DATA-source decisions | Create targeted PoCs only for later evidence-qualified evaluation/case studies, not a broad 5–10-per-class Phase-4 program | Outcome-validation/case-study evidence | PROPOSED | Non-blocking future evidence opportunity; not required for first-baseline G4 |
| R4-GAP-007 | V3 `TRAIN_UNLABELED` groups / eight enabled classes | Confirmed-negative class-specific evaluation evidence sufficient to observe false-positive behavior and support later evaluation-design decisions | Historical zeros, source silence, unlabeled cells, V2/V3 positive evidence, tool outputs, R4-GAP-002 review, and V2/V3 pilot-queue mechanics | None of the prior evidence establishes target `0`; historical/source absence is explicitly non-negative, and V2 queue reservations are obsolete after grouping correction | Start with the deterministic V3 pilot queue: 25 UNKNOWN/PENDING_REVIEW candidates per enabled class (200 distinct groups). Any `CONFIRMED_NEGATIVE` requires complete class-specific primary review plus independent agreeing verification under `confirmed_negative_evaluation.py`. Use observed pilot yield before expanding; the planning-only zero-FP bound is 59 confirmed negatives/class at 5% max FPR and 95% confidence, not a final gate. | Confirmed-negative **evaluation-only** evidence; any optimizer/training authority requires a separate later policy/ADR | APPROVED | 2026-08-16 — delegated technical approval after V3 acceptance; pilot queue generated cleanly with 200 PENDING_REVIEW cells, all target `None`, all `TRAIN_UNLABELED`, `negative_truth_claim=false`; adjudication not started |

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

## R4-GAP-007 entry evidence and stop line

- current logical authority: R4-D-009 / `sentinel-r4-vnext-v3`;
- generated queue: `data_module/data/r4-v3-logical-build/confirmed_negative_review_queue_v1.json`;
- queue state at authorization: 200 cells, 25 per enabled class, 200 reserved groups, all `PENDING_REVIEW`, all target `None`, all role `TRAIN_UNLABELED`, `negative_truth_claim=false`;
- detailed checkpoint: `runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`.

Queue membership is a review reservation only. Do not convert source absence, unlabeled state, static-tool silence, or a failed/ambiguous review into target `0`. Accepted negatives remain evaluation-only unless a later versioned decision explicitly grants optimizer authority.

## Approval delegation note

On 2026-08-11, the human owner delegated routine technical and project-governance approvals to the AI assistant, with the expectation that the assistant choose the technically strongest, scope-minimal option. This delegation does not authorize the assistant to fabricate evidence or to bypass explicit safety, integrity, irreversible-external-action, or value-dependent decision boundaries.

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
