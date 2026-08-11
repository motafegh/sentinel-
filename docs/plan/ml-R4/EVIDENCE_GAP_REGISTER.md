# Evidence Gap Register

No new contract review is permitted without an approved gap entry.

| Gap ID | Source/class/population | Missing decision evidence | Prior evidence searched | Why reuse is insufficient | Smallest required review | Intended role decision | Status | Approval |
|---|---|---|---|---|---|---|---|---|
| R4-GAP-001 | Web3Bugs/all classes | Entire source corpus, crosswalk, parser, connector (~3,500 contest-verified bugs with PoE) | Full repo search across data/ crosswalks/ config/ parsers/ | Declared Tier-1 Gold `enabled: true` but zero artifacts exist; cannot inform any current class decision | Acquire/configure the source only if first-baseline inclusion is explicitly desired; otherwise leave absent and propose explicit exclusion in Phase 5 | Tier-1 Gold or explicitly excluded | PROPOSED | — |
| R4-GAP-002 | DIVE/Arithmetic,DoS,UncheckedReturn,TimeManip,FrontRunning | Source-specific semantic precision evidence for five mapped DIVE strata not covered by prior EB/RE reviews | Recovered DIVE EB/RE manual reviews; Slither/Aderyn corroboration; DIVE labels/crosswalk; Phase-2 source authority and zero-semantics reconstruction; Phase-3 ledger | These five DIVE folder assertions actively map to `IntegerUO`, `DenialOfService`, `UnusedReturn`, `Timestamp`, and `TransactionOrderDependence`, but source-specific semantic precision is not established. BCCC/tool evidence cannot be transferred as DIVE label authority. | After explicit approval: deterministic leakage/project-group-aware DIVE-positive sample per stratum; blind semantic review first, evidence reconciliation second, adaptive expansion only for ambiguous role decisions. `Bad Randomness` is excluded from this review because the current DIVE crosswalk drops it and it can remain unknown/masked. | Strong/weak/masked/excluded role per mapped DIVE stratum | PROPOSED | — |
| R4-GAP-003 | BCCC/CallToUnknown,ExternalBug,GasException | ML-propagated verification (GraphCodeBERT+HDBSCAN) deferred in Phase 5 Stage 5.5 | BCCC v1.4 verified labels are PROVISIONAL for these 3 classes | Stage 5.5 never ran; however BCCC remains deferred rather than an active first-baseline source | Run Stage 5.5 only if Phase 5 proposes importing these BCCC strata; otherwise keep deferred/masked | PROVISIONAL->VERIFIED or deferred/excluded | PROPOSED | — |
| R4-GAP-004 | BCCC/all classes | 2-tool consensus baseline (Slither+Aderyn intersection) | `consensus.py` skeleton exists but patterns/results are empty; historical BCCC verification evidence recovered | Tool intersection is not ground truth and detector coverage is absent for some classes, including DoS/GasException | Execute only if a later benchmark-role decision specifically requires this baseline | Tier-C benchmark baseline | PROPOSED | — |
| R4-GAP-005 | All classes | Echidna/fuzzing-based precision estimates | Full repo search — no Echidna cache or results recovered | No fuzzing evidence, but fuzzing is complementary rather than required to decide current active source roles | Define targeted fuzzing only for a later evaluation/case-study gap | Complementary fuzz evidence | PROPOSED | — |
| R4-GAP-006 | All classes | Exploit-reproduction PoC tests | No retained Foundry/Hardhat vulnerability PoC corpus recovered | Functional exploit evidence would strengthen outcome validation but is not the smallest evidence needed for current DATA-source decisions | Create targeted PoCs only for later evidence-qualified evaluation/case studies, not a broad 5–10-per-class Phase-4 program | Outcome-validation/case-study evidence | PROPOSED | — |

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
