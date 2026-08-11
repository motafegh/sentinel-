# Evidence Gap Register

No new contract review is permitted without an approved gap entry.

| Gap ID | Source/class/population | Missing decision evidence | Prior evidence searched | Why reuse is insufficient | Smallest required review | Intended role decision | Status | Approval |
|---|---|---|---|---|---|---|---|---|
| GAP-001 | Web3Bugs/all classes | Entire source corpus, crosswalk, parser, connector (~3,500 contest-verified bugs with PoE) | Full repo search across data/ crosswalks/ config/ parsers/ | Declared Tier-1 Gold `enabled: true` but zero artifacts exist; cannot inform ANY class decision | Acquire or configure Web3Bugs source; define crosswalk; or explicitly disable and accept absence | Tier-1 Gold or explicitly excluded | PROPOSED | — |
| GAP-002 | DIVE/Arithmetic,DoS,UncheckedReturn,TimeManip,BadRandom,FrontRunning | Per-class precision/recall evidence for 6 of 8 DIVE classes | DIVE review mds cover EB/RE only | 75% of DIVE classes have zero precision evidence; cannot determine label quality | Source-code review of random sample per class (e.g. 50 contracts each = 300 total) | KEEP/DROP per class | PROPOSED | — |
| GAP-003 | BCCC/CallToUnknown,ExternalBug,GasException | ML-propagated verification (GraphCodeBERT+HDBSCAN) deferred in Phase 5 Stage 5.5 | BCCC v1.4 verified labels are PROVISIONAL for these 3 classes | VRAM conflict on RTX 3070 8GB during active SENTINEL training; Stage 5.5 never ran | Re-run Stage 5.5 ML propagation when VRAM available | PROVISIONAL->VERIFIED or alternative approach | PROPOSED | — |
| GAP-004 | BCCC/all classes | 2-tool consensus baseline (Slither+Aderyn intersection) | consensus.py skeleton exists but patterns/ and results/ are empty | Current BCCC verification used Phase 5 rules, not 2-tool consensus; GasException+DoS have no Aderyn coverage | Execute consensus.py on BCCC subset; design replacement for tools with no detector coverage | Tier C benchmark baseline | PROPOSED | — |
| GAP-005 | All classes | Echidna/fuzzing-based precision estimates | Full repo search — no echidna cache or results found | Zero fuzzing evidence for any class | Define fuzzing targets for highest-impact classes (Reentrancy, ExternalBug) | Complementary fuzz evidence | PROPOSED | — |
| GAP-006 | All classes | Exploit reproduction PoC tests | No Foundry/Hardhat vulnerability PoC files in contracts/test/ | No functional proof that predicted vulnerabilities are exploitable | Write PoC for 5-10 most confident predictions per class | Outcome-validation evidence | PROPOSED | — |

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
