# Previous Evidence Register

Register every prior investigation, review batch, tool output, report, script, and conclusion before starting new review.

| Evidence set ID | Source/class scope | Artifact IDs | Method | Contract-class coverage | Raw evidence retained? | Conclusion | Status | Imported to ledger? | Duplicate work risk |
|---|---|---|---|---:|---|---|---|---|---|
| R4-PREV-DIVE-EBRE | DIVE ExternalBug/Reentrancy | R4-P0-EVD-001 | Manual + source/tool investigation (150 contracts, seed=42) | 75 EB + 75 RE | YES (per-contract TP/FP tables) | Folder TP rates: EB 5.3%, RE 4.2%. Prior DROP recommendation. | TO_RECOVER | NO | HIGH |
| R4-PREV-DIVE-SLITHER | DIVE ExternalBug/Reentrancy | R4-P0-EVD-002 | Manual review of DIVE-Slither agreed subset (175 contracts, seed=7) | 100 EB + 75 RE | YES (per-contract TP/FP criteria + verdicts) | Second-round review of agreed subset. | TO_RECOVER | NO | HIGH |
| R4-PREV-DIVE-CORROB | DIVE ExternalBug/Reentrancy | R4-P0-EVD-003 | Tool corroboration (Slither agreement) | 6804 EB agreed (42.7%), 8258 RE agreed (75.0%) | YES (agreed_shas lists) | Slither corroboration of DIVE automated labels. | TO_RECOVER | NO | MEDIUM |
| R4-PREV-DIVE-ADERYN1 | DIVE ExternalBug/Reentrancy | R4-P0-EVD-001 (audit dir) | Tool output (Aderyn per-contract) | 573 contracts | YES (per-contract Aderyn outputs) | Aderyn per-contract results on DIVE subset. | TO_RECOVER | NO | MEDIUM |
| R4-PREV-DIVE-ADERYN2 | DIVE ExternalBug/Reentrancy | R4-P0-EVD-001 (audit dir) | Tool output (Aderyn on disagreed) | Disagreed subset | YES | Aderyn run on Slither-disagreed DIVE subset. | TO_RECOVER | NO | MEDIUM |
| R4-PREV-DIVE-SLITHER-CACHE | DIVE all classes | R4-P0-EVD-001 (slither_cache) | Tool output (Slither per-contract) | 17287 files | YES | Slither per-contract outputs for DIVE. | TO_RECOVER | NO | LOW |
| R4-PREV-DIVE-LABELS | DIVE all classes | R4-P0-LBL-001 | Source labels (automated CSV) | 22330 rows (8 DASP classes) | YES (original CSV) | Source-native automated multi-label CSV. | TO_RECOVER | NO | LOW |
| R4-PREV-DIVE-OTHER | DIVE non-EB/RE classes | — | — | 0 classes reviewed | NO | No per-class review for Arithmetic/DoS/UncheckedReturn/TimeManip/BadRandom/FrontRunning. | UNAVAILABLE | NO | HIGH |
| R4-PREV-DIVE-2NDREVIEW | DIVE all reviewed | — | — | — | NO | Both DIVE review mds are single-author; no second reviewer. | UNAVAILABLE | NO | MEDIUM |
| R4-PREV-BCCC | BCCC all classes | R4-P0-EVD-005 | Deep dive (5-phase) | Full 10-class | YES (CSVs/parquet/scripts/definitions/batches) | Reentrancy 89% FP, CallToUnknown 86.9% FP. v1.4 verified labels. | TO_RECOVER | NO | HIGH |
| R4-PREV-BCCC-V14 | BCCC all classes | R4-P0-EVD-004 | Verified labels | All 10 BCCC classes | YES | BCCC Phase 5 verified labels. | TO_RECOVER | NO | HIGH |
| R4-PREV-BCCC-2TOOL | BCCC 2-tool consensus | R4-P0-EVD-005 (benchmarks) | Benchmark consensus | — | NO (patterns/results empty) | consensus.py exists but run never executed. | TO_RECOVER | NO | MEDIUM |
| R4-PREV-SOLIDIFI | SolidiFI all classes | R4-P0-EVD-006 (raw dir) | Source framework | 7 vuln types | YES (injection framework + tool results) | ISSTA 2020 injection benchmark. | TO_RECOVER | NO | LOW |
| R4-PREV-SMARTBUGS-CUR | SmartBugs Curated all classes | R4-P0-EVD-006 | Source corpus + recall test | 143 hand-labeled | YES (vulnerabilities.json) | 143 hand-labeled DASP contracts (ICSE 2020). 94.4% recall. | TO_RECOVER | NO | LOW |
| R4-PREV-SMARTBUGS-WILD | SmartBugs Wild | R4-P0-EVD-006 (ml/data) | Source corpus | 47K contracts | YES (results_wild.json) | 47K mainnet contracts (pretraining only). | TO_RECOVER | NO | LOW |
| R4-PREV-WEB3BUGS | Web3Bugs all classes | — | — | ~3500 contest-verified | NO | Declared Tier-1 Gold but never acquired. No data/crosswalk/parser. | UNAVAILABLE | NO | HIGH |
| R4-PREV-DEFIHACKLABS | DeFiHackLabs | R4-P0-EVD-007 (preprocessed) | Source corpus | 47 processed (715 dropped) | PARTIAL | Foundry project; forge-std prevents solc. DEFERRED. | TO_RECOVER | NO | MEDIUM |
| R4-PREV-MANUAL | Manual hand-written | R4-P0-EVD-006 (manual dir) | Manual review | 83 .sol + 83 .json (11 classes) | YES | Hand-written contract library with paired labels. | TO_RECOVER | NO | LOW |
| R4-PREV-AI-REPORTS | AI audit reports | R4-P0-EVD-006 (agents) | AI review output | 22 report JSONs | YES | AI-agent audit runs over manual corpus. | TO_RECOVER | NO | LOW |
| R4-PREV-BENCHMARK | Benchmark case studies | R4-P0-BMK-001 | Benchmark | 66 contracts 5-tier OOD | YES | 66-contract 0%-contamination quickstart. | TO_RECOVER | NO | LOW |
| R4-PREV-ECHIDNA | Echidna tool | — | Tool output | — | NO | No echidna cache or results found. | UNAVAILABLE | NO | LOW |
| R4-PREV-EXPLOIT-POC | Exploit reproductions | — | Exploit reproduction | — | NO | No Foundry/Hardhat vulnerability PoC tests. | UNAVAILABLE | NO | LOW |
| R4-PREV-DATA-AUDIT | Data module audit | R4-P0-EVD-007 | Audit report | 45% leakage + DoS patch | YES | Documents leakage finding and DoS/Reentrancy co-occurrence patch. | TO_RECOVER | NO | LOW |
| R4-PREV-CROSSWALKS | Crosswalks (4 of 5) | R4-P0-XWK-001..003 | Pipeline config | dive/solidifi/smartbugs/defihacklabs | YES | 4 crosswalk YAMLs exist; web3bugs.yaml UNAVAILABLE. | TO_RECOVER | NO | LOW |

## Status

- `TO_RECOVER`
- `RECOVERED_VERIFIED`
- `RECOVERED_PARTIAL`
- `CONCLUSION_ONLY`
- `UNAVAILABLE`
- `SUPERSEDED`
