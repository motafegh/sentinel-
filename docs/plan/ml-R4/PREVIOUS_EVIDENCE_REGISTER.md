# Previous Evidence Register

Register every prior investigation, review batch, tool output, report, script, and conclusion before starting new review.

| Evidence set ID | Source/class scope | Artifact IDs | Method | Contract-class coverage | Raw evidence retained? | Conclusion | Status | Imported to ledger? | Duplicate work risk |
|---|---|---|---|---:|---|---|---|---|---|
| R4-PREV-DIVE-EBRE | DIVE ExternalBug/Reentrancy | R4-P0-EVD-001 | Manual + source/tool investigation (150 contracts, seed=42) | 75 EB + 75 RE | YES (per-contract TP/FP tables) | Folder TP rates: EB 5.3%, RE 4.2%. Prior DROP recommendation. Contradiction: per-table shows 3 TP / 72 FP but tally says 4 TP / 71 FP. | RECOVERED_VERIFIED | YES | HIGH |
| R4-PREV-DIVE-SLITHER | DIVE ExternalBug/Reentrancy | R4-P0-EVD-002 | Manual review of DIVE-Slither agreed subset (175 contracts, seed=7) | 100 EB + 75 RE | YES (per-contract TP/FP criteria + verdicts) | DIVE+Slither agreement does NOT improve precision (EB 4.0%, RE 2.7%). Disjoint sampling frame from crosswalk review. | RECOVERED_VERIFIED | YES | HIGH |
| R4-PREV-DIVE-CORROB | DIVE ExternalBug/Reentrancy | R4-P0-EVD-003 | Tool corroboration (Slither agreement) | 6804 EB agreed (42.7%), 8258 RE agreed (75.0%) | YES (agreed_shas lists) | Slither corroboration of DIVE automated labels. No per-contract detail. Tool-correlated with D-A group. | RECOVERED_VERIFIED | YES | MEDIUM |
| R4-PREV-DIVE-ADERYN1 | DIVE ExternalBug/Reentrancy | data_module/audit/2026-06-18_dive_aderyn_per_contract_v1.json | Tool output (Aderyn per-contract) | 175 contracts (same as Slither-agreed review) | YES (per-contract Aderyn detector firings) | Triple-agreement (DIVE+Slither+Aderyn) worse than Slither alone (EB 3.0%, RE 1.7%). FULL overlap with Slither-agreed review. | RECOVERED_VERIFIED | YES | MEDIUM |
| R4-PREV-DIVE-ADERYN2 | DIVE ExternalBug/Reentrancy | data_module/audit/2026-06-18_dive_aderyn_on_slither_disagreed_v1.json | Tool output (Aderyn on disagreed) | 400 contracts (200 EB + 200 RE Slither-disagreed) | YES (per-contract JSON) | Aderyn-only on Slither-disagreed: 0.0% TP (0/30 EB sampled). Aderyn added signal is noise. | RECOVERED_VERIFIED | YES | MEDIUM |
| R4-PREV-DIVE-SLITHER-CACHE | DIVE all classes | data_module/data/slither_cache/dive/ (17,287 files) | Tool output (Slither per-contract) | 17287 files | YES | Slither per-contract outputs. Tool-correlated with DIVE-CORROB. No individual SHA-256 hashes. | RECOVERED_PARTIAL | NO | LOW |
| R4-PREV-DIVE-ADERYN-CACHE | DIVE all classes | data_module/data/aderyn_cache/dive/ (573 files) | Tool output (Aderyn per-contract) | 573 .aderyn.json files | YES | Aderyn per-contract outputs. Tool-correlated. No individual SHA-256 hashes. | RECOVERED_PARTIAL | NO | LOW |
| R4-PREV-DIVE-LABELS | DIVE all classes | R4-P0-LBL-001 | Source labels (automated CSV) | 22330 rows (8 DASP classes) | YES (original CSV) | Source-native automated multi-label CSV. 170 fewer per-contract labels due to compile failures. | RECOVERED_VERIFIED | NO | LOW |
| R4-PREV-DIVE-OTHER | DIVE non-EB/RE classes | — | — | 0 classes reviewed | NO | No per-class review for Arithmetic/DoS/UncheckedReturn/TimeManip/BadRandom/FrontRunning. Proposed gap GAP-002. | UNAVAILABLE | NO | HIGH |
| R4-PREV-DIVE-2NDREVIEW | DIVE all reviewed | — | — | — | NO | Both DIVE review mds are single-author; no second reviewer. | UNAVAILABLE | NO | MEDIUM |
| R4-PREV-BCCC | BCCC all classes | R4-P0-EVD-005 + full deep dive directory | Deep dive (5-phase) | Full 10-class 67,311 contracts | YES (CSVs/parquet/scripts/definitions/batches/class-definitions/decisions) | Reentrancy 89% FP, CallToUnknown 86.9% FP. v1.4 verified labels. 5-phase: exploration through verification. Stage 5.5 deferred (VRAM). | RECOVERED_VERIFIED | YES | HIGH |
| R4-PREV-BCCC-V14 | BCCC all classes | R4-P0-EVD-004 | Verified labels (v1.4) | All 10 BCCC classes, 67,311 rows x 30 cols | YES | BCCC Phase 5 verified labels: 46,977 dropped, 7,403 kept, 18,751 reclassified to NV. CTU/EB/Gas PROVISIONAL. | RECOVERED_VERIFIED | YES | HIGH |
| R4-PREV-BCCC-2TOOL | BCCC 2-tool consensus | data_module/benchmarks/sources/tier_c_bccc_2tool/consensus.py | Benchmark consensus | — | NO (patterns/results empty) | consensus.py skeleton exists but run never executed. GasException+DoS have no Aderyn coverage. Proposed gap GAP-004. | RECOVERED_PARTIAL | NO | MEDIUM |
| R4-PREV-SOLIDIFI | SolidiFI all classes | ml/data/SolidiFI/ (386 files) + SolidiFI-benchmark/ (5,135 files, 1,700 .sol) | Injection framework (ISSTA 2020) | 7 vuln types, 1,700 buggy .sol, 283 pipeline labels | YES (injection framework + tool results) | T0 confidence (100% ground-truth certainty). Only source with mathematically guaranteed labels. No GasException/UnusedReturn. | RECOVERED_VERIFIED | NO | LOW |
| R4-PREV-SMARTBUGS-CUR | SmartBugs Curated all classes | ml/data/smartbugs-curated/dataset/ (143 .sol) + report.json | Hand-labeled corpus + recall test | 143 hand-labeled, 10 categories | YES (vulnerabilities.json + recall report) | 143 hand-labeled DASP contracts (ICSE 2020). 94.4% recall gate PASS. 8 misses caused by crosswalk quality. T1 tier. | RECOVERED_VERIFIED | NO | LOW |
| R4-PREV-SMARTBUGS-WILD | SmartBugs Wild | ml/data/smartbugs-wild/ | Source corpus | 47K contracts | YES (results_wild.json) | 47K mainnet contracts. 97% FP as labeled. Pretraining only. DISABLED in config. | RECOVERED_PARTIAL | NO | LOW |
| R4-PREV-WEB3BUGS | Web3Bugs all classes | — | — | ~3500 contest-verified | NO | Declared Tier-1 Gold but never acquired. No data/crosswalk/parser. Config-vs-reality contradiction. Proposed gap GAP-001. | UNAVAILABLE | NO | HIGH |
| R4-PREV-DEFIHACKLABS | DeFiHackLabs | data_module/data/preprocessed/defihacklabs/ (47 files, ~23 contracts) | Source corpus | 47 preprocessed (715 dropped) | PARTIAL | Foundry project; forge-std prevents solc. DEFERRED. Crosswalk is PLACEHOLDER. | RECOVERED_PARTIAL | NO | MEDIUM |
| R4-PREV-MANUAL | Manual hand-written | manual_hand_written_contracts/ (83 .sol + 83 .json, 11 classes) | Hand-written contract library | 83 .sol + 83 .json across 11 classes | YES | Hand-written contract library with paired labels. Teaching/validation contracts. Not in active export. | RECOVERED_VERIFIED | NO | LOW |
| R4-PREV-MANUAL-BCCC | BCCC-injected manual | manual_hand_written_contracts/bccc_injected/ (8 .sol + 8 .json) | BCCC-style injected contracts | 8 injected contracts | YES | BCCC-style injected contracts for cross-testing. | RECOVERED_VERIFIED | NO | LOW |
| R4-PREV-AI-REPORTS | AI audit reports | agents/test_audit_reports/ (72 files) | AI review output | 22 report JSONs over manual corpus | YES | AI-agent audit runs over manual_hand_written_contracts corpus. 12 single-vuln + 3 tricky + 3 multi-vuln. | RECOVERED_VERIFIED | NO | LOW |
| R4-PREV-BENCHMARK | Benchmark case studies | data_module/benchmarks/benchmark_v0.1_quickstart/tier_a_manifest.json | Benchmark | 74 contracts 5-tier OOD (66 documented, 74 actual) | YES | 74-contract 0%-contamination quickstart benchmark. Count discrepancy: docs say 66, manifest has 74. | RECOVERED_VERIFIED | NO | LOW |
| R4-PREV-ECHIDNA | Echidna tool | — | Tool output | — | NO | No echidna cache or results found anywhere in repo. Proposed gap GAP-005. | UNAVAILABLE | NO | LOW |
| R4-PREV-EXPLOIT-POC | Exploit reproductions | — | Exploit reproduction | — | NO | No Foundry/Hardhat vulnerability PoC test files. Proposed gap GAP-006. | UNAVAILABLE | NO | LOW |
| R4-PREV-DATA-AUDIT | Data module audit | R4-P0-EVD-007 + v2_full_audit/ (7 files + 6 plans) + 8-file audit suite | Audit report | 45% leakage + DoS patch (2,655 zeroed) | YES | Documents 45% leakage finding and DoS/Reentrancy co-occurrence patch. Run 10 checkpoint inflated by leaky splits. | RECOVERED_VERIFIED | NO | LOW |
| R4-PREV-CROSSWALKS | Crosswalks (4 of 5) | R4-P0-XWK-001..003 + defihacklabs.yaml | Pipeline config | dive/solidifi/smartbugs/defihacklabs | YES | 4 crosswalk YAMLs exist; web3bugs.yaml UNAVAILABLE. defihacklabs.yaml is PLACEHOLDER. | RECOVERED_VERIFIED | NO | LOW |

## Status

- `TO_RECOVER`
- `RECOVERED_VERIFIED`
- `RECOVERED_PARTIAL`
- `CONCLUSION_ONLY`
- `UNAVAILABLE`
- `SUPERSEDED`
