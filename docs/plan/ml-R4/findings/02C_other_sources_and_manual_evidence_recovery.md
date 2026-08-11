# 02C — Other Sources and Manual Evidence Recovery

- **Run ID:** R4-P1-OTHER-20260716
- **Phase:** 1
- **Date:** 2026-07-16
- **Status:** COMPLETE

## SolidiFI

| Field | Value |
|---|---|
| Source path | ml/data/SolidiFI/ (386 files, tool source) + ml/data/SolidiFI-benchmark/ (5,135 files, 1,700 .sol) |
| Unit of analysis | Injection tool results: bugs injected into clean contracts |
| Positive mechanism | Guaranteed ground truth (T0): each contract verified as having exactly the injected vulnerability |
| Negative semantics | No negative controls in injection set (every contract has the injected bug) |
| Class coverage | 7 types: Overflow-Underflow, Re-entrancy, TOD, Timestamp-Dependency, Unchecked-Send, Unhandled-Exceptions, tx.origin |
| Pipeline labels | 283 per-contract generated labels in data_module/data/labels/solidifi/ |
| Compiler coverage | Multiple versions via injection tool framework |
| Crosswalk | solidifi.yaml: 1-to-1 mapping for 6 classes; tx.origin → ExternalBug |
| Status | ACTIVE in current export (T0 tier, 283 contracts) |

**Key finding:** SolidiFI is the ONLY source with mathematically guaranteed ground truth (T0 confidence tier). However, with only 283 contracts (after 7 failed preprocessing), it provides very sparse coverage. No GasException or UnusedReturn injection exists.

## SmartBugs Curated

| Field | Value |
|---|---|
| Source path | ml/data/smartbugs-curated/dataset/ (143 .sol files, 10 categories) |
| Unit of analysis | Hand-labeled contracts from ICSE 2020 empirical study |
| Positive mechanism | Hand-labeled by domain experts |
| Negative/absence | NonVulnerable category (4 contracts) |
| Class coverage | 10 categories mapped to 10 SENTINEL classes (some lossy) |
| Crosswalk | smartbugs_curated.yaml: DASP-direct mapping; front_running→TOD (fixed from Timestamp) |
| Recall gate | 94.4% aggregate recall verified (report.json: 8 missed due to crosswalk quality, not extraction) |
| Tier | T1 (structural benchmark, hand-labeled) |
| Status | ENABLED in current export |

**Recall degradation per class:**
- Reentrancy: 1.0 (31/31)
- CallToUnknown: 1.0 (52/52)
- NonVulnerable: 1.0 (4/4)
- IntegerUO: 1.0 (15/15)
- DenialOfService: 0.833 (5/6, miss: auction.sol)
- ExternalBug: 0.833 (15/18, 3 access_control/ misses)
- Timestamp: 0.765 (13/17, 4 front_running/ misses via lossy crosswalk)

## SmartBugs Wild

| Field | Value |
|---|---|
| Source path | ml/data/smartbugs-wild/ (47K+ contracts) |
| Status | AVAILABLE but NOT in active export (DISABLED in config) |
| Role | Pretraining corpus only (results_wild.json reports 97% FP as labeled) |
| Note | Not suitable for supervised training due to extreme label noise |

## Web3Bugs

**Status: UNAVAILABLE.** Declared Tier-1 Gold in config.yaml but no data, crosswalk, parser, or connector exists. ~3,500 contest-verified bugs with proof-of-exploit per the original description, but material was never acquired.

## Manual Hand-Written Contracts

| Field | Value |
|---|---|
| Path | manual_hand_written_contracts/ |
| Corpus | 83 .sol files + 83 .json label files, 11 vulnerability classes + Safe |
| BCCC injected | 8 .sol + 8 .json in bccc_injected/ (BCCC-style injections for cross-testing) |
| Training role | These are hand-written teaching/validation contracts, NOT in the active export |
| Crosswalk relevance | Manual .json labels use SENTINEL 10-class taxonomy directly |

## AI Audit Reports

| Field | Value |
|---|---|
| Path | agents/test_audit_reports/ (72 files) |
| Scope | 12 single-vuln, 3 tricky/edge-case, 3 multi-vuln AI-generated reports |
| Training use | AI agents auditing the manual_hand_written_contracts corpus |
| Format | Structured JSON audit reports |

## Benchmark (Tier A Quickstart)

| Field | Value |
|---|---|
| Path | data_module/benchmarks/benchmark_v0.1_quickstart/ |
| Corpus | 74 contracts (8 smartbugs_curated + 66 solidifi_benchmark) |
| Classes | NonVulnerable (11), Reentrancy (11), CallToUnknown (22), TOD (10), Timestamp (10), MishandledException (10) |
| Purpose | 0%-contamination OOD evaluation benchmark |
| Tier | A (quickstart, no overlap with training export) |

## DeFiHackLabs

| Field | Value |
|---|---|
| Path | data_module/data/preprocessed/defihacklabs/ (47 files, ~23 contracts) |
| Status | DISABLED in config (Foundry compile issue: forge-std prevents solc) |
| Crosswalk | defihacklabs.yaml — PLACEHOLDER (major categories mapped; full 400+ entries deferred) |
| Used in export | NO (DISABLED) |

## Crosswalks (all 5)

| Source | Path | Status | Classes |
|---|---|---|---|
| dive | dive.yaml (92 lines) | ACTIVE | 8 DASP -> 10 SENTINEL; BadRandomness DROPPED |
| solidifi | solidifi.yaml (76 lines) | ACTIVE | 7 injection types -> 10 SENTINEL; 1-to-1 |
| smartbugs_curated | smartbugs_curated.yaml (32 lines) | ACTIVE | 10 DASP folders -> 10 SENTINEL |
| web3bugs | — | UNAVAILABLE | Declared enabled:true in config but file does not exist |
| defihacklabs | defihacklabs.yaml (37 lines) | EXISTS, source DISABLED | PLACEHOLDER, 400+ entries deferred |

## Data Module Audit (v2, 45% leakage finding)

| Field | Value |
|---|---|
| Leakage rate | 45% val/test graph-level leakage in v1 splits |
| Root cause | `_run_split` never set `dedup_group` on Contract objects |
| Patch applied | Graph-hash dedup: 21,523 -> 12,577 unique groups; 0% leakage post-patch |
| DoS patch | 2,655 DoS labels zeroed (BCCC-era noise propagated into labels.parquet) |
| Run 10 inflation | GCB-P1-Run10-v2clean checkpoint has INFLATED F1=0.683 due to v1 leaky splits |
| GasException | 0 positives in current export (SmartBugs Curated skipped for this class) |
| Residual issues | 677 dup groups with conflicting labels (~4,700 contracts) accepted as irreducible DIVE noise |

## Contradictions

1. **Web3Bugs declared enabled but entirely absent:** Config says `web3bugs: enabled: true` but no data/crosswalk/parser/connector exists. Crosswalk web3bugs.yaml does not exist. This is the most serious config-vs-reality contradiction in the active pipeline.

2. **SmartBugs Curated recall vs crosswalk losses:** The 94.4% recall gate passes, but 8 missed contracts are all crosswalk quality issues (front_running→Timestamp lossy mapping; access_control→ExternalBug ambiguous mapping), not extraction bugs.

3. **DeFiHackLabs DEFERRED but crosswalk exists:** Source disabled due to Foundry compile issue, but crosswalk was already created. If resolved, crosswalk needs full 400+ entry mapping.

## Unavailable Artifacts

| Item | Impact |
|---|---|
| Web3Bugs entire source (data/crosswalk/parser/connector) | Declared Tier-1 Gold; config says enabled; cannot support any class claim |
| DeFiHackLabs full source (715 of 738 dropped) | 47/762 preprocessed; Foundry compile blocks pipeline |
| Echidna tool outputs | No fuzzing evidence for any class |
| Exploit reproduction PoCs | No functional exploit tests for any vulnerability class |
