# 02B — BCCC Previous Evidence Recovery

- **Run ID:** R4-P1-BCCC-20260716
- **Phase:** 1
- **Date:** 2026-07-16
- **Status:** COMPLETE

## Evidence Sources Recovered

| Evidence set ID | Artifacts | Method | Status |
|---|---|---|---|
| R4-PREV-BCCC | R4-P0-EVD-005 (5-phase overview md) | Deep dive: 5-phase protocol, 67,311 contracts | RECOVERED_VERIFIED |
| R4-PREV-BCCC-V14 | R4-P0-EVD-004 (contracts_clean_v1.4.csv, 67,311 rows x 30 cols) | Phase 5 verified labels with verdicts + confidence | RECOVERED_VERIFIED |
| R4-PREV-BCCC-2TOOL | data_module/benchmarks/sources/tier_c_bccc_2tool/consensus.py | Skeleton implementation, never executed | RECOVERED_PARTIAL |
| BCCC class definitions | p5_s0_class_definitions/ (9 markdown files) | Class definitions for all 9 non-NV classes | RECOVERED_VERIFIED |
| BCCC per-class decisions | D-I-11 (drop NV with vuln), D-I-12 (drop NV with IntegerUO) | Registered decisions in decisions/ directories | RECOVERED_VERIFIED |
| BCCC batch files/review | Phase 3 WS-I (808 contracts, slither), Phase 4 (10,693 contracts, slither+aderyn) | Tool-assisted label validation | RECOVERED_PARTIAL |

## Historical Claims Recovered

### Class-by-class FP/P rates from Phase 4 Stage 1 (500-contract Reentrancy audit)

| Class | FP Rate | TP Rate | n | Source |
|---|---|---|---|---|
| Reentrancy | 89.4% | 10.6% | 500/17,698 | Phase 4 Stage 1 manual audit |
| CallToUnknown | 91% (FN rate) | 9% | 11 (FN-only sample) | Phase 4 Stage 1 expanded review |
| ExternalBug | 100% (tiny) | 0% | 1 | Phase 4 Stage 1 |
| GasException | 67% | 0% KEEP | 9 | Phase 4 Stage 1 |
| DenialOfService | 56% | 0% KEEP | 18 | Phase 4 Stage 1 |
| Timestamp | 50% FN / 33% TP | ~17% | Referenced | Phase 4 Stage 1 |
| IntegerUO | 0% | 100% | Confirmed manual+2tools | Phase 4 Stage 1 |
| UnusedReturn | 0% | 100% | Confirmed manual+2tools | Phase 4 Stage 1 |
| MishandledException | 0% | 100% | Confirmed manual+2tools | Phase 4 Stage 1 |

### Phase 5 verification results (ALL 67,311 contracts, automated + manual rules)

| Class | Before P5 | After P5 | % Kept | Gate |
|---|---|---|---|---|
| Reentrancy | 17,698 | 1,699 | 9.6% | VERIFIED |
| CallToUnknown | 11,131 | 239 | 2.1% | PROVISIONAL (Stage 5.5 deferred) |
| Timestamp | 2,674 | 1,075 | 40.2% | BEST-EFFORT |
| ExternalBug | 3,604 | 344 | 9.5% | PROVISIONAL |
| GasException | 6,879 | 2,794 | 40.6% | PROVISIONAL |
| DenialOfService | 12,394 | 1,252 | 10.1% | BEST-EFFORT |
| MishandledException | 5,154 | 5,154 | 100% | VERIFIED |
| UnusedReturn | 3,229 | 3,229 | 100% | VERIFIED |
| IntegerUO | 16,740 | 16,740 | 100% | VERIFIED |
| NonVulnerable | 26,148 | 44,899 | 171.7% | — |

### Key structured finding: Reentrancy FP root cause
- `.call.value()` = true positive (10.6%)
- `.transfer()` only = false positive (41.0%)
- `.send()` only = BORDERLINE (14.2%)
- No external call = false positive (34.2%)

## Contradictions and Ambiguities

1. **BCCC status in active config vs evidence:** config.yaml declares BCCC as `DEFERRED` in sources, yet the active export uses `disl` (nonvulnerable pool) which is BCCC-derived. The BCCC v1.4 verified labels (67,311 contracts x 10 classes) exist but are NOT loaded by the current export pipeline.

2. **Tool coverage gaps:** GasException and DenialOfService have NO Aderyn detector coverage. 5 of 9 classes have no reliable static analysis coverage for at least one tool. This means the 2-tool consensus approach (Slither+Aderyn intersection) cannot produce coverage for these classes.

3. **Stage 5.5 ML propagation not executed:** GraphCodeBERT embedding + HDBSCAN clustering was deferred (VRAM conflict with active SENTINEL training on RTX 3070 8GB). This means the CallToUnknown, ExternalBug, GasException phases remain PROVISIONAL rather than VERIFIED.

4. **Phase 5 halving vs original BCCC:** After Phase 5 verification, 46,977 labels dropped, 7,403 kept, 18,751 reclassified to NonVulnerable. The entire BCCC dataset shrank from 67,311 raw to essentially the `contracts_clean_v1.4.csv` with verified+provisional entries.

## Duplicate/Overlap Detection

| Evidence set pair | Overlap detected | Notes |
|---|---|---|
| R4-PREV-BCCC vs R4-PREV-BCCC-V14 | FULL (v1.4 is the OUTPUT of BCCC Phase 5) | Not independent; v1.4 is the verified product |
| R4-PREV-BCCC-2TOOL (consensus.py) vs BCCC Phase 4 | DIFFERENT | 2-tool consensus was a SEPARATE planned run (never executed); Phase 4 used Slither+Aderyn on sampled subset |
| BCCC manual review vs DIVE manual review | NONE (different corpuses) | BCCC 67,311 contracts vs DIVE 22,330 contracts — different sourcing |

## Dependency Groups

- **Group B-A:** R4-PREV-BCCC (5-phase overview) → R4-PREV-BCCC-V14 (v1.4 verified labels) → all class definitions and decisions. These are the same chain, not independent.
- **Group B-B:** BCCC Phase 3 WS-I (808 slither), Phase 4 (10,693 slither+aderyn) → both are tool-assisted sampling from the same BCCC population. Phase 4 is a scaled-up version of Phase 3 methodology.
- **Group B-C:** BCCC class definitions (9 .md files), decisions (D-I-11, D-I-12), and verification scripts → design and process documents. Independent from measurement evidence.

## Unavailable Artifacts

| Item | Impact |
|---|---|
| BCCC 2-tool consensus RUN (patterns/results empty) | Consensus approach defined but never executed; no 2-tool baseline for BCCC |
| BCCC 2-tool audit memory file (outside repo) | Lives in Claude memory; not reproducible from repo alone |
| Stage 5.5 GraphCodeBERT propagation run | Deferred; 3 classes remain PROVISIONAL instead of VERIFIED |
| BCCC original acquisition source | 67,311 contracts acquired from BCCC repository; acquisition commit/timestamp not recorded |

## Recommended Phase 2 Actions

- Consider whether BCCC v1.4 verified labels should become the BCCC training source (replacing raw BCCC) — this would reduce Reentrancy from 17,698 to 1,699 (-90.4%)
- Register needed: 2-tool consensus execution (gap for GasException, DenialOfService without Aderyn coverage)
- Note: BCCC v1.4 is DEFERRED in config; no pipeline changes needed yet
