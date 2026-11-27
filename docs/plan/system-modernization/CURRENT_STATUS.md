# S4 current execution status

Date: 2026-11-26

Program: ACTIVE; governing documents established; S4-00 complete with limitations; S4-G00 PASS_WITH_LIMITATIONS.

Reconnaissance source baseline: 61854a32a2badbcd84b98cef3f3976f070b1341e. The program-establishment commit is 2c4dd3cee0a1313d1f23d87fdb17e682340cacd6; later evidence must record its own exact source SHA.

## Governing navigation

- [Master program](00_MASTER_PROGRAM.md)
- [Audit protocol](01_AUDIT_PROTOCOL.md)
- [Technical completion](../system-finalization/technical-completion/00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md)
- [R4 status](../ml-R4/PLAN_STATUS_MATRIX.md)

## Stage state

| Stage | State |
|---|---|
| S4-00 Baseline | COMPLETE / G00 PASS_WITH_LIMITATIONS |
| S4-01 Architecture/usefulness | IN_PROGRESS (001/002/003A complete; 003B ready) |
| S4-02 Dependencies/CI | READY (parallel eligible) |
| S4-03 DATA | L0A/L0B/L1A/L1B COMPLETE; L-G0 PASS_WITH_LIMITATIONS; L1C INDEPENDENT REVIEW PENDING |
| S4-04 ML/inference | NOT_STARTED |
| S4-05 AGENTS/LLM/RAG | NOT_STARTED |
| S4-06 ZKML/contracts | NOT_STARTED |
| S4-07 Security/reliability | NOT_STARTED |
| S4-08 Maintainability/governance | NOT_STARTED |
| S4-09 Experiments/decisions | NOT_STARTED |
| S4-10 Implementation | NOT_STARTED |
| S4-11 Integrated acceptance | NOT_STARTED |

G00 passed with limitations. F-0004 CLOSED on exact remote CI; F-0011/F-0012 locally validated, CI coverage pending. F-0013 locally validated under S4-D-0002; dedicated AGENTS CI pending. S4-D-0001 product direction accepted; no model/toolchain promotion. Existing historical domain gates retain their own authority.

## Completed slice: S4-00-001

Inspected source: 2c4dd3cee0a1313d1f23d87fdb17e682340cacd6 (local and remote main matched). Evidence is published in the commit containing this updated record.

- [Baseline inventory](02_BASELINE_INVENTORY.md) and [findings register](03_FINDINGS_REGISTER.md); five evidence records E-0001 through E-0005.
- Five core roots: 292 source files, 64,738 physical lines; 189 test-source files. Inventory, not exhaustive audit.
- Root/AGENTS Poetry checks failed; DATA/ML passed lock checks. DATA 69 and AGENTS 10 common installed/locked version mismatches; ML zero common differences, not full environment equivalence.
- D-011/D-013 roots and selected runtime/proof artifacts present; physical integrity/live operation not reverified.
- Six of seven local dependency-light checks passed; exact historical G4 replay failed at line 75. GitHub three triggered workflows at the inspected SHA passed; G4 not triggered there.
- D2 86 raw rows, 84 unique queued UNVERIFIED, two duplicate tombstones retained. No historical closures asserted.

## Completed slice: S4-00-002

Source: 5d8dd776ad647c0c32778653342f84ab5b5d27df. [Evidence and gate decision](evidence/2026-11-26_s4_00_002/WORKING_RECORD.md). G4 typo traced to b7deb49f6; one-character correction, exact local and remote gate pass (run 45836727069); all seven prior workflows green. Five bounded module test slices pass 153 tests; AGENTS initial harness collection failure retained. Full test suites, clean installs, live tools, physical rehash and proof generation remain NOT_RUN/unestablished. Local Foundry library lock conformance unverified. G00 PASS_WITH_LIMITATIONS permits S4-01 and S4-02 research, not product or domain promotion.

## Completed slice: S4-01-001

Inspected 9eaaae171c09650f6cca6eae3f4b56c7965cf050. [Architecture and scope](module-audits/S4-01_ARCHITECTURE_AND_SCOPE.md), [evidence](evidence/2026-11-26_s4_01_001/WORKING_RECORD.md). S4-D-0001: evidence-first off-chain investigation as primary product, DATA/ML and optional trust tracks separately governed. User delegated design/direction and engineering authority; no bypass of evidence/safety/launch gates. Reproduced and repaired F-0011 per-job LLM disable and F-0012 screen eligibility/actual routing trace. Final selected runs 28 + 46 + 47 passed (overlap); initial failures preserved. F-0013 no-evidence SAFE exposed and confirmed; F-0014 ZK claim mismatch confirmed; F-0015 durable recovery candidate. G01 remains IN_PROGRESS; no empirical product usefulness established.

## Completed slice: S4-01-002

Inspected 154896459942c772b267e2de4e11670a1d812819. [Evidence and decision](evidence/2026-11-26_s4_01_002/WORKING_RECORD.md). S4-D-0002 evidence-adequacy-v1 projects report verdicts separately from raw fusion, abstains for insufficient evidence, preserves target signals, prevents RAG-only promotion, explicitly scopes no findings; no safety assurance. Legacy evaluation rejects new report policy until coverage-aware metrics exist. 135 selected test executions pass; prior commit three workflows green. F-0013 VALIDATING; narrow F-0014 wording corrected, broader sweep pending. No physical artifact, scientific label or training change.

[S4-03-L Solidity source and label recovery](module-audits/S4-03_LABEL_RECOVERY_PROGRAM.md) approved for planning/bounded evidence research. L0 independently ready: recover exact current class/source/outcome/group/role counts, source provenance and review queue gaps. Committed logical V3 seed: 225,400 cells, 932 effective loss, 143 outcome metric, zero confirmed negatives. Do not infer physical revalidation or admit labels from counts. R4 authority and independent GAP-007 remain intact.

## Completed slices: S4-01-003A and S4-03-L0A

Inspected ee9a2e0202bb0a729c60a05906b71bd3195e67b6. [Evidence](evidence/2026-11-26_s4_01_003/WORKING_RECORD.md). Actual V3 label and target rows read-only reconciled (seven hashes, 225,400 cells, no checked projection mismatches or cross-role groups). 474 confirmed positive and 606 weak target cells; zero negatives. GasException/UnusedReturn no positive targets, DoS six; TOD 633 of 932 effective loss. L0A complete, L-G0 remains open pending native source provenance. Actual Slither timestamp detector on three synthetic inputs: positive observation, completed zero finding, failed compilation. All 15 preregistered checks pass; current report WATCH/INCONCLUSIVE/INCONCLUSIVE, unrelated synthetic context not promoted. Raw legacy fusion comparator CONFIRMED on all with synthetic context. No actual ML/RAG/LLM/full graph comparison, no security label truth, no empirical user utility. Prior exact SHA three workflows green.

## Completed slice: S4-03-L0B

Inspected ca7e47c6ce7a1aeaf49f9ee5689e0862bf0dbf06. [Native provenance evidence](evidence/2026-11-26_s4_03_l0b/WORKING_RECORD.md). 22,823 raw bytes/size manifest checks, 22,540 normalized source identities, 22,801 source records and all 58,056 native claims reconcile. DIVE 22 omitted paths match historical compile-failure count; per-path linkage not revalidated. Original CSV, SmartBugs 207 annotations and SolidiFI 350 injection logs available; exact local pins/license notices captured. DIVE portable origin/license unresolved; per-contract rights, annotation semantics and fresh compilation unestablished. L-G0 PASS_WITH_LIMITATIONS for local recovery research, not scientific acceptance. User's DATA quality priority advanced; no new labels or protected data changes. Prior exact SHA CI 2/2 green.

## Completed slice: S4-03-L1A

Inspected 604d882dac7958baed50c3cafb957b5d060ae28e. [Evidence](evidence/2026-11-26_s4_03_l1a/WORKING_RECORD.md). Frozen 16 distinct DIVE groups (all TRAIN_UNLABELED, excluded R4 reserved groups), 8 annotation calibration files. All 57 spans in bounds; partial exact coordinate anchoring. 16/16 live Slither runs complete with six detectors; DoS native 3/4 and UnusedReturn native 1/4 relevant signals, no mapped claim strata 0/4 each. Three DoS spot-checks insufficient for class confirmation; two source-supported UnusedReturn positive candidates, one cross-class post hoc missing native claim. No labels admitted. L1A complete for triage, L-G1 remains open; reservation recorded not enforced in successor. Previous CI 2/2 green.

## Completed slice: S4-03-L1B

Inspected 820bc1457f3260c54c946c704f7695240b19f1d4. [Evidence](evidence/2026-11-27_s4_03_l1b/WORKING_RECORD.md). Complementary direct-return helper fixture pass after retained compilation/API failures, 16/16 parsed, 730 function/modifier inventory entries, 10 direct sites/four contracts. 11 primary reviewed cells (five UnusedReturn positive candidates including two missing native claim, six insufficient); no admission, no complete semantic scope claim, risk unestablished. 11/11 neutral source-only local packets validated with tamper and extra-member rejection. Independent review pending and no DIVE redistribution. Prior exact CI 2/2 success. L1B complete for primary preparation, L-G1 open.

## Completed slice: S4-03-L3A

Inspected 40ee9feb93c54ae303fe5b0f0a64106e503aa7b4. [External source acquisition plan](module-audits/S4-03_L3_EXTERNAL_SOURCE_ACQUISITION.md), [research catalog](evidence/2026-11-27_s4_03_l3a/source_catalog.json). 13 discovery entries compared to current registry and legacy config; new official DIVE locator/dataset license identified, exact archive equivalence unresolved. Gas Gauge, CVE Smart Contracts, FORGE Curated prioritized; Bastet/ScaBench and new EVMbench for project evaluation, Sourcify/DISL for targeted source research only. No download/ingestion, new label or training authorization. Prior CI 2/2 success.

## Completed slice: S4-03-L3B0/B1 preflight

Inspected 099b9d66c8a7edaae5323dcbb09cbe6b48b76a4d. [Evidence](evidence/2026-11-27_s4_03_l3b/WORKING_RECORD.md). DIVE official label ZIP verified, local CSV exact match all 22,330 records; additional six-tool results (26,071 rows) profiled, semantics pending. Gas Gauge 2,000 paths/1,683 distinct blobs; frozen 10 paths, 9 source paths and 10 reports verified, four author aggregate logs verified. Source rights and exact run-source identity unresolved, no staging/compile or class truth. Prior exact CI 2/2 green. Both preflight slices PASS_WITH_LIMITATIONS; no new labels or training.

## Completed slice: S4-03-L3B0A

Inspected 38d15e1b9ce281193eb064f12278cf22b196571c. [Evidence](evidence/2026-11-27_s4_03_l3b0a/WORKING_RECORD.md). DIVE 3,741 repeated IDs with complementary observations and no conflicting nonblank values. First-row published-rule replay contains all published positives, four categories exact; remaining extra positives consistent with heuristic removal, exact validator not rerun. Nine published Time positives lack positive tool observation in either row; 880 native Unchecked Return Values zeros have positive observation. No ground truth inference. All 22,330 original source names/size/CRC32 match official raw ZIP directory fetched by bounded ranges; full archive checksum not verified. PASS_WITH_LIMITATIONS; prior CI 2/2 green.

## Completed slice: S4-03-L3B0B

Inspected d267df59bf92a744a13fc98a094dedb76d29fd7e. [Evidence](evidence/2026-11-27_s4_03_l3b0b/WORKING_RECORD.md). Implemented strict evidence-only DIVE observations module and local Parquet ledger. 26,071 rows, 1,251,408 cells (696,540 missing), 178,640 native-category rows, 22,308 original path links/22,054 normalized artifacts, 22 unmatched IDs retained. All 21 frozen review quotas filled, unique groups, combined reservations 237; overlap guard tested but not integrated in successor. 34 focused tests plus governance passed; prior CI 2/2 green. No label/role changes.

## Completed slice: S4-03-L3B2

Inspected f5c0dcd415ed376c71b28b3c68e9afffd2a7e29f. [Evidence](evidence/2026-11-27_s4_03_l3b2/WORKING_RECORD.md). CVE 2026-08-14 pinned catalog/index/labels/localizations reconciled; 491 catalog records, 437 arithmetic, rare-class coverage limited. Five frozen cases, 19/20 artifacts verified, 4/4 inspected explorer location ranges in bounds; one DIVE raw source overlap, distributed source byte identity differs. Source rights/run linkage and grouping unresolved; no staging, execution, admission. PASS_WITH_LIMITATIONS; prior CI 5/5 green.

## Next bounded slices

L3B3 FORGE Curated preflight next; L3B0C DIVE candidate primary source reviews independently ready; L3B1 Gas Gauge and L3B2 CVE source/run/rights follow-up; L1C independent reviews pending. S4-01-003B and S4-02 independently eligible. Audit legacy integration before enabling source connectors. L-G1/G01 open; no training/promotion.

## Known initial blockers and independent work

- G4 reference repaired, local and remote exact gate passed. F-0004 closed; prior baseline evidence retained.
- Root Poetry config and AGENTS manifest/lock mismatch observed; actual dependency environment drift. No broad upgrades authorized.
- Nine pre-existing untracked WSL paths are preserved; their disposition remains unreviewed.
- R4-GAP-007 isolated independent review remains pending; confirmed negatives zero, M5/full training and G8 not authorized. That separate evidence path can proceed without S4 context.
- ZKML proof scope, read-only audit MCP, production signing/deployment and accepted DATA/ML roots unchanged.

Next actions: L3B3 FORGE preflight, L3B0C candidate review, L3B1/B2 follow-up; L1C, S4-01-003B and S4-02 independently eligible. No training without admissible evidence.
