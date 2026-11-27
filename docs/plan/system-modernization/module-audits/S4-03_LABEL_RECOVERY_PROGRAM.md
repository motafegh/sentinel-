# S4-03-L — Solidity source and label recovery program

**Status:** APPROVED FOR PLANNING AND BOUNDED EVIDENCE RESEARCH; no new label admitted, no optimizer authorization.
**Parent:** [S4 master](../00_MASTER_PROGRAM.md), S4-03 DATA and S4-04 ML.
**Authority:** [R4 status](../../ml-R4/PLAN_STATUS_MATRIX.md), [label policy](../../ml-R4/LABEL_STATE_AND_DATASET_ROLE_POLICY.md), [Phase 2 reconstruction](../../ml-R4/findings/03_label_corruption_reconstruction_summary.md), [negative evaluation policy](../../ml-R4/specs/confirmed_negative_evaluation_v1.json). R4 decisions control semantic/role changes and training. This program does not replace them.

## 1. Problem and objective

R4 established that historical zeros collapsed missing, unsupported, unmapped, unknown and explicit negative source states. The historical reconstruction identified DIVE folderization loss, unsupported source classes, inappropriate NonVulnerable mapping, 2,655 DoS positive-to-zero cell changes, and missing representation/export issues. These are historical reconstructed mechanisms, not counts of currently wrong labels. Accepted logical V3 and D-011/D-013 physical roots improve lineage; they do not manufacture class truth. Confirmed negatives remain zero; D-015 bounded positive-fit result did not establish discrimination. D-016 governs current CallToUnknown; GAP-007 independent review remains separate.

### Existing accepted logical V3 support (starting evidence)

The committed [logical V3 summary](../../ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_summary.json) reports 22,540 contracts, 22,394 leakage groups and 225,400 contract-class cells across ten classes. There are 932 effective loss cells and 143 outcome-metric cells, with zero confirmed negatives. Roles: TRAIN_STRONG 343 contracts; TRAIN_WEAK 602; TRAIN_UNLABELED 21,449; MODEL_SELECTION 73; INTERNAL_AUDIT 73. These are exact reported snapshot counts, not newly verified physical contents or class support. [Seed record](../evidence/2026-11-26_s4_01_002/label_baseline_seed.json) binds the summary bytes to the inspected SHA. L0 must recover detailed per-class/source/group denominators and outcome-state counts before choosing acquisition targets. The very sparse eligible supervision, not simply the contract count, is the key planning constraint.

Objective: improve evidence-qualified label correctness, coverage, class balance and reproducibility per contract identity × vulnerability class, and measure what additional supervision/evaluation becomes feasible. More rows or model confidence alone do not meet this objective.

## 2. Parallel work packages

| ID | Work package | Method and exit evidence |
|---|---|---|
| L0 | Reconstruct current source/label inventory | Trace raw source claim -> crosswalk -> review outcome -> role/mask -> exact physical identity. Count per class/source/compiler/group and outcome, distinguish historical versus accepted current. Record source license/version/hashes, missing records, contradictions and acquisition constraints. |
| L1 | Recover original claims | Recover native CSV/JSON/annotations, exploit/injection and patch metadata from verified source versions. Compare SmartBugs, DIVE, SolidiFI and other actually acquired sources. Preserve unknown/missing/unsupported distinctions; no default zeros. Verify compiler/source scope. |
| L2 | Correct existing label errors | Stratified review of suspected wrong positives, wrong historical zeros, out-of-taxonomy mappings, contradictory claims, duplicate/pair cases. Record source locations, class criteria, reasoning and disagreement. Separate label error from representation failure and taxonomy ambiguity. Publish append-only adjudication candidates. |
| L3 | Acquire new Solidity evidence | Research current licensed datasets, verified source snapshots, public audit reports, known exploit/fix pairs, curated challenge suites and reproducible compiler/tool fixtures. Check availability, licensing, commit/address/chain/source identity and complete source/build context before ingestion. No automatic trust of published labels, no blind scraping or credential use. |
| L4 | AI-assisted candidate review | Compare local code-capable models, retrieval of exact class definitions, static/dynamic analysis, constrained evidence extraction and targeted adjudication. Use full relevant code scope or explicit abstention, source ranges, compiler/import context, tool result IDs, alternative explanations. LLM output is a proposal, not gold truth; same model/agents sharing context do not establish independent verification. |
| L5 | Class-specific negative evidence | Continue existing approved R4-GAP-007 queue and independent process; candidate #2 is not adjudicated by this session. Use class-relevant manual/static arguments, trusted explicit negative sources, applicable property tests as corroboration. Preserve not-applicable vs negative vs unknown; no global safe labels. Reserve leakage groups before review; evaluation-only unless later authority. |
| L6 | Quality and review operations | Design reviewer schema, versioned rubric, conflict handling, blind packets, agreement and review cost, deterministic sampling, random control plus active-learning triage. Assess tool disagreement and model uncertainty as queue priority only. Include false positives/false negatives and missing-class strata. |
| L7 | Candidate publication | Immutable successor semantic release with provenance, reproducible builder, exact accepted adjudication IDs, source/group/role/mask validation, leakage and duplicate checks, rejection logs, delta report and rollback. Never overwrite D-011/D-013 or prior label history. Gate new representation only if meaning/identity changes require it. |
| L8 | ML handoff | Quantify role-eligible per-class positives/negatives, coverage/uncertainty, independent group counts, supported metrics and explicit remaining gaps. Compare alternatives (positive-unlabeled, weak supervision, self-supervision, transfer learning) only in separately approved experiments. No new optimizer or calibration/threshold authority without R4 decision. |

## 3. Evidence acquisition and adjudication design

Use an evidence ledger, not an LLM-generated label spreadsheet. The record should bind exact source hash and contract/component scope, source origin/license/version, vulnerability class and governing definition version, native label and crosswalk, candidate evidence (line ranges, functions, claim types, compilation/tool status, reproduction), reviewer/independence and conflict, current outcome and reason, role eligibility, leakage group, adjudication identity and publication version. Existing R4 schema/validators remain authoritative; design successor schema only if an explicit ADR is necessary.

Evidence hierarchy is conditional on class: a reproducible exploit/injection or property counterexample can establish a scoped positive when the class matches; a fix pair does not establish unrelated negatives. Public audit prose must match exact code/version and class. Static analyzers and LLMs may suggest candidates, not silently promote labels. A tool run with no findings, missing source label, LLM agreement or low ML score is not negative evidence. Independent primary/review evidence and conflict policy apply under R4; when insufficient, retain UNKNOWN/NOT_REVIEWED/CONFLICTING_EVIDENCE/NOT_APPLICABLE as appropriate. Do not collapse to 0.

Human review bottleneck: prioritize high-information cells, combine stratified random controls with disagreement/uncertainty sampling, and estimate actual cost/yield before scaling. Blind review packets must include neutral current definitions and required scope, not prior verdicts. Separate reviewers must actually be independent; record contamination. Never reuse this session for blind candidate #2 review.

## 4. Measurements and gates

Start with L0/L1 read-only inventory; no arbitrary acquisition count. Report denominator and unit for each measure:

- unique source identities, group count, eligible compiled source coverage and provenance completeness;
- class-by-source outcome counts and mask eligibility; label conflict and unknown rates;
- verified correction counts (old state -> new candidate state), acceptance yield, reviewer disagreement, time/cost per accepted cell, abstention;
- independent positive/negative group support per class and role, leakage violations, class/source/compiler skew;
- representation compatibility and source-location coverage; versioned changes and reproducibility.

Freeze a bounded pilot design before labeling: select classes based on inventory, include randomized controls and under-supported strata, explicit budgets and failure/stop criteria. Existing negative evaluation policy uses a 25-candidate-per-enabled-class pilot to estimate yield; the 59 independent confirmed negatives per class planning number assumes zero observed false positives and 95% confidence for 5% upper bound. It is not a universal training target and does not establish threshold/calibration eligibility. Separate development, selection and untouched evaluation groups; measure sampling bias and avoid optimizing on review/acceptance cases.

Gate L-G0: source/label inventory reconciled with existing R4 evidence. L-G1: candidate sources and review method validated on bounded pilot; measured yield and limitations. L-G2: adjudicated candidate release validates provenance, policy, independence and leakage; acceptance only through owning R4 authority. L-G3: explicit ML handoff decision stating eligible roles and supported metrics. If no admissible negative evidence can be acquired, retain hold and evaluate alternative research paths; do not lower standards.

## 5. Execution order and relationship to S4

This is an early, independent DATA priority: S4-03-L0 can start alongside S4-01/S4-02 once its exact baseline is captured; do not wait for all runtime architecture work. S4-01-003 analyst utility evaluation must remain separate from R4 training/acceptance roles; no leakage into future optimizer populations. L3/L4 current external dataset/tool/model research should occur when L0 reveals gaps, using primary sources and actual license/version verification. No external models or tools are selected yet. Prior accepted data and review records remain intact. Record every candidate experiment and decision under S4/R4, preserve unsuccessful review attempts, publish only admissible evidence. User explicitly requested this priority on 2026-11-26.

## 6. L0A measured inventory (2026-11-26)

[Exact inventory evidence](../evidence/2026-11-26_s4_01_003/label_inventory.json) at ee9a2e0202bb0a729c60a05906b71bd3195e67b6. Seven logical artifact digests agree with the committed manifest, local manifest bytes match committed manifest, 225,400 long cells match 22,540 wide target rows across checked fields, no recorded group spans incompatible roles (22,394 groups). This is read-only byte/structural reconciliation, not class adjudication, raw source validation or a full representation rehash.

| Class | Positive targets | Effective loss cells | Outcome-metric cells | Confirmed negatives |
|---|---:|---:|---:|---:|
| CallToUnknown | 101 | 76 | 25 | 0 |
| DenialOfService | 6 | 3 | 3 | 0 |
| ExternalBug | 67 | 43 | 24 | 0 |
| GasException | 0 | 0 | 0 | 0 |
| IntegerUO | 64 | 41 | 23 | 0 |
| MishandledException | 49 | 40 | 9 | 0 |
| Reentrancy | 80 | 59 | 21 | 0 |
| Timestamp | 54 | 37 | 17 | 0 |
| TransactionOrderDependence | 659 | 633 | 21 | 0 |
| UnusedReturn | 0 | 0 | 0 | 0 |
| Total | 1,080 | 932 | 143 | 0 |

171,083 UNKNOWN; 53,843 NOT_REVIEWED (includes 606 weak TOD targets); 474 CONFIRMED_POSITIVE. No target-zero cells, 224,320 null targets. Primary sources: DIVE 22,054 contracts (606 weak positive targets, 601 effective loss, no metric cells), SolidiFI 343 (343 strong targets, 240 effective loss, 103 metric), SmartBugs curated 143 (131 strong targets, 91 effective loss, 40 metric). All 225,400 historical_state fields in this V3 projection are HISTORICAL_MISSING; native claims are preserved separately, not evidence of all historical source labels missing. L0B must trace this projection and reconstruct native label semantics from source claim/ledger artifacts and raw inputs. Claims are counted as occurrences across class cells, not unique evidence; SHA fields present does not establish content/licensing.

Priority: first investigate GasException and UnusedReturn (no target-positive or metric support), then DenialOfService (only six strong positives). TOD has 633/932 effective loss cells, 601 from weak DIVE claims, so measure source/label shortcuts and bias before acquiring more TOD examples. ExternalBug has 16,605 NOT_REVIEWED cells; prioritize stratified review for evidence yield, not automatic relabeling. Negative acquisition remains class-specific under R4; these counts do not authorize training, threshold fitting or new labels. Counts of distinct recorded groups do not prove statistical independence.

L0A complete; L0B next: native claim state/crosswalk/unique source records and actual source/license/compiler availability, deduplicated by source identity, verify representative raw source hashes and trace historical-state projection. Then select bounded pilot for L1/L2/L3, prioritizing under-supported classes and negative evaluation coverage. L-G0 remains open.

## 7. L0B original native evidence reconciliation

[Working record and evidence](../evidence/2026-11-26_s4_03_l0b/WORKING_RECORD.md), exact baseline ca7e47c6ce7a1aeaf49f9ee5689e0862bf0dbf06.

| Source | Raw manifest files matched | Normalized identities matched | Distinct native claim rows |
|---|---:|---:|---:|
| DIVE | 22,330 | 22,054 | 57,563 |
| SmartBugs Curated | 143 | 143 | 143 |
| SolidiFI | 350 | 343 | 350 |
| Total | 22,823 | 22,540 | 58,056 |

All 58,056 native claim rows replay exactly from protected source metadata and original CSV under accepted policy. No duplicate exact claim rows. 22 DIVE omitted paths match historical compile-failure count; exact per-path rejection linkage not revalidated. Source identity hashes and metadata ingestion references reconcile. Historical-state projection is intentionally HISTORICAL_MISSING; native claims exist separately.

Native evidence inventory: DIVE 5,911 Unchecked Return Values (5,901 represented claims), 3,781 DoS (all retained) are available for triage, not confirmed positives. Current UnusedReturn and GasException supervision explicitly disabled; DIVE DoS masks remain. SmartBugs 143 annotated files / 207 entries, all line lists in bounds and category-folder consistent. SolidiFI 350 injection logs, literal spelling/encoding variants need semantic verification. Local exact pins and license notice hashes recorded; DIVE portable archive and original license unresolved; individual contract rights unverified. Existing source acquisition spec remains historical; the full SmartBugs pin is now recorded as observed local evidence, not a newly tested external reacquisition.

L-G0 PASS_WITH_LIMITATIONS: accepted logical label and source provenance inventory reconciled, local bounded recovery research may proceed. No new label or source publication authorization; no claim of independent truth, recompile or physical representation revalidation.

### L1A bounded recovery pilot design

- Verify current canonical class definitions before review; isolate UnusedReturn from MishandledException and CallToUnknown, and DenialOfService from GasException. Do not silently change D-016 or other taxonomy.
- Validate line coordinate conventions of SmartBugs JSON and SolidiFI BugLog CSV against exact raw and normalized source on small non-blind calibration samples; preserve literal strings, use mapping only after evidence. Compare locations with source structure and explicit semantics rather than trust filenames.
- For UnusedReturn and DoS, design a small stratified DIVE candidate pilot using randomized controls, compiler era, source family/group, native claim and analyzer disagreement. Freeze case list and rubric before execution. Reserve groups for review; do not inspect or re-adjudicate GAP-007 blind candidates. Use source-grounded evidence, tool capability and independence rules; no automatic promotion or training target.
- For GasException, first define class-specific evidence requirements and identify licensed candidate source families that actually support that class. Separate gas failure from generic DoS; no crosswalk by keyword. External source availability/licensing research is L3, not yet performed.
- Measure candidate-review yield, disagreement, scope failures, source locations, cost and class/group support. If native evidence is insufficient, keep UNKNOWN; seek other sources rather than weakening acceptance. L-G1 requires pilot evidence. No negative/optimizer/threshold authority.

## 8. L1A bounded native annotation and recovery triage

[Working record](../evidence/2026-11-26_s4_03_l1a/WORKING_RECORD.md) at 604d882dac7958baed50c3cafb957b5d060ae28e. Frozen 16 DIVE groups, TRAIN_UNLABELED only, 200 existing R4 reserved groups excluded, two per class x claim status x compiler era. 8 native coordinate calibration files; all 57 spans in bounds, 314/352 annotated line occurrences exact nonempty text at same raw/normalized coordinate, 201 unique exact anchors, all same coordinate. No full source map or semantic equivalence inferred. Six real Slither detectors, 16/16 completed. DoS native claim 3/4 relevant signals, no mapped claim 0/4; UnusedReturn native claim 1/4, no mapped claim 0/4. Signals are not ground truth. Three DoS hit cases lack sufficient evidence of permanent attacker-induced nonfunctionality in source spot-checks; two UnusedReturn positive candidates with typed calls discarding results, one absent the native category (post hoc cross-class finding, excluded from fixed rates). All remain unadjudicated; no new labels. L1B full-source primary and independent reviews, coverage-aware candidate extraction and role reservation enforcement; L-G1 remains open. L3 GasException source research independently eligible. No target changes or full training.

## 9. L1B primary review and complementary call-return coverage

[Evidence](../evidence/2026-11-27_s4_03_l1b/WORKING_RECORD.md), baseline 820bc1457f3260c54c946c704f7695240b19f1d4. Verified actual Slither unused-return coverage limit (HighLevelCall, named token-transfer exclusions). Synthetic fixture corrected after retained compile/API failures; direct expression helper finds four expected functions, Slither finds external/full tuple/partial tuple. Complementary scope only. All 16 frozen sources parsed, 730 function/modifier entries, 10 direct sites in four contracts. Targeted primary source review of 11 contract-class cells, five UnusedReturn source-supported candidates (two without native claim), six insufficient (four DoS, two UnusedReturn); no admitted label or confirmed security impact. Assigned-unused and nested-call examples expose helper limitations. Constant-success booleans show structural applicability and material security risk are separate; do not redefine canonical class. Full structural inventory is not exhaustive manual review; exact review scope recorded per cell.

11/11 neutral local-only packets verified, tampered source and unexpected prior-review member rejected; packet-to-case mapping private and ignored. Reviewer independence pending, one isolated review per packet, no external redistribution given DIVE provenance/rights limitation. L1B complete for primary research/packet preparation only; L1C independent adjudication and explicit successor group reservation enforcement required before any admission. L-G1 open. No new targets, no optimizer/threshold/production changes.

## 10. L3A refreshed external evidence research

[Acquisition plan](S4-03_L3_EXTERNAL_SOURCE_ACQUISITION.md), [research catalog](../evidence/2026-11-27_s4_03_l3a/source_catalog.json). 13 source entries reviewed, including newly located 2026 DIVE archive and dataset/framework license distinction, Gas Gauge, CVE Smart Contracts August 2026, FORGE Curated, EVMbench and Sourcify; existing deferred Bastet/ScaBench/DISL refreshed. Exact upstream Git metadata and selected directory listings checked, no source files downloaded. Prioritize DIVE provenance, GasException benchmark and CVE/FORGE localization. Separate ML candidates, RAG knowledge and project-level evaluation. Legacy source configuration not current R4 authority; do not enable stale negative defaults or checkout scripts. L3A complete, L3B bounded preflight ready; L1C independent review pending, L-G1 open. No labels/roles/artifact changes.

## 11. L3B0/B1 bounded external preflight

[Evidence](../evidence/2026-11-27_s4_03_l3b/WORKING_RECORD.md). DIVE official label archive checksum verified; exact local/published CSV equality, 22,330 records. Additional six-tool results table available with repeated IDs and incomplete observations; no automatic label reconstruction. Gas Gauge pinned 2,000 paths/1,683 blobs; 9/10 frozen source paths, 10/10 Slither reports and four author aggregate reports verified. Rights/run binding and raw/normalized group overlap unresolved, no new source staged or label admitted. L3B0/B1 preflight PASS_WITH_LIMITATIONS; L3B0A raw/tool lineage and L3B1 follow-up, L3B2 independently ready. L-G1 open.

## 12. L3B0A original tool lineage and raw directory consistency

[Evidence](../evidence/2026-11-27_s4_03_l3b0a/WORKING_RECORD.md). DIVE 3,741 repeated IDs with disjoint tool observations, no nonblank conflicts. First-row Power-based replay matches all published positives, four categories exact; remaining extra positives consistent with filtering, no site-level reproduction. Nine published Timestamp native positives lack any positive tool vote in both rows; 880 native Unchecked Return Values zeros have positive observation. Candidate triage only. All local 22,330 DIVE originals match published ZIP member names/size/CRC32; not cryptographic identity. L3B0A complete within scope; L3B0B observation ledger and triage, L3B2 CVE independently ready, L1C review pending; L-G1 open. No target changes.

## 13. L3B0B evidence-only observation ledger and review queue

[Evidence](../evidence/2026-11-27_s4_03_l3b0b/WORKING_RECORD.md). Implemented strict DIVE row/category preservation with 1,251,408 cells, 696,540 nulls, 178,640 native-category records, 22,308 original path links/22,054 artifacts; 22 unmatched IDs explicit. Separate ignored Parquet root and manifest, no truth inference. 21 unique eligible review groups (9 Time unsupported positives, 4 each UR/DoS zero with positive observation, 2 each no-positive comparison), prior 216 groups excluded, combined prospective reservations 237. Explicit optimizer-overlap guard tested but not integrated into successor. 34 focused tests and governance pass; V3 claims and targets unchanged. L3B0B complete; L3B0C primary review and L3B2 CVE preflight independently ready, prioritize external source next. L-G1 open.

## 14. L3B2 CVE acquisition preflight

[Evidence](../evidence/2026-11-27_s4_03_l3b2/WORKING_RECORD.md). Exact catalog selection/labels/localization keys reconciled, 491 records dominated by 437 arithmetic claims; five frozen cases, 19/20 artifacts verified, 4/4 source unit ranges in bounds, one raw DIVE overlap. Rights, source coordinate transformation, runtime and group lineage unresolved; no admission. L3B3 FORGE next; L3B0C review independent.
