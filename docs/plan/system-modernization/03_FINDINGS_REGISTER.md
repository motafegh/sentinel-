# S4 findings register

**Baseline:** `2c4dd3cee0a1313d1f23d87fdb17e682340cacd6`

**State:** updated through S4-01-001; initial observations preserved and later dispositions appended. [Baseline and evidence index](02_BASELINE_INVENTORY.md) | [Protocol](01_AUDIT_PROTOCOL.md)

Findings below are scoped to the observed source and environment. CONFIRMED means the narrow stated issue is established, not that all consequences or exploitability have been demonstrated. CANDIDATE means investigation only. Severity is provisional where impact remains unmeasured; delivery priority is separate. No historical item is closed by this register.

| ID | Seed | Finding / invariant and observed behavior | Evidence state / confidence | Lifecycle | Severity / priority | First owner |
|---|---|---|---|---|---|---|
| S4-F-0001 | 01 | Root workspace manifest must parse under the supported tooling. Local Poetry 2.2.1 rejects deprecated source priority default. | REPRODUCED / HIGH for local failure | CONFIRMED | P2 / NEXT | S4-02 |
| S4-F-0002 | 01,02 | AGENTS reproducible dependency declarations: poetry check --lock fails and 10 common installed package versions differ from lock. | REPRODUCED / HIGH for mismatch | CONFIRMED | P1 / NEXT | S4-02 |
| S4-F-0003 | 02 | DATA reproducible environment: manifest/lock check passes but 69 common installed package versions differ from lock. | REPRODUCED / HIGH for mismatch | CONFIRMED | P1 / NEXT | S4-02 |
| S4-F-0004 | 07 | Historical G4 replay: one-character review digest typo traced and corrected; exact unchanged workflow passes locally. | REPRODUCED / HIGH | CLOSED (local and remote PASS) | P1 / NOW (baseline gate) | S4-00-002 |
| S4-F-0005 | 06 | Current authority projections lag accepted R4 decisions; DATA/ML instructions and handbook present superseded selector/review state. | SOURCE_CONFIRMED / HIGH | CONFIRMED | P1 / NEXT | S4-08 with R4 |
| S4-F-0006 | 03 | Artifact deserialization: unrestricted pickle and torch loads warrant pre-load trust/provenance review. No exploitability established. | SOURCE_CONFIRMED call sites / LOW exposure confidence | CANDIDATE | Pending / NEXT investigation | S4-07 with modules |
| S4-F-0007 | 05 | Silent exception paths: DATA 15 and ML 2 broad handlers with direct pass; potential default-feature ambiguity. | SOURCE_CONFIRMED pattern / LOW defect confidence | CANDIDATE | Pending / PLANNED | S4-03/04 |
| S4-F-0008 | 04 | Large core files may combine responsibilities; file size alone not a defect. | OBSERVATION / LOW defect confidence | CANDIDATE | Pending / PLANNED | S4-03/04/05/08 |
| S4-F-0009 | 08 | Historical docs/.bin path concentration may affect navigation/index/clone; cost and safe retention options not yet established. | OBSERVATION / LOW impact confidence | CANDIDATE | Pending / PLANNED | S4-08 |
| S4-F-0010 | 08 | Nine local untracked paths have no reviewed disposition; preserve and determine relevance safely. | OBSERVATION / HIGH for inventory only | CANDIDATE (evidence gap) | Pending / PLANNED | S4-00/08 |

## Reproduction, scope, and acceptance notes

- **F-0001:** E-0001 environments root; pyproject.toml tool.poetry.source priority. The root is a compatibility workspace, not universal runtime. Cause of observed rejection is invalid priority in current Poetry; supported-tooling policy/need for root remain open. Acceptance: decide supported root role/version; valid manifest/lock resolution and relevant clean install/check, or explicit retirement of root active use. Rollback via versioned manifest/lock. Historical relation: D2-DATA-012, D2-X-007 (candidate relation, not adjudicated).
- **F-0002:** E-0001 agents/pyproject.toml, agents/poetry.lock and installed metadata. Manifest/lock stale; local env differs (LangChain 1.3.1 vs 0.3.28, LangGraph 1.2.0 vs 0.2.76). Clean clone/runtime compatibility not established; do not blindly regenerate lock or update packages. Acceptance: compatible supported resolution, clean isolated environment, relevant tests/seams and drift check; preserve prior lock/environment rollback. Historical relation D2-X-007 pending.
- **F-0003:** E-0001 data_module manifests/lock and installed metadata; compare common names only, no marker resolution claim. Proven drift, not proof of corrupt accepted data. Acceptance: establish supported groups/toolchain and clean reproducibility with equivalence checks or explicit distinct historical environment identities; never regenerate accepted roots in place. Historical D2-DATA-012 and D2-X-007 pending.
- **F-0004:** E-0002 g4_replay and E-0005 run 45817256741. Workflow .github/workflows/r4-phase4-g4.yml, embedded Python line 75; current docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md lacks exact expected review digest. No hash mismatch observed before assertion. Root cause traced to one-character substitution in b7deb49f6; original hash matches actual frozen file and other records. E-0006 records before/after exact replay PASS locally. Exact-commit G4 run 45836727069 succeeded; all seven prior workflows passed (E-0009). CLOSED. Acceptance: trace prior record/authority, restore correct reference or equivalent validated registration without weakening semantic assertions, replay G4 plus affected checks and exact-commit CI. Keep frozen sample/bundle/review untouched. No historical gate reversal.
- **F-0005:** E-0004 exact paths/line evidence. data_module/CLAUDE.md lines 78-98 and ml/CLAUDE.md current stop language omit D-013/D-015/D-016; handbook chapter 16 and current_r4 metadata omit D-015/D-016. Root authority/R4 status override stale text. Impact: assistant misdirection and duplicated work/semantic misunderstanding; no training violation observed. Acceptance: align owner projections and validator coverage to exact D-013/015/016 and published v2 state, preserve historical prose explicitly, test claims and restart navigation; no semantic changes.
- **F-0006:** E-0004 AST call sites (DATA 4, ML 7, AGENTS 6, ZKML 4) includes pickle and torch.load where weights_only is not explicitly True; not all are necessarily unsafe in the installed version. Example ml/src/inference/predictor.py and agents/src/rag/retriever.py. Trace ingress, path ownership, verification before load, privilege and legacy compatibility. D2-ML-004/005 and D2-X-007 are candidate links, not closure decisions. Acceptance depends on exposure; safe synthetic fixture, trust boundary and parity, or documented trusted-only constraint. Do not deserialize untrusted artifacts in baseline.
- **F-0007:** E-0004 broad-handler pattern. graph_extractor.py unchecked-scope branch returns 0.0 after broad catch; need caller/schema semantics, historical compatibility and reproducible failure injection. Candidate relation D2-DATA-010, D2-ML-001. Acceptance: demonstrate correct distinct unknown/error state under governing version or justify historical behavior; immutable artifact identities unchanged.
- **F-0008:** E-0001 per-source physical lines and master seed; graph_extractor.py 2,327, trainer.py 2,091, cli.py 1,053, predictor.py 931, _submit.py 888. Investigate cohesion, coupling, change/test friction, reachability; compare retain vs decomposition using behavior-preserving tests. No automatic file-length threshold as a finding.
- **F-0009:** E-0001 86,168/90,069 tracked paths under docs/.bin. Need actual clone, indexing/navigation cost and historical artifact references/licensing. Prior portfolio history retention controls apply. Acceptance: documented retention design with measurable benefit, no history rewrite or protected data loss; no-change acceptable.
- **F-0010:** E-0001 path/type/size only; no content read in this slice. Determine provenance and ownership in scoped review, classify retain/archive/commit/ignore only with evidence and safe handling. No automatic staging/deletion.

All entries verified or observed at the inspected SHA on 2026-11-26. For confirmed findings, severity is a provisional bounded judgment based on reproducibility/authority impact, not exploit severity; no P0 exposure established in this slice. At S4-00-001 no remediation existed; S4-00-002 repaired F-0004 locally; S4-01-001 verified remote success and closed it. Candidates have no accepted repair, rollback or severity until adjudicated. Source leads do not claim independent corroboration.

## Historical reconciliation

[E-0003 D2 queue](evidence/2026-11-26_s4_00_001/d2_reconciliation_queue.json) contains 86 original IDs, 84 unique and two duplicate tombstones. All unique current dispositions remain UNVERIFIED; original severity/owner/wave preserved, no implication that historical P0/P1 issues are currently open. Assign each owner stage and reconcile original acceptance matrix, R0/V3/R4 closeouts and current regression evidence. Add cross-links only after verification, merge root causes without discarding IDs. Later findings from technical-completion/portfolio tracks are indexed in the baseline and require bounded relevance review.

S4-01-001 performed architecture/usefulness inventory; S4-02 dependency investigation independently eligible. Retain earlier evidence. Do not launch dependency migrations or full training from this register.


## S4-01-001 architecture findings

Source: 9eaaae171c09650f6cca6eae3f4b56c7965cf050. [Architecture map](module-audits/S4-01_ARCHITECTURE_AND_SCOPE.md) and [E-0009 through E-0012](evidence/2026-11-26_s4_01_001/WORKING_RECORD.md). Local repairs are source/test deltas in this record's commit.

| ID | Invariant and actual behavior | Evidence / confidence | Lifecycle | Severity / priority | Owner |
|---|---|---|---|---|---|
| S4-F-0011 | Per-request LLM disable flag must govern execution. Gateway recorded metadata only, node gates process-only, app-level monkeypatch global. | REPRODUCED / HIGH | VALIDATING (local acceptance) | P1 / NOW | S4-01/05 |
| S4-F-0012 | Fast path must require actual screen completion; missing/failed screen with empty hits routed fast; path inferred from result presence. | REPRODUCED / HIGH | VALIDATING (local acceptance) | P1 / NOW | S4-01/05 |
| S4-F-0013 | No usable evidence is not SAFE. compute_overall_verdict({}) returns SAFE; observed unknown ML + no evidence yet SAFE report. | REPRODUCED / HIGH | VALIDATING (local acceptance) | P1 / NOW | S4-01-002 |
| S4-F-0014 | Deterministic-only verdict is not ZK proof. AGENTS verdict/fuse.py and orchestration README describe ZK-anchorable/anchored verdict; retained EZKL proves proxy only. | SOURCE_CONFIRMED / HIGH for claim mismatch | PARTIAL REPAIR (claim sweep pending) | P1 / NEXT | S4-06/08 |
| S4-F-0015 | Gateway default checkpointer disabled; pending jobs marked FAILED on restart, no resume. Whether durable resume is required not yet established. | SOURCE_CONFIRMED / HIGH behavior, LOW defect confidence | CANDIDATE | Pending / PLANNED | S4-05/07 |

F-0011: gateway.py submit_audit/_run_job, state.py, nodes/_helpers.py and three LLM nodes. Before E-0010 failed. Fix job-owned bool, prevent caller metadata override, preserve process controls, remove gateway monkeypatch. Acceptance E-0012: query and app default correctly bind state, next job independent, spy asserts no LLM calls across all three nodes. No external transfer observed. Relevant CI coverage absent; S4-02 track. Rollback narrow source/test delta.

F-0012: graph.py edge, routing.py, nodes/evidence_router.py, synthesizer.py path. Before E-0010 failed. Repair shared route decision and explicit successful ran=True + explicit empty hit list for both tools; branch and trace align; preserve thresholds. Legacy direct synthesizer calls retain old path fallback. E-0012 passes. No overall SAFE/coverage policy resolved; F-0013 separate. Relevant CI coverage pending. Rollback narrow delta.

F-0013: routing.compute_overall_verdict and synthesizer final report. E-0011 trace shows empty fusion + unknown label -> SAFE. Also low-risk narrative driven by ML label; distinguish actual security finding, coverage and abstention; inspect consumers, version report policy, test failed/missing tools, contradictory results and no-evidence paths. No live exploit or scientific negative claim. Immediate next action, not deferred to distant stage.

F-0014: source claim mismatches R4/ZKML proof boundary. Correct documentation/claim tests; retain compatibility keys until explicit versioned decision; no changes to proxy/verifier needed to correct claim. F-0015: source observation; evaluate requirement and current state ownership; no unsupported severity or premature framework replacement.

Historical D2 candidate mappings for these new findings must be reconciled against original records in owner stages; no historical closures inferred. S4-D-0001 accepted scope direction, comparative usefulness not yet tested.

## S4-01-002 disposition

F-0013: E-0013/14 reproduce and repair report SAFE default using S4-D-0002 evidence-adequacy-v1. Projected verdicts distinguish no findings in completed screen scope, insufficient evidence and direct positive signals. Historical raw fusion preserved. 135 selected tests pass; legacy evaluation of new reports intentionally blocked pending coverage-aware S4-01-003. Dedicated remote AGENTS CI not yet established; VALIDATING. F-0014: corrected narrow misleading ZK wording; broader claim sweep remains open. No changes to R4 label truth or protocol. S4-03-L0 label inventory independent priority; see [label program](module-audits/S4-03_LABEL_RECOVERY_PROGRAM.md).
