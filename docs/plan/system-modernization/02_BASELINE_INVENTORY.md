# S4 baseline inventory — S4-00-001

**Recorded:** 2026-11-26
**Inspected source:** `2c4dd3cee0a1313d1f23d87fdb17e682340cacd6` (local WSL main and remote main matched at capture).
**Slice disposition:** inventory complete; G00 not yet evaluated.
**Authority:** [master](00_MASTER_PROGRAM.md), [protocol](01_AUDIT_PROTOCOL.md), [status](CURRENT_STATUS.md).

This is an inventory and bounded validation record, not a function-by-function audit, full test result or release acceptance. Scope includes the five implementation roots, module environments, test entry points, GitHub workflows, selected artifact prerequisites, prior evidence and local untracked state. No dependency installation, live service execution, GPU training, deserialization of checkpoints, physical-root regeneration or on-chain action was performed.

## 1. Evidence index

All captured records are under [evidence/2026-11-26_s4_00_001](evidence/2026-11-26_s4_00_001/).

| Evidence | Record | Establishes |
|---|---|---|
| S4-E-0001 | [inventory.json](evidence/2026-11-26_s4_00_001/inventory.json) | Source identity, exact paths/counts, manifest and lock hashes, package-version inventories, tool probes, artifact presence, workflow names/triggers/hashes, local untracked paths. |
| S4-E-0002 | [baseline_checks.json](evidence/2026-11-26_s4_00_001/baseline_checks.json) | Exact commands, exit codes, elapsed time, output and scope of seven local checks. |
| S4-E-0003 | [d2_reconciliation_queue.json](evidence/2026-11-26_s4_00_001/d2_reconciliation_queue.json) | All original D2 row identities, original dispositions and first review owner; no premature closure. |
| S4-E-0004 | [source_leads.json](evidence/2026-11-26_s4_00_001/source_leads.json) | AST scan leads and specific stale authority text; source patterns are not reachability/exploit findings. |
| S4-E-0005 | [github_runs.json](evidence/2026-11-26_s4_00_001/github_runs.json) | Exact GitHub head SHAs, run IDs and outcomes at observation. |

Capture scripts are retained as executed-method evidence in the same directory; they assume a scratch checkout of the inspected source, WSL and installed tools, and write a new evidence directory. They are not production entry points or new gates. Re-execution must not overwrite this evidence.

## 2. Repository and coverage baseline

Git tracked **90,069 paths** at the inspected SHA; 86,168 reside under docs/.bin. Compared with reconnaissance (90,066 paths at 61854a32a), only the three new S4 documents increased path count. No tracked changes existed before capture. Nine pre-existing untracked paths are listed by path/type/size only in S4-E-0001; preserve them without staging, opening secret material or deleting them. These include three local audit files, R4 diagnostic reports, two earlier plans and an environment example. Disposition is not decided.

| Module | Core source root | Files | Physical lines | Test source files | Review level |
|---|---|---:|---:|---:|---|
| DATA | data_module/sentinel_data | 111 | 25,872 | 82 | Inventory + selected static leads |
| ML | ml/src | 50 | 13,791 | 22 | Inventory + selected static leads |
| AGENTS | agents/src | 114 | 20,761 | 66 | Inventory + selected static leads |
| ZKML | zkml/src | 13 | 2,282 | 8 | Inventory + selected static leads |
| Contracts | contracts/src | 4 | 2,032 | 11 | Inventory only |
| Total | five core roots | 292 | 64,738 | 189 | Not exhaustive |

Counts use tracked .py/.sol paths within stated source/test roots, including comments, blank lines, test helpers and generated verifier; not logical LOC or test-case counts. Detailed source file list and tracked subarea counts are in E-0001. Scope classification at this point is **provisional by directory**; callers/entry-point reachability are for later module audits.

| Scope | Responsibilities and primary lifecycle | Follow-up |
|---|---|---|
| DATA | ingestion, preprocessing, labeling, splitting, graph/token representation, vnext, verification/export/registry. Active build/research plus historical DVC compatibility. Benchmarks, docs, temp and audits are research/fixtures/history until reachability established. | S4-03 |
| ML | datasets, models, training, preprocessing/data extraction, inference and utilities. Run12 historical runtime vs R4 research; scripts include active tools and archived experiments. | S4-04 |
| AGENTS | API/gateway, contracts/schema, eval, ingestion, LLM, MCP, orchestration, persistence, RAG, security. Active runtime and research/eval; historical reports/fixtures are not live execution. | S4-05 |
| ZKML | distillation and EZKL code; retained proof/proxy lineage is supported historical evidence. New teacher handoff gated. | S4-06 |
| Contracts | AuditRegistry, SentinelToken, verifier interface and generated verifier; current protocol plus V1/V2 compatibility. Scripts, dependencies and test fixtures separate from runtime. | S4-06 |
| Cross-system | schemas, tools, scripts, root manifests, CI, handbook and plans; historical .bin/archive material not active code by default. | S4-01/02/07/08 |

No per-file runtime lifecycle certification is claimed; evidence records exact source locations to support later trace.

## 3. Environment fingerprint and reproducibility limits

WSL Python 3.12.3; Poetry 2.2.1; Git 2.43.0; Docker CLI 29.9.3; Foundry 1.5.1-stable available at ~/.foundry/bin/forge (not on observed noninteractive PATH); NVIDIA RTX 3070 Laptop GPU, 8,192 MiB, driver 621.24 reported by nvidia-smi. uv and ruff not found as host commands; this does not establish absence in all virtual environments. Docker daemon, CUDA compute, compilers and analyzers were not exercised.

| Environment | Lock check | Unique locked names | Installed names | Common names with different version |
|---|---|---:|---:|---:|
| Root | FAIL: invalid source priority | 197 | Not probed | N/A |
| DATA | PASS | 177 | 153 | 69 |
| ML | PASS | 301 | 372 | 0 |
| AGENTS | FAIL: manifest/lock drift | 172 | 204 | 10 |

Comparison uses package names lowercased as recorded by the lock and installed metadata; common-name version differences are direct observations, not an exhaustive marker/optional-group resolution or compatibility diagnosis. Missing/extra packages may result from platform markers, optional groups or research installs. Do not infer ML reproducibility simply from zero common-version differences (27 locked names not observed and 98 extra names). Root environment installed packages not probed. All four manifest/lock SHA-256 fingerprints and package versions are in E-0001.

Examples: DATA locked/installed PyTorch 2.14.0/2.12.0, Transformers 5.16.1/5.11.0, Slither 0.10.0/0.11.5. AGENTS locked/installed LangChain 0.3.28/1.3.1, LangGraph 0.2.76/1.2.0, Starlette 1.0.0/0.46.2; FastAPI 0.116.2 installed but absent from lock. These are local observations, not upstream release/advisory verification. ZKML uses ML environment, contracts use Foundry; no environment was upgraded.

## 4. Artifact and external prerequisite inventory

Both D-011 and D-013 accepted physical roots exist at exact paths referenced by the tracked acceptance records; recorded digests are d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd and 9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881 respectively. The acceptance record hashes and paths are in E-0001. Presence is not a re-verification of 22,540 identities or physical bytes.

| Artifact / prerequisite | Availability at capture | Limit |
|---|---|---|
| Historical Run12 checkpoint | Present, 281,562,156 bytes, untracked/ignored | No load, full hash or quality claim. |
| RAG FAISS/BM25/chunks/metadata | All four present, untracked/ignored | No deserialization or index coherence test. |
| ZKML proxy/compiled/settings/VK | Present, tracked, small-file hashes captured | No proof regeneration or key compatibility claim. |
| ZKML SRS and PK | Present, untracked/ignored | No hash or load; generation/verification not exercised. |
| Candidate #2 v2 review archive | Present, tracked, hash 9036d5744ef903f7ba1204edb3a248f234ca1675349b8b8d19a0e6f9bab4fc36 | No candidate adjudication; independent review still pending. |
| Foundry lock and local libraries | Lock tracked/hashed; library directories present | Tool/library/compiler compatibility and build not exercised. |
| LM Studio, MCP services, RPC, compiler/analyzer availability, GPU inference | NOT_RUN / not established | No secret checks, networking or live runtime claims. |

Protected data, local model/cache/index state and historical artifacts were not mutated. Fresh clone does not guarantee these local prerequisites.

## 5. Test and CI coverage

Module commands are taken from current README/DEVELOPMENT/VALIDATION or workflow definitions. Full suites and runtime tests are **NOT_RUN in this slice**; schedule safe isolated baselines with exact environment and prerequisites in S4-00-002. Do not compare raw test counts across incompatible environments.

| Scope | Candidate baseline command / source | Prerequisites and limitation |
|---|---|---|
| DATA | cd data_module && .venv/bin/python -m pytest -q; Phase-8 repair workflow | Core/pipeline/ML extras and compatible tools; protected artifact-dependent tests only under explicit safe scope. |
| ML | ml/.venv/bin/python -m pytest ml/tests -q | ML env, fixtures; GPU/model/artifact tests require separate validation; no training. |
| AGENTS | cd agents && .venv/bin/python -m pytest -q | Local env differs from lock; external ML, RAG, MCP, analyzers, LLM unavailable unless established; isolate tests from real services. |
| ZKML | ml/.venv/bin/python -m pytest zkml/tests -q | ML/EZKL deps, retained fixtures; proof generation separately scoped. |
| Contracts | cd contracts && ~/.foundry/bin/forge test | Local libraries, compiler, offline fixture configuration; no broadcast, install or network by default. |
| Cross-system | system-alignment.yml, v3-runtime-feedback.yml, showcase and handbook | CI dependency-light boundary tests are not full service coverage. |

Executed at the exact source SHA (E-0002):

| Check | Result and scope |
|---|---|
| Handbook static | PASS 145/145 |
| Current R4 authority validator | PASS 56/56 (encoded checks only; authority drift still found) |
| Handbook unit tests | PASS 13/13 |
| Showcase | PASS; retained proxy check PASS_WITH_LIMITATION (UNSAFE); live ML, analyzers, LangGraph audit, proving and signing/broadcast NOT_RUN |
| Historical G6 frozen partitions | PASS, 22,493 contracts, 0 confirmed negatives; historical role scope only |
| Committed logical V3 snapshot | PASS, 11 checksums and 60 coherence checks; 200 queue cells/groups; no training authority |
| Historical G4 exact workflow Python body | FAIL, exit 1, line 75 missing expected review digest reference in current EVIDENCE_GAP_REGISTER.md |

G4 reproduces [run 45817256741](https://github.com/motafegh/sentinel-/actions/runs/45817256741) on the prior code baseline, after frozen sample/bundle/review hash assertions succeeded. This is evidence of a present replay/registration inconsistency, not proof of historical review corruption. Trace history and reconcile reference without weakening gate in a bounded subsequent slice.

Three triggered CI workflows at inspected SHA all completed SUCCESS (handbook 45819267557, security hygiene 45819267576, showcase 45819267646). Prior SHA 61854a32a had six successes and one G4 failure. No new G4 run at this SHA; local replay establishes its present failure. 25 top-level workflows are inventoried in E-0001; full path/filter/permission and historical-vs-current review belongs to S4-02. Exact run data in E-0005.

## 6. Historical evidence and reconciliation queue

[D2 canonical registry](../../audit/system-engineering/2026-07-14_SYSTEM_D2_source_engineering_audit/registry/2026-07-14_SYSTEM_D2_FINDINGS_registry.md) and [acceptance matrix](../../audit/system-engineering/2026-07-14_SYSTEM_D2_source_engineering_audit/acceptance/2026-07-14_SYSTEM_D2_ACCEPTANCE_matrix.md) are the original baseline. All 86 raw identities are indexed in E-0003 (84 unique; D2-AGT-013 -> D2-ZKC-004 and D2-X-009 -> D2-AGT-001 historical duplicate tombstones). Each unique row starts UNVERIFIED/QUEUED, not open or resolved by default; owner stages assigned by original module. Cross-module items assigned S4-07; coordinate original closure waves and linked findings.

Additional required reconciliation sources: [R0 main integration closeout](../system-finalization/2026-07-16_SYSTEM_R0_PLAN_main_integration_closeout.md), [V3/ZKML contracts alignment closeout](../system-finalization/2026-08-11_ZKML_CONTRACTS_ALIGNMENT_cross_module_closeout.md), [R4 status/decisions](../ml-R4/PLAN_STATUS_MATRIX.md), [technical completion](../system-finalization/technical-completion/00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md), [portfolio status](../portfolio-professionalization/CURRENT_STATUS.md). These are indexed, not yet fully reconciled; do not substitute historical report claims for current tests. Local untracked historical audits remain unreviewed for disposition.

## 7. Slice close and next work

S4-00-001 establishes a bounded baseline: local/remote SHA matched, five modules inventoried, test prerequisites and unexecuted scope explicit, artifact presence distinguished from integrity, dependency drift reproduced, seven safe checks run, historical queue indexed. New leads and narrow confirmed defects are registered in [03_FINDINGS_REGISTER.md](03_FINDINGS_REGISTER.md).

G00 remains IN_PROGRESS. Next S4-00-002: baseline test safety/coverage and historical G4 registration repair investigation; review test commands/fixtures before running, isolate state, capture available bounded module checks, and decide whether to fix the missing digest reference based on original immutable G4 evidence. No install/upgrade or protected artifact work without explicit scoped decision. Once baseline discrepancies have disposition and adequate coverage, evaluate G00; S4-01/02 may then start.

## 8. S4-00-002 bounded validation addendum (2026-11-26)

Source at slice start: 5d8dd776ad647c0c32778653342f84ab5b5d27df. [Working record and G00 decision](evidence/2026-11-26_s4_00_002/WORKING_RECORD.md); [E-0006 repair](evidence/2026-11-26_s4_00_002/g4_repair.json), [E-0007 module runs](evidence/2026-11-26_s4_00_002/module_baseline.json), [E-0008 post-repair checks](evidence/2026-11-26_s4_00_002/post_repair_checks.json). Earlier E-0001/2 results remain immutable observations.

| Module | Selected tests passed | Boundary |
|---|---:|---|
| DATA | 42 | Archive containment, V3 grouping, policy |
| ML | 10 | Synthetic compatibility/role masks, tracked historical projection |
| AGENTS | 40 | Read-only MCP, mocked RPC reads, V3 policy; initial harness collection error retained |
| ZKML | 11 | Structure and lineage; no real proving |
| Contracts | 50 | Selected local registry/token/V3/upgrade tests; no real proof, invariant or golden digest suite |
| Total | 153 | Bounded selection, not full module suite |

Network-disabled scratch source, existing local venvs, isolated test environments and local offline compiler. No install, training, live service, external chain, protected-root or real checkpoint use. Library copies fingerprinted but lock conformance unverified. Exact G4 unchanged workflow now passes locally after one-character correction to current digest reference; remote validation pending. Historical artifacts, D-011/D-013 semantics and launch hold preserved. S4-G00 PASS_WITH_LIMITATIONS permits architecture/dependency investigation only; full suites, clean installs, actual runtime and physical integrity remain unestablished.
