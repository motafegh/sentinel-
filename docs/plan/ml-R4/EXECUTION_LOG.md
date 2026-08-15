# R4 Execution Log

Append one entry for each work package.

## Entry

### R4-LOG-YYYYMMDD-NNN — <title>

- **Phase:**
- **Gap ID, if review work:**
- **Operator:**
- **Date/timezone:**
- **Repository branch/commit:**
- **Worktree status before:**
- **Input artifact IDs/hashes:**
- **Command(s):**
- **Environment and seed(s):**
- **Expected outputs:**
- **Actual outputs/hashes:**
- **Result:** `PASS` / `FAIL` / `PARTIAL` / `BLOCKED`
- **Historical evidence reused:**
- **New evidence created:**
- **Protected artifacts changed:** `NO` / `YES`
- **Register updates:**
- **Gate effect:**
- **Next permitted action:**

---

### R4-LOG-20260716-002 — Phase 0 Closure Corrections

- **Phase:** 0 (closure correction)
- **Gap ID, if review work:** N/A
- **Operator:** AI implementation agent
- **Date/timezone:** 2026-07-16 UTC
- **Repository branch/commit:** r4/phase1-previous-evidence-recovery / 6febb4283f92e801bf70e33de2cb00c409e8284f
- **Worktree status before:** clean (no modified tracked files beyond DVC tmp locks)
- **Input artifact IDs/hashes:** Same as R4-LOG-20260716-001
- **Command(s):** jsonschema validate(instance=baseline_manifest.json, schema=baseline_manifest.schema.json); sha256sum on 26 protected artifacts; python3 scripts/p0_baseline_freeze.py --validate
- **Environment and seed(s):** WSL2 Ubuntu 24.04, Python 3.12.1, jsonschema 4.10.3
- **Expected outputs:** Corrected findings/01_baseline_and_evidence_location.md (audited baseline vs R4 output commit distinction, 26 protected, 30 evidence sets, lineage note); schema validation record; updated protected_artifacts.json (both commits recorded); updated p0_baseline_freeze.py (--validate flag); this log entry
- **Actual outputs/hashes:** All corrections applied. Schema validation: PASS (15 artifacts, all type assertions PASS). Hash re-verification: 26/26 OK.
- **Result:** PASS
- **Historical evidence reused:** None
- **New evidence created:** None
- **Protected artifacts changed:** NO
- **Register updates:** EXECUTION_LOG (this entry)
- **Gate effect:** None (G0 already PASS; corrections close Phase 0 properly)
- **Next permitted action:** Begin Phase 1 — Previous Evidence Recovery

### R4-LOG-20260716-001 — Phase 0 Baseline Freeze and Evidence Location

- **Phase:** 0
- **Gap ID, if review work:** N/A (no contract review)
- **Operator:** AI implementation agent
- **Date/timezone:** 2026-07-16 UTC
- **Repository branch/commit:** main / 4b5bd333c63ab7a7ec83810fbbae54f3ebf1b493
- **Worktree status before:** dirty (untracked files only: docs/plan/ml-R4/, audit .md files, R0/R4 plan files). No modified or staged tracked files.
- **Input artifact IDs/hashes:** N/A (Phase 0 is the first phase; inputs are the existing repository state)
- **Command(s):** git status; git rev-parse HEAD; git worktree list; sha256sum (17 artifacts); dvc status; pyarrow parquet reads; systematic file search for 14 evidence categories
- **Environment and seed(s):** WSL2 Ubuntu 24.04, Python 3.12.1, ml/.venv with pyarrow. No seed (read-only investigation).
- **Expected outputs:** baseline_manifest.json, protected_artifacts.json, availability_inventory.csv, evidence_location_inventory.csv, findings/01_baseline_and_evidence_location.md, scripts/p0_baseline_freeze.py
- **Actual outputs/hashes:** All 6 outputs created under docs/plan/ml-R4/. See manifests/ for SHA-256 hashes of protected artifacts.
- **Result:** PASS
- **Historical evidence reused:** None consumed; 24 evidence sets located and registered (DIVE 8, BCCC 4, SolidiFI 2, SmartBugs 3, Web3Bugs 1 UNAVAILABLE, DeFiHackLabs 1, Manual 3, Benchmark 1, tools 1 UNAVAILABLE, exploit PoC 1 UNAVAILABLE)
- **New evidence created:** None (Phase 0 prohibits contract review)
- **Protected artifacts changed:** NO
- **Register updates:** EXECUTION_LOG (this entry), ARTIFACT_INDEX (24 artifacts), PREVIOUS_EVIDENCE_REGISTER (24 evidence sets), RISK_AND_BLOCKER_REGISTER (R4-B001 closed, R4-R006/R4-R007 added), PLAN_STATUS_MATRIX (Phase 0 -> PASSED)
- **Gate effect:** G0 PASS — all 8 pass criteria met. Phase 1 entry condition satisfied.
- **Next permitted action:** Begin Phase 1 — Previous Evidence Recovery (DIVE workstream first)

---

### R4-LOG-20260716-003 — Phase 1 Previous Evidence Recovery

- **Phase:** 1
- **Gap ID, if review work:** N/A (no contract review)
- **Operator:** AI implementation agent
- **Date/timezone:** 2026-07-16 UTC
- **Repository branch/commit:** r4/phase1-previous-evidence-recovery / (current working commit)
- **Worktree status before:** clean (committed Phase 0 closure corrections)
- **Input artifact IDs/hashes:** R4-P0-EVD-001 through R4-P0-EVD-007, R4-P0-LBL-001, R4-P0-XWK-001..003, plus all evidence source directories
- **Command(s):** Task agents for DIVE, BCCC, other sources, lineage investigations; jsonschema validate; sha256sum; structured recovery reports
- **Environment and seed(s):** WSL2 Ubuntu 24.04, Python 3.12.1, jsonschema 4.10.3
- **Expected outputs:** findings/02A, 02B, 02C, 02D, 02_previous_evidence_recovery_summary.md; manifests/evidence_inventory.jsonl; updated registers (PREVIOUS_EVIDENCE_REGISTER, ARTIFACT_INDEX, EVIDENCE_GAP_REGISTER, RISK_AND_BLOCKER_REGISTER, EXECUTION_LOG, PLAN_STATUS_MATRIX)
- **Actual outputs/hashes:** All expected outputs created. Detailed findings in workstream-specific files. 27 evidence items in JSONL.
- **Result:** PASS
- **Historical evidence reused:** 30 evidence sets analyzed: 17 RECOVERED_VERIFIED, 6 RECOVERED_PARTIAL, 7 UNAVAILABLE. DIVE review mds, BCCC Phase 5 v1.4, SolidiFI, SmartBugs, manual contracts, benchmark, AI reports, data audit all recovered.
- **New evidence created:** None (Phase 1 prohibits contract review). 6 evidence gaps proposed (not approved).
- **Protected artifacts changed:** NO
- **Register updates:** PREVIOUS_EVIDENCE_REGISTER (all statuses updated to RECOVERED_VERIFIED/PARTIAL/UNAVAILABLE), ARTIFACT_INDEX (5 Phase 1 artifacts added), EVIDENCE_GAP_REGISTER (6 PROPOSED gaps), RISK_AND_BLOCKER_REGISTER (3 new risks: R4-R010-R013), EXECUTION_LOG (this entry), PLAN_STATUS_MATRIX (Phase 1 -> PASSED)
- **Gate effect:** G1 PASS — all 5 pass criteria met. See findings/02_previous_evidence_recovery_summary.md for detailed assessment.
- **Next permitted action:** Begin Phase 2 — Label Corruption Reconstruction, subject to approved EVIDENCE_GAP_REGISTER entries only

---

### R4-LOG-20260811-004 — Phase 2 Label-Corruption Mechanism Reconstruction

- **Phase:** 2
- **Gap ID, if review work:** N/A — no new contract review performed
- **Operator:** ChatGPT + repository source audit
- **Date/timezone:** 2026-08-11 Europe/Berlin / repository work performed remotely through GitHub
- **Repository branch/commit:** `r4/phase2-label-corruption-reconstruction` / Phase-2 branch commits beginning at `96037edd8`
- **Worktree status before:** Remote branch created from canonical `main` commit `253cbdec0`; local runtime/DVC state intentionally excluded from this remote-only reconstruction
- **Input artifact IDs/hashes:** Phase-0 protected baseline and crosswalks; Phase-1 findings 02A/02B/02C/02D; current executable DATA/ML source on branch
- **Command(s):** repository source tracing across config, ingestion folderization, label parsers/crosswalks, merger, verification gate, split, NonVulnerable cap, export/chunk/shard writers, SentinelDataset/collate, and AsymmetricLoss; retained-evidence reconciliation
- **Environment and seed(s):** GitHub-tracked source/evidence only; no stochastic execution and no new dataset review
- **Expected outputs:** source authority matrix; source semantics cards; crosswalk effect table; merger sensitivity table; all-zero decomposition; population reconciliation; quantitative matrix; end-to-end trace JSONL; Phase-2 summary
- **Actual outputs/hashes:** All expected semantic reconstruction outputs created. Protected source/data artifacts were not modified. Exact DVC-only source/class sub-cross-tabs remain explicitly unavailable where not retained in Git.
- **Result:** PASS
- **Historical evidence reused:** DIVE source/crosswalk/audit evidence; SolidiFI source semantics; SmartBugs evidence; BCCC v1.4/deep-dive evidence; June 13 data audit; Run12 lineage; Phase-0 protected counts
- **New evidence created:** Source-code-derived reconstruction only; no new contract adjudication, tool scan, exploit reproduction, or label correction
- **Protected artifacts changed:** NO
- **Register updates:** PLAN_STATUS_MATRIX Phase 2 -> PASSED; RISK_AND_BLOCKER_REGISTER R4-R010 narrowed/mitigated and R4-R014 added; ARTIFACT_INDEX Phase-2 outputs; EXECUTION_LOG this entry
- **Gate effect:** **G2 PASS** — every mandatory historical positive/zero origin category has a named transformation path. Contract×class adjudication may remain unknown as permitted by the Phase-2 gate.
- **Next permitted action:** Begin Phase 3 — Evidence Ledger. Build a sidecar contract×class evidence ledger without rewriting the protected historical export.

---

### R4-LOG-20260811-005 — Phase 3 Evidence-Ledger Framework Checkpoint

- **Phase:** 3
- **Gap ID, if review work:** N/A — no new contract review performed
- **Operator:** ChatGPT + GitHub repository implementation
- **Date/timezone:** 2026-08-11 Europe/Berlin
- **Repository branch/commit:** `r4/phase3-evidence-ledger` / branch created from canonical G2-passed main `380c6f468cc1971e9ca995af0bd48895797573a5`
- **Worktree status before:** remote branch; no protected DATA/ML/runtime artifact modifications
- **Input artifact IDs/hashes:** R4-P0-EXP-002 and R4-P0-SPL-001..004 identities/counts; Phase-2 findings/trace manifest; Label State and Dataset Role Policy; Phase-3 specification/templates
- **Command(s):** design versioned ledger/evidence/manifest schemas; implement semantic validator; create deterministic valid/invalid fixtures and unittest harness; map Phase-2 mechanisms into conservative ledger states; seed category/source-scoped evidence; probe protected v3 row artifacts through GitHub contents API
- **Environment and seed(s):** GitHub-tracked repository only; deterministic fixtures; no stochastic execution
- **Expected outputs:** Phase-3 execution plan, row/evidence/manifest schemas, validator, fixtures/tests, state mapping, evidence seed, draft production manifest, schema report
- **Actual outputs/hashes:** Framework outputs created. Production manifest declares expected 22,493 contracts / 224,930 rows but actual 0 / DRAFT. GitHub reads returned 404 for protected v3 train.jsonl, split_manifest.json, and labels.parquet, proving row identities are unavailable in ordinary remote contents.
- **Result:** BLOCKED
- **Historical evidence reused:** Phase-0 protected artifact identities/counts and Phase-2 transformation evidence only
- **New evidence created:** deterministic schema/validator fixtures and source-derived transformation evidence; no contract adjudication
- **Protected artifacts changed:** NO
- **Register updates:** PLAN_STATUS_MATRIX Phase 3 -> BLOCKED; RISK_AND_BLOCKER_REGISTER adds R4-B002; Phase-3 framework report records exact unblock condition
- **Gate effect:** **G3 NOT PASSED.** Schema/state representation requirement is implemented, but the required 224,930-row production ledger and validation cannot be produced from aggregate counts/hashes.
- **Next permitted action:** Make the protected v3 split/labels row population available to an execution environment, verify hashes against R4-P0 artifacts, materialize the full ledger, run validator/self-tests, and reassess G3. Phase 4 remains WAITING.

---

### R4-LOG-20260811-006 — Phase 3 Protected-Population Materialization and G3 Closure

- **Phase:** 3
- **Gap ID, if review work:** N/A — no new contract adjudication performed
- **Operator:** User local execution + ChatGPT repository review/registration
- **Date/timezone:** 2026-08-11 Europe/Berlin
- **Repository branch/commit:** `r4/phase3-evidence-ledger`; materialization generation commit `b8911daed077db573a2c421fb5e21a9811b62526`; publication commit `17fa204955e1228b1d2f691f2f7e3fe76875085a`
- **Worktree status before:** local Phase-3 branch with protected Phase-0 split/export/representation artifacts available; no protected artifact modifications
- **Input artifact IDs/hashes:** R4-P0-SPL-002 `03f2a237...`, R4-P0-SPL-003 `cf9a7b45...`, R4-P0-SPL-004 `b9bb4649...`, R4-P0-EXP-002 `26e739b5...`, Phase-3 evidence items `f0b2684d1b59272a549e61801287cf381e312b3af429507fcd06e60a3705f36d`
- **Command(s):** `ml/.venv/bin/python docs/plan/ml-R4/scripts/p3_run_local_gate.py`; this ran semantic validator self-tests, strict schema-surface self-tests, artifact-binding self-tests, frozen protected-population verification, staged ledger materialization, strict production validation, staged binding, canonical promotion, and canonical binding validation
- **Environment and seed(s):** existing local SENTINEL ML/data environment with `pyarrow`; deterministic transformation; no stochastic seed and no DVC command required
- **Expected outputs:** complete 224,930-row ledger, materialized manifest, semantic report, strict report, artifact-binding report
- **Actual outputs/hashes:** 22,493 contracts; 224,930 rows / 224,930 unique keys; 21,657 represented contracts; ledger SHA-256 `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`; strict report SHA-256 `acd54021e8ff614c5517b1dbc0eecbcf20ac076aa43a12d9713837e4a2427b2b`; evidence SHA-256 `f0b2684d1b59272a549e61801287cf381e312b3af429507fcd06e60a3705f36d`; all semantic/strict/binding reports PASS with zero errors; 51,546 historical positives remain `NOT_REVIEWED`; 173,384 historical zeros remain `UNKNOWN`; all 224,930 rows are initially `TRAIN_UNLABELED` and excluded from outcome metrics
- **Result:** PASS
- **Historical evidence reused:** frozen Phase-0 population/artifact identities and Phase-2 transformation evidence only
- **New evidence created:** materialized sidecar ledger and validation/binding reports; no vulnerability outcome adjudication
- **Protected artifacts changed:** NO
- **Register updates:** ARTIFACT_INDEX production Phase-3 identities; RISK_AND_BLOCKER_REGISTER R4-B002 -> CLOSED and R4-R002/R4-R003 mitigation state clarified; PLAN_STATUS_MATRIX Phase 3 -> PASSED and Phase 4 -> READY; Phase-3 specification -> G3 PASS; EXECUTION_LOG this entry
- **Gate effect:** **G3 PASS.** The complete export-relevant contract×class population is represented without forcing historical unknowns/zeros into confirmed negatives.
- **Next permitted action:** Begin Phase 4 — Targeted Gap Adjudication. Every new review action requires an authorized Gap ID; no Phase-4 adjudication is implied by Phase-3 materialization.

---

### R4-LOG-20260811-007 — Phase 4 Targeted DIVE Gap Adjudication and G4 Closure

- **Phase:** 4
- **Gap ID, if review work:** `R4-GAP-002`
- **Operator:** ChatGPT / GPT-5.6 Sol primary semantic reviewer + user local protected-source bundle materialization; routine technical/governance approval delegated by the human owner
- **Date/timezone:** 2026-08-11 Europe/Berlin (execution crossed into 2026-08-12 in the user's local timezone)
- **Repository branch/commit:** `r4/phase4-targeted-gap-adjudication`; frozen source bundle commit `02f254249f16b2f940dca0c9a9309e6b38bade12`; machine-readable review publication commit `c8f283f5961f2955c7738409bf8298dc41c599bd`
- **Worktree status before:** Phase-3 G3-passed canonical ledger on `main`; protected local DIVE preprocessed source available without modifying protected historical artifacts
- **Input artifact IDs/hashes:** R4-P3-LED-001 `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`; DIVE crosswalk/evidence R4-P0-XWK-001 and R4-P0-EVD-001..003; frozen sample SHA-256 `2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9`; blind source bundle SHA-256 `2b1ce12fdd96819c89bbb9fe1dfb2d9aa992ec0a05ce32f651c6b834b97ddf38`
- **Command(s):** `p4_freeze_gap002_sample.py`; `p4_build_gap002_review_bundle.py`; checksum verification and safe-unpack CI; source-only blind semantic review; `p4_publish_gap002_blind_review.py`; review/sample binding CI
- **Environment and seed(s):** deterministic SHA-ranked group-aware sample from committed Phase-3 ledger; TRAIN-only; groups touching val/test excluded; no stochastic review seed; blind source-only semantic pass with model/tool/merger/non-target-label evidence hidden
- **Expected outputs:** authorization record; frozen population/sample; checksum-bound blind source bundle; 100 semantic verdicts; role recommendation and uncertainty report; explicit gap/G4 disposition
- **Actual outputs/hashes:** 100 unique contracts/review groups, 20 per stratum; review rows SHA-256 `7d7f0cce287c847df2376ac0f580abf6be05b46f6a2c90b5f00f9b34f8fc5473`. Blind results: DenialOfService 0 support / 20 not-support; IntegerUO 3 / 16 / 1 unclear; Timestamp 4 / 15 / 1 unclear; TransactionOrderDependence 12 / 5 / 3 class-boundary conflicts; UnusedReturn 9 / 11. CI regenerated and bound all review identities successfully.
- **Result:** PASS
- **Historical evidence reused:** Phase-1 DIVE EB/RE manual reviews and correlated Slither/Aderyn findings; Phase-2 source/crosswalk semantics; Phase-3 ledger. Historical/tool evidence was revealed only after the blind semantic verdicts were locked.
- **New evidence created:** single-AI primary source-only semantic review of 100 checksum-bound DIVE contracts; explicit source/stratum reliability evidence. This is not human/inter-rater or untouched-acceptance evidence.
- **Protected artifacts changed:** NO
- **Register updates:** R4-GAP-002 -> RESOLVED; Web3Bugs and provisional inactive BCCC first-baseline populations -> MASK_OR_EXCLUDE/deferred; Phase 4 -> PASSED; Phase 5 -> READY; Phase-4 artifacts registered in ARTIFACT_INDEX
- **Gate effect:** **G4 PASS.** DIVE DoS/Arithmetic/Time manipulation/Unchecked Return Values source assertions are masked/excluded for the first baseline; DIVE Front Running/TOD is limited to `TRAIN_WEAK` and barred from outcome metrics, model selection, threshold/calibration fitting, and untouched acceptance. `DOES_NOT_SUPPORT_POSITIVE` does not create a confirmed negative.
- **Next permitted action:** Begin Phase 5 — DATA vNext Policy and Design. Encode the Phase-0–4 source/class/state/role decisions in versioned ADRs/specification before any implementation makes semantic choices.

---

### R4-LOG-20260812-008 — Phase 5 DATA vNext Policy and G5 Closure

- **Phase:** 5
- **Operator:** GPT-5.6 Sol under delegated routine technical/governance approval
- **Date:** 2026-08-12
- **Repository branch/commit:** `r4/phase5-data-vnext-policy-design` at `104dd4f6f8a186f28a5d2a0f34aa960d274295b8` before deterministic closeout commit
- **Inputs:** Phase-0–4 evidence, Phase-3 ledger `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`, Phase-4 R4-GAP-002 review `7d7f0cce287c847df2376ac0f580abf6be05b46f6a2c90b5f00f9b34f8fc5473`
- **Outputs:** accepted machine policy SHA-256 `b1cfce9cf85c49e4eea533808005d466e0872e98737d366641e287e2a8cfe094`; label-state schema SHA-256 `14e414a568f090891cb39b4a9a16b3c710d9d69e2279aace50c310aece98959b`; implementation specification SHA-256 `fdf236a4bf8729a4bf3ee5e3c2c9b0a4dce2efc8666a25fae007204b12a913d4`; five Accepted ADRs; R4-D-001..005 decision-register entries
- **Validation:** JSON-Schema row invariants + machine policy assertions + design-only branch scope guard passed in `R4 Phase 5 DATA vNext policy` CI
- **Key decisions:** target zero requires confirmed negative; no blanket negative source; SolidiFI/approved SmartBugs direct categories strong positive; DIVE weak TOD only and otherwise unlabeled/masked; SmartBugs bad_randomness/short_addresses/other no canonical target; GasException and UnusedReturn supervision disabled; export format v2 explicit; historical v1 immutable
- **Protected artifacts changed:** NO
- **Implementation code changed:** NO
- **Gate effect:** **G5 PASS.** Phase 6 is authorized to create leakage-safe dataset roles/partitions and freeze or explicitly declare unsupported acceptance support.
- **Next permitted action:** Phase 6 only; Phase-7 DATA implementation remains blocked until G6.

---

### R4-LOG-20260812-009 — Phase 6 Leakage-Safe Roles and G6 Closure

- **Phase:** 6
- **Operator:** GPT-5.6 Sol under delegated routine technical/governance approval
- **Date:** 2026-08-12
- **Inputs:** Phase-3 ledger `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`; accepted Phase-5 policy `b1cfce9cf85c49e4eea533808005d466e0872e98737d366641e287e2a8cfe094`; measured role-support inventory; repository exposure audit
- **Partition:** `r4-vnext-roles-v1` / status `FROZEN_G6`
- **Population:** 22493 contracts / 13509 leakage groups, exactly one role each
- **Roles:** TRAIN_STRONG 275 contracts; MODEL_SELECTION 56; INTERNAL_AUDIT 62; TRAIN_WEAK 773; TRAIN_UNLABELED 20,491; EXCLUDED 836
- **Evidence limitations:** zero confirmed-negative rows; MODEL_SELECTION positive-only; THRESHOLD_FIT and CALIBRATION_FIT unsupported empty; UNTOUCHED_ACCEPTANCE unsupported empty frozen
- **Exposure findings:** manual suite exposed to historical ML/AGENTS validation; quickstart contains invalid NonVulnerable mappings; Tier-E BCCC/tool-silence design is not confirmed-negative evidence; unavailable/deferred sources not imported
- **Protected artifacts changed:** NO
- **Implementation code changed:** NO
- **Decision:** R4-D-006 / ADR-R4-006 Accepted
- **Gate effect:** **G6 PASS.** Phase 7 is authorized to implement DATA vNext exactly from the frozen policy and role manifests; threshold/calibration/acceptance limitations must remain explicit.

---

### R4-LOG-20260812-010 — Phase 7 DATA vNext Implementation and G7 Closure

- **Phase:** 7
- **Operator:** GPT-5.6 Sol under delegated routine technical/governance approval; local physical gate executed by project owner
- **Date:** 2026-08-12
- **Implementation merge:** `81d9c547d3610e2cfb12a5927a7a78b5693430c2`
- **Local G7 evidence commit:** `5bd9c19eb46cd804b34ac0c2cd598767f10c7fad`
- **Dataset:** `sentinel-r4-vnext-v1` / export schema `v2` / graph schema `v9`
- **Semantic population:** 22,493 contracts / 224,930 contract×class rows / 1,007 positive targets / 0 negative targets / 403 STRONG / 604 WEAK
- **Physical representation validation:** 21,657/21,657 contracts; 64,971/64,971 files; missing=0; mismatches=0; physical path not recorded
- **Representation binding digest:** `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`
- **Unsupported roles preserved:** THRESHOLD_FIT empty; CALIBRATION_FIT empty; UNTOUCHED_ACCEPTANCE empty/frozen
- **Historical artifacts changed:** NO
- **Legacy v1 semantic path changed:** NO
- **Decision:** R4-D-007 / ADR-R4-007 Accepted
- **Gate effect:** **G7 PASS.** Phase 8 is authorized to adapt the existing frozen training consumer to this exact v2 lineage and retrain without acceptance leakage.

---

### R4-LOG-20260814-011 — Phase 8 Real-Data Readiness Audit and Launch Hold

- **Phase:** 8
- **Operator:** Codex / GPT-5.6 Sol under user-authorized read-only data audit and durable documentation scope
- **Date/timezone:** 2026-08-14 Asia/Tehran
- **Repository branch/commit:** `main` at `8dc81e865a8291dc84cb2d0bf1d4661e54fc150c` before audit documentation
- **Inputs:** all 22,823 active raw manifest records; DIVE labels/folder links; all 22,493 preprocessed Solidity/metadata pairs; `sentinel-r4-vnext-v1`; all 21,657 graph/token/sidecar triplets; SolidiFI injection logs; current Phase-8 runtime/backbone
- **Commands:** `p8_audit_real_data.py`; `p8_audit_representations.py`; offline `p8_audit_token_coverage.py`; `p8_run_micro_smoke.py --train-batches 2 --selection-batches 1 --batch-size 1`; handbook static/unit/inventory validation
- **Exact source findings:** 65 compile-valid content-distinct positives removed only by address equality; one valid SmartBugs contract removed by an unconditional legacy-incompatible compiler flag; five direct SmartBugs Timestamp records physically recoverable; 120 identical normalized-code groups / 288 records, ten with multiple group IDs
- **Exact representation findings:** 21,657 / 21,657 triplets load with zero hard structural failures; 341 selected-contract mismatches; 18,491 contracts over the four-window token limit; median retained token coverage 44.3%; 612 / 852 effective loss cells over the limit
- **Exact exclusion findings:** all 836 excluded rows have no representation component; current retry yields 790 direct DIVE normalization-syntax failures, seven SolidiFI top-level normalization failures, two now-successful SmartBugs Reentrancy graphs, and 37 excluded positive target cells
- **SolidiFI injection-line check:** 275 / 276 represented files have at least one logged injection in the GNN-selected contract; selected graphs cover 4,335 / 7,203 logged injection sites
- **Runtime result:** bounded GPU smoke PASS; two optimizer steps; finite losses; 970.04 MB peak allocated; no Run12 weights and no checkpoint written
- **Protected DATA/model artifacts changed:** NO
- **New evidence created:** three read-only deterministic audit profilers and `2026-08-14_PHASE8_real_data_readiness_audit.md`
- **Register updates:** R4-R008 reopened; R4-R017 through R4-R020 opened; Phase-8 handoff/status surfaces changed to launch hold
- **Result:** **HOLD** — execution wiring is valid, but known source loss, post-compile corruption, graph-target mismatch, and token omission make the current corpus unsuitable for the full evidence-generating run
- **Gate effect:** G7 remains the valid historical binding result for `sentinel-r4-vnext-v1`; no G8 pass is claimed. Repair and publish a new versioned DATA/representation/role lineage before launching the full Phase-8 horizon.

### R4-LOG-20260815-012 — Phase-8 repository real-DATA repair implemented; local physical rebuild boundary opened

- Trigger: the 2026-08-14 real-data readiness audit invalidated the historical v1 physical DATA/representation lineage as the full-retrain input while preserving G7 as immutable historical evidence.
- Repository-repair base: `a10fae041cc5f436b5607b6fd54fcabf63386059`.
- Repository-safe repair completed on canonical `main`:
  - lexical Solidity-safe, line-preserving repaired normalization;
  - provenance-preserving exact/normalized-code identity with address equality removed as a deletion rule;
  - deterministic parent-process source-record aggregation;
  - version-aware solc flags and compile-the-promoted-bytes ordering;
  - fail-closed application-contract graph target resolution and requested/actual assertion;
  - `[4,512]` preserved with explicit pre-subsampling token/window coverage telemetry;
  - repaired leakage grouping, source-native claims, evidence-ledger, role-freeze, DATA publication and representation-binding interfaces;
  - separate repaired-v2 ML dataset/run-binding and bounded GPU-smoke seam without weakening historical v1 guards;
  - portable source-acquisition descriptor, raw-byte verifier, local rebuild driver, repaired-lineage acceptance profiler and bounded-window experiment.
- New lineage identifiers: `sentinel-preprocessed-r4-v2`, `r4-provenance-v1`, `evidence-ledger-r4-v2`, `r4-leakage-groups-v2`, `r4-vnext-roles-v2`, `sentinel-r4-vnext-v2`, extractor `v2.2-r4-repaired`, token coverage `r4-token-coverage-v1`; graph schema remains `v9`, token shape remains `[4,512]`, model remains `four_eye_v8` / `v8.1`.
- Repository validation evidence before final governance sync: repaired focused suite reached `83 passed, 4 warnings`; frozen historical G6 validation passed. The remaining whitespace-only CI defect was corrected before final gate rerun.
- Historical v1 artifacts, G7 evidence, Run12 artifacts and old representation/preprocessing roots were not overwritten.
- Physical recovery/acceptance is intentionally **not** claimed from repository tests. Next authority is `runs/2026-08-15_PHASE8_local_data_rebuild_handoff.md`.
- Current blockers: protected raw corpus/labels, historical solc binaries, generated repaired representations/parquets, physical representation binding, repaired-lineage acceptance, long-contract evidence experiment and bounded repaired-data GPU smoke.
- Full 100-epoch training remains explicitly unauthorized pending local evidence review and a later governance re-authorization.

### R4-LOG-20260815-013 — Local repaired-DATA gate re-audit and fail-closed corrections

- **Phase:** 8
- **Repository inspected:** `main` at `433c5cd021b608d37929578102e0a4d2fa445fdb` before this correction tranche
- **Raw evidence:** 22,823/22,823 manifest records passed size and SHA-256 verification (DIVE 22,330; SmartBugs 143; SolidiFI 350)
- **Defects corrected:** intentional raw symlinks misclassified as path escapes; `--limit` builds not marked/rejected as incomplete; single-target graph rule unresolved for 4,241 files; binder did not deserialize graph bytes or prove token/sidecar parity; publication declared but did not hash-consume the evidence ledger; role freeze/acceptance/GPU smoke lacked final coverage and exact-state bindings
- **Graph decision:** file-level inheritance-leaf union; full census resolves 22,823 files / 29,556 components / 4,256 multi-component files / maximum 28 components
- **Physical smoke:** two DIVE files, including a five-component 1,595-node/2,485-edge graph, compiled/tokenized and passed strict graph/token/sidecar validation
- **Repository validation:** repaired focused suite `93 passed`; corrected raw verifier PASS on all three full manifests; frozen G6 validator PASS; handbook validator `11 passed`; `git diff --check` PASS
- **Generated production artifacts:** none; only a disposable `/tmp` representation smoke was used
- **Gate effect:** physical repaired-v2 rebuild remains pending; G8 remains open; full training remains unauthorized
- **Durable evidence:** `runs/2026-08-15_PHASE8_local_gate_reaudit_and_corrections.md`

### R4-LOG-20260815-014 — First DIVE rebuild attempt rejected; compiler constraint gate corrected

- **Phase:** 8 local repaired-v2 rebuild
- **Attempt result:** complete 22,330-record DIVE reconciliation produced 22,249 prepared records, 81 compile drops and 21,995 unique artifacts; candidate rejected before claims/grouping
- **Drop audit:** 57 no-pragma sources were rejected without compilation; `<0.6.0` and one flattened adjacent comparator expression selected zero compilers; remaining compile failures retained explicit attempted versions/errors
- **Correction:** deterministic Solidity constraint evaluation for exact/comparator/adjacent/caret/tilde/OR clauses; no-pragma sources try installed compilers newest-first and bind the successful version
- **Real recovery checks:** no-pragma `10824.sol` -> solc 0.4.26; `11811.sol` `<0.6.0` -> solc 0.5.17; `1115.sol` flattened adjacent constraints -> solc 0.6.12
- **Validation:** repaired focused suite `98 passed`; frozen G6 validator PASS; handbook static validator `145 passed`; `git diff --check` PASS
- **Gate effect:** first DIVE generated root is rejected/archived; full DIVE preprocessing must restart from a fresh root; claims/grouping/representations remain blocked until the corrected full rebuild completes

### R4-LOG-20260815-015 — Second DIVE rebuild attempt rejected; partial pragmas corrected

- **Attempt result:** 22,304 prepared / 26 drops / 22,050 artifacts, recovering 55 of the first attempt's 81 drops
- **Residual gate finding:** four drops had zero attempted compilers; every one used valid shorthand `pragma solidity ^0.8`
- **Correction:** one/two/three-component constraint versions are normalized and evaluated under exact/comparator/caret/tilde semantics
- **Real recovery check:** DIVE `15105.sol` with `^0.8` prepared successfully using solc 0.8.35
- **Validation:** repaired focused suite `99 passed`; `git diff --check` PASS
- **Gate effect:** second output rejected/archived; final full DIVE preprocessing restarts from a fresh root before any downstream stage
