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
