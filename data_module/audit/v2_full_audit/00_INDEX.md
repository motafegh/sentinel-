# SENTINEL v2 Data Module — Full Audit (Pre-Run-11)

**Audit owner:** SENTINEL data engineering
**Audit window:** 2026-06-12 onward
**Goal:** ship a single document that gates Run 11 promotion ("v2 is ready to train")
**Scope:** all 9 subpackages of `sentinel_data/` (12,569 LOC, 77 .py files) + CLI + config + DVC + Docker + tests

---

## Why this audit exists

The existing audit folder (`data_module/audit/00..08`) covers **Stages 0–4 only**, with a deep dive on Stages 0–2. **Stages 5, 6, 7, the seam swap, and end-to-end integration have never been audited.** The prior audit also left 23 FAIL and 42 WARN items un-tracked (compliance status unknown).

The README at `data_module/README.md:4` claims "Stage 7 ⏳ STUB" but the `export/` folder has 7 files with ~1,000 LOC — the README is out of date with the code. The README also flags a **known two-taxonomy divergence** (representation order vs labeling order) with no ADR and no regression test guarding it.

Run 11 launch: **2026-08-18.** This audit must finish in time to fix any blockers it finds.

---

## Phases

| Phase | Doc | Goal | Sessions | Output file |
|---|---|---|---|---|
| A | [01](plans/01_phase_a_foundation_recon.md) | Foundation recon — know what's shipping vs claimed | 1 | `v2_full_audit/01_phase_a_foundation_recon.md` |
| B | [02](plans/02_phase_b_prior_audit_compliance.md) | Re-check 23 FAIL / 42 WARN from prior audit | 1 | `v2_full_audit/02_phase_b_prior_audit_compliance.md` |
| C1 | [03](plans/03_phase_c1_stages_5_6_audit.md) | Deep source review of Stages 5 (splitting+registry) + 6 (analysis) | 1 | `v2_full_audit/03_phase_c1_stages_5_6_audit.md` |
| C2 | [04](plans/04_phase_c2_stage_7_export_audit.md) | Deep source review of Stage 7 (export + seam swap) | 1 | `v2_full_audit/04_phase_c2_stage_7_export_audit.md` |
| D | [05](plans/05_phase_d_integration_and_taxonomy.md) | Integration, CLI/DVC/Docker round-trip, two-taxonomy question | 1 | `v2_full_audit/05_phase_d_integration_and_taxonomy.md` |
| E | [06](plans/06_phase_e_master_report.md) | Master consolidated report — Run 11 readiness verdict | 1 | `v2_full_audit/06_FINAL_master_report.md` |

**Total: 6 sessions, 6 audit docs, 1 master report.**

---

## How to execute a phase

1. Open the phase's plan doc (e.g. `plans/01_phase_a_foundation_recon.md`).
2. Follow the **Tasks** section in order. Each task has a verifiable exit condition.
3. Update the phase's `## Status` block at the bottom of the plan doc as you go (LIVE / DONE / BLOCKED).
4. Produce the phase's output file at the path listed in the table above.
5. Mark the phase DONE in the master todo list before starting the next phase.
6. **Do not start Phase B until Phase A is DONE.** Phases B and C1 can be reordered if time-pressed; phases D and E are sequential.

---

## Reference material (load once at session start)

| Doc | Why |
|---|---|
| `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` | v2 facts, schema, training history |
| `data_module/README.md` | Module-level overview, taxonomy divergence note |
| `docs/proposal/Data_Module_Proposals/00_INDEX.md` | Stage plan index |
| `data_module/audit/00_EXECUTIVE_SUMMARY.md` | Prior audit P0–P3 list (Phase B consumes this) |
| `data_module/audit/08_stages_0_2_deep_audit.md` | Prior deep-audit template (Phases C1, C2 follow this format) |

---

## Hostile Verification Protocol (applies to all phases)

Per the session-0 directive, every check in this audit is **hostile**: assume broken, prove correct, only flag what is provably broken. The protocol is:

### Three rules

1. **No trust without evidence.** Every claim (PASS / FIXED / "works") needs ONE of:
   - A `file:line` quote of the relevant code
   - A command output (test result, grep match, ls output)
   - A test name + its pass/fail status

2. **No limit to the prior audit.** The prior audit found 23 FAIL + 42 WARN + ~136 PASS. The prior audit's PASS verdicts are NOT trusted. **Re-verify ~10% of the PASS items** in each phase (random sample + high-impact ones). The prior audit's "not in scope" items (Stages 5/6/7, seam swap, full integration) are now in scope.

3. **No invented problems.** Every finding must be reproducible:
   - A bug needs a command that triggers it
   - A discrepancy needs both sides quoted (the README claim AND the source reality)
   - A "what if" is not a finding. A "reproduced" is.

### Beyond the prior audit — what to look for that wasn't listed

| Category | Examples of "not in prior audit" |
|---|---|
| **Implicit contracts** | Does the catalog *find* a registered dataset, or just *store* it? Does the splitter *use* the merged label schema, or assume its own? Does the export's manifest *match* the parquet schemas? |
| **Cross-thing consistency** | Class order in `graph_schema.py` vs `taxonomy.yaml` vs trainer vs checkpoint. Hash algorithm in `manifest.py` vs `catalog.py` vs `lineage_tracker.py`. Confidence tier names across all 5 places they're referenced. |
| **End-to-end behavior** | Does `sentinel-data run` work on the ScaBench fixture? Does a v2 export load in a v9-trained model? Does `dvc repro` complete? |
| **Test quality** | Do the tests *test* the right thing? Are the fixtures real or mocked? Are the assertions specific (assertEqual) or weak (assertTrue)? |
| **Reproducibility** | Same input → same output? (Random seeds, timestamps, dict iteration order.) |
| **Error messages** | When something fails, does the error tell you *what* to fix? Or is it a bare `Exception: failed`? |
| **Dormant code** | The `_backup_pre_seam_swap_2026-06-12_graph_schema.py` file — is it bit-identical to live? Is it dead code, a reference, or a rollback point? |
| **Hidden assumptions** | Hardcoded path components (`/home/motafeq`, `/tmp`, `__file__` relativity). Magic numbers without context. Hardcoded class indices (the magic `9` for NonVulnerable). |
| **Missing tests** | The prior audit lists 8 fixed bugs with regression tests. What about the other 28 (A1–A38 minus 8 fixed minus 2 open)? Are they tested or just claimed-fixed? |
| **CLI vs code drift** | `cli.py` subparser declares a flag the handler doesn't read. Handler reads an arg the subparser doesn't declare. `--help` text doesn't match what the code does. |

### How to record a finding

Every finding in any phase output gets a `FINDING-X:N` ID. The minimum body is:

```markdown
- **FINDING-X:N** [SEVERITY]
  - **File:** `path/to/file.py:LINE` (or `FINDING-CROSS:N` if cross-file)
  - **Evidence:** exact quote from the file, or command + output
  - **Reproduction:** exact command(s) to trigger
  - **Impact:** what breaks (concrete, not speculative)
  - **Fix sketch:** 1-3 lines, not a full design
```

A finding without all 5 fields is rejected. The hostile-lens check: would another auditor reading just this finding agree it's a real problem? If not, it's not a finding.

---

## Out of scope (per session-0 agreement)

- `ml/src/inference/predictor.py` (F8/F10) — `sentinel-ml` fix, owned by seam swap
- The 5 critical-path source corpora themselves (SolidiFI, DIVE, DeFiHackLabs, SmartBugs Curated, Web3Bugs) — only their parsers and ingestion paths are in scope
- The active Run 9 training pipeline — read-only observation only
- Re-training Run 9 or launching Run 10/11 — the audit gates Run 11, doesn't launch it

---

## Glossary

- **v9 schema** — `FEATURE_SCHEMA_VERSION="v9"`, `NODE_FEATURE_DIM=12`, `NUM_NODE_TYPES=14`, `NUM_EDGE_TYPES=12`, `_MAX_TYPE_ID=13.0`. Defined in `ml/src/preprocessing/graph_schema.py:161,175,218` and re-exported by the thin-adapter in `sentinel_data/representation/graph_schema.py:131-134`.
- **Two taxonomies** — `representation/graph_schema.py:73-84` (9 vuln + NonVulnerable, no TOD) vs `labeling/schema/taxonomy.yaml` (9 vuln + UnusedReturn, no NonVulnerable). See `data_module/README.md:218-223` for the full divergence write-up.
- **Thin-adapter** — `sentinel_data/representation/{graph_extractor,graph_schema,tokenizer}.py` re-export from `ml/src/preprocessing/` and `ml/src/data_extraction/`. Bug fixes apply once and propagate. Seam swap (Stage 7) deletes the wrappers and rebinds the import.
- **Seam swap** — Stage 7 task 7.8. Deletes `ml/src/preprocessing/{graph_extractor,graph_schema}.py` and `ml/src/datasets/dual_path_dataset.py`; `sentinel-ml` reads from the v2 export instead. `_backup_pre_seam_swap_2026-06-12_graph_schema.py` in `representation/` is the pre-swap snapshot.
- **Critical-path sources** (Run 11): DeFiHackLabs, SolidiFI, DIVE, SmartBugs Curated, Web3Bugs + DISL as NonVulnerable pool (3:1 cap).
