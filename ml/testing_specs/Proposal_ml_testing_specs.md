
***

```markdown
# Proposal: ml/testing_specs/ — Modular Validation Spec Suite
**Status:** DRAFT — Pre-writing proposal  
**Date:** 2026-06-11  
**Author:** Sentinel project  
**Next action:** Review → approve → write files sequentially

---

## Why We Are Doing This

### Problem with the current single-file spec
The existing `ML-validation-instructions-Spec.md` is a single ~300-line file with
correct intent but several structural problems:

1. **Context bloat** — Claude loads the entire file even when only one section is
   relevant. This causes drift and selective attention failures on long files.
2. **Missing coverage** — Scripts in `ml/scripts/` (smoke suite, promote_model,
   vram_gate, check_contamination, validate_graph_dataset, interpretability suite)
   are not referenced. The spec describes procedures that already have implementations.
3. **No meta-validation** — The spec tells Claude what to validate but never requires
   Claude to verify that its own validation was performed correctly.
4. **Mixed concerns** — Universal rules, per-task procedures, and checklists are
   mixed in one file with no clear loading strategy.
5. **Static structure** — Adding a new procedure requires editing one large file,
   making changes hard to review and track.

### Why a folder structure solves this
- Each file is scoped to one concern — Claude loads only what the current task needs
- `README.md` acts as a routing map — Claude reads it first to decide which spec to load
- `00_rules.md` contains universal invariants — always loaded, never duplicated
- Files stay under ~80 lines — short enough to be followed completely without drift
- Version control granularity — PRs show exactly which procedure changed and why
- Extensible — adding J, K, L, M requires no edits to existing files

---

## Guiding Principles for All 14 Files

These principles govern every file in the suite. Any writing that violates them
must be revised before the file is considered complete.

1. **No hardcoding** — No schema constants, checkpoint paths, metric numbers,
   run names, or file paths that change over time. Every value is delegated
   to its source file with a read instruction.

2. **No dictating outcomes** — Procedures describe *how* to check, never *what
   the answer should be*. The spec must remain valid even when values change.

3. **Procedures, not knowledge** — If a fact belongs in `MEMORY.md`, `graph_schema.py`,
   or a docs file, it must not be duplicated here. The spec only says where to read it.

4. **Dynamic and adoptable** — Every procedure must work for Run 10, Run 11, v2 data,
   and future schema versions without modification. If a procedure would need editing
   when a run number changes, it is written wrong.

5. **Completeness without length** — Every required step must be present. But if a
   step can be expressed in one line with a file reference, it must not be expanded
   into a paragraph.

6. **Self-validating** — Every multi-step procedure must end with a completion
   attestation written to a file. Claude must not consider a procedure done until
   the attestation exists.

---

## File Map and Rationale

### `README.md` — Routing Index
**Purpose:** The only file Claude always reads first. Maps tasks to spec files.
**Must contain:**
- One-line description of each spec file
- Task-to-file routing table (e.g., "launching a run → F", "contract wrong → B+C")
- How to load spec files (reference from CLAUDE.md via @import)
- When to load multiple files simultaneously
- What `00_rules.md` is and why it always applies

**Why needed:** Without a routing index, Claude must scan all files or guess which
one applies. The README eliminates that ambiguity entirely.

---

### `00_rules.md` — Universal Invariants
**Purpose:** Rules that apply to every procedure in every other file.
**Must contain:**
- **Rule 0 — Read Before Claiming:** Never assert a value without reading its source
- **Rule 1 — Where Things Live:** Lookup table mapping information types to source files
- **Rule 2 — Validate Your Validation:** The meta-validation requirement:
  - Layer 1: Gate assertions — every validation result must be written to a file,
    not just printed. Unverifiable results must be marked UNVERIFIED, not assumed pass.
  - Layer 2: Cross-check — any reported metric must be confirmed from two independent
    sources where possible. Disagreement between sources is itself a finding.
  - Layer 3: Completion attestation — at the end of any multi-step procedure, produce
    a structured summary (steps completed, skipped, unverified, findings, written-to path)
    and append it to the relevant doc before the session ends.
- **Rule 3 — No Floating Findings:** Any finding, decision, or open question discovered
  during a procedure must be written to a named doc immediately, not deferred.

**Why needed:** These rules apply universally. Duplicating them in every section file
would create drift. A single source that all other files implicitly inherit is cleaner.

---

### `A_benchmark_runs.md` — Benchmark Procedures
**Purpose:** How to run and interpret external benchmark evaluations correctly.
**Must contain:**
- A.1: Contamination check — use `check_contamination.py`, not manual hashing
- A.2: Environment check — venv, env vars, schema match, open-issues scan,
  `vram_gate_test.py` for inference runs
- A.3: Metric selection — Top-K for OOD/injected benchmarks, FP probe requirement,
  documentation of what results prove and don't prove

**Why needed:** Benchmarks have produced misleading results before due to skipped
contamination checks and wrong metric choices (threshold-based on OOD contracts).

---

### `B_contract_deep_dive.md` — Per-Contract Debugging
**Purpose:** Ordered procedure for diagnosing individual mispredicted contracts.
**Must contain:**
- B.1: Dataset-level integrity first via `validate_graph_dataset.py`, then per-contract
  node/edge checks, AST integrity via `ast_extractor.py` when relevant,
  re-extraction via `reextract_graphs.py` if graph is suspect
- B.2: Per-eye breakdown via `return_aux=True`; reference `diag_per_eye_solidifi.py`
  as canonical implementation; read `sentinel_model.py` for eye-to-output relationship
- B.3: Contract source reading — pragma version, comment density, interface/abstract
  body injection, library verbosity, vulnerability location vs. schema coverage
- B.4: Classify and record — match to known failure mode or create new finding entry

**Why needed:** Per-contract diagnosis without this order wastes effort. Graph integrity
must be confirmed before model output is interpretable.

---

### `C_diagnostic_checks.md` — Training Log and Model Behaviour
**Purpose:** How to read training logs and verify model behaviour post-run.
**Must contain:**
- C.1: Log analysis — read `training_logger.py` for exact field names first;
  JK entropy interpretation (compute max from layer count, not from memory);
  gnn_share interpretation (read warmup schedule from source);
  per-class F1 convergence (compare against split label counts)
- C.2: Model behaviour — SmartBugs Curated smoke inference, FP probe,
  two-step threshold verification (`calibrate_temperature.py` then `tune_threshold.py`),
  `predictor.py` threshold load path verification, known open bug check

**Why needed:** Training logs have been misread before due to assumed field names
and assumed warmup behaviour. Source-first reading prevents this.

---

### `D_smoke_preflight.md` — Smoke Tests and Pre-Flight Gates
**Purpose:** When and how to run smoke tests, VRAM gate, and compile validation.
**Must contain:**
- D.1: Smoke suite triggers — run `smoke/run_all.py` before: schema changes,
  checkpoint loads, new training runs, any `ml/src/` modification.
  Read `_common.py` first. Each `smoke_fix<N>` targets a specific historical bug —
  read its header before interpreting its failure.
- D.2: VRAM pre-flight — run `vram_gate_test.py`; document headroom; do not proceed
  on failure. Record the VRAM result in the run pre-flight log.
- D.3: Compile validation — run `compile_smoke_test.py` after any checkpoint load;
  verify `._orig_mod.` strip is applied; read the script before running.

**Why needed:** Smoke tests exist precisely because these checks were skipped before
and caused bugs. The spec formalizes them as non-negotiable gates.

---

### `E_preprocessing_consistency.md` — Train/Eval Path Alignment
**Purpose:** Ensuring training and evaluation use identical preprocessing.
**Must contain:**
- Reading order: `tokenizer.py` + `windowed_tokenizer.py` → `dual_path_dataset.py`
  → `inference/preprocess.py`
- What to confirm identical: window count, truncation behaviour, comment stripping
- Re-tokenization trigger conditions — when `retokenize_windowed.py` is required
- Re-extraction trigger conditions — when `reextract_graphs.py` is required
  (schema change, Slither version change, new edge types, new node features)
- Destructive side-effect warning for both scripts

**Why needed:** Train/eval preprocessing mismatch has already caused silent metric
corruption. The `--relabel-timestamp` omission from v10 is a documented example.

---

### `F_new_run_checklist.md` — Pre/Post Training Run Gates
**Purpose:** Complete checklist for launching and closing a training run correctly.
**Must contain:**
- F.1 Pre-launch: data checks (label file + verification status, split zero-overlap,
  cache version match, token cache settings, inference cache invalidation);
  schema checks (read `graph_schema.py`, confirm encoding output);
  checkpoint/resume checks (read `trainer.py` resume logic, stale name check,
  schema version recorded); pre-flight gates (vram, smoke, open-issues, loss
  function confirmed from `trainer.py` instantiation)
- F.2 Promotion gate: what `promote_model.py` checks; pre-conditions
  (contamination done, behaviour checks passed, calibration files in place);
  post-promotion documentation requirements
- F.3 Post-run: diagnostic checks (Section C), contamination check before benchmark,
  `MEMORY.md` Training History update, findings externalized before session close

**Why needed:** Multiple past run failures trace to skipped pre-launch steps.
Formalizing this as a gated checklist prevents recurrence.

---

### `G_drift_detection.md` — Inference Drift Procedures
**Purpose:** How to interpret and respond to drift signals from `drift_detector.py`.
**Must contain:**
- Read `drift_detector.py` before interpreting any signal — understand detection method
- Triage procedure: check for shared characteristics among drifting contracts
  (Solidity version, size, edge distribution, label class)
- Graph integrity check on drifting contracts via `validate_graph_dataset.py`
- Distinguishing input drift vs. model drift and required response for each
- Documentation requirements: drift type, affected contracts, graph stats,
  mapping to known failure mode or new finding

**Why needed:** `drift_detector.py` exists in the repo but was completely absent
from the original spec. Unhandled drift signals will produce silent false confidence.

---

### `H_api_validation.md` — API Endpoint Validation
**Purpose:** How to validate `api.py` responses and end-to-end inference correctness.
**Must contain:**
- Read `api.py` response schema before writing assertions — do not assume field names
- Threshold source verification — trace load path in `predictor.py`
- Drift flag surfacing check
- Known-positive + known-negative contract round-trip test
- Cache invalidation verification on checkpoint change

**Why needed:** `api.py` is a production surface. An endpoint that silently uses
fallback thresholds or stale cache produces incorrect security classifications
without any visible error.

---

### `I_interpretability.md` — Interpretability Experiment Procedures
**Purpose:** How to run and record interpretability experiments correctly.
**Must contain:**
- Pre-read requirement: `utils.py` + `EXPERIMENT_INDEX.md` before any experiment
- Experiment family map (A, B, E, L, S, val_finding) with one-line purpose per family
- `run_training_ablation.sh` warning — read before running, triggers multiple runs
- Recording requirement: findings to `EXPERIMENT_INDEX.md` immediately, not deferred
- How to distinguish a new finding from confirmation of an existing one

**Why needed:** 25 interpretability scripts exist with no procedure for when to run
which family or how to record results. Without this, experiments are run ad hoc
and findings accumulate only in conversation history.

---

### `J_schema_migration.md` — Schema Version Change Protocol
**Purpose:** Safe procedure for advancing the graph schema version.
**Must contain:**
- What constitutes a schema change (new feature dim, new node type, new edge type,
  changed encoding logic)
- Complete downstream impact checklist: which source files reference schema constants
  (read `graph_schema.py` imports to find all consumers)
- Required re-extraction and re-tokenization (reference Section E trigger conditions)
- Schema-dim gate test requirement before training
- How to version the transition: checkpoint must record old and new schema version,
  cache files must be rebuilt not patched
- ADR requirement: every schema change must have a corresponding ADR entry

**Why needed:** v8→v9 transition caused multiple silent failures (wrong BinaryType
names, missing `in_unchecked_block`, wrong `_MAX_TYPE_ID`). A formal migration
protocol prevents recurrence for future transitions.

---

### `K_label_validation.md` — Label File Validation Before Training
**Purpose:** How to validate a new label file before it enters any training run.
**Must contain:**
- Per-class distribution check — read class counts from split index, not the label file alone
- Co-occurrence matrix check — the 99% DoS↔Reentrancy co-occurrence trap is a known
  failure mode; flag any class pair above a co-occurrence threshold (read threshold
  from v2 config, not hardcoded here)
- Confidence gate check — confirm all labels meet the VERIFIED/PROVISIONAL/BEST-EFFORT
  gate defined in the v2 data module proposal
- NonVulnerable ratio check — confirm the 3:1 cap rule from v2 config is respected
- CallToUnknown merge rule check — if count < threshold, confirm merge decision is
  documented (read threshold from v2 config)
- Go/No-Go gate — read `pipeline.min_viable_corpus` from config before declaring
  label file ready; do not proceed if gate fails

**Why needed:** BCCC label noise (89% Reentrancy FP, 86.9% CallToUnknown FP) was the
root cause of the F1 ceiling in Runs 1–9. v2 is built to fix this — label validation
is the enforcement mechanism.

---

### `L_reproducibility.md` — Exact Reproducibility Protocol
**Purpose:** How to confirm a result is genuinely reproducible, not run-specific.
**Must contain:**
- RNG seed pinning check — confirm seed is recorded in run config and checkpoint
- `TRANSFORMERS_OFFLINE=1` confirmation — required for deterministic tokenizer behaviour
- DVC data hash verification — confirm tracked data files match their recorded hashes
- Poetry lockfile drift check — confirm `poetry.lock` hasn't changed between runs
- Two-source metric confirmation — F1 from MLflow AND from JSONL epoch summary must agree
- Tolerance definition — read tolerance from run config or docs, not hardcoded here
- What to do when results are non-reproducible: document delta, investigate seed
  or data drift before drawing conclusions

**Why needed:** "Run 10 is better than Run 9" is only a defensible claim if both
runs are reproducible. Without this protocol, metric comparisons are anecdotal.

---

### `M_session_handoff.md` — Session Close and Handoff Protocol
**Purpose:** What must be written and verified before a session ends.
**Must contain:**
- Mandatory writes before close:
  - `MEMORY.md` Current State section updated to reflect actual state
  - Any completed run added to Training History table
  - Any open bugs or findings appended to the relevant audit doc
  - Any decisions recorded as ADR entries
  - Any open questions added to an `## Open Questions` section in the relevant doc
- Verification before close:
  - Confirm no findings exist only in the conversation
  - Confirm no plan exists only in the conversation
  - Confirm the next session has enough context in written docs to resume without
    re-reading this conversation
- Handoff summary format — brief structured block written to `MEMORY.md`:
  ```
  ## Session Handoff — <date>
  Completed: [what was done]
  Open: [what was not finished]
  Next session must read: [specific files]
  Blockers: [anything preventing progress]
  ```

**Why needed:** This is the enforcement mechanism for Rule 4 (no floating findings)
and the harness engineering principle of treating session context as ephemeral.
Without a formal handoff, every new session re-discovers context that was already known.

---

## Writing Order and Dependencies

Write files in this order — later files reference earlier ones:

1. `00_rules.md` — no dependencies
2. `README.md` — depends on knowing all file names (write last among routing files)
3. `D_smoke_preflight.md` — referenced by F, A, J
4. `E_preprocessing_consistency.md` — referenced by F, J
5. `A_benchmark_runs.md` — references D, E
6. `B_contract_deep_dive.md` — references D, E
7. `C_diagnostic_checks.md` — references B
8. `F_new_run_checklist.md` — references A, C, D, E
9. `G_drift_detection.md` — standalone
10. `H_api_validation.md` — references C
11. `I_interpretability.md` — references C
12. `J_schema_migration.md` — references E, F
13. `K_label_validation.md` — references F
14. `L_reproducibility.md` — references F, C
15. `M_session_handoff.md` — references all (write last)

---

## Success Criteria

The suite is complete when:
- [ ] Every script in `ml/scripts/` is referenced by at least one spec file
- [ ] Every source file in `ml/src/` that has validation significance is named
      in at least one spec file with a specific read instruction
- [ ] No spec file contains a hardcoded value that would require editing when
      a run number, schema version, or checkpoint path changes
- [ ] `00_rules.md` Rule 2 (meta-validation) is implementable for every
      multi-step procedure in every section file
- [ ] A new team member (or fresh Claude session) could execute any procedure
      correctly using only the spec files and the source code — no prior context needed
- [ ] `README.md` routing table covers every realistic task a developer would
      arrive with
```

***

This proposal is the thing you save right now before we write a single file. A few things to note about it:

- The **writing order section** is the most important part — `00_rules.md` first, `M_session_handoff.md` last, because M references everything
- The **success criteria checklist** at the bottom is a self-check you can run after all 14 files are written — it makes the completion condition explicit and objective
- The **rationale paragraph in each file section** captures the specific past incident or failure mode that justifies that file's existence — so future Claude sessions understand *why* each file exists, not just *what* it contains

