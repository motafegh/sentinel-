# Phase C1 — Stages 5 + 6 Deep Source Audit

**Sessions:** 1 (~3h)
**Output:** `v2_full_audit/03_phase_c1_stages_5_6_audit.md`
**Status:** PENDING (gated on Phase B DONE)

> **Apply the [Hostile Verification Protocol](../../00_INDEX.md#hostile-verification-protocol-applies-to-all-phases).** The prior audit never touched Stages 5/6, so there are no prior verdicts to re-verify — every check is fresh. For each public function, try to actually call it (or read the test that does). Cross-check implicit contracts (does the catalog actually *find* a dataset; does the splitter actually *use* the merger schema).

---

## Goal

Apply the `08_stages_0_2_deep_audit.md` format (PASS / WARN / FAIL with file:line evidence) to the two subpackages the prior audit never touched: Stage 5 (splitting + registry) and Stage 6 (analysis). The output style should match the prior audit's tables so the master report (Phase E) can merge them.

---

## What this phase touches

### Stage 5 — splitting (4 files, 883 LOC)

| File | LOC | What |
|---|---|---|
| `splitting/splitters.py` | 441 | 4 splitter strategies + Contract/Splits/SplitMetadata dataclasses |
| `splitting/dedup_enforcer.py` | 116 | BCCC-failure pattern fix; cross-split dedup |
| `splitting/leakage_auditor.py` | 163 | Post-split text-shingle audit |
| `splitting/nonvulnerable_cap.py` | 163 | 3:1 NonVulnerable cap (friend review) |

### Stage 5 — registry (3 files, 800 LOC)

| File | LOC | What |
|---|---|---|
| `registry/catalog.py` | 541 | SQLite + YAML mirror; 4+2 tables; `hash_artifact` / `verify_artifact` |
| `registry/dataset_diff.py` | 161 | Per-class metric projection |
| `registry/lineage_tracker.py` | 98 | DAG of transformations |

### Stage 6 — analysis (6 files, 1,344 LOC)

| File | LOC | What |
|---|---|---|
| `analysis/balance_viz.py` | 134 | Per-class / per-source / per-tier counts |
| `analysis/cooccurrence.py` | 187 | Directed + conditional matrices |
| `analysis/drift_monitor.py` | 298 | KS test for feature + label distribution |
| `analysis/feature_dist.py` | 436 | The Run-9-failure catcher; `complexity_proxy_risk.md` output |
| `analysis/overlap_detector.py` | 267 | Pairwise Jaccard between source datasets |
| `analysis/probe_dataset.py` | 22 | Re-export only (verify it matches `verification/probe_dataset.py`) |

---

## Tasks (ordered, each with exit condition)

### C1.1 — Read the Stage 5 + 6 plan docs

```bash
wsl -- bash -c 'cat ~/projects/sentinel/docs/proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md'
```

Build a check-list of every **design decision (D-X.Y)** and **exit criterion**. The audit will compare code to this list.

**Exit condition:** design-decision inventory table in working notes. ~15-20 rows.

### C1.2 — Stage 5 splitting: per-file deep review

For each of the 4 files, do a structured review:

1. **Read top to bottom** (or in 100-line chunks if file is >300 lines)
2. **For each public function / class**, verify:
   - Signature matches what `cli.py` calls (cross-check the dispatcher)
   - Docstring exists for public symbols
   - Error handling: are exceptions specific, or broad `except Exception`?
   - Path handling: uses `pathlib.Path` (not `os.path` strings)?
   - Determinism: same inputs → same outputs (no `datetime.now()` without seed)?
3. **For each file**, look for the prior-audit-style issues:
   - Mutable default args
   - Hardcoded paths
   - `datetime.utcnow()` deprecated
   - Broad `except Exception`
   - Path traversal (using `Path.resolve()` against a base)
   - Unused imports
   - Dead code

**Output per file:** a markdown table of checks, e.g.:

```markdown
### splitters.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| 4 splitter functions exported | PASS | 89-441 | … |
| Contract dataclass has all D-5.1 fields | PASS | 12-30 | … |
| Deterministic with seed | … | … | … |
```

**Exit condition:** 4 review tables, one per file.

### C1.3 — Stage 5 registry: per-file deep review

Same format as C1.2. Special focus areas:

- **`catalog.py`**: SQLite schema (4+2 tables per MEMORY); does the YAML mirror stay in sync? Is there a `hash_artifact` that handles the SHA-256 correctly? Does `verify_artifact` actually re-hash or just trust the manifest?
- **`dataset_diff.py`**: per-class metric projection — what metrics? Are they the ones Run 9's `epoch_summary.jsonl` produces?
- **`lineage_tracker.py`**: DAG of transformations — is it actually a DAG (acyclic) or a tree? Does it survive a `dvc repro` rerun?

**Exit condition:** 3 review tables.

### C1.4 — Stage 6 analysis: per-file deep review

Same format. Special focus areas:

- **`feature_dist.py`** (the 436-LOC one) — this is the "Run 9 failure catcher" and produces `complexity_proxy_risk.md`. Verify:
  - It computes the per-class mean of `feat[6]` (LoC) — the Run 9 complexity proxy
  - The `complexity_proxy_risk.md` threshold (1.5σ per Stage 7 plan D-7.6) is implemented
  - The output is a single green/yellow/red signal, not 20 individual metrics
- **`cooccurrence.py`** — verify it computes directed P(A|B) not just undirected correlation
- **`drift_monitor.py`** — KS test: is it 2-sample? Per-feature or aggregate? What threshold flags drift?
- **`probe_dataset.py`** (the 22-LOC re-export) — verify it re-exports from `verification/probe_dataset.py` (NOT a divergent copy)

**Exit condition:** 6 review tables.

### C1.5 — Cross-cutting checks for Stages 5 + 6

- [ ] **CLI dispatch** — does `cli.py:_handle_split` call `splitters.py` with the same signature? Does `cli.py:_handle_register` actually invoke `catalog.register()`?
- [ ] **DVC DAG** — are `split` and `register` stages in `dvc.yaml`?
- [ ] **Test coverage** — `tests/test_splitting/test_splitters.py` exists (per `tests/` listing). Does it cover all 4 splitter strategies? Are there tests for `tests/test_registry/`?
- [ ] **Cross-stage contracts** — does `splitters.py` consume the same merged-label schema that `merger.py` produces? (Class names, confidence_tier, source list.)
- [ ] **Schema version** — does the registry stamp the export with `FEATURE_SCHEMA_VERSION` and refuse to load a mismatched one?

**Exit condition:** cross-cutting section in output doc with 5-row check table.

### C1.6 — Severity assignment

For every FAIL found, assign:
- **CRITICAL** — silently breaks Run 11 (e.g. wrong class order, schema gate missing)
- **HIGH** — produces wrong results in some cases (e.g. dedup misses near-dupes)
- **MEDIUM** — works but has a bug that will surface in edge cases (e.g. empty input not handled)
- **LOW** — cosmetic / docs / dead code

For every WARN, assign:
- **HIGH-WARN** — P1-equivalent (blocks Stage 7)
- **MED-WARN** — P2-equivalent (post-Run-11)
- **LOW-WARN** — P3-equivalent (nice to have)

**Exit condition:** every finding has a severity tag.

### C1.7 — Output doc structure

Author `v2_full_audit/03_phase_c1_stages_5_6_audit.md` with:

1. **Executive summary** — totals table (Stage 5 / Stage 6 / cross-cutting) with PASS / WARN / FAIL counts
2. **Design-decision compliance** — D-X.Y checklist from C1.1
3. **Stage 5 splitting** — 4 file review tables from C1.2
4. **Stage 5 registry** — 3 file review tables from C1.3
5. **Stage 6 analysis** — 6 file review tables from C1.4
6. **Cross-cutting checks** — from C1.5
7. **Findings inventory** — `FINDING-C1:N` numbered list, each with severity + file:line
8. **Run 11 blockers from C1** — items that must be fixed before Run 11
9. **Deferred-to-Phase-D items** — items that need integration testing, not just file review

---

## What this phase will NOT touch

- Stage 7 (export + seam swap) — that's Phase C2
- The two-taxonomy divergence — Phase D
- Anything in `ml/` — out of scope

---

## Required inputs

- `v2_full_audit/01_phase_a_foundation_recon.md` — test pass/fail rate, parser state
- `v2_full_audit/02_phase_b_prior_audit_compliance.md` — items still open from prior audit
- `docs/proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md` — Stage 5 plan
- `data_module/audit/08_stages_0_2_deep_audit.md` — format template

## Outputs

- `v2_full_audit/03_phase_c1_stages_5_6_audit.md` (consumed by Phase E)
- Updated todo list

---

## Exit criteria checklist

- [ ] Design-decision inventory table authored
- [ ] All 4 splitting files reviewed
- [ ] All 3 registry files reviewed
- [ ] All 6 analysis files reviewed
- [ ] Cross-cutting checks done (CLI, DVC, tests, contracts, schema)
- [ ] All findings have severity tags
- [ ] Output doc authored with all 9 sections
- [ ] All findings numbered as `FINDING-C1:N`
- [ ] Run 11 blockers from C1 identified (subset of total)
