# Phase A — Foundation Recon

**Sessions:** 1 (~2h)
**Output:** `v2_full_audit/01_phase_a_foundation_recon.md`
**Status:** LIVE

---

## Goal

Know what is actually shipping in `sentinel_data/` vs what is claimed in the README, the prior audit, and the action plans. This phase does NOT find bugs — it establishes the baseline numbers the later phases need.

---

## Beyond the prior audit — Phase A additions

The prior audit's Phase A equivalent is `08_stages_0_2_deep_audit.md` which only covered Stages 0-2. The prior audit did **not** do a foundation recon. Phase A here is the first time the full-module baseline is established. Additions to the standard Phase A tasks:

| # | Beyond-prior-audit check | Why it matters |
|---|---|---|
| A.X1 | Count `*.py` files in `sentinel_data/` and `tests/` — does the count match the README and the claim in `00_INDEX.md` (77 source, 41 test)? | A file count mismatch means drift; either code is missing or the audit is wrong. |
| A.X2 | `python -m py_compile sentinel_data/**/*.py` — do all 77 source files parse? | A parse error means the file is corrupt or unfinished. |
| A.X3 | `from sentinel_data.X import Y` for every public symbol listed in the 9 subpackage READMEs — does the import resolve? | An unresolved import means the README lies. |
| A.X4 | `git ls-files \| wc -l` vs filesystem find — is git tracking the files we think it is? | Untracked files mean commits were dropped or work happened but wasn't saved. |
| A.X5 | Does `poetry install` succeed from a clean venv? (Skip if venv already exists and works.) | A broken install means the package is unshippable. |
| A.X6 | Cross-check MEMORY's claims ("80/80 tests", "91/91 tests", "65/65 pass") against the actual pytest output. | Stale memory docs are dangerous — Run 11 plan might be based on wrong numbers. |
| A.X7 | Read the .gitignore — does it exclude the right things? Is `data/` excluded (DVC-tracked) or accidentally gitignored? | Wrong .gitignore = secrets in git OR 2.5 GB of cache in commits. |
| A.X8 | `find . -name "*.py" -path "*/sentinel_data/*" \| xargs wc -l` — does the LOC match the 12,569 claim? | LOC drift = uncommitted code or stale docs. |
| A.X9 | For each subpackage, count actual files vs README's file map. | README says 8 files in `representation/`; if there are 10, something is unannounced. |
| A.X10 | Check if there's a `data_module/docs/decisions/ADR-0008-*.md` already. | If yes, the two-taxonomy question is partially decided; don't re-debate it. |

These are folded into A.1–A.10 below as inline steps, not separate tasks.

---

## What this phase touches

| File / Path | Why |
|---|---|
| All 41 test files under `tests/` | Count pass/fail/skip |
| `sentinel_data/cli.py` (1,037 lines) | Subcommand inventory + dry-run behavior |
| `dvc.yaml` | DVC DAG vs CLI subcommand match |
| `config.yaml` | Hardcoded paths (F1) + defihacklabs enabled flag (F2) |
| 9 subpackage READMEs under `sentinel_data/*/README.md` | Doc-vs-source drift |
| `docs/decisions/*.md` | ADR coverage of 8 fixed + 2 open bugs |
| `docker/Dockerfile.data` | Build prerequisites |
| `pyproject.toml` | Dependency footprint |
| `git log --oneline -30` | Commit cadence, force-pushes, rebase history |

---

## Tasks (ordered, each with exit condition)

### A.1 — Run the full test suite

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && poetry run pytest tests/ -v 2>&1 | tee /tmp/phase_a_tests.log'
```

Record the result in a table:

| Test file | Pass | Fail | Skip | Note |
|---|---|---|---|---|
| `test_ingestion/test_connector.py` | … | … | … | |
| … (one row per test file) | | | | |

**Exit condition:** all 41 test files have a row; aggregate `X / Y passed` in the output doc. If >5 tests fail, that's a Phase A finding (not a Phase B finding) — log it as `FINDING-A:N`.

### A.2 — CLI subcommand inventory

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && poetry run sentinel-data --help && echo "---" && poetry run sentinel-data run --help && echo "---" && for cmd in ingest preprocess represent label verify split register analyze export freshness; do echo "===$cmd==="; poetry run sentinel-data $cmd --help 2>&1 | head -30; done'
```

Build a table:

| Subcommand | Claimed in README? | `cli.py` subparser? | `--help` works? | `--dry-run` works? |
|---|---|---|---|---|
| `ingest` | yes | yes | yes | … |
| `export` | yes | … | … | … |
| `freshness` | yes | … | … | … |

**Exit condition:** every subcommand has a row. If README claims a subcommand and it doesn't exist → `FINDING-A:N`. If subcommand exists but `--help` errors → `FINDING-A:N`. If `--dry-run` raises an exception → `FINDING-A:N`.

### A.3 — DVC DAG vs CLI subcommand match

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && cat dvc.yaml'
```

Compare `dvc.yaml` stages to `cli.py` `_STAGE_FN` dispatch table. Diff them:

- `dvc.yaml` stage exists but `cli.py` has no `_handle_<stage>` → DVC dead stage
- `cli.py` `_STAGE_FN` has a stage not in `dvc.yaml` → not DVC-tracked (likely intentional for export/registry/etc., but flag it)

**Exit condition:** diff table in output doc.

### A.4 — Hardcoded paths in config.yaml

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && grep -n "/home/motafeq" config.yaml || echo "no hardcoded paths found"'
```

Also check test files:

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && grep -rn "/home/motafeq\|C:\\\\\\\\Users" tests/ || echo "clean"'
```

**Exit condition:** all matches listed by file:line. If F1 (config paths) is fixed, the only matches should be the `pipeline.project_root` reference in the comment, not actual absolute paths.

### A.5 — defihacklabs / Web3Bugs / SmartBugs parser state

Per `data_module/audit/00_EXECUTIVE_SUMMARY.md:42,165` and `00_EXECUTIVE_SUMMARY.md:174`, three parsers were "not implemented":

1. **DeFiHackLabs parser** — `labeling/parsers/` currently has only `solidifi.py` and `dive.py`. Check.
2. **Web3Bugs parser** — MEMORY lists as critical-path source. Was it ever implemented? Check both `labeling/parsers/` and `ingestion/ingest.py` for any web3bugs reference.
3. **SmartBugs Curated parser** — verification has a test `test_smartbugs_recall.py` (per `tests/test_verification/`), which implies at least ingestion. Check `labeling/parsers/` for any `smartbugs_*.py`.

For each, report: **FOUND** (with file:line) or **MISSING** (becomes a Phase A critical finding).

**Exit condition:** 3-row table in output doc, one per parser.

### A.6 — Subpackage README vs source drift

For each of the 9 subpackages, open the README and the `__init__.py`. Check:

- Public API listed in README matches symbols exported in `__init__.py`
- File map in README matches files actually in the directory
- Status indicators (e.g. "STUB", "✅") are accurate

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module/sentinel_data && for d in ingestion preprocessing representation labeling verification splitting registry analysis export; do echo "===$d==="; head -50 $d/README.md; echo "---init---"; cat $d/__init__.py; echo; done'
```

**Exit condition:** per-subpackage drift table. Each drift item gets a `FINDING-A:N`.

### A.7 — ADR coverage

List all ADRs:

```bash
wsl -- bash -c 'ls -la ~/projects/sentinel/data_module/docs/decisions/ 2>/dev/null && echo "===" && ls -la ~/projects/sentinel/docs/ml/adr/ 2>/dev/null'
```

Per `data_module/README.md:264-268`, the 4 important ADRs are: 0001, 0002, 0007, and "implicit (no ADR yet)" for the two-taxonomy divergence. Check if more exist.

The 8 fixed bugs (A9, A15, A20, A34, A38, resume, def_use, return_ignored) and 2 still-open bugs (predictor tier threshold, EMITS edge / Interp-6) per MEMORY should be referenced in at least one ADR.

**Exit condition:** ADR inventory table. If the two-taxonomy divergence has no ADR, that's `FINDING-A:N` (priority HIGH — it gets deep attention in Phase D).

### A.8 — Docker image build prerequisites

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && cat docker/Dockerfile.data'
```

Verify:
- Base image is `python:3.12.1-bookworm` (per Stage 7 plan, AUDIT_PATCHES 7-P10)
- All 6 solc baseline versions are referenced
- `slither-analyzer` install path is correct
- No `pip install` without `poetry.lock` reference

**Exit condition:** Dockerfile review table. Actual `docker build` is **out of scope** for Phase A (Docker build is in Phase D, since it requires the full export working first).

### A.9 — Commit cadence + HEAD state

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && git log --oneline -30 && echo "===" && git status && echo "===" && git diff --stat HEAD~5..HEAD'
```

- Has there been a force-push? (`git reflog` shows it.)
- Are there uncommitted changes in `sentinel_data/`?
- Is the working tree clean?

**Exit condition:** state-of-HEAD section in output doc.

### A.10 — Output doc structure

Author `v2_full_audit/01_phase_a_foundation_recon.md` with:

1. **Executive summary** — 5-bullet TL;DR
2. **Test suite result** — table from A.1
3. **CLI inventory** — table from A.2
4. **DVC vs CLI diff** — table from A.3
5. **Hardcoded paths** — list from A.4
6. **Parser state** — table from A.5
7. **Subpackage README drift** — table from A.6
8. **ADR coverage** — table from A.7
9. **Dockerfile review** — table from A.8
10. **HEAD state** — output from A.9
11. **Phase A findings** — `FINDING-A:1` through `FINDING-A:N` numbered list, each with severity HIGH/MED/LOW and file:line

---

## What this phase will NOT touch

- Per-file source code review (Phases B, C1, C2)
- Actual `docker build` execution (Phase D)
- Test code review (Phase D — only counted in this phase, not reviewed)
- The two-taxonomy divergence itself (Phase D — only flagged here)

---

## Required inputs

None — this is the first phase.

## Outputs

- `v2_full_audit/01_phase_a_foundation_recon.md` (consumed by Phases B, C1, C2, D)
- Updated todo list (mark Phase A DONE before starting Phase B)

---

## Exit criteria checklist

- [ ] All 41 test files have a pass/fail row
- [ ] Every CLI subcommand has a `--help` + `--dry-run` row
- [ ] `dvc.yaml` vs `cli.py` diff is documented
- [ ] No hardcoded paths in `config.yaml` OR matches are explicitly listed
- [ ] DeFiHackLabs, Web3Bugs, SmartBugs parser state is known (FOUND/MISSING)
- [ ] All 9 subpackage READMEs reviewed for drift
- [ ] ADR inventory + taxonomy-divergence ADR gap is documented
- [ ] Dockerfile reviewed (build NOT executed)
- [ ] HEAD state captured
- [ ] Output doc authored with all 11 sections
- [ ] All findings numbered as `FINDING-A:N`
