# `data_module/tests/` — Test Architecture Overview

> **Status: ~94 tests across 8 sub-packages + 3 top-level integration tests.** Mirrors the `sentinel_data/` source layout 1:1. Run with `poetry run pytest tests/ -v` from `data_module/`.

## 1. Purpose

This directory is the **single test entry point** for the `sentinel-data` package. The test layout **mirrors the source layout 1:1** — every sub-package of `sentinel_data/` has a corresponding `test_<subpackage>/` directory here. This makes it easy to find the test for any given module: the test for `sentinel_data/labeling/merger.py` lives at `tests/test_labeling/test_merger.py`.

The tests are organized into three layers:

1. **Unit tests** (per sub-package) — test individual classes and functions in isolation
2. **Integration tests** (top-level) — end-to-end on real corpora (SolidiFI 283 contracts, DIVE 22,073 contracts)
3. **Skeleton / smoke test** — basic import check that catches wiring regressions

All tests use **pytest** with a `conftest.py` at the top level that mirrors the `cli.py:62-68` `sys.path` bootstrap pattern.

## 2. Source map

```
data_module/tests/
├── __init__.py
├── conftest.py                          # sys.path bootstrap + shared fixtures
├── test_skeleton.py                     # import smoke test
│
├── test_analysis/                       # analysis tools coverage
│   ├── __init__.py
│   └── test_analysis.py
│
├── test_ingestion/
│   ├── __init__.py
│   ├── test_connector.py                # base + registry + find_sol_files include/exclude
│   ├── test_manifest.py                 # SHA-256 verification + append-only semantics
│   └── test_label_folderize.py          # flat-vs-folderized layout
│
├── test_labeling/
│   ├── __init__.py
│   ├── test_parser_solidifi.py          # folder extraction, single-label write
│   ├── test_parser_dive.py              # multi-folder index, multi-label write
│   ├── test_merger.py                   # single-source, multi-source, conflict resolution, 99% flag
│   ├── test_gate.py                     # every criterion pass/fail, blocking vs warn-only
│   ├── test_crosswalk_solidifi.py       # crosswalk YAML structure validation
│   ├── test_crosswalk_dive.py           # crosswalk YAML structure validation
│   └── test_taxonomy.py                 # class order / count / string equality regression guard
│
├── test_preprocessing/
│   ├── __init__.py
│   ├── test_pipeline.py                 # full 5-step pipeline on synthetic Solidity
│   └── test_retry_failed.py             # --retry-failed merge semantics
│
├── test_representation/
│   ├── test_13_issue_preservation.py    # 13 bug-fix regression tests
│   ├── test_byte_identical_regression.py # schema-dim gate + old-is-new guard
│   ├── test_emits_fixture.py            # EMITS edge type fixture test
│   ├── test_orchestrator.py             # full orchestrator flow with synthetic Solidity
│   ├── test_solidifi_fixes.py           # A-1, A-2, A-3 fix regression tests
│   └── test_thin_adapter.py             # is equality between sentinel_data.X and ml.X
│   # NOTE: no __init__.py — pytest still discovers via rootdir
│
├── test_registry/
│   ├── __init__.py
│   └── test_catalog.py                  # 6-table CRUD + retirement + YAML mirror + diff + changelog
│
├── test_splitting/
│   ├── __init__.py
│   └── test_splitters.py                # 4 strategies + dedup_enforcer + NonVuln cap + manifest
│
├── test_verification/
│   ├── __init__.py
│   ├── test_class_auditor.py            # co-occurrence matrix, threshold flagging
│   ├── test_semantic_checker.py         # per-class PASS/FAIL/NOT_EXTRACTABLE
│   ├── test_tool_validator.py           # Slither agreement, NO_DETECTOR skip
│   ├── test_fp_estimator.py             # stratified sampling, FP rate math
│   ├── test_negative_checker.py         # hit rate, status thresholds
│   ├── test_gate.py                     # every verdict path, hard fail list
│   ├── test_report_generator.py         # markdown structure
│   ├── test_patterns.py                 # every YAML parses, every class has a YAML
│   ├── test_probe_dataset.py            # 40+2 per class, BCCC KEEP filter
│   ├── test_cli_verify.py               # CLI argument parsing, --skip flags
│   ├── test_bccc_regression.py          # the BCCC Phase 5 regression test
│   └── test_smartbugs_recall.py         # recall on SmartBugs Curated (≥ 70% on major classes)
│
├── test_export/
│   ├── __init__.py
│   ├── test_chunker.py                  # chunk_export orchestration, manifest generation
│   ├── test_export.py                   # SentinelDatasetExport consumer API, hash verification
│   ├── test_graph_token_writer.py       # graph + token shard writing
│   ├── test_label_writer.py             # labels.parquet schema and content
│   └── test_metadata_writer.py          # metadata.parquet schema and enrichment
│
├── test_integration_solidifi.py         # full end-to-end on SolidiFI (283 contracts)
└── test_integration_dive.py             # full end-to-end on DIVE (22,073 contracts)
```

**Sub-total: 34 test files** across 10 directories + 3 top-level files.

## 3. Test counts per stage (per MEMORY.md, 2026-06-11)

| Stage | New tests added | Cumulative | Notes |
|-------|-----------------|------------|-------|
| Stage 0 (skeleton) | 27 | 27/27 | Initial smoke + schema constants |
| Stage 1 (ingestion + preprocessing) | 35 | 65/65 | Includes 3-level dedup tests |
| Stage 2 (representation) | 20 | 85/85 | Includes thin-adapter + byte-identical regression |
| Stage 3 (labeling) | 15 | 100/100 — actually 80/80 per MEMORY.md | Includes crosswalk + taxonomy |
| Stage 4 (verification) | 11 | 91/91 | Includes BCCC regression + SmartBugs recall |
| **Total (unit + skeleton)** | — | **~94** | (count varies; re-run `pytest --collect-only` for exact number) |
| Top-level integration | 2 | +2 | `test_integration_solidifi.py` + `test_integration_dive.py` |

> **Discrepancy note:** MEMORY.md lists "100/100" for Stage 3 cumulative, but the actual line for Stage 4 says "91/91" — so Stage 3 ended at 80/80 (the 11 added in Stage 4 brings it to 91). The "100" may have been a typo or a planned-but-not-shipped count.

## 4. The most important tests

### `tests/test_representation/test_byte_identical_regression.py` — Stage 2 gate

> **This is the merge gate for any extractor change.** It enforces two guarantees:
> 1. **Schema-dim gate**: `x.shape[-1] == NODE_FEATURE_DIM == 12` (NOT 11) — prevents the silent shape mismatch that would silently break training
> 2. **Old-is-new guard**: the bytes output by `sentinel_data.representation.graph_extractor.extract_contract_graph` are byte-identical to the bytes output by `ml.src.preprocessing.graph_extractor.extract_contract_graph` (same object via thin-adapter)

The Stage 2 integration test report (`docs/training/stage_0_2_integration_test_2026-06-11.md`) is the proof that the thin-adapter port produced identical output to the old `ml/src/preprocessing/` path on 3 corpora (SolidiFI 97.5%, DIVE 97.2%, DeFiHackLabs 60.9% — the DeFiHackLabs gap is an import issue, not a byte-difference).

### `tests/test_verification/test_bccc_regression.py` — Stage 4 gate

> **This is the proof that the v2 verification module would have caught the BCCC failure mode.**

The test runs the new `verification/` module on the legacy BCCC corpus and compares the output `verification_report.md` to the Phase 5 report (frozen in `docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/`):

- Per-class drop counts must match within ±0.5%
- Per-class gate verdicts must match exactly

If this test passes, the verification module is provably equivalent to the 14-day manual investigation that fixed BCCC. If it fails, the v2 module disagrees with the Phase 5 ground truth on some class — a bug to fix before Stage 5 launches.

### `tests/test_verification/test_smartbugs_recall.py` — Stage 4 functional gate

Per-class recall ≥ 70% on SmartBugs Curated (143 real contracts). The 70% threshold is the "the model is actually learning patterns" bar; below 70% means we're back to BCCC-level noise.

### `tests/test_representation/test_13_issue_preservation.py` — 13 bug-fix regression

A single file with 13 test cases — one per previously-fixed bug in `ml/src/preprocessing/graph_extractor.py`. The test ensures that the thin-adapter port doesn't re-introduce any of them. Critical: A-9 (now-keyword), A-15 (RETURN_TO for terminal calls), A-20, A-34, A-38 — all 8 "fixed bugs" listed in MEMORY.md should be in here (along with 5 others).

### `tests/test_labeling/test_gate.py` — Stage 3 Go/No-Go gate

Every criterion path:

- `total_contracts_min` pass / fail
- `per_class_positive_min_major` (Reentrancy, DoS, IntegerUO) pass / fail — BLOCKING
- `per_class_positive_min_minor` (7 others) pass / fail — warn-only
- `call_to_unknown_min` < 300 → human-review NOTE (not a fail)
- Empty merged dir → all fail
- All blocking failures → `gate_passed=False`

### `tests/test_integration_dive.py` — end-to-end on 22K contracts

The biggest integration test. Runs the full pipeline (ingest → preprocess → represent → label → verify → split → register → analyze) on the DIVE corpus (22,073 contracts). Validates:

- DIVE folderize: 22,073 files in 9 class folders
- Preprocess: ~22K successes, 0 compile failures (DIVE is pre-cleaned)
- Represent: ~22K graphs + tokens, content-addressed cache hits on re-run
- Label: 22K merged labels with multi-label (DIVE is the only multi-label source in the v2 corpus)
- Verify: Slither runs on all 22K (cached after first run)
- Split: 70/15/15 with dedup_enforcer + NonVulnerable cap (3:1)
- Register: catalog.db has the registered version + the manifest hash
- Analyze: complexity_proxy_risk.md has 0 HIGH-RISK pairs (DIVE is the cleanest source)

### `tests/test_integration_solidifi.py` — end-to-end on 283 contracts

Smaller integration test. SolidiFI is the T0 (injection-verified) source, so the gate should pass with **all 10 classes VERIFIED**. The test asserts:

- `verification.gate.verdits[<class>].verdict == Verdict.VERIFIED` for every class with SolidiFI positives
- NonVulnerable cap respects 3:1 ratio
- Manifest is reproducible (same seed → byte-identical split_manifest.json)

## 5. Key concepts

### The `conftest.py` sys.path bootstrap

`tests/conftest.py` mirrors the `cli.py:62-68` pattern:

```python
# conftest.py (paraphrased)
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_ML_ROOT   = _REPO_ROOT / "ml"
for _p in (_REPO_ROOT, _ML_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
```

This is required because the `representation` subpackage uses the **thin-adapter pattern** — it re-exports the graph and tokenizer code from `ml/src/preprocessing/` and `ml/src/data_extraction/`. Without these paths on `sys.path`, the re-exports fail with `ModuleNotFoundError` whenever pytest is invoked from outside the repo root.

This is one of **two parallel strategies** for the same problem:

- `cli.py:62-68` — for the production CLI (must work without pytest)
- `tests/conftest.py` — for the test suite (must work without the CLI)

### The fixture pattern (if any)

Most tests are **self-contained** — they create their own `tmp_path`-scoped fixtures, write synthetic Solidity, and clean up. There are no shared `data/` fixtures (those would require a clean DIVE / SolidiFI clone in CI). The integration tests (`test_integration_*`) do use real data — they need a DIVE clone in `data/raw/dive/` to run.

### The Stage 0–4 progression

The test counts grew with each stage (see §3 above). The progression:

1. **Stage 0**: 27 tests — just the skeleton (`test_skeleton.py` + a few `tests/test_*.py` import tests)
2. **Stage 1**: +35 — full preprocessing pipeline tests (pipeline.py, compiler.py, flattener.py, deduplicator.py, normalizer.py, segmenter.py, parallel.py)
3. **Stage 2**: +20 — thin-adapter regression + byte-identical + orchestrator + 13 issue preservation + A-1/A-2/A-3 fixes
4. **Stage 3**: +15 — parsers + merger + gate + crosswalks + taxonomy
5. **Stage 4**: +11 — 5 sub-checkers + gate + report + patterns + probe + CLI + BCCC regression + SmartBugs recall

### The "no `__init__.py`" wrinkle

`tests/test_representation/` has no `__init__.py` (verified via `find`). Pytest still discovers the tests via the rootdir pattern, but this is **inconsistent** with the other 7 test packages which all have `__init__.py`. Adding the missing `__init__.py` would be a 1-line fix; until then, the test_representation package relies on pytest's rootdir mechanism.

## 6. Pipeline interactions

| Component | Direction | What |
|-----------|-----------|------|
| `tests/conftest.py` | → | `sys.path` bootstrap (mirrors `cli.py:62-68`) |
| `pyproject.toml` | → | Defines the `poetry` venv + pytest config |
| `sentinel_data` source | ← | Imports the modules under test |
| `ml/` source | ← | Imported by the `representation` thin-adapter (needs `conftest.py` sys.path) |
| `data/raw/`, `data/preprocessed/`, `data/representations/`, `data/labels/`, `data/splits/` | ↔ | Integration tests read/write these; unit tests use `tmp_path` |

## 7. Commands

### Run all tests

```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/ -v
```

### Run only unit tests (skip the slow integration tests)

```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/ -v --ignore=tests/test_integration_dive.py --ignore=tests/test_integration_solidifi.py
```

### Run a single sub-package's tests

```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_verification/ -v
```

### Run a single test file

```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_verification/test_bccc_regression.py -v
```

### Run a single test by name

```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_verification/test_bccc_regression.py -v -k "test_class_auditor_matches_phase5"
```

### Collect (count) tests

```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/ --collect-only -q
```

### Run the BCCC regression specifically (the most important gate)

```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_verification/test_bccc_regression.py -v
```

This is the test that proves the v2 verification module would have caught the BCCC failure mode. **Run this before any Run 11 launch.**

### Run the byte-identical regression (the most important Stage 2 gate)

```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_representation/test_byte_identical_regression.py -v
```

This is the test that proves the thin-adapter port is byte-identical to the old `ml/src/preprocessing/` path. **Run this before any extractor change.**

## 8. See also

- **Per-subpackage READMEs** for what each module does (and what's tested): see the 9 sub-package READMEs in `sentinel_data/`
- **The package-level overview**: `sentinel_data/README.md`
- **The test gate per stage**: the 9 sub-package READMEs each have a §"Tests" section listing the relevant test files
- **Stage 2 integration test report**: `docs/training/stage_0_2_integration_test_2026-06-11.md`
- **BCCC Phase 5 retrospective** (the failure we're guarding against): `docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/`
- **The v2 data module build context**: `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` §"Sentinel v2 Data Module Build"
- **Run 11 launch plan**: targets 2026-08-18; all BCCC regression + byte-identical + SmartBugs recall tests must pass before launch
- **The conftest pattern**: parallels `cli.py:62-68`; both are required because the thin-adapter re-export needs `ml/` on `sys.path`
