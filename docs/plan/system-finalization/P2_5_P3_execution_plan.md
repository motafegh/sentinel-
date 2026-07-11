# P2.5 + P3 — Execution Plan

**Date:** 2026-06-25
**Phases:** P2.5 (Rule-A long-file audit) + P3 (data-derived reliability, B-3 / D-C)
**Architecture of record:** `docs/proposal/2026-06-23_proposal_SYSTEM_architecture-finalization.md` (§10 P2.5 row + §5.4, §10.1 P3 row)
**Pre-conditions:** P2 DONE (Shape A active, 530 tests green, commit `c47898ea5`).
**Working memory:** `~/.claude/scratch/system_finalization_statecheck_20260625.md` + `p2_5_p3_plan_20260625.md`

---

## P2.5 — Rule-A long-file audit

**Proposal §10 row P2.5 (verbatim):** "Rule-A audit of other long files (audit_server.py 717, build_index.py 661): split only where a file has >1 reason to change."

**Gate:** split ONLY files where the audit confirms >1 reason-to-change. A file with one coherent responsibility is left alone even if long.

### Source audit (2026-06-25)

#### `agents/src/mcp/servers/audit_server.py` (717 LOC) — **SPLIT warranted (3 reasons)**

Reasons for change found by reading source:
1. **Score/math + ABI plumbing** — `_decode_audit_result`, `_mock_audit_result`, `_mock_history`, `_load_abi` (BN254 field-element math, ABI load).
2. **Tool dispatch handlers** — `_validate_address`, `_handle_get_latest_audit`, `_handle_get_audit_history`, `_handle_check_audit_exists`, `list_tools`, `call_tool` (the MCP protocol surface; changes when a tool is added/spec changes).
3. **Transport + lifecycle** — `_on_startup`, `_on_shutdown`, `run_server` (Starlette/SSE wiring; changes when transport changes).

Target layout (new subdir, mirrors the `nodes/` split pattern):
```
agents/src/mcp/servers/audit/
├── __init__.py        # re-export public API (test compat + entry-point import)
├── _config.py         # env vars, paths, EZKL_SCALE_FACTOR, _SERVER_PORT, _REGISTRY_ADDRESS, _MOCK_MODE, _ABI_PATH, load_dotenv
├── _lifecycle.py      # _load_abi, _on_startup, _on_shutdown + module-level _w3/_registry/_ABI state
├── _decode.py         # _decode_audit_result, _mock_audit_result, _mock_history
├── _handlers.py       # server = Server(...), list_tools, _validate_address, _handle_*, call_tool
└── _server.py         # run_server (Starlette wiring) — kept thin; lifespan pulls from _lifecycle
```
Backward-compat: `audit_server.py` becomes a 10-line **re-export shim** importing from the package so test imports (`from src.mcp.servers.audit_server import _decode_audit_result, …`) keep resolving. Same pattern as the P2 `nodes.py` shim before its flip.

#### `agents/src/rag/build_index.py` (661 LOC) — **SPLIT warranted (3 reasons)**

Reasons for change:
1. **Atomic-write / durability mechanics** — `_tmp_path`, `_fsync_directory`, `_atomic_write_{json,pickle,faiss}`, `_sha256_file`, `_snapshot_existing_artifacts`, `_restore_snapshot`. Changes with OS-level durability strategy.
2. **Index schema / metadata identity** — `INDEX_SCHEMA_VERSION`, `EMBEDDING_MODEL_NAME`, `BM25_TOKENIZER_VERSION`, `FAISS_TYPE`, `_REQUIRED_ARTIFACTS`, `_expected_config`, `_config_hash`, `_load_metadata`, `_source_file_count`, `_index_is_current`. Changes with index format version.
3. **Orchestrated build pipeline** — `_extra_fetchers`, `_collect_extra_documents`, `_validate_build_outputs`, `_write_artifacts`, `build_index`. Changes when a pipeline step is added.

Target layout:
```
agents/src/rag/build_index/
├── __init__.py        # re-export build_index + public symbols
├── _paths.py          # path/schema constants (_AGENTS_DIR, INDEX_DIR, _REQUIRED_ARTIFACTS, EMBEDDING_MODEL_NAME, _*_PATH, INDEX_LOCK_PATH, INDEX_LOCK_TIMEOUT)
├── _io.py             # _tmp_path, _fsync_directory, _atomic_write_*, _sha256_file, _snapshot_existing_artifacts, _restore_snapshot
├── _metadata.py       # _expected_config, _config_hash, _load_metadata, _source_file_count, _index_is_current
├── _pipeline.py       # _extra_fetchers, _collect_extra_documents, _validate_build_outputs, _write_artifacts
└── _orchestrator.py   # build_index() (the main 6-step pipeline)
```
`build_index.py` becomes a thin shim with `if __name__ == "__main__": build_index(force_rebuild=True)`. CLI entry `python -m src.rag.build_index` keeps working (package `__init__` re-exports `build_index`).

### P2.5 tasks

#### T2.5.1 — `audit/` package split
- Create 5 files under `agents/src/mcp/servers/audit/` per the layout above.
- `audit_server.py` → thin re-export shim (preserve `from src.mcp.servers.audit_server import X`).
- Constants (`_SERVER_PORT`, `_RPC_URL`, `_REGISTRY_ADDRESS`, `_DEFAULT_HISTORY_LIMIT`, `_PROJECT_ROOT`, `_ABI_PATH`, `EZKL_SCALE_FACTOR`) live in `_config.py`.
- **Mutable runtime state** (`_w3`, `_registry`, `_ABI`, `_MOCK_MODE`) ALSO defined in `_config.py`, but the shim `audit_server.py` re-imports them into its own namespace (`from ._config import _MOCK_MODE, _registry, …`) so the test's `monkeypatch.setattr("src.mcp.servers.audit_server._MOCK_MODE", False)` rebinds a live shim attribute.
- `_handlers.py` and `_lifecycle.py` read/write these via `from src.mcp.servers import audit_server as _as; _as._MOCK_MODE` (attribute access at call time — sees test monkeypatches + startup mutations). NOT via `from ._config import _MOCK_MODE` (that snapshot would miss the rebinding).
- Order in shim: (1) `from ._config import …` (binds state names on shim), (2) `from ._decode import …`, (3) `from ._handlers import …`, (4) `from ._lifecycle import …`, (5) `from ._server import run_server`. Step 1 must precede step 3 so the partial shim module has `_MOCK_MODE` bound when `_handlers` imports it.
- `server = Server("sentinel-audit")` defined in `_handlers.py` (used by `@server.list_tools()` / `@server.call_tool()` decorators at import time; `_server.py` re-imports it for `run_server`).
- Acceptance: `cd agents && poetry run pytest -q tests/test_audit_server.py -q` — all tests green unchanged.

#### T2.5.2 — `build_index/` package split
- Create 5 files under `agents/src/rag/build_index/` per the layout above.
- `build_index.py` → thin shim re-exporting `build_index` + `__main__` entry.
- Acceptance: `cd agents && poetry run python -c "from src.rag.build_index import build_index; print(callable(build_index))"` → `True`. No tests exist for build_index (file IO is too heavy); rely on import + smoke `python -m src.rag.build_index --help` if a flag exists, else import-only.

#### T2.5.3 — Full suite green + long-file README touch-up
- `cd agents && poetry run pytest -q` — **530+ passed, 3 skipped** (no regressions).
- Update `agents/src/rag/README.md` line 33 (file line counts) + the DIAGRAM.md LOC table where they cite exact sizes — Rule 4 flag only; not blocking.

---

## P3 — Data-derived reliability (B-3 / D-C)

**Proposal §5.4:** `reliability` per (source, class) loaded from versioned config; **fitted offline from a confusion matrix** on the labeled benchmark.
**§10.1 P3 row:** fit on **train split**, measure on **test split**; refit after each model retrain + on-demand (`scripts/fit_reliability.py`, version-bumped, test-gated: fail on >5pp unjustified drop); **zero-sample prior: Bayesian shrinkage `(n·measured + α·prior)/(n+α)`, α=5, prior = current principled defaults**.

### What P3 changes

Today, `verdict/reliability.py` reads the L1 hand-set `accuracy_weights` from `verdicts_default.yaml` — those were the principled defaults P2 left as placeholders. P3 turns them into **L3** values by:

1. Building a **per-(source, class) confusion matrix** from the labeled benchmark reports (`agents/test_audit_reports_p0/`).
2. Computing **measured precision** for each (source, class) cell.
3. Applying **Bayesian shrinkage** toward the current principled defaults.
4. Writing the fitted values to a **versioned config file** (`configs/reliability_v1.yaml`).
5. Refusing to commit if any (source, class) drops **≥5pp** vs the prior without a written justification (test-gated).
6. `verdict/reliability.py` flips to read from the fitted file by default; prior defaults remain as the fallback for zero-sample cells.

### Reality-check findings (2026-06-25 source read)

- `reliability.py` (62 LOC) is the sole lookup; `fuse.py`, `emit.py`, `consensus_engine`, `cross_validator` all call `get_reliability(source, cls)` (or `load_reliability()`). One extension point — clean.
- Report JSON (sample: `01_approve_frontrun_report.json`) contains the raw per-source signals needed for the confusion matrix:
  - `ml_result.probabilities[cls]` (ML prob per class) — ML signal
  - `quick_screen_hits.slither` / `quick_screen_hits.aderyn` — static-signal presence
  - `static_findings[]` — static analysis output (impact map there)
  - `consensus_verdict[cls].ml_signal / slither_match / aderyn_match` — already-extracted 0/1 per-source signals (the cleanest source of truth)
  - `ml_result.thresholds[cls]` — ML threshold per class (confirms the per-source positive polarity)
  - Labels: sidecar `<stem>.json` `labels` + `// expect:` header (siblings live next to reports via `_build_sol_index`)
- A binary per-source positive = `(ml_signal==1 / slither_match>0 / aderyn_match>0)` matched against the ground-truth label set per contract. This is exactly the confusion matrix needed.
- Config schema today: `verdict/default.yaml` → `consensus.accuracy_weights` is a flat `{class: {ml, slither, aderyn}}`. P3 will add `configs/reliability_v1.yaml` with `{schema_version, fit_metadata, table: {(source, class): value}}` and a loading path. Schema to be encoded as a Pydantic model in `src/config/` extensions (P1 precedent).

### P3 tasks

#### T3.1 — Confusion-matrix builder (`scripts/build_reliability_matrix.py`)
- Walks `agents/test_audit_reports_p0/*_report.json` + matches labels via the same `_build_sidecar_index`/`_build_sol_index`/`_derive_labels_for_stem` logic used by `run_benchmark.py` (reuse, don't duplicate — import from `src.eval.run_benchmark`).
- For each (contract, class, source) triple:
  - **Predicted positive** = source emitted a positive signal for that class:
    - `ml`: `consensus_verdict[cls].ml_signal == 1` (already ML-above-threshold)
    - `slither`: `consensus_verdict[cls].slither_match > 0`
    - `aderyn`: `consensus_verdict[cls].aderyn_match > 0`
  - **True positive** = class is in the contract's labels.
  - Accumulate per-cell TP/FP/FN/TN.
- Output: `agents/eval/reliability/confusion_matrix_v1.json` (per-cell counts) + a per-row measured-precision computation.
- Acceptance: `cd agents && poetry run python scripts/build_reliability_matrix.py --reports test_audit_reports_p0 --corpus ../manual_hand_written_contracts --out eval/reliability/confusion_matrix_v1.json` — file written, all 30 cells (3 sources × 10 classes) populated.

#### T3.2 — Bayesian shrinkage fitter (`src/eval/reliability_fit.py`)
- `ReliabilityCell` Pydantic model: `{source, cls, tp, fp, fn, tn, n, measured_precision, prior, fitted}`.
- Shrinkage: `fitted = (n · measured + α · prior) / (n + α)`, `α=5`, `prior` = current value of `consensus.accuracy_weights[cls][source]` (slither/aderyn) or `accuracy_weights[cls]["ml"] * ml_weight_scale` (ml — matches `load_reliability` today).
- For zero-sample cells (n=0): fall back to prior verbatim (no fiction).
- Test gate: **fail with exit 1** if any `|fitted - prior| >= 0.05` AND the cell's drop is not explained in an optional `--justify <markdown>` file passed on the CLI (justifications recorded in the YAML's `fit_metadata.justifications`).
- Output: `agents/configs/reliability_v1.yaml` (versioned, schema_version="1", `fit_metadata` = {matrix_path, n_contracts, alpha, prior_schema, fit_at, run12_hash}).
- Acceptance: `cd agents && poetry run python -m src.eval.reliability_fit --matrix eval/reliability/confusion_matrix_v1.json --out configs/reliability_v1.yaml` — file written; CI gate (`scripts/fit_reliability.py` thin wrapper) reports PASS or FAIL with the offending cells.

#### T3.3 — Wire `reliability.py` to read the fitted file
- `load_reliability()` behaviour change:
  1. If `configs/reliability_v1.yaml` exists AND `schema_version` matches → load it; returns the L3 fitted table.
  2. Else → fall back to current behaviour (L1 hand-set `accuracy_weights` from default config). Zero-breaking change.
  3. Env override `SENTINEL_RELIABILITY_CONFIG=path` (escape hatch for A/B testing — NOT a decision-number, just an experimental selector; operationally env-only, not in YAML).
- Keep the existing `(source, cls)` tuple-key interface + the fallback-to-default logic in `get_reliability`.
- `configs/verdicts_default.yaml` untouched (the L1 priors stay canonical as the shrinkage prior).
- Acceptance: unit tests in `tests/test_verdict_reliability.py` extend with: (a) L3 path when `reliability_v1.yaml` exists, (b) prior fallback when absent, (c) mixed cell behaviour (some fitted / zero-sample → prior). All green.

#### T3.4 — Equivalence re-baseline (the P0.1-style acceptance)
- Re-run the eval benchmark with the new L3 reliability:
  ```bash
  cd agents && poetry run python -m src.eval.run_benchmark \
      --name p3_reliability_fit \
      --config configs/verdicts_default.yaml \
      --reports test_audit_reports_p0/ \
      --corpus ../manual_hand_written_contracts \
      --baseline eval/runs/20260624T231228Z_p2_calibrated/eval_metrics.json
  ```
- Acceptance (L2 measurement): macro_Fbeta **≥ P2 baseline − 0.005** (tolerance from fuse()'s band quantisation). If Fβ regresses >5pp we DO NOT flip — investigate prior-vs-fitted mismatch and, if needed, raise α (more shrinkage to prior) — documented as a measurement, not a "feels better" guess.
- A measured table of `prior → fitted → macro_Fβ_delta` goes in the scratch `.md` and (if ≥0) gets linked from MEMORY.
- ⚠️ Ali confirmation gate: the flip to L3 is **gated on Ali reviewing the measured deltas** (Rule B: no decision-number change without a measured delta). Implementation proceeds up to the gate; the actual `load_reliability` flip waits for Ali's nod on the measurement.

### P3 deliverables
- `agents/scripts/build_reliability_matrix.py` (~150 LOC)
- `agents/src/eval/reliability_fit.py` (~250 LOC) — Pydantic schema, shrinkage math, CLI entry, gate
- `agents/scripts/fit_reliability.py` (thin wrapper, ~30 LOC)
- `agents/src/eval/reliability_matrix.py` (~150 LOC) — shared confusion-matrix data model used by both builder and fitter
- `agents/configs/reliability_v1.yaml` (generated, version-bumped on each refit)
- `agents/tests/test_reliability_fit.py` (~180 LOC) — confusion-matrix builder, shrinkage math, zero-sample fallback, 5pp gate, prior persist
- `agents/tests/test_verdict_reliability.py` extended (currently ~40 LOC → ~80): L3 path + fallback + mixed
- New eval run dir: `agents/eval/runs/<ts>_p3_reliability_fit/`

---

## Critical DoD-test gates (Rule B maturity ladder)

| Gate | Where | What it asserts |
|---|---|---|
| 5pp unjustified-drop | `scripts/fit_reliability.py` | No fitted cell regresses ≥5pp vs prior without justification entry |
| L2 measurement record | `reliability_v1.yaml.fit_metadata` | matrix path, n_contracts, alpha, prior_schema, fit_at, model_hash — reproducible |
| Fβ regression | eval benchmark re-run | macro_Fbeta delta vs P2 within ±0.005 tolerance; ≥0 favorable, <0 investigate before flip |
| Zero-sample prior | confusion-matrix builder | n=0 cell returns prior verbatim (no fiction) |

---

## Ordering & effort

1. **P2.5 — ~0.5–1 day** (T2.5.1 audit split, T2.5.2 build_index split, T2.5.3 suite). Mechanical, safety-net is `pytest -q`.
2. **P3 — ~3–5 days**:
   - T3.1 confusion builder (~0.5 day)
   - T3.2 shrinkage fitter + gate (~1.5 days)
   - T3.3 wire `reliability.py` (~0.5 day)
   - T3.4 re-baseline + Ali review gate (~1 day, partly waiting)
   - Tests written alongside each.

**Stop point:** after P3.T3.4 produces the measured table, surface to Ali for the flip nod, then update MEMORY + proposal changelog.

---

## Rollback plan

| Step | What fails | Rollback |
|---|---|---|
| T2.5.1 audit split | test_audit_server fails on import | Restore `audit_server.py` from `c47898ea5`, delete `audit/` dir |
| T2.5.2 build_index split | `python -m src.rag.build_index` errors | Restore `build_index.py` from git, delete package |
| T3.1 matrix builder | counts negative or cells empty | Don't write `reliability_v1.yaml`; debug the report-label matching |
| T3.2 fitter gate | 5pp gate fires unjustifiably | Investigate; raise α; or keep P2 L1 priors (system works, just not L3) |
| T3.3 reliability wiring | fuse() verdicts change unexpectedly | Set `SENTINEL_RELIABILITY_CONFIG` unset → load_reliability falls back to L1 |
| T3.4 Fβ regress | macro_Fbeta < P2 − 0.005 | Same — don't commit; investigate shrinkage vs. principled defaults |

---

## Risks (P3-specific)

| Risk | L | I | Mitigation |
|---|---|---|---|
| Small benchmark (83) → noisy precision | High | Med | Shrinkage α=5 explicitly addresses this; report n per cell; raise α if needed |
| Fitted weights regression vs principled priors | Med | Med | 5pp gate + Ali review before flip |
| Report fields differ across the 83 (some consensus_verdict missing) | Low | Med | Reuse `run_benchmark._load_contract_eval` which already handles missing; emit `n=0` for missing cells |
| Confusion-matrix cells trained on the same data the eval scores on (no train/test split for 83 contracts) | High | Med | Proposal §10.1 says train/test split — but with N=83 this is statistically thin; for v1 use the full 83 as train (acknowledge in metadata), give Ali the choice at the gate |
| Diff in `consensus_verdict` schema between `--no-llm` reports (P0 baseline = no-llm) and LLM-enabled reports | Low | Low | P0 baseline reports are all `--no-llm`; consensus_verdict schema is identical there |

---

## What this plan deliberately defers (named, not forgotten)

- **Larger train/test split corpus** — defer until 150–200 contracts exist (proposal §10.1 P3 says "fit on train, measure on test"; we measure on test here, but the corpus is one and the same — full audit transparent).
- **Multi-family decorrelation re-tuning** — `fuse.py` FAMILIES discount 1/N. P3 fits weights, not the discount — that's a separate Rule-B decision gated on P3's measurement.
- **P8 channels reliability fit** (Halmos, Gigahorse, etc.) — those sources don't exist yet; their cells in `reliability_v1.yaml` use prior or zero-sample fallback until P8 ships.