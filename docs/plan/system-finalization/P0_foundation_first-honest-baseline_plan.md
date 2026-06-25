# P0 FOUNDATION — Execution Plan (P0.0 + P1 + P0.1)

**Date:** 2026-06-23
**Phase:** P0 FOUNDATION = P0.0 (labels + hygiene) -> P1 (config externalization) -> P0.1 (measurement loop). The first milestone: **"first honest baseline."**
**Architecture of record:** `docs/proposal/2026-06-23_proposal_SYSTEM_architecture-finalization.md` (decisions FINAL — §4, §5, §6, §10.1). This is a *how* doc; it does not re-litigate decisions. Cite the proposal by section instead of repeating reasoning.
**Reality-check source:** `~/.claude/scratch/p0_foundation_plan_20260623.md` (file-reality vs docs, written 2026-06-23).

> **Ordering (proposal §10):** P0.0 -> P1 -> P0.1. P1 is pulled ahead of P0.1 because the eval must score *named configurations* (the config schema must exist first). Behavior is preserved before capability is added (proposal §5.2 migration discipline).

---

## 1. Scope — files touched / created

All paths verified against the repo on 2026-06-23. `agents/` is the working root for the agents venv (`cd agents && poetry run ...`).

### P0.0 — Labels & hygiene
| Action | Path | Verified |
|--------|------|----------|
| Audit (read-only) | `manual_hand_written_contracts/**/*.sol` (83 files) | exists |
| Create (generated) | `manual_hand_written_contracts/<class>/<stem>.json` (83 sidecars) | none yet |
| Edit (stale docstrings) | `agents/src/llm/client.py` (:10, :151, :153) | exists |
| Edit (stale docstrings) | `agents/src/llm/README.md` (:16, :39) | exists |
| Edit (stale comments) | `agents/src/orchestration/nodes.py` (:1147, :1581) | exists |
| Edit (stale docstring) | `agents/src/eval/__init__.py` (:19-20, refs non-existent `gates`) | exists |
| Edit (stale present-tense table) | `agents/src/llm/DIAGRAM.md` (:311) | exists; keep FIX-18 changelog at :62,:216 |
| Read (record open findings) | `docs/plan/agents/2026-06-21-agents-redesign/00_FINDINGS.md`, `04_LIVE_BASELINE_FINDINGS.md` | exist |
| Create (audit tool) | `agents/scripts/audit_labels.py` | none yet |

### P1 — Config externalization
| Action | Path | Verified |
|--------|------|----------|
| Create (package) | `agents/src/config/__init__.py`, `schema.py`, `loader.py` | none yet |
| Create (default config) | `agents/configs/verdicts_default.yaml` | dir none yet |
| Edit (add dep) | `agents/pyproject.toml` (add `pyyaml`) | exists; pydantic ^2.0 present |
| Edit (wire to config) | `agents/src/orchestration/consensus.py` | exists |
| Edit (wire to config) | `agents/src/orchestration/confidence.py` | exists |
| Edit (wire to config) | `agents/src/orchestration/routing.py` | exists |
| Edit (wire to config) | `agents/src/orchestration/attribution.py` | exists |
| Edit (wire to config) | `agents/src/eval/pipeline_metrics.py` (BORDERLINE_BAND, positive_verdicts) | exists |
| Create (tests) | `agents/tests/test_config.py` | none yet |

### P0.1 — Measurement loop
| Action | Path | Verified |
|--------|------|----------|
| Edit (add F-beta) | `agents/src/eval/pipeline_metrics.py` (ClassMetrics + PipelineMetrics) | exists |
| Create (runner module) | `agents/src/eval/run_benchmark.py` | none yet |
| Create (gates module — fixes stale __init__ ref) | `agents/src/eval/gates.py` | none yet (gates inline in script today) |
| Edit (thin wrapper) | `agents/scripts/eval_benchmark.py` | exists at `agents/scripts/` (NOT `scripts/`) |
| Edit (export gates) | `agents/src/eval/__init__.py` | exists |
| Edit (compare macro_fbeta) | `agents/src/eval/regression.py` | exists |
| Create (tests) | `agents/tests/test_eval_fbeta.py`, `agents/tests/test_run_benchmark.py` | none yet |
| Create (first baseline) | `agents/eval/baselines/p0_honest_baseline.json` | dir exists |
| Create (first run) | `agents/eval/runs/<timestamp>_p0_honest/` | dir exists |

---

## 2. Pre-conditions

- **Decisions are FINAL** (proposal §4/§5/§10.1). No re-design in this phase.
- The 438-function test suite is green at start (`cd agents && poetry run pytest -q`). If not, stop and fix before P0.0.
- LM Studio + the 5 MCP servers are NOT required for P0.0/P1 (pure logic + config). P0.1's first baseline run DOES require the ML inference server (port 8010) up; LLM debate is optional for a deterministic-tier baseline (a no-LLM run is a valid first data point; proposal §6 scores the full config).
- **Inter-phase dependency:** P0.0 (honest corpus) -> P1 (config schema, because the eval scores *named configurations*) -> P0.1 (eval wired to config). Each sub-phase's acceptance gate must pass before the next starts.

---

## 3. Ordered task list

Each task is independently verifiable. Acceptance = the exact command that proves it (Rule B teeth). Rollback noted where risky.

### P0.0 — Labels & hygiene

#### T0.0.1 — Audit `// expect:` headers on all 83 contracts
- **What:** Write `agents/scripts/audit_labels.py` that walks `manual_hand_written_contracts/`, parses every `// expect:` header, and reports: (a) count per class, (b) any contract with a missing/unparseable header, (c) the inferred ground_truth split.
- **Acceptance:** `cd agents && poetry run python scripts/audit_labels.py --corpus ../manual_hand_written_contracts` prints a table with 83 rows parsed, 0 missing headers, and the per-class support counts. Exit 0.
- **Rollback:** Read-only; delete the script if unwanted.

#### T0.0.2 — Generate JSON sidecars from headers (one canonical source)
- **What:** Extend `audit_labels.py` (or a sibling `generate_sidecars.py`) to write `<stem>.json` next to each `.sol`, shape `{"labels": [...], "ground_truth": "vulnerable"|"safe", "source": "expect_header"}`. Header remains canonical; sidecar is generated (proposal §10.1 P0.0).
- **Acceptance:** (a) 83 JSON files written. (b) `Benchmark` loads the corpus via sidecars and per-class support matches T0.0.1 exactly. (c) Re-running is idempotent (no diff on a clean run).
- **Rollback:** `find ../manual_hand_written_contracts -name '*.json' -delete` (none existed before).

#### T0.0.3 — Drop unresolvable contracts (contingency)
- **What:** If T0.0.1 finds any contract whose label cannot be resolved honestly, record it in `agents/eval/corpus_exclusions.json` and exclude from the benchmark. Per proposal §10.1: "accept a lower N; the eval reports actual N."
- **Acceptance:** `corpus_exclusions.json` either is empty/absent (N=83) or lists each dropped stem with a one-line reason; `audit_labels.py` reports the post-drop N. **Expected per Ali (MEMORY): N stays 83.** This task is a no-op if all 83 are honest — that is the preferred outcome.
- **Rollback:** Remove the exclusion file.

#### T0.0.4 — Stale docstring purge (Rule 4)
- **What:** Replace present-tense `qwen3.5-9b-ud` references with the current `MODEL_STRONG = "gemma-4-e2b-it"` (verified at `llm/client.py:72`). Files: `client.py:10,151,153`; `llm/README.md:16,39`; `nodes.py:1147,1581`; `DIAGRAM.md:311`. **Keep** the FIX-18 changelog entries (`DIAGRAM.md:62,216`; `client.py:66`) — they are historical records, not present-tense claims. Also fix `eval/__init__.py:19-20` (remove the `gates` module reference — P0.1 recreates it).
- **Acceptance:** `cd agents && rg -n "qwen3.5-9b-ud" src/` returns matches ONLY inside FIX-18 changelog comments (lines containing "FIX-18" or "changed from"). No present-tense docstring/table claims remain. `poetry run pytest -q` still green.
- **Rollback:** `git checkout -- <file>` per file.

#### T0.0.5 — Read findings docs; record what's still open
- **What:** Read `00_FINDINGS.md` (#1-12) and `04_LIVE_BASELINE_FINDINGS.md` (#13-16). Record in `~/.claude/scratch/p0_foundation_plan_20260623.md` which findings are (a) resolved by shipped WS1-WS5, (b) resolved by the upcoming P2 fuse() (per proposal §5.2 — #13/14/15 are the 8-case reconciliation), (c) still open and in-scope for P0, (d) ML-side / out of agents scope (#16).
- **Acceptance:** The scratch file has a "Findings status" table with one row per finding and a disposition. No code change.
- **Rollback:** N/A (documentation).

**P0.0 gate:** T0.0.1-T0.0.5 all pass -> corpus is honest + sidecars exist + docstrings clean. Proceed to P1.

---

### P1 — Externalize decision-numbers (proposal §10.1 P1, B-2)

#### T1.1 — Add `pyyaml` dependency
- **What:** Add `pyyaml = ">=6.0"` to `agents/pyproject.toml` `[tool.poetry.dependencies]`. Run `poetry lock && poetry install`.
- **Acceptance:** `cd agents && poetry run python -c "import yaml; print(yaml.__version__)"` succeeds.
- **Rollback:** Revert pyproject + `poetry lock`.

#### T1.2 — Create the config schema (Pydantic; proposal §10.1: YAML + Pydantic, eager, fail-fast)
- **What:** Create `agents/src/config/schema.py` with Pydantic v2 models mirroring the decision-number groups (verified line refs in §4 below): `ConsensusConfig`, `ConfidenceConfig`, `RoutingConfig`, `AttributionConfig`, `EvalConfig`, and a top-level `SentinelConfig` nesting them + `schema_version: str`. Each field carries the *current* code value as its default (behavior preservation).
- **Acceptance:** `cd agents && poetry run python -c "from src.config.schema import SentinelConfig; c=SentinelConfig(); assert c.consensus.confirmed_band==0.70 and c.eval.fbeta_beta==2.0"` passes.
- **Rollback:** Delete `src/config/`.

#### T1.3 — Implement the eager loader
- **What:** Create `agents/src/config/loader.py`: `load_config(path|None=None) -> SentinelConfig` — reads YAML (default `agents/configs/verdicts_default.yaml`, overridable via `SENTINEL_CONFIG` env), validates via Pydantic (raises on invalid -> fail-fast), caches a singleton for the process (no hot-reload — proposal §10.1). `__init__.py` exposes `get_config() -> SentinelConfig`.
- **Acceptance:** (a) `get_config()` returns a `SentinelConfig`. (b) A malformed YAML raises `ValidationError`. (c) Two calls return the same object (cached). (d) `SENTINEL_CONFIG=/nonexistent poetry run python -c "from src.config import get_config"` fails loudly.
- **Rollback:** Delete `src/config/`.

#### T1.4 — Create the default YAML config
- **What:** Create `agents/configs/verdicts_default.yaml` with every decision-number from the schema, using the *current* code values (behavior preservation). Comment each as a Rule-B decision-number (L1 externalized; L2 measured in P0.1; L3 learned in P3). Include `schema_version: "1"`.
- **Acceptance:** `load_config()` reads it; `SentinelConfig(**yaml.safe_load(...))` round-trips with no validation errors; `get_config()` values match the current constants exactly.
- **Rollback:** Delete the file.

#### T1.5 — Wire `consensus.py` to config (preserve testability)
- **What:** Replace module-level constants (`ACCURACY_WEIGHTS`, `DEFAULT_WEIGHTS`, `DEFAULT_ML_WEIGHT_SCALE`, `ML_POSITIVE_THRESHOLD`, `CONFIRMED_BAND`/`LIKELY_BAND`/`DISPUTED_BAND`) with reads from `get_config().consensus` **at call time** (not import time), so monkeypatching still works (the existing `_ml_scale()` pattern at `consensus.py:74` is the template). Keep the old names as aliases for backward-compat. Remove the `os.getenv("ML_WEIGHT_SCALE")` read — decision-numbers come from YAML now (reproducibility; proposal §10.1).
- **Acceptance:** `cd agents && poetry run pytest tests/test_consensus_voting.py -q` green. New assertion: `python -c "from src.orchestration.consensus import CONFIRMED_BAND; from src.config import get_config; assert CONFIRMED_BAND==get_config().consensus.confirmed_band"`.
- **Rollback:** `git checkout -- src/orchestration/consensus.py`.

#### T1.6 — Wire `confidence.py` to config
- **What:** Same call-time pattern for `SLITHER_AGREE/DISAGREE`, `ADERYN_AGREE/DISAGREE`, `RAG_AGREE`, `RAG_RELEVANCE`. Keep names as aliases.
- **Acceptance:** `poetry run pytest tests/test_confidence_tracking.py -q` green.
- **Rollback:** `git checkout -- src/orchestration/confidence.py`.

#### T1.7 — Wire `routing.py` to config
- **What:** Same pattern for `DEEP_THRESHOLDS`, `ROUTING_RULES`, `prob_to_severity` thresholds, `OVERALL_VERDICT_RANK`. **Also externalize the 3 hardcoded constants in `compute_verdict` (lines 244-248: 0.50/0.80/0.50)** as `routing.compute_verdict_prob_cutoff` / `rag_confirmed_cutoff` / `rag_likely_cutoff` — these are decision-numbers NOT listed in proposal §4.2 (see §6 Contradictions). `CLASS_TO_DETECTORS` + `DETECTOR_TO_CLASSES` stay in code (structural mapping, not thresholds).
- **Acceptance:** `poetry run pytest tests/test_routing_phase0.py tests/test_graph_routing.py -q` green.
- **Rollback:** `git checkout -- src/orchestration/routing.py`.

#### T1.8 — Wire `attribution.py` to config
- **What:** Same pattern for `RAG_RELEVANCE_FLOOR`.
- **Acceptance:** `poetry run pytest tests/test_metric_attribution.py -q` green.
- **Rollback:** `git checkout -- src/orchestration/attribution.py`.

#### T1.9 — Wire `eval/pipeline_metrics.py` to config
- **What:** `BORDERLINE_BAND` and `DEFAULT_POSITIVE_VERDICTS` read from `get_config().eval`. Keep names as aliases.
- **Acceptance:** `poetry run pytest tests/test_eval_framework.py -q` green.
- **Rollback:** `git checkout -- src/eval/pipeline_metrics.py`.

#### T1.10 — Config tests
- **What:** Create `agents/tests/test_config.py` asserting: (a) default load matches current constants for every group; (b) invalid YAML raises; (c) env override of path works; (d) schema_version round-trips; (e) a value changed in YAML is reflected by `get_config()` after a fresh process.
- **Acceptance:** `poetry run pytest tests/test_config.py -q` green.
- **Rollback:** Delete the test file.

**P1 gate:** T1.1-T1.10 all pass + the **full suite** `cd agents && poetry run pytest -q` is green (behavior preserved: no decision-number value changed, only its source). Proceed to P0.1.

---

### P0.1 — Close the measurement loop (proposal §6, D-D)

#### T0.1.1 — Add F-beta (beta=2) to PipelineMetrics
- **What:** In `agents/src/eval/pipeline_metrics.py`: add `fbeta: float` field to `ClassMetrics` + a `beta` param to `compute()` (default from `get_config().eval.fbeta_beta` = 2.0); `fbeta = (1+beta^2)*p*r / (beta^2*p + r)`. Add `macro_fbeta` to `PipelineMetrics` (macro mean over classes with support>0, mirroring the existing `macro_f1` NaN-aware logic at `pipeline_metrics.py:242-243`). Keep `f1`/`macro_f1` for backward-compat. Add `fbeta`/`macro_fbeta` to `as_dict()` serialisation.
- **Acceptance:** `poetry run pytest tests/test_eval_fbeta.py -q` green (new test: known confusion matrix -> expected F2 value within 1e-6; macro_fbeta = mean of per-class F2).
- **Rollback:** `git checkout -- src/eval/pipeline_metrics.py`.

#### T0.1.2 — Move gates into `src/eval/gates.py` (fixes stale __init__ ref)
- **What:** Extract the 9 gate functions + `GateResult` from `agents/scripts/eval_benchmark.py` (lines 105-542) into `agents/src/eval/gates.py`. Update `eval/__init__.py` to export `GateResult` + the gate functions (now the `gates` docstring at :19-20 becomes true). The script imports them from the module.
- **Acceptance:** `poetry run pytest tests/test_eval_framework.py -q` green; `python -c "from src.eval import gate_ws1a_silent_safe_on_flagged"` succeeds.
- **Rollback:** `git checkout -- agents/scripts/eval_benchmark.py agents/src/eval/__init__.py`; delete `src/eval/gates.py`.

#### T0.1.3 — Create the runner module `src/eval/run_benchmark.py`
- **What:** Create `agents/src/eval/run_benchmark.py` with CLI `python -m src.eval.run_benchmark --name <id> --config <yaml> --reports <dir> --corpus <dir> [--baseline <json>]`. It: loads the named config (validates via the P1 loader), runs the gates + `PipelineMetrics` (now with macro_fbeta), writes `eval/runs/<timestamp>_<name>/eval_metrics.json` + `eval_report.md`, and exits nonzero on regression vs baseline. This is the proposal §10.1 runner.
- **Acceptance:** `cd agents && poetry run python -m src.eval.run_benchmark --name smoke --config configs/verdicts_default.yaml --reports eval/runs/<existing> --corpus ../manual_hand_written_contracts` produces a metrics JSON containing `macro_fbeta` and a non-empty `per_class` block; exit 0 (no baseline) or 1 (regression).
- **Rollback:** Delete `src/eval/run_benchmark.py`.

#### T0.1.4 — Make `scripts/eval_benchmark.py` a thin wrapper
- **What:** Reduce the script to: parse args, call `src.eval.run_benchmark.main()` (or a shared entry point). Remove the now-duplicated logic (gates moved in T0.1.2; metrics already delegate). Keep CLI compatibility (`--reports`, `--corpus`, `--baseline`).
- **Acceptance:** `poetry run python scripts/eval_benchmark.py --reports eval/runs/<existing> --corpus ../manual_hand_written_contracts` produces identical metrics JSON to T0.1.3 on the same inputs (diff-stable).
- **Rollback:** `git checkout -- agents/scripts/eval_benchmark.py`.

#### T0.1.5 — Extend `RegressionBaseline.compare()` to report macro_fbeta
- **What:** In `agents/src/eval/regression.py`: add `baseline_macro_fbeta`/`current_macro_fbeta` to `RegressionResult` and a `macro_fbeta` entry to `metric_deltas`. Keep `regressed` keyed on macro_f1 for now (do not change pass/fail semantics without a measured delta — Rule B); report macro_fbeta as informational so P0.1 establishes the number without gating on it yet.
- **Acceptance:** `poetry run pytest tests/test_eval_framework.py -q` green (existing regression tests); new assertion that `RegressionResult.metric_deltas` contains `macro_fbeta`.
- **Rollback:** `git checkout -- src/eval/regression.py`.

#### T0.1.6 — Run the first honest baseline (the milestone)
- **What:** Start the ML inference server (port 8010). Run the audit pipeline over the post-P0.0 corpus (83 contracts) producing `<stem>_report.json` in a fresh runs dir. Then score it: `python -m src.eval.run_benchmark --name p0_honest --config configs/verdicts_default.yaml --reports <runs_dir> --corpus ../manual_hand_written_contracts`. Save the resulting `eval_metrics.json` as `agents/eval/baselines/p0_honest_baseline.json`. A no-LLM run is acceptable for the first data point (document which).
- **Acceptance:** `p0_honest_baseline.json` exists, `contract_count` equals the post-P0.0 N (83 expected), `macro_fbeta` is a real number (not 0.0/NaN), and the per-class block has all 10 classes with support>0. Record the headline (macro_fbeta, macro_f1) in the scratch file + MEMORY.
- **Rollback:** Delete the run dir + baseline file. The old `pre_redesign.json` baseline is untouched.

#### T0.1.7 — Run-benchmark tests
- **What:** Create `agents/tests/test_run_benchmark.py`: (a) CLI parses `--name`/`--config`/`--reports`/`--corpus`; (b) invalid config path fails fast; (c) output dir is `<timestamp>_<name>`; (d) metrics JSON contains `macro_fbeta`; (e) exit code is 1 on a synthetic regression.
- **Acceptance:** `poetry run pytest tests/test_run_benchmark.py -q` green.
- **Rollback:** Delete the test file.

**P0.1 gate:** T0.1.1-T0.1.7 all pass + `cd agents && poetry run pytest -q` green + `p0_honest_baseline.json` exists with a real macro_fbeta. **This is the "first honest baseline" milestone** (proposal §17.2 critical gate).

---

## 4. Decision-number handling (Rule B)

Every threshold/weight/band touched ships in `agents/configs/verdicts_default.yaml` (L1 externalized) and is validated by the Pydantic schema (T1.2). **No new hand-set constant in any `.py`** — the lone exception is the F-beta `beta` default (2.0) which is a Pydantic field default *in the schema*, sourced from proposal §6 (the committed default), and overridable via YAML.

| Decision-number | Current location (verified) | Config path (P1) | Maturity target |
|-----------------|-----------------------------|------------------|-----------------|
| `ACCURACY_WEIGHTS` (per-class {ml,slither,aderyn}) | consensus.py:46-57 | `consensus.accuracy_weights` | L3 (P3: fitted from confusion matrix) |
| `DEFAULT_WEIGHTS` | consensus.py:61 | `consensus.default_weights` | L3 (P3) |
| `ML_WEIGHT_SCALE` (was env) | consensus.py:71,77 | `consensus.ml_weight_scale` | L2 (P0.1 measures; raise on retrain) |
| `ML_POSITIVE_THRESHOLD` | consensus.py:82 | `consensus.ml_positive_threshold` | L2 |
| `CONFIRMED/LIKELY/DISPUTED_BAND` | consensus.py:85-87 | `consensus.confirmed_band` etc. | L2 (PR-curve selection) |
| `SLITHER/ADERYN_AGREE/DISAGREE`, `RAG_AGREE`, `RAG_RELEVANCE` | confidence.py:22-27 | `confidence.*` | L2 |
| `DEEP_THRESHOLDS` (per-class) | routing.py:23-34 | `routing.deep_thresholds` | L2 |
| `ROUTING_RULES` (per-class) | routing.py:43-54 | `routing.routing_rules` | L1 (structural; L2 if measured) |
| `compute_verdict` cutoffs (0.50/0.80/0.50) | routing.py:244-248 | `routing.compute_verdict_*` | L2 (NOTE: not in proposal §4.2 — see §6) |
| `prob_to_severity` (0.85/0.70/0.50/0.35) | routing.py:264-269 | `routing.prob_to_severity` | L1 (severity display) |
| `OVERALL_VERDICT_RANK` | routing.py:272-275 | `routing.overall_verdict_rank` | L1 (structural order) |
| `RAG_RELEVANCE_FLOOR` | attribution.py:24 | `attribution.rag_relevance_floor` | L2 |
| `BORDERLINE_BAND` | eval/pipeline_metrics.py:40 | `eval.borderline_band` | L1 (gate definition) |
| `DEFAULT_POSITIVE_VERDICTS` | eval/pipeline_metrics.py:27 | `eval.positive_verdicts` | L1 (metric definition) |
| `fbeta_beta` (NEW, =2.0) | (proposal §6) | `eval.fbeta_beta` | L1 (committed default; tunable w/ measured justification) |

**Measurement (gives Rule B teeth):** P0.1's `run_benchmark` scores a named config end-to-end and reports per-class P/R/F1/F-beta + macro aggregates (proposal §6). Any later change to a decision-number is accepted only if the delta vs `p0_honest_baseline.json` is favorable (enforced by `RegressionBaseline.compare()`; T0.1.5). **Stays put (not config):** ML three-tier thresholds (0.55/0.25 — model-side, in `ml/`); `timeouts.py` (operational env vars); `CLASS_TO_DETECTORS` (structural detector mapping).

---

## 5. Test plan

| File (new/changed) | Asserts |
|--------------------|---------|
| `tests/test_config.py` (new, T1.10) | default config == current constants per group; invalid YAML raises; env path override works; schema_version round-trips; YAML value change reflected after fresh process. |
| `tests/test_eval_fbeta.py` (new, T0.1.1) | F-beta formula on a known confusion matrix (beta=1 -> equals F1; beta=2 weights recall); `macro_fbeta` = NaN-aware mean over support>0 classes; `as_dict()` includes `fbeta`+`macro_fbeta`. |
| `tests/test_run_benchmark.py` (new, T0.1.7) | CLI arg parsing; invalid config fails fast; output dir naming `<ts>_<name>`; metrics JSON has `macro_fbeta`; exit 1 on synthetic regression. |
| `tests/test_eval_framework.py` (changed) | Existing 471-LOC suite stays green after T0.1.2 (gates move) + T0.1.5 (regression adds macro_fbeta). No assertion weakened. |
| `tests/test_consensus_voting.py` (unchanged) | Stays green after T1.5 — proves behavior preservation (config-backed constants yield identical verdicts). |
| `tests/test_confidence_tracking.py` (unchanged) | Stays green after T1.6. |
| `tests/test_routing_phase0.py`, `test_graph_routing.py` (unchanged) | Stay green after T1.7. |
| `tests/test_metric_attribution.py` (unchanged) | Stays green after T1.8. |

**Regression net:** after every P1/P0.1 task that touches a wired module, run the *full* suite `cd agents && poetry run pytest -q`. Green = behavior preserved. The 438 functions are the safety net; no test is deleted or weakened in this phase.

---

## 6. Risks & rollback

| Risk | Likelihood | Impact | Mitigation / rollback |
|------|-----------|--------|----------------------|
| Config wiring changes a verdict silently (constant value drift) | Med | High | T1.4 uses *current* values verbatim; T1.5-T1.9 run the per-module test suites; P1 gate runs the full 438-function suite. Per-file `git checkout` rollback. |
| Eager load breaks an import path (circular import) | Low | Med | `src/config/` depends only on pydantic+yaml, not on orchestration; orchestration imports config, not vice versa. If circular, lazy-import `get_config` inside functions. |
| Monkeypatching tests break (constants no longer module-level) | Med | Med | Read config **at call time** (T1.5 pattern = existing `_ml_scale()`); keep names as aliases. If a test patches the constant directly, switch it to patch `get_config()` return value. |
| F-beta formula bug mis-ranks configs | Low | Med | T0.1.1 unit test with beta=1 must equal existing F1 (exact regression check); beta=2 against a hand-computed value. |
| First baseline run needs infra (ML server) that is down | Med | Low | No-LLM deterministic-tier run is a valid first data point (T0.1.6 documents which). Baseline can be re-run later with LLM on. |
| Sidecar generation overwrites a hand-written JSON | Low | Low | T0.0.2 is idempotent; no `<stem>.json` exists in the corpus today (verified). Rollback = delete generated JSONs. |

**Phase rollback:** every task has a per-file `git checkout` or delete. The phase introduces NO schema migration, NO state-shape change, NO deletion of existing verdict logic (that is P2). If P1 destabilizes, revert the 5 wired modules + delete `src/config/` — the system returns to its current constant-based behavior unchanged.

---

## 7. Done-when checklist (objective, checkable)

- [ ] `python scripts/audit_labels.py` reports N=83 (or documented post-drop N) with 0 missing headers.
- [ ] 83 `<stem>.json` sidecars exist and `Benchmark` loads them with support matching the audit.
- [ ] `rg "qwen3.5-9b-ud" src/` matches only inside FIX-18 changelog comments.
- [ ] `agents/configs/verdicts_default.yaml` exists; `get_config()` returns a valid `SentinelConfig`.
- [ ] No decision-number constant remains hand-set in `consensus.py`/`confidence.py`/`routing.py`/`attribution.py`/`pipeline_metrics.py` (all read from config at call time).
- [ ] `cd agents && poetry run pytest -q` is green (full 438-function suite).
- [ ] `PipelineMetrics` emits `macro_fbeta` (beta=2) alongside `macro_f1`.
- [ ] `python -m src.eval.run_benchmark --name <id> --config configs/verdicts_default.yaml ...` runs end-to-end and writes `eval/runs/<ts>_<name>/eval_metrics.json`.
- [ ] `agents/eval/baselines/p0_honest_baseline.json` exists with `contract_count`=N and a real `macro_fbeta`.
- [ ] `scripts/eval_benchmark.py` is a thin wrapper (no duplicated gate/metric logic).
- [ ] Scratch file + MEMORY record the headline baseline numbers and the findings status table.

---

## 8. Contradictions found (doc vs source) — surfaced, not silently changed

1. **P0.0 "label completion / ~250 gap decisions"** (proposal §10 + §10.1 P0.0) vs **Ali: "all contracts fully checked with ground-truth labels"** (MEMORY 2026-06-23; BUILD_questions changelog removed the 65% gap finding). -> P0.0 is planned as **audit + sidecar generation + drop-if-needed contingency**, NOT a 250-decision gap-fill. The `// expect:` headers exist on all 83 (verified). If T0.0.1 reveals gaps, STOP and raise it.
2. **Eval script path:** proposal §10.1 says `scripts/eval_benchmark.py`; actual path is `agents/scripts/eval_benchmark.py`. Plan uses the real path.
3. **`eval/__init__.py:19-20`** docstring references a `gates` module that does not exist (gates are inline in the script). T0.1.2 creates `src/eval/gates.py`, making the docstring true; T0.0.4 removes the stale ref in the interim.
4. **`routing.py:compute_verdict` (lines 244-248)** has 3 hardcoded decision-numbers (0.50/0.80/0.50) NOT listed in proposal §4.2 config contents. T1.7 externalizes them (Rule B: no hand-set constants in `.py`). Flagging because the proposal's config-contents list is incomplete — the decision to externalize them is consistent with Rule B, not a re-design.
5. **`PipelineMetrics` has no F-beta** — the proposal §6 headline metric (macro F-beta, beta=2) is not yet implemented. This is the core P0.1 deliverable (T0.1.1), not a contradiction, but it means the "C.2 is built" framing in §10 overstates what exists: the framework computes F1, not the committed recall-weighted metric.

No decision in proposal §4/§5/§10.1 was changed. Items 1 and 4 are raised as questions for Ali's awareness; the rest are path/fact corrections.
