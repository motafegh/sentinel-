# Actionable Plan — Stage 0: Skeleton + Data/ Restructure

**Date:** 2026-06-09
**Stage:** 0 of 8 (Week 1: Jun 9–15)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §2, §5 (Week 1), §10 items 1–3
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §0 (F1, F14, F19, F20), §1 (0-P1 through 0-P8), §2 (C-2, C-6)
**Exit criteria:** `poetry install` works in `Data/.venv/`; `sentinel-data --help` runs; DVC initialized; `Data/README.md` and `Data/docs/architecture.md` written; legacy BCCC deep-dive moved; **schema v9 constants in the stub**; **solc 98-version note + on-demand install in Dockerfile**; **MLflow `sqlite:///` backend noted in config**.

**⚠ CRITICAL: Schema is v9, not v8** (per `ml/src/preprocessing/graph_schema.py:161` verified 2026-06-08). The stub in task 0.4 must use v9 constants: `FEATURE_SCHEMA_VERSION="v9"`, `NODE_FEATURE_DIM=12`, `NUM_NODE_TYPES=14`, `NUM_EDGE_TYPES=12`, `_MAX_TYPE_ID=13.0`. See `00_INDEX.md` for the full schema table.

---

## Goal

Promote the existing `~/projects/sentinel/Data/` directory to a first-class installable Python package named `sentinel-data` with a working DVC pipeline scaffold. After this stage, every subsequent stage (1–7) builds inside the skeleton created here. No data is moved, no extraction logic is touched — this stage is purely structural.

---

## Why this stage first

The BCCC deep-dive outputs already live under `Data/Deep_Dive/`. They are the historical foundation (Phase 0 of the new module's docs). Promoting `Data/` to a real package now means the deep-dive work is preserved as the legacy section of the new module, and every later stage references the new structure as the only canonical layout. Doing the structural move first also means the package boundary — the only enforcement mechanism for "data layer cannot import from ml" — is locked into `pyproject.toml` before any code crosses it.

---

## Design decisions

### D-0.1 — Module location

The new package is rooted at the existing `~/projects/sentinel/Data/` directory. It is renamed internally (`Data/sentinel_data/...`) but keeps the folder name `Data/` on disk for continuity with the deep-dive work and the root README's cross-link. The BCCC deep-dive subfolder is moved once to `Data/docs/legacy/bccc_deep_dive/` and stays frozen as historical record.

### D-0.2 — Package boundary

`Data/pyproject.toml` declares `name = "sentinel-data"`, `packages = [{include = "sentinel_data"}]`, and lists no dependency on `sentinel-ml`. The one-way dependency rule is enforced at install time: `sentinel-ml` may add `sentinel-data = "^0.1.0"`, never the reverse. A future CI job will run `poetry show --tree` in `Data/` and fail if any `sentinel-ml` reference appears.

### D-0.3 — Standalone venv

`Data/.venv/` is separate from `ml/.venv/`. Different dependency sets (the data side needs `slither-analyzer`, `solc-select`, `huggingface-hub`, `dvc`; the ML side does not) and different update cadences. Both venvs coexist on the same machine without conflict.

### D-0.4 — Stub vs real code in Stage 0

`representation/graph_schema.py` and `representation/graph_extractor.py` ship as **stubs** in Stage 0. The stubs re-export the existing v8 schema constants from the `ml/src/preprocessing/` originals (re-exported by value — not imported — to keep the package self-contained at this stage). The real port from `ml/` happens in Stage 3. The seam swap (delete old path, switch ML module to import from new path) happens in Stage 7 with a byte-identical regression test as the gate.

This split exists because moving the graph extraction code in Stage 0 would risk regressions in the active Run 9 training pipeline; isolating the move to Stage 3, with a dedicated regression stage before the seam swap in Stage 7, is the safer sequence.

### D-0.5 — DVC as the orchestrator

`Data/dvc.yaml` defines all 9 stages from the proposal §3 as named nodes in a DAG, each with placeholder commands in Stage 0. Every later stage replaces its placeholder with the real command. DVC's caching means re-running unchanged stages is free; bumping a crosswalk YAML only re-runs `label` and downstream, not `ingest` or `represent`.

### D-0.6 — Config-as-data

`Data/config.yaml` is the single source of truth for: which sources are enabled, which versions are pinned, what the verification thresholds are, what the export shard size is. All 12 sources + Zenodo 16910242 are listed; only `scabench` and `smartbugs_curated` and `defihacklabs` are `enabled: true` initially (the ones we need for Stages 1–4 end-to-end tests). BCCC is recorded under a separate `deferred_sources:` block, not as a regular source.

### D-0.7 — CLI surface area

`Data/sentinel_data/cli.py` exposes a top-level `sentinel-data` command with two modes:
- `sentinel-data run [--from-stage N] [--config config.yaml] [--dry-run]` — runs the full pipeline (or resumes from a stage)
- `sentinel-data <stage> [--config ...] [--dry-run]` — runs a single stage

In Stage 0 the implementations are placeholders that print the stage name. Stages 1–7 fill in real logic, stage by stage.

### D-0.8 — Docker for reproducibility

`Data/docker/Dockerfile.data` is authored in Stage 0 but not built yet. The build + run is verified in Stage 7 as part of the export + seam swap. The Dockerfile pins `solc-select install` for the 6 most common Solidity versions (0.4.26, 0.5.17, 0.6.12, 0.7.6, 0.8.20, 0.8.24) so the offline batch extraction is byte-identical across machines.

### D-0.9 — Documentation as code

`Data/README.md` and `Data/docs/architecture.md` are committed in Stage 0 because they are the design artifacts that every later PR references. Architecture diagrams use Mermaid (rendered in GitHub Markdown). The README is the entry point for any future contributor; the architecture doc is the entry point for any reviewer.

### D-0.10 — ADR for the split

The "why we have a separate data module" decision is the highest-stakes architectural call in the v2 build. It is documented as `ADR-0001-sentinel-data-skeleton.md` in `Data/docs/decisions/`. The ADR cites the BCCC failure (89% Reentrancy FP, 86.9% CallToUnknown FP), references the proposal, and lists the integration seam as the explicit trade-off.

---

## Tasks — ordered, each with verifiable exit condition

### 0.1 — Restructure `Data/Deep_Dive/` → `Data/docs/legacy/bccc_deep_dive/`

Move the entire `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/` subtree (5 phase subdirs + 4 plan markdown files + 1 README) to `Data/docs/legacy/bccc_deep_dive/` using `git mv`. Update any internal references (paths in READMEs, links in plan docs) to reflect the new location. The empty `Data/Deep_Dive/` directory is removed.

**Why first:** every later stage references the legacy deep-dive outputs from the new location. Doing this move first means the new structure is the only canonical layout from day 1, and no PR later in the build needs to update internal links.

**Exit condition:** `test ! -e Data/Deep_Dive && test -d Data/docs/legacy/bccc_deep_dive`; the file `contracts_clean_v1.4.csv` is reachable at its new path.

**Commit:** `chore(data): relocate BCCC deep-dive to docs/legacy/bccc_deep_dive`

---

### 0.2 — Create `pyproject.toml` for `sentinel-data`

Author `Data/pyproject.toml` declaring `sentinel-data` as a Poetry package with `packages = [{include = "sentinel_data"}]`. The dependency list covers everything the v2 build needs (data sources, Solidity tooling, graph export, DVC, HF, verification). The dependency list explicitly excludes any reference to `sentinel-ml`.

**Why:** the package boundary is the only enforcement mechanism for "data layer cannot import from ml." Locking it in `pyproject.toml` makes the boundary visible and machine-checked.

**Exit condition:** `poetry install` succeeds in `Data/.venv/`; `poetry show --tree | grep -i sentinel-ml` returns empty.

**Commit:** `feat(data): add pyproject.toml for sentinel-data package`

---

### 0.3 — Create `sentinel_data/` package skeleton

Create empty `__init__.py` files for all 9 subpackages and their nested subpackages (15 in total, per the proposal §2 tree). The `representation/` subpackage gets a non-empty `__init__.py` that re-exports the v8 schema constants from its stub. The other 8 subpackages are empty for now.

**Why:** every later stage imports from this package. Empty `__init__.py` files make the layout real and the import paths testable.

**Exit condition:** `poetry run python -c "from sentinel_data.representation import FEATURE_SCHEMA_VERSION"` succeeds; all 9 subpackage dirs exist.

**Commit:** `feat(data): create sentinel_data package skeleton`

---

### 0.4 — Add stubs for `representation/graph_schema.py` and `representation/graph_extractor.py`

Create the two files as **stubs**. The schema stub holds the **active v9** constants by value (copied from `ml/src/preprocessing/graph_schema.py:161,175,218`) — **NOT v8 as the proposal §2 says**. The constants to replicate:
- `FEATURE_SCHEMA_VERSION = "v9"`
- `NODE_FEATURE_DIM = 12`
- `NUM_NODE_TYPES = 14`
- `NUM_EDGE_TYPES = 12`
- `_MAX_TYPE_ID = 13.0`
- `VISIBILITY_MAP`, `NODE_TYPES` (14 entries), `EDGE_TYPES` (12 entries), `FEATURE_NAMES` (12 entries)

Mark the stub `STUB = True` so future stages know to replace it. The extractor stub defines the function signature and raises `NotImplementedError` with a pointer to Stage 2 (not Stage 3 — the port is Stage 2, the seam swap is Stage 7).

Also create `Data/sentinel_data/representation/_schema_constants.md` (per AUDIT_PATCHES N-1) — a single source of truth markdown file listing all active schema constants for cross-stage reference. This prevents drift.

Also create `Data/sentinel_data/representation/_schema_version_registry.json` (per AUDIT_PATCHES N-2) — a JSON file tracking the active schema version, used by the versioner in Stage 2.

**Why:** the real port from `ml/` happens in Stage 2. Stubbing in Stage 0 means the package can be installed and imported without touching the active Run 9 training pipeline. The stub is the placeholder that the Stage 7 seam swap removes.

**Why the v9 constant is critical:** if the stub says v8, the Stage 2 byte-identical regression test (which compares new path output to old path output) will fail because the old path uses v9. The stub must match the active schema.

**Exit condition:** `from sentinel_data.representation.graph_extractor import GraphExtractionError` imports; `extract_contract_graph()` raises `NotImplementedError`; **all constants match v9 (not v8)**; `_schema_constants.md` and `_schema_version_registry.json` exist.

**Commit:** `feat(data): add representation stub with v9 schema constants`

---

### 0.5 — Add `dvc.yaml` with 9 stage placeholders

Author `Data/dvc.yaml` defining the 9-stage DAG from the proposal §3 (ingest → preprocess → represent → label → verify → split → register → export → analyze). Each stage has a placeholder `cmd:` and a `deps:` list referencing the subpackage directory; `outs:` points to a `.gitkeep` file in the corresponding `data/` subdir. Real commands are added per stage.

**Why:** DVC is the runtime orchestrator. Initializing the DAG now (even with placeholders) means `dvc repro --dry` is testable from Stage 0 forward, and every later stage adds its real commands to a working scaffold.

**Exit condition:** `dvc repro --dry` succeeds (no actual execution); `dvc dag` prints the 9-stage DAG; each `data/<stage>/.gitkeep` file is referenced as an `out`.

**Commit:** `feat(data): scaffold 9-stage dvc.yaml pipeline`

---

### 0.6 — Add `config.yaml` with all 17 sources + Zenodo 16910242 + BCCC deferred (per friend's suggestions)

Author `Data/config.yaml` listing **17 sources** (12 from the original proposal + **5 new from friend's recommendations**: DIVE-promoted-to-Tier-1, FORGE, ScrawlD, Code4rena, DeFi Hacks REKT) + Zenodo 16910242 with their connector type, URL, pin placeholder, and crosswalk path. Only `scabench`, `smartbugs_curated`, and `defihacklabs` are `enabled: true` initially (the small low-risk subset for Stages 1–4). BCCC is recorded under a separate `deferred_sources:` block. The top-level `pipeline:` section holds the global settings (schema version, export format, shard size, verification thresholds).

**Friend's recommendation (per `datasources_suggestions.md`, applied 2026-06-08):**

| Source | Friend's tier | Why added | Plan tier |
|---|---|---|---|
| **DIVE** (Nature Scientific Data 2025) | TIER 1 (promoted from v2 dataset proposal §6 Tier 4) | 22,330 real-world contracts, 8 DASP classes, multi-label (avoids BCCC's "one folder = one vuln" fiction), peer-reviewed | Tier 1 (gold) |
| **FORGE** (ICSE 2026) | TIER 1 (new) | LLM-driven extraction from real audit reports; CWE classification (industry standard, not ad-hoc DASP); no tool-induced FPs | Tier 1 (gold) |
| **SolidiFI Benchmark** (ISSTA 2020) | TIER 1 (already in plan) | Bug-injection ground truth (100% mathematically certain); 7 vulnerability types | Tier 1 (gold) |
| **ScrawlD** (MSR 2022) | TIER 2 (new) | 5-tool majority voting (requires 3/5 agreement); 6,780 mainnet contracts | Tier 2 (silver) |
| **Code4rena Audit Reports** | TIER 2 (new) | Human expert auditors under financial incentive; covers logic/access-control bugs tools miss | Tier 2 (gold) — needs report-scraping connector |
| **DeFi Hacks REKT Database** | TIER 2 (new) | Verified real exploits; perfect ground truth | Tier 2 (gold) — T0 tier assignment |
| **DISL** (TBD) | TIER 3 (new) | 514,506 unique Solidity files; unlabeled pretraining + NonVulnerable class | Tier 4 (bronze) |
| **ReentrancyStudy-Data** (TBD) | TIER 2 (new) | 230,548 reentrancy-labeled contracts; massive scale for the priority class | Tier 4 (bronze) — tool-based, single class |
| **EVMbench** (referenced in friend doc) | TIER 2 (new) | 120 curated vulnerabilities from 40 Code4rena audits | Subset of Code4rena; may not need separate connector |
| **DeFiVulnLabs** (TBD) | TIER 2 (new) | 48 vulnerability types | Tier 3 (structural) |

**Friend's 3-tool ensemble warning** (per datasources_suggestions.md Part 1): Conkas + Slither + Smartcheck only detects 76.78% of actual vulnerabilities; Slither reentrancy precision is 51.97%. **This reinforces** Stage 4's tool_validator design (D-4.3 in §3.5): tool agreement is corroborative, NOT authoritative; human audit overrides tool labels.

**Friend's 97% SmartBugs Wild FP rate warning** (per Part 4 "What to Avoid"): SmartBugs Wild as labeled data is a "97% FP" trap. The Stage 3 crosswalk for `smartbugs_wild` is already in the plan as Tier 3 (T3 tool-generated, conservative); the friend confirms this is correct. **No change needed** — the existing crosswalk design is validated by the friend.

**Why:** the config is the single source of truth for what runs in the pipeline. Locking the structure now means Stage 1's ScaBench work doesn't have to invent config from scratch.

**Exit condition:** `config.yaml` is valid YAML; `sources.scabench.enabled == true`; `deferred_sources.bccc` exists; `pipeline.verification.fail_threshold` is set to 0.30; `pipeline.solc.baseline_versions` lists the 6 solc versions; **`pipeline.mlflow.uri` is `sqlite:///mlruns.db`** (per F20).

**Commit:** `feat(data): add config.yaml with all 12 sources + BCCC deferred`

---

### 0.7 — Initialize DVC and create `data/` directory layout

Run `dvc init` in `Data/`. Create the 9 `data/<stage>/` subdirectories (`raw/`, `preprocessed/`, `representations/`, `labels/`, `verification/`, `splits/`, `registry/`, `exports/`, `analysis/`) with `.gitkeep` files. Add `data/**/` to `.gitignore` (DVC tracks the directory pointers, not the contents).

**Why:** the physical layout must exist before any stage writes to it. DVC needs to be initialized before any `dvc add` or `dvc repro` works.

**Exit condition:** `dvc status` is clean; all 9 subdirs exist; `.dvc/config` exists; `.gitignore` excludes `data/**/`.

**Commit:** `chore(data): initialize DVC and scaffold data/ directory layout`

---

### 0.8 — Add `cli.py` skeleton with 9 stage subcommands

Author `Data/sentinel_data/cli.py` with a top-level `argparse` parser that supports the two modes from D-0.7: `sentinel-data run [--from-stage N] [--config config.yaml] [--dry-run]` and per-stage `sentinel-data <stage> [--config ...] [--dry-run]`. In Stage 0 the implementation is a placeholder that prints the stage name(s) it would run. Wire the `sentinel-data` entry point in `pyproject.toml` under `[tool.poetry.scripts]`.

**Why:** the CLI is the user-facing entry point. The skeleton supports `--help` and `--dry-run` from this stage forward; later stages fill in real implementations.

**Exit condition:** `sentinel-data --help` exits 0 and lists all 9 stage subcommands; `sentinel-data run --dry-run` lists all 9 stages; `sentinel-data run --dry-run --from-stage verify` lists only stages 5–9.

**Commit:** `feat(data): add sentinel-data CLI skeleton with 9 stage subcommands`

---

### 0.9 — Add `Dockerfile.data` with pinned solc/slither versions

Author `Data/docker/Dockerfile.data` based on **`python:3.12.1-bookworm`** (per AUDIT_PATCHES 7-P10 — `slither-analyzer` requires `build-essential` + `libpq-dev` for `psycopg2-binary` transitive dep; slim doesn't have them). Install Poetry, copy `pyproject.toml` + `poetry.lock` for layer caching, run `poetry install --without dev`. Pin **6 baseline solc versions** via `solc-select install` (0.4.26, 0.5.17, 0.6.12, 0.7.6, 0.8.20, 0.8.24) and `solc-select use 0.8.24`. **The 92 other solc versions (0.4.0–0.8.35) are installed on-demand by `solc-select install` at Stage 1 compile time** (per F14 — 98 versions are pre-installed in `~/.solc-select/artifacts/` on dev machines; the Dockerfile is a minimal baseline, the rest are on-demand). Set entrypoint to `sentinel-data`. Author `Data/docker/.dockerignore` excluding `.venv/`, `data/`, `.dvc/cache/`, etc.

**Why:** reproducibility. Every developer machine and every CI run should produce byte-identical representations for the same input. Pinned Docker image is the enforcement mechanism. The Dockerfile is authored now but the actual build verification is deferred to Stage 7 (it requires the full pipeline to be functional before a meaningful test).

**Why bookworm over slim:** `slither-analyzer` won't install cleanly on `slim` because of the `psycopg2-binary` transitive dependency that needs `libpq-dev` + `build-essential`. The image is ~200 MB larger but the install is reliable. (See AUDIT_PATCHES 7-P10.)

**Why 6 baseline + on-demand:** the 98-version full install is ~1.2 GB and would bloat the image unnecessarily. The 6 baselines cover the most common Solidity eras; the rest are downloaded on demand by the Stage 1 preprocessor.

**Exit condition:** `Dockerfile.data` and `.dockerignore` exist; `docker build` syntax is valid (no actual build in Stage 0); image is `python:3.12.1-bookworm` (not slim).

**Commit:** `feat(data): add Dockerfile.data with pinned solc/slither versions`

---

### 0.10 — Write `Data/README.md` (module map)

Author `Data/README.md` covering: what `sentinel-data` is; why it exists (link to proposal §1); installation steps; quickstart (`sentinel-data run --dry-run`); the module map (copy the directory tree from proposal §2); the 9 pipeline stages with one-line descriptions; the datasets in scope (link to proposal §6); the BCCC legacy pointer; current status (Stage 0 complete, Stages 1–7 in progress); contributing guide.

Add a **"WSL2 caveats"** section (per F19, C-1): all WSL commands must use the `wsl -- bash -c '...'` wrapper. The PowerShell host `wsl.exe` errors on inline commands. Example pattern in all docs.

Add a **"MLflow backend"** section (per F20): the v2 build's MLflow uses `sqlite:///mlruns.db` only. The `file:///` backend is corrupt (experiments 1, 2, 3 are corrupted in the file backend). Any plan that runs an experiment must use sqlite.

Add a **"Schema version"** section: the active schema is v9 (verified 2026-06-08). See `sentinel_data/representation/_schema_constants.md` for the full table. The proposal §2 incorrectly says v8; the v2 build uses v9 throughout.

**Why:** the README is the entry point for any future contributor. It must reflect the package structure created in Stage 0 and link to the proposal for full context. The WSL + MLflow + schema callouts prevent the operational issues documented in MEMORY.

**Exit condition:** file exists, > 100 lines, contains > 5 references to `sentinel-data`; has WSL2 + MLflow + schema-version sections.

**Commit:** `docs(data): add module README with directory map and quickstart`

---

### 0.11 — Write `Data/docs/architecture.md` (data flow + DAG diagrams)

Author `Data/docs/architecture.md` covering: a Mermaid data-flow diagram (raw source → connector → ingestion → preprocessing → representation → labeling → verification → splitting → registry → export → analysis); per-stage contracts (input file, output file, schema version, side effects — drawn from proposal §3); the hard contract with `sentinel-ml` (proposal §3.8 + §4); the one-way dependency boundary and why it matters; why DVC and Docker; the confidence tier system; the 6 verification gates (proposal §9).

**Why:** every later stage's PR references this doc. The architecture must be committed before any stage starts adding code, so reviewers can see the intended design.

**Exit condition:** file exists, contains Mermaid diagrams, references the v2-gold catalog name.

**Commit:** `docs(data): add architecture.md with data-flow + DAG diagrams`

---

### 0.12 — Add `tests/` directory with smoke tests for the skeleton

Author `Data/tests/test_skeleton.py` with the following smoke tests:
- All 9 `sentinel_data` subpackages are importable
- The `representation` stub exposes the **v9** constants (regression test for the port in Stage 2)
- `config.yaml` is valid YAML and has the expected structure
- `sentinel-data --help` exits 0 and lists the 9 stages
- `sentinel-data run --dry-run` lists all 9 stages
- `sentinel-data run --dry-run --from-stage verify` resumes from verify and excludes earlier stages
- `pipeline.mlflow.uri == "sqlite:///mlruns.db"` (per F20)
- `pipeline.solc.baseline_versions` contains exactly 6 versions (per F14)

Add `Data/pytest.ini` with `testpaths = tests` and `python_files = test_*.py`.

**Why:** Stage 0 must be testable in isolation. Each later stage adds tests for its own code; Stage 0's job is to prove the skeleton loads and the CLI works.

**Exit condition:** `poetry run pytest tests/ -v` passes all smoke tests.

**Commit:** `test(data): add Stage 0 smoke tests for skeleton + CLI + config`

---

### 0.13 — Initialize `docs/decisions/ADR-0001-sentinel-data-skeleton.md`

Author the first ADR documenting the package split decision. Standard format (Context / Decision / Consequences). The "Context" cites the BCCC failure (89% Reentrancy FP, 86.9% CallToUnknown FP), the proposal, and the verification regression test from Stage 4. The "Decision" is the package split + location. The "Consequences" lists the integration seam, the one-way dependency rule, and the deferred BCCC decision. Create `Data/docs/decisions/INDEX.md` as a registry of all future ADRs (append-only).

### 0.14 (NEW) — Add a "Code-Bug State at Build Start" ADR

Author `Data/docs/decisions/ADR-0002-code-bug-state-at-build-start.md` (per AUDIT_PATCHES 0-P8, I-5). The ADR catalogs the 8 already-fixed bugs (A9, A15, A20, A34, A38, resume, def_use, return_ignored) and the 3 still-open ones (CALL_ENTRY cross-function, predictor tier threshold, BCCC-vs-version skew). For each, list: the file:line, the original bug, the fix commit, the regression test that guards it, and which plan stage must preserve the fix. The ADR is the first design artifact that downstream stages reference when they see "do not fix this; the test in Stage N guards it."

**Why:** the build creates irreversible architectural decisions. The first ADR documents why we have a separate `sentinel-data` package at all; the second ADR documents the current state of the codebase that the build is moving. ADRs are append-only and never edited after merge.

**Exit condition:** both files exist; ADR-0001 references the BCCC failure; ADR-0002 references all 8 fixed bugs + 3 open bugs; INDEX.md has the standard template.

---

### 0.14 — Update root `sentinel/README.md` with `sentinel-data` cross-link

Add a new section to `~/projects/sentinel/README.md` titled "📦 Data Engineering Module" between the existing project intro and the next section. The section contains: a one-paragraph summary, a pointer to `Data/README.md`, a pointer to the integration proposal, and a note about the BCCC legacy at `Data/docs/legacy/bccc_deep_dive/`.

**Why:** the repo root README is the first thing any visitor sees. Stage 0 must add a top-level section pointing to the new data module so the package boundary is visible at the project level.

**Exit condition:** root README contains > 1 reference to `sentinel-data` and > 1 reference to `Data/README.md`.

**Commit:** `docs: link sentinel-data module from root README`

---

## What NOT to fix (preservation list)

Per AUDIT_PATCHES C-4 + INDEX §"8 code bugs already fixed":

| Bug | Status | File:line | Stage 0 action |
|---|---|---|---|
| **A20** label=0 hardcode | ✅ FIXED | `ml/src/data_extraction/ast_extractor.py:290,342,395` | Do not re-fix. The Stage 2 port preserves the fix via regression test. |
| **A9** `now` keyword miss | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:587-605` | Do not re-fix. The Stage 2 port preserves the fix via regression test. |
| **A15** def_map by name | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:1147-1179` | Do not re-fix. The Stage 2 port preserves the fix via regression test. |
| **A34** prefix sort dim | ✅ FIXED | `ml/src/models/sentinel_model.py:356,367` | Do not re-fix. The Stage 2 port preserves the fix via regression test. |
| Resume overwrite | ✅ FIXED | `ml/src/training/trainer.py:383,1184,1206,1212` | Do not re-fix. Stage 8 uses the full-resume default. |
| **EMITS edge bug** | ⚠ OPEN | `ml/src/preprocessing/graph_extractor.py` (Interp-6) | Stage 7 seam swap must fix this (per AUDIT_PATCHES 7-P6). |
| **Predictor tier threshold** | ⚠ OPEN | `ml/src/inference/predictor.py:150,168,752` | Stage 7 seam swap must fix this (per AUDIT_PATCHES 7-P7). |

**Verification:** the Stage 0 stub must NOT contain code that re-implements any of these fixes. The stub uses NotImplementedError for the extractor and copies the v9 schema constants by value.

---

## Final exit criteria check

From a clean clone, verify all of the following hold:

| # | Check |
|---|---|
| 1 | `Data/docs/legacy/bccc_deep_dive/` exists; `Data/Deep_Dive/` does not exist |
| 2 | `cd Data && poetry install` succeeds |
| 3 | `poetry run python -c "from sentinel_data.representation import FEATURE_SCHEMA_VERSION; assert FEATURE_SCHEMA_VERSION == 'v9'"` succeeds (NOT v8) |
| 4 | `poetry run python -c "from sentinel_data.representation import NODE_FEATURE_DIM, NUM_EDGE_TYPES, NUM_NODE_TYPES; assert (NODE_FEATURE_DIM, NUM_EDGE_TYPES, NUM_NODE_TYPES) == (12, 12, 14)"` succeeds |
| 5 | `poetry run sentinel-data --help` exits 0 with all 9 stage subcommands |
| 6 | `poetry run sentinel-data run --dry-run` lists all 9 stages |
| 7 | `dvc status` is clean |
| 8 | `poetry run pytest tests/ -v` passes all smoke tests |
| 9 | All 9 subpackage dirs exist under `sentinel_data/` |
| 10 | `Data/README.md`, `Data/docs/architecture.md`, `Data/docs/decisions/ADR-0001-sentinel-data-skeleton.md`, `Data/docs/decisions/ADR-0002-code-bug-state-at-build-start.md` exist |
| 11 | `Data/dvc.yaml`, `Data/config.yaml` (with `pipeline.mlflow.uri=sqlite:///mlruns.db`), `Data/docker/Dockerfile.data` (with `python:3.12.1-bookworm`) exist |
| 12 | `poetry show --tree | grep -i sentinel-ml` returns empty (boundary enforced) |
| 13 | `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/contracts_clean_v1.4.csv` is reachable at its new path |
| 14 | `Data/sentinel_data/representation/_schema_constants.md` and `_schema_version_registry.json` exist (per N-1, N-2) |
| 15 | `config.yaml` lists **17 sources** (12 original + 5 new from friend's suggestions: DIVE-promoted, FORGE, ScrawlD, Code4rena, DeFi Hacks REKT) + Zenodo 16910242; BCCC is in `deferred_sources:` |

All 14 pass → **Stage 0 complete**. Merge to main, tag `data-stage-0`, proceed to Stage 1.

---

## Risk register

| Risk | Mitigation |
|---|---|
| `git mv` of `Deep_Dive/` fails because the path contains spaces (e.g. "Source Codes" inside the BCCC source dir) | Quote the path; verify with `git mv -n` dry-run first |
| Empty `Data/Deep_Dive/` left behind if `rmdir` fails (e.g. hidden files) | `test ! -e Data/Deep_Dive` is in the exit criteria; merge blocked if it fails |
| `poetry install` in `Data/` conflicts with the existing root `~/projects/sentinel/.venv/` | The two venvs are independent; `Data/.venv/` is created in `Data/` and uses `Data/pyproject.toml` only |
| The CLI skeleton uses `print` for stage output; later stages need structured logging | Stage 0 is `print` for visible diffs; Stage 1 switches to `logging.getLogger("sentinel_data")` |
| The `representation/__init__.py` re-exports from the stub; if the stub is removed without updating the init, every import breaks | Stage 3 (port) keeps the stub path stable; Stage 7 (seam swap) removes the stub only after the regression test passes |
| Docker build fails in Stage 0 because `solc-select` versions are not yet pinned in CI | The Dockerfile is authored but not built in Stage 0; the build verification is deferred to Stage 7 |

---

**End of Stage 0 actionable plan. Total estimated time: 4–5 working days (Jun 9–13), with Jun 14–15 as buffer.**
