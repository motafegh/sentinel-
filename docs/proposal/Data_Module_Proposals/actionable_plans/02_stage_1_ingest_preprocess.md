# Actionable Plan — Stage 1: Ingestion + Preprocessing

**Date:** 2026-06-16
**Stage:** 1 of 8 (Week 2: Jun 16–22)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.1, §3.2, §5 (Week 2), §6
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §1 (1-P1 through 1-P7), §2 (C-7 performance budget)
**Exit criteria:** `sentinel-data ingest --source scabench` clones ScaBench at its pinned commit with SHA-256 verification; preprocessing passes 30 hand-picked raw `.sol` files through flatten + two-pass compile + dedup + normalize + segment + version-bucket with all sidecar `meta.json` fields populated correctly; **performance budget: 30 files in < 5 min on 8 cores**.

---

## Goal

Implement the **Ingestion** and **Preprocessing** submodules end-to-end against ScaBench (the smallest-effort Tier-1 source). After this stage, raw contracts from a pinned upstream source are reliably pulled to disk, validated, flattened, compiled, deduplicated, normalized, segmented, and version-bucketed — with a per-file sidecar `meta.json` recording provenance. This is the foundation that Stages 2–7 build on.

The stage targets ScaBench as the first source because it is the smallest-effort end-to-end test (per proposal §6). The other 12 sources are *not* enabled in this stage; they are added in Stages 2–4 with the same connector and preprocessor scaffolding, but Stage 1 only proves the design works for one source.

---

## Why this stage second

Stage 0 created the package skeleton. Stage 1 is the first stage that touches real data. Doing ingestion + preprocessing together (rather than separating them into two stages) is intentional: a connector that cannot feed a preprocessor is useless, and a preprocessor with nothing to consume is untestable. Proving the seam between ingestion and preprocessing on one source is the fastest way to validate the design before adding 11 more sources.

---

## Design decisions

### D-1.1 — One connector per family, not per dataset

The 5 connector classes (Git, HuggingFace, Zenodo, Etherscan, Manual) are the *only* connector classes in the module. Every source plugs into one of them. ScaBench, Web3Bugs, Bastet, DeFiHackLabs, SmartBugs (Curated + Wild), OpenZeppelin (Contracts + Ethernaut), and Messi-Q all use the Git connector. Solidly Defi Vulnerabilities and Slither-Audited use the HF connector. Zenodo 16910242 uses the Zenodo connector. Etherscan and Manual are reserved for future use (Tier-3 / reconstruction sources).

This decision means the per-source integration work is **connector configuration + crosswalk YAML**, not new code. The connector is reusable.

### D-1.2 — Ingestion manifest is the source of truth for "what was pulled"

Every `sentinel-data ingest --source <name>` run produces an `ingestion_manifest.json` per source containing: source name, connector type, URL or HF dataset name or Zenodo ID, pinned commit / version / record, the SHA-256 of every downloaded file, the total contract count, and the timestamp. The manifest is registered in the catalog. A subsequent `sentinel-data ingest --source <name>` re-validates the SHA-256; if any file's hash has changed, the run fails loud.

The manifest is **append-only** — past ingestions are never deleted, they are versioned. This is the audit trail that lets us answer "what version of ScaBench did Run 11 train on?" six months from now.

### D-1.3 — Pinned versions are non-negotiable

Every connector requires a pin in `config.yaml`. Wildcard versions ("HEAD", "latest", "main") are rejected at config-load time. The pin is what makes the pipeline reproducible. The `freshness.py` tool (W1 of the stage) compares the pinned version to the upstream HEAD and emits a "stale" alert when they diverge, but the alert does **not** auto-update the pin — that requires a human review and a new manifest entry.

### D-1.4 — Preprocessing is a 5-step pipeline, each step isolated

The preprocessor runs 5 transformations in fixed order on each raw `.sol` file:
1. **Flatten** — resolve `import` chains to a single file (`solc --flatten` or `hardhat-flatten`); skip if no imports
2. **Compile** — resolve `pragma solidity` to a specific `solc` version, install via `solc-select` if missing, compile to verify parseability; drop files that fail to compile
3. **Deduplicate** — three-level (exact SHA-256 → address-level → AST near-dup clustering); assign a `dedup_group_id` per cluster
4. **Normalize** — strip comments, SPDX headers, non-semantic decorators; standardize whitespace
5. **Segment + version-bucket** — for multi-contract files, split into per-contract units with inheritance chain; tag each as `legacy` (<0.6) / `transitional` (0.6–0.7) / `modern` (0.8+)

Each step is a separate function in `preprocessing/` that takes the previous step's output and writes a new artifact. This makes failures debuggable (the step that failed is named) and makes it possible to re-run individual steps (e.g. re-bucket after a version policy change without re-compiling).

### D-1.5 — Drop-not-fix for compile failures

A file that fails to compile is **dropped**, not passed through with a warning. The reasoning: an un-parseable contract cannot be reliably represented as a graph, and forcing a graph from a broken parse produces silent model degradation (the BCCC dataset has 8,232 BCCC contracts that did not compile cleanly in Phase 5 retry — they were re-classified as `NonVulnerable` only because they could not be verified to be vulnerable, which is not the same thing as clean).

The drop is recorded with the compile error in the sidecar `meta.json` so the manifest carries a full audit trail. The dropped file's source path and reason are emitted to a per-source `dropped.csv` for human review.

### D-1.6 — Three-level dedup is the highest-leverage step

The BCCC duplication rate was 38.8% per the Phase 1 inventory. Without dedup, a contract that appears in ScaBench *and* in Slither-Audited gets two different label sets, and the merge step (Stage 3) has to resolve them. With dedup, the merge step sees one canonical contract and merges the labels with provenance intact.

The three levels matter for different reasons:
- **Exact SHA-256** — catches whitespace/comment-only differences across sources; the cheapest check
- **Address-level** — same Ethereum address across sources; rare but important for cross-source validation
- **AST near-dup** — SoliDiffy-style AST similarity above 0.92 cosine merges into the same `dedup_group_id`; catches the "minor edits" leakage that bit us on BCCC

The near-dup threshold (0.92) is in `config.yaml` under `pipeline.dedup.ast_similarity_threshold` so it can be tuned without code changes.

### D-1.7 — Sidecar `meta.json` is the contract between preprocessing and downstream stages

Every preprocessed file is accompanied by a `meta.json` carrying: `sha256`, `source`, `original_path`, `contract_count`, `version_bucket`, `inheritance_root`, `dedup_group_id`, `parent_sha256`, `pragma`, `solc_version`, `compile_status`, `compile_error` (if any), `n_imports`, `n_normalized_lines`. Downstream stages (representation, labeling, splitting) read this sidecar to make decisions. The sidecar schema is the contract; if the schema changes, the schema version bumps and downstream stages opt in explicitly.

---

## Tasks — ordered, each with verifiable exit condition

### 1.1 — Implement `sentinel_data/ingestion/manifest.py`

Author the ingestion manifest module. Reads `config.yaml` for source definitions; writes `data/raw/<source>/ingestion_manifest.json` after a successful ingest. The manifest contains: source name, connector type, URL/HF/Zenodo ID, pinned version, per-file SHA-256, total contract count, fetch timestamp, fetch duration. Provides `load_manifest(source) -> dict` and `verify_manifest(source) -> bool` (re-checks every SHA-256 against the current files on disk).

**Why first:** the manifest is the contract between ingestion and every later stage. The connectors and preprocessor both read from and write to it.

**Exit condition:** `manifest.py` imports; `load_manifest` and `verify_manifest` work against a manually-constructed manifest fixture.

**Commit:** `feat(data-ingestion): add manifest module with SHA-256 verification`

---

### 1.2 — Implement the 5 connector base classes

Author `sentinel_data/ingestion/connectors/{git,huggingface,zenodo,etherscan,manual}.py`. Each connector exposes a `pull(config: SourceConfig) -> PullResult` interface where `PullResult` contains the local path, manifest data, and any sidecar files. The base class `BaseConnector` defines the interface and the `verify_sha256` helper that all subclasses use.

The Git connector clones at a pinned commit (no `--depth 1` to keep the full history for audit), checks out the pin, walks the repo for `.sol` files, and computes SHA-256 per file. The HF connector loads a HF dataset, writes each row's source code to a `.sol` file, and computes SHA-256. The Zenodo connector downloads the record zip, extracts, walks for `.sol` files, computes SHA-256. The Etherscan and Manual connectors are stubbed with `NotImplementedError` and a TODO pointer to a later milestone.

**Why next:** connectors are the lowest-level primitives; the ingestion CLI (1.3) wraps them.

**Exit condition:** `BaseConnector` and all 5 subclass shells import; Git connector pulls a real repo (ScaBench) at a pinned commit and writes SHA-256s correctly.

**Commit:** `feat(data-ingestion): add 5 connector base classes with Git implementation`

---

### 1.3 — Wire the `sentinel-data ingest` CLI subcommand

Connect `cli.py` `ingest` subcommand to the connector system. The CLI reads `--source <name>` from the user, looks up the source in `config.yaml`, instantiates the right connector, calls `pull()`, and writes the ingestion manifest. Add `--dry-run` support that prints the planned source, pin, and expected destination without executing.

Update `dvc.yaml` stage `ingest` to call `sentinel-data ingest` with the current `config.yaml`. (The DVC stage reads the enabled sources from the config and processes them serially.)

**Why:** the CLI is the user-facing entry point; the DVC wiring makes it part of the reproducible pipeline.

**Exit condition:** `sentinel-data ingest --source scabench --dry-run` prints the planned action; `sentinel-data ingest --source scabench` (real run) clones ScaBench, writes `data/raw/scabench/ingestion_manifest.json`, and registers the source in the catalog.

**Commit:** `feat(data-ingestion): wire CLI + DVC for the ingest stage`

---

### 1.4 — Implement the 5 preprocessor steps (with two-pass compile + pragma tolerance)

Author `sentinel_data/preprocessing/{flattener,compiler,deduplicator,normalizer,segmenter,version_bucketer}.py`. The flattener resolves imports using `solc --flatten` (preferred) or `hardhat-flatten` (fallback for hardhat-style projects). The compiler uses `solc-select` to install + activate the right solc version per file and verifies the file compiles — **with two-pass compile (initial + retry with relaxed pragma parsing) per AUDIT_PATCHES 1-P1, 1-P2, F24, F26**. The deduplicator runs the 3-level dedup with **`ast_similarity_threshold=0.85`** (not 0.92; the lower threshold catches BCCC-style copy-paste-with-edits per 1-P4). The normalizer strips comments/headers and standardizes whitespace. The segmenter splits multi-contract files. The version bucketer tags files with `legacy | transitional | modern` **AND distinguishes 0.8+ with `unchecked{}` from 0.8+ without** (per 1-P5 — this is the dim-9 / `in_unchecked_block` signal that was rightly dropped from v7 because 87.9% of BCCC is pre-0.8, but v2 sources will have more 0.8.x contracts).

A new `sentinel_data/preprocessing/pipeline.py` orchestrates the 5 steps per file. Each step writes a sidecar `meta.json` field; the orchestrator merges the per-step meta dicts into one final `meta.json` per preprocessed file. The drop log includes the *attempted* solc versions and the *reason for failure* (per 1-P3), not just `compile_failed`.

**Why the two-pass compile:** the Phase 5 retry script recovered 2,488 contracts that failed initial compile because of BCCC-style pragma oddities (`^ 0.4 .9` with whitespace, exact-version pragmas like `0.4.25`). ScaBench and Web3Bugs audit reports have similar patterns. A single-pass compile would lose these contracts silently.

**Why the pragma tolerance:** the original `pick_solc_version()` was found to have 2 bugs in Phase 5 Session 3 — (a) it didn't handle spaced pragmas like `^ 0.4 .9`, (b) it compiled `0.4.25` with `0.4.26` (wrong exact-version). The new compiler must (a) strip whitespace before regex, (b) try the requested version first, (c) fall back to a known-good version if not available.

**Why ast_similarity_threshold=0.85 not 0.92:** the BCCC near-dup cases (the 38.8% duplication rate) were at 0.85–0.95 AST similarity. 0.92 is too strict — it misses the copy-paste-with-minor-edits cases that produced the duplication rate. 0.85 is the empirical sweet spot for Solidity contracts.

**Why distinguish 0.8+ with `unchecked{}`:** the v7 schema rightly dropped `in_unchecked_block` (was dim 9) because 87.9% of BCCC is pre-0.8 where `unchecked{}` doesn't exist. v2 sources will have more 0.8.x contracts; the bucketer must record `has_unchecked_block: bool` in the sidecar so Stage 4's semantic checker can use it for IntegerUO detection in 0.8.x era.

**Why in one stage:** the 5 steps are co-dependent (a bucketed file is meaningless without dedup, etc.). Designing them together is faster than splitting across stages.

**Exit condition:** 30 hand-picked ScaBench raw `.sol` files flow through the pipeline end-to-end; the resulting `data/preprocessed/scabench/<sha256>.sol` and `.meta.json` files have all sidecar fields populated correctly; `has_unchecked_block` is populated for 0.8.x files; performance budget: 30 files in < 5 min on 8 cores.

**Commit:** `feat(data-preprocessing): implement 5-step pipeline (flatten→two-pass-compile→dedup@0.85→normalize→segment+bucket)`

---

### 1.5 — Wire the `sentinel-data preprocess` CLI subcommand

Connect `cli.py` `preprocess` subcommand to the pipeline. The CLI reads sources from `config.yaml`, iterates over the enabled ones, runs the pipeline per file with multiprocessing, and writes the final `data/preprocessed/<source>/` output. Add `--source <name>` to limit to one source; default is "all enabled sources." Add `--skip-dedup` and `--skip-compile` flags for fast iteration during development.

Update `dvc.yaml` stage `preprocess` to call `sentinel-data preprocess` with the current `config.yaml`.

**Why:** CLI + DVC wiring is the same pattern as ingest. Doing it now means every later stage follows the same template.

**Exit condition:** `sentinel-data preprocess --source scabench` processes the 30 files; `data/preprocessed/scabench/` contains 30 `.sol` + 30 `.meta.json` files; the sidecar schema matches the design (D-1.7).

**Commit:** `feat(data-preprocessing): wire CLI + DVC for the preprocess stage`

---

### 1.6 — Author the ScaBench crosswalk YAML (placeholder)

Author `sentinel_data/labeling/crosswalks/scabench.yaml` as a *placeholder* — the real crosswalk is authored in Stage 3 alongside the parser. The placeholder defines the YAML structure (per-class mapping table) and lists the open questions for Stage 3 (e.g. "how do we map ScaBench's 555 free-text descriptions to our 10 classes — LLM-assist or manual review?").

**Why now:** the crosswalk is referenced by `config.yaml` and the labeling CLI (Stage 3). Having a placeholder file from this stage means the import path is testable and the structure is settled before Stage 3 fills in the content.

**Exit condition:** YAML is valid; structure matches the labeling/crosswalk schema; comments mark open questions for Stage 3.

**Commit:** `chore(data-labeling): scaffold scabench crosswalk YAML placeholder`

---

### 1.7 — Add `freshness.py` (staleness alerts) — including Slither version

Author `sentinel_data/ingestion/freshness.py` that compares the pinned version of each enabled source to the upstream HEAD (via `git ls-remote` for Git sources, via the HF API for HF sources, via the Zenodo record metadata for Zenodo). Emits a `freshness_report.md` per run listing "behind by N commits / versions" per source. The report is informational, not blocking. **The report also checks the pinned `slither-analyzer` version against the latest release** (per AUDIT_PATCHES 1-P6) — Slither API changes can break the graph extractor silently, so the freshness report is the early-warning system for the Stage 2 port.

**Why:** per the proposal §3.1, the freshness checker alerts when upstream sources have new commits and the pinned version is stale. This is the trigger for the human review process that decides whether to bump the pin. Adding the Slither version check is the proactive catcher for the "extractor broke because Slither changed" failure mode.

**Exit condition:** `sentinel-data freshness` produces a report; ScaBench pin is reported correctly; Slither version check is included in the report.

**Commit:** `feat(data-ingestion): add freshness checker for upstream pin + Slither version staleness`

---

### 1.8 — Add tests for the full ingest + preprocess path

Author `Data/tests/test_ingestion/` and `Data/tests/test_preprocessing/` with the following test categories:
- **Connector tests** — each connector's `pull()` against a tiny fixture (a 5-file mini-repo for Git; a 5-row mini-dataset for HF; a tiny zip for Zenodo). SHA-256 verification passes.
- **Manifest tests** — load/save/verify round-trip; tampered SHA-256 is detected.
- **Pipeline tests** — 30 ScaBench files flow through; expected `meta.json` schema matches; compile failures are dropped with reason recorded (including attempted solc versions); dedup cluster sizes are correct.
- **Sidecar tests** — every field in the sidecar schema is present after preprocessing.
- **A20 regression test** (per AUDIT_PATCHES 1-P7) — assert that a fixture contract with known labels in the label CSV produces those labels in the graph `.y` tensor, not a hardcoded 0. This guards the A20 fix through Stage 1's preprocessor.
- **Two-pass compile test** — a fixture file with a spaced pragma (`^ 0.4 .9`) and an exact-version pragma (`0.4.25`) compiles on the second pass.
- **Pragma tolerance test** — whitespace in pragma doesn't break version resolution.
- **`has_unchecked_block` test** — a 0.8.x file with `unchecked{}` is tagged correctly in the sidecar.
- **Dedup threshold test** — two near-dup files at AST similarity 0.87 are merged (not at 0.93, which would be missed).

The tests run against ScaBench (real source) plus a tiny 5-file fixture repo for fast unit tests.

**Exit condition:** `poetry run pytest tests/test_ingestion tests/test_preprocessing -v` passes; coverage > 80% for the new code; **performance budget: 30-file pipeline runs in < 5 min on 8 cores** (per C-7).

**Commit:** `test(data): add ingest + preprocess test suites (A20 regression + two-pass compile + pragma tolerance)`

---

### 1.9 — Update `Data/README.md` and `docs/architecture.md` for Stage 1

Add to `Data/README.md` a "Quickstart: ingest a source" section showing the exact commands a new contributor would run to pull ScaBench and preprocess it. Add to `Data/docs/architecture.md` a per-stage contract table (input file → output file → schema version → side effects) for ingest and preprocess.

**Why:** the docs must stay in sync with the implementation. Every stage update touches the docs.

**Exit condition:** both files have a Stage-1 section; the quickstart example actually runs.

**Commit:** `docs(data): document Stage 1 (ingest + preprocess) usage`

---

### 1.10 — Author `ADR-0002-ingestion-and-preprocessing-design.md`

Document the key design decisions of this stage: connector-per-family (D-1.1), pinned versions mandatory (D-1.3), drop-not-fix for compile failures (D-1.5), three-level dedup (D-1.6), sidecar `meta.json` contract (D-1.7). The ADR follows the standard Context/Decision/Consequences format.

**Why:** the design decisions made in this stage are the foundation of every later stage. An ADR makes them explicit and reviewable.

**Exit condition:** file exists; references all 5 design decisions; lists trade-offs.

**Commit:** `docs(data): add ADR-0002 for ingestion + preprocessing design`

---

## What NOT to fix (preservation list)

| Bug | Status | File:line | Stage 1 action |
|---|---|---|---|
| **A20** label=0 hardcode | ✅ FIXED | `ml/src/data_extraction/ast_extractor.py:290,342,395` | Do not re-fix. The A20 regression test in 1.8 guards the fix through Stage 1's preprocessor. |
| Slither API differences between ml/ venv and sentinel-data venv | ⚠ Monitor | `slither-analyzer` version in `pyproject.toml` | The freshness check (1.7) monitors Slither version staleness. If a major version bump happens, the Stage 2 byte-identical regression test will catch it. |
| 99% DoS↔Reentrancy co-occurrence in BCCC | Source: BCCC folder-based labeling | (not in v2 corpus) | The dedup threshold (0.85) catches the duplication, but the *label co-occurrence* is a Stage 3 merger concern, not a Stage 1 preprocessing concern. |
| 0.8.x `unchecked{}` invisibility in v7 schema (was dim 9) | N/A (schema change is post-Run-11) | `ml/src/preprocessing/graph_schema.py` | Stage 1 only records `has_unchecked_block` in the sidecar; the schema change is v2.1. |
| 38.8% BCCC duplication | Source: BCCC | (not in v2 corpus) | The 3-level dedup with threshold 0.85 handles it for v2 sources. |

---

## Final exit criteria check

| # | Check |
|---|---|
| 1 | `sentinel-data ingest --source scabench` clones ScaBench at the pinned commit and writes `data/raw/scabench/ingestion_manifest.json` |
| 2 | `sentinel-data ingest --source scabench --dry-run` prints the planned action without executing |
| 3 | `sentinel-data preprocess --source scabench` produces 30 `.sol` + 30 `.meta.json` files under `data/preprocessed/scabench/` with all sidecar fields populated |
| 4 | Compile-failure files in the test fixture are dropped with reason in `dropped.csv` (including attempted solc versions) and not present in `data/preprocessed/scabench/` |
| 5 | Dedup correctly groups near-duplicate files into a single `dedup_group_id`; **threshold is 0.85 (not 0.92)** |
| 6 | Version bucketer assigns `legacy` / `transitional` / `modern` tags correctly for the 3 representative Solidity versions in the fixture; **`has_unchecked_block` is populated for 0.8.x files** |
| 7 | `sentinel-data freshness` produces a `freshness_report.md` listing ScaBench behind-by-N-commits **and the Slither version check** |
| 8 | `dvc repro ingest preprocess` runs the pipeline end-to-end |
| 9 | **Two-pass compile test passes** — a fixture file with a spaced pragma (`^ 0.4 .9`) and an exact-version pragma (`0.4.25`) compiles on the second pass |
| 10 | **A20 regression test passes** — fixture contract with known labels in the CSV produces those labels, not 0 |
| 11 | **Performance budget: 30-file pipeline runs in < 5 min on 8 cores** |
| 9 | `poetry run pytest tests/test_ingestion tests/test_preprocessing -v` passes with > 80% coverage |
| 10 | The Stage-1 sections in `README.md` and `docs/architecture.md` are present and accurate |
| 11 | `ADR-0002-ingestion-and-preprocessing-design.md` is committed |

All 11 pass → **Stage 1 complete**. Tag `data-stage-1`, proceed to Stage 2.

---

## Risk register

| Risk | Mitigation |
|---|---|
| `solc-select install` for older solc versions (0.4.x) fails on WSL2 / requires manual binary download | The preprocessor logs the exact `solc-select install` command and its error; a fallback path uses the Docker image's pinned solc (Stage 7's hard gate) |
| `solc --flatten` is unstable for some projects (e.g. circular imports) | Fall back to `hardhat-flatten`; if both fail, the file is dropped with reason `flatten_failed` and the source path is recorded |
| The 3-level dedup is computationally expensive for large corpora (47K SmartBugs Wild contracts) | The near-dup clustering is opt-in per source via `config.yaml`; SmartBugs Wild is left for Stage 4 when the volume justifies the cost |
| Compile step is slow (each file may need a different solc version) | Multiprocessing pool; per-source worker count in `config.yaml`; pre-warm the most common solc versions via `solc-select install` in the Dockerfile |
| `hardhat-flatten` is not in the system PATH on all dev machines | Listed as a build dependency in the Dockerfile; the system install instructions in `README.md` cover macOS + WSL2 + native Linux |
| Sidecar `meta.json` schema drifts as the stage progresses | Schema is versioned via `meta_schema_version` field; downstream readers opt in to a specific version and reject unknown ones |

---

**End of Stage 1 actionable plan. Total estimated time: 5 working days (Jun 16–20), with Jun 21–22 as buffer.**
