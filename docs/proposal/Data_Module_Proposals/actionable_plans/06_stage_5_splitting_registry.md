# Actionable Plan — Stage 5: Splitting + Registry

**Date:** 2026-07-14
**Stage:** 5 of 8 (Week 6: Jul 14–20)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.6, §3.7, §5 (Week 6)
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §1 (5-P1 through 5-P7)
**Exit criteria:** `sentinel-data split --config split-config.yaml` produces `splits/v1/{train,val,test}.parquet` with a versioned manifest; `sentinel_data.registry.load_artifact("sentinel-v2-dryrun-2026-07")` returns the artifact; the leakage auditor reports 0 near-duplicates across split boundaries; the catalog has at least one named dataset version; **the catalog has a schema migrations table + a dataset version retirement chain (preserving the v1.4 BCCC labels, v8 BCCC graphs, etc.)**.

---

## Goal

Implement the **Splitting** submodule (5 splitter types + leakage auditor) and the **Registry** submodule (catalog + lineage + hasher + diff + changelog). After this stage, the dataset has versioned, leak-free, stratified train/val/test splits, and every exported artifact is registered in the catalog with full lineage and a SHA-256 hash that the ML module verifies on load.

The two submodules are coupled: a split is a registry artifact, and the registry tracks which splits are referenced by which dataset versions. Building them together avoids a half-step where splits exist but are not yet registered.

---

## Why this stage sixth

Stages 1–4 produced verified labeled data. Stage 5 is the first stage that prepares the data for *training*: the split is the contract that says "this contract is in train, that one is in test, and no contract is in both." Doing splitting after verification is critical — a split on unverified labels is a split on noise.

The registry is the catalog that makes every artifact discoverable. Without it, "what dataset version did Run 11 train on?" is unanswerable 6 months later. The registry is also the gate that the ML module's `SentinelDataset` checks at load time — without the registry, the seam swap in Stage 7 has no enforcement mechanism.

---

## Design decisions

### D-5.1 — Three split strategies, with project-level as the default for audit datasets

The splitter supports 4 strategies: random, stratified, project-level, and temporal. For audit datasets (Bastet, ScaBench, Web3Bugs, DeFiHackLabs), the default is **project-level** (a project is entirely in one split, not split across train/val/test). For tool-derived datasets (Slither-Audited, SmartBugs Wild, Messi-Q), the default is **stratified with project-level fallback** (per AUDIT_PATCHES 5-P1). The fallback is to use the source's own project grouping if available — pure stratified may put 90% of one tool's contracts in one split.

The strategy is per-source in `config.yaml`: `sources.bastet.split.strategy: project_level`. The splitter applies the per-source strategy and merges.

### D-5.2 — Splitting is a two-pass operation

Pass 1: stratified splitter assigns contracts to splits per the strategy. Pass 2: `dedup_enforcer` reassigns any near-dup group that straddles a split boundary. This is the BCCC failure pattern: dedup is run *after* splitting, so near-dup groups leak across train/test.

The two-pass design means the splitter is a `StratifiedSplitter` + `DedupEnforcer` composition, not a single function. The output of pass 1 is the "stratified splits"; the output of pass 2 is the "leak-free splits."

### D-5.3 — Leakage auditor is an independent post-split check

After the two-pass split, the `leakage_auditor` does its own similarity check using a different algorithm (e.g. shingled text similarity instead of AST similarity) and reports any leak it finds. If the auditor finds something the `dedup_enforcer` missed, that's a bug to fix — the auditor is a safety net.

The auditor's threshold (default 0.5 shingled similarity) is in `config.yaml`. The auditor's output is a `leakage_report.md` per split version.

### D-5.4 — Registry is SQLite + YAML mirror with migrations + retirement chain

The catalog is a SQLite database at `data/registry/catalog.db` for fast lookup, with a human-readable mirror at `data/registry/catalog.yaml` for version control. Every change to the DB writes a corresponding YAML entry; CI checks that the two stay in sync.

The DB schema has **4 base tables + 2 system tables** (per AUDIT_PATCHES 5-P2, 5-P3):
- `sources` (per-source pin + last-fetched)
- `artifacts` (per-exported-artifact hash + lineage)
- `splits` (per-split-version seed + strategy)
- `dataset_versions` (named composite: source set + preprocessing config + split version)
- **`schema_migrations`** — tracks every schema change (initial = 4 tables; subsequent migrations add columns). First migration creates the 4 base tables; future migrations are tracked here.
- **`dataset_version_retirements`** — old dataset versions are marked as "superseded" but not deleted. The audit trail is permanent. The v1.4 BCCC labels, the v8 BCCC graphs, the v9 graphs, the v10 graphs — all are preserved in the catalog with their `superseded_by` chain.

The `dataset_versions` table is the entry point — a `sentinel_data.registry.load_artifact(name)` call joins across the 4 base tables to return the full artifact.

### D-5.5 — Lineage is a graph, not a flat list

Every artifact in the registry has a lineage: which ingestion connectors, which preprocessing steps, which labeling parsers, which verification components, which splitting strategy, which export writers produced it. The lineage is a DAG (not a list) because some steps (e.g. verification) can run in parallel with others (e.g. splitting). The lineage is stored as a JSON field on each `artifacts` row; `lineage_tracker.py` provides the API to query the graph.

### D-5.6 — Hash verification is the load-time gate

The ML module's `SentinelDataset.__init__` (the new file in `sentinel-ml/src/datasets/sentinel_dataset.py`, written in Stage 7) calls `sentinel_data.registry.verify_artifact_hash(export_path)` before loading. If the hash doesn't match the registered hash, the load fails. This is the mechanism that prevents "I edited the export file by hand and the model trained on the wrong data" — the registry says what the file should be, and any deviation is a load error.

### D-5.7 — Dataset versions are named and append-only

A dataset version is a named entry in the catalog: `sentinel-v2-gold-2026-08`. The name is a *promise* — it means "this is the gold-standard v2 dataset as of August 2026." Once registered, the version is immutable; updates create a new version (`sentinel-v2-gold-2026-09`).

The `dataset_diff` tool shows what changed between two versions: which contracts were added/removed, which labels changed, which class distribution shifted. This is the audit trail for "why did Run 12 train differently from Run 11?" — the answer is in the diff.

---

## Tasks — ordered, each with verifiable exit condition

### 5.1 — Implement the 4 splitter strategies

Author `sentinel_data/splitting/{stratified_splitter,project_splitter,random_splitter,temporal_splitter}.py`. Each splitter takes a list of contracts + labels and returns a `{train: [...], val: [...], test: [...]}` assignment per the strategy. The split ratios are in `config.yaml` (default 70/15/15). The seed is recorded for reproducibility.

**Why first:** the 4 splitters are the primitives; the dedup_enforcer and leakage_auditor wrap them.

**Exit condition:** each splitter runs against a small fixture (100 contracts) and produces splits that match the expected assignment (e.g. stratified splitter preserves per-class distribution within ±2%).

**Commit:** `feat(data-split): add 4 splitter strategies (random, stratified, project, temporal)`

---

### 5.2 — Implement `dedup_enforcer.py` (the BCCC-failure pattern fix)

Author `sentinel_data/splitting/dedup_enforcer.py` that takes the output of any splitter and reassigns any near-dup group that straddles a split boundary. The reassignment rule: the group goes to the split where the majority of its members are; ties go to train. The enforcer records all reassignments in the split manifest.

**Why:** the BCCC 38.8% duplication rate meant many contracts appeared in both train and test, inflating Run 9's F1 by ~0.05 (estimated). The enforcer eliminates this.

**Exit condition:** a fixture with a known near-dup group straddling the split boundary is correctly reassigned; the manifest records the reassignment.

**Commit:** `feat(data-split): add dedup_enforcer to prevent split leakage`

---

### 5.3 — Implement `leakage_auditor.py` (the safety net)

Author `sentinel_data/splitting/leakage_auditor.py` that does an independent post-split similarity check (shingled text similarity, default threshold 0.5) and reports any leak. The output is a `leakage_report.md` per split version with the list of any near-dup pairs across the boundary.

**Why:** the dedup_enforcer uses AST similarity; the auditor uses text similarity. The two methods can disagree, and the auditor is the safety net for cases the enforcer misses.

**Exit condition:** the auditor runs against a fixture with a known leak (a near-dup pair placed across train/test) and reports it; the report includes the pair's similarity score and the splits involved.

**Commit:** `feat(data-split): add leakage_auditor for post-split safety check`

---

### 5.4 — Implement `split_manifest.py` (coexist with old splits for 1 release)

Author `sentinel_data/splitting/split_manifest.py` that writes the versioned JSON contract per the proposal §3.6. The manifest contains: version, seed, strategy (per source), contract counts per split, class distributions per split, dedup groups resolved, reassignments, leakage auditor results, generated_at.

**Per AUDIT_PATCHES 5-P5:** the existing `ml/data/splits/v10_deduped/` and the new `Data/data/splits/v1/` **coexist for 1 release**. The old splits are used by Run 9 / any active training; the new splits are for Run 11. The seam swap in Stage 7 retires the old.

**Why:** the manifest is the contract that downstream stages and the registry depend on.

**Exit condition:** the manifest is written for a fixture split; all fields are populated; the manifest is loadable by the registry; both old (v10_deduped) and new (v1) splits are reachable.

**Commit:** `feat(data-split): add split_manifest with versioned JSON contract (coexist with v10 for 1 release)`

---

### 5.5 — Implement the registry catalog (SQLite + YAML mirror + migrations + retirement + shared hash)

Author `sentinel_data/registry/catalog.py` with the 4 base tables + 2 system tables (per D-5.4): sources, artifacts, splits, dataset_versions, schema_migrations, dataset_version_retirements. The catalog exposes: `add_source()`, `add_artifact()`, `add_split()`, `add_dataset_version()`, `load_artifact(name) -> Artifact`, `verify_artifact_hash(path) -> bool`, `migrate()`, `retire_dataset_version(name, superseded_by)`. Every DB write produces a corresponding YAML entry in `data/registry/catalog.yaml`; CI checks that the two stay in sync.

**Per AUDIT_PATCHES 5-P4:** the `verify_artifact_hash()` function must use the **same algorithm** as `ml/src/inference/cache.py:InferenceCache` for content addressing. Both are SHA-256 of the file bytes; the key includes the schema version. The two functions call the same shared `sentinel_data.registry.compute_hash()` to avoid drift. (The inference cache continues to use its own helper until the seam swap in Stage 7; after the swap, both call `sentinel_data.registry.compute_hash`.)

**Per AUDIT_PATCHES 5-P6 (N-8):** the catalog has a **Python client API** (`sentinel_data.registry.CatalogClient`) that wraps the catalog for use by the ML module and the inference server. The ML module's `SentinelDataset.__init__` (Stage 7) uses this client; the inference server uses it for `/health` and `/metrics` endpoints.

**Why:** the catalog is the single source of truth for "what dataset version is the v2 baseline?" The retirement chain preserves the v1.4 BCCC labels, v8/v9/v10 graphs, etc. — the audit trail is permanent.

**Exit condition:** the catalog is created with 4 base tables + 2 system tables; a fixture artifact is added, loaded, and verified; the YAML mirror is generated and matches the DB; the v1.4 BCCC labels are added to the retirement chain (preserved, not deleted).

**Commit:** `feat(data-registry): add SQLite catalog with YAML mirror + migrations + retirement + shared hash`

---

### 5.6 — Implement `lineage_tracker.py` and `artifact_hasher.py`

Author `sentinel_data/registry/lineage_tracker.py` that records the DAG of transformations for every artifact. Author `sentinel_data/registry/artifact_hasher.py` that computes SHA-256 of every exported file and verifies the hash on load.

**Why:** lineage is the audit trail; hashing is the load-time gate.

**Exit condition:** an artifact's lineage is recorded and queryable; the hasher computes and verifies SHA-256 correctly for a fixture file.

**Commit:** `feat(data-registry): add lineage_tracker + artifact_hasher`

---

### 5.7 — Implement `dataset_diff.py` (with per-class metric projection) and `changelog.md`

Author `sentinel_data/registry/dataset_diff.py` that takes two dataset versions and reports the diff (added/removed contracts, label changes, class distribution deltas). **Per AUDIT_PATCHES 5-P7, the diff must produce a per-class metric projection** — for each class, show the new vs old count, the new vs old label distribution, the new vs old confidence tier breakdown. The model team uses this to predict "Run 11's per-class F1 will likely be X% better than Run 9's because the v2 corpus has 30% more Reentrancy positives." Author the `changelog.md` template that is updated with every dataset version registration.

**Why:** the diff is the answer to "why did Run N+1 train differently from Run N?" The per-class metric projection is the proactive F1 estimate.

**Exit condition:** the diff tool reports a clean diff for two fixture versions (one empty, one with 10 contracts); the per-class metric projection is generated; the changelog is updated with the new version entry.

**Commit:** `feat(data-registry): add dataset_diff with per-class metric projection + changelog`

---

### 5.8 — Wire the `sentinel-data split` and `sentinel-data register` CLI subcommands

Connect `cli.py` `split` and `register` subcommands to the new modules. The `split` command reads sources from `config.yaml`, runs the per-source splitter, runs the dedup_enforcer, runs the leakage_auditor, and writes the splits + manifest. The `register` command reads a completed split manifest + verification report + export, computes hashes, and registers the dataset version in the catalog.

Update `dvc.yaml` stages `split` and `register` to call the new CLI commands.

**Exit condition:** `sentinel-data split` produces `data/splits/v1/{train,val,test}.parquet` + `split_manifest.json`; `sentinel-data register --name sentinel-v2-dryrun-2026-07` adds the version to the catalog; `sentinel_data.registry.load_artifact("sentinel-v2-dryrun-2026-07")` returns the artifact.

**Commit:** `feat(data-split,data-registry): wire CLI + DVC for split + register stages`

---

### 5.9 — Add tests for splitting + registry

Author `Data/tests/test_splitting/` and `Data/tests/test_registry/` with:
- **Splitter tests** — each of the 4 strategies produces correct splits; project-level splitter keeps projects in one split; stratified preserves per-class distribution
- **Dedup enforcer tests** — straddling groups are reassigned; reassignments are recorded
- **Leakage auditor tests** — known leaks are reported; clean splits pass
- **Catalog tests** — add/load/verify round-trip; YAML mirror matches DB
- **Lineage tests** — lineage graph is queryable
- **Hash tests** — SHA-256 computed and verified correctly
- **Diff tests** — clean diff for two fixture versions

**Exit condition:** `poetry run pytest tests/test_splitting tests/test_registry -v` passes; coverage > 80%.

**Commit:** `test(data): add full test suites for splitting + registry`

---

### 5.10 — Author `ADR-0006-splitting-and-registry-design.md`

Document the key design decisions: 3 strategies (D-5.1), two-pass splitting (D-5.2), leakage auditor as safety net (D-5.3), SQLite + YAML catalog (D-5.4), lineage as graph (D-5.5), load-time hash verification (D-5.6), append-only dataset versions (D-5.7).

**Exit condition:** file exists; references the BCCC duplication failure as motivation; cites the leakage auditor as the safety net.

**Commit:** `docs(data): add ADR-0006 for splitting + registry design`

---

## What NOT to fix (preservation list)

| Bug / Decision | Status | File:line | Stage 5 action |
|---|---|---|---|
| 99% DoS↔Reentrancy co-occurrence in BCCC | Source: BCCC | (not in v2 corpus) | The Stage 3 merger de-duplicates; the Stage 4 co-occurrence matrix flags it. Stage 5's dedup_enforcer is a complementary safety net (catches near-dup *groups*, not label co-occurrence). |
| Inference cache hash mismatch (per F-cache) | N/A (Stage 7 swaps the cache) | `ml/src/inference/cache.py` | The shared `compute_hash()` (5-P4) is the unification point. Until Stage 7, the inference cache uses its own helper. |
| 38.8% BCCC duplication | Source: BCCC | (not in v2 corpus) | The Stage 1 dedup at 0.85 handles it; Stage 5's dedup_enforcer is the post-split safety net. |
| Cross-split leakage in v6 era (34.9%) | Source: BCCC | (not in v2 corpus) | The two-pass split (stratified → dedup_enforcer) is the structural fix. Stage 5 implements it; the leakage_auditor is the safety net. |

## Final exit criteria check

| # | Check |
|---|---|
| 1 | All 4 splitter strategies (random, stratified, project, temporal) run against fixtures and produce correct splits |
| 2 | `dedup_enforcer` correctly reassigns straddling near-dup groups; reassignments are recorded |
| 3 | `leakage_auditor` reports 0 leaks on a clean split; reports known leaks on a contaminated fixture |
| 4 | `split_manifest.json` is written with all fields populated |
| 5 | SQLite catalog is created with 4 tables; YAML mirror is generated and matches DB |
| 6 | `sentinel_data.registry.load_artifact("sentinel-v2-dryrun-2026-07")` returns the fixture artifact |
| 7 | `verify_artifact_hash()` returns True for a valid file; False for a tampered file |
| 8 | `dataset_diff` reports a clean diff for two fixture versions |
| 9 | `dvc repro split register` runs end-to-end |
| 10 | `poetry run pytest tests/test_splitting tests/test_registry -v` passes with > 80% coverage |
| 11 | `ADR-0006-splitting-and-registry-design.md` is committed |

All 11 pass → **Stage 5 complete**. Tag `data-stage-5`, proceed to Stage 6.

---

## Risk register

| Risk | Mitigation |
|---|---|
| The dedup_enforcer is slow (recomputing near-dup similarity for every straddling group) | The near-dup clusters are pre-computed in Stage 1 (preprocessing); the enforcer looks up the cluster, doesn't recompute |
| The leakage auditor's text-similarity method is too strict or too lenient | The threshold is in `config.yaml` and the auditor's behavior is unit-tested; the safety net is a *report*, not a *block* — it warns but doesn't fail the split |
| The catalog YAML mirror drifts from the DB if a write fails mid-operation | The DB write + YAML write are wrapped in a transaction; if either fails, both are rolled back |
| The hash verification is slow for large artifacts (1 GB+ sharded exports) | The hash is computed incrementally as the file is written; verification is a streaming read of the same size |
| Splitting 41K contracts with project-level strategy puts 90% of contracts in one split (if the corpus is dominated by one project) | The splitter detects this and emits a warning; the user can choose to enable sub-project splitting (split by file, not by project) |
| The `load_artifact` API is slow for large catalogs (1000+ versions) | The catalog has an index on `dataset_versions.name`; the typical query is O(log n) |

---

**End of Stage 5 actionable plan. Total estimated time: 5 working days (Jul 14–18), with Jul 19–20 as buffer.**
