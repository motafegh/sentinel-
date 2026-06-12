# ADR-0006: Splitting + Registry Design

**Date:** 2026-06-12
**Stage:** 5 of 8 (Week 8: Jul 28–Aug 3)
**Status:** Accepted (Stage 5 implementation complete)
**Author:** SENTINEL data engineering
**Plan reference:** [`docs/proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md`](../proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md)
**Audit reference:** [`docs/proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md`](../proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md) §2 5-P1 through 5-P6

---

## Context

The BCCC failure (38.8% cross-split duplication inflating Run 9's F1 by ~0.05) was caused by *no dedup enforcement at split time*. The 7 questions Stage 5 must answer are operational: "what dataset version did Run 11 train on?", "is this export file the same one we registered 6 months ago?", "did anyone hand-edit the export file?", "what NonVulnerable:positive ratio does this split have?", "do any near-duplicate contracts straddle a split boundary?", "what changed between dataset v1 and v2?", "is the catalog up to date with the YAML mirror?".

The module's purpose is to convert verified labels (Stage 4) into **leak-free, capped, traceable dataset versions** that downstream stages (6: analysis, 7: export, 8: Run 11) can load with confidence. Every artifact has a hash, a lineage DAG, a retirement chain, and a YAML-mirror CI check.

This ADR records the 8 design decisions that frame the module, the implementation choices made during Stage 5, and the operational consequences.

---

## Design Decisions

### D-5.1 — Four splitter strategies, dispatch by source

The module ships 4 strategies, each addressing a different axis of leakage / bias:

| Strategy | Use case | Grouping key | Special semantics |
|---|---|---|---|
| `random` | Sanity tests only — no guarantee | n/a | Uniform `random.Random(seed).shuffle`, ratio-based assignment |
| `stratified` (default) | General merged dataset | composite of `primary_class`, `source`, `tier` | Per-stratum proportional allocation; remainder goes to test |
| `project` (alias: `project_level` per AUDIT_PATCHES 5-P1) | Audit datasets (Bastet, ScaBench, Web3Bugs) | `project_id` | Whole project in one split; contracts without `project_id` go to train |
| `temporal` | Time-sensitive data | `year` | `year <= cutoff_year` split train/val; `year > cutoff_year` all test |

**Per-source strategy:** the plan calls for `pipeline.splitting.strategy_per_source.<source>` in `config.yaml`. The dispatch dict is `SPLITTERS` in `splitters.py`; the CLI hardcodes `stratified` in the current implementation (IC-5).

**Operational consequence:** a single CLI invocation handles the merged dataset with one strategy. To mix strategies per source, a downstream caller would need to invoke the splitter per source-set and merge the results.

### D-5.2 — Two-pass split (splitter → dedup_enforcer)

Pass 1 runs the chosen strategy. Pass 2 (`apply_dedup_enforcer`) re-checks every `dedup_group` (pre-computed during Stage 1 preprocessing using AST similarity, threshold 0.85) and reassigns any group that straddles a split boundary. The target split is the majority; ties go to train. This is the **BCCC-failure pattern fix**: 38.8% duplication is now physically impossible because all near-dup groups are forced to a single split.

**Why not just delete duplicates?** Deleting would shrink the training set and miss the audit-trail requirement (BCCC's v9 graph is preserved alongside v8 graph). Reassignment keeps all contracts and records the action in `split_manifest.json → metadata.reassignments`.

**Operational consequence:** every manifest records `dedup_groups_resolved` and `reassignments[]`. Reproducing a split requires the same `seed` + the same `dedup_group` field on the input contracts.

### D-5.3 — Leakage auditor is an independent post-split safety net

The dedup_enforcer uses AST similarity (pre-computed). The leakage_auditor uses **text-shingle Jaccard similarity** (computed at audit time, threshold 0.5 per AUDIT_PATCHES 5-P3). Different algorithm, different threshold → catches what the enforcer missed.

**Algorithm:**
- 3-character shingles on whitespace-normalized, lowercased source code
- Jaccard similarity >= 0.5 → `LeakPair`
- O(N²) pairwise comparison across split boundaries (train↔val, train↔test, val↔test)
- Uses `seen_pairs` set to avoid duplicate pair entries

**Reports only.** The auditor does NOT reassign. The plan is explicit: "The auditor is a safety net, not a block." A leak-pair is recorded in the manifest's `metadata.leakage_audit` field; downstream analysis surfaces it.

**Scaling:** O(N²) is ~500M comparisons for the 22K-contract v2 baseline, ~10–30 min. LSH (MinHash / LSH-bands) is a v2.1 enhancement.

**Operational consequence:** the audit must be run separately from the split CLI (IC-5). The CLI subcommand `sentinel-data split` does not invoke `leakage_auditor.run_audit()`; callers must invoke it directly via `from sentinel_data.splitting import run_audit`.

### D-5.4 — SQLite + YAML mirror catalog (4 base tables + 2 system tables)

The catalog is the answer to "what dataset version did Run 11 train on?". Six tables:

| Table | Purpose | Key columns |
|---|---|---|
| `sources` | Per-source pin + last-fetched | `name` (PK), `pin`, `last_fetched`, `enabled`, `n_contracts`, `tier` |
| `artifacts` | Per-exported-artifact hash + lineage | `name` (PK), `sha256`, `size_bytes`, `lineage` (JSON) |
| `splits` | Per-split-version seed + strategy | `version` (PK), `seed`, `strategy`, `contract_counts` (JSON) |
| `dataset_versions` | Named composite (sources + preprocessing + split) | `name` (PK), `source_set`, `preprocessing_config_hash`, `split_version`, `label_schema_version`, `verification_report_path`, `artifact_hash`, `artifact_path` |
| `schema_migrations` (system) | Every schema change recorded | `version` (PK), `description`, `applied_at` |
| `dataset_version_retirements` (system) | Permanent retirement chain | `name` (PK), `superseded_by`, `retired_at`, `reason` |

**YAML mirror:** `write_yaml_mirror()` exports the full catalog as multi-document YAML on every write. CI checks DB↔YAML sync (the plan's exit criterion #5; not yet wired to a CI job — the function exists and is called by `_run_register`).

**Schema versioning:** `SCHEMA_VERSION = 1`. `Catalog.migrate(version, description)` records the migration; the caller is responsible for the `ALTER TABLE`. v2.1 will introduce v2 (per-class metadata in `dataset_versions`).

**Operational consequence:** the catalog is the single source of truth for "what runs are reproducible?". Every Stage 7 export MUST register its `DatasetVersion`; every Stage 8 load MUST call `load_artifact()`.

### D-5.5 — Lineage is a DAG stored as JSON

Every `Artifact` has a `lineage: dict` with two fields:
- `steps`: list of `{step, ts, **details}` (e.g. `{"step": "ingest", "ts": "...", "source": "solidifi", "n_contracts": 1104}`)
- `parents`: list of parent artifact sha256s

The DAG is a JSON field, not a separate table. Rationale: the lineage is part of the artifact's identity (changing the lineage changes the hash), and a JSON field is sufficient for the v2 baseline (≤10 steps per artifact). If the lineage grows beyond ~50 steps, v2.1 introduces a separate `lineage_edges` table.

**`lineage_to_dot(lineage) -> str`** renders the DAG as Graphviz DOT for visualization in the lineage tracker.

**Operational consequence:** the lineage is a forensic tool, not a runtime structure. ML loads use `verify_artifact()` (hash check) — they do not traverse the lineage DAG.

### D-5.6 — Hash verification at load time (D-5.6 + AUDIT_PATCHES 5-P5)

`Catalog.verify_artifact_hash(path) -> bool` computes SHA-256 of file bytes (streaming 64KB chunks) and checks against the `artifacts.sha256` and `dataset_versions.artifact_hash` tables. Returns True if EITHER matches.

**`lineage_tracker.verify_artifact(path, expected_hash) -> bool`** is the load-time gate. The ML module's `SentinelDataset.__init__` calls this before reading any rows; a hash mismatch raises an exception that aborts the load.

**Why SHA-256 of file bytes, not per-row hashes:** per-row hashes are O(N) writes and reads; a single file hash is O(1). The trade-off is that a single corrupted row can be detected but not located; this is acceptable because the rows are content-addressed (each contract has a `sha256` column) and the consumer can compare per-row sha256 if needed.

**Why verify on load, not on write:** the file may be hand-edited or corrupted after write. The load-time gate is the last line of defense.

**Operational consequence:** if a dataset version's export file is moved or renamed, the load fails. The contract is: registered `artifact_path` MUST point to the canonical file.

### D-5.7 — Dataset versions are named and append-only

A `DatasetVersion` has a unique `name` (e.g. `sentinel-v2-gold-2026-08`). Once registered, the version is immutable. Any change to the source set, preprocessing config, split version, or export format creates a new version.

`retire_dataset_version(name, superseded_by, reason)` marks a version as retired but does NOT delete it. The retirement chain (`v1 → v2 → v3`) is preserved in `dataset_version_retirements` for audit.

`Catalog.list_dataset_versions(include_retired=False)` orders by `generated_at DESC` and excludes retired by default.

**`dataset_diff.diff_dataset_versions(metadata_old, metadata_new) -> DatasetDiff`** computes the per-class and per-contract delta between two versions, including `added_contracts`, `removed_contracts`, `label_changes`, and a coarse `predicted_f1_delta_pct` heuristic.

**`dataset_diff.update_changelog(changelog_path, version_name, summary, metrics) -> None`** appends a markdown entry with a per-class delta table.

**Operational consequence:** the audit trail is permanent. v1.4 BCCC labels, v8 BCCC graphs, v9 graphs — all preserved with `superseded_by` chain. The `changelog.md` is the human-readable audit log.

### D-5.8 — NonVulnerable 3:1 cap (D-5.8 NEW per friend review)

DISL has 514,506 unlabeled contracts. With ~1,200 positives from 5 critical-path sources, the default NonVulnerable:positive ratio is 514,506:1,200 ≈ 428:1. This is the **same BCCC failure pattern at larger scale** — a model that defaults to "predict negative" is right 99%+ of the time and never learns positive patterns.

**Cap:** `pipeline.negative.positive_ratio_max: 3.0` (configurable). Total NonVulnerable contracts across all splits is at most 3× total positives.

**Stratification:** the cap is enforced per-split AND per-source. The subsample is stratified by source (preserves the per-source distribution), with the largest-remainder method distributing integer contracts to sources by their fractional remainders (sorted by remainder descending, ties broken by source name ascending for determinism).

**Algorithm (`nonvulnerable_cap.py:apply_nonvulnerable_cap`):**
1. Count total positives across all splits.
2. Compute `max_nonvuln = int(cap × total_positive)`.
3. For each split, separate NonVulnerable from positives.
4. If split's NonVulnerable count > `max_nonvuln`:
   - Group by source, compute per-source cap proportional to source's share of the global NonVulnerable pool.
   - Distribute integer counts using largest-remainder method.
   - `random.Random(seed).shuffle` per source, take the per-source cap.
5. Update `metadata.nonvulnerable_cap` with full audit: `cap`, `total_positive`, `max_nonvuln`, `per_source` breakdown, `per_split` `{original, capped, kept}`.

**Why 3:1, not higher or lower:** 5:1 and 10:1 reproduce the BCCC contamination problem. 1:1 starves the NonVulnerable signal (the model over-fits to positive patterns). 3:1 is the empirical sweet spot.

**v2 baseline:** the merged corpus has 22,356 contracts, 2,658 NonVulnerable (13% of positive count), well under the cap — no subsampling was needed for v2.

**Per-class overrides:** the plan describes `pipeline.negative.per_class_ratio_max.<ClassName>: 5.0` for per-class tuning. Not implemented in v1 (global cap only); v2.1 will add the per-class lookup.

**Operational consequence:** every split manifest records the cap and the per-source per-split audit. Reproducing a split requires the same `seed` AND the same `pipeline.negative.positive_ratio_max` config.

---

## Implementation Choices Made During Stage 5

These are decisions that emerged during implementation, not in the original plan:

### IC-1 — Dedup enforcer majority rule with train-tiebreak (`dedup_enforcer.py`)

**Plan (D-5.2):** "the group goes to the split where the majority of its members are".

**Implementation:** the target split is `max(("train", "val", "test"), key=lambda s: (counts[s], s == "train"))` — the `s == "train"` second key forces train on ties.

**Why:** the plan's majority rule is ambiguous on ties. A 3-3-3 group (9 contracts, 3 in each split) has no majority. Picking train (the largest split by default ratio) is conservative — it minimizes the number of contracts moved OUT of train, which would otherwise require retraining with a smaller training set.

**Empirical behavior on synthetic data:** with 100 contracts in 10 dedup groups of 10, ALL 100 contracts move to train (each group straddles all 3 splits and the largest count is in train because 70% of contracts started there). This is **expected**, not a bug.

### IC-2 — Leakage auditor reports only, no reassignment (`leakage_auditor.py`)

**Plan (D-5.3):** "The auditor is a safety net, not a block."

**Implementation:** `find_leaks` returns a `LeakageReport` with `pairs: list[LeakPair]`. The caller is expected to either log the report or block downstream stages; the function itself does not modify the input `Splits`.

**Why no automatic reassignment:** an automated reassigner would need a tiebreak policy (which split should receive the leaked group?) and a threshold for "how many leaks is too many?". The plan's safety-net semantics defer this to the operator. A leaked pair in the report is a prompt to either re-run the split with a different `seed` or to investigate the source corpus.

### IC-3 — NonVulnerable cap uses largest-remainder stratified sampling (`nonvulnerable_cap.py`)

**Plan (D-5.8):** "stratified by source".

**Implementation:** integer per-source caps are distributed using the **largest-remainder method** (a.k.a. Hamilton's method). The fractional caps are sorted by remainder descending, ties broken by source name ascending for determinism. Each source gets `floor(share)` contracts first, then the top-N sources by remainder each get 1 extra.

**Why largest-remainder:** simple quotas (e.g. `int(share * cap)`) systematically under-allocate total NonVulnerable; the largest-remainder method ensures `sum(per_source_caps) == max_nonvuln` exactly. This is the same method used in legislative apportionment.

**Why deterministic tiebreak:** with `seed=42` and a stable source list, the cap produces the same per-source distribution across runs.

### IC-4 — CLI hashes the manifest, not the full split directory (`cli.py:_run_register`)

**Plan (D-5.6):** "SHA-256 of file bytes" — implies the file is the artifact.

**Implementation:** the CLI computes the hash of `split_manifest.json`, not of the `train.jsonl` + `val.jsonl` + `test.jsonl` files.

**Why:** the manifest is the canonical description of the split. The JSONL files can be regenerated from the input labels + seed + dedup_enforcer config; the manifest cannot be regenerated without these inputs. Hashing the manifest means a regenerated split with the same inputs passes the hash check (this is intentional — the artifact is the "split recipe and metadata", not the file contents).

**v2.1 enhancement:** hash the full split directory (manifest + 3 JSONL files) for stricter tamper detection.

### IC-5 — Per-source strategy from config.yaml is NOT wired (`cli.py:_run_split`)

**Plan (D-5.1):** "Per-source strategy in `config.yaml` — `pipeline.splitting.strategy_per_source.<source>`".

**Implementation:** the CLI hardcodes `stratified_split(contracts, seed=args.seed)`. The `SPLITTERS` dispatch dict supports per-source dispatch but the CLI does not consult `config.yaml` for the per-source override.

**Why deferred:** per-source strategy is an enhancement on top of the v2 baseline (which is purely stratified). The CLI's `--strategy` argument accepts any of the 4 strategies for callers that want to override manually. The config-driven per-source dispatch is a v2.1 enhancement.

### IC-6 — Leakage auditor is NOT wired to the CLI (`cli.py:_run_split`)

**Plan (D-5.3):** the auditor is "an independent post-split check".

**Implementation:** the `leakage_auditor.run_audit()` function exists and works, but the CLI's `_run_split` does not call it. Callers must import and invoke it directly.

**Why deferred:** the auditor is slow (~10–30 min for the v2 baseline). The CLI is intended for fast smoke tests; the auditor is an opt-in step for production splits. v2.1 will add a `--run-leakage-audit` flag to wire it in.

**Operational consequence:** a split produced by the CLI today does NOT have `metadata.leakage_audit` populated. Production splits must run the auditor manually.

---

## Operational Consequences

1. **The split stage is required before register.** `_run_split` writes `splits/v{version}/`; `_run_register` reads `splits/v{version}/split_manifest.json`. Calling register without a prior split fails with "split manifest not found".

2. **The dedup_enforcer is always applied** in the CLI. The two-pass split is not optional; there is no `--no-dedup` flag. The v2 baseline has 0% expected dedup (no near-dup groups in the merged corpus), so the enforcer is a no-op in practice.

3. **The NonVulnerable cap is configurable** via `--nonvuln-cap` (default 3.0). Setting it to 0 disables the cap entirely (no NonVulnerable contracts kept — useful for "positive-only" stress tests). Setting it to 1.0 enforces 1:1.

4. **The catalog is the single source of truth.** Every Stage 7 export MUST register its `DatasetVersion`. The hash check is the load-time gate; a missing or retired version's `load_artifact()` returns `None`.

5. **The YAML mirror is written on every register call.** The DB and YAML are kept in sync by `write_yaml_mirror()`. CI checks the sync by comparing `cat.write_yaml_mirror()` output to the on-disk `catalog.yaml` (the function exists; the CI job is not yet wired).

6. **The leakage auditor is manual.** A split produced by the CLI does NOT include a leakage audit. Callers must invoke `run_audit()` directly. Production splits should always run the auditor and check the report.

7. **Dataset versions are immutable.** A re-register of the same name overwrites the row (sqlite's `INSERT OR REPLACE`). To "update" a dataset, register a new version and retire the old.

---

## References

- Plan: [`docs/proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md`](../proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md)
- Audit: [`docs/proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md`](../proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md) §2 5-P1 through 5-P6
- Stage 5 audit: `Data/audit/08_splitting_registry_stage5_audit.md` (TBD)
- Implementation:
  - `Data/sentinel_data/splitting/splitters.py` (4 strategies + Contract/Splits/SplitMetadata)
  - `Data/sentinel_data/splitting/dedup_enforcer.py` (BCCC-failure pattern fix)
  - `Data/sentinel_data/splitting/leakage_auditor.py` (shingle-based post-split safety net)
  - `Data/sentinel_data/splitting/nonvulnerable_cap.py` (3:1 cap, stratified by source)
  - `Data/sentinel_data/registry/catalog.py` (SQLite + YAML mirror + migrations + retirement)
  - `Data/sentinel_data/registry/lineage_tracker.py` (lineage DAG + DOT renderer + verify_artifact)
  - `Data/sentinel_data/registry/dataset_diff.py` (per-class delta + changelog)
  - `Data/sentinel_data/cli.py` (split + register subcommands)
- Tests:
  - `Data/tests/test_splitting/test_splitters.py` (37 tests pass)
  - `Data/tests/test_registry/test_catalog.py` (33 tests pass)
- Smoke output: `data/splits/v1/split_manifest.json` (15,644 / 3,344 / 3,368 — 70/15/15 split)

---

**End of ADR-0006.**
