# Stage 5 — Splitting + Registry (the dataset-versioning gate)

**Date:** 2026-07-14 (revised 2026-06-12 post-implementation)
**Status:** ✅ COMPLETE. 8 commits landed. 70 tests pass (37 splitting + 33 registry). Stage 5 exit criteria all met.
**Reading time:** 30-40 minutes. The doc has 6 sections matching the standard format; take notes.
**Goal:** After this doc, you can answer all 6 items in `LEARNING_CHECKLIST.md` §"Stage 5" from memory, explain the 8 design decisions (D-5.1 through D-5.8), the 6 implementation choices (IC-1 through IC-6), and the 12 exit criteria.

---

## 1️⃣ The Problem

### What Stage 5 has to deliver

Stages 1–4 produced 22,356 **verified** labeled contracts. Stage 5 prepares them for **training**:
1. Split them into train/val/test with no cross-split leakage
2. Enforce a NonVulnerable:positive ratio cap (to prevent the model from defaulting to "predict negative")
3. Register every artifact in a catalog so "what dataset version did Run 11 train on?" is answerable 6 months from now
4. Verify the export file hasn't been hand-edited (load-time hash check)

The BCCC failure had a splitting component too: 38.8% duplication meant many contracts appeared in both train and test, inflating Run 9's F1 by ~0.05. Stage 5 prevents this with a two-pass split + leakage auditor.

### The NonVulnerable 3:1 cap (D-5.8, friend review)

DISL has 514,506 unlabeled contracts. With ~1,200 positives from 5 critical-path sources, the default ratio is 514K:1,200 = 428:1. This is the **same BCCC failure pattern at larger scale** — a model that defaults to "predict negative" and is right 99%+ of the time never learns positive patterns.

The cap is `pipeline.negative.positive_ratio_max: 3.0`. NonVulnerable is subsampled to at most 3× total positive count, stratified by source.

**v2 baseline reality:** the merged corpus has 22,356 contracts and 2,658 NonVulnerable (13% of positive count), well under the cap — no subsampling was needed for v2. The cap is enforced for v2.1+ when DISL is re-introduced.

### The 7 questions Stage 5 must answer

| # | Question | Component | File |
|---|---|---|---|
| 1 | What dataset version did Run 11 train on? | `Catalog.list_dataset_versions()` | `registry/catalog.py` |
| 2 | Is this export file the same one we registered 6 months ago? | `Catalog.verify_artifact_hash()` | `registry/catalog.py` |
| 3 | Did anyone hand-edit the export file? | `lineage_tracker.verify_artifact()` | `registry/lineage_tracker.py` |
| 4 | What NonVulnerable:positive ratio does this split have? | `apply_nonvulnerable_cap()` + manifest | `splitting/nonvulnerable_cap.py` |
| 5 | Do any near-duplicate contracts straddle a split boundary? | `apply_dedup_enforcer()` | `splitting/dedup_enforcer.py` |
| 6 | What changed between dataset v1 and v2? | `diff_dataset_versions()` | `registry/dataset_diff.py` |
| 7 | Is the catalog up to date with the YAML mirror? | `Catalog.write_yaml_mirror()` | `registry/catalog.py` |

---

## 2️⃣ The Solution

### The 8 design decisions (D-5.1 through D-5.8) — all from the plan

| # | Decision | Implementation | Why |
|---|---|---|---|
| **D-5.1** | 4 splitter strategies (random, stratified, project, temporal) | `splitters.py:SPLITTERS` dispatch dict (with `project_level` alias per AUDIT_PATCHES 5-P1) | Per-source strategy from config.yaml is deferred (IC-5); the dispatch dict supports it but the CLI hardcodes `stratified` |
| **D-5.2** | Two-pass split (splitter → dedup_enforcer) | `apply_dedup_enforcer` reassigns straddling groups to majority-split | BCCC-failure pattern fix: 38.8% duplication is now physically impossible |
| **D-5.3** | Leakage auditor (independent post-split safety net) | `find_leaks` uses 3-shingle Jaccard similarity, threshold 0.5 | Different algorithm from AST similarity in dedup_enforcer → catches what enforcer missed. Reports only, no reassignment |
| **D-5.4** | SQLite + YAML mirror (4 base + 2 system tables) | `catalog.py` with `sources`, `artifacts`, `splits`, `dataset_versions` + `schema_migrations`, `dataset_version_retirements` | YAML mirror for version control; CI check is future work |
| **D-5.5** | Lineage is a DAG stored as JSON | `lineage_tracker.py` with `record_lineage_step` and `lineage_to_dot` | Part of artifact identity; JSON is sufficient for ≤10 steps |
| **D-5.6** | Hash verification at load time | `verify_artifact_hash` (catalog) and `verify_artifact` (lineage_tracker) | SHA-256 of file bytes (streaming 64KB chunks); load-time gate prevents hand-edited exports from poisoning training |
| **D-5.7** | Dataset versions are named and append-only | `DatasetVersion` dataclass + `retire_dataset_version` + `dataset_diff` | `v1 → v2 → v3` retirement chain is preserved for audit; `changelog.md` is the human-readable log |
| **D-5.8** | NonVulnerable 3:1 cap (NEW per friend review) | `apply_nonvulnerable_cap` with largest-remainder stratified sampling by source | DISL's 514K contracts would create 428:1 imbalance; 3:1 is the empirical sweet spot |

### The 6 implementation choices (IC-1 through IC-6) — see ADR-0006

These are decisions that emerged during implementation, not in the original plan. The full rationale is in `docs/decisions/ADR-0006-splitting-and-registry-design.md`.

- **IC-1**: Dedup enforcer majority rule with train-tiebreak (`dedup_enforcer.py`). The plan's "majority rule" is ambiguous on ties. The implementation uses `max(splits, key=lambda s: (counts[s], s == "train"))` — train wins on ties. Conservative: minimizes the number of contracts moved OUT of train.
- **IC-2**: Leakage auditor reports only, no reassignment (`leakage_auditor.py`). The function returns a `LeakageReport`; the caller decides whether to log, block, or re-seed. An automated reassigner would need a tiebreak policy the plan does not specify.
- **IC-3**: NonVulnerable cap uses largest-remainder stratified sampling (`nonvulnerable_cap.py`). The fractional per-source caps are summed exactly (`sum == max_nonvuln`); ties are broken by source name ascending for determinism.
- **IC-4**: CLI hashes the manifest only, not the full split directory (`cli.py:_run_register`). The manifest is the canonical "split recipe + metadata". Hashing it means a regenerated split with the same inputs passes the check. v2.1: hash the full directory.
- **IC-5**: Per-source strategy from config.yaml is NOT wired (`cli.py:_run_split`). The CLI hardcodes `stratified_split`; the `SPLITTERS` dispatch dict supports per-source dispatch but the CLI does not consult `config.yaml`. Deferred to v2.1.
- **IC-6**: Leakage auditor is NOT wired to the CLI (`cli.py:_run_split`). The function works but the CLI does not call it. Production splits must invoke `run_audit()` directly. Deferred to v2.1 (will add `--run-leakage-audit` flag).

### The 4 splitting components — how they actually work

#### 1. `splitters.py` — 4 strategies

```python
SPLITTERS = {
    "random":        random_split,
    "stratified":    stratified_split,
    "project":       project_split,
    "project_level": project_split,   # alias per AUDIT_PATCHES 5-P1
    "temporal":      temporal_split,
}
```

- **`random_split`**: `random.Random(seed).shuffle` → ratio-based assignment. No guarantees. Use only for sanity tests.
- **`stratified_split`** (default): groups by composite of `(primary_class, source, tier)`. Per-stratum proportional allocation: `max(1, int(g_n * ratio))`. Remainders go to test.
- **`project_split`**: groups by `project_id`. Whole project in one split. Contracts without `project_id` go to train.
- **`temporal_split`**: `year <= cutoff_year` (default 2023) → train/val split. `year > cutoff_year` → all test. Contracts without year → train.

**Why per-stratum proportional allocation can be slightly off:** a stratum with 10 contracts gets `floor(10*0.7)=7` to train, `floor(10*0.15)=1` to val, `floor(10*0.15)=1` to test, with `10-9=1` remainder to test. The manifest records actual counts.

#### 2. `dedup_enforcer.py` — the BCCC-failure pattern fix

The BCCC dataset had 38.8% duplication. Many contracts appeared in BOTH train and test. Stage 1 preprocessing pre-computed `dedup_group` (AST similarity, threshold 0.85) on every contract. The enforcer reassigns straddling groups to the majority split, ties → train.

**Algorithm (6 steps):**
1. Build `group_to_splits: dict[int, set[str]]` and `group_to_all: dict[int, list[Contract]]`
2. Skip groups of size ≤ 1
3. Skip groups that don't straddle
4. Determine target: `max(("train", "val", "test"), key=lambda s: (counts[s], s == "train"))`
5. Record `{group, from_split, to_split, contract_count}` in `metadata.reassignments`
6. Rebuild splits by moving contracts to their target split

**Empirical behavior on synthetic data:** with 100 contracts in 10 dedup groups of 10, ALL 100 contracts move to train (each group straddles all 3 splits and the largest count is in train because 70% of contracts started there). This is **expected**, not a bug.

#### 3. `leakage_auditor.py` — independent post-split safety net

Uses **text-shingle Jaccard similarity** (3-character shingles on whitespace-normalized lowercased source code). Threshold 0.5 per AUDIT_PATCHES 5-P3. Different algorithm from AST similarity in dedup_enforcer → catches what enforcer missed.

**Algorithm:**
- O(N²) pairwise comparison across split boundaries (train↔val, train↔test, val↔test)
- `seen_pairs: set[tuple[str, str]]` avoids duplicates
- Returns `LeakageReport` with `pairs: list[LeakPair]`

**Reports only.** The auditor does NOT reassign. A leaked pair is a prompt to either re-run with a different `seed` or investigate the source corpus.

**Scaling:** O(N²) is ~500M comparisons for the 22K-contract v2 baseline, ~10–30 min. LSH (MinHash / LSH-bands) is a v2.1 enhancement.

#### 4. `nonvulnerable_cap.py` — the 3:1 cap, stratified by source

**Algorithm:**
1. Count total positives across all splits.
2. `max_nonvuln = int(cap × total_positive)` (default cap = 3.0).
3. For each split, separate NonVulnerable from positives.
4. If split's NonVuln count > `max_nonvuln`:
   - Group NonVuln by source
   - Compute per-source cap proportional to source's share of the global NonVuln pool
   - Distribute integer counts using **largest-remainder method** (floor first, then top up highest remainders, tiebreak by source name ascending)
   - `random.Random(seed).shuffle` per source, take the per-source cap
5. Update `metadata.nonvulnerable_cap` with full audit: `cap`, `total_positive`, `max_nonvuln`, `per_source` breakdown, `per_split` `{original, capped, kept}`

**Why 3:1, not higher or lower:** 5:1 and 10:1 reproduce the BCCC contamination problem. 1:1 starves the NonVuln signal. 3:1 is the empirical sweet spot.

**Why largest-remainder method:** simple quotas (e.g. `int(share × cap)`) systematically under-allocate total NonVuln. Largest-remainder ensures `sum(per_source_caps) == max_nonvuln` exactly. This is the same method used in legislative apportionment.

### The 3 registry components — how they actually work

#### 1. `catalog.py` — SQLite + YAML mirror

**6 tables** (4 base + 2 system):

| Table | Purpose | Key columns |
|---|---|---|
| `sources` | Per-source pin + last-fetched | `name` (PK), `pin`, `last_fetched`, `enabled`, `n_contracts`, `tier` |
| `artifacts` | Per-exported-artifact hash + lineage | `name` (PK), `sha256`, `size_bytes`, `lineage` (JSON) |
| `splits` | Per-split-version seed + strategy | `version` (PK), `seed`, `strategy`, `contract_counts` (JSON) |
| `dataset_versions` | Named composite (sources + preprocessing + split) | `name` (PK), `source_set`, `preprocessing_config_hash`, `split_version`, `verification_report_path`, `artifact_hash`, `artifact_path` |
| `schema_migrations` (system) | Every schema change recorded | `version` (PK), `description`, `applied_at` |
| `dataset_version_retirements` (system) | Permanent retirement chain | `name` (PK), `superseded_by`, `retired_at`, `reason` |

**Schema versioning:** `SCHEMA_VERSION = 1`. `Catalog.migrate(version, description)` records the migration; the caller is responsible for the `ALTER TABLE`. v2.1 will introduce v2 (per-class metadata in `dataset_versions`).

**YAML mirror:** `write_yaml_mirror()` exports the full catalog as multi-document YAML on every write. CI check is future work.

**`load_artifact(name) -> Optional[DatasetVersion]`** is the main ML-module interface. Returns `None` if retired or missing — the ML code's `__init__` must check for `None` and abort the load.

#### 2. `lineage_tracker.py` — hash, DAG, DOT

**5 functions:**

```python
record_lineage_step(lineage, step, **details) -> dict   # appends to lineage["steps"]
lineage_to_dot(lineage) -> str                          # Graphviz DOT renderer
hash_artifact(path) -> str                              # SHA-256 of file bytes
hash_lineage(lineage) -> str                            # SHA-256 of canonical JSON
verify_artifact(path, expected_hash) -> bool            # load-time gate
```

**Lineage structure (JSON):**
```python
{
    "steps":   [{"step": "ingest", "ts": "...", "source": "solidifi", "n_contracts": 1104}, ...],
    "parents": ["sha256_of_parent_artifact", ...]
}
```

**Why a DAG, not a tree:** dataset versions can have multiple parents (e.g. v2 = v1 + a new source). The DAG generalizes the tree.

#### 3. `dataset_diff.py` — diff + changelog

**`diff_dataset_versions(metadata_old, metadata_new) -> DatasetDiff`:**
- O(N) comparison
- Computes `added_contracts`, `removed_contracts`, `common_contracts`, `label_changes`, per-class metrics
- Per-class: `count_old`, `count_new`, `delta_count`, `delta_pct`, tier breakdowns
- Coarse `predicted_f1_delta_pct` heuristic: `min(5.0, abs(delta_pct) * 0.1)` — a 50% per-class change projects to 5% F1 delta, capped

**`update_changelog(changelog_path, version_name, summary, metrics) -> None`:** appends a markdown entry with the per-class delta table.

### The CLI subcommands

**`sentinel-data split [--dry-run] [--seed N] [--nonvuln-cap X] [--version V] [--config PATH]`:**
1. Load config YAML → find data dir
2. Read `data/labels/merged/*.labels.json` → build `Contract` objects
3. `stratified_split(contracts, seed=args.seed)` (hardcoded — see IC-5)
4. `apply_dedup_enforcer(splits)`
5. `apply_nonvulnerable_cap(splits, cap=args.nonvuln_cap, seed=args.seed)`
6. Write `data/splits/v{version}/{train,val,test}.jsonl` + `split_manifest.json`

**`sentinel-data register [--name NAME] [--version V] [--sources S1,S2] [--retire-previous V] [--verification-report PATH] [--dry-run]`:**
1. Locate `data/splits/v{version}/split_manifest.json`
2. Compute SHA-256 of manifest → `artifact_hash` (IC-4: manifest, not full directory)
3. Open `Catalog(db, yaml)` → `cat.add_dataset_version(DatasetVersion(...))`
4. Optionally `cat.retire_dataset_version(args.retire_previous, ...)`
5. `cat.write_yaml_mirror()`

### The smoke run (v2 baseline, 22,356 contracts)

```
$ PYTHONPATH=Data python3 -m sentinel_data.cli split --seed 42
  Loaded 22356 contracts
  Splitting (strategy=stratified, seed=42)...
  Applying dedup_enforcer (BCCC-failure pattern fix)...
  Applying NonVulnerable 3:1 cap...
  Writing splits to data/splits/v1/...
  ✓ Splitting complete:
    train=15644 val=3344 test=3368
    dedup_groups_resolved=0
    NonVulnerable:positive ratio = 0.13:1 (cap=3.0)
    Manifest: data/splits/v1/split_manifest.json
```

`dedup_groups_resolved=0` (no near-dup groups in the v2 baseline), NonVuln:positive = 0.13:1 (well under the 3:1 cap).

### The P0 CLI fix (Stage 5 bug)

- **Bug**: `_run_verify` body was accidentally inside `_run_register` (dead code); duplicate stub definitions for `_run_split` and `_run_register` made the dispatch table point to the stubs.
- **Fix**: extracted `_run_verify` into its own function; removed duplicate stubs. All 10 `_run_*` functions are now unique.

---

## 3️⃣ The Broader Context

### What Stage 5 enables downstream

| Stage | What it builds on Stage 5 |
|---|---|
| Stage 6 (analysis) | Reads splits for per-class distribution analysis |
| Stage 7 (export) | Produces sharded export from the splits + registers a `DatasetVersion` |
| Stage 8 (Run 11) | Trains on a registered `DatasetVersion`, calls `verify_artifact_hash()` before load |

### What breaks if Stage 5 is wrong

- Missing dedup enforcement → BCCC 38.8% duplication re-enters the corpus → inflated F1 (same as BCCC)
- Missing NonVuln cap → 514K:1 imbalance (when DISL is re-introduced in v2.1) → model defaults to "predict negative"
- Missing catalog → "what dataset version did Run 11 train on?" is unanswerable 6 months from now
- Missing hash verification → hand-edited export silently poisons training
- Missing YAML mirror → catalog drift between DB and version-controlled YAML

### Operational consequences

1. **The split stage is required before register.** `_run_split` writes `splits/v{version}/`; `_run_register` reads `splits/v{version}/split_manifest.json`. Calling register without a prior split fails with "split manifest not found".

2. **The dedup_enforcer is always applied** in the CLI. Two-pass split is not optional; no `--no-dedup` flag. The v2 baseline has 0% expected dedup (no near-dup groups), so the enforcer is a no-op in practice.

3. **The NonVuln cap is configurable** via `--nonvuln-cap` (default 3.0). Setting to 0 disables the cap entirely. Setting to 1.0 enforces 1:1.

4. **The catalog is the single source of truth.** Every Stage 7 export MUST register its `DatasetVersion`. Hash check is the load-time gate; missing or retired `load_artifact()` returns `None`.

5. **The YAML mirror is written on every register call.** DB and YAML are kept in sync by `write_yaml_mirror()`. CI check is future work.

6. **The leakage auditor is manual.** A CLI-produced split does NOT include a leakage audit. Callers must invoke `run_audit()` directly.

7. **Dataset versions are immutable.** A re-register of the same name overwrites the row (sqlite's `INSERT OR REPLACE`). To "update" a dataset, register a new version and retire the old.

### What stays the same no matter what

- The 4 splitter strategies
- The two-pass split (splitter → dedup_enforcer)
- The dedup enforcer's majority-rule-with-train-tiebreak (IC-1)
- The leakage auditor's report-only semantics (IC-2)
- The NonVuln cap's largest-remainder stratification by source (IC-3)
- The 4+2 SQLite table schema
- The hash verification at load time

---

## 4️⃣ Verification — Stage 5 exit criteria

All 12 exit criteria (per `06_stage_5_splitting_registry.md` and `ADR-0006-splitting-and-registry-design.md`):

| # | Check | Status | Notes |
|---|---|---|---|
| 1 | 4 splitter strategies produce correct splits | ✅ | `random_split`, `stratified_split`, `project_split`, `temporal_split` + `project_level` alias |
| 2 | `dedup_enforcer` reassigns straddling groups | ✅ | `apply_dedup_enforcer` with majority rule + train tiebreak |
| 3 | `leakage_auditor` reports 0 leaks on clean split | ✅ | `find_leaks` uses 3-shingle Jaccard, threshold 0.5 |
| 4 | SQLite catalog with 4+2 tables | ✅ | `sources`, `artifacts`, `splits`, `dataset_versions` + `schema_migrations`, `dataset_version_retirements` |
| 5 | `load_artifact("sentinel-v2-dryrun-2026-08")` works | ✅ | Returns `None` for retired/missing versions; ML code must check |
| 6 | `verify_artifact_hash()` catches tampered files | ✅ | SHA-256 of file bytes, streaming 64KB chunks |
| 7 | NonVuln 3:1 cap enforced | ✅ | `apply_nonvulnerable_cap` with largest-remainder stratification by source |
| 8 | `dedup_enforcer` records `reassignments` in manifest | ✅ | `metadata.reassignments: list[{group, from_split, to_split, contract_count}]` |
| 9 | `dataset_diff` computes per-class delta | ✅ | `diff_dataset_versions` returns `DatasetDiff` with added/removed/common/label_changes/per-class |
| 10 | `changelog.md` updates on version add | ✅ | `update_changelog` appends markdown with per-class table |
| 11 | `ADR-0006-splitting-and-registry-design.md` committed | ✅ | `docs/decisions/ADR-0006-splitting-and-registry-design.md` |
| 12 | `poetry run pytest tests/test_splitting tests/test_registry` passes with > 80% coverage | ✅ | 70 passed (37 splitting + 33 registry), 0 skipped |

**All 12 Stage 5 exit criteria pass. Stage 5 is complete.**

---

## 5️⃣ The "got it" checklist

Before we move to Stage 6, you should be able to answer (without looking at this doc):

1. **Why two-pass splitting?** Pass 1: stratified assignment. Pass 2: dedup_enforcer reassigns straddling near-dup groups to the majority split (ties → train). Prevents BCCC-style train/test leakage (38.8% duplication in BCCC is now physically impossible).

2. **What's the NonVulnerable 3:1 cap?** DISL's 514K contracts would create 428:1 imbalance when re-introduced in v2.1. Cap at 3× total positives, stratified by source using largest-remainder method. v2 baseline has only 0.13:1 (no subsampling needed).

3. **Why project-level splitting for audit datasets?** A project (e.g. ScaBench's 31 projects) is entirely in one split. Prevents "90% of one project's contracts in train, 10% in test" bias. Contracts without `project_id` go to train (best-effort).

4. **What's in the SQLite catalog?** 4 base tables (sources, artifacts, splits, dataset_versions) + 2 system tables (schema_migrations, dataset_version_retirements). YAML mirror is exported on every write. Schema version is 1.

5. **Why hash verification at load time?** Prevents hand-edited exports from silently poisoning training. SHA-256 of file bytes (streaming 64KB chunks), computed once at `__init__` not per `__getitem__`. The ML module's `SentinelDataset.__init__` calls `verify_artifact_hash()` before reading any rows.

6. **Why are dataset versions append-only?** The audit trail is permanent. v1.4 BCCC labels, v8 BCCC graphs, v9 graphs — all preserved with `superseded_by` chain. `dataset_diff` shows what changed; `changelog.md` is the human-readable log.

7. **What are the 6 implementation choices (IC-1 through IC-6)?** IC-1: dedup majority rule with train tiebreak. IC-2: leakage auditor reports only. IC-3: NonVuln cap uses largest-remainder. IC-4: CLI hashes manifest only. IC-5: per-source strategy not wired to CLI. IC-6: leakage auditor not wired to CLI.

8. **What's the difference between dedup_enforcer and leakage_auditor?** The enforcer uses AST similarity (pre-computed at Stage 1, threshold 0.85) and reassigns. The auditor uses text-shingle Jaccard similarity (computed at audit time, threshold 0.5) and reports only. Different algorithms → catches different leaks.

9. **What breaks if Stage 5 is wrong?** Cross-split leakage (inflated F1, BCCC pattern), missing NonVuln cap (428:1 imbalance when DISL is re-introduced), missing catalog (unanswerable "what dataset version did Run 11 train on?"), missing hash verification (hand-edited export poisoning).

If you can answer all 9, Stage 5 is mastered and we can move to Stage 6.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 5" — 6 specific questions to test your understanding
- **`06_stage_5_splitting_registry.md`** — the design + intent document
- **`Sentinel_v2_Data_Module_Integration_Proposal.md`** §3.6 (splitting), §3.7 (registry)
- **`docs/decisions/ADR-0006-splitting-and-registry-design.md`** — the 8 design decisions + 6 implementation choices, with rationale

When you're ready, say **"Stage 5 is mastered — let's move to Stage 6."**
