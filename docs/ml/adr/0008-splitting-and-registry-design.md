# ADR-0008 — Splitting and Registry Design (Stage 5)

**Status:** Accepted  
**Date:** 2026-06-12  
**Deciders:** Ali Motafegh, Claude (assistant)  
**Supersedes:** —  
**Superseded by:** —  

---

## Context

Stage 5 establishes the train/val/test boundary and the artifact registry for
the v2 data module.  Before this ADR the split was produced by a one-off script
(`ml/scripts/make_splits_v10.py`) with no deduplication guarantees, no cap on
the dominant NonVulnerable class, and no record of which dataset version a
training run consumed.

Three concrete problems drove the design:

1. **BCCC data leakage** — the original BCCC labels are grouped by vulnerability
   folder; contracts sharing the same source code across folders were placed in
   different splits, creating near-identical pairs across train/test.  A dedup
   enforcer must re-assign contracts in the same group to the same split.

2. **NonVulnerable class dominance** — before the v1.4 verification pass,
   67 % of contracts had only `NonVulnerable=1`.  A 3:1 NonVulnerable-to-positive
   cap prevents the model from collapsing to the majority class.

3. **Traceability** — "which data did Run 11 train on?" was unanswerable in the
   v1 pipeline.  A named, content-addressed dataset version with a retirement
   chain and lineage graph closes this gap.

Additional constraints:

- The active pipeline (Run 9, F1=0.3081) must not be disrupted — v2 splits are
  produced in `data_module/data/splits/` and do not touch `ml/data/splits/`.
- Label verification (Stage 5 Phase 5) reduced positives by 69.8 %; the
  splitter must handle severely imbalanced classes without error.
- The registry must be human-readable (YAML mirror) for version control review
  and machine-queryable (SQLite) for fast lookup.

---

## Decision

### D-5.1 — Four split strategies, `stratified` as default

`sentinel_data/splitting/splitters.py` implements four strategies:

| Strategy | When to use |
|----------|-------------|
| `random` | Baseline / ablation |
| `stratified` | Default (per-class ratio preserved) |
| `project` | Same project's contracts stay in one split |
| `temporal` | Year-based boundary (pre-2019 train, 2019–2020 val, 2021+ test) |

`stratified_split` uses 70/15/15 proportions with a fixed seed.  The CLI
`sentinel-data split` always runs `stratified` unless `--strategy` overrides.

**Rejected alternatives:** a single hard-coded 70/15/15 random split (no
guarantee that rare classes like CallToUnknown survive in val/test); temporal
only (year metadata is missing for ~40 % of BCCC contracts).

### D-5.2 — Two-pass pipeline: splitter → dedup_enforcer

Splitting and deduplication are separate passes:

1. `stratified_split()` assigns contracts to splits by class ratio.
2. `apply_dedup_enforcer()` reads the pre-computed `dedup_group` field from
   Stage 1 and reassigns contracts: within each group, the majority split wins;
   ties go to `train`.

Separating the passes keeps the splitter pure (no dependency on dedup metadata)
and makes the enforcer independently testable.

**Rejected alternative:** integrate dedup logic into the splitter.  Rejected
because the splitter needs to be correct in isolation for the `project` and
`temporal` strategies too, and mixing dedup logic into the splitter complicates
all four implementations.

### D-5.3 — NonVulnerable 3:1 cap applied after dedup

`apply_nonvulnerable_cap(splits, cap=3.0)` downsamples `NonVulnerable=1`
contracts in each split to at most `cap × positive_count`.  The cap is applied
after dedup to avoid reassigning dedup-enforced groups.

The 3:1 default was chosen to match the v10 training distribution (positive
contracts 25 % of 41,576 = ~10,400; NonVulnerable ~31,176 = ~3:1).

### D-5.4 — Leakage auditor is reporting-only (no reassignment)

`sentinel_data/splitting/leakage_auditor.py` computes Jaccard shingle
similarity between all pairs of `.sol` source texts.  Pairs with similarity
≥ 0.5 that land in different splits are flagged in the audit report.

The auditor does **not** reassign contracts — that is the dedup enforcer's job
(which acts on exact SHA-256 duplicates from Stage 1).  The auditor catches
near-duplicates that survived dedup (e.g. contracts with minor variable renames).

**Rejected alternative:** merge auditor + enforcer.  Rejected because the
auditor is an O(N²) post-hoc check that is too slow to run in the critical path;
the enforcer is a fast O(N) pre-split step.

### D-5.5 — SQLite catalog + YAML mirror; append-only versions

`sentinel_data/registry/catalog.py` maintains:

- 4 base tables: `sources`, `artifacts`, `splits`, `dataset_versions`
- 2 system tables: `schema_migrations`, `dataset_version_retirements`

Every `add_dataset_version()` call appends a new row; nothing is overwritten.
Retired versions are recorded in `dataset_version_retirements` with a
`superseded_by` chain.  `load_artifact(name)` returns `None` for retired
artifacts.

The YAML mirror (`data/registry/catalog.yaml`) is written on every mutation and
is suitable for `git diff` review.

**Rejected alternatives:**
- **JSON files only** — no query capability; no schema enforcement.
- **Postgres** — operational overhead for a single-machine research pipeline.
- **DVC** — adds a new dependency and remote storage requirement; the YAML
  mirror achieves the same version-control goal.

### D-5.6 — Hash verification as the load-time gate

`catalog.verify_artifact_hash(path)` checks the SHA-256 of the file bytes
against the stored hash.  The ML module's `SentinelDataset.__init__` (Stage 7)
will call this before loading.  A mismatch raises `ValueError`.

The shared `compute_hash()` function lives in `sentinel_data.registry.catalog`
and is re-exported from `sentinel_data.registry`.  The ML module's
`InferenceCache` will import it from here after the Stage 7 seam swap
(currently uses its own helper).

### D-5.7 — Lineage as a DAG with training-run links (5-P6)

`sentinel_data/registry/lineage_tracker.py` records each transformation step
as a list entry in the artifact's `lineage_json` field.  A dedicated
`record_training_run(lineage, run_name, dataset_version, ...)` function adds the
training-run → dataset-version link, answering "what data did Run 11 train on?"

This is stored on the artifact; the `lineage_to_dot()` helper renders it as
Graphviz DOT for visualization.

---

## Consequences

### Positive

- Near-duplicate leakage caught and logged before training begins.
- NonVulnerable dominance bounded; rare classes (CallToUnknown: 239 positives)
  survive in all three splits.
- Full provenance chain: every artifact has a SHA-256 hash, a lineage DAG, and
  a retirement chain.
- YAML mirror makes split metadata reviewable in a standard `git diff`.

### Negative

- O(N²) leakage auditor is too slow for large corpora (>50K contracts); must be
  run as a background job, not in the CLI critical path.
- The 3:1 cap is heuristic; the optimal ratio depends on the class-imbalance
  mitigation strategy chosen for Run 11 (WeightedRandomSampler vs ASL weights).
- SQLite is single-writer; parallel registration of multiple versions is not
  supported.

### Neutral

- `split_manifest.json` in each split directory duplicates some data from
  `catalog.db` — intentional redundancy so the split is self-describing without
  requiring a catalog lookup.
- The four split strategies share the `Contract` dataclass and `Splits` output
  container; adding a fifth strategy is ~50 lines.

---

## Alternatives Considered

**Single pass: integrate dedup into splitter**  
Rejected: makes each of the four splitter implementations depend on dedup
metadata, complicating all four and making unit testing harder.

**Temporal-only splits**  
Rejected: year metadata is absent for ~40 % of BCCC contracts; temporal splits
would produce severely unbalanced folds.

**DVC for artifact tracking**  
Rejected: adds remote storage dependency and a new tool in the critical path.
The SQLite + YAML mirror achieves the same goal with zero new dependencies.

**Jaccard similarity threshold < 0.5**  
Considered 0.3 and 0.4 — rejected because Solidity contracts share many
boilerplate patterns (SafeMath, Ownable, ERC20 interface); 0.5 avoids
flagging unrelated contracts that happen to share the same library imports.

---

## References

- Stage 5 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md`
- AUDIT_PATCHES 5-P1 through 5-P7: `docs/proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md`
- Splitters: `data_module/sentinel_data/splitting/splitters.py`
- Dedup enforcer: `data_module/sentinel_data/splitting/dedup_enforcer.py`
- Leakage auditor: `data_module/sentinel_data/splitting/leakage_auditor.py`
- NonVulnerable cap: `data_module/sentinel_data/splitting/nonvulnerable_cap.py`
- Catalog: `data_module/sentinel_data/registry/catalog.py`
- Lineage tracker: `data_module/sentinel_data/registry/lineage_tracker.py`
- [ADR-0005 — BCCC dataset choice](0005-bccc-dataset-choice.md) (context for label noise)
- [ADR-0007 — Representation port design](0007-representation-port-design.md) (Stage 2 predecessor)

---

**When this ADR is superseded, do NOT delete it.** Add "Superseded by NNNN" at the top
and link forward. The record of why we changed our minds is as valuable as the original
decision.
