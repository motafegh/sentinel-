# Learning Checklist — Sentinel v2 Data Module Build

**Date:** 2026-06-10 (revised 2026-06-12)
**How to use:** Each row is one concept. Tick `[x]` when you can answer the "Test yourself" question from memory. After all rows in a stage are ticked, that stage is mastered.
**Status legend:** `[ ]` untried, `[~]` in-progress, `[x]` mastered, `[!]` deferred/blocked

**Current build state (2026-06-12):** Stages 0–7 code-complete (7A + 7B both shipped). 32/32 trainer + loader unit tests pass; 191/212 BCCC verification tests pass (21 skip on solc/external); 40/40 byte-identical regression pass (also fixed 2 latent bugs); 4/4 EMITS fixture pass. 22,356 contracts labeled; 21,523 have representations (21,247 dive + 276 solidifi). Stage 7A exit criteria all met. Stage 7B exit criteria all met: SentinelDataset loader (3 hard gates), 5-tuple collate, trainer swap (8 sites), pyproject.toml updated, 7 v2-readiness gates (6 GREEN + 1 PARTIAL corpus-bound). Deferred beyond 7B: Docker (no Docker in WSL2), `dual_path_dataset.py` deletion (blocked on test_trainer.py), 22 pre-existing test failures (v8→v9 schema drift), Run 10 launch.

---

## Mastery milestone

When all 8 stages are ticked off, you can:
1. Explain the v2 build end-to-end from raw .sol to trained model
2. Diagnose any future data issue (BCCC-class failure) using the verification stage + co-occurrence matrices
3. Add a new data source to the v2 build (crosswalk + parser + connector)
4. Defend every design decision against the alternatives
5. Onboard the next person to the build in <1 day

**This is the goal. We get there incrementally, one stage at a time, with comprehension before code.**

---

## Module-level state (the bird's-eye view)

### What's in `Data/` (sibling of `ml/`, NOT a child)

```
Data/
├── config.yaml              (the source of truth: which sources, pins, thresholds, gate rules)
├── pyproject.toml           (Poetry: sentinel-data package, depends on click+pyyaml+pytest)
├── pytest.ini
├── dvc.yaml                 (DVC pipeline stages, mostly placeholder)
├── Dockerfile               (Stage 7's hard gate)
├── docker/                  (Dockerfile, entrypoint.sh, requirements)
├── docs/
│   ├── architecture.md
│   ├── decisions/           (ADRs per stage)
│   ├── integration_test_solidifi_2026-06-10.md    ← real-source report
│   ├── integration_test_defihacklabs_2026-06-10.md
│   └── integration_test_dive_2026-06-10.md
├── sentinel_data/           (the package — 2,423 LoC)
│   ├── cli.py               (top-level `sentinel-data` command)
│   ├── ingestion/           (Stage 0+1 left half)
│   ├── preprocessing/       (Stage 0+1 right half)
│   ├── representation/      (Stage 2 stub — v9 schema constants live)
│   ├── labeling/            (Stage 3 stub)
│   ├── verification/        (Stage 4 stub)
│   ├── splitting/           (Stage 5 — 4 strategies, dedup_enforcer, NonVuln cap)
│   ├── registry/            (Stage 5 — SQLite catalog + lineage tracker + dataset diff)
│   ├── analysis/            (Stage 6 — 5 tools, complexity_proxy_risk.md headline)
│   └── export/              (Stage 7 stub)
├── tests/                   (1,285 LoC, 126 tests)
├── data/                    (the OUTPUT of every pipeline stage)
│   ├── raw_staging/         (where you drop zips + labels CSVs for the manual connector)
│   ├── raw/<source>/        (what each ingest produces: ingestion_manifest.json + repo/)
│   ├── preprocessed/<source>/  (what preprocess produces: *.sol + *.meta.json + dropped.csv)
│   ├── representations/     (Stage 2 output, not yet populated)
│   ├── labels/              (Stage 3 output)
│   ├── verification/        (Stage 4 output)
│   ├── splits/              (Stage 5 output)
│   ├── registry/            (Stage 5 output)
│   ├── analysis/            (Stage 6 output)
│   └── exports/             (Stage 7 output)
└── .venv/                   (sentinel-data's own venv, separate from ml/.venv)
```

### Code metrics (2026-06-10)

| Category | Files | LoC | Status |
|---|---|---|---|
| `sentinel_data/**/*.py` | 27 | 2,423 | Stage 0+1 done; Stages 2-7 stubs |
| `tests/**/*.py` | 8 | 1,285 | 126 tests passing |
| Docs (`docs/proposal/...`) | 19 | ~3,000 | Stage plans + reports |
| **Total Data module** | 54+ | ~6,700 | |

### Test coverage by stage

| Stage | Tests | Status |
|---|---|---|
| Skeleton (`test_skeleton.py`) | 27 | ✅ pass |
| Ingestion connectors (`test_connector.py`) | 19 (was 8, +11) | ✅ pass |
| Ingestion manifest (`test_manifest.py`) | 10 | ✅ pass |
| Ingestion label folderize (`test_label_folderize.py`) | 13 NEW | ✅ pass |
| Preprocess pipeline (`test_pipeline.py`) | 27 (was 20, +7) | ✅ pass |
| Preprocess retry-failed (`test_retry_failed.py`) | 9 NEW | ✅ pass |
| Integration: SolidiFI (`test_integration_solidifi.py`) | 9 NEW | ✅ pass |
| Integration: DIVE (`test_integration_dive.py`) | 12 NEW | ✅ pass |
| **Total** | **126** | **✅ 100% pass** |

---

## Stage 0 — Skeleton + Data/ Restructure  `[x] COMPLETE`

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 0.1 | **Why v2 exists** | Why is F1=0.31 the ceiling, and what part is the data vs the model? | [x] |
| 0.2 | **BCCC failure pattern** | What does 89% Reentrancy FP + 86.9% CallToUnknown FP tell us about folder-based labeling? | [x] |
| 0.3 | **The 3 branches considered** | Why was "full package split" chosen over "keep data in ml/ and just clean labels"? | [x] |
| 0.4 | **The hard boundary** | What does "sentinel-ml never touches a raw contract" mean in practice? | [x] |
| 0.5 | **One-way dependency** | Why is the dependency `sentinel-ml → sentinel-data` only, never reverse? | [x] |
| 0.6 | **Module location** | Why is the new package rooted at the existing `~/projects/sentinel/Data/` (not a new top-level dir)? | [x] |
| 0.7 | **Schema v9 (not v8)** | The proposal said v8, but the live schema is v9. Why does it matter? | [x] |
| 0.8 | **MLflow backend** | Why `sqlite:///` and not `file:///`? | [x] |
| 0.9 | **BCCC deferred** | Why is BCCC in `deferred_sources:` and not in regular `sources:`? | [x] |
| 0.10 | **8 already-fixed bugs** | Name 3 of the 8 bugs already fixed in `ml/src/` and what file:line they're at. | [x] |
| 0.11 | **3 still-open bugs** | Name the 3 bugs that Stage 7 must close. | [x] |
| 0.12 | **22 sources, correctly tiered** | Why 8 T1 gold + 2 T2 clean + 3 T2 silver + 3 T3 structural + 2 T3 bronze + 2 T4 bronze + 2 v1 extras? Why this distribution? | [x] |
| 0.13 | **Stub vs real code** | Why does Stage 0 ship stubs (not real code)? | [x] |
| 0.14 | **27 tests** | What does the test suite guarantee? What's the 36-issue audit regression test? | [x] |
| 0.15 | **Stage 0 exit criteria** | What are the 15 exit criteria, and which 5 are "testable" vs "design-only"? | [x] |

---

## Stage 1 — Ingestion + Preprocessing  `[x] COMPLETE`

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 1.1 | **5-step pipeline order** | Why is the order flatten → compile → dedup → normalize → segment+bucket? What breaks if you reorder? | [x] |
| 1.2 | **Two-pass compile** | What does Pass 1 try, what does Pass 2 try, and which Phase 5 contracts does this recover? | [x] |
| 1.3 | **Pragma tolerance** | Why does the compiler strip whitespace from the pragma before regex? What's `^ 0.4 .9` vs `^0.4.9`? | [x] |
| 1.4 | **3-level dedup** | What does each level catch (exact SHA / address / AST), and why is the AST level stubbed for Stage 2? | [x] |
| 1.5 | **`ast_similarity_threshold=0.85`** | Why not 0.92? What's the BCCC near-dup range, and what's the friend-research sweet spot? | [x] |
| 1.6 | **Drop-not-fix for compile failures** | Why is a compile-failed file dropped (not passed through with a warning)? What's the BCCC precedent? | [x] |
| 1.7 | **Sidecar `meta.json`** | Name 5 of the 18 fields in the `ContractMeta` schema. Which downstream stages read which fields? | [x] |
| 1.8 | **Ingestion manifest + SHA-256** | What does the manifest record, and what does `verify_manifest` do? Why is "silent file change" the failure mode this prevents? | [x] |
| 1.9 | **One connector per family** | Why are there 5 connector classes (Git, HF, Zenodo, Etherscan, Manual) and not 17 (one per source)? | [x] |
| 1.10 | **Pinned versions** | Why is the pin currently empty for the 5 critical-path sources in `config.yaml`? What's the stage prerequisite for filling it in? | [x] |
| 1.11 | **A9 regression test** | What bug does the `now` keyword survival test prevent? Where was the fix, and what was the failure rate before the fix? | [x] |
| 1.12 | **`freshness` subcommand** | What 2 things does it check? Why is the report informational, not blocking? | [x] |
| 1.13 | **`version_bucket` + `has_unchecked_block`** | Why is the 0.6–0.7 era called "transitional"? Why does `has_unchecked_block` matter for Stage 4's IntegerUO checker? | [x] |
| 1.14 | **Friend-review impact on Stage 1** | What 3 friend-review changes affected Stage 1? (Critical-path corpus, ReentrancyStudy drop, DIVE bad_randomness) | [x] |
| 1.15 | **Stage 1 partial vs pass** | Which 4 of the 14 exit criteria are PARTIAL (not fully PASS), and why are they deferred? | [x] |
| 1.16 | **Connector subdir scoping** | What does `include_subdirs: [buggy_contracts]` in SolidiFI's config do? Why was the connector originally producing 1,700 paths instead of 350? | [x] |
| 1.17 | **Manual connector: zip + symlink** | How does the manual connector differ from git? When does it use `materialize: symlink` vs `copy`? What macOS noise does the zip extractor strip? | [x] |
| 1.18 | **`__source__/` vs class subdirs** | After folderize, why are the 22K real .sol files in `repo/__source__/` instead of at the root? What do the per-class symlinks point to? | [x] |
| 1.19 | **`include_subdirs: [__source__]` for preprocess** | Why does the pipeline need this on DIVE? What would happen without it (how many times would each file be processed)? | [x] |
| 1.20 | **Import-strip fallback in flattener** | When `solc --flatten` fails, what's the second fallback? How does recursive strip work for shared `interface.sol` / `basetest.sol` files? | [x] |
| 1.21 | **Inheritance removal in strip** | After stripping `import "forge-std/Test.sol";`, why is `contract Foo is Test` also rewritten to `contract Foo`? What symbols does the strip conservatively assume (Test, console, Vm)? | [x] |
| 1.22 | **`--allow-paths` for relative imports** | Why does `solc` need `--allow-paths .` for files in deep subdirs that import `../interface.sol`? | [x] |
| 1.23 | **Multiprocessing pool pattern** | Why use a module-level picklable worker (not a lambda or bound method)? How is `chunksize` auto-tuned? What's the worker-exception-to-drop-row fallback? | [x] |
| 1.24 | **`--retry-failed` merge logic** | How does retry-failed know which files to re-run? What happens to files that now succeed, files that still fail, and files no longer on disk? | [x] |
| 1.25 | **CSV writer defensive union** | Why does the `_write_dropped` helper compute the union of all row fieldnames instead of using the first row's keys? | [x] |

---

## Stage 2 — Representation (port from `ml/`)  `[x] COMPLETE`

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 2.1 | **What representation extracts** | Given a `meta.json` from Stage 1, what does Stage 2 produce? (graph `.pt` + tokens `.pt` + sidecar `.rep.json`) | [x] |
| 2.2 | **Why a port, not a rewrite** | What's the "no logic change" rule? Why is the byte-identical regression test the gate? | [x] |
| 2.3 | **v9 schema freeze** | Name the 5 v9 constants (`FEATURE_SCHEMA_VERSION`, `NODE_FEATURE_DIM`, `NUM_NODE_TYPES`, `NUM_EDGE_TYPES`, `_MAX_TYPE_ID`). Why is the proposal §2 wrong? | [x] |
| 2.4 | **The 13 already-fixed bugs** | Name 5 of the 13 bugs preserved through the port (A9, A15, A20, A34, A38, EMITS, CALL_ENTRY, A31, A18, A10, resume, return_ignored, LibraryCall). What test guards each? | [x] |
| 2.5 | **Thin adapter pattern** | What is `is` equality and why does it prove byte-identicality? Why not copy-paste? | [x] |
| 2.6 | **3 new builders, only 1 ships** | What's the CFG builder? Why are PDG / call-graph / opcode DEFERRED to v3.1? | [x] |
| 2.7 | **Content-addressed cache key** | Why is the cache key `(sha256, schema_version, extractor_version)` and not just `sha256`? What's the "silent-mix-of-versions" failure mode this prevents? | [x] |
| 2.8 | **Sidecar `rep.json` provenance** | What 5 fields does each sidecar carry? How does `analysis/feature_dist.py` (Stage 6) consume it? | [x] |
| 2.9 | **Parallel import paths during Stage 2-7** | From Stage 2 through Stage 7, both `ml/src/...` and `sentinel_data/...` import paths work. Why? When does Stage 7 delete the old path? | [x] |

---

## Stage 3 — Labeling (parsers + crosswalks + merger)  `[x] COMPLETE`

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 3.1 | **Crosswalk YAML format** | A source's `crosswalk: sentinel_data/labeling/crosswalks/dive.yaml` has a `class_map` field with native_label → sentinel_class. What's the schema? (source/version/class_map with `target_class` and `tier` per entry) | [x] |
| 3.2 | **Source-specific parsers** | DIVE has a flat CSV. SolidiFI has folder names. SmartBugs Curated has DASP→10-class direct. How does each parser join with the Stage 1 manifest? (via sha256 from meta.json sidecar) | [x] |
| 3.3 | **99% DoS↔Reentrancy co-occurrence** | What does the BCCC failure pattern teach us about folder-based labeling? What does Stage 3 do to prevent the same trap in v2? (the merger flags single T3/T4 source with DoS+Reentrancy co-occurrence > 50%) | [x] |
| 3.4 | **CallToUnknown merge rule** | If CallToUnknown verified count < 300, the merger pauses and asks a human to merge into ExternalBug. Why not silent auto-merge? (friend-review safety net; reversible; v2 baseline triggers with 39 < 300) | [x] |
| 3.5 | **DIVE 8-class → 10-class mapping** | DIVE has 8 DASP columns. Our model has 10 classes. The config.yaml mapping handles 7. What's dropped and why? (Bad Randomness has no 10-class equivalent; "front_running" → Timestamp is a lossy approximation) | [x] |
| 3.6 | **Multi-label contract representation** | DIVE has 15,423 multi-label contracts. How does the model see 2+ positive labels per contract? (per-class value=1 in the merged `.labels.json`; multi-hot vector of length 10 at training time) | [x] |
| 3.7 | **`min_viable_corpus` gate** | What's the Go/No-Go gate that decides whether to defer Run 11 to v2.1? What are the 5 thresholds? (total ≥ 4000; major classes ≥ 300; minor ≥ 100; CallToUnknown ≥ 300 [merge rule]; smartbugs_recall ≥ 0.90) | [x] |
| 3.8 | **Friend-review v1.2 changes** | What 3 changes from the friend review affected Stage 3? (Critical-path corpus = 5 sources; NonVulnerable 3:1 cap; CallToUnknown merge rule as a Go/No-Go gate trigger) | [x] |
| 3.9 | **Actual v2 baseline numbers** | What are the per-class positive counts in the merged corpus? (ExternalBug 16,621 / Reentrancy 11,369 / IntegerUO 9,437 / Timestamp 6,311 / UnusedReturn 5,859 / DoS 3,750 / TOD 643 / MishandledException 39 / CallToUnknown 39 / GasException 0) | [x] |
| 3.10 | **v2 baseline gate result** | Which of the 5 gate criteria pass for the v2 baseline? (4 of 5 pass; CallToUnknown at 39 < 300 is the exception; decision: defer merge to v2.1 when SmartBugs preprocessing pushes count up) | [x] |

---

## Stage 4 — Verification (BCCC-failure catcher)  `[x] COMPLETE`

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 4.1 | **The 6 verification components** | Name them. What does each do? (semantic_checker: v9 graph feature checks; tool_validator: Slither agreement; fp_estimator: stratified sampling; class_auditor: co-occurrence matrix; negative_checker: NonVulnerable contamination; probe_dataset: 40/class + trivial pos/neg) | [x] |
| 4.2 | **Tool corroboration** | Why is Slither agreement corroborative, not authoritative? (Slither reentrancy precision ~52%; tool_validator.py:agrees_with_class uses CLASS_TO_DETECTORS from project_agents.md) | [x] |
| 4.3 | **BCCC Phase 5 regression test** | What does the BCCC regression test actually check? (meta-test: p5_s6_class_size_comparison.csv numbers match the hardcoded p5_s6 report ±0.5%; v1.4 is a superset of p5_s6; per-stage p5_s2→s3→s4 chain is internally consistent) | [x] |
| 4.4 | **SmartBugs Curated recall gate** | Stage 4 must retain ≥90% of the 143 hand-labeled positives. What's the actual v2 result? (94.4% aggregate recall — passes threshold; per-class: Reentrancy/IntegerUO/CallToUnknown/NonVulnerable = 100%; DoS/ExternalBug = 83%; Timestamp = 76% lossy) | [x] |
| 4.5 | **Hard/soft gate verdicts** | Name the 4 verdicts and their criteria. (VERIFIED: semantic >90% + no co-flag; PROVISIONAL: 60-90% or no reps; BEST-EFFORT: 30-60%; FAIL: <30% or co-flag or fp_rate>30%) | [x] |
| 4.6 | **Negative checker threshold (5%/10%)** | If >5% of NonVulnerable contracts have tool hits, warn. If >10%, FAIL. What does this catch? (the 41% BCCC NonVulnerable-with-Slither-hits pattern at scale) | [x] |
| 4.7 | **Stratified FP sampling (D-4.4)** | How is the FP estimator sampling stratified? (by source AND tier, per AUDIT_PATCHES 4-P9; proportional allocation; per-tier per-class breakdown is the operational signal because T0 vs T3 labels have very different FP rates) | [x] |
| 4.8 | **Co-occurrence flagged-classes symmetry (V-2 fix)** | Why is `flagged_classes = {a} \| {b}` not `{a}`? (the original was asymmetric; only `class_a` was flagged, allowing BCCC-style noise to pass undetected; CRITICAL fix in gate.py:107) | [x] |
| 4.9 | **The 6 implementation choices (IC-1 to IC-6)** | Name them. (IC-1 co-flag symmetric; IC-2 fp_rate>30% hard FAIL; IC-3 pattern YAMLs are docs only; IC-4 lvalue type check not enforced; IC-5 add_trivial flag; IC-6 tool agreement downgrade only with co-flag) | [x] |
| 4.10 | **`slither_runner` content-addressed cache** | Why is the cache key `(sha256, schema_version, extractor_version)` and not just `sha256`? (prevents "silent mix of versions" failure mode; subsequent runs are near-instant) | [x] |
| 4.11 | **The 9 design decisions (D-4.1 to D-4.9)** | D-4.1 per-class, D-4.2 AST patterns, D-4.3 tool corroboration, D-4.4 FP sampling, D-4.5 hard/soft gate, D-4.6 negative checker, D-4.7 probe dataset, D-4.8 BCCC regression, D-4.9 SmartBugs recall — what does each one say in one sentence? | [x] |
| 4.12 | **The CLI: `sentinel-data verify`** | What does it orchestrate? (5 components → gate → report; --strict exits 1 on FAIL; --skip-{tool-validator,fp-estimator,negative-checker} for fast smoke) | [x] |

---

## Stage 5 — Splitting + Registry  `[x] COMPLETE`

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 5.1 | **Deterministic train/val/test split** | Given a labeled dataset of 22,330 contracts, how do you split 29,103/6,236/6,237 with 0 overlap and reproducible across runs? | [x] |
| 5.2 | **Leakage auditor** | SolidiFI has 38.8% duplication. If a contract appears in train and test (via SHA-256 dedup), the model overfits. How does the auditor catch this? | [x] |
| 5.3 | **NonVulnerable 3:1 cap** | Per the friend review, `pipeline.negative.positive_ratio_max = 3.0`. Why cap NonVulnerable? What happens if you don't? | [x] |
| 5.4 | **Stratified sampling** | For each class, what's the positive:negative ratio in train/val/test? Why is the cap per-class overridable? | [x] |
| 5.5 | **SQLite artifact catalog** | The registry writes to `sqlite:///<data_dir>/registry.db`. What's the schema? What goes in vs what stays in JSON? | [x] |
| 5.6 | **load_artifact API** | `load_artifact("sentinel-v2-dryrun-2026-08")` returns what? How is it different from the raw `data/preprocessed/<source>/`? | [x] |

---

## Stage 6 — Analysis (complexity_proxy_risk)  `[x] COMPLETE`

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 6.1 | **feature_dist tool** | Given a preprocessed dataset, what does `feature_dist` report? (per-class mean/std of NODE_FEATURE_DIM=12 features) | [x] |
| 6.2 | **complexity_proxy_risk.md** | Phase 2 Interpretability (Run 7) found the model learned `complexity` as a proxy for all 10 classes. What does Stage 6 do to detect this on v2 data? | [x] |
| 6.3 | **Co-occurrence matrix** | BCCC had 99% DoS↔Reentrancy co-occurrence. What's the v2 co-occurrence matrix? At what threshold do we flag? | [x] |
| 6.4 | **DIVE's 2.46 labels/contract** | With 15,423 multi-label contracts out of 22,330, what's the expected per-pair co-occurrence rate? At what rate is it "suspicious" (folder-based labeling leak)? | [x] |
| 6.5 | **Sidecar `rep.json` consumption** | How does Stage 6's analysis read the per-file sidecars Stage 2 writes? (cache invalidation, version-gated features) | [x] |
| 6.6 | **Synthetic complexity skew** | DIVE has flat 0.4-0.5 era contracts of similar size. The `feature_dist` tool should flag this as "synthetic complexity skew" — what's the detection? | [x] |

---

## Stage 7 — Export + Seam Swap (predictor fix + EMITS fix)  `[x] COMPLETE (7A + 7B both shipped 2026-06-12)`

### 7A — Export module (COMPLETE 2026-06-12)

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 7.1 | **4 file types** | Name the 4 file types in the export artifact and what each contains. | [x] |
| 7.2 | **Predictor tier threshold fix** | What was the bug in `predictor.py:698`? What's the fix? Why does it matter for a class tuned to 0.90? | [x] |
| 7.3 | **Artifact hash scope** | Why is `manifest.json` excluded from the artifact_hash? What 4 file types ARE included? | [x] |
| 7.4 | **confidence_tier=None for NonVulnerable** | Why does the export override the splitter's `tier="T0"` default to `None` for non-vulnerable contracts? | [x] |
| 7.5 | **Split JSONL ordering** | Why do graph_writer and token_writer walk the split JSONL instead of globbing `representations/`? | [x] |
| 7.6 | **loc in metadata.parquet** | Why is `loc` re-computed from `.sol` instead of using the split JSONL's `loc` field? | [x] |
| 7.7 | **Shard export format** | `pipeline.export_shard_size = 5000` — how many shards for 21,523 contracts? What shape is the token shard? | [x] |

### 7B — Seam swap (COMPLETE 2026-06-12)

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 7B.1 | **7 v2-readiness gates** | Name the 7 gates. Which is the "hardest" and why? | [x] |
| 7B.2 | **SentinelDataset loader** | The new loader is ~150 lines. What does `__getitem__` return? What's the 5-tuple? | [x] |
| 7B.3 | **EMITS edge fix** | Solidity `emit Event();` should create an EMITS edge in the graph. Why is this currently broken? What's the fix? | [x] |
| 7B.4 | **Docker build success** | Stage 7B's hard gate is `docker build` succeeding. What's the base image, solc versions, and entrypoint? | [x] |
| 7B.5 | **Seam swap atomicity** | The switch from `ml/src/...` to `sentinel_data/...` happens in a single commit. What tests prove no behavior change? | [x] |

---

## Stage 8 — Run 11 launch  `[ ] NOT STARTED`

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 8.1 | **12-condition pre-launch checklist** | Name 6 of the 12. What's the one that's hardest to satisfy? (sqlite backend + timestamped run-name + watcher F1 floor + ... ) | [ ] |
| 8.2 | **Timestamped `--run-name`** | Why? (Run 9's silent-overwrite incident, 2026-06-06) What does `mlflow` use the name for? | [ ] |
| 8.3 | **Watcher script (`run8_watcher.sh`)** | The F1>0.1 floor catches degenerate runs. What's the action when the floor is breached? (kill + alert) | [ ] |
| 8.4 | **First-epoch val F1 logged** | Why is first-epoch F1 a leading indicator? What's the expected range for v2 data? | [ ] |
| 8.5 | **Per-class P/R reported separately** | The F1=0.31 was macro. What are the per-class breakdowns? Which classes are the "easy" ones (high P/R) and which are the "hard" ones? | [ ] |
| 8.6 | **DeFiHackLabs re-enablement** | Once `compile_required: false` is added to config (v2.1), what's the expected DeFiHackLabs yield? (~80% per the DeFiHackLabs report) | [ ] |

---

## Data sources: what's integrated, what's deferred

### Integration status (2026-06-10)

| # | Source | Tier | Status | Yield | Notes |
|---|---|---|---|---|---|
| 1 | **SolidiFI** | T1 Gold | ✅ INTEGRATED | 283 / 350 (19% drop) | `include_subdirs: [buggy_contracts]` excludes 1,350 analysis-tool copies |
| 2 | **DIVE** | T1 Gold | ✅ INTEGRATED | 22,263 / 22,330 (0.3% drop) | 5 min 25s with 4 workers; 67 dropped are DIVE source data issues |
| 3 | **DeFiHackLabs** | T1 Gold | ⏸️ DEFERRED to v2.1 | 23 / 738 (3% drop) | Foundry project; needs `compile_required: false` or forge-std clone |
| 4 | **SmartBugs Curated** | T3 structural | ⏳ NOT STARTED | — | 143 hand-labeled contracts; smallest T1-T3 source |
| 5 | **Web3Bugs** | T1 Gold | ⏳ NOT STARTED | — | ~3,500 contest reports; largest T1 source |
| 6 | **DISL** | T4 Bronze | ⏳ NOT STARTED | — | NonVulnerable pool only; needs etherscan connector |
| 7 | **Bastet** | T1 Gold | ⏸️ DEFERRED (v2.1) | — | Replaces Code4rena scraper; legal risk |
| 8 | **FORGE** | T1 Gold | ⏸️ DEFERRED (v2.1) | — | Conditional on 50-entry agreement test ≥85% |
| 9 | **ScrawlD** | T2 Silver | ⏸️ DEFERRED (v2.1) | — | 5-tool majority voting |
| 10 | **DeFi Hacks REKT** | T2 Gold | ⏸️ DEFERRED (v2.1) | — | Verified real exploits |
| 11 | **Ethernaut** | T2 Clean | ⏸️ DEFERRED (v2.1) | — | 30 pedagogical CTF levels |
| 12 | **OpenZeppelin** | T2 Clean | ⏸️ DEFERRED (v2.1) | — | Multi-audit-verified negatives |
| 13 | **solidity_defi_vulns** | T1 | ⏸️ DEFERRED (v2.1) | — | 270 bridge examples |
| 14 | **DeFiVulnLabs** | T3 | ⏸️ DEFERRED (v2.1) | — | 48 vuln types |
| 15 | **SC-Bench** | T3 | ⏸️ DEFERRED (v2.1) | — | Audited + err-inj subsets |
| 16 | **SmartBugs Wild** | T3 | ⏸️ DEFERRED (v2.1) | — | 47K contracts (pretraining only) |
| 17 | **slither-audited** | T3 | ⏸️ DEFERRED (v2.1) | — | 467K Slither-labeled |
| 18 | **Messi-Q** | T4 | ⏸️ DEFERRED (v2.1) | — | Older Solidity era |
| 19 | **Zenodo 16910242** | T4 | ⏸️ DEFERRED (v2.1) | — | CLEAR ICSE 2025 |
| 20 | **BCCC-SCsVul-2024** | BCCC | ⏸️ DEFERRED (v2.1) | — | The dataset that caused the F1=0.31 ceiling |

### Data currently on disk (2026-06-10)

```
data/raw/
├── solidifi/         441 MB  (ingested 2026-06-10, commit 4b0573e1, 350 contracts)
├── defihacklabs/      22 MB  (ingested 2026-06-10, commit b3bc4a4a, 738 contracts; DEFERRED)
└── dive/             3.6 MB  (ingested 2026-06-10, "v1.0" pin, 22,330 contracts; symlink to staging)

data/preprocessed/
├── solidifi/         4.6 MB  (283 .sol + 283 .meta.json + dropped.csv with 67 duplicates)
├── defihacklabs/     728 KB  (23 .sol + 23 .meta.json + dropped.csv with 715 compile failures)
└── dive/             425 MB  (22,263 .sol + 22,263 .meta.json + dropped.csv with 67 compile failures)
```

---

## Real-source lessons learned (the 3 integration reports)

### SolidiFI (✅ the first success, 19% drop)
**Key finding:** the connector was scanning the entire repo, including `results/{Mythril,Slither,Smartcheck}/` (1,350 tool analysis copies). 60% of "files" were duplicates of the 350 real ones. **Fix:** `include_subdirs: [buggy_contracts]` in config. 8 new unit tests added.

**Lesson:** always run a real-source integration test on a small source first. Unit tests on synthetic fixtures don't catch scoping bugs.

### DeFiHackLabs (⏸️ deferred, 3% drop)
**Key finding:** DeFiHackLabs is a Foundry project. Every PoC imports `forge-std/Test.sol` and uses forge cheatcodes (`vm.expectRevert`, `console.log`). Standalone `solc` can't resolve them.

**Built but couldn't fix:** recursive import-strip fallback (`flattener.py` + `_transitive_strip.py` + inheritance removal). Handles the `Test` base class but can't strip `vm.*` cheatcodes from the test body without losing the actual exploit code.

**Lesson:** not all "drop" is a pipeline bug. Some are data-tooling mismatches. Defer to v2.1 for forge-std clone or `compile_required: false` flag.

### DIVE (✅ the big success, 0.3% drop)
**Key finding:** DIVE is real Ethereum mainnet contracts, pre-flattened, multi-label, with a separate labels CSV. Required:
- Generic ManualConnector (zip + symlink)
- Label-aware folderization (54,919 symlinks for 15,423 multi-label contracts)
- `__source__/` canonical subdir for clean root
- Multiprocessing pool (5 min 25s vs 18 min serial)
- `--retry-failed` mode (build-system-style incremental)

**The 67 dropped files are all genuine DIVE source data issues** (broken pragmas like `pragma solidity^0.4.24;` with no space). Reporting them to the DIVE authors is out of scope.

**Lesson:** big real datasets (22K files) need multiprocessing from day one. Retrying 67 files in 2.4s vs reprocessing all 22K in 5 min is the difference between "iterating on a config bug" and "waiting 5 min to find out it didn't help."

---

## Cross-cutting concepts (apply to multiple stages)

### The "include_subdirs" pattern
Used by every connector (git, manual). SolidiFI uses `[buggy_contracts]`, DIVE uses `[__source__]`. The contract: include only canonical source files; downstream stages can opt in to class subdirs (e.g. Stage 3 sampling from `Reentrancy/`).

### The "sidecar JSON" pattern
Stage 1 writes `*.meta.json` next to each `*.sol`. Stage 2 will write `*.rep.json` next to each `*.pt`. Every sidecar carries provenance (sha256, schema_version, extractor_version). This is what makes caching and version invalidation work.

### The "drop-not-fix" policy (with --retry-failed escape hatch)
Stage 1's plan §1.5 says drop on compile failure. That policy was right for SolidiFI/DIVE (production contracts). It was wrong for DeFiHackLabs (forge PoCs). The `--retry-failed` flag is the escape hatch: when you realize the policy was too strict, install the missing solc, retry, recover the recovered files.

### The "data/raw/" → "data/preprocessed/" → "data/<stage-output>/" chain
Each stage takes the previous stage's output as input. The directory structure mirrors the pipeline. DVC tracks this; `sentinel-data run --from-stage X` resumes from any point.

---

## What's missing (for Run 11 launch)

| Item | Stage | Effort | Blocker? |
|---|---|---|---|
| Stage 2 (representation port) | 2 | 5 days | YES — Run 11 needs graphs+tokens |
| Stage 3 (labeling for SolidiFI, DIVE) | 3 | 2 weeks | YES — Run 11 needs labels |
| Stage 4 (verification) | 4 | 1 week | YES — Run 11 needs verified labels |
| Stage 5 (splitting) | 5 | 1 week | YES — Run 11 needs train/val/test |
| Stage 6 (analysis) | 6 | 2 days | NO — can be done after Run 11 launches |
| Stage 7 (export + seam swap) | 7 | 1.5 weeks | YES — Run 11 needs the seam in place |
| Stage 8 (Run 11 launch) | 8 | 1 day | Final step |
| 4 more sources (SmartBugs, Web3Bugs, DISL, DIVE) | 1 | 2-3 days | Optional — bigger corpus = better model |
| DeFiHackLabs re-enablement | 1.5 | 2 days (with `compile_required: false`) | Optional |
| **Total to Run 11** | | **~8 weeks** | Per the v2 build schedule |

**Critical path:** Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 7 → Stage 8 (with Stage 6 in parallel after Stage 2).

---

## What changed in this revision (2026-06-10)

**Old checklist:** placeholders for Stages 2-8.
**New checklist:** captures the actual state — what's built, what's tested, what's deferred, what 3 sources have been integrated, and the cross-cutting patterns that emerged.

**New sections added:**
- Module-level state (the bird's-eye view)
- Code/test metrics
- Data sources table (20 sources with status)
- Real-source lessons learned (3 reports)
- Cross-cutting concepts (include_subdirs, sidecar JSON, drop-not-fix, pipeline chain)
- What's missing for Run 11

**Stage 1 expanded from 15 to 25 concepts** to cover the new code we wrote (manual connector, label folderize, multiprocessing, retry-failed, subdir scoping).

**Stage 2 still 9 placeholders** — won't be filled in until we start Stage 2. Currently scheduled for Jun 23-29 in the plan.
