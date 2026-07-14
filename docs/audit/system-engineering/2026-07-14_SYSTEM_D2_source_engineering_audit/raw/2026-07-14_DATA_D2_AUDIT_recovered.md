DATA audit completed read-only. No tracked files were changed.

## Architecture inventory

- 78 production Python files under `data_module/sentinel_data`, plus 9 benchmark scripts and 51 test files.
- Package ownership:
  - analysis 7
  - export 7
  - ingestion 12
  - labeling 8
  - preprocessing 10
  - registry 4
  - representation 11
  - splitting 5
  - verification 12
  - root CLI/package 2
- Actual CLI pipeline has nine stages: ingest, preprocess, represent, label, verify, split, register, analyze, export. Freshness is a separate utility, not a tenth stage.
- Implemented but unwired: labeling parsers/merger/gate, leakage auditor, Aderyn, probe utilities.
- Hard stubs: Etherscan, Hugging Face and Zenodo connectors; call graph, PDG and opcode builders.
- Enabled-source mismatch: SolidiFI, DIVE, SmartBugs, Web3Bugs and DISL are enabled, but only the first three have parsers. Web3Bugs is unpinned and references a missing crosswalk; DISL selects the stub Etherscan connector.
- DATA owns the graph/class schema; ML re-exports it. Tokenization still reaches into ML through runtime path injection.
- Main persisted state: ignored raw/preprocessed/representation/split/export directories, `.pt` pickle artifacts, Parquet tables, JSON manifests and a SQLite/YAML catalog.

Complexity hotspots:

- `representation/graph_extractor.py`: 2,057 lines; `extract_contract_graph` 490 lines and roughly 95 branching constructs.
- `cli.py`: 1,054 lines.
- `registry/catalog.py`: 542 lines.
- 40 broad `Exception`/bare handlers in ten DATA files; 23 are in `graph_extractor.py`.
- One production path-bootstrap site inserts both repository and ML roots into `sys.path`.

## Verification evidence

Clean D2 worktree command:

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp \
PYTHONPATH=data_module:ml \
/home/motafeq/projects/sentinel/.venv/bin/python \
-m pytest data_module/tests -q
```

Result: **465 passed, 13 failed, 111 skipped; 589 collected in 10.24s.**

Failures:

- Four genuine ICFG regressions: no `CALL_ENTRY` or `RETURN_TO` edges.
- Seven orchestrator tests require ignored/local `data_module/data/preprocessed/solidifi`.
- Fresh clone lacks the `data_module/data/*` directories expected by the skeleton test.
- SmartBugs recall requires ignored `ml/data/smartbugs-curated/dataset`.

No credible fresh-clone DATA throughput measurement is possible because production corpora and released artifacts are unavailable.

## Findings

### DATA-P0-01 — Archive extraction escapes its destination

- Severity/classification: **P0 defect/security exploit**
- Source: `data_module/sentinel_data/ingestion/connectors/manual_connector.py::_extract_zip`
- Evidence: containment uses string `startswith`. A ZIP member `../repo_evil/pwned.txt` with destination `.../repo` passed validation and wrote to the sibling directory. Probe result: `no_exception`, `escaped True`.
- Violated invariant: an ingested archive must never write outside its allocated workspace.
- Impact: operator filesystem overwrite during manual ZIP ingestion.
- Recommendation: use resolved `Path.is_relative_to`, reject symlinks and special entries, enforce extraction limits, extract into a temporary directory, then atomically promote it.
- Rejected alternatives: string prefix checks and `os.path.commonprefix`.
- Migration/rollback: compatible connector hardening; no artifact schema change. Discard and re-ingest archives processed from untrusted origins.
- Tests: sibling-prefix traversal, `..`, absolute paths, symlink entries, nested traversal and decompression limits.

### DATA-P0-02 — Export integrity checks do not bind the released dataset

- Severity/classification: **P0 corrupted-evidence defect**
- Sources:
  - `export/chunker.py::_HASH_EXCLUDED`
  - `export/export.py::SentinelDatasetExport.verify_artifact_hash`
- Evidence:
  - `manifest.json` is excluded from the artifact hash, although it owns split membership, shard positions, schemas, sources and label order. An existing test explicitly accepts manifest tampering.
  - After creating a valid export and warming the cache, deleting `graphs-00000.pt` still produced `verify_artifact_hash() == True`.
  - The cache iterates surviving files only and trusts mtime/size metadata.
- Violated invariant: a release commitment must bind every semantic field and the complete expected file set.
- Impact: split or shard mapping can be altered, or files removed, while the integrity gate reports success.
- Recommendation: commit a canonical manifest core and exact file inventory; place the resulting digest in a separately signed/content-addressed release descriptor. Treat mtime caches as performance hints only, never evidence.
- Rejected alternatives: excluding the manifest to avoid a circular hash, or trusting mutable filesystem metadata.
- Migration/rollback: introduce a new export format version; retain old exports as explicitly legacy/untrusted.
- Tests: manifest-field mutation, deletion, addition, same-size/same-mtime replacement, reordered index and truncated shard.

### DATA-P1-03 — Graph and token shards can silently describe different contracts

- Severity/classification: **P1 correctness/scientific risk**
- Sources:
  - `export/graph_writer.py::write_graphs_shards`
  - `export/token_writer.py::write_tokens_shards`
  - `export/chunker.py::chunk_export`
  - `ml/src/datasets/sentinel_dataset.py::SentinelDataset.__getitem__`
- Evidence: writers independently skip missing artifacts, but chunking publishes the graph index as the token index. With the middle token absent, contract 1 received contract 2’s tokens and contract 2 raised `IndexError`.
- Violated invariant: graph, tokens, label and contract ID must be an atomic row.
- Impact: silent training-label corruption and invalid evaluation.
- Recommendation: build one validated join over the graph/token/label intersection and fail publication on any unexplained asymmetry.
- Rejected alternative: independent graceful skipping.
- Migration/rollback: regenerate exports; previously published exports require an alignment audit before use.
- Tests: graph-only, token-only, missing-middle, different shard-boundary and duplicate-ID cases.

### DATA-P1-04 — The advertised full pipeline cannot complete and fails open

- Severity/classification: **P1 integration/availability defect**
- Sources:
  - `cli.py::STAGES`
  - `cli.py::_run_label`
  - `cli.py::_handle_run`
  - `cli.py::_run_split`
- Evidence:
  - Label merely prints `NOT IMPLEMENTED`.
  - `run` constructs a shared namespace without split/register/analyze/export arguments. A split-start probe raised `AttributeError: Namespace has no attribute seed`.
  - Missing prerequisites usually print and return success.
  - The documented `compute-dedup-groups` command does not exist.
  - DVC register omits required `--name`.
- Violated invariant: a successful orchestrator run must produce every declared output or return failure.
- Impact: automation can report success with no labels, splits, registration or export.
- Recommendation: typed stage inputs/results, explicit output postconditions, a real label dispatcher, configuration-derived arguments and nonzero failure propagation.
- Rejected alternative: print-and-return error handling.
- Migration/rollback: preserve individual stage commands while introducing a compatibility wrapper around the corrected orchestrator.
- Tests: full fixture pipeline, every resume point, missing prerequisite, failed gate and output-postcondition cases.

### DATA-P1-05 — Preprocessing can mutate valid Solidity into invalid or semantically different source

- Severity/classification: **P1 correctness/scientific risk**
- Sources:
  - `preprocessing/normalizer.py::normalize`
  - `preprocessing/flattener.py::flatten_contract`
  - `preprocessing/pipeline.py::PreprocessingPipeline._process_one`
  - `preprocessing/_transitive_strip.py::apply_sub_strips_to_source`
- Evidence:
  - `"https://example.com"` becomes an unterminated `"https:` string.
  - `"/*not comment*/"` becomes `""`.
  - Normalized output is compiled before normalization, not after it.
  - `solc --flatten` is unsupported by the installed compiler.
  - A valid relative-import contract passed preprocessing but was stored without its dependency and failed representation.
  - Transitive stripping writes local absolute paths and leaves `.sentinel_stripped.sol` files outside cleanup scope.
  - The contract ID hashes pre-normalized content, not the stored `.sol` bytes.
- Violated invariant: the released source must preserve Solidity lexical semantics, compile independently and have an unambiguous content identity.
- Impact: dropped representations, host-specific outputs, false deduplication and source/graph mismatch.
- Recommendation: use a Solidity-aware lexer for comment removal; either use a proven flattener or preserve a content-addressed source bundle/import graph. Compile the exact stored artifact. Record raw, flattened and normalized hashes separately.
- Rejected alternative: regex comment stripping and lossy parent/import deletion in canonical training data.
- Migration/rollback: schema/version bump, regenerate representations and retrain affected models. Preserve prior releases for rollback.
- Tests: strings containing comment delimiters/URLs, escaped quotes, imports, remappings, nested dependencies and byte/hash identity.

### DATA-P1-06 — Ingestion provenance is mutable and folderization invalidates manifests

- Severity/classification: **P1 provenance/operational defect**
- Sources:
  - `ingestion/ingest.py::ingest_source`
  - `ingestion/manifest.py::IngestionManifest.save`
  - `ingestion/connectors/git_connector.py::GitConnector._pull`
  - `preprocessing/preprocess.py::preprocess_source`
  - `ingestion/label_folderize.py::folderize_by_labels`
- Evidence:
  - Manifest documentation says append-only, but `save` overwrites one file.
  - Existing Git clones are reused without verifying remote or requested pin.
  - Previous manifests are not verified before replacement or preprocessing.
  - DIVE manifest paths are loaded before folderization moves flat files. A fixture reproduced `FileNotFoundError`.
  - With `materialize: symlink`, folderization mutates the user’s original staging directory.
- Violated invariant: raw acquisition is immutable and every derived record traces to a verified source snapshot.
- Impact: stale/wrong commits, destroyed source layout and unverifiable labels.
- Recommendation: immutable versioned ingestion snapshots; verify remote, resolved commit and prior manifest; folderize into a derived workspace before building the canonical manifest; hash label files and crosswalks.
- Rejected alternative: reusing any existing directory as authoritative.
- Migration/rollback: retain old manifests as legacy; re-ingest into new content-addressed roots.
- Tests: wrong existing commit/remote, changed manual staging, first-run flat DIVE, manifest overwrite and crosswalk change.

### DATA-P1-07 — Parallel preprocessing disables deterministic deduplication

- Severity/classification: **P1 correctness/provenance defect**
- Sources:
  - `preprocessing/parallel.py::_process_one_worker`
  - `preprocessing/parallel.py::run_preprocess_parallel`
- Evidence: every task creates a fresh `PreprocessingPipeline` and `Deduplicator`. Two identical files produced:
  - serial: 1 processed, 1 duplicate dropped
  - two workers: 2 processed, 0 dropped
  - the shared metadata file’s `original_path` depended on the last writer.
- Violated invariant: worker count must not change corpus membership or lineage.
- Impact: nondeterministic labels, leakage and overwritten provenance.
- Recommendation: parallelize compilation/normalization into immutable candidates, then perform deterministic global dedup and atomic publication centrally.
- Rejected alternative: “best-effort” process-local dedup.
- Migration/rollback: reprocess any corpus built with multiple workers.
- Tests: exact/normalized/address duplicates across worker counts and repeated-run byte equality.

### DATA-P1-08 — Split leakage and class-balance controls are not enforced

- Severity/classification: **P1 scientific risk**
- Sources:
  - `cli.py::_run_split`
  - `splitting/dedup_enforcer.py::apply_dedup_enforcer`
  - `splitting/leakage_auditor.py::run_audit`
  - `splitting/nonvulnerable_cap.py::apply_nonvulnerable_cap`
  - `splitting/splitters.py::stratified_split`
- Evidence:
  - Split relies on local `dedup_groups_graph_hash.json`; its advertised generator is absent.
  - Missing dedup groups only warn and continue.
  - Leakage auditor is never invoked by the CLI.
  - CLI ignores configured per-source project/temporal strategies and does not populate project/year fields.
  - The cap applies the global allowance independently to each split; a cap of 3 reproduced a global 9:1 ratio.
  - Post-hoc majority movement can weaken temporal/project boundaries.
- Violated invariant: release splits must be group-disjoint, policy-compliant and reproducible from tracked code.
- Impact: inflated evaluation and biased training.
- Recommendation: compute canonical group IDs in the pipeline, require complete group coverage, allocate groups directly, run independent leakage checks as a publication gate and apply the negative cap globally. Derive limits only from measured experiments.
- Rejected alternative: untracked one-off dedup files and warning-only audits.
- Migration/rollback: produce a new split version and retrain; never overwrite old split names.
- Tests: absent/stale/partial group file, cross-source duplicates, project/temporal boundaries, global cap and multilabel stratification.

### DATA-P1-09 — Verification can pass with no usable evidence and is disconnected from export

- Severity/classification: **P1 scientific/trust defect**
- Sources:
  - `verification/gate.py::run_gate`
  - `verification/negative_checker.py::NonVulnResult.status`
  - `cli.py::_run_verify`
  - `cli.py::_run_export`
- Evidence: empty `AuditResult` plus empty semantic/negative results produced `gate_passed=True`, ten provisional verdicts and negative status `OK`. Verify defaults to non-strict, while export never reads its result.
- Violated invariant: absent, skipped or errored validation cannot be interpreted as successful evidence.
- Impact: unverified or empty corpora can be exported and promoted.
- Recommendation: fail closed on absent evidence and insufficient measured coverage; emit a machine-readable verification artifact whose digest is required by export/registration.
- Rejected alternative: optional `--strict` and “no data = OK”.
- Migration/rollback: preserve report Markdown for compatibility, add a versioned quality manifest.
- Tests: empty corpus, all skipped, all tool errors, stale report, failed gate followed by export and partial coverage.

### DATA-P1-10 — Representation versioning does not uniquely determine tensors

- Severity/classification: **P1 scientific/reproducibility risk**
- Sources:
  - `representation/graph_extractor.py::_build_solc_args`
  - `representation/graph_extractor.py::_add_icfg_edges`
  - `representation/orchestrator.py::_extract_one`
  - `representation/cache_manager.py::stale_entries`
  - `representation/graph_schema.py::FEATURE_SCHEMA_VERSION`
- Evidence:
  - Orchestrator passes `allow_paths` as a list; generated solc args are `--allow-paths .,['/path']`.
  - Four clean-suite ICFG tests show missing `CALL_ENTRY`/`RETURN_TO`.
  - Current extractor encodes pre-0.8 unchecked state as `0.5` while retaining schema `v9`; comments acknowledge a deferred v10 bump.
  - Tokenizer model revision is not pinned.
  - Cache eviction derives `<sha>.rep` rather than `<sha>`, so files are not removed.
  - Feature fallbacks often return ordinary zero/safe values after broad exceptions; counters are not a release gate.
- Violated invariant: a schema/toolchain manifest must identify one canonical tensor for one source.
- Impact: operator disagreement, training/inference drift and stale graph export.
- Recommendation: pin exact extractor/compiler/Slither/tokenizer commits and environment image; make feature failures explicit; fix typed configuration and validate sidecars before export.
- Rejected alternative: relying on a broad `v9` label and mutable third-party model names.
- Migration/rollback: bump graph/extractor version and rebuild; keep old checkpoint/artifact pair available.
- Tests: current ICFG regressions, pre-0.8 golden tensors, cache-failure paths, tokenizer revision and cross-environment byte comparison.

### DATA-P1-11 — Registry is neither immutable nor connected to the final export

- Severity/classification: **P1 provenance/integration defect**
- Sources:
  - `registry/catalog.py::Catalog.add_dataset_version`
  - `registry/catalog.py::Catalog.load_artifact`
  - `cli.py::_run_register`
  - `cli.py::_resolve_corpus_paths`
- Evidence:
  - `INSERT OR REPLACE` overwrites supposedly append-only versions; probe changed the same name’s hash from `first` to `second`.
  - `load_artifact` does not verify a hash.
  - Register hashes only `split_manifest.json`, leaves config hash empty and stores the split directory as the artifact.
  - Directory paths cannot be checked by the file-only verifier.
  - Registered split directories do not contain the labels/representations/preprocessed layout expected by `analyze --corpus`.
  - Registration happens before export, and the final export is not registered.
- Violated invariant: a dataset version is an immutable, verifiable release object.
- Impact: silent version replacement and unusable lineage.
- Recommendation: register only finalized export descriptors using insert-once semantics, transactional publication, foreign-keyed split/verification/toolchain records and load-time verification.
- Rejected alternative: registering a pre-export split directory as the dataset.
- Migration/rollback: introduce catalog schema v2 and import old rows as legacy records.
- Tests: duplicate-name rejection, final export registration, directory verification, retired load and corpus analysis.

### DATA-P1-12 — Fresh-clone and package reproducibility are absent

- Severity/classification: **P1 operational/packaging risk**
- Sources:
  - `data_module/.dvc/config`
  - `data_module/dvc.yaml`
  - `data_module/config.yaml`
  - `data_module/pyproject.toml`
  - `cli.py` path bootstrap
- Evidence:
  - Zero `data_module/data` files are tracked.
  - No useful DVC data pointers or `dvc.lock`; stages track `.gitkeep`, not produced artifacts.
  - DVC is `no_scm=True` with private absolute remote `/mnt/d/sentinel-dvc-remote`.
  - Manual source paths are host-specific.
  - Standalone package omits direct dependencies required by active export/analysis code, including PyArrow, Matplotlib, NumPy and SciPy.
  - Representation eagerly imports ML code despite no package dependency.
- Violated invariant: a declared environment plus artifact manifest must reproduce the same release.
- Impact: clean operators cannot build, test or acquire the canonical DATA release.
- Recommendation: define a real workspace/package dependency graph, lock direct dependencies, publish portable DVC/CAS descriptors and classify private inputs explicitly.
- Rejected alternative: relying on the root developer virtualenv and hidden local directories.
- Tests: isolated DATA installation, offline artifact bootstrap and clean-clone module suite.

### DATA-P1-13 — Benchmark material is not valid release evidence

- Severity/classification: **P1 scientific risk**
- Sources:
  - `benchmarks/evaluate.py::main`
  - `benchmarks/contamination_check.py::build_export_index`
  - `benchmarks/sources/tier_a_existing_ood/build.py`
  - Tier C/D builders
- Evidence:
  - Evaluator prints “ACTUAL EVALUATION NOT YET IMPLEMENTED”.
  - Missing reference splits produce an empty contamination index and therefore zero reported overlap.
  - Quickstart maps known access-control and `tx.origin` vulnerabilities to `NonVulnerable`; access control conflicts with the canonical DATA mapping to `ExternalBug`.
  - Tier C and D are skeletons.
  - The tracked quickstart contains 66 Solidity contracts and 66 sidecars, but its historical contamination report embeds local absolute paths.
- Violated invariant: benchmark labels, contamination checks and metrics must be executable and schema-consistent.
- Impact: false OOD/quality claims.
- Recommendation: fail if the reference release is unavailable; align labels to the canonical taxonomy or mark out-of-scope explicitly; implement the evaluator before citing metrics.
- Rejected alternative: treating an unknown vulnerability as safe or accepting zero indexed training contracts.
- Migration/rollback: retain quickstart as historical/noncanonical material and create a new benchmark version.
- Tests: missing reference release, known overlap, taxonomy consistency and end-to-end metric generation.

### DATA-P2-14 — Responsibility and performance debt obscure failures

- Severity/classification: **P2 maintainability/performance debt**
- Sources: `graph_extractor.py`, `cli.py`, `catalog.py`, `leakage_auditor.py`, export hash paths.
- Evidence: large monolithic functions, 40 broad handlers, serial representation despite a parsed workers option, quadratic leakage checking and whole-file hashing.
- Impact: difficult review, poor observability and unclear scaling to larger corpora.
- Recommendation: split extraction into typed feature/edge passes; separate CLI orchestration from stages; stream hashes; introduce measured candidate indexing for leakage and structured error telemetry.
- Rejected alternative: arbitrary file-size rewrites without stabilizing contracts first.
- Verification: golden-contract suites, fault injection, memory/throughput benchmarks and profile-guided changes.

## Ranked DATA roadmap

1. Fix ZIP containment and export commitment/alignment before accepting any new artifacts.
2. Make verification and stage orchestration fail closed.
3. Stabilize immutable ingestion, Solidity-aware preprocessing, deterministic representation and packaging.
4. Replace local dedup/split machinery with a reproducible group-aware release pipeline.
5. Publish one immutable DATA release manifest binding sources, labels, tools, representations, verification and splits.
6. Regenerate affected data, resplit and retrain only after compatibility tests pass.
7. Implement truthful benchmark evaluation and then measure throughput/quality before setting new policy values.

For the V3 decentralized architecture, DATA should contribute a signed/content-addressed release manifest containing source snapshot hashes, taxonomy/crosswalk hashes, exact compiler/Slither/tokenizer revisions, graph schema, split/group commitments and verification evidence. Operators should consume this pinned release; mutable local DATA rebuilds must not enter consensus.