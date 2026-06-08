
## Why a Separate Data Engineering Module

Before anything else: the lesson from your five BCCC phases is that **data quality failures are silent and catastrophic** — they don't produce errors, they produce a model that learns the wrong thing. A dedicated data engineering module enforces a **hard boundary** between "what data we trust" and "what the ML module trains on." The ML module should never touch a raw dataset; it only ever consumes validated, versioned, labeled artifacts produced by this module. [slideshare](https://www.slideshare.net/slideshow/data-versioning-and-reproducible-ml-with-dvc-and-mlflow/239445797)

***

## Module Name Proposal: `sentinel-data` (or `sentinel.data`)

A standalone Python package, independently testable and deployable, with its own CLI, configuration schema, and pipeline orchestration — completely decoupled from the ML training code.

***

## High-Level Architecture

```
sentinel-data/
├── ingestion/          ← Pull from all source datasets
├── preprocessing/      ← Normalize, flatten, deduplicate .sol files
├── representation/     ← AST, CFG, PDG, opcode extraction
├── labeling/           ← Label ingestion, normalization, consensus
├── verification/       ← Label quality checks, confidence scoring
├── splitting/          ← Train/val/test splits with guarantees
├── registry/           ← Dataset versioning, lineage, catalog
├── export/             ← Output artifacts for ML module
└── analysis/           ← Exploratory tooling, class balance, drift
```

Each submodule is a pipeline stage with defined inputs, outputs, and contracts (schemas). Below is every submodule in detail.

***

## Submodule 1: Ingestion

**Purpose:** Pull raw data from every upstream source into a local raw store, reproducibly.

**Components:**

- **Source connectors** — one per dataset:
  - `GitConnector` — clones repos at pinned commits (ScaBench, Web3Bugs, DeFiHackLabs, SmartBugs, OZ, Ethernaut, Bastet)
  - `HuggingFaceConnector` — pulls HF datasets (slither-audited, solidity-defi-vulnerabilities)
  - `ZenodoConnector` — downloads Zenodo ZIP records
  - `EtherscanConnector` — fetches verified contract source by address (for reconstruction pipelines)
  - `ManualConnector` — handles offline ZIPs like Bastet's `dataset.zip` with checksums
- **Ingestion manifest** — a YAML file per source declaring URL, commit/version pin, expected checksum, and last-fetched timestamp
- **Raw store** — all ingested data lands in `data/raw/<source_name>/` with the original file structure intact; nothing is modified here
- **Freshness checker** — alerts when upstream sources have new commits and the pinned version is stale [slideshare](https://www.slideshare.net/slideshow/data-versioning-and-reproducible-ml-with-dvc-and-mlflow/239445797)

***

## Submodule 2: Preprocessing

**Purpose:** Clean and normalize raw `.sol` files into a canonical, ML-ready code corpus.

**Components:**

- **Flattener** — resolves `import` chains and produces single-file contracts using tools like `solc --flatten` or hardhat-flatten; essential for contracts that import OpenZeppelin libraries [gist.github](https://gist.github.com/dougbtv/8c8383a2442f7b7575368322eb8a5390)
- **Compiler version resolver** — detects `pragma solidity` version per file and maps to the right `solc` binary; handles version ranges gracefully
- **Syntax validator** — compiles each contract with the appropriate `solc` version and flags those that fail to parse; rejects contracts that can't be compiled rather than silently passing broken ASTs [pypi](https://pypi.org/project/solc-ast-parser/)
- **Deduplicator** — multi-level deduplication:
  - **Exact hash** — SHA-256 of normalized source text (whitespace/comment stripped)
  - **Address-level** — same Ethereum address appearing in multiple datasets maps to one canonical entry
  - **Near-duplicate detection** — AST-similarity based clustering to catch copy-paste contracts with minor edits, which would otherwise leak across train/test [themoonlight](https://www.themoonlight.io/en/review/solidiffy-ast-differencing-for-solidity-smart-contracts)
- **Normalizer** — strips comments, standardizes whitespace, removes SPDX headers and non-semantic decorators to reduce spurious feature variation
- **Contract segmenter** — for multi-contract files, splits into individual contract-level units; records parent file and inheritance chains
- **Solidity version bucketing** — tags each contract: `legacy` (< 0.6), `transitional` (0.6–0.7), `modern` (0.8+); lets the ML module filter or weight by era [arxiv](https://arxiv.org/html/2606.03387v1)

***

## Submodule 3: Representation Extraction

**Purpose:** Transform `.sol` source into every graph and token representation that Sentinel's architecture can consume.

**Components:**

- **AST extractor** — uses `solc-ast-parser` or `py-solc-x` to produce JSON ASTs per contract; stores as `<id>.ast.json` [pypi](https://pypi.org/project/solc-ast-parser/)
- **CFG builder** — extracts Control Flow Graphs per function using Slither's internal IR or a custom builder; stores as `<id>_<function>.cfg.json` [dial.uclouvain](https://dial.uclouvain.be/pr/boreal/object/boreal:311929/datastream/PDF_01/view)
- **PDG builder** — Program Dependence Graph (data + control dependencies); critical for reentrancy and access control detection where dependency chains matter [techscience](https://www.techscience.com/cmc/v86n2/64772/html)
- **Call graph extractor** — cross-contract call graph; essential for detecting inter-contract vulnerabilities (delegatecall abuse, flash loan patterns) [github](https://github.com/QuangNguyen711/GraphFusionVulDetect)
- **Opcode extractor** — compiles to bytecode and extracts opcode sequences for DIVE-style bytecode features; optional but useful for ensemble inputs [nature](https://www.nature.com/articles/s41597-026-07025-5)
- **Token sequence producer** — produces raw token sequences (BPE-tokenized Solidity code) for Transformer path; respects contract/function boundaries
- **Representation cache** — all representations are cached as files alongside the source; a manifest records which representations have been computed for each contract ID, so re-runs skip already-processed entries
- **Representation versioner** — when extractor tools change version, the cache is invalidated and recomputed; ensures representation consistency across the dataset [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0950584925001053)

***

## Submodule 4: Labeling

**Purpose:** Ingest, normalize, and harmonize labels from all upstream sources into a single unified label schema.

**Components:**

- **Label schema definition** — a YAML/JSON schema defining the canonical Sentinel vulnerability taxonomy (your 8–12 class scheme); all sources must be mapped into this schema or rejected
- **Source-specific parsers** — one per dataset:
  - `BastetLabelParser` — maps 46 Tags → Sentinel classes using a defined crosswalk table
  - `ScaBenchLabelParser` — maps severity + description → Sentinel classes (may use LLM-assist for description-to-class mapping)
  - `Web3BugsLabelParser` — maps O/L/S labels + report text → Sentinel classes
  - `DeFiHackLabsLabelParser` — maps exploit type to Sentinel class via incident metadata
  - `SmartBugsLabelParser` — direct DASP→Sentinel crosswalk
  - `SlitherLabelParser` — maps Slither detector names → Sentinel classes via `label_mappings.json` [github](https://github.com/mwritescode/slither-audited-smart-contracts)
- **Crosswalk tables** — version-controlled YAML files that define every source-to-Sentinel class mapping; human-reviewed, not auto-generated
- **Multi-label merger** — for contracts labeled by multiple sources, merges labels with explicit conflict resolution rules (e.g., expert-audit overrides tool label)
- **Label confidence scorer** — assigns a confidence tier per label:
  - `T0` = exploit-verified (DeFiHackLabs)
  - `T1` = expert-audited (Bastet, ScaBench, Web3Bugs)
  - `T2` = curated/manual (SmartBugs Curated, Ethernaut)
  - `T3` = tool-generated (Slither-Audited, Messi-Q)
  - `T4` = inferred/heuristic (anything derived from proximity to labeled code)
- **NonVulnerable label strategy** — explicit policy for how "clean" labels are assigned:
  - OZ Contracts: `T1_clean` (multi-audit verified)
  - Audited code regions with no reported findings: `T2_clean`
  - Wild/tool-clean: `T3_clean`
  - Never auto-assigns NonVulnerable without explicit source tracing

***

## Submodule 5: Verification

**Purpose:** The module that would have caught your BCCC problems before they entered training. Runs systematic checks on labels and representations.

**Components:**

- **Semantic consistency checker** — for each vulnerability class, verifies that labeled contracts actually contain the code patterns implied by that label:
  - Reentrancy → must contain external call + state change after call pattern (via AST check)
  - CallToUnknown → must contain low-level call opcode or `call{}`
  - Timestamp dependency → must reference `block.timestamp` in a condition
  - This directly catches the "86.9% CallToUnknown had no low-level call" class of failure [arxiv](https://arxiv.org/html/2606.03387v1)
- **Tool cross-validation** — runs Slither (and optionally Mythril, Semgrep) on labeled positive contracts; reports what fraction of expert-labeled positives are also flagged by tools; flags mismatches for review rather than auto-correcting
- **False positive estimator** — for each class, samples N labeled positives and runs them through multiple tools; estimates a per-class false positive rate; any class exceeding a threshold (e.g., >30% false positive rate) is flagged as low-confidence [arxiv](https://arxiv.org/html/2606.03387v1)
- **Class distribution auditor** — reports label counts per class, per source, per confidence tier; immediately surfaces class imbalance and suspicious distributions (e.g., if one source contributes 90% of a class)
- **Confidence score reporter** — outputs per-class confidence scores summarizing label quality; provides the "Timestamp 52.6%, DoS 64.5%" type of diagnostic you discovered manually in Phase 5, now automated before training
- **Negative quality checker** — for contracts labeled NonVulnerable, runs all enabled tools and reports what fraction have tool hits; anything above a configurable threshold gets flagged (catching your "41% of NonVulnerable have Slither hits" issue automatically)
- **Verification report** — a human-readable report generated after every pipeline run summarizing all findings; any run that produces a report with high-failure classes should block export to the ML module unless explicitly overridden with a documented reason

***

## Submodule 6: Splitting

**Purpose:** Produce train/val/test splits that are sound, reproducible, and leak-free.

**Components:**

- **Contract-level deduplication enforcer** — ensures a contract does not appear in both train and test (critical given multi-source corpora where the same contract may appear in several datasets under different addresses)
- **Project-level split** — for audit datasets (ScaBench, Web3Bugs, Bastet), splits at project level rather than contract level; prevents the model from memorizing project-specific style and calling it generalization
- **Stratified splitter** — maintains label distribution across splits per class and per confidence tier; ensures test set has the same T0/T1/T2/T3 mix as the training set
- **Temporal split support** — for time-sensitive experiments, can split by contract deployment date or audit date (pre-2023 train, post-2023 test), to test against distribution shift [arxiv](https://arxiv.org/html/2606.03387v1)
- **Leakage audit** — runs deduplication checks across the split boundary after splitting; reports any near-duplicates that crossed the boundary
- **Split manifest** — a versioned JSON file recording exactly which contract IDs go into train/val/test, the random seed, and the split strategy used; reproducible by anyone with the same data version

***

## Submodule 7: Registry (Data Versioning & Lineage)

**Purpose:** Make every dataset artifact traceable, versioned, and reproducible — the "DVC layer" for this module. [discuss.dvc](https://discuss.dvc.org/t/integrating-mlflow-and-dvc-for-robust-machine-learning-lifecycle-management/2644)

**Components:**

- **DVC integration** — all large files (raw corpora, representation caches, labeled CSVs) tracked via DVC with a remote backend (S3/GCS/local); Git tracks `.dvc` pointer files [discuss.dvc](https://discuss.dvc.org/t/integrating-mlflow-and-dvc-for-robust-machine-learning-lifecycle-management/2644)
- **Dataset catalog** — a central YAML/JSON catalog of every named dataset version:
  - `sentinel-v2-gold-train-2026-06` as an example entry
  - Records: source versions, preprocessing config hash, label schema version, split seed, statistics summary
- **Lineage graph** — for every output artifact, records the full chain of transformations applied (which ingestion connectors, which preprocessing steps, which label parsers, which crosswalk table version)
- **Artifact hashing** — SHA-256 of every exported artifact is recorded and verified on load; the ML module refuses to train on an artifact whose hash doesn't match the registry
- **Changelog** — human-readable log of every time a dataset version changes and why; helps you explain model performance differences across runs by tracing them to data changes
- **Dataset diff tool** — given two dataset versions, shows what changed: which contracts were added/removed, which labels changed, class distribution deltas [slideshare](https://www.slideshare.net/slideshow/data-versioning-and-reproducible-ml-with-dvc-and-mlflow/239445797)

***

## Submodule 8: Export

**Purpose:** Produce the final artifacts consumed by the ML module, in the exact formats it expects.

**Components:**

- **Graph exporter** — serializes AST/CFG/PDG graphs to PyTorch Geometric `Data` objects or DGL graphs; one file per contract
- **Token sequence exporter** — produces tokenized sequences with attention masks ready for Transformer input
- **Label tensor exporter** — produces multi-label tensors with confidence-tier masks; ML module can use tier masks for loss weighting
- **Metadata exporter** — CSV with contract ID, source dataset, Solidity version, label class(es), confidence tier, split assignment
- **Split-aware export** — always exports train/val/test as separate artifacts; never exports a single unsplit blob
- **Format versioner** — records the exact export format schema version; the ML module declares which schema version it expects; mismatches are caught at load time, not silently corrupted
- **Compression & chunking** — large corpora (slither-audited) are chunked into manageable shards with an index file; prevents OOM during loading in training

***

## Submodule 9: Analysis & Exploration

**Purpose:** Tooling for understanding the data before and after pipeline runs; critical for catching the "model learns complexity proxy" class of failure early.

**Components:**

- **Class balance visualizer** — plots per-class label counts by source and confidence tier; instant alert on severe imbalance
- **Feature distribution analyzer** — for each class, computes and plots distributions of graph-level statistics (node count, edge count, cyclomatic complexity, call depth); if distributions differ wildly between classes, the model is likely to learn complexity rather than semantics — your exact Run 9 problem [arxiv](https://arxiv.org/html/2606.03387v1)
- **Inter-dataset overlap detector** — pairwise similarity analysis between source datasets to quantify how much they share; informs weighting decisions
- **Label co-occurrence matrix** — for multi-label contracts, shows which vulnerability classes tend to co-occur; helps design label-aware loss functions
- **Solidity version distribution** — breakdown of contracts by version across sources; informs whether you need version-stratified training
- **Data drift monitor** — compares statistical properties of a new dataset version against the current one; flags if distributions have shifted significantly before the ML module is retrained
- **Interpretability probe dataset** — a small hand-curated set of ~50 contracts per class where the vulnerability is visually obvious in the code; used to probe the model's attention/attribution after training to verify it's actually learning the right patterns, not shortcuts [arxiv](https://arxiv.org/html/2606.03387v1)

***

## Pipeline Orchestration

The entire module is orchestrated as a **DAG pipeline** where each stage is a node with declared inputs and outputs. Recommended stack:

- **DVC pipelines** (`dvc.yaml`) for data version-aware orchestration and caching [discuss.dvc](https://discuss.dvc.org/t/integrating-mlflow-and-dvc-for-robust-machine-learning-lifecycle-management/2644)
- Each stage is independently re-runnable; changing a crosswalk table re-runs only labeling + downstream stages, not ingestion or representation extraction
- A single CLI command (`sentinel-data run`) executes the full pipeline end-to-end or from a specific stage
- **Docker container** wraps the entire module with pinned tool versions (`solc`, `slither`, `py-solc-x`); ensures that representation extraction is reproducible across machines

***

## Summary: Module Component Map

```
sentinel-data/
│
├── ingestion/
│   ├── connectors/        (Git, HF, Zenodo, Etherscan, Manual)
│   ├── manifest.yaml      (source pins + checksums)
│   └── freshness.py       (staleness alerts)
│
├── preprocessing/
│   ├── flattener.py
│   ├── compiler.py        (version resolver + syntax validator)
│   ├── deduplicator.py    (hash + address + near-dup)
│   ├── normalizer.py
│   └── segmenter.py       (multi-contract splitter)
│
├── representation/
│   ├── ast_extractor.py
│   ├── cfg_builder.py
│   ├── pdg_builder.py
│   ├── call_graph.py
│   ├── opcode_extractor.py
│   ├── token_producer.py
│   └── cache_manager.py
│
├── labeling/
│   ├── schema/            (canonical taxonomy YAML)
│   ├── parsers/           (one per source)
│   ├── crosswalk/         (version-controlled mapping tables)
│   ├── merger.py          (multi-source conflict resolution)
│   └── confidence.py      (T0–T4 tier assignment)
│
├── verification/
│   ├── semantic_checker.py
│   ├── tool_validator.py  (Slither/Mythril cross-check)
│   ├── fp_estimator.py
│   ├── class_auditor.py
│   └── report_generator.py
│
├── splitting/
│   ├── dedup_enforcer.py
│   ├── stratified_splitter.py
│   ├── temporal_splitter.py
│   ├── leakage_auditor.py
│   └── split_manifest.py
│
├── registry/
│   ├── catalog.yaml
│   ├── lineage_tracker.py
│   ├── artifact_hasher.py
│   └── dataset_diff.py
│
├── export/
│   ├── graph_exporter.py  (PyG / DGL)
│   ├── token_exporter.py
│   ├── label_exporter.py
│   ├── metadata_exporter.py
│   └── format_schema/
│
├── analysis/
│   ├── balance_viz.py
│   ├── feature_dist.py
│   ├── overlap_detector.py
│   ├── cooccurrence.py
│   ├── drift_monitor.py
│   └── probe_dataset/
│
├── dvc.yaml               (pipeline DAG)
├── config.yaml            (global config)
└── cli.py                 (sentinel-data CLI)
```

The most critical submodule for Sentinel's specific history is **Verification** — if this had existed before Phase 1 of BCCC, you would have caught the CallToUnknown and Reentrancy problems in hours rather than after five full training phases. That submodule alone is worth the entire architectural split. [arxiv](https://arxiv.org/html/2606.03387v1)