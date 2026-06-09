# Sentinel v2 — Data Module Integration Proposal

**Date:** 2026-06-08
**Status:** Approved Path A (full split), pending build
**Author:** SENTINEL data engineering track
**Supersedes:**
- `docs/proposal/Data_Module_Proposals/first_proposal_data_module.md` (architecture outline, v1)
- `docs/proposal/Data_Module_Proposals/Sentinel_v2_Dataset_Proposal.md` (dataset tier list, v1)

**This document is binding for the v2 build.** Both predecessors are kept for historical reference.

---

## 0. TL;DR

The SENTINEL model ceiling on Run 9 (ep52, fixed F1 = 0.2965, tuned F1 = 0.3081) is not a model-architecture problem — it is a **label-quality** problem caused by 89% Reentrancy FP, 86.9% CallToUnknown FP, and 87.9% pre-0.8 / 12.1% 0.8.x Solidity version skew in BCCC. Phase 5 verification (2026-06-08) reduced 67,311 raw BCCC labels down to 7,403 verified positives, but the cleanest path forward is to **stop relying on BCCC as the primary ground-truth source** and rebuild the data pipeline against a curated multi-source corpus.

We are creating a new standalone package, **`sentinel-data`**, hosted at the existing `~/projects/sentinel/Data/` directory (which already contains the BCCC deep-dive outputs from Phases 1–5 and is the natural home for it). This package owns every concern from raw contract ingestion through verified, versioned, multi-label dataset export. **`sentinel-ml` never sees a raw contract** — it only consumes registry-versioned, hash-verified artifacts emitted by `sentinel-data`.

**Build window:** ~10 weeks (2026-06-09 → 2026-08-17); Run 11 launches 2026-08-18.
**First v2 training run:** Run 11 ("v2-baseline"), launch 2026-08-18, on **17 sources** (ScaBench + Web3Bugs + Bastet + DeFiHackLabs + DIVE + FORGE + SolidiFI + ScrawlD + Code4rena + DeFi Hacks REKT + Zenodo 16910242 + SmartBugs Curated + 5 more Tier 3-4 — see §6 below).
**Window extension reason:** 5 new Tier-1/Tier-2 sources from `datasources_suggestions.md` (applied 2026-06-08) added 3 weeks to Stage 3 (labeling crosswalks + parsers). The extension is worth it: 5 new high-quality sources (peer-reviewed DIVE, ICSE 2026 FORGE, 100%-ground-truth SolidiFI, 5-tool-consensus ScrawlD, human-auditor Code4rena, real-exploit REKT) directly address the BCCC failure pattern (89% Reentrancy FP, 86.9% CallToUnknown FP).

---

## 1. Why a separate data engineering module

The five BCCC phases (2026-06-06 → 2026-06-08) cost 14 calendar days of deep work and produced a 1,000-line debugging session whose root cause was a single architectural mistake: the ML module was **directly reading a raw CSV with folder-based labels** that no one had verified end-to-end before training. The data failures did not raise errors; they raised a model that learned the wrong thing. Run 9's per-class snapshot — `IntegerUO = 0.698, GasException = 0.376, Reentrancy = 0.164, DoS = 0.164` — is the exact signature of a model that has learned `complexity` as a proxy for vulnerability (see `ml/interpretability_results/phase2_run7_ep39_v10_2026-06-04/` and the Phase 2 interpretability report).

A dedicated data engineering module enforces a **hard boundary** between "what data we trust" and "what the ML module trains on." The ML module should never touch a raw dataset; it only ever consumes validated, versioned, labeled artifacts produced by this module. Concretely:

- Every artifact written by `sentinel-data` carries a SHA-256 of its bytes, a content schema version, and a lineage manifest recording which ingestion connectors, preprocessing steps, and label parsers produced it.
- `sentinel-ml`'s `SentinelDataset` refuses to load an artifact whose hash is not in the registry's `known_artifacts` table.
- The two packages have independent `pyproject.toml` files, independent test suites, independent lockfiles, and independent Docker base images.
- `sentinel-ml` has a *runtime* dependency on `sentinel-data` (consumes the exported artifacts), but `sentinel-data` has zero runtime dependency on `sentinel-ml` (the data layer is allowed to know nothing about the model).

This boundary is the only structural defense against the Phase-4 / Phase-5 class of silent-failure.

---

## 2. Module name and physical location

| Property | Value |
|---|---|
| **Package name (import)** | `sentinel_data` |
| **Distribution name (PyPI/poetry)** | `sentinel-data` |
| **CLI command** | `sentinel-data` |
| **Physical root** | `~/projects/sentinel/Data/` (existing folder, repurposed) |
| **Git-tracked?** | Yes (this folder) |
| **Large artifacts** | Tracked by DVC under `Data/.dvc/`, NOT committed raw |
| **Python venv** | New `Data/.venv/` (separate from `ml/.venv/`) |
| **Docker image** | `Data/docker/Dockerfile.data` — pinned `solc-select`, `slither-analyzer`, `py-solc-x`, `solc-ast-parser` |

**Why `Data/` and not a new top-level `sentinel-data/` directory:** the existing `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/` folder already contains 5 phases (120+ files) of validation work that *is* the historical foundation of this module. Promoting `Data/` to a first-class package — adding `pyproject.toml`, `src/`, `tests/`, `dvc.yaml`, and a `connectors/` subpackage — preserves the deep-dive work as the "Phase 0: BCCC post-mortem" section of the new module's docs. The deep-dive phase folders become a `Data/docs/legacy/bccc_deep_dive/` subdirectory.

**Layout of the new module:**

```
Data/                                                    ← sentinel-data root
├── pyproject.toml                                       ← name = "sentinel-data", packages = [{include = "sentinel_data"}]
├── poetry.lock
├── dvc.yaml                                             ← full pipeline DAG
├── Dockerfile.data                                      ← pinned solc/slither build env
├── README.md                                            ← module map
├── sentinel_data/                                       ← installable package
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── connectors/
│   │   │   ├── git.py                ← GitConnector (ScaBench, Web3Bugs, Bastet, DeFiHackLabs, SmartBugs, OZ, Ethernaut)
│   │   │   ├── huggingface.py        ← HFConnector (slither-audited, solidity-defi-vulnerabilities)
│   │   │   ├── zenodo.py             ← ZenodoConnector (record 16910242)
│   │   │   ├── etherscan.py          ← EtherscanConnector (verified source by address)
│   │   │   └── manual.py             ← ManualConnector (offline ZIPs with checksums)
│   │   ├── manifest.py               ← ingestion manifest (YAML): source, URL/pin, checksum, last-fetched
│   │   └── freshness.py              ← staleness alert: when pinned version is N commits behind upstream HEAD
│   ├── preprocessing/
│   │   ├── flattener.py              ← resolves import chains → single-file (solc --flatten / hardhat-flatten)
│   │   ├── compiler.py               ← pragma → solc version resolution + syntax validation
│   │   ├── deduplicator.py           ← SHA-256 exact + Ethereum-address + AST-near-dup clustering
│   │   ├── normalizer.py             ← strip comments/headers, normalize whitespace
│   │   ├── segmenter.py              ← multi-contract file → per-contract units + inheritance chain
│   │   └── version_bucketer.py       ← legacy (<0.6) / transitional (0.6–0.7) / modern (0.8+) tags
│   ├── representation/
│   │   ├── ast_extractor.py          ← REWRITTEN from ml/src/data_extraction/ast_extractor.py
│   │   ├── graph_extractor.py        ← MOVED from ml/src/preprocessing/graph_extractor.py
│   │   ├── graph_schema.py           ← MOVED from ml/src/preprocessing/graph_schema.py (v9 → v10 bump here)
│   │   ├── tokenizer.py              ← MOVED + EXTENDED from ml/src/data_extraction/tokenizer.py (CodeBERT windowed)
│   │   ├── cfg_builder.py            ← NEW: standalone CFG via Slither IR
│   │   ├── pdg_builder.py            ← NEW: program-dependence graph (data + control deps)
│   │   ├── call_graph.py             ← NEW: cross-contract call graph (delegatecall, flash loans)
│   │   ├── opcode_extractor.py       ← NEW: DIVE-style bytecode features
│   │   ├── cache_manager.py          ← content-addressed rep cache + manifest
│   │   └── versioner.py              ← invalidates cache on schema/extractor version change
│   ├── labeling/
│   │   ├── schema/
│   │   │   ├── taxonomy.yaml         ← CANONICAL 10-class taxonomy (frozen until Run 11 baseline lands)
│   │   │   └── crosswalks/           ← per-source source→Sentinel-class mapping YAMLs
│   │   │       ├── bastet.yaml
│   │   │       ├── scabench.yaml
│   │   │       ├── web3bugs.yaml
│   │   │       ├── defihacklabs.yaml
│   │   │       ├── solidity_defi_vulns.yaml
│   │   │       ├── smartbugs_curated.yaml
│   │   │       ├── smartbugs_wild.yaml
│   │   │       ├── openzeppelin.yaml
│   │   │       ├── ethernaut.yaml
│   │   │       ├── slither_audited.yaml
│   │   │       ├── messi_q.yaml
│   │   │       └── zenodo_16910242.yaml
│   │   ├── parsers/                  ← one per source (each consumes crosswalk + raw)
│   │   │   ├── bastet.py
│   │   │   ├── scabench.py
│   │   │   ├── web3bugs.py
│   │   │   ├── defihacklabs.py
│   │   │   ├── solidity_defi_vulns.py
│   │   │   ├── smartbugs_curated.py
│   │   │   ├── smartbugs_wild.py
│   │   │   ├── openzeppelin.py
│   │   │   ├── ethernaut.py
│   │   │   ├── slither_audited.py
│   │   │   ├── messi_q.py
│   │   │   └── zenodo_16910242.py
│   │   ├── merger.py                 ← multi-source label merger with conflict resolution
│   │   └── confidence.py             ← T0 (exploit-verified) → T4 (heuristic) tier assignment
│   ├── verification/                 ← the module that would have caught the BCCC problems in hours
│   │   ├── semantic_checker.py       ← AST-level: Reentrancy must have call+state-after; CallToUnknown must have low-level call; etc.
│   │   ├── tool_validator.py         ← runs Slither + (optional) Mythril + Semgrep on positives
│   │   ├── fp_estimator.py           ← samples N positives per class, runs all tools, reports per-class FP rate
│   │   ├── class_auditor.py          ← per-class count, source breakdown, confidence-tier breakdown
│   │   ├── negative_checker.py       ← NonVulnerable contracts: what fraction have Slither hits? threshold flag
│   │   ├── probe_dataset.py          ← ~50 hand-curated positive contracts per class for model interpretability probes
│   │   └── report_generator.py       ← human-readable verification_report.md (the Phase 5 output template)
│   ├── splitting/
│   │   ├── dedup_enforcer.py         ← prevents same contract across train/val/test
│   │   ├── project_splitter.py       ← project-level split for audit datasets (Bastet, ScaBench, Web3Bugs)
│   │   ├── stratified_splitter.py    ← per-class + per-confidence-tier stratification
│   │   ├── temporal_splitter.py      ← optional: split by deployment date / audit date
│   │   ├── leakage_auditor.py        ← post-split near-dup check across split boundary
│   │   └── split_manifest.py         ← versioned JSON: contract IDs, seed, strategy
│   ├── registry/
│   │   ├── catalog.yaml              ← named dataset versions: sentinel-v2-gold-2026-08 etc.
│   │   ├── lineage_tracker.py        ← per-artifact lineage graph (which connectors/steps/parsers produced it)
│   │   ├── artifact_hasher.py        ← SHA-256 every exported artifact; ML module verifies on load
│   │   ├── dataset_diff.py           ← v_old vs v_new: added/removed contracts, label changes, distribution deltas
│   │   └── changelog.md              ← human-readable log of every dataset version change
│   ├── export/
│   │   ├── graph_writer.py           ← PyG Data → shard writer (one .pt per contract OR one sharded .pt per N)
│   │   ├── token_writer.py           ← tokenized windowed sequences → shard writer
│   │   ├── label_writer.py           ← multi-label tensor + confidence-tier mask → parquet
│   │   ├── metadata_writer.py        ← contract_id, source, solc_version, classes, confidence, split
│   │   ├── format_schema/            ← exported artifact schema versions (v1.0, v1.1, …)
│   │   └── chunker.py                ← large corpora → sharded with index file (prevents OOM at load time)
│   ├── analysis/
│   │   ├── balance_viz.py            ← per-class / per-source / per-tier counts
│   │   ├── feature_dist.py           ← node-count / edge-count / complexity / call-depth per class — CATCHES the complexity-proxy problem
│   │   ├── overlap_detector.py       ← pairwise source overlap (e.g., is SmartBugs Curated a subset of Wild?)
│   │   ├── cooccurrence.py           ← class co-occurrence matrix (informs multi-label loss design)
│   │   ├── drift_monitor.py          ← dataset version drift (new version vs baseline)
│   │   └── probe_dataset.py          ← aliased from verification/probe_dataset.py
│   ├── cli.py                        ← `sentinel-data run [--from-stage N] [--config config.yaml]`
│   └── dvc_dag.py                    ← programmatic DAG definition (also rendered to dvc.yaml)
├── tests/                            ← standalone pytest suite, no ml/ imports allowed
│   ├── test_ingestion/
│   ├── test_preprocessing/
│   ├── test_representation/
│   ├── test_labeling/
│   ├── test_verification/
│   ├── test_splitting/
│   ├── test_registry/
│   └── test_export/
├── docs/
│   ├── INDEX.md                      ← module map (like ml/README.md)
│   ├── architecture.md               ← DAG + data flow diagrams
│   ├── legacy/
│   │   └── bccc_deep_dive/           ← MOVED from Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/
│   │       ├── README.md
│   │       ├── Phase1_Exploration_2026-06-05/
│   │       ├── Phase2_Validation_2026-06-06/
│   │       ├── Phase3_DeepAnalysis_2026-06-06/
│   │       ├── Phase4_LabelValidation_2026-06-07/
│   │       └── Phase5_LabelVerification_2026-06-08/
│   ├── decisions/                    ← ADRs (Data/ADR-0001, etc.)
│   └── changes/                      ← per-stage changelogs
├── config.yaml                       ← global pipeline config (sources enabled, version pins, thresholds)
├── data/                             ← (gitignored except for .dvc pointers)
│   ├── raw/                          ← per-source raw corpora (immutable mirror of upstream)
│   ├── preprocessed/                 ← flattened, deduped, normalized .sol files
│   ├── representations/              ← .pt graphs, token shards
│   ├── labels/                       ← per-source label CSVs, merged canonical labels
│   ├── verification/                 ← per-run verification reports
│   ├── splits/                       ← versioned split manifests + .npy index files
│   └── exports/                      ← registry-versioned training artifacts (this is what ml reads)
├── .dvc/                             ← DVC tracking
├── .venv/                            ← standalone venv
├── logs/                             ← run logs (gitignored)
└── scripts/                          ← top-level orchestrators (the rest of the code lives in sentinel_data/)
    ├── run_full_pipeline.sh
    ├── run_stage.sh
    └── verify_artifact.sh
```

**Total surface area:** ~22 new Python files in `sentinel_data/` + 11 source-specific label parsers + 11 crosswalk YAMLs + 6 verification scripts + 6 splitting scripts + 5 export writers + 5 analysis tools + ~25 test files. The DVC pipeline has 9 stages. The CLI exposes 1 `run` command plus per-stage subcommands.

---

## 3. The 9 submodules — inputs, outputs, contracts

Each submodule is a DVC pipeline stage. The `dvc.yaml` file is the authoritative DAG. The submodules are loosely coupled: each takes an artifact path as input and writes an artifact path as output. The contract between them is the artifact schema, versioned explicitly in `sentinel_data/export/format_schema/`.

### 3.1 Ingestion

| Property | Value |
|---|---|
| **Input** | `config.yaml` source list (URLs / HF dataset names / Zenodo IDs / local ZIPs) |
| **Output** | `data/raw/<source_name>/...` — unmodified mirror of upstream; checksum file per source |
| **Contract** | ingestion manifest: `{source, version, url, sha256, fetched_at, contract_count}` |
| **Re-run trigger** | manual OR `freshness.py` alert (upstream HEAD moved > N commits) |
| **Connectors** | Git, HuggingFace, Zenodo, Etherscan, Manual (5 connectors, 1 per dataset family) |

Source connectors are **one per dataset family, not one per dataset** — `GitConnector` handles ScaBench, Web3Bugs, Bastet, DeFiHackLabs, SmartBugs, OpenZeppelin Contracts, OpenZeppelin Ethernaut (all git-cloneable with pinned commit). The HF and Zenodo connectors each handle one family. The `EtherscanConnector` is for the reconstruction pipeline (fetching verified source by address — needed for some Tier-3 audit datasets). The `ManualConnector` is the fallback for offline ZIPs.

### 3.2 Preprocessing

| Property | Value |
|---|---|
| **Input** | `data/raw/<source_name>/` |
| **Output** | `data/preprocessed/<source_name>/<sha256>.sol` + sidecar `meta.json` (per-file metadata) |
| **Side effects** | flattens imports, strips comments, normalizes whitespace, deduplicates, segments multi-contract files, tags Solidity version bucket |
| **Sidecar fields** | `{sha256, source, original_path, contract_count, version_bucket, inheritance_root, dedup_group_id, parent_sha256}` |

Deduplication is **three-level**:
1. **Exact SHA-256** of normalized source text — drops whitespace/comment-only differences.
2. **Address-level** — same Ethereum address appearing in multiple sources maps to one canonical entry.
3. **Near-dup AST clustering** — SoliDiffy-style AST similarity score; anything above 0.92 cosine similarity merges into the same `dedup_group_id`. This is the most important level: it catches the copy-paste-with-minor-edits leakage that bit us on BCCC (38.8% duplication rate).

The `version_bucketer` adds `legacy | transitional | modern` tag per file. The audit found Run 8's failure was partly a version skew (training 87.9% pre-0.8 vs test_contracts 100% 0.8.x) — version-stratified splitting will be **opt-in but on by default** for v2.

### 3.3 Representation

| Property | Value |
|---|---|
| **Input** | `data/preprocessed/<source_name>/<sha256>.sol` |
| **Output** | `data/representations/<source_name>/<sha256>.pt` (PyG Data) + `<sha256>.tokens.pt` (windowed CodeBERT) |
| **Sidecar** | `data/representations/<source_name>/<sha256>.rep.json` — `{schema_version, extractor_version, node_count, edge_count, window_count, compute_time_ms}` |
| **Cache key** | `(sha256 of source, schema_version, extractor_version)` — auto-invalidate on either bump |

This is the module that absorbs the existing `ml/src/data_extraction/*` and `ml/src/preprocessing/*` code. **No extraction logic is rewritten in v1** — the only changes are (a) module rename (`sentinel_data.representation.ast_extractor` etc.), (b) add `cfg_builder.py` / `pdg_builder.py` / `call_graph.py` / `opcode_extractor.py` as new builders, (c) the schema version bumps from `v8` → `v9` → `v10` happen here (the v9/v10 schema additions live in `ml/src/preprocessing/graph_schema.py` — they move to `sentinel_data/representation/graph_schema.py`).

The `versioner.py` ensures that changing the schema version invalidates the entire representation cache. This is critical — silently mixing v9 and v10 graphs in the same dataset is exactly the failure mode that produced Run 8's regression.

**New builders in v2 (do not exist in v1):**
- `cfg_builder.py` — uses Slither's internal IR; produces per-function CFG in a normalized form.
- `pdg_builder.py` — data + control dependence graph; needed for reentrancy and access-control where dependency chains matter.
- `call_graph.py` — cross-contract call graph; needed for delegatecall abuse, flash loan patterns, proxy patterns.
- `opcode_extractor.py` — compiles to bytecode, extracts opcode sequences; provides DIVE-style features for ensemble use.

These four are **additive** for the v2 baseline — Run 11 trains on AST+CFG as v1 did, and PDG / call graph / opcode are exposed for v3+ architectures.

### 3.4 Labeling

| Property | Value |
|---|---|
| **Input** | `data/preprocessed/<source>/<sha256>.sol` + per-source `crosswalks/<source>.yaml` + per-source raw label file |
| **Output** | `data/labels/<source>/<sha256>.labels.json` — `{source, contract_id, classes: {cls: {value, confidence, source_tier}}, primary_class, n_pos}` |
| **Crosswalk format** | version-controlled YAML, human-reviewed, never auto-generated |
| **Confidence tiers** | T0 (exploit-verified) → T1 (expert-audited) → T2 (curated/manual) → T3 (tool-generated) → T4 (heuristic) |

This is the module that fixes the BCCC failure. The 11 source-specific parsers each consume their crosswalk YAML and the raw label file from the source, and emit a per-contract JSON label file with the **canonical 10-class taxonomy** as the key set. The `merger.py` handles multi-source contracts (a single contract appearing in Bastet and ScaBench gets labels from both, with explicit conflict resolution — expert audit overrides tool label by default).

**The crosswalk YAMLs are the highest-leverage artifact in the whole module.** They are the human decisions that map a source's idiosyncratic taxonomy (Bastet has 46 finding tags; Web3Bugs uses O/L/S; ScaBench uses free-text descriptions) into the canonical 10 classes. Each crosswalk is reviewed by a human (Ali) before being committed. Per source:

- `bastet.yaml` — 46 tags → 10 classes (Bastet has a partial crosswalk in its README; we extend it)
- `scabench.yaml` — 555 free-text descriptions → 10 classes (this is the hardest; LLM-assist for description→class)
- `web3bugs.yaml` — O/L/S severity + report text → 10 classes
- `defihacklabs.yaml` — exploit type → 10 classes (high confidence — these are real exploits)
- `solidity_defi_vulns.yaml` — `is_real` flag + `attack_title` → 10 classes
- `smartbugs_curated.yaml` — DASP categories → 10 classes (direct mapping)
- `smartbugs_wild.yaml` — derived from Slither detector hits (T3)
- `openzeppelin.yaml` — clean negative source (T1_clean, multi-audit verified)
- `ethernaut.yaml` — pedagogical positives (T2)
- `slither_audited.yaml` — Slither detector JSON → 10 classes (T3)
- `messi_q.yaml` — derived labels (T3)
- `zenodo_16910242.yaml` — Yizhou Chen's CLEAR dataset (T2/T3 mix)

The `confidence.py` module assigns T0–T4 per (contract, class) pair. Multi-source contracts get the **max tier** across sources (DeFiHackLabs T0 + Slither T3 → T0 for the class).

### 3.5 Verification — the module that would have caught BCCC in hours

| Property | Value |
|---|---|
| **Input** | `data/labels/<source>/*.labels.json` + `data/representations/<source>/*.pt` |
| **Output** | `data/verification/<run_id>/verification_report.md` + per-class gate tables |
| **Hard gate** | any class with `gate == "FAIL"` blocks downstream export unless explicitly overridden in `config.yaml` with documented reason |
| **Soft gate** | any class with `gate == "PROVISIONAL"` or `"BEST-EFFORT"` exports but emits a warning into the registry catalog |

This is the module that the Phase 5 work was the prototype for. The 6 components:

1. **semantic_checker.py** — for each (class, contract) pair, verifies the contract actually contains the code pattern implied by the class label:
   - Reentrancy → external call + state change after call (via AST pattern match)
   - CallToUnknown → must have low-level `.call{}` or `.delegatecall{}` or `.send()` or `.transfer()` (this single check would have caught 86.9% of the BCCC CallToUnknown FPs in minutes)
   - Timestamp → must reference `block.timestamp` / `now` in a condition
   - Reentrancy + CallToUnknown + ExternalBug + GasException + DoS — all have AST-level patterns
2. **tool_validator.py** — runs Slither (and optionally Mythril, Semgrep) on every labeled positive; reports the per-class tool agreement rate
3. **fp_estimator.py** — samples N positives per class, runs all tools, computes a per-class empirical FP rate; flags anything > 30%
4. **class_auditor.py** — per-class count, per-source breakdown, per-confidence-tier breakdown; surfaces "one source contributes 90% of a class" automatically
5. **negative_checker.py** — for `NonVulnerable` contracts, runs all enabled tools and reports what fraction have tool hits; flags anything > threshold (catches the BCCC "41% of NonVulnerable have Slither hits" pattern)
6. **probe_dataset.py** — exports a hand-curated set of ~50 contracts per class where the vulnerability is visually obvious in the code (Phase 5's `review_batches/` is the seed; we expand it); used downstream by the model interpretability suite

The `report_generator.py` produces the human-readable `verification_report.md` (the same template Phase 5 used: per-class gate, confidence histogram, per-class drop counts, top reasons for drops). This is the document the data module emits to the registry as a release gate.

### 3.6 Splitting

| Property | Value |
|---|---|
| **Input** | `data/labels/merged/labels.parquet` + `data/preprocessed/dedup_groups.json` (dedup group → [sha256s]) |
| **Output** | `data/splits/<version>/{train,val,test}.parquet` + `split_manifest.json` + `splits.dvc` |
| **Guarantees** | (a) no contract in 2 splits, (b) no dedup group straddles a split, (c) no project straddles a split (for audit datasets), (d) per-class distribution within ±2% across splits |
| **Strategies** | random, stratified, project-level, temporal (pre-2023 / post-2023), per-confidence-tier stratified |
| **Default** | stratified per class + per confidence tier, project-level for audit datasets |

`dedup_enforcer.py` runs AFTER `stratified_splitter.py` and reassigns any near-dup group that straddles a split. `leakage_auditor.py` is a separate post-split pass that does its own similarity check (independent of `dedup_enforcer`) and reports any leak it finds — if `leakage_auditor` finds something `dedup_enforcer` missed, that's a bug to fix.

The `split_manifest.json` is the contract: `{version, seed, strategy, contract_counts, class_distributions, dedup_groups_resolved, generated_at}`. The ML module reads the manifest before loading — refuses to load a split that doesn't match the registered version.

### 3.7 Registry

| Property | Value |
|---|---|
| **Storage** | SQLite database at `data/registry/catalog.db` + human-readable mirror at `data/registry/catalog.yaml` |
| **Tables** | `sources` (per-source pin), `artifacts` (per-exported-artifact hash + lineage), `splits` (per-split-version seed+strategy), `dataset_versions` (named composite: source set + preprocessing config + split version) |
| **Catalog entry** | `sentinel-v2-gold-2026-08` → `{sources: [bastet@<sha>, scabench@<sha>, ...], preprocessing_config: <hash>, split_version: <hash>, label_schema_version: v2, export_format: v1, generated_at: <iso>, verification_report: <path>, artifact_hash: <sha>}` |
| **ML-module interface** | `sentinel_data.registry.load_artifact(name) → Artifact` validates hash and returns path. The ML module's `SentinelDataset.__init__` calls this on construction. |

DVC tracks the raw + intermediate files; the registry tracks the **composite dataset versions** that combine them. The dataset diff tool (`dataset_diff.py`) shows what changed between two versions: added/removed contracts, label changes, class distribution deltas.

### 3.8 Export

| Property | Value |
|---|---|
| **Input** | `data/labels/merged/labels.parquet` + `data/splits/<version>/` + `data/representations/<source>/*.pt` |
| **Output** | `data/exports/<dataset_version>/{graphs, tokens, labels, metadata}/` + `manifest.json` |
| **Format** | sharded: `graphs-{shard:05d}.pt` (PyG `Batch`), `tokens-{shard:05d}.pt`, `labels.parquet`, `metadata.parquet` |
| **Shard size** | 5,000 contracts per shard (configurable); index file maps `contract_id → shard` |
| **Schema version** | `export_format_schema_version` in `sentinel_data/export/format_schema/v1.yaml` |
| **ML-module reader** | `sentinel_data.export.SentinelDatasetExport` — what `sentinel-ml`'s new thin dataset class wraps |

This is the **hard contract** between the two packages. The `format_schema/v1.yaml` is the spec:

```yaml
# sentinel_data/export/format_schema/v1.yaml
# Field dimensions are Pinned to the live v9 schema constants; do NOT edit
# without bumping export_format_schema_version and adding a migration test.
# Source of truth: sentinel_data/representation/graph_schema.py
#   FEATURE_SCHEMA_VERSION="v9", NODE_FEATURE_DIM=12, NUM_EDGE_TYPES=12, NUM_CLASSES=10
version: "1.0"
graphs_shard:
  type: torch_geometric.data.Batch
  fields: {x: float32[12], edge_index: int64[2,E], edge_attr: int64[E], y: int64[1]}
tokens_shard:
  type: torch.Tensor
  shape: [N, 4, 512]
  fields: {input_ids: int64, attention_mask: int64}
labels_parquet:
  type: pandas.DataFrame
  columns: {contract_id: str, source: str, class_<i>: int8 (for i in 0..9), confidence_<i>: float32, split: str}
metadata_parquet:
  type: pandas.DataFrame
  columns: {contract_id: str, source: str, solc_version: str, version_bucket: str, loc: int32, n_functions: int16, n_pos: int8, primary_class: str, confidence_tier: str}
shard_index:
  type: dict[str, int]  # contract_id → shard_number
manifest:
  type: dict
  required: [dataset_version, export_format_schema_version, n_contracts, n_shards, graph_schema_version, label_schema_version, artifact_sha256]
```

The ML module's `SentinelDataset` (in `ml/src/datasets/sentinel_dataset.py`, replacing `dual_path_dataset.py`) takes a `SentinelDatasetExport` and yields `(graph, tokens, y, contract_id, confidence_tier)` per `__getitem__`. The collate function is identical to the existing `dual_path_collate_fn` modulo the new `confidence_tier` field.

**Schema-dim gate test (v2-readiness gate, added 2026-06-09 post-friend-review):** the `SentinelDatasetExport` schema's `x.shape[-1]` MUST equal `sentinel_data.representation.NODE_FEATURE_DIM` (12, not 11). This test runs in Stage 7's CI: if a future schema bump changes the dim, the test fails loud and the loader refuses the export. Prevents the silent shape mismatch that would have occurred if the v1 spec (`x: float32[11]`) propagated into code.

### 3.9 Analysis

| Property | Value |
|---|---|
| **Input** | any pipeline stage output |
| **Output** | `data/analysis/<run_id>/` — plots + CSVs |
| **Tools** | `balance_viz`, `feature_dist`, `overlap_detector`, `cooccurrence`, `drift_monitor`, `probe_dataset` |
| **Hard rule** | the `feature_dist` tool emits a `complexity_proxy_risk.md` report — if the per-class node-count / edge-count / cyclomatic complexity distributions differ by more than 1.5σ between any pair of classes, the report flags it as a high-risk indicator that the model is likely to learn a complexity proxy (the Run 9 failure mode) |

The `drift_monitor` is the gate between two dataset versions: if the new version has feature distributions that differ significantly from the baseline, the ML module is warned before retraining.

---

## 4. The integration seam with `sentinel-ml`

`dual_path_dataset.py` does two jobs at once and is the only file in `sentinel-ml` that needs a real change. The split:

**Before (single file, two concerns):**
```
sentinel-ml/src/datasets/dual_path_dataset.py:
  - reads <md5>.pt + <md5>.pt from disk
  - reads cached_dataset_v10.pkl (paired dict)
  - validates schema version
  - returns torch Dataset
  - collate_fn batches into Batch + token tensors
```

**After (split across the two packages):**
```
sentinel-data/src/sentinel_data/export/graph_writer.py:
  - reads raw <sha256>.pt + tokens + labels.parquet + metadata.parquet
  - writes sharded graphs/tokens/labels/metadata files
  - returns SentinelDatasetExport (just paths + manifest)

sentinel-ml/src/datasets/sentinel_dataset.py:  (NEW FILE, replaces dual_path_dataset.py)
  - takes a SentinelDatasetExport
  - validates hash against sentinel_data.registry
  - returns torch Dataset wrapping the export's shards
  - collate_fn is identical to dual_path_collate_fn
  - additionally returns (graph, tokens, y, contract_id, confidence_tier)
```

The ML module keeps a `from sentinel_data.export import SentinelDatasetExport` import and a `sentinel-data >= 0.1.0` dependency in its `pyproject.toml`. The data module has zero imports from `sentinel-ml`. The boundary is enforced at install time.

---

## 5. Build order — ~10 weeks (revised 2026-06-09 with critical-path corpus)

> **🚨 POST-FRIEND-REVIEW (2026-06-09):** The build order is updated to reflect the **5 critical-path + 12 additive** source split (§6.1–6.2). The 3-week Stage 3 budget is now realistic (5 critical-path crosswalks × 1-2 days = 5-10 days; the 12 additive are v2.1 work). New Stage 3 exit criteria include the **Go/No-Go minimum-viable-corpus gate** (§6.5). New Stage 4 task: **SmartBugs Curated recall test** (≥90%). New Stage 5 task: **NonVulnerable 3:1 cap**. New Stage 7 task: **slither transitive dep test** (per §8).

| Week | Stage | Deliverable | Exit criteria |
|---|---|---|---|
| **1** (Jun 9–15) | **0. Skeleton + Data/ restructure** | `Data/pyproject.toml`, `Data/sentinel_data/__init__.py`, `Data/dvc.yaml` (no remote; local-only per §10 item 3), `Data/Dockerfile.data` (bookworm), move `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/` to `Data/docs/legacy/bccc_deep_dive/`, write `Data/README.md` and `Data/docs/architecture.md`. **Config lists 5 critical-path + 12 additive sources + 1 critical-path negative (DISL) + 2 deferred (BCCC, ReentrancyStudy) per §6.1–6.2.** | `poetry install` works in `Data/.venv/`; `sentinel-data --help` runs; DVC initialized; config.yaml lists sources with `critical_path`/`additive` flags |
| **2** (Jun 16–22) | **1. Ingestion + Preprocessing** | All 5 original connectors + 3 new connectors (AuditReportConnector for FORGE, RektScraper for DeFi Hacks, EtherscanConnector for DISL); flattener, two-pass compiler, deduplicator (threshold 0.85), normalizer, segmenter, version_bucketer (with `has_unchecked_block`); first test on DeFiHackLabs clone (critical-path #1) | `sentinel-data ingest --source defihacklabs` clones at pinned commit; SHA-256 verified; 30 raw .sol files pass preprocessing |
| **3** (Jun 23–29) | **2. Representation (port from ml/)** | Move `ml/src/data_extraction/*` and `ml/src/preprocessing/graph_extractor.py` + `graph_schema.py` to `sentinel_data/representation/` (v9 schema); **36-issue pre-Run-8 audit regression test**; **schema-dim gate test** (x.shape[-1] == 12 per §3.8); CFG builder only (PDG/call-graph/opcode DEFERRED to v3.1); representation cache + versioner (invalidates on Slither version) | `sentinel-data represent --source defihacklabs` produces 30 .pt files; existing Run 9 graphs re-loadable through the new path with byte-identical output; all 13 critical A1-A38 fixes preserved |
| **4-6** (Jun 30–Jul 20) | **3. Labeling (3 weeks for 5 critical-path + up to 12 additive crosswalks + parsers)** | `taxonomy.yaml` (10-class v1-checkpoint order), **5 critical-path crosswalk YAMLs (DeFiHackLabs, SolidiFI, DIVE, SmartBugs Curated, Web3Bugs)** + DIVE "bad randomness" dropped (§6.3.3) + DIVE class mapping documented (§6.4); 5 critical-path parsers; merger (with 99% DoS↔Reentrancy de-duplication); **CallToUnknown merge rule** (§6.3.2, human-checked); **FORGE 50-entry agreement test** (§6.5); confidence.py (T0-T4); **Go/No-Go minimum-viable-corpus gate** (§6.5) | `sentinel-data label --source defihacklabs` produces 30 per-contract `.labels.json` files; merged labels for 30 contracts match a hand-checked reference; 99% co-occurrence regression test passes; **minimum-viable-corpus gate returns 0** (or decision to defer Run 11 documented) |
| **7** (Jul 21–27) | **4. Verification (the BCCC-failure catcher)** | `semantic_checker.py` (with CEI ordering for Reentrancy, library-call detection), `tool_validator.py`, `fp_estimator.py` (stratified by source+tier), `class_auditor.py` (with co-occurrence matrix), `negative_checker.py` (5% threshold), `probe_dataset.py` (40 per class + trivial pos/neg), `report_generator.py`; per-stage p5_s1→p5_s6 regression; **SmartBugs Curated recall test** (≥90% on the 143 hand-labeled contracts per friend review) | Re-run the Phase 5 verification logic on BCCC through the new module; output matches Phase 5 report to within 0.5% per-class; **semantic_checker retains ≥90% of SmartBugs Curated positives** |
| **8** (Jul 28–Aug 3) | **5. Splitting + Registry** | `dedup_enforcer.py`, `project_splitter.py`, `stratified_splitter.py` (with **NonVulnerable 3:1 cap per §6.3.1**), `temporal_splitter.py`, `leakage_auditor.py`, `split_manifest.py`, `catalog.yaml` + SQLite catalog (with schema_migrations + dataset_version_retirements tables), `lineage_tracker.py`, `artifact_hasher.py` (shared with inference cache), `dataset_diff.py` (with per-class metric projection) | `sentinel-data split --config split-config.yaml` produces `splits/v1/{train,val,test}.parquet` with manifest; **NonVulnerable class count ≤ 3× total positive count**; `sentinel_data.registry.load_artifact("sentinel-v2-dryrun-2026-08")` returns the artifact; 0 leakage; v1.4 BCCC labels preserved in retirement chain |
| **9** (Aug 4–5) | **6. Analysis** | `balance_viz.py`, `feature_dist.py` (with **label-conditional distribution** catching L4 complexity-proxy on data side), `cooccurrence.py` (directed + conditional), `overlap_detector.py`, `drift_monitor.py` (with label distribution KS test), `probe_dataset.py` re-export; DVC-tracked outputs | `feature_dist` flags synthetic complexity skew; `complexity_proxy_risk.md` GREEN; 99% DoS↔Reentrancy directed co-occurrence visible |
| **10** (Aug 6–17) | **7. Export + Seam Swap + Predictor fix + EMITS fix** | Export writers + SentinelDatasetExport; new `sentinel-ml/src/datasets/sentinel_dataset.py` (with v9 schema gate + tier field + threshold warning + **slither transitive dep test** per §8); **predictor.py tier-threshold fix (F8/F10)**; **EMITS edge fix (Interp-6)**; archive deleted scripts to `ml/scripts/_legacy_data_pipeline/`; bookworm Docker base | Export→import round-trip byte-identical; 8 fixed bugs preserved; 7 v2-readiness gates GREEN; Docker build succeeds; **`pip install sentinel-data` brings slither-analyzer transitively** |
| **+1 day** (Aug 18) | **8. Run 11 launch ("v2-baseline")** | `ml/scripts/train.py --run-name v2-baseline-$(date +%Y%m%d-%H%M%S) --dataset-version sentinel-v2-gold-2026-08` (sqlite MLflow, gnn-prefix-warmup=5, lambda=0.005, per-class P/R report, watcher copy of run8 with F1>0.1 floor) | Training starts cleanly; first epoch val F1 logged; per-class P/R documented; 12-condition pre-launch checklist verified |

**Risk register for the build:**

| Risk | Mitigation |
|---|---|
| Phase 5 verification logic is in 7 ad-hoc scripts in `Data/Deep_Dive/.../Phase5_LabelVerification_2026-06-08/scripts/` — they need to be refactored into the 6 verification modules | Week 5 has a "regression test" that the new module reproduces the old per-class drop counts; if it diverges, halt and fix before continuing |
| Moving `graph_extractor.py` and `graph_schema.py` from `ml/src/preprocessing/` to `sentinel_data/representation/` is a circular import risk if `ml/src/inference/preprocess.py` still imports from the old location | Week 3 includes a parallel-import test: both `from sentinel_data.representation.graph_extractor import ...` and the old `from src.preprocessing.graph_extractor import ...` must produce identical graphs for the same input; the old import path is removed in Week 7 |
| `ml/src/inference/preprocess.py` calls `extract_contract_graph()` for single-contract inference — the new data module needs to support online single-contract mode | Add a `sentinel_data.representation.extract_single(source_text, schema_version=None) → Data` thin wrapper; the inference path uses this directly, bypassing the cache |
| Schema version bump (v8 → v10) will invalidate the 41K .pt graphs in `ml/data/graphs/` | They get re-extracted in Week 7 as part of the end-to-end run; old graphs are archived in `Data/data/archive/graphs_v8_pre_run11/` |
| The 41K BCCC graphs on disk are referenced from `ml/src/datasets/dual_path_dataset.py` (path constants) — moving the loader breaks every active run script | The `ml/data/` directory stays in place physically; only the *loader* changes; existing graphs are re-homed into `Data/data/exports/sentinel-v1-bcc-2026-06/` and the old path is deprecated |

**Friend-review risks (added 2026-06-09):**

| Risk | Mitigation |
|---|---|
| 5 critical-path crosswalks still take 5-10 working days (1-2 days each); if 1 source's crosswalk reveals a structural issue (e.g. SolidiFI's injection patterns don't generalize), the source may be removed from critical-path, falling below 5 sources | The Go/No-Go gate (§6.5) catches this; if 1 critical source is removed, the gate decision is documented and the v2.1 launch is accepted as Run 12 |
| DISL URL is TBD; if no download mirror is found, NonVulnerable 3:1 cap cannot be filled | The 3:1 cap is a ceiling, not a floor; if DISL is unavailable, the cap is filled from OZ Contracts (deferred to v2.1) at a lower scale |
| FORGE 50-entry agreement test may return <85%; if so, FORGE is deferred to v2.2 | The 5-source critical path is independent of FORGE; if FORGE is deferred, Run 11 ships without it (per §6.5 gate) |
| CallToUnknown < 300 verified (per Phase 5 having 245); if all 5 critical-path sources still leave us below 300, the merge rule triggers | The merge rule (§6.3.2) is a 1-line config change; the merger pauses and asks Ali before applying; the merge is reversible in v2.1 |
| BCCC 1.4 verified labels may be re-introduced as a gold supplement in v2.1; if the v1.4 labels are themselves noisy (Phase 5 had its own limitations), the v2.1 re-introduction may reproduce a smaller version of the BCCC failure | The v1.4 labels are used as **negative cross-validation** only in v2.1 (run a Stage 4 verification on them; if they fail semantic_checker at >30%, they are not re-introduced) |

---

## 6. The dataset sources — what's in and what's out (updated 2026-06-09 with friend-review critical-path corpus)

> **🚨 POST-FRIEND-REVIEW (2026-06-09):** The 2026-06-08 design treated all 17 sources as equally load-bearing for Run 11. Friend-review identified this as the **single biggest timeline risk** (Stage 3's 17 crosswalks × 1-3 days each = 3-6 weeks, not the budgeted 3 weeks). This section is rewritten with:
>
> 1. **Critical-path corpus** — 5 sources that Run 11 ships with (Stage 3 budget: 5-15 working days, realistic)
> 2. **Additive sources** — 12 sources deferred to v2.1 (post-Run-11); included in v2 build only if Stage 3 finishes early
> 3. **Negative-pool source** — DISL added for NonVulnerable scale, capped at 3:1 ratio
> 4. **Code4rena → Bastet substitution** — Bastet (curated C4 dataset) replaces the C4 scraper (legal risk + redundant work)
> 5. **ReentrancyStudy dropped entirely** — 230K single-class labels recreates the BCCC imbalance pattern; URL was TBD anyway
> 6. **DIVE "bad randomness" dropped from DIVE labels** — no equivalent in 10-class taxonomy
> 7. **CallToUnknown < 300 verified → merge into ExternalBug** — human-checked decision rule, not silent auto-merge
>
> See [`actionable_plans/learning_docs/friend_review_2026-06-09.md`](actionable_plans/learning_docs/friend_review_2026-06-09.md) for the per-idea analysis + verdicts. (If file doesn't exist, see the conversation history from 2026-06-09 where the analysis was performed.)

### 6.1 Critical-path corpus (Run 11 ships with these 5 + DISL negatives)

**5 critical-path sources (Run 11 baseline):**

| # | Source | Tier | Why critical | Crosswalk difficulty |
|---|---|---|---|---|
| 1 | **DeFiHackLabs** | T1 Gold | Exploit-PoC `_exp.sol` files are self-documenting; folder name = exploit type → class is direct. **T0 confidence** on real exploits. | LOW (1 day) |
| 2 | **SolidiFI** | T1 Gold (promoted) | 9,369 injected bugs, **100% ground-truth certainty** (mathematically guaranteed by injection). 7 types map to 7 of our 10. | MEDIUM (1-2 days) |
| 3 | **DIVE** | T1 Gold (promoted) | 22,330 contracts, **peer-reviewed** (Nature Scientific Data 2025), 8 DASP classes, multi-label. The only source that addresses the "one folder = one vuln" BCCC fiction. | MEDIUM (1-2 days) |
| 4 | **SmartBugs Curated** | T3 Structural | **143 hand-labeled contracts** (already on disk at `ml/data/smartbugs-curated/`); DASP → 10 classes direct. Critical because it's the **ground-truth probe for the semantic_checker recall test** (Stage 4.11). | LOW (0.5 day) |
| 5 | **Web3Bugs** | T1 Gold | Contest-verified real exploits (C4/Sherlock/Immunefi); O/L/S severity filter. | MEDIUM (1-2 days) |

**DISL as negative-pool source (3:1 cap, per §6.3 below):**
- DISL = 514,506 unlabeled Solidity files; used for **NonVulnerable class only**
- Capped at 3× the total positive count (subsample in Stage 5)
- **No crosswalk needed** (no labels)
- **Defer to v2.1 if 5-source positive volume already hits NonVulnerable target**

**Total critical-path volume estimate:** ~1,200 positive contracts + 3,600 DISL negatives (3:1 cap, conservative) = **~4,800 contracts for Run 11 v2-baseline**. This is **explicitly smaller than Run 9's BCCC corpus** (60K+) — Run 11 is a *cleaner* baseline, not a *bigger* one. The bigger Run 12 (post-this-launch) will add the 12 deferred sources for scale.

### 6.2 Additive sources (deferred to v2.1, post-Run-11)

These 12 sources are **not gating Run 11**. They are added to the v2.1 build *if* Stage 3 finishes the 5 critical-path crosswalks + 1-2 more within the 3-week budget. The order below is the recommended v2.1 priority (highest-impact first):

| # | Source | Tier | Why deferred | Re-introduction gate |
|---|---|---|---|---|
| 1 | **Bastet** (replaces C4 scraper) | T1 Gold | 849 findings from 394 C4 reports; HIGH-effort crosswalk (46 tags → 10 classes). Friend: Code4rena scraper has legal risk + Bastet covers the same ground. | v2.1 (W11) |
| 2 | **FORGE** | T1 Gold | ICSE 2026, CWE-classified audit reports. **50-entry agreement test required** before shipping; if <85% LLM agreement, defer to v2.2. | v2.1 (gated on 50-entry test) |
| 3 | **ScrawlD** | T2 Silver | 6,780 mainnet contracts, 5-tool majority voting. Useful scale. | v2.1 (W12) |
| 4 | **DeFi Hacks REKT** | T2 Gold (T0) | Real-exploit scale (3,216 incidents); needs `RektScraper` connector. | v2.1 (W12) |
| 5 | **Ethernaut** | T2 Clean | 30 pedagogical CTF levels. Easy to add (LOW effort). | v2.1 (W12) |
| 6 | **OpenZeppelin Contracts** | T2 Clean | Multi-audit-verified negatives. Easy to add (VERY LOW effort). | v2.1 (W12) |
| 7 | **SolidiFI** *(if not already in critical-path)* | T1 | Already in critical-path #2. | (no-op) |
| 8 | **solidity_defi_vulns** (HF) | T1 Gold | 270 bridge examples. LOW effort. | v2.1 (W12) |
| 9 | **DeFiVulnLabs** | T3 Structural | 48 vuln types, Foundry-style. MEDIUM effort. | v2.1 (W13) |
| 10 | **SC-Bench** | T3 Structural | Two sub-parsers (audited + err-inj). MEDIUM effort. | v2.2 |
| 11 | **SmartBugs Wild** | T3 Bronze | 47K contracts, friend warns 97% FP. Use for **pretraining only**, not as labeled data. | v2.2 (pretraining) |
| 12 | **slither-audited (HF)** | T3 Bronze | 467K Slither-labeled, friend confirms corroborative-not-authoritative. Down-weight in loss. | v2.2 (corroboration) |
| 13 | **Messi-Q** | T4 Bronze | 40K raw + 12K labeled, older Solidity era. | v2.2 (backup) |
| 14 | **Zenodo 16910242 (CLEAR)** | T4 Bronze | 88.3 MB, Yizhou Chen's ICSE 2025 dataset. | v2.1 (W12) |
| 15 | **ReentrancyStudy-Data** | DROPPED | 230K single-class reentrancy labels recreates BCCC imbalance. URL TBD. **NOT in v2 build.** | NEVER (use DeFiHackLabs + DIVE reentrancy examples) |
| 16 | **DISL** *(as NonVulnerable scale)* | T4 Bronze | 514K unlabeled; 3:1 cap subsample. | v2.1 (W12, after critical-path stabilizes) |
| 17 | **BCCC-SCsVul-2024** | DEFERRED | 89% Reentrancy FP, 86.9% CallToUnknown FP. v1.4 verified labels preserved at `Data/docs/legacy/bccc_deep_dive/.../contracts_clean_v1.4.csv`; may re-introduce as gold supplement in v2.1. | v2.1 (decision pending v2 baseline) |

**Total 5 critical + 12 additive = 17 sources** matches the 2026-06-08 source count. The framing change is: critical-path **gates** Run 11; additive **gates** v2.1.

### 6.3 Class-distribution rules (NEW — friend review)

#### 6.3.1 NonVulnerable 3:1 cap (prevents BCCC-style class imbalance)

**Rule:** in the final train split, the NonVulnerable class count is capped at **3× the total positive count across all 10 classes**.

**Why 3:1, not higher:** the BCCC failure had a 51:1 negative:positive ratio (515K DISL-style negatives vs ~10K positives). A 3:1 ratio forces the model to learn positive patterns (because it can't default to "predict negative" and be right 97%+ of the time) while still providing enough negative signal for the NonVulnerable class.

**Implementation:** in `Stage 5 (Splitting)`, the `stratified_splitter.py` enforces `pipeline.negative.positive_ratio_max: 3.0` from `config.yaml`. The subsample is stratified by source so the OZ clean contract distribution is preserved within the cap.

**Default override:** the 3:1 cap is the v2 default. If a class needs more negatives (e.g. NegativeVulnerable is the dominant signal for a class), the ratio is per-class overridable in `config.yaml`.

#### 6.3.2 CallToUnknown < 300 verified → merge into ExternalBug (human-checked)

**Rule:** if Stage 3 verification produces **fewer than 300 verified CallToUnknown positives** across all enabled sources, the merger de-duplicates CallToUnknown → ExternalBug. **This is a human-checked decision, not a silent auto-merge.**

**Why:** Phase 5 had 245 CallToUnknown positives; <300 in v2 is the same statistical-unlearnable problem. A 10-class softmax with 1 class at <500 samples forces the loss to be dominated by the larger classes (same BCCC failure pattern at smaller scale).

**Why merge, not keep-10-classes:** "Call to an address whose code we couldn't resolve" and "interaction with an untrusted external contract" are semantically adjacent. The merge is **reversible** in v2.1: if we accumulate 1000+ verified CallToUnknown, split them back.

**Why human-checked, not silent:** the Stage 3 plan adds a per-class count check after verification. If CallToUnknown < 300, the merger **pauses and asks Ali** before applying the merge. The merge is a 1-line config change (`labels.class_map: {CallToUnknown: ExternalBug}`); it's cheap to do at the last minute but irreversible without retraining.

**Decision rule, formalized in `config.yaml`:**
```yaml
pipeline:
  class:
    merge_rules:
      - trigger: if CallToUnknown_verified_count < 300
        action: pause_and_ask_human
        apply: labels.class_map.CallToUnknown = ExternalBug
        reversible: true
```

#### 6.3.3 DIVE "bad randomness" → drop from DIVE labels

DIVE has 8 DASP classes. The 10-class Sentinel taxonomy maps cleanly for 7 of them (reentrancy, access control, arithmetic, unchecked low-level calls, DoS, time manipulation, front-running). The 8th — "bad randomness" — has **no equivalent in the 10-class taxonomy** (it's not Timestamp, not CallToUnknown, not DoS).

**Decision:** drop "bad randomness" from DIVE labels (loses ~80 contracts of data). Alternatives considered and rejected: (a) merge into ExternalBug (semantically wrong), (b) add 11th class (breaks taxonomy lock). Documented in the DIVE crosswalk YAML as a comment for v2.1 to consider.

#### 6.3.4 ReentrancyStudy → dropped (no 5K cap, full drop)

Friend suggested a 5K cap. Decision: **drop entirely** (not cap). Reasons: (a) URL is TBD anyway, (b) 230K single-class recreates the 99% co-occurrence pattern at larger scale, (c) DeFiHackLabs + DIVE + SolidiFI already give plenty of reentrancy examples. A 5K cap is a band-aid; the underlying imbalance is structural.

### 6.4 DIVE class mapping (per-class decisions, pre-Stage 3)

Per the friend review and §6.3.3 above, the DIVE → Sentinel 10-class crosswalk is:

| DIVE DASP class | Maps to | Notes |
|---|---|---|
| reentrancy | Reentrancy | Direct |
| access_control | ExternalBug | Sub-tier note: "may include UnusedReturn edge cases" |
| arithmetic | IntegerUO | Direct |
| unchecked_low_level_calls | CallToUnknown | Direct (subject to merge rule §6.3.2) |
| denial_of_service | DoS | Direct |
| **bad_randomness** | **DROPPED** | No equivalent — see §6.3.3 |
| front_running | TOD | Direct |
| time_manipulation | Timestamp | Direct |

### 6.5 Critical-path minimum viable corpus gate (NEW — friend review)

**Definition:** the "minimum viable corpus" (MVC) is the set of criteria that, if not met, **Run 11 is deferred to v2.1 (Run 12)** rather than training on a broken corpus.

| Criterion | Threshold | If below |
|---|---|---|
| Total contracts (5 critical-path + DISL negatives) | ≥ 4,000 | Defer Run 11 |
| Per-class positive count (Reentrancy, DoS, IntegerUO) | ≥ 300 each | Defer Run 11 |
| Per-class positive count (other 7 classes) | ≥ 100 each | Defer Run 11 OR apply merge rule §6.3.2 |
| Total verified CallToUnknown | ≥ 300 | Apply merge rule §6.3.2 (NOT a defer trigger) |
| Tool agreement on SmartBugs Curated (143 contracts) | ≥ 80% recall | Defer Run 11 (semantic_checker is broken) |
| FORGE LLM agreement (50-entry test, if FORGE included) | ≥ 85% | Defer FORGE to v2.1 (not Run 11) |

**Go/No-Go gate location:** end of Stage 3 (after all 5 critical-path crosswalks + parsers + merger + confidence tiers are applied). The gate is automated: `sentinel-data verify --min-viable-corpus` exits 0 if all 6 criteria are met, non-zero otherwise. The Stage 3 plan adds this gate.

**If the gate fails:** the 12 additive sources are evaluated for fast re-introduction (deferring the affected source rather than the whole run). If no fast re-introduction is possible, the Run 11 launch slips to Run 12 and the 10-week timeline is re-baselined.

### 6.6 Original source table (preserved for reference, see v1.1 below)

**Friend's key contributions** (see `datasources_suggestions.md` for full context):
- **DIVE promoted from Tier 4 → Tier 1** (Nature Scientific Data 2025, peer-reviewed, multi-label, 22,330 contracts)
- **FORGE added Tier 1** (ICSE 2026, LLM-driven audit-report extraction, CWE classification)
- **SolidiFI promoted to Tier 1** (ISSTA 2020, injection ground truth, 100% certainty)
- **ScrawlD added Tier 2 silver** (MSR 2022, 5-tool majority voting)
- **Code4rena added Tier 2 gold** (human expert auditors, requires audit-report scraper connector)
- **DeFi Hacks REKT added Tier 2 gold** (verified real exploits, T0 tier)
- **DISL + ReentrancyStudy-Data + DeFiVulnLabs added Tier 3-4** (volume/structural sources)
- **Friend's 3-tool ensemble warning** validates the existing tool-validator design (D-4.3): tool agreement is corroborative, NOT authoritative
- **Friend's 97% SmartBugs Wild FP rate** confirms the existing Tier 3 conservative design for `smartbugs_wild` is correct — no change needed

| Tier | Source | Connector | Custom work | Build week | Notes |
|---|---|---|---|---|---|
| **Critical-path (Run 11)** | DeFiHackLabs · SolidiFI · DIVE · SmartBugs Curated · Web3Bugs | (5 sources) | See §6.1 | W3 | Run 11 ships with these 5 |
| **Critical-path (neg)** | DISL (3:1 cap) | (1 source) | See §6.1 | W2 (no crosswalk) | NonVulnerable pool |
| **Additive (v2.1, deferred)** | Bastet · FORGE · ScrawlD · DeFi Hacks REKT · Ethernaut · OZ Contracts · solidity_defi_vulns · DeFiVulnLabs · SC-Bench · SmartBugs Wild · slither-audited · Messi-Q · Zenodo 16910242 | (12 sources) | See §6.2 | v2.1 (W11–13) | Deferred to v2.1 |
| **DROPPED** | ReentrancyStudy · DIVE bad_randomness | — | — | — | See §6.3.3–6.3.4 |
| **DEFERRED** | BCCC-SCsVul-2024 | — | — | — | v2.1 (decision pending) |

**v1.1 (2026-06-08) original 17-source table — preserved for audit trail:**

| Tier | Source | Connector | Custom work | Build week | Notes |
|---|---|---|---|---|---|
| **T1 Gold** | Bastet | `GitConnector` | **HIGH** — needs audit-report parser to map `dataset.csv` findings → `.sol` source locations via Bastet's `reports/<id>.md` mapping. 46 finding tags → 10 classes is the crosswalk work. | W3 | 849 fully annotated findings; modern DeFi; the gold anchor |
| **T1 Gold** | ScaBench | `GitConnector` (clone + run `checkout_sources.py`) | **MEDIUM** — `checkout_sources.py` exists; needs wrapper. 555 free-text descriptions → 10 classes is LLM-assist work. | W2 (connector), W3 (parser) | Real audit ecosystems (C4/Cantina/Sherlock) |
| **T1 Gold** | Web3Bugs | `GitConnector` | **MEDIUM** — `bugs.csv` + `contests.csv` + report text; needs report-text → class mapper (LLM-assist). O/L/S severity filter. | W2 (connector), W3 (parser) | Contest-verified; semantic vs simple bug separation is valuable |
| **T1 Gold** | DeFiHackLabs | `GitConnector` | **LOW** — exploit PoC `_exp.sol` files are self-documenting; per-folder incident type → class is direct. | W2 | Highest-confidence positives (real exploits) |
| **T1 Gold** | solidity-defi-vulnerabilities (HF) | `HFConnector` | **LOW** — HF row schema is structured; `is_real` flag + `attack_title` → class is mostly direct. | W2 | Bridge dataset; 270 examples |
| **T1 Gold (NEW)** | **DIVE** | `ManualConnector` (Nature Sci. Data download) | **MEDIUM** — peer-reviewed, multi-label (avoids BCCC's "one folder = one vuln" fiction). 22,330 contracts, 8 DASP classes → 10 Sentinel classes (with 2 extras: bad randomness, time manipulation). Per-contract multi-hot label. | W3 | **Friend-suggested, promoted from Tier 4 to Tier 1** because peer-reviewed. |
| **T1 Gold (NEW)** | **FORGE** | `AuditReportConnector` (NEW — LLM extraction from audit PDFs/Markdown) | **HIGH** — needs a new `audit_report_connector` that ingests real audit reports. CWE-classified (industry standard). 46 CWE IDs → 10 classes is the crosswalk work. LLM-assist for the initial draft; human review for every mapping. | W3 (connector + parser) | **Friend-suggested Tier 1** (ICSE 2026). Highest-quality labels (human auditor). |
| **T1 Gold (PROMOTED)** | **SolidiFI Benchmark** | `GitConnector` (already cloned to `ml/data/SolidiFI-benchmark/`) | **MEDIUM** — 9,369 injected bugs with 100% ground-truth certainty (mathematically guaranteed by injection). 7 types map to our 10. | W3 (parser) | **Friend-suggested promotion from Tier 4 → Tier 1** because of guaranteed ground truth. |
| **T2 Clean** | OpenZeppelin Contracts | `GitConnector` | **VERY LOW** — clean negative source; OZ's own `audits/` folder provides audit evidence. `T1_clean` tier for all contracts. | W2 | Multi-audit verified; the strongest practical negative pool |
| **T2 Clean** | OpenZeppelin Ethernaut | `GitConnector` | **VERY LOW** — 30 levels; each is a known vulnerability pattern → positive label. | W2 | Pedagogical; useful for debugging class definitions |
| **T2 Silver (NEW)** | **ScrawlD** | `GitConnector` | **MEDIUM** — 6,780 mainnet contracts, 5-tool majority voting (3/5 agreement). The 5 tools: Slither, Mythril, Manticore, Securify, Smartcheck. Crosswalk requires 3/5 agreement for positive label. | W3 | **Friend-suggested Tier 2 silver** (MSR 2022). Tool-based, but with consensus. |
| **T2 Gold (NEW)** | **Code4rena Audit Reports** | `AuditReportScraper` (NEW — Markdown scrape from code4rena.com) | **HIGH** — needs a new `audit_report_scraper` connector that ingests C4 Markdown reports. 1,361 high-risk findings, 499 confirmed high-severity. Crosswalk: finding severity + description → 10 classes. **Includes EVMbench subset** (120 vulns from 40 audits). | W3 (connector + parser) | **Friend-suggested Tier 2 gold** — human expert auditors under financial incentive. |
| **T2 Gold (NEW)** | **DeFi Hacks REKT Database** | `RektScraper` (NEW — Markdown scrape from REKT database) | **MEDIUM** — needs a new `rekt_scraper` connector. 3,216 hacked incidents + 181 curated high-impact. Crosswalk: incident type → 10 classes. **T0 tier assignment** (real exploits, per D-3.3). | W3 | **Friend-suggested Tier 2 gold** — verified real exploits. |
| **T3 Structural** | SmartBugs Curated | `GitConnector` | **LOW** — DASP categories → 10 classes is direct; line-level annotations are an extra metadata field. | W2 | 143 contracts; classic benchmark |
| **T3 Structural** | SC-Bench | `GitConnector` | **MEDIUM** — `dataset/audited/` and `dataset/err-inj/` are two distinct subsets with different semantics; need separate sub-parsers. | W2 | Useful for mutation studies |
| **T3 Structural (NEW)** | **DeFiVulnLabs** | `GitConnector` | **MEDIUM** — 48 vulnerability types, Foundry-style exploits. Same org as DeFiHackLabs. | W3 | **Friend-suggested Tier 3**. DeFi-focused. |
| **T4 Volume** | SmartBugs Wild | `GitConnector` | **MEDIUM** — 47K contracts, no labels; needs Slither run to derive T3 labels. **Friend confirms this is a "97% FP" trap**; the existing Tier 3 conservative design is correct. | W2 (connector), W3 (parser) | Raw verified contracts; volume for pretraining/contrastive |
| **T4 Volume** | slither-audited-smart-contracts (HF) | `HFConnector` | **LOW** — HF row schema is structured; detector JSON → class is direct. 467K contracts is the largest single source. **Crosswalk rule: confidence > 0.8 + canonical detector** (per AUDIT_PATCHES 3-P9). | W2 | Tool-derived labels; down-weight by source tier in loss |
| **T4 Volume** | Messi-Q Smart-Contract-Dataset | `GitConnector` | **MEDIUM** — multiple subfolders with different schemas; needs per-folder sub-parser. | W2 | Older Solidity era; treat as backup |
| **T4 Bronze (NEW)** | **DISL** | `EtherscanConnector` (NEW — or download mirror) | **LOW** — 514,506 unique Solidity files, **unlabeled**. Used for NonVulnerable class + pretraining. **No crosswalk needed** (no labels). | W3 | **Friend-suggested Tier 4 bronze**. Unlabeled volume. |
| **T4 Bronze (NEW)** | **ReentrancyStudy-Data** | `GitConnector` (or Etherscan) | **MEDIUM** — 230,548 Etherscan contracts, reentrancy-labeled (single class, tool-based). Crosswalk: per-tool label → Reentrancy class only. | W3 | **Friend-suggested Tier 4 bronze**. Single-class scale. |
| **T4 Volume** | Zenodo 16910242 (CLEAR dataset) | `ZenodoConnector` | **LOW** — 88.3 MB zip; `contracts.zip` with `labels.csv`; per-contract `is_vulnerable_*` columns. | W2 | Yizhou Chen's CLEAR ICSE 2025 dataset; high quality |
| **DEFERRED** | BCCC-SCsVul-2024 | `ManualConnector` (already on disk at `~/projects/sentinel/BCCC-SCsVul-2024/`) | **N/A in v2 build** — kept as T4 fallback candidate for the *next* iteration once the v2 corpus is built. The Phase 5 verified labels (`contracts_clean_v1.4.csv`) are preserved at `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/contracts_clean_v1.4.csv` for reference but are NOT consumed by the v2 pipeline. | DEFERRED | Per your decision: "we will put this data source at the end, when we have perfect enough data we will decide about it" |

**New connectors required** (per friend suggestions, beyond the 5 in the original proposal):
- `AuditReportConnector` (FORGE) — ingests real audit PDFs/Markdown
- `AuditReportScraper` (Code4rena) — scrapes code4rena.com Markdown
- `RektScraper` (DeFi Hacks) — scrapes REKT database Markdown
- `EtherscanConnector` (DISL/ReentrancyStudy) — fetches verified source by address

These 4 new connectors join the 5 in the original proposal (Git, HF, Zenodo, Etherscan, Manual). The Etherscan connector was a stub in the original; it's now active for DISL/ReentrancyStudy.

**Source schema-versioning policy:** every connector pins to a specific commit / version / record ID. The `manifest.yaml` records the pin + SHA-256 of the downloaded artifact. Re-running ingestion re-validates the SHA-256; if the pin is bumped, `freshness.py` emits a warning and a manual re-approval is required.

**Friend's critical warnings** (carried into the design):
- **97% FP on SmartBugs Wild as labeled data** → Tier 4 conservative design (T3 confidence + consensus where possible) is correct, no change
- **76.78% detection for 3-tool ensemble (Conkas + Slither + Smartcheck)** → confirms tool_validator is corroborative, not authoritative
- **51.97% Slither precision on reentrancy** → Slither-Audited crosswalk has the "confidence > 0.8 + canonical detector" rule, drops low-precision hits

**v1.1 → v1.2 changelog (2026-06-09, post-friend-review):**
- 5 new connectors from 2026-06-08 design (`AuditReportConnector` for FORGE, `AuditReportScraper` for Code4rena, `RektScraper` for DeFi Hacks, `EtherscanConnector` for DISL/ReentrancyStudy) → **Code4rena scraper REMOVED (use Bastet instead); ReentrancyStudy DROPPED entirely; FORGE conditional on 50-entry agreement test; DISL remains**
- Original 17 active sources → **5 critical-path + 12 additive (v2.1) + 1 critical-path negative (DISL)**
- New §6.3 (class-distribution rules) + §6.5 (Go/No-Go gate) added

---

## 7. Schema decisions (deferred to post-Build)

Per your decision, all schema changes (`FEATURE_SCHEMA_VERSION` bump, class taxonomy changes, feature dimension changes, edge type changes) are **frozen until Run 11 baseline lands**. The build plan above moves the existing `v8`/`v9` schema code into the new module without changing the schema. The post-Run-11 schema review will use `sentinel_data/analysis/feature_dist.py` to identify which features are uninformative (the Phase 2 interpretability finding was "complexity is the only thing the model learned") and propose targeted schema changes for v2.1.

This is a deliberate discipline: the v2 corpus is being built against the *current* schema to establish a clean baseline, and the schema is then refined *based on what the new model learns*. Building both corpus and schema at the same time would reproduce the BCCC failure mode (changing two things at once and not knowing which change caused the result).

---

## 8. What `sentinel-ml` looks like after the split

After the build, the `ml/` module's data-related surface area shrinks to:

```
ml/src/datasets/
  sentinel_dataset.py        ← NEW, ~150 lines, replaces dual_path_dataset.py
  collate.py                  ← NEW, ~50 lines, extracted from dual_path_collate_fn
  (dual_path_dataset.py → DELETED in W7)

ml/src/inference/
  preprocess.py               ← UNCHANGED signature, but its call to extract_contract_graph() now imports from sentinel_data.representation
  cache.py                    ← UNCHANGED
  drift_detector.py           ← UNCHANGED
  api.py, predictor.py        ← UNCHANGED

ml/scripts/
  train.py                    ← UNCHANGED signature, but --cache-path / --label-csv flags are replaced by --dataset-version sentinel-v2-gold-2026-08
  tune_threshold.py           ← UNCHANGED
  calibrate_temperature.py    ← UNCHANGED
  promote_model.py            ← UNCHANGED
  (reextract_graphs.py, retokenize_windowed.py, build_multilabel_index.py, create_splits.py, create_cache.py, validate_graph_dataset.py, archive_v8_data.py → DELETED in W7, all moved to sentinel-data)

ml/pyproject.toml             ← ADDS: sentinel-data = "^0.1.0" (path or PyPI dep); REMOVES: slither-analyzer, solc-select, py-solc-ast, solc (all now data-only deps)

ml/data/                      ← FROZEN at its current state; the active data moves to Data/data/exports/sentinel-v2-gold-2026-08/; old paths kept as symlinks for 1 release, then removed
```

**Slither transitive dependency (NEW 2026-06-09, friend review):** `ml/src/inference/preprocess.py` calls `sentinel_data.representation.extract_single(source_text)` for single-contract inference (per Stage 7's seam swap). The new `sentinel_data.representation` uses Slither to extract the graph. **The inference path therefore depends on `slither-analyzer` transitively, via `sentinel-data`.** The two valid options are:

| Option | Mechanism | Trade-off |
|---|---|---|
| **(A) `sentinel-data` declares `slither-analyzer` as a runtime dep** | `pip install sentinel-data` pulls in slither-analyzer automatically | ✅ Works out of the box. (B) becomes redundant. The inference path has no extra install step. |
| **(B) `sentinel-ml`'s inference Docker image inherits from the `sentinel-data` Docker image** | Layer caching; slither comes from the parent image | ✅ Cleanest Docker topology. ❌ Tighter coupling — if the data Docker image is updated, the inference image is implicitly updated. |

**Default: option (A).** `sentinel-data`'s `pyproject.toml` keeps `slither-analyzer` as a runtime dep (it was always going to be a data-side dep for graph extraction). `sentinel-ml`'s `pyproject.toml` only needs `sentinel-data = "^0.1.0"` — no need to list slither-analyzer, solc-select, etc. directly. The transitive resolution gives the inference path everything it needs.

**Verification:** Stage 7 includes a test that runs `pip install sentinel-data` in a clean `sentinel-ml`-style venv and asserts `pip show slither-analyzer` returns a version. If slither is not transitively available, the test fails loud before any inference path is exercised.

**Net effect:** `sentinel-ml` loses ~5,000 lines of data-pipeline code and ~5 GB of data artifacts; gains a single `sentinel-data` dependency. The `ml/` README's "Data Pipeline" section becomes a single line: "all data produced by `sentinel-data`; see Data/README.md for the pipeline."

---

## 9. Verification gates — when do we say "v2 is ready"?

The v2 build is "ready to train Run 11" when **all** of the following are green:

1. **Schema regression test:** every existing graph .pt on disk from Run 9 is byte-identical after re-extraction through the new module (proves the move didn't change extraction logic).
2. **Phase 5 regression test:** running the new `verification/` stage on the legacy BCCC corpus produces a `verification_report.md` that matches `Data/docs/legacy/bccc_deep_dive/.../p5_s6_verification_report.md` to within ±0.5% per class.
3. **End-to-end round-trip:** `sentinel-data run --config config.v2-baseline.yaml` from scratch produces a catalog entry; loading it through `SentinelDataset` and running a 1-batch forward pass in `sentinel-ml` succeeds with the same loss as the v1 loader.
4. **Feature distribution report:** `sentinel_data.analysis.feature_dist.complexity_proxy_risk.md` is GREEN (no class-pair has > 1.5σ difference in node-count / edge-count / cyclomatic complexity).
5. **All 10 classes pass verification gate:** every class is either `VERIFIED` or `PROVISIONAL` in the v2 verification report; no `FAIL` classes.
6. **No leakage across splits:** `sentinel_data.splitting.leakage_auditor` reports 0 near-duplicates across split boundaries.

When all 6 pass, the v2-baseline catalog entry is registered as `sentinel-v2-gold-2026-08` and Run 11 launches.

---

## 10. Open items after this proposal

| # | Item | Owner | Blocking? |
|---|---|---|---|
| 1 | Choose the path of the `sentinel-data` distribution (path dep, PyPI, or git tag) for `sentinel-ml`'s `pyproject.toml` | Ali | W1 |
| 2 | Confirm Data/docker/Dockerfile.data base image (recommend `python:3.12-slim` + `solc-select install` + `pip install slither-analyzer`) | Ali | W1 |
| 3 | **RESOLVED 2026-06-09 (friend review):** DVC remote is **local-only for v2 build**; S3/GCS deferred to v2.1 (post-Run-11). Single-developer WSL2 doesn't need a remote. | ✅ Done | (no longer blocks) |
| 4 | Confirm the taxonomy YAML's 10 classes are exactly the v1 taxonomy (no changes) | Ali | W4 |
| 5 | Pre-commit the canonical `taxonomy.yaml` + 5 critical-path crosswalk YAMLs once they're drafted (revised: 5 not 11 per §6.1) | Ali | W4 |
| 6 | Decide the export shard size default (recommend 5,000 contracts/shard) | Ali | W7 |
| 7 | Approve Run 11 launch date (recommend 2026-08-18, after friend's 5 new sources expand Stage 3 to 3 weeks) | Ali | W10 |
| 8 | **NEW (friend review):** Decide whether to apply CallToUnknown → ExternalBug merge if <300 verified (per §6.3.2). Decision is **human-checked, not auto-merge**; reviewed at Stage 3 gate. | Ali (Stage 3 gate) | W4–5 |
| 9 | **NEW (friend review):** Decide whether to defer FORGE based on 50-entry agreement test (per §6.5). Test is part of Stage 3. | Ali (Stage 3 gate) | W4–5 |
| 10 | **NEW (friend review):** Decide whether to defer Run 11 to Run 12 if minimum viable corpus gate (§6.5) fails. | Ali (Stage 3 gate) | W4–5 |

---

**End of proposal. Awaiting approval to proceed with Week 1 (skeleton + Data/ restructure).**
