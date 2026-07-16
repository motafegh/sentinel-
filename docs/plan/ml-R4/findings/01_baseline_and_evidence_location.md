# Phase 0 — Baseline Freeze and Previous-Evidence Location Findings

- **Run ID:** R4-P0-20260716-001
- **Repository commit:** 4b5bd333c63ab7a7ec83810fbbae54f3ebf1b493
- **Input artifact IDs:** R4-P0-CHK-001 through R4-P0-EVD-007 (see manifests)
- **Date:** 2026-07-16
- **Status:** COMPLETE

## Objective

Freeze the exact current DATA/ML baseline, hash all protected artifacts, locate all prior contract-level evidence without reviewing contracts, and produce an availability inventory with separated population counts. No label correction, no export regeneration, no architecture work.

## Reused previous evidence

No previous evidence was *consumed* in Phase 0 — it was *located and registered*. The evidence location inventory (24 evidence sets) was built by searching repository history and local storage. Key prior evidence sets identified:

- **DIVE** (2 review mds + 2 tool-corroboration JSONs + 17,287 Slither + 573 Aderyn + 22,073 per-contract labels + 22,330-row source CSV)
- **BCCC** (full 5-phase deep dive: integrity, validation, analysis, label-validation, verification — 28 scripts, verified labels v1.4, class definitions, review batches)
- **SolidiFI** (injection framework + 7-tool benchmark results + 283 per-contract labels)
- **SmartBugs Curated** (143 hand-labeled + recall test report)
- **Manual** (83 hand-written contracts + 83 paired labels + 6 BCCC-injected + 22 AI audit reports)
- **Benchmark** (66-contract 5-tier OOD v0.1)

## Method

1. Read all governing files (START_HERE_AGENT, MASTER_PLAN, policies, phase spec, templates, workstreams, registers).
2. Resolved repository identity via `git status`, `git rev-parse`, `git worktree list`, `git submodule status`, `dvc status`.
3. Resolved active DATA bundle by reading `data_module/config.yaml`, export `manifest.json`, split `split_manifest.json`, and parquet schemas via the ML venv's pyarrow.
4. Resolved active ML bundle by reading `ml/mlops_config.json`, checkpoint `state.json`, `thresholds.json`, `locked_files.sha256`, DVC pointer files, and the checkpoints README.
5. Located prior evidence via systematic file search (Task agent + direct exploration) across 14 evidence categories.
6. Hashed all protected artifacts with `sha256sum`.
7. Created deterministic manifest files and a validation script.
8. Verified no protected artifact was modified during Phase 0 (hashes match on-disk state).

## Populations

All counts are kept separate per the Phase 0 spec. No distinct populations are combined into one corpus number.

| Boundary | Count | Source |
|---|---|---|
| DIVE source records (DIVE_Labels.csv rows) | 22,330 | data_module/data/raw_staging/dive_labels/DIVE_Labels.csv |
| DIVE __source__ .sol files | 22,330 | data_module/data/raw_staging/dive/__source__/ |
| DIVE per-contract generated labels | 22,073 | data_module/data/labels/dive/ |
| SolidiFI per-contract generated labels | 283 | data_module/data/labels/solidifi/ |
| SmartBugs Curated per-contract generated labels | 137 | data_module/data/labels/smartbugs_curated/ |
| Merged-label contracts | 22,493 | data_module/data/labels/merged/ |
| Export label rows (labels.parquet) | 22,493 | export manifest n_contracts |
| Export metadata rows (metadata.parquet) | 22,493 | parquet read |
| Export contracts with representations | 21,657 | export manifest n_contracts_with_reps |
| Export contracts without representations | 836 | 22493 - 21657 |
| Split train rows | 18,596 | split_manifest.json |
| Split val rows | 1,983 | split_manifest.json |
| Split test rows | 1,914 | split_manifest.json |
| Split total | 22,493 | 18596 + 1983 + 1914 |
| Dedup groups resolved | 3,036 | split_manifest.json |
| DIVE representations | 63,741 | data_module/data/representations/dive/ (includes per-class symlinks) |
| SolidiFI representations | 828 | data_module/data/representations/solidifi/ |
| SmartBugs Curated representations | 402 | data_module/data/representations/smartbugs_curated/ |
| DeFiHackLabs representations | 42 | data_module/data/representations/defihacklabs/ (disabled source) |
| ML processed index rows (legacy) | 41,577 | ml/data/processed/multilabel_index.csv (v2-era, not active export) |
| ML loaded classes | 10 | thresholds.json (CallToUnknown through UnusedReturn) |
| GasException support in split | 0 | No GasException positives in train/val/test class distributions |

### Per-class positive counts in active split (from split_manifest.json)

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| ExternalBug | 13,720 | 1,491 | 1,427 | 16,638 |
| Reentrancy | 9,644 | 899 | 856 | 11,399 |
| IntegerUO | 7,786 | 851 | 815 | 9,452 |
| Timestamp | 5,065 | 641 | 618 | 6,324 |
| UnusedReturn | 4,757 | 549 | 553 | 5,859 |
| DenialOfService | 845 | 129 | 127 | 1,101 |
| TransactionOrderDependence | 478 | 86 | 83 | 647 |
| CallToUnknown | 66 | 10 | 11 | 87 |
| MishandledException | 27 | 5 | 7 | 39 |
| GasException | 0 | 0 | 0 | 0 |

## Findings

### F0.1 — Active DATA bundle is explicit and bound by evidence

- **Export:** `sentinel-v3-smartbugs-2026-06-13/` with artifact_hash `5cc5cfcb...`, 22,493 contracts, 5 shards, graph_schema v9.
- **Split:** v3, seed=42, stratified 70/15/15, 3,036 dedup groups resolved (leakage prevention reassignments logged in split_manifest.json).
- **Config:** `data_module/config.yaml` hash `543e37cc...` — sources enabled: solidifi, dive, smartbugs_curated, web3bugs (declared but absent), disl (NonVulnerable pool only).
- **Parquet schemas:** labels.parquet has `contract_id, source, split, class_0..class_9, confidence_tier`; metadata.parquet has `contract_id, source, split, solc_version, version_bucket, loc, n_functions, n_pos, primary_class, node_count, edge_count, has_unchecked_block, dedup_group_id, confidence_tier`.

### F0.2 — Active ML bundle is explicit and bound by evidence

- **Checkpoint:** `GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` (281MB, SHA-256 `6a220c6b...`, DVC-tracked, state: epoch=51, best_f1=0.6801).
- **Thresholds:** per-class F1-tuned thresholds (architecture `four_eye_v8`, 10 classes, overall F1_macro=0.6823).
- **MLOps config:** `ml/mlops_config.json` binds checkpoint + thresholds + drift_baseline + experiment name.
- **Calibration:** **explicit absence** — no calibration sidecar is referenced in mlops_config.json. Temperature scaling artifacts exist in `ml/calibration/` but are NOT loaded by the inference API.
- **Schema:** v9 (NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14, NUM_EDGE_TYPES=12).
- **MLflow:** sqlite:///mlruns.db, experiment `sentinel-retrain-v2`.

### F0.3 — Prior evidence is extensive and largely AVAILABLE

24 evidence sets identified across DIVE, BCCC, SolidiFI, SmartBugs, manual, tool, benchmark, and audit categories. 16 are AVAILABLE_VERIFIED or AVAILABLE_UNVERIFIED; 8 are UNAVAILABLE.

### F0.4 — Web3Bugs is entirely UNAVAILABLE

Declared `enabled: true` in config.yaml as Tier-1 Gold, but no crosswalk, parser, connector, data directory, or source acquisition exists anywhere in the repository. This is a config-vs-reality contradiction: the active export cannot contain Web3Bugs contracts despite the config declaring it enabled.

### F0.5 — GasException has zero support in the active split

No GasException positives exist in train, val, or test. The threshold is 0.05 with F1=0.0, precision=0.0, recall=0.0, support=0, predicted_positives=1831. This class is effectively unsupported in the current model.

### F0.6 — Stale locked_files.sha256

`ml/locked_files.sha256` records hashes from the v4-sprint era. 4 of 5 referenced source files (sentinel_model.py, gnn_encoder.py, graph_extractor.py, graph_schema.py) have changed since; 1 file (val_indices.npy) was moved to docs/.bin/ during the 2026-06-15 DVC cleanup. The locked file is NOT a protected R4 artifact — it is a stale legacy lock. Current on-disk hashes (recorded in protected_artifacts.json) are the R4 baseline.

### F0.7 — BCCC config path mismatch

`config.yaml` declares `legacy_outputs: "Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/"` but the actual repo path is `data_module/docs/legacy/...`. The v1.4 verified labels file exists at the data_module path. The config path string is stale but does not affect the active export (BCCC is DEFERRED, not enabled).

### F0.8 — 836 contracts without representations

Export manifest reports 22,493 contracts but only 21,657 with representations (n_contracts_with_reps). 836 contracts have labels but no graph/token representations and therefore cannot be loaded by the ML training pipeline.

## Contradictions

1. **Web3Bugs enabled-but-absent:** config.yaml declares `web3bugs: enabled: true` but no crosswalk/parser/data/connector exists. The active export was built from solidifi + dive + smartbugs_curated only (confirmed by source_distributions in split_manifest.json: dive, solidifi, smartbugs_curated — no web3bugs).
2. **BCCC config path:** `legacy_outputs` path in config.yaml uses `Data/docs/...` prefix but actual path is `data_module/docs/...`.
3. **locked_files.sha256:** 4/5 source file hashes are stale (files modified after locks were recorded).

## Missing evidence

| Item | Status | Impact |
|---|---|---|
| Web3Bugs data/crosswalk/parser | UNAVAILABLE | Declared Tier-1 Gold but never acquired; cannot appear in DATA vNext without new acquisition |
| DIVE non-EB/RE class reviews | UNAVAILABLE | Arithmetic, DoS, UncheckedReturn, TimeManip, BadRandom, FrontRunning have no per-class review |
| DIVE second independent reviewer | UNAVAILABLE | Both DIVE review mds are single-author |
| DIVE /tmp sample lists | UNAVAILABLE | Temp files not committed; membership recoverable from md tables + documented seeds |
| BCCC 2-tool consensus run | UNAVAILABLE | consensus.py exists but patterns/ and results/ are empty |
| BCCC 2-tool audit memory file | UNAVAILABLE | Lives outside repo in Claude memory directory |
| Echidna tool outputs | UNAVAILABLE | No echidna cache or results anywhere |
| Exploit reproduction PoCs | UNAVAILABLE | No Foundry/Hardhat vulnerability PoC tests |

## Limitations

- Export graph/token shard files (5 shards each, ~1.6GB total) are recorded as AVAILABLE_UNVERIFIED — individual file SHA-256 hashes were not computed for all shards due to size; the shard_index.json hash is recorded and the export manifest artifact_hash covers the full export.
- The ML processed index (`ml/data/processed/multilabel_index.csv`, 41,577 rows) is a legacy v2-era artifact, not the active export. It is recorded but not protected.
- `cached_dataset_v9.pkl` is a legacy paired cache from the v2 export era, not the active export. It is recorded but not protected.
- DVC remote (`/mnt/d/sentinel-dvc-remote`) is a local Windows D: drive mount; DVC push/pull was not tested in Phase 0 (read-only scope).

## Outputs and hashes

| Output | Path | SHA-256 |
|---|---|---|
| Baseline manifest | manifests/baseline_manifest.json | computed at write |
| Protected artifacts manifest | manifests/protected_artifacts.json | computed at write |
| Availability inventory | manifests/availability_inventory.csv | computed at write |
| Evidence location inventory | manifests/evidence_location_inventory.csv | computed at write |
| Validation script | scripts/p0_baseline_freeze.py | computed at write |
| Phase 0 findings | findings/01_baseline_and_evidence_location.md | this file |

## Gate assessment

**G0 PASS** — all G0 pass criteria are met:

| Criterion | Status | Evidence |
|---|---|---|
| Actual local baseline is explicit | PASS | baseline_manifest.json records branch/commit/dirty/worktrees/DVC/config |
| Active DATA/ML bundle is bound by evidence | PASS | Export hash, split hashes, checkpoint hash, threshold hash, MLOps config hash all recorded and verified |
| Prior evidence locations are inventoried | PASS | evidence_location_inventory.csv with 24 evidence sets |
| Missing artifacts are explicit | PASS | 8 UNAVAILABLE items listed with impact |
| Population counts are separated | PASS | 17 distinct population boundaries recorded separately |
| Protected artifacts are recorded | PASS | protected_artifacts.json with 24 protected entries + SHA-256 hashes |
| Manifests validate | PASS | JSON manifests are valid JSON; CSV inventories are well-formed |
| No protected artifact changed | PASS | All hashes computed at Phase 0 start; no modifications made during Phase 0 |

## Next permitted action

Begin Phase 1 — Previous Evidence Recovery, starting with the DIVE workstream (`docs/plan/ml-R4/workstreams/DIVE_PREVIOUS_EVIDENCE_RECOVERY.md`), then BCCC, then other sources. Phase 1 recovers and structures the evidence located in Phase 0 into structured evidence items with artifact identity, method, reviewer, date, conclusion, limitations, and contract-class scope. No new contract review is permitted without an approved EVIDENCE_GAP_REGISTER entry.
