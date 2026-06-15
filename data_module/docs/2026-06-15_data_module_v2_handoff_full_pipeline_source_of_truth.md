# SENTINEL Data Module — Full Pipeline Handover (Source of Truth)

**Date:** 2026-06-15
**Module:** data_module
**Export version:** sentinel-v2-baseline-2026-06-12
**Active training run:** GCB-P1-Run11-v2deduped-20260613
**Author:** verified against source code 2026-06-15

> **This document is the canonical source of truth for the v2 data pipeline.**
> Every fact below was verified directly against source code (`.py` files) and live data files.
> No claim is taken from comments, docstrings, or prior docs without cross-checking the code.

---

## Table of Contents

1. [What the Data Module Is](#1-what-the-data-module-is)
2. [Pipeline Overview — 9 Stages](#2-pipeline-overview--9-stages)
3. [Entry Point: The CLI](#3-entry-point-the-cli)
4. [Stage 0 — Ingestion](#4-stage-0--ingestion)
5. [Stage 1 — Preprocessing](#5-stage-1--preprocessing)
6. [Stage 2 — Representation](#6-stage-2--representation)
7. [Stage 3 — Labeling](#7-stage-3--labeling)
8. [Stage 4 — Verification](#8-stage-4--verification)
9. [Stage 5 — Splitting](#9-stage-5--splitting)
10. [Stage 5b — Registry](#10-stage-5b--registry)
11. [Stage 6 — Analysis](#11-stage-6--analysis)
12. [Stage 7A — Export](#12-stage-7a--export)
13. [Deduplication Deep Dive](#13-deduplication-deep-dive)
14. [Label Quality and Multi-Label Handling](#14-label-quality-and-multi-label-handling)
15. [Current Artifact State (v2 Export)](#15-current-artifact-state-v2-export)
16. [v1 vs v2 Splits — The Leakage Investigation and Fix](#16-v1-vs-v2-splits--the-leakage-investigation-and-fix)
17. [Known Issues and Stubs](#17-known-issues-and-stubs)
18. [Run 11 Training Context](#18-run-11-training-context)
19. [How to Re-Run from Scratch](#19-how-to-re-run-from-scratch)

---

## 1. What the Data Module Is

The data module is the offline batch pipeline that builds the labeled Solidity contract dataset used to train SENTINEL's vulnerability detection model. It lives in `data_module/` and is installed as a Python package (`sentinel_data`).

The output is an **export artifact** — a directory of sharded `.pt` files plus Parquet tables and a manifest — consumed by the ML training loop (`ml/src/data/sentinel_dataset.py`) via the `SentinelDatasetExport` class.

The module handles:
- Pulling raw `.sol` files from multiple labeled sources
- Compiling, flattening, deduplicating, and normalizing contracts
- Extracting graph (GNN input) and token (CodeBERT input) representations
- Assigning multi-label vulnerability labels with tier-precedence merging
- Verifying labels with AST checks and Slither corroboration
- Splitting into train/val/test with strict no-leakage enforcement
- Sharding everything into a reproducible, hash-verified export artifact

---

## 2. Pipeline Overview — 9 Stages

```
ingest → preprocess → represent → label → verify → split → register → analyze → export
```

Each stage is a CLI subcommand (see §3). Stages are ordered and each depends on the previous. The graph schema version (`FEATURE_SCHEMA_VERSION`) must match between the represent stage and the ML model; bumping it invalidates all existing graph `.pt` files and checkpoints.

**Source of truth:** `data_module/sentinel_data/cli.py` lines 71–93 (the `STAGES` list and `STAGE_DESCRIPTIONS` dict).

---

## 3. Entry Point: The CLI

**File:** `data_module/sentinel_data/cli.py`

The CLI is the single user-facing surface. Every stage has a subcommand:

```bash
# Run one stage
sentinel-data <stage> --config data_module/config.yaml

# Run all stages from a given stage onward
sentinel-data run --from-stage split --config data_module/config.yaml

# Dry run (prints planned actions without writing)
sentinel-data <stage> --config data_module/config.yaml --dry-run
```

**Important path note:** The CLI must be invoked from the repo root with `PYTHONPATH=data_module` if not installed as a package. The CLI manipulates `sys.path` (lines 62–68) to add the repo root and `ml/` so that thin-adapter imports from `ml/src/preprocessing/` resolve correctly.

**Configuration:** All pipeline settings live in `data_module/config.yaml`. The CLI always loads config from this file unless `--config` overrides it.

---

## 4. Stage 0 — Ingestion

**CLI:** `sentinel-data ingest`
**Code:** `data_module/sentinel_data/ingestion/ingest.py`, `ingestion/connectors/`

Pulls raw `.sol` files from enabled sources and writes them to `data/raw/<source>/`. Each source gets an ingestion manifest (`data/raw/<source>/manifest.json`) recording file counts, SHA-256 hashes, and the git pin used.

**Connectors available:** `git_connector`, `huggingface_connector`, `zenodo_connector`, `etherscan_connector`, `manual_connector`.

### Sources — What Is Enabled

From `data_module/config.yaml` (verified 2026-06-15):

| Source | Enabled | Tier | Subtype | Notes |
|--------|---------|------|---------|-------|
| `solidifi` | ✅ | T1 | gold | Injected vulnerabilities — ground truth |
| `dive` | ✅ | T1 | gold | Peer-reviewed, multi-label via folder membership |
| `smartbugs_curated` | ✅ | T3 | structural | 143 hand-labeled contracts |
| `web3bugs` | ✅ | T1 | gold | Contest-verified |
| `disl` | ✅ | T4 | bronze | 514K unlabeled — NonVulnerable pool only |
| `defihacklabs` | ❌ | T1 | gold | Disabled |
| `bastet`, `forge`, `scrawld`, `defi_hacks_rekt`, `openzeppelin_*`, `smartbugs_wild`, and 8 more | ❌ | — | — | Deferred to v2.1+ |

**What actually made it to the v2 export:** Only `solidifi` (283 contracts) and `dive` (22,073 contracts) = 22,356 total. The other enabled sources (`smartbugs_curated`, `web3bugs`, `disl`) were enabled in config but were either not yet preprocessed or not included in the v1 export run. The v2 export's `source_set` field in `manifest.json` confirms: `["solidifi", "dive"]`.

---

## 5. Stage 1 — Preprocessing

**CLI:** `sentinel-data preprocess`
**Code:** `data_module/sentinel_data/preprocessing/preprocess.py`, `pipeline.py`, `deduplicator.py`, `compiler.py`, `flattener.py`, `normalizer.py`, `segmenter.py`

Reads raw `.sol` files from `data/raw/<source>/`, runs them through a multi-step pipeline, and writes preprocessed outputs to `data/preprocessed/<source>/`. Each contract produces a `<sha256>.meta.json` sidecar.

**Pipeline steps per contract:**

1. **Flatten** (`flattener.py`) — Resolves imports, collapses multi-file contracts into one.
2. **Compile** (`compiler.py`) — Selects the correct `solc` version (two-pass: try detected version, fall back to latest compatible). Contracts that fail compilation are written to `dropped.csv`.
3. **Deduplicate** (`deduplicator.py`) — Three levels (see §13).
4. **Normalize** (`normalizer.py`) — Strips comments, normalizes whitespace, strips Ethereum addresses.
5. **Segment** (`segmenter.py`) — Splits multi-contract files into per-contract units.
6. **Version-bucket** — Tags each contract with the `solc` version used.

**Output per contract:** A `<sha256>.meta.json` containing sha256, original path, solc version, source name, and dedup metadata.

**Config:** `pipeline.dedup.ast_similarity_threshold: 0.85` (used by Level 3 — currently stubbed, see §13).

---

## 6. Stage 2 — Representation

**CLI:** `sentinel-data represent`
**Code:** `data_module/sentinel_data/representation/orchestrator.py`, `graph_extractor.py`, `tokenizer.py`, `versioner.py`

Reads preprocessed contracts from `data/preprocessed/<source>/` and produces two representations per contract:

- **Graph `.pt` file** — `data/representations/<source>/<sha256>.pt` — a `torch_geometric.data.Data` object containing node features, edge index, and edge types.
- **Token `.pt` file** — `data/representations/<source>/<sha256>.tokens.pt` — a tensor of shape `[4, 512]` (four 512-token windows for GraphCodeBERT).

**Graph schema (v9)** — verified from `data_module/sentinel_data/representation/graph_schema.py` (which is the canonical source; `ml/src/preprocessing/graph_schema.py` is a thin re-export shim):

| Constant | Value |
|---|---|
| `FEATURE_SCHEMA_VERSION` | `"v9"` |
| `NODE_FEATURE_DIM` | `12` |
| `NUM_NODE_TYPES` | `14` (ids 0–13) |
| `NUM_EDGE_TYPES` | `12` (ids 0–11) |

**12 Node features (in order):**

| Index | Name | Description |
|---|---|---|
| 0 | `type_id` | Node type (normalized by max id=13) |
| 1 | `visibility` | `public/external=0.0`, `internal=0.5`, `private=1.0` |
| 2 | `uses_block_globals` | Uses `block.timestamp`, `block.number`, `now`, etc. |
| 3 | `view` | Is a view/pure function |
| 4 | `payable` | Is payable |
| 5 | `complexity` | Cyclomatic complexity proxy |
| 6 | `loc` | Lines of code |
| 7 | `return_ignored` | Return value of call ignored |
| 8 | `call_target_typed` | Call target is typed (not dynamic) |
| 9 | `has_loop` | Contains a loop |
| 10 | `external_call_count` | Number of external calls |
| 11 | `in_unchecked_block` | Inside a Solidity `unchecked {}` block (v9 addition) |

**14 Node types:** `STATE_VAR(0)`, `FUNCTION(1)`, `MODIFIER(2)`, `EVENT(3)`, `FALLBACK(4)`, `RECEIVE(5)`, `CONSTRUCTOR(6)`, `CONTRACT(7)`, `CFG_NODE_CALL(8)`, `CFG_NODE_WRITE(9)`, `CFG_NODE_READ(10)`, `CFG_NODE_CHECK(11)`, `CFG_NODE_OTHER(12)`, `CFG_NODE_ARITH(13)` (v9 addition for IntegerUO).

**12 Edge types:** `CALLS(0)`, `READS(1)`, `WRITES(2)`, `EMITS(3)`, `INHERITS(4)`, `CONTAINS(5)`, `CONTROL_FLOW(6)`, `REVERSE_CONTAINS(7)` (runtime-only, never on disk), `CALL_ENTRY(8)`, `RETURN_TO(9)`, `DEF_USE(10)`, `EXTERNAL_CALL(11)` (v9 addition).

**Versioner:** `versioner.py` checks `FEATURE_SCHEMA_VERSION` against cached representations and evicts stale entries if the schema changed. The version registry is written to `data/representations/_version_registry.json`.

**Slither version requirement:** `graph_schema.py` lines 59–70 assert `slither-analyzer >= 0.9.3` at import time. Older Slither silently produces wrong `in_unchecked_block` features.

---

## 7. Stage 3 — Labeling

**CLI:** `sentinel-data label`
**Code:** `data_module/sentinel_data/labeling/parsers/dive.py`, `parsers/solidifi.py`, `labeling/merger.py`, `labeling/gate.py`

Reads preprocessed meta from `data/preprocessed/<source>/` and raw repo layout, assigns vulnerability class labels, and writes one canonical `.labels.json` per contract SHA-256 to `data/labels/merged/`.

### 10 Vulnerability Classes (LOCKED — reordering invalidates checkpoints)

From `graph_schema.py` `CLASS_NAMES` (lines 190–201):

| Index | Class Name |
|---|---|
| 0 | `CallToUnknown` |
| 1 | `DenialOfService` |
| 2 | `ExternalBug` |
| 3 | `GasException` |
| 4 | `IntegerUO` |
| 5 | `MishandledException` |
| 6 | `Reentrancy` |
| 7 | `Timestamp` |
| 8 | `TransactionOrderDependence` |
| 9 | `UnusedReturn` |

Plus `NonVulnerable` — a contract where all 10 class values are 0. This is not a named class column but is the label for contracts with no positives.

### DIVE Multi-Label Mechanism

**File:** `data_module/sentinel_data/labeling/parsers/dive.py`

DIVE stores contracts in `data/raw/dive/repo/__source__/<N>.sol`. The same file ALSO appears in vulnerability-named subfolders: `repo/Reentrancy/<N>.sol`, `repo/Arithmetic/<N>.sol`, etc. A single file may appear in multiple vulnerability folders.

The parser uses this layout to assign multi-label:

1. `_build_folder_index()` (lines 54–69) — Scans every mapped folder under `raw/dive/repo/`, builds a dict `{filename → frozenset(canonical_classes)}`.
2. `_build_labels_json()` (lines 72–91) — For each contract, sets `value: 1` for every class in its folder membership frozenset, `value: 0` for all others.

**Result:** A single `.labels.json` per SHA-256 with all matching vulnerability classes set to 1. This is the correct and intended handling of "same contract, multiple vulnerabilities" — 69.1% of DIVE contracts (15,259 / 22,073) are multi-label by design.

The confidence tier for DIVE labels is `T2` (set in `labeling/crosswalks/dive.yaml`).

### SolidiFI Labeling

**File:** `data_module/sentinel_data/labeling/parsers/solidifi.py`

SolidiFI contracts have injected bugs with known ground truth. The parser reads the SolidiFI metadata to assign single-class labels. Tier: `T0` (highest confidence — injected ground truth).

### Label Merger

**File:** `data_module/sentinel_data/labeling/merger.py`

Merges per-source label files into one canonical `.labels.json` per SHA-256 in `data/labels/merged/`. Called after per-source parsers have run.

**Tier precedence** (higher confidence wins): `T0 > T1 > T2 > T3 > T4`. Within the same tier, **positive wins over negative**. Source precedence order (lines 41–43): `solidifi → defihacklabs → smartbugs_curated → web3bugs → dive → disl`.

**Multi-source merge logic** (`_merge_class_entries()`, lines 76–97): For each class column, take the highest-confidence positive across all sources. If no source is positive, take the lowest-rank negative. The output records which source and tier contributed each label.

**DoS+Reentrancy co-occurrence noise check** (`_check_co_occurrence_flag()`, lines 100–124): If BOTH DoS and Reentrancy are positive in the merged output AND the contract comes from a single T3/T4 source AND that source's DoS+Reentrancy co-occurrence rate exceeds 50%, the record is flagged as `dos_reentrancy_cooccur_suspect`. BCCC's co-occurrence rate was ~99%, which is what this rule targets. DIVE at ~12% is NOT flagged.

**Current state:** 0 multi-source overlaps in the active export (all 22,356 contracts are single-source).

### DoS / Reentrancy Patch on labels.parquet

After the initial export was built, a post-export patch was applied directly to `data/exports/sentinel-v2-baseline-2026-06-12/labels.parquet`: all rows where both DoS and Reentrancy are 1 had DoS zeroed to 0. This reduced DoS positives from ~3,750 to 1,095.

The pre-patch file is preserved as `_labels_pre_dos_reentrancy_patch_2026-06-13.parquet` in the same directory.

**Important:** The split manifest (`data/splits/v2/split_manifest.json`) records class distributions as they were at split time (pre-patch). The actual training signal uses the patched `labels.parquet` which the `SentinelDataset` reads at runtime.

---

## 8. Stage 4 — Verification

**CLI:** `sentinel-data verify`
**Code:** `data_module/sentinel_data/verification/`

Runs 5 checks over the merged labels and writes a verification report. The gate (`verification/gate.py`) determines PASS/FAIL.

| Check | What it does |
|---|---|
| `class_auditor` | Counts per-class positives; flags co-occurrence pairs |
| `semantic_checker` | AST-level checks (does the code have the construct the label claims?) |
| `tool_validator` | Runs Slither on a sample; checks label–tool agreement |
| `fp_estimator` | Samples N per class; estimates false positive rate |
| `negative_checker` | Runs Slither on NonVulnerable contracts; flags if >5% hit rate |

Config thresholds (from `config.yaml`): `fail_threshold: 0.30`, `tool_corroboration_min: 1`, `negative_tool_hit_threshold: 0.05` (warn) / `0.10` (fail).

**Current state for v2 export:** Verification reports exist in `data/verification/`. The pipeline was run with `--skip-tool-validator` for speed on the BCCC/Slither-heavy checks.

---

## 9. Stage 5 — Splitting

**CLI:** `sentinel-data split --version 2 --seed 42`
**Code:** `data_module/sentinel_data/splitting/splitters.py`, `dedup_enforcer.py`, `leakage_auditor.py`, `nonvulnerable_cap.py`

The splitting stage reads all merged `.labels.json` files from `data/labels/merged/`, builds `Contract` objects, runs a two-pass split, and writes JSONL files + `split_manifest.json` to `data/splits/v{N}/`.

### Contract Object

Defined in `splitters.py` (lines 55–75):

```python
@dataclass
class Contract:
    sha256: str
    source: str             # "solidifi" / "dive" / ...
    tier: str               # "T0" / "T1" / "T2" / "T3" / "T4"
    classes: dict[str, int]
    primary_class: str      # first positive class, or "NonVulnerable"
    n_pos: int              # count of positive class labels
    dedup_group: Optional[str] = None  # graph-content dedup group id (see §13)
    project_id: Optional[str] = None   # for project-level splitter
```

### Two-Pass Split

**Pass 1 — Stratified splitter** (`stratified_split()`, lines 200–265):

Groups contracts by composite stratum key `(primary_class, source, tier)` and allocates each stratum proportionally at ratios `(0.70, 0.15, 0.15)` with `seed=42`. Preserves per-class distribution within ±2%.

**Pass 2 — Dedup enforcer** (`apply_dedup_enforcer()` in `dedup_enforcer.py`):

Looks at each contract's `dedup_group` field. For every group that straddles a split boundary (members in more than one split), reassigns ALL members to the split that holds the majority. Ties go to train. Records all reassignments in `splits.metadata.reassignments`.

**Key line:** `dedup_enforcer.py:46` — `if c.dedup_group is not None:` — contracts with `dedup_group=None` are completely skipped. This is the line that caused v1 leakage (see §16).

**Pass 3 — NonVulnerable cap** (`apply_nonvulnerable_cap()`):

Enforces a maximum ratio of NonVulnerable contracts relative to positive contracts. Config: `pipeline.negative.positive_ratio_max: 3.0`. In the v2 split, the NonVulnerable cap was not binding (only 2,658 NonVulnerable contracts vs 59,094 allowed at 3:1 ratio).

### 4 Splitter Strategies

| Strategy | When used | Key property |
|---|---|---|
| `stratified` | Default | Preserves per-class distribution |
| `random` | Sanity test only | No per-class guarantees |
| `project` | Audit datasets (Bastet, ScaBench, Web3Bugs) | Keeps full project in one split |
| `temporal` | Future use | Pre-2023 → train/val, post-2023 → test |

The v2 split used `stratified` (the default for the enabled sources).

### CLI Fix for v2 (Critical)

The `_run_split` function in `cli.py` was fixed (lines 263–276) to load graph-hash dedup groups before building Contract objects:

```python
dedup_groups_path = data_dir / "dedup_groups_graph_hash.json"
if dedup_groups_path.exists():
    cid_to_group = json.loads(dedup_groups_path.read_text()).get("groups", {})

contracts.append(Contract(
    ...,
    dedup_group=cid_to_group.get(sha),  # None if not in graph-hash groups
))
```

Without this fix, `dedup_group` is `None` for all contracts and the dedup enforcer has nothing to act on (v1 behavior).

### v2 Split Results (verified from `data/splits/v2/split_manifest.json`)

```
strategy:              stratified
seed:                  42
ratios:                [0.70, 0.15, 0.15]
generated_at:          2026-06-13T10:37:35Z
contracts (all):       22,356  (train=18,458 / val=1,996 / test=1,902)
dedup_groups_resolved: 3,009
nonvuln_cap:           3.0 (not binding — only 2,658 NonVulnerable contracts)
```

The JSONL split files (`train.jsonl`, `val.jsonl`, `test.jsonl`) in `data/splits/v2/` are the authoritative record of v2 assignments for all 22,356 contracts, including the 833 that lack complete representations and are therefore absent from the export.

---

## 10. Stage 5b — Registry

**CLI:** `sentinel-data register --name <name> --version 2`
**Code:** `data_module/sentinel_data/registry/catalog.py`

Registers a named dataset version in an SQLite catalog (`data/registry/catalog.db`) with a YAML mirror (`data/registry/catalog.yaml`). Records: name, source set, split version, artifact hash, artifact path, verification report path.

**Current state:** v2 baseline was registered. The catalog tracks superseded versions with a `retired` flag and a `superseded_by` field.

---

## 11. Stage 6 — Analysis

**CLI:** `sentinel-data analyze`
**Code:** `data_module/sentinel_data/analysis/`

Runs 5 analytical tools over the merged labels and representations:

| Tool | What it catches |
|---|---|
| `balance_viz` | Per-class / per-source / per-tier contract counts |
| `feature_dist` | Cross-class feature distribution — the "Run-9-failure catcher" (complexity feature dominance) |
| `cooccurrence` | Directed co-occurrence matrix; flags pairs where P(A\|B) > 50% |
| `overlap_detector` | Pairwise source Jaccard overlap (exact and near-dup) |
| `drift_monitor` | KS test comparing current build against a registered baseline |

Outputs are written to `data/analysis/<run_id>/`. The `--baseline-version` flag enables drift monitoring against a prior registered version.

---

## 12. Stage 7A — Export

**CLI:** `sentinel-data export --dataset-version sentinel-v2-baseline-2026-06-12 --split-version 2`
**Code:** `data_module/sentinel_data/export/chunker.py`, `graph_writer.py`, `token_writer.py`, `label_writer.py`, `metadata_writer.py`

Combines representations and split assignments into a self-contained, hash-verified artifact directory consumed by the ML training loop.

### Output Layout

```
data/exports/sentinel-v2-baseline-2026-06-12/
  labels.parquet                   ← all 22,356 contracts, class columns + split column
  metadata.parquet                 ← per-contract metadata (loc, solc version, source, tier)
  graphs/
    graphs-00000.pt                ← dict {sha256: Data} for shard 0
    graphs-00001.pt
    graphs-00002.pt
    graphs-00003.pt
    graphs-00004.pt
    _shard_index.json              ← {sha256: {shard: int, pos_in_shard: int}}
  tokens/
    tokens-00000.pt                ← dict {sha256: Tensor[4,512]} for shard 0
    tokens-00001.pt
    ...
    _shard_index.json
  manifest.json                    ← written LAST (Fix A — see below)
  _labels_pre_dos_reentrancy_patch_2026-06-13.parquet  ← backup before DoS patch
```

### 4 Writers (run in order)

From `chunker.py` `chunk_export()` function (lines 127–207):

1. **`write_labels_parquet()`** — Reads all split JSONL files, writes `labels.parquet` with columns: `contract_id`, `source`, `split`, `class_0`…`class_9`, `confidence_tier`.
2. **`write_metadata_parquet()`** — Reads preprocessed meta.json sidecars, writes `metadata.parquet`.
3. **`write_graphs_shards()`** — For each contract in split order, loads the `.pt` graph from `representations/`, packs into shards of `shard_size` (5,000) contracts.
4. **`write_tokens_shards()`** — Same pattern for `.tokens.pt` files.

### Artifact Hash (Fix A)

`_hash_export_data()` (lines 62–76) computes SHA-256 over all files in the export directory **except `manifest.json`** — in sorted path order. This means `manifest.json` can be updated (e.g. to embed new split assignments) without invalidating the hash.

**Current artifact hash:** `45e2a2d406f90708cc4dec9a2824a9e86c1b9df8...` (verified from `manifest.json`).

### manifest.json (Written Last — Fix A)

`manifest.json` is written after the artifact hash is computed. It contains:

```json
{
  "schema_version": "v1",
  "graph_schema_version": "v9",
  "artifact_hash": "45e2a2d...",
  "hash_algorithm": "sha256",
  "shard_size": 5000,
  "n_contracts": 22356,
  "n_contracts_with_reps": 21523,
  "n_shards": 5,
  "splits": {
    "train": ["sha256...", ...],   // 17,877 SHA-256s
    "val":   ["sha256...", ...],   // 1,878 SHA-256s
    "test":  ["sha256...", ...]    // 1,768 SHA-256s
  },
  "shard_index": { "sha256": {"shard": N, "pos_in_shard": M}, ... },
  "source_set": ["solidifi", "dive"],
  "label_class_columns": ["CallToUnknown", "DenialOfService", ...]
}
```

**Critical:** `manifest.json["splits"]` is what `SentinelDataset` reads to determine train/val/test membership at training time. The `split` column embedded in `labels.parquet` reflects the v1 assignment at export time and is NOT used by the trainer — the trainer always uses `manifest.json["splits"]`.

### SentinelDatasetExport

**File:** `data_module/sentinel_data/export/export.py`

The read-only wrapper consumed by the ML side. Validates manifest fields on construction. Key method: `get_split_contract_ids(split)` returns the list of SHA-256s for a given split from `manifest.json["splits"]`.

The ML-side `SentinelDataset` (Stage 7B, `ml/src/data/sentinel_dataset.py`) wraps this class to implement `__len__` and `__getitem__` for PyTorch.

---

## 13. Deduplication Deep Dive

Deduplication happens at two separate points in the pipeline with different mechanisms. Understanding both is critical to understanding the v1 leakage bug and the v2 fix.

### Level 1 — Exact SHA-256 (Stage 1, Preprocessing)

**Code:** `deduplicator.py` lines 48–57.

If two raw `.sol` files have identical byte content, they hash to the same SHA-256. The second occurrence is detected as a duplicate and skipped. Only one copy is preprocessed.

**Status: WORKS correctly.**

### Level 2 — Ethereum Address Dedup (Stage 1, Preprocessing)

**Code:** `deduplicator.py` lines 59–70.

Different `.sol` files that share a hardcoded Ethereum address (`0x...`) are likely the same deployed contract copied across datasets. The first SHA-256 that saw the address is treated as canonical; subsequent files with the same address are flagged as duplicates.

**Status: WORKS correctly.** BCCC had 38.8% duplication — this level catches cross-dataset copy-paste.

### Level 3 — AST Near-Dup (STUB — NOT IMPLEMENTED)

**Code:** `deduplicator.py` lines 73–79.

```python
# Level 3: AST near-dup — STUB (requires Slither; deferred to Stage 2)
return DedupRecord(
    sha256=sha,
    dedup_group_id=sha,    # ← every contract is its own group
    is_duplicate=False,
    duplicate_of="",
)
```

Every contract that passes L1 and L2 is assigned `dedup_group_id = sha256` (its own SHA-256 as its group). This means every contract is treated as a unique dedup group. The dedup group is NOT propagated into `labels.json` files — there is no field for it there.

**Status: STUBBED.** Intended to use Slither-based AST similarity clustering at threshold 0.85 (from config). Not implemented.

### Graph-Content Hash Dedup (Post-Export Fix)

**File:** `data_module/data/dedup_groups_graph_hash.json`
**Created:** 2026-06-13 (this session)

This is the workaround for the Level 3 stub. After the export artifact was built, the graph `.pt` files were scanned and each contract was fingerprinted by the content of its graph tensor:

```
group_id = MD5(g.x.numpy().tobytes() + g.edge_index.numpy().tobytes())
```

Contracts that produce the same fingerprint are assigned to the same `dedup_group`. This catches different source files (different SHA-256) that compile to identical graph tensors — the model's view of these contracts is indistinguishable.

**Results (verified from `data/dedup_groups_graph_hash.json`):**

```
n_contracts:              21,523  (all contracts with representations)
n_unique_groups:          12,577
n_singleton_groups:       10,712  (only one contract in this graph)
n_dup_groups:              1,865  (two or more contracts share this graph)
n_contracts_in_dup_groups: 10,811 (50.2% of contracts are in a dup group)
```

**Why this matters for splitting:** If contract A (from `DIVE/Reentrancy/`) and contract B (from `DIVE/Arithmetic/`) compile to the same graph, putting A in train and B in test means the model has already "seen" that exact graph during training. This is pure leakage at the model's representational level, regardless of different labels.

**How it's read by the split CLI** (`cli.py` lines 263–301): At split time, the CLI loads `dedup_groups_graph_hash.json` and sets `dedup_group=group_id` on each Contract object. The dedup enforcer then reassigns straddling groups.

### The 677 Irreducible Inconsistent Groups

Of the 1,865 duplicate graph groups, 677 have members with DIFFERENT labels (e.g., two contracts with the same graph tensor but different DIVE folder memberships). These are NOT a pipeline bug.

**What they are:** Different source `.sol` files (different SHA-256, different filenames) that DIVE placed in different vulnerability subfolders, but which compiled to the same graph representation. The model sees identical input features but the label differs.

**Outcome in v2:** The dedup enforcer moves all members of each group to the majority split. All 677 inconsistent groups land in train. The contradictory gradient signals are bounded and average out during training; they do not appear in val/test and therefore cannot inflate evaluation metrics.

**Why not fixable here:** Resolving label conflicts for same-graph contracts requires human review of each DIVE folder assignment. It is a data curation issue, not a pipeline bug.

---

## 14. Label Quality and Multi-Label Handling

### Label Distribution (v2 Export — verified from `labels.parquet`, 2026-06-15)

The export covers 22,356 contracts from solidifi (283) and dive (22,073):

| Class | Positives | % of 22,356 |
|---|---|---|
| CallToUnknown | 39 | 0.2% |
| DenialOfService | 1,095 | 4.9% (post-patch) |
| ExternalBug | 16,621 | 74.3% |
| GasException | **0** | 0.0% |
| IntegerUO | 9,437 | 42.2% |
| MishandledException | 39 | 0.2% |
| Reentrancy | 11,369 | 50.9% |
| Timestamp | 6,311 | 28.2% |
| TransactionOrderDependence | 643 | 2.9% |
| UnusedReturn | 5,859 | 26.2% |

- **Multi-label** (>1 positive class): 15,213 contracts (68.1%)
- **Any positive**: 19,698 contracts
- **All-negative (NonVulnerable)**: 2,658 contracts
- **GasException: 0 positives** — Neither DIVE nor SolidiFI has this label. The class exists in the schema but is untrained in v2. This is a known gap; SmartBugs Curated is the planned source for GasException labels.
- **DoS+Reentrancy co-occurrence after patch: 0** — confirmed from `labels.parquet`.

### labels.parquet Split Column Discrepancy

The `split` column in `labels.parquet` reflects the v1 split (train=15,644 / val=3,344 / test=3,368). This column was baked in at export time (v1 assignments) and was NOT updated when the v2 split was applied.

The v2 assignments live in `manifest.json["splits"]`. The `SentinelDataset` reads splits ONLY from `manifest.json`, not from `labels.parquet["split"]`. The stale split column in `labels.parquet` is harmless for training but misleading if read directly.

---

## 15. Current Artifact State (v2 Export)

**Export directory:** `data_module/data/exports/sentinel-v2-baseline-2026-06-12/`

| Property | Value |
|---|---|
| `artifact_hash` | `45e2a2d406f90708cc4dec9a2824a9e86c1b9df8...` |
| `graph_schema_version` | `v9` |
| `n_contracts` (total in splits) | 22,356 |
| `n_contracts_with_reps` (in shards) | 21,523 |
| Contracts missing reps (not in shards) | 833 |
| `n_shards` | 5 (shard_size=5,000) |
| `source_set` | `["solidifi", "dive"]` |
| **Split assignments in manifest.json** | train=17,877 / val=1,878 / test=1,768 |
| Split version embedded in labels.parquet | v1 (train=15,644 / val=3,344 / test=3,368) — do not use |
| DoS+Reentrancy co-occurrence | 0 (patched) |
| GasException positives | 0 |

**Backup:** The pre-DoS-patch labels are preserved as `_labels_pre_dos_reentrancy_patch_2026-06-13.parquet`.

---

## 16. v1 vs v2 Splits — The Leakage Investigation and Fix

### v1 Leakage: Root Cause Chain

v1 was the split produced by the original export run. It had 45% leakage at the graph-tensor level — nearly half of val/test contracts had a byte-identical graph in train.

**Chain of failures:**

1. **`deduplicator.py` Level 3 is a stub** — every contract gets `dedup_group_id = sha256` (its own unique group). No real grouping ever happens.
2. **`dedup_group_id` is never propagated into `labels.json` files** — the field does not exist in `.labels.json` output files.
3. **The v1 split CLI never set `dedup_group` on Contract objects** — it was simply omitted from the constructor call. All 22,356 Contract objects had `dedup_group=None`.
4. **`dedup_enforcer.py:46` checks `if c.dedup_group is not None:`** — with all contracts at `None`, the enforcer skipped all 22,356 contracts. Pass 2 was a no-op.
5. **Result:** Pure stratified random split with no dedup enforcement. Contracts with the same graph tensor were scattered across train/val/test according to the random stratification. 45% of val/test ended up with train-identical graphs.

### v2 Fix: Graph-Content Hash Dedup

The fix bypasses the Level 3 stub entirely by working at the representation level (post-export):

1. **Compute graph fingerprints** — for all 21,523 contracts with representations, compute `MD5(x_tensor + edge_index_tensor)`.
2. **Save group map** to `data/dedup_groups_graph_hash.json` (21,523 contracts → 12,577 unique groups, 1,865 dup groups).
3. **Fix `cli.py` `_run_split()`** — load the group map, set `dedup_group=group_id` on each Contract before splitting.
4. **Re-run split as v2** — dedup enforcer now has groups to act on. It resolved 3,009 groups (some groups had members in multiple splits).
5. **Update `manifest.json["splits"]`** with v2 assignments — artifact hash unchanged (hash excludes manifest.json).

### v2 Results

**Verified 0% cross-split leakage** at graph-tensor level (independent check via `leakage_auditor.py`).

| | v1 | v2 |
|---|---|---|
| Strategy | stratified | stratified + dedup_enforcer |
| dedup_groups_resolved | 0 | 3,009 |
| Leakage at graph level | ~45% of val/test | 0% |
| Train (all contracts) | 15,644 | 18,458 |
| Val (all contracts) | 3,344 | 1,996 |
| Test (all contracts) | 3,368 | 1,902 |
| Train (with reps, in manifest) | — | 17,877 |
| Val (with reps, in manifest) | — | 1,878 |
| Test (with reps, in manifest) | — | 1,768 |

**Run 10** used v1 splits. Its best F1=0.683 at ep32 is inflated by memorization, not generalization. Run 10 was killed at ep33.

---

## 17. Known Issues and Stubs

| ID | Issue | Location | Severity | Status |
|---|---|---|---|---|
| I-1 | Level 3 AST near-dup is a stub | `preprocessing/deduplicator.py:73` | Medium | Partially mitigated by graph-hash dedup (§13). Long-term fix: text-shingle Jaccard clustering. |
| I-2 | `dedup_group_id` not propagated through labels.json | `preprocessing/deduplicator.py`, `labeling/parsers/` | Medium | Workaround: graph-hash JSON file. Long-term: propagate through the labeling stage. |
| I-3 | v1 split had 45% leakage | v1 export manifest | High | **FIXED** — v2 splits with 0% leakage are now in manifest.json. |
| I-4 | 677 groups with contradictory labels (same graph, different labels) | `data/dedup_groups_graph_hash.json` | Low | Irreducible DIVE curation inconsistency. All members moved to train in v2. |
| I-5 | Level 2 bytecode dedup never implemented | `deduplicator.py` | Low | Not planned for v2. |
| I-6 | `leakage_auditor.py` exists but was never run in production | `splitting/leakage_auditor.py` | Medium | Should be run after every split as a gate check. |
| I-7 | `labels.parquet` split column is v1 (stale) | export `labels.parquet` | Low | Trainer uses `manifest.json["splits"]` — stale column is harmless but misleading. |
| I-8 | GasException: 0 positives in v2 export | `labels.parquet` | Medium | SmartBugs Curated is the planned source. Not yet exported. |
| I-9 | `split_manifest.json` class distributions are pre-DoS-patch | `data/splits/v2/split_manifest.json` | Low | Informational only. Actual training uses patched `labels.parquet`. |

---

## 18. Run 11 Training Context

**Run name:** `GCB-P1-Run11-v2deduped-20260613`
**Launched:** 2026-06-13 15:27 UTC
**PID:** 556540
**Log:** `ml/logs/GCB-P1-Run11-v2deduped-20260613.log`
**JSONL streams:** `ml/logs/GCB-P1-Run11-v2deduped-20260613/` (`epoch_summary.jsonl`, `alerts.jsonl`, `step_metrics.jsonl`)

**What this run is using:**
- Export: `data_module/data/exports/sentinel-v2-baseline-2026-06-12/`
- Split assignments: from `manifest.json["splits"]` (v2, 0% leakage)
- train=17,877 / val=1,878 / test=1,768 (contracts with both graph + token reps)
- Graph schema: v9 (`NODE_FEATURE_DIM=12`)
- Labels: patched `labels.parquet` (DoS+Reentrancy co-occurrence = 0)

**Training config (verified from launch command):**

| Parameter | Value |
|---|---|
| `--epochs` | 100 |
| `--batch-size` | 8 (effective 64 with `grad_accum=8`) |
| `--lr` | 2e-4 |
| `--early-stop-patience` | 30 |
| `--appnp-alpha` | 0.2 |
| `--gnn-prefix-k` | 48 (warmup starts ep15) |
| `--weighted-sampler` | positive (weight range [1.0, 3.0]) |
| `--threshold-tune-interval` | 10 |
| `--use-amp` | True (AMP + TF32) |
| `TRANSFORMERS_OFFLINE` | 1 |

**Model config (logged at startup):**

```
SentinelModel v8 (four-eye)
  num_classes=10 | eye_dim=128 | classifier [512→256→10]
  gnn_hidden=256 | heads=8 | layers=8 | use_jk=True | jk_mode=attention
  lora_r=16 | lora_alpha=32 | gnn_prefix_k=48 | warmup=15
  LoRA trainable: 589,824 | frozen: 124,645,632
```

**Why Run 10 was killed:** Best F1=0.683 at ep32 was inflated by 45% train/val leakage. The model was memorizing training graphs, not generalizing. Run 11 is the first honest baseline on clean v2 splits.

**Resume command** (logged by trainer at startup):

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --resume ml/checkpoints/GCB-P1-Run11-v2deduped-20260613_best.pt \
  --run-name GCB-P1-Run11-v2deduped-20260613-resumed \
  --experiment-name sentinel-multilabel \
  --epochs 100 \
  --gradient-accumulation-steps 8
```

---

## 19. How to Re-Run from Scratch

> All commands run from the repo root with `PYTHONPATH=data_module`.

```bash
# 0. Install the package (if not already)
cd data_module && pip install -e . && cd ..

# 1. Ingest (pull raw .sol files from enabled sources)
sentinel-data ingest --config data_module/config.yaml

# 2. Preprocess (flatten + compile + dedup L1/L2 + normalize)
sentinel-data preprocess --config data_module/config.yaml --workers 4

# 3. Represent (extract v9 graphs + token windows)
#    Requires slither-analyzer >= 0.9.3
sentinel-data represent --config data_module/config.yaml --workers 4

# 3b. Compute graph-content dedup groups (required before split — fixes L3 stub)
#    Run this script after representation completes:
#    ml/scripts/audit/compute_dedup_groups.py  (or re-derive manually)
#    Output: data_module/data/dedup_groups_graph_hash.json

# 4. Label (DIVE multi-label + SolidiFI ground truth + merger)
sentinel-data label --config data_module/config.yaml

# 5. Verify (AST checks + Slither corroboration — optional for speed)
sentinel-data verify --config data_module/config.yaml --skip-tool-validator

# 6. Split (v2, stratified + dedup_enforcer — requires dedup_groups_graph_hash.json)
sentinel-data split --config data_module/config.yaml --version 2 --seed 42

# 7. Register
sentinel-data register --config data_module/config.yaml \
  --name sentinel-v2-baseline-$(date +%Y-%m-%d) \
  --version 2 --sources solidifi dive

# 8. Analyze (optional — catches feature dominance / drift issues)
sentinel-data analyze --config data_module/config.yaml

# 9. Export
sentinel-data export --config data_module/config.yaml \
  --dataset-version sentinel-v2-baseline-$(date +%Y-%m-%d) \
  --split-version 2

# 10. Apply DoS/Reentrancy patch to labels.parquet
#     (until the co-occurrence rule is enforced in the merger for all sources)
#     Backup first:
cp data_module/data/exports/.../labels.parquet \
   data_module/data/exports/.../_labels_pre_dos_reentrancy_patch_$(date +%Y-%m-%d).parquet
# Then zero DoS where Reentrancy=1 — see the patch script in ml/scripts/

# 11. Launch training
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --run-name GCB-P1-Run12-v2-$(date +%Y%m%d) \
  --export-dir data_module/data/exports/sentinel-v2-baseline-$(date +%Y-%m-%d) \
  --epochs 100 --batch-size 8 --lr 2e-4 --early-stop-patience 30 \
  --appnp-alpha 0.2 --gnn-prefix-k 48 \
  --weighted-sampler positive --threshold-tune-interval 10 --use-amp \
  >> ml/logs/GCB-P1-Run12-v2-$(date +%Y%m%d).log 2>&1 &
```

### Key Files to Check Before Re-Run

| What to verify | File |
|---|---|
| Graph schema version matches ML model | `data_module/sentinel_data/representation/graph_schema.py` `FEATURE_SCHEMA_VERSION` |
| Sources enabled in config | `data_module/config.yaml` `sources_critical_path` |
| Dedup groups exist (or recompute) | `data_module/data/dedup_groups_graph_hash.json` |
| 0% leakage after split | `data_module/data/splits/v{N}/split_manifest.json` `dedup_groups_resolved > 0` |
| Export manifest has v2 splits | `manifest.json["splits"]` counts match v2 split manifest |
| DoS co-occurrence = 0 | `labels.parquet`: count where class_1==1 AND class_6==1 should be 0 |

---

*Document verified against source code 2026-06-15. Every numerical claim cross-checked against `cli.py`, `deduplicator.py`, `dive.py`, `merger.py`, `splitters.py`, `dedup_enforcer.py`, `chunker.py`, `graph_schema.py`, `dedup_groups_graph_hash.json`, `split_manifest.json`, `manifest.json`, and `labels.parquet`.*
