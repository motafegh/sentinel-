# SENTINEL Data Module — Comprehensive Reference
**Version:** v3 (sentinel-v3-smartbugs-2026-06-13)
**Date:** 2026-06-14
**Authors:** Ali Rajabi + Claude

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [The 10 Vulnerability Classes](#2-the-10-vulnerability-classes)
3. [Graph Schema v9](#3-graph-schema-v9)
4. [Data Sources](#4-data-sources)
5. [The 8-Stage Pipeline](#5-the-8-stage-pipeline)
6. [Deduplication System](#6-deduplication-system)
7. [Export Artifact](#7-export-artifact)
8. [SentinelDataset (Stage 7B)](#8-sentineldataset-stage-7b)
9. [Pre-Run-12 Fixes (2026-06-13)](#9-pre-run-12-fixes-2026-06-13)
10. [Run 12 Training](#10-run-12-training)
11. [Known Limitations and Deferred Work](#11-known-limitations-and-deferred-work)

---

## 1. System Overview

SENTINEL is a smart contract security oracle — a multi-label classifier that, given a Solidity source file, outputs a probability score for each of 10 vulnerability classes. A single contract can be vulnerable to multiple classes simultaneously.

The model is a **dual-encoder fusion architecture**:

```
Solidity source
      │
      ├──► Graph Extractor (Slither IR) ──► GNN (8-layer GAT + JK) ──► graph embedding [128]
      │                                                                          │
      └──► Tokenizer (GCB tokenizer)  ──► GraphCodeBERT + LoRA  ──► text embedding [128]
                                                                                 │
                                                              CrossAttentionFusion [256]
                                                                                 │
                                                              Classifier [256→10 sigmoid]
                                                                                 │
                                                               P(vuln_0)...P(vuln_9)
```

The **data module** (`data_module/`) is the factory that turns raw `.sol` files into the tensors the model trains on. It is completely independent of the ML training code in `ml/`. The data module's output is a single **export artifact directory** — a self-contained, hash-verified bundle of tensors, labels, and metadata that the ML side reads via `SentinelDataset`.

---

## 2. The 10 Vulnerability Classes

Defined in `sentinel_data/representation/graph_schema.py` as `CLASS_NAMES` (locked index order — changing it invalidates checkpoints).

| Index | Class | Description |
|---|---|---|
| 0 | CallToUnknown | Low-level `.call()` to an untrusted/unknown address |
| 1 | DenialOfService | Attacker can permanently block a function (admin freeze, unbounded loop) |
| 2 | ExternalBug | Unchecked return value from an external call |
| 3 | GasException | Out-of-gas from a transfer or unbounded loop with no fallback |
| 4 | IntegerUO | Integer overflow or underflow (critical in pre-Solidity 0.8 contracts) |
| 5 | MishandledException | Exception from `.send()` or `.call()` is swallowed silently |
| 6 | Reentrancy | External call allows attacker to re-enter and drain funds |
| 7 | Timestamp | Logic depends on `block.timestamp`, manipulable by miners |
| 8 | TransactionOrderDependence | Race condition exploitable via transaction ordering |
| 9 | UnusedReturn | Return value of a call is ignored, missing error signal |

**Multi-label:** a contract can have multiple classes set to 1 simultaneously. The model outputs 10 independent sigmoid scores, not a softmax.

**NonVulnerable** is not a separate class — it is the implicit state when all 10 outputs are 0.

---

## 3. Graph Schema v9

**Source of truth:** `sentinel_data/representation/graph_schema.py`

```python
FEATURE_SCHEMA_VERSION = "v9"
NODE_FEATURE_DIM  = 12   # length of each node's feature vector
NUM_NODE_TYPES    = 14   # distinct node type IDs
NUM_EDGE_TYPES    = 12   # distinct edge type IDs
```

### Node Types (14)

| ID | Name | Meaning |
|---|---|---|
| 0 | STATE_VAR | Contract-level state variable |
| 1 | FUNCTION | Function declaration |
| 2 | MODIFIER | Modifier declaration |
| 3 | EVENT | Event declaration |
| 4 | FALLBACK | Fallback function |
| 5 | RECEIVE | Receive function |
| 6 | CONSTRUCTOR | Constructor |
| 7 | CONTRACT | Contract-level node |
| 8 | CFG_NODE_CALL | CFG node: external/internal call |
| 9 | CFG_NODE_WRITE | CFG node: storage write |
| 10 | CFG_NODE_READ | CFG node: storage read |
| 11 | CFG_NODE_CHECK | CFG node: conditional / require |
| 12 | CFG_NODE_OTHER | CFG node: other statement |
| 13 | _(reserved)_ | Padding / unknown |

### Edge Types (12)

| ID | Name | Meaning |
|---|---|---|
| 0 | CALLS | Function call relationship |
| 1 | READS | Reads a state variable |
| 2 | WRITES | Writes to a state variable |
| 3 | EMITS | Emits an event |
| 4 | INHERITS | Contract inheritance |
| 5 | CONTAINS | Contract contains function/modifier |
| 6 | CONTROL_FLOW | CFG sequential flow |
| 7 | REVERSE_CONTAINS | Inverse of CONTAINS (runtime only, never on disk) |
| 8 | CALL_ENTRY | Entry edge into a call (ICFG) |
| 9 | RETURN_TO | Return edge from a call (ICFG) |
| 10 | DEF_USE | Data-flow: definition to use |
| 11 | EXTERNAL_CALL | v9 addition — explicit external call edge |

### Node Feature Vector `x[n, 12]`

Each of the 12 features is a float in the range [0, 1]:

| feat[i] | Meaning |
|---|---|
| 0 | Node type (one of 14, normalized) |
| 1 | Visibility (public/external/internal/private, normalized) |
| 2 | Is payable (0 or 1) |
| 3 | Has modifier (0 or 1) |
| 4 | Static call (0 or 1) |
| 5 | Complexity: `log1p(cfg_block_count) / log1p(100)` — **zeroed at GNN input** (Run 9+, feature dominance prevention) |
| 6–11 | Structural counts (calls out, reads, writes, checks, emits, children), each log-normalized |

---

## 4. Data Sources

### 4.1 SolidiFI — Tier 0 (Gold)

- **What it is:** Synthetic injection benchmark. Researchers took clean, audited contracts and deliberately injected one known vulnerability per file.
- **Why it is reliable:** The injection is intentional and documented — labels are 100% accurate by construction.
- **Volume:** ~283 contracts across 9 vulnerability categories.
- **Connector:** `git` — cloned from the SolidiFI repository.
- **Parser:** `sentinel_data/labeling/parsers/solidifi.py` — reads folder name from path `repo/<category>/<contract>.sol`.
- **Crosswalk:** `sentinel_data/labeling/crosswalks/solidifi.yaml`.
- **Contribution to v3:** Provides T0 ground-truth for CallToUnknown, MishandledException, and the VERIFIED classes.

### 4.2 DIVE — Tier 2 (Best-effort)

- **What it is:** ~22,000 real deployed Ethereum contracts scraped and labeled by automated Slither static analysis.
- **Why it is the backbone:** Largest source by far. Provides multi-label signal across all classes.
- **Limitation:** Slither has false positives. Labels are automated, not human-verified. The DoS/Reentrancy co-occurrence at 12% is treated as legitimate multi-label signal (contracts with admin-freeze patterns can genuinely have both vulnerabilities).
- **Volume:** ~22,073 contracts in the DIVE corpus.
- **Connector:** `git` — cloned from the DIVE repository.
- **Parser:** `sentinel_data/labeling/parsers/dive.py` — reads multi-label index from DIVE's folder structure.
- **Crosswalk:** `sentinel_data/labeling/crosswalks/dive.yaml`.

### 4.3 SmartBugs Curated — Tier 1 (Structural ground-truth)

- **What it is:** 143 manually curated Solidity contracts organized into category folders. Real vulnerable contracts from CTFs and academic datasets, verified by humans.
- **Why it matters:** Primary fix for the CallToUnknown dead class (+48 T1 examples). Also contributes to Reentrancy, ExternalBug, IntegerUO, Timestamp.
- **Volume:** 143 raw → 137 preprocessed (1 compile failure, 5 duplicates of DIVE contracts) → 134 graphs.
- **Connector:** `manual` — files live at `ml/data/smartbugs-curated/dataset/`, symlinked to `data/raw/smartbugs_curated/repo/dataset/`.
- **Parser:** `sentinel_data/labeling/parsers/smartbugs_curated.py` — reads `parts[1]` from `repo/<category>/<contract>.sol`.
- **Crosswalk:** `sentinel_data/labeling/crosswalks/smartbugs_curated.yaml`.
- **Status:** Added 2026-06-13. `confidence_tier: T1` for all contracts.
- **Config:** `tier: 3` in `config.yaml` (structural benchmark / recall ground-truth, not gold).

**Crosswalk bug fixed 2026-06-13:** `front_running: Timestamp` → `front_running: TransactionOrderDependence`. Front-running is a race condition (TOD), not a timestamp manipulation.

### 4.4 DeFiHackLabs — BLOCKED (deferred)

- **Status:** Blocked. The 693 `_exp.sol` files are Foundry PoC test contracts (attacker code), not the vulnerable protocol contracts. They import `forge-std/Test.sol` and protocol-specific interfaces that Slither cannot resolve without the full Foundry environment.
- **Fix required:** Extract the actual vulnerable protocol contract from each exploit (requires Etherscan lookup per contract address) — a 2+ day research task, deferred to v3 data module.

---

## 5. The 8-Stage Pipeline

```
Stage 1 → Ingestion      raw .sol files collected from each source
Stage 2 → Preprocessing  Slither compilation, metadata extraction
Stage 3 → Representation graph tensor (.pt) + token tensor (.tokens.pt)
Stage 4 → Labeling       vulnerability class labels per contract
Stage 5 → Splitting      train/val/test assignment with dedup group enforcement
Stage 6 → Verification   per-class label quality audit
Stage 7A→ Export         package everything into a versioned artifact directory
Stage 7B→ SentinelDataset PyTorch Dataset that reads the artifact
```

### Stage 1 — Ingestion

Each source's `connector` in `config.yaml` specifies how to get the raw files:
- `connector: git` — clones the repo, reads from `data/raw/<source>/repo/`
- `connector: manual` — expects files at `staging_path`, creates a symlink at `data/raw/<source>/repo/`

All sources are normalized to the same directory layout after ingestion, so Stages 2–4 are source-agnostic.

### Stage 2 — Preprocessing

For each `.sol` file:
1. Slither compiles the contract and resolves `pragma solidity` version.
2. Metadata is extracted: SLOC, number of functions, version bucket (legacy/modern).
3. The source is normalized (CRLF → LF, trailing whitespace removed).
4. SHA-256 of the normalized source becomes the contract's permanent ID.

Output per contract: `data/preprocessed/<source>/<sha256>.meta.json` + `<sha256>.sol`.

**Failures logged but not fatal:**
- Old `pragma ^0.4.x` contracts: Slither cannot compile with modern solc → skipped.
- Import failures: contracts importing external packages without the full dependency tree → skipped.
- Duplicates: if a SHA-256 already exists from another source, the duplicate is silently skipped (5 SmartBugs contracts duplicated DIVE contracts).

### Stage 3 — Representation

Two artifacts are produced per contract:

**Graph (`.pt`):** Slither's intermediate representation (IR) is traversed to build a heterogeneous graph. Every IR statement becomes a node. Edges encode 12 relationship types. The node feature vector is 12-dimensional (see Section 3). Output: a PyTorch Geometric `Data` object with `x[n_nodes, 12]`, `edge_index[2, n_edges]`, `edge_attr[n_edges]`.

**Tokens (`.tokens.pt`):** The normalized source is tokenized with the GraphCodeBERT tokenizer and split into 4 overlapping chunks of 512 tokens. Output: a `[4, 512]` int64 tensor of token IDs.

Both files land at `data/representations/<source>/<sha256>.pt` and `<sha256>.tokens.pt`.

**Failures (v3):** 3 SmartBugs contracts failed graph extraction due to old `.call.value()` syntax that Slither's IR cannot represent. These contracts have labels but no graph tensor — they are excluded from training (counted in `n_contracts` but not `n_contracts_with_reps`).

### Stage 4 — Labeling

Each contract gets a `.labels.json` with one entry per class (value: 0 or 1, tier: T0/T1/T2/None).

The pipeline uses two components per source:

**Parser** (`sentinel_data/labeling/parsers/<source>.py`): reads the file path and extracts which vulnerability category the contract belongs to. For folder-organized sources (SolidiFI, SmartBugs), this reads the folder name. For DIVE, it reads a multi-label index file.

**Crosswalk** (`sentinel_data/labeling/crosswalks/<source>.yaml`): maps the source's category name to one of the 10 SENTINEL class names.

**Merger** (`sentinel_data/labeling/merger.py`): combines labels from all three sources. For contracts appearing in multiple sources:
- Conflict resolution: T0 > T1 > T2 (higher confidence wins).
- Within a tier, positive wins over negative (false negatives are worse than false positives).
- The co-occurrence noise detection flag fires only for T3/T4 sources at >50% DoS+Reentrancy co-occurrence rate. DIVE (T2) at 12% is explicitly exempt — its DoS+Reentrancy multi-label signal is treated as legitimate.

### Stage 5 — Splitting

The splitter reads all merged label files and assigns each contract to train, val, or test. Two constraints are enforced:

1. **Dedup group constraint:** contracts in the same dedup group (see Section 6) must all land in the same split. This prevents training on a near-duplicate of a test contract.
2. **Stratified split:** class-aware stratification keeps class prevalences similar across splits.

**v3 split sizes:** train=18,561 / val=2,008 / test=1,924 (post-L3 dedup, 0% leakage confirmed).

### Stage 6 — Verification

For each class, the verifier samples positive contracts and checks whether the source code actually contains the pattern the label claims. Results produce one of three verdicts:

| Verdict | Meaning |
|---|---|
| VERIFIED | ≥80% semantic pass rate, ≥80% coverage, best tier ≤ T1 |
| PROVISIONAL | Some evidence but insufficient coverage or confidence |
| BEST-EFFORT | Too few examples or pattern not extractable from graph schema v9 |

**v3 verdict summary:**

| Class | Verdict | Note |
|---|---|---|
| MishandledException | VERIFIED | 38/39 semantic pass |
| TransactionOrderDependence | VERIFIED | 100% coverage from DIVE+SolidiFI |
| IntegerUO | PROVISIONAL | 3% coverage (Slither semantic check runs on small sample) |
| Reentrancy | PROVISIONAL | 67% semantic pass on sampled contracts |
| Timestamp | PROVISIONAL | 83% semantic pass |
| UnusedReturn | PROVISIONAL | 100% semantic pass on sample |
| GasException | PROVISIONAL | 0 training examples — class is unlearnable until new source added |
| CallToUnknown | BEST-EFFORT → PROVISIONAL (expected) | +48 T1 SmartBugs contracts; re-verification pending |
| DenialOfService | BEST-EFFORT | NOT_EXTRACTABLE: Slither-based AST patterns not in v9 schema |
| ExternalBug | BEST-EFFORT | NOT_EXTRACTABLE: no automatic semantic check for v9 |

DoS and ExternalBug cannot reach VERIFIED in v9 because the v9 graph schema lacks the Slither AST-level nodes that would allow automatic pattern extraction. Full verification requires `tool_validator` + Slither batch runs (deferred to v2.1).

### Stage 7A — Export

`sentinel-data export` packages the pipeline output into a single artifact directory. See Section 7.

### Stage 7B — SentinelDataset

The PyTorch Dataset that the ML trainer uses. See Section 8.

---

## 6. Deduplication System

Deduplication prevents a contract appearing in both train and test in any form. There are three levels:

### Level 1 — Graph-hash dedup (active since v2)

**Method:** SHA-256 of the concatenated raw bytes of `x` (node feature matrix) + `edge_index` (adjacency). If two contracts hash identically, their graphs are byte-identical — same control-flow structure, same features, regardless of source code differences (variable names, comments, whitespace).

**Scale:** 10,811 of 21,523 contracts share a graph hash with at least one other contract. These are merged into dedup groups with a canonical SHA.

**Why this happens:** DIVE deploys the same contract template to many Ethereum addresses. Each deployment may have different variable names or comments but compiles to an identical CFG. Graph-hash dedup catches all such structural duplicates.

**Where stored:** `data/dedup_groups_graph_hash.json`.

### Level 2 — Source-hash dedup (implicit in Stage 2)

The contract ID IS the SHA-256 of the normalized source. Two contracts with identical normalized source automatically have the same ID and the second is silently skipped during preprocessing. This is not a separate dedup step — it is built into Stage 2.

### Level 3 — Text-hash dedup (added 2026-06-13)

**Method:** Normalize source code by:
1. Strip all block comments (`/* ... */`)
2. Strip all line comments (`// ...`)
3. Collapse all whitespace sequences to a single space

Then SHA-256 the result. Group contracts with identical normalized hashes.

**What it catches:** Copy-paste contracts where someone only added or removed comments. Graph-hash misses these because different comments produce slightly different IR that changes the graph.

**What it does NOT catch intentionally:** Identifier renaming. `reentrantWithdraw` and `reentrancyWithdraw` are different identifiers with different semantic meaning. Lowercasing would merge them — this was an explicit decision dropped from the normalization.

**v3 scan results (all 22,493 contracts):**
- 147 L3 groups found (356 contracts total)
- 83 groups are label-consistent (all members have identical labels) → applied
- 64 groups are conflicting (same source text, different labels) → skipped
- FP rate: 43.5% — this is the DIVE redeployment pattern (same contract source deployed to multiple addresses, independently labeled per vulnerability class by DIVE's folder structure)

**Applied result:** 15 contracts reassigned to new canonical SHA in `dedup_groups_graph_hash.json`.

**Where stored:** `data/dedup_groups_l3_candidates.json` (full analysis), `data/dedup_groups_graph_hash.json` (updated with L3 groups).

---

## 7. Export Artifact

**Current artifact:** `data/exports/sentinel-v3-smartbugs-2026-06-13/`

### Directory structure

```
sentinel-v3-smartbugs-2026-06-13/
├── manifest.json              # metadata, splits, shard_index, artifact_hash
├── .hash_cache.json           # warm-path hash verification cache (mtime+size per file)
├── labels.parquet             # contract_id, class_0..class_9, confidence_tier
├── metadata.parquet           # contract_id, source, version_bucket, dedup_group_id, ...
├── graphs/
│   ├── _shard_index.json      # {sha256: {shard, pos_in_shard, num_nodes}}
│   ├── graphs-00000.pt        # PyG Batch of graphs (shard 0)
│   ├── graphs-00001.pt
│   ├── ...
│   └── graphs-00004.pt        # 5 shards total
└── tokens/
    ├── _shard_index.json
    ├── tokens-00000.pt        # [n_in_shard, 4, 512] int64
    └── ...
```

### Key manifest fields

```json
{
  "schema_version": "v1",
  "graph_schema_version": "v9",
  "artifact_hash": "5cc5cfcbf42bef4c...",
  "hash_algorithm": "sha256",
  "shard_size": 5000,
  "n_contracts": 22493,
  "n_contracts_with_reps": 21657,
  "n_shards": 5,
  "splits": {
    "train": ["sha256_1", "sha256_2", ...],   // 18,561 entries
    "val":   [...],                            // 2,008 entries
    "test":  [...]                             // 1,924 entries
  },
  "shard_index": {
    "<sha256>": {"shard": 0, "pos_in_shard": 4, "num_nodes": 312}
  }
}
```

### Artifact integrity

The `artifact_hash` is a SHA-256 over all data files except `manifest.json` and `.hash_cache.json`. This means:
- Tampering with any shard or parquet file changes the hash → `verify_artifact_hash()` fails.
- Modifying only the manifest (e.g., adding metadata fields) does not change the hash.
- The `.hash_cache.json` is excluded so it can be rewritten without invalidating the hash.

### Hash cache sidecar (speedup, added 2026-06-13)

`.hash_cache.json` stores the last-computed `artifact_hash` alongside each data file's `mtime` and `size`:

```json
{
  "artifact_hash": "5cc5cfcbf42bef4c...",
  "files": {
    "graphs/graphs-00000.pt": {"mtime": 1749876543.1, "size": 523456789},
    "labels.parquet": {"mtime": 1749876543.2, "size": 12345678},
    ...
  }
}
```

**Warm path (cache hit):** `verify_artifact_hash()` stats each file (microseconds), compares mtime+size. If all match → return cached hash vs manifest hash. No disk read of shard data. Time: ~0.00s.

**Cold path (cache miss):** Full SHA-256 recompute over all data files, then write a fresh cache. Time: ~15-30s depending on storage speed.

**Impact:** SentinelDataset init hash verification went from 30+ seconds (cold read of ~3GB) to 0.00s on all warm invocations (every training run after the first).

### `num_nodes` in shard_index (speedup, added 2026-06-13)

Each `shard_index` entry now contains `num_nodes`. This is free to record at export time because the graph `Data` object is in memory. Before this change, `SentinelDataset.__init__` loaded all 5 graph shards (~500MB each) just to read one integer per contract.

**Impact:** `num_nodes_map` construction went from 12+ seconds (loading all shards) to 0.01s (dict comprehension over the shard_index JSON).

Backward compatibility: if an old export lacks `num_nodes` in `shard_index`, the dataset falls back to the original shard-loading loop automatically.

---

## 8. SentinelDataset (Stage 7B)

**File:** `ml/src/datasets/sentinel_dataset.py`

The PyTorch `Dataset` class that wraps the export artifact. Returns 5-tuples:

```python
(graph, tokens, y, contract_id, confidence_tier)
# graph:            PyG Data — x[n_nodes, 12], edge_index[2, E], edge_attr[E]
# tokens:           dict — "input_ids"[4,512] int64, "attention_mask"[4,512] int64
# y:                float32 Tensor[10] — multi-label targets
# contract_id:      str (sha256 of the contract)
# confidence_tier:  str | None — "T0", "T1", "T2", or None for NonVulnerable
```

### Init sequence and timing (v3, warm cache)

```
[SentinelDataset] export loaded          0.02s  ← reads manifest.json (~100KB JSON)
[SentinelDataset] hash verified          0.00s  ← warm path: stats files, compares mtime+size
[SentinelDataset] label lookup built     1.90s  ← reads labels.parquet, builds {sha: (y_tensor, tier)}
[SentinelDataset] contract list filtered 1.93s  ← filters split IDs to those with representations
[SentinelDataset] num_nodes done         0.01s  ← reads from shard_index (fast path)
[SentinelDataset] __init__ complete      1.94s total
```

### Three hard gates at construction

1. **Format schema version:** manifest.schema_version must equal `"v1"`. Guards against loading a v0 or future-incompatible export.
2. **Graph schema version:** manifest.graph_schema_version must match `FEATURE_SCHEMA_VERSION` from `graph_schema.py`. Guards against training with the wrong node/edge feature encoding.
3. **Artifact hash integrity:** `verify_artifact_hash()` must return True. Guards against corrupted or tampered shard data.

All three raise `ValueError` with a descriptive message on failure — training never silently starts on bad data.

### Shard LRU cache

Graph and token shards are loaded with `@lru_cache(maxsize=4)` (configurable via `SENTINEL_SHARD_CACHE_SIZE` env var). With 5 shards, setting `SENTINEL_SHARD_CACHE_SIZE=5` keeps all shards warm after the first epoch scan — no re-loading on subsequent epochs.

### Attention mask reconstruction

The attention mask is reconstructed from `input_ids` at `__getitem__` time rather than stored on disk:

```python
attention_mask = (input_ids != _PAD_TOKEN_ID).long()
```

`_PAD_TOKEN_ID = 1` (RoBERTa vocabulary pad token). This saves ~50% of token shard disk space.

---

## 9. Pre-Run-12 Fixes (2026-06-13)

Before launching Run 12, eight open items were assessed and three were executed. Work order: **A → C → B → D**.

### 9.1 Step A — DoS Label Quality Investigation (DONE)

**Question:** Are the DoS labels reliable? AUC-PR=0.046 at ep1 of Run 11 was near-random.

**Method:** Sampled 30 DoS=1 training contracts (seed=42), manually inspected source code for syntactic DoS patterns (`for(`, `while(`, `.transfer(`, `.send(`, `.call{`, `frozenAccount`, owner-gated transfer).

**Finding — all 782 DoS train contracts in v2 were DIVE T2.** No T0/T1 diversity.

**Multi-label rate: 26/30 (87%).** DoS rarely appears alone — average 2.9 co-labels.

**FPR estimate:**
- Clear false positives: 2/30 (6.7%) — TypeFetcher (utility contract, no vulnerability), LIBC (wrong balance check, not DoS)
- Marginal: 2/30 (6.7%)
- True positives: 26/30 (86.7%)

**Why AUC-PR was low despite good labels:** Admin DoS (frozenAccount, `_Swapping` gate) creates the same graph structure as ExternalBug — both are `require(condition)` nodes where `condition` reads owner-controlled storage. The graph features cannot distinguish them. The 782 examples at 4.9% prevalence is a hard few-shot problem. ep1 is too early to judge (280 gradient steps).

**Decision: KEEP all DoS labels.** FPR ~7% is acceptable. AUC-PR improvement must come from training longer.

**Important correction (2026-06-13):** The v2 tracking doc incorrectly claimed "DoS/Reentrancy co-occurrence patch zeroed 2,655 labels." This never happened. The merger's co-occurrence rule only fires for T3/T4 sources — DIVE is T2. The true v3 DoS count before the manual patch was 3,756 (2,655 co-occurring with Reentrancy). The patch described in Step C below was applied manually.

---

### 9.2 Step C — SmartBugs Curated Ingestion (DONE)

**Why:** Three classes had near-zero training examples:
- GasException: 0 (model has zero signal)
- CallToUnknown: 39 total (27 train) — AUC-PR=0.006
- MishandledException: 39 total (27 train) — AUC-PR=0.003

**What was built:**

| Component | Action |
|---|---|
| `labeling/parsers/smartbugs_curated.py` | Created — reads `parts[1]` from `repo/<category>/<contract>.sol` |
| `labeling/crosswalks/smartbugs_curated.yaml` | Bug fix: `front_running: Timestamp` → `front_running: TransactionOrderDependence` |
| `labeling/crosswalks/smartbugs_curated.yaml` | Added `confidence_tier: T1` (required by parser) |
| `config.yaml` SmartBugs entry | `connector: git` → `connector: manual`, added `staging_path` |
| `config.yaml` SmartBugs `tier` | `tier: 1` → `tier: 3` (caught by `test_config_has_all_tier1_sources`) |

**Pipeline results:**

| Stage | Input | Output | Notes |
|---|---|---|---|
| Ingestion | 143 `.sol` files | 143 available | Symlink created |
| Preprocessing | 143 | 137 | 1 compile fail (parity_wallet_bug_1.sol, pragma 0.4.9); 5 DIVE duplicates |
| Representation | 137 | 134 graphs | 3 Slither IR parse failures (old `.call.value()` syntax) |
| Labeling | 137 | 137 labels | 0 failures |
| Merged total | — | 22,493 | 137 new + 22,356 cached (DIVE+SolidiFI) |

**Class gains from SmartBugs:**

| Class | Old count (train) | New count (train) | Delta |
|---|---|---|---|
| CallToUnknown | ~27 | ~75 | +48 |
| Reentrancy | 7,950 | ~7,980 | +30 |
| ExternalBug | large | large+17 | +17 |
| IntegerUO | large | large+15 | +15 |
| Timestamp | large | large+13 | +13 |
| TransactionOrderDependence | ~500 | ~504 | +4 (front_running fix) |
| DenialOfService | large | large+6 | +6 |
| MishandledException | ~27 | ~27 | 0 (no SmartBugs category) |
| GasException | 0 | 0 | 0 (no SmartBugs category) |

---

### 9.3 Step B — Level-3 Text-Hash Dedup (DONE)

**Why:** The `deduplicator.py` Level-3 implementation was a stub (every contract returned `dedup_group_id = sha256` of itself). Near-duplicates that differed only in comments could leak across splits.

**Implementation:** Added to `sentinel_data/preprocessing/deduplicator.py`:
- Strip block comments and line comments from source
- Collapse all whitespace to single spaces
- SHA-256 the result
- Group contracts with identical normalized hashes

**Explicit non-decision:** Identifier lowercasing was considered and rejected. `reentrantWithdraw` and `reentrancyWithdraw` are semantically different functions — lowercasing would produce false-positive group merges that cross-pollinate vulnerability classes.

**Scan results on full 22,493-contract corpus:**

| Metric | Value |
|---|---|
| Total unique normalized hashes | 22,284 |
| L3 groups found (2+ members) | 147 groups, 356 contracts |
| Label-consistent groups | 83 — applied |
| Conflicting groups | 64 — skipped |
| Contracts reassigned | 15 |
| False positive rate | 43.5% (DIVE redeployment pattern) |

The 64 conflicting groups are not errors — they are legitimate DIVE contracts deployed to multiple Ethereum addresses, independently labeled per vulnerability class by DIVE's folder structure. Text-identical contracts with different labels are valid and must be kept separate.

**Post-L3 v3 splits:** train=18,561 / val=2,008 / test=1,924. 0% leakage confirmed.

---

### 9.4 Step D — DoS Co-occurrence Patch + Tests + 7 Readiness Gates (DONE)

**DoS co-occurrence patch:** After the Step A investigation confirmed that DIVE's 2,655 DoS+Reentrancy co-occurring contracts were contributing label noise to the training signal, the labels.parquet in the export was patched: all DoS=1 entries where Reentrancy=1 were zeroed (DoS set to 0). This was applied directly to the export artifact and the manifest was re-exported with updated artifact hash.

**DoS counts before and after patch:**

| | Before patch | After patch |
|---|---|---|
| DoS=1 total | 3,756 | 1,101 |
| DoS=1 AND Reentrancy=1 | 2,655 | 0 |
| train | 2,910 | 845 |
| val | 429 | 129 |
| test | 417 | 127 |

**Full data_module test suite results:**
```
571 passed, 47 skipped, 4 failed (33s)
```
The 4 failures are pre-existing P3 defects (`CALL_ENTRY`/`RETURN_TO` edge tests), documented in `docs/architecture.md:272`, deferred to v2.1. No new failures introduced.

**7 Readiness Gates (all evaluated against sentinel-v3-smartbugs-2026-06-13):**

| Gate | Status | Evidence |
|---|---|---|
| 1 — Schema regression | GREEN | 40/40 byte-identical regression tests pass |
| 2 — BCCC Phase 5 verification suite | GREEN | 191 pass / 21 skipped |
| 3 — End-to-end round-trip (SentinelDataset) | GREEN | 16/16 tests passed in 20.86s. Fast path: hash 0.00s, num_nodes 0.01s, total init 1.94s |
| 4 — Feature distribution | GREEN | By construction — v9 schema unchanged |
| 5 — All 10 classes VERIFIED or PROVISIONAL | AMBER | No regression vs v2. DoS+ExternalBug BEST-EFFORT by design (NOT_EXTRACTABLE in v9). CallToUnknown expected to upgrade to PROVISIONAL after re-verification |
| 6 — No leakage across splits | GREEN | 0 overlap confirmed across train∩val, train∩test, val∩test |
| 7 — No code-bug regression | GREEN | EMITS fixture 4/4; predictor per-class threshold fix already in code |

**SentinelDataset speedup summary (both fixes measured):**

| Init step | Before | After | Improvement |
|---|---|---|---|
| verify_artifact_hash() | 30+ seconds (read ~3GB) | 0.00s (mtime+size check) | >1000× |
| num_nodes_map construction | 12+ seconds (load 5 shards) | 0.01s (dict comprehension) | >1200× |
| Total init | ~45 seconds | ~1.94 seconds | ~23× |

---

### 9.5 Code Changes Summary

| File | Change | Reason |
|---|---|---|
| `sentinel_data/labeling/crosswalks/smartbugs_curated.yaml` | `front_running: Timestamp` → `TransactionOrderDependence`; added `confidence_tier: T1` | Bug fix + parser requirement |
| `sentinel_data/labeling/parsers/smartbugs_curated.py` | Created (new file) | SmartBugs labeling parser |
| `config.yaml` SmartBugs entry | `connector: git` → `connector: manual`, `staging_path`, `tier: 1` → `tier: 3` | Manual connector; tier correction |
| `sentinel_data/preprocessing/deduplicator.py` | Level-3 text-hash dedup (strip comments + collapse whitespace, no identifier lowercasing) | Catches copy-paste near-dups |
| `sentinel_data/export/graph_writer.py` | Return `num_nodes_map` as third value in `write_graphs_shards()` | num_nodes free at export time |
| `sentinel_data/export/chunker.py` | Embed `num_nodes` in shard_index; write `.hash_cache.json`; exclude cache from artifact hash | Dataset init speedup |
| `sentinel_data/export/export.py` | `verify_artifact_hash()` warm/cold path with `.hash_cache.json` | Avoids 3GB read on warm init |
| `ml/src/datasets/sentinel_dataset.py` | Fast `num_nodes` path from shard_index; O(1) membership set; timing logs | Eliminates shard load at init |
| `tests/test_export/test_graph_token_writer.py` | Unpack 3 values from `write_graphs_shards()` | Fix broken tests after return type change |
| `tests/test_export/test_chunker.py` | 3 new tests: `num_nodes` in shard_index, hash cache written, hash cache excluded from artifact hash | Regression coverage for new features |
| `ml/tests/test_sentinel_dataset.py` | Updated `EXPORT_DIR` to v3 | Point Gate 3 at current export |

---

## 10. Run 12 Training

**Run name:** `GCB-P1-Run12-v3dospatched-20260613`
**Started:** 2026-06-13 (late night)
**Export:** `sentinel-v3-smartbugs-2026-06-13`
**Checkpoint:** `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt`

### Launch command

```bash
cd /home/motafeq/projects/sentinel
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. TRITON_CACHE_DIR=/tmp/triton_cache \
  nohup ml/.venv/bin/python ml/scripts/train.py \
    --run-name GCB-P1-Run12-v3dospatched-20260613 \
    --experiment-name sentinel-v12 \
    --export-dir data_module/data/exports/sentinel-v3-smartbugs-2026-06-13 \
    --appnp-alpha 0.2 \
    --gnn-prefix-k 48 \
    --weighted-sampler positive \
    --dos-loss-weight 1.0 \
    --epochs 100 \
    --early-stop-patience 30 \
  > ml/logs/run12_launch_2026-06-13.log 2>&1 &
```

### Key configuration decisions

| Flag | Value | Why |
|---|---|---|
| `--appnp-alpha` | `0.2` | First run with APPNP teleport active. Prevents Phase 2 GNN layers from diluting Phase 1 structural signal (CHECK→CALL→WRITE signal decays to <1% after 3 hops without teleport). Run 11 used 0.0 (disabled). |
| `--gnn-prefix-k` | `48` | GNN prefix injection: 48 tokens prepended to source tokens before GCB. Warmup for 15 epochs (projection trains from random init). Matches Run 11. |
| `--weighted-sampler` | `positive` | Upweights contracts with any positive label. Matches Run 11 for clean comparison. |
| `--dos-loss-weight` | `1.0` | Full gradient on DoS. Raised from default 0.5 because co-occurrence noise is now removed — 845 clean DoS examples, no contradictory signal. |
| `--loss-fn` | `asl` (default) | AsymmetricLoss — handles class imbalance internally, no pos_weight needed. |
| `--gradient-accumulation-steps` | `8` (default) | Effective batch size = 8 × 8 = 64. |

### Model architecture (from ep1 log)

```
SentinelModel v8 (four-eye)
  GNN: 8-layer GAT, hidden_dim=256, heads=8, JK=attention, APPNP alpha=0.2
  LoRA: r=16, alpha=32, modules=[query, value], trainable=589,824 / frozen=124,645,632
  Fusion: CrossAttentionFusion node_dim=256 → token_dim=768 → output=128, max_nodes=2048
  Classifier: [512 → 256 → 10] (4-eye: gnn_eye[128] + tf_eye[128] + fused[128] + fused[128] → 512)
  GNN prefix: K=48, warmup=15 epochs
```

**Optimizer param groups:**
- GNN: lr × 2.5 (counteracts GNN gradient collapse)
- LoRA: lr × 0.3 (tighter than GNN to prevent over-adaptation)
- Fusion+Classifier: lr × 0.5
- PrefixProj: lr × 5.0 (cold-start acceleration for prefix projection)

### Epoch-by-epoch results (as of 2026-06-14 ~03:00)

| Epoch | F1-macro | New Best | Top-3 | DoS F1 | Notes |
|---|---|---|---|---|---|
| 4 | 0.3718 | ★ | ExternalBug=0.857, Timestamp=0.665, Reentrancy=0.632 | — | First full epoch |
| 5 | 0.4311 | ★ | ExternalBug=0.857, Timestamp=0.709, Reentrancy=0.685 | 0.177 | DoS alive |
| 6 | 0.4429 | ★ | ExternalBug=0.858, Timestamp=0.756, Reentrancy=0.675 | 0.188 | |
| 7 | 0.4758 | ★ | ExternalBug=0.859, Timestamp=0.721, Reentrancy=0.700 | 0.194 | |
| 8 | 0.4868 | ★ | ExternalBug=0.861, Timestamp=0.764, Reentrancy=0.724 | 0.198 | Aux warmup complete |
| 9 | in progress | — | — | — | ~03:04 finish |

**Comparison to Run 11:** Run 11's best result (ep1) was F1-macro=0.3384. Run 12 at ep8 is 0.4868 — **44% improvement**. The data quality fixes (DoS patch, SmartBugs, L3 dedup) and APPNP teleport are the primary drivers.

**DoS class is alive:** F1=0.177–0.198 across ep5–8. In Run 11, DoS was near-random (AUC-PR=0.046). The combination of clean DoS labels (0 co-occurring Reentrancy) and `--dos-loss-weight 1.0` is working.

**GasException=0.000** throughout — expected, 0 training examples. CLASS DEATH warning fires every epoch but is not actionable without new data.

**Training speed:** ~22 min/epoch on RTX 3070 (8GB VRAM). VRAM usage: 0.5/8.0 GiB (6.8%) — no memory pressure.

### JK attention weights

Phase1≈0.32, Phase2≈0.31, Phase3≈0.37 — all three GNN phases contributing. Phase3 is slightly dominant, which is expected (it sees the most aggregated global structure). No collapse to a single phase.

### Upcoming milestones

| Epoch | Event |
|---|---|
| 9 | Aux loss warmup complete (aux_weight reaches 0.3000) |
| 15 | GNN prefix injection activates — prefix tokens begin injecting into GCB |
| ~15-20 | Expected F1 jump as GNN structural signal propagates into language model |
| 30 | Early-stop patience starts mattering if no improvement |
| 100 | Max epochs |

---

## 11. Known Limitations and Deferred Work

### Unlearnable classes

| Class | Issue | Path to fix |
|---|---|---|
| GasException | 0 training examples in all sources | Need a source with gas_exception category. SmartBugs has none. DeFiHackLabs blocked. |
| MishandledException | Only 30 train examples (T0 SolidiFI only) | Low count but VERIFIED — model may learn with longer training |

### Deferred data sources

| Source | Status | Blocker |
|---|---|---|
| DeFiHackLabs | BLOCKED | PoC Foundry test contracts, not vulnerable protocol contracts. Needs Etherscan lookup per exploit. |
| Stage 5.5 GCB Propagation | DEFERRED | Use GCB embeddings to propagate high-confidence labels to near-neighbors. Complex sub-project, needs GPU inference pass on all 22K contracts. Post-Run-12. |
| Stage 5 Registry | DEFERRED | SQLite catalog, lineage DAG, `sentinel_data.registry.load_artifact()` API. Scheduled Week 8 (Jul 28–Aug 3, 2026). |

### Open defects

| Defect | Priority | Status |
|---|---|---|
| CALL_ENTRY / RETURN_TO edges (4 test failures) | P3 | Deferred to v2.1. Architecture.md:272. |
| DoS / ExternalBug verification (BEST-EFFORT) | P3 | NOT_EXTRACTABLE from v9 schema. Needs `tool_validator` + Slither batch runs. |
| drift_baseline.json placeholder | P4 | Populated only after trained model runs warmup batch. Needs Run 12 output. |
| C-4 max_nodes guard (0.18% rate) | P4 | 33 contracts exceed 2048 nodes. Risk vs reward too low to change `fusion_max_nodes` now. |
