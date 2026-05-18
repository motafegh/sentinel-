# SENTINEL ML Dataset Pipeline — Technical Reference

## Overview

The 68,555-sample training dataset was built offline from the **BCCC-SCsVul-2024** benchmark dataset.
The pipeline runs **once** — the resulting `.pt` files are the training data. Understanding this
pipeline is essential because the inference preprocessor (`preprocess.py`) must replicate it exactly.

```
BCCC-SCsVul-2024/SourceCodes/<VulnType>/*.sol   (raw contracts, 111,798 files total)
      │
      ▼
[1] slither_wrapper / Slither — solc version detection
      │
      ▼
[2] ast_extractor_v4_production.py  →  ml/data/graphs/<hash>.pt  (68,556 files)
      │
      ▼
[3] tokenizer_v1_production.py      →  ml/data/tokens/<hash>.pt  (68,570 files)
      │
      ▼
[4] create_label_index.py           →  ml/data/processed/label_index.csv
      │
      ▼
[5] create_splits.py                →  ml/data/splits/{train,val,test}_indices.npy
```

`fix_labels_from_csv.py` was a one-time patch that corrected labels in the graph files using
the ground-truth CSV. It is a historical artefact — the current `.pt` files already have correct
labels.

---

## Primary Data Source: BCCC-SCsVul-2024

**Location:** `BCCC-SCsVul-2024/` (project root)

BCCC-SCsVul-2024 is an academic benchmark dataset of labelled Solidity smart contracts. It is
organised as **12 subdirectories under `SourceCodes/`** — one directory per vulnerability class
(plus one for safe contracts):

| Folder | Vulnerability | Files |
|--------|--------------|-------|
| `CallToUnknown/` | External call to unknown/untrusted address | ~11,131 |
| `DenialOfService/` | DoS via gas exhaustion, failed calls, or blocked state | ~12,394 |
| `ExternalBug/` | Bugs arising from untrusted external contract behaviour | ~3,604 |
| `GasException/` | Unexpected out-of-gas or gas-related exceptions | ~6,879 |
| `IntegerUO/` | Integer overflow and underflow | ~16,740 |
| `MishandledException/` | Unchecked return values or silently caught exceptions | ~5,154 |
| `Reentrancy/` | Classic reentrancy (CEI pattern violation) | ~17,698 |
| `Timestamp/` | Block timestamp manipulation / dependence | ~2,674 |
| `TransactionOrderDependence/` | Front-running / transaction ordering attacks | ~3,562 |
| `UnusedReturn/` | Return values discarded without checking | ~3,229 |
| `WeakAccessMod/` | Missing or incorrect access modifiers | ~1,918 |
| `NonVulnerable/` | Safe contracts (negative class) | ~26,914 |
| **Total** | | **~111,798 files** |

The dataset also ships with `BCCC-SCsVul-2024.csv` (71.6 MB), a feature-rich CSV with 68 columns:
56 statistical features (lines of code, Solidity language features, bytecode stats) plus 12 binary
class indicator columns (Class01–Class12, one per vulnerability type).

---

## Critical Finding: The Dataset Is Genuinely Multi-Label

**This is the most important thing to understand about the data.**

Contracts are **not** mutually exclusive to one folder. The same contract file (identified by its
SHA256 hash as the filename) appears in **multiple** vulnerability folders when it exhibits multiple
vulnerability types simultaneously. A measurement across all 12 folders shows:

```
Folder appearances per unique contract hash:
  1 folder  (single-label): 40,267 contracts  (58.8%)
  2 folders:                19,068 contracts
  3 folders:                 5,473 contracts
  4 folders:                 1,871 contracts
  5 folders:                 1,138 contracts
  6–9 folders:               1,587 contracts
  ─────────────────────────────────────────────────
  Multi-label total:        28,166 contracts  (41.2%)

  Total unique hashes in dataset: 68,433
```

**Real example:** hash `48e59d16542b5861...` appears in seven folders: Timestamp, UnusedReturn,
Reentrancy, IntegerUO, MishandledException, TransactionOrderDependence, CallToUnknown. That is a
single Solidity file carrying all seven vulnerability types.

Additionally, **766 contracts** appear in both at least one vulnerability folder and the
`NonVulnerable` folder — a direct label contradiction in the raw dataset.

### What This Means for the Model

The current model is trained on **binary labels only** (safe=0 / vulnerable=1). The multi-label
nature of the raw data is not exploited. This was a deliberate preprocessing simplification — see
the section below.

---

## How Binary Labels Were Produced: The Simplification

`contract_labels_correct.csv` contains 44,442 rows with **zero duplicate hashes and exactly one
class column = 1 per row**. Here is how that happened:

### Step 1 — Hash deduplication

When the preprocessing script scanned the 12 SourceCodes subfolders, it encountered the same SHA256
hash in multiple folders. It processed each unique hash **once**, assigning the label from the
folder where it was **first encountered**. All other folder memberships for that hash were silently
discarded.

This lost the multi-label signal for 14,176 multi-folder contracts that ended up in the CSV
(assigned a single primary class). The remaining 28,166 - 14,176 = ~14,000 multi-folder contracts
were excluded entirely (Slither extraction failures during graph building).

### Step 2 — Slither failures drop ~24,000 contracts

Of 68,433 unique hashes in BCCC, only 44,442 made it into the CSV. The missing ~24,000 were
dropped because Slither could not parse them (unsupported solc version, import resolution failures,
malformed source). The graph extractor runs Slither with version-matched `solc` binaries, but some
contracts still fail.

### Step 3 — 12-class → binary collapse

The CSV has a `class_label` column (integer 0–11) and a `binary_label` column (0 or 1). Binary
label assignment:
- `class_label == 11` (NonVulnerable) → `binary_label = 0`
- `class_label 0–10` (any vulnerability type) → `binary_label = 1`

Only `binary_label` is baked into `graph.y` in the final `.pt` files. The `class_label` column
exists in the CSV but is **not used** by the training pipeline.

### Why Binary Was Chosen

1. **Preprocessing design** — the pipeline was written with binary in mind from the start.
   Single-label-per-hash deduplication made it a multi-class problem; collapsing to binary was then
   straightforward.

2. **Multi-label is genuinely harder with this data.** Treating it correctly would require:
   - A multi-hot `[12]` label vector per unique hash (scanning all folder memberships)
   - `BCEWithLogitsLoss` with per-class weighting
   - Per-class threshold tuning (12 separate sweeps)
   - Handling extreme class imbalance per type (Reentrancy: ~122 samples in CSV, IntegerUO: ~16K)
   - Deciding what to do with the 766 contradictory labels

3. **The core oracle question is answered by binary.** "Does this contract have any vulnerability?"
   is what on-chain registries and auditors need first. Per-type classification is a natural
   future extension, not a prerequisite.

**Important:** The ZKML (zero-knowledge proof) pipeline and AuditRegistry smart contract were
built *after* the binary classification decision and were designed to match it — they are not
the reason binary was chosen.

---

## The Processed Ground-Truth CSV

**File:** `ml/data/processed/contract_labels_correct.csv`
**Size:** ~9.1 MB, 44,442 rows

```
contract_path, file_hash, binary_label, class_label,
Class01:ExternalBug, Class02:GasException, Class03:MishandledException,
Class04:Timestamp, Class05:TransactionOrderDependence, Class06:UnusedReturn,
Class07:WeakAccessMod, Class08:CallToUnknown, Class09:DenialOfService,
Class10:IntegerUO, Class11:Reentrancy, Class12:NonVulnerable
```

**Example row:**
```
BCCC-SCsVul-2024/SourceCodes/CallToUnknown/f805dcc...sol,
f805dcc8abff966430ef42299b248d571a37bd0eef42be078ff82355dfe3251a,
1,7,0,0,0,0,0,0,0,1,0,0,0,0
```
- `binary_label=1` → vulnerable
- `class_label=7` → CallToUnknown (0-indexed: ExternalBug=0, ..., CallToUnknown=7, ..., NonVulnerable=11)
- `Class08:CallToUnknown=1` → the one-hot class indicator

**Distribution in CSV (44,442 rows):**
```
Safe       (binary=0, class=11 NonVulnerable): 24,461  (55.0%)
Vulnerable (binary=1, classes 0–10):           19,981  (45.0%)

  By vulnerability type:
  Class00 ExternalBug:               ~3,409
  Class01 GasException:              ~5,008
  Class02 MishandledException:       ~2,729
  Class03 Timestamp:                   ~944
  Class04 TransactionOrderDependence:  ~848
  Class05 UnusedReturn:                ~349
  Class07 CallToUnknown:             ~2,025
  Class08 DenialOfService:             ~138
  Class09 IntegerUO:                 ~4,409
  Class10 Reentrancy:                  ~122   ← most severe class imbalance
```

Note that class indices 0–10 in `class_label` do **not** align one-to-one with Class01–Class12
column names. The `class_label` integer is 0-indexed from ExternalBug, while the Class columns are
named 01–12. Always use the Class column names to identify vulnerability types unambiguously.

---

## Shared Foundation: `hash_utils.py`

**File:** `ml/src/utils/hash_utils.py`

All pipeline components use MD5 hashing for contract identification. The hash is the filename stem
for both graph and token files — the only pairing mechanism.

```python
from ml.src.utils.hash_utils import get_contract_hash, get_filename_from_hash

hash_id  = get_contract_hash("path/to/Contract.sol")   # MD5 of full path string
filename = get_filename_from_hash(hash_id)             # f"{hash_id}.pt"
```

**Key functions:**

| Function | Input | Output | Usage |
|---|---|---|---|
| `get_contract_hash(path)` | File path (str or Path) | 32-char MD5 hex | Graph + token file naming |
| `get_contract_hash_from_content(code)` | Source code string | 32-char MD5 hex | API-layer content-based caching |
| `get_filename_from_hash(hash)` | 32-char MD5 | `f"{hash}.pt"` | Filename generation |
| `validate_hash(hash_str)` | Any string | bool | Integrity checks |
| `extract_hash_from_filename(fname)` | `.pt` filename | hash or None | Reverse lookup |

**Critical design decision:** Hash is computed from the **full absolute path string**, not file
content. Two files at different paths with identical content get different hashes. This was
intentional — the dataset uses path-based labels from a CSV, so two copies of the same source code
may have different vulnerability labels.

This differs from `process_source()` in `preprocess.py`, which hashes content (for
content-addressable caching in the API layer).

**Note on BCCC file naming:** The original BCCC filenames are **SHA256 hashes** of the contract
content, not MD5 hashes. The internal pipeline uses MD5 hashes of the **file path** as the `.pt`
stem. These are different hash schemes and should not be confused.

---

## Step 2: Graph Extraction

**File:** `ml/scripts/ast_extractor_v4_production.py`
**Class:** `ASTExtractorV4`
**Version:** V4.2 (February 15, 2026)

### What it does

Runs Slither on every `.sol` file, extracts AST/CFG structure, builds a PyG `Data` object, and
saves it as `<hash>.pt` in `ml/data/graphs/`.

**Input:** `ml/data/processed/contracts_metadata.parquet` — contains `contract_path`,
`detected_version`, `success` columns for all contracts.

**Output:** `ml/data/graphs/<md5_hash>.pt` — PyG `Data(x=[N,8], edge_index=[2,E],
edge_attr=[E,1], y=[label], contract_hash, contract_path, contract_name)`

### What goes into `graph.y`

The label baked into `graph.y` is the **binary label only** (0 or 1). Class-level labels from the
CSV are not stored in the `.pt` files. This means:
- Re-extracting with multi-class or multi-label labels would require re-running this script
- All 68,556 graph files would need to be regenerated
- The CSV (`contract_labels_correct.csv`) retains the `class_label` and Class01–12 columns and is
  the source of truth for the richer label information

### Node feature vector (8-dim, float32)

Identical to `preprocess.py`'s `_extract_graph()`. Must stay in sync — any drift causes inference
results to not match training.

| Index | Feature | Encoding |
|---|---|---|
| 0 | `type_id` | 0=STATE_VAR, 1=FUNCTION, 2=MODIFIER, 3=EVENT, 4=FALLBACK, 5=RECEIVE, 6=CONSTRUCTOR, 7=CONTRACT |
| 1 | `visibility` | 0=public/external, 1=internal, 2=private |
| 2 | `pure` | 1.0 if pure function, else 0.0 |
| 3 | `view` | 1.0 if view function, else 0.0 |
| 4 | `payable` | 1.0 if payable, else 0.0 |
| 5 | `reentrant` | Slither's `is_reentrant` flag |
| 6 | `complexity` | `float(len(func.nodes))` — CFG node count |
| 7 | `loc` | `float(len(source_mapping.lines))` — lines of code |

### Parallelisation strategy

Contracts are grouped by Solidity version (each version needs a different `solc` binary). Within
each version group, a `multiprocessing.Pool` of 11 workers runs `contract_to_pyg()` in parallel
via `pool.imap()`.

```python
for version, group in groups:
    solc_bin = get_solc_binary(version)
    worker = partial(contract_to_pyg, solc_binary=solc_bin, solc_version=version)
    with mp.Pool(processes=11) as pool:
        for result in pool.imap(worker, group["contract_path"].tolist(), chunksize=50):
            torch.save(result, output_dir / f"{result.contract_hash}.pt")
```

**Why version-grouped?** Each Slither call invokes `solc`. Different contracts require different
compiler versions. Grouping avoids constant version-switching overhead.

### Checkpoint system

A `checkpoint.json` file in the output directory tracks processed hashes. Checkpoints every 500
contracts. On resume (`--resume`), already-processed hashes are filtered from the DataFrame.

### Solc version compatibility fix (V4.2)

`--allow-paths` was introduced in solc 0.5.0. For contracts targeting solc 0.4.x, this flag causes
a compiler error. V4.2 adds:

```python
def solc_supports_allow_paths(version: str) -> bool:
    major, minor, _ = parse_solc_version(version)
    return (major, minor) >= (0, 5)
```

### Running (if rebuilding from scratch)

```bash
poetry run python ml/scripts/ast_extractor_v4_production.py \
    --input ml/data/processed/contracts_metadata.parquet \
    --output ml/data/graphs \
    --workers 11

poetry run python ml/scripts/ast_extractor_v4_production.py --test    # first 100 only
poetry run python ml/scripts/ast_extractor_v4_production.py --resume  # after interruption
```

---

## Step 3: Tokenisation

**File:** `ml/scripts/tokenizer_v1_production.py`
**Date:** February 15, 2026

### What it does

Reads each `.sol` file, tokenises with CodeBERT, and saves `<hash>.pt` in `ml/data/tokens/`.

**Input:** `ml/data/processed/contracts_metadata.parquet`
**Output:** `ml/data/tokens/<md5_hash>.pt` — dict with `input_ids [512]`, `attention_mask [512]`,
metadata

### Tokenisation parameters

```python
TOKENIZER_MODEL = "microsoft/codebert-base"
MAX_LENGTH      = 512    # CodeBERT hard limit
PADDING         = "max_length"   # Always pad to 512
TRUNCATION      = True   # Cut long contracts
```

These **must** match `preprocess.py`'s `_tokenize()` exactly.

### Worker initialisation pattern

The tokenizer (500 MB) is loaded **once per worker process**, not once per contract:

```python
tokenizer = None

def init_worker():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)

with mp.Pool(processes=11, initializer=init_worker) as pool:
    pool.imap(tokenize_single_contract, contract_paths, chunksize=50)
```

### Token tensor shapes

```python
input_ids      = encoded["input_ids"].squeeze(0)        # [1,512] → [512]
attention_mask = encoded["attention_mask"].squeeze(0)   # [1,512] → [512]
```

Token files store `[512]` tensors (1D). The training collate function stacks them to `[B, 512]`.
The inference `_tokenize()` method keeps `[1, 512]` for direct model input.

### Truncation detection

```python
truncated = num_real_tokens >= (MAX_LENGTH - 2)  # 510 = 512 - [CLS] - [SEP]
```

If 510 or more positions are real tokens, the contract almost certainly exceeded 512 and was
truncated. Long contracts lose their tail — inference marks these with `truncated=True`.

### Running

```bash
poetry run python ml/scripts/tokenizer_v1_production.py
poetry run python ml/scripts/tokenizer_v1_production.py --test     # 100 contracts
poetry run python ml/scripts/tokenizer_v1_production.py --resume
```

---

## Step 4: Label Index

**File:** `ml/scripts/create_label_index.py`

Scans all graph files, reads the `y` attribute from each, and writes `label_index.csv`:

```csv
hash,label
a1b2c3...,1
d4e5f6...,0
```

This is a lightweight (~2.4 MB) index mapping each paired hash to its binary label. It is used
by `create_splits.py` to enable stratified splitting without loading all 68,555 graph files.

**Labels here are binary only** — the richer class-level information from the CSV is not
propagated forward. This is a consequence of `graph.y` storing only binary labels.

```bash
poetry run python ml/scripts/create_label_index.py
```

Output: `ml/data/processed/label_index.csv`

---

## Step 5: Train/Val/Test Splits

**File:** `ml/scripts/create_splits.py`

Reads `label_index.csv`, applies stratified splitting (70/15/15, seed=42), and saves three `.npy`
index arrays.

### Stratified splitting

```python
train_indices, temp = train_test_split(
    all_indices,
    test_size=0.30,
    stratify=all_labels,   # preserves class ratio in each split
    random_state=42
)
val_indices, test_indices = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)
```

**Why two passes?** `sklearn` `train_test_split` produces two splits at a time only.

### Output files

```
ml/data/splits/train_indices.npy   # int64 array, 47,988 positions
ml/data/splits/val_indices.npy     # int64 array, 10,283 positions
ml/data/splits/test_indices.npy    # int64 array, 10,284 positions
```

**These are position indices** into the sorted paired-hash list in `DualPathDataset`, not boolean
masks.

### Real class distribution after pairing

The actual distribution observed in the 68,555 paired `.pt` files (not the 44,442-row CSV) is
different because some contracts appear in the training data multiple times if their graph file was
built from different occurrences:

```
Vulnerable (label=1): 44,099  (64.33%)  ← majority class
Safe       (label=0): 24,456  (35.67%)  ← minority class
```

The discrepancy vs the CSV (55%/45%) is because multi-folder contracts that were each processed as
separate graph files (with the same content but different path hashes) contribute multiple times to
the graph count. The splits preserve the 64.33%/35.67% ratio via stratified sampling.

```bash
poetry run python ml/scripts/create_splits.py
```

---

## Historical: `fix_labels_from_csv.py`

**File:** `ml/scripts/fix_labels_from_csv.py`
**Status:** Completed — do not re-run

This one-time script re-assigned labels in all graph files using `contract_labels_correct.csv` as
ground truth. It was needed because the initial AST extraction used placeholder or incorrect labels.

The current graph files already have correct labels. Re-running would be harmless but slow.

---

## Known Limitations and Issues

| Issue | Impact | Notes |
|-------|--------|-------|
| Multi-label signal discarded | Model cannot predict *which* vulnerability type is present, only *whether* any exist | 41.2% of raw contracts have multiple vulnerability types; preprocessing picked one label per hash |
| ~24K contracts lost to Slither failures | Smaller training set than BCCC contains | Likely older/malformed Solidity; no fix without changing the parser |
| 766 contradictory labels in raw data | Minor noise | Contracts in both a vuln folder and NonVulnerable; deduplication picks one label |
| Severe per-class imbalance if multi-class were used | — | Reentrancy: ~122 samples, IntegerUO: ~16K samples; would need per-class over/undersampling |
| Binary labels stored in graph.y only | Upgrading to multi-class requires full dataset rebuild | `class_label` column in CSV is preserved but not in `.pt` files |
| BCCC SHA256 vs internal MD5 confusion | — | BCCC filenames are SHA256 of content; internal `.pt` stems are MD5 of file path. Do not mix. |
| 512-token truncation | Long contracts have their tail dropped | `truncated=True` flag is set in inference output; no recovery mechanism |
| Node features do not include Slither detector output | Model cannot directly use Slither's reentrancy flag for a specific call pattern | `is_reentrant` is in node features (dim 5) but only as a binary per-function flag |

---

## Upgrade Path: Binary → Multi-Class (12 classes)

If you want to use the full label richness of BCCC in the future:

1. **Rebuild label vectors from raw folders:**
   ```python
   # For each unique hash, scan all 12 folders for presence
   hash_to_classes = defaultdict(set)
   for folder in SourceCodes.iterdir():
       for f in folder.glob("*.sol"):
           hash_to_classes[f.stem].add(folder.name)
   # Build multi-hot [12] vector per hash
   ```

2. **Decide on multi-hot vs primary-class:** Multi-hot (true multi-label) or single primary class
   (multi-class, picking the rarest/most severe type as primary).

3. **Rebuild graph `.pt` files** with `graph.y = torch.tensor(multi_hot_vector, dtype=torch.float)`

4. **Change model classifier head** from `Linear(64→1) + Sigmoid` to `Linear(64→12)` with
   `BCEWithLogitsLoss` (for multi-label) or `CrossEntropyLoss` (for multi-class).

5. **Rebuild splits** — `create_label_index.py` and `create_splits.py` both assume binary labels.

6. **Rebuild EZKL proxy model** — the ZK circuit is compiled from the model architecture; changing
   the output head requires a new circuit, new proving/verification keys, and a new on-chain
   verifier contract.

---

## The `ml/src/data/graphs/ast_extractor.py` Question

This file contains an older `ASTExtractor` class (text-based `ASTNode`/`ASTEdge` objects, not PyG
Data). It is imported in `preprocess.py`:

```python
from ml.src.data.graphs.ast_extractor import ASTExtractor
```

However, `preprocess.py`'s `_extract_graph()` method never calls `ASTExtractor` — it uses Slither
directly to build PyG Data objects. The import is a **stale reference** from an earlier version.
`ASTExtractor` produces intermediate data structures, not the 8-dim float vectors the model
expects.

`graph_builder.py` (which uses `ASTExtractor`) produces 17-dim one-hot features — incompatible
with the trained model. Neither file is part of the production data pipeline.

---

## File Status Reference

| File | Status | Role |
|---|---|---|
| `ast_extractor_v4_production.py` | **Complete / run once** | Built `ml/data/graphs/` |
| `tokenizer_v1_production.py` | **Complete / run once** | Built `ml/data/tokens/` |
| `create_label_index.py` | **Complete / run once** | Built `label_index.csv` |
| `create_splits.py` | **Complete / run once** | Built `{train,val,test}_indices.npy` |
| `fix_labels_from_csv.py` | **Historical** | One-time label correction, do not re-run |
| `ml/src/utils/hash_utils.py` | **Active** | Used by extractors + tokeniser |
| `ml/src/data/graphs/ast_extractor.py` | **Stale** | Imported but unused in production |
| `ml/src/data/graphs/graph_builder.py` | **Archived** | 17-dim features, wrong for current model |
| `ml/data/archive/old_extractors/` | **Archived** | v2, v3 extractors (superseded) |
