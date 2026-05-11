# SENTINEL — Comprehensive Project Documentation

> **Author:** Ali Motafegh
> **Last Updated:** April 2026
> **Status:** M3.4 complete. Dual-path model trained + inference API live. Production checkpoint: `run-alpha-tune_best.pt` (val F1-macro: 0.6686). ZKML pipeline complete (EZKL proof generated + verified). Agents pipeline in progress.

---

## Table of Contents

1. [What SENTINEL Is](#1-what-sentinel-is)
2. [The Problem Being Solved](#2-the-problem-being-solved)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Project Directory Structure](#4-project-directory-structure)
5. [Data Pipeline — End to End](#5-data-pipeline--end-to-end)
6. [Model Architecture — Every Component](#6-model-architecture--every-component)
7. [Dataset and DataLoader Layer](#7-dataset-and-dataloader-layer)
8. [Dependencies and Environment](#8-dependencies-and-environment)
9. [Test Coverage](#9-test-coverage)
10. [What Remains to Be Done](#10-what-remains-to-be-done)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Numbers at a Glance](#12-numbers-at-a-glance)

---

## 1. What SENTINEL Is

SENTINEL is a **deep learning system for automated smart contract vulnerability detection**. Given a Solidity smart contract as input, SENTINEL outputs a single number between 0 and 1 representing the probability that the contract contains at least one security vulnerability.

It is not a rule-based scanner. It does not run Slither or Mythril at inference time. It is a trained neural network that has learned the structural and semantic patterns that distinguish vulnerable contracts from safe ones.

The system is built around a **dual-path architecture**: two completely different representations of the same contract are computed in parallel and then fused into a single prediction.

- **Path 1 — Structure:** The contract's AST (Abstract Syntax Tree) is converted into a graph. A Graph Attention Network reads this graph and learns how contract components interact — which functions call which, which state variables are read or written by which functions.
- **Path 2 — Semantics:** The raw Solidity source code is tokenized and fed through a frozen CodeBERT model (a transformer pretrained on code). CodeBERT captures the natural-language meaning of variable names, comments, and coding patterns.

Both representations are fused and passed through a classification head that produces the final vulnerability score.

---

## 2. The Problem Being Solved

Smart contract vulnerabilities have caused billions of dollars in losses (DAO hack, Parity wallet freeze, countless DeFi exploits). Traditional static analysis tools like Slither and Mythril work by pattern-matching against known vulnerability signatures. They are:

- **Fast** but have high false-positive rates
- **Rule-based**, meaning they can only catch what they were explicitly programmed to detect
- **Blind** to novel vulnerability patterns

SENTINEL's hypothesis is that a model trained on tens of thousands of labeled vulnerable and safe contracts can learn representations that generalize beyond known patterns. The dual-path design specifically targets the two complementary information sources:

- **Graph structure** captures *how* the contract is wired — reentrancy requires a specific call-back pattern in the call graph; integer overflows appear in arithmetic nodes; access control bugs show up in modifier-to-function relationships.
- **Token semantics** captures *what* the contract says — variable names like `owner`, `locked`, `balance`, function names like `transfer`, `withdraw`, coding idioms the author used.

Neither source alone is sufficient. The fusion is the core innovation.

---

## 3. High-Level Architecture

```
Raw Solidity Contract (.sol file)
          │
          ├──────────────────────────────────┐
          │                                  │
          ▼                                  ▼
  AST Extraction (Slither)          CodeBERT Tokenizer
  ↓ graph: nodes + edges             ↓ tokens: [512] IDs
  Stored as {hash}.pt                Stored as {hash}.pt
  (ml/data/graphs/)                  (ml/data/tokens/)
          │                                  │
          ▼                                  ▼
    GNNEncoder                     TransformerEncoder
  3-layer GAT                       frozen CodeBERT
  [N,8] → [B,64]                    [B,512] → [B,768]
          │                                  │
          └──────────────┬───────────────────┘
                         ▼
                    FusionLayer
              concat [B,832] → MLP → [B,64]
                         │
                         ▼
                 Classifier Head
              Linear(64→1) + Sigmoid
                         │
                         ▼
              Score ∈ [0,1] per contract
```

---

## 4. Project Directory Structure

```
sentinel/
├── ml/
│   ├── pyproject.toml              # All dependencies (Poetry)
│   ├── data/
│   │   ├── graphs/                 # 68,556 PyG graph files ({hash}.pt)
│   │   ├── tokens/                 # 68,570 token files ({hash}.pt)
│   │   ├── splits/
│   │   │   ├── train_indices.npy   # 47,988 int64 position indices
│   │   │   ├── val_indices.npy     # 10,283 int64 position indices
│   │   │   └── test_indices.npy    # 10,284 int64 position indices
│   │   └── processed/
│   │       ├── label_index.csv         # hash → label (68,555 rows)
│   │       ├── contract_labels.csv     # original label mapping
│   │       ├── contract_labels_correct.csv  # corrected labels
│   │       └── contracts_metadata.parquet   # full Slither analysis results
│   ├── src/
│   │   ├── models/
│   │   │   ├── gnn_encoder.py          # 3-layer GAT → [B,64]
│   │   │   ├── transformer_encoder.py  # frozen CodeBERT → [B,768]
│   │   │   ├── fusion_layer.py         # concat + MLP → [B,64]
│   │   │   └── sentinel_model.py       # full model orchestrator
│   │   ├── datasets/
│   │   │   └── dual_path_dataset.py    # DualPathDataset + collate_fn
│   │   ├── data/
│   │   │   ├── bccc_dataset.py         # CSV-based BCCC dataset loader
│   │   │   └── graphs/
│   │   │       ├── ast_extractor.py    # v1 (prototype)
│   │   │       ├── ast_extractor_v2.py # v2 (iteration)
│   │   │       └── ast_extractor_v3.py # v3 (pre-production)
│   │   ├── utils/
│   │   │   └── hash_utils.py           # MD5 hashing — shared by all pipeline stages
│   │   ├── validation/
│   │   │   ├── statistical_validation.py  # IQR outlier detection
│   │   │   └── models.py, models_v2.py    # Pydantic validation schemas
│   │   └── tools/
│   │       └── slither_wrapper.py         # Slither analysis wrapper
│   └── scripts/
│       ├── ast_extractor_v4_production.py  # Graph extraction (production)
│       ├── tokenizer_v1_production.py      # CodeBERT tokenization (production)
│       ├── create_label_index.py           # Scan graphs → label_index.csv
│       ├── fix_labels_from_csv.py          # Patch labels in graph files
│       ├── create_splits.py                # Stratified 70/15/15 split
│       ├── comprehensive_data_validation.py # Full pre-training validation
│       ├── analyze_token_stats.py          # Token length analysis
│       ├── test_gnn_encoder.py             # Unit test: GNNEncoder shapes
│       ├── test_fusion_layer.py            # Unit test: FusionLayer shapes
│       ├── test_dataset.py                 # Unit test: DualPathDataset loading
│       ├── test_dataloader.py              # Unit test: DataLoader batching
│       └── test_sentinel_model.py          # End-to-end: full forward pass
└── BCCC-SCsVul-2024/               # Raw dataset (primary data source)
    ├── BCCC-SCsVul-2024.csv        # 71.6 MB — full features + 12 class columns
    └── SourceCodes/
        ├── CallToUnknown/          # ~11,131 contracts
        ├── DenialOfService/        # ~12,394 contracts
        ├── ExternalBug/            # ~3,604 contracts
        ├── GasException/           # ~6,879 contracts
        ├── IntegerUO/              # ~16,740 contracts
        ├── MishandledException/    # ~5,154 contracts
        ├── NonVulnerable/          # ~26,914 contracts (safe class)
        ├── Reentrancy/             # ~17,698 contracts
        ├── Timestamp/              # ~2,674 contracts
        ├── TransactionOrderDependence/ # ~3,562 contracts
        ├── UnusedReturn/           # ~3,229 contracts
        └── WeakAccessMod/          # ~1,918 contracts
        # NOTE: 41.2% of contracts appear in MULTIPLE folders — the dataset is genuinely
        # multi-label. Current training uses binary labels only (any vuln = 1, safe = 0).
        # See ML_DATASET_PIPELINE.md for details.
```

---

## 5. Data Pipeline — End to End

### 5.1 Raw Data Sources

The primary training data source is **BCCC-SCsVul-2024**:

| Source | Description | Location | Role |
|---|---|---|---|
| **BCCC-SCsVul-2024** | Primary dataset. 111,798 labelled Solidity contracts across 12 vulnerability categories (11 types + NonVulnerable). | `BCCC-SCsVul-2024/` | **Main training data** |
| **SolidiFI benchmark** | Injected vulnerability benchmark | `ml/data/SolidiFI/` | Reference/analysis only |
| **SmartBugs curated** | Curated labeled contracts | `ml/data/smartbugs-curated/` | Reference/analysis only |
| **SmartBugs wild** | Real-world contracts from Ethereum mainnet | `ml/data/smartbugs-wild/` | Reference/analysis only |

**Critical property of BCCC:** 41.2% of the 68,433 unique contracts appear in **multiple**
vulnerability folders simultaneously (up to 9 folders for a single contract). The dataset is
genuinely multi-label. However, the preprocessing pipeline assigns **binary labels only** — one
label per unique hash — losing the multi-label information. See ML_DATASET_PIPELINE.md.

The `BCCC-SCsVul-2024.csv` file (71.6 MB) ships with 68 columns: 56 statistical features plus 12
per-class binary indicators (Class01:ExternalBug through Class12:NonVulnerable). The processed
ground-truth file (`ml/data/processed/contract_labels_correct.csv`) is a 44,442-row subset of this
CSV, with `binary_label` and `class_label` columns added, used to patch the labels in graph files.

### 5.2 Step 1 — Static Analysis with Slither

**Script:** `ml/src/tools/slither_wrapper.py` (and `slither_wrapper_turbo.py`)

Before graph extraction, each contract is run through **Slither**, the leading static analysis framework for Solidity. Slither:
- Detects vulnerability patterns (reentrancy, integer overflow, access control, etc.)
- Exposes the contract's full AST and call graph programmatically
- Classifies findings by impact level (High, Medium, Low)

Results from this analysis are stored in `ml/data/processed/contracts_metadata.parquet` (101,897 rows × 11 columns):

| Column | Description |
|---|---|
| `contract_path` | Path to .sol file |
| `success` | Whether Slither analysis succeeded |
| `detected_version` | Solidity compiler version found |
| `analysis_time` | Seconds taken |
| `high_impact_count` | Number of high-severity findings |
| `medium_impact_count` | Number of medium-severity findings |
| `vulnerability_types` | List of vulnerability categories found |
| `findings` | Full raw Slither findings |

A `StatisticalValidator` (`ml/src/validation/statistical_validation.py`) was applied to the results to identify outliers using **IQR analysis** — contracts with anomalous analysis times, extremely high finding counts, or other statistical anomalies were flagged.

### 5.3 Step 2 — Graph Extraction (AST → PyG Data)

**Script:** `ml/scripts/ast_extractor_v4_production.py`
**Output:** `ml/data/graphs/{hash}.pt` — 68,556 files

This is the most complex step. For each Solidity contract, Slither's programmatic API is used to traverse the contract's structure and build a **heterogeneous graph** where:

- **Nodes** represent contract components
- **Edges** represent relationships between them

#### Node Types (8 types → encoded as integer 0–7)

| ID | Type | Description |
|---|---|---|
| 0 | `STATE_VAR` | State variable declaration |
| 1 | `FUNCTION` | Regular function |
| 2 | `MODIFIER` | Function modifier |
| 3 | `EVENT` | Event declaration |
| 4 | `FALLBACK` | Fallback function |
| 5 | `RECEIVE` | Receive function |
| 6 | `CONSTRUCTOR` | Constructor |
| 7 | `CONTRACT` | Contract-level node |

#### Edge Types (5 types → encoded as integer 0–4)

| ID | Type | Meaning |
|---|---|---|
| 0 | `CALLS` | Function A calls Function B |
| 1 | `READS` | Function reads state variable |
| 2 | `WRITES` | Function writes state variable |
| 3 | `EMITS` | Function emits event |
| 4 | `INHERITS` | Contract inherits from another |

#### Node Features — 8-Dimensional Vector

Each node is represented as an 8-dimensional float vector:

```
[type_id, visibility, is_pure, is_view, is_payable, is_reentrant, complexity, loc]
```

| Feature | Type | Meaning |
|---|---|---|
| `type_id` | float (0–7) | Node type encoding |
| `visibility` | float (0–2) | public/external=0, internal=1, private=2 |
| `is_pure` | 0.0/1.0 | Function has pure modifier |
| `is_view` | 0.0/1.0 | Function has view modifier |
| `is_payable` | 0.0/1.0 | Function is payable |
| `is_reentrant` | 0.0/1.0 | Slither flags this function as reentrant |
| `complexity` | float | Number of CFG nodes in function |
| `loc` | float | Lines of code (from source mapping) |

#### Handling Multiple Solidity Versions

A major engineering challenge was supporting contracts written in Solidity versions 0.4.x through 0.8.x. The `--allow-paths` flag in the Solidity compiler was only introduced in 0.5.0. The extractor detects the compiler version using `solc-select` and conditionally applies the flag:

```python
def solc_supports_allow_paths(version: str) -> bool:
    major, minor, patch = parse_solc_version(version)
    return (major, minor) >= (0, 5)
```

`solc-select` binary paths are resolved inside the Poetry virtualenv at:
`.venv/.solc-select/artifacts/solc-{version}/`

#### Output Format

Each graph is saved as a PyTorch Geometric `Data` object:

```python
Data(
    x=torch.Tensor([N, 8]),          # node feature matrix
    edge_index=torch.LongTensor([2, E]),  # COO edge connectivity
    edge_attr=torch.Tensor([E, 1]),   # edge type
    y=torch.LongTensor([1]),          # binary label (0=safe, 1=vulnerable)
    contract_hash="<md5>",            # for integrity verification
)
```

Saved to `ml/data/graphs/{md5_hash}.pt`.

#### Iteration History

The extractor went through 4 versions before reaching production:
- `v1`, `v2`, `v3` — prototype iterations in `ml/src/data/graphs/`
- `v4` (current) — production-grade, multiprocessing, MD5 naming, checkpoint/resume, bug fixes for older Slither APIs

### 5.4 Step 3 — Tokenization (Source Code → CodeBERT Tokens)

**Script:** `ml/scripts/tokenizer_v1_production.py`
**Output:** `ml/data/tokens/{hash}.pt` — 68,570 files

The raw Solidity source code of each contract is tokenized using the **CodeBERT tokenizer** (`microsoft/codebert-base`) from HuggingFace Transformers.

**Parameters:**
- `max_length = 512` — CodeBERT's maximum sequence length; contracts longer than 512 tokens are truncated
- `padding = "max_length"` — contracts shorter than 512 tokens are padded
- `truncation = True` — long contracts are cut at token 512

**Engineering details:**
- 11 parallel worker processes for speed
- Checkpoint/resume system: progress saved every 500 contracts; the script can be killed and restarted
- CodeBERT tokenizer is initialized **once per worker process** (not once per contract) to avoid repeated model loading overhead
- Input read from `ml/data/processed/contracts_metadata.parquet`

**Output format** (saved as `{hash}.pt`):

```python
{
    'input_ids':      torch.LongTensor([512]),   # token IDs
    'attention_mask': torch.LongTensor([512]),   # 1 for real tokens, 0 for padding
    'contract_hash':  str,                        # for integrity verification
}
```

### 5.5 Step 4 — Label Correction and Index Creation

**Scripts:** `ml/scripts/fix_labels_from_csv.py`, `ml/scripts/create_label_index.py`

During processing it was discovered that some graph files had incorrect labels embedded in their `y` attribute. A dedicated script (`fix_labels_from_csv.py`) patched all graph files using the ground-truth labels from `ml/data/processed/contract_labels_correct.csv`.

After correction, `create_label_index.py` scanned all 68,556 graph files and extracted the `contract_hash → label` mapping into a lightweight 2-column CSV (`ml/data/processed/label_index.csv`, ~3 MB). This file is used by everything downstream — the split script never has to load the full graph files just to get labels.

**Label distribution:**

| Label | Count | Percentage |
|---|---|---|
| 1 — Vulnerable | 44,099 | 64.3% |
| 0 — Safe | 24,456 | 35.7% |
| **Total** | **68,555** | |

The dataset is **class-imbalanced**: there are 1.8× more vulnerable contracts than safe ones. This is why FocalLoss is planned for training — it down-weights easy examples and forces the model to focus on hard cases near the decision boundary.

### 5.6 Step 5 — Stratified Splits

**Script:** `ml/scripts/create_splits.py`
**Output:** `ml/data/splits/*.npy`

The 68,555 paired samples are divided into three non-overlapping splits using **sklearn's `train_test_split` with stratification**. Stratification ensures the vulnerable/safe ratio is preserved in each split.

| Split | Samples | Percentage | File |
|---|---|---|---|
| Train | 47,988 | 70.0% | `train_indices.npy` |
| Validation | 10,283 | 15.0% | `val_indices.npy` |
| Test | 10,284 | 15.0% | `test_indices.npy` |

**Format:** Each file is a 1D `int64` numpy array of position indices — positions in the sorted list of all paired hashes. Not boolean masks. Values range from 0 to 68,554. Random seed: 42.

**Verification built in:**
```python
assert len(train_set & val_set) == 0      # no train/val overlap
assert len(train_set & test_set) == 0     # no train/test overlap
assert len(val_set & test_set) == 0       # no val/test overlap
assert total == 68555                      # full coverage
```

### 5.7 The MD5 Hash System — Critical Design Decision

Every artifact in the pipeline (graph file, token file, label index row) is identified by the same **MD5 hash of the contract's full path**, computed by `ml/src/utils/hash_utils.py`:

```python
def get_contract_hash(contract_path) -> str:
    return hashlib.md5(str(contract_path).encode('utf-8')).hexdigest()
```

This guarantees that for any given contract:
- Its graph is at `ml/data/graphs/{hash}.pt`
- Its tokens are at `ml/data/tokens/{hash}.pt`
- Its label is in `label_index.csv` row where `hash == {hash}`

Pairing in `DualPathDataset` is done by taking the **set intersection** of graph hashes and token hashes. Any contract missing from either side is automatically excluded. Cross-verification at load time (`graph_hash != token_hash → raise ValueError`) catches any data corruption.

---

## 6. Model Architecture — Every Component

### 6.1 GNNEncoder (`ml/src/models/gnn_encoder.py`)

A 3-layer **Graph Attention Network (GAT)** that reads the contract graph and produces a single fixed-size embedding per contract.

```
Input: x [N, 8], edge_index [2, E], batch [N]

Layer 1: GATConv(in=8,  out=8,  heads=8, concat=True, dropout=0.2)
         → [N, 64]   ReLU → Dropout

Layer 2: GATConv(in=64, out=8,  heads=8, concat=True, dropout=0.2)
         → [N, 64]   ReLU → Dropout

Layer 3: GATConv(in=64, out=64, heads=1, concat=False, dropout=0.2)
         → [N, 64]

Global mean pool (collapses N nodes → 1 per graph)
         → [B, 64]
```

**Why GAT over vanilla GCN?** Attention heads learn *which* neighboring nodes are most important. In a contract graph, a function that calls a sensitive state-variable write should attend more strongly to that variable than to unrelated utility functions. Vanilla GCN treats all neighbors equally.

**Why 3 layers?** Each layer expands the receptive field by one hop. With 3 layers, every node can see patterns up to 3 hops away — enough to capture most cross-function interaction patterns in smart contracts, which are typically shallow graphs.

### 6.2 TransformerEncoder (`ml/src/models/transformer_encoder.py`)

A thin wrapper around **microsoft/codebert-base** used as a frozen feature extractor.

```
Input: input_ids [B, 512], attention_mask [B, 512]

CodeBERT (125M parameters, ALL FROZEN — zero gradients)
→ last_hidden_state [B, 512, 768]

CLS token: [:, 0, :]
→ [B, 768]
```

**Why frozen?** CodeBERT was pretrained on millions of code files across multiple languages. Fine-tuning it on 47,988 contracts would take enormous compute and risk catastrophic forgetting. Using it as a frozen extractor gives high-quality semantic embeddings for free. Only the GNN, FusionLayer, and classifier head are trained.

**Why CLS token?** In BERT-style models, the `[CLS]` token at position 0 is specifically designed to aggregate sequence-level information. It is the canonical representation for classification tasks.

**Why `torch.no_grad()` inside forward?** Even though CodeBERT parameters have `requires_grad=False`, wrapping in `no_grad()` prevents PyTorch from building any computation graph through the transformer forward pass, which saves memory proportional to sequence length × batch size.

### 6.3 FusionLayer (`ml/src/models/fusion_layer.py`)

A two-layer MLP that combines the GNN and Transformer representations.

```
Input: gnn_out [B, 64], transformer_out [B, 768]

torch.cat([gnn_out, transformer_out], dim=1)
→ [B, 832]

MLP:
  Linear(832 → 256)   ← large compression step
  ReLU
  Dropout(0.3)         ← regularization
  Linear(256 → 64)    ← final projection
  ReLU

Output: [B, 64]
```

**Why concatenation over addition/cross-attention?** Concatenation is the simplest fusion that preserves all information from both paths. Addition would force the two embeddings to be in the same semantic space, which they are not (GNN lives in a 64-dim space trained from scratch; CodeBERT CLS lives in a 768-dim pretrained space). Cross-attention fusion would be more expressive but requires far more parameters and training data to converge reliably.

**Why 832 → 256 → 64 (not direct 832 → 64)?** The intermediate 256-dim layer allows the MLP to learn cross-modal interactions before projecting down. A direct 832 → 64 compression is too aggressive — it forces the model to discard information before it can learn what to keep.

### 6.4 SentinelModel (`ml/src/models/sentinel_model.py`)

The full model. Orchestrates all three sub-modules.

```python
def forward(graphs, input_ids, attention_mask) -> Tensor[B]:
    gnn_out         = self.gnn(graphs.x, graphs.edge_index, graphs.batch)  # [B, 64]
    transformer_out = self.transformer(input_ids, attention_mask)           # [B, 768]
    fused           = self.fusion(gnn_out, transformer_out)                 # [B, 64]
    scores          = self.classifier(fused).squeeze(1)                     # [B]
    return scores  # values in [0, 1]
```

**Classifier head:**
```python
nn.Sequential(
    nn.Linear(64, 1),
    nn.Sigmoid(),
)
```

**CRITICAL:** The output is already a **probability in [0, 1]**. It is NOT a raw logit. This means:
- Loss function must be `BCELoss`, not `BCEWithLogitsLoss`
- Labels must be cast to `float` before loss computation
- FocalLoss implementation must be BCE-based

**Parameter breakdown:**

| Component | Trainable | Frozen |
|---|---|---|
| GNNEncoder | ~16K | 0 |
| TransformerEncoder | 0 | ~125M |
| FusionLayer | ~222K | 0 |
| Classifier | ~65 | 0 |
| **Total** | **~239,041** | **~125M** |

---

## 7. Dataset and DataLoader Layer

### 7.1 DualPathDataset (`ml/src/datasets/dual_path_dataset.py`)

A standard PyTorch `Dataset` that serves paired (graph, tokens, label) samples.

**Initialization:**
```python
dataset = DualPathDataset(
    graphs_dir="ml/data/graphs",
    tokens_dir="ml/data/tokens",
    indices=train_indices,   # list of int positions
    validate=True,           # loads sample[0] to catch issues early
)
```

**What happens at init:**
1. Glob all `*.pt` files in both directories
2. Extract hash (filename stem) from each file
3. Compute set intersection → paired hashes only
4. Sort the intersection for deterministic ordering across runs
5. Filter to `indices` positions if provided
6. Optionally validate by loading `dataset[0]`

**What `__getitem__` returns:**

```python
(
    graph: torch_geometric.data.Data,    # .x [N,8], .edge_index [2,E], .edge_attr [E,1]
    tokens: dict,                        # {'input_ids': [512], 'attention_mask': [512]}
    label: torch.Tensor,                 # scalar, dtype=long, value 0 or 1
)
```

Labels come from `graph.y`. If the graph has no `y` attribute, `KeyError` is raised.

**Lazy loading:** Files are only opened and deserialized when `__getitem__` is called — not at init time. This means a dataset of 47,988 samples does not load 47,988 files into RAM at startup.

**Integrity check:** After loading both files for a given index, the embedded `contract_hash` field in both the graph and token file are compared:
```python
if graph_hash != token_hash:
    raise ValueError(f"Hash mismatch at index {idx}")
```

### 7.2 dual_path_collate_fn

The DataLoader cannot use the default PyTorch collate function because graphs have variable numbers of nodes — you cannot `torch.stack` graphs with different N. The custom collate function handles this:

```python
def dual_path_collate_fn(batch):
    graphs  = [item[0] for item in batch]
    tokens  = [item[1] for item in batch]
    labels  = [item[2] for item in batch]

    batched_graphs = Batch.from_data_list(graphs)   # PyG merges N variable graphs
    batched_tokens = {
        "input_ids":      torch.stack([t["input_ids"] for t in tokens]),       # [B, 512]
        "attention_mask": torch.stack([t["attention_mask"] for t in tokens]),  # [B, 512]
    }
    batched_labels = torch.stack(labels).squeeze(1)  # [B]

    return batched_graphs, batched_tokens, batched_labels
```

`Batch.from_data_list` merges multiple graphs into one disconnected supergraph and adds a `batch` tensor that maps each node to its original graph index. The GNNEncoder uses this `batch` tensor in `global_mean_pool` to produce one embedding per graph.

**Usage:**
```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=dual_path_collate_fn,
    num_workers=4,
)
```

### 7.3 BCCCDataset (`ml/src/data/bccc_dataset.py`)

A separate, simpler dataset for the CSV-based BCCC features. **Not used in the main dual-path pipeline.** This was built during the initial BCCC EDA phase and handles the 241 hand-crafted numeric features from the BCCC CSV file.

```python
dataset = BCCCDataset(csv_path="BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv")
features, label = dataset[0]
# features: Tensor[241], label: int
```

Labels are multi-class (one-hot across vulnerability categories), converted to integer class indices via `argmax`.

---

## 8. Dependencies and Environment

**Environment manager:** Poetry
**Python:** 3.12.x
**Virtual env:** `ml/.venv/`
**Activate:** `source ml/.venv/bin/activate`

**Core dependencies (all present in `ml/pyproject.toml`):**

| Package | Version | Purpose |
|---|---|---|
| `torch` | ^2.5.0 | Core deep learning |
| `torch-geometric` | ^2.6.0 | Graph neural networks, PyG Batch/Data |
| `transformers` | ^4.45.0 | CodeBERT tokenizer and model |
| `scikit-learn` | ^1.4.0 | Stratified splits, F1-score computation |
| `mlflow` | ^2.17.0 | Experiment tracking (ready, not yet used) |
| `numpy` | ^1.26.0 | Array operations, split files |
| `pandas` | ^2.2.0 | Metadata parquet, label CSV |
| `tqdm` | ^4.66.0 | Progress bars in scripts |
| `loguru` | (implied) | Structured logging in models |
| `slither-analyzer` | (system) | AST extraction (offline, already done) |
| `solc-select` | ^1.2.0 | Multi-version Solidity compiler |

---

## 9. Test Coverage

All tests are in `ml/scripts/test_*.py`. Run them with:
```bash
cd /path/to/sentinel
poetry run python ml/scripts/test_sentinel_model.py
```

| Test Script | What It Tests |
|---|---|
| `test_gnn_encoder.py` | GNNEncoder: output shape [B,64] with fake graph inputs |
| `test_fusion_layer.py` | FusionLayer: output shape [B,64] with fake tensor inputs |
| `test_dataset.py` | DualPathDataset: loads real files, checks shapes and types |
| `test_dataloader.py` | DataLoader + collate_fn: batch shapes, label dtypes |
| `test_sentinel_model.py` | **End-to-end:** real DataLoader batch → full forward pass → shape and range assertions |

The end-to-end test (`test_sentinel_model.py`) specifically verifies:
- No shape errors through the full pipeline
- `scores.shape == labels.shape` (both `[B]`)
- `0.0 ≤ scores ≤ 1.0` (sigmoid is working)
- Ready for loss computation

---

## 10. What Remains to Be Done

### Active: M3.4 Close-Out

1. Sweep `run-more-epochs_best.pt` threshold — compare vs production checkpoint (0.6686)
2. Run final **test-set** evaluation on the winner (10,284 never-touched samples)
3. Smoke test `Predictor` on a real contract from `contracts/`

### Pending Future Work

- **Per-type vulnerability classification:** Upgrade from binary to 12-class. Requires rebuilding
  `graph.y` from `contract_labels_correct.csv` class columns, retraining with CrossEntropyLoss,
  and rebuilding the EZKL circuit. The multi-label ground truth (41.2% of BCCC contracts have
  multiple types) is preserved in the CSV.
- **Error analysis:** Which vulnerability types does the model miss most? Requires correlating
  val-set false negatives back to `class_label` in the CSV.
- **Unfreeze CodeBERT top layers:** Potential F1 improvement; requires ~10× GPU memory.
- **SHAP explainability:** Highlight which lines contributed to the vulnerability score.
- **Drift detection:** Evidently AI on incoming production contracts.

---

## 11. Key Design Decisions

| Decision | Choice Made | Why |
|---|---|---|
| **Graph representation** | PyTorch Geometric `Data` objects | Native batching support via `Batch.from_data_list`; GPU-ready |
| **Graph encoder** | 3-layer GAT with 8 attention heads | Attention mechanism learns which relationships matter most; chosen over original DR-GCN plan |
| **Text encoder** | Frozen CodeBERT CLS token | 125M pretrained params for free; fine-tuning risks catastrophic forgetting |
| **Fusion strategy** | Concatenation + 2-layer MLP | Simpler than planned GMU; preserves both modalities; negligible performance difference at this scale |
| **Output activation** | Sigmoid (in model) | Probability in [0,1]; enables threshold tuning at inference |
| **Classification type** | Binary (vulnerable/safe) | BCCC multi-label simplified to binary during preprocessing; binary answers the primary oracle question; ZK proof simpler with scalar output |
| **Loss function** | Binary FocalLoss (γ=2, α=0.25) | Handles 64%/36% imbalance; alpha=0.25 down-weights vulnerable (majority), 0.75 up-weights safe (minority) |
| **Split strategy** | Stratified 70/15/15 | Preserves 64%/36% class ratio in each split; val and test equally sized |
| **File naming** | MD5 hash of contract path | Guarantees graph/token pairing; different from BCCC's SHA256 content hash |
| **Loading strategy** | Lazy (per `__getitem__`) | 68K files × ~50KB = 3.4GB; cannot fit in RAM; lazy avoids OOM at startup |
| **Label storage** | Binary label in graph `y` | Single source of truth per sample; no CSV lookup at training time; class-level labels stay in CSV only |

---

## 12. Numbers at a Glance

### Dataset (BCCC-SCsVul-2024)

| Metric | Value |
|---|---|
| Raw contracts in BCCC | ~111,798 total files across 12 folders |
| Unique contract hashes in BCCC | 68,433 |
| Single-folder (clean single-label) | 40,267 (58.8%) |
| Multi-folder (genuinely multi-label) | 28,166 (41.2%) |
| Contradictory labels (vuln + NonVulnerable) | 766 |
| Processed CSV rows | 44,442 (after Slither failures dropped ~24K) |

### Training Data

| Metric | Value |
|---|---|
| Paired graph+token samples | 68,555 |
| Graph files | 68,556 |
| Token files | 68,570 |
| Vulnerable contracts (label=1) | 44,099 (64.33%) |
| Safe contracts (label=0) | 24,456 (35.67%) |
| Train split | 47,988 samples |
| Val split | 10,283 samples |
| Test split | 10,284 samples (never touched during training) |

### Model

| Metric | Value |
|---|---|
| Node feature dimensions | 8 |
| Token sequence length | 512 (fixed, padded/truncated) |
| GNNEncoder output | [B, 64] |
| TransformerEncoder output | [B, 768] |
| FusionLayer input/output | [B, 832] → [B, 64] |
| Final model output | [B] ∈ [0, 1] (sigmoid-activated, binary) |
| Trainable parameters | 239,041 |
| Frozen parameters | 124,645,632 (CodeBERT) |

### Training Results

| Metric | Value |
|---|---|
| Batch size | 32 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-2) |
| Loss | Binary FocalLoss (γ=2.0, α=0.25) |
| Checkpoint criterion | F1-macro on val split |
| Production checkpoint | `run-alpha-tune_best.pt` |
| Val F1-macro (production) | **0.6686** |
| Val precision (vulnerable) | 0.7797 (78% of flagged contracts truly vulnerable) |
| Val recall (vulnerable) | 0.7147 (71% of vulnerable contracts caught) |
| Production threshold | 0.50 (swept from 0.30–0.70) |
| Experiment tracker | MLflow (SQLite backend, `mlruns.db`) |
