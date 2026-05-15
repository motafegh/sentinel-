# SENTINEL — Comprehensive Project State Report
**Generated:** February 22, 2026
**Scope:** Full codebase audit + architecture analysis
**Status:** Module 1 ML Core — Milestone M3.2 at 75%

---

## 1. What SENTINEL Is

SENTINEL is a decentralised smart contract security oracle. Its core value proposition is that any user can submit a Solidity contract and receive a verifiable vulnerability score — verifiable meaning a zero-knowledge proof accompanies every result, so the score can be checked on-chain without trusting the operator or revealing the model weights.

The system is designed as six interconnected modules:

| Module | Name | Status |
|--------|------|--------|
| 1 | Deep Learning Intelligence Core (GNN + CodeBERT) | 🔄 In progress — M3.2 at 75% |
| 2 | ZKML — Zero-Knowledge Proof of Inference | ❌ Not started |
| 3 | MLOps — Experiment tracking, versioning, monitoring | ❌ Not started |
| 4 | AI Agents — LangChain, CrewAI, RAG | ❌ Not started |
| 5 | Solidity Contracts — AuditRegistry, SentinelToken | ✅ MVP complete |
| 6 | System Integration — FastAPI, Docker, CI/CD | ❌ Not started |

Only Module 1 (ML core) and Module 5 (contracts) have real code. Everything else is architectural spec.

---

## 2. Where It Started

### Initial Dataset Exploration (January 2026)

The project began with exploration of the **BCCC-SCsVul-2024** dataset — a Kaggle-sourced collection of ~111,897 Solidity contracts with 241 pre-extracted features and one-hot encoded vulnerability labels across multiple classes.

The first approach was straightforward supervised learning on the pre-extracted features:

```
BCCCDataset  →  features tensor [241]  →  label [13 classes, one-hot]
```

This is still present in `ml/src/data/bccc_dataset.py` but has been superseded. It never made it to training — the team pivoted before any model was trained on these pre-extracted features.

**Why pivoted:** The 241 features were too opaque (pre-extracted, source unknown), the multi-class setup required understanding the full label taxonomy, and the approach didn't leverage source code directly. The team decided to build graph and token representations from the raw `.sol` files instead.

### Secondary Dataset Experiments (Late January / Early February 2026)

Two additional datasets were explored in parallel:

- **SolidiFI-benchmark** — a curated vulnerability injection benchmark (~`ml/data/SolidiFI/`)
- **SmartBugs** — a collection of annotated vulnerable contracts (~`ml/data/smartbugs-*`)

Both datasets produced separate pipeline code (`preprocess_solidifi.py`, `solidifi_dataset.py`, `validate_solidifi.py`) and are still present on disk. They were parked — not deleted — as potential secondary training data. The primary pipeline focuses entirely on BCCC.

---

## 3. The Data Pipeline — How It Evolved

The data pipeline went through four major generations before landing on the current production version.

### Generation 1 — Direct Slither (abandoned)
Early scripts ran Slither per-contract and tried to extract features inline. No file naming convention, no checkpointing, no parallelism. These scripts (`extract_ast_features_slither.py`, `enrich_dataset_with_ast.py`) produced output that was immediately thrown away due to naming collisions.

### Generation 2 — Chunked enrichment (abandoned)
`enrich_chunked.py` and `enrich_chunked_safe.py` attempted to process contracts in chunks to recover from crashes. The "safe" variant added basic error handling. Still used path-based naming that created duplicate collisions when the same filename appeared in different directories.

### Generation 3 — AST Extractor v3 (abandoned)
`ast_extractor_v3.py` and `ast_extractor_v3_production.py` introduced a structured graph representation but used the contract filename stem as the output filename. This caused silent collisions: `contract_001.sol` from `Reentrancy/` and `contract_001.sol` from `Timestamp/` would both write to `contract_001.pt`, with the second silently overwriting the first.

**The duplicates and old graphs are still on disk** as `graphs_old_stem_naming/` and `graphs_old_duplicates/`.

### Generation 4 — Production pipeline (current)
**`ast_extractor_v4_production.py`** solved the naming problem by introducing MD5 hashing of the full contract path. Two files with the same name but different directories now get different hashes. The hash function lives in `ml/src/utils/hash_utils.py` and is shared by every pipeline component.

The tokenizer (`tokenizer_v1_production.py`) was built alongside v4 using the same hash function, guaranteeing that `{hash}.pt` in `graphs/` and `{hash}.pt` in `tokens/` always refer to the same contract.

---

## 4. Current Data Assets

All live data is in `ml/data/`:

| Asset | Path | Format | Count | Size |
|-------|------|--------|-------|------|
| **Graph files** | `ml/data/graphs/` | PyG `.pt` | 68,555 | ~GB |
| **Token files** | `ml/data/tokens/` | dict `.pt` | 68,568 | ~GB |
| **Label index** | `ml/data/processed/label_index.csv` | CSV | 68,555 rows | 2.3 MB |
| **Train indices** | `ml/data/splits/train_indices.npy` | NumPy | 47,988 | tiny |
| **Val indices** | `ml/data/splits/val_indices.npy` | NumPy | 10,283 | tiny |
| **Test indices** | `ml/data/splits/test_indices.npy` | NumPy | 10,284 | tiny |

**Total paired samples:** 68,555 (13 unpaired tokens exist and are ignored by the dataset loader)

**Label distribution:** 35.7% safe / 64.3% vulnerable (binary classification)

**Important note on label sources:** 24,113 contracts received their labels from folder structure (the parent directory name e.g. `Reentrancy/`). The remainder came from the BCCC CSV file. Folder-based labels have not been spot-checked against ground truth. This is flagged as a medium-priority concern before full training.

**Split strategy:** 70/15/15 stratified split (sklearn `train_test_split` with `stratify=labels`, seed=42). Stratification preserves the 35.7/64.3 ratio in every split. Verified: no overlap between splits, full coverage of all 68,555 samples.

### Dead data (still on disk, not used)
- `graphs_old_stem_naming/` — Gen 3 graphs with collision-prone naming
- `graphs_old_duplicates/` — Gen 3 graphs with confirmed duplicates
- `graphs_old_backup/` — earlier backup
- `graphs_v4_test/` — 100-contract test run of v4 extractor
- `tokens_test/` — small token test batch
- `SolidiFI/`, `SolidiFI-benchmark/`, `SolidiFI-processed/` — parked secondary dataset
- `smartbugs-curated/`, `smartbugs-results-master/`, `smartbugs-wild/` — parked secondary dataset
- `processed/bccc_full_dataset_results.json` (375MB) — raw Slither output, pre-graph extraction
- `processed/bccc_v04*_only.json` — version-specific intermediate outputs (~303MB total)
- `processed/contracts_ml_ready_csv.parquet` (92MB) — superseded by `contracts_ml_ready_clean.parquet`
- `processed/bccc_full_dataset_results_OLD.json.bak` (43MB) — explicit backup

---

## 5. The Graph Representation — What Each Contract Becomes

Every `.sol` file is parsed by **Slither** (the static analysis framework) and converted to a **PyTorch Geometric `Data` object** with the following schema:

### Nodes
Each node represents a contract element. There are 8 node types:

| ID | Type | Description |
|----|------|-------------|
| 0 | STATE_VAR | State variable declaration |
| 1 | FUNCTION | Regular function |
| 2 | MODIFIER | Function modifier |
| 3 | EVENT | Event declaration |
| 4 | FALLBACK | Fallback function |
| 5 | RECEIVE | Receive function |
| 6 | CONSTRUCTOR | Constructor |
| 7 | CONTRACT | The contract itself |

**Node features are 8-dimensional:**
```
[type_id, visibility, pure, view, payable, is_reentrant, cyclomatic_complexity, lines_of_code]
```

Visibility encoding: public/external=0, internal=1, private=2.

### Edges
Five relationship types are extracted:

| ID | Type | Meaning |
|----|------|---------|
| 0 | CALLS | Function calls another function internally |
| 1 | READS | Function reads a state variable |
| 2 | WRITES | Function writes a state variable |
| 3 | EMITS | Function emits an event |
| 4 | INHERITS | Contract inherits from parent |

### Stored attributes per graph
```python
Data(
    x            = [N, 8],          # node feature matrix
    edge_index   = [2, E],          # COO edge connectivity
    edge_attr    = [E, 1],          # edge type per edge
    y            = [1],             # binary label (0=safe, 1=vulnerable)
    contract_hash   = str,          # 32-char MD5 hash (file pairing key)
    contract_path   = str,          # original .sol path
    contract_name   = str,          # Slither-parsed contract name
    num_nodes    = int,
    num_edges    = int
)
```

### Solidity version handling
The BCCC dataset contains contracts from solc 0.4.x through 0.8.x. Each version requires its own solc binary. The extractor resolves the binary from `solc-select` artifacts stored in `.venv/.solc-select/artifacts/`. The `--allow-paths` flag is only passed for solc ≥ 0.5.0 (introduced in that version — a bug that was discovered and fixed in v4).

---

## 6. The Token Representation — CodeBERT Tokenization

Every `.sol` file is also tokenized with the **CodeBERT tokenizer** (`microsoft/codebert-base`). Each token file is a plain Python dict saved as a `.pt`:

```python
{
    'input_ids':       tensor [512],    # token IDs, padded/truncated to 512
    'attention_mask':  tensor [512],    # 1 for real tokens, 0 for padding
    'contract_hash':   str,             # MD5 hash (matches graph file)
    'contract_path':   str,
    'num_tokens':      int,             # real token count before padding
    'truncated':       bool,            # True if contract exceeded 512 tokens
    'tokenizer_name':  str,             # "microsoft/codebert-base"
    'max_length':      int              # 512
}
```

**Truncation reality:** 96.5% of contracts exceed 512 tokens and are truncated from the end. The decision was to keep this truncation as-is for the MVP baseline — head+tail truncation (first 256 + last 254 tokens) is a known improvement but is parked until after a baseline F1 is established.

The tokenizer runs with 11 parallel workers, each loading the CodeBERT tokenizer once at init time (the `init_worker` pattern — loading per-contract would be catastrophic).

---

## 7. The Dataset Loader — `DualPathDataset`

**File:** `ml/src/datasets/dual_path_dataset.py` (236 lines)

This is the bridge between the raw `.pt` files and the training loop. It is a standard PyTorch `Dataset` with several important design choices:

**Lazy loading:** Files are read in `__getitem__`, not in `__init__`. With 68,555 pairs totalling multiple GB, loading everything into RAM at init would be impossible.

**Hash-based pairing:** The dataset sorts graph hashes and intersects them with token hashes. The 13 unpaired tokens (68,568 - 68,555) are silently dropped. Only paired samples are indexed.

**Split support:** The constructor accepts an `indices` array (e.g. `train_indices.npy`). It maps those positions into the sorted hash list, so a single `DualPathDataset` class handles train, val, and test simply by passing different index arrays.

**Integrity check:** On every `__getitem__`, the `contract_hash` stored inside the graph file is compared against the one inside the token file. A mismatch raises `ValueError` — this catches any data corruption or mis-saved files.

**Verified output shapes (batch_size=32):**
```
graphs.x:         [N_total, 8]     (N_total = sum of nodes across batch)
graphs.batch:     [N_total]        (maps each node to its graph index)
input_ids:        [32, 512]
attention_mask:   [32, 512]
labels:           [32]
```

**Custom collate function:** `dual_path_collate_fn` (defined at module level, not inside the class) handles the heterogeneous batch. PyG graphs are batched with `Batch.from_data_list()`, which merges variable-size graphs into a single disconnected graph. Token tensors are stacked normally. Labels use `.squeeze(1)` not `.squeeze()` — the latter would collapse a size-[1] batch dimension, which breaks batch_size=1.

---

## 8. The Model Architecture — Current State

### GNNEncoder — `ml/src/models/gnn_encoder.py` ✅ VERIFIED

Three-layer Graph Attention Network. Operates on the 8-dim node features extracted by Slither.

```
Input: x [N, 8], edge_index [2, E], batch [N]

Layer 1: GATConv(8 → 8, heads=8, concat=True)   → [N, 64]  + ReLU + Dropout(0.2)
Layer 2: GATConv(64 → 8, heads=8, concat=True)  → [N, 64]  + ReLU + Dropout(0.2)
Layer 3: GATConv(64 → 64, heads=1, concat=False) → [N, 64]
global_mean_pool(x, batch)                       → [B, 64]

Output: graph embeddings [B, 64]
```

**Architecture decisions:**
- 3 layers = 3-hop neighbourhood = most vulnerability patterns covered (reentrancy typically involves 2–3 function calls)
- `heads=8, concat=True` on layers 1–2: eight independent attention mechanisms capture diverse relationship types in parallel
- `heads=1, concat=False` on layer 3: collapses to a single clean 64-dim embedding for fusion
- `global_mean_pool` over all nodes: doesn't require knowing which node is the vulnerability entry point

**Verified:** Output [32, 64] on a real DataLoader batch. Real class imbalance confirmed: 18 vulnerable / 14 safe in a batch of 32.

**Note on dimension divergence from spec:** The architecture document (`SENTINEL-modules.md`) described a DR-GCN with [79 → 256 → 128] dim. The implementation uses GAT with [8 → 64]. This is a deliberate simplification — 8-dim features instead of 79 (the full spec required full AST feature extraction that wasn't completed), and GAT instead of DR-GCN for better attention-based interpretability.

### TransformerEncoder — `ml/src/models/transformer_encoder.py` ✅ VERIFIED

Wraps `microsoft/codebert-base` as a frozen feature extractor.

```
Input: input_ids [B, 512], attention_mask [B, 512]

CodeBERT (12 transformer layers, 768-dim hidden)
  All 199 named parameters frozen (requires_grad=False)
  Forward pass under torch.no_grad()

last_hidden_state[:, 0, :]  →  [CLS] token  →  [B, 768]

Output: semantic embeddings [B, 768]
```

**Architecture decisions:**
- Frozen entirely: 68,555 contracts is too small to fine-tune 125M parameters without catastrophic overfitting. LoRA fine-tuning is parked as a post-baseline improvement.
- `AutoModel` (base), not `AutoModelForSequenceClassification`: the ForSequence variant adds a classification head that would be discarded — we build our own fusion + head.
- `torch.no_grad()` in `forward()`: saves ~40% memory and speeds up the forward pass because no computation graph is built for the frozen encoder.
- CLS token (`[:, 0, :]`): BERT's built-in sequence-level summary, always at position 0.

**Verified:** Output [4, 768] on a test batch. Frozen params: 199. Trainable params: 0.

### FusionLayer — ❌ NOT YET WRITTEN

This is the next file to be created. The specification from the handover is complete:

```
Input: gnn_out [B, 64], transformer_out [B, 768]

torch.cat([gnn_out, transformer_out], dim=1)  →  [B, 832]
Linear(832 → 256) + ReLU + Dropout(0.3)       →  [B, 256]
Linear(256 → 64)  + ReLU                      →  [B, 64]

Output: fused embedding [B, 64]
```

Target file: `ml/src/models/fusion_layer.py`

**Architecture decisions:**
- Simple concat + MLP, not Gated Multimodal Unit (GMU): GMU is the stretch goal from the architecture spec. Concat+MLP is the MVP — establish baseline F1 first.
- Output 64-dim: matches GNN output dimension; keeps the classification head trivially simple.
- Dropout(0.3) after first linear only: regularisation where the dimensionality reduction is sharpest.

### SentinelModel — ❌ NOT YET WRITTEN

The top-level wrapper that wires all three encoders together plus the classification head.

```
Input: graph (PyG Data), tokens (dict)

GNNEncoder(graph.x, graph.edge_index, graph.batch)  →  [B, 64]
TransformerEncoder(tokens['input_ids'], tokens['attention_mask'])  →  [B, 768]
FusionLayer(gnn_out, transformer_out)  →  [B, 64]
Linear(64 → 1) + Sigmoid  →  [B]  (risk score 0–1)

Output: vulnerability probability per contract [B]
```

Target file: `ml/src/models/sentinel_model.py`

**Training target:** Binary cross-entropy with Focal Loss (γ=2, α=0.25), AdamW lr=1e-4. F1-macro ≥ 85%.

---

## 9. The Solidity Contracts — Module 5

Two contracts are implemented and compiled with Foundry.

### SentinelToken (`contracts/src/SentinelToken.sol`)
Standard ERC20 governance token (symbol: SENT).

- **Extensions:** ERC20Permit (EIP-2612 gasless approvals) + ERC20Votes (delegation + governance checkpointing)
- **Supply:** 1,000,000,000 SENT minted to deployer at construction
- **Use:** Agents must stake SENT to submit audits; voting power for future governance
- **C3 linearisation override:** `_update()` and `nonces()` both have diamond inheritance conflicts resolved correctly

### AuditRegistry (`contracts/src/AuditRegistry.sol`)
The core on-chain registry. Currently at v1.1.

- **Pattern:** UUPS upgradeable (OpenZeppelin) — `_authorizeUpgrade` restricted to `onlyOwner`
- **Storage gap:** `uint256[43] private __gap` — reserves upgrade space (50 slots - 7 used = 43 gap)
- **Staking:** Agents call `stake(amount)` using SafeERC20 (`safeTransferFrom`). Min stake enforced on audit submission.
- **Audit submission:** `submitAudit(contractAddress, riskScore, vulnerabilities[], zkProof)`. Validates score ≤ 100, stake ≥ minStake, then calls `_verifyProof()`.
- **ZK verification:** `_verifyProof()` is currently a **placeholder** — it returns `true` if `zkProof.length > 0`. Real ZK verification requires the EZKL-generated `ZKMLVerifier.sol` contract which depends on Module 2 being complete.
- **Vulnerability taxonomy:** 13-type enum matching the ML model's output classes (Reentrancy, IntegerOverflow, AccessControl, etc.)
- **SafeERC20:** Added in v1.1 — handles tokens that don't return bool from `transfer()` (e.g. USDT)
- **CEI pattern:** All functions follow Checks → Effects → Interactions for reentrancy safety

**Tests:** Unit tests exist for both contracts in `contracts/test/`. The test suite covers happy paths and basic revert conditions. Fuzz and invariant tests (as planned in the architecture spec) are not yet written.

---

## 10. The Technology Stack

### Python / ML Environment
```
Runtime:          Python 3.12.1 (pinned by pyproject.toml)
Package manager:  Poetry (ml/pyproject.toml)
Virtual env:      ml/.venv (not at project root)

Deep Learning:    PyTorch 2.5+
Graph Learning:   PyTorch Geometric 2.6+
Transformers:     HuggingFace Transformers 4.45+, Tokenizers 0.20+
Static Analysis:  Slither (installed separately, runs via subprocess in extractor)
Compiler mgmt:    solc-select (multiple solc versions in .venv/.solc-select/)
Data:             Pandas 2.2+, NumPy 1.26+, scikit-learn 1.4+, PyArrow
MLOps (ready):    MLflow 2.17+ (installed, not yet configured)
Visualisation:    Matplotlib, Seaborn
```

### Solidity / Contracts Environment
```
Compiler:   solc ^0.8.20
Framework:  Foundry (forge)
Libraries:  OpenZeppelin Contracts + OpenZeppelin Upgradeable Contracts
Testing:    forge test (unit tests complete, fuzz/invariant pending)
```

### Infrastructure
```
OS:     Linux (WSL2 on Windows)
Shell:  zsh
IDE:    VSCode with Claude Code extension
Git:    Main branch only so far
```

---

## 11. The File Structure — What's Real vs Noise

### Active source files (the real codebase)

```
sentinel/
├── Project-docs/
│   ├── SENTINEL-architecture.md     ← System design reference
│   ├── SENTINEL-modules.md          ← Per-module technical specs
│   ├── Module 3, Session 3.md       ← Last handover (Feb 20, 2026)
│   └── SENTINEL-STATE-REPORT.md     ← This document
│
├── contracts/
│   ├── src/
│   │   ├── AuditRegistry.sol        ← Core registry contract (v1.1)
│   │   └── SentinelToken.sol        ← SENT governance token
│   ├── test/
│   │   ├── AuditRegistry.t.sol      ← Unit tests
│   │   └── SentinelToken.t.sol      ← Unit tests
│   └── foundry.toml
│
└── ml/
    ├── pyproject.toml               ← Poetry config + all dependencies
    │
    ├── src/
    │   ├── datasets/
    │   │   └── dual_path_dataset.py  ← DualPathDataset + collate_fn ✅
    │   ├── models/
    │   │   ├── gnn_encoder.py        ← 3×GAT → [B, 64] ✅ VERIFIED
    │   │   ├── transformer_encoder.py← Frozen CodeBERT → [B, 768] ✅ VERIFIED
    │   │   ├── fusion_layer.py       ← ❌ NOT YET WRITTEN
    │   │   └── sentinel_model.py     ← ❌ NOT YET WRITTEN
    │   ├── tools/
    │   │   └── slither_wrapper.py    ← Full Slither API wrapper (63KB)
    │   ├── utils/
    │   │   └── hash_utils.py         ← MD5 hash functions (shared by pipeline)
    │   └── validation/
    │       ├── models_v2.py          ← Pydantic validation models
    │       └── statistical_validation.py
    │
    ├── scripts/
    │   ├── ast_extractor_v4_production.py ← Graph extraction (DONE, ran once)
    │   ├── tokenizer_v1_production.py     ← Token extraction (DONE, ran once)
    │   ├── create_splits.py               ← Train/val/test split generation
    │   ├── create_label_index.py          ← label_index.csv generation
    │   ├── comprehensive_data_validation.py← Full dataset validation
    │   ├── train_baseline_rf.py           ← RF baseline (not yet run)
    │   ├── test_dataset.py                ← DualPathDataset shape tests ✅
    │   ├── test_gnn_encoder.py            ← GNN end-to-end shape test ✅
    │   └── test_dataloader.py             ← Collate fn shape test ✅
    │
    └── data/
        ├── graphs/         ← 68,555 graph .pt files  [LIVE]
        ├── tokens/         ← 68,568 token .pt files  [LIVE]
        ├── splits/         ← train/val/test .npy     [LIVE]
        └── processed/
            └── label_index.csv  ← 68,555 hash→label pairs [LIVE]
```

### Files that exist but are no longer needed

**Root level (should be in ml/ or removed):**
- `ast_extractor.py` — superseded by `ml/scripts/ast_extractor_v4_production.py`
- `eda_analysis.py`, `comprehensive_multiclass_eda.py` — EDA phase complete
- `BCCC-SCsVul-2024_*.md` — EDA reports, phase complete
- `run_full_dataset_overnight.py` — one-off overnight run, done
- `test_fixed_export.py`, `test_one_chunk.py` — temporary tests
- `validate_project_state.py`, `validation_full_output.txt` — one-off validation
- `slither_output.json` — one-off test output
- `graphstatus.py`, `check_status.sh`, `status.sh` — status checking scripts
- `graph_extraction.pid`, `graph_extraction_final.pid` — stale PID files
- `src/hello_world.py` — placeholder, never deleted
- `code` — unknown temporary file
- `pyproject.toml` + `poetry.lock` (root) — shadow of ml/pyproject.toml

**ml/scripts/ — one-off scripts, all completed:**
- `explore_bccc.py` — empty (0 bytes)
- `explore_csv.py`, `DiagnosetheDuplicates.py`
- `analyze_compilation_errors.py`, `analyze_enrichment.py`, `analyze_token_stats.py`
- `categorize_all_errors.py`, `investigate_reentrancy_failures.py`, `quick_category_analysis.py`
- `split_by_version.py`, `fix_merge_dataset.py`, `fix_labels_from_csv.py`
- `enrich_dataset_with_ast_old.py`, `enrich_dataset_with_ast.py` (superseded by v4)
- `enrich_chunked.py` (superseded by enrich_chunked_safe.py, which is itself done)
- `ast_extractor_v3.py`, `ast_extractor_v3_production.py` (superseded by v4)
- `extract_ast_features_slither.py` (superseded by v2, then v4)
- `build_graph_dataset.py` — replaced by v4
- `test_folder_structure.py`, `test_preprocessing.py` — early sanity checks, done
- `test_slither_wrapper.py`, `test_slither_bccc.py`, `test_ast_pipeline.py` — early tests, done

**ml/src/ — old/backup files:**
- `src/tools/slither_wrapper_backup_20260206_160828.py` — explicit dated backup
- `src/tools/slither_wrapper_turbo.py` — experimental variant, not used in pipeline
- `src/validation/models.py` — 1KB stub, superseded by `models_v2.py` (14KB)
- `src/validation/test_full_dataset_final.py`, `test_models.py`, `test_real_data.py` — test scripts incorrectly placed inside `src/`
- `src/data/validate_dataset.py`, `validate_solidifi.py` — superseded by `src/validation/`
- `src/data/solidifi_dataset.py` — SolidiFI path parked
- `src/data/bccc_dataset.py` — original feature-based approach, not used

**ml/data/ — large dead data:**
- `graphs_old_stem_naming/`, `graphs_old_duplicates/`, `graphs_old_backup/`
- `graphs_v4_test/`, `tokens_test/`
- `SolidiFI/`, `SolidiFI-benchmark/`, `SolidiFI-processed/`
- `smartbugs-curated/`, `smartbugs-results-master/`, `smartbugs-wild/`
- `processed/bccc_full_dataset_results.json` (375MB)
- `processed/bccc_full_dataset_results_OLD.json.bak` (43MB)
- `processed/bccc_v04*_only.json` files (~303MB)
- `processed/contracts_ml_ready_csv.parquet` (92MB)
- `processed/test_enriched.json`, `processed/test_fix_100.json`

---

## 12. Open Issues Before Training

| Priority | Issue | Detail |
|----------|-------|--------|
| 🔴 HIGH | FusionLayer not written | `fusion_layer.py` must be created before SentinelModel |
| 🔴 HIGH | SentinelModel not written | `sentinel_model.py` must be created to close M3.2 |
| 🟡 MEDIUM | Label verification | 24,113 folder-based labels unverified — spot-check 50 samples |
| 🟡 MEDIUM | Training loop not started | `train.py` with Focal Loss + AdamW + MLflow needed for M3.3 |
| 🟢 LOW | DVC not set up | graphs/tokens/splits unversioned — reproducibility risk |
| 🟢 LOW | MLflow not configured | needed before training experiments begin |

---

## 13. What's Parked (Conscious Decisions to Not Do Yet)

These are explicitly deferred, not forgotten:

| Feature | Where it goes | Condition for revival |
|---------|---------------|-----------------------|
| Head+tail CodeBERT truncation | TransformerEncoder | After baseline F1 established |
| LoRA fine-tuning of CodeBERT | TransformerEncoder | After baseline F1 established |
| GMU (Gated Multimodal Unit) | FusionLayer → stretch goal | After simple fusion baseline |
| 79-dim node features | GNNEncoder | Currently 8-dim; full feature set deferred |
| Multi-label classification (13 classes) | SentinelModel head | Currently binary; multi-label is stretch |
| Optuna hyperparameter search | training config | After first training run |
| Continual learning (EWC + replay) | separate module | After stable model |
| Module 2 (ZKML) | zkml/ | After Module 1 model trained |
| Module 3 (MLOps) | mlops/ | After Module 1 first training run |
| Module 4 (Agents) | agents/ | After Module 1 inference API exists |
| Module 6 (Integration) | api/, docker-compose | After Modules 1–4 working |

---

## 14. The Immediate Roadmap (M3.2 → M3.3)

### Close M3.2 (this session)
1. **Write `fusion_layer.py`** — concat + MLP, output [B, 64], verify shape
2. **Write `sentinel_model.py`** — wire GNN + Transformer + Fusion + head, verify [B] output
3. **Write `test_sentinel_model.py`** — end-to-end on real DataLoader batch; confirm gradients flow through GNN and head only (not frozen BERT)

### Open M3.3 (next session)
4. **Spot-check 50 folder-based labels** — confirm label quality before spending GPU hours
5. **Set up MLflow** — `mlflow server --host 0.0.0.0 --port 5000`; wrap training in `mlflow.start_run()`
6. **Write `train.py`** — Focal Loss (γ=2, α=0.25), AdamW lr=1e-4, train loop with val F1 tracking
7. **First training run** — target F1-macro ≥ 85% on val set

---

*End of report. Generated by full codebase audit — February 22, 2026.*
