# SENTINEL Data Pipeline — Ground-Truth Audit

**Date:** 2026-05-16  
**Approach:** No trust in written docs. Verified everything from source code and actual files.  
**Goal:** Understand exactly what data the model is trained on, from raw BCCC files to graph tensors.

---

## Audit Plan

| Stage | Topic | Status |
|-------|-------|--------|
| 1 | BCCC Dataset Origin | ✅ DONE |
| 2 | Label Assignment | ✅ DONE |
| 3 | Deduplication | ✅ DONE |
| 4 | Graph Extraction | ✅ DONE |
| 5 | Token Extraction | ✅ DONE |
| 6 | Dataset Construction | ✅ DONE |
| 7 | Splits | ✅ DONE |
| 8 | Augmented Data (CEI) | ✅ DONE |
| 9 | What the Model Sees | ✅ DONE |
| 10 | Known Problems | ✅ DONE |

---

## Stage 1 — BCCC Dataset Origin

### 1.1 What is BCCC-SCsVul-2024?

`BCCC-SCsVul-2024.csv` has **111,898 rows** — one row per (vulnerability_class, contract) occurrence. This means the same contract file can appear in multiple rows if it was labelled under multiple vulnerability classes.

```
BCCC-SCsVul-2024/
  BCCC-SCsVul-2024.csv          ← 111,898 rows, 12+ columns
  SourceCodes/
    CallToUnknown/    ← .sol files named by SHA256 of their content
    DenialOfService/
    ExternalBug/
    GasException/
    IntegerUO/
    MishandledException/
    NonVulnerable/
    Reentrancy/
    Timestamp/
    TOD/
    UnusedReturn/
    WeakAccessMod/
```

Each `.sol` filename IS its SHA256 content hash. Identical contracts across directories share the same filename.

### 1.2 Cross-directory duplication (root cause of label co-occurrence)

The same physical `.sol` file appears in multiple class directories. From audit (verified in previous session):

- ~68K unique SHA256 contracts across ~111K directory entries
- **41.2% of unique contracts appear in 2+ class directories**

This is not a bug — it's BCCC's design. A single contract can legitimately demonstrate multiple vulnerability types. However it means any model trained on BCCC will see highly correlated labels (see Stage 10).

### 1.3 / 1.5 / 1.6 — NonVulnerable and WeakAccessMod

**NonVulnerable**: Contracts with no vulnerability label. They exist in `SourceCodes/NonVulnerable/` and are extracted to graphs. They appear in `multilabel_index.csv` with all-zero label vectors. Not excluded — they ARE part of training (as "safe" examples, ~55% of training set).

**WeakAccessMod**: 1,918 `.sol` files in BCCC but **zero were extracted** to `.pt` files in the original graph extraction pass. `build_multilabel_index.py` explicitly excludes it:
> "Including a class with zero training examples would produce undefined gradients and a permanently near-zero output node."

Current output is 10 classes (WeakAccessMod excluded). If extracted in future, append as index 9.

---

## Stage 2 — Label Assignment

**Script:** `ml/scripts/build_multilabel_index.py`

### How labels are assigned

Two hash systems — never mix:
- **BCCC SHA256**: hash of file CONTENTS → `.sol` filename in BCCC SourceCodes/
- **Internal MD5**: hash of file PATH → `.pt` filename in `ml/data/graphs/`

**Bridge:** `graph.contract_path` inside each `.pt` → `Path(contract_path).stem` = SHA256

The pipeline:
1. Load BCCC CSV → group by SHA256, `max()` (OR) per class column → `{sha256: [0/1,×10]}` dict
2. For each `.pt` file: read `graph.contract_path` → extract SHA256 stem → lookup dict → multi-hot vector
3. Write `ml/data/processed/multilabel_index.csv` (68,523 rows pre-dedup)

### ⚠️ CRITICAL BUG: contract_path not stored in current graphs

**Verified:** Current graph `.pt` files have keys:
```
['edge_attr', 'node_metadata', 'num_edges', 'contract_name', 'edge_index', 'x', 'num_nodes']
```
No `contract_path`. The `graph_extractor.py` docstring explicitly says:
> "Caller-specific metadata (.contract_hash, .contract_path, .y) is NOT set here; each caller attaches its own values after the call returns."

But `reextract_graphs.py` (v5.1 Phase 1) **never sets `contract_path`** after extraction. If `build_multilabel_index.py` is run today, `getattr(graph, "contract_path", "")` returns `""` → all labels = `[0]*10` (all-safe).

**Current state is valid** because `multilabel_index.csv` was built from the ORIGINAL graphs (v4/v5.0 extraction) that DID have `contract_path`. The deduped CSV is correct. But running `build_multilabel_index.py` again would silently corrupt all labels.

**Fix needed in `reextract_graphs.py`:** add `g.contract_path = str(sol_path.relative_to(PROJECT_ROOT))` before `torch.save(g, tmp)`.

### Class ordering (training vector indices 0-9)

```python
CLASS_NAMES = [
    "CallToUnknown",              # 0
    "DenialOfService",            # 1
    "ExternalBug",                # 2
    "GasException",               # 3
    "IntegerUO",                  # 4
    "MishandledException",        # 5
    "Reentrancy",                 # 6
    "Timestamp",                  # 7
    "TransactionOrderDependence", # 8
    "UnusedReturn",               # 9
]
```

---

## Stage 3 — Deduplication

**Script:** `ml/scripts/dedup_multilabel_index.py`

### What counts as a duplicate?

Two `.pt` files with **identical `.sol` source content** (content-MD5 via `get_contract_hash_from_content()`) are duplicates, regardless of path. This captures the case where BCCC stores the same `.sol` in multiple class directories — each gets a different path-MD5 and thus a different `.pt` file.

### How duplicates are resolved

For each content-MD5 group:
1. OR all label vectors (union of all class memberships)
2. Keep the alphabetically-first path-MD5 as the canonical row
3. Discard the rest

### Before/after

| | Rows |
|--|------|
| `multilabel_index.csv` (original) | 68,523 |
| `multilabel_index_deduped.csv` (deduped) | 44,420 |
| CEI augmented rows added (indices 44420-44469) | +50 |
| **Total in deduped CSV** | **44,470** |

**35.2% reduction** (24,103 rows removed). These were content-duplicates — same Solidity code appearing under multiple BCCC class directories.

### Does deduplication fix cross-split leakage?

**Yes, completely.** Verified:
- `unique md5_stems` = 44,470 = total rows → 0 path-MD5 duplicates
- Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0

The original 68K dataset had 34.9% of content groups spanning multiple splits. After dedup: zero.

---

## Stage 4 — Graph Extraction

### 4.1 The 12 node features (NODE_FEATURE_DIM=12, schema v2/v3)

Verified in `ml/src/preprocessing/graph_schema.py`:

| Index | Name | Type/Range | Notes |
|-------|------|-----------|-------|
| 0 | type_id | float, type_id/12.0 | NODE_TYPE / _MAX_TYPE_ID(=12) |
| 1 | visibility | int {0=priv/int, 1=pub, 2=ext, 3=pub+ext} | function visibility |
| 2 | pure | bool {0,1} | |
| 3 | view | bool {0,1} | |
| 4 | payable | bool {0,1} | |
| 5 | complexity | int (cyclomatic) | 0 for non-function nodes |
| 6 | loc | int (lines of code) | source_mapping.lines length |
| 7 | return_ignored | float {-1.0=N/A, 0=checked, 1=ignored} | -1.0 for non-call nodes |
| 8 | call_target_typed | float {-1.0=N/A, 0=untyped/addr, 1=typed} | -1.0 for non-call nodes |
| 9 | in_unchecked | bool {0,1} | whether inside `unchecked {}` block |
| 10 | has_loop | bool {0,1} | function contains a loop |
| 11 | external_call_count | int | number of external calls in function |

### 4.2 Node type ordering (verified from code)

```python
NODE_TYPES = {
    "STATE_VAR":      0,   # normalized = 0/12 = 0.000
    "FUNCTION":       1,   # normalized = 1/12 = 0.083
    "MODIFIER":       2,   # normalized = 2/12 = 0.167
    "EVENT":          3,   # normalized = 3/12 = 0.250
    "FALLBACK":       4,   # normalized = 4/12 = 0.333
    "RECEIVE":        5,   # normalized = 5/12 = 0.417
    "CONSTRUCTOR":    6,   # normalized = 6/12 = 0.500
    "CONTRACT":       7,   # normalized = 7/12 = 0.583
    "CFG_NODE_CALL":  8,   # normalized = 8/12 = 0.667
    "CFG_NODE_WRITE": 9,   # normalized = 9/12 = 0.750
    "CFG_NODE_READ":  10,  # normalized = 10/12 = 0.833
    "CFG_NODE_CHECK": 11,  # normalized = 11/12 = 0.917
    "CFG_NODE_OTHER": 12,  # normalized = 12/12 = 1.000
}
```

**Node 0 is always the CONTRACT node** (feat[0]=0.583). Then STATE_VARs. Then FUNCTION+CFG nodes interleaved (each function followed immediately by its CFG children).

### 4.3 Edge types (verified from code)

```python
EDGE_TYPES = {
    "CALLS":            0,  # function → called function
    "READS":            1,  # CFG_NODE_READ → STATE_VAR
    "WRITES":           2,  # CFG_NODE_WRITE → STATE_VAR
    "EMITS":            3,  # function → EVENT
    "INHERITS":         4,  # contract → parent contract
    "CONTAINS":         5,  # function → its CFG_NODE children
    "CONTROL_FLOW":     6,  # CFG_NODE → successor CFG_NODE
    "REVERSE_CONTAINS": 7,  # CFG_NODE → parent function (RUNTIME ONLY — not in .pt files)
}
```

**REVERSE_CONTAINS (type 7) is injected at runtime** in `trainer.py` during each batch, not stored in `.pt` files.

### 4.4 Actual graph structure (observed from 9 real graphs)

Typical ranges:
- Nodes: 40–250 (median ~130)
- Edges: 60–430 (median ~200)
- Dominant edge types: CONTAINS (43–50%), CONTROL_FLOW (35–45%), READS+WRITES (10–20%)
- Dominant node types: CFG_NODE_OTHER (30–45%), FUNCTION (10–25%), CFG_NODE_WRITE/READ (5–15%)

### 4.5 Structural differences: Reentrancy vs Safe (sample n=100 each)

| Metric | Reentrancy | Safe | Ratio |
|--------|-----------|------|-------|
| n_nodes | 111 | 112 | 0.99× |
| n_edges | 184 | 184 | 1.00× |
| n_functions | 15.8 | 16.6 | 0.95× |
| **payable_fns** | **1.03** | **0.31** | **3.32×** |
| **has_loop_fns** | **0.55** | **0.32** | **1.72×** |
| **ret_ignored_fns** | **0.95** | **0.37** | **2.57×** |
| ext_calls_total | 0.91 | 0.90 | 1.01× |
| mean_complexity | 4.38 | 4.29 | 1.02× |

**Overall topology is nearly identical.** Distinguishing signals are in node features (payable, return_ignored, has_loop), NOT in graph structure. The GNN must learn from feature-weighted message passing, not from topological patterns.

`ext_calls_total` being identical (1.01×) is notable — one would expect Reentrancy to have more external calls, but safe ERC-20 tokens also have many external calls. This is why co-occurrence is so high.

### 4.6 Ghost graphs

66 total ghost graphs (num_nodes ≤ 3) post-extraction — 0.1% of dataset. Gate passed.

### ⚠️ BUG: node_metadata type field is systematically wrong in current graphs

**Root cause:** In `graph_extractor.py`, `_add_node()` computes `actual_type_name` for metadata by:
```python
# OLD (before fix a0576fb, 2026-05-12 21:19):
actual_type_id = int(x_list[-1][0])    # int(0.583) = 0 → always STATE_VAR!

# NEW (after fix):
actual_type_id = int(round(x_list[-1][0] * 12))  # int(round(0.583*12)) = 7 → CONTRACT ✅
```

Graphs on disk were extracted at **16:24 on 2026-05-12** (v5.1 Phase 1). The fix was committed at **21:19 on 2026-05-12**. All graphs were extracted with the old code.

**Effect:** `node_metadata[i]['type']` is "STATE_VAR" for ALL nodes (not just state vars). Contract node (node 0) has `feat[0]=0.583` (correct, CONTRACT) but `metadata.type="STATE_VAR"` (wrong).

**Impact on training:** ZERO. The model sees only `x` tensors — `node_metadata` is never used during training or inference. The feature vector is correct.

**Impact on debugging:** Misleading. Any tool that reads `metadata.type` to display node types will show "STATE_VAR" for everything.

**Fix:** Re-run `reextract_graphs.py` with current code. Not urgent unless debugging tools rely on metadata.

---

## Stage 5 — Token Extraction

**Script:** `ml/scripts/tokenizer_v1_production.py` (original) / `ml/scripts/extract_augmented.py` (CEI)

### Token .pt file structure

```
{
    'input_ids':              torch.int64 [512]   # CodeBERT token IDs, padded to 512
    'attention_mask':         torch.int64 [512]   # 1=real, 0=padding
    'contract_hash':          str                 # path-MD5 (matches .pt filename)
    'contract_path':          str                 # BCCC/SourceCodes/<class>/<sha256>.sol
    'num_tokens':             int                 # actual tokens before padding
    'truncated':              bool                # True if source > 512 tokens
    'tokenizer_name':         str                 # "microsoft/codebert-base"
    'max_length':             int                 # 512
    'feature_schema_version': str                 # "v1"
}
```

Note: token files DO have `contract_path` (unlike graph files). Token extraction was run from a different script that correctly sets it.

### Truncation

Sample of 5,000 token files:
- **Truncated (hit 512 limit): 96.6%** — almost every contract is truncated
- Full (< 512 tokens): 3.4%

The CodeBERT encoder sees at most the first ~512 tokens of each contract. For a typical Solidity contract with 200–500 lines, this captures roughly 30–60% of the source. Functions near the end of the file are invisible to the text encoder.

**This is a fundamental limitation:** the GNN path has full structural coverage; the CodeBERT path has severe truncation bias toward the start of each file.

### Token count: 44,470 files

Exactly matches graph count. All graphs have a paired token file.

---

## Stage 6 — Dataset Construction (DualPathDataset)

**Script:** `ml/src/datasets/dual_path_dataset.py`

### Pairing mechanism

Files are paired by **MD5 stem** (path-MD5). `graph_hashes ∩ token_hashes` = paired set.

If a graph has no token (or vice versa), it's silently skipped with a warning log.

### What `__getitem__` returns

```
(graph: Data,          # PyG Data object with x[N,12], edge_index[2,E], edge_attr[E]
 input_ids: [512],     # int64
 attention_mask: [512],# int64
 labels: [10])         # float32, 0/1 multi-hot
```

The REVERSE_CONTAINS edges (type 7) are added to `graph.edge_attr` during batch collation in `trainer.py`, not here.

### Cache

`ml/data/cached_dataset_deduped.pkl` — 1.42 GB. Stores all 44,470 (graph, tokens) pairs in RAM. Pre-loading eliminates disk I/O during training.

---

## Stage 7 — Splits

**Script:** `ml/scripts/dedup_multilabel_index.py` (splits are built inside the dedup script)

### Split method

Stratified by **label-count bucket** (0 labels = safe, 1 label, 2 labels, 3+ labels), seed=42. Proportions: 70% train / 15% val / 15% test.

**Split files:** `ml/data/splits/deduped/` — numpy index arrays (not hash lists).

### Verified split sizes

| Split | Size | Safe | % Safe |
|-------|------|------|--------|
| Train | 31,142 | 17,141 | 55.0% |
| Val | 6,661 | 3,667 | 55.1% |
| Test | 6,667 | 3,669 | 55.1% |
| **Total** | **44,470** | **24,477** | **55.0%** |

### Cross-split leakage

| Check | Result |
|-------|--------|
| Train ∩ Val | **0** |
| Train ∩ Test | **0** |
| Val ∩ Test | **0** |

No leakage. All splits are disjoint by index.

### Per-class stratification quality (% of total class positives in each split)

All classes achieve ~68–72% train, ~14–16% val/test. DoS shows slightly more variance (68%/14%/18%) due to very small sample size (377 total).

---

## Stage 8 — Augmented Data (CEI Pairs)

**50 CEI contracts** added at indices 44420–44469 in `multilabel_index_deduped.csv`:
- 25 with `Reentrancy=1` (Check-Effects-Interactions violations)
- 25 with all-zero labels (safe CEI pattern)

**All 50 are in the training split** (confirmed — none in val or test).

The safe CEI contracts are structurally similar to real BCCC Reentrancy contracts (payable functions, external calls present) but with correct CEI ordering. This provides contrastive signal for the Reentrancy class.

---

## Stage 9 — What the Model Actually Sees

### REVERSE_CONTAINS injection

At runtime, for each batch, `trainer.py` adds REVERSE_CONTAINS edges (type 7) between each CFG node and its parent function. This is done by reversing all CONTAINS edges (type 5):

```python
# For each (func_node → cfg_node) CONTAINS edge, add (cfg_node → func_node) REVERSE_CONTAINS
```

This is why `NUM_EDGE_TYPES=8` and `edge_emb = Embedding(8, 32)` — the embedding table must include type 7 even though `.pt` files only store types 0–6.

### A complete batch

Each training batch from the DataLoader produces:
```
batch.x:            [N_total, 12]   float32  — all nodes in batch concatenated
batch.edge_index:   [2, E_total]    int64    — + REVERSE_CONTAINS added
batch.edge_attr:    [E_total]       int64    — edge type IDs (0-7)
batch.batch:        [N_total]       int64    — which graph each node belongs to
input_ids:          [B, 512]        int64    — CodeBERT tokens
attention_mask:     [B, 512]        int64    — CodeBERT mask
labels:             [B, 10]         float32  — multi-hot targets
```

### Feature value ranges (verified)

- `feat[0]` (type_id): {0.000, 0.083, 0.167, 0.250, 0.333, 0.417, 0.500, 0.583, 0.667, 0.750, 0.833, 0.917, 1.000}
- `feat[6]` (loc): ranges from 1 to 500+ (raw count, not normalized) — **POTENTIAL DOMINANCE ISSUE**
- `feat[5]` (complexity): 0–50+ (raw count, not normalized) — **POTENTIAL DOMINANCE ISSUE**
- `feat[11]` (ext_calls): 0–20+ (raw count, not normalized)
- Binary features (1–4, 7–10): {0, 1} or {-1, 0, 1}

`feat[6]` (LoC) with values up to 500+ is 3–4 orders of magnitude larger than the binary features. This CAN dominate the first linear layer dot product unless the GNN's initial weight matrix is calibrated accordingly. The model learns weights so this may self-correct, but it is a known risk.

---

## Stage 10 — Known Problems (Quantified)

### 10.1 Co-occurrence matrix (training set, n=31,142)

Percentage of row-class contracts that ALSO have col-class label:

|  | CTU | DoS | ExtB | GasE | IntUO | MisH | Rentr | Time | TOD | UnuR |
|--|-----|-----|------|------|-------|------|-------|------|-----|------|
| **CallToUnknown** [2502] | — | 0% | 13% | 26% | 70% | 14% | 42% | 7% | 14% | 9% |
| **DenialOfService** [257] | 0% | — | 1% | 64% | 0% | 2% | **99%** | 0% | 0% | 0% |
| **ExternalBug** [2422] | 13% | 0% | — | 17% | 67% | 24% | 47% | 8% | 21% | 41% |
| **GasException** [3921] | 17% | 4% | 11% | — | 77% | 28% | 30% | 16% | 26% | 18% |
| **IntegerUO** [10886] | 16% | 0% | 15% | 28% | — | 29% | 16% | 10% | 19% | 12% |
| **MishandledException** [3337] | 11% | 0% | 17% | 33% | **96%** | — | 20% | 13% | 25% | 18% |
| **Reentrancy** [3483] | 30% | 7% | 33% | 33% | 51% | 19% | — | 14% | 18% | 42% |
| **Timestamp** [1493] | 12% | 0% | 13% | 41% | 72% | 28% | 34% | — | 26% | 21% |
| **TOD** [2366] | 14% | 0% | 21% | 44% | **86%** | 36% | 27% | 16% | — | 17% |
| **UnusedReturn** [2118] | 11% | 0% | 47% | 34% | 62% | 29% | **69%** | 15% | 19% | — |

**Critical co-occurrences:**
- DoS ↔ Reentrancy: **99%** — virtually every DoS contract is also labeled Reentrancy. The model cannot distinguish these.
- MishandledException ↔ IntegerUO: **96%** — same problem.
- TOD ↔ IntegerUO: **86%** — model will conflate TOD with IntegerUO.
- UnusedReturn ↔ Reentrancy: **69%** — severe.

This is a BCCC labeling artifact: contracts in multiple class directories get all their class labels ORed together. It does NOT mean these vulnerabilities co-occur in real contracts.

### 10.2 Data starvation

| Class | Train positives | Status |
|-------|-----------------|--------|
| **DenialOfService** | **257** | ⛔ SEVERE — barely enough for one batch |
| **Timestamp** | 1,493 | ⚠️ LOW |
| UnusedReturn | 2,118 | OK |
| CallToUnknown | 2,502 | OK |
| ExternalBug | 2,422 | OK |
| TransactionOrderDependence | 2,366 | OK |
| MishandledException | 3,337 | OK |
| Reentrancy | 3,483 | OK (+25 CEI) |
| GasException | 3,921 | OK |
| IntegerUO | 10,886 | OK (dominant) |

DoS at 257 training positives is critically data-starved. With batch_size=16 and 99% Reentrancy co-occurrence, the model effectively cannot learn a DoS-specific signal.

### 10.3 IntegerUO dominance

- IntegerUO: **35.0%** of all training rows have IntegerUO=1
- Safe contracts: **55.0%** of all training rows

With these proportions, a batch of 64 (effective batch after accumulation) would typically contain:
- ~35 contracts with IntegerUO=1
- ~35 safe contracts
- ~11 Reentrancy contracts
- ~8 GasException contracts
- ~1 DoS contract (with probability < 50% per batch)

IntegerUO and safe contracts dominate every batch.

### 10.4 Safe contract clustering

Safe contracts are evenly distributed across splits (55.0% / 55.1% / 55.1%). No clustering — they're mixed uniformly.

---

## Summary of Critical Findings

### Bugs Found

| # | Severity | Description | Impact |
|---|----------|-------------|--------|
| B1 | HIGH | `reextract_graphs.py` never saves `contract_path` on graph object | Re-running `build_multilabel_index.py` today would produce all-zero labels for 44K graphs. CSV currently valid because built from old graphs. |
| B2 | LOW | `node_metadata[i]['type']` is "STATE_VAR" for all nodes in current graphs | Training unaffected (model uses `x` tensor). Debugging/visualization misleading. Fixed in code (a0576fb) but graphs pre-date the fix. |

### Structural Data Quality Issues

| # | Issue | Quantified |
|---|-------|-----------|
| Q1 | DoS/Reentrancy inseparability | 99% co-occurrence in training set |
| Q2 | MishandledException/IntegerUO inseparability | 96% co-occurrence |
| Q3 | DoS data starvation | 257 training positives |
| Q4 | 96.6% of contracts truncated at 512 tokens | CodeBERT sees <35-60% of each contract |
| Q5 | LoC feature (feat[6]) is raw count, not normalized | Range 1–500+, potentially dominates early dot products |
| Q6 | IntegerUO at 35% of training rows | Every batch dominated by IntegerUO samples |

### What Is Confirmed Correct

- Deduplication: complete, no residual leakage
- Split stratification: clean (0 cross-split overlap, well-stratified per class)
- Feature vectors: correct (type_id normalized to [0,1], all 12 features as documented)
- Edge type encoding: correct (0-6 in .pt files, 7 added at runtime)
- CEI augmentation: all 50 contracts in training split, labels correct
- Token files: 44,470 matching graph files, content correct
- Label assignment: CSV is valid (built from graphs that had `contract_path`)
