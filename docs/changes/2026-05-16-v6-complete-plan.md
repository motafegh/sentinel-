# SENTINEL v6 — Complete Implementation Plan
**Date:** 2026-05-16
**Design Principle:** No constants or architectural choices are locked. Every improvement,
no matter how small, will be included. If changing anything helps even a little, we change it.

This plan covers every code change, data change, and training change required to produce
a v6 model that passes behavioral gates (≥ 80% detection, ≥ 80% safe specificity).

---

## Summary Change Table

| Change | File(s) | Phase | Priority |
|--------|---------|-------|----------|
| return_ignored fix | graph_extractor.py | 0.1 (DONE) | P0 |
| Transfer/Send in ext_calls | graph_extractor.py | 0.2 | P0 |
| Transfer/Send in CFG typing | graph_extractor.py | 0.3 | P0 |
| uses_block_globals feature (replace pure) | graph_extractor.py, graph_schema.py | 0.4 | P0 |
| loc normalization (log1p) | graph_extractor.py | 0.5 | P0 |
| FEATURE_SCHEMA_VERSION bump to v4 | graph_schema.py | 0.6 | P0 |
| Full re-extraction | reextract_graphs.py | 0.7 | P0 |
| Validate extraction | validate_graph_dataset.py | 0.8 | P0 |
| Windowed tokenization (W×512) | transformer_encoder.py, preprocessor | 1.1 | P0 |
| TransformerEncoder.forward() windowed | transformer_encoder.py | 1.2 | P0 |
| Token cache format [W,512] | dual_path_dataset.py, tokenize_contracts.py | 1.3 | P0 |
| Re-tokenize all contracts | tokenize_contracts.py | 1.4 | P0 |
| Rebuild cache | create_cache.py | 1.5 | P0 |
| GNN hidden_dim 128→256 | gnn_encoder.py, trainer.py | 2.1 | P1 |
| 2nd CF phase layer (Phase2: 2 layers) | gnn_encoder.py | 2.2 | P1 |
| 2nd REVERSE_CONTAINS layer (Phase3: 2 layers) | gnn_encoder.py | 2.2 | P1 |
| Edge embedding dim 32→64 | gnn_encoder.py | 2.3 | P1 |
| Window attention pooler module | transformer_encoder.py | 2.8 | P1 |
| Classifier hidden layer [384→192→10] | sentinel_model.py | 2.5 | P1 |
| GNN attention pooling over function nodes | gnn_encoder.py | 2.6 | P1 |
| LoRA r: 16→32, alpha: 32→64 | transformer_encoder.py | 2.7 | P1 |
| model_version = "v6.0" | sentinel_model.py | 2.9 | P1 |
| Pos_weight: freq-based, floor=1.0 (no cap) | trainer.py | 3.1 | P0 |
| AsymmetricLoss (replace BCE) | trainer.py, new loss file | 3.2 | P1 |
| Label smoothing keep 0.05 | trainer.py | 3.3 | P0 |
| Epochs 100, patience 30 | trainer.py, train.py | 3.4 | P0 |
| Gradient accumulation 8 steps | trainer.py | 3.5 | P1 |
| eval_threshold 0.35 | trainer.py | 3.6 | P0 |
| Warmup 5 epochs | trainer.py | 3.7 | P1 |
| GNN LR 2.5×, Fusion LR 0.5×, LoRA LR 0.3× | trainer.py | 3.8 | P0 |
| DoS augmentation (~500 clean single-label) | data augmentation script | 4.1 | P1 |
| Timestamp augmentation (~500 clean single-label) | data augmentation script | 4.2 | P1 |
| Verify CEI augmentation in deduped CSV | multilabel_index_deduped.csv | 4.3 | P0 |
| Per-class oversampling DoS+Timestamp | dual_path_dataset.py | 4.4 | P2 |
| Re-extract with Phase 0 fixes | reextract_graphs.py | 5.1 | P0 |
| Validate graphs | validate_graph_dataset.py | 5.2 | P0 |
| Re-tokenize windowed | tokenize_contracts.py | 5.3 | P0 |
| Rebuild deduped CSV (labels stable) | create_cache.py | 5.4 | P0 |
| Rebuild cache | create_cache.py | 5.5 | P0 |
| v6.0 training start | train.py | 6.1 | P0 |

---

## Phase 0: Feature Schema v4

All changes in this phase are in `ml/src/preprocessing/graph_extractor.py` and
`ml/src/preprocessing/graph_schema.py`. The goal is to fix every known feature bug
and add every useful feature signal before re-extraction.

### 0.1 return_ignored Fix — DONE (commit bef1f2a)

Status: already applied.

**What was fixed:** `op.lvalue is None` replaced with correct check: does the lvalue ID of a
call's return value appear in any subsequent `op.read` set within the function? If not,
the return value is unused → `return_ignored = 1`.

**Correct implementation:**
```python
def _compute_return_ignored(self, function) -> float:
    for node in function.nodes:
        for op in node.irs:
            if isinstance(op, (HighLevelCall, LowLevelCall)):
                if op.lvalue is not None:
                    lval_id = id(op.lvalue)
                    # Check if this lvalue is read in any subsequent op in this function
                    used = False
                    for other_node in function.nodes:
                        for other_op in other_node.irs:
                            if hasattr(other_op, 'read') and \
                               any(id(v) == lval_id for v in other_op.read):
                                used = True
                                break
                        if used:
                            break
                    if not used:
                        return 1.0
    return 0.0
```

**Affected classes:** MishandledException, UnusedReturn
**Expected improvement:** MishandledException: 0.342 → ≥ 0.50, UnusedReturn: 0.238 → ≥ 0.45

### 0.2 Transfer/Send in ext_calls

**File:** `ml/src/preprocessing/graph_extractor.py`
**Function:** `_compute_external_call_count()`

**Current:**
```python
from slither.slithir.operations import HighLevelCall, LowLevelCall

def _compute_external_call_count(self, function) -> int:
    count = 0
    for node in function.nodes:
        for op in node.irs:
            if isinstance(op, (HighLevelCall, LowLevelCall)):
                count += 1
    return count
```

**New:**
```python
from slither.slithir.operations import HighLevelCall, LowLevelCall, Transfer, Send

def _compute_external_call_count(self, function) -> int:
    count = 0
    for node in function.nodes:
        for op in node.irs:
            if isinstance(op, (HighLevelCall, LowLevelCall, Transfer, Send)):
                count += 1
    return count
```

**Why:** ETH-transfer DoS loops use `recipient.transfer(amount)` or `recipient.send(amount)`,
which produce Slither `Transfer` and `Send` IR ops. These are NOT HighLevelCall or LowLevelCall.
Without this fix, a DoS-vulnerable ETH-transfer loop has ext_calls=0, making it look like
a pure state-manipulation contract to the GNN.

**Affected classes:** DenialOfService, Reentrancy (some ETH-transfer reentrancy variants)

### 0.3 Transfer/Send in CFG Node Typing

**File:** `ml/src/preprocessing/graph_extractor.py`
**Function:** `_cfg_node_type()`

**Current priority ordering for CFG node type:**
1. `has LowLevelCall op` → CFG_NODE_CALL
2. `has HighLevelCall op` → CFG_NODE_CALL
3. `has Write op` → CFG_NODE_WRITE
4. `has Condition op` → CFG_NODE_COND
5. `is function entry` → CFG_NODE_ENTRY
6. `is function return` → CFG_NODE_RETURN
7. `default` → CFG_NODE_ENTRY

**New — add Transfer/Send to the CALL group:**
```python
from slither.slithir.operations import HighLevelCall, LowLevelCall, Transfer, Send

def _cfg_node_type(self, node) -> int:
    has_call = any(isinstance(op, (HighLevelCall, LowLevelCall, Transfer, Send))
                   for op in node.irs)
    if has_call:
        return CFG_NODE_TYPES.index('CFG_NODE_CALL')
    # ... rest of priority ordering unchanged
```

**Why:** CFG node type is critical for Phase 2 (CONTROL_FLOW) message passing. If a Transfer
op's CFG node is typed as CFG_NODE_WRITE instead of CFG_NODE_CALL, the GNN cannot see
"call before write" (CEA) vs "write before call" (CEI) patterns involving ETH transfers.
Transfer-based reentrancy (classic Dao hack) would be invisible to Phase 2.

**Affected classes:** Reentrancy (Transfer-based variants), DenialOfService

### 0.4 Add uses_block_globals Feature (Replace pure)

**File:** `ml/src/preprocessing/graph_extractor.py`
**Change:** Replace feature index 1 (`pure`) with `uses_block_globals`

**Why replace pure:** The `pure` function modifier means the function cannot read state and
cannot be vulnerable to any state-based exploit. Its discriminative value is as a negative
indicator only (pure=1 → almost certainly not vulnerable to state-based bugs). However,
it is rarely 1 for vulnerable functions (by definition, vulnerable functions access state).
In practice, `pure` contributes almost zero gradient signal on vulnerable examples.

**Why uses_block_globals is better:** Direct signal for Timestamp class (block.timestamp),
TOD class (block.number for ordering), and some GasException cases (block.gaslimit).

**Implementation:**
```python
from slither.core.variables.state_variable import StateVariable
from slither.slithir.variables import Constant
from slither.slithir.variables.solidity_variable import SolidityVariableComposed

BLOCK_GLOBALS = frozenset({
    'timestamp', 'number', 'difficulty', 'gaslimit',
    'coinbase', 'basefee', 'prevrandao'
})

def _compute_uses_block_globals(self, function) -> float:
    for node in function.nodes:
        for op in node.irs:
            if hasattr(op, 'read'):
                for var in op.read:
                    if isinstance(var, SolidityVariableComposed):
                        name = var.name.split('.')[-1].lower()  # 'block.timestamp' → 'timestamp'
                        if name in BLOCK_GLOBALS:
                            return 1.0
    return 0.0
```

**Feature array update (feat[1]):**
```
feat[1] = uses_block_globals  # was: pure (function is pure)
```

**Note:** FEATURE_SCHEMA_VERSION must be bumped to "v4" (see 0.6) because the feature
at index 1 is semantically different. Any cached graphs or inference data using schema v3
will have incorrect feature[1] interpretation.

**Affected classes:** Timestamp (direct signal), TOD (indirect — block.number), GasException (block.gaslimit)

### 0.5 Normalize loc Feature (feat[6])

**File:** `ml/src/preprocessing/graph_extractor.py`
**Function:** `_build_node_features()`

**Current:**
```python
feat[6] = float(node.loc)  # raw line count, range [0, 2538]
```

**New:**
```python
import math
# log1p normalization: log1p(loc) / log1p(1000) clamped to [0, 1]
# log1p(1000) ≈ 6.908
_LOC_NORM = math.log1p(1000.0)
feat[6] = min(1.0, math.log1p(float(node.loc)) / _LOC_NORM)
```

**Mapping:**
- loc=0 → 0.000
- loc=10 → 0.342
- loc=50 → 0.573
- loc=100 → 0.683
- loc=133 → 0.722
- loc=1000 → 1.000
- loc=2538 → 1.000 (clamped)

**Why:** Raw loc=133 (typical CONTRACT node) dominates GAT attention dot products where all
other features are 0/1. This creates "attend to large contracts" as a spurious attention
pattern. After normalization, CONTRACT nodes have feat[6]≈0.72, comparable in scale to
the type_id feature (feat[7] = type_id/12.0 ∈ [0, 1]).

**Affected classes:** All (global attention quality improvement)

### 0.6 Bump FEATURE_SCHEMA_VERSION to "v4"

**File:** `ml/src/preprocessing/graph_schema.py`

```python
FEATURE_SCHEMA_VERSION = "v4"   # was "v3" (2026-05-12, commit a0576fb)
                                  # v3→v4: uses_block_globals replaces pure (feat[1])
                                  #         loc normalized with log1p
                                  #         return_ignored correctly computed
                                  #         Transfer/Send counted in ext_calls
```

**Impact:** Any inference using old v3 graphs will get incorrect feature[1]. The dataset
loader should check FEATURE_SCHEMA_VERSION and refuse to load v3 graphs with a clear error.

Add to `dual_path_dataset.py`:
```python
EXPECTED_FEATURE_SCHEMA_VERSION = "v4"

def _verify_graph(self, graph_data):
    schema = graph_data.get('feature_schema_version', 'unknown')
    if schema != EXPECTED_FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Graph at {path} has schema {schema}, expected {EXPECTED_FEATURE_SCHEMA_VERSION}. "
            f"Re-run reextract_graphs.py."
        )
```

### 0.7 Re-run Full Extraction

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/reextract_graphs.py \
    --workers 16 \
    --output-dir ml/data/graphs \
    --index ml/data/processed/multilabel_index_deduped.csv
```

Expected time: ~4–6 hours for 44,420 contracts on 16 workers.
Expected output: 44,420 `.pt` graph files, schema version "v4" embedded in each.

### 0.8 Validate Extraction

```bash
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py \
    --index ml/data/processed/multilabel_index_deduped.csv \
    --graph-dir ml/data/graphs \
    --check-dim 12 \
    --check-edge-types 7 \
    --check-schema-version v4
```

Check that:
- All 44,420 graphs exist
- NODE_FEATURE_DIM=12 everywhere
- No ghost graphs
- feature[1] (uses_block_globals) is non-zero for at least 10% of FUNCTION nodes (sanity check)
- feature[6] (loc normalized) is in [0, 1] for all nodes
- ext_calls > 0 for known Transfer-pattern contracts (manual spot check)

---

## Phase 1: Token Pipeline Overhaul

The CodeBERT path currently sees only 512 tokens (3.9% of contracts fit). This phase implements
windowed tokenization so each contract is represented by up to 8 non-overlapping-but-strided
windows of 512 tokens, covering 83% of the median contract.

### 1.1 Implement tokenize_windowed()

**File:** `ml/src/preprocessing/tokenizer.py` (new function, or add to existing preprocessor)

```python
from typing import List, Tuple
import torch

def tokenize_windowed(
    text: str,
    tokenizer,
    max_len: int = 512,
    stride: int = 256,
    max_windows: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a contract with a sliding window approach.

    Returns:
        input_ids:      [W, max_len] int64 tensor
        attention_mask: [W, max_len] int64 tensor
    where W = min(max_windows, ceil(full_token_length / stride))
    """
    # First tokenize the full text without truncation (up to a large limit)
    full_encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors='pt',
    )
    full_ids = full_encoding['input_ids'][0]  # [N]
    total_len = full_ids.shape[0]

    windows_ids = []
    windows_mask = []

    pos = 0
    while pos < total_len and len(windows_ids) < max_windows:
        # Reserve 2 tokens for [CLS] and [SEP]
        chunk = full_ids[pos : pos + max_len - 2]
        chunk_len = chunk.shape[0]

        # Build window: [CLS] + chunk + [SEP] + [PAD]*remainder
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        window_ids = torch.cat([
            torch.tensor([cls_id], dtype=torch.long),
            chunk,
            torch.tensor([sep_id], dtype=torch.long),
        ])
        pad_len = max_len - window_ids.shape[0]
        if pad_len > 0:
            window_ids = torch.cat([
                window_ids,
                torch.full((pad_len,), pad_id, dtype=torch.long)
            ])
        mask = torch.ones(max_len, dtype=torch.long)
        if pad_len > 0:
            mask[max_len - pad_len:] = 0

        windows_ids.append(window_ids)
        windows_mask.append(mask)
        pos += stride

    W = len(windows_ids)
    return (
        torch.stack(windows_ids),   # [W, max_len]
        torch.stack(windows_mask),  # [W, max_len]
    )
```

**Parameters:**
- `max_len=512`: CodeBERT hard position embedding limit — cannot change without retraining CodeBERT
- `stride=256`: 50% overlap between windows; ensures boundary tokens appear in 2 windows
- `max_windows=8`: caps VRAM usage; covers 83% of median (2,469-token) contract

### 1.2 Update TransformerEncoder.forward() for Windowed Input

**File:** `ml/src/models/transformer_encoder.py`

**New forward signature:**
```python
def forward(
    self,
    input_ids: torch.Tensor,       # [B, W, 512] or [B, 512] (backward compat)
    attention_mask: torch.Tensor,  # [B, W, 512] or [B, 512]
) -> torch.Tensor:                 # [B, 768]
```

**Implementation sketch:**
```python
def forward(self, input_ids, attention_mask):
    if input_ids.dim() == 2:
        # Legacy single-window path: [B, 512] → add W=1 dim
        input_ids = input_ids.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1)

    B, W, L = input_ids.shape

    # Flatten batch×window for CodeBERT forward
    flat_ids = input_ids.view(B * W, L)          # [B*W, 512]
    flat_mask = attention_mask.view(B * W, L)    # [B*W, 512]

    # CodeBERT forward (frozen backbone + LoRA adapters)
    outputs = self.codebert(
        input_ids=flat_ids,
        attention_mask=flat_mask,
    )
    cls_emb = outputs.last_hidden_state[:, 0, :]  # [B*W, 768] — CLS token

    # Reshape to [B, W, 768]
    cls_emb = cls_emb.view(B, W, 768)

    # Window attention pooling: [B, W, 768] → [B, 768]
    pooled = self.window_pooler(cls_emb, attention_mask)  # see WindowAttentionPooler

    return pooled  # [B, 768]
```

**Window attention mask for pooler:** need to know which windows are real vs padding windows
(if a contract has W < max_windows, remaining "windows" should be masked out).
The `attention_mask` for padding windows should be all-zeros (no tokens). The pooler uses
`any(attention_mask[b, w] > 0)` to determine valid windows.

### 1.3 WindowAttentionPooler Module

**File:** `ml/src/models/transformer_encoder.py` (new inner class or separate module)

```python
class WindowAttentionPooler(nn.Module):
    """
    Learnable attention pooling over W window CLS embeddings.
    Masks out padding windows (all-zero attention_mask).
    """
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        cls_embs: torch.Tensor,    # [B, W, 768]
        window_mask: torch.Tensor, # [B, W, 512] — use to detect valid windows
    ) -> torch.Tensor:             # [B, 768]
        # Determine valid windows: a window is valid if it has any non-zero token
        # window_mask: [B, W, 512] → [B, W] (1 if valid, 0 if padding window)
        valid = (window_mask.sum(-1) > 0).float()  # [B, W]

        # Attention scores: [B, W, 768] → [B, W, 1] → [B, W]
        scores = self.score(cls_embs).squeeze(-1)  # [B, W]

        # Mask padding windows with large negative before softmax
        scores = scores + (1.0 - valid) * (-1e9)

        # Softmax attention weights: [B, W]
        weights = torch.softmax(scores, dim=-1)  # [B, W]

        # Weighted sum: [B, W] × [B, W, 768] → [B, 768]
        pooled = (weights.unsqueeze(-1) * cls_embs).sum(dim=1)  # [B, 768]
        return pooled
```

### 1.4 Update Token Cache Format

**File:** `ml/scripts/tokenize_contracts.py` (or equivalent tokenization script)

Change output format:
```python
# OLD format:
torch.save({'input_ids': tensor_512, 'attention_mask': tensor_512}, path)

# NEW format:
torch.save({
    'input_ids': tensor_W_512,      # [W, 512] — W = min(max_windows, windows_needed)
    'attention_mask': tensor_W_512, # [W, 512]
    'num_windows': W,               # int
    'token_schema_version': 'v2',   # bumped from v1
}, path)
```

**Note:** `num_windows` varies per contract (W ∈ [1, max_windows=8]).
The DataLoader must handle variable W by padding batches to `max_windows` or using per-batch
maximum W. Recommended: pad all contracts to W=8 (fill extra windows with all-zero tensors
and mask out in pooler). This keeps tensor shapes uniform within a batch.

### 1.5 Update DualPathDataset

**File:** `ml/src/datasets/dual_path_dataset.py`

```python
def _load_tokens(self, token_path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(token_path, weights_only=True)
    schema = data.get('token_schema_version', 'v1')

    if schema == 'v1':
        # Legacy single-window: [512] → pad to [8, 512]
        ids = data['input_ids']     # [512]
        mask = data['attention_mask']  # [512]
        # Expand to [1, 512] then pad with zeros to [8, 512]
        ids = ids.unsqueeze(0)
        mask = mask.unsqueeze(0)
        ids = torch.cat([ids, torch.zeros(7, 512, dtype=ids.dtype)], dim=0)
        mask = torch.cat([mask, torch.zeros(7, 512, dtype=mask.dtype)], dim=0)
        return {'input_ids': ids, 'attention_mask': mask, 'num_windows': 1}

    elif schema == 'v2':
        W = data['num_windows']
        ids = data['input_ids']     # [W, 512]
        mask = data['attention_mask']  # [W, 512]
        max_W = 8
        if W < max_W:
            # Pad to max_W with zero windows
            pad = torch.zeros(max_W - W, 512, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
            mask = torch.cat([mask, pad], dim=0)
        return {'input_ids': ids, 'attention_mask': mask, 'num_windows': W}
```

### 1.6 Re-tokenize All Contracts

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/tokenize_contracts.py \
    --index ml/data/processed/multilabel_index_deduped.csv \
    --output-dir ml/data/tokens \
    --max-len 512 \
    --stride 256 \
    --max-windows 8
```

Expected time: ~2–4 hours for 44,420 contracts.

### 1.7 Rebuild Cache

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/create_cache.py \
    --index ml/data/processed/multilabel_index_deduped.csv \
    --graph-dir ml/data/graphs \
    --token-dir ml/data/tokens \
    --output ml/data/cached_dataset_deduped_v2.pkl
```

Verify:
```bash
python -c "
import pickle
with open('ml/data/cached_dataset_deduped_v2.pkl', 'rb') as f:
    d = pickle.load(f)
sample = d[0]
assert sample['input_ids'].shape == (8, 512), f'Bad shape: {sample[\"input_ids\"].shape}'
assert sample['x'].shape[1] == 12, f'Bad feat dim: {sample[\"x\"].shape}'
print('Cache OK:', len(d), 'samples')
"
```

---

## Phase 2: Model Architecture Updates

### 2.1 GNN hidden_dim: 128 → 256

**File:** `ml/src/models/gnn_encoder.py`

Update default:
```python
class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 12,
        hidden_dim: int = 256,    # was 128
        num_edge_types: int = 8,
        ...
    ):
```

**Impact on GNN parameter count (approximate):**
- Phase 1 GATConv (heads=8, dim=256): ~8 × (256+256) × 256 / 8 = ~65K per layer × 2 = ~130K
- Phase 2 GATConv (heads=1, dim=256): ~256 × 256 = ~65K per layer × 2 = ~130K
- Phase 3 GATConv (heads=1, dim=256): ~256 × 256 = ~65K per layer × 2 = ~130K
- Total GNN: ~600K params (was ~150K at dim=128) — well within VRAM budget
- Full model: ~600K (GNN) + 590K (LoRA) + ~2M (CrossAttentionFusion+classifier) = ~3.2M trainable
- With CodeBERT frozen backbone: ~124M total params in memory, ~3.2M trainable

**VRAM estimate at hidden_dim=256:**
- Activations per batch of 8: ~O(256 × N_nodes) ≈ 256 × 500 × 8 × 4 bytes ≈ 4MB (negligible)
- No significant VRAM issue on RTX 3070 8GB

### 2.2 GNN Depth: 2nd CF Layer and 2nd RC Layer

**File:** `ml/src/models/gnn_encoder.py`

**Phase structure change:**
```
Phase 1: layers 1+2 (structural + CONTAINS) — unchanged, 2 layers
Phase 2: layers 3+4 (CONTROL_FLOW directed) — ADD layer 4 (was layer 3 only)
Phase 3: layers 5+6 (REVERSE_CONTAINS) — ADD layer 6 (was layer 4 only)
Total: 6 layers (was 4)
```

**Why 2 CF layers:** 1 CF hop can only propagate signal from a node to its direct CF successor.
A typical reentrancy function has:
```
ENTRY(0) → CHECK(1) → CALL(2) → ASSIGN_TMP(3) → WRITE(4) → RETURN(5)
```
With 1 hop: CALL(2) sends message to ASSIGN_TMP(3) only.
With 2 hops: CALL(2) message reaches WRITE(4) via ASSIGN_TMP(3). This is the "call before write"
signal needed for CEI/CEA discrimination.

**Why 2 RC layers:** REVERSE_CONTAINS allows child nodes to send messages upward to their
CONTAINS parent (function→contract reverse). 2 RC hops allows grandchild→grandparent propagation,
which is useful for multi-function vulnerability patterns.

**Update JK aggregation to include 3 phases × 2 layers = 6 total inputs:**
If JK uses 3 phase outputs (one per phase), keeping this is fine — each phase's output is
the aggregation of its 2 layers. The JK attention weights over 3 phases (128+256 → 3-way
attention if dim=256) remain valid.

### 2.3 Edge Embedding Dimension: 32 → 64

**File:** `ml/src/models/gnn_encoder.py`

```python
self.edge_emb = nn.Embedding(num_edge_types, 64)   # was 32
```

The edge embedding is concatenated with node features in GATConv. With hidden_dim=256, the
input to Phase 2 GATConv is `node_emb(256) + edge_emb(64) = 320`. Update GATConv input_channels
accordingly.

**Why:** 32 dimensions for 8 edge types is sparse (32/8 = 4 dims per type). At 64 dims,
each edge type gets 8 dims of dedicated representational capacity. This is especially important
for distinguishing CONTROL_FLOW (edge_type=6) from CONTAINS (edge_type=5) which share similar
structural roles but different semantic meanings.

### 2.4 loc Normalization Already in Phase 0.5

(no additional work here — covered by Phase 0.5 change in graph_extractor.py)

### 2.5 Classifier Hidden Layer: [384 → 192 → 10]

**File:** `ml/src/models/sentinel_model.py`

**Current:**
```python
self.classifier = nn.Linear(384, num_classes)  # [B, 384] → [B, 10]
```

**New:**
```python
self.classifier = nn.Sequential(
    nn.Linear(384, 192),
    nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(192, num_classes),
)
```

The hidden layer allows non-linear combinations of the three-eye representations. Specifically,
the model can learn:
- "GNN eye detects call-before-write AND TF eye detects 'transfer' keyword → high Reentrancy"
- "GNN eye detects large loop AND ext_calls > 0 AND TF sees 'gas' keyword → high DoS"
These non-linear AND patterns are impossible with a single linear layer.

**Parameter cost:** 384×192 + 192×10 = 73,728 + 1,920 = 75,648 params. Negligible.

### 2.6 GNN Pooling: Add Attention Pooling for Function Nodes

**File:** `ml/src/models/gnn_encoder.py`

**Current GNN eye:** Max+mean pool over function nodes → concat [256+256=512] → Linear(512, 128)

**New GNN eye:** Max+mean+attention pool over function nodes → concat [256+256+256=768] → Linear(768, 256)

```python
class FunctionNodeAttentionPooler(nn.Module):
    """Learnable attention pooling over function-type nodes in a batch."""
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, fn_mask: torch.Tensor) -> torch.Tensor:
        # x: [N_fn, dim]; fn_mask: [N_fn] boolean
        fn_embs = x[fn_mask]  # [N_fn_valid, dim]
        scores = self.score(fn_embs)  # [N_fn_valid, 1]
        weights = torch.softmax(scores, dim=0)  # [N_fn_valid, 1]
        return (weights * fn_embs).sum(0)  # [dim]
```

Attention pooling finds the "most important" function node (by learned scoring) rather than
averaging or max-pooling. This is useful for contracts with many functions where the
vulnerability is localized to one or two functions.

### 2.7 LoRA r: 16 → 32, alpha: 32 → 64

**File:** `ml/src/models/transformer_encoder.py`

```python
lora_config = LoraConfig(
    r=32,            # was 16
    lora_alpha=64,   # was 32
    lora_dropout=0.05,
    target_modules=['query', 'value'],  # CodeBERT Q+V in all 12 layers
    bias='none',
)
```

**Parameter count increase:**
- 12 layers × 2 matrices (Q, V) × 2 LoRA matrices (A, B) × r × d
- With r=32, d=768: 12 × 2 × 2 × 32 × 768 = 1,179,648 params (was 589,824 at r=16)
- ~1.2M trainable LoRA params — still manageable

**Why:** With windowed tokenization (up to 8 windows), the LoRA must encode vulnerability
patterns from any position in the contract. Higher r gives the LoRA more capacity to represent
varied positional contexts. The scaling factor (alpha/r = 64/32 = 2.0) remains unchanged.

### 2.8 Window Attention Pooler

(already described in Phase 1.3 — see WindowAttentionPooler class)

### 2.9 Update model_version

**File:** `ml/src/models/sentinel_model.py`

```python
MODEL_VERSION = "v6.0"
```

In checkpoint saving:
```python
checkpoint_dict = {
    'model_state_dict': model.state_dict(),
    'model_version': MODEL_VERSION,
    'feature_schema_version': 'v4',
    'token_schema_version': 'v2',
    ...
}
```

---

## Phase 3: Training Pipeline Changes

### 3.1 Pos_weight Strategy: Frequency-Based with Floor=1.0

**File:** `ml/src/training/trainer.py`

**Current (v5.3 with pos_weight_min_samples=3000):**
- If class has > 3000 training positives → pos_weight capped to 1.0
- This removed imbalance correction for 5 of 10 classes

**New:**
```python
def _compute_pos_weights(class_counts, total_negatives_per_class, device, floor=1.0):
    """
    pos_weight[i] = (n_negative[i] / n_positive[i]) clipped to [floor, max_weight=20.0]
    No upper cap based on sample count — let actual frequency dictate the weight.
    """
    pos_weights = []
    for cls_idx in range(NUM_CLASSES):
        n_pos = class_counts[cls_idx]
        n_neg = total_samples - n_pos
        if n_pos == 0:
            weight = floor
        else:
            weight = max(floor, min(20.0, n_neg / n_pos))
        pos_weights.append(weight)
    return torch.tensor(pos_weights, dtype=torch.float32, device=device)
```

**Expected weights with this formula:**
| Class | Positives | Negatives | pos_weight |
|-------|-----------|-----------|------------|
| DenialOfService | 257 | 30,885 | 20.0 (capped at max) |
| Timestamp | 1,493 | 29,649 | 19.86 |
| CallToUnknown | 2,527 | 28,615 | 11.32 |
| ExternalBug | 2,383 | 28,759 | 12.07 |
| TOD | 2,374 | 28,768 | 12.12 |
| UnusedReturn | 2,126 | 29,016 | 13.65 |
| Reentrancy | 3,500 | 27,642 | 7.90 |
| MishandledException | 3,296 | 27,846 | 8.45 |
| GasException | 3,918 | 27,224 | 6.95 |
| IntegerUO | 10,886 | 20,256 | 1.86 |

These weights reflect actual class imbalance without artificial capping by sample count.

### 3.2 Loss Function: AsymmetricLoss (Replace BCE)

**File:** `ml/src/training/losses.py` (new file)

AsymmetricLoss (ASL) applies asymmetric focusing: harder penalty on false negatives for
minority classes, softer penalty on false positives. This naturally handles multi-label
class imbalance without requiring pos_weight tuning.

```python
import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Paper: Ridnik et al., 2021 — "Asymmetric Loss For Multi-Label Classification"

    L_pos = -(1 - p)^gamma_pos * log(p)
    L_neg = -(p_m)^gamma_neg * log(1 - p_m)
    where p_m = max(p - m, 0)  (margin shifts probability down for negative samples)

    Recommended defaults:
    - gamma_neg=4, gamma_pos=1, m=0.05 for standard multi-label
    - With label_smoothing, use m=0.0 (smoothing already regularizes negatives)
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,         # margin shift for negatives (=m in paper)
        eps: float = 1e-8,
        disable_torch_grad: bool = True,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad = disable_torch_grad

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B, C] float (already label-smoothed if applicable)
        probs = torch.sigmoid(logits)

        # Asymmetric clipping: shift down negative probabilities by margin m
        probs_pos = probs
        probs_neg = (probs - self.clip).clamp(min=0)

        # log probabilities
        log_pos = torch.log(probs_pos + self.eps)
        log_neg = torch.log(1.0 - probs_neg + self.eps)

        # Focal weights
        if self.disable_torch_grad:
            with torch.no_grad():
                pt_pos = probs_pos
                pt_neg = (1.0 - probs_neg)
        else:
            pt_pos = probs_pos
            pt_neg = (1.0 - probs_neg)

        loss_pos = -(torch.pow(1.0 - pt_pos, self.gamma_pos)) * log_pos
        loss_neg = -(torch.pow(pt_neg, self.gamma_neg)) * log_neg

        # Combine: targets selects positive/negative branch per element
        loss = targets * loss_pos + (1.0 - targets) * loss_neg

        return loss.mean()
```

**Why ASL over BCE:**
- BCE treats all false negatives equally regardless of class frequency
- ASL's gamma_neg=4 means high-confidence false negatives (p≈1 but target=0) get penalized 4× more
  than low-confidence ones (p≈0.5 but target=0). This naturally suppresses spurious high-confidence
  negative predictions for minority classes.
- The margin `m=0.05` shifts negative probabilities down by 0.05 before computing the negative
  loss, which has the effect of ignoring low-probability negatives (reduces false negative penalty
  when the model is uncertain but not confidently wrong).

**Integration with pos_weight:** pos_weight and ASL serve overlapping purposes. With ASL,
use pos_weight only as a mild correction (floor=1.0, max=5.0) to avoid doubling up on the
same correction mechanism.

### 3.3 Label Smoothing: Keep 0.05

```python
# In trainer.py, apply label smoothing before loss computation:
def _smooth_targets(targets: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    return targets * (1.0 - smoothing) + smoothing * 0.5
```

Label smoothing prevents the model from becoming overconfident (logits → ±∞) even with
ASL. Combined with ASL's gamma_neg, this provides robust training signal.

### 3.4 Epochs: 100 with patience=30

```bash
--epochs 100 --early-stop-patience 30
```

**Rationale:** Deeper architecture (6-layer GNN, windowed CodeBERT, larger hidden dims) needs
more epochs to converge. With 973 steps/epoch (31,142 samples / batch 32), 100 epochs = 97,300
steps. Learning rate cosine annealing over 100 epochs with 5-epoch warmup → LR decays to ~2%
of base by epoch 100.

### 3.5 Gradient Accumulation: 8 Steps (effective batch=64)

```bash
--gradient-accumulation-steps 8
```

**Rationale:** With effective batch=64 (batch_size=8 × accum=8), each step includes ~0.5 DoS
positive examples (vs 0.26 at accum=4). This is still very low but 2× better. More importantly,
smoother gradient estimates benefit minority class learning. The cosine annealing scheduler
is based on the number of optimizer steps (after accumulation), so epochs remain meaningful.

### 3.6 eval_threshold: 0.35

```bash
--eval-threshold 0.35
```

During training validation, report macro-F1 at threshold=0.35. Post-training, use
`tune_threshold.py` to find per-class optimal thresholds for final results.

### 3.7 Warmup: 5 Epochs

```python
# In trainer.py:
warmup_epochs = 5      # was 3
warmup_steps = warmup_epochs * steps_per_epoch
```

**Rationale:** Deeper architecture (6 layers GNN, higher hidden_dim, larger LoRA) has more
parameters to coordinate during early training. Longer warmup prevents the randomly-initialized
GNN layers (especially the 2nd CF and RC layers) from making large gradient updates before
the rest of the model has stabilized.

### 3.8 LR Group Configuration

```python
optimizer_groups = [
    # GNN encoder: randomly initialized → higher LR
    {'params': gnn_params,    'lr': base_lr * 2.5},
    # LoRA adapters: small perturbation from pre-trained → conservative LR
    {'params': lora_params,   'lr': base_lr * 0.3},   # was 0.5× in v5.3
    # Fusion + classifier: new components, start from base LR but with 0.5× cap
    {'params': fusion_params, 'lr': base_lr * 0.5},   # fixed from v5.2 (was 1.0×)
    # Everything else (JK weights, poolers, etc.): base LR
    {'params': other_params,  'lr': base_lr},
]
```

**Why LoRA 0.3× (reduced from 0.5×):** With 8 windows × 44,420 contracts, the LoRA weights
receive 8× more gradient signal per contract than in single-window training. Without a
compensating LR reduction, LoRA would effectively update 8× faster. Reducing to 0.3× keeps
LoRA update magnitude comparable to v5.2 on a per-contract basis.

---

## Phase 4: Data Quality Fixes

### 4.1 DoS Augmentation: ~500 Clean Single-Label Contracts

**Goal:** Break the 99% DoS→Reentrancy co-occurrence signal.

**Source:** SmartBugs SWC-128 (block gas limit) or hand-written contracts:
```solidity
// Example: DoS without Reentrancy
// ETH distribution loop — reverts if one recipient can't accept ETH
pragma solidity ^0.8.0;
contract Distributor {
    address[] public recipients;
    function distribute() external payable {
        for (uint i = 0; i < recipients.length; i++) {
            recipients[i].transfer(msg.value / recipients.length);
            // Any recipient can add a revert() in fallback to DoS this
        }
    }
    // NOTE: No reentrancy possible because we don't read/write state after call
    // AND we don't check re-entry. This is DoS-only, not Reentrancy.
}
```

**Verification:** Run each candidate contract through Slither and confirm:
- Slither reentrancy detector: NO
- Slither gas-estimation: WARN on loop with external call
- Label: DoS=1, Reentrancy=0, all others=0

**Target:** 500 clean DoS-only contracts. This would bring DoS training positives from
257 to 757 — still sparse but 3× improvement. The key benefit is breaking the co-occurrence:
with 500 DoS-without-Reentrancy examples, the 99% co-occurrence drops to ~40%.

### 4.2 Timestamp Augmentation: ~500 Clean Single-Label Contracts

**Goal:** Provide direct Timestamp signal without other co-occurring vulnerabilities.

**Example pattern:**
```solidity
pragma solidity ^0.8.0;
contract RandomLottery {
    // Bad: uses block.timestamp as randomness source
    function pickWinner(address[] memory participants) external view returns (address) {
        uint256 randomIndex = uint256(block.timestamp) % participants.length;
        return participants[randomIndex];
    }
}
```

**Verification:** Run through Slither:
- Timestamp detector: WARN
- No other detectors triggering
- Label: Timestamp=1, all others=0

### 4.3 Verify CEI Augmentation in Deduped CSV

Confirm the 50 CEI augmentation pairs (50 safe + 50 vulnerable Reentrancy) in
`ml/data/processed/multilabel_index_deduped.csv`:

```python
import pandas as pd
df = pd.read_csv('ml/data/processed/multilabel_index_deduped.csv')
# Find CEI augmentation entries (contract hash starts with 'CEI_' prefix or in augmentation list)
aug = df[df['contract_hash'].str.startswith('CEI_')]
assert len(aug) == 100, f"Expected 100 CEI entries, got {len(aug)}"
safe_count = aug[aug['Reentrancy'] == 0].shape[0]
vuln_count = aug[aug['Reentrancy'] == 1].shape[0]
print(f"CEI augmentation: {safe_count} safe, {vuln_count} vulnerable")
```

### 4.4 Per-Class Oversampling for DoS and Timestamp

**File:** `ml/src/datasets/dual_path_dataset.py`

For the train split only, apply oversampling on the DataLoader level:
```python
from torch.utils.data import WeightedRandomSampler

def build_sampler(dataset, target_classes=[0, 8]):  # 0=DoS, 8=Timestamp
    """
    Over-sample contracts with rare target classes to appear ~3× more often.
    """
    weights = torch.ones(len(dataset))
    for idx in range(len(dataset)):
        labels = dataset.labels[idx]
        for cls in target_classes:
            if labels[cls] == 1:
                weights[idx] = 3.0
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
```

This ensures each epoch, DoS and Timestamp positives appear ~3× more often than their
natural frequency, without introducing duplicate gradients for other classes.

### 4.5 Co-occurrence Penalty (Experimental)

**Status:** Experimental — implement as an optional flag, default disabled.

The idea: penalize the loss when DoS and Reentrancy BOTH fire above threshold in inference,
unless the contract actually demonstrates both. In practice, add a soft regularization term:

```python
# In trainer.py, after main loss computation:
if use_cooccurrence_penalty:
    # probs: [B, 10] after sigmoid
    dos_prob = probs[:, DOS_IDX]      # [B]
    reen_prob = probs[:, REEN_IDX]    # [B]
    dos_label = targets[:, DOS_IDX]
    reen_label = targets[:, REEN_IDX]
    # Penalize when model fires BOTH but only one is labeled
    both_fired = dos_prob * reen_prob
    should_be_separate = (dos_label != reen_label).float()
    cooc_penalty = (both_fired * should_be_separate).mean()
    total_loss += 0.1 * cooc_penalty
```

This is a soft constraint, not a hard one — it discourages the model from always co-firing
DoS and Reentrancy but doesn't prevent it when both are truly present.

---

## Phase 5: Re-extraction and Cache Rebuild

Sequential steps (each depends on the previous):

### 5.1 Re-extract Graphs with Phase 0 Fixes

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/reextract_graphs.py \
    --workers 16 \
    --output-dir ml/data/graphs \
    --index ml/data/processed/multilabel_index_deduped.csv \
    --feature-schema-version v4
```

**Expected changes vs v3 graphs:**
- feat[1]: was `pure` (0.02% non-zero) → `uses_block_globals` (expected 15–25% non-zero)
- feat[4]: `ext_calls` now includes Transfer/Send (will increase for ~15% of contracts)
- feat[6]: `loc` now log1p-normalized, values in [0, 1] instead of [0, 2538]
- feat[9]: `return_ignored` now correctly non-zero for contracts with ignored return values

### 5.2 Validate Graph Dataset

```bash
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py \
    --index ml/data/processed/multilabel_index_deduped.csv \
    --graph-dir ml/data/graphs \
    --check-dim 12 \
    --check-schema-version v4 \
    --check-loc-normalized \
    --check-uses-block-globals-nonzero
```

Spot checks to add:
- Load 10 known Timestamp contracts: confirm feat[1]=1.0 (uses_block_globals)
- Load 10 known IntegerUO 0.8.x contracts: confirm feat[8]=1.0 (in_unchecked)
- Load 10 known Transfer-DoS contracts: confirm feat[4]>0 (ext_calls)
- Confirm all feat[6] values are in [0, 1]

### 5.3 Re-tokenize All Contracts with Windowed Tokenizer

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/tokenize_contracts.py \
    --index ml/data/processed/multilabel_index_deduped.csv \
    --source-dir ml/data/raw_contracts \
    --output-dir ml/data/tokens_v2 \
    --max-len 512 --stride 256 --max-windows 8 \
    --schema-version v2
```

### 5.4 Rebuild Deduped CSV (Labels Stable)

The labels in `multilabel_index_deduped.csv` are derived from BCCC directory assignments.
They have not changed. Only need to update the CSV if augmentation contracts are added
(DoS, Timestamp, additional CEI pairs from Phase 4).

Add augmentation rows to CSV:
```bash
python ml/scripts/add_augmentation.py \
    --csv ml/data/processed/multilabel_index_deduped.csv \
    --dos-contracts ml/data/augmentation/dos_clean/ \
    --timestamp-contracts ml/data/augmentation/timestamp_clean/ \
    --output ml/data/processed/multilabel_index_v6.csv
```

### 5.5 Rebuild Cache

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/create_cache.py \
    --index ml/data/processed/multilabel_index_v6.csv \
    --graph-dir ml/data/graphs \
    --token-dir ml/data/tokens_v2 \
    --output ml/data/cached_dataset_v6.pkl
```

Verify:
```bash
python -c "
import pickle, torch
with open('ml/data/cached_dataset_v6.pkl', 'rb') as f:
    ds = pickle.load(f)
s = ds[0]
assert s['input_ids'].shape == (8, 512), 'Wrong token shape'
assert s['x'].shape[1] == 12, 'Wrong feature dim'
assert s['x'][:, 6].max() <= 1.0, 'loc not normalized'
print(f'Cache OK: {len(ds)} samples')
print(f'Sample feat[1] (uses_block_globals): {s[\"x\"][:, 1].sum().item():.0f} non-zero nodes')
print(f'Sample feat[6] (loc normalized) max: {s[\"x\"][:, 6].max().item():.3f}')
"
```

---

## Phase 6: Training and Evaluation

### 6.1 Launch v6.0 Training

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v6.0-20260517 \
    --experiment-name sentinel-v6 \
    --cache-path ml/data/cached_dataset_v6.pkl \
    --loss-fn asl \
    --asl-gamma-neg 4.0 \
    --asl-gamma-pos 1.0 \
    --asl-clip 0.05 \
    --label-smoothing 0.05 \
    --epochs 100 \
    --early-stop-patience 30 \
    --gradient-accumulation-steps 8 \
    --eval-threshold 0.35 \
    --warmup-epochs 5 \
    --gnn-lr-mult 2.5 \
    --fusion-lr-mult 0.5 \
    --lora-lr-mult 0.3 \
    --gnn-hidden-dim 256 \
    --lora-r 32 \
    --lora-alpha 64 \
    --model-version v6.0
```

### 6.2 Training Health Monitors

At epoch 1, check:
- GNN gradient share ≥ 15% (formula: RMS GNN grads / RMS all grads)
- Phase 2 JK weight ≥ 5% (CF phase is meaningful)
- Phase 3 JK weight ≥ 5% (RC phase is meaningful)
- uses_block_globals feature gradient non-zero for Timestamp-labeled batches
- return_ignored feature gradient non-zero for MishandledException-labeled batches

At epoch 5 (after warmup):
- Loss should be declining (< 0.75 from initial ~0.85)
- DoS class F1 > 0.0 (was 0 at epoch 1 in v5.2 due to starvation)
- Reentrancy F1 < 0.50 at threshold=0.35 (if higher, likely overfitting external-call proxy)

### 6.3 Per-Class F1 Targets

| Class | v5.2 Tuned F1 | v6.0 Target | Key Fix |
|-------|---------------|-------------|---------|
| IntegerUO | 0.732 | ≥ 0.75 | Windowed tokenization + hidden_dim=256 |
| GasException | 0.407 | ≥ 0.45 | Deeper GNN + better CF signal |
| Reentrancy | 0.322 | ≥ 0.40 | RC1 fix + augmentation + 2nd CF layer |
| MishandledException | 0.342 | ≥ 0.50 | return_ignored fix (was always 0) |
| UnusedReturn | 0.238 | ≥ 0.45 | return_ignored fix (was always 0) |
| Timestamp | 0.174 | ≥ 0.30 | uses_block_globals feature + windowed |
| DenialOfService | 0.329 | ≥ 0.35 | Transfer/Send fix + augmentation |
| CallToUnknown | 0.284 | ≥ 0.35 | Windowed tokenization |
| TOD | 0.283 | ≥ 0.30 | uses_block_globals (block.number) + windowed |
| ExternalBug | 0.262 | ≥ 0.30 | Cleaner training data + deeper GNN |
| **Macro avg** | **0.3422** | **≥ 0.43** | All fixes combined |

### 6.4 Behavioral Gates (Primary Pass/Fail Criterion)

Run `ml/scripts/manual_test.py` after tuning:
- **Detection rate:** ≥ 80% (currently 36%)
- **Safe specificity:** ≥ 80% (currently 33%)
- **No class always-on:** no single class fires on every input regardless of content

These gates are the ONLY honest judge of model quality. Val F1 ≥ 0.43 is a necessary but
not sufficient condition.

### 6.5 Post-Training Steps

1. Run `ml/scripts/tune_threshold.py` on val split to find per-class optimal thresholds
2. Run `ml/scripts/manual_test.py` with tuned thresholds
3. If behavioral gates PASS:
   - Save thresholds to `ml/checkpoints/v6.0-YYYYMMDD_thresholds.json`
   - Export ONNX proxy MLP for ZKML pipeline (Module 2)
   - Tag checkpoint as production checkpoint
   - Update MEMORY.md active checkpoint
   - Push to origin/main
4. If behavioral gates FAIL:
   - Identify failing class pattern from manual_test.py output
   - Diagnose: co-occurrence residual? feature still missing? LR imbalance?
   - Apply targeted fix and run v6.1 (smaller scope than full v6)

---

## Implementation Order and Dependencies

```
Phase 0 (graph_extractor.py fixes)
    ↓
Phase 5.1 (re-extraction — CONSUMES Phase 0 fixes)
    ↓
Phase 5.2 (validate)
    ↓
Phase 1.1–1.4 (windowed tokenization)
    ↓
Phase 5.3 (re-tokenize — CONSUMES Phase 1)
    ↓
Phase 4.1–4.3 (augmentation data prep — PARALLEL with phases above)
    ↓
Phase 5.4 (rebuild CSV — CONSUMES Phase 4)
    ↓
Phase 5.5 (rebuild cache — CONSUMES all Phase 5 above)
    ↓
Phase 2 (model architecture — code changes, no data dependency)
Phase 3 (training config — code changes, no data dependency)
    ↓
Phase 6.1 (training — CONSUMES Phase 5.5 cache + Phase 2+3 code)
```

Phases 2 and 3 can be coded in parallel with the data pipeline steps (Phases 0, 5, 1, 4).
The training launch (Phase 6) must wait for both the cache rebuild AND the code changes.
