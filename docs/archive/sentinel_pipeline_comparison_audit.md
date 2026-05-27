# SENTINEL Offline vs Online Pipeline — Full Comparison Audit

## Executive Summary

The offline (training) and online (inference) pipelines share a **canonical graph extraction module** (`ml/src/preprocessing/graph_extractor.py`) which was explicitly refactored to prevent divergence. However, the **tokenization paths** have critical differences that affect feature parity. The `compare_pipelines.py` script was **not found** in the uploaded zip — it must be created or provided separately.

---



##  Tokenization: DIFFERENT (Critical Divergence)

This is where the pipelines differ most significantly.

###  Offline Tokenization (Training)

**Script**: `ml/scripts/retokenize_windowed.py`

| Aspect | Value |
|--------|-------|
| Tokenizer model | `microsoft/codebert-base` |
| Output shape | `[4, 512]` (always exactly 4 windows) |
| Method | `tokenizer(code, max_length=512, stride=256, return_overflowing_tokens=True)` |
| Window selection | If W > 4: `np.linspace(0, W-1, 4)` sub-sampling |
| Padding | If W < 4: zero-padded windows (attention_mask=0) |
| Hash key | `get_contract_hash(sol_path.relative_to(PROJECT_ROOT))` — path-based MD5 |
| File format | `.pt` dict with `input_ids [4,512]`, `attention_mask [4,512]`, `num_windows`, `stride` |

###  Online Tokenization (Inference)

**Module**: `ml/src/inference/preprocess.py` → `ContractPreprocessor`

| Aspect | `process_source()` (single) | `process_source_windowed()` (sliding) |
|--------|---------------------------|--------------------------------------|
| Tokenizer | `microsoft/codebert-base` | `microsoft/codebert-base` |
| Output shape | `[1, 512]` | `list[dict]` each `[1, 512]` |
| Method | `tokenizer(code, max_length=512, truncation=True, padding="max_length")` | Manual sliding: encode without special tokens → slide by stride → add [CLS]/[SEP] per window |
| Max windows | N/A (1 window) | `max_windows=8` (default) |
| Window overlap | None | stride=256 |
| Hash key | `get_contract_hash_from_content(source_code)` — **content-based** MD5 | Same |
| Padding | Standard max_length | Manual pad to 512 per window |

###  Critical Tokenization Differences

| # | Difference | Offline (Training) | Online (Inference) | Impact |
|---|-----------|-------------------|-------------------|--------|
| **T1** | **Number of windows** | Fixed at **4** | Variable up to **8** (default) | ⚠️ **CRITICAL** — model trained on 4 windows, but inference may produce up to 8. See below. |
| **T2** | **Window selection for long contracts** | `linspace` sub-sampling (evenly spaced) | Sequential sliding (start-to-end) | ⚠️ Different windows selected for same contract — different token coverage. |
| **T3** | **CLS/SEP framing** | HuggingFace tokenizer handles [CLS]/[SEP] automatically per window via `return_overflowing_tokens` | Manual: `encode(add_special_tokens=False)` → prepend [CLS], append [SEP] per window | ✅ Should be equivalent after fix E1 (C4), but implementation path differs. |
| **T4** | **Hash method** | Path-based MD5 | Content-based MD5 | ⚠️ Same source at different paths = different hash offline, same hash online. Cache keys don't match. |
| **T5** | **Output tensor format** | `[4, 512]` stacked tensor in .pt file | List of `[1, 512]` dicts | Different but handled by Predictor (iterates list). |
| **T6** | **Window aggregation** | Not applicable (all 4 fed to model as `[B, 4, 512]`) | Per-window forward pass → max-pool across windows | ⚠️ **CRITICAL** — see analysis below. |

###  Window Count Mismatch (T1) — Deep Analysis

**Training**: The model always receives exactly `[B, 4, 512]` token input. DualPathDataset loads `[4, 512]` tensors. The DataLoader stacks them to `[B, 4, 512]`. TransformerEncoder processes all 4 windows in one fused call.

**Inference (windowed path)**: `predict_source()` calls `process_source_windowed()` which can return up to 8 windows. Then `_score_windowed()` runs a **separate forward pass per window**, each with the same graph but a single `[1, 512]` token input. Results are max-pooled across windows.

**This is a fundamentally different execution path**:
- Training: `[B, 4, 512]` → reshape `[B*4, 512]` → one GraphCodeBERT call → reshape `[B, 4*512, 768]` → WindowAttentionPooler
- Inference: per-window `[1, 512]` → one GraphCodeBERT call each → `[1, 512, 768]` → per-window sigmoid → max across windows

The model was **never trained** with the per-window-scoring-then-max-pool pattern. During training, all 4 windows are processed simultaneously with cross-window attention through the WindowAttentionPooler's learned attention weights. At inference, windows are scored independently — the WindowAttentionPooler never sees multiple windows simultaneously.

**Severity**: High. The aggregation strategy (learned attention pooler vs max-pool) differs. The model may produce different confidence distributions than training.

###  Window Selection Mismatch (T2) — Deep Analysis

For a contract with 8 raw windows:
- **Offline**: `linspace(0, 7, 4)` → windows [0, 2, 5, 7] — covers start, early-mid, late-mid, end
- **Online**: sequential sliding → windows [0, 1, 2, 3, 4, 5, 6, 7] (all 8 if max_windows=8)

The model was trained on the linspace-selected windows. At inference, it may see windows it never saw during training (e.g., window 1 or 6), and the window attention weights learned during training are calibrated for 4 evenly-spaced windows, not 8 sequential ones.

###  Hash Mismatch (T4) — Deep Analysis

- **Offline**: `get_contract_hash(sol_path.relative_to(PROJECT_ROOT))` — hash of the **path string**
- **Online**: `get_contract_hash_from_content(source_code)` — hash of the **source content**

These produce **completely different hashes** for the same contract. This is by design (online needs content-addressable caching), but it means:
- Inference cache keys can never match training data keys
- No direct pairing between online cache and offline dataset entries
- The `FEATURE_SCHEMA_VERSION` suffix is appended to both, so schema changes still invalidate both caches

---

## . Model Forward Path: Training vs Inference

### 4.1 Token Input Shape

| Phase | Shape | Notes |
|-------|-------|-------|
| Training | `[B, 4, 512]` | Windowed via DualPathDataset + collate_fn |
| Inference (single) | `[1, 512]` | No window dimension |
| Inference (windowed) | `[1, 512]` per window | Iterated, not batched |

###  GNN Prefix Injection

| Aspect | Training | Inference |
|--------|----------|-----------|
| Warmup suppression | `self._current_epoch < self.gnn_prefix_warmup_epochs` → prefix=None | `self.model._current_epoch = 9999` → always active |
| Prefix shape | `[B, K, 768]` | `[1, K, 768]` per window (same graph reused) |
| K value | 48 | 48 (from checkpoint config) |

**Issue**: In windowed inference, the same graph is reused for each window, but the prefix is computed fresh each time (same result). This is correct but wasteful — the prefix should be computed once and cached.

###  CrossAttentionFusion

| Aspect | Training | Inference (single) | Inference (windowed) |
|--------|----------|--------------------|---------------------|
| Key/value shape | `[B, 4*512, 768]` | `[1, 512, 768]` | `[1, 512, 768]` per window |
| Key_padding_mask | `[B, 2048]` | `[1, 512]` | `[1, 512]` per window |
| Node truncation | `_scatter_to_dense` truncates >1024 nodes silently | Same | Same |

**Critical**: During training, CrossAttentionFusion sees 2048 token positions (4 windows × 512). During inference, it sees only 512 positions (1 window). The attention pattern over a 2048-length sequence is different from a 512-length sequence — the query (GNN node) attends to a much larger context during training.

###  WindowAttentionPooler

| Aspect | Training | Inference (single) | Inference (windowed) |
|--------|----------|--------------------|---------------------|
| Input | `[B, 2048, 768]` | `[1, 512, 768]` | `[1, 512, 768]` per window |
| CLS positions | 4 CLS tokens at positions [0+K, 512+K, 1024+K, 1536+K] | 1 CLS at position K | 1 CLS at position K |
| Path taken | Multi-window path: learned attention over 4 CLS tokens | Single-window fallback: `token_embs[:, prefix_k, :]` | Single-window fallback per window |

**Issue**: The learned attention weights in `WindowAttentionPooler.attn` are trained to weight 4 window-CLS tokens. At inference with single-window input, the attention mechanism is bypassed entirely (fallback path). The transformer eye projection receives a single CLS token instead of a learned-attention-weighted combination — different signal composition than training.

---

##  Post-Processing Differences

| Aspect | Training | Inference |
|--------|----------|-----------|
| Loss function | ASL + BCE aux + label smoothing + per-class gamma/clip + DoS scaling | N/A |
| Sigmoid | Applied in loss (ASL logits interface) | Applied manually: `torch.sigmoid(logits)` |
| Thresholding | Per-class thresholds for eval metric | Per-class thresholds from JSON file (or 0.5 fallback) |
| Window aggregation | N/A (all windows processed together) | Max-pool across windows for per-class probs |

---



##  Critical Findings Summary

### 🔴 HIGH SEVERITY

| # | Finding | Location | Impact |
|---|---------|----------|--------|
| **H1** | **Solc version mismatch**: Online uses system solc (no pinning), offline uses solc-select pinned binaries. Different solc versions produce different ASTs → different graph features. | `preprocess.py` line 394: `config = GraphExtractionConfig()` with no solc_binary/solc_version | Silent feature drift — model trained on different features than it receives at inference. |
| **H2** | **Window aggregation strategy differs**: Training uses 4 simultaneous windows with learned WindowAttentionPooler. Inference uses per-window forward passes with max-pool. | `predictor.py` `_score_windowed()` | Model never trained with per-window-max-pool pattern — different probability calibration. |
| **H3** | **CrossAttentionFusion context length differs**: Training sees 2048 token positions (4×512). Inference sees 512. | `fusion_layer.py` called with different key/value lengths | Attention distribution over tokens is fundamentally different. |

### 🟡 MEDIUM SEVERITY

| # | Finding | Location | Impact |
|---|---------|----------|--------|
| **M1** | **Window count mismatch**: Training = 4 windows always. Inference = up to 8 windows. | `preprocess.py` `process_source_windowed(max_windows=8)` | Extra windows introduce tokens the model never saw in training. |
| **M2** | **Window selection strategy differs**: Training uses linspace sub-sampling; inference uses sequential sliding. | `retokenize_windowed.py` `_select_windows()` vs `preprocess.py` `_tokenize_sliding_window()` | For long contracts, different code regions are captured. |
| **M3** | **GraphExtractionConfig allow_paths not set online**: Without allow_paths, solc may fail to resolve imports in contracts that use relative imports. | `preprocess.py` line 394 | Some valid contracts may fail compilation at inference but succeeded offline. |
| **M4** | **InferenceCache validates single-window shape only**: `cache.py` line 109: `expected_tok = torch.Size([1, 512])` — cached windowed tokens with `[4, 512]` shape would be evicted. | `cache.py` `get()` | Windowed inference can't use the cache for multi-window tokens. |

### 🟢 LOW SEVERITY (Already Handled)

| # | Finding | Location | Impact |
|---|---------|----------|--------|
| **L1** | Hash method differs (path vs content) | `hash_utils.py` | By design; cache keys differ but schema versioning is consistent. |
| **L2** | `weights_only=False` in predictor.py checkpoint loading | `predictor.py` line 146 | Security concern (not pipeline parity). LoRA state dict requires it. |
| **L3** | edge_attr shape normalization in DualPathDataset | `dual_path_dataset.py` line 353 | `.squeeze(-1)` handles both [E] and [E,1] — correct. |
| **L4** | Dead `_compute_in_unchecked` code in graph_extractor | `graph_extractor.py` line 334 | Not called in feature vector; no impact. |

---
3. Online vs Offline Pipeline Discrepancies
DISCREPANCY-1: solc Version Pinning (Severity: CRITICAL)
Path
solc Strategy
Config
Offline (reextract_graphs.py)	Pragma detection → version-pinned binary	GraphExtractionConfig(solc_version=ver, solc_binary=path)
Offline (compare_pipelines.py)	Same as above	Same via _make_config()
Online (preprocess.py)	System PATH solc — NO version pinning	GraphExtractionConfig()

Code evidence:

reextract_graphs.py line 120+: detects pragma version, resolves binary from .solc-select/artifacts/
preprocess.py line 394: config = GraphExtractionConfig() — no arguments, no solc pinning
Impact: If the system solc version differs from the pragma-specified version:

Different solc versions produce different AST structures
Different AST → different graph topology (nodes, edges)
Different graph → different node features (complexity, loc, has_loop, etc.)
Different features → different GNN embeddings → different predictions
This is the single biggest risk for training↔inference divergence. A contract compiled with solc 0.4.26 during training but solc 0.8.31 during inference will produce a fundamentally different graph.

What compare_pipelines.py does: The G-series checks compare the output of version-pinned offline extraction vs system-solc online extraction. If the solc versions differ, G2 (feature values) will catch it. This is correct — it tests the actual production divergence.

Recommendation: preprocess.py._extract_graph() should implement pragma detection and version pinning, matching reextract_graphs.py. The compare_pipelines.py P1b check was intended to verify this but crashes (BUG-CP1).

DISCREPANCY-2: Windowed Tokenization Algorithm (Severity: HIGH — Known/Documented)
Aspect
Offline (retokenize_windowed.py)
Online (preprocess._tokenize_sliding_window)
Algorithm	HuggingFace return_overflowing_tokens=True	Manual encode + slide + re-frame
Window 0 format	[CLS] content... [SEP] [PAD]...	[CLS] content... [SEP] [PAD]...
Window 1+ format	content_overlap... (NO CLS, NO SEP)	[CLS] content_overlap... [SEP] [PAD]...
CLS at pos 0	Window 0 only	Every window
SEP at end	Window 0 only	Every window
Max windows	4	8
Sub-sampling	np.linspace	Truncation at max_windows

Code evidence:

retokenize_windowed.py line 198-206: tokenizer(code, ..., return_overflowing_tokens=True, ...)
preprocess.py line 530: self.tokenizer.encode(source_code, add_special_tokens=False) → manual framing
Impact: For multi-window contracts:

Window 0 token IDs: identical between offline and online
Window 1+ token IDs: fundamentally different — offline has raw overflow tokens (no CLS/SEP), online has proper framing
This means CodeBERT sees different input for windows 1+ during inference vs training
The online fix (Fix E1) is an improvement — CodeBERT was pre-trained to always see [CLS]...[SEP], so missing CLS in window 1+ degraded those windows' representations
What compare_pipelines.py does: W1-W9 checks fully document this discrepancy, correctly marking it as DIFF (known design difference) rather than FAIL. W2/W3 explicitly check for CLS/SEP presence per window.

Note: This discrepancy means the model was trained on "broken" multi-window tokens (windows 1+ without CLS/SEP), but inference uses "fixed" tokens (all windows with CLS/SEP). The model learned to handle the broken format; the fixed format may actually produce worse results for windows 1+ because the model never saw that pattern during training. This is a subtle form of train/test mismatch.

DISCREPANCY-3: Training vs Inference Window Aggregation (Severity: HIGH — Known/Documented)
Aspect
Training
Inference
Input format	[B, 4, 512] (batch of multi-window)	W × [1, 512] (separate per-window calls)
Window pooling	WindowCLSPooler (learned attention over W CLS embeddings)	max(sigmoid(logits)) per class across W windows
Forward passes per contract	1 (all windows in one pass)	W (one per window)
Graph sharing	Same graph used for all W windows simultaneously	Same graph re-used for each window (same batch)

Code evidence:

predictor.py line 420-431: _score_windowed() loops over windows, calls model(batch, input_ids, attention_mask) per window
dual_path_dataset.py line 390: torch.stack([t["input_ids"] for t in tokens]) → [B, W, 512]
Training collation: [B, 4, 512] → single forward pass through TransformerEncoder → WindowCLSPooler
Impact:

WindowCLSPooler's learned attention weights are never used at inference for multi-window contracts
Max-pooling probabilities is a fundamentally different aggregation than learned attention — it always keeps the "most alarming" window rather than weighting by learned importance
The model was trained to have WindowCLSPooler decide which windows matter; inference bypasses this mechanism entirely
What compare_pipelines.py does: O2 check correctly documents this as DIFF. M5/M5b tests both formats for crash safety but cannot test semantic equivalence without a trained checkpoint.

DISCREPANCY-4: Hashing Strategy (Severity: MEDIUM — Known/Documented)
Path
Hash Function
Key Format
Offline (graph/token files)	MD5(path.relative_to(PROJECT_ROOT))	a1b2c3...pt
Online (process via file path)	MD5(absolute_path) + "_" + FEATURE_SCHEMA_VERSION	d4e5f6..._v8
Online (process_source via string)	MD5(source_content) + "_" + FEATURE_SCHEMA_VERSION	g7h8i9..._v8

Code evidence:

retokenize_windowed.py line 72: get_contract_hash(sol_path.relative_to(PROJECT_ROOT))
preprocess.py line 191: f"{get_contract_hash(sol_path)}_{FEATURE_SCHEMA_VERSION}"
preprocess.py line 253: f"{content_hash}_{FEATURE_SCHEMA_VERSION}"
Impact: The hash values are completely different between offline and online for the same contract. This is not a problem for pipeline correctness (each pipeline is self-consistent), but:

Online-generated graphs/tokens cannot be paired with offline training data by filename
The inference cache (InferenceCache) uses the online hash format, so it will never match offline-generated .pt files
The FEATURE_SCHEMA_VERSION suffix is a good online safeguard (invalidates cache on schema change) but is absent from offline filenames
What compare_pipelines.py does: H1-H3b checks fully document this, correctly marking H3 as DIFF.

DISCREPANCY-5: predict(sol_path) Never Uses Sliding Window (Severity: MEDIUM)
API
Tokenization
Window Count
Predictor.predict(sol_path)	_tokenize() → [1, 512]	Always 1
Predictor.predict_source(src)	_tokenize_sliding_window() → list of [1, 512]	1 to 8

Code evidence:

predictor.py line 381: graph, tokens = self.preprocessor.process(sol_path) → _tokenize() → single window
predictor.py line 396: graph, windows = self.preprocessor.process_source_windowed(source_code) → sliding window
Impact: If the production API calls predict(sol_path) on a long contract (>512 tokens), the tail is silently truncated. Patterns in the truncated portion (e.g., withdrawal logic at line 400+) are completely invisible to the model. This is only an issue if the API endpoint uses predict() instead of predict_source().

What compare_pipelines.py does: O3 check correctly warns about this truncation risk for long contracts.

DISCREPANCY-6: Edge Type Count Mismatch (Severity: LOW)
Aspect
Offline (v7 era)
Current Schema (v8)
NUM_EDGE_TYPES	8 (types 0-7)	11 (types 0-10)
New edge types	—	CALL_ENTRY(8), RETURN_TO(9), DEF_USE(10)

Code evidence: graph_schema.py line 208: NUM_EDGE_TYPES: int = 11

The training data was re-extracted with v8 schema (including CALL_ENTRY, RETURN_TO, DEF_USE edges), but the predictor.py has a compatibility shim at lines 247-255 that resizes the edge embedding when a checkpoint has fewer edge types than the current schema. This means old checkpoints (trained with 8 edge types) can still be loaded, but the new edge type embeddings will be randomly initialized — the model never learned to use them.

4. Additional Findings
FINDING-1: _warmup() Does Not Exercise Prefix Injection (Severity: MEDIUM)
Code: predictor.py lines 343-377

The warmup dummy graph has only 2 nodes (both type_id=0=STATE_VAR). The select_prefix_nodes() function only selects FUNCTION/MODIFIER/CONSTRUCTOR/FALLBACK/RECEIVE nodes for prefix injection. With no eligible nodes in the dummy graph, the prefix code path is NEVER exercised during warmup. Bugs in gnn_to_bert_proj or prefix_type_embedding would survive warmup and only surface on the first real contract.

What compare_pipelines.py does: O5 correctly warns about this.

FINDING-2: weights_only=False in Predictor (Severity: MEDIUM — Security)
Code: predictor.py line 146

python

raw = torch.load(checkpoint, map_location=self.device, weights_only=False)
The DualPathDataset was fixed to use weights_only=True (line 316-317), but the predictor still uses weights_only=False because LoRA state dicts contain peft-specific classes that are blocked by the safe globals list. This is a known security risk — a malicious checkpoint could execute arbitrary code during loading.

FINDING-3: CFG Node Feature Inheritance Not Tested for Online Path
The BUG-C3 fix (CFG nodes inherit dims [1,3,4,5,9] from parent FUNCTION) is implemented in graph_extractor.py's _build_control_flow_edges() which passes parent_features=x_list[fn_idx]. Both offline and online paths call the same extract_contract_graph(), so they should produce identical inheritance. The G19 check validates this on the offline graph only (not the online graph), but G5 (edge_attr values) would catch any difference if the graphs diverge.

FINDING-4: FEATURE_SCHEMA_VERSION Documentation Inconsistency
Code: graph_schema.py

Line 160: FEATURE_SCHEMA_VERSION: str = "v8" (code value)
Line 119: "v7 — 11 features (in_unchecked dropped — BUG-L2)" (docstring says v7)
Line 433: "Human-readable labels for each node feature dimension (v7 — 11 dims)" (comment says v7)
The actual code version is "v8" (bumped when CALL_ENTRY/RETURN_TO/DEF_USE edges were added), but multiple comments and docstrings still say "v7". This is a documentation issue, not a functional bug — the code uses the correct "v8" value for cache invalidation.

FINDING-5: Online process() Uses Absolute Path for Hashing
preprocess.py line 191:

python

contract_hash = f"{get_contract_hash(sol_path)}_{FEATURE_SCHEMA_VERSION}"
Since sol_path is resolved to an absolute path, get_contract_hash(sol_path) hashes the absolute path string. The same contract at different filesystem locations will produce different hashes. This differs from the offline pipeline which hashes relative paths. The compare_pipelines.py H-series checks document this difference.

FINDING-6: DualPathDataset Excludes node_metadata from Batch
dual_path_dataset.py line 384:

python

_EXCLUDE = ["contract_hash", "contract_path", "contract_name",
            "node_metadata", "num_edges", "num_nodes", "y"]
The node_metadata list (used by _find_function_node() for prefix node selection) is excluded from the training batch. This means select_prefix_nodes() cannot use node_metadata to identify FUNCTION nodes during training — it must rely on type_id from x[:, 0]. The online path also excludes node_metadata from Batch.from_data_list(), so both paths are consistent. However, the G17 check verifies node_metadata is present on the online graph (which it is — just excluded during batching).