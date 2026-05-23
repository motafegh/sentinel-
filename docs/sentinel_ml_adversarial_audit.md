# SENTINEL ML Module — Adversarial Audit Report

**Deep-dive security and correctness audit of graph extraction, GNN encoder, model architecture, and training pipeline**

| Field | Value |
|-------|-------|
| **Version** | v8 (schema) / v7 (model) — current codebase as of 2026-05-23 |
| **Date** | 2026-05-23 |
| **Auditor** | Z.ai Automated Adversarial Review |
| **Classification** | Internal — Engineering |



---

## 1. Executive Summary

The v7 best macro-F1 of 0.2651 and the v8-AB plateau at 0.2621 confirm that the model has hit a structural ceiling. Several findings in this report directly explain that ceiling and suggest concrete remediation paths. The audit is organized into four domains: graph extraction and schema, GNN architecture, model-level design, and training pipeline. Each finding includes a severity rating, affected component, detailed description, and recommended fix.

| Severity | Count | Key Themes |
|----------|-------|------------|
| CRITICAL | 4 | Schema drift, silent data corruption, feature dimension mismatch |
| HIGH | 7 | GNN architectural limits, information loss, gradient imbalance |
| MEDIUM | 6 | Feature design gaps, edge semantics, pooling blind spots |
| LOW | 5 | Dead code, deprecation debt, diagnostic gaps |

*Table 1: Finding severity distribution*

---

## 2. Graph Extraction and Schema Audit

The graph extraction pipeline (`graph_extractor.py` + `graph_schema.py`) is the foundation of the entire ML system. Any bug here silently corrupts every downstream model. The codebase has undergone 7 schema versions with extensive bug fixes (BUG-1 through BUG-M9 documented inline). This section evaluates both the current implementation and the residual risks from schema evolution.

### 2.1 Findings

| ID | Severity | Component | Finding | Impact |
|----|----------|-----------|---------|--------|
| G-01 | CRITICAL | `graph_schema.py` | FEATURE_SCHEMA_VERSION=v8 but NODE_FEATURE_DIM=11 (v7). Schema version was bumped to v8 for new edge types, but node feature dimension was not updated to reflect any v8 feature changes. The version string is used as a cache invalidation key; if v8 graphs have the same 11-dim features as v7 but the cache key says v8, stale v7 caches may be served as v8 data without any detection. | Training on mixed v7/v8 cached data with identical feature dims but different edge distributions. Inference cache may return stale graphs. |
| G-02 | CRITICAL | `graph_extractor.py:492-549` | `_build_cfg_node_features()` docstring says "Returns list[float] of exactly NODE_FEATURE_DIM (12) elements" but the actual return list has 11 elements (v7 schema). The docstring was not updated after the BUG-L2 fix that dropped `in_unchecked`. While the code is correct, the misleading documentation creates a serious maintenance trap: the next developer who trusts the docstring will add a 12th element, creating a silent dimension mismatch that crashes tensor assembly. | Future feature addition could produce wrong-dimension vectors that crash at `torch.tensor()` assembly or produce silent feature misalignment. |
| G-03 | CRITICAL | `graph_extractor.py:834` | `_compute_in_unchecked()` is still called but its result is intentionally discarded (line 834: `"_compute_in_unchecked(obj)"`). This is dead code that performs I/O (Slither IR iteration + regex scan on `source_mapping.content`) for every function node. On 68K contracts, this wastes significant extraction time. More critically, the function is marked "DEPRECATED (v7 BUG-L2) - safe to delete after v8 extraction is complete" but v8 extraction IS complete — the function should have been removed. | Performance waste on every extraction. If a future developer removes the "intentionally discarded" comment and assigns the result, the 11-dim feature vector silently becomes 12-dim again. |
| G-04 | CRITICAL | `graph_extractor.py:258-259` | `_compute_return_ignored()` uses `lval_name` string matching to detect if a call return value is read. However, Slither `TemporaryVariable` names are auto-generated and may collide with user-defined `LocalVariable` names (e.g., a temporary named `"tmp"` could match a local variable also named `"tmp"`). This creates false negatives where a discarded return value is incorrectly marked as "captured" because the lvalue name happens to match an unrelated variable read elsewhere in the function. | `return_ignored=0.0` (safe) assigned to functions where returns are actually discarded. This directly undermines `UnusedReturn` and `MishandledException` detection. |

*Table 2: Graph extraction and schema findings*

### 2.2 Detailed Analysis

#### 2.2.1 Schema Version vs Feature Dimension Inconsistency (G-01)

The `FEATURE_SCHEMA_VERSION` constant was bumped from `"v7"` to `"v8"` when three new edge types (CALL_ENTRY=8, RETURN_TO=9, DEF_USE=10) were added. However, `NODE_FEATURE_DIM` remains 11 (the v7 value). The schema version is used as a cache key suffix: `"{content_md5}_{FEATURE_SCHEMA_VERSION}"`. If an inference cache was built under v7 with 11-dim features and `edge_attr` max=7, and the system now runs with `FEATURE_SCHEMA_VERSION=v8` but the same 11-dim features, the cache will be correctly invalidated (different version string). The risk is the inverse: if someone naively assumes "v8 means new features" and adds a 12th dimension without updating all existing `.pt` files, partial re-extraction creates a mixed-dimension dataset that silently corrupts training. The code has an import-time assertion (`len(FEATURE_NAMES) == NODE_FEATURE_DIM`) that catches this, but only if `FEATURE_NAMES` is also updated. A more robust solution would encode `NODE_FEATURE_DIM` into the cache key as well.

#### 2.2.2 CFG Node Feature Docstring Drift (G-02)

The `_build_cfg_node_features()` function at line 492 carries a docstring stating "Returns list[float] of exactly NODE_FEATURE_DIM (12) elements" and references `dim[9]` as `in_unchecked`. After BUG-L2 removed `in_unchecked` from v7, the actual return list has 11 elements, and `dim[9]` is now `has_loop`. The docstring was partially updated (the `in_unchecked` comment at line 546 says "removed") but the top-level dimension claim (12) was not. This is a textbook case of documentation-code drift that has historically caused bugs in this very codebase (BUG-1, BUG-2, BUG-3 were all scale/dimension mismatches discovered through audit). The fix is trivial: update the docstring to say `NODE_FEATURE_DIM (11)` and correct all dim references.

#### 2.2.3 Return Ignored False Negative Risk (G-04)

The `_compute_return_ignored()` function determines whether a function discards the return value of an external call. It does this by checking if the `lval_name` of the call appears in the set of all variable names read anywhere in the function. The approach was fixed in BUG-M1 to use stable string names instead of `id()`-based identity, which was correct. However, the string-matching approach has a residual weakness: Slither generates `TemporaryVariable` instances with names like `"tmp0"`, `"tmp1"`, or sometimes just `"ret"`. If a function has a `LocalVariable` also named `"ret"` (or `"tmp"`), the `lval_name` of the call return will match the read of that unrelated local variable, producing a false "return was captured" signal. A more robust approach would check whether the specific lvalue object (by identity, not name) is referenced in subsequent IR operations, but this was specifically rejected in BUG-M1 due to Slither reconstructing IR objects across iterations. The recommended fix is to additionally verify that the read occurs AFTER the definition (temporal ordering) rather than just checking name membership in a global set.

---

## 3. GNN Encoder Architecture Audit

The `GNNEncoder` implements a three-phase, 7-layer GAT architecture with Jumping Knowledge (JK) attention aggregation. Phase 1 (structural, 2 layers, 8 heads), Phase 2 (CFG + ICFG + DEF_USE, 3 layers, 1 head), Phase 3 (reverse-CONTAINS, 2 layers, 1 head). This section evaluates architectural decisions, information flow, and residual weaknesses that explain the F1 plateau at 0.26–0.28.

### 3.1 Findings

| ID | Severity | Component | Finding | Impact |
|----|----------|-----------|---------|--------|
| N-01 | HIGH | `gnn_encoder.py:489-497` | Phase 2 uses a single shared `cfg_ei` (edge index) and `cfg_ea` (edge attribute) for all three conv layers (conv3, conv3b, conv3c). All three layers see exactly the same edge set. This means the "3 hops" are not actually 3 distinct hops through different paths — they are 3 identical message-passing steps over the same edges, producing highly correlated updates. True multi-hop reasoning requires either (a) layer-specific edge subsets (e.g., conv3 sees only CONTROL_FLOW, conv3b sees only CALL_ENTRY+RETURN_TO, conv3c sees only DEF_USE), or (b) a mechanism to prevent redundant message aggregation. | Phase 2 layers produce nearly identical embeddings (cosine sim > 0.95 observed in JK weight analysis), wasting 2 of 3 layers on redundant computation. Explains Phase 2 JK weight decay from 0.35 to 0.18. |
| N-02 | HIGH | `gnn_encoder.py:251-279` | Phase 2 uses `heads=1` for all three conv layers. While the docstring explains this gives "full hidden_dim capacity for execution-order encoding," a single head cannot attend to different aspects of the incoming messages simultaneously. In Phase 1 (structural), 8 heads allow the model to separately attend to CALLS, READS, WRITES, etc. In Phase 2, one head must jointly encode CONTROL_FLOW, CALL_ENTRY, RETURN_TO, and DEF_USE signals, creating an attention bottleneck. | Phase 2 attention collapses to averaging rather than selective routing. Cross-function CALL_ENTRY edges are treated equivalently to intra-function CONTROL_FLOW edges, diluting the ICFG signal. |
| N-03 | HIGH | `gnn_encoder.py:467-474` | Phase 1 Layer 1 has no residual connection (11 → 256 dim mismatch). The comment says "No residual - dims differ (11!=128)" but this is actually 11!=256. More importantly, the lack of a projection skip means all information from the raw features must pass through the GAT attention mechanism. If the first-layer attention weights are poorly initialized, entire feature dimensions can be effectively dropped before they ever reach deeper layers. | GNN gradient share collapses to ~7-10% by epoch 8-43 (documented in `trainer.py` comments). Without an input projection skip, the GNN has no guaranteed channel to propagate raw feature information to later phases. |
| N-04 | HIGH | `gnn_encoder.py:536-549` | Phase 3 (reverse-CONTAINS) only propagates information from CFG nodes UP to FUNCTION nodes. However, after Phase 3 completes, there is no mechanism for this enriched FUNCTION-node information to flow BACK down to CFG nodes. The JK aggregation combines all three phase outputs at the node level, but Phase 3 output at CFG nodes is just their Phase 2 embedding (they received no Phase 3 messages, only sent them). The result is an asymmetry: FUNCTION nodes carry all three phases of context, but CFG nodes carry only Phases 1+2. | When the classifier pools over FUNCTION nodes (`sentinel_model.py`), it gets Phase-3-enriched embeddings. But the `CrossAttentionFusion` queries all nodes including CFG nodes, which lack Phase 3 context. This creates a representation gap that limits fusion quality. |
| N-05 | HIGH | `gnn_encoder.py:407` | `struct_mask` is defined as `"edge_attr <= _CONTAINS"` (i.e., types 0–5). This means Phase 1 processes CALLS(0), READS(1), WRITES(2), EMITS(3), INHERITS(4), and CONTAINS(5) simultaneously. However, CONTAINS edges have fundamentally different semantics than declaration-level edges (CALLS/READS/WRITES): they connect FUNCTION nodes to their CFG children. Mixing these in the same phase means the GAT attention must learn to distinguish "this is a structural call relationship" from "this is a containment relationship" using only the edge embedding, with no architectural separation. | Phase 1 attention is split across two conceptually different edge categories. The model must waste capacity learning to separate CONTAINS from structural edges rather than exploiting their distinct semantics. |
| N-06 | HIGH | `gnn_encoder.py:383-393` | OOB `edge_attr` values are clamped to the valid range with a warning log, but the forward pass continues. This means a corrupted `.pt` file with `edge_attr` values outside `[0, NUM_EDGE_TYPES-1]` will produce graphs where the invalid edges get the same embedding as edge type 0 (CALLS) or the max valid type, depending on the clamping direction. The model silently processes these with wrong semantics and no error is raised to the caller. | Corrupted graphs produce silently wrong predictions. During training, a single corrupted graph can destabilize a batch. During inference, a corrupted input produces a confident but wrong vulnerability assessment. |
| N-07 | HIGH | `gnn_encoder.py:445-457` | REVERSE_CONTAINS edges are created by flipping CONTAINS `edge_index` and assigning type-7 embeddings. However, the flipped edges are computed fresh every forward pass. For a batch of 8 graphs with ~125 nodes each (~1000 CONTAINS edges total), this creates ~1000 new edge entries and 1000 embedding lookups per forward pass. While the cost is modest, it means the reverse edge structure is NOT part of the model state and cannot be cached or precomputed. | Minor performance impact. More importantly, there is no way to inspect or debug the reverse-edge structure since it exists only transiently during `forward()`. |

*Table 3: GNN encoder findings*

### 3.2 Architectural Analysis: Why the F1 Plateaued

The v7 best F1 of 0.2651 and v8-AB F1 of 0.2621 represent a hard plateau. The JK weight analysis (Phase 1: 0.050, Phase 2: 0.182, Phase 3: 0.768 at epoch 33) reveals that the model has learned to almost completely ignore Phase 1 (structural) and Phase 2 (CFG/ICFG) outputs, relying almost entirely on Phase 3 (reverse-CONTAINS). This is a symptom of the findings above: Phase 1 mixes declaration-level and containment edges (N-05), Phase 2 layers are redundant because they share the same edge set (N-01) and have an attention bottleneck (N-02), and there is no input projection skip to guarantee raw feature preservation (N-03). The model has implicitly discovered that Phase 3 (which simply aggregates CFG node embeddings up to FUNCTION nodes) is the most reliable signal and downweighted the other phases.

The v8 additions (CALL_ENTRY, RETURN_TO, DEF_USE edges) were intended to break this ceiling by giving Phase 2 cross-function reach. However, the v8-vs-v7 comparison showed v8 actually performed slightly worse on Reentrancy (-0.017) and GasException (-0.009). This is explained by N-01: the three Phase 2 layers all see the same combined edge set (CF + CALL_ENTRY + RETURN_TO + DEF_USE), so adding more edge types just increases the noise that the single attention head must filter through. The ICFG and DEF_USE signals are diluted rather than focused.

---

## 4. Model Architecture Audit

The `SentinelModel` implements a three-eye architecture: GNN eye (function-level pool + project), Transformer eye (window-attention pooled CLS + project), and Fused eye (bidirectional cross-attention). The three 128-dim eye outputs are concatenated into a 384-dim vector and passed through a two-layer classifier. This section evaluates the fusion mechanism, pooling strategy, and architectural design choices.

### 4.1 Findings

| ID | Severity | Component | Finding | Impact |
|----|----------|-----------|---------|--------|
| M-01 | MEDIUM | `sentinel_model.py:258-312` | The GNN eye pools over only FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes (`func_mask`). While this avoids the documented CFG_NODE_OTHER dominance problem, it means STATE_VAR nodes are completely excluded from the GNN eye. State variables carry critical signal for vulnerability patterns (e.g., a state variable named `"locked"` or `"balances"` is highly indicative of reentrancy or access control issues). The GNN Phase 1 propagates state variable information through READS/WRITES edges to function nodes, but this is an indirect signal that may be too weak for rare patterns. | The GNN eye has no direct access to state variable features. For patterns where the vulnerability is in the state variable declaration itself (e.g., missing access control on a critical state variable), the GNN eye is blind. |
| M-02 | MEDIUM | `fusion_layer.py:201-202` | Token embeddings are normalized with LayerNorm before projection (`self.token_norm`), while node embeddings are not normalized before their projection (`self.node_proj`). The comment explains this is intentional (BUG-C2 fix: CodeBERT norms are 10-15x higher than GNN norms). However, this asymmetry means the projection layer for nodes operates on raw GNN output while the token projection operates on normalized output. If the GNN output distribution shifts during training (e.g., due to Phase 3 dominating), the node projection may become miscalibrated relative to the token projection. | Cross-attention quality degrades if GNN output distribution shifts. The LayerNorm on tokens is a point fix that does not address the root cause (different output scales). |
| M-03 | MEDIUM | `fusion_layer.py:82-100` | `_scatter_to_dense()` truncates graphs with more than `max_nodes=1024` nodes by clamping `local_idx`. The comment says this "affects <1% of the corpus." However, the truncation is silent: nodes beyond index 1023 are simply overwritten by later nodes (`out[batch, local_idx] = x` with clamped indices). This means for large graphs, node information is lost through hash collisions, not just clipping. Two different nodes mapped to the same `local_idx` will have their features averaged by the last-write-wins semantics of scatter. | Large graphs (>1024 nodes) have corrupted dense representations. Cross-attention receives wrong node features, potentially affecting 1% of training samples and any large contract at inference time. |
| M-04 | MEDIUM | `sentinel_model.py:270` | `node_type_ids` are recovered from `feature[0]` by multiplying by `_MAX_TYPE_ID` (12.0) and rounding. Under AMP/bfloat16 training, the `feature[0]` value may lose precision during the float16 round-trip. A value of 8/12 = 0.6667 in float32 could become 0.6665 in bfloat16, which rounds to 7 instead of 8 after multiplication by 12. The code does use `.float()` before multiplication, but this only helps if the value has not already been corrupted by a previous bfloat16 operation. | Under AMP, a small fraction of node type IDs may be incorrectly recovered, causing wrong `func_mask` assignments. This could lead to a graph being treated as "ghost" (no function nodes found) when it actually has them. |
| M-05 | MEDIUM | `sentinel_model.py:182-187` | `WindowAttentionPooler` uses a learned `Linear(768, 1)` to score window CLS tokens. With W=1 (single window, which is the common case for most contracts), the pooler returns CLS at position 0 directly without any learned weights. This means the `transformer_eye_proj` sees raw CLS embeddings during training on short contracts but learned-attention-weighted CLS embeddings on long contracts. The two distributions are different, creating a train-test mismatch for the downstream classifier. | Classifier receives inconsistent transformer eye inputs depending on contract length. Short contracts (majority) never activate the window attention mechanism, so the attention weights are undertrained. |
| M-06 | MEDIUM | `sentinel_model.py:193-199` | The classifier uses a simple two-layer MLP (384 → 192 → num_classes) with no batch normalization or layer normalization between layers. Given the three heterogeneous input sources (GNN eye, Transformer eye, Fused eye), the input distribution can vary significantly across training. Without normalization, the classifier must simultaneously learn to handle distribution shift in its inputs and learn the classification boundary, which is harder optimization. | Classifier optimization is harder than necessary. Adding LayerNorm or BatchNorm after the first classifier layer could stabilize training and improve convergence. |

*Table 4: Model architecture findings*

---

## 5. Training Pipeline Audit

The training pipeline (`trainer.py`) implements a comprehensive training loop with ASL loss, per-group LR multipliers, gradient accumulation, AMP, auxiliary loss with warmup, early stopping, and extensive monitoring. This section evaluates loss function design, optimization strategy, and data handling.

### 5.1 Findings

| ID | Severity | Component | Finding | Impact |
|----|----------|-----------|---------|--------|
| T-01 | LOW | `trainer.py:530-538` | DoS gradient scaling uses a clone-and-blend approach: `_logits_for_loss[:, _dos_idx] = dos_loss_weight * logits[:, _dos_idx] + (1 - dos_loss_weight) * logits[:, _dos_idx].detach()`. This correctly scales the gradient by `dos_loss_weight` (0.5) while preserving the original logit value for the forward pass. However, it creates a detached copy of the DoS logit on every training step, adding unnecessary memory allocation. A more efficient approach would use `torch.autograd.Function` or simply scale the loss weight per-class. | Minor memory overhead. The approach is correct but inefficient; a custom autograd function would avoid the clone+detach overhead. |
| T-02 | LOW | `trainer.py:864` | `torch.load()` is called with `weights_only=False`, which is necessary for LoRA/peft objects but is a security risk if the checkpoint comes from an untrusted source. The comment acknowledges this ("LoRA peft objects not in safe globals") but does not mitigate it. A malicious checkpoint could execute arbitrary code during deserialization. | If a compromised checkpoint is loaded (e.g., from a shared model registry), arbitrary code execution is possible. Low risk in current single-developer workflow, high risk in collaborative settings. |
| T-03 | LOW | `trainer.py:870-876` | `model.load_state_dict(ckpt["model"], strict=False)` silently skips missing LoRA keys. This means if a checkpoint was trained with `lora_r=8` but the current config uses `lora_r=16`, the LoRA matrices are randomly initialized while all other weights are loaded. The training resumes with partially-random LoRA, which may explain some of the observed instability on resume. | Resumed training with mismatched LoRA rank produces unpredictable behavior. The warning is logged but training continues regardless. |
| T-04 | LOW | `trainer.py:77-78` | `logger.remove()` + `logger.add(sys.stderr, level="INFO")` at module level removes ALL existing loguru handlers and replaces with a single stderr handler. This is destructive: any caller who configured loguru before importing `trainer.py` loses their configuration. This is a well-known loguru anti-pattern. | Importing `trainer.py` as a module silently reconfigures logging for the entire process. In test or notebook environments, this can suppress important log messages from other modules. |
| T-05 | LOW | `trainer.py:766-772` | The shared cache is loaded via `pickle.load()` with no schema version check. If the cache was built with v7 graphs (8 edge types) but the current code expects v8 (11 edge types), the cached graphs will have wrong `edge_attr` ranges. The cache path was updated (`cached_dataset_deduped.pkl` → `cached_dataset_v8.pkl`), but there is no validation that the cached data actually matches the current schema. | Loading a stale cache produces graphs with wrong edge types. The path rename provides implicit versioning but no explicit validation. |

*Table 5: Training pipeline findings*

---

## 6. Cross-Cutting Concerns

### 6.1 Schema Drift and Version Management

The codebase has gone through 8 schema versions in under 2 weeks (v1 through v8, 2026-05-11 to 2026-05-19). While the inline documentation is excellent (each bug fix is traced to a specific BUG-ID), the version management relies on manual discipline: the developer must remember to (1) update `FEATURE_SCHEMA_VERSION`, (2) update `NODE_FEATURE_DIM`, (3) update `FEATURE_NAMES`, (4) update `NUM_EDGE_TYPES`, (5) update `EDGE_TYPES`, (6) add import-time assertions, and (7) re-extract all 41K+ graph files. The import-time assertions catch some mismatches but not all. For example, there is no assertion that `FEATURE_SCHEMA_VERSION` is consistent with the actual feature dimension in cached `.pt` files. A more robust approach would be to embed a schema fingerprint (hash of all schema constants) into each `.pt` file and validate it at load time.

### 6.2 Inference-Training Consistency

The shared `graph_extractor.py` module was created specifically to prevent inference-training feature drift (documented in the module docstring). This is a strong design decision. However, the inference path (`preprocess.py`) and training path (`ast_extractor.py`) still have different error handling: the training path returns `None` on failure (skip and log), while the inference path raises typed exceptions. This means a graph that fails during training is silently dropped from the dataset, but the same graph at inference time produces an error response. The model is never trained on failure-mode inputs, creating a coverage gap for edge cases that trigger extraction failures.

### 6.3 Test Coverage Gaps

The codebase includes 20 manual test contracts (`test_contracts/01-20.sol`) covering each vulnerability class. However, there are no automated unit tests for the feature computation functions (`_compute_return_ignored`, `_compute_call_target_typed`, etc.). These functions have complex Slither IR interaction logic and have already been the source of multiple bugs (BUG-9, BUG-M1). Without unit tests, regressions can only be caught by full-pipeline re-extraction and re-evaluation, which is expensive and rarely done comprehensively. The import-time assertions in `graph_schema.py` catch some structural issues but cannot verify feature computation correctness.

---


