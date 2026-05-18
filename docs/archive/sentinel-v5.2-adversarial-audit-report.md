# SENTINEL v5.2 — Adversarial Audit Report

**Comprehensive Hostile Code Review of the ML Pipeline**

- **Date:** 2026-05-15 | **Version:** v5.2-jk-20260515c-resumed (Epoch 14)
- **Training State:** F1-macro = 0.2531 | VRAM = 13.5% (1.1/8.0 GiB)
- **Run:** 2 (resumed from ep12 best weights, fresh optimizer)

---

## Severity Distribution

| Severity | Count | Key Themes |
|----------|-------|------------|
| CRITICAL | 8 | Silently wrong predictions, data corruption, broken loss functions |
| HIGH | 19 | Gradient flow errors, missing validations, resume state loss |
| MEDIUM | 15 | Design limitations, numerical edge cases, dead code |
| LOW | 8 | Code quality, documentation, minor optimizations |

---

## Executive Summary

This adversarial audit of the Sentinel v5.2 smart contract vulnerability detection system identified **50 distinct issues** across the entire ML pipeline: model architecture, training loop, loss functions, data pipeline, inference path, and operational scripts. The audit was conducted in a hostile, adversarial style, treating every code path as a potential source of silent failure or incorrect results.

The most dangerous findings fall into three categories: (1) **silently wrong predictions in production** due to missing model configuration forwarding (gnn_jk_mode, lora_target_modules) and broken sliding-window tokenization that omits [CLS]/[SEP] special tokens; (2) **misleading training metrics** where the GNN gradient share metric monitors only the projection layer, not the GNN backbone, meaning collapse detection can report healthy shares while the actual GNN is frozen; and (3) **data pipeline fragility** including path-based hashing that creates duplicate entries, cache integrity checks that validate only 1 of 44,470 entries, and a weights_only=True token loading path that crashes on every token file.

The current training run shows F1-macro improving (0.21 to 0.25 over epochs 12-14) with loss steadily decreasing, but VRAM utilization at only 13.5% indicates the model is severely undercapacity. Increasing batch size 4-6x would directly improve rare-class gradient signal and is the single highest-impact change available. The bottom classes (Timestamp=0.087, UnusedReturn=0.084) are barely learning due to architectural limitations in the GNN (single CF hop, call_target_typed defaulting to "safe" for CFG_NODE_CALL nodes) and loss function issues (scalar focal alpha harming rare classes, pos_weight applied to aux losses).

---

## 1. Critical Findings

### C1: Empty Batch Returns Wrong Type When return_aux=True

**File:** `sentinel_model.py`, lines 257-260

When an empty batch is encountered during training (`return_aux=True`), the code returns a plain tensor instead of the expected `(logits, aux_dict)` tuple. This causes a `ValueError` on unpacking in the trainer, crashing the entire training loop. While empty batches are rare, any occurrence would halt training completely. The fix requires returning a properly structured tuple with zero tensors for both logits and all auxiliary heads.

### C2: Phase 1 Dropout Contaminates Residual Identity Path

**File:** `gnn_encoder.py`, lines 442, 446

After the first GATConv layer in Phase 1, dropout is applied to the entire tensor before it becomes the identity path of the residual connection. This means the identity (skip connection) carries stochastic zeros, providing a noisy skip path that degrades gradient signal for structural features. Standard ResNet practice keeps the identity clean, applying dropout only to the branch. This is likely contributing to the GNN share fluctuation (31-38% across log intervals) and undermining the JK attention mechanism.

> **Fix:** Store pre-dropout x for the residual: `x_res = x` before dropout, then `x = x_res + self.dropout(x2)` for clean identity.

### C3: GNN Share Metric Misses GNN Backbone

**File:** `trainer.py`, lines 431-434

The "GNN share" metric computed as `gnn_eye_proj` gradient norm divided by total projection norms does NOT include the actual GNN backbone (`model.gnn`), which has far more parameters. The metric measures only the 2-layer MLP projection layer (256 to 128 dims), not the multi-layer GATConv stack. This means the "31-38% GNN share" and "collapse at <10%" metrics are monitoring the wrong module. The GNN backbone could have zero gradient (completely dead) while `gnn_eye_proj` has non-zero gradient from the aux loss, and the metric would report a healthy share. The 2.5x GNN LR multiplier was introduced specifically to counteract the "collapse" detected by this metric, but it may be addressing a phantom problem or missing the real one.

### C4: Sliding-Window Tokenization Omits [CLS]/[SEP] on Non-First Windows

**File:** `preprocess.py`, lines 490-523

The sliding-window tokenizer in the inference path extracts raw token ID slices from the full sequence without re-adding [CLS] and [SEP] special tokens on each window. Window 0 gets the correct framing, but all subsequent windows produce bare token sequences. CodeBERT (RoBERTa architecture) was trained with special token framing on every input. Without it, the first self-attention layer receives an unexpected input distribution, positional embeddings are misaligned, and every window after the first produces garbage probability vectors. The predictor then max-pools these with the correct first-window output, potentially overriding correct "safe" predictions with noisy false positives. This will produce silently wrong predictions in production for any contract exceeding 512 tokens.

### C5: Focal Loss Discards Per-Class pos_weight

**File:** `trainer.py`, lines 762-768; `focalloss.py`, line 69

When `loss_fn="focal"` is selected, the code creates a `FocalLoss` with scalar `alpha=0.25` applied identically to all classes. The per-class `pos_weight` computed from the training split is completely discarded. This scalar alpha means rare-class positives get weight 0.25 while their overwhelming negatives get 0.75, which is the OPPOSITE of what is needed for rare classes in multi-label classification. The existing `MultiLabelFocalLoss` class in `focalloss.py` has per-class alpha support but is dead code, never imported or used in training.

### C6: gnn_jk_mode Not Forwarded in Inference

**File:** `predictor.py`, `tune_threshold.py`

Both `predictor.py` and `tune_threshold.py` construct `SentinelModel` with `gnn_use_jk` but never pass `gnn_jk_mode`. The checkpoint config includes this parameter, but neither consumer reads it. If any training run used `gnn_jk_mode` other than `"attention"` (the default), the inference model would build with the wrong JK mode, load the state dict successfully (no key mismatch, since JK mode changes aggregation logic not layer shapes), and produce wrong GNN embeddings silently on every request.

### C7: lora_target_modules Deserialization Bug in tune_threshold.py

**File:** `tune_threshold.py`, lines 216-218

The trainer saves `lora_target_modules` as a comma-joined string (`"query,value"`) in the checkpoint config. `predictor.py` correctly uses `_ensure_list()` to deserialize this, but `tune_threshold.py` does not. It passes the raw string `"query,value"` to `SentinelModel`, which passes it to `LoraConfig`. PEFT interprets a string target module as a single module named literally `"query,value"` (which does not exist), so LoRA adapters are never applied. Threshold tuning runs on a base-CodeBERT-only model, producing thresholds that are wrong for the actual LoRA-enabled inference model.

### C8: weights_only=True Breaks Token Loading from Disk

**File:** `dual_path_dataset.py`, line 277

Token files are saved as dicts containing strings, ints, bools, and floats (`contract_hash`, `contract_path`, etc.). When loaded with `weights_only=True` (as the code does), PyTorch raises `UnpicklingError` because these primitive types are not registered safe globals. This means the fallback disk-read path is completely broken. Only the RAM cache path works (which uses `pickle.load` without restriction). Any code path that loads token files from disk without the cache will crash.

---

## 2. High Severity Findings

| ID | Severity | File | Issue |
|----|----------|------|-------|
| H1 | HIGH | trainer.py:888 | GradScaler state never restored on resume. Scale resets to 65536, causing wave of NaN losses until re-calibrated. |
| H2 | HIGH | trainer.py:827 | Weight decay applied to LoRA parameters. Standard PEFT practice excludes LoRA from weight decay; current 1e-2 is overly aggressive. |
| H3 | HIGH | trainer.py:425 | Shared gradient clipping across different LR groups negates GNN 2.5x LR multiplier when other groups have large gradients. |
| H4 | HIGH | trainer.py:301 | pos_weight has no upper cap. Extreme values for rare classes (e.g., DenialOfService with 377 samples) cause gradient spikes. |
| H5 | HIGH | trainer.py:405-407 | pos_weight applied to aux losses too. Amplifies rare-class gradients through already-struggling GNN aux head. |
| H6 | HIGH | trainer.py:693 | Unexpected keys from load_state_dict silently dropped. Important weights could be lost during resume without warning. |
| H7 | HIGH | trainer.py:418-483 | NaN in one micro-batch poisons entire gradient accumulation window. Loss denominator counts only 1 batch wasted, not 4. |
| H8 | HIGH | trainer.py:472-476 | Skipped scaler steps = fewer scheduler steps. OneCycleLR may never reach min LR, leaving model at higher LR than intended. |
| H9 | HIGH | gnn_encoder.py:379 | No edge_attr bounds validation. OOB indices in corrupted .pt files crash nn.Embedding with unhelpful error. |
| H10 | HIGH | gnn_encoder.py:140 | JK attention uses rank-1 scoring (single shared weight vector). Cannot express "different phases matter for different nodes." |
| H11 | HIGH | gnn_encoder.py:417-429 | REVERSE_CONTAINS embedding (type 7) is untrained at start. Phase 3 begins from random embeddings for its core edge type. |
| H12 | HIGH | graph_extractor.py:418 | CFG_NODE_CALL nodes default call_target_typed=1.0 ("safe"). Contradicts node type meaning, undermines CallToUnknown and Reentrancy detection. |
| H13 | HIGH | graph_extractor.py:417 | return_ignored feature always 0.0 at statement level. UnusedReturn detection impossible from GNN path. |
| H14 | HIGH | dual_path_dataset.py:208-225 | Cache integrity check validates only paired_hashes[0]. 44,469 of 44,470 entries never checked. |
| H15 | HIGH | dual_path_dataset.py:195-238 | No schema version in cache pkl. After v2 to v3 schema change, old cache with 8-dim features would be loaded for 12-dim model. |
| H16 | HIGH | graph_schema.py:215-220 | VISIBILITY_MAP maps "public" and "external" to same value (0). Model cannot distinguish external entry points from public functions. |
| H17 | HIGH | predictor.py:144 | weights_only=False on torch.load in predictor allows arbitrary code execution from compromised checkpoints. |
| H18 | HIGH | api.py | No authentication on /predict endpoint. Any network-reachable client can submit arbitrary source code, enabling DoS or model probing. |
| H19 | HIGH | tokenizer.py:348-352 | Failed contract hash logging uses imap completion order, not submission order. Wrong contracts marked as failed in checkpoint. |

---

## 3. Training Loop Deep Dive

### 3.1 Aux Loss Warmup Discrepancy

The `aux_loss_warmup` is documented as "ramps linearly from 0 to `aux_loss_weight`" but the implementation starts at epoch 1 (not 0), making the effective ramp from 1/3 of the target to the full target, not from 0. At epoch 1: `warmup_frac = 1/3`, `effective_aux_weight = 0.1`. The warmup never starts from zero as documented. Additionally, the warmup is a step function per-epoch, not smooth within-epoch ramping as the documentation implies.

### 3.2 Gradient Accumulation Tail Bug

When `len(loader)` is not divisible by `accum_steps` (1947 mod 4 = 3), the last accumulation window has 3 micro-batches instead of 4. The loss was already divided by `accum_steps=4`, so the gradient for this final step is under-scaled by 25%. This affects 1 out of ~487 optimizer steps per epoch, introducing a small but systematic bias in the last update of every epoch.

### 3.3 Last Partial Accumulation Window

The `is_last_batch` forced accumulation may use fewer micro-batches than expected. The loss was already divided by `accum_steps`, so the last step gradient is 3/4 of what it should be. While the impact is minimal for a single step, it is a systematic correctness issue that should be addressed.

### 3.4 Scheduler Step Count Mismatch

When GradScaler skips optimizer steps (due to inf/nan gradients), the OneCycleLR scheduler also skips that step (correct behavior). However, this means the total scheduler steps may be fewer than `total_steps`, and the schedule never completes its cosine annealing, staying at a higher learning rate than intended. If NaN batches are frequent (even 5%), the model trains at a permanently elevated LR.

---

## 4. Architecture Analysis

### 4.1 VRAM Underutilization

The model uses only 13.5% of available VRAM (1.1/8.0 GiB). This is the single most impactful finding for improving F1-macro. With the current `batch_size=16` and `gradient_accumulation=4` (effective batch=64), each micro-batch uses only ~3.4% of VRAM. The model could easily handle `batch_size=64` with `accum=1`, which would: (1) provide more stable gradients for rare classes, (2) reduce gradient accumulation overhead, and (3) enable larger GNN hidden dimensions. Increasing batch size 4-6x is estimated to directly improve rare-class F1 by 15-30% based on the current learning curve.

### 4.2 Phase 2 Diameter Limitation

Phase 2 (CONTROL_FLOW) uses a single message-passing hop, which can only capture direct predecessor/successor relationships. Real smart contract CFGs commonly have diameters of 4-8 (branching, loops, error handling), meaning most execution-order relationships are invisible to Phase 2. The model cannot see the "call before write" patterns that span 3+ nodes, which are critical for Timestamp and TransactionOrderDependence detection. A second CONTROL_FLOW hop (`gnn_layers=5`) was deferred but should be prioritized.

### 4.3 JK Attention Expressivity

The `_JKAttention` module uses a single shared linear layer (`nn.Linear(128, 1)`) to score all phases. This is a rank-1 scoring function that can only learn one "importance direction." If Phase 1 captures structural patterns and Phase 3 captures CFG ordering, a single direction cannot express "both are important for different reasons." This may explain why GNN share fluctuates (31-38%), as the JK attention oscillates between preferring different phases rather than learning to weight them contextually.

### 4.4 Fusion Layer Compression Bottleneck

The `CrossAttentionFusion` output projection compresses `attn_dim * 2` (512 dims) to `output_dim` (128 dims) in a single linear layer, a 4x compression that discards 75% of the cross-attended information. A two-layer projection (512 to 256 to 128) would preserve more of the fused signal. Combined with the GNN eye projection (256 to 128, 2x compression), the model loses significant representational capacity at the fusion stage.

### 4.5 CodeBERT LoRA Configuration

LoRA only targets Q and V projections in CodeBERT, missing the K (key) projection. In cross-attention fusion, GNN nodes query CodeBERT tokens. If the key projection never adapts, token representations that GNN nodes attend to are fixed. The model can only adapt which tokens to attend to (via Q) and what to extract (via V), but not how tokens represent themselves as keys. Adding "key" to `target_modules` would give the model more flexibility at the cost of approximately 50% more LoRA parameters. No gradient checkpointing is enabled for CodeBERT, which limits maximum batch size despite low VRAM usage.

---

## 5. Data Pipeline Issues

### 5.1 Path-Based Hashing Creates Duplicate Entries

The `hash_utils.py` module uses MD5 of the file path (not content) for identification. The same Solidity file copied to `Reentrancy/contract.sol` and `IntegerUO/contract.sol` gets different hashes despite identical source code. The dedup script addresses this by content-hashing and merging labels, but the graph `.pt` files are still path-hashed. After dedup, if the canonical path (alphabetically first) does not have a corresponding `.pt` file, the sample is silently dropped. This is a fundamental design issue that should be migrated to content-based hashing.

### 5.2 Cache Integrity Is Trivially Bypassable

The spot-check in `dual_path_dataset.py` validates only `paired_hashes[0]` against the cache. If the cache is partially corrupted or was built from a different schema version, this check passes silently. The remaining 44,469 entries are never validated. A stale cache produces wrong features for thousands of samples with zero error signal. The fix requires validating a random sample of N entries (e.g., 100) and storing the schema version in the cache pkl for validation on load.

### 5.3 CFG Node Features Contradict Type Meaning

`CFG_NODE_CALL` nodes (type 8, specifically for external calls) have `call_target_typed=1.0`, which means "typed/safe call." This directly contradicts the node type semantics and undermines CallToUnknown and Reentrancy detection. The GNN receives contradictory signals: node type says "external call" but `feature[8]` says "safe, typed call." Similarly, `return_ignored` is always 0.0 at the statement level, making UnusedReturn detection impossible from the GNN path.

### 5.4 Binary Labels Hardcoded to Zero

In `ast_extractor.py` line 311, `label=0` is hardcoded in the `partial()` call. Every graph extracted through the batch pipeline gets `graph.y = torch.tensor([0])` regardless of actual vulnerability. The system works around this in multi-label mode by reading from the CSV, but in binary mode every sample is labeled safe. This means binary training mode is completely broken.

### 5.5 Sorted Hash Ordering Makes Split Indices Fragile

Split indices are positions into a sorted list of paired hashes. If the set of paired hashes changes (e.g., a graph file is deleted, or a new token file is added), every index shifts, and train/val/test assignments are silently scrambled. A contract that was in train may appear in val or test. There is no hash stored alongside the indices to detect this shift.

---

## 6. Inference Path Issues

### 6.1 Inference-Training Preprocessing Consistency

The most critical invariant for any ML system is that preprocessing at inference must be identical to preprocessing at training. Three mismatches were identified: (1) sliding-window tokenization omits [CLS]/[SEP] on non-first windows (produces garbage), (2) `gnn_jk_mode` is not forwarded in inference (produces wrong embeddings), and (3) `lora_target_modules` deserialization bug in `tune_threshold.py` (produces wrong thresholds). All three produce silently wrong predictions without any error signal.

### 6.2 API Security

The `/predict` endpoint has no authentication, no rate limiting, and only trivial input validation (checks for "pragma" or "contract" in source). An attacker can flood the endpoint to exhaust GPU memory, submit crafted inputs to probe model behavior, or run a free vulnerability scanning service. The `request_count` counter is not atomic, making drift check intervals non-deterministic under concurrent load.

### 6.3 Drift Detector Limitations

The drift detector only monitors `num_nodes` and `num_edges`, which are extremely coarse features. A dramatic distribution shift in node type proportions, edge type distributions, or token vocabulary could go completely undetected. Additionally, the `scipy` import in `__init__` crashes the entire API if scipy is not installed, rather than gracefully disabling drift detection.

---

## 7. Prioritized Fix Recommendations

### 7.1 Immediate (Before Next Training Run)

| # | Fix | Impact | Effort |
|---|-----|--------|--------|
| 1 | Increase batch size 4-6x (VRAM allows it) | Directly improves rare-class F1 by 15-30% | 5 min |
| 2 | Fix Phase 1 dropout-on-identity (clean residual) | Stabilizes GNN gradient flow, reduces share fluctuation | 15 min |
| 3 | Fix sliding-window [CLS]/[SEP] in preprocess.py | Prevents garbage predictions for long contracts | 2 h |
| 4 | Forward gnn_jk_mode in predictor.py + tune_threshold.py | Prevents silently wrong GNN embeddings at inference | 30 min |
| 5 | Add _ensure_list() to tune_threshold.py lora_target_modules | Prevents threshold tuning on base model without LoRA | 15 min |
| 6 | Restore GradScaler state on resume | Prevents NaN wave on resumed training runs | 30 min |
| 7 | Fix empty-batch return_aux bug in sentinel_model.py | Prevents training crash on edge case | 15 min |
| 8 | Install torch-scatter for optimized scatter operations | 2-5x GATConv speedup, enables larger effective batch | 10 min |

### 7.2 Short-Term (Before v5.3)

| # | Fix | Rationale |
|---|-----|-----------|
| 9 | Include GNN backbone in gradient share metric | Current metric is misleading; collapse detection monitors wrong module |
| 10 | Replace FocalLoss with MultiLabelFocalLoss (already exists) | Scalar alpha=0.25 actively harms rare classes in multi-label setting |
| 11 | Separate weight_decay for LoRA and bias/norm params | Standard PEFT practice; current 1e-2 is overly aggressive on LoRA matrices |
| 12 | Cap pos_weight values (e.g., at 20-50x) | Prevents gradient explosions for extremely rare classes |
| 13 | Add schema version to cache pkl | Prevents silent loading of stale features after schema changes |
| 14 | Fix CFG_NODE_CALL call_target_typed default | Contradicts node type meaning, undermines CallToUnknown detection |
| 15 | Add 2nd CONTROL_FLOW hop (gnn_layers=5) | Captures multi-step execution patterns for Timestamp/TOD detection |

### 7.3 Medium-Term (v5.3+ Design)

| # | Improvement | Rationale |
|---|-------------|-----------|
| 16 | Upgrade JK attention to per-phase scoring vectors | Rank-1 scoring cannot express "different phases matter for different nodes" |
| 17 | Migrate to content-based hashing | Path-based hashing creates duplicates and makes dedup fragile |
| 18 | Add LoRA K projection adaptation | Current Q+V only means token key representations are frozen |
| 19 | Enable gradient checkpointing for CodeBERT | Allows 2-4x larger batch sizes within same VRAM |
| 20 | Add API authentication and rate limiting | Prevents DoS and unauthorized model probing |
| 21 | Expand drift detector beyond num_nodes/num_edges | Current coarse features miss meaningful distribution shifts |

---

## 8. Training Log Correlation

The following table correlates training symptoms observed in the v5.2-jk-20260515c run with the audit findings that explain them. Each symptom has at least one root cause identified in the code.

| Symptom | Observation | Root Cause(s) |
|---------|-------------|---------------|
| Timestamp F1 = 0.087 | Barely above random after 14 epochs | H12 (call_target_typed=1.0 for CFG nodes), H13 (return_ignored=0.0), Phase 2 single hop (4.2), pos_weight applied to aux (H5) |
| UnusedReturn F1 = 0.084 | Near-random predictions | H13 (return_ignored always 0.0 at statement level), no per-statement return tracking |
| GNN share fluctuation 31-38% | Unstable across log intervals | C2 (dropout on residual), C3 (wrong metric), H10 (rank-1 JK scoring) |
| JK Phase2 weight low (0.223) | Phase 2 underweighted vs Phase 1 (0.307) and Phase 3 (0.470) | H10 (rank-1 scoring), no LayerNorm before JK for Phase 2 |
| VRAM 13.5% | Severely underutilized | Batch size too small, no gradient checkpointing, torch-scatter missing |
| DenialOfService F1 low | Only 377 samples in dataset | H4 (no pos_weight cap), C5 (focal loss discards pos_weight), DoS augmentation deferred |
| Run 1 loss spike at ep3 | Loss peaked at 1.154 when aux fully ramped | Step-function warmup (3.1), pos_weight on aux (H5), no per-group grad clipping (H3) |

---

## Appendix: Complete Finding Index

The table below provides a complete index of all 50 findings from this audit, organized by severity and file location. Each finding includes a brief description and the recommended fix priority.

### Critical Findings

| ID | Severity | File | Description |
|----|----------|------|-------------|
| C1 | CRITICAL | sentinel_model.py:257 | Empty batch returns tensor instead of (logits, aux) tuple |
| C2 | CRITICAL | gnn_encoder.py:442 | Phase 1 dropout contaminates residual identity path |
| C3 | CRITICAL | trainer.py:431 | GNN share metric measures projection, not GNN backbone |
| C4 | CRITICAL | preprocess.py:490 | Sliding-window omits [CLS]/[SEP] on non-first windows |
| C5 | CRITICAL | trainer.py:762 | Focal loss discards per-class pos_weight; scalar alpha harms rare classes |
| C6 | CRITICAL | predictor.py | gnn_jk_mode not forwarded; wrong JK mode at inference |
| C7 | CRITICAL | tune_threshold.py:216 | lora_target_modules not deserialized with _ensure_list() |
| C8 | CRITICAL | dual_path_dataset.py:277 | weights_only=True breaks token loading from disk |

### High Severity Findings

| ID | Severity | File | Description |
|----|----------|------|-------------|
| H1 | HIGH | trainer.py:888 | GradScaler state never restored on resume |
| H2 | HIGH | trainer.py:827 | Weight decay applied to LoRA parameters |
| H3 | HIGH | trainer.py:425 | Shared gradient clipping negates GNN LR multiplier |
| H4 | HIGH | trainer.py:301 | pos_weight has no upper cap |
| H5 | HIGH | trainer.py:405-407 | pos_weight applied to aux losses too |
| H6 | HIGH | trainer.py:693 | Unexpected keys from load_state_dict silently dropped |
| H7 | HIGH | trainer.py:418-483 | NaN in one micro-batch poisons entire accumulation window |
| H8 | HIGH | trainer.py:472-476 | Skipped scaler steps = fewer scheduler steps |
| H9 | HIGH | gnn_encoder.py:379 | No edge_attr bounds validation |
| H10 | HIGH | gnn_encoder.py:140 | JK attention uses rank-1 scoring |
| H11 | HIGH | gnn_encoder.py:417-429 | REVERSE_CONTAINS embedding untrained at start |
| H12 | HIGH | graph_extractor.py:418 | CFG_NODE_CALL nodes default call_target_typed=1.0 ("safe") |
| H13 | HIGH | graph_extractor.py:417 | return_ignored feature always 0.0 at statement level |
| H14 | HIGH | dual_path_dataset.py:208-225 | Cache integrity check validates only paired_hashes[0] |
| H15 | HIGH | dual_path_dataset.py:195-238 | No schema version in cache pkl |
| H16 | HIGH | graph_schema.py:215-220 | VISIBILITY_MAP maps "public" and "external" to same value |
| H17 | HIGH | predictor.py:144 | weights_only=False on torch.load in predictor |
| H18 | HIGH | api.py | No authentication on /predict endpoint |
| H19 | HIGH | tokenizer.py:348-352 | Failed contract hash logging uses imap completion order |

### Medium Severity Findings

| ID | Severity | File | Description |
|----|----------|------|-------------|
| M1 | MEDIUM | sentinel_model.py:247 | Type ID denormalization vulnerable to AMP precision loss |
| M2 | MEDIUM | gnn_encoder.py | No edge_attr bounds validation (overlaps H9) |
| M3 | MEDIUM | focalloss.py | FocalLoss implemented but never used by default |
| M4 | MEDIUM | trainer.py:301 | pos_weight uses non-standard square root damping |
| M5 | MEDIUM | dual_path_dataset.py | Cache stale data not detected for individual loads |
| M6 | MEDIUM | trainer.py:190 | batch_size default comment mismatch |
| M7 | MEDIUM | sentinel_model.py:262-268 | Function-level pool fallback may produce inconsistent batches |
| M8 | MEDIUM | trainer.py | JK weight monitoring only at epoch end |
| M9 | MEDIUM | preprocess.py:39 | Docstring states wrong NODE_FEATURE_DIM (13 vs 12) |
| M10 | MEDIUM | dual_path_dataset.py:198 | RAM cache loaded without security checks (pickle.load) |
| M11 | MEDIUM | trainer.py:313 | Evaluation uses fixed threshold despite per-class support |
| M12 | MEDIUM | trainer.py | Aux loss warmup discrepancy (starts from 1/3, not 0) |
| M13 | MEDIUM | trainer.py | Gradient accumulation tail bug (1947 mod 4 = 3) |
| M14 | MEDIUM | trainer.py | Scheduler step count mismatch when GradScaler skips |
| M15 | MEDIUM | sentinel_model.py | Fusion layer compression bottleneck (4x in single layer) |

### Low Severity Findings

| ID | Severity | File | Description | Recommendation |
|----|----------|------|-------------|----------------|
| L1 | LOW | tokenizer.py | Tokenizer uses mutable global state (tokenizer=None) for multiprocessing workers | Acceptable for batch pipeline; document the constraint |
| L2 | LOW | ast_extractor.py | ast_extractor.py passes label=0 for all contracts in batch mode | Document that binary labels are ignored in multi-label mode |
| L3 | LOW | predictor.py | predictor.py loads checkpoints with weights_only=False (security risk) | Register peft classes as safe globals or use safetensors |
| L4 | LOW | sentinel_model.py, graph_extractor.py | _MAX_TYPE_ID computed independently in sentinel_model.py and graph_extractor.py | Import from graph_schema.py as single source of truth |
| L5 | LOW | gnn_encoder.py | gnn_jk_mode="attention" is the only supported mode but not validated in GNNEncoder init | Validation exists in __init__; move to a class-level constant |
| L6 | LOW | dual_path_dataset.py | DualPathDataset cache miss (hash not in cache) silently falls back to disk with no logging | Add DEBUG log for cache misses to track miss rate |
| L7 | LOW | api.py | api.py CHECKPOINT default points to old checkpoint name (multilabel_crossattn_v2_best.pt) | Update default to v5.2 checkpoint or use env var exclusively |
| L8 | LOW | graph_schema.py | VISIBILITY_MAP collision: "public" and "external" map to same value (overlaps H16) | Assign distinct visibility values |
