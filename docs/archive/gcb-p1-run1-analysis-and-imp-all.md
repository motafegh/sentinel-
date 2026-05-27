# GCB-P1 Run 1 Autopsy + IMP-* Architectural Fixes

**Date:** 2026-05-24  
**Run killed:** `graphcodebert-v1-prefix48-20260524` at ep28 (best ep27 F1-macro=0.2628)  
**Checkpoint:** `ml/checkpoints/graphcodebert-v1-prefix48-20260524_best.pt`  
**Log:** `ml/logs/graphcodebert-v1-prefix48-20260524.log`  
**Follow-up run:** `GCB-P1-Run2` — PID 80610, log `ml/logs/graphcodebert-p1-run2-20260524.log`

---

## Run 1 Configuration

| Parameter | Value |
|-----------|-------|
| GNN layers | 7 (2+3+2) |
| GNN hidden_dim | 256 |
| prefix K | 48 |
| Warmup epochs | 1–15 (prefix suppressed) |
| Prefix active from | ep16 |
| Batch size | 8 × grad_accum=8 = eff. 64 |
| Loss | ASL (γ_neg=2.0, γ_pos=1.0, clip=0.01) |
| Backbone | GraphCodeBERT + LoRA r=16 α=32 |
| Phase 2 edges | CF(6) + CALL_ENTRY(8) + RETURN_TO(9) |

---

## JK Weight Trajectory (training log)

| Epoch | F1-macro | proj_norm | JK Ph1 | JK Ph2 | JK Ph3 |
|-------|----------|-----------|--------|--------|--------|
| 1 | 0.1832 | 16.0000 | 0.063 | 0.387 | 0.550 |
| 21 | 0.2570 | 16.1250 | — | — | — |
| 24 | 0.2496 | 16.2500 | 0.063 | 0.245 | 0.692 |
| 27 | **0.2628** | 16.2500 | 0.058 | 0.234 | **0.707** |

---

## Root Cause Analysis

### RC-1: Phase 2 JK Collapse (0.387 → 0.234)

All three Phase 2 layers (conv3/conv3b/conv3c) used the same `cfg_mask` edge set
(`CF ∪ CALL_ENTRY ∪ RETURN_TO`). When layers receive identical input features and operate
on identical edges, they produce identical output representations. JK attention correctly
downweights all three — it cannot distinguish them, so it routes around Phase 2 entirely.

**Fix (IMP-G1):** Assign distinct edge subsets per Phase 2 layer:
- conv3: `cf_only` — `edge_attr == 6` (intra-function control flow only)
- conv3b: `icfg_only` — `edge_attr ∈ {8, 9}` (cross-function CALL_ENTRY + RETURN_TO only)
- conv3c: `cfg_joint` — CF ∪ ICFG (full combined mask, as before)

Each layer now operates on a different neighbourhood — JK attention can learn to combine
them meaningfully.

### RC-2: Phase 3 JK Dominance Growing (0.550 → 0.707)

Phase 3 uses REVERSE_CONTAINS edges — CFG→FUNCTION upward aggregation. Only FUNCTION nodes
receive cross-function context from Phase 3; CFG nodes are enriched only via their outgoing
edges to FUNCTION, which in REVERSE_CONTAINS direction means nothing flows back down.

Result: a systematic representation gap between FUNCTION and CFG nodes entering
CrossAttentionFusion. FUNCTION nodes carry Phase 3 context; CFG nodes don't. JK overweights
Phase 3 because it's the only path to function-level signals.

**Fix (IMP-G3):** Add a downward CONTAINS pass (conv4c) after the existing upward passes.
`fwd_contains_ei` = CONTAINS edges in FUNCTION→CFG direction (original extractor direction).
After the two upward REVERSE_CONTAINS passes, conv4c broadcasts the enriched FUNCTION
representations back down to all CFG children:
```python
x4c = self.conv4c(x, fwd_contains_ei, fwd_contains_ea)
x   = x + self.dropout(x4c)
x   = self.phase_norm[2](x)
```
Model is now 8 layers (2+3+3). Default `gnn_num_layers` updated 7→8 everywhere.

### RC-3: Phase 1 JK Flat (0.058–0.063, no change)

conv1 projects 11→256. The dimension change discards all structural information in the 11-dim
raw feature vector without a skip connection — the only way Phase 1 information survives is
through whatever conv1 learns to retain, which JK apparently finds uninformative compared to
CFG signal.

**Fix (IMP-G2):** Add `input_proj = nn.Linear(11, 256, bias=False)` skip around conv1:
```python
x_init = x
x_skip = self.input_proj(x_init)
x = self.conv1(x_init, struct_ei, struct_ea)
x = self.relu(x + x_skip)
```
Raw 11-dim features are linearly projected to 256-dim and added directly. Phase 1 output
now always carries the raw feature signal regardless of what conv1 learns.
Parameter cost: 11 × 256 = 2,816 params.

### RC-4: BF16 proj_norm Stagnation (2 ULPs over 13 epochs)

`gnn_to_bert_proj` weight norm moved only 16.0000 → 16.2500 over 13 post-warmup epochs.
At norm≈16, one BF16 ULP = 0.125 — the quantization floor prevents fine-grained gradient
accumulation. This was traced to a BF16 dtype pollution bug:

`AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)` calls
`torch.set_default_dtype(bfloat16)` as a global side effect. All `nn.Linear` created after
the BERT load (including `gnn_to_bert_proj`, `gnn_eye_proj`, `classifier`) were created in
BF16, causing the weight to stagnate at the BF16 quantization floor.

**Fix (DTYPE FIX):** Wrap BERT load in `transformer_encoder.py.__init__`:
```python
_prev_default_dtype = torch.get_default_dtype()
try:
    self.bert = AutoModel.from_pretrained("microsoft/graphcodebert-base",
                                         torch_dtype=torch.bfloat16, ...)
finally:
    torch.set_default_dtype(_prev_default_dtype)
```
All post-BERT `nn.Linear` now created in float32. `GNNEncoder.forward()` also gains a dtype
guard to ensure GNN input tensor matches GNN parameter dtype.

---

## Other IMP Fixes Applied

### IMP-M1 — FUNCTION Node Secondary Sort

`select_prefix_nodes()` sorts FUNCTION nodes by `external_call_count` (feature index 10)
descending when K truncation occurs. Sort key: `(priority, -ext_call_count, local_idx)`.
Functions with more external calls are more likely to contain reentrancy/CEI patterns.

### IMP-M2 Tier 2 — prefix_attention_mean

`TransformerEncoder.forward(output_attentions=True)` extracts mean attention from code
positions → prefix positions: `attn[:, :, :, K:, :K].mean()` across all 12 layers and
12 heads. Logged by trainer to MLflow as `prefix_attention_mean` post-warmup.
Warning threshold: < 0.002 (transformer ignoring prefix).

### IMP-M3 — Padded Prefix Mask

`select_prefix_nodes()` now returns `(prefix [B,K,768], node_counts [B])`.
`TransformerEncoder` constructs `prefix_mask[g, :node_counts[g]] = 1.0` so padded positions
(graphs with fewer than K declaration nodes) get `attention_mask=0`.

### IMP-D1 — return_ignored Temporal Ordering

`_compute_return_ignored()` rewritten from global-set to CFG-ordered per-call scan.

**Old:** Built `all_read_names` as a global set — false negative when a TemporaryVariable
name collided with an unrelated read elsewhere in the function.

**New:** Iterate `func.nodes` in topological order, collect `all_ops_ordered`. For each
call at index `call_idx`, check if `lval.name` appears in any `later_op.read` at
`all_ops_ordered[call_idx + 1:]`. Uses direct `func.nodes` (not `getattr`) so
`AttributeError` propagates to sentinel return.

Re-extraction of 41K graphs pending.

---

## P1-TRAIN Run 2 Expectations

| Epoch | Expected |
|-------|----------|
| ep1–14 | Warmup: prefix suppressed, JK Phase 1 should grow (IMP-G2 active) |
| ep15 | Warmup ends |
| ep16 | Prefix activates — brief loss spike expected |
| ep17–20 | `prefix_attention_mean` > 0.005; Phase 2 JK > 0.10 (vs 0.234 declining in Run 1) |
| ep20+ | Expect Phase 1 JK to remain > 0.10 (IMP-G2 hypothesis) |
| ep20+ | Expect Phase 2 JK stable or growing (IMP-G1 hypothesis) |
| ep20+ | Expect Phase 3 JK < 0.60 (IMP-G3 gives CFG nodes their own Phase 3 context) |

**Target:** Tuned F1-macro > 0.30 (breaks v7/v8/PLAN-3A architectural ceiling of 0.287).
