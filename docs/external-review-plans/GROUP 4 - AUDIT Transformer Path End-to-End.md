
---

# GROUP 4 AUDIT: Transformer Path End-to-End
**Files:** `transformer_encoder.py` + `fusion_layer.py` + `sentinel_model.py`

---

## 4.1 [HIGH] `lora_target_modules` checkpoint config deserialization: string→list type trap

**Location:** `trainer.py:1008` saves, `predictor.py:225-227` reads

**Bug:** In the checkpoint, `dataclasses.asdict(config)` at line 1082 serializes `lora_target_modules` as a Python `list[str]`, e.g. `["query", "value"]`. But the MLflow `log_params` at line 1008 serializes it as `",".join(config.lora_target_modules)` → the string `"query,value"`. The predictor at line 225 reads:

```python
lora_target_modules=saved_cfg.get("lora_target_modules", ["query", "value"]),
```

If a user or script ever reconstructs `saved_cfg` from MLflow params (which is a natural thing to do for experiment tracking), the string `"query,value"` would be passed directly to `TransformerEncoder.__init__` → `LoraConfig(target_modules="query,value")`. `peft` would interpret this as a single module name `"query,value"` that doesn't exist in CodeBERT, silently creating **zero LoRA adapters** — all 295K+ trainable transformer params vanish. The model trains with 0 adaptive transformer capacity and the user has no way to notice until evaluation, when the transformer eye is effectively a frozen CodeBERT feature extractor.

**Impact:** Silent total LoRA destruction if config is reconstructed from MLflow instead of the checkpoint file. The `TransformerEncoder.__init__` doesn't validate that `lora_target_modules` is a list or that the resulting LoRA model has any trainable parameters.

**Fix:** In `TransformerEncoder.__init__`, after `get_peft_model()`, assert that `trainable > 0`. If zero, raise `RuntimeError("LoRA produced 0 trainable parameters — check lora_target_modules")`. Also add a type guard: `if isinstance(lora_target_modules, str): lora_target_modules = lora_target_modules.split(",")`.

---

## 4.2 [HIGH] `README.md` describes v4 architecture — dangerously stale for production on-call

**Location:** `ml/src/models/README.md` (entire file)

**Bug:** The README describes the **v4 architecture**:
- Line 13: `FusionLayer` with "Concat + MLP: `[B, 832]` → `[B, 64]`" — but v5 uses `CrossAttentionFusion` with cross-attention `[B,128]`
- Line 50: "Wraps `microsoft/codebert-base` as a **frozen** feature extractor. No weights are updated during training." — LoRA fine-tuning is the core v5 upgrade
- Line 73: "The `torch.no_grad()` block in `forward()` prevents computation graph construction" — v5 explicitly documents that no `no_grad()` is used because LoRA needs gradient flow
- Line 138: "output is **already sigmoid-activated**. Use `BCELoss`, not `BCEWithLogitsLoss`." — v5 outputs raw logits, uses `BCEWithLogitsLoss`
- Line 148-154: Constructor defaults show `gnn_dim=64, transformer_dim=768, fusion_output_dim=64` — v5 uses `gnn_hidden_dim=128, fusion_output_dim=128`

This README is the primary reference for anyone debugging the model. If an on-call engineer reads "use `BCELoss`" and switches from `BCEWithLogitsLoss`, it would apply sigmoid twice → all gradients become near-zero → training silently dies. If they read "frozen CodeBERT" and add `torch.no_grad()`, it kills LoRA gradient flow.

**Impact:** The README is an active hazard. It directly contradicts the v5 code on three critical points (loss function, LoRA gradient flow, architecture dimensions). Any engineer consulting it during incident response will be misled.

**Fix:** Rewrite `README.md` to match the v5 three-eye architecture with LoRA, or at minimum stamp a giant **"DEPRECATED — describes v4 only"** warning at the top.

---

## 4.3 [HIGH] `SentinelModel.__init__` default `lora_r=16` but `TransformerEncoder.__init__` default `lora_r=8` — silent parameter mismatch if TransformerEncoder used standalone

**Location:** `sentinel_model.py:109` vs `transformer_encoder.py:103`

**Bug:** `SentinelModel.__init__` sets `lora_r: int = 16` as its default, which it passes through to `TransformerEncoder`. But `TransformerEncoder.__init__` has its own independent default of `lora_r: int = 8`. If anyone constructs `TransformerEncoder()` directly (e.g. in a test, a probing script, or the `gnn_encoder.py` docstring mentions standalone use), they get `r=8` → 295K trainable params and scale 2.0, while the production model uses `r=16` → 590K trainable params and scale 2.0. The parameter count logged at line 133-138 would say "295K trainable" but the real production model has 590K.

This is especially dangerous because `r=8` vs `r=16` changes the effective LoRA capacity by 2x, meaning probing experiments run with the wrong capacity and produce misleading conclusions about what CodeBERT has learned.

**Impact:** Any standalone use of `TransformerEncoder` gets the wrong LoRA rank. Tests and experiments produce misleading results.

**Fix:** Either make `TransformerEncoder`'s defaults match `SentinelModel`'s (r=16, alpha=32), or remove the defaults from `TransformerEncoder` entirely and require them to be passed explicitly. Add a comment in `TransformerEncoder` noting that `SentinelModel` sets different defaults.

---

## 4.4 [HIGH] `aux_loss_weight=0.1` in `sentinel_model.py` docstring contradicts `0.3` from Phase 0 fix

**Location:** `sentinel_model.py:44-45`

**Bug:** The module docstring says:
```
λ=0.1 keeps each eye's gradient signal alive even if the main classifier
learns to weight one eye heavily.
```

But per the conversation context, v5.1 Phase 0 fixes raised `aux_loss_weight` from `0.1 → 0.3` specifically to combat the GNN gradient collapse to 6.7%. The `TrainConfig` default is also `0.1` at line 288. If Phase 0 was applied to `trainer.py` but not to the `SentinelModel` docstring, or if the value was changed back, this is a documentation-integrity issue. If the value was never actually changed to 0.3, the gradient collapse fix was never applied.

**Impact:** If the actual running value is still 0.1, the Phase 0 fix for gradient collapse was not applied, and the GNN eye is still being starved. If it was changed to 0.3, the docstring misleads anyone reading the model code.

**Fix:** Verify the actual `aux_loss_weight` in the running config. If Phase 0 raised it to 0.3, update `TrainConfig.aux_loss_weight` default and the `SentinelModel` docstring.

---

## 4.5 [MEDIUM] Transformer eye uses CLS token only — duplicates v4's "blurry summary" problem that cross-attention was designed to solve

**Location:** `sentinel_model.py:250-252`

```python
transformer_eye = self.transformer_eye_proj(
    token_embs[:, 0, :]   # [B, 768]
)
```

**Bug:** The entire motivation for `CrossAttentionFusion` (as stated in `fusion_layer.py:16-18`) was:
> "CLS is a blurry summary — withdraw() needs to find 'call.value' and 'transfer' specifically, not an averaged contract embedding."

Yet the `transformer_eye` still extracts only the CLS token at position 0, which is exactly the "blurry summary" the fusion layer was designed to improve upon. The transformer eye has zero fine-grained token access. The three-eye architecture was supposed to provide three independent opinions, but the transformer eye's "opinion" is the same diluted CLS that v4 had.

The fusion layer enriches both modalities before pooling, but the transformer eye bypasses all that and goes straight to CLS. This means the transformer eye contributes no new information beyond what CLS already provides to the fused eye (since the fused eye also attends to all 512 tokens including CLS).

**Impact:** The transformer eye is semantically redundant with the CLS component of the fused eye. It provides an independent projection of CLS, but no new token-level signal. The three-eye architecture effectively has two CLS-derived views and one cross-attention view, not three truly independent modalities.

**Fix:** Consider using a masked-mean pool for the transformer eye (like the fused eye does for tokens) instead of CLS-only. Or add a learnable attention pool over token positions. At minimum, document this architectural choice and its tradeoffs.

---

## 4.6 [MEDIUM] `node_type_ids` denormalisation uses `*12.0` but `_FUNC_TYPE_IDS` is a Python set with magic numbers — no schema coupling

**Location:** `sentinel_model.py:220-225`

```python
_FUNC_TYPE_IDS = {1, 2, 4, 5, 6}   # FUNCTION MODIFIER FALLBACK RECEIVE CONSTRUCTOR
node_type_ids = (graphs.x[:, 0] * 12.0).round().long()
```

**Bug (same class as Group 1's 1.2):** The magic `12.0` and the set `{1, 2, 4, 5, 6}` are both hardcoded here with no import from `graph_schema.py`. If the schema adds a new node type (e.g. `CFG_NODE_EMIT=13`), the `12.0` needs to change to `13.0` but nothing in this file will break or warn. Similarly, if `FUNCTION` stops being type_id=1, the set silently selects wrong nodes.

The `.round()` is also concerning: `graphs.x[:, 0]` was normalized by dividing by 12.0. With floating-point arithmetic, `5 / 12.0 * 12.0` could be `4.9999999...` which `.round()` gives `5.0` → `.long()` gives `5` ✓. But `6 / 12.0 * 12.0` could be `6.0000001` which is fine. However, if the normalization divisor ever changes to a float that doesn't round-trip cleanly (e.g. if `NUM_NODE_TYPES` becomes 7), the `.round()` could misclassify nodes at the boundary.

**Impact:** Silent misclassification of function-level nodes if the schema changes. The pool_mask could select zero nodes (ghost graph fallback) or wrong nodes.

**Fix:** Import `NUM_NODE_TYPES` from `graph_schema.py` and use it instead of `12.0`. Import `_FUNC_TYPE_IDS` from the schema or define a `NODE_TYPE_NAMES` enum that maps to integer IDs.

---

## 4.7 [MEDIUM] Ghost graph fallback in `sentinel_model.py` silently pools ALL nodes — includes CFG_RETURN noise that function-level pool was designed to exclude

**Location:** `sentinel_model.py:230-234`

```python
else:
    # Ghost graph (interface-only extraction) — fall back to all nodes
    pool_embs  = node_embs
    pool_batch = batch
```

**Bug:** The function-level pool was specifically introduced because "Pooling over all nodes was dominated by CFG_RETURN (77% of CFG node mass, median 93%), drowning the CFG_CALL/WRITE/COND signal that encodes execution order" (line 216). When a ghost graph (interface-only extraction from Group 2's Finding 2.12) falls back to all-node pooling, it re-introduces exactly this domination problem for those samples.

Ghost graphs from interfaces have no function-level nodes because `_add_node` (Finding 2.1) labels everything as "STATE_VAR" due to the type-id roundtrip bug. So `pool_mask.any()` returns False, triggering the fallback. But these ghost graphs still go through the full GNN forward pass and produce a GNN eye embedding that's dominated by noise.

**Impact:** Ghost graph samples get a garbage GNN eye that's dominated by CFG_RETURN node mass, but the model has no way to signal "this sample has no structural signal." The classifier still uses this garbage as 1/3 of its input, likely contributing to v5.0's 0% specificity.

**Fix:** Instead of pooling all nodes, produce a learned "no-structure" embedding (e.g. a trainable parameter) when `pool_mask.any()` is False. This gives the classifier a clean signal that structural information is absent. Also fix Finding 2.1 so that real function nodes are correctly labelled.

---

## 4.8 [MEDIUM] `CrossAttentionFusion` bidirectional attention has asymmetric gradient flow — token→node gets stronger gradient than node→token

**Location:** `fusion_layer.py:197-230`

**Bug:** In the forward pass, `node_to_token` attention (Step 3) runs first, producing `enriched_nodes`. Then `token_to_node` attention (Step 4) uses the **original** `padded_nodes` (projected but not enriched) as K/V, not the `enriched_nodes`. This means:

1. **Node→Token direction**: Q=original_nodes, K=V=tokens → enriched_nodes carry token signal
2. **Token→Node direction**: Q=tokens, K=V=original_nodes → enriched_tokens carry node signal, but from *un-enriched* nodes

This is architecturally intentional (avoiding serial dependency), but it creates an asymmetry: the enriched_nodes are immediately used in pooling, while the enriched_tokens attend to nodes that haven't been enriched yet. During backpropagation, the gradient path to `self.node_proj` is:
- Through `enriched_nodes` (from node→token attention output) → pooled_nodes → output
- Through `padded_nodes` as K/V in token→node attention → enriched_tokens → pooled_tokens → output

The second path (K/V gradient) is attenuated by the softmax in attention (gradients through keys/values are scaled by softmax probabilities, which are typically small). This means `self.node_proj` gets most of its gradient from the node→token direction, and `self.token_proj` gets most from the token→node direction, but the gradient to node_proj from the token→node path is weak.

**Impact:** The gradient asymmetry may slow convergence of the node projection relative to the token projection, contributing to the GNN gradient collapse observed in v5.0. This is a theoretical concern; empirical measurement of per-layer gradient norms would confirm.

**Fix:** Consider using `enriched_nodes` as K/V for token→node attention (stacked attention), or add a skip connection from `padded_nodes` to `enriched_nodes`. Document the asymmetry and its gradient implications.

---

## 4.9 [MEDIUM] `CrossAttentionFusion` returns no attention weights — debuggability gap for production

**Location:** `fusion_layer.py:197, 225-226`

**Bug:** Both `node_attn_weights` and `token_attn_weights` are computed but discarded — `forward()` returns only the `[B, 128]` output. During training or inference, there is no way to inspect which tokens each node attended to, or which nodes each token found relevant. This is a significant debuggability gap:

1. If the model is producing false positives for reentrancy, you cannot check whether the `withdraw()` node actually attended to `call.value` tokens.
2. If the GNN eye is collapsing, you cannot verify whether nodes are receiving meaningful token signal through cross-attention.
3. The test file has no tests for attention weight correctness.

**Impact:** Production debugging of misclassifications requires adding code to extract attention weights, which means modifying the model or adding a debug mode. This is slow and error-prone during incident response.

**Fix:** Add an optional `return_attn_weights=False` parameter (like `return_aux` in SentinelModel). When True, return `(output, {"node_to_token": ..., "token_to_node": ...})`. Zero inference overhead when False.

---

## 4.10 [MEDIUM] `TransformerEncoder` hardcoded to `microsoft/codebert-base` — no `from_pretrained` path configurability

**Location:** `transformer_encoder.py:123`

```python
self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
```

**Bug:** The model name is a hardcoded string. There's no constructor argument or config option to change it. This means:
1. Cannot use a different pretrained model (e.g. `microsoft/codebert-base-mlm`, GraphCodeBERT, or a domain-fine-tuned checkpoint)
2. Cannot point to a local model directory (important for air-gapped production environments)
3. The `TRANSFORMERS_OFFLINE=1` env var in trainer.py means if the model isn't cached, the process crashes with no fallback

**Impact:** Production deployments must have the exact model cached before first run, and can never upgrade the base model without editing source code.

**Fix:** Add a `pretrained_model_name_or_path: str = "microsoft/codebert-base"` constructor argument and pass it to `from_pretrained()`.

---

## 4.11 [MEDIUM] `gnn_dropout` is passed to `CrossAttentionFusion` as `dropout` — semantic confusion and different effective rates

**Location:** `sentinel_model.py:141`

```python
self.fusion = CrossAttentionFusion(
    ...
    dropout=dropout,     # This is SentinelModel's `dropout` param, default 0.3
)
```

But the `TransformerEncoder` uses a separate `lora_dropout` (default 0.1). And `GNNEncoder` uses `gnn_dropout` (default 0.2). So the three sub-modules use three different dropout rates: 0.3 (fusion), 0.2 (GNN), 0.1 (LoRA). This is fine architecturally, but the `SentinelModel.__init__` parameter name `dropout` is ambiguous — it could be confused with `gnn_dropout`. A reader looking at:

```python
SentinelModel(dropout=0.3, gnn_dropout=0.2, lora_dropout=0.1)
```

would naturally assume `dropout` applies to all sub-modules, but it only applies to the fusion layer and the eye projections.

**Impact:** Low direct bug risk, but high confusion risk. If someone sets `dropout=0.5` thinking it affects all components, only the fusion layer gets 0.5 dropout while GNN stays at 0.2 — a 2.5x difference in regularization that could cause overfitting in GNN while the fusion path is regularized.

**Fix:** Rename the parameter to `fusion_dropout` to match `TrainConfig.fusion_dropout` and make the scope explicit.

---

## 4.12 [MEDIUM] `sentinel_model.py` denormalisation uses `.round().long()` but `graphs.x` may be BF16/FP16 under AMP

**Location:** `sentinel_model.py:221`

```python
node_type_ids = (graphs.x[:, 0] * 12.0).round().long()
```

**Bug:** Under AMP autocast, `graphs.x` may be cast to BF16 by PyTorch's autocast policy. BF16 has only 7 bits of mantissa (vs 23 for FP32). For a value like `5/12.0 = 0.4166666...`, BF16 represents this as approximately `0.4167` which `*12.0` gives `5.0004` → `.round()` → `5` ✓. But for larger type IDs near the boundary, BF16 precision loss could cause misclassification. For example, type_id=11 → `11/12.0 = 0.9166...` → BF16 `0.9167` → `*12.0 = 11.0004` → `.round() = 11` ✓. This seems safe for current values, but the `graphs.x` tensor from PyG's Batch is typically kept in FP32 by autocast exceptions for embedding-like inputs. If PyTorch's autocast policy changes, this could break silently.

**Impact:** Currently safe due to autocast keeping embeddings in FP32, but fragile to PyTorch version upgrades.

**Fix:** Add an explicit `.float()` before the denormalisation: `(graphs.x[:, 0].float() * 12.0).round().long()`. This is defensive and zero-cost.

---

## 4.13 [LOW] `TransformerEncoder` logs `trainable` and `frozen` counts but doesn't log total or percentage

**Location:** `transformer_encoder.py:133-138`

**Bug:** The log message shows `trainable: 295,296 | frozen: 124,705,536` but doesn't show the total or the percentage of trainable parameters. At r=16 (SentinelModel default), the trainable count is ~590K, but the log would show the same format. A quick scan of logs doesn't tell you whether LoRA is 0.2% or 0.5% of total, which is the key metric for capacity assessment.

**Impact:** Minor — requires mental arithmetic to assess LoRA capacity from logs.

**Fix:** Add `pct = 100 * trainable / (trainable + frozen)` and log it.

---

## 4.14 [LOW] `CrossAttentionFusion.output_proj` uses `ReLU` before `Dropout` — dead neuron risk

**Location:** `fusion_layer.py:128-132`

```python
self.output_proj = nn.Sequential(
    nn.Linear(attn_dim * 2, output_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
)
```

**Bug:** ReLU can kill neurons permanently (dying ReLU problem). When a neuron's weights drive its pre-activation consistently negative, ReLU outputs zero, gradient is zero, and the neuron never recovers. This is well-documented for deep networks. While this is a single projection layer (not deep), the combination of ReLU + Dropout means that during training, some neurons that were already producing near-zero outputs get further suppressed by dropout, and may not recover because their gradients are zeroed by ReLU.

**Impact:** Minor in a single projection layer, but could reduce the effective dimensionality of the output from 128 to something smaller, reducing the fused eye's capacity.

**Fix:** Consider `GELU` or `LeakyReLU` as alternatives, which don't have the dying neuron problem. Or use the standard `Linear → Dropout → ReLU` ordering (dropout before activation is unusual).

---

## 4.15 [LOW] `SentinelModel.parameter_summary()` logs but doesn't return — unusable in programmatic checks

**Location:** `sentinel_model.py:285-311`

**Bug:** `parameter_summary()` logs the parameter counts but returns `None`. This means:
1. Cannot use it in assertions or tests (e.g. `assert model.parameter_summary()["total_trainable"] > 0`)
2. Cannot compare parameter counts across checkpoints programmatically
3. Cannot feed it into MLflow or other tracking

The predictor calls `self.model.parameter_summary()` at line 231 purely for logging, but if the LoRA adapters silently vanish (Finding 4.1), there's no programmatic way to detect it.

**Impact:** Minor — but would have caught Finding 4.1 if it returned a dict.

**Fix:** Return the `components` dict (or a summary dict) from `parameter_summary()`, so callers can use it programmatically.

---

## Summary Table

| # | Severity | File | Finding |
|---|----------|------|---------|
| 4.1 | **HIGH** | transformer_encoder.py / predictor.py | `lora_target_modules` string→list deserialization silently destroys all LoRA adapters |
| 4.2 | **HIGH** | README.md | README describes v4 architecture; "use BCELoss" and "frozen CodeBERT" contradict v5 code |
| 4.3 | **HIGH** | sentinel_model.py / transformer_encoder.py | Default `lora_r` mismatch (16 vs 8) between SentinelModel and standalone TransformerEncoder |
| 4.4 | **HIGH** | sentinel_model.py / trainer.py | `aux_loss_weight=0.1` docstring contradicts Phase 0 fix (0.3); gradient collapse fix may not be applied |
| 4.5 | MEDIUM | sentinel_model.py | Transformer eye uses CLS-only — same "blurry summary" problem cross-attention was designed to fix |
| 4.6 | MEDIUM | sentinel_model.py | `*12.0` and `{1,2,4,5,6}` magic numbers with no schema coupling |
| 4.7 | MEDIUM | sentinel_model.py | Ghost graph fallback re-introduces CFG_RETURN domination that function-level pool was designed to fix |
| 4.8 | MEDIUM | fusion_layer.py | Bidirectional attention gradient asymmetry — may contribute to GNN gradient collapse |
| 4.9 | MEDIUM | fusion_layer.py | Attention weights discarded — no debuggability for production misclassifications |
| 4.10 | MEDIUM | transformer_encoder.py | Hardcoded `microsoft/codebert-base` — no config for model path or local cache |
| 4.11 | MEDIUM | sentinel_model.py | `dropout` parameter name ambiguous — applies only to fusion, not GNN/LoRA |
| 4.12 | MEDIUM | sentinel_model.py | `.round().long()` on potentially BF16 tensor under AMP — fragile |
| 4.13 | LOW | transformer_encoder.py | Parameter log doesn't show trainable % — requires mental arithmetic |
| 4.14 | LOW | fusion_layer.py | ReLU + Dropout ordering risks dead neurons in output projection |
| 4.15 | LOW | sentinel_model.py | `parameter_summary()` returns None — unusable for programmatic assertions |

**Cross-reference with Group 3 overlap:**
- Finding 4.6 (magic `12.0` / `{1,2,4,5,6}`) connects to Group 1 Finding 1.2 (`/12.0` not derived from schema)
- Finding 4.7 (ghost graph fallback) connects to Group 2 Finding 2.12 (interface ghost graphs) and Finding 2.1 (type-id roundtrip bug)
- Finding 4.4 (aux_loss_weight) connects to the v5.0 gradient collapse narrative
Group 4 audit complete — **15 findings** (4 HIGH, 7 MEDIUM, 4 LOW).

Key takeaways for the transformer path end-to-end:

1. **Most dangerous**: Finding 4.1 — `lora_target_modules` can silently deserialize as a string instead of a list, destroying all 590K LoRA trainable params. This is the exact same failure mode that v5.0 was supposed to fix (frozen CodeBERT can't adapt).

2. **Most operationally dangerous**: Finding 4.2 — the README tells engineers to use `BCELoss` instead of `BCEWithLogitsLoss` and to wrap CodeBERT in `torch.no_grad()`. Both are wrong for v5 and would silently kill training.

3. **Architectural concern**: Finding 4.5 — the transformer eye uses only CLS, which is the exact "blurry summary" problem that cross-attention fusion was designed to solve. The three-eye architecture has two CLS-derived views, not three truly independent modalities.

4. **Connection to v5.0 failure**: Findings 4.4 (aux_loss_weight possibly still 0.1), 4.7 (ghost graph fallback with CFG_RETURN domination), and 4.8 (gradient asymmetry in cross-attention) all contribute to the GNN gradient collapse narrative. The ghost graph fallback is particularly concerning because it re-introduces the exact signal domination that function-level pooling was designed to fix.

Ready for Group 5 when you are.