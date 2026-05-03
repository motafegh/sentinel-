# SENTINEL ML — Roadmap 2 of 3: Architecture & Training

> **Covers:** GNN encoder · Transformer encoder (CodeBERT + LoRA) · Cross-attention fusion · Training loop · Loss functions · Threshold tuning
> **Weeks:** 4–6 of the 8-week plan
> **Previous:** ← [Roadmap 1: Foundations & Data Pipeline](SENTINEL_Roadmap_1_Foundations_and_Data.md)
> **Next:** → [Roadmap 3: Production, MLOps & Interview Prep](SENTINEL_Roadmap_3_Production_and_Interview.md)

---

## Quick Reference

**Depth signals:** 🔴 Master (2–8h) · 🟡 Understand (1–2h) · 🟢 Survey (15–30min)

**The Senior's Angle — apply to every file before moving on:**
1. Why this architecture and not the alternative?
2. What are the input and output shapes?
3. What would break if this changed?
4. What is this component protecting against?
5. How does this connect to the file I read before?

**⚡ Principle vs Project-Specific:** Throughout this roadmap, ⚡ flags moments where you must learn both the project-specific value AND the generalizable principle behind it. See Roadmap 1 for the full explanation.

---

## Phase 3 — The Two Encoders

**Theme:** Each encoder in isolation — what it consumes, produces, and why it was designed this way.
**Goal:** Explain GAT with edge attributes, LoRA parameter math from first principles, and what would be lost if either encoder were simplified.
**Time:** 3–4 hours

---

### Concept Injection — Before Opening Any File

**`nn.Module` structure — read this before `gnn_encoder.py`**
Every PyTorch model follows this pattern:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 64)   # registered as a parameter
        self.layer2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)
```

`__init__` registers sub-modules and their parameters. `forward()` defines the computation. Calling `model(x)` dispatches to `forward(x)` automatically. Open `gnn_encoder.py` and map every `self.X` in `__init__` to exactly where it is used in `forward()`. Do this mapping on paper before reading the questions.

**GATConv parameters — SENTINEL-specific**
GATConv in SENTINEL is called with specific parameters. Find the call in `gnn_encoder.py` and map each:
- `in_channels` = input node feature dimension (must match `NODE_FEATURE_DIM=8` on layer 1)
- `out_channels` = output features *per head*
- `heads` = number of attention heads
- `concat=True` → output shape is `[N, heads × out_channels]`
- `concat=False` → output shape is `[N, out_channels]` (heads averaged)
- `edge_dim` = edge feature dimension (must match SENTINEL's edge embedding size)

Layer 3 uses `concat=False` and collapses heads. Layers 1–2 use `concat=True`. Derive the output shape at each layer before reading the questions section.

**LoRA through the PEFT library**
Open `transformer_encoder.py` and find where `LoraConfig` and `get_peft_model()` are used. The PEFT library wraps the CodeBERT model and replaces the target attention matrices with their LoRA equivalents. `target_modules=["query", "value"]` means only Q and V projection matrices get LoRA adapters — K is left frozen. Confirm you can find these lines before proceeding.

**PyG Batch mechanics — draw this on paper before reading `fusion_layer.py`**
This is the mechanism underlying the entire model. Draw it manually for a batch of 3 graphs.

`Batch.from_data_list([g1, g2, g3])`:
- Merges all graphs into one large disconnected graph with `N1 + N2 + N3` total nodes
- Adds `batch` tensor `[N_total]` where `batch[i]` = which graph node `i` belongs to
- `edge_index` values are offset by cumulative node counts so edges stay internal to each graph
- Result: one `Batch` object that passes through GATConv as one large graph

`to_dense_batch(x, batch)` (used in `fusion_layer.py`):
- Reverses the above: produces padded tensor `[B, N_max, F]` and mask `[B, N_max]`
- Graphs with fewer nodes than `N_max` get zero-padded rows; mask marks real vs padded
- This enables cross-attention to process all graphs in a batch simultaneously

---

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/models/gnn_encoder.py` | 🔴 Master | 3-layer GAT + edge embeddings; no pooling decision; graceful degradation |
| `ml/src/models/transformer_encoder.py` | 🔴 Master | CodeBERT + LoRA; the no_grad trap; why all token positions |
| `ml/tests/test_gnn_encoder.py` | 🔴 Master | edge_attr shapes; graceful degradation; head-divisibility |

### Questions to answer

**gnn_encoder.py:**
- Pooling was removed from `forward()` (→ `gnn_encoder.py:121`). What information is preserved by NOT pooling? Give a concrete example using a `withdraw()` function.
- The edge type embedding is at `gnn_encoder.py:91` (`nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)`). Layer 3 uses `heads=1, concat=False`; layers 1–2 use `heads=8, concat=True`. Why does layer 3 collapse heads? What would the output shape be if layer 3 also used `concat=True`?
- `edge_attr=None` triggers graceful degradation to zero vectors (→ `gnn_encoder.py:134`). What would an operator observe in production metrics before realising edge embeddings weren't being used?

**transformer_encoder.py:**
- `get_peft_model()` raises `RuntimeError` if the `peft` library is not installed (→ `transformer_encoder.py:66`). Is LoRA optional or required for training? What does this tell you about how SENTINEL treats LoRA?
- "Never wrap `self.bert()` in `torch.no_grad()`" (→ `transformer_encoder.py:35`). Why would this silently kill LoRA training even though CodeBERT weights have `requires_grad=False`?
- Returning `[B, 512, 768]` (all positions) not `[B, 768]` (CLS) (→ `transformer_encoder.py:168`). What would reentrancy detection specifically lose if you reverted to CLS-only?
- With `r=8` and `lora_alpha=16` (→ `transformer_encoder.py:113`): what is the scale factor applied to `BA`? Why does this decoupling of alpha and r matter?

**⚡ Interpretability note on the GNN encoder:**
The node-level embeddings from `gnn_encoder.py` are never pooled before fusion — this is deliberate. A consequence is that you can inspect per-node attention weights after cross-attention to understand which graph nodes (which functions, state variables) contributed most to a vulnerability prediction. If asked "how would you explain a Reentrancy prediction to a developer?", the answer starts here: extract node attention weights from the cross-attention, map them back to AST nodes, and highlight the top-k nodes in the source code. SENTINEL does not implement this visualisation, but be able to describe it.

### Code Directing Exercise

Write the prompt you would give an AI to generate `gnn_encoder.py`. Your prompt must specify: why three GAT layers, why edge attributes must be embedded before being passed to GATConv, why layer 3 uses `concat=False`, why pooling is explicitly absent from `forward()`, and what graceful degradation for `edge_attr=None` must look like. If you can write this prompt correctly from memory, you own the encoder design.

### Teach-Back Exercise

"Why does reentrancy detection specifically benefit from keeping node-level embeddings unfused AND returning all 512 token positions instead of just CLS?" Use the `withdraw()` example. Walk through which attention weights in Phase 4 let the vulnerability signal survive to the classifier.

---

## Phase 4 — Fusion: Where Structure Meets Semantics

**Theme:** `CrossAttentionFusion` completely — bidirectional design, masking, the 8 audit fixes.
**Goal:** Walk through the forward pass on a whiteboard, explain each fix, describe masked mean pooling.
**Time:** 3–4 hours

> Read `ml/tests/test_fusion_layer.py` BEFORE reading `fusion_layer.py`. The tests are the spec — they show you exactly what correct behaviour looks like before you see the implementation.

---

### Concept Injection — Before Opening Any File

**`nn.MultiheadAttention` — the exact parameters that appear in SENTINEL**
This is the core operator in `fusion_layer.py`. Open the PyTorch docs or an AI session and understand these specific parameters as they appear in the codebase:
- `query` shape: `[B, N_q, embed_dim]`
- `key` shape: `[B, N_k, embed_dim]`
- `value` shape: `[B, N_k, embed_dim]`
- `key_padding_mask` shape: `[B, N_k]` — True positions are *ignored* by attention

The last parameter is the most important one for SENTINEL. In node→token cross-attention: nodes are queries (`[B, N_nodes, dim]`), tokens are keys and values (`[B, 512, dim]`), and `key_padding_mask` is the tokenizer's `attention_mask` (True where PAD). Without this mask, every node wastes attention weight on PAD tokens. This is exactly what Fix #2 and Fix #6 correct.

**`to_dense_batch` — run this yourself before reading the file**
Open a Python shell with PyG installed:

```python
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

# Graph 1: 5 nodes with 4 features each
g1 = Data(x=torch.ones(5, 4), edge_index=torch.zeros(2, 0, dtype=torch.long))
# Graph 2: 3 nodes
g2 = Data(x=torch.ones(3, 4) * 2, edge_index=torch.zeros(2, 0, dtype=torch.long))

batch = Batch.from_data_list([g1, g2])
print("batch.x shape:", batch.x.shape)       # [8, 4]
print("batch.batch:", batch.batch)            # [0,0,0,0,0,1,1,1]

padded, mask = to_dense_batch(batch.x, batch.batch)
print("padded shape:", padded.shape)          # [2, 5, 4]
print("mask shape:", mask.shape)              # [2, 5]
print("mask:", mask)                          # [[T,T,T,T,T],[T,T,T,F,F]]
```

Run this. See the padded zeros and the False positions in the mask. This is what `fusion_layer.py` handles. You will never misread the masking code after doing this exercise.

**`torch.autocast` as a context manager**
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input)   # matmul/attention ops run in BF16
    loss = criterion(output, target)  # loss stays in FP32
```
Ops inside the block are cast to BF16 automatically. The gradient graph is preserved — `loss.backward()` still works. Open `trainer.py` and find this context manager to confirm the pattern.

---

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/models/fusion_layer.py` | 🔴 Master | The most complex component; 8 audit fixes; masking decisions |
| `ml/src/models/sentinel_model.py` | 🔴 Master | Composition; num_classes fix; attention_mask threading |
| `ml/tests/test_fusion_layer.py` | 🔴 Master | Read FIRST — masked pooling correctness; attn_dim divisibility; device detection |
| `ml/tests/test_model.py` | 🟡 Understand | Full forward pass with stub TransformerEncoder |

### Questions to answer

**fusion_layer.py:**
- Fix #2 threads `token_padding_mask` into node→token cross-attention as `key_padding_mask` (→ `fusion_layer.py:186`). Before this fix, what happened to a node's enriched representation? Error, wrong results, or both?
- Fix #7 replaces a Python for-loop with `to_dense_batch()` (→ `fusion_layer.py:181`). What was the loop doing and why is the Python version a GPU utilisation problem specifically?
- Fix #8 zeros padded node positions after node→token attention (→ `fusion_layer.py:215`), even though pooling already excludes them via `node_real_mask` (→ `fusion_layer.py:237`). Under what future code change would omitting Fix #8 silently reintroduce the bug?

**sentinel_model.py:**
- Fix #3 changed `num_classes` default from 1 → 10 (→ `sentinel_model.py:85`). What would a developer observe loading a 10-class checkpoint into `SentinelModel()` with no args under the old default? Python error, shape error, or wrong predictions?
- `SentinelModel.forward()` outputs raw **logits** — there is no `sigmoid()` call inside the model (→ `sentinel_model.py:126`). Why? What applies the sigmoid, and when?
- Trace `attention_mask` from `SentinelModel.forward()` through `CrossAttentionFusion` (→ `sentinel_model.py:155`). List the 3 distinct places it is used and what breaks at each if it is missing.

### Code Directing Exercise

Write the prompt you would give an AI to regenerate `CrossAttentionFusion` from scratch. Your prompt must specify: bidirectional cross-attention (both directions), masked mean pooling on both sides with correct mask sources, why `to_dense_batch` must be used instead of a loop, why padded node positions must be zeroed after attention even if excluded from pooling, and the final output shape. A prompt precise enough to produce correct code means you own the fusion design.

### Teach-Back Exercise

On a whiteboard: draw tensor shapes at each step of `CrossAttentionFusion.forward()` for B=2 contracts where contract 1 has 5 nodes and contract 2 has 3 nodes. Label:
`nodes_proj [N=8, 256]` → `to_dense_batch` → `padded_nodes [2, 5, 256]` + `node_real_mask [2, 5]` → both cross-attention outputs → final `[2, 128]`.

---

## Phase 5 — Training: Loss, Optimisation, Stability

**Theme:** Every training decision and why it was made.
**Goal:** Explain class imbalance handling from scratch; configure a TrainConfig for a new run; diagnose a resume bug; describe what AMP does to the gradient graph.
**Time:** 3–4 hours

---

### Concept Injection — Before Opening Any File

**`OneCycleLR` — visualise the curve before reading the resume bug**
`OneCycleLR` follows a specific LR schedule over `epochs` total steps: rises from `base_lr` to `max_lr` in the first 30% of steps, then anneals down to `min_lr` for the remaining 70%. Draw this curve on paper. Now imagine resuming at epoch 20 of a 40-epoch run. If you reinitialise the scheduler with `epochs=40` (the old bug), the curve resets to the beginning — the LR rises again when it should be annealing. If you reinitialise with `epochs=remaining_epochs=20` (the fix), the curve continues from where it was. This is Fix #8. You now understand it without reading a single line of `trainer.py`.

**MLflow — the only experiment tracker in this project**
`trainer.py` uses MLflow exclusively (→ `trainer.py:607`). WandB is in `pyproject.toml` as a declared dependency but is not called anywhere in Python source. Open `trainer.py` and find where MLflow is called. Note what is logged per epoch (metrics, artifacts, tags) — this is what you look at to diagnose training runs.

**The FocalLoss BF16 underflow bug — before reading `focalloss.py`**
Inside a `torch.autocast` BF16 region, small float values can underflow to exactly 0.0. Specifically: `sigmoid(-10.0)` in BF16 → `0.0`, then `log(0.0)` → `-inf`, then loss = `nan`, then training diverges silently. The fix is `logits.float()` before the sigmoid — manually overriding the autocast for that specific numerically sensitive operation. Open `focalloss.py` and find this line immediately. This one bug and fix tells you more about numerical stability in practice than any tutorial.

---

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/training/focalloss.py` | 🟡 Understand | FocalLoss mechanics; BF16 underflow bug and fix |
| `ml/src/training/trainer.py` | 🔴 Master | `TrainConfig` (`:133`, all 34 fields); `CLASS_NAMES` (`:107`); `compute_pos_weight()` (`:222`); AMP autocast (`:349`); OneCycleLR resume fix (`:575`); `resume_model_only` flag; MLflow logging (`:607`) |
| `ml/scripts/tune_threshold.py` | 🔴 Master | Per-class threshold tuning; why not on test split |
| `ml/scripts/create_splits.py` | *(revisit from Phase 2)* | Focus on how `multilabel_index.csv` is now the stratification source — not `label_index.csv`. Confirm the `random_seed=42` contract for reproducibility. |
| `ml/scripts/build_multilabel_index.py` | *(revisit from Phase 2)* | Focus only on pos_weight flow: computed at `build_multilabel_index.py:204`, consumed at `trainer.py:222`, passed to loss at `trainer.py:551`. |
| `ml/scripts/run_overnight_experiments.py` | 🟡 Understand | Hyperparameter sweep orchestration; use as experiment management template |
| `ml/scripts/train.py` | 🟢 Survey | CLI wrapper; know the flags |
| `ml/tests/test_trainer.py` | 🟡 Understand | pos_weight; evaluate(); FocalLoss BF16 fix |

### Deep study: AMP and GradScaler 🔴

**What `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` does:**
- Casts specific ops (matmul, conv, attention) to BF16 during the forward pass
- Keeps accumulation, batch norm, and loss in FP32
- **BF16 vs FP16:** BF16 has 8 exponent bits (same as FP32) vs FP16's 5 — same dynamic range as FP32 but lower precision. BF16 does not overflow on large gradient values. FP16 does.

**Why BF16 on Ampere (RTX 3070+):** Native BF16 tensor cores — same hardware speed as FP16 without overflow risk.

**`GradScaler` in BF16:** Essentially a no-op in BF16 (designed for FP16 overflow). Stays in the code for API consistency and future-proofing.

### Deep study: FocalLoss vs BCE+pos_weight 🔴

`FL(p) = -α(1-p)^γ · log(p)`

The `(1-p)^γ` term down-weights easy negatives. In SENTINEL: safe contracts vastly outnumber vulnerable ones — Focal Loss prevents "always predict safe."

**When to switch from BCE+pos_weight to FocalLoss:** When you observe the model assigning uniformly high confidence to negative predictions (high precision, collapsed recall), and pos_weight alone isn't enough. Focal Loss directly modulates gradient flow; pos_weight only scales the loss magnitude.

### Questions to answer

**focalloss.py + trainer.py:**
- What is `pos_weight` doing mathematically? Under what training metric pattern would you switch from `bce` to `focal`?
- Audit Fix #8: `OneCycleLR(epochs=remaining_epochs)` not `config.epochs` (→ `trainer.py:575`). Draw the LR curve for resuming at epoch 20/40 with the old code. Why does this hurt convergence?
- Audit Fix #7: filter `trainable_params` before `clip_grad_norm_`. What extra computation was happening, and why was it *wrong* not just slow?
- `TrainConfig` has a `resume_model_only: bool = True` field (→ `trainer.py:133`). What is the difference between `True` and `False`? When would you want `False`?
- MLflow is the only experiment tracker (→ `trainer.py:607`). What does it log per epoch? What would you add to `trainer.py` to debug a run where val F1-macro plateaus at epoch 5 and never improves?

**tune_threshold.py:**
- Thresholds are tuned on the validation split. Why the same split used during training, not the test split?
- Tie-break rule: higher F1 → higher recall → lower threshold. Why does "prefer lower threshold" make sense for a vulnerability detector vs a spam filter?

**⚡ Project-specific vs generalizable:**
The tie-break rule "prefer recall" is correct for security tools but NOT universal. The generalizable principle: threshold tie-breaking should reflect the asymmetry of error costs in your domain. For security/safety tools: missing a real vulnerability (false negative) is more costly than a false alarm → prefer recall → lower threshold. For fraud detection at scale: alerting on every transaction is operationally infeasible → prefer precision → higher threshold. Understand the cost function, not just the rule.

### Code Directing Exercise

Write the prompt you would give an AI to generate the core training loop in `trainer.py`. Your prompt must specify: why `BCEWithLogitsLoss` not `CrossEntropyLoss`, what `pos_weight` does and where it comes from (`trainer.py:222`), why the FocalLoss path requires `logits.float()` (`focalloss.py:55`), why `trainable_params` must be filtered before `clip_grad_norm_`, how `OneCycleLR` must be initialised on resume using remaining not total epochs (`trainer.py:575`), what `resume_model_only` controls, and what MLflow logs per epoch. If you can write this prompt, you own the training loop design.

### Teach-Back Exercise

Explain the full training loop to a colleague who knows image classifiers but not multi-label. Cover: sigmoid not softmax; `BCEWithLogitsLoss` not `CrossEntropyLoss`; what `pos_weight` does; why training-time threshold (0.5) is replaced by per-class tuned values; what "early stopping on F1-macro" means in a 10-class multi-label setting; what AMP buys; and what MLflow logs for each training run.

---

## Cross-Cutting Concerns

Own these without being prompted which phase they belong to.

### 1. The Masked Mean Pooling Pattern (appears in 3 places — `fusion_layer.py:237`)

Naive `.mean(dim=...)` includes PAD positions, diluting signal with zeros.

| Location | Mask source | What it excludes |
|---|---|---|
| `CrossAttentionFusion` node pooling | `to_dense_batch` mask | Padded node positions |
| `CrossAttentionFusion` token pooling | `attention_mask` from tokenizer | PAD tokens ([PAD]=0) |
| `DualPathDataset` collate | — | Not pooling, but same precision awareness |

When reading any new pooling code: ask "what is the mask and is it applied?"

### 2. The `weights_only` Split (`torch.load` in `predictor.py`, `trainer.py`, `dual_path_dataset.py`)

| Where | `weights_only` | Reason |
|-------|---------------|--------|
| Checkpoint loading (predictor, trainer, tune_threshold, promote_model) | `False` | LoRA peft classes cannot be loaded with `weights_only=True` |
| Graph/token `.pt` files (dataset `__getitem__`) | `True` | Safe — PyG classes registered via `add_safe_globals()` at module import |

### 3. The Three Locked Architecture Contracts (`graph_schema.py:56`, `transformer_encoder.py:168`, `sentinel_model.py:85`)

Changing any of these requires a full retrain:
- `in_channels=8` in `GNNEncoder` — locked by `NODE_FEATURE_DIM`
- `token_dim=768` in `CrossAttentionFusion` — locked by CodeBERT hidden size
- `num_classes=10` in `SentinelModel.classifier` — locked by `CLASS_NAMES`

`_ARCH_TO_FUSION_DIM = {"cross_attention_lora": 128, "legacy": 64, "legacy_binary": 64}` (→ `predictor.py:65`) lets old and new checkpoints coexist — adding a new architecture = one dict entry. Note there are three entries, not two.

### 4. No Sigmoid Inside `SentinelModel.forward()` (→ `sentinel_model.py:126`)

`SentinelModel` outputs **raw logits** — there is no `sigmoid()` call in the model's forward pass. Sigmoid is applied:
- **During training**: inside `BCEWithLogitsLoss` (numerically stable combined form)
- **During inference**: explicitly in `predictor.py` after the forward pass, before threshold comparison

This is a common source of confusion when reading the code. When you see `model(graph, tokens, attention_mask)`, the output is logits in `(-∞, +∞)`, not probabilities in `[0, 1]`.

### 5. `resume_model_only` — What Gets Restored on Checkpoint Load (→ `trainer.py:133`)

`TrainConfig.resume_model_only: bool = True` (default) controls what `resume_from` restores:
- `True` (default): only model weights are loaded. Optimizer state, scheduler step count, and best_f1 are reinitialised fresh. Use when you want to continue training with a different learning rate or config.
- `False`: full state restored — model weights + optimizer momentum + scheduler step position + best_f1. Use when you need to exactly resume an interrupted run.

The default `True` is deliberate: most resume scenarios in SENTINEL involve parameter changes (new `lr`, `epochs`, `batch_size`), making optimizer state invalid anyway.

### 7. CLASS_NAMES — the append-only registry (→ `trainer.py:107`)

Never insert in the middle. Adding class 10 at the end is safe. Inserting at index 3 silently maps "GasException" predictions to "ExternalBug" in all existing checkpoints.

### 8. Owning AI-Generated Code (the 2026 meta-skill)

Practice explaining each audit fix as:
*"The original implementation did X, which would cause Y under Z conditions. I identified it and changed it to Z."*

**Key audit fixes to memorise:**

| Fix | File | Original problem | Consequence |
|-----|------|-----------------|-------------|
| Fix #3 | `sentinel_model.py` | `num_classes` default 1→10 | Silent wrong-shape on checkpoint load |
| Fix #5 | `predictor.py` | warmup used 0-edge graph | GAT propagate never called on 0-edge |
| Fix #6 | `fusion_layer.py` | token PAD mask not applied in pooling | Naive mean diluted with PAD zeros |
| Fix #7 | `fusion_layer.py` | Python loop → `to_dense_batch` | Per-sample loop cannot batch on GPU |
| Fix #8 | `trainer.py` | `remaining_epochs` not `config.epochs` for resume | LR curve restarted from epoch 1 |
| Audit #3 | `dual_path_dataset.py` | `weights_only=True` for graph files | Pickle security |
| Audit #11 | `dual_path_dataset.py` | RAM cache integrity check missing | Stale cache silently served wrong data |

**The general skill beyond the specific fixes:** When reviewing AI-generated ML code, the high-risk areas are always: default argument values (Fix #3), gradient flow through frozen components (the no_grad trap in transformer_encoder.py), masking in attention/pooling (Fix #6), LR scheduler state on resume (Fix #8), and numerical precision at the boundary of AMP regions (FocalLoss BF16 bug). These patterns repeat across projects.

**Be able to extend the system correctly:**
- Add vulnerability class 10 → append to `CLASS_NAMES`, rebuild `multilabel_index.csv`, retrain
- Add edge type 5 → update `EDGE_TYPES` in `graph_schema.py`, bump `FEATURE_SCHEMA_VERSION`, rebuild graphs, retrain
- Change fusion output dim → update `_ARCH_TO_FUSION_DIM`, create new architecture key, retrain
- Add per-class precision/recall to API response → extend `PredictResponse`, thread thresholds through

### 9. Evaluation Metrics Deep Study 🔴

| Metric | How computed | When it misleads |
|--------|-------------|-----------------|
| **F1-macro** | F1 per class, unweighted average | Gives rare classes equal weight — may look good when rare classes are poorly predicted |
| **F1-micro** | Aggregate TP, FP, FN across all classes | Dominated by frequent classes — hides failures on rare vulnerabilities |
| **Per-class F1** | F1 for each of 10 classes independently | Most informative; reveals which classes the model actually handles |
| **Precision-Recall trade-off** | Controlled by per-class threshold | Lower threshold → higher recall. For security tools, prefer recall. |

Early stopping on **F1-macro** prevents overfitting to the majority class and forces performance across rare vulnerability types.

### 10. Model Interpretability — Beyond the Codebase

SENTINEL does not implement interpretability features, but you should be able to describe them if asked. For a security tool, interpretability is not optional in a real production context — a developer receiving a "Reentrancy detected" alert needs to understand *where* in the contract the vulnerability is.

**What you could add:**
- **Attention weight extraction from cross-attention:** After `CrossAttentionFusion.forward()`, the node→token attention weights `[B, N_nodes, 512]` tell you which tokens each node attended to. Highlight those tokens in the source.
- **Graph node importance:** The token→node attention weights `[B, 512, N_nodes]` tell you which graph nodes (functions, state variables) the token sequence found most relevant. Map high-weight nodes back to AST nodes and highlight those function definitions.
- **SHAP values on the final classifier:** Apply SHAP to the final linear layer's input `[B, 128]` to understand which dimensions of the fused representation drive each vulnerability class.

If asked in an interview: "SENTINEL doesn't currently expose these, but the architecture naturally supports it because node-level representations are preserved through fusion. I would extract attention weights from both directions of the cross-attention and map them back to source code positions."

---

## Weeks 4–6 Timeline

| Week | Phases | Focus | Deliverable |
|------|--------|-------|-------------|
| **Week 4** | Phase 3 | Both encoders. Run PyG Batch exercise in Python shell first. | Tensor shape diagram for the full forward pass — drawn from memory |
| **Week 5** | Phase 4 | Fusion. Run `to_dense_batch` exercise in Python shell first. Read `test_fusion_layer.py` before `fusion_layer.py`. | Fusion whiteboard teach-back with all 8 fixes narrated |
| **Week 6** | Phase 5 | Training. Draw OneCycleLR curve before reading trainer.py. | Training loop explanation + audit fix narration |

**Continue with → [Roadmap 3: Production, MLOps & Interview Prep](SENTINEL_Roadmap_3_Production_and_Interview.md)**
