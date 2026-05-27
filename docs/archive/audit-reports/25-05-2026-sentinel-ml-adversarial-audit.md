# SENTINEL ML Module — Adversarial Audit Report
**Scope:** `ml-sourcescode/` + `ml-doc/` | **Date:** 2026-05-24 | **Lens:** Hostile / Adversarial

---

## Executive Summary

This is a technically ambitious project. The architecture is genuinely non-trivial, the documentation is unusually honest, and the bug-tracking discipline is real. But under adversarial scrutiny, the module has **five systemic problems** that compound each other, and a cluster of code-level bugs that could silently corrupt training or inference. The headline: **the model is not production-ready, and the path to production has a gate that architecture cannot unlock** — the data is wrong.

---

## TIER 1 — CRITICAL: Will corrupt results silently

### C-1 · BF16 dtype pollution bug (partially fixed, partially live)
**File:** `transformer_encoder.py`, `gnn_encoder.py`, `sentinel_model.py`

`AutoModel.from_pretrained(..., torch_dtype=bfloat16)` calls `torch.set_default_dtype(bfloat16)` as a global side effect. The `transformer_encoder.py` fix (`_prev_default_dtype` save/restore) was applied — but **only for the BERT load path**. Any `nn.Linear` or `nn.Embedding` instantiated *inside* `SentinelModel.__init__` *before* `TransformerEncoder.__init__` runs will inherit the polluted dtype if `TransformerEncoder` is constructed after any BF16 context. The `gnn_encoder.py` forward has a dtype guard (`x.to(_param_dtype)`) but no construction-time guard. If `gnn_to_bert_proj` (created in `SentinelModel.__init__` after the `TransformerEncoder` is constructed) ends up BF16, the projection weights will be at BF16 quantization precision (ULP ≈ 0.125 at norm=16), exactly the bug that killed GCB-P1 Run 1.

**Evidence:** GCB-P1 Run 1 autopsy confirms: `proj_norm` moved only 0.25 over 13 post-warmup epochs at BF16. The "DTYPE FIX" patch description is accurate but the scope of the fix is narrower than documented.

**Adversarial verdict:** The fix is in the right place but the construction order problem is not fully closed. Any future refactor that changes construction order re-opens the bug silently.

**Recommended fix:** Move `torch.set_default_dtype(torch.float32)` to a module-level guarantee in `sentinel_model.py` before any submodule construction, or assert parameter dtype at the end of `__init__`.

---

### C-2 · Label contamination via OR-labeling (structural, not fixable in code)
**File:** `ml-doc/data-quality-findings.md`, `label_cleaner.py`

The BCCC dataset uses folder-level labeling. A benign ERC20 token in a "reentrancy" folder gets `Reentrancy=1, CallToUnknown=1, MishandledException=1, IntegerUO=1` simultaneously. This is not a bug in the pipeline — it is **the training signal the model is trained on**. The label cleaner removes structural impossibilities (3,859 in the v8.0-B run, 4,304 after the stricter pass), but:

1. The estimated remaining Tier-2 semantic noise is **20–35% of Timestamp=1** contracts and **14% of Reentrancy=1** contracts after structural cleaning.
2. The model's behavioral test result — **0/3 safe contracts score clean** — is a direct consequence: the model learned "ERC20 structure → multiple vulnerabilities" because that is what the labels said.
3. Three architecturally distinct runs (v7, v8-AB, PLAN-3A) all converged to **the same F1 ceiling (0.2875–0.2877)**. This is empirical proof that the bottleneck is in the data, not the model.

**Adversarial verdict:** Any model trained on this dataset is, at best, a noisy approximator. Deploying it as a "security oracle" for smart contracts without the false-positive rate being explicitly communicated to users is a liability. The 0/3 safe contract result is alarming for a security tool.

---

### C-3 · JK attention collapse is architectural, not a training artifact
**File:** `gnn_encoder.py` (`_JKAttention`), `ml-doc/jk-attention-collapse-findings.md`

The per-node JK weight analysis on 123,139 val nodes shows Phase 3 dominates **100% of nodes** (dominant%=99.99%). The std across nodes is 0.097 — the model learned a fixed global weighting, not per-node routing. The `_JKAttention` module's stated purpose — "each node routes to the phase most relevant to it" — **is not being achieved**.

This means the 3-phase architecture is, in practice, a weighted average with learned global constants ≈ [0.065, 0.247, 0.688]. This is equivalent to a much simpler linear interpolation of the three phase outputs. The ~2.4M GNN parameters are not being utilized at their designed capacity.

The IMP-G1/G2/G3 fixes address *why* Phase 1 and Phase 2 collapse (homogeneous edges, missing skip connection, no downward CONTAINS), but they do not address the fundamental incentive: Phase 3 (reverse-CONTAINS) is the only path from leaf nodes to the classifier, so always weighting it highest is rational from the model's perspective.

**Adversarial verdict:** The planned PLAN-3D fix (`gnn_jk_mode='cat'`) is the correct architectural response. The current attention mode has been proven to collapse in both v7 and v8-AB — it is not a configuration issue, it is structural. The IMP-* patches applied in GCB-P1-Run2 are incremental improvements that may delay collapse, not prevent it.

---

### C-4 · `_scatter_to_dense` silently truncates large graphs
**File:** `fusion_layer.py`

```python
local_idx = local_idx.clamp(max=max_nodes - 1)  # truncate oversized graphs
```

With `max_nodes=1024`, contracts with >1024 nodes have their excess nodes silently dropped from the cross-attention step. The comment says "<1% of corpus" but this estimate is not validated in the audit documents. More importantly, oversized graphs are disproportionately **large, complex contracts** — exactly the ones most likely to have sophisticated vulnerability patterns. Silently truncating them means the GNN fusion eye receives incomplete structural information for the hardest cases.

Additionally, the `CrossAttentionFusion.forward()` receives the full `node_embs [N, D]` from the GNN (correctly), but `_scatter_to_dense` discards nodes for the *fusion* step only. This creates an asymmetry: the GNN eye pool uses all nodes, but the fused eye misses nodes for large graphs. This inconsistency is undocumented and likely unintentional.

---

### C-5 · `select_prefix_nodes` iterates over graphs in a Python loop
**File:** `sentinel_model.py`

```python
for g in range(num_graphs):
    g_mask = batch == g
    ...
```

This is O(num_graphs × N) Python-level iteration. With batch_size=8, this is 8 iterations, which is tolerable. But with the sliding-window dataset where W=4 and effective sequences per graph multiply, this loop becomes a CPU bottleneck during training. More critically, this loop runs inside `forward()` on the hot training path when `gnn_prefix_k > 0` and epoch ≥ warmup. **No profiling data is presented in any doc** confirming this is not a bottleneck.

For GCB-P1-Run2 with `prefix_k=48`, every training forward pass calls this loop. The sort operation inside (`sort_keys.sort()`) is also Python-level per-graph. This should be vectorized.

---

## TIER 2 — HIGH: Significant risk to training validity or deployment

### H-1 · Gradient scaling double-counting in auxiliary loss
**File:** `trainer.py`, `train_one_epoch()`

The loss accumulation logic:
```python
loss = (main_loss + aux_loss_weight * aux_loss) / _actual_window
```

The `_actual_window` division is correct for gradient accumulation. However, the **running sums** `_run_main`, `_run_gnn_a`, etc. accumulate the un-divided loss values and are averaged with `_run_n` at log time. This means the logged per-eye losses are **not** divided by `_actual_window`. They represent raw per-step loss, not per-effective-batch loss. The metric is internally consistent (always raw) but the absolute values differ from what a researcher comparing across different `gradient_accumulation_steps` configs would expect.

This also means the "GNN share" calculation uses post-clip gradient norms which are correct, but the logged loss values in MLflow do not correspond to any standard interpretation.

---

### H-2 · `dos_loss_weight` gradient blending is mathematically odd
**File:** `trainer.py`

```python
_logits_for_loss[:, _dos_idx] = (
    dos_loss_weight * logits[:, _dos_idx]
    + (1.0 - dos_loss_weight) * logits[:, _dos_idx].detach()
)
```

This blends the gradient-flowing logit with its own detached copy. The gradient through this expression is `dos_loss_weight * grad_output` — which is correct, it scales the gradient. But the **forward value** of `_logits_for_loss[:, _dos_idx]` equals `logits[:, _dos_idx]` in all cases (the blend of a value and its own detached copy is identical to the original value). The semantic intent is clear, but the code is redundant and could mislead future readers into thinking the forward pass is modified. It only modifies the backward pass. This should be documented or replaced with a cleaner implementation using `logits[:, _dos_idx] * dos_loss_weight + logits[:, _dos_idx].detach() * (1 - dos_loss_weight)` with an explicit comment that the forward values are identical.

---

### H-3 · Weighted sampler interacts badly with deduplication
**File:** `trainer.py`, `_build_weighted_sampler()`

The weighted sampler gives 3× weight to "any-vuln" rows. After the label cleaner removes 4,304 labels, some rows that *were* any-vuln may become all-zero (all their labels removed). The `_build_weighted_sampler` reads from `label_csv_path` (the cleaned CSV) and checks `CLASS_NAMES` columns — so it should pick up the cleaned labels correctly. **But** the DualPathDataset's labels are loaded separately inside the dataset constructor, also from `label_csv`. If there is a race condition or path inconsistency between what the sampler sees and what the dataset loads, weight assignments will be misaligned. This is not tested and no assertion exists to verify they agree.

---

### H-4 · `compute_pos_weight` cap at 20.0 is arbitrary and undocumented
**File:** `trainer.py`

```python
pos_weight_vals.append(min(float(raw_ratio ** 0.5), 20.0))
```

The sqrt scaling + cap at 20.0 are engineering choices with no theoretical justification. DoS with ~260 positives in 29K training samples has raw_ratio ≈ 110, sqrt ≈ 10.5 — well below cap. But if a new rare class were added with <10 positives, raw_ratio would hit ~3000, sqrt ≈ 55, capped to 20. The cap is a safety valve, not a principled choice. More critically, the `pos_weight_min_samples=3000` threshold that clamps large classes to 1.0 is completely undocumented as to why 3000 was chosen over, say, 1000 or 5000.

---

### H-5 · `_ckpt_state` reference is kept across the entire `train()` function
**File:** `trainer.py`

```python
_ckpt_state: dict | None = None
if config.resume_from:
    ckpt = torch.load(...)
    _ckpt_state = ckpt
```

The checkpoint (which can be several hundred MB) is held in `_ckpt_state` for the entire duration of `train()`, which runs for 100 epochs. The checkpoint dict is never `del`-ed or released. This wastes RAM for the entire training run. On an 8GB GPU system this is significant.

---

### H-6 · `torch.load(..., weights_only=False)` in predictor.py is a security risk
**File:** `inference/predictor.py`, `trainer.py`

Both files use `weights_only=False` with the stated justification "LoRA peft objects not in safe globals". This is correct for LoRA checkpoints. However, `weights_only=False` allows arbitrary pickle deserialization — a malicious checkpoint file could execute arbitrary code on load. For a production security oracle, the checkpoint loading surface is an attack vector. The LoRA state dict should be extracted once at save time as a plain tensor dict (via `peft.get_peft_model_state_dict`) and saved with `weights_only=True`.

---

### H-7 · The `preprocess.py` NODE_FEATURE_DIM comment is stale
**File:** `inference/preprocess.py`

The docstring says:
```
graph.x  [N, NODE_FEATURE_DIM]  float32  (13 in v5; was 8 in v4)
```

But `graph_schema.py` v7 explicitly dropped `in_unchecked`, reducing `NODE_FEATURE_DIM` from 12 to **11**. The "13 in v5" comment was already wrong (v5 had 12), and it's now doubly wrong (current is 11). The shape contract comment in an inference-critical file is incorrect. A developer trusting this comment would write code expecting 13-dim inputs.

---

### H-8 · Empty batch guard in `sentinel_model.py` is unreachable in practice but wrong in theory
**File:** `sentinel_model.py`, `forward()`

```python
if batch.numel() == 0:
    B = input_ids.size(0)
    zeros = torch.zeros(B, self.num_classes, device=dev)
```

When `batch.numel() == 0`, there are no nodes — but `B` (number of graphs) is inferred from `input_ids.size(0)`, which is the transformer batch dimension. If `graphs` is empty but `input_ids` is not (an unlikely but possible data loading race), this returns zero logits without running the transformer path. The transformer eye contribution is silently dropped. This is the wrong behavior — it should either raise or handle the empty graph case by running only the transformer eye.

---

## TIER 3 — MODERATE: Scientific validity and design concerns

### M-1 · F1 evaluation uses `eval_threshold=0.35` — this is a scientifically problematic choice
**File:** `trainer.py`, `TrainConfig`

The rationale: "Setting eval_threshold=0.35 moves minority classes away from the boundary so patience receives a real learning signal." This is a valid engineering pragmatism for stable training signals. **However**, this means early stopping is based on F1 computed at threshold=0.35, while final performance is reported at per-class tuned thresholds from `tune_threshold.py`. The model is selected based on metric A (eval_threshold=0.35) but reported on metric B (tuned thresholds). The best checkpoint by metric A is not guaranteed to be the best checkpoint by metric B. There is no validation of this assumption in the documentation.

---

### M-2 · `tune_threshold.py` on validation set inflates reported performance
**File:** `scripts/tune_threshold.py`

Per-class thresholds are tuned on the validation split using the same data the checkpoint was selected on. This is double-dipping: the checkpoint is selected because it has the best val F1, and then thresholds are optimized on the same val set. True generalization performance requires threshold tuning on a held-out test set, with the val set used only for checkpoint selection. The "tuned F1-macro" numbers (0.2875–0.2877) are optimistic estimates, not generalization estimates.

---

### M-3 · `complexity_correlation.py` shortcut detection threshold is arbitrary
**File:** `ml-doc/data-quality-findings.md`

The report uses r > 0.40 as the "shortcut evidence" threshold with no citation. For a behavioral test on a model used as a security oracle, the MishandledException r=0.402 result was dismissed as "expected and not alarming." But a security model that predicts vulnerability based on "how many external calls the contract makes" rather than "whether those calls are mishandled" is a liability in production — it will alarm on correctly-written complex contracts and miss simple but buggy ones. The threshold of 0.40 should be 0.20 for a security tool.

---

### M-4 · GNN Phase 2 IMP-G1 layer specialization is correct in direction but incomplete
**File:** `gnn_encoder.py`

The IMP-G1 fix assigns distinct edge subsets per Phase 2 layer:
- conv3: CF only
- conv3b: ICFG only  
- conv3c: CF+ICFG joint

But conv3c uses `cfg_ei` which already includes all Phase 2 edges (including DEF_USE when not ablated). If DEF_USE is in the graph, conv3c is NOT "CF+ICFG joint" — it's "CF+ICFG+DEF_USE joint". This is inconsistent with the GCB-P1 autopsy description. The variable naming `cfg_ei` is misleading — it should be `phase2_ei` or `cf_icfg_ei` at minimum.

---

### M-5 · `select_prefix_nodes` fallback for empty graphs is silent
**File:** `sentinel_model.py`

```python
if not eligible_local:
    continue  # no declaration nodes; prefix stays zero
```

Ghost graphs (no FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes) get a zero prefix silently. The prefix mask returned includes `node_counts[g] = 0` for such graphs, causing the transformer to see K=48 zero prefix tokens with attention weight 0. This is harmless for the attention (masked out), but the zero prefix embeddings still pass through `gnn_to_bert_proj` position lookup — which they don't, since the prefix is not passed for zero-count graphs. The logic is correct but the code comment "no declaration nodes; prefix stays zero" should explicitly note that `node_counts[g]=0` causes the prefix mask to zero out attention, otherwise future maintainers may try to "fix" the continue by injecting fallback embeddings.

---

### M-6 · `WindowAttentionPooler` single-window fallback is untested
**File:** `transformer_encoder.py`, `sentinel_model.py`

The docstring says "single-window fallback returns CLS at position 0 with zero overhead." This fallback path is not exercised in any of the training runs documented (all use windowed mode). The `prefix_k` shift of CLS position (from 0 to `prefix_k` within each window) also applies to the single-window fallback. If someone runs single-window inference with `prefix_k=48`, the CLS at position 48 (not 0) would be returned — but is this correct? No test or doc addresses this.

---

### M-7 · `_build_weighted_sampler` mode "all-rare" is mathematically inverted
**File:** `trainer.py`

```python
elif mode == "all-rare":
    n_pos = max(1, sum(float(row.get(cls, 0)) for cls in CLASS_NAMES))
    w = 1.0 / n_pos
```

This gives **lower** weight to contracts with more positive labels, and **higher** weight to contracts with fewer labels (approaching 1.0 for single-class contracts). This is the opposite of what "all-rare" implies — rare classes (DoS, Timestamp) tend to appear alone, so single-label contracts get the *highest* weight, which is what is intended. But contracts with, say, only GasException (a common class with 3000+ samples) also have n_pos=1 and get weight 1.0, while mixed-vulnerability contracts get lower weight. The mode name and behavior don't match the intuition, and this is never tested in the documented runs.

---

## TIER 4 — LOW: Code hygiene, maintainability, documentation

### L-1 · `.backup` file committed to source
**File:** `ml-sourcescode/src/datasets/dual_path_dataset.py.backup`

A 20KB backup of `dual_path_dataset.py` from 2026-04-20 is in the repo. This is dead code in version control. The backup predates the v8 architecture — the backup contains old code that could mislead a developer into thinking it's authoritative.

### L-2 · `__pycache__` directories are committed
**Path:** Multiple `__pycache__/` directories with `.cpython-312.pyc` files are in the zip

This is a `.gitignore` failure. The .pyc files are binary artifacts from a specific Python 3.12 installation — they are not portable and add 400KB+ of binary noise to the repository. They should be excluded.

### L-3 · `label_cleaner.py` duplicates schema constants instead of importing
**File:** `scripts/label_cleaner.py`

```python
# Constants matching graph_schema.py (reproduced to avoid circular imports
# when running this script standalone)
EDGE_CALLS = 0
CLASS_NAMES = [...]
```

The comment acknowledges the duplication. But when `graph_schema.py` changes (e.g., adding a new edge type that shifts IDs), `label_cleaner.py` will silently use wrong values. The correct fix is to restructure `label_cleaner.py` to import from a standalone constants module that has no Slither/PyTorch dependency, not to duplicate constants manually.

### L-4 · Architecture string `"three_eye_v7"` in `trainer.py` does not match actual version
**File:** `trainer.py`, `predictor.py`

`ARCHITECTURE = "three_eye_v7"` and `MODEL_VERSION = "v7.0"` are hardcoded. The source code is clearly running v8+ experiments (8-layer GNN, IMP-G1/G2/G3, etc.) but the model is tagged as v7. This means checkpoints from the GCB-P1-Run2 (which uses the IMP-patched v8 architecture) will be labeled `architecture="three_eye_v7"` in the checkpoint. Future resume logic checks `ARCHITECTURE` for mismatch — this will silently pass for architecture-incompatible checkpoints.

### L-5 · `monitor.sh` is undocumented
**File:** `scripts/monitor.sh`

A shell script exists in the scripts directory with no README entry explaining what it monitors, when to run it, or what its outputs mean. Its 3KB content likely contains useful monitoring logic but is invisible to anyone not manually inspecting the directory.

### L-6 · `src/tools/` and `src/validation/` directories are empty
**Path:** `ml-sourcescode/src/tools/`, `ml-sourcescode/src/validation/`

These directories exist but are empty (no files, no `__init__.py`). They suggest planned functionality that was never implemented. Empty directories in a repo are usually `.gitkeep` placeholders — but there are none, and `src/validation/` in a security-critical ML system is not a nice-to-have.

---

