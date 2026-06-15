# Pre-Run-8 Human Understanding Checklist

**Purpose:** A running record of everything the human should understand before launching Run 8. Covers the problem, the solution, and the broader context for each issue. Updated during the investigation session 2026-06-04.

---

## Section 1 — Why Run 7 Got Nothing

### 1.1 The Complexity Proxy (Root Cause)

**What happened:**  
The model learned `feat[5]=complexity` (log-normalised CFG block count) as a universal proxy for all 10 vulnerability classes. All 10 classes show 34–36% gradient share from `complexity` with zero elevation for their specific features (`return_ignored` for UnusedReturn, `external_call_count` for Reentrancy, etc.).

**Why this happened:**  
- `complexity` correlates loosely with every vulnerability class (larger contracts have more bugs)
- `complexity` is a single continuous scalar → gradient descent prefers it over multiple sparse binary features
- The model has no reason to learn structural patterns when `complexity` explains 35% of the variance cheaply

**What the model SHOULD have learned:**  
- UnusedReturn → `return_ignored` (feat[7]) is elevated
- Reentrancy → `external_call_count` (feat[10]) + CFG path (CALL→WRITE without RETURN in between)
- Timestamp → `uses_block_globals` (feat[2]) + CFG path through `block.timestamp`

**The fix:**  
Zero out `feat[5]` at GNN input (`--drop-complexity-feature`). The model is forced to use class-specific features or learn structural patterns from graph topology.

**Design decision — why zero instead of remove:**  
Zeroing preserves the feature schema (NODE_FEATURE_DIM=11 stays fixed). The model is schema-locked (ZKML proxy MLP hardcoded, cached graphs pre-extracted). Removing feat[5] would require full re-extraction of all 41,576 graphs and rebuilding the ZKML circuit. Zeroing is equivalent at the GNN input (the weight matrix column for feat[5] still exists but receives zero gradient through that feature).

**Edge cases:**  
- The `.clone()` before zeroing is mandatory (`.to(dtype)` is a no-op when dtype matches, so without clone we'd corrupt cached training tensors in-place)
- Zeroing must happen BEFORE the type embedding concat — otherwise feat[5] position would be zeroed in `x` but `x_init` would still have the original value passed into `conv1` and `input_proj`

---

### 1.2 Edge Topology Ignored (L2 Finding)

**What happened:**  
All 11 edge types were ablated one at a time. Max delta across ALL types was 0.013 F1. The model effectively acts as a node-feature MLP — it could use only CONTAINS edges and get the same output.

**Why this happened:**  
When `complexity` dominates (35% of gradient), the GNN's attention weights learn to attend to high-complexity nodes regardless of edge type. The edge-specific masking in Phase 2 (CF-only, ICFG-only, joint) builds different representations, but if the JK aggregator weights all three phases nearly equally (L1 finding: 99.5% max entropy) AND the classifier ignores fine-grained phase differences (because `complexity` explains most of the variance), then ablating any single edge type has minimal effect.

**What should change in Run 8:**  
With `complexity` zeroed, the model MUST use structural patterns. Edge topology should become discriminative. Run 8's L2-equivalent experiment (if run) should show larger deltas for CONTROL_FLOW and CALL_ENTRY edges.

---

### 1.3 JK Routing Near-Uniform (L1 Finding)

**What happened:**  
JK attention weights were Phase1=0.334→0.304, Phase2=0.319→0.301, Phase3=0.347→0.395 at ep40. Entropy = 1.094/1.099 = 99.5% of maximum — nearly indistinguishable from random equal routing.

**Why this happened:**  
The JK attention module learns to route each node to the "most useful" phase. But if all phases produce similar representations (because they all feed through the same `complexity`-dominated GNN), the router has no incentive to specialise. Entropy regularizer (λ=0.005) penalises LOW entropy (collapse to one phase), so it was working correctly — but it doesn't reward USEFUL specialisation.

**Phase 3 drift:**  
Phase 3 weight drifted from 0.347 to 0.395 (ep40). This is the REVERSE_CONTAINS + CONTAINS bidirectional pass. When Phase 3 dominates, signals from Phase 2 (CFG/ICFG) have less influence on the final embedding. This explains why the ISSUE-1 fix (cfg_eye_proj gradient path) was necessary — without it, the only way to give Phase 2 a gradient signal was through JK, and JK was deprioritising Phase 2.

**What changes in Run 8:**  
λ=0.0075 (increased from 0.005) + `--drop-complexity-feature`. With complexity gone, different phases will process different structural information → the router should learn to specialise. Entropy regularizer with higher λ more strongly resists Phase 3 dominance drift.

---

### 1.4 DoS Sawtooth Masked the 9-Class Plateau

**What happened:**  
The reported best F1 of 0.3074 (fixed threshold) was driven by DoS variance: 65 val positives (1.04% prevalence). One correct DoS prediction = +0.008 macro F1. The "new best" epochs (sawtooth peaks) were DoS-high epochs, not genuine improvements in other classes.

**True underlying trend (excluding DoS):**  
Flat at 0.287–0.295 since ep20. The 9-class system was already plateaued by ep20.

**Why it happened:**  
DoS is a STRUCTURAL class (cross-contract, complex control flow). The model detected it via `complexity` proxy + some structural signal. Val set had only 65 DoS positives with 1.04% prevalence — extreme variance. Every 8 epochs, by chance, the model's DoS threshold straddled the boundary and caught a few extra positives.

**What this means for interpreting Run 8:**  
- Track BOTH fixed-threshold and per-class tuned F1 separately
- Track DoS separately from the 9-class average
- A "new best" that comes only from DoS improvement is noise, not signal
- Real improvement = improvement in UnusedReturn, IntegerUO, Reentrancy (the 3 classes with structural signals that `--drop-complexity-feature` should unlock)

---

### 1.5 BUG-SL-1: 40 Epochs of Blind Training

**What happened:**  
`training_logger.py:check_aux_head()` tried to access `model.aux_phase2[-1]` (subscript on `nn.Sequential`). After `torch.compile`, `model.aux_phase2` is an `OptimizedModule` wrapper. `OptimizedModule` implements `__getattr__` for attribute access but NOT `__getitem__` for subscript access. The `TypeError` was silently caught by the outer `try/except Exception` block in the epoch loop.

**Result:**  
All structured logging data (AUC-ROC, Brier score, ECE, aux head norms, per-eye probability distributions) was empty for all 40 Run 7 epochs. The team was effectively flying blind on model health.

**Why it matters:**  
- AUC-ROC would have shown whether the model was learning ordering (even if threshold-based F1 was poor)
- ECE would have shown calibration divergence — the per-eye ECE 0.04 vs ensemble ECE 0.23 finding was only discovered post-hoc via Phase 2 experiments
- Aux head weight norms would have shown if the Phase 2 head was being trained

**The fix (confirmed in code):**
```python
head = getattr(head, "_orig_mod", head)  # unwrap OptimizedModule before [-1]
```

**Broader lesson:**  
Any code that accesses `model.some_submodule[-1]` after `torch.compile` MUST unwrap `_orig_mod`. Attribute access (`model.gnn.jk`) works through `__getattr__` forwarding. Subscript access (`head[-1]`) does NOT.

---

## Section 2 — The Run 8 Solution

### 2.1 The Core Change: `--drop-complexity-feature`

**What it does:**  
At `GNNEncoder.forward()` entry, after dtype normalisation, zero-out `x[:, 5]` (the `complexity` feature) before any convolution. The rest of the graph — topology, edge types, all other 10 features — remains intact.

**Why this specific approach:**  
Three alternatives were considered:
1. **Zero-out (chosen):** Preserves schema, no re-extraction, backward compatible
2. **Batch normalise feat[5]:** Makes complexity mean-zero but still variance-present — model can still use it
3. **Remove feat[5] entirely:** Requires full re-extraction of 41,576 graphs, ZKML circuit rebuild — months of work

Option 1 is the minimal targeted intervention. The model's `input_proj` linear layer (IMP-G2 skip) has a weight column for feat[5] — that column gets zero gradient through feat[5], but the other columns adapt. The model learns the right thing or doesn't: there's no intermediate proxy shortcut.

**What the model is forced to do:**  
Learn from:
- `return_ignored` (feat[7]) for UnusedReturn
- `external_call_count` (feat[10]) for Reentrancy / ExternalBug
- `uses_block_globals` (feat[2]) for Timestamp
- CFG ordering via CONTROL_FLOW + CALL_ENTRY edges in Phase 2
- Contract-level aggregation via Phase 3 reverse/forward CONTAINS

---

### 2.2 Secondary Change: `--gnn-prefix-k 48`

**What it does:**  
Injects the top-48 GNN node embeddings as prefix tokens before the GraphCodeBERT input sequence. CONSTRUCTOR/FALLBACK/RECEIVE/MODIFIER/FUNCTION nodes (sorted by priority × external_call_count) are projected via `gnn_to_bert_proj` (Linear 256→768) and prepended to BERT's input_embeds.

**Why disabled in Run 7:**  
`gnn_prefix_k=0` was set in Run 7. The GNN prefix was tested in Run 1 but never carried into the final architecture because the implementation was built AFTER Run 1 and the subsequent runs were fixing other bugs.

**Design of prefix selection:**  
The priority order (CONSTRUCTOR > FALLBACK > RECEIVE > MODIFIER > FUNCTION) was chosen because:
- CONSTRUCTOR is the unique entry point — vulnerabilities originate from constructor state
- FALLBACK/RECEIVE are reentrancy-critical entry points (ETH transfer handlers)
- Within FUNCTION: secondary sort by `external_call_count` (feat[10]) — functions that make external calls are higher-risk for CEI violations

**Why `warmup_epochs=5` instead of 15:**  
Run 8 starts fresh. The GNN will be learning from scratch alongside the prefix projection. 5 epochs of GNN warming before prefix injection is sufficient — the GNN doesn't need the full 15 epochs since `drop_complexity_feature` means the GNN is learning something meaningful from epoch 1.

---

### 2.3 Secondary Change: `fusion_lr_multiplier=0.3`

**Why reduce from 0.5 to 0.3:**  
The CrossAttentionFusion (821K params) was running at 0.5× base LR in Run 7 and still producing gradient spikes (0.09–0.165) at step 100–200 of each epoch. Fusion spikes are transient (normalise by step 300), but they indicate fusion is learning faster than the GNN and may be locking in a suboptimal representation before GNN has enough epochs.

At 0.3×, fusion trains at the same rate as LoRA — both are "secondary learners" adapting on top of well-established representations.

---

### 2.4 Secondary Change: `λ=0.0075` (JK entropy reg)

**Why increase from 0.005:**  
Phase 3 JK weight drifted to 0.395 by ep40. The regularizer penalises `(H_max - H)`. At uniform [0.33, 0.33, 0.33], penalty=0. At [0.30, 0.30, 0.40], penalty=λ×0.0036. At λ=0.005, this is 0.018 — barely detectable against the main loss. At λ=0.0075, it's 0.027. The goal is not to force exact uniformity, but to provide a meaningful signal against Phase 3 dominance.

**Why not higher (e.g., λ=0.01):**  
Run 3 used λ=0.01 and collapsed JK to exact 33/33/33 uniform (zero specialisation). The entropy regularizer is a soft penalty, not a hard constraint. λ=0.0075 sits between "ineffective" (0.005) and "too strong" (0.01).

---

## Section 3 — Code Changes Impact Analysis

### 3.1 Files That Need Changes Before Run 8

| File | Change | Risk |
|------|--------|------|
| `ml/src/models/gnn_encoder.py` | Add `drop_complexity` param + zero-out in forward | Low — well-isolated |
| `ml/src/models/sentinel_model.py` | Accept + pass `drop_complexity_feature` | Low — one parameter thread |
| `ml/src/training/trainer.py` | Add `drop_complexity_feature: bool = False` to TrainConfig | Low — dataclass field |
| `ml/scripts/train.py` | Add `--drop-complexity-feature` argparse flag | Low — one arg |
| `ml/src/inference/predictor.py` | Read flag from `saved_cfg` in SentinelModel construction | Low — one line |
| `ml/src/training/trainer.py` | Fix `use_weighted_sampler` default: `"positive"` → `"timestamp-size"` | Very low |
| `ml/src/training/trainer.py` | Fix `import math` inside loop → module level | Very low |
| `ml/scripts/train.py` | Fix dead `--aux-cei-loss-weight` arg | Very low |

### 3.2 Files That Do NOT Need Changes Before Run 8

| File | Reason |
|------|--------|
| `ml/src/training/training_logger.py` | BUG-SL-1 already fixed |
| `ml/src/models/fusion_layer.py` | fusion_max_nodes change is a runtime flag, not code |
| `ml/data/graphs/*.pt` | No re-extraction needed (zero-out is at GNN input, not data) |
| ZKML circuit | No schema change (NODE_FEATURE_DIM=11 stays 11) |
| `ml/scripts/interpretability/*.py` | Diagnostic scripts, not training path |

### 3.3 Checkpoint Compatibility

A Run 8 checkpoint with `drop_complexity_feature=True` can be loaded by:
- **Predictor** (after fix F1): reads flag from `saved_cfg`, passes to `SentinelModel`
- **Resuming Run 8**: full config is in `checkpoint["config"]` via `dataclasses.asdict(config)`
- **Interpretability scripts**: use `GNNEncoder.forward()` directly — they must pass `drop_complexity=True` manually or the diagnostic will see different input distribution than training

The flag is saved in `checkpoint["config"]["drop_complexity_feature"]` because `dataclasses.asdict(config)` includes all fields.

---

## Section 4 — What Will and Won't Change in Run 8

### 4.1 Expected Improvements

| Class | Why it should improve | Expected |
|-------|-----------------------|---------|
| UnusedReturn | `return_ignored` (feat[7]) is now the primary signal | +0.05–0.15 F1 |
| Reentrancy | `external_call_count` + CFG CEI sequence forced to matter | +0.02–0.06 |
| Timestamp | `uses_block_globals` + `block.timestamp` CFG path | +0.01–0.04 |
| IntegerUO | Arithmetic CFG patterns + `complexity` was noise here | +0.01–0.03 |
| DoS | Structural patterns + no complexity shortcut | Unpredictable (noisy) |

### 4.2 What Won't Change

- **UnusedReturn F1 ceiling**: Still ~0.234 without RC5 DEF_USE edges (def-use chains not in current schema)
- **Timestamp F1 ceiling**: Still ~0.17 without data-flow provenance of `block.timestamp`
- **TransactionOrderDependence/ExternalBug**: Need cross-contract reasoning (not in current architecture)
- **DoS sawtooth**: 65 val positives → still high variance per epoch
- **Ensemble ECE 5.8× worse than per-eye ECE**: The final Linear(512→256→10) classifier is miscalibrated. Not addressed in Run 8.

### 4.3 What Might Temporarily Regress

- **Early epoch F1**: The model no longer has the complexity shortcut. It may spend more epochs learning the right features, producing lower F1 at ep5–20 than Run 7 had.
- **GNN share / Ph2 gradient**: The zeroed feature means fewer raw gradient paths → watch the Ph2/Ph1 ratio in early epochs.
- **Transition at gnn_prefix_k warmup (ep5)**: The prefix projection starts from random init. Expect a ~0.01–0.02 F1 dip at ep5–8 while the projection aligns.

---

## Section 5 — Pre-Run Checklist

### Must-do before launching Run 8

- [ ] **F1: Implement `--drop-complexity-feature`** in 5 files (see FINDINGS.md F1)
- [ ] **F5: Set `--fusion-max-nodes 2048`** in the run command (no code change needed)
- [ ] **F2: Fix `use_weighted_sampler` default** in TrainConfig
- [ ] **Extract Run 7 tuned thresholds** from ep39 checkpoint (for reference/comparison)
- [ ] **Close MLflow ghost run**: `mlflow runs set-terminated --run-id 541345bab6864f738e484794122607bc --status KILLED`
- [ ] **Verify BUG-SL-1 fix works** on first epoch structured log (AUC/Brier/ECE should now be non-empty)

### Nice-to-have before launching (won't block)

- [ ] F3: Remove dead `--aux-cei-loss-weight` arg from train.py
- [ ] F6: Add prefix modules to torch.compile list
- [ ] F15: Move `import math` to module level in trainer.py
- [ ] F12: Update train.py docstring

---

## Section 6 — Concepts to Understand Deeply

### 6.1 torch.compile and OptimizedModule

`torch.compile(module, dynamic=True)` returns an `OptimizedModule`. This wrapper:
- **Forwards attribute access** (`module.submodule`) via `__getattr__` → `_orig_mod.submodule` ✅
- **Does NOT support subscript access** (`module[-1]`) ❌ → raises TypeError

Always unwrap with `getattr(module, "_orig_mod", module)` before any `[-1]` indexing.

### 6.2 JK Attention and Entropy Regularization

JK collects [phase1_out, phase2_out, phase3_out] as 3 tensors of [N, 256]. For each node, a learned Linear(256,1) scores each phase, softmax gives weights [w1, w2, w3] summing to 1. The output is the weighted sum.

Entropy = -sum(w * log(w)). Maximum entropy = log(3) ≈ 1.099 when all weights = 1/3.

The regulariser adds `λ × (H_max - H)` to the loss. When H=H_max (uniform), penalty=0. When one phase dominates (H→0), penalty→λ×H_max ≈ λ×1.1.

**Key insight:** The regulariser prevents COLLAPSE (one phase taking all attention) but does NOT prevent near-uniform routing. In Run 7, near-uniform was the "wrong" solution — not because JK was collapsed, but because uniform routing can still work when `complexity` explains most variance. With `complexity` zeroed, the model has incentive to route different nodes to different phases (structural nodes to Phase 1, CFG nodes to Phase 2, contract-level nodes to Phase 3).

### 6.3 Four-Eye Gradient Flow

The 4 eyes create 4 independent gradient paths to the classifier:
1. **GNN eye** (`gnn_eye_proj` → pool over FUNCTION nodes) → gradient reaches conv1/conv2
2. **TF eye** (`transformer_eye_proj` → WindowAttentionPooler → BERT CLS) → gradient reaches LoRA
3. **Fused eye** (`CrossAttentionFusion` → node_embs × token_embs) → gradient reaches both GNN + LoRA
4. **CFG eye** (`cfg_eye_proj` → pool over CFG_NODE types) → gradient reaches conv3/conv3b/conv3c DIRECTLY

Without the CFG eye (BUG-R7-1), Phase 2 gradient only reached conv3 via: (1) JK attention (deprioritised) → (2) Phase 3 reverse-CONTAINS (2 hops) → (3) GNN eye pool over FUNCTION nodes. With the CFG eye, Phase 2 has a DIRECT gradient path.

### 6.4 Why `aux_phase2` Uses Phase 2 NOT Phase 3

`aux_phase2` pools `_phase2_x` (the Phase 2 output tensor, BEFORE Phase 3 aggregation) over CFG_NODE types. This gives Phase 2 a gradient signal BEFORE its output is blended with Phase 1 and Phase 3 by JK. The phase 2 tensor is the output of conv3c (Layer 5) after the LayerNorm.

`_phase2_x` is kept with gradients attached (`return_phase2_embs=True` always in SentinelModel). At inference under `torch.no_grad()`, this is free.
