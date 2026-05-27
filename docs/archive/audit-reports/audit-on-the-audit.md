
---

# SENTINEL ML Module — Full Adversarial Audit & Data Strategy Report
**Date:** 2026-05-25 | **Lens:** Hostile / Adversarial | **Source of Truth:** Live source code + training logs (not docs)

---

## Executive Summary

I have audited every source file (`sentinel_model.py`, `gnn_encoder.py`, `transformer_encoder.py`, `fusion_layer.py`, `losses.py`, `focalloss.py`, `trainer.py`, `graph_schema.py`, `dual_path_dataset.py`), the training log for GCB-P1-Run2 (epochs 1–3 running), the project CHANGELOG, and both external reviewer documents. My findings confirm and extend the prior audit, but I also found **new issues the previous reviewer missed**, and I disagree with some of their assessments. The report is structured by severity with concrete evidence from the actual code.

**The core problem remains data.** The architecture is genuinely sophisticated — three-eye, cross-attention fusion, GNN prefix injection, IMP-G1/G2/G3 fixes — but the model is being trained on labels that are systemically wrong. No architecture can learn patterns the training signal doesn't contain.

---

## TIER 0 — META: Where I Disagree With the Prior Audit

### D-1 · Prior Audit C-3 (JK Collapse) Is Correct in Diagnosis but Wrong in Severity

The prior audit says "JK attention collapse is architectural" and recommends `jk_mode='cat'`. I agree JK collapses — the P1-Run2 log already shows Phase 3 dominance at epoch 2 (86.6% attention weight with warning). **But** calling this a "critical" finding overstates the case. JK in attention mode is equivalent to a learned global weighting `[0.058, 0.075, 0.866]` — which is functionally a linear interpolation. The GNN still produces useful embeddings; it's just that Phases 1 and 2 contribute less than designed. The model still achieves F1=0.2877 with JK collapsed. Switching to `cat` mode would triple the GNN-to-classifier parameter count (3×256=768 input dim vs 256) and could cause its own problems on this dataset size. **Recommendation:** Keep JK attention mode but add a JK entropy regularizer to prevent full collapse, before trying `cat`.

### D-2 · Prior Audit C-5 (Python loop in select_prefix_nodes) Is Overstated

With `batch_size=8` and `num_graphs ≤ 8`, this loop is 8 iterations per forward pass. The sort inside is over a few dozen nodes per graph. At batch=8, this is <0.1ms overhead vs ~450ms per step. The prior audit says "No profiling data is presented" — that's true, but the claim that this is a "CPU bottleneck" is unsubstantiated. The real bottleneck is the GraphCodeBERT forward pass through 12 attention layers. **Verdict:** Nice-to-vectorize, not critical.

### D-3 · Prior Audit M-4 (conv3c includes DEF_USE) Is Correct and Important

Looking at the actual code in `gnn_encoder.py` line 538:
```python
x2 = self.conv3c(x, cfg_ei, cfg_ea)  # Layer 5: joint integration
```
`cfg_ei` is built from `cfg_mask` which, when `phase2_edge_types` is `None`, includes `DEF_USE(10)`:
```python
cfg_mask = (
    (edge_attr == _CONTROL_FLOW) |
    (edge_attr == _CALL_ENTRY)   |
    (edge_attr == _RETURN_TO)    |
    (edge_attr == _DEF_USE)
)
```
So conv3c is NOT "CF+ICFG joint" — it's "CF+ICFG+DEF_USE joint". The docstring in the GNN encoder says "Layer 5 (conv3c): CF+CALL_ENTRY+RETURN_TO joint — integration layer" which is **factually wrong** when `phase2_edge_types=None`. However, in the actual P1-Run2 config, `--phase2-edge-types 6 8 9` explicitly excludes DEF_USE, so `cfg_mask` only contains {6,8,9}. **The bug only fires when running with default config** (no `--phase2-edge-types` flag). All documented v8+ runs use the explicit flag. Still, this is a latent bug.

---

## TIER 1 — CRITICAL: New Findings From Source Code Audit

### NC-1 · `gnn_to_bert_proj` Receives Zero Gradient During Warmup — But Its Parameters Are Still in the Optimizer

**File:** `trainer.py` (train function), `sentinel_model.py`

When `gnn_prefix_k=48` and epoch < `gnn_prefix_warmup_epochs=15`, the prefix path is completely skipped:
```python
# sentinel_model.py line 434
if self.gnn_prefix_k > 0 and self._current_epoch >= self.gnn_prefix_warmup_epochs:
    gnn_prefix, gnn_prefix_counts = self.select_prefix_nodes(...)
```

This means `self.gnn_to_bert_proj` and `self.prefix_type_embedding` receive **zero gradient** for 15 epochs. But the trainer creates a separate param group `PrefixProj` with its own LR multiplier. The `AdamW` optimizer maintains momentum (exp_avg) and variance (exp_avg_sq) states for these parameters even when gradients are zero. After 15 epochs of zero-gradient updates, the momentum buffer is zero but the variance buffer is non-zero (initialized from `weight_decay`). When warmup ends and gradients start flowing at epoch 16, the optimizer's adaptive learning rate is cold-started from a distorted state.

**The P1-Run2 log confirms this:** `gnn_to_bert_proj weight norm: 15.9853` stays identical across epochs 1–3 (as expected — no gradient). But the P1-Run1 autopsy shows it only moved from 16.0→16.25 over 13 post-warmup epochs. The cold-start problem is real.

**Recommended fix:** Either (a) skip adding `gnn_to_bert_proj` and `prefix_type_embedding` to the optimizer until warmup ends (reconstruct optimizer at epoch 16), or (b) add a small L2 regularizer to `gnn_to_bert_proj` during warmup so it at least drifts slightly and the AdamW state accumulates meaningful variance.

### NC-2 · `_FUNC_IDS_CPU` Is Built at Module Import Time But NODE_TYPES Could Change

**File:** `sentinel_model.py` line 89

```python
_FUNC_IDS_CPU: torch.Tensor = torch.tensor(sorted(_FUNC_TYPE_IDS), dtype=torch.long)
```

This is a module-level tensor built from `NODE_TYPES`. If any test or script mutates `NODE_TYPES` (unlikely but possible — Python dicts are mutable), this tensor would be silently wrong. More importantly, this tensor is never validated against the actual `graph_schema.NODE_TYPES` at runtime. If a schema version mismatch existed between the model code and the graph files, the pooling mask would silently select wrong node types, producing zero GNN eye output for affected graphs.

**Evidence:** The graph schema has been through v1→v2→v3→v4→v5→v6→v7→v8 with multiple renumberings. The current code assumes v8 (NODE_TYPES values are hardcoded). If someone loads a v7-era graph file (12-dim features), the GNNEncoder guard catches it. But if someone loads a v8 graph where the type IDs are correct but the _FUNC_IDS_CPU mapping doesn't match (e.g., after a schema refactor), this would be a silent accuracy regression.

**Recommended fix:** Add a `__post_init__` or `forward()` assertion that `_FUNC_IDS_CPU` values are a subset of `set(NODE_TYPES.values())`.

### NC-3 · `AsymmetricLoss` and `MultiLabelFocalLoss` Are Both in the Codebase But Only ASL Is Used — Dead Code Risk

**Files:** `losses.py`, `focalloss.py`, `trainer.py`

The trainer imports both:
```python
from ml.src.training.focalloss import FocalLoss
from ml.src.training.losses import AsymmetricLoss
```

But `FocalLoss` is never instantiated in `train()` — only `AsymmetricLoss` and `BCEWithLogitsLoss` are used based on `config.loss_fn`. The `FocalLoss` class expects **post-sigmoid** inputs (line 21: "IMPORTANT — this class expects POST-SIGMOID probabilities"), while `AsymmetricLoss` expects **raw logits** (line 81: "logits: Raw model output before sigmoid"). If someone switches `loss_fn="focal"` without reading the docstring carefully, they'll pass raw logits to a function that applies `F.binary_cross_entropy` (which expects probabilities), producing NaN losses silently.

**Evidence:** The `_FocalFromLogits` wrapper mentioned in the `FocalLoss` docstring (line 22-23) does not exist anywhere in the codebase. It was apparently planned but never implemented.

**Recommended fix:** Either remove `FocalLoss` entirely or implement the `_FocalFromLogits` wrapper. The current state is a trap.

### NC-4 · `compute_pos_weight` Is Never Actually Used by the ASL Loss

**File:** `trainer.py`

The `compute_pos_weight()` function (lines 356–399) is called and stored in `pos_weight`:
```python
pos_weight = compute_pos_weight(str(label_csv_path), train_indices, ...)
```

But then looking at how `loss_fn` is constructed in the trainer, when `loss_fn="asl"`, the loss is created as `AsymmetricLoss(...)` — which takes **no `pos_weight` argument**. The `pos_weight` tensor is computed, logged, and then **completely ignored**. For `loss_fn="bce"`, `pos_weight` IS passed to `BCEWithLogitsLoss`. But the default and recommended config is `"asl"`.

This means the ASL loss has no per-class positive/negative rebalancing beyond its own gamma/clip mechanism. The carefully computed sqrt-scaled pos_weights are dead code in the default configuration. The ASL gamma parameters partially compensate, but they're global (gamma_neg=2.0 for all classes) rather than per-class. DoS (260 positives) gets the same gamma_neg as IntegerUO (13,797 positives).

**Recommended fix:** Either (a) pass `pos_weight` to ASL as a per-class `gamma_neg` modifier, or (b) remove the `compute_pos_weight` call when `loss_fn="asl"` to avoid confusion.

---

## TIER 1 — CRITICAL: Training Log Analysis (GCB-P1-Run2)

### TL-1 · JK Phase 3 Dominance at Epoch 2 Is Worse Than Expected

The log at epoch 2 shows:
```
JK attention weights — Phase1=0.058±0.108 Phase2=0.075±0.104 Phase3=0.866±0.212
⚠ JK phase dominance: Phase 3 has 86.6% attention weight.
```

Phase 3 at 86.6% at epoch 2 is **worse** than PLAN-3A at convergence (68.8%). The IMP-G1/G2/G3 fixes were supposed to improve Phase 1 and Phase 2 representation quality, which should slow JK collapse. Instead, JK is collapsing faster. This suggests the 8-layer architecture (IMP-G3's downward CONTAINS pass) is making Phase 3 even more dominant by giving it an additional aggregation step.

**Adversarial interpretation:** The IMP-G3 fix (conv4c: downward CONTAINS) was designed to enrich CFG nodes with Phase 3 context, but it also makes Phase 3 the deepest and most processed representation. JK attention rationally weights it higher because it has the most signal. This is a fundamental tension: adding depth to Phase 3 makes it stronger, which makes JK collapse faster.

### TL-2 · `gnn_to_bert_proj weight norm: 15.9853` Is Stagnant

Across epochs 1–3, the projection weight norm stays at 15.9853 — exactly as expected during warmup (no gradient flows). But this also means that when warmup ends at epoch 16, the projection starts from a completely random initialization while the rest of the model has 15 epochs of training. The P1-Run1 autopsy showed this projection barely moved (16.0→16.25 over 13 epochs), suggesting the cold-start problem is severe.

**Adversarial interpretation:** The warmup design is self-defeating. The GNN prefix projection is supposed to translate GNN node embeddings into BERT's input space. But after 15 epochs, the GNN has adapted to the CrossAttentionFusion path — its embeddings are optimized for fusion, not for projection into BERT. When the projection starts training, it must simultaneously (a) learn to project GNN embeddings into a space BERT can use, and (b) not disrupt the already-trained attention patterns. This is a harder optimization problem than training everything from scratch.

### TL-3 · Bottom-3 Classes at Epoch 2 Are Zero — DoS, Timestamp, UnusedReturn

```
Bottom3: DenialOfService=0.040 | UnusedReturn=0.035 | Timestamp=0.000
```

Timestamp is at 0.000 at epoch 2. This class has historically been one of the hardest (F1≈0.25 at best). The external reviewer's Solution 3 (CFG-path gating for Timestamp) directly addresses this — contracts labeled Timestamp=1 but where `block.timestamp` doesn't gate an external call should have their labels removed. The current model sees so many false-positive Timestamp labels that it can't learn the real pattern.

---

## TIER 2 — HIGH: Source Code Bugs and Design Issues

### NH-1 · `dual_path_collate_fn` Excludes `contract_path` — Label Cleaner Can't Cross-Reference

**File:** `dual_path_dataset.py` line 384

```python
_EXCLUDE = ["contract_hash", "contract_path", "contract_name",
            "node_metadata", "num_edges", "num_nodes", "y"]
```

The `contract_path` attribute is excluded from the batched graph. This is correct for training (the model doesn't need it), but it means the ensemble label audit (Solution 4 in the external reviewer doc) cannot access `contract_path` through the DataLoader. The audit script would need to load `.pt` files separately. This is not a bug, but it makes the audit solution harder to implement.

### NH-2 · `class_label_smoothing` in TrainConfig Is a `dict` — Not Reproducible Across Runs

**File:** `trainer.py` line 270

```python
class_label_smoothing: dict = field(default_factory=lambda: {
    "CallToUnknown": 0.10,
    ...
})
```

This dict is not sorted. Python dicts preserve insertion order (3.7+), but if someone modifies the dict or adds entries in a different order, the `class_eps` tensor constructed from it could have misaligned values. More critically, this dict is not validated against `CLASS_NAMES` — if a class is missing from the dict, it silently gets `eps=0.0` (no smoothing). If a typo is introduced (e.g., "CalltoUnknown" instead of "CallToUnknown"), the same silent failure.

**Recommended fix:** Validate `class_label_smoothing` keys against `CLASS_NAMES` in `__post_init__`.

### NH-3 · `torch.load(..., weights_only=False)` in Trainer — Security Surface

**File:** `trainer.py` line 879

```python
ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
```

The dual_path_dataset.py correctly uses `weights_only=True` with safe globals registered. But the trainer and predictor use `weights_only=False` because LoRA/peft objects aren't in the safe globals. This is a known issue flagged by the prior audit. However, there's a deeper concern: **the checkpoint dict contains the optimizer state, scheduler state, and potentially the entire TrainConfig**. A malicious checkpoint could inject arbitrary objects through these paths.

**Recommended fix:** Save checkpoints as `{"model": model_state_dict, "optimizer": opt_state_dict, ...}` with only plain tensors (extract LoRA state dict separately via `peft.get_peft_model_state_dict`). Then load with `weights_only=True`.

### NH-4 · `_ckpt_state` Is Never Deleted — Memory Leak

**File:** `trainer.py`

As flagged by the prior audit, `_ckpt_state` holds the full checkpoint dict (model + optimizer + scheduler states = potentially 500MB+) for the entire 100-epoch training run. On an 8GB GPU system where total system RAM may also be constrained, this is significant.

**Recommended fix:** Add `del _ckpt_state` after loading all necessary components.

### NH-5 · The `gnn_prefix_proj_lr_mult=1.0` Default Means the Projection Learns at Full LR From Cold Start

**File:** `trainer.py` line 306

The prefix projection starts from random init at epoch 16, while the rest of the model has 15 epochs of training. Giving it `lr × 1.0` (same as base LR) means it will make large initial updates that could destabilize the transformer's attention patterns. A higher warmup multiplier (e.g., 3.0–5.0) for the first few epochs after warmup would help it catch up faster, similar to how `gnn_lr_multiplier=2.5` helps the GNN.

**Recommended fix:** Add a separate `prefix_proj_warmup_lr_mult` that ramps from 5.0 → 1.0 over 3 epochs after warmup ends.

---

## TIER 3 — MODERATE: Data Strategy and Scientific Validity

### NM-1 · The 0.287 Ceiling Is Real and the Data Is the Primary Cause

Three architecturally distinct runs (v7: 7-layer GNN, v8-AB: 7-layer + DEF_USE, PLAN-3A: 7-layer ICFG-only) all converge to 0.2875–0.2877 tuned F1. This is the strongest possible evidence that the ceiling is not architectural but **data-determined**. The v8.0-B run (with stricter label cleaning) actually performed WORSE (0.2460), which the project interpreted as "data cleaning alone cannot break the ceiling." But this interpretation is wrong — the v8.0-B run was killed at epoch 11 with only 10 epochs of training. It was still improving. The early kill was because `early_stop_patience` was set too low relative to the harder optimization landscape created by cleaner (less noisy) labels.

**Adversarial interpretation:** The project has a confirmation bias problem. When v8.0-B underperformed, the conclusion was "data quality isn't the bottleneck." But v8.0-B was never given a fair chance — it was killed while still improving. The correct conclusion is "cleaner labels make the optimization harder and require more epochs." The F1 ceiling at 0.287 likely reflects the maximum achievable F1 on this dataset with this label noise level, and it will require BOTH cleaner labels AND more training epochs to break.

### NM-2 · The External Reviewer's Solutions Are Sound — Prioritize Differently

The seven solutions in `sentinel-c2-concrete-data-fixing-solutions.md` are all technically correct. However, I recommend a different sequencing based on cost-benefit:

**Revised Priority Order:**

1. **Solution 5 (Safe Contract Injection) — HIGHEST PRIORITY.** The 0/3 behavioral test failure is the single most alarming finding. It means the model cannot distinguish safe contracts from vulnerable ones. This is not a "nice-to-have" — it's a showstopper for any deployment. 100 clean anchors from OpenZeppelin/Solmate will teach the model what "safe" looks like. **Expected impact: Immediate behavioral test improvement.**

2. **Solution 1 (CEI Order Detection for Reentrancy) — SECOND.** This is the most straightforward label cleaning improvement. The implementation is clean, uses existing graph structure, and targets the highest-impact class (Reentrancy has the most training samples and the most noise). **Expected impact: ~200-400 label removals, Reentrancy F1 +0.01-0.03.**

3. **Solution 2 (Solidity ≥0.8.0 IntegerUO Cleaning) — THIRD.** IntegerUO is the model's best class (F1≈0.70). Cleaning it won't dramatically improve F1 but will reduce false positives on modern contracts. The implementation requires source file access which some graphs lack (8.5% have empty `contract_path`). **Expected impact: Cleaner signal, F1 stable or slightly improved.**

4. **Solution 4 (Cross-Checkpoint Ensemble Label Audit) — FOURTH.** This is the most powerful technique (Confident Learning) but requires the most infrastructure. It should be applied AFTER Solutions 1-3 because those remove the most obvious noise, leaving the ensemble audit to find subtler mislabels. **Expected impact: ~500 additional mislabels found.**

5. **Solution 3 (Timestamp CFG-Path Gating) — FIFTH.** Timestamp is the hardest class and the smallest positive set. The CFG-path filter is correct but the expected removal count (~100-200) is small. Still worth doing, but after the higher-impact fixes.

6. **Solution 7 (Threshold Validation on Held-Out Test Set) — SIXTH.** This is a scientific validity fix, not a model improvement. It will make reported numbers LOWER but more honest. Apply after all label cleaning is done and before any deployment decision.

7. **Solution 6 (Pragma-Aware Temporal Data Splitting) — SEVENTH.** This is important for preventing version-based overfitting but requires re-extracting version metadata. It should be done as part of a dataset rebuild, not as a standalone fix.

### NM-3 · DoS Has Only 243 Training Positives — No Loss Engineering Can Fix This

After all augmentation, DoS has ~243 training positives out of ~29,000 training samples (0.84%). The `dos_loss_weight=0.5` setting gives it 50% gradient, but the fundamental problem is that 243 examples is simply not enough for the model to learn a robust pattern. The BCCC dataset has inherently few DoS examples.

**Recommended data addition strategy:**
- Source from **SmartBugs Wild** dataset (contains real-world DoS vulnerabilities from mainnet)
- Source from **SWC Registry** (SWC-128: Denial of Service with Failed Call)
- Generate additional synthetic DoS contracts: focus on `payable(addr).transfer()` in loops, `selfdestruct`-based DoS, and gas-limit DoS patterns
- Target: 500+ additional DoS positives (triples the training set for this class)

### NM-4 · The Model Has No Explicit Negative Anchor — It Doesn't Know What "Safe" Means

This is the root cause of the 0/3 behavioral test failure. The BCCC dataset's "benign" contracts are in vulnerability-labeled folders and receive OR-labels. The model has never seen a contract with ground-truth all-zero labels. The weighted sampler gives 3× weight to any-vuln rows and 1× to zero-label rows, but the zero-label rows are mostly contaminated (OR-labeled but the specific vulnerability happens to be 0).

**Concrete addition plan:**

| Source | Contracts | Labels | Notes |
|--------|-----------|--------|-------|
| OpenZeppelin v4.9 (audited) | 50 | All zero | ERC20, ERC721, Ownable, etc. |
| Solmate (audited) | 20 | All zero | ERC20, ERC721, Auth |
| Manual safe contracts | 30 | All zero | Simple storage, getters, pure functions |
| Existing test contracts | 8 | All zero | `12_safe_contract.sol` through `19_safe_no_calls.sol` |
| **Total** | **~108** | **All zero** | Weighted at 10-15× in sampler |

### NM-5 · Additional Data Sources Beyond BCCC

The project currently relies solely on BCCC (~44K contracts). For a production security oracle, this is insufficient. Here are concrete additional data sources:

| Source | Size | Access | Key Value | Cost |
|--------|------|--------|-----------|------|
| **SmartBugs Wild** | ~47K contracts | GitHub (open) | Real-world deployed contracts with known vulnerabilities | Free, requires extraction pipeline |
| **SWC Registry** | ~400 test cases | GitHub (open) | Per-vulnerability canonical examples with exact labels | Free, small but high-quality |
| **SolidiFI** | ~16K injected vulnerabilities | GitHub (open) | Synthetic but precisely labeled; good for rare classes | Free, requires graph extraction |
| **Slither-audited contracts** | ~5K contracts | Etherscan + Slither | Slither's own classification as labels; same pipeline as training | Free but requires Etherscan API |
| **Expert-labeled subset** | ~200-500 contracts | Manual | Gold-standard labels for evaluation | Expensive (~$50-100/contract) |

**Recommended approach:**
1. Start with SmartBugs Wild (largest, free, most diverse)
2. Add SWC Registry (small but perfectly labeled — use as validation anchors)
3. Add SolidiFI for rare classes (DoS, Timestamp)
4. Build expert-labeled test set for final evaluation

---

## TIER 4 — LOW: Code Quality and Maintainability

### NL-1 · `ARCHITECTURE = "three_eye_v7"` and `MODEL_VERSION = "v7.0"` Are Stale

The code is running v8+ architecture (8-layer GNN, IMP-G1/G2/G3, GraphCodeBERT) but labels itself as v7. Checkpoints saved from GCB-P1-Run2 will have `architecture="three_eye_v7"` in their metadata, making resume logic unable to distinguish them from actual v7 checkpoints.

### NL-2 · `.backup` and `__pycache__` Files in the Source Tree

`dual_path_dataset.py.backup` (20KB, from 2026-04-20) and multiple `__pycache__/` directories with `.cpython-312.pyc` files are in the zip. These should be excluded from version control.

### NL-3 · `focalloss.py`'s `FocalLoss` Class Is Unreachable Dead Code

As detailed in NC-3, the `FocalLoss` class (post-sigmoid) is never instantiated in the training pipeline. The `_FocalFromLogits` wrapper mentioned in its docstring doesn't exist. This is a maintenance hazard.

### NL-4 · `EMITS(3): 12` — Near-Zero Edge Count

From the CHANGELOG edge statistics, EMITS edges have only 12 instances across 41,576 graphs. This means the EMITS edge type is effectively dead — the GNN's edge embedding for type 3 is trained on 12 examples. This edge type should either be fixed (the extractor may not be emitting EMITS edges correctly — the BUG-H7 fix was applied but only adds EventCall fallback) or removed from the schema.

### NL-5 · Empty `src/validation/` and `src/tools/` Directories

A security-critical ML system has an empty validation directory. This is a gap — validation should include schema compatibility checks, label distribution sanity checks, and inference-determinism tests.

---

## Training Log: GCB-P1-Run2 Epochs 1–3 Deep Analysis

| Metric | Ep 1 | Ep 2 | Ep 3 | Trend |
|--------|------|------|------|-------|
| Loss | 0.1606 | 0.2046 | ~0.146 | Decreasing (ep2 spike from ASL warmup) |
| F1-macro | 0.1402 | 0.1678 | — | Improving |
| JK Phase1 | 0.109 | 0.058 | — | Collapsing (bad) |
| JK Phase2 | 0.147 | 0.075 | — | Collapsing (bad) |
| JK Phase3 | 0.744 | 0.866 | — | Dominating (bad) |
| GNN share | 86%→86% | 76%→76% | 76% | Stable but declining slightly |
| `gnn_to_bert_proj` norm | 15.985 | 15.985 | 15.985 | Frozen (expected during warmup) |

**Key observations:**
1. The aux loss warmup is working correctly — epoch 1 has `aux_weight=0.0000`, epoch 3 has `aux_weight=0.0750` (ramping to 0.3 over 8 epochs).
2. JK collapse is faster than in v7/v8 runs, confirming that IMP-G3's deeper Phase 3 amplifies the dominance problem.
3. GNN gradient share is healthy (76%), much better than the v5.1 10% collapse that prompted the `gnn_lr_multiplier=2.5` fix.
4. The per-eye losses are converging (gnn=0.43, tf=0.43, fused=0.43 at ep3 step 300), suggesting the three eyes are learning similar representations — this may indicate the eyes aren't learning complementary features.

---

## Concrete Data Addition Strategy (Final Recommendation)

### Phase A: Immediate (1 week, no re-extraction needed)

1. **Inject 100+ clean anchor contracts** (Solution 5)
   - Source: OpenZeppelin v4.9, Solmate, existing test_contracts
   - Labels: All-zero
   - Sampler weight: 10-15×
   - Run GCB-P1-Run3 with clean anchors + existing cleaned labels

2. **Apply CEI-order Reentrancy filter** (Solution 1)
   - Add `check_reentrancy_cei_order()` to `label_cleaner.py`
   - Re-run label cleaner → expect ~300 fewer Reentrancy positives
   - No re-extraction needed

3. **Apply pragma-based IntegerUO filter** (Solution 2)
   - Add `check_integer_uo()` to `label_cleaner.py`
   - Re-run label cleaner → expect ~1,500 fewer IntegerUO positives
   - Requires source file access for ~91.5% of graphs

### Phase B: Short-term (2-3 weeks)

4. **SmartBugs Wild dataset integration**
   - Download ~47K contracts from SmartBugs GitHub
   - Run Slither analysis for automated labeling
   - Extract graphs using existing pipeline
   - Merge into training set with per-contract Slither labels (NOT folder-level OR-labels)
   - Target: 10,000+ new contracts with higher-quality labels

5. **SWC Registry integration**
   - Add ~400 canonical vulnerability examples
   - These serve as high-confidence positive anchors
   - Weight at 5× in sampler

6. **Run ensemble label audit** (Solution 4)
   - Use v7, PLAN-3A, and GCB-P1-Run2 checkpoints
   - Apply Confident Learning to find remaining mislabels
   - Manual review of top 500 flagged samples

### Phase C: Medium-term (1-2 months)

7. **SolidiFI dataset for rare classes**
   - Focus on DoS (243→700+ positives) and Timestamp (1500→2000+)
   - These are synthetically injected but precisely labeled

8. **Pragma-aware stratified splitting** (Solution 6)
   - Rebuild splits with version stratification
   - Prevents version-based shortcut learning

9. **Held-out test set for honest evaluation** (Solution 7)
   - Reserve 15% as untouched test set
   - Tune thresholds on separate holdout
   - Report final numbers on test set only

### Expected Cumulative Impact

| Phase | Expected Tuned F1-macro | Behavioral Test | Notes |
|-------|------------------------|-----------------|-------|
| Current (GCB-P1-Run2) | 0.28-0.30 (projected) | 8-10/19 | Prefix injection may lift ceiling slightly |
| After Phase A | 0.30-0.33 | 12-14/19 | Clean anchors fix specificity; CEI filter helps Reentrancy |
| After Phase B | 0.33-0.38 | 14-17/19 | SmartBugs adds scale; SWC adds precision |
| After Phase C | 0.35-0.40 | 15-18/19 | Honest evaluation; rare class improvement |

---

## Summary of All Findings

| ID | Severity | Category | Source | Status |
|----|----------|----------|--------|--------|
| NC-1 | CRITICAL | `gnn_to_bert_proj` cold-start with AdamW state | Source code | **NEW** |
| NC-2 | CRITICAL | `_FUNC_IDS_CPU` no runtime validation | Source code | **NEW** |
| NC-3 | CRITICAL | `FocalLoss` expects post-sigmoid — trap | Source code | **NEW** |
| NC-4 | CRITICAL | `pos_weight` computed but unused by ASL | Source code | **NEW** |
| TL-1 | CRITICAL | JK collapse faster with IMP-G3 | Training log | **NEW** |
| TL-2 | HIGH | Projection weight norm stagnant | Training log | Confirms prior |
| TL-3 | HIGH | Timestamp/DoS/UnusedReturn at 0.000 | Training log | Confirms prior |
| NH-1 | HIGH | `contract_path` excluded from collation | Source code | **NEW** |
| NH-2 | HIGH | `class_label_smoothing` dict unvalidated | Source code | **NEW** |
| NH-3 | HIGH | `weights_only=False` security surface | Source code | Confirms prior |
| NH-4 | HIGH | `_ckpt_state` memory leak | Source code | Confirms prior |
| NH-5 | HIGH | Prefix proj LR too conservative for cold start | Source code | **NEW** |
| NM-1 | MODERATE | 0.287 ceiling is data-determined | Analysis | Contradicts project conclusion |
| NM-2 | MODERATE | Solution sequencing needs revision | Analysis | **NEW** |
| NM-3 | MODERATE | DoS starvation (243 samples) | Data | Confirms prior |
| NM-4 | MODERATE | No negative anchors | Data | Confirms prior |
| NL-1 | LOW | Version strings stale | Source code | Confirms prior |
| NL-2 | LOW | Backup/cache files in source tree | Source code | Confirms prior |
| NL-3 | LOW | Dead `FocalLoss` code | Source code | **NEW** (extends NC-3) |
| NL-4 | LOW | EMITS edges near-zero (12 total) | Data | **NEW** |
| NL-5 | LOW | Empty validation/tools directories | Source code | Confirms prior |

**Total new findings: 10 | Confirmed from prior audit: 7 | Contradictions: 1 (v8.0-B interpretation)**
The full adversarial audit report is above. Here's a quick summary of the most actionable takeaways:

**5 most urgent actions:**

1. **Inject 100+ clean anchor contracts immediately** (OpenZeppelin + Solmate with all-zero labels, weighted at 10-15× in sampler) — this directly fixes the 0/3 behavioral test failure where safe contracts are scored as vulnerable.

2. **Fix the `gnn_to_bert_proj` cold-start problem** — either skip adding it to the optimizer until warmup ends (reconstruct optimizer at epoch 16), or add a higher initial LR multiplier (5.0→1.0 ramp over 3 epochs post-warmup). The current design means the projection starts from random init with a stale AdamW state.

3. **Apply CEI-order Reentrancy filter** to label_cleaner.py (Solution 1 from the external reviewer) — uses existing CFG edges in .pt files, no re-extraction needed, expects ~200-400 label removals.

4. **Fix the `FocalLoss` / `pos_weight` dead code trap** — `FocalLoss` expects post-sigmoid inputs but would receive raw logits if anyone switches `loss_fn="focal"`. `pos_weight` is computed but never used by ASL (the default loss). Both are maintenance hazards.

5. **Add SmartBugs Wild dataset** (~47K contracts with per-contract Slither labels instead of folder-level OR-labels) — this is the single highest-impact data addition. The BCCC dataset's OR-labeling is the fundamental constraint.

**Key disagreement with project:** The v8.0-B run (F1=0.2460, killed at ep11) was interpreted as "data cleaning can't break the ceiling." I believe this is wrong — it was killed too early while still improving. Cleaner labels make optimization harder and require more epochs, not fewer.