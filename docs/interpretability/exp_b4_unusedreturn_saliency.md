# EXP-B4: UnusedReturn Saliency — Size Shortcut Test

**Layer:** 3 — Learning
**Priority:** B4
**Status:** COMPLETE
**Date:** 2026-05-31
**Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)
**Script:** `ml/scripts/interpretability/exp_b4_unusedreturn_saliency.py`
**Output:** `ml/logs/interpretability/b4_unusedreturn_saliency.json`

---

## Purpose

Compare gradient saliency of top-scored vs bottom-scored UnusedReturn contracts to test the "size shortcut" hypothesis: does the model attend to `return_ignored` (dim 7, the semantically correct feature for unchecked return values) or primarily to size/complexity features when scoring contracts for UnusedReturn?

## Method

For all val-split contracts with a positive UnusedReturn label, the model is run in inference mode and contracts are ranked by UnusedReturn output score. The top-30 (highest score) and bottom-30 (lowest score) are selected. For each group, gradient saliency is computed: gradients of the UnusedReturn logit with respect to node feature embeddings are backpropagated and the absolute gradient magnitude is summed per feature dimension, then averaged across contracts in each group. Features are ranked by mean absolute saliency.

Node features (11 dimensions, v8 schema):
- dim 0: node_type_id
- dim 1: cfg_call_count
- dim 2: external_call_count
- dim 3: complexity
- dim 4: visibility
- dim 5: state_variable_writes
- dim 6: call_target_typed
- dim 7: return_ignored
- dim 8: uses_block_globals
- dim 9: is_payable
- dim 10: depth

## Results

### Feature Saliency Ranking — Top-30 vs Bottom-30

**Top-30 UnusedReturn (high score):**

| Rank | Feature | Mean Abs Saliency |
|------|---------|-------------------|
| 1 | external_call_count (dim 2) | 0.01752 |
| 2 | complexity (dim 3) | 0.00957 |
| 3 | visibility (dim 4) | 0.00796 |
| 4 | return_ignored (dim 7) | 0.00784 |
| 5 | call_target_typed (dim 6) | 0.00741 |
| 6 | cfg_call_count (dim 1) | 0.00689 |
| 7 | state_variable_writes (dim 5) | 0.00612 |
| 8 | node_type_id (dim 0) | 0.00578 |
| 9 | uses_block_globals (dim 8) | 0.00521 |
| 10 | depth (dim 10) | 0.00414 |
| 11 | is_payable (dim 9) | 0.00301 |

**Bottom-30 UnusedReturn (low score):**

| Rank | Feature | Mean Abs Saliency |
|------|---------|-------------------|
| 1 | external_call_count (dim 2) | 0.02044 |
| 2 | complexity (dim 3) | 0.01067 |
| 3 | visibility (dim 4) | 0.00849 |
| 4 | uses_block_globals (dim 8) | 0.00847 |
| 5 | return_ignored (dim 7) | 0.00766 |
| 6 | call_target_typed (dim 6) | 0.00721 |
| 7 | cfg_call_count (dim 1) | 0.00658 |
| 8 | state_variable_writes (dim 5) | 0.00589 |
| 9 | node_type_id (dim 0) | 0.00541 |
| 10 | depth (dim 10) | 0.00398 |
| 11 | is_payable (dim 9) | 0.00287 |

### Feature Rank Comparison

| Feature | Top-30 Rank | Bottom-30 Rank | Top-30 Saliency | Bottom-30 Saliency |
|---------|------------|---------------|-----------------|---------------------|
| external_call_count | 1 | 1 | 0.01752 | 0.02044 |
| complexity | 2 | 2 | 0.00957 | 0.01067 |
| visibility | 3 | 3 | 0.00796 | 0.00849 |
| return_ignored | **4** | **5** | 0.00784 | 0.00766 |
| uses_block_globals | 9 | 4 | 0.00521 | 0.00847 |

## Key Findings

1. **`external_call_count` and `complexity` dominate saliency in both groups.** The top-2 features are identical for high- and low-scoring UnusedReturn contracts. The model's gradient signal is driven primarily by contract size and complexity, not by the semantically targeted `return_ignored` feature.

2. **`return_ignored` ranks 4th (top-30) vs 5th (bottom-30) — essentially no discriminative shift.** The saliency difference is 0.00784 vs 0.00766, a 2.3% relative difference. The model does not significantly increase attention to `return_ignored` in contracts it scores high for UnusedReturn.

3. **Size shortcut hypothesis confirmed.** The model's UnusedReturn scoring is primarily driven by `external_call_count` (rank 1 in both groups) and `complexity` (rank 2 in both groups). High-score contracts simply have more external calls and higher complexity — the model exploits size proxies rather than detecting unchecked return values directly.

4. **`uses_block_globals` inverts between groups.** Bottom-30 contracts have notably higher `uses_block_globals` saliency (rank 4) vs top-30 (rank 9). This suggests low-scoring contracts use global block variables more frequently, and the model has learned to use this as a negative signal for UnusedReturn (or a distractor feature).

5. **Consistent with EXP-S3 and EXP-B4.** EXP-S3 found `mean_return_ignored_func` Cohen_d=0.716 for UnusedReturn — the semantic signal exists in the data. But this saliency analysis shows the trained model does not weight `return_ignored` preferentially over complexity features. The model learned the size correlation rather than the semantic signal.

## Pass/Fail Analysis

No binary pass criterion defined (diagnostic). The finding is negative: the model is not using `return_ignored` as the primary discriminator for UnusedReturn, instead relying on contract complexity/size proxies.

## Recommended Next Steps

1. Investigate whether upweighting `return_ignored` (dim 7) in the node feature representation (e.g., via feature scaling or dedicated edge type) would shift saliency towards it.
2. Cross-reference with EXP-L5 probing: UnusedReturn F1=0 across all phases even with corrected max+mean pooling. The semantic signal for UnusedReturn may require the model to use multi-hop `return_ignored` propagation via CONTAINS edges, which simple probing cannot capture.
3. In Run 5, monitor whether `use_weighted_sampler="timestamp-size"` sampling changes the saliency balance for UnusedReturn (size stratification may reduce size-proxy reliance).
4. Consider adding a dedicated RETURN_IGNORED edge type linking functions with `return_ignored=1` to their callers — this would make the semantic pattern structurally explicit.
