# EXP-L8: Node Feature Permutation Importance

**Layer:** 3  **Priority:** 2  **Status:** COMPLETE — PARTIALLY CONFIRMED (2026-05-30, corrected 2026-05-31)  
**Checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (Run 4, ep32, F1=0.3362)  
**Output:** `ml/logs/interpretability/exp_l8_permutation/`

---

## Correction Note (2026-05-31)

The original run of this experiment used stale hardcoded feature names that did not match the v8 schema. The script was fixed to import `FEATURE_NAMES` directly from `ml/src/preprocessing/graph_schema.py`. This document records the corrected results.

The v8 schema feature order (dims 0–10) is:
`type_id, visibility, uses_block_globals, view, payable, complexity, loc, return_ignored, call_target_typed, has_loop, external_call_count`

Several findings from the uncorrected run were artifacts of name mismatch (e.g., `has_loop` was printed in place of `view`, `is_modifier` in place of `complexity`). The importance values themselves were always correct — only the printed names were wrong.

---

## Hypothesis

Permuting each of the 11 node feature dimensions within a graph (preserving structure but destroying feature signal) and measuring the resulting GNN-only prediction change will reveal the feature importance ranking. Based on vulnerability semantics and prior experiments:

- `type_id` (dim 0) and `return_ignored` (dim 7) should rank in the top 4 overall — node identity is fundamental to graph learning; return_ignored is the explicit UnusedReturn signal.
- `uses_block_globals` (dim 2) and `has_loop` (dim 9) should rank in the top 6 — these are the primary Timestamp and loop-structure signals.
- If `type_id` dominates the ranking but `uses_block_globals` is low, it confirms that the GNN uses node type as the primary routing signal and specific vulnerability features have low marginal value.

---

## Method

For each of 277 sampled val-split graphs (300 requested, 23 skipped due to cache misses): GNN-only inference is run to produce baseline sigmoid probabilities `[10]` for each graph. The GNN-only path uses the GNN encoder's output, function-level mean/max pooling, the `gnn_eye_proj` projection, and `aux_gnn` classifier head — matching the GNN-eye contribution to the full model. For each of the 11 feature dimensions, each graph's feature column is permuted in-place (within-graph shuffle, preserving marginal distribution) and GNN inference re-run. Importance for feature d = mean over all graphs of `|permuted_pred - baseline_pred|` per class, yielding a `[11, 10]` matrix. Mean across classes gives the overall ranking.

---

## Results

### Feature Importance Ranking (Mean Across 10 Classes)

| Rank | Feature (v8 schema) | Dim | Mean Importance | Top Class (Importance) |
|------|---------------------|-----|-----------------|------------------------|
| 1 | type_id | 0 | 0.07860 | IntegerUO (0.1563) |
| 2 | external_call_count | 10 | 0.02623 | IntegerUO (0.0526) |
| 3 | view | 3 | 0.01932 | IntegerUO (0.0584) |
| 4 | complexity | 5 | 0.01912 | IntegerUO (0.0517) |
| 5 | visibility | 1 | 0.01831 | IntegerUO (0.0551) |
| 6 | return_ignored | 7 | 0.01623 | UnusedReturn (0.0238) |
| 7 | payable | 4 | 0.01408 | IntegerUO (0.0364) |
| 8 | loc | 6 | 0.01114 | IntegerUO (0.0315) |
| 9 | call_target_typed | 8 | 0.01038 | CallToUnknown (0.0267) |
| 10 | uses_block_globals | 2 | 0.00553 | IntegerUO (0.0145) |
| 11 | has_loop | 9 | 0.00532 | IntegerUO (0.0126) |

### Per-Class Importance for Key Features

| Feature | CallToUnknown | ExternalBug | IntegerUO | Reentrancy | Timestamp | UnusedReturn |
|---------|---------------|-------------|-----------|------------|-----------|--------------|
| type_id | 0.0816 | 0.0657 | 0.1563 | 0.0860 | 0.0380 | 0.0568 |
| external_call_count | 0.0326 | 0.0280 | 0.0526 | 0.0375 | 0.0107 | 0.0104 |
| uses_block_globals | 0.0036 | 0.0027 | 0.0145 | 0.0042 | **0.0113** | 0.0031 |
| return_ignored | 0.0134 | 0.0111 | 0.0463 | 0.0156 | 0.0030 | **0.0238** |
| has_loop | 0.0064 | 0.0046 | 0.0126 | 0.0057 | 0.0021 | 0.0029 |

### Pass/Fail Checks

| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| top-4 contains `type_id` (dim 0) | Yes | Yes (rank 1) | PASS |
| top-4 contains `return_ignored` (dim 7) | Yes | No (rank 6) | FAIL |
| **top-4 check** | both in top-4 | only `type_id` | **FAIL** |
| top-6 contains `uses_block_globals` (dim 2) | Yes | No (rank 10) | FAIL |
| top-6 contains `has_loop` (dim 9) | Yes | No (rank 11) | FAIL |
| **top-6 check** | all in top-6 | neither present | **FAIL** |

**Overall pass: FAIL**

### Timestamp-Specific Check (Post-Hoc)

`uses_block_globals` (dim 2) ranks **2nd for Timestamp** with importance 0.0113, compared to rank 10 globally (mean 0.0055). For Timestamp contracts, the block-globals feature is the 2nd most informative GNN feature after `type_id`. This means the "uses_block_globals ranks last" finding from the original (bugged) run was an artifact of stale feature names — the feature IS used for Timestamp, just not globally.

**Timestamp-specific check: PARTIALLY CONFIRMED** — `uses_block_globals` is informative for Timestamp (rank 2 per-class) but remains globally low because Timestamp is a rare class (n=3 in this val slice).

---

## Key Findings

1. **`type_id` dominates by a factor of 3:** Mean importance 0.0786 vs next-highest `external_call_count` at 0.0262 — a 3× gap. Node type is by far the most important feature for GNN-only predictions. GAT message-passing is type-conditioned via the node type embedding; permuting types fundamentally disrupts routing. The model uses node identity as the primary discriminant.

2. **`return_ignored` ranks 6th (not 8th as reported in the original run):** With corrected feature names, `return_ignored` (dim 7) moves from reported rank 8 to actual rank 6 with mean importance 0.0162. Its highest per-class importance is for UnusedReturn (0.0238), which is the expected class — the feature IS relevant for its intended class, just not enough to crack the top 4 overall.

3. **`uses_block_globals` ranks 10th globally but 2nd for Timestamp:** Mean importance 0.0055 overall, but 0.0113 for Timestamp — the 2nd-highest feature for that class. The low global rank is because Timestamp is rare (n=3 in this slice) and `uses_block_globals` is low-importance for other classes. This partially vindicates the feature: the model does use it for its intended class.

4. **`external_call_count` (rank 2) and structural features dominate globally:** The raw external call count is the second most important feature. `view`, `complexity`, `visibility` (ranks 3–5) are function-level structural properties. This confirms the model is learning structural complexity as its primary non-type signal, not fine-grained semantic vulnerability indicators.

5. **IntegerUO is the top class for most features:** This reflects IntegerUO's high prevalence (best-trained class, F1=0.647 from Run 4 ep30), not a semantic relationship. Permuting almost any feature shifts the GNN's IntegerUO predictions because it's the class the GNN is most sensitive to. Notably, `return_ignored`'s top class is UnusedReturn (0.0238) — the one correct semantic alignment in the ranking.

6. **`has_loop` and `call_target_typed` rank last:** Loop presence and call target typing (whether a call target is a typed interface vs. unknown) provide the least signal overall. `call_target_typed` is most important for CallToUnknown (0.0267) and Reentrancy (0.0220), which is semantically sensible.

---

## Implications for Architecture

- **`uses_block_globals` is class-specific, not globally important:** Its importance is concentrated in Timestamp (rank 2 per-class). Adding a targeted auxiliary loss for Timestamp on `uses_block_globals`-positive nodes could reinforce this signal without disrupting other classes.

- **`return_ignored` is in the right tier:** Rank 6 with top-class UnusedReturn confirms the feature is used correctly, but its low magnitude relative to `type_id` and `external_call_count` means the model treats it as a supplementary signal. A targeted auxiliary loss (as proposed in Sol-1) could strengthen it.

- **Size-proxy features dominate the global ranking:** `external_call_count` (rank 2) correlates with contract size/complexity, consistent with the size-shortcut hypothesis from EXP-S3. Removing or capping size-correlated features would force the model to attend to genuinely discriminative signals.

- **GNN-only path is type-routing, not feature-routing:** The 3× dominance of `type_id` means the GNN primarily routes messages by node type, then adjusts based on other features. Semantic binary features (`uses_block_globals`, `has_loop`) are being underweighted relative to their vulnerability-diagnostic value.

---

## Known Caveats

- This experiment uses GNN-only inference (`aux_gnn` head), not the full three-eye model. The transformer eye (GraphCodeBERT) may compensate for weak GNN feature usage.
- Permutation is within-graph (preserving marginal distributions) — cross-graph permutation would measure a different form of importance.
- 277 graphs is a small sample; feature importance rankings for near-tied features (e.g., view vs complexity at 0.01932 vs 0.01912) should be treated as approximate.
- Timestamp slice is n=3, making per-class Timestamp numbers noisy. The rank-2 finding for `uses_block_globals` on Timestamp is directionally consistent but not statistically robust.
- `type_id` is stored as `float(type_id) / 12.0` — permuting it permutes the type IDs themselves, which affects edge embedding routing. Its importance is partly a structural effect (disrupting GNN layer routing) rather than a pure feature value effect.
