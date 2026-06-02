# EXP-L4: Gradient Saliency — Node Features and Node Identity

**Layer:** 3  **Priority:** 1  **Status:** COMPLETE (2026-05-30)  
**Checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (Run 4, ep32, F1=0.3362)  
**Output:** `ml/logs/interpretability/exp_l4_gradient_saliency/`

---

## Hypothesis

The gradient of each class logit with respect to the input node feature matrix X will reveal which node features and node types drive class predictions. Specifically:

- **Timestamp:** `uses_block_globals` (dim 2) should account for ≥20% of total saliency, because `block.timestamp` access is the defining signal.
- **Reentrancy:** Saliency should concentrate on `CFG_NODE_CALL` (type 8) and `CFG_NODE_WRITE` (type 9) nodes (≥20% combined), because reentrancy exploits the external-call / state-write pattern at CFG level.

If saliency goes to generic features (external_call_count, fn_call_count) or high-level node types (FUNCTION, CONTRACT) instead, it indicates the model is not leveraging fine-grained CFG structure.

---

## Method

For each class C with at least one positive val-split example, up to 200 positive contracts are selected. For each contract the node feature tensor `x` is detached and re-attached with `requires_grad=True`, then a single forward pass is run and `logits[0, C].backward()` is called. Saliency per node is `|x.grad|` (shape `[N, 11]`). All contracts per class are aggregated by mean: per-feature saliency is normalised to sum to 1.0 (fraction of total); per-node-type saliency is computed as each type's share of the total absolute gradient. The primary method is backward (gradient), with permutation as fallback — all 200 contracts used backward successfully.

---

## Results

### Check Results

| Class | Check | Value | Threshold | Result |
|-------|-------|-------|-----------|--------|
| Timestamp | `uses_block_globals` (dim 2) fraction | 10.1% | ≥20% | **FAIL** |
| Reentrancy | CFG_NODE_CALL + CFG_NODE_WRITE combined fraction | 9.5% | ≥20% | **FAIL** |
| Reentrancy | CONTRACT + FUNCTION fraction >50% (shortcut flag) | ~71.7% | >50% triggers flag | **FLAG RAISED** |

### Per-Class Top-3 Feature Dims (normalized saliency fraction)

| Class | Rank 1 | Rank 2 | Rank 3 |
|-------|--------|--------|--------|
| CallToUnknown | external_call_count (21.3%) | fn_call_count (10.9%) | state_reads (9.0%) |
| DenialOfService | external_call_count (23.9%) | fn_call_count (10.1%) | uses_block_globals (9.2%) |
| ExternalBug | external_call_count (22.8%) | fn_call_count (11.4%) | visibility (9.1%) |
| GasException | external_call_count (23.9%) | fn_call_count (10.5%) | uses_block_globals (8.8%) |
| IntegerUO | external_call_count (21.5%) | fn_call_count (10.4%) | uses_block_globals (9.3%) |
| MishandledException | external_call_count (23.5%) | fn_call_count (10.7%) | visibility (8.9%) |
| Reentrancy | external_call_count (23.5%) | fn_call_count (10.9%) | visibility (8.7%) |
| Timestamp | external_call_count (21.4%) | fn_call_count (10.9%) | uses_block_globals (10.1%) |
| TransactionOrderDependence | external_call_count (22.8%) | fn_call_count (11.1%) | visibility (9.0%) |
| UnusedReturn | external_call_count (21.3%) | fn_call_count (10.8%) | visibility (10.0%) |

### Per-Class Top-3 Node Types (fraction of total gradient)

| Class | Rank 1 | Rank 2 | Rank 3 |
|-------|--------|--------|--------|
| CallToUnknown | FUNCTION (50.2%) | CFG_NODE_OTHER (22.6%) | CFG_NODE_WRITE (4.6%) |
| DenialOfService | FUNCTION (42.6%) | CFG_NODE_OTHER (27.2%) | FALLBACK (6.5%) |
| ExternalBug | FUNCTION (45.5%) | CFG_NODE_OTHER (25.6%) | CFG_NODE_READ (4.8%) |
| GasException | FUNCTION (45.5%) | CFG_NODE_OTHER (24.9%) | CONSTRUCTOR (5.0%) |
| IntegerUO | FUNCTION (45.1%) | CFG_NODE_OTHER (24.6%) | CONSTRUCTOR (4.7%) |
| MishandledException | FUNCTION (47.2%) | CFG_NODE_OTHER (24.7%) | CFG_NODE_CALL (5.0%) |
| Reentrancy | FUNCTION (46.4%) | CFG_NODE_OTHER (26.3%) | CFG_NODE_CALL (5.0%) |
| Timestamp | FUNCTION (48.6%) | CFG_NODE_OTHER (20.4%) | CFG_NODE_READ (6.2%) |
| TransactionOrderDependence | FUNCTION (47.7%) | CFG_NODE_OTHER (24.8%) | CFG_NODE_WRITE (4.5%) |
| UnusedReturn | FUNCTION (46.8%) | CFG_NODE_OTHER (27.8%) | CFG_NODE_CALL (4.7%) |

---

## Key Findings

1. **Feature monoculture:** `external_call_count` (dim 10) dominates saliency for all 10 classes at 21–24%, followed by `fn_call_count` (dim 5) at 10–11%. This is class-undifferentiated — the model attends to the same two features regardless of the vulnerability type being predicted. This is the strongest finding from this experiment.

2. **Timestamp shortcut confirmed:** `uses_block_globals` (dim 2) ranks 3rd for Timestamp at only 10.1%, far below the 20% threshold. EXP-S3 previously found a graph-size shortcut (Cohen's d=1.657); this experiment confirms the feature-level signal is also weak — the model is not using the semantically correct feature for Timestamp classification.

3. **CFG structure not used for Reentrancy:** `CFG_NODE_CALL + CFG_NODE_WRITE` combined saliency is only 9.5% for Reentrancy (threshold: 20%). FUNCTION nodes account for 46.4% and CFG_NODE_OTHER for 26.3% — the shortcut flag is raised (FUNCTION+CONTRACT > 50%). The model predicts reentrancy based on function-level features (external_call_count on FUNCTION nodes), not on CFG-level call-write patterns.

4. **CFG_NODE_OTHER dominance:** This catch-all CFG node type consistently ranks 2nd (20–28%) across all classes, absorbing gradient that should be going to semantically meaningful CFG subtypes (CALL, WRITE, READ). This suggests the model's CFG signal is generic rather than structured.

5. **No class differentiation at node-type level:** The top-2 node types (FUNCTION, CFG_NODE_OTHER) account for 68–74% of saliency for every class, with only minor variation in the 3rd rank. This means the GNN is not exploiting the full node type vocabulary in a class-specific way.

---

## Implications for Architecture

- **GNN is not learning CFG semantics:** Saliency concentrates on FUNCTION nodes and generic CFG nodes rather than type-specific CFG nodes, regardless of class. Phase 2 (CFG/ICFG) is architecturally active but not producing class-discriminative CFG-level gradients. This is consistent with EXP-L1 finding Phase 2 has the lowest JK weight (0.322) and EXP-L2 finding CFG ablation has near-zero effect.

- **`external_call_count` is a proxy shortcut:** This feature correlates with contract complexity, not specifically with any vulnerability. Removing or normalising this feature might force the model to learn more specific signals.

- **Timestamp fix requires data-level intervention:** Even with `uses_block_globals` present in the feature vector, gradient saliency shows only 10.1% attention to it. The size-shortcut found in EXP-S3 is likely suppressing the feature-level signal. Data balancing for Timestamp (Sol-3 in EXECUTION_PLAN) should be prioritised.

- **Phase 3 (REVERSE_CONTAINS hierarchy):** No node type corresponding to hierarchical structure shows up prominently, suggesting Phase 3's main value is normalisation/refinement rather than structural routing.

---

## Known Caveats

- Only 48 positive Timestamp examples exist in val split — saliency statistics for this class are noisier than others.
- Gradient saliency is linear: it measures the first-order sensitivity of the logit to input changes at the current model point. Non-linear interactions between features are not captured.
- `external_call_count` is correlated with graph size (larger contracts tend to have more external calls), so part of its saliency may be absorbing the size-shortcut signal from EXP-S3.
- The backward method was used for all 200 contracts per class — no permutation fallback was needed — so results are consistent.
