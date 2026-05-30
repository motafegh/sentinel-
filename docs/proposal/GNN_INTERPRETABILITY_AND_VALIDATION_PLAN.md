# SENTINEL GNN — Interpretability & Architecture Validation Plan

**Status:** Proposed — ready to execute  
**Date:** 2026-05-30  
**Scope:** GNN encoder + graph preprocessing layer  
**Motivation:** Run 4 finished at 0.3 F1. Before adding more architecture, verify that the architecture we already built is actually doing what we designed it to do.

---

## 0. The Core Problem

We built a sophisticated architecture with specific theoretical hopes:

| What we added | What we hoped it would learn |
|---|---|
| CFG nodes + CONTROL_FLOW edges | Execution order within a function |
| ICFG-Lite (CALL_ENTRY / RETURN_TO) | Cross-function CEI violation (reentrancy) |
| DEF_USE edges | Value flow from arithmetic ops into state writes |
| INHERITS edges | Vulnerability propagation through inheritance |
| Reverse CONTAINS (Phase 3) | CFG signal rising up to FUNCTION nodes |
| JK aggregation (3 phases) | Each phase capturing a different granularity |
| Parent feature inheritance for CFG nodes | CFG nodes carry their function's context |

The problem: the only test so far has been "run training, look at F1." That conflates three completely separate questions that need different diagnostics.

---

## 1. The Three-Layer Framework

Every architecture assumption lives at exactly one of these three layers. Confusing them wastes weeks of debugging time.

```
Layer 1: STRUCTURE
  "Does the extracted graph actually encode the vulnerability pattern?"
  → Testable WITHOUT any model. Pure graph analysis.
  → If this fails, no GNN can ever learn it.

Layer 2: EXPRESSIVITY
  "Can message-passing on this graph structure even distinguish the pattern?"
  → Testable with graph properties + k-hop analysis. No training.
  → If this fails, training more won't help. Architecture needs to change.

Layer 3: LEARNING
  "Did the trained model actually use the encoded pattern?"
  → Requires a trained model (even poorly trained).
  → If this passes but F1 is low, the problem is data / loss / training, not architecture.
```

The experiments below are grouped by layer. **Always run Layer 1 first.** A Layer 3 failure is meaningless if Layer 1 already fails.

---

## 2. Experiment List

Each experiment has:
- **ID** — reference handle
- **Layer** — which of the three questions it answers
- **Priority** — P0 (must do first), P1 (high value), P2 (if time allows)
- **Effort** — hours of work to implement and run
- **What it proves / disproves**
- **Pass/fail criteria** — concrete threshold, not subjective

---

### LAYER 1 — Does the Graph Structure Encode the Pattern?

---

#### EXP-S1: CEI Structural Trace Audit
**Layer:** 1 (Structure)  
**Priority:** P0  
**Effort:** 2–4 hours

**What it does:**  
For each vulnerability class, manually trace whether the expected vulnerability pattern exists as a reachable path in the extracted graph. Pick 10 ground-truth positive contracts per class from the validation set.

**Patterns to trace:**

| Class | Expected graph path |
|---|---|
| Reentrancy | CFG_CALL → [CONTROL_FLOW] → CFG_WRITE (without intervening CHECK) **or** CFG_CALL → [CALL_ENTRY] → callee CFG → [RETURN_TO] → CFG_WRITE |
| IntegerUO | arithmetic CFG_OTHER → [DEF_USE] → CFG_WRITE with no CFG_CHECK between them |
| Timestamp | node with `uses_block_globals=1.0` → [CONTROL_FLOW or DEF_USE] → CFG_CHECK or CFG_WRITE |
| UnusedReturn | CFG_CALL node with `return_ignored=1.0` |
| TOD | two FUNCTION nodes that both WRITE the same STATE_VAR with payable=1.0 on at least one |
| MishandledException | CFG_CALL node → no CFG_CHECK as CONTROL_FLOW successor |

**How to run:**  
Use `full_graph_diagnostic.py` on known positive contracts. Read `node_metadata` to identify nodes by name and type. Trace edges manually or write a simple DFS path finder.

**Pass criteria:**  
≥7/10 positive contracts show the expected pattern in their graph structure. Any class below 5/10 means the extractor is NOT encoding the pattern — architecture changes won't help.

**Script to write:** `ml/scripts/interpretability/exp_s1_structural_trace.py`

---

#### EXP-S2: Edge Presence Rate Per Vulnerability Class
**Layer:** 1 (Structure)  
**Priority:** P0  
**Effort:** 1 hour (already partially done by `edge_activation.py`)

**What it does:**  
For each of the 11 edge types, compute: in positive contracts for class C, what fraction have ≥1 edge of that type? Then compare to the negative contracts. A high positive/negative ratio means the edge type correlates structurally with the class.

**Key question:** Do CONTROL_FLOW and CALL_ENTRY edges appear more frequently in reentrancy positives than negatives? Do DEF_USE edges appear more in IntegerUO positives?

**Pass criteria per class:**  
The edges we designed for a class should show ≥1.3× enrichment in positives vs. negatives. If CONTROL_FLOW edges are equally present in reentrancy positives and negatives, the structural encoding is not class-specific.

**Note:** `edge_activation.py` already computes presence rate but does not compute the positive/negative ratio. Extend it with this ratio column.

**Script to extend:** `ml/scripts/edge_activation.py` — add `enrichment_ratio = per_class_pct / baseline_pct`

---

#### EXP-S3: Graph Size and Feature Distribution Per Class
**Layer:** 1 (Structure)  
**Priority:** P1  
**Effort:** 2 hours

**What it does:**  
For each vulnerability class, compare the distributions of:
- Number of CFG nodes
- Number of external calls (sum of `external_call_count` across FUNCTION nodes)
- Number of CFG_NODE_CALL nodes specifically
- Number of DEF_USE edges
- Max depth of CONTROL_FLOW DAG (topological longest path)

**Why it matters:**  
If reentrancy positives have CFG_NODE_CALL count = 1.2 (barely more than negatives), the node-level feature signal is weak and the model must rely on structural path information. If they have count = 8.5, the model can cheat by just counting call nodes without learning any path.

**Pass criteria:**  
This experiment has no hard pass/fail — it maps the exploitable shortcuts in your feature space. Any feature that cleanly separates classes will be learned by the model regardless of graph structure.

**Script to write:** `ml/scripts/interpretability/exp_s3_feature_distribution.py`

---

#### EXP-S4: CFG Path Validity — Does ICFG-Lite Close the Reentrancy Loop?
**Layer:** 1 (Structure)  
**Priority:** P0  
**Effort:** 3 hours

**What it does:**  
For known reentrancy contracts, verify the full ICFG-Lite path closes correctly:
```
[caller function]
  CFG_CALL node (external call)
    → [CALL_ENTRY] → [callee ENTRYPOINT CFG node]
                         ↓ (callee CFG path)
                    [callee terminal CFG node]
                         → [RETURN_TO] → [caller's next CFG node after the call]
                                              ↓
                                         CFG_WRITE node (the re-entered state write)
```

This checks whether the ICFG-Lite actually connects the re-entry site to the vulnerable write. Multi-level reentrancy (A→B→A) is especially important: the second CALL_ENTRY and RETURN_TO must exist.

**Pass criteria:**  
The full path above exists in ≥6/10 known reentrancy contracts. Failure means reentrancy cannot be detected structurally at all — the graph is missing the cross-function connection.

**Script to write:** `ml/scripts/interpretability/exp_s4_icfg_path_audit.py`

---

### LAYER 2 — Can Message-Passing Even See the Pattern?

---

#### EXP-E1: K-Hop Receptive Field Analysis
**Layer:** 2 (Expressivity)  
**Priority:** P1  
**Effort:** 3 hours

**What it does:**  
For a given contract graph, compute: for each FUNCTION node, what is the set of nodes reachable within k hops (k=1…8) following the edge types used in each phase?

For reentrancy: can the FUNCTION node "see" both the CFG_CALL node and the CFG_WRITE node in its k-hop neighborhood at the depth used by Phase 2? Can the CONTRACT node see the CFG chain at the depth used by Phase 3?

This is computable without any trained weights — it is purely graph topology.

**Key test:**  
Pick one known reentrancy contract. For the FUNCTION that contains the vulnerable CFG:
- After Phase 1 (depth 2, STRUCT edges): what CFG nodes are visible?
- After Phase 2 (depth 3 more, CFG edges): has the ordered signal propagated correctly?
- After Phase 3 (depth 3 more, CONTAINS/REVERSE): did it rise to FUNCTION level?

**Pass criteria:**  
The CFG_CALL and CFG_WRITE nodes that form the reentrancy pattern are within the receptive field of the contract-level representation after 8 hops. If they are not reachable at all within 8 hops, message-passing cannot learn the pattern regardless of weights.

**Script to write:** `ml/scripts/interpretability/exp_e1_receptive_field.py`

---

#### EXP-E2: Weisfeiler-Lehman (WL) Distinguishability Test
**Layer:** 2 (Expressivity)  
**Priority:** P1  
**Effort:** 4 hours

**What it does:**  
The 1-WL test is the expressivity ceiling of any standard GNN (including GAT). Two graphs that are WL-equivalent will produce identical GNN outputs regardless of weights or training. The WL test computes colored-neighborhood hashes iteratively.

Compare WL hash sequences between:
- A reentrancy-positive contract vs. a structurally-similar negative contract
- An IntegerUO positive vs. a negative with similar function count and features
- A known-vulnerable vs. known-safe contract of the same vulnerability type

**How:**  
Implement k-WL hashing using node type + sorted neighbor hashes at each round. Compare hash distributions at rounds 1–8.

**Pass criteria:**  
Vulnerable and safe contract pairs that are WL-equivalent at round 8 cannot be distinguished by any standard GNN. If ≥30% of your reentrancy pairs are WL-equivalent → there is a fundamental expressivity ceiling that no amount of training will overcome. You would need higher-order GNNs or explicit path features.

**Script to write:** `ml/scripts/interpretability/exp_e2_wl_distinguishability.py`

---

#### EXP-E3: Message-Passing Information Propagation Simulation
**Layer:** 2 (Expressivity)  
**Priority:** P1  
**Effort:** 3 hours

**What it does:**  
Initialize the GNN with **random weights** and run a forward pass. For each node, record which other nodes contributed most to its final embedding (approximated by correlation between source node features and target node embedding after the pass).

This shows the actual information flow through your specific graph structure, with your specific phase/edge-type routing, without needing trained weights.

**Key question:**  
After Phase 2, do CFG_WRITE nodes in reentrancy contracts receive a meaningfully different contribution from CFG_CALL nodes than CFG_WRITE nodes in safe contracts? If random-weight information flow is identical → the graph topology doesn't distinguish them.

**Pass criteria:**  
For reentrancy pairs (vulnerable vs. safe), the Phase 2 output of the CFG_WRITE node that is the "vulnerable write" differs by >0.1 cosine distance from the same node in safe contracts, even with random weights.

**Script to write:** `ml/scripts/interpretability/exp_e3_message_propagation_sim.py`

---

#### EXP-E4: Direction Sensitivity Test (Directed vs. Undirected CFG)
**Layer:** 2 (Expressivity)  
**Priority:** P2  
**Effort:** 2 hours

**What it does:**  
Your CONTROL_FLOW edges are directed (CFG_A → CFG_B means A executes before B). This directional signal is the key theoretical advantage over undirected graph methods. But GAT aggregation is over the incoming neighbor set — the direction only matters if the incoming vs. outgoing sets are different.

Test: for known reentrancy pairs, compute whether the ordered set of CFG nodes (sorted by CONTROL_FLOW predecessors) differs between vulnerable and safe contracts even using only local 1-hop neighborhoods.

If reversing the CONTROL_FLOW edges (making them undirected) produces the same WL hash distribution → direction is not actually helping.

**Pass criteria:**  
Directed CFG produces ≥10% more WL-distinguishable pairs than undirected CFG.

---

### LAYER 3 — Did the Trained Model Actually Learn to Use the Pattern?

All Layer 3 experiments require a trained model checkpoint. They work even with the current 0.3 F1 checkpoint — the model learned *something*, and these tools reveal what.

---

#### EXP-L1: JK Attention Weight Distribution Per Vulnerability Class
**Layer:** 3 (Learning)  
**Priority:** P0  
**Effort:** 1 hour

**What it does:**  
Your JK attention module already stores `last_weights[K]` (mean per-phase attention) and `last_weight_stds[K]` (std). In eval mode with `use_jk=True`, it stores full per-node weights.

Run inference on your validation split, separated by ground-truth class. For each class, collect the per-contract JK weights [phase1, phase2, phase3] and plot the distribution.

**Key hypotheses to test:**

| Class | Expected JK pattern | Rationale |
|---|---|---|
| Reentrancy | High Phase 2 weight | CEI detection requires CFG/ICFG |
| IntegerUO | High Phase 2 weight | DEF_USE chains |
| Timestamp | High Phase 1 weight | Node feature `uses_block_globals` is sufficient |
| UnusedReturn | High Phase 1 weight | Node feature `return_ignored` is sufficient |
| INHERITS-dependent | High Phase 1 weight | Structural inheritance pattern |

**Pass criteria:**  
At least 3 of the 10 classes show Phase 2 as the dominant phase (mean weight >0.40) when they are expected to. Uniform Phase 1 dominance across all classes would indicate the CFG/ICFG phases are being ignored.

**JK entropy** (already computed as `jk_entropy`): values near 0 mean one phase dominates entirely (collapsed). Values near `log(3)=1.099` mean all phases equally used. Report per class.

**Script to write:** `ml/scripts/interpretability/exp_l1_jk_weight_analysis.py`

---

#### EXP-L2: Inference-Time Edge Type Ablation
**Layer:** 3 (Learning)  
**Priority:** P0  
**Effort:** 2 hours

**What it does:**  
For each of the 11 edge types (0–10), run inference with that edge type zeroed out (replace its edge embeddings with the zero vector or remove those edges entirely). Measure the prediction change for each vulnerability class.

```
For edge_type in range(11):
    ablated_edge_attr = edge_attr.clone()
    ablated_edge_attr[edge_attr == edge_type] = MASK_VALUE
    pred_ablated = model(x, edge_index, ablated_edge_attr, batch)
    delta[edge_type][class_c] = pred_original[:, c] - pred_ablated[:, c]
```

Run on 200 contracts per class (100 positive, 100 negative).

**Expected results:**

| Edge type | Class it should matter for | Expected delta magnitude |
|---|---|---|
| CONTROL_FLOW (6) | Reentrancy, IntegerUO | High (>0.05 probability shift) |
| CALL_ENTRY (8) | Reentrancy | High |
| RETURN_TO (9) | Reentrancy | Moderate |
| DEF_USE (10) | IntegerUO, UnusedReturn | Moderate |
| READS/WRITES (1,2) | TOD, Reentrancy | Moderate |
| INHERITS (4) | All classes | Low (structural only) |
| EMITS (3) | Nothing specific | Near zero |

**Pass criteria:**  
CONTROL_FLOW + CALL_ENTRY ablation causes ≥0.05 mean probability drop for Reentrancy positives. If ablating ALL CFG edges (types 6,8,9,10 together) causes <0.02 total drop → the model is not using the control-flow architecture at all.

**Script to write:** `ml/scripts/interpretability/exp_l2_edge_ablation.py`

---

#### EXP-L3: GAT Attention Weight Visualization
**Layer:** 3 (Learning)  
**Priority:** P1  
**Effort:** 3 hours

**What it does:**  
PyG's GATConv supports `return_attention_weights=True`. Modify each phase's forward pass to return attention weights, then visualize which edges receive the highest attention for specific contracts.

For Phase 2 (CFG phase), the question is: for a reentrancy-positive contract, do the CONTROL_FLOW and CALL_ENTRY edges on the path from CFG_CALL to CFG_WRITE receive higher attention than other edges?

Visualize as a subgraph with edge width = attention weight. Use `node_metadata` for node labels.

**For each of:**
- Phase 1 (structural): which CALLS/READS/WRITES edges have highest attention?
- Phase 2 (CFG): which CONTROL_FLOW and ICFG edges have highest attention?
- Phase 3 (aggregation): which REVERSE_CONTAINS edges carry the most signal upward?

**Pass criteria:**  
Qualitative — does the highest-attention subgraph for a reentrancy contract form a path that resembles a CEI violation? If high attention is on random structural edges unrelated to the vulnerability → Phase 2 is not doing CEI detection.

**Script to write:** `ml/scripts/interpretability/exp_l3_attention_visualization.py`

---

#### EXP-L4: Gradient Saliency — Node Features and Node Identity
**Layer:** 3 (Learning)  
**Priority:** P1  
**Effort:** 2 hours

**What it does:**  
Compute the gradient of the prediction for class C with respect to the node feature matrix X. High gradient on node i, feature j means "changing this feature would most change the prediction."

```python
x = graph.x.clone().requires_grad_(True)
logits = model.gnn_encoder(x, edge_index, batch, edge_attr)
logits[:, class_idx].sum().backward()
saliency = x.grad.abs()  # [N, 11]
```

For each vulnerability class:
1. Which node TYPES have the highest mean saliency? (CFG_NODE_CALL? FUNCTION? CONTRACT?)
2. Which feature DIMENSIONS have the highest mean saliency? (external_call_count? uses_block_globals? return_ignored?)

**The key diagnostic:**  
If `external_call_count` (dim 10) accounts for >60% of gradient magnitude for reentrancy predictions, the model is learning "contracts with many external calls → reentrancy," not learning the CEI pattern. This is a valid but shallow signal — it means more contract-level features, not graph structure, are driving predictions.

**Pass criteria:**  
For reentrancy: saliency is distributed across CFG_NODE_CALL and CFG_NODE_WRITE nodes (not concentrated on CONTRACT or FUNCTION nodes). For Timestamp: high saliency on `uses_block_globals` feature (expected, this is by design).

**Script to write:** `ml/scripts/interpretability/exp_l4_gradient_saliency.py`

---

#### EXP-L5: Probing Classifiers Per Phase
**Layer:** 3 (Learning)  
**Priority:** P1  
**Effort:** 4 hours

**What it does:**  
After training, freeze the entire model. Extract node embeddings at the output of each phase (Phase 1, Phase 2, Phase 3). For each phase, train a simple **linear probe** (logistic regression on top of pooled phase embeddings) to predict each vulnerability class.

```
Phase 1 probe accuracy → structural signal
Phase 2 probe accuracy → structural + control-flow signal  
Phase 3 probe accuracy → full aggregated signal
```

If Phase 2 probe outperforms Phase 1 probe on reentrancy → Phase 2 (CFG/ICFG) adds learnable signal beyond structure alone. If Phase 2 probe performs the same as Phase 1 → the CFG phase is not adding new separable information.

**The probes are linear, so this directly tests whether the signal is linearly accessible in the embeddings (not just present but buried).**

**Pass criteria:**  
Phase 2 probe accuracy on Reentrancy exceeds Phase 1 probe by ≥3 percentage points. Any phase where the probe performs at chance level means that phase is not encoding the vulnerability pattern in a linearly-accessible way.

**Script to write:** `ml/scripts/interpretability/exp_l5_probing_classifiers.py`

---

#### EXP-L6: Counterfactual Contract Testing
**Layer:** 3 (Learning)  
**Priority:** P1  
**Effort:** 6–8 hours

**What it does:**  
Create minimal synthetic Solidity contracts that are **identical except for the single structural difference** being tested. Run them through the full pipeline and compare model predictions.

**Test pairs to write:**

**Pair A: CEI violation vs. correct (reentrancy)**
```solidity
// Vulnerable: call before write
function withdraw(uint amount) external {
    msg.sender.call{value: amount}("");  // call first
    balances[msg.sender] -= amount;      // write after
}

// Safe: write before call
function withdraw(uint amount) external {
    balances[msg.sender] -= amount;      // write first
    msg.sender.call{value: amount}("");  // call after
}
```
Expected: vulnerable scores higher on Reentrancy class.

**Pair B: Unchecked arithmetic vs. checked (IntegerUO)**
```solidity
// Vulnerable
uint256 result = a + b;  // no overflow check
balances[msg.sender] = result;

// Safe
require(a + b >= a, "overflow");
uint256 result = a + b;
balances[msg.sender] = result;
```

**Pair C: block.timestamp in condition vs. not (Timestamp)**
```solidity
// Vulnerable
require(block.timestamp > deadline);

// Safe
require(block.number > deadlineBlock);
```

**Pair D: Inheritance chain with vulnerability**
```solidity
// Contract A has the reentrancy bug
// Contract B inherits A
// Expected: B also scores high on Reentrancy
```

**Pass criteria:**  
For each pair, the vulnerable variant scores higher than the safe variant on the corresponding vulnerability class. Failure means the model is not learning the specific structural difference being tested — it's relying on something else.

**Script to write:** `ml/scripts/interpretability/exp_l6_counterfactual_contracts.py`  
**Contracts to write:** `ml/scripts/interpretability/test_contracts/` (12 files)

---

#### EXP-L7: Per-Class Calibration and Decision Boundary Analysis
**Layer:** 3 (Learning)  
**Priority:** P2  
**Effort:** 2 hours

**What it does:**  
Plot calibration curves for each class (predicted probability vs. actual positive rate). Also plot the precision-recall curve broken down by contract size (number of nodes), contract complexity (CFG depth), and edge count.

**Key question:**  
Does performance degrade on contracts with many CFG nodes? If small contracts (few nodes, short CFG) show much higher F1 than large contracts → the model may be exploiting size-correlated features (has_loop, external_call_count) rather than structural patterns.

**Pass criteria:**  
F1 on contracts with >100 CFG nodes is within 10 percentage points of F1 on contracts with <20 CFG nodes, per class.

**Script to write:** `ml/scripts/interpretability/exp_l7_calibration_size_analysis.py`

---

#### EXP-L8: Feature Importance via Permutation (Node Feature Shuffle)
**Layer:** 3 (Learning)  
**Priority:** P2  
**Effort:** 2 hours

**What it does:**  
For each of the 11 node feature dimensions, shuffle that dimension's values across all nodes in the batch (destroying its signal) and measure prediction change.

This complements gradient saliency (EXP-L4) — gradient measures local sensitivity, permutation measures global reliance.

```python
for feat_dim in range(11):
    x_permuted = graph.x.clone()
    x_permuted[:, feat_dim] = x_permuted[torch.randperm(N), feat_dim]
    pred_permuted = model(x_permuted, ...)
    importance[feat_dim] = (pred_original - pred_permuted).abs().mean()
```

**Expected finding:**  
`external_call_count` (dim 10) and `uses_block_globals` (dim 2) will likely be the most important features overall. If `type_id` (dim 0) is most important → the model is primarily learning node-type statistics, not feature-informed patterns.

---

#### EXP-L9: Attention Rollout (Layer-Collapsed Attribution)
**Layer:** 3 (Learning)  
**Priority:** P2  
**Effort:** 4 hours

**What it does:**  
Attention rollout propagates attention weights backward through all 8 GATConv layers to produce a single score per node representing "how much did this original node contribute to the final contract embedding."

The result is a ranked list of nodes by contribution to the vulnerability prediction.

**Key test:**  
For a known reentrancy contract:
- Are CFG_NODE_CALL and CFG_NODE_WRITE nodes in the top-10 contributors?
- Is the FUNCTION containing the vulnerable CFG in the top-3 contributors?

For a known safe contract:
- Are those same node types NOT in the top contributors?

This is the most direct evidence of whether the model learned the right abstraction.

**Script to write:** `ml/scripts/interpretability/exp_l9_attention_rollout.py`

---

#### EXP-L10: Edge Type Training Ablation (Small Dataset)
**Layer:** 3 (Learning)  
**Priority:** P2  
**Effort:** 1–2 days

**What it does:**  
Train 12 versions of the model on a small subset (3,000 contracts balanced across classes):
- 1 baseline (all edge types)
- 11 versions each with one edge type removed at training time

Compare F1 per class across all 12 runs. This is the ground-truth answer to "which edge types actually contribute to learning."

**Unlike EXP-L2 (inference ablation), this shows whether the edge type is needed during TRAINING, not just inference.** A model can learn patterns using an edge type during training and then not need it at inference (it encoded the pattern in weights). Training ablation catches this.

**Pass criteria:**  
Removing CONTROL_FLOW (type 6) causes ≥5% drop in Reentrancy F1. Removing EMITS (type 3) causes <1% drop across all classes (validates that EMITS is correctly low-impact for vulnerability detection).

**Note:** This is the most expensive experiment but gives ground-truth attribution. Run only after Layer 1 and Layer 2 experiments have identified which edge types are most suspect.

---

### SUPPLEMENTARY — Architecture-Specific Checks

---

#### EXP-A1: Phase 3 Pooling Node-Type Audit
**Layer:** 3 (Learning)  
**Priority:** P0  
**Effort:** 1 hour

**What it does:**  
The SentinelModel pools only over FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes after Phase 3. Verify that:
1. These nodes are actually present in every contract
2. Their Phase 3 embeddings carry non-trivial variance across vulnerability classes
3. The fallback (all-node pool) is not being triggered frequently

Run the model in eval mode and inspect:
```python
# In sentinel_model.py forward pass
fn_mask = torch.isin(node_types, FUNCTION_LIKE_NODE_TYPES)
print(f"fn_mask True rate: {fn_mask.float().mean():.3f}")  # should be >0.15
```

**Pass criteria:**  
FUNCTION-like nodes present in ≥95% of graphs. Fallback pool triggered in <5% of contracts.

---

#### EXP-A2: CFG Node Feature Inheritance Validation
**Layer:** 1 (Structure)  
**Priority:** P1  
**Effort:** 1 hour

**What it does:**  
Verify that BUG-C3 (CFG node inheriting dims [1,3,4,5,9] from parent FUNCTION) is working correctly in the extracted graphs. For a CFG_NODE_CALL inside a `payable` function, does the CFG node have `payable=1.0` (dim 4)?

Spot-check 20 CFG nodes across 5 contracts.

**Pass criteria:**  
100% of CFG nodes inside payable functions have dim[4]=1.0. Any failure means the inheritance is broken and CFG nodes carry incomplete features.

---

#### EXP-A3: JK Entropy Monitoring During Training
**Layer:** 3 (Learning)  
**Priority:** P1  
**Effort:** 1 hour (add to trainer logging)

**What it does:**  
The `jk_entropy` loss term is supposed to prevent JK attention from collapsing to a single phase. Add explicit logging of `last_weights` and `jk_entropy` per training step to TensorBoard/wandb.

If `jk_entropy` decays toward 0 during training → the regularizer is too weak and one phase is dominating. This should trigger either increasing the entropy regularizer weight or investigating why one phase is dominating.

**Pass criteria:**  
Mean `jk_entropy` at convergence ≥ 0.5 (out of max `log(3)=1.099`). Below 0.3 means JK has effectively collapsed and you are running a 1-phase model.

---

#### EXP-A4: Cross-Attention Fusion Contribution Analysis
**Layer:** 3 (Learning)  
**Priority:** P1  
**Effort:** 2 hours

**What it does:**  
The SentinelModel has three independent predictions: `aux_gnn`, `aux_transformer`, `aux_fused`. Log these per-class at inference time to understand which eye is making correct predictions.

For each class, compute:
- GNN-only F1 (from aux_gnn head)
- Transformer-only F1 (from aux_transformer head)  
- Fused F1 (from aux_fused head)
- Joint F1 (from main classifier)

This directly answers: "Is the GNN eye carrying useful signal at all, or is the transformer doing all the work?"

**Pass criteria:**  
GNN-only F1 is ≥5 percentage points above random baseline (0.5 × positive_rate) for at least 5 of 10 classes. If GNN-only F1 is at baseline for all classes → the GNN encoder is contributing nothing to the final prediction.

---

## 3. Execution Sequence

### Phase 1 — Structural Validation (Do First, No Model Needed)
| ID | Experiment | Time |
|---|---|---|
| EXP-S1 | CEI Structural Trace Audit | 3h |
| EXP-S2 | Edge Presence Rate Per Class | 1h |
| EXP-A2 | CFG Feature Inheritance Check | 1h |
| EXP-S4 | ICFG-Lite Path Closure Audit | 3h |
| EXP-S3 | Feature Distribution Per Class | 2h |

**Gate:** If EXP-S1 or EXP-S4 fail, the graph encoding is broken and must be fixed before any model training.

### Phase 2 — Expressivity Validation (Graph Topology, No Training)
| ID | Experiment | Time |
|---|---|---|
| EXP-E1 | K-Hop Receptive Field | 3h |
| EXP-E2 | WL Distinguishability Test | 4h |
| EXP-E4 | Direction Sensitivity Test | 2h |

**Gate:** If EXP-E2 shows >40% WL-equivalent pairs, the architecture needs path encoding or positional encoding of CFG nodes before training more.

### Phase 3 — Learning Validation (Requires Trained Weights)
| ID | Experiment | Time | Priority |
|---|---|---|---|
| EXP-A4 | Aux Head Contribution Analysis | 2h | P0 |
| EXP-L1 | JK Weight Distribution Per Class | 1h | P0 |
| EXP-L2 | Inference-Time Edge Ablation | 2h | P0 |
| EXP-A1 | Phase 3 Pooling Node-Type Audit | 1h | P0 |
| EXP-L4 | Gradient Saliency | 2h | P1 |
| EXP-L5 | Probing Classifiers Per Phase | 4h | P1 |
| EXP-L3 | GAT Attention Visualization | 3h | P1 |
| EXP-L6 | Counterfactual Contracts | 8h | P1 |
| EXP-L7 | Calibration and Size Analysis | 2h | P2 |
| EXP-L8 | Feature Permutation Importance | 2h | P2 |
| EXP-L9 | Attention Rollout | 4h | P2 |
| EXP-L10 | Edge Type Training Ablation | 2 days | P2 |

### Add During Next Training Run
| ID | Experiment | Time |
|---|---|---|
| EXP-A3 | JK Entropy Monitoring in Trainer | 1h |

---

## 4. Decision Tree — What to Do Based on Results

```
EXP-S1/S4 FAIL (graph doesn't encode pattern)
  └─→ Fix graph extractor first. No training work until structure is right.

EXP-S1/S4 PASS, EXP-E2 FAIL (WL-equivalent pairs >40%)
  └─→ Graph encodes it but GNN can't distinguish it.
      Options:
        A. Add topological position encoding to CFG nodes (fast)
        B. Add explicit path features as node features (e.g., "distance to nearest CALL")
        C. Switch Phase 2 to path-based aggregation (PathNN or GraphGPS)

EXP-S1/S4/E2 PASS, EXP-L2 FAIL (ablating CFG edges has no effect)
  └─→ Graph is correct, GNN is expressible, but model didn't learn it.
      Options:
        A. Increase CFG phase capacity (more layers or wider hidden_dim)
        B. Add auxiliary CEI supervision loss (explicitly penalize wrong CEI ordering)
        C. Add curriculum — first train on easy classes, then reentrancy

EXP-L2 PASS but EXP-L5 FAIL (probing classifiers don't separate)
  └─→ Edge types are being used, but the signal is not linearly separable at phase boundaries.
      This is a Phase 3 aggregation / pooling problem.

EXP-L4 shows features dominate (gradient mostly on type_id, external_call_count)
  └─→ Node features are so predictive that the model doesn't need graph structure.
      Options:
        A. Add stronger graph-structure loss (enforce that edge ablation hurts)
        B. Reduce the discriminative power of node-level features
        C. Accept this as valid — node features are a legitimate signal

EXP-A4 shows GNN aux head at baseline
  └─→ GNN is contributing nothing. Transformer is doing all the work.
      This is the most severe failure. Likely cause: Phase 2/3 signal is being washed
      out by Phase 1's 8-head output before JK can separate them.
```

---

## 5. Script Organization

All scripts go in: `ml/scripts/interpretability/`

```
ml/scripts/interpretability/
├── exp_s1_structural_trace.py
├── exp_s2_edge_enrichment.py          (extend edge_activation.py)
├── exp_s3_feature_distribution.py
├── exp_s4_icfg_path_audit.py
├── exp_e1_receptive_field.py
├── exp_e2_wl_distinguishability.py
├── exp_e3_message_propagation_sim.py
├── exp_e4_direction_sensitivity.py
├── exp_l1_jk_weight_analysis.py
├── exp_l2_edge_ablation.py
├── exp_l3_attention_visualization.py
├── exp_l4_gradient_saliency.py
├── exp_l5_probing_classifiers.py
├── exp_l6_counterfactual_contracts.py
├── exp_l7_calibration_size_analysis.py
├── exp_l8_permutation_importance.py
├── exp_l9_attention_rollout.py
├── exp_l10_training_ablation.py
├── exp_a1_pooling_audit.py
├── exp_a3_jk_entropy_logging.py
└── test_contracts/                    (minimal synthetic contracts for EXP-L6)
    ├── reentrancy_vulnerable.sol
    ├── reentrancy_safe.sol
    ├── integer_uo_vulnerable.sol
    ├── integer_uo_safe.sol
    ├── timestamp_vulnerable.sol
    ├── timestamp_safe.sol
    ├── unused_return_vulnerable.sol
    ├── unused_return_safe.sol
    ├── tod_vulnerable.sol
    ├── tod_safe.sol
    ├── inheritance_propagation.sol
    └── README.md
```

---

## 6. Shared Infrastructure Needed

All scripts share these utilities — write them once in `ml/scripts/interpretability/utils.py`:

```python
# Load model checkpoint
def load_model(checkpoint_path, device) -> SentinelModel

# Load validation data split
def load_val_split(cache_path, label_csv, splits_dir) -> DataLoader

# Get node type tensor from graph (node_metadata-based)
def get_node_types(graph) -> Tensor  # [N] int

# Collect predictions with auxiliary heads
def collect_predictions(model, dataloader, return_aux=True) -> dict

# Plot per-class heatmap
def plot_class_heatmap(matrix, row_labels, col_labels, title, output_path)
```

---

## 7. Expected Summary Output

After all experiments are complete, produce a single summary table:

```
| Experiment | Checks | Result | Verdict |
|---|---|---|---|
| EXP-S1 | CEI path in reentrancy graphs | X/10 contracts | PASS/FAIL |
| EXP-S2 | CONTROL_FLOW enrichment for reentrancy | X.Xx ratio | PASS/FAIL |
| EXP-S4 | ICFG-Lite loop closure | X/10 contracts | PASS/FAIL |
| EXP-E2 | WL distinguishability reentrancy pairs | X% equivalent | PASS/FAIL |
| EXP-L1 | Phase 2 JK weight for reentrancy | X.XX mean | PASS/FAIL |
| EXP-L2 | Prediction drop on CFG ablation | X.XX delta | PASS/FAIL |
| EXP-A4 | GNN aux F1 above baseline | X classes passing | PASS/FAIL |
| EXP-L5 | Phase 2 probe > Phase 1 probe on reentrancy | +X pp | PASS/FAIL |
| EXP-L6 | Counterfactual CEI pair | correct/wrong | PASS/FAIL |
```

This table is the answer to the original question: "Did what we designed actually work?"

---

## 8. What Each Experiment Validates (Architecture → Experiment Map)

| Architecture Decision | What We Hoped | Validated By |
|---|---|---|
| CFG nodes added | Model sees intra-function execution | EXP-S1, EXP-E1, EXP-L2 (type 6) |
| CONTROL_FLOW edges directed | Model learns execution order | EXP-E4, EXP-L2 (type 6), EXP-L6 Pair A |
| ICFG-Lite (CALL_ENTRY/RETURN_TO) | Model sees cross-function CEI | EXP-S4, EXP-L2 (types 8,9), EXP-L6 Pair A |
| DEF_USE edges | Model traces value flow | EXP-L2 (type 10), EXP-L6 Pair B |
| INHERITS edges | Model propagates parent vulnerabilities | EXP-L6 Pair D, EXP-L2 (type 4) |
| Parent feature inheritance (BUG-C3) | CFG nodes carry function context | EXP-A2 |
| Phase 2 (single-head, no self-loops) | Directional signal preserved | EXP-E4, EXP-L1 Phase 2 weight |
| Phase 3 REVERSE_CONTAINS | CFG signal rises to FUNCTION nodes | EXP-L3 (Phase 3 attention), EXP-L4 |
| JK aggregation | All phases contribute | EXP-L1 (entropy), EXP-A3 |
| Function-level pooling (not all nodes) | Function context dominates contract embedding | EXP-A1, EXP-L4 |
| Auxiliary heads (3 eyes) | No eye collapse during training | EXP-A4 |
| JK entropy regularizer | No phase collapse | EXP-A3 |

---

*Document written: 2026-05-30*  
*Author: Analysis session — GNN interpretability discussion*  
*Next step: Begin Phase 1 experiments (EXP-S1 through EXP-S4 + EXP-A2)*
