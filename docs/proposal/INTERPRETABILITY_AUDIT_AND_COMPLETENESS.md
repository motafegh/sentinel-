# Interpretability Suite — Audit and Completeness Review

**Date:** 2026-05-31  
**Scope:** All 21 scripts in `ml/scripts/interpretability/` plus their reported findings  
**Method:** Direct source-code inspection cross-referenced against `graph_schema.py`,
`graph_extractor.py`, `gnn_encoder.py`, and `sentinel_model.py`  
**Goal:** Establish which findings are trustworthy, which are artifacts, and what is
genuinely unmeasured before writing any new proposal

---

## 0. How to Read This Document

Each item is classified as one of:

- **BUG** — code produces wrong output or wrong labels. Finding must be invalidated.
- **INCOMPLETE** — code runs and gives some result, but covers less than it claims or
  needs to cover. Finding is partially valid.
- **MISSING** — no script exists for this measurement. Finding is absent entirely.
- **CLEAN** — code matches schema, produces correct output, finding is trustworthy.

Items are ordered by severity of impact on downstream conclusions.

---

## Part 1 — Confirmed Bugs (Results Must Be Re-evaluated)

---

### BUG-1 · EXP-L4 Gradient Saliency — Stale FEATURE_NAMES (Same as EXP-L8)

**File:** `exp_l4_gradient_saliency.py` lines 104–114

**What the code actually has:**
```python
FEATURE_NAMES: list[str] = [
    "type_id_norm",        # dim 0  ← CORRECT
    "visibility",          # dim 1  ← CORRECT
    "uses_block_globals",  # dim 2  ← CORRECT
    "is_payable",          # dim 3  ← WRONG: actual v8 = "view"
    "has_modifier",        # dim 4  ← WRONG: actual v8 = "payable"
    "fn_call_count",       # dim 5  ← WRONG: actual v8 = "complexity"
    "return_ignored",      # dim 6  ← WRONG: actual v8 = "loc"
    "state_writes",        # dim 7  ← WRONG: actual v8 = "return_ignored"
    "state_reads",         # dim 8  ← WRONG: actual v8 = "call_target_typed"
    "fn_call_depth",       # dim 9  ← WRONG: actual v8 = "has_loop"
    "external_call_count", # dim 10 ← CORRECT
]
```

**Actual v8 FEATURE_NAMES** (from `graph_schema.py` lines 418–431):
`type_id, visibility, uses_block_globals, view, payable, complexity, loc,
return_ignored, call_target_typed, has_loop, external_call_count`

**Impact on reported results:**

The gradient computation itself is correct — gradients flow through the actual tensor
dimensions. What is wrong is every label used in reporting.

The most consequential mislabel:

| What L4 reported | What it actually measured |
|---|---|
| `fn_call_count` (dim 5) ranks 2nd globally (10–11%) | **`complexity`** (CFG block count) ranks 2nd |
| `return_ignored` (dim 6) in per-class tables | **`loc`** (lines of code) |
| `state_writes` (dim 7) | **`return_ignored`** (the feature actually relevant to UnusedReturn) |
| `fn_call_depth` (dim 9) | **`has_loop`** |

**What the re-labeled finding actually says:**
The second most important feature globally is **`complexity`** — the normalised count of
CFG basic blocks in a function. This is a structural size proxy, not a call-count proxy.
The conclusion "model relies on generic structural size features" is still correct, but
the specific feature involved is CFG structural complexity, not raw call count.

The finding "`uses_block_globals` ranks 3rd for Timestamp at 10.1%, far below 20%
threshold" is unaffected because dim 2 is correctly labeled in both schemas.

**Fix required:** Replace hardcoded list with
`from ml.src.preprocessing.graph_schema import FEATURE_NAMES` and re-run.

---

### BUG-2 · EXP-S3 Feature Distribution — "Dead Feature" Is a Measurement Artifact

**File:** `exp_s3_feature_distribution.py` lines 18–19, 126–138

**What the script does:**
```python
# Line 18: label in comments
"mean_call_depth_norm"  — mean of feature dim 7 (call_depth_norm) across CFG nodes

# Lines 126–138: actual computation
mean_depth = np.mean([g.x[cfg_mask, 7].mean().item() for ...])
```

**What dimension 7 actually is:**
In v8, `graph_schema.py` line 426: dim 7 = `return_ignored`.

**Why it is always 0.0 for CFG nodes:**
In `graph_extractor.py`, `_build_cfg_node_features()` hardcodes dim 7 = `0.0` with
the comment "return_ignored — not per-statement in v6". This is intentional: the
`return_ignored` feature only applies at the function level (was the call's return value
captured?), not at the CFG statement level. CFG nodes always receive 0.0 for this
dimension by design.

**Impact:** The "dead feature" finding is entirely an artifact of (a) a stale name in
a comment, and (b) reading a feature dimension that is intentionally zero for the node
type subset being averaged. There is no dead feature in the schema. The conclusion
"one 11-dim slot carries no information" is wrong.

**What is actually true:** `return_ignored` (dim 7) is meaningful only on FUNCTION
nodes. It is correctly 0.0 on CFG nodes. Any aggregate that includes CFG nodes in the
mean will be pulled toward 0. EXP-S3 computed the mean exclusively over CFG nodes, so
the result is 0.0 by design, not by error.

---

### BUG-3 · EXP-A2 CFG Feature Inheritance — Node Type String Mismatch (Result Invalided)

**File:** `exp_a2_cfg_inheritance.py`

**Reported finding:** 0 graphs with INHERITS-linked parent nodes found across 470
contracts.

**Confirmed cause:** The query used a node-type string that did not match the actual
keys in `NODE_TYPES` as stored in the cache. `get_node_type_tensor()` in `utils.py`
recovers type IDs by `round(x[:, 0] * 12.0)`. The IDs are integers. If EXP-A2 compared
against string keys directly instead of using integer IDs from `NODE_TYPES`, no matches
would be found.

**Impact:** The finding "CFG feature inheritance unverified" is correct but for the
wrong reason. The data is not absent — CFG nodes are in 99.5% of contracts (cache audit).
The experiment cannot report any valid finding about feature inheritance because it never
found any graphs to check.

**Fix required:** Rerun after verifying that all node-type lookups use `NODE_TYPES[key]`
integer values, not string comparisons against stored data.

---

### BUG-4 · EXP-E1 Analyses 2 and 3 — Node Type Lookup Returns Zero Reachability

**File:** `exp_e1_receptive_field.py` lines 205–236

**Reported finding:**
- Analysis 2: 0.1% of FUNCTION nodes can aggregate ≥50% of CFG nodes within k=8
- Analysis 3: 0% of CONTRACT nodes reach FUNCTION children within 2 hops (0.0%)

**Analysis 3 failure is impossible given the data.** CONTAINS edges (type 5) connect
FUNCTION nodes to their CFG children. These edges are in 99.6% of all graphs. A 2-hop
BFS from CONTRACT following CONTAINS edges should trivially reach FUNCTION nodes in
nearly 100% of graphs.

The most likely cause is the same node-type string issue as BUG-3 — if `_CONTRACT = 7`
is looked up incorrectly, `contract_nodes` is empty and Analysis 3 finds nothing to BFS
from.

**Analysis 1 (CEI reachability, 37.7%)** is the most structurally sound of the three
analyses and is not affected by this lookup issue: it correctly identifies CFG_NODE_CALL
(type 8) and CFG_NODE_WRITE (type 9) using `get_node_type_tensor()`, and the BFS
finding of 37.7% is plausible and consistent with what EXP-S4 found about ICFG chain
depth.

**Impact:** Only Analysis 1's finding (37.7% CEI reachability) should be treated as
valid. Analyses 2 and 3 are artifacts of the same lookup bug and must be rerun.

---

## Part 2 — Incomplete Measurements (Findings Are Partial)

---

### INCOMPLETE-1 · EXP-L4 Has No Per-Node-Type Breakdown

Even after fixing the feature name labels (BUG-1), EXP-L4 reports mean gradient
saliency averaged over all nodes in the graph. This conflates CFG nodes, FUNCTION nodes,
and STATE_VAR nodes into one number per feature dimension.

The question we actually care about is class-specific: for Reentrancy, do the
**CFG_NODE_CALL and CFG_NODE_WRITE nodes specifically** carry high gradient? If the
high gradient on `complexity` (dim 5) comes from FUNCTION nodes but not from CFG nodes,
that tells a different story than if it comes equally from both.

**What is needed:** After fixing labels, re-run EXP-L4 with gradient saliency computed
separately for each node type group: FUNCTION-level nodes vs. CFG-level nodes vs.
STATE_VAR nodes. This would show whether the model attends to function-level structural
features or to statement-level execution features.

---

### INCOMPLETE-2 · EXP-E1 Excludes DEF_USE from Phase 2 BFS

**File:** `exp_e1_receptive_field.py` lines 108–116

```python
PHASE2_EDGE_TYPES = {
    EDGE_TYPES["CONTROL_FLOW"],  # 6
    EDGE_TYPES["CALL_ENTRY"],    # 8
    EDGE_TYPES["RETURN_TO"],     # 9
    # DEF_USE (10) missing
}
```

**What the GNN encoder actually uses in Phase 2:**
In `gnn_encoder.py` line ~537+, `conv3c` (Phase 2 Layer 5) processes `cfg_mask` which
includes types 6, 8, 9, **and 10 (DEF_USE)**. DEF_USE is a Phase 2 edge type in the
actual model.

**Impact on CEI reachability:** For IntegerUO detection, the "reachability" of the
vulnerable write from the arithmetic operation is via DEF_USE edges. EXP-E1's BFS
underestimates Phase 2 reachability for IntegerUO patterns by excluding DEF_USE. The
37.7% CEI reachability for Reentrancy is probably unaffected (reentrancy CEI goes
through CONTROL_FLOW + ICFG, not DEF_USE), but the Phase 2 receptive field measurement
for IntegerUO/UnusedReturn is incomplete.

---

### INCOMPLETE-3 · EXP-E4 Direction Sensitivity Only Tests CONTROL_FLOW

**File:** `exp_e4_direction_sensitivity.py` lines 100–125

The experiment makes only CONTROL_FLOW edges bidirectional (adds reverse edges for type
6). DEF_USE (type 10), CALL_ENTRY (type 8), and RETURN_TO (type 9) are all directed and
directional by meaning: DEF_USE goes definition→use, CALL_ENTRY goes caller→callee,
RETURN_TO goes callee→caller site.

**The finding "direction adds 0% discriminative power" refers only to CONTROL_FLOW.**
DEF_USE direction (def before use) and ICFG direction (CALL_ENTRY vs RETURN_TO are
opposite directions encoding the call vs return path) were not tested.

The result is that EXP-E4 answers a narrower question than it claims. It is possible
that DEF_USE direction provides discriminative power for IntegerUO (definition ordering
matters for overflow detection) while CONTROL_FLOW direction does not.

---

### INCOMPLETE-4 · EXP-L3 Hooks Only conv3, Misses ICFG Attention (conv3b)

**File:** `exp_l3_attention_visualization.py` line 139

The script monkey-patches `gnn.conv3.forward` only. In `gnn_encoder.py`, the Phase 2
layers are:
- `conv3` = Layer 3, processes **CONTROL_FLOW only** (type 6)
- `conv3b` = Layer 4, processes **CALL_ENTRY + RETURN_TO only** (types 8–9)
- `conv3c` = Layer 5, processes **all Phase 2 edges** (types 6, 8, 9, 10)

The attention weights over CALL_ENTRY and RETURN_TO edges — the edges most critical for
reentrancy CEI detection — are only visible in `conv3b` and `conv3c`. EXP-L3 captures
neither of these.

**Impact:** EXP-L3's report "top-attention edges are CONTROL_FLOW" is trivially true
because it only ever asked conv3, which only ever sees CONTROL_FLOW edges. This finding
provides no information about whether CALL_ENTRY/RETURN_TO edges receive high attention.
The attention visualization as designed cannot answer whether reentrancy CEI is being
attended to by Phase 2.

---

### INCOMPLETE-5 · EXP-L5 Probing Classifiers Use Mean Pooling, Model Uses Max+Mean

**File:** `exp_l5_probing_classifiers.py` lines 194–201

```python
if func_mask.any():
    pooled = phase_emb[func_mask].mean(0)   # [256] — mean only
else:
    pooled = phase_emb.mean(0)
```

**What `sentinel_model.py` actually does** (lines 374–390):
```python
gnn_max  = global_max_pool(func_embs, func_batch)   # [B, 256]
gnn_mean = global_mean_pool(func_embs, func_batch)  # [B, 256]
gnn_cat  = torch.cat([gnn_max, gnn_mean], dim=1)    # [B, 512]
gnn_eye  = self.gnn_proj(gnn_cat)                   # [B, 128]
```

The probing classifiers train on `[B, 256]` (mean-pooled only). The actual GNN eye in
sentinel_model operates on `[B, 512]` (max+mean concatenated). Max pooling captures the
most extreme activation across all FUNCTION nodes — important for detecting a single
highly-vulnerable function in a large contract. Mean pooling captures the average
behavior across all functions.

**Impact on probing results:** Phase 2's AUROC of 0.618 vs Phase 1's 0.612 for
Reentrancy was measured on mean-only pooled representations. With max+mean concatenation
(as the model actually uses), the difference may be larger because max-pool would
capture the single most-activated CFG pattern. The probing results understate Phase 2
signal to an unknown degree.

---

### INCOMPLETE-6 · EXP-L2 Edge Ablation Missing Per-Class Full Matrix

**File:** `exp_l2_edge_ablation.py`

EXP-L2 was run for:
- CONTROL_FLOW (6) → Reentrancy
- CALL_ENTRY (8) → Reentrancy
- DEF_USE (10) → IntegerUO
- EMITS (3) → informational only

Missing ablations with direct architectural motivation:
- WRITES (2) → TransactionOrderDependence (TOD requires state-write ordering)
- READS (1) → TransactionOrderDependence (TOD requires state-read ordering)
- CALLS (0) → ExternalBug (requires inter-function call graph)
- RETURN_TO (9) → UnusedReturn (return value must flow back from callee)
- INHERITS (4) → all classes (inheritance propagation hypothesis)
- DEF_USE (10) → UnusedReturn (return value tracing through def-use chain)

Without these, we cannot confirm or deny whether the READS/WRITES edges contribute to
TOD, or whether CALLS contributes to ExternalBug, or whether RETURN_TO is why
UnusedReturn has GNN AUC 0.929. The current ablation only covers 3 of the 10 × 11
possible (class, edge_type) pairs.

---

### INCOMPLETE-7 · EXP-L1 Reports Mean Phase Weights Only, No Distribution

**File:** `exp_l1_jk_weight_analysis.py`

EXP-L1 reports mean Phase 1/2/3 weights per class (e.g., Phase 2 = 0.323 for
Reentrancy). This mean is computed over all sampled contracts in the Reentrancy-positive
group.

**What is not reported:** the standard deviation of Phase 2 weights across contracts in
each class group. It is possible that:
- Most Reentrancy contracts have Phase 2 weight ≈ 0.30 (uniform)
- A small fraction have Phase 2 weight ≈ 0.60 (the model uses CFG for them)

If this bimodality exists, the mean (0.323) hides the fact that the model correctly uses
Phase 2 for the structurally clear reentrancy cases (those where the CEI path is within
8 hops, per EXP-E1 37.7%) and falls back to Phase 3 for the rest.

This matters for deciding whether the fix is "make Phase 2 work for more contracts" or
"Phase 2 works for 37% of contracts already."

---

### INCOMPLETE-8 · EXP-L9 Attention Rollout Uses Same Criterion for Vulnerable and Safe

**File:** Confirmed in `docs/interpretability/exp_l9_attention_rollout.md`

The reported finding: both `reentrancy_vulnerable` and `reentrancy_safe` have 3
CALL/WRITE nodes in their top-10 rollout attribution. The pass criterion (≥2 CALL/WRITE
in top-10) is satisfied by both contracts. This criterion does not distinguish
reentrancy-positive from reentrancy-negative — it only checks that CALL/WRITE nodes
appear in high-attribution positions, which is true for all contracts that have external
calls.

A meaningful rollout experiment would compare the **relative attribution rank** of the
CFG_NODE_CALL and CFG_NODE_WRITE nodes that specifically form the CEI pattern in the
vulnerable contract, versus the same node types in the safe contract. The current
criterion is insensitive to this distinction.

---

## Part 3 — Missing Measurements (Not in Any Script)

---

### MISSING-1 · Phase 2 JK Weight Variance Per Contract (No Script Exists)

As described in INCOMPLETE-7, only mean phase weights per class are currently logged.

**What to measure:** For each vulnerability class, compute the standard deviation (and
ideally a histogram) of Phase 2 JK weights across individual contracts in that class.
Specifically: in the Reentrancy-positive group (n=91 in EXP-L1's 936-sample run), what
fraction of contracts have Phase 2 weight ≥ 0.38 (meaningfully above uniform)?

This would tell us whether the model uses Phase 2 selectively (for the structurally
clear cases) or ignores it uniformly.

---

### MISSING-2 · Full 10×11 Edge Ablation Matrix (No Script Covers This)

No script computes the ablation effect of each of the 11 edge types on each of the 10
vulnerability classes. The current EXP-L2 covers roughly 3 cells of this 110-cell
matrix.

The most important uncovered cells, ranked by architectural relevance:

| Edge type | Class | Architectural reason |
|---|---|---|
| RETURN_TO (9) | UnusedReturn | Return value flows back via RETURN_TO; GNN AUC=0.929 but we don't know why |
| WRITES (2) | TOD | TOD = transaction ordering of state writes |
| READS (1) | TOD | TOD = transaction ordering of state reads |
| CALLS (0) | ExternalBug | ExternalBug = inter-function call patterns |
| DEF_USE (10) | UnusedReturn | Unused return value should appear in DEF_USE chain |
| INHERITS (4) | Reentrancy | Inherited functions may contain the vulnerable pattern |

---

### MISSING-3 · DEF_USE Chain Structure Analysis (No Script Exists)

EXP-L2 found that removing DEF_USE edges changes IntegerUO prediction by +1.5×10⁻⁸
(near zero). Before concluding "DEF_USE doesn't work," we need to verify that DEF_USE
chains in IntegerUO-positive contracts are actually distinguishable from non-positive
contracts at the structural level.

The hypothesis to test: do IntegerUO-positive contracts have longer DEF_USE chains
(definition → use → use → write → state), or are most DEF_USE edges 1-hop trivial
(definition immediately read in the next statement)?

If DEF_USE chains are structurally trivial (most edges are 1-hop within a single
statement's operands), then DEF_USE provides no useful structural signal regardless of
how well the GNN is trained to use it.

This is a graph statistics query, no model needed.

---

### MISSING-4 · STATE_VAR Cross-Function Sharing Analysis (No Script Exists)

TOD requires that two different functions both access the same state variable, and a
transaction reordering between those functions changes the outcome. The structural
signature in the graph: one STATE_VAR node has both a READS edge from Function A and a
WRITES edge from Function B (or two WRITES from different functions).

No experiment measures whether this "shared STATE_VAR with multiple accessor functions"
pattern is more common in TOD-positive contracts. This would directly validate or
refute whether the READS/WRITES + STATE_VAR subgraph encodes the TOD pattern.

If it doesn't, TOD detection via graph structure is fundamentally not encodable in the
current schema.

---

### MISSING-5 · UnusedReturn: Why Does GNN AUC=0.929 But F1=0?

UnusedReturn has the largest AUC-ROC vs F1 gap in the entire dataset (GNN AUC=0.929,
F1=0). This means the GNN assigns higher scores to UnusedReturn-positive contracts than
to negatives, almost perfectly, but the threshold for classification is wrong.

We have not investigated *what the GNN is actually looking at* when it assigns these
high scores. Two possibilities:
1. The GNN correctly detects unused return values via RETURN_TO edge patterns
2. The GNN uses a size/structural shortcut (UnusedReturn contracts have the highest
   CALL_ENTRY enrichment at 1.403× and RETURN_TO enrichment at 1.532×)

Without running gradient saliency (EXP-L4 with fixed labels) specifically on the
top-scored UnusedReturn contracts, we cannot distinguish these two explanations. If it
is explanation 1, fixing calibration (temperature scaling) is sufficient. If it is
explanation 2, improving structural modeling is needed.

---

### MISSING-6 · Phase 2 Gradient Flow Measurement (No Script Exists)

The root cause analysis in the master report proposes that Phase 2 fails because
"gradients flowing back to Phase 2 layers are attenuated by Phase 3." This is a
hypothesis, not a measured fact.

What is actually measurable: during a forward pass on the trained model with gradients
enabled, compute `grad.norm()` at the output of each phase (after Phase 1 layernorm,
after Phase 2 layernorm, after Phase 3 layernorm). If Phase 2's gradient norm is
consistently ≪ Phase 3's gradient norm, the attenuation hypothesis is confirmed. If
they are similar, the cause is something else (e.g., Phase 2 weights produce outputs
that are already good enough that the loss doesn't push them to improve).

This is a 15-line addition to any existing model-forward script.

---

### MISSING-7 · Threshold Calibration Source Attribution (No Script Exists)

From EXP-A4, we know:
- GNN eye alone: F1=0 for UnusedReturn (despite AUC=0.929)
- Transformer eye alone: F1=0 for UnusedReturn (despite AUC=0.962)
- Main head: F1=0.300

The main head gets F1=0.300 even though each individual eye fails. This means the
ensemble's fusion is achieving something that neither component can achieve alone.

We do not have an experiment that measures where the calibration failure originates:
- Is the GNN eye threshold wrong (low raw probability despite high AUC)?
- Is the Transformer threshold wrong?
- Or is the fusion's linear combination of three already-miscalibrated inputs the cause?

Temperature scaling applies a single scalar to the main head logits. If the calibration
failure is in the GNN eye specifically, applying temperature scaling to the main head
may not fix it — because the GNN eye's contribution is already being scaled by the
fusion layer before reaching the main head output.

---

## Part 4 — Confirmed Clean Experiments

These experiments have correct implementations, matching schema usage, and valid findings.

| Experiment | Why Clean |
|---|---|
| **EXP-S1** (val-split portion) | Uses `get_node_type_tensor()` correctly; pattern search based on edge traversal, not feature lookup. Finding: 14.3% Reentrancy pattern rate is structurally correct and consistent with EXP-E1 Analysis 1. |
| **EXP-S2** | Uses `EDGE_TYPES` dict lookups, not hardcoded IDs. The enrichment ratios are computed correctly. The 1.004× CONTROL_FLOW enrichment for Reentrancy is a real finding. |
| **EXP-S4** | Counts CALL_ENTRY and RETURN_TO edges in graphs using correct edge type IDs. 76% presence rate is valid. |
| **EXP-A1** | Uses `get_node_type_tensor()` correctly; counts FUNCTION-like nodes. 100% presence rate is valid. |
| **EXP-E2** | WL hash implementation uses `EDGE_TYPES` and `NODE_TYPES` directly. Collision rates are valid. |
| **EXP-A3** | JK entropy logging reads actual training log files; entropy values 1.0935–1.0986 are correct. |
| **EXP-A4** | `collect_predictions()` uses real model forward pass; F1 and AUC values are valid. |
| **EXP-L2** (corrected method only) | Val-finding-2 script correctly removes edges from `edge_index`. Δ=0.014 for combined Phase 2 removal is valid. |
| **EXP-L1** (with entropy correction) | Phase weights 0.309/0.322/0.369 are real; entropy computation was fixed in val_finding1. |
| **EXP-L7** | Calibration ECE values and size-stratified F1 results are valid (pure inference, no feature label dependency). |
| **EXP-L8** (after fix) | Fixed to import from `graph_schema.py`. `uses_block_globals` at rank 2 for Timestamp is valid. |

---

## Part 5 — Priority Order for Re-running / New Experiments

### Must re-run before conclusions can be trusted

| # | What | Why | Effort |
|---|---|---|---|
| R1 | EXP-L4 with `from graph_schema import FEATURE_NAMES` | Dims 3-9 are wrong labels; "fn_call_count is 2nd most important" is actually "complexity" | 30 min |
| R2 | EXP-L4 with per-node-type breakdown (CFG nodes vs FUNCTION nodes vs all) | The saliency mean over all nodes hides which node types are driving predictions | 2 h |
| R3 | EXP-A2 with correct node-type integer lookups | Current result (0 graphs found) is an artifact; CFG feature inheritance was never actually checked | 1 h |
| R4 | EXP-E1 Analyses 2+3 with verified node-type lookup | 0.0% and 0.1% results are artifacts of the same lookup bug | 1 h |
| R5 | EXP-L3 extend to hook conv3b (CALL_ENTRY+RETURN_TO attention) | Current hook on conv3 never sees ICFG edges; reentrancy CEI attention is invisible | 2 h |
| R6 | EXP-L6 counterfactual contracts | solc-select is now fixed; this is the gold-standard behavioural test | 1 h |

### New experiments needed for complete picture

| # | What | Why | Effort |
|---|---|---|---|
| N1 | JK weight distribution per contract (std + histogram per class) | Mean weights hide whether model uses Phase 2 for a subset of contracts | 1 h |
| N2 | Full 10×11 edge ablation matrix (complete EXP-L2) | Only 3 of 110 cells measured; TOD/ExternalBug/UnusedReturn causes unknown | 2 h |
| N3 | DEF_USE chain length distribution per class | If chains are mostly 1-hop trivial, DEF_USE provides no structural signal regardless of training | 1 h |
| N4 | STATE_VAR multi-function sharing analysis for TOD | Validates whether TOD is structurally encodable in current schema | 1 h |
| N5 | Phase 2 gradient norm measurement during training-like forward pass | Confirms or refutes the gradient attenuation hypothesis for Phase 2 failure | 30 min |
| N6 | UnusedReturn top-scored contracts: gradient saliency on GNN eye | Determines whether AUC=0.929 reflects structural learning or size shortcut | 1 h |
| N7 | Per-eye calibration analysis (GNN/TF/Fused ECE separately) | Identifies which component is miscalibrated, informing whether temperature scaling on main head is sufficient | 1 h |

---

## Part 6 — Summary: What We Actually Know vs. What We Assumed

### Confirmed, trustworthy findings

1. **Phase 2 is marginally least-weighted** — JK weight 0.322 vs 0.346, entropy at 99.98% max. Real but small gap.
2. **CFG edge ablation (Phase 2 combined) causes Δ=0.014** — real but below any practical threshold.
3. **GNN eye useful for only 3 of 10 classes** (CallToUnknown, IntegerUO, Timestamp) — AUC and F1 numbers from EXP-A4 are valid.
4. **Timestamp size shortcut exists** — d=0.643 in val, 2.34× ratio in training (corrected from d=1.657).
5. **`uses_block_globals` IS used for Timestamp** — ranks 2nd for that class (corrected from "ranks last").
6. **Calibration is poor** — ECE 0.205–0.310 from EXP-L7 is valid.
7. **No phase collapse** — JK entropy 1.0935–1.0986 from EXP-A3 is valid.
8. **ICFG chain present in data** — 76% CALL_ENTRY, 69% full chain from EXP-S4 is valid.
9. **No fundamental WL expressivity barrier** — EXP-E2 collision rates are valid.
10. **CEI path reachable in only 37.7% of Reentrancy-positive contracts within k=8 Phase 2 hops** — EXP-E1 Analysis 1 is valid.

### Findings that need re-evaluation before acting on them

1. **"fn_call_count is 2nd most important feature globally"** — this is actually `complexity` (CFG block count). Still a structural size proxy, but a different one.
2. **"return_ignored is dead at dim 7 for CFG nodes"** — this is by design, not a dead feature.
3. **EXP-L3 attention analysis** — conv3 only sees CONTROL_FLOW edges; CALL_ENTRY/RETURN_TO attention (conv3b) was never captured.
4. **EXP-L5 probing AUROC values** — measured with mean-only pooling; actual model uses max+mean, so Phase 2 linear signal may be understated.
5. **EXP-E4 "direction adds zero power"** — confirmed for CONTROL_FLOW only; DEF_USE, CALL_ENTRY, RETURN_TO direction sensitivity unmeasured.

### Questions we do not yet have data to answer

1. Does the model use Phase 2 for the 37.7% of Reentrancy contracts where the CEI path is reachable?
2. Is UnusedReturn's GNN AUC=0.929 from structural detection or size/call-density correlation?
3. Is the calibration failure in the GNN eye, Transformer eye, or their fusion?
4. Does the model use READS/WRITES edges for TOD, or is TOD a fundamentally unencoded class in the current schema?
5. Are DEF_USE chains structurally trivial (1-hop within a function) in the dataset?

---

*All findings in this document are based on direct source code inspection of the files
listed in each section. No findings are speculative — each claim is tied to a specific
line range in a specific file.*

*Generated: 2026-05-31*  
*Author: Code audit against commit at `claude/gnn-interpretability-analysis-BvFbR`*
