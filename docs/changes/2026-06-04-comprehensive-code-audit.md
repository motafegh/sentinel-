# Comprehensive Source Code Audit — Preprocessing, GNN Architecture & Training Pipeline

**Date:** 2026-06-04
**Scope:** `ml/src/preprocessing/`, `ml/src/models/`, `ml/src/training/`, `ml/src/inference/`, `ml/src/data_extraction/`
**Trigger:** GCB-P1-Run7 analysis (ep39, F1=0.3074 fixed / 0.3329 tuned) revealed structural ceilings requiring source-code root cause investigation
**Mode:** Read-only audit, cross-referenced with training metrics and `docs/CHANGELOG.md` (2,079 lines)
**Revision 2:** Added CHANGELOG cross-references — interpretability suite findings (EXP-L2, A4, E1, L4, L8) validate and extend all preprocessing audit findings

---

## Table of Contents
1. Executive Summary
2. Methodology
3. Cross-Reference: Run 7 Findings × Source Code
4. Issue Inventory
5. Preprocessing Deep Dives
   5.1 Complexity Normalization
   5.2 MODIFIER CFG Gap
   5.3 CALLS Edges to Modifiers
   5.4 `return_ignored` Branch Sensitivity
   5.5 `call_target_typed` Regex Fallback
   5.6 `_select_contract` Heuristic
   5.7 ICFG Edge Failure Isolation
   5.8 CEI BFS Cross-Function Gap
6. GNN Architecture Deep Dives
   6.1 DEF_USE Routing in Phase 2
   6.2 JK Entropy Regularization
   6.3 GNN Prefix Injection
   6.4 CFG Eye (IMP-R7-2)
7. Training Pipeline Issues
   7.1 StructuredLogger `torch.compile` Bug
   7.2 `fusion_lr_multiplier` Calibration
   7.3 `fusion_max_nodes` Truncation
   7.4 DoS Loss Weight
   7.5 Post-Training Calibration
8. Priority Fix Plan
9. Schema-Change Blueprints for v9
   9.1 Multi-Value `uses_block_globals`
   9.2 Modifier CFG Extraction
   9.3 Cross-Contract Edges
   9.4 `fusion_max_nodes` Upgrade
10. Parked Topics

---

## 1. Executive Summary

The GCB-P1-Run7 training run reached F1=0.3074 (fixed threshold) / 0.3329 (tuned) at ep39 — effectively at parity with the best previous run (Run 4: 0.3362) despite using v10 data with schema improvements. Cross-referencing the training doc with source code reveals:

- **DEF_USE edges are in fact implemented and extracted** (Run 7 doc §6 is stale — "deferred RC5" was completed). However, they only get 1 GAT hop in the joint conv3c layer, limiting their utility.
- **The highest-ROI fix without schema changes** is the complexity normalization (`x[:,5] *= log1p(N)/N`) which crushes the complexity feature to ~1% of original value for a 500-node contract. This directly harms DoS, the class with the most remaining headroom.
- **The largest architectural gap** is MODIFIER nodes: they're extracted as featureless placeholders with no CFG subgraph and no CALLS edges from their parent functions. This hides `nonReentrant` guards, `onlyOwner` access control, and modifier-modulated vulnerability patterns.
- **~8% multi-contract files map to the wrong contract's AST**, injecting direct training noise that caps all-class performance.

---

## 2. Methodology

Files audited:

| Module | File | Lines | Coverage |
|--------|------|-------|----------|
| Preprocessing | `graph_schema.py` | 476 | Full — schema constants, version tracking, invariants |
| Preprocessing | `graph_extractor.py` | 1753 | Full — all `_compute_*` helpers, edge construction, tensor assembly |
| Preprocessing | `ast_extractor.py` | ~200 | Wrapper only — confirmed identical code path |
| Inference | `preprocess.py` | ~650 | Extraction path (identical), tokenizer diff verified |
| Models | `gnn_encoder.py` | 628 | Full — 3-phase routing, edge masks, JK attention |
| Models | `sentinel_model.py` | 646 | Prefix injection, 4-eye classifier, aux heads |
| Training | `trainer.py` (TrainConfig) | ~355 | Config defaults, optimizer groups, loss params |
| Training | `training_logger.py` | ~35 | StructuredLogger bug |

Cross-referenced against: `docs/training/GCB-P1-Run7-analysis-2026-06-04.md` (380 lines), `ml/scripts/`, `docs/CHANGELOG.md`.

---

## 3. Cross-Reference: Run 7 Findings × Source Code

| Run 7 Finding | § | Source Code Location | Status | Gap |
|---|---|---|---|---|
| UnusedReturn ceiling — "no DEF_USE edges" | §6 | `graph_extractor.py:887-997` | **Stale** — DEF_USE IS implemented | Only 1 hop in conv3c joint layer |
| Timestamp ceiling — "binary flag only" | §6 | `graph_extractor.py:455-488` | **Confirmed** | `uses_block_globals` is boolean, no propagation tracking |
| Reentrancy improving — ICFG helps | §6 | `graph_extractor.py:824-884` | **Confirmed** | Cross-function CALL_ENTRY/RETURN_TO working |
| DoS sawtooth — 65 val positives | §5 | `graph_extractor.py:1609-1621` | **Amplified by code bug** | Complexity normalization crushes signal |
| Threshold tuning gap growing | §7 | `trainer.py:241,253` | **Config issue** | `fusion_lr_multiplier=0.5` too high for 4-eye |
| JK Phase 3 drift | §8 | `gnn_encoder.py:298-327`, `trainer.py:346` | **Config issue** | λ=0.005 not sufficient |
| StructuredLogger silent | §10 | `training_logger.py:305` | **Bug** | `OptimizedModule` not subscriptable |
| GNN share 91%→28% healthy | §9 | `trainer.py:230-241` | **Confirmed** | LR multipliers working correctly |
| Ph2/Ph1 ratio 0.47-0.71 stable | §9 | `trainer.py:249` (comment) | **Confirmed** | ISSUE-1 fix working |
| Sawtooth = DoS noise | §5 | — | **Accepted** | Requires sample-level fix, not code fix |
| `gnn_prefix_k=0` disabled | §14 (R8-2) | `trainer.py:342` | **Available but unused** | Prefix injection code ready |

---

## 4. Issue Inventory

### HIGH Severity (affects F1 ceiling directly)

| ID | Issue | File:Line | Impact | Evidence |
|----|-------|-----------|--------|----------|
| A-1 | **Complexity normalization crushes feature** — `x[:,5] *= log1p(N)/N` reduces complexity to ~0.008 for 500-node contract | `graph_extractor.py:1617-1621` | DoS F1 suppressed; complexity-dependent pattern recognition (gas loops, unbounded ops) near-zero | `_size_factor = log1p(500)/500 = 0.0124`; original complexity `0.66` → `0.008` |
| A-2 | **MODIFIER CFG not extracted** — `_build_control_flow_edges()` only called for `contract.functions`, not `contract.modifiers` | `graph_extractor.py:~1450` (call site for functions only) | `nonReentrant` guard invisible → Reentrancy misses defensive signal; `onlyOwner` invisible → ExternalBug misses access control | Modifier nodes with type_id=2 exist but have no CONTAINS/CONTROL_FLOW children |
| A-3 | **CALLS edges to modifiers never created** — `func.internal_calls` doesn't include `func.modifiers` | `graph_extractor.py:1658` | Modifiers disconnected from their parent functions; GNN sees no edge from function to its `onlyOwner` or `nonReentrant` | Slither stores modifiers in `Function.modifiers`, not `Function.internal_calls` |
| A-4 | **`_select_contract` 8% wrong-AST** — `_derivation_score` heuristic misses contracts that inherit from out-of-file parents | `graph_extractor.py:1236-1246` | ~3,500/44K training samples have wrong AST-label alignment — direct noise across all 10 classes | ~92% accuracy on BCCC; 8% failure rate directly input to F1 ceiling |
| A-5 | **`return_ignored` scan is CFG-topological-order, not execution-path-order** | `graph_extractor.py:304-331` | False negatives on branch-heavy contracts: a return that is captured on one branch but ignored on another is classified as captured | `all_ops_ordered` is flat CFG post-order, doesn't branch-split |

### MEDIUM Severity

| ID | Issue | File:Line | Impact |
|----|-------|-----------|--------|
| B-1 | **DEF_USE only 1 hop in Phase 2** — not in per-layer sub-masks (only in joint conv3c) | `gnn_encoder.py:509-514` | DEF_USE signal gets 1 GAT hop vs CF's 3 hops; 4 heads in conv3c split capacity across CF+ICFG+DEF_USE |
| B-2 | **ICFG failure kills ALL edges for contract** — single try/except wraps entire `_add_icfg_edges` call | `graph_extractor.py:1539-1543` | One problematic function in a contract disables cross-function signal for the entire contract |
| B-3 | **`call_target_typed` regex misses multi-line calls, `addr.call{value}`, assembly** | `graph_extractor.py:383` | `address\s*\(\s*(?!this\b)[^)]+\)\s*\.call` doesn't match `addr.call{value:x}(data)` or nested casts |
| B-4 | **CEI BFS intra-function only** — `_compute_has_cei_path` only follows CONTROL_FLOW(6) edges | `graph_extractor.py:1014-1016` | Misses cross-function CEI: function A calls B which writes state — BFS stays in function A |
| B-5 | **`fusion_max_nodes=1024`** — 227 contracts exceed this, truncated in cross-attention | `trainer.py:200` | Upper tail of contract size distribution loses CFG nodes |
| B-6 | **StructuredLogger broken under `torch.compile`** | `training_logger.py:305` | No AUC, ECE, Brier scores for entire Run 7 |
| B-7 | **DoS loss weight 0.5** reduces gradient for class with most headroom | `trainer.py:326` | Conservative; may be suppressing DoS improvement below what complexity normalization fix would enable |

### LOW Severity

| ID | Issue | File:Line | Impact |
|----|-------|-----------|--------|
| C-1 | **Formula vs code mismatch** in complexity normalization comment | `graph_extractor.py:1614` | Doc rot; actual code correct |
| C-2 | **`_add_icfg_edges` docstring typo** — function name wrong in docstring | `graph_extractor.py:832` | Cosmetic |
| C-3 | **Version tracking drift** — `FEATURE_SCHEMA_VERSION="v8"` but cache paths reference `v10` | cross-file | Non-functional; cache key uses actual content MD5 |
| C-4 | **loc feature dim[6] nearly constant for CFG nodes** — single-statement CFG nodes all have `log1p(1)/log1p(1000)` ≈ 0.033 | `graph_extractor.py:1100-1104` | Negligible discriminative signal |

---

## 5. Preprocessing Deep Dives

### 5.1 Complexity Normalization (A-1, HIGH)

**Location:** `graph_extractor.py:1609-1621`

**Current code:**
```python
_CONTRACT_SIZE = float(x.shape[0])
if _CONTRACT_SIZE > 1.0:
    import math as _math
    _size_factor = _math.log1p(_CONTRACT_SIZE) / _CONTRACT_SIZE
    x[:, 5] = x[:, 5] * _size_factor
```

**Comment claims:** `complexity_norm[i] = complexity[i] * (1 + log(x.shape[0])) / (1 + x.shape[0])` — but actual formula is `log1p(N)/N`.

**Empirical effect:**

| Contract nodes | _size_factor | Complexity (20 CFG blocks) → post-norm |
|---------------|-------------|----------------------------------------|
| 10 | 0.240 | 0.66 → 0.158 |
| 50 | 0.079 | 0.66 → 0.052 |
| 100 | 0.047 | 0.66 → 0.031 |
| 500 | 0.012 | 0.66 → 0.008 |

For the ~41K training dataset (mean nodes=48.7, median ~21 for declaration-only), the complexity feature is crushed to 5-8% of its original value for most samples. Since complexity is the primary signal for loop-based DoS patterns (gas exhaustion via unbounded loops), this directly suppresses DoS F1.

**The comment's stated intent** ("Large contracts have many functions, each with moderate complexity, so the raw mean is size-correlated") is legitimate — complexity does correlate with contract size. But the current formula over-corrects. A milder alternative: `_size_factor = 1.0 / math.log1p(_CONTRACT_SIZE)`, which gives 0.217 at N=500 vs the current 0.012.

**History:** This was added as "E2 / Interp-3: Contract-size normalisation (Timestamp size shortcut fix)". The Timestamp connection is unclear — Timestamp uses feature[2] (`uses_block_globals`), not feature[5] (`complexity`). This may have been a proxy fix for a different observed correlation.

**Recommendation:** Either remove the normalization entirely or replace with `1/log1p(N)`.

---

### 5.2 MODIFIER CFG Gap (A-2, HIGH)

**Location:** `graph_extractor.py:~1450` (where `_build_control_flow_edges` is called)

**What happens currently:**
1. Modifier nodes ARE registered: `_add_node(mod, NODE_TYPES["MODIFIER"])` at line 1434
2. Modifier features are extracted: `_build_node_features(mod, type_id)` at line 1440
3. But `_build_control_flow_edges` is only called for `contract.functions` — NOT for `contract.modifiers`

**Consequences:**
- Modifiers are featureless beyond declaration-level attributes: `complexity=0`, `has_loop=0`, `external_call_count=0`, `return_ignored=0`
- Modifiers have NO CONTAINS edges to CFG children (no internal structure visible)
- Modifiers have NO CONTROL_FLOW edges between CFG children
- The GNN receives a type_id=2 node with a uniform feature vector

**What modifiers contain (Slither):**
- `modifier.nodes` — same structure as function CFG nodes
- `modifier.irs` — Slither IR operations (reads, writes, calls, conditions)
- Common patterns: `_status != _ENTERED` (reentrancy guard read), `_status = _ENTERED` (state write), `require(msg.sender == owner)` (access control)

**Impact on vulnerability classes:**
- **Reentrancy** — `nonReentrant` modifier contains the guard's state read/write. Without modifier CFG, the GNN cannot learn "this function has a reentrancy guard". Reentrancy detection relies entirely on external-call patterns in the function body, missing the guard signal.
- **ExternalBug** — `onlyOwner` modifier enforces access control. Without modifier CFG, the GNN cannot distinguish `onlyOwner`-protected functions from unprotected ones.
- **All classes** — modifier-modulated functions (e.g., `whenNotPaused`, `afterDeadline`) lose their structural context.

**Fix scope:**
- Add `_build_control_flow_edges(contract, contract.modifiers, ...)` call alongside the existing `contract.functions` call
- This requires `_func_entry_map`, `_func_terminal_map`, `_func_cfg_maps` to handle modifiers as well
- Increases node count per contract (modifier CFG nodes add to total N)
- No schema version bump needed (type_id=2 already exists, edge types are unchanged)

---

### 5.3 CALLS Edges to Modifiers (A-3, HIGH)

**Location:** `graph_extractor.py:1655-1660`

**Current code:**
```python
for func in contract.functions:
    fn = getattr(func, "canonical_name", None) or func.name
    for call in (getattr(func, "internal_calls", None) or []):
        if hasattr(call, "canonical_name"):
            _add_edge(fn, call.canonical_name, EDGE_TYPES["CALLS"])
```

`func.internal_calls` returns only `Function` objects. `func.modifiers` is a separate attribute returning `Modifier` objects.

**Slither API:**
- `Function.internal_calls` — list of internally-called `Function` objects
- `Function.modifiers` — list of `Modifier` objects applied to this function
- Modifiers are NOT included in `internal_calls`

**Fix:**
```python
for func in contract.functions:
    fn = getattr(func, "canonical_name", None) or func.name
    for call in (getattr(func, "internal_calls", None) or []):
        if hasattr(call, "canonical_name"):
            _add_edge(fn, call.canonical_name, EDGE_TYPES["CALLS"])
    for mod in (getattr(func, "modifiers", None) or []):  # NEW
        mod_key = getattr(mod, "canonical_name", None) or mod.name
        if mod_key and mod_key != fn:
            _add_edge(fn, mod_key, EDGE_TYPES["CALLS"])
```

---

### 5.4 `return_ignored` Branch Sensitivity (A-5, HIGH)

**Location:** `graph_extractor.py:304-331`

**Current algorithm:**
1. Build `all_ops_ordered`: flat list of `(node, op)` pairs in CFG topological order
2. For each call op, scan subsequent ops in flat order to check if `lval_name` is read
3. If never read, return 1.0 (ignored)

**Problem:** CFG topological order is NOT execution order across branches. Example:
```
tmp = addr.call{value: x}("");   // returns bool
if (condition) {
    require(tmp);                 // captures return — but only on this path
} else {
    // tmp never read here        // return IS ignored on this path
}
```

The flat scan finds `require(tmp)` after the call and returns 0.0 (captured), even though the else branch ignores the return. This produces false negatives for UnusedReturn and MishandledException on branch-heavy contracts.

**Fix scope:** This would require path-sensitive analysis — for each call site, determine if there exists ANY complete execution path through the function where `lval_name` is never read. This is equivalent to the "unused variable" data-flow problem and is NP-hard in the general case. A practical approximation:
- After scanning all ops in flat order, do a BFS from the call site through CONTROL_FLOW edges
- Check if `lval_name` appears in reads on ALL paths (conservative: only flag as ignored if no path reads it)
- This is a `_compute_has_cei_path`-style approach but for variable uses instead of CFG_NODE types

**Current IMP-D1 fix** (replacing global set with sequential scan) was correct for the specific bug (local variable names matching unrelated reads), but created this branch-sensitivity false negative. A better approach: per-path analysis using Slither's CFG node successors.

---

### 5.5 `call_target_typed` Regex Fallback (B-3, MEDIUM)

**Location:** `graph_extractor.py:~383`

**Regex:** `r"address\s*\(\s*(?!this\b)[^)]+\)\s*\.call"`

**Doesn't match:**
- `(address(addr)).call{value: x}("")` — outer parentheses
- `address(uint160(addr)).call(...)` — nested casts
- Assembly-style calls
- `receiver.call{value: amount}("")` — no `address(...)` wrapper at all
- Multi-line calls spanning source_line boundaries

**Fallback path (A6):** When type resolution fails, the regex is used to scan source. When it misses, `call_target_typed` defaults to 1.0 (typed = safe) — a false negative for CallToUnknown.

**Fix:** Instead of regex on source, use Slither's IR types directly. In Slither, `LowLevelCall` has a `.function_name` attribute. If the call is through an interface type, it's a `HighLevelCall`. The type system is already available via `op.call_type`. The regex fallback should be a last resort only.

---

### 5.6 `_select_contract` Heuristic (A-4, HIGH)

**Location:** `graph_extractor.py:1174-1246`

**Heuristic:** Picks the contract with the highest `_derivation_score` — number of in-file parent contracts it inherits from.

**~92% accuracy** on BCCC means ~3,500 training samples have the wrong contract's AST. Since labels are column-specific (one column per contract name), these samples have the GNN learning patterns from contract A's structure while the labels describe contract B's vulnerabilities. This is direct noise in the training signal for ALL 10 classes.

**Improvement ideas:**
- Add out-of-file inheritance to the derivation score (check `contract.inheritance` against all parsed contracts, not just in-file candidates)
- Use file name heuristics: if the file is named `ERC20Token.sol`, prefer the contract whose name matches the file stem
- Fall back to `most_funcs` only when `most_derived` fails (currently done, but `most_funcs` is 47.4% wrong)
- The true fix: multi-contract graph extraction (build N graphs per file for N candidate contracts). This multiplies extraction cost but eliminates the heuristic entirely.

**Note from schema history (v5 line 98-99):** "most_derived composite heuristic (~92%+ accurate)" was already a 2.3× improvement over the previous 52.6%. The remaining 8% gap is harder but the ROI is now higher.

---

### 5.7 ICFG Edge Failure Isolation (B-2, MEDIUM)

**Location:** `graph_extractor.py:1535-1543`

**Current code:**
```python
try:
    _add_icfg_edges(contract, ...)
except Exception as exc:
    logger.warning("ICFG edge extraction failed for '%s' ...", contract.name, exc)
```

If ANY function in the contract causes an exception during ICFG edge building, ALL ICFG edges for the ENTIRE contract are dropped. One problematic function (e.g., a library stub with missing `internal_calls`) disables cross-function signal for the whole contract.

**Fix:** Move the try/except inside `_add_icfg_edges` to per-function granularity, so a failing function only loses its own ICFG edges, not the entire contract's.

---

### 5.8 CEI BFS Cross-Function Gap (B-4, MEDIUM)

**Location:** `graph_extractor.py:1000-1026`

**Current BFS** only traverses CONTROL_FLOW(6) edges. From the code:
```python
# Only CONTROL_FLOW edges (type 6) are traversed — CALL_ENTRY/RETURN_TO/DEF_USE
# are ignored so we stay intra-function (not inter-procedural).
```

This is an explicit design choice, but it means the CEI label is false for cross-function patterns:
```
function A() {
    B();    // A has no WRITE — no CEI detected
}
function B() {
    state = 1;  // WRITE here — but A→B is ICFG, not CF
}
```

**Fix:** Extend the BFS to optionally follow CALL_ENTRY(8) edges (going from call site → callee) when computing `has_cei_path`. The RETURN_TO(9) direction already returns, so following CALL_ENTRY and then CF in the callee would catch cross-function CEI.

---

## 6. CHANGELOG Cross-References — Interpretability Suite Validates Audit Findings

The CHANGELOG's interpretability suite (2026-05-30, 21 experiments against Run 4 best checkpoint F1=0.3362) provides independent empirical validation of many audit findings. Key connections:

### 6.1 EXP-L2: CFG Ablation Near-Zero Effect → Validates A-1, A-2, A-5

**CHANGELOG line 1165:** `CFG ablation has near-zero effect (1.08×10⁻⁶)`

The interpretability suite confirmed that removing ALL Phase 2 edges (CONTROL_FLOW, CALL_ENTRY, RETURN_TO) changes Reentrancy predictions by ~1×10⁻⁶ — effectively zero. This means Phase 2 is dead weight.

**Connection to audit:**
- **A-1 (Complexity normalization)**: The complexity feature (dim 5) is crushed to ~1% of original value, so CFG node features carry no discriminative signal → Phase 2 has nothing useful to propagate
- **A-2 (MODIFIER CFG missing)**: Modifiers have no CFG children, so reentrancy-guard patterns are invisible → Phase 2's CF edges have no guard-related structure to learn from
- **A-5 (return_ignored branch-insensitive)**: Even if Phase 2 wanted to learn return-capture patterns, the feature is a flat heuristic with false negatives → Phase 2 cannot correct it

**Additional EXP-L2 finding:** Removing Phase 2 edges *increases* Reentrancy scores (CF: +0.020, CALL_ENTRY: +0.010, RETURN_TO: +0.018). The model has learned a **suppression shortcut**: dense CFG → "well-engineered contract" → "not Reentrancy". Phase 2 is actively harmful for some classes.

### 6.2 EXP-A4: GNN Eye Useful for Only 3/10 Classes → Validates GNN Underperformance

**CHANGELOG line 1166:** `GNN eye F1=0 for 7 classes including Reentrancy`

Only CallToUnknown, IntegerUO, and Timestamp benefit from the GNN eye. For the other 7 classes, the GNN eye contributes zero discriminative power — the Transformer eye and Fused eye carry all the signal.

**Classes where GNN is dead (per-class GNN F1 ≈ 0):**
- Reentrancy, GasException, MishandledException, UnusedReturn, TOD, ExternalBug, DenialOfService

Every one of these is a class whose detection requires understanding MODIFIER structure, cross-function data flow, or CFG branch paths — exactly the gaps identified in this audit (A-2, A-3, A-5, B-1).

### 6.3 EXP-E1: CEI Reachability Only 37.7% at k=8 Hops → Validates Phase 2 Depth Gap

**CHANGELOG line 1173:** `CEI reachability 37.7% at k=8 hops (Reentrancy-positive)`

Even when CEI paths exist in the graph data, the GNN's `_compute_has_cei_path` BFS with max_hops=8 only finds 37.7% of Reentrancy-positive paths. Since the GNN's Phase 2 only has 3 layers (conv3/3b/3c, each being 1 hop), the effective reachability is even lower — the model can aggregate at most 3 CFG hops.

**Connection to audit:**
- **B-1 (DEF_USE only 1 hop)**: Even when DEF_USE edges exist, they get only 1 hop → def-use chains longer than 1 hop are invisible
- **B-4 (CEI BFS intra-function)**: Cross-function CEI is not even attempted — the BFS stops at function boundaries by design

### 6.4 EXP-L4: `external_call_count` Dominates ALL Classes → Confounded Shortcut Confirmed

**CHANGELOG line 1337:** `external_call_count dominates gradient for ALL classes (21-24%)`

Every vulnerability class — including Timestamp, DoS, IntegerUO, where external calls are semantically irrelevant — learns primarily from `external_call_count` (feature[10]). This is a **dataset shortcut**: the BCCC dataset correlates external calls with vulnerability labels because the dataset was collected from exploit-prone contracts. The model learns "many external calls = vulnerable" rather than the true structural patterns.

**Connection to audit:**
- **A-1 (Complexity normalization)**: If complexity were a useful signal, it would compete with `external_call_count` for gradient. Instead, complexity is crushed → `external_call_count` dominates unchallenged
- **A-4 (8% wrong-contract)**: 8% of samples have the wrong contract's CFG, amplifying noise in all non-external-call features and further concentrating gradient on the one reliable signal (`external_call_count`)

### 6.5 EXP-L8: `type_id_norm` Dominates Permutation Importance 3× → Node Type Bias

**CHANGELOG line 1170:** `type_id_norm dominates permutation importance by 3× (0.079 vs next 0.026)`

The model's most important feature is "what type of node is this" (type_id/12.0), not any vulnerability-relevant signal. This means the GNN is learning to distinguish CONTRACT nodes from FUNCTION nodes from CFG_NODE_CALL nodes — but not learning WHY a particular CFG_NODE_CALL is part of a reentrancy pattern.

**Root cause:** The type embedding (BUG-R7-2, `nn.Embedding(13, 16)`) and per-phase edge routing mean the GNN sees different edge types per phase, so node type becomes the primary discriminative feature available in all phases.

### 6.6 C-1 Fix Already in v10 (Per-Statement CFG Features)

**CHANGELOG §33 (2026-06-02):** C-1 fix added four per-node helper functions (`_node_uses_block_globals`, `_node_return_ignored`, `_node_call_target_typed`, `_node_external_call_count`) in `graph_extractor.py`. These provide per-statement features for CFG nodes, fixing the previous behavior where CFG nodes inherited function-level defaults (0.0/1.0/0.0).

**Status:** This fix is already in v10 graph data and active in Run 7. The audit's A-5 finding (`return_ignored` branch-insensitive scan) is a separate issue — C-1 was about CFG nodes inheriting function-level feature values, not about the scan being branch-insensitive.

### 6.7 H-2 Fix Already in v10 (ReferenceVariable DEF_USE)

**CHANGELOG §33 (2026-06-02):** `_add_def_use_edges` now resolves `ReferenceVariable` lvalues (mapping/array writes like `balances[msg.sender]`) back to the underlying `StateVariable` via `points_to`/`points_to_origin` traversal.

**Status:** Active in v10. Our audit's DEF_USE analysis (B-1) still stands — the edge EXISTS but only gets 1 hop in Phase 2.

---

## 7. Architectural Synthesis — Why the Ceiling Exists

The CHANGELOG's interpretability suite (Run 4, F1=0.3362) and our preprocessing audit converge on the same root causes:

### Layer 1: Input Features Are Starved
- Complexity (dim 5) crushed by size normalization (A-1) → CFG node feature vectors carry ~1% original signal
- `return_ignored` (dim 7) heuristic is branch-insensitive (A-5) → unreliable for branch-heavy contracts
- `external_call_count` (dim 10) dominates ALL gradient via dataset shortcut (EXP-L4) → model ignores other features

### Layer 2: Modifier Structure Is Invisible
- MODIFIER nodes have no CFG children (A-2) → reentrancy guards (`nonReentrant`), access controls (`onlyOwner`) provide zero structural signal
- No CALLS edges from functions to modifiers (A-3) → modifier-function relationships invisible to all 3 GNN phases

### Layer 3: Phase 2 Cannot Route Useful Signal
- Phase 2 layers receive CFG features that are near-constant or confounded → GAT attention collapses to uniform weights (EXP-L3)
- DEF_USE only 1 hop (B-1) → def-use chains beyond 1 hop invisible
- CEI paths found at only 37.7% for k=8 hops (EXP-E1) → GNN's 3 Phase 2 hops cover even less

### Layer 4: Training Noise Caps Performance
- ~8% wrong-contract mapping (A-4) → direct noise across all 10 classes
- 4,304 label corrections applied (Phase 3.5) but remaining noise still significant
- 2,606/3,887 Reentrancy positives would be removed by aggressive label cleaning (too risky without cross-function BFS)

---

## 8. Update to Issue Inventory (Post-CHANGELOG)

| ID | Change | New Status |
|----|--------|------------|
| A-1 | Complexity normalization — CHANGELOG confirms EXP-L2: CFG ablation near-zero effect | **Elevated** — Phase 2 dead weight directly linked to missing complexity signal |
| A-2 | MODIFIER CFG gap — CHANGELOG confirms EXP-A4: GNN eye F1=0 for 7 classes | **Elevated** — Missing modifier CFG explains 7/10 dead GNN classes |
| A-5 | return_ignored — C-1 fix already in v10 but only fixes CFG node defaults, not branch sensitivity | **Unchanged** — C-1 is orthogonal; the branch scan bug remains |
| B-1 | DEF_USE 1 hop — CHANGELOG confirms EXP-E1: CEI reachability 37.7% at k=8 | **Elevated** — Hop depth confirmed as bottleneck |
| B-5 | external_call_count dominance — CHANGELOG EXP-L4 confirms 21-24% across ALL classes | **New severity: HIGH** — Confounded shortcut is model's primary learning signal |
</parameter>
