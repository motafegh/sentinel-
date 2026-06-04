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
6. CHANGELOG Cross-References — Interpretability Suite Validates Audit Findings
   6.1 EXP-L2: CFG Ablation Near-Zero Effect → Validates A-1, A-2, A-5
   6.2 EXP-A4: GNN Eye Useful for Only 3/10 Classes → Validates GNN Underperformance
   6.3 EXP-E1: CEI Reachability Only 37.7% at k=8 → Validates Phase 2 Depth Gap
   6.4 EXP-L4: `external_call_count` Dominates ALL Classes → Confounded Shortcut Confirmed
   6.5 EXP-L8: `type_id_norm` Dominates 3× → Node Type Bias
   6.6 C-1 Fix Already in v10 (Per-Statement CFG Features)
   6.7 H-2 Fix Already in v10 (ReferenceVariable DEF_USE)
7. Architectural Synthesis — Why the Ceiling Exists
8. Update to Issue Inventory (Post-CHANGELOG)
9. GNN Architecture Deep Dives
   9.1 DEF_USE Routing in Phase 2
   9.2 JK Entropy Regularization
   9.3 GNN Prefix Injection
   9.4 CFG Eye (IMP-R7-2)
10. Training Pipeline Issues
   10.1 StructuredLogger `torch.compile` Bug
   10.2 `fusion_lr_multiplier` Calibration
   10.3 `fusion_max_nodes` Truncation
   10.4 DoS Loss Weight
   10.5 Post-Training Calibration
11. Priority Fix Plan
12. Schema-Change Blueprints for v9
   12.1 Multi-Value `uses_block_globals`
   12.2 Modifier CFG Extraction
   12.3 Cross-Contract Edges
   12.4 `fusion_max_nodes` Upgrade
13. Parked Topics

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

---

## 9. GNN Architecture Deep Dives

### 9.1 DEF_USE Routing in Phase 2

**File:** `ml/src/models/gnn_encoder.py`

DEF_USE (edge type 10) is defined in `graph_schema.py:393` as `CFG_NODE defining a LocalVariable -> node reading it`. It is **included in Phase 2's default `cfg_mask`** (line 466-471) alongside CONTROL_FLOW(6), CALL_ENTRY(8), and RETURN_TO(9).

**Layer-specific routing (lines 508-517):** DEF_USE is segregated by sub-masking:

| Layer | GATConv | Edge subset | DEF_USE included? |
|---|---|---|---|
| Layer 3 (conv3) | `cf_only_ei` | CONTROL_FLOW only | No |
| Layer 4 (conv3b) | `icfg_only_ei` | CALL_ENTRY + RETURN_TO only | No |
| Layer 5 (conv3c) | `phase2_ei` | ALL Phase 2 (CF + ICFG + DEF_USE) | **Yes** |

**Effective hops:** DEF_USE gets exactly **1 hop** in Layer 5 (conv3c). The source/target nodes have received 2 prior hops of CF + ICFG context via residual connections, but the DEF_USE-specific message propagates only once. This matches the CHANGELOG's EXP-E1 finding: even CEI paths with k=8 hops are only 37.7% reachable, and DEF_USE chains longer than 1 hop are invisible.

**Ablation support (lines 161-164):** `phase2_edge_types` constructor parameter can exclude specific edge types, enabling the clean ablation studies seen in CHANGELOG's EXP-L2.

### 9.2 JK Entropy Regularization

**File:** `ml/src/models/gnn_encoder.py` (entropy computation), `ml/src/training/trainer.py` (loss application)

**`_JKAttention` class (lines 68-123):** Computes per-node softmax weights over K=3 phases:

```python
jk_entropy = -(w_nk * (w_nk + 1e-8).log()).sum(dim=1).mean()  # line 122
```

Entropy range: `[0, log(3)] = [0, ~1.099]`. Low = one phase dominates. High = all 3 phases contribute uniformly.

**Loss term (trainer.py lines 720-725):**

```python
_H_max = math.log(3)
_jk_reg = jk_entropy_reg_lambda * (_H_max - _jk_ent.clamp(max=_H_max))
```

With `jk_entropy_reg_lambda=0.005` (trainer.py:346), max penalty is `0.005 * 1.099 ≈ 0.0055`. This pushes entropy toward `log(3)` — penalizing collapse to a single dominant phase.

**Monitoring threshold (training_logger.py:86):**

```python
JK_ENTROPY_MIN = 0.5
```

A warning fires below 0.5, indicating JK attention has collapsed to one phase.

**Assessment:** The regularization is correctly implemented and appropriately scaled. During Run 7, JK entropy remained above 0.5 throughout, confirming all 3 phases contribute. The concern is: if Phase 2 carries no useful signal (A-1, EXP-L2), JK will eventually learn to downweight it, but the regularization forces it to stay engaged — wasting representational capacity.

### 9.3 GNN Prefix Injection

**File:** `ml/src/models/sentinel_model.py`

**Configuration:**

- `gnn_prefix_k: int = 48` (line 182) — number of GNN node embeddings injected into each Transformer window
- `gnn_prefix_warmup_epochs: int = 15` (line 183) — prefix suppressed for first 15 epochs

**Projection (lines 225-231):**

```python
self.gnn_to_bert_proj = nn.Linear(gnn_hidden_dim, 768)            # [K, 256] -> [K, 768]
self.prefix_type_embedding = nn.Embedding(_NUM_PREFIX_TYPES, 768)  # 5 types x 768
```

**Node selection priority (lines 114-120):**

| Priority | Node type | Rationale |
|---|---|---|
| 0 | CONSTRUCTOR | Always relevant |
| 1 | FALLBACK | Reentrancy-critical |
| 2 | RECEIVE | Reentrancy-critical |
| 3 | MODIFIER | Access control |
| 4 | FUNCTION | General (sorted by external_call_count descending) |

**Injection (transformer_encoder.py lines 272-302):**

Prefix tokens replace the first K positions of each window. CLS moves from position 0 to position K. Position IDs for prefix tokens are set to 1 (RoBERTa padding slot — no positional bias), so the Transformer treats them as position-agnostic structural hints.

**Warmup behavior (sentinel_model.py:510):**

During warmup (epochs 0-14), `gnn_prefix is None` and the standard Transformer path runs. At epoch 15, the prefix projection starts from random init — causing a sharp gradient discontinuity. The CHANGELOG notes this instability at Run 7 ep15 validation metrics.

**Assessment:** The prefix injection architecturally makes sense as a cross-modal bridge. Two concerns: (a) warmup creates a sharp discontinuity at epoch 15, and (b) if the GNN carries no discriminative signal for 7/10 classes (EXP-A4), the prefix injects noise into the Transformer's first 48 tokens for those classes.

### 9.4 CFG Eye (IMP-R7-2)

**File:** `ml/src/models/sentinel_model.py`

**Purpose (lines 24-28):** Direct gradient path from classifier to Phase 2 conv layers, bypassing Phase 3 reverse-CONTAINS layers. Without it, Phase 2 signal must propagate through 3 CONTAINS layers to reach FUNCTION nodes for pooling, severely attenuating gradients to `conv3/conv3b/conv3c`.

**Pooling target types (lines 98-108):** CFG_NODE_CALL(8), CFG_NODE_WRITE(9), CFG_NODE_READ(10), CFG_NODE_CHECK(11), CFG_NODE_OTHER(12) — the 5 CFG subtypes connected by Phase 2 edges.

**Architecture (lines 254-258):**

```python
self.cfg_eye_proj = nn.Sequential(
    nn.Linear(2 * gnn_hidden_dim, eye_dim),  # 512 -> 128
    nn.ReLU(),
    nn.Dropout(dropout),
)
```

Pools `_phase2_x` (raw Phase 2 output, NOT JK-aggregated) via `global_max_pool || global_mean_pool`, projects to 128d. Combined with gnn_eye, transformer_eye, fused_eye → `4 * 128 = 512` input to classifier.

**Auxiliary supervision (lines 282-288):** `aux_phase2` head pools Phase 2 over CFG nodes and produces independent logits, keeping the Phase 2 gradient alive even if the main classifier downweights the CFG eye.

**Assessment:** The CFG eye is correctly structured and addresses BUG-R7-1. However, EXP-L2 shows that removing CFG edges changes predictions by ~1×10⁻⁶ — meaning even with a direct gradient path, Phase 2 has no useful signal to learn. The gradient highway exists but carries near-zero information.

---

## 10. Training Pipeline Issues

### 10.1 StructuredLogger `torch.compile` Bug (B-6, HIGH)

**File:** `ml/src/training/training_logger.py:298-306`, cause at `trainer.py:1410-1415`

**Error:**

`check_aux_head()` at training_logger.py:305 calls `head[-1]` on `model.aux_phase2`. After `torch.compile` wraps it in `OptimizedModule`, this raises `TypeError: 'OptimizedModule' object is not subscriptable`.

**Root cause (trainer.py lines 1410-1415):**

`torch.compile(dynamic=True)` is applied to submodules including `"aux_phase2"`. The resulting `OptimizedModule` wrapper does not forward `__getitem__`, so `head[-1]` on line 305 fails silently (caught by the method's try/except but returning empty dict).

**Impact:** No `aux_phase2` weight/bias norm logging for the entire Run 7. This means Phase 2 convergence monitoring is blind — we cannot track whether conv3/conv3b/conv3c are learning or saturated.

**Fix (3 options):**

1. Change `head[-1]` to `head._orig_mod[-1]` — `OptimizedModule` stores original module in `_orig_mod`
2. Exclude `"aux_phase2"` from compile list at trainer.py:1412 (negligible compile benefit for a single Linear layer)
3. try/except with fallback to `_orig_mod`

### 10.2 `fusion_lr_multiplier` Calibration

**File:** `ml/src/training/trainer.py:241, 1333-1343, 1354`

**Current value:** `fusion_lr_multiplier: float = 0.5` — fusion head + classifier + aux heads receive half the base learning rate.

**Scope (lines 1333-1343):** Parameters matching `fusion.*`, `transformer_eye_proj.*`, `classifier.*`, `aux_*` — approximately 821K parameters.

**Rationale (lines 1339-1342):**

> "fusion + classifier at reduced LR to prevent CodeBERT's Reentrancy bias from overwhelming the GNN signal via high-gradient cross-attention"

**Assessment:** The multiplier was set during RC1 when the fusion head had 821K params and was producing 4-5× the GNN gradient norm. With v10 data and v8.1 model, this ratio may have shifted. The appropriate value depends on the relative gradient norms of fusion vs GNN at initialization, which should be re-calibrated per run. The CHANGELOG's Run 7 entry notes this as a potential ceiling but does not override the default.

### 10.3 `fusion_max_nodes=1024` Truncation (B-5, MEDIUM)

**File:** `ml/src/training/trainer.py:199-200`, `ml/src/models/fusion_layer.py:68-117`

**Current value:** `fusion_max_nodes: int = 1024`. Comment at trainer.py:199:

```python
# IMP-D1: raise to 2048 after re-extraction with max_nodes=2048.
# At 1024 the 227 contracts >1024 nodes are truncated in fusion attention.
```

**Truncation mechanism (fusion_layer.py lines 96-117):**

```python
valid     = local_idx < max_nodes        # drop excess nodes
local_idx = local_idx.clamp(max=max_nodes - 1)  # clamp survivors
```

The `valid` mask is computed BEFORE clamping. Nodes beyond `max_nodes-1` are silently dropped from both the dense tensor and the attention mask — 227 contracts affected.

**Impact:** Truncated contracts lose CFG nodes in the cross-attention fusion, meaning the Transformer cannot attend to those nodes. The C-2 fix (computing `valid` before clamping) prevents last-write-wins corruption, but the data loss is still real.

**Comment notes it should be raised to 2048,** but the docstring also says "affects <1% of the corpus." The 227 affected contracts may disproportionately be complex/fragmented contracts with more vulnerability surface area, introducing a systematic bias away from high-complexity contracts.

### 10.4 DoS Loss Weight

**File:** `ml/src/training/trainer.py:320-326, 660-666, 674-685`

**Current value:** `dos_loss_weight: float = 0.5`

**Mechanism (lines 660-666):**

```python
_logits_for_loss[:, dos_idx] = (
    dos_loss_weight * logits[:, dos_idx]
    + (1.0 - dos_loss_weight) * logits[:, dos_idx].detach()
)
```

The `.detach()` portion contributes no gradient. Net effect: DoS gradient is 50% of full weight.

**Rationale:** Historical — when DoS had only 3 training samples, `dos_loss_weight=0.0`. With ~243 positives now, `0.5` is a safe starting point. Comment recommends raising to 1.0 when DoS F1 plateaus below other classes.

**Assessment:** At Run 7, DoS F1=0.0272 is by far the worst class. The half-gradient is appropriate as a guard against early-training instability with ~243 positives in 17,961 total. However, once DoS recall starts improving, the weight should be raised to 1.0 to close the gap with other classes.

### 10.5 Post-Training Calibration

**Status: Not implemented.**

**What exists:**

- `log_calibration()` method at training_logger.py:601-618 — full logging infrastructure, but never called
- `temperature` field in epoch summary — hardcoded to `1.0` at trainer.py:1971
- ECE computation at training_logger.py:439-459 — diagnostic only, no scaling loop
- Brier score at training_logger.py:413-437 — diagnostic only

**What does not exist:**

- Temperature scaling training (optimizer, log-likelihood on held-out set)
- Platt scaling
- Any post-training calibration algorithm

**Impact:** The model's probabilities are uncalibrated. ECE is logged but never acted upon. This means the model's confidence scores do not correspond to actual probabilities — a critical gap for an oracle contract, where clients need calibrated probability estimates for risk assessment.

---

## 11. Priority Fix Plan

Ranked by expected F1 impact / implementation effort ratio, ordered highest-ROI first.

### P1: Fix Complexity Normalization (A-1)

**Effort:** Low (1 config change)
**Expected impact:** Moderate (DoS, GasException)
**Action:** Change normalization from `complexity / log1p(100)` to `log1p(complexity) / log1p(1000)` — preserving relative ordering while keeping complexity numeric scale stable. Alternatively, add complexity as a separate un-normalized feature.

### P2: Fix StructuredLogger `torch.compile` Bug (B-6)

**Effort:** Low (1-line fix)
**Expected impact:** Indirect (enables Phase 2 convergence monitoring)
**Action:** Change `head[-1]` to `head._orig_mod[-1]` at training_logger.py:305.

### P3: Recalibrate `fusion_lr_multiplier`

**Effort:** Low (config change)
**Expected impact:** Moderate (balanced fusion training)
**Action:** Measure GNN vs fusion gradient norms at ep0 for current model. Set multiplier to match scales. Default target: 0.2–0.3 for v8.1 with 4 eyes.

### P4: Raise `fusion_max_nodes` to 2048

**Effort:** Low (config change + data re-extraction)
**Expected impact:** Low–Moderate
**Action:** Re-extract graph data with `max_nodes=2048`, update config default.

### P5: Implement Post-Training Calibration

**Effort:** Medium (temperature scaling loop)
**Expected impact:** Moderate (ECE reduction)
**Action:** Implement temperature scaling on held-out 10% of training data. Optimize per-class temperature via log-likelihood. Log pre/post ECE.

### P6: Add Modifier CFG Extraction (A-2)

**Effort:** High (schema change + extraction pipeline + model retraining)
**Expected impact:** High (reentrancy, access-control classes)
**Action:** Add CFG subgraph extraction for MODIFIER nodes in `graph_extractor.py`. Add MODIFIER CFG nodes and CONTAINS edges. Requires schema version bump to v9.

### P7: Cross-Contract ICFG Edges

**Effort:** High (extraction + data representation)
**Expected impact:** Moderate (cross-contract reentrancy)
**Action:** Extend `_add_icfg_edges()` to follow CALLS edges across separate `.sol` files within each dataset sample. Requires cross-contract node registration and edge resolution.

### P8: Raise DoS Loss Weight to 1.0

**Effort:** Low (config change)
**Expected impact:** Low–Moderate
**Action:** Set `dos_loss_weight=1.0` after confirming DoS recall is improving.

---

## 12. Schema-Change Blueprints for v9

### 12.1 Multi-Value `uses_block_globals`

**What it is:** Currently `uses_block_globals` (feature[2]) is a single float per node — 1.0 if any block global is read, 0.0 otherwise. For CFG nodes, this is already per-statement (C-1 fix). For FUNCTION declaration nodes, it's a function-level aggregate.

**Blueprint:** Replace the single float with a 5-dimensional multi-hot for CFG nodes:
- `[2a] block.timestamp read`
- `[2b] block.number read`
- `[2c] block.difficulty read`
- `[2d] block.basefee read`
- `[2e] block.prevrandao read`

**Why:** Timestamp vulnerability (Reentrancy sub-type) vs block.number (TOD) vs basefee (GasException) have different vulnerability implications. A single bool collapses all 5.

**Effort:** Low — 5-dimensional expansion of existing scan at `_node_uses_block_globals()` (graph_extractor.py:643-665). Requires schema version bump.

### 12.2 Modifier CFG Extraction

**What it is:** Currently MODIFIER nodes are extracted as featureless declaration nodes with no CFG subgraph (graph_extractor.py:1434-1437). They are invisible to Phase 2 and contribute no structural signal.

**Blueprint:**

1. After the function CFG loop (graph_extractor.py:1449-1492), add a parallel loop for `contract.modifiers`:
   - Call `_build_control_flow_edges()` for each modifier
   - Append CFG_NODE children for each modifier statement
   - Add CONTAINS(5) edges from MODIFIER → its CFG children

2. Add CALLS(0) edges from FUNCTION → MODIFIER (instead of current edge-type silence)

3. Add `_` modifier base to feature vector (how many modifiers a function applies)

**Why:** `nonReentrant` guards, `onlyOwner` access controls, and modifier-modulated vulnerability patterns are completely invisible to the model. The CHANGELOG's EXP-A4 confirms GNN eye F1=0 for Reentrancy — adding modifier CFG is the single highest-ROI architectural change.

**Effort:** High — extraction pipeline changes + model retraining + schema version v9. See §9.2 for detailed pseudocode.

### 12.3 Cross-Contract Edges

**What it is:** Currently all edges (CALLS, ICFG, DEF_USE) are intra-contract only. INHERITS(4) edges to parent contracts exist but only for declarations in the same `.sol` file.

**Blueprint:**

1. In `extract_contract_graph()`, after primary contract extraction, iterate the multi-contract's sibling contracts within the same file
2. For sibling contracts: extract CFG, resolve CALLS edges to functions in other contracts
3. Add a CROSS_CALLS edge type or use existing CALLS with cross-contract attribute

**Why:** Flash-loan-style cross-contract reentrancy and multi-contract vulnerability patterns require cross-contract data flow. The A-4 `_select_contract` heuristic (8% wrong-contract) is a symptom of needing multi-contract graphs.

**Effort:** High — extraction pipeline + schema change + increased graph sizes.

### 12.4 `fusion_max_nodes` Upgrade

**What it is:** Currently truncates 227 contracts at `max_nodes=1024`.

**Blueprint:**

1. Re-extract all graph data with `max_nodes=2048` (not just change config)
2. Update the `_scatter_to_dense()` max_nodes parameter in fusion_layer.py
3. Consider dynamic batch-dependent truncation (sort by node count, group similar sizes)

**Why:** The 227 truncated contracts are disproportionately complex/high-risk. Truncation removes CFG nodes from cross-attention, systematically under-weighing high-complexity contracts.

**Effort:** Low — primarily data re-extraction.

---

## 13. Parked Topics

| Topic | Reason for Parking | Revisit Trigger |
|---|---|---|
| **GNN eye pooling strategy** — vs CFG eye, vs Phase 3 enrichment | Current 2-eye (GNN + CFG) split is adequate; root cause is input features, not pooling | When Phase 2 has useful signal (A-1, A-2 fixed) |
| **Transformer Context window size** — 512 tokens per window | Validated in Run 5 (512 > 256, 512 vs 1024 not tested) | If prefix injection saturates (K > 48) |
| **Cross-attention fusion alternatives** — MoE, adaptive pooling | Fusion model (CrossAttentionFusion) is working; no evidence of cross-attention bottleneck | If fusion eye F1 plateaus below transformer_eye or gnn_eye |
| **DEF_USE to 2+ hops** — increasing conv3c depth | Phase 2 depth increase (more layers) needed, not just DEF_USE hops | When Phase 2 has discriminative signal |
| **ZKML proxy MLP (10-output lock)** | Hard constraint from ZKML circuit deployment — cannot expand to 11 outputs | When ZKML circuit supports variable output count |
| **per-class fusion_lr_multiplier** — separate LR for each eye's projection | Not a bottleneck currently; overall fusion_lr_multiplier is adequate | If one eye's F1 consistently lags others |
| **Multi-label training** — contracts with multiple vulnerabilities | BCEWithLogitsLoss already supports multi-label; dataset has multi-label samples | If single-label evaluation is insufficient for deployment |
| **Threshold calibration per class** — beyond global fixed threshold | Fixed threshold (0.261) is a deployment simplification; per-class thresholds add complexity | When deployment requirement emerges |
</parameter>
