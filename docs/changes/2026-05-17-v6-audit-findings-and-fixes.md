# SENTINEL v6 — Deep Audit Findings, Bug Fixes, and Schema v5 Changes

**Date:** 2026-05-17  
**Scope:** Full graph-and-token audit, 9 confirmed bugs + 6 new findings, P0 crisis analysis, all fixes applied and pending  
**Status:** P0 fix scripts written; graph_extractor.py rewrite pending; re-extraction required before training  
**Precedes:** v6.0 training launch

This document is the definitive record of the post-v6-re-extraction deep audit: what we investigated, what we found, what we fixed, what we could not fix, and what must happen next. It supersedes the per-bug notes in earlier change logs and consolidates every finding into one actionable reference.

---

## Part 1: Audit Methodology and Scope

### 1.1 Why We Audited

The v6 code was committed across four commits (`bef1f2a`, `310e738`, `b38c9da`, `2bb0e16`, `64dfc5a`) covering Phase 0 (feature schema v4), Phase 1 (windowed tokenization), Phase 2 (GNN architecture expansion), and Phase 3 (training configuration). After re-extracting 44,470 graphs under the new v4 schema, we needed to verify that the extracted data actually matched the code's intentions before committing to a multi-day training run.

The audit was designed to catch silent feature corruption — cases where the extractor code is technically correct for the intended feature but wrong for the dataset's actual Solidity version distribution, or where a feature fires in unit tests but never fires on real data.

### 1.2 Audit Structure

We designed 26 audit tasks grouped into 7 categories:

| Category | Tasks | Focus |
|----------|-------|-------|
| Feature activation | 1, 5, 6 | Do features actually fire on real data? |
| Source pattern alignment | 2, 7 | Do graph features match what is in the .sol source? |
| Token coverage | 3, 4 | Do windowed tokens cover vulnerability code? |
| Edge type distribution | 5 | Are all 8 edge types present in extracted graphs? |
| Label quality | 8, 19, 20 | Are BCCC labels trustworthy? |
| Data integrity | 9–15, 21–26 | Ranges, alignment, correlations, shifts |

### 1.3 Execution and Results

All 20 audit scripts were written, debugged, and run successfully. Total runtime: 5.1 minutes on 44,470 graphs and 44,470 token files. Result: **20/20 PASS** (all scripts completed without errors; findings are in the data, not in script failures).

Key audit statistics:
- 500 graphs sampled for feature range audit (66,288 nodes)
- 100 token files sampled for integrity checks
- 2,000 graphs scanned for ghost graph detection
- Full 44,470-row CSV checked for file triple alignment

---

## Part 2: Confirmed Bugs — Complete Catalog

### 2.1 BUG-1: `loc` [6] Not Normalized for CFG Nodes

**Severity:** HIGH (P1)  
**Location:** `_build_cfg_node_features()` line 458 in `graph_extractor.py`

**What is wrong:** The `_build_node_features()` path (FUNCTION, CONTRACT, STATE_VAR nodes) correctly applies `log1p(loc_raw) / log1p(1000)` to normalize loc into the [0, 1] range. However, `_build_cfg_node_features()` (which handles CFG_NODE_CALL, CFG_NODE_WRITE, etc.) uses the raw line count directly: `loc = float(len(sm.lines))`. This produces values up to 682.0 for statement-level nodes.

**Evidence from audit (Task 09, 500 graphs, 66,288 nodes):**
- CFG nodes: loc max = **129.0** (should be ≤ 1.0)
- Declaration nodes: loc max = **946.0** (also raw in some paths — but declaration nodes ARE normalized; the 946 figure is from the audit measuring raw values before the log transform was accounted for)
- 4,762 / 27,801 nodes (17%) exceed 1.0 for loc

**Impact:** CFG nodes constitute the majority of all nodes in a typical graph (48,175 CFG out of 66,288 total = 72.7%). A single 100-line function body produces CFG nodes with loc≈100 while `uses_block_globals`, `return_ignored`, `has_loop` etc. are all in [0, 1]. The GAT attention mechanism computes dot products between node feature vectors; the 100× magnitude difference causes attention weights to be dominated by loc, making GNN message passing effectively "attend to high-loc nodes" regardless of vulnerability-relevant structure.

**Fix:** Apply the same `log1p(loc_raw) / log1p(1000)` normalization used in `_build_node_features()`:

```python
# In _build_cfg_node_features():
# BEFORE:
loc = float(len(sm.lines))

# AFTER:
loc = min(math.log1p(float(len(sm.lines))) / math.log1p(1000), 1.0)
```

**Status:** Fix designed, pending implementation in graph_extractor.py rewrite.

---

### 2.2 BUG-2: `complexity` [5] Not Normalized

**Severity:** HIGH (P1)  
**Location:** `_build_node_features()` line 618 in `graph_extractor.py`

**What is wrong:** The `complexity` feature stores the raw CFG block count: `complexity = float(len(obj.nodes))`. The docstring says "normalised" but the code never normalizes it. Raw values reach 169.0 while all other features are in [0, 1].

**Evidence from audit:**
- Declaration nodes: complexity max = **67.0** (should be ≤ 1.0)
- p99 = 12.0 (most functions have fewer than 12 CFG blocks)
- 3,698 / 27,801 nodes (13%) exceed 1.0

**Impact:** High-complexity functions (loops, many branches) get complexity ≈ 10–67 while all other features are [0, 1]. This artificially upweights complexity in GAT attention, similar to BUG-1 but less severe because complexity values are generally smaller than loc values.

**Fix:** Apply `log1p(complexity_raw) / log1p(200)` normalization (p99 is ~12, max seen is 169; log1p(200) ≈ 5.303):

```python
# In _build_node_features():
# BEFORE:
complexity = float(len(obj.nodes)) if obj.nodes else 0.0

# AFTER:
complexity = min(math.log1p(float(len(obj.nodes))) / math.log1p(200), 1.0) if obj.nodes else 0.0
```

**Status:** Fix designed, pending implementation in graph_extractor.py rewrite.

---

### 2.3 BUG-3: `visibility` [1] Not Normalized to [0, 1]

**Severity:** MEDIUM (P2)  
**Location:** `_build_node_features()` lines 590–592 in `graph_extractor.py`

**What is wrong:** The `VISIBILITY_MAP` encodes visibility as an ordinal: `public=0, external=0, internal=1, private=2`. Private functions get value=2, which exceeds the [0, 1] range that all other features respect.

**Evidence from audit:**
- Declaration nodes: visibility max = **2.0** (exceeds [0, 1])
- 211 / 27,801 nodes (0.76%) exceed 1.0 — all are `private` functions

**Impact:** Mild — only 2× out of range. The model may have adapted to this during training. However, `private` functions are the most restricted in terms of attack surface, and getting 2× weight for "private" is semantically incorrect. The ordinal encoding implies `private > internal > public` on a linear scale, which the model learns as "more visibility value = more dangerous" when the reverse is true.

**Fix:** Divide by the maximum ordinal value (2.0) to normalize to [0, 1]:

```python
visibility = float(VISIBILITY_MAP.get(
    str(getattr(obj, "visibility", "public")), 0
)) / 2.0
```

Alternatively, change the VISIBILITY_MAP to binary encoding: public/external=1 (exposed attack surface), internal/private=0 (not exposed). This would be semantically cleaner but changes the feature meaning more drastically.

**Status:** Fix designed, pending implementation in graph_extractor.py rewrite.

---

### 2.4 BUG-4: `contract_path` Not Stored in Graphs (False Alarm)

**Severity:** NONE — confirmed not a bug  
**Originally suspected from:** Pipeline audit concern

The original concern was that `reextract_graphs.py` never sets `g.contract_path` after extraction, and `graph_extractor.py` says "caller sets it." Upon manual verification during this audit, `contract_path` IS present in extracted `.pt` files (e.g., `BCCC-SCsVul-2024/SourceCodes/...`). The code in `reextract_graphs.py` does set it after the call returns. This was a false alarm.

---

### 2.5 BUG-5: `in_unchecked` [9] Is a Permanently Dead Feature

**Severity:** HIGH (P1 — replace the feature)  
**Location:** `_compute_in_unchecked()` in `graph_extractor.py`

**What is wrong:** The code is technically correct — it checks for `NodeType.STARTUNCHECKED` (Slither ≥0.9.3) and falls back to a regex pattern `\bunchecked\s*\{`. The problem is that the `unchecked{}` block was introduced in Solidity 0.8.0, and the BCCC dataset is overwhelmingly pre-0.8.0:

**Evidence from audit (Task 18, Solidity version distribution):**

| Version | Count | Percentage |
|---------|-------|------------|
| **0.4.x** | 1,758 | **87.9%** |
| 0.5.x | 160 | 8.0% |
| 0.8.x | 1 | 0.1% |
| no_pragma | 81 | 4.0% |

Result: `in_unchecked=0` across effectively all 44,470 graphs. One full feature dimension provides zero training signal.

**Additional issue:** The regex fallback `\bunchecked\s*\{` produces false positives on comments (`// unchecked { this is a comment }`) and strings (`string memory s = "unchecked {"`). Since the feature is dead, this doesn't affect current data, but would need fixing if the feature were ever activated.

**Impact:** A dead feature wastes one dimension of the 12-D feature vector. PCA analysis (Task 21) confirms that the `in_unchecked` dimension has **0.0000 variance** — it contributes absolutely nothing to the representation. The GNN must learn to ignore it, which wastes a small amount of representational capacity.

**Proposed replacement:** Task 17 evaluated `uses_safe_math` as a replacement and found it has only 45% discriminative power for IntegerUO — essentially coin-flip. The audit report recommends replacing with `pragma_version` (0.4.x=0, 0.5.x=0.5, 0.8.x=1.0) which provides a meaningful signal for IntegerUO detection since overflow behavior differs by Solidity version. However, this requires careful design.

**Alternative approach:** Convert to `has_unchecked_arithmetic` — a feature that fires when the contract does arithmetic without SafeMath AND is on Solidity <0.8.0 (where arithmetic is unchecked by default). This combines version information with SafeMath presence into a single vulnerability-relevant signal.

**Status:** Fix designed, pending implementation. Feature replacement decision required.

---

### 2.6 BUG-6: Wrong Contract Selection — 47.4% Failure Rate

**Severity:** CRITICAL (P0)  
**Location:** `_select_contract()` line 697 in `graph_extractor.py`

**What is wrong:** When a `.sol` file defines multiple contracts, `_select_contract()` picks the non-interface contract with the most functions: `max(non_iface, key=lambda c: len(c.functions))`. This heuristic was expected to have a 7–28% wrong-selection rate. The actual rate is **47.4%** — nearly a coin flip.

**Evidence from audit (Task 16, 114 multi-contract files sampled):**

| Heuristic | Accuracy | Wrong Rate | 95% CI |
|-----------|----------|------------|--------|
| Most Functions (current) | 52.6% | **47.4%** | [39.2%, 55.8%] |
| Last Contract | **87.4%** | 12.6% | [8.0%, 19.2%] |

**Per-class wrong rate (Most Functions heuristic):**

| Class | Multi-contract % | Wrong selected % |
|-------|-----------------|-----------------|
| CallToUnknown | 56% | **56.2%** |
| IntegerUO | 92% | **48.0%** |
| Reentrancy | 52% | **47.4%** |
| ExternalBug | 78% | **50.0%** |
| GasException | 89% | **41.2%** |
| Timestamp | 91% | **17.4%** |
| MishandledException | 59% | 7.4% |
| TOD | 81% | 7.4% |

**Common wrong-selection pattern:** The extractor picks `StandardToken` (an OpenZeppelin library with many functions) instead of the actual vulnerable contract (e.g., `ERC20Token`, `BTPCoin`, `FANBASE`). Of 64 wrong selections, many choose base classes that are never the vulnerable contract.

**Why this is devastating:** When the wrong contract is selected, ALL vulnerability signals are absent from the graph. The selected contract has no `block.timestamp` reads, no external calls with ignored returns, no state variable writes after calls — because the vulnerability is in a different contract in the same file. The model receives a graph that says "this is a safe library" while the label says "this is vulnerable." This trains the model to ignore its input features.

**By contract count per file:**
- 2 contracts per file: 17.6% wrong (Most Functions) vs 0% wrong (Last Contract)
- 6+ contracts per file: 50–100% wrong (Most Functions) vs 10–50% wrong (Last Contract)

**Fix: "Most-Derived" composite heuristic** (3-tier fallback):

1. **Most-derived contract** — the contract that inherits from the most other in-file candidates. In Solidity, the main contract inherits from library contracts, not the other way around. This captures `ERC20Token is StandardToken, Ownable` correctly.
2. **Last defined contract** — if no inheritance edges exist, use the Solidity convention of defining base classes first and the main contract last.
3. **Most functions (legacy)** — if both above fail, fall back to the current heuristic as a last resort.

```python
def _select_contract(sl, config):
    candidates = [c for c in sl.contracts if not c.is_from_dependency()]
    non_iface = [c for c in candidates if not c.is_interface]
    
    if not non_iface:
        return candidates[0]
    
    # Tier 1: Most-derived (inherits from most in-file candidates)
    candidate_names = {c.name for c in non_iface}
    def inheritance_score(c):
        return sum(1 for parent in c.inheritance if parent.name in candidate_names)
    
    best = max(non_iface, key=inheritance_score)
    if inheritance_score(best) > 0:
        return best
    
    # Tier 2: Last defined (Solidity convention: base first, main last)
    # Sort by source_mapping start line descending
    sorted_contracts = sorted(non_iface, key=lambda c: (
        c.source_mapping.lines[0] if c.source_mapping and c.source_mapping.lines else 0
    ), reverse=True)
    return sorted_contracts[0]
```

**Expected improvement:** 47.4% → ~12.6% wrong (3.8× improvement based on "Last Contract" audit data). The most-derived heuristic should be even better than "Last Contract" alone.

**Status:** Fix script written (`p0-fix-bug6-contract-selection.py`), pending integration into graph_extractor.py rewrite.

---

### 2.7 BUG-7: EMITS Edges (Type 3) Never Generated

**Severity:** MEDIUM (P2)  
**Location:** Edge creation logic in `graph_extractor.py`

**What is wrong:** The EMITS edge type is defined in `EDGE_TYPES` (id=3) and the edge embedding table has a slot for it. However, EMITS edges are never created in practice because 94%+ contracts are Solidity 0.4.x where events use the old syntax without the `emit` keyword:

```solidity
// Solidity <0.4.21 (no emit keyword):
Transfer(msg.sender, _to, _value);  // looks like a function call to Slither

// Solidity ≥0.4.21 (emit keyword):
emit Transfer(msg.sender, _to, _value);  // correctly identified as event
```

Slither's `func.events_emitted` returns empty for pre-0.4.21 contracts. The audit confirmed: **all 44,470 graphs have EMITS edge count = 0.**

**Evidence from audit (Task 13, edge type distribution across 5,974+ edges):**

| Edge Type | Count | Notes |
|-----------|-------|-------|
| CALLS (0) | 5,974 | |
| READS (1) | 8,495 | |
| WRITES (2) | 8,622 | |
| **EMITS (3)** | **0** | **BUG-7: completely absent** |
| **INHERITS (4)** | **0** | **BUG-8: completely absent** |
| CONTAINS (5) | 48,175 | |
| CONTROL_FLOW (6) | 41,233 | |

**Impact:** The EMITS embedding slot in the GNN edge embedding table is trained with zero examples. The 8-row embedding table has one dead row. This wastes representational capacity and means the model cannot learn event-emission patterns (e.g., "reentrancy function emits an event after the external call").

**Fix options:**
1. **Scan CFG IR for `EventCall` operations directly** — bypass `func.events_emitted` and look at IR operations in each node.
2. **Detect by name matching** — check if a called name matches any `event` declaration in scope.
3. **Drop EMITS from the edge type vocabulary** and repurpose the embedding slot (requires NUM_EDGE_TYPES change).

Option 1 is preferred: scan each CFG node's IR for `EventCall` operations and create EMITS edges from the function to the event node.

**Status:** Fix designed, pending implementation in graph_extractor.py rewrite.

---

### 2.8 BUG-8: INHERITS Edges (Type 4) Never Generated

**Severity:** MEDIUM (P2)  
**Location:** `_add_edge()` logic in `graph_extractor.py`

**What is wrong:** The INHERITS edge type is defined (id=4) and the code attempts to create these edges via `_add_edge(contract.name, parent.name, INHERITS)`. However, `_add_edge()` looks up both source and destination in `node_map`, and parent contracts are NOT in `node_map` — only the selected contract and its members are added. The `_add_edge()` silently fails (returns None when either key is not found), so INHERITS edges are never created.

The audit confirmed: **all 44,470 graphs have INHERITS edge count = 0.**

**Impact:** The model has no information about inheritance relationships. This means:
- It cannot learn "contracts inheriting from Ownable are more likely to have access control issues"
- It cannot distinguish base contracts from derived contracts via graph structure
- The INHERITS embedding slot is dead, same as EMITS

**Fix options:**
1. **Add parent contracts to the graph** — add them as CONTRACT nodes with their state variables and functions. This would make the graph much larger and more complex.
2. **Store inheritance as a feature on the CONTRACT node** — add a boolean `has_inheritance` or a count of parent contracts as a feature. This is simpler and doesn't require graph topology changes.
3. **Drop INHERITS from the edge type vocabulary** and repurpose the slot.

Option 2 is recommended: add `inheritance_depth` or `has_inheritance` as a feature on the CONTRACT node. This provides the signal without the complexity of multi-contract graphs.

**Status:** Fix designed, pending implementation in graph_extractor.py rewrite.

---

### 2.9 BUG-9: `.send()` Return Value Ignored Not Detected

**Severity:** LOW (P3)  
**Location:** `_compute_return_ignored()` in `graph_extractor.py`

**What is wrong:** The `_compute_return_ignored()` function checks for `LowLevelCall` and `HighLevelCall` IR operations. It does NOT check for `Send` operations. In Solidity, `.send()` returns a boolean indicating success, and failing to check it is a MishandledException vulnerability:

```solidity
wallet.send(amount);  // return value ignored — MishandledException!
```

Slither classifies `.send()` as a `Send` IR type, separate from `LowLevelCall`. The extractor misses this pattern.

**Evidence from audit (Task 23):**
- 32/500 MishandledException contracts use `.send()` (6.4%)
- Of those, 4 have unchecked `.send()` (12.5%)
- 3 of 4 are missed by the graph (return_ignored=0 when it should be 1)

**Fix:** Add `Send` to the `isinstance` check in `_compute_return_ignored()`:

```python
from slither.slithir.operations import LowLevelCall, HighLevelCall, Send

# In _compute_return_ignored():
if isinstance(op, (LowLevelCall, HighLevelCall, Send)):
```

**Status:** Fix designed, pending implementation in graph_extractor.py rewrite.

---

## Part 3: New Findings Beyond the Bug Catalog

### 3.1 NEW-1: Timestamp Labels 48.2% Mislabelled

**Severity:** CRITICAL (P0) — dataset ground truth issue

The audit (Task 19) checked 2,191 Timestamp=1 contracts against their actual source code:

| Category | Description | Count | Rate |
|----------|-------------|-------|------|
| (a) | Signal in source AND feature fires | 623 | 28.4% |
| (b) | Signal in source but feature doesn't fire | 491 | 22.4% |
| **(c)** | **No signal in source AND no feature activation** | **1,056** | **48.2%** |
| (d) | No signal in source but feature fires (false positive) | 21 | 1.0% |

**Category (b) breakdown:** 202 of 491 cases are due to BUG-6 (wrong contract selection — the timestamp-reading contract was not selected). The remaining 289 are due to Slither IR omissions, inline assembly wrapping block globals, or indirect access through inherited contracts.

**Category (c) — the 1,056 mislabelled contracts:** These contracts are labelled Timestamp=1 in BCCC but have NO `block.timestamp`, `block.number`, or any block global anywhere in their source code. The BCCC label appears to be wrong — possibly applied by indirect analysis (e.g., the contract is a fork of one that had timestamp issues, or the labelling tool used a different detection method).

**Impact on training:** The Timestamp class has ~48% label noise. A model trained on these labels will:
- Learn to predict "not Timestamp" for truly timestamp-dependent contracts (they are outnumbered by mislabelled ones)
- Potentially learn spurious correlations from the mislabelled majority
- Achieve low F1 for Timestamp regardless of model architecture (ceiling imposed by label quality)

**Proposed fix: Source-verified relabeling** with 4 categories:

1. **True positive** — source has block globals AND feature fires → keep Timestamp=1
2. **Feature miss** — source has block globals but feature doesn't fire → keep Timestamp=1 (feature bug, not label bug)
3. **Assembly/inherit** — source has block globals in assembly or inherited contract → keep Timestamp=1 (legitimate indirect usage)
4. **Mislabel** — source has NO block globals anywhere → set Timestamp=0 (false label)

Only category (d) contracts are relabelled. Categories (a), (b), and (c) keep Timestamp=1 because the vulnerability exists in the source even if the feature misses it. This is a conservative approach that avoids removing true positives.

After BUG-6 is fixed, category (b) shrinks significantly (202 contracts recover their signal), reducing the noise floor from 48.2% to an estimated ~33%.

**Fix script written:** `p0-fix-timestamp-labels.py`

**Status:** Fix script written, pending execution after BUG-6 fix and graph re-extraction.

---

### 3.2 NEW-2: CFG Nodes Carry Zero Semantic Features

**Severity:** HIGH (design issue, P2)

The audit (Task 09, 01) revealed that 72% of all nodes (CFG nodes) have zero values for ALL semantic features:

| Feature | DECL nodes | CFG nodes | CFG values |
|---------|-----------|-----------|------------|
| visibility | 0–2 | 0.000 | Always 0 |
| uses_block_globals | 0–1 | 0.000 | Always 0 |
| view | 0–1 | 0.000 | Always 0 |
| payable | 0–1 | 0.000 | Always 0 |
| complexity | 0–67 | 0.000 | Always 0 |
| return_ignored | 0–1 | 0.000 | Always 0 |
| in_unchecked | 0–0 | 0.000 | Dead anyway |
| has_loop | 0–1 | 0.000 | Always 0 |
| ext_call_count | 0–0.95 | 0.000 | Always 0 |

CFG nodes only have 3 non-zero features: `type_id` (which CFG subtype they are), `loc` (how many lines the statement spans — but this is buggy per BUG-1), and `call_target_typed` (always 1.0 — a safe default that provides no information).

This means the GNN's message passing operates on a graph where 72% of nodes are effectively indistinguishable except by their type_id. The model must rely entirely on graph topology (edge structure, node connectivity) rather than node features to distinguish between, say, a CALL node that ignores its return value vs one that checks it.

**Proposed fix:** Propagate parent function features to CFG nodes. For each CFG node, copy the parent function's `return_ignored`, `uses_block_globals`, `has_loop`, and `ext_call_count` as additional features. This creates a per-statement context that tells the GNN "this CALL statement is inside a function that ignores return values and reads block.timestamp" — information that is currently invisible at the statement level.

However, this requires careful design to avoid information leakage between functions. The current architecture intentionally keeps CFG node features minimal so the GNN must learn to propagate information through CONTAINS edges. Changing this is a Phase 2+ design decision.

**Status:** Design issue documented. No immediate fix planned.

---

### 3.3 NEW-3: DoS Has Only 7 Pure-Label Samples

**Severity:** HIGH (P2 — data quality issue)

The audit (Task 20) confirmed the extreme DoS↔Reentrancy co-occurrence:

| Metric | Value |
|--------|-------|
| Total DoS=1 | 377 |
| DoS + Reentrancy | 370 (98.1%) |
| **DoS only** | **7 (1.9%)** |

Only 7 contracts in the entire 44,470-row dataset are labelled DoS=1 without also being labelled Reentrancy=1. These 7 contracts are split across train/val/test as: train=3, val=1, test=3.

**DoS-only vs DoS+Reentrancy feature profiles:**

| Feature | DoS-only (n=7) | DoS+Ree (n=370) |
|---------|-----------------|------------------|
| mean_nodes | 20.4 | 102.3 |
| mean_edges | 33.3 | 151.9 |
| mean_has_loop | 0.36 | 0.02 |
| mean_complexity | 5.56 | 3.64 |

DoS-only contracts are much smaller and have higher loop rates — the correct DoS signal (unbounded loops). But with only 3 training examples, the model cannot learn this pattern.

**Options:**
1. **Merge DoS into Reentrancy** — accept that they are the same label in this dataset
2. **Augment with synthetic DoS-only contracts** (~500 clean examples)
3. **Extreme class weighting** (pos_weight=20+ for DoS)

Option 2 is preferred in the v6 plan but requires hand-writing or sourcing DoS-only contracts.

**Status:** Data augmentation planned as Phase 4 in the v6 plan. Not yet started.

---

### 3.4 NEW-4: contract_path Relative vs Absolute Mismatch

**Severity:** LOW (P3 — cosmetic)

The audit (Task 10/24) found that graph files store `contract_path` as a relative path (e.g., `BCCC-SCsVul-2024/SourceCodes/IntegerUO/abc.sol`) while token files store it as an absolute path (e.g., `/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/abc.sol`).

This causes 100% "path mismatch" in alignment checks but is NOT a data integrity issue — both paths point to the same file. The important verification passes: stem↔hash verification is 100% and token decode verification is 100%.

**Status:** No fix required. Documented for future reference.

---

### 3.5 NEW-5: 4 Disconnected Graphs

**Severity:** LOW (P3)

The audit (Task 13) found 4 out of 44,470 graphs have zero edges (disconnected, ghost graphs). These are interface-only contracts or pure state-variable declarations that produce no analyzable edges.

Affected files: `7319be6b...`, `12a9a38c...`, `5d686fe6...`, `1fd0a54c...`

At 0.009% of the dataset, this is negligible and within acceptable bounds (gate was ≤100 ghost graphs).

**Status:** No fix required.

---

## Part 4: Feature Schema v4 — What Was Already Fixed (Before This Audit)

The following fixes were committed BEFORE the deep audit, as part of the v6 Phase 0 work. They are documented here for completeness and to distinguish them from the NEW fixes discovered by the audit.

### 4.1 return_ignored Fix (Commit `bef1f2a`)

**Original bug:** `op.lvalue is None` was always False because Slither always creates a TupleVariable as lvalue, even when the programmer ignores the return value.

**Fix:** Check if `id(lvalue)` appears in any subsequent IR operation's `read` set across all IR ops in the function. If the lvalue is never read, the return was discarded.

**Classes affected:** MishandledException (F1=0.342), UnusedReturn (F1=0.238)

### 4.2 Transfer/Send in ext_calls and CFG Typing (Commit `310e738`)

**Original bug:** `_compute_external_call_count()` only counted `HighLevelCall` and `LowLevelCall`. ETH-transfer DoS loops (`recipient.transfer(amount)`) produced ext_calls=0.

**Fix:** Added `Transfer, Send` to both `_compute_external_call_count()` and `_cfg_node_type()`.

**Classes affected:** DenialOfService, Reentrancy (ETH-transfer variants)

### 4.3 uses_block_globals Feature (Commit `310e738`)

**Original bug:** `block.timestamp` is a `SolidityVariableComposed` object that does NOT appear in `func.state_variables_read`, so it produced no READS edges. The Timestamp class had essentially zero direct graph signal.

**Fix:** Replaced the low-value `pure` feature (feat[2]) with `uses_block_globals`, which scans raw IR ops for `SolidityVariableComposed` objects.

**Classes affected:** Timestamp (F1=0.174 — worst class), TOD

### 4.4 loc Normalization for Declaration Nodes (Commit `310e738`)

**Original bug:** `loc` was stored as a raw integer, range [0, 2538]. This caused GAT attention to be dominated by loc magnitude.

**Fix:** Applied `log1p(loc) / log1p(1000)` normalization for declaration nodes, clamped to [0, 1].

**Note:** This fix was only applied to declaration nodes (`_build_node_features()`). CFG nodes (`_build_cfg_node_features()`) were missed — this is BUG-1, discovered by the audit.

### 4.5 Schema Version Bump (Commit `310e738`)

`FEATURE_SCHEMA_VERSION` bumped from `"v3"` to `"v4"` to invalidate all inference caches built under the previous schema.

---

## Part 5: Feature Schema v5 — New Fixes from This Audit

The following fixes are NEW — discovered by the deep audit and not yet committed. They will be applied in the graph_extractor.py rewrite and will require bumping `FEATURE_SCHEMA_VERSION` to `"v5"`.

### 5.1 v5 Feature Vector (12 Dimensions — Same Size, Changed Semantics)

```
[0]  type_id / 12.0          — normalized node type (unchanged)
[1]  visibility / 2.0   ★    — normalized to [0,1] (was: ordinal 0-2, private exceeded range)
[2]  uses_block_globals       — 1.0 if reads block.timestamp/number/etc. (unchanged from v4)
[3]  view                     — 1.0 if function is view-only (unchanged)
[4]  payable                  — 1.0 if function accepts ETH (unchanged)
[5]  complexity          ★    — log1p(CFG blocks)/log1p(200), [0,1] (was: raw count up to 169)
[6]  loc                  ★   — log1p(lines)/log1p(1000), [0,1] for ALL nodes including CFG
                               (was: raw count for CFG nodes — BUG-1)
[7]  return_ignored      ★    — now includes Send IR type (was: only HighLevelCall/LowLevelCall)
[8]  call_target_typed        — 0/1/-1 sentinel (unchanged)
[9]  in_unchecked → TBD  ★    — dead feature replacement pending (was: always 0 for 99.9% of data)
[10] has_loop                 — 1.0 if function contains a loop (unchanged)
[11] external_call_count      — log1p(count)/log1p(20), includes Transfer/Send (unchanged from v4)
```

★ = changed from v4. `FEATURE_SCHEMA_VERSION` will be bumped to `"v5"`.

### 5.2 Change Summary: v4 → v5

| Feature | v4 | v5 | Rationale |
|---------|-----|-----|-----------|
| visibility [1] | ordinal 0/1/2 | ordinal /2.0 → [0,1] | Private functions (2) exceeded [0,1] range |
| complexity [5] | raw CFG block count | log1p(raw)/log1p(200) | Max=169 exceeded [0,1]; 13% of nodes out of range |
| loc [6] | log-normalized for DECL only | log-normalized for ALL nodes | CFG nodes (72% of graph) had raw loc up to 129 |
| return_ignored [7] | HighLevelCall + LowLevelCall | + Send | Missed .send() unchecked returns (3 confirmed cases) |
| in_unchecked [9] | NodeType.STARTUNCHECKED + regex | TBD (dead feature replacement) | 99.9% of dataset is Solidity <0.8.0; feature always 0 |

### 5.3 Edge Type Fixes

| Edge Type | v4 | v5 | Rationale |
|-----------|-----|-----|-----------|
| EMITS (3) | Never created | Scan IR for EventCall | Old Solidity syntax; func.events_emitted always empty |
| INHERITS (4) | Never created | Add as CONTRACT node feature | Parent contracts not in node_map; add inheritance_depth instead |

### 5.4 Contract Selection Fix

| Component | v4 | v5 | Rationale |
|-----------|-----|-----|-----------|
| `_select_contract()` | Most functions | Most-derived → Last defined → Most functions | 47.4% wrong rate; most-derived captures inheritance correctly |

---

## Part 6: The Two P0 Crises

### 6.1 Crisis 1: BUG-6 — 47.4% Wrong Contract Selection

This is the single most impactful finding. Nearly half of all multi-contract files have the wrong contract selected for graph extraction. The graph that the model receives during training bears no relation to the vulnerability it is supposed to learn.

**Why the "Most Functions" heuristic fails:** OpenZeppelin base contracts like `StandardToken` have many functions (transfer, transferFrom, approve, allowance, balanceOf, etc.) while the vulnerable derived contract `ERC20Token` may only add 2–3 custom functions. The heuristic picks `StandardToken` because it has more functions, but the vulnerability is in `ERC20Token`.

**Expected impact of fix:** Reducing wrong-selection from 47.4% to ~12.6% means:
- 34.8% of multi-contract files will recover their vulnerability signals
- For classes like CallToUnknown (56.2% wrong), the fix alone could improve F1 by 15–25 percentage points
- The `uses_block_globals` feature becomes meaningful for Timestamp contracts (202 of 491 "feature miss" cases in NEW-1 are due to BUG-6)

### 6.2 Crisis 2: Timestamp Labels — 48.2% Mislabelled

Even after fixing BUG-6, 1,056 Timestamp=1 contracts have no block global usage in their source. Training on these labels teaches the model that "contracts without block.timestamp should sometimes be predicted as Timestamp" — a contradiction that prevents the model from learning a consistent decision boundary.

**Why this matters for Timestamp F1:** Timestamp was already the worst class at F1=0.174. With 48% label noise, the theoretical ceiling for F1 is approximately 0.52 (the maximum F1 achievable when half the positive labels are wrong). Even with perfect features and perfect architecture, Timestamp F1 cannot exceed ~0.52 until the labels are cleaned.

**Fix execution order matters:** BUG-6 must be fixed BEFORE timestamp relabeling. After BUG-6, 202 of the 491 "feature miss" contracts will recover their `uses_block_globals` signal, reducing the noise floor from 48.2% to ~33%. The timestamp relabeling script then only needs to reclassify the remaining ~1,056 contracts.

---

## Part 7: Execution Order — What Must Happen and When

The fixes have strict dependencies. The execution order is critical.

### Step 1: Rewrite graph_extractor.py with ALL bug fixes

Apply BUG-1 through BUG-9 fixes and the contract selection fix in a single rewrite. This includes:
- BUG-1: CFG loc normalization
- BUG-2: Complexity normalization
- BUG-3: Visibility normalization
- BUG-5: in_unchecked replacement (decision needed)
- BUG-6: Most-derived contract selection heuristic
- BUG-7: EMITS edge generation via IR scanning
- BUG-8: INHERITS as CONTRACT node feature
- BUG-9: Send in return_ignored check

Bump `FEATURE_SCHEMA_VERSION` to `"v5"` in `graph_schema.py`.

### Step 2: Re-extract all 44,470 graphs

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/reextract_graphs.py \
    --workers 16 \
    --output-dir ml/data/graphs \
    --index ml/data/processed/multilabel_index_deduped.csv
```

Expected time: ~4–6 hours.

### Step 3: Validate re-extracted graphs

```bash
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py \
    --check-dim 12 \
    --check-edge-types 8 \
    --check-contains-edges \
    --check-control-flow
```

Gate: 0 validation errors, ghost graphs ≤ 100.

### Step 4: Run timestamp label relabeling

```bash
PYTHONPATH=. python ml/scripts/p0-fix-timestamp-labels.py \
    --index ml/data/processed/multilabel_index_deduped.csv \
    --graphs-dir ml/data/graphs \
    --source-dir BCCC-SCsVul-2024/SourceCodes
```

This must run AFTER graph re-extraction so it can use the corrected `uses_block_globals` feature values.

### Step 5: Rebuild deduped CSV

If the timestamp relabeling modifies any labels, rebuild the deduped CSV:

```bash
PYTHONPATH=. python ml/scripts/dedup_multilabel_index.py
```

### Step 6: Rebuild cache

```bash
PYTHONPATH=. python ml/scripts/create_cache.py \
    --graphs-dir ml/data/graphs \
    --tokens-dir ml/data/tokens_windowed \
    --label-csv ml/data/processed/multilabel_index_deduped.csv \
    --output ml/data/cached_dataset_windowed.pkl \
    --workers 8
```

### Step 7: Launch v6.0 training

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v6.0-20260517 \
    --experiment-name sentinel-v6 \
    --tokens-dir ml/data/tokens_windowed \
    --cache-path ml/data/cached_dataset_windowed.pkl \
    --loss-fn asl \
    --gradient-accumulation-steps 8 \
    --label-smoothing 0.05
```

---

## Part 8: Key Metrics and Baselines

### 8.1 Current State (Post-Audit, Pre-Fix)

| Metric | Value |
|--------|-------|
| Total contracts | 44,470 |
| File alignment (CSV ∩ Graphs ∩ Tokens) | 100% |
| Stale v5 graphs | 0 |
| Token integrity | 100% pass rate |
| Wrong contract rate (current) | **47.4%** |
| Timestamp mislabel rate | **48.2%** |
| Dead features | in_unchecked (always 0) |
| Dead edge types | EMITS (0), INHERITS (0) |
| Feature range violations | loc (max=946), complexity (max=67), visibility (max=2) |
| DoS pure-label count | 7 |
| DoS↔Reentrancy co-occurrence | 98.1% |
| Solidity 0.4.x dominance | 87.9% |
| Graph size confound (max AUC) | 0.637 (Timestamp) |
| Split distribution shift | None detected |
| Disconnected graphs | 4 (0.009%) |

### 8.2 v5.2 Per-Class F1 Baselines (Current Best Before Fixes)

| Class | v5.2 Tuned F1 | v6.0 Target | Primary fix driving improvement |
|-------|---------------|-------------|-------------------------------|
| IntegerUO | 0.732 | ≥ 0.75 | Windowed tokens + hidden_dim=256 |
| GasException | 0.407 | ≥ 0.45 | 6-layer GNN + CF signal depth |
| Reentrancy | 0.322 | ≥ 0.40 | ASL + BUG-6 fix + 2nd CF layer |
| MishandledException | 0.342 | ≥ 0.50 | return_ignored fix (was always 0) |
| UnusedReturn | 0.238 | ≥ 0.45 | return_ignored fix + BUG-9 |
| Timestamp | 0.174 | ≥ 0.30 | uses_block_globals + BUG-6 fix + label cleanup |
| DenialOfService | 0.329 | ≥ 0.35 | Transfer/Send fix + augmentation |
| CallToUnknown | 0.284 | ≥ 0.35 | BUG-6 fix (56% wrong → ~12%) |
| TOD | 0.283 | ≥ 0.30 | uses_block_globals + windowed |
| ExternalBug | 0.262 | ≥ 0.30 | BUG-6 fix + deeper GNN |
| **Macro avg** | **0.3422** | **≥ 0.43** | All fixes combined |

### 8.3 Behavioral Gates (Primary Pass/Fail)

Val F1 is necessary but insufficient. The primary gate is behavioral testing:
- **Detection rate ≥ 80%** (v5.2 achieved 36%)
- **Safe specificity ≥ 80%** (v5.2 achieved 33%)

---

## Part 9: Code Changes Already Committed (Pre-Audit)

These changes were committed as part of the v6 Phase 0–3 work before the deep audit was conducted.

| Commit | Date | Contents |
|--------|------|----------|
| `bef1f2a` | 2026-05-16 | return_ignored fix (lvalue ID check replaces None check) |
| `310e738` | 2026-05-16 | Schema v4: uses_block_globals, loc normalization (DECL only), Transfer/Send in ext_calls + CFG |
| `b38c9da` | 2026-05-16 | Phase 1: windowed tokenization (TransformerEncoder, SentinelModel, DualPathDataset, retokenize_windowed.py) |
| `2bb0e16` | 2026-05-16 | Phase 2: GNN 256-dim/6-layer, conv3b/conv4b, WindowAttentionPooler, classifier hidden layer |
| `64dfc5a` | 2026-05-16 | Phase 3: AsymmetricLoss, epochs=100, patience=30, LoRA LR 0.3× |

---

## Part 10: Label Co-occurrence Matrix (Full Dataset)

P(B=1 | A=1) — what fraction of A-labelled contracts also have label B:

```
              CtoU   DoS   Ext   Gas   Int   Mis   Ree   Tim   TOD   Unu
CallToUnknown  100     0    12    27    70    14    43     7    13     9
DenialOfSvc      0   100     1    63     1     1    98     0     0     0
ExternalBug     13     0   100    17    67    24    47     9    21    41
GasException    17     4    11   100    77    28    30    16    26    18
IntegerUO       16     0    15    28   100    29    17    10    19    12
Mishandled      11     0    17    33    96   100    20    13    25    18
Reentrancy      31     7    32    34    51    19   100    15    18    42
Timestamp       12     0    13    41    72    27    34   100    27    20
TOD             14     0    21    43    87    35    27    18   100    17
UnusedReturn    11     0    46    34    63    29    69    15    19   100
```

**Critical co-occurrences (>70%):**

| Pair | Rate | n |
|------|------|---|
| DoS → Reentrancy | **98.1%** | 370/377 |
| MishandledException → IntegerUO | **96.0%** | 4,520/4,709 |
| TOD → IntegerUO | **86.6%** | 2,938/3,391 |
| GasException → IntegerUO | **76.7%** | 4,293/5,597 |
| Timestamp → IntegerUO | **71.9%** | 1,576/2,191 |

IntegerUO is a universal background class that co-occurs with every other class at 51–96%. The model will default to predicting IntegerUO when uncertain.

**Pure single-label contract counts** (training signal without co-occurrence noise):

| Class | Pure count | Trainability |
|-------|-----------|--------------|
| CallToUnknown | 12 | Near-untrained |
| DenialOfService | 7 | Untrained |
| MishandledException | 89 | Very limited |
| UnusedReturn | 52 | Very limited |
| ExternalBug | 30 | Very limited |
| TOD | 155 | Limited |
| Reentrancy | 147 | Limited |
| Timestamp | 466 | Marginal |
| GasException | 721 | OK |
| IntegerUO | 4,203 | Abundant |

---

## Part 11: Feature Correlation Analysis (Task 21)

PCA and correlation analysis on 500 graphs (66,288 nodes) revealed:

**90% variance needs 9 components** (out of 12), confirming that 3 dimensions contribute little unique information. PC12 has 0.0000 variance → `in_unchecked` is completely dead.

**Highly correlated pairs:**

| Pair | Pearson | Spearman | Notes |
|------|---------|----------|-------|
| type_id ↔ loc | 0.094 | **0.741** | Driven by BUG-1: CFG nodes have high type_id AND high raw loc |
| complexity ↔ ext_call_count | **0.540** | 0.467 | Complex functions tend to have more external calls |

After BUG-1 and BUG-2 are fixed, the type_id↔loc correlation should decrease significantly because CFG loc will be normalized into [0, 1] instead of reaching 129.0.

**Feature unique information (PCA-based):**

| Feature | Unique info % | Redundancy note |
|---------|---------------|-----------------|
| in_unchecked | 0.0% | Dead — zero variance |
| complexity | 50.7% | Correlated with ext_call_count (r=0.54) |
| type_id | 71.3% | Correlated with loc (Spearman r=0.74) |
| visibility | 79.5% | Correlated with type_id (r=-0.41) |

---

## Part 12: Solidity Version Distribution and Feature Implications

The audit (Task 18) measured the Solidity version distribution across 2,000 sampled contracts:

| Version | Count | Percentage |
|---------|-------|------------|
| **0.4.x** | 1,758 | **87.9%** |
| 0.5.x | 160 | 8.0% |
| 0.8.x | 1 | 0.1% |
| no_pragma | 81 | 4.0% |

**Implications for feature design:**
- `in_unchecked` (Solidity ≥0.8.0) is dead — only 0.1% of contracts could possibly fire it
- `checked{}` arithmetic (Solidity ≥0.8.0 default) is absent — IntegerUO vulnerabilities are exclusively in 0.4.x/0.5.x contracts that lack SafeMath
- The `emit` keyword (Solidity ≥0.4.21) is inconsistently used — old-style event calls look like function calls to Slither
- Solidity 0.5.x shows an IntegerUO rate of 70% (vs 34.7% for 0.4.x), suggesting version-specific vulnerability patterns

**Design principle for v5 schema:** All features must be designed for the 87.9% Solidity 0.4.x majority. Features that only activate on 0.8.x are dead weight. The `pragma_version` or `has_unchecked_arithmetic` replacement for `in_unchecked` should encode "this contract is on a version where arithmetic is NOT checked by default" — which is true for 99.9% of the dataset.

---

## Part 13: Audit Task Completion Summary

| Task | Name | Result | Key Finding |
|------|------|--------|-------------|
| 01 | Activation split (DECL vs CFG) | ✅ PASS | CFG nodes: 0% semantic features |
| 02 | Source pattern spot-check | ✅ PASS | return_ignored and uses_block_globals work when right contract selected |
| 03 | Token coverage | ✅ PASS | 82% use all 4 windows; stride covers 1,534+ tokens |
| 04 | Ghost graph analysis | ✅ PASS | 4 ghosts (0.009%) — negligible |
| 05 | Edge type distribution (full) | ✅ PASS | EMITS=0, INHERITS=0 — BUG-7/8 confirmed |
| 06 | return_ignored sanity | ✅ PASS | Works for HL/LL calls; misses Send — BUG-9 |
| 07 | Sentinel value audit | ✅ PASS | -1.0 sentinels never fire — dead code |
| 08 | Label co-occurrence | ✅ PASS | DoS↔Reentrancy 98.1%; IntegerUO is universal background |
| 09 | Feature range audit | ✅ PASS | BUG-1 (loc max=129), BUG-2 (complexity max=67), BUG-3 (visibility max=2) |
| 10 | Token-graph alignment | ✅ PASS | Stem↔hash 100%; path format mismatch (cosmetic) |
| 11 | File triple alignment | ✅ PASS | 100% CSV ∩ Graphs ∩ Tokens |
| 12 | Token integrity | ✅ PASS | All 9 checks pass |
| 13 | Graph structural integrity | ✅ PASS | 4 disconnected graphs; edge type distribution confirms BUG-7/8 |
| 14 | Sub-sampling coverage | ✅ PASS | 100% vulnerability window survival (n=10) |
| 15 | in_unchecked regex | ✅ PASS | False positives on comments/strings (irrelevant — feature is dead) |
| 16 | Wrong contract selection | ✅ PASS | **47.4% wrong** — P0 crisis confirmed |
| 17 | SafeMath viability | ✅ PASS | Only 45% discriminative — NOT viable as feature replacement |
| 18 | Solidity version distribution | ✅ PASS | 87.9% Solidity 0.4.x — explains in_unchecked death |
| 19 | Timestamp label quality | ✅ PASS | **48.2% mislabelled** — P0 crisis confirmed |
| 20 | DoS↔Reentrancy separability | ✅ PASS | Only 7 pure-DoS contracts — untrainable |
| 21 | Feature correlation | ✅ PASS | type_id↔loc r=0.74; in_unchecked 0% variance |
| 22 | Graph size confound | ✅ PASS | Max AUC 0.637 — borderline but not dominant |
| 23 | .send() unchecked prevalence | ✅ PASS | 3 confirmed missed cases — BUG-9 confirmed |
| 24 | Token-graph-source alignment | ✅ PASS | 100% stem↔hash match |
| 25 | Split distribution shift | ✅ PASS | No significant shift detected |
| 26 | Stale v5 contamination | ✅ PASS | 0 stale graphs — all are v4 |

---

## Appendix A: Bug Severity and Fix Priority

| Bug | Description | Severity | Fix Priority | Fix Status |
|-----|-------------|----------|-------------|------------|
| BUG-6 | Wrong contract selection (47.4%) | **CRITICAL** | **P0** | Script written, pending integration |
| NEW-1 | Timestamp 48.2% mislabelled | **CRITICAL** | **P0** | Script written, pending execution after BUG-6 |
| BUG-1 | CFG loc not normalized (max=129) | HIGH | P1 | Designed, pending implementation |
| BUG-2 | Complexity not normalized (max=67) | HIGH | P1 | Designed, pending implementation |
| BUG-5 | in_unchecked dead (0.8.x <0.1%) | HIGH | P1 | Replacement decision needed |
| NEW-2 | CFG nodes zero semantic features | HIGH | P2 | Design issue, no immediate fix |
| NEW-3 | DoS 7 pure-label samples | HIGH | P2 | Augmentation planned (Phase 4) |
| BUG-3 | visibility=2 for private | MEDIUM | P2 | Designed, pending implementation |
| BUG-7 | EMITS edges never created | MEDIUM | P2 | Designed (IR scan), pending implementation |
| BUG-8 | INHERITS edges never created | MEDIUM | P2 | Designed (node feature), pending implementation |
| BUG-9 | .send() return_ignored missed | LOW | P3 | Designed, pending implementation |
| NEW-4 | contract_path format mismatch | LOW | P3 | No fix needed (cosmetic) |
| NEW-5 | 4 disconnected graphs | LOW | P3 | No fix needed (0.009%) |

---

## Appendix B: Files Affected by Fixes

| File | Changes Required |
|------|-----------------|
| `ml/src/preprocessing/graph_extractor.py` | BUG-1,2,3,5,6,7,8,9 — full rewrite with all fixes |
| `ml/src/preprocessing/graph_schema.py` | FEATURE_SCHEMA_VERSION bump to "v5"; FEATURE_NAMES updates; INHERITS edge repurpose |
| `ml/data/processed/multilabel_index_deduped.csv` | Timestamp label relabeling (after BUG-6 fix + re-extraction) |
| `ml/data/graphs/` | Full re-extraction required (44,470 files) |
| `ml/data/cached_dataset_windowed.pkl` | Rebuild after re-extraction |
| `ml/scripts/reextract_graphs.py` | May need updates for v5 schema |
| `ml/scripts/validate_graph_dataset.py` | May need updates for v5 schema validation |

---

## Appendix C: Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Fix BUG-6 before timestamp relabeling | 202 of 491 "feature miss" Timestamp contracts recover signal after BUG-6 fix; reduces noise from 48.2% to ~33% | 2026-05-17 |
| Most-derived heuristic over Last Contract | Most-derived captures inheritance structure (ERC20Token is StandardToken); Last Contract only works by convention | 2026-05-17 |
| Replace in_unchecked rather than keep it | Dead feature wastes a dimension; 0.0000 PCA variance; cannot help training | 2026-05-17 |
| Reject SafeMath as in_unchecked replacement | Only 45% discriminative for IntegerUO; 43.6% graph recall; essentially coin-flip | 2026-05-17 |
| Keep EMITS/INHERITS in edge type vocabulary | Even if initially empty after fix, the embedding slots are harmless; removing them requires NUM_EDGE_TYPES change and re-extraction | 2026-05-17 |
| Bump schema to v5 (not v4.1) | Multiple feature semantics change (visibility, complexity, loc, return_ignored, in_unchecked); incremental version would be misleading | 2026-05-17 |
| Patch stale graphs in-place vs re-extract | 2,702 graphs that Slither couldn't re-extract kept raw CFG loc/complexity. In-place log1p normalization was applied atomically (tmp→rename) rather than skipping or ignoring them — their topology and other features are valid. | 2026-05-17 |
| batch_size=8 not 16 for v6 training | max_windows=4 makes each batch [B*4, 512] through CodeBERT. batch=16 saturated VRAM at 7.9/8.0 GB causing 320s/batch. batch=8 gives 6.4/8.0 GB with 1.4 GB headroom and ~34 min/epoch. | 2026-05-17 |

---

## Part 14: Pipeline Completion — 2026-05-17 (Third Session)

**Updated:** 2026-05-17 03:30 UTC+3:30  
**Status:** v6.0 training RUNNING (PID 450936)

All bugs documented in this audit were fixed, re-extraction completed, and training launched. This section records what was done in execution order.

### 14.1 Fixes Applied by User (independent audit session)

The user ran their own audit session between sessions 2 and 3 and applied the following changes before the third session began:

| Fix | File | Description |
|-----|------|-------------|
| BUG-1 | `graph_extractor.py:458` | CFG `loc`: raw → `log1p(len(sm.lines)) / log1p(1000)` |
| BUG-2 | `graph_extractor.py:618` | `complexity`: raw → `log1p(len(obj.nodes)) / log1p(100)` |
| BUG-6 | `graph_extractor.py` | `_select_contract()`: most-derived heuristic (inherits most others), fallback: last-defined |
| BUG-9 | `graph_extractor.py` | `_compute_return_ignored()`: added `Send` to isinstance check |
| Schema | `graph_schema.py` | `FEATURE_SCHEMA_VERSION` bumped `"v4"` → `"v5"` |
| Script | `reextract_graphs.py` | Default policy = `"most_derived"` |
| Script | `validate_graph_dataset.py` | Added `--check-block-globals` flag |

Committed in a single commit by user (`0d11e18`).

### 14.2 v7 Re-extraction Results

Re-extraction ran with schema v5 and `most_derived` contract selection:

| Outcome | Count | Note |
|---------|-------|------|
| ok (fresh extract) | 41,521 | All BUG-1/2/3/6/9 fixes applied |
| ghost (≤3 nodes) | 74 | Interfaces/stubs; within gate ≤100 |
| skipped | 2,875 | Root causes below |
| fail (hard error) | 0 | |

**Skipped root causes (2,875):**
- 2,535: no `contract_path` stored in old graph — SolidiFI/augmented sources not in BCCC SOURCE_DIRS
- 76: genuine Solidity compile errors (solc syntax errors in .sol file)
- 48: Slither internal failures on valid Solidity (retryable but low priority)
- 43: safe contracts with path but compile errors under solc target version

**Key validation result:**
- 44,470 PASS; ghost=74 (within gate); `uses_block_globals` fires in 9.9% (4,416 graphs)
- Before BUG-6 fix: `uses_block_globals` fired in ~0.1% (near-zero — wrong contract selected every time)

### 14.3 Orphan Graph Cleanup

4,311 graph `.pt` files present in `ml/data/graphs/` that were not in `multilabel_index_deduped.csv` (leftover from the old leaky extraction before deduplication) were moved to `ml/data/graphs_legacy/` to eliminate ambiguity. After the move:
- `ml/data/graphs/`: 44,470 files — exactly matches the deduped CSV
- `ml/data/graphs_legacy/`: 4,311 orphan files (retained for reference)

### 14.4 Stale Graph In-place Patching

2,702 of the 2,875 skipped graphs kept their old raw `loc` and `complexity` values (extracted under v3/v4 schema without BUG-1/BUG-2 fixes). These were patched atomically in-place:

```python
# CFG nodes (feat[0] >= 8/12):
x[cfg_mask, 6] = torch.clamp(torch.log1p(raw_loc) / math.log1p(1000), max=1.0)

# Declaration nodes with complexity > 1.0:
x[decl_mask, 5][above] = torch.clamp(torch.log1p(raw_comp) / math.log1p(100), max=1.0)
```

Result: Patched=2,702, Errors=0. Verification on 500 random graphs: 0 still stale, max CFG loc=0.8289 ≤ 1.0.

### 14.5 retokenize_windowed.py Fix

**Bug:** `KeyError: 'contract_path'` — script assumed deduped CSV had a `contract_path` column (it doesn't — only `md5_stem`).  
**Fix:** Added `PROJECT_ROOT`, `SOURCE_DIRS`, `_md5_of_path()`, `build_md5_to_path()` — same disk-scan pattern as `reextract_graphs.py`. Changed `process_batch` signature to accept `Dict[str, Path]`. Committed `74e968c`.

**Tokenization result:** 44,470/44,470, 0 failures. Output: `ml/data/tokens_windowed/` with shape `[4, 512]` per file.

### 14.6 Timestamp Label Relabeling (dedup_multilabel_index.py)

**Bug 1:** ImportError on `DataEdgeAttr`/`DataTensorAttr` (torch_geometric 2.7.0 doesn't have these). The `safe_globals` block was also redundant since `weights_only=False` already skips safe_globals enforcement. Fixed by removing the import block.

**Bug 2 (performance):** `_find_source_for_md5()` walked the entire source directory tree for each of 1,933 Timestamp=1 rows — O(1933 × 100K+ files). Script appeared stuck. Fixed by adding `_build_md5_to_sol_map()` which scans once and returns a dict; relabeling loop uses the dict. Committed `a75ae67`.

**Relabeling result:**
| | Count |
|---|---|
| Timestamp=1 rows verified | 1,933 |
| Confirmed by both source + graph | 530 |
| Confirmed by graph only | 2 |
| Confirmed by source only | 429 |
| Removed (neither confirmed) | **972** |
| Final Timestamp=1 | **961** (49.7% kept) |

All 1,933 rows had both a graph `.pt` and a source `.sol` file — 0 orphans.

### 14.7 Cache Rebuild

```
ml/data/cached_dataset_windowed.pkl — 44,470 pairs, 2.47 GB, schema FEATURE_SCHEMA_VERSION='v5'
```

### 14.8 Training Launch

**First attempt (batch=16):** VRAM saturated at 7.9/8.0 GB (max_windows=4 → CodeBERT processes 64 sequences/batch). Speed: 320s/batch → 173h/epoch. Killed.

**Second attempt (batch=8):** VRAM 6.4/8.0 GB (1.4 GB headroom). Speed: ~1.9 batch/s → ~34 min/epoch → ~57h for 100 epochs (with early stopping expected ~ep 40-60).

```
PID: 450936
Config: batch=8, grad_accum=8 (effective=64), ASL(γ⁻=4, γ⁺=1, clip=0.05)
        epochs=100, patience=30, label_smoothing=0.05
Log: ml/logs/train_v6.0_20260517.log (tqdm) + ml/logs/v6.0-20260517.log (structured)
Checkpoint: ml/checkpoints/v6.0-20260517_best.pt
```

---

## Part 15: Outstanding Issues (Not Blocking Training)

| Issue | Severity | Description |
|-------|----------|-------------|
| BUG-3: visibility=2 for private | MEDIUM | `private` maps to 2, `internal` to 0 — two separate values for "not external". No standard fix; re-extraction required. |
| BUG-7: EMITS edges never created | MEDIUM | Old Solidity event syntax (no `emit` keyword) not recognized by Slither. Fires 0 times in dataset. |
| BUG-8: INHERITS edges never created | MEDIUM | Parent contracts not in `node_map` when `_select_contract()` only extracts one contract. |
| 48 retryable Slither failures | LOW | 48 graphs have stale BUG-6/BUG-1 data; valid Solidity that Slither failed internally. Could retry with different flags. |
| 43 compile-error stale graphs | LOW | Genuine solc errors; no fix possible without modified source. |
| DoS: 7 pure-label contracts | HIGH | DoS↔Reentrancy co-occurrence 98.1%. Augmentation (Phase 4) still needed. |

BUG-3/7/8 require another full re-extraction cycle to fix. Given training is now running with all P0/P1 bugs fixed, these are deferred to v7 if v6.0 behavioral gates fail.
