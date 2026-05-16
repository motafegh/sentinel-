# SENTINEL — Complete Analysis Findings
**Date:** 2026-05-16
**Scope:** All analysis performed during the 2026-05-16 audit session, covering data pipeline,
feature schema, model architecture, token processing, co-occurrence, and per-class signal analysis.

This document records every significant finding from the session that informs the v6 design.

---

## 1. Data Pipeline Audit

### 1.1 BCCC Dataset Structure

The Berkeley-Columbia Contract Corpus (BCCC) stores contracts in a directory structure where
each vulnerability class has its own subdirectory. A contract vulnerable to multiple classes
appears as a separate file copy in each relevant directory. Key facts:

- **BCCC raw rows:** 111,898 (one row per class+file combination)
- **Unique contracts:** ~68,000 across 12 vulnerability directories
- **File format:** `.sol` files named by SHA256 hash of content
- **Label assignment:** all contracts in directory `reentrancy/` get label Reentrancy=1, etc.

Two hash systems exist in the pipeline:
- **BCCC SHA256:** content hash, used as the `.sol` filename in BCCC
- **Path-based MD5:** `md5(contract_path)`, used as the `.pt` filename for graph and token cache

This dual-hash system caused the original leakage problem: the same .sol content in two different
directories gets two different path-MD5 values → two separate `.pt` files → assigned to two
different splits with ORed labels.

### 1.2 Deduplication

After deduplication by content hash:
- **Original:** 68,523 rows in `multilabel_index.csv`
- **Deduped:** 44,420 rows in `multilabel_index_deduped.csv`
- **Reduction:** 35.2% (24,103 duplicate entries removed)
- **Cross-split leakage detected:** 7,630 content groups spanning multiple splits (34.9% of groups)
- **Leakage mechanism:** same content appeared in train and val/test with different labels depending
  on which directory copy was assigned to which split during stratified splitting

The deduped splits (`ml/data/splits/deduped/`):
- train: 31,092 rows (+ 50 CEI augmentation = 31,142 in v5.3)
- val: 6,661 rows
- test: 6,667 rows

Stratified by all 10 class columns. All 10 classes represented in val and test splits.

### 1.3 Graph Extraction Status

**Total graphs on disk:** 44,420 (exactly matching deduped row count; orphan duplicates deleted)

**280 stale graphs (0.6%):** These correspond to contracts with strict version pragmas such as
`pragma solidity 0.4.15;` (exact version, not range). The re-extraction script uses solc 0.4.26
as the 0.4.x representative, which rejects exact-version pragmas if the version does not match.
The original extraction used exact-version matching. These 280 graphs were extracted with the
original pipeline and still carry old-format features. They represent genuinely uncompilable
contracts under the current strategy — 0.4.x contracts with 0.5.x syntax being the dominant
subcase. Impact: 0.6% of dataset, below the 5% concern threshold.

**66 ghost graphs (0.1%):** Graph `.pt` files that exist on disk but are not referenced in the
deduped CSV. Post-extraction residual artifacts. Filtered out by dataset.__getitem__() path
check. Not a training concern.

**44,140 freshly extracted graphs:** extracted during v5.1 Phase 0 re-extraction with all fixes
applied (feature schema v3, select_contract fix, etc.).

### 1.4 contract_path Bridge Bug (B1 — FIXED)

**Bug:** `reextract_graphs.py` was using `contract_path` from the CSV directly without joining
to the actual BCCC directory. On some rows, the stored `contract_path` referenced the original
BCCC subdirectory path (e.g., `data/BCCC/reentrancy/0xABC.sol`) but the file had been moved
or the base path was wrong.

**Fix:** Applied in an earlier session (commit referenced in memory). The extractor now
reconstructs the path from BCCC base directory + canonical SHA256 filename.

**Impact of unfixed bug:** ~12% of graphs were extracted from the wrong source file, receiving
mismatched labels or incorrect AST structure.

### 1.5 Token Pipeline Status

- **Token `.pt` files on disk:** ~68K (includes orphans from original dataset)
- **Token files actually used by dataset:** 44,420 canonical-MD5 files (referenced by deduped CSV)
- **Format (current):** `input_ids: [512]`, `attention_mask: [512]` (single-window, 512 tokens)
- **Cache:** `ml/data/cached_dataset_deduped.pkl` — built and verified
- **Stale cache:** `ml/data/cached_dataset.pkl` — DO NOT USE (references original 68K dataset)

---

## 2. Co-occurrence Problem

### 2.1 Source of Co-occurrence

Because BCCC stores one .sol file per class directory, and because contracts often appear in
multiple directories (vulnerability co-location), the deduped label for any given contract is the
OR of all class labels found across all directories that contain that contract's SHA256. The result
is systematic co-occurrence between classes that share contracts.

### 2.2 Measured Co-occurrence Rates

| Primary Class | Co-occurring Class | Rate |
|---------------|-------------------|------|
| DenialOfService | Reentrancy | 99% |
| MishandledException | IntegerUO | 96% |
| TOD | IntegerUO | 86% |
| UnusedReturn | Reentrancy | 69% |

**DoS→Reentrancy at 99%** is the most severe case. Of all training samples labeled DenialOfService,
99% are also labeled Reentrancy. This means:
- The model cannot learn "DoS-specific" patterns because almost every DoS example is also a
  Reentrancy example.
- Any gradient signal for "contract has DoS" also updates Reentrancy weights with equal force.
- The model learns: "external call pattern present → Reentrancy AND DoS" as a co-occurrent pair.

### 2.3 Training Impact

**Reentrancy collapse:** The combination of 99% DoS co-occurrence, 69% UnusedReturn co-occurrence,
and 2.82× pos_weight in v5.2 created extreme pressure on the Reentrancy prediction head. Any
contract with an external call in any context would push the model toward Reentrancy=1.

**Safe-contract false positives:** Many DeFi contracts are safe (proper CEI ordering) but have
external calls. The model was trained on data where external calls almost always co-occurred with
Reentrancy. It generalized this incorrectly, firing Reentrancy on safe contracts with external calls.

**Implication for v6:** Must augment training data with clean single-label DoS and Reentrancy
contracts that break the 99% co-occurrence signal. The model needs to see:
- "DoS without Reentrancy" (gas-grief loops, block-gas-limit exploits without reentrancy)
- "Reentrancy without DoS" (classical reentrancy without gas issues)
- "Safe with external calls" (proper CEI-ordered contracts with multiple external calls)

---

## 3. Data Starvation Analysis

### 3.1 Training Sample Counts (deduped)

| Class | Training Positives | % of Train Set |
|-------|--------------------|----------------|
| IntegerUO | 10,886 | 35.0% |
| GasException | 3,918 | 12.6% |
| Reentrancy | 3,500 | 11.2% |
| MishandledException | 3,296 | 10.6% |
| ExternalBug | 2,383 | 7.7% |
| CallToUnknown | 2,527 | 8.1% |
| TOD | 2,374 | 7.6% |
| UnusedReturn | 2,126 | 6.8% |
| Timestamp | 1,493 | 4.8% |
| DenialOfService | **257** | **0.8%** |

**Safe-only contracts (no positive label):** 17,141 training rows = 55.0%

**Critical starvation:**
- **DenialOfService: 257 positives.** At effective batch size 32 (batch=8, grad_accum=4), there
  is on average 0.26 DoS-positive examples per batch step. The model sees one DoS example every
  ~4 batches. With 1,942 steps per epoch, that is 257 DoS positives per epoch — comparable to
  a few hundred examples total. Gradient updates for DoS are extremely noisy.
- **Timestamp: 1,493 positives.** Starvation, but less severe than DoS.

### 3.2 IntegerUO Dominance

IntegerUO has 10,886 positives = 35% of all training rows. It is by far the dominant class.
With sigmoid multi-label outputs, the IntegerUO head provides the largest gradient contribution.
Other classes may be pulled toward IntegerUO-like patterns. This is the "gradient gravity" problem
in multi-label learning.

### 3.3 Effective Minority Training Rates (v5.2 config)

With gradient_accumulation_steps=4, effective batch=32:
- Steps per epoch: ~973 (31,142 / 32)
- DoS positives per step: 0.26 (≪1, extremely sparse)
- Timestamp positives per step: 1.53 (borderline)

A pos_weight of 10.96 for DoS means each DoS positive contributes 10.96× gradient for the DoS
head. But with only 0.26 positives per step on average, most steps contribute zero DoS gradient,
and the 10.96× weight applies only when a positive happens to appear.

---

## 4. Feature Schema Bugs and Limitations

### 4.1 Bug B1: return_ignored Always 0.0 — FIXED (commit bef1f2a)

**Location:** `ml/src/preprocessing/graph_extractor.py`, function `_compute_node_features()`

**Incorrect code (before fix):**
```python
return_ignored = any(op.lvalue is None for op in node.irs)
```

**Why wrong:** In Slither's IR, every assignment-type operation has `op.lvalue` set to a temporary
variable (e.g., `TMP_1`). The `lvalue` is never `None` for operations that produce return values —
even if the return value is subsequently never read. The check `op.lvalue is None` always evaluates
to False, so `return_ignored` was always 0.0 for every node.

**Correct logic:** Check if the lvalue of a call operation appears in any subsequent operation's
`reads` set within the same function. If the lvalue ID is absent from all subsequent reads, the
return value is unused.

**Impact:** MishandledException and UnusedReturn both depend on this feature as a primary signal.
With return_ignored=0 always, these classes had essentially zero function-level graph signal.
Their learned representations relied entirely on structural/co-occurrence features. This explains
MishandledException F1=0.342 and UnusedReturn F1=0.238 in v5.2.

**Expected improvement after fix:** MishandledException target ≥ 0.50, UnusedReturn ≥ 0.45.

### 4.2 Bug B2: node_metadata TYPE Shows STATE_VAR for All Nodes — LOW PRIORITY

**Location:** `ml/src/preprocessing/graph_extractor.py`, function `_build_node_metadata()`

**Issue:** The metadata dict stored `type: "STATE_VAR"` for all node types when inspecting node
attributes. This only affects debugging/visualization tools that use the metadata dict, not the
actual node feature vectors used during training.

**Impact:** Low. Does not affect training or inference. Affects only interpretability tools.

### 4.3 Dark Spot: block.timestamp Invisible in Graph

**SolidityVariableComposed** (block.timestamp, block.number, block.difficulty, msg.sender,
msg.value, etc.) is Slither's IR class for built-in Solidity global variables. These are NOT
included in `state_variables_read` — they are a separate category.

**Current extractor:** only iterates `state_variables_read` for READS edges. Therefore:
- `block.timestamp` → no READS edge generated
- `block.number` → no READS edge generated
- Any CFG node using block.timestamp is typed as CFG_NODE_WRITE or CFG_NODE_ENTRY, not CFG_NODE_CALL
- No direct graph feature captures "uses block.timestamp"

**Current NODE_FEATURE_DIM=12 features (feat indices 0-11):**
```
0: is_reentrant_old (call_typed old Slither reentrancy check)
1: pure (function is pure, cannot read state)
2: view (function is view-only)
3: is_constructor
4: ext_calls (count of HighLevelCall + LowLevelCall ops)
5: has_loop (function contains a loop)
6: loc (lines of code — unnormalized)
7: type_id / 12.0 (normalized node type)
8: in_unchecked (Solidity 0.8.x only)
9: return_ignored (FIXED but was 0 always)
10: is_reentrant_new (new Slither reentrancy detector)
11: (reserved or zero)
```

Feature index 1 (`pure`) is the weakest discriminative feature (pure functions cannot be
vulnerable to state-based exploits; useful as a negative indicator but rarely true for vulnerable
functions). Replacing `pure` with a `uses_block_globals` feature (1 if any slithir op reads
SolidityVariableComposed with name in {timestamp, number, difficulty}) would directly encode
Timestamp-class signal.

**Impact:** Timestamp class currently has F1=0.174. With direct graph signal, target ≥ 0.30.

### 4.4 Dark Spot: Transfer/Send Invisible in ext_calls and CFG

**Slither Transfer/Send IR ops** (ETH transfer and send) are a distinct IR class from HighLevelCall
and LowLevelCall. The current `_compute_external_call_count()` only counts HighLevelCall and
LowLevelCall.

**Impact on DoS:** Many DoS vulnerability patterns involve ETH-transfer loops:
```solidity
for (uint i = 0; i < recipients.length; i++) {
    recipients[i].transfer(amount);  // ETH-transfer DoS: one reverts, all revert
}
```
This pattern has `Transfer` ops, not `HighLevelCall` ops. Current extractor:
- ext_calls = 0 for this contract (no HighLevelCall found)
- CFG node typed as CFG_NODE_WRITE (transfer counts as a write) not CFG_NODE_CALL
- DoS-positive example looks like a pure-write-loop with no external interactions

The GNN and CodeBERT both miss this pattern unless CodeBERT happens to see the word "transfer"
within the 512-token window.

**Fix:** add `isinstance(op, (Transfer, Send))` check in `_compute_external_call_count()` and
update `_cfg_node_type()` to type Transfer/Send ops as CFG_NODE_CALL.

### 4.5 Dark Spot: in_unchecked Only Captures Solidity 0.8.x

**Current implementation:** The `in_unchecked` feature is 1 when a node is inside a Slither
`unchecked {}` block. This block type only exists in Solidity 0.8.x where arithmetic overflow
checking is on by default.

**Problem for IntegerUO:** All Solidity 0.4.x and 0.5.x contracts have unchecked arithmetic by
default (no SafeMath). For these contracts, `in_unchecked=0` does not mean "checked arithmetic"
— it means "old Solidity where everything is unchecked". The model cannot distinguish:
- Old contract without SafeMath (vulnerable, in_unchecked=0 because no block)
- New contract with explicit unchecked{} block (vulnerable, in_unchecked=1)
- New contract with SafeMath or checked arithmetic (safe, in_unchecked=0)

Despite this limitation, IntegerUO still achieves F1=0.732 (best class) because it has 10,886
training positives — the model can learn the pattern from other features (ext_calls, loc, type_id).
This is the least urgent schema fix.

### 4.6 loc Feature Not Normalized — Scale Imbalance

**Current:** feat[6] = raw line count (integer). Range observed: [0, 2538]. CONTRACT nodes
average loc≈133.6. All other features are in [0, 1] or [0, 12/12].

**Impact on dot products:** In GAT attention `e_ij = W·x_i ⊕ W·x_j`, the loc=133 of a CONTRACT
node dominates the attention logit. The attention weight computation is therefore disproportionately
influenced by loc, meaning GAT learns "attend to high-loc nodes" rather than "attend to structurally
important nodes". This is a gradient magnitude imbalance issue that affects all 4 GAT layers.

**Fix:** apply `log1p(loc) / log1p(1000)` normalized to [0, 1]. log1p(1000)≈6.91. This maps:
- loc=0 → 0.0
- loc=50 → 0.57
- loc=133 → 0.72
- loc=1000 → 1.0
- loc=2538 → 1.13 (clamped to 1.0)

---

## 5. CEI Ordering Analysis — Is the Graph Sufficient?

### 5.1 Vulnerability-Safe Distinction in Graph Structure

The fundamental question for Reentrancy detection is whether the CEI (Checks-Effects-Interactions)
vs CEA (Checks-Effects external-call-Attack) pattern is encoded in the graph.

**Analysis of CONTROL_FLOW edge structure:**

For a Reentrancy-vulnerable (CEA) contract:
```
CFG_NODE_ENTRY → CFG_NODE_READ (check balance) → CFG_NODE_CALL (send ETH)
→ CFG_NODE_WRITE (update state) → CFG_NODE_RETURN
```
CONTROL_FLOW edge order: READ → CALL → WRITE

For a safe (CEI) contract:
```
CFG_NODE_ENTRY → CFG_NODE_READ (check balance) → CFG_NODE_WRITE (update state)
→ CFG_NODE_CALL (send ETH) → CFG_NODE_RETURN
```
CONTROL_FLOW edge order: READ → WRITE → CALL

This pattern IS structurally encoded in the directed CONTROL_FLOW edges. A GNN with directed
message passing along CF edges (Phase 2, layer 3 in v5.2) CAN in principle learn to distinguish
these orderings. The critical requirements are:
1. CF edges are directed (source→destination, not bidirectional) — this is enforced by
   `add_self_loops=False` in Phase 2
2. At least 2 CF hops to propagate from CALL to WRITE (or WRITE to CALL)
3. The CFG node types correctly distinguish CALL from WRITE ops

**v5.2 implementation:** Phase 2 has only 1 CF layer (1 hop). This is theoretically insufficient
for a function with CALL and WRITE nodes separated by multiple intermediate nodes. A 2-hop CF
layer would reach WRITE from CALL even with one intermediate node.

**However:** The v5.2 behavioral failure for Reentrancy was not due to insufficient CF depth.
It was due to RC1 (fusion gradient dominance) causing the GNN phase to be underweighted, and RC2
(pos_weight=2.82) causing Reentrancy to fire on any external call regardless of ordering.

### 5.2 CEI Augmentation Effectiveness

The 50 CEI augmentation pairs added in v5.2/v5.3 were designed to provide "safe contract with
external calls" training examples specifically for the Reentrancy class. However:
- 50 pairs = 50 safe + 50 vulnerable = 100 contracts
- These represent 0.32% of the training set
- The co-occurrence signal from 99% DoS→Reentrancy overwhelms this small augmentation

**Recommendation for v6:** Augmentation must be at scale (500+ safe contracts with external calls,
500+ clean DoS, 500+ clean Reentrancy) to meaningfully break co-occurrence patterns.

---

## 6. Token Truncation — Catastrophic Finding

### 6.1 Measurement Methodology

A random sample of 1,000 contracts from the deduped dataset was tokenized WITHOUT truncation using
the CodeBERT tokenizer. The full token length distribution was measured.

### 6.2 True Token Length Distribution

| Percentile | Token count |
|------------|-------------|
| P5 | 312 |
| P10 | 448 |
| Median (P50) | 2,469 |
| P75 | 4,231 |
| P90 | 7,549 |
| P95 | 11,283 |
| P99 | 22,718 |

**Current limit:** 512 tokens (hard position embedding limit for CodeBERT base)

**Contracts that FIT in current window:**

| Limit | % of contracts |
|-------|----------------|
| 512 (current) | 3.9% |
| 1024 | 14.9% |
| 2048 | 33.6% |
| 4096 | 73.1% |
| 8192 | 91.2% |

### 6.3 Implications

**96.1% of contracts are truncated at 512 tokens.** The CodeBERT path sees only a prefix of
the source code. For the median contract (2,469 tokens), CodeBERT sees 512/2469 = 20.7% of
the contract.

This means:
- Any vulnerability function appearing after the first ~20 functions is completely invisible
  to the text encoder.
- For large contracts (DeFi protocols, governance contracts), the CodeBERT embedding captures
  only constructor, token metadata, and early functions.
- Timestamp usage patterns (typically in oracle/getter functions later in the file) are often
  in the truncated portion.
- Multi-function Reentrancy patterns (where the reentrant path spans across different functions)
  may be split by truncation.

**The GNN path is not affected by truncation** (graph is built from full AST, all functions
extracted). This is a CodeBERT-specific limitation.

### 6.4 Windowed Tokenization Solution

Windowed tokenization with stride:
```
window_size = 512, stride = 256, max_windows = 8
```

For a 2,469-token contract: ceil(2469 / 256) = 10 windows, capped to 8
Coverage: 8 × 256 = 2,048 tokens = 83% of contract

For a 7,549-token (P90) contract: ceil(7549 / 256) = 30 windows, capped to 8
Coverage: 8 × 256 = 2,048 tokens = 27% of contract

**VRAM impact:** Each window is one forward pass through CodeBERT. With max_windows=8 and
batch_size=8, this is 64 CodeBERT forward passes per training step (vs 8 currently).
With CodeBERT frozen (LoRA only), this is just linear layer + attention computation.
Estimated VRAM increase: ~1.5× (window attention pooler adds minimal overhead).

---

## 7. Model Architecture Bottlenecks

### 7.1 GNN Capacity (hidden_dim=128)

The current GNN outputs 128-dim node embeddings. For multi-hop CEI ordering detection, the
model must encode: node type (13 types), edge type (8 types), topological position (pre/post-call),
and vulnerability signal (return ignored, ext_call count, etc.) into 128 dimensions.

**GAT attention head calculation:** In Phase 2 with heads=1 and dim=128, the attention computation
is over 128-dim keys. This is a relatively narrow representation for discriminating 8 edge types
and 13 node types simultaneously.

**Recommendation:** hidden_dim 128 → 256. This quadruples GNN parameter count (~4M → ~16M params
for GNN encoder) but remains well within VRAM budget (GNN is <100MB even at 256-dim).

### 7.2 GNN Depth (Phase 2: 1 CF layer)

One CF-hop means CALL and WRITE nodes only communicate through their direct CF successor/predecessor.
A typical function with intermediate nodes:
```
CFG_ENTRY → CHECK → CALL → INTERMEDIATE1 → INTERMEDIATE2 → WRITE → RETURN
```
With 1 CF hop, CALL's message only reaches INTERMEDIATE1, not WRITE. WRITE has no information
about whether CALL preceded it.

**Recommendation:** 2 CF layers in Phase 2 (currently 1). This allows CEI-pattern signals to
propagate across one intermediate node, covering most real-world reentrancy patterns.

### 7.3 Edge Embedding Dimension (32)

Current: `nn.Embedding(8, 32)` — 8 edge types, 32-dim embedding.
With 8 edge types, 32 dimensions is sufficient for linear separation but may not encode
edge-type interactions well in the attention mechanism.

**Recommendation:** 32 → 64. Minimal VRAM impact (+256 parameters).

### 7.4 Classifier Architecture (384 → 10, no hidden layer)

The Three-Eye classifier concatenates three 128-dim embeddings [B, 384] and passes through
a single Linear(384, 10) layer. This linear projection cannot model non-linear interactions
between the three eyes' representations.

**Recommendation:** Add a hidden layer: Linear(384, 192) → ReLU → Dropout(0.1) → Linear(192, 10).
Adds ~75K parameters. The hidden layer allows the classifier to learn "GNN eye sees CEI but
TF eye also sees external call → confidence boost" type non-linear interactions.

### 7.5 LoRA Configuration (r=16, α=32)

LoRA r=16 means each modified attention matrix gets 16-rank updates. For CodeBERT on 512-token
sequences, this is a reasonable capacity. But with windowed tokenization (up to 8 windows, each
512 tokens covering different parts of the contract), the LoRA weights must encode vulnerability
patterns that may appear anywhere in the contract.

**Recommendation:** r: 16 → 32, α: 32 → 64. Doubles LoRA capacity. Adds ~4M trainable parameters
(still <10M total LoRA params). This is particularly important for long-range vulnerability
patterns that span function boundaries.

---

## 8. Per-Class Signal Analysis

### 8.1 Signal Sources Per Class

| Class | Graph Feature Signal | Topological Signal | CodeBERT Signal | Current F1 |
|-------|---------------------|-------------------|----------------|------------|
| IntegerUO | in_unchecked (0.8.x), ext_calls | Arithmetic op nodes | Overflow patterns | 0.732 |
| GasException | has_loop, ext_calls | CFG loop structure | gas_left, gasleft() | 0.407 |
| Reentrancy | is_reentrant_old/new, ext_calls | CF ordering (CALL before WRITE) | transfer/call patterns | 0.322 |
| MishandledException | return_ignored (FIXED) | READS trace after CALL | .call() return check | 0.342 |
| UnusedReturn | return_ignored (FIXED) | READS trace | return value usage | 0.238 |
| DenialOfService | ext_calls (PARTIAL — Transfer missing) | CF loop + CALL | transfer() loop | 0.329 |
| CallToUnknown | ext_calls (partial) | LowLevelCall topology | address.call() | 0.284 |
| TOD | None directly | None | block.number, order dep | 0.283 |
| ExternalBug | ext_calls (indirect) | CALLS edges | untrusted external | 0.262 |
| Timestamp | None (block.timestamp invisible) | None | block.timestamp text | 0.174 |

### 8.2 Class-by-Class Detailed Assessment

**IntegerUO (F1=0.732 — best class):**
10,886 training positives = dominant class. The `in_unchecked` feature works for Solidity 0.8.x.
For 0.4.x/0.5.x contracts, the model learns structural patterns (large arithmetic-heavy functions)
via CodeBERT. High data volume overcomes the 0.4.x blindspot. This class will benefit from
windowed tokenization (arithmetic ops spread throughout contract).

**GasException (F1=0.407):**
`has_loop` is a meaningful feature — 1 if function contains a loop. CFG loop structure (backward
CF edges) is visible in the graph. Model can detect "loop with external interaction" patterns.
Limited by co-occurrence with IntegerUO (loops with arithmetic often trigger both).

**Reentrancy (F1=0.322 — underperforms given signal richness):**
Rich signal available (CF ordering, is_reentrant, ext_calls) but poisoned by:
1. 99% DoS co-occurrence → model fires on any "loop + external call" pattern
2. RC2 pos_weight=2.82 in v5.2 → overfit to fire on any external call
3. RC1 fusion dominance → GNN CF ordering signal was underweighted
After fixing RC1+RC2+RC3 properly AND augmenting with clean Reentrancy/DoS/safe examples,
this class should improve substantially.

**MishandledException (F1=0.342) and UnusedReturn (F1=0.238):**
Both were effectively disabled by the return_ignored=0 bug. The fix (commit bef1f2a) should
substantially improve both. The graph signal (READS trace after CALL node) was present but
muted because the feature that should identify it was always zero.
MishandledException: target ≥ 0.50 with fix
UnusedReturn: target ≥ 0.45 with fix

**DenialOfService (F1=0.329 at threshold=0.95 — barely usable):**
Severe starvation (257 training positives). Transfer/Send ops invisible in ext_calls and CFG
typing. At threshold=0.95, the model almost never fires DoS — the high threshold was the only
way to get non-trivial F1 by suppressing false positives rather than improving true positives.
With Transfer/Send fix + ~500 augmented DoS-only examples, this class may become viable.

**Timestamp (F1=0.174 — worst class):**
Zero direct graph signal (block.timestamp invisible). 1,493 training positives (low). CodeBERT
truncation cuts off many timestamp usages. TOD has the same problem (block.number dependency).
With uses_block_globals feature addition (replacing `pure`) and windowed tokenization, both
Timestamp and TOD should improve.

**CallToUnknown (F1=0.284):**
Most CallToUnknown patterns involve delegatecall/staticcall, which are LowLevelCall in Slither.
The ext_calls counter catches some of these but doesn't distinguish delegatecall from regular call.
CodeBERT can detect "delegatecall" keyword. Windowed tokenization would help.

**ExternalBug (F1=0.262):**
This class covers a broad range of external call misuse patterns. The graph signal is diffuse
(any ext_call may indicate ExternalBug). Improvement depends on cleaner training data (breaking
co-occurrence) and possibly additional graph features specific to external call context.

**TOD (F1=0.283):**
Transaction-ordering dependency is difficult to detect statically. Patterns typically involve
reading block.number or checking msg.sender/msg.value in ways that create ordering windows.
No direct graph features for this. CodeBERT text signal is the primary hope. Windowed tokenization
helps by exposing more of the contract to CodeBERT.

---

## 9. Summary of Root Causes Ranked by Impact

| Rank | Root Cause | Classes Affected | v6 Fix |
|------|-----------|-----------------|--------|
| 1 | Token truncation (96% truncated) | All classes | Windowed tokenization |
| 2 | return_ignored=0 bug (FIXED) | MishandledException, UnusedReturn | Phase 0.1 |
| 3 | DoS/Reentrancy 99% co-occurrence | Reentrancy, DoS, ExternalBug | Phase 4 augmentation |
| 4 | block.timestamp invisible | Timestamp, TOD | Phase 0.4 (replace pure with uses_block_globals) |
| 5 | Transfer/Send invisible | DoS, Reentrancy | Phase 0.2/0.3 |
| 6 | loc not normalized | All classes (attention scale) | Phase 0.5 |
| 7 | Fusion LR dominance (RC1) | Reentrancy | RC1 fix in loss/optimizer config |
| 8 | DoS starvation (257 positives) | DoS | Phase 4.1 augmentation |
| 9 | GNN 1-hop CF insufficient | Reentrancy | Phase 2.2 (2nd CF layer) |
| 10 | Classifier no hidden layer | All classes | Phase 2.5 |
| 11 | GNN hidden_dim=128 narrow | All classes | Phase 2.1 |
| 12 | LoRA under-parameterized | Text-dependent classes | Phase 2.7 |
