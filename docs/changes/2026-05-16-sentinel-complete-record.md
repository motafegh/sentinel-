# SENTINEL — Complete Project Record
**Date written:** 2026-05-16  
**Covers:** v4 baseline through v6 architecture (all training runs, all analysis, all decisions)  
**Status at time of writing:** v6 code committed; graph re-extraction 51% complete

This document is the single unified reference for everything that happened in the SENTINEL
ML pipeline. It answers: what did we find, why did we make each decision, what did we change,
what is the current state, and what do we expect.

---

## Part 1: What SENTINEL Is and What It Is Trying to Do

SENTINEL is a decentralised security oracle for Ethereum smart contracts. It takes a Solidity
source file as input and outputs a multi-label vulnerability classification: which of 10 known
vulnerability classes are present in the contract.

**The 10 classes:**

| ID | Class | Description |
|----|-------|-------------|
| 0 | CallToUnknown | External call to an unknown/untrusted address |
| 1 | DenialOfService | Gas-griefing, block-gas-limit, ETH-transfer loop patterns |
| 2 | ExternalBug | General external call misuse |
| 3 | GasException | Gas-related issues, loops without gas bounds |
| 4 | IntegerUO | Integer overflow/underflow |
| 5 | MishandledException | Return value of call not checked |
| 6 | Reentrancy | CEA ordering: call before state update |
| 7 | Timestamp | block.timestamp manipulation |
| 8 | TOD | Transaction-ordering dependency |
| 9 | UnusedReturn | Return value of function silently discarded |

**The pipeline output:** per-class binary predictions with confidence scores, proved via
ZK circuit (EZKL/Groth16) so the prediction can be verified on-chain without revealing the
source code.

**Why it is hard:**

1. **Multi-label:** a contract can be vulnerable to multiple classes simultaneously (often is).
2. **Class imbalance:** DenialOfService has 257 training positives vs IntegerUO's 10,886.
3. **Co-occurrence:** 99% of DoS contracts are also labeled Reentrancy — the model cannot
   easily learn to distinguish them.
4. **Source diversity:** 44,420 contracts spanning Solidity 0.4.x through 0.8.x, ranging from
   10 to 2,538 lines of code.
5. **Dual-modality:** structural patterns (graph) and semantic patterns (text) are both needed.

**Architecture approach:** dual-path model.
- **GNN path:** graph-theoretic analysis of Slither's program IR; encodes function call graphs,
  data flow, control flow ordering.
- **CodeBERT path:** text embedding of the Solidity source; picks up keyword patterns and
  natural language structure.
- **CrossAttentionFusion:** lets GNN node queries attend to specific token positions; fuses
  "this function has an external call" (graph) with "the word 'transfer' appears near 'for'"
  (text).
- **Three-Eye Classifier:** three independent 128-dim embeddings (GNN eye, Transformer eye,
  Fused eye) → concatenated → classifier. Auxiliary heads on each eye prevent one eye from
  dominating.

---

## Part 2: Dataset — What We Found and Fixed

### 2.1 The Original Dataset and Its Fatal Flaw

The Berkeley-Columbia Contract Corpus (BCCC) stores contracts by vulnerability class: each
class has a subdirectory of `.sol` files named by SHA256 hash. A contract vulnerable to
multiple classes appears as a separate file copy in each relevant directory.

When we built the dataset, we assigned a row per file and used `md5(file_path)` as the unique
key for graph and token cache files. This dual-hash system introduced a catastrophic problem:

- Contract X (SHA256=`abc`) appears in `reentrancy/abc.sol` AND `dos/abc.sol`
- Path-based MD5 produces MD5-A for `reentrancy/abc.sol` and MD5-B for `dos/abc.sol`
- During stratified splitting, MD5-A lands in train, MD5-B lands in val
- The model trains on the Reentrancy label and validates on the DoS label of the same contract

**The result:** 7,630 content groups (34.9% of unique contracts) spanned multiple splits.
The original 68,523-row dataset had ~35% of its "validation" set memorized in training.

**What the leak did to metrics:** v4 reported tuned F1=0.5422 and v5.0 reported F1=0.5828.
Both numbers are meaningless — they measured how well the model memorized training data, not
how well it detected vulnerabilities. We discovered the leakage when we deduped the dataset
and saw F1 drop from 0.5828 to 0.34.

### 2.2 The Fix: Content-Based Deduplication

We deduped by SHA256 content hash (the BCCC filename), merged all class labels for duplicate
contracts, and produced a single clean CSV:

- **Before dedup:** 68,523 rows, 34.9% cross-split leakage
- **After dedup:** 44,420 rows, 0% cross-split leakage
- **Splits:** train=31,092 / val=6,661 / test=6,667 (stratified, seed=42)
- **File:** `ml/data/processed/multilabel_index_deduped.csv`

All v5.x and v6 training uses this deduped CSV. The original 68K CSV is kept for reference
only — training on it is prohibited.

### 2.3 Co-occurrence Problem

Deduplication exposed the true label distribution. The same co-location mechanism that caused
leakage also caused extreme multi-label co-occurrence:

| Class pair | Co-occurrence rate |
|------------|-------------------|
| DoS → Reentrancy | **99%** |
| MishandledException → IntegerUO | 96% |
| TOD → IntegerUO | 86% |
| UnusedReturn → Reentrancy | 69% |

A 99% DoS→Reentrancy co-occurrence means:
- Every training signal for "this is a DoS contract" also fires Reentrancy updates
- The model cannot learn "DoS specifically" — it only learns "DoS co-occurs with Reentrancy"
- Any contract with an external call loop pattern becomes both DoS=1 and Reentrancy=1 in predictions

This explains why Reentrancy fires on safe contracts with external calls: the model learned that
"external call pattern → DoS AND Reentrancy" as a single co-occurrent cluster.

### 2.4 Data Starvation

The deduped label counts reveal extreme class imbalance:

| Class | Train positives | Rate |
|-------|-----------------|------|
| IntegerUO | 10,886 | 35.0% |
| GasException | 3,918 | 12.6% |
| Reentrancy | 3,500 | 11.2% |
| MishandledException | 3,296 | 10.6% |
| ExternalBug | 2,383 | 7.7% |
| CallToUnknown | 2,527 | 8.1% |
| TOD | 2,374 | 7.6% |
| UnusedReturn | 2,126 | 6.8% |
| Timestamp | 1,493 | 4.8% |
| **DenialOfService** | **257** | **0.8%** |

DenialOfService has 257 training positives in a 31,092-row training set. At effective batch
size 32 (batch=8, gradient accumulation=4), there is on average 0.26 DoS positives per batch
step — one DoS example every 4 batches. This is fundamentally incompatible with stable gradient
learning for this class.

---

## Part 3: Feature Schema Bugs — What We Investigated and Found

Before the v5.x training series launched, we had already fixed the interface selection bug
(`_select_contract()`). During the v5.2 behavioral failure analysis we discovered four more bugs
in the feature extractor (`graph_extractor.py`) that were silently corrupting the training signal.

### 3.1 Bug: `return_ignored` Always 0.0 (the lvalue bug)

**Classes affected:** MishandledException (F1=0.342), UnusedReturn (F1=0.238)

**What the feature is supposed to do:** `return_ignored = 1.0` if a function calls an external
contract and does not use the return value. This is the primary signal for MishandledException
("you called `.call()` and didn't check if it succeeded") and UnusedReturn ("you called a
function that returns a value and discarded it").

**How we found the bug:** We traced MishandledException's poor F1 and noticed both classes
clustered around the same low threshold (0.35–0.45) with no discrimination power. We then read
the extractor code and found this check:

```python
return_ignored = any(op.lvalue is None for op in node.irs)
```

**Why it was wrong:** In Slither's IR, every assignment-type operation stores its left-hand-side
value in `op.lvalue`. This is always set — even for temporary variables produced by call
expressions. Slither creates a `TMP_1` (TupleVariable) and assigns it as the lvalue; it is never
`None`. The condition `op.lvalue is None` always returns False, so `return_ignored` was always
`0.0` for every node, for every contract.

**The correct logic:** The return value is "unused" if the temporary variable assigned by the
call (`op.lvalue`) does not appear in the `reads` set of any subsequent IR operation in the same
function. We rewrite it as:

```python
def _compute_return_ignored(function) -> float:
    for node in function.nodes:
        for op in node.irs:
            if isinstance(op, (HighLevelCall, LowLevelCall)):
                if op.lvalue is not None:
                    lval_id = id(op.lvalue)
                    used = any(
                        any(id(v) == lval_id for v in getattr(other_op, 'read', []))
                        for other_node in function.nodes
                        for other_op in other_node.irs
                    )
                    if not used:
                        return 1.0
    return 0.0
```

**Fix committed:** commit `bef1f2a`

### 3.2 Bug: Transfer/Send Invisible in `ext_calls` Counter

**Classes affected:** DenialOfService (F1=0.329 at threshold=0.95), Reentrancy variants

Slither has four types of "external interaction" IR operations:
- `HighLevelCall`: typed interface call, e.g. `token.transfer(amount)`
- `LowLevelCall`: untyped `.call{value: x}("")`
- `Transfer`: native ETH transfer, e.g. `recipient.transfer(amount)`
- `Send`: native ETH send, e.g. `recipient.send(amount)`

The extractor only counted `HighLevelCall` and `LowLevelCall` in the `ext_calls` feature.
ETH-transfer-loop DoS patterns (the most common DoS pattern in BCCC) use `Transfer` ops:

```solidity
for (uint i = 0; i < recipients.length; i++) {
    recipients[i].transfer(amount);   // Transfer op, not HighLevelCall
}
```

This contract had `ext_calls=0` in the graph, making it look like a pure state-manipulation
contract. The GNN had no signal that it contained external interactions.

**Fix:** Added `Transfer, Send` to both `_compute_external_call_count()` and `_cfg_node_type()`.
Now Transfer/Send nodes are typed as `CFG_NODE_CALL` in the control-flow graph, making
call-before-write (CEA) vs write-before-call (CEI) patterns visible for ETH-transfer reentrancy.

**Fix committed:** commit `310e738`

### 3.3 Bug: `block.timestamp` Invisible in Graph

**Classes affected:** Timestamp (F1=0.174 — worst class), TOD (F1=0.283)

In Slither's type system, Solidity global variables (`block.timestamp`, `block.number`,
`msg.sender`, etc.) are `SolidityVariableComposed` objects. These are **not** `StateVariable`
objects and therefore do not appear in `function.state_variables_read`.

The extractor built READS edges from `state_variables_read`. Since `block.timestamp` is a
`SolidityVariableComposed` and not in `state_variables_read`, it produced no READS edges.
A CFG node that reads `block.timestamp` was typed as `CFG_NODE_ENTRY` (default), not
`CFG_NODE_CALL` or `CFG_NODE_READ`. The Timestamp class had essentially zero direct graph signal.

**The old `pure` feature (feat[2]):** The existing feature at index 2 was `pure` (whether the
function declares itself as pure/view). This has essentially zero discriminative value for
vulnerable functions: pure functions cannot write state and therefore cannot be vulnerable to
most exploits. In practice, `pure=1` was a near-certain indicator of non-vulnerability — useful
as a negative but contributing near-zero gradient on positive examples.

**Fix:** Replaced `pure` at feat[2] with `uses_block_globals`. The new feature is `1.0` if any
IR operation in the function reads a `SolidityVariableComposed` in the set
{timestamp, number, difficulty, gaslimit, coinbase, basefee, prevrandao}. This directly encodes
"this function uses block globals" — the primary signal for Timestamp and TOD.

**Fix committed:** commit `310e738`

### 3.4 Bug: `loc` Not Normalized — Scale Dominance in GAT Attention

**Classes affected:** All (global attention quality)

The `loc` feature (lines of code) was stored as a raw integer, range [0, 2538]. All other
features are in [0.0, 1.0] or [0, 12/12]. In GAT attention, the dot product between node
feature vectors includes `loc=133` for a typical CONTRACT node against binary {0,1} features
elsewhere. The 133× magnitude difference causes the attention weights to be dominated by loc:
`softmax(W·x_i · W·x_j)` has its largest component from the loc feature.

The net effect is that GAT layers learned "attend to high-loc (large) nodes" as the primary
attention pattern, regardless of vulnerability-relevant structure. This is a gradient magnitude
imbalance that affects all four GAT layers in v5.x.

**Fix:** `log1p(loc) / log1p(1000)`, clamped to [0, 1]. This maps:
- loc=0 → 0.000
- loc=50 → 0.573
- loc=133 → 0.722
- loc=1000 → 1.000
- loc=2538 → 1.000 (clamped)

CONTRACT nodes now have feat[6]≈0.72, comparable in scale to the type_id feature.

**Fix committed:** commit `310e738`

### 3.5 Summary of v4 Feature Schema

After all fixes, the 12-dimensional node feature vector is:

```
[0]  type_id / 12.0          — normalized node type (CONTRACT=0, FUNCTION=1, ..., CFG_RETURN=12)
[1]  visibility               — 1.0 if function is public
[2]  uses_block_globals  ★   — 1.0 if reads block.timestamp/number/etc. (was: pure)
[3]  view                     — 1.0 if function is view-only
[4]  payable                  — 1.0 if function accepts ETH
[5]  complexity               — cyclomatic complexity, normalized
[6]  loc                 ★   — log1p(lines) / log1p(1000), range [0,1] (was: raw count)
[7]  return_ignored      ★   — 1.0 if call return value unused in subsequent ops (was: always 0)
[8]  call_target_typed        — 1.0 if interface-typed call (vs dynamic address.call)
[9]  in_unchecked             — 1.0 if inside unchecked{} block (Solidity 0.8.x)
[10] has_loop                 — 1.0 if function contains a loop
[11] external_call_count ★   — log1p(count)/log1p(20), includes Transfer/Send (was: 0 for ETH loops)
```

★ = changed from v3. `FEATURE_SCHEMA_VERSION` bumped from `"v3"` to `"v4"`.

---

## Part 4: Token Truncation — A Catastrophic Finding

### 4.1 What We Measured

We tokenized a random 1,000-contract sample from the deduped dataset WITHOUT any truncation
to measure the true token length distribution:

| Percentile | Token count |
|------------|-------------|
| P5 | 312 |
| P10 | 448 |
| **Median (P50)** | **2,469** |
| P75 | 4,231 |
| P90 | 7,549 |
| P99 | 22,718 |

**Current CodeBERT limit:** 512 tokens (hard position embedding limit in CodeBERT-base).

**Percentage of contracts that fit within various limits:**

| Window size | Contracts that fit |
|-------------|-------------------|
| 512 (v5.x) | **3.9%** |
| 1024 | 14.9% |
| 2048 | 33.6% |
| 4096 | 73.1% |

**96.1% of contracts were truncated in v5.x.** The CodeBERT path was seeing only the first
~20% of a median-length contract. Any vulnerability function appearing after the first ~20
functions was completely invisible to the text encoder.

### 4.2 Why This Matters

- **Timestamp and TOD:** These vulnerability patterns often appear in oracle/pricing functions
  that are later in the contract file. With 512-token truncation, most timestamp usages are cut.
- **Multi-function Reentrancy:** Reentrancy patterns that span two functions (withdraw calls
  transfer calls fallback) may be split by the 512-token boundary.
- **Large DeFi contracts:** Governance and DeFi contracts with hundreds of functions have the
  vulnerability buried deep in the file, past the truncation point.

The GNN path is NOT affected by truncation — Slither parses the full AST and all functions
are present in the graph regardless of contract length.

### 4.3 The Fix: Windowed Tokenization

We implemented a sliding-window tokenizer (`retokenize_windowed.py`):

```
window_size = 512 tokens
stride      = 256 tokens (50% overlap — boundary tokens appear in two windows)
max_windows = 4  (constrained by 8GB VRAM: 4×512×B=8 = 16,384 positions; 8× pushes OOM)
```

For a 2,469-token contract with max_windows=4:
- Window 0: tokens [0, 512) — pragma, imports, contract declaration, first functions
- Window 1: tokens [256, 768) — functions 3-8 (with overlap)
- Window 2: tokens [512, 1024) — middle functions
- Window 3: tokens [768, 1280) — later functions

Coverage: 4 windows × 256 stride ≈ 1,280 unique tokens = 52% of median contract vs 21% before.

All contracts produce exactly `[4, 512]` tensors (short contracts get zero-padded windows
with `attention_mask=0`). The `key_padding_mask` in CrossAttentionFusion masks out padding
windows so they contribute zero to cross-attention.

The `WindowAttentionPooler` in the model extracts the CLS token from each window (positions
0, 512, 1024, 1536) and combines them via learned attention weights, so the transformer eye
represents all four windows rather than just window 0's CLS.

---

## Part 5: Training History — Every Run, What Happened, Why We Stopped

### 5.1 v4 Baseline — `multilabel-v4-finetune-lr1e4`

**Architecture:** Simpler dual-path without Three-Eye design, CrossAttentionFusion not yet present  
**Dataset:** 68,523-row leaky dataset  
**Result:** Tuned F1-macro = **0.5422**

This number is inflated by 34.9% cross-split leakage. The v4 checkpoint is kept as a fallback
reference (`ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`) but its numbers are not
honest baselines. We use "v4 F1 − 0.05 per class" as a floor for v5/v6 comparison.

### 5.2 v5.0 — `v5-full-60ep`

**Architecture:** Three-Eye + CrossAttentionFusion; GNN 4-layer 128-dim; no JK; schema v3  
**Dataset:** 68K leaky (not yet deduped)  
**Best val F1:** 0.5828 (epoch 44) — **inflated by leakage**  
**Behavioral result:** 15% detection rate, **0% safe-contract specificity**

Every single safe contract was flagged as vulnerable. The model had degenerated to an
all-vulnerable predictor.

**Root causes diagnosed:**
- **Interface selection bug:** `_select_contract()` selected the wrong contract in multi-contract
  .sol files, sending wrong node features to the GNN for a significant fraction of training examples.
- **GNN pool dominated by CFG_RETURN nodes:** All-node pooling for the GNN eye. CFG_RETURN nodes
  make up ~77% of all nodes in typical contracts. They carry only structural information (exit
  points) and completely washed out function-level vulnerability signals.
- **aux_loss_weight=0.1 too low:** Auxiliary heads contributed negligible gradient. GNN and TF
  eyes converged to nearly identical representations.
- **No JK connections:** Without Jumping Knowledge, gradients did not flow effectively to early
  GNN layers.

**Lesson:** Val F1 on leaky data is meaningless. Behavioral tests are the only honest metric.

### 5.3 v5.1 — INVALID

Multiple resume cycles corrupted the OneCycleLR scheduler (total_steps mismatch). GNN
gradients collapsed at epoch 8. Best F1=0.2794 was achieved with a partially-broken model.
No behavioral test run. Results discarded.

### 5.4 v5.2 Series — Three Sub-runs on Clean Data

All v5.2 sub-runs share: 44,470-row deduped dataset, JK attention aggregation, function-node
pooling, aux_loss=0.3, NUM_EDGE_TYPES=8, schema v3.

**Sub-run 1 (v5.2-jk-20260515c):** Found that eval_threshold=0.5 was masking minority class
improvements. Early-stopped at epoch 16, F1=0.1872 (threshold=0.5 noise).

**Sub-run 2 (r2):** Continued from r1. Best epoch 20, val F1=0.2823 (still noisy). After
post-hoc threshold tuning: **tuned F1=0.3373**. Key observation: DoS required threshold=0.95
to get any F1 — model almost never fires DoS confidently. Timestamp F1=0.174 worst class.

**Sub-run 3 (r3) — FINAL:** Fixed eval_threshold=0.35. Ran to epoch 52, best at epoch 32.
**Tuned F1-macro = 0.3422** with per-class thresholds. Behavioral test results:
- Detection rate: **36%** (7 of 19 test contracts correctly flagged) — FAIL (gate: ≥80%)
- Safe specificity: **33%** (fires on 67% of safe contracts) — FAIL (gate: ≥80%)

**Root causes of v5.2 behavioral failure:**
1. **Fusion gradient dominance (RC1):** CrossAttentionFusion ran at full base LR, producing
   4–5× higher gradient norms than the GNN. Fusion found the shortcut: "external call present
   → Reentrancy=1" and overwhelmed the GNN's CEI ordering signal.
2. **Reentrancy pos_weight=2.82 (RC2):** Combined with RC1, the model became aggressively
   Reentrancy-positive for any contract with an external call.
3. **No label smoothing (RC3):** Hard targets pushed Reentrancy probability on safe contracts
   to 0.97 with no regularization penalty.
4. **DoS/Timestamp starvation:** 257 DoS positives cannot produce stable gradients at any
   reasonable batch size.
5. **return_ignored=0 always (code bug):** MishandledException and UnusedReturn had zero
   function-level graph signal.
6. **block.timestamp invisible (code bug):** Timestamp class had zero direct graph signal.

### 5.5 v5.3 — `v5.3-bce-smooth-20260516` — KILLED at epoch 47

**Applied fixes vs v5.2:** fusion_lr=0.5× (RC1 fix), label_smoothing=0.05 (RC3 fix),
pos_weight_min_samples=3000 (intended RC2 fix).

**The pos_weight_min_samples mistake:** Setting min_samples=3000 capped 5 of 10 classes to
pos_weight=1.0:
- Reentrancy (3,500 positives → capped)
- GasException (3,918 → capped)
- IntegerUO (10,886 → capped)
- MishandledException (3,296 → capped)
- GasException (3,918 → capped)

With 5 classes having no imbalance correction, those classes trained under equal BCE weighting
despite still being imbalanced. This removed discriminative gradient signal precisely for the
mid-frequency classes that need it most.

**Training trajectory:** F1 plateau at 0.2559 from epoch 31. Loss moved from 0.84 to 0.807
in 47 epochs — expected convergence would be ~0.60 at epoch 47 if training normally.

**Kill decision:** Flat F1 for 16+ epochs, worse than v5.2 (0.2559 vs 0.3422), abnormally
slow loss decay. Hyperparameter tuning on a broken feature schema cannot work.

**Decision:** Fix the feature schema first. Do not retrain until the extractor is correct.

---

## Part 6: Root Cause Analysis — Ranked by Impact

After the v5.3 kill, we systematically ranked all root causes by expected F1 impact:

| Rank | Root cause | Classes affected | Fix |
|------|-----------|-----------------|-----|
| 1 | Token truncation (96% contracts truncated at 512) | All | Windowed tokenization |
| 2 | `return_ignored` always 0 (lvalue bug) | MishandledException, UnusedReturn | Phase 0 fix ✅ |
| 3 | DoS/Reentrancy 99% co-occurrence | Reentrancy, DoS | Phase 4 augmentation |
| 4 | `block.timestamp` invisible | Timestamp, TOD | Phase 0 fix ✅ |
| 5 | Transfer/Send invisible in ext_calls + CFG | DoS, Reentrancy | Phase 0 fix ✅ |
| 6 | `loc` not normalized (2538× scale imbalance) | All (attention quality) | Phase 0 fix ✅ |
| 7 | Fusion LR dominance (RC1) | Reentrancy | Config fix ✅ |
| 8 | DoS data starvation (257 positives) | DoS | Phase 4 augmentation |
| 9 | GNN 1-hop CF insufficient for CEI detection | Reentrancy | Phase 2 (2nd CF layer) ✅ |
| 10 | Classifier single linear layer | All | Phase 2 (hidden layer) ✅ |
| 11 | GNN hidden_dim=128 too narrow | All | Phase 2 (256-dim) ✅ |
| 12 | LoRA under-parameterized | Text-dependent classes | Phase 2 (r=16, deferred to r=32) |

Fixes marked ✅ are implemented. Phase 4 augmentation is planned but not yet executed.

---

## Part 7: The v6 Decision — What We Chose to Do and Why

After ranking root causes, we designed v6 as a six-phase plan. Every decision had an explicit
rationale:

### Phase 0: Fix the Feature Schema (do first — everything depends on this)

**Decision:** Fix all four code bugs before touching any architecture or hyperparameters.

**Why first:** Training on graphs where `return_ignored` is always 0 means MishandledException
and UnusedReturn have near-zero feature signal regardless of model capacity. Adding more GNN
layers cannot compensate for corrupt input features. The feature bugs are bugs — they produce
incorrect ground truth features. No amount of architecture improvement can overcome that.

**What changed in `graph_extractor.py`:**
- `_compute_return_ignored()`: rewrote to check if `id(lvalue)` appears in any subsequent op's
  `read` set, rather than checking `lvalue is None` (which is never true)
- `_compute_external_call_count()`: added `Transfer, Send` to the isinstance check
- `_cfg_node_type()`: added `Transfer, Send` to the CALL priority check
- `_compute_uses_block_globals()`: new function scanning raw IR ops for `SolidityVariableComposed`
- `_build_node_features()`: feat[2] = `uses_block_globals` (was `pure`); feat[6] = log1p-normalized
- `FEATURE_SCHEMA_VERSION` bumped to `"v4"` in `graph_schema.py`

**Why `uses_block_globals` replaces `pure`:** The `pure` feature is only 1 for functions that
cannot write state. These functions are almost never vulnerable by construction. It provides
signal only as a negative indicator (pure=1 → not vulnerable) but contributes near-zero gradient
on positive examples. `uses_block_globals` replaces it with a direct Timestamp/TOD signal.

### Phase 1: Windowed Tokenization (do before re-tokenization)

**Decision:** Implement max_windows=4, stride=256, always pad to [4, 512] shape.

**Why max_windows=4 not 8:** 8GB VRAM. With batch=8 and max_windows=8, each training step
processes 64 CodeBERT forward passes. At max_windows=4 we process 32 passes. The difference
is not linear in VRAM because activations accumulate, but empirically 4 windows stays
comfortably within 8GB while 8 windows would likely OOM during training.

**Why always pad to max_windows:** PyTorch's DataLoader `torch.stack()` requires all batch
elements to have identical shapes. Variable W per contract (W=1 for a 50-token contract,
W=4 for a 3000-token contract) would crash collation. The solution is to always produce
exactly max_windows windows, with zero-attention-mask padding for unused windows. The
`key_padding_mask` in CrossAttentionFusion correctly zeroes out padding window contributions.

**Why stride=256 (50% overlap):** Tokens at window boundaries appear in two adjacent windows.
This prevents vulnerability patterns that span a window boundary from being cut in half. For
a function that starts at token 480 and ends at token 550, the function appears fully in
both window 0 (tokens 0–511) and window 1 (tokens 256–767).

**What changed:**
- `transformer_encoder.py`: `forward()` handles `[B,L]` (legacy) and `[B,W,L]` (windowed);
  added `WindowAttentionPooler` class
- `sentinel_model.py`: `forward()` builds `flat_mask = [B, W*L]` for CrossAttentionFusion;
  `self.window_pooler` replaces `token_embs[:, 0, :]` for transformer eye
- `dual_path_dataset.py`: shape validation accepts `[512]` or `[max_windows, 512]`
- `retokenize_windowed.py`: new script producing windowed token files

### Phase 2: GNN Architecture Expansion

**Decision:** Expand GNN to 256-dim, 6-layer with two second hops; add classifier hidden layer;
implement WindowAttentionPooler.

**Why hidden_dim 128 → 256:** The GNN must encode 13 node types, 8 edge types, 12 feature
dimensions, and multi-hop topological context into a single embedding vector. At 128 dims,
the attention computation for Phase 2 (heads=1, full 128-dim) is working with a very narrow
representation for discriminating CALL from WRITE nodes in CFG ordering tasks. 256-dim provides
2× the representational capacity for essentially no VRAM cost (GNN activations are tiny vs
CodeBERT activations).

**Why 6 layers (2 per phase) instead of 4:**
- **Phase 2: 1 CF hop was insufficient for CEI detection.** A typical reentrancy function:
  `ENTRY → CHECK → CALL → TMP → WRITE → RETURN`. With 1 CF hop, the CALL node's message
  reaches TMP but not WRITE. The model cannot see "CALL precedes WRITE" because the signal
  stops at the intermediate TMP node. With 2 CF hops, CALL's message reaches WRITE via TMP.
  This is the minimum depth needed to detect CEA (Checks-Effects-Attack) vs CEI ordering.
- **Phase 3: 2 RC hops allows deeper signal propagation.** REVERSE_CONTAINS edges go from
  child nodes upward to their CONTAINS parent (CFG node → function → contract). 2 RC hops
  allows deeper CFG nodes to influence the contract-level embedding.

**Why classifier hidden layer [384→192→10]:** The single linear layer `Linear(384, 10)` cannot
model non-linear interactions between the three eyes. With the hidden layer, the model can learn
"GNN detects call-before-write AND TF detects 'transfer' keyword → boost Reentrancy" as a
non-linear AND relationship. Added ~76K parameters — negligible.

**Why WindowAttentionPooler:** The previous transformer eye used `token_embs[:, 0, :]` — the
CLS token of window 0. This only represents the first 512 tokens (pragma, imports, first few
functions). With 4 windows, there are 4 CLS tokens. The pooler learns per-window attention
weights and combines them. For a contract where the vulnerability is in window 2 (function 5–8),
window 2's CLS gets a higher weight than window 0's CLS.

### Phase 3: Training Configuration

**Decision:** ASL loss, 100 epochs, patience=30, LoRA LR 0.3×.

**Why AsymmetricLoss instead of BCE:**
- BCE treats all (sample, class) cells equally: 440K cells, 85%+ are negative (label=0).
  The optimizer spends most gradient budget suppressing easy negatives (DoS=0 on a safe contract)
  rather than learning positive patterns.
- ASL with `gamma_neg=4, gamma_pos=1, clip=0.05`:
  - Easy negatives (p≈0.05, y=0): `(0.05)^4 × log(0.95) ≈ 0` — zero gradient
  - Hard negatives (p=0.4, y=0): `(0.4)^4 × log(0.6) ≈ 0.013` — moderate gradient
  - Positives (y=1): `(1-p)^1 × log(p)` — full focus on genuine positives
- The `clip=0.05` margin shift: any negative with p < 0.05 contributes zero to the loss,
  eliminating trivial negatives from gradient computation entirely.

**Why we did NOT fix the pos_weight_min_samples mistake with another pos_weight change:**
ASL subsumes the positive weighting mechanism. With gamma_neg=4, the effective weight ratio
between positives and negatives is already implicit in the asymmetric focusing. Adding
pos_weight on top would double-count the correction. We set `pos_weight_min_samples=0`
(disabled) for v6 and let ASL handle class imbalance.

**Why 100 epochs / patience=30:** The deeper architecture (256-dim, 6-layer GNN, windowed
CodeBERT) has more parameters to coordinate. At 973 steps/epoch (31K samples / batch 32),
100 epochs = 97K optimizer steps. Early v5.x runs were stopped too early by patience=10 when
the true metric was still improving.

**Why LoRA LR 0.3× (reduced from 0.5×):** With 4 windows, the LoRA weights receive 4× more
gradient signal per contract (one forward pass per window). Without compensating LR reduction,
LoRA would effectively train 4× faster than in single-window mode. Reducing to 0.3× keeps
LoRA update magnitude comparable to v5.2 single-window training on a per-contract basis.

---

## Part 8: What We Built — Concrete Code Changes

### Phase 0 — Feature Schema v4

**Files changed:** `ml/src/preprocessing/graph_extractor.py`, `ml/src/preprocessing/graph_schema.py`  
**Commits:** `bef1f2a` (return_ignored fix), `310e738` (all remaining schema fixes)

| What | Before | After |
|------|--------|-------|
| `return_ignored` | `lvalue is None` (always False) | ID-based check across all subsequent IR ops |
| `ext_calls` count | HighLevelCall + LowLevelCall only | + Transfer + Send |
| CFG node typing | Transfer typed as CFG_NODE_WRITE | Transfer typed as CFG_NODE_CALL |
| feat[2] | `pure` (function is pure) | `uses_block_globals` (reads block.timestamp/etc.) |
| feat[6] `loc` | raw integer [0, 2538] | `log1p(loc)/log1p(1000)` clamped to [0, 1] |
| `FEATURE_SCHEMA_VERSION` | `"v3"` | `"v4"` |

### Phase 1 — Windowed Tokenization

**Files changed:** `transformer_encoder.py`, `sentinel_model.py`, `dual_path_dataset.py`,
`train.py`  
**Files created:** `retokenize_windowed.py`  
**Commit:** `b38c9da`

Key architectural change in `TransformerEncoder.forward()`:
```python
# Before: [B, 512] → CodeBERT → [B, 512, 768]
# After:  [B, W, 512] → reshape → [B*W, 512] → CodeBERT → reshape → [B, W*512, 768]
```

Key change in `SentinelModel.forward()`:
```python
# Build flat_mask for CrossAttentionFusion
if input_ids.dim() == 3:
    flat_mask = attention_mask.view(B, W * L)   # [B, W*L] for fusion key_padding_mask
else:
    flat_mask = attention_mask                   # [B, L] unchanged
```

### Phase 2 — GNN Architecture Expansion

**Files changed:** `gnn_encoder.py`, `sentinel_model.py`, `transformer_encoder.py`,
`trainer.py`, `train.py`, `reextract_graphs.py`  
**Commit:** `2bb0e16`

**GNN changes (6-layer, 256-dim):**
```python
# New convolutions added
self.conv3b = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                      add_self_loops=False, edge_dim=edge_emb_dim)  # 2nd CF hop
self.conv4b = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                      add_self_loops=False, edge_dim=edge_emb_dim)  # 2nd RC hop

# Phase 2 forward (with 2 CF hops)
x2 = self.conv3(x, cfg_ei, cfg_ea);  x = x + self.dropout(self.relu(x2))
x2 = self.conv3b(x, cfg_ei, cfg_ea); x = x + self.dropout(self.relu(x2))
x  = self.phase_norm[1](x)

# Phase 3 forward (with 2 RC hops)
x2 = self.conv4(x, rev_ei, rev_ea);  x = x + self.dropout(self.relu(x2))
x2 = self.conv4b(x, rev_ei, rev_ea); x = x + self.dropout(self.relu(x2))
x  = self.phase_norm[2](x)
```

**Classifier hidden layer:**
```python
# Before
self.classifier = nn.Linear(3 * eye_dim, num_classes)

# After
self.classifier = nn.Sequential(
    nn.Linear(3 * eye_dim, 192),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(192, num_classes),
)
```

**WindowAttentionPooler (in `transformer_encoder.py`):**
```python
class WindowAttentionPooler(nn.Module):
    def __init__(self, hidden_dim=768, window_size=512):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, token_embs):          # [B, W*L, 768]
        B, WL, D = token_embs.shape
        if WL <= self.window_size:
            return token_embs[:, 0, :]      # single-window fallback
        W = WL // self.window_size
        cls_indices = torch.arange(W) * self.window_size
        window_cls = token_embs[:, cls_indices, :]   # [B, W, 768]
        weights = torch.softmax(self.attn(window_cls), dim=1)  # [B, W, 1]
        return (weights * window_cls).sum(dim=1)     # [B, 768]
```

**Default updates:**

| Parameter | v5.x | v6 |
|-----------|------|-----|
| `gnn_hidden_dim` | 128 | **256** |
| `gnn_layers` | 4 | **6** |
| `gnn_edge_emb_dim` | 32 | **64** |
| `MODEL_VERSION` | `"v5.2"` | `"v6.0"` |

### Phase 3 — Training Configuration

**Files changed:** `trainer.py`, `train.py`  
**Files created:** `ml/src/training/losses.py`  
**Commit:** `64dfc5a`

**AsymmetricLoss (`losses.py`):**
```python
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
        ...
    def forward(self, logits, labels):
        prob = torch.sigmoid(logits.float())
        prob_neg = (prob - self.clip).clamp(min=0.0)
        loss_pos = -labels       * (1.0 - prob) ** self.gamma_pos * torch.log(prob.clamp(1e-8))
        loss_neg = -(1 - labels) * prob_neg ** self.gamma_neg * torch.log((1-prob_neg).clamp(1e-8))
        return (loss_pos + loss_neg).mean()
```

**Config defaults updated:**

| Setting | v5.x | v6 |
|---------|------|-----|
| `epochs` | 60 | **100** |
| `early_stop_patience` | 10 | **30** |
| `lora_lr_multiplier` | 0.5 | **0.3** |
| `loss_fn` | `"bce"` | `"bce"` (default; use `--loss-fn asl` for v6 run) |
| `_VALID_LOSS_FNS` | bce, focal | bce, focal, **asl** |

---

## Part 9: Current State (2026-05-16 22:46)

### Code — All Committed

| Commit | Contents |
|--------|----------|
| `bef1f2a` | return_ignored fix |
| `310e738` | Schema v4: uses_block_globals, loc normalization, Transfer/Send |
| `b38c9da` | Phase 1: windowed tokenization (TransformerEncoder, SentinelModel, DualPathDataset, retokenize_windowed.py) |
| `2bb0e16` | Phase 2: GNN 256-dim/6-layer, conv3b/conv4b, WindowAttentionPooler, classifier hidden layer |
| `64dfc5a` | Phase 3: AsymmetricLoss, epochs=100, patience=30, LoRA LR 0.3× |

No uncommitted code changes. All architecture and training config changes for v6 are in place.

### Data Pipeline — In Progress

| Step | Status |
|------|--------|
| Graph re-extraction (v4 schema) | ⏳ RUNNING — PID 187896, 51% complete @ 22:46 |
| Validate re-extracted graphs | ⏳ waiting for re-extraction |
| Windowed re-tokenization | ⏳ waiting for validation |
| Rebuild cache (`cached_dataset_windowed.pkl`) | ⏳ waiting for retokenization |
| v6.0 training launch | ⏳ waiting for cache |

### Re-extraction Details

The current re-extraction was launched from `multilabel_index_deduped.csv` (44,420 contracts),
producing graphs with `FEATURE_SCHEMA_VERSION="v4"`. At 21 contracts/second it will complete
in approximately 20–25 minutes from now.

An earlier accidental re-extraction from the 68K leaky index (PID 184883) was identified and
killed. Only PID 187896 (deduped index) is running.

### Commands to Run After Re-extraction Completes

```bash
# Step 1: Validate (gate: 0 errors, ghost graphs ≤ 100)
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py \
    --check-dim 12 --check-edge-types 8 \
    --check-contains-edges --check-control-flow

# Step 2: Windowed retokenization (~1–2 hours)
PYTHONPATH=. python ml/scripts/retokenize_windowed.py \
    --input ml/data/processed/multilabel_index_deduped.csv \
    --output ml/data/tokens_windowed --max-windows 4 --workers 11

# Step 3: Rebuild cache
PYTHONPATH=. python ml/scripts/create_cache.py \
    --graphs-dir ml/data/graphs \
    --tokens-dir ml/data/tokens_windowed \
    --label-csv ml/data/processed/multilabel_index_deduped.csv \
    --output ml/data/cached_dataset_windowed.pkl --workers 8

# Step 4: Launch v6 training
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v6.0-20260517 \
    --experiment-name sentinel-v6 \
    --tokens-dir ml/data/tokens_windowed \
    --cache-path ml/data/cached_dataset_windowed.pkl \
    --loss-fn asl \
    --gradient-accumulation-steps 8 \
    --label-smoothing 0.05
```

All other v6 values (gnn_hidden_dim=256, gnn_layers=6, edge_emb_dim=64, epochs=100,
patience=30, lora_lr=0.3×, asl_gamma_neg=4.0, asl_gamma_pos=1.0, asl_clip=0.05)
are now the TrainConfig defaults — no flags needed.

---

## Part 10: What We Expect from v6 and Why

### Per-Class Targets

| Class | v5.2 F1 | v6 Target | Primary mechanism |
|-------|---------|-----------|------------------|
| IntegerUO | 0.732 | ≥ 0.75 | Windowed tokens (arithmetic spread through contract); larger GNN |
| GasException | 0.407 | ≥ 0.45 | 2nd CF hop captures gas-loop patterns; windowed tokenization |
| Reentrancy | 0.322 | ≥ 0.40 | ASL removes easy-negative dominance; 2nd CF hop for CEI/CEA; LoRA LR corrected |
| MishandledException | 0.342 | ≥ 0.50 | `return_ignored` fix (was always 0 — primary feature now active) |
| UnusedReturn | 0.238 | ≥ 0.45 | `return_ignored` fix (same cause) |
| Timestamp | 0.174 | ≥ 0.30 | `uses_block_globals` feature (was invisible) + windowed tokenization |
| DenialOfService | 0.329 | ≥ 0.35 | Transfer/Send now in ext_calls + CFG; minimal without augmentation |
| CallToUnknown | 0.284 | ≥ 0.35 | Windowed tokenization exposes more delegatecall/staticcall patterns |
| TOD | 0.283 | ≥ 0.30 | `uses_block_globals` captures block.number; windowed tokenization |
| ExternalBug | 0.262 | ≥ 0.30 | Cleaner gradients from ASL; 256-dim GNN depth |
| **Macro avg** | **0.3422** | **≥ 0.43** | All above combined |

### Behavioral Gates (Primary Pass/Fail)

Val F1 is a necessary but insufficient condition. The primary gate is behavioral:

```bash
python ml/scripts/manual_test.py
```

- **Detection rate ≥ 80%:** correctly identify vulnerability class in known-vulnerable contracts
- **Safe specificity ≥ 80%:** correctly predict safe for contracts with no known vulnerabilities

v5.2 achieved 36% detection / 33% specificity. v5.0 achieved 15% detection / 0% specificity.
The behavioral gates expose what val F1 hides: a model that fires on every contract scores well
on val when the val set is class-heavy but fails completely on safe-contract specificity.

### What We Believe Will Work

1. **return_ignored fix** should immediately improve MishandledException and UnusedReturn.
   The feature was zero for 100% of contracts. It will now correctly fire for any contract
   with an unchecked return value. These two classes should see the largest F1 improvement.

2. **uses_block_globals** should improve Timestamp from 0.174 to at least 0.25. It gives the
   GNN a direct boolean signal for "this function reads block.timestamp". Previously the GNN
   had zero node features for this.

3. **Windowed tokenization** should improve all classes modestly and help most for Timestamp
   and large-contract vulnerability classes that appear in the truncated portion of the file.

4. **ASL** should reduce the Reentrancy false-positive rate by de-weighting easy negatives.
   The model currently fires Reentrancy on safe contracts because BCE equally weighted the
   abundant easy-negative "safe contract, no Reentrancy" examples, causing the model to learn
   a conservative threshold. ASL reduces this pressure.

### What May Still Struggle

1. **DenialOfService:** 257 training positives is fundamentally insufficient for reliable
   gradient learning. The Transfer/Send fix helps but the class still needs ~500 augmented
   clean DoS-only examples to break the 99% DoS→Reentrancy co-occurrence. Without Phase 4
   augmentation, DoS F1 may remain below 0.35 even with all other fixes.

2. **Behavioral specificity on Reentrancy:** The 99% DoS→Reentrancy co-occurrence means
   the model has seen DoS and Reentrancy together in 99% of DoS training cases. Even with
   ASL, the model will have a strong prior to predict both together. Without augmentation
   adding clean "DoS without Reentrancy" and "safe with external calls" examples, the
   false-positive Reentrancy rate on safe DeFi contracts may persist.

3. **TOD:** Transaction-ordering dependency has no strong structural graph signal. The primary
   hope is CodeBERT text signal (block.number appears in token stream + windowed tokenization
   exposes more of the contract). The uses_block_globals feature helps but TOD patterns are
   subtle enough that text-only detection may cap F1 around 0.30.

### If v6 Training Fails Behavioral Gates

If v6 achieves good val F1 (≥ 0.43) but fails behavioral gates:

1. Check Reentrancy specifically: if it fires on every contract → remaining co-occurrence issue
   → proceed to Phase 4 augmentation immediately
2. Check DoS: if threshold=0.95+ for any F1 → starvation persists → Phase 4 augmentation
3. Check MishandledException and UnusedReturn: if still near v5.2 F1 → the re-extraction may
   have not reflected the return_ignored fix → validate spot-check some graphs directly

If val F1 is below 0.43 but converging:
- Extend patience (more epochs)
- Check GNN gradient share at epoch 1 (should be ≥ 15%)
- Check JK weight distribution (all three phases should contribute > 5%)

---

## Part 11: Key Files Reference

| Purpose | Path |
|---------|------|
| Feature extractor | `ml/src/preprocessing/graph_extractor.py` |
| Feature schema constants | `ml/src/preprocessing/graph_schema.py` |
| GNN encoder | `ml/src/models/gnn_encoder.py` |
| Transformer encoder | `ml/src/models/transformer_encoder.py` |
| Full model | `ml/src/models/sentinel_model.py` |
| CrossAttention fusion | `ml/src/models/fusion_layer.py` |
| Dataset loader | `ml/src/datasets/dual_path_dataset.py` |
| Trainer | `ml/src/training/trainer.py` |
| ASL loss | `ml/src/training/losses.py` |
| Graph re-extraction | `ml/scripts/reextract_graphs.py` |
| Windowed tokenizer | `ml/scripts/retokenize_windowed.py` |
| Graph validation | `ml/scripts/validate_graph_dataset.py` |
| Cache builder | `ml/scripts/create_cache.py` |
| Training launcher | `ml/scripts/train.py` |
| Threshold tuning | `ml/scripts/tune_threshold.py` |
| Behavioral test | `ml/scripts/manual_test.py` |
| Deduped CSV | `ml/data/processed/multilabel_index_deduped.csv` |
| Deduped splits | `ml/data/splits/deduped/` |
| Graph files (v4 schema) | `ml/data/graphs/*.pt` (re-extraction in progress) |
| Windowed tokens | `ml/data/tokens_windowed/*.pt` (pending retokenization) |
| Cache (windowed) | `ml/data/cached_dataset_windowed.pkl` (pending) |
| Fallback checkpoint | `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` |
| MLflow database | `mlruns.db` |

---

## Part 12: Lessons Summary

1. **Feature bugs compound silently.** `return_ignored=0` for 100% of contracts means two
   classes were blind in training for months. Always verify feature non-triviality before
   training (histogram of each feature across classes).

2. **Val F1 is a necessary but insufficient metric.** v5.0 had F1=0.5828 and 0% behavioral
   specificity. Always run manual_test.py as the final gate.

3. **Dataset leakage inflates metrics by 20–30%.** An F1 of 0.5422 on leaky data corresponds
   to roughly 0.34 on clean data. Never publish or gate on leaky metrics.

4. **Co-occurrence is a data problem, not a model problem.** The 99% DoS→Reentrancy link
   cannot be fixed by loss functions or architecture. It requires augmentation data that breaks
   the co-occurrence signal.

5. **Token truncation at 512 is catastrophic for long contracts.** 96% of contracts are
   truncated. Windowed tokenization is not optional for meaningful text-path signal.

6. **Loss function tuning on corrupt features is wasted effort.** v5.3's pos_weight_min_samples
   experiment failed because the underlying feature schema was broken. Fix the root cause first.

7. **GNN gradient collapse is detectable and fixable.** Per-group LR multipliers, JK connections,
   and per-phase LayerNorm together keep the GNN from collapsing. Monitor GNN gradient share
   at epoch 1 — it should be ≥ 15%.

8. **eval_threshold during training must match expected inference probability range.** Using
   threshold=0.5 when minority class probabilities cluster at 0.35–0.45 creates noise that
   triggers premature early stopping.
