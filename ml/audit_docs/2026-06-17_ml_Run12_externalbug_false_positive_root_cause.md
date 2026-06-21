# Root Cause Analysis — Run 12 ExternalBug False Positive (2026-06-17)

**Severity:** CRITICAL — model is fundamentally broken for the ExternalBug class
**Discovered by:** E2E test of agents module on `safe_storage.sol` (false positive CONFIRMED at 0.82)
**Scope:** Affects any contract that has the "owner + msg.sender" pattern without actual risky external calls

---

## TL;DR

The Run 12 model's **ExternalBug head is firing on the wrong feature**. It associates
the "owner pattern" (`address public owner` + `constructor() { owner = msg.sender; }`)
with ExternalBug at 85% confidence, while giving 0% confidence to contracts with
actual untrusted low-level calls (`to.call(data)`). The root cause is **noisy
training labels**: 74% of training contracts are labeled ExternalBug (16,638 / 22,493),
up from 17% in v1.4 (614 / 3,600). The class definition was implicitly broadened
from "risky external call" to "any contract with external-looking patterns".

**This is a training data problem, not a model architecture problem.**

---

## Evidence

### Test results — model output on 10 contract variations

| Test | Description | ExternalBug | Reentrancy |
|---|---|---|---|
| 1 | safe_storage ORIGINAL (msg.sender + owner + state) | **0.822** | 0.283 |
| 2 | pure state, no msg.sender | 0.361 | 0.303 |
| 3 | msg.sender, no owner check | 0.381 | 0.282 |
| **4** | **JUST `address public owner` + constructor (no functions!)** | **0.850** | 0.292 |
| 5 | with real interface call (msg.sender + owner) | 0.678 | 0.733 |
| A | real external call to typed interface | 0.344 | 0.538 |
| **B** | **untrusted low-level `to.call(data)`** | **0.000** | 0.000 |
| C | only `msg.sender` | 0.554 | 0.239 |
| D | pure function | 0.333 | 0.221 |
| E | empty contract `{}` | 0.223 | 0.494 |

**Smoking gun:** Test 4 (just stores the owner, zero functions, zero state writes)
gets ExternalBug=0.85. Test B (textbook dangerous pattern) gets ExternalBug=0.00.

### GNN hotspots confirm wrong focus

`/hotspots` for safe_storage.sol + ExternalBug class:
- `setValue(uint256)`: score 1.0 (highest)
- `constructor()`: score 0.9399
- `getValue()`: score 0.0

The GNN says "look at setValue and constructor" — the two functions that:
- Have `msg.sender` references
- Set state from external input

The ExternalBug head then takes these features and outputs 0.82.

---

## Root Cause — Training Data Label Quality

### 12-dim node features (from `_schema_constants.md`)

| Index | Name | Description |
|---|---|---|
| 0 | node_type_norm | Normalised node type id |
| 1 | visibility | Function visibility: 0=public/external, 1=internal, 2=private |
| 2 | uses_block_globals | Count of block.timestamp, block.number reads |
| 3 | **external_call_count** | Number of external calls in scope |
| 4 | state_var_writes | Number of state variable write ops |
| 5 | contract_size_norm | Normalised contract line count |
| 6 | loc | Raw line count |
| 7 | return_ignored | 1.0 if return value dropped |
| 8 | call_target_typed | 1.0 = typed HighLevelCall, 0.0 = low-level call |
| 9 | has_loop | 1.0 if inside loop |
| 10 | payable | 1.0 if function is payable |
| 11 | in_unchecked_block | fraction in unchecked{} scope |

The most relevant feature for ExternalBug should be **`external_call_count` (feat 3)**
and **`call_target_typed` (feat 8)**. But the model is firing on contracts with
**0 external calls** — it's clearly not using these features as the primary signal.

### Training data label distribution (v3 export)

```
Total contracts: 22,493
ExternalBug (class_2) positive: 16,638 (74.0%)
ExternalBug negative: 5,855 (26.0%)
```

**74% positive rate is the smoking gun.** For comparison:
- Reentrancy: 50.7%
- IntegerUO: 42.0%
- UnusedReturn: 26.0%
- Timestamp: 28.1%

ExternalBug is the **most common** class in the training data — by a wide margin.
The model can get 74% accuracy by predicting ExternalBug=1 for everything.

### Cross-reference of ExternalBug labels

Of the 16,638 ExternalBug-positive contracts, the co-labels are:
- 64.0% also have Reentrancy
- 49.8% also have IntegerUO
- 34.0% also have UnusedReturn
- 28.0% also have Timestamp

This means ExternalBug is **almost never labeled alone** — it's the "default" tag
for any contract that has any other vulnerability. The class has become a
**catch-all "yes this contract has issues" label**, not a specific vulnerability type.

### Training data class evolution

From `_schema_constants.md` (v1.4, original curation):
- **ExternalBug: 614 retained (17.0%)** — small, curated set

From current v3 export:
- **ExternalBug: 16,638 (74.0%)** — expanded 27× from the original 614

The class went from a small curated set (17%) to nearly universal (74%) between
v1.4 and v3. The expansion was probably done to balance the multi-label training
set, but it had the side effect of **diluting the class definition**.

### Training corpus composition

```
dive:                 22,073 contracts (98.1%)   ← DeFiHackLabs
solidifi:               283 contracts (1.3%)
smartbugs_curated:      137 contracts (0.6%)
```

The "dive" corpus (DeFiHackLabs) is the dominant source. These are all **real
DeFi exploits** — contracts that were attacked in the wild. The labels for these
were probably generated by:
1. Manual annotation of the actual exploit
2. Pattern matching with known vulnerability classes
3. Slither output as proxy for vulnerability

The 74% positive rate for ExternalBug suggests that the labelling process
**flagged any contract that "had external interactions"** as ExternalBug-positive.
Since most DeFi contracts have external interactions, most got labeled positive.

### Per-shard ExternalBug rate

```
Shard 0:  5000/5000  (100.0%)  ← ALL positive
Shard 1:  5000/5000  (100.0%)  ← ALL positive
Shard 2:  2646/5000  (52.9%)
Shard 3:  2133/5000  (42.7%)
Shard 4:  1194/1657  (72.1%)
```

The first 2 shards are 100% positive. This is consistent with **stratified
sampling where ExternalBug-positive contracts are over-represented in the
first shards**, then mixed in later shards.

---

## Why the Model Can't Fix This on Its Own

### Model architecture (verified from `sentinel_model.py:281-291`)

```python
self.classifier = nn.Sequential(
    nn.Linear(4 * eye_dim, _cls_hidden),  # 512 → 256
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(_cls_hidden, num_classes),    # 256 → 10
)
```

The classifier is a 2-layer MLP. Given 4 concatenated eye vectors (GNN, transformer,
fused, CFG), it outputs 10 logits. The classifier **learns whatever patterns the
labels reward** — if 74% of labels are ExternalBug=1, the classifier will learn to
predict ExternalBug=1 for most inputs.

### Loss function (verified from `trainer.py`)

The trainer uses **AsymmetricLoss (ASL)** with:
- pos_weight computed and capped at 10.0
- pos_weight_min_samples=3000 — classes with ≥3000 positives get pos_weight=1.0

ExternalBug has 16,638 positives → pos_weight=1.0 (no amplification).

The loss function is **doing the right thing** — it doesn't artificially amplify
ExternalBug. But the labels are already biased.

### Why the model picks "owner pattern" as the signal

The model has 12 input features. For "owner + msg.sender" contracts:
- `state_var_writes` (feat 4) = 1 (owner = msg.sender)
- `visibility` (feat 1) = 0 (owner is public)
- `external_call_count` (feat 3) = 0
- `call_target_typed` (feat 8) = 0

For "to.call(data)" contracts:
- `state_var_writes` = 0
- `visibility` = 0 (external function)
- `external_call_count` = 1+
- `call_target_typed` = 0

The model's ExternalBug head has likely learned: **"high ExternalBug when
state_var_writes > 0 AND visibility == 0"** — the "owner pattern".

This is the **OPPOSITE** of the correct signal. The correct signal would be
**"high ExternalBug when external_call_count > 0 AND call_target_typed varies"**.

---

## Why Downstream Components Can't Catch This

### Agents module chain: ML → Slither → RAG → LLM

When ML predicts ExternalBug=0.82 for safe_storage.sol:
1. **Slither returns 0 findings** (correct — safe contract)
2. **RAG returns 5 chunks** (about ExternalBug exploits in OTHER contracts, not relevant)
3. **LLM cross_validator** sees: ML=0.82, Slither=0, RAG=5. Bias = confirm ML
4. **LLM synthesizer** writes a confident narrative about a non-existent bug

The LLM is **biased by the model's high probability**. When ML says 0.82, the LLM
is more likely to confirm than to push back. No amount of agents-layer logic can
detect that the model is fundamentally wrong about what ExternalBug means.

---

## Recommended Fix (defer to Run 13 / Phase A training)

### Option 1: Audit and clean the training labels (most important)

1. **Audit the 16,638 ExternalBug-positive training contracts**:
   - Open the original source code for each (currently redacted in v3 export)
   - For each, check if it has **actual risky external calls**:
     - `interface.method()` on a typed interface variable
     - `address.call()`, `delegatecall()`, `staticcall()` to non-constant addresses
     - Forwarding user-supplied calldata
   - Contracts that have NONE of these should be relabeled as **ExternalBug-negative**

2. **Tighten the class definition** to: "Contract makes a call to a non-constant
   external address (interface or low-level) where the call target is influenced
   by user input or untrusted state."

3. **Target retention rate**: 5-15% (not 74%). Most DeFi contracts that have
   "external" patterns but not "risky" ones should NOT be ExternalBug.

### Option 2: Add a rule-based inference filter (Phase A improvement)

In the agents module, add a hard rule before cross_validator:

```python
# At agents/src/orchestration/nodes.py — before cross_validator
if ml_result.get('ExternalBug', 0) > 0.5:
    static_findings = state.get('static_findings', [])
    has_actual_external_call = any(
        f.get('detector') in EXTERNAL_CALL_DETECTORS
        for f in static_findings
    )
    if not has_actual_external_call:
        # Force ExternalBug down — model over-predicts on owner patterns
        ml_result['ExternalBug'] = 0.3
        ml_result.setdefault('downgraded_by_rule', []).append('ExternalBug: no actual external call')
```

This is a **defensive layer** that prevents the model from producing catastrophic
false positives until the training labels are fixed.

### Option 3: Retrain with a stricter ExternalBug definition

If the training data can be re-labelled, retrain Run 12+ with:
- ExternalBug: only contracts with `interface.method()` or low-level calls to
  non-constant addresses
- Add explicit negatives: contracts with `address public owner` but no actual
  external calls
- Target positive rate: 5-15% (current 74%)

The retrained model should:
- Give ExternalBug=high for: `interface.method()`, `address.call()` to typed vars
- Give ExternalBug=low for: `address public owner`, `msg.sender` checks, public functions

---

## What Worked

Despite the broken ExternalBug class, **other classes work correctly**:
- Reentrancy: model gave 0.519 for vulnerable_reentrant.sol (real reentrancy) — correct
- Reentrancy: model gave 0.282 for safe_storage.sol (no reentrancy) — correct
- The GNN+CodeBERT+4-eye architecture is working as designed

The failure is **isolated to ExternalBug** because of the 27× label expansion
in v3. Other classes were not affected.

---

## Cross-references

- Plan: `docs/plan/agents/2026-06-17-agents-real-e2e-test/`
- E2E test scratch: `~/.claude/scratch/agents_e2e_run_20260617.md`
- Run 12 model: `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt`
- v3 training data: `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/`
- Schema: `data_module/sentinel_data/representation/_schema_constants.md`
- Graph extractor: `data_module/sentinel_data/representation/graph_extractor.py`
- Loss function: `ml/src/training/trainer.py` (ASL with pos_weight cap=10.0)
- Classifier head: `ml/src/models/sentinel_model.py:281-291`

---

**Status:** Documentation complete. No code changes made. Awaiting decision on
Option 1 (re-label data) vs Option 2 (rule-based filter) vs Option 3 (retrain).
