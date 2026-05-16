# SENTINEL Model Failure — Root Cause Analysis

**Date:** 2026-05-16  
**Method:** Live Slither IR inspection of all 20 manual test contracts + CONTROL_FLOW topology analysis  
**Context:** v5.2 behavioral test: 7/19 = 36% detection; v5.3 training in progress

---

## Executive Summary

The v5.2 model fails on specific vulnerability classes because the graph feature
schema has fundamental dark spots — entire vulnerability patterns that are either
invisible in the feature vector or require multi-hop graph reasoning to detect.
Six root causes are identified: one was a code bug (now fixed), five are structural
schema limitations.

| Class | Root Cause | Direct signal? | Severity |
|-------|-----------|---------------|----------|
| Reentrancy (modern) | CEI ordering not in features; only in CF topology | Via GNN 2-hop CF | High |
| MishandledException | `return_ignored` always returned 0.0 (BUG) | **Fixed** ✓ | Critical |
| UnusedReturn | Same `return_ignored` bug | **Fixed** ✓ | Critical |
| Timestamp | `block.timestamp` is invisible (SolidityVariableComposed) | None | High |
| DenialOfService | ETH `transfer()` inside loops invisible | None | High |
| IntegerUO (0.4.x) | `in_unchecked` only catches Solidity 0.8.x explicit blocks | None | Medium |

---

## Bug Fix: `_compute_return_ignored()` Always Returned 0.0

### Root cause

The original code checked `op.lvalue is None` to detect discarded return values:

```python
for op in func.slithir_operations:
    if isinstance(op, (LowLevelCall, HighLevelCall)):
        if op.lvalue is None:   # ← NEVER TRUE
            return 1.0
```

Slither **always** creates a TupleVariable or TemporaryVariable for every call's
return value — even when the programmer ignores it. The distinction is not whether
`lvalue is None`, but whether the lvalue variable is ever *read* in subsequent IR.

**Contract 06 (MishandledException):**
```solidity
r.call{value: amount}("");   // ← return completely discarded
```
Slither IR: `TUPLE_0 = r.call(amount)` — lvalue is not None, but TUPLE_0 is never
referenced in any subsequent op.

**Contract 08 (UnusedReturn):**
```solidity
token.transfer(vault, amount);   // ← TemporaryVariable TMP_1, never read
token.approve(spender, amount);  // ← TemporaryVariable TMP_2, never read
```

**Contract 12 (SafeVault — SAFE):**
```solidity
(bool ok,) = msg.sender.call{value: amount}("");
require(ok, "transfer failed");   // ← TUPLE_0 IS read → correctly returns 0.0
```

### Fix applied

`ml/src/preprocessing/graph_extractor.py` — `_compute_return_ignored()`:

```python
# Collect all id(variable) that appear in any op's read list
all_read_vars: set = set()
for node in (getattr(func, "nodes", None) or []):
    for op in (getattr(node, "irs", None) or []):
        for rv in (getattr(op, "read", None) or []):
            all_read_vars.add(id(rv))

# Call is ignored if lvalue never appears in any read
for node in (getattr(func, "nodes", None) or []):
    for op in (getattr(node, "irs", None) or []):
        if isinstance(op, (LowLevelCall, HighLevelCall)):
            lval = op.lvalue
            if lval is None or id(lval) not in all_read_vars:
                return 1.0
```

### Verified output after fix

| Function | Before | After | Correct? |
|----------|--------|-------|----------|
| `PayoutManager.flushAll()` | `ret_ign=0.0` | `ret_ign=1.0` | ✓ |
| `ReturnIgnorer.sweep()` | `ret_ign=0.0` | `ret_ign=1.0` | ✓ |
| `ReturnIgnorer.approveAndForget()` | `ret_ign=0.0` | `ret_ign=1.0` | ✓ |
| `ReturnIgnorer._tryNotify()` | `ret_ign=0.0` | `ret_ign=0.0` | ✓ (captures ok) |
| `SafeVault.withdraw()` | `ret_ign=0.0` | `ret_ign=0.0` | ✓ (require(ok)) |

### Impact

This fix requires re-extraction of all 44,420 graphs — the feature values in
existing .pt files are wrong for any function with discarded call returns.
`FEATURE_SCHEMA_VERSION` should be bumped to "v4" to invalidate the old cache.

---

## Dark Spot 1: CEI Ordering IS in the Graph (Reentrancy)

### What "CEI violation" looks like in the graph

```
Reentrancy (CEA — vulnerable):
[ENTRY] → [CFG_READ] → [CFG_OTHER] → [CFG_CALL] → [CFG_OTHER] → [CFG_WRITE]
                                          ↑ external call BEFORE state write → reentrancy possible

Safe (CEI — correct):
[ENTRY] → [CFG_READ] → [CFG_WRITE] → [CFG_OTHER] → [CFG_CALL] → [CFG_OTHER]
                            ↑ state write BEFORE external call → reentrancy impossible
```

The ordering IS encoded in the directed CONTROL_FLOW edges. The critical pattern:
- **Vulnerable**: `CFG_CALL →(CF)→ CFG_WRITE` — call precedes write
- **Safe**: `CFG_WRITE →(CF)→ CFG_CALL` — write precedes call

### What the GNN needs

The Phase 2 GAT layer (directed CONTROL_FLOW, heads=1, layer 3) in principle can
detect this pattern with 1 directed message-passing step:
- CFG_WRITE node aggregates messages from predecessors — if predecessor is CFG_CALL,
  the write "knows" a call came before it
- CFG_CALL node aggregates messages from predecessors — if predecessor is CFG_WRITE,
  the call "knows" a write came before it

Phase 3 (REVERSE_CONTAINS) then propagates CFG-level evidence back to the FUNCTION
node. After 2 total message hops (CF + RC), the function node "knows" its CEI status.

### Why the model fails anyway

Despite the structural signal being present, v5.2 cannot use it reliably due to:

1. **99% DoS↔Reentrancy co-occurrence**: In the training set, 99% of DoS samples also
   have the Reentrancy label (same .sol file in multiple BCCC directories). The model
   learns the DoS structural pattern (loop + array access) as a Reentrancy proxy.

2. **Identical function-level features**: At the function-node level (payable, ext_calls,
   call_target_typed), vulnerable and safe contracts are feature-identical:
   - `EtherBank.withdraw()`: call_target_typed=0.0, ext_calls=0.228
   - `SafeVault.withdraw()`: call_target_typed=0.0, ext_calls=0.228
   The CEI signal is entirely in the CFG subgraph, which requires the GNN Phase 2/3
   to propagate correctly.

3. **Reentrancy pos_weight was 2.82× in v5.2**: Over-emphasised Reentrancy label during
   training, causing the model to associate any external call with Reentrancy.
   Fixed in v5.3 (pos_weight_min_samples=3000 caps Reentrancy weight at 1.0).

### Contract-level comparison (all features identical at function level)

```
                           pay  ext_calls  call_typed  ret_ign  loops  CFG_CALL_before_WRITE?
EtherBank.withdraw()  [VUL]  0    0.228       0.0       0.0      0      YES
SafeVault.withdraw()  [SAFE] 0    0.228       0.0       0.0      0      NO
PullPayment.withdraw()[SAFE] 0    0.228       0.0       0.0      0      NO
```

---

## Dark Spot 2: `block.timestamp` is Completely Invisible

### The problem

`block.timestamp` in Slither IR is a `SolidityVariableComposed` — it is NOT a
user-defined state variable. Therefore:

- It does NOT appear in `func.state_variables_read`
- It creates NO `READS` edge to any node in the graph
- The `TimestampLottery` contract and `draw()` function look identical to any other
  contract at the graph feature level — no feature fires, no edge is added

### Verified

```python
# Slither output for TimestampLottery.draw():
func.state_variables_read = ['jackpot', 'unlockTime']  # block.timestamp NOT here
# IR scan finds:
op.read = [..., SolidityVariableComposed:block.timestamp]  # visible only in raw IR
```

### What signal the model has

None from the graph. The model must infer Timestamp vulnerability from:
- Lottery/game contract structural pattern (jackpot, prize, payable functions)
- Correlation with other features (payable=1, multiple state writes in draw())

This is an indirect correlation that will not generalise to Timestamp vulnerabilities
in non-lottery patterns.

### Proposed fix (requires schema bump + re-extraction)

Add a new feature `uses_block_globals` (replaces or augments an existing feature):

```python
def _compute_uses_block_globals(func: Any) -> float:
    """1.0 if function reads block.timestamp, block.number, or block.difficulty."""
    try:
        for op in func.slithir_operations:
            for rv in (getattr(op, 'read', None) or []):
                name = getattr(rv, 'name', '') or ''
                if 'timestamp' in name or 'number' in name or 'difficulty' in name:
                    if type(rv).__name__ == 'SolidityVariableComposed':
                        return 1.0
    except Exception:
        pass
    return 0.0
```

Cost: requires bumping FEATURE_SCHEMA_VERSION to "v4" and re-extracting all graphs.

---

## Dark Spot 3: ETH `transfer()` and `send()` are Invisible

### The problem

Slither classifies `address.transfer(amount)` and `address.send(amount)` as a special
`Transfer` IR operation — NOT `LowLevelCall` or `HighLevelCall`.

Consequences for the DoS contract (`distribute()`):

```solidity
for (uint256 i = 0; i < participants.length; i++) {
    payable(participants[i]).transfer(share);   // ← Transfer op, NOT LowLevelCall
}
```

- `func.low_level_calls = []` → `ext_calls = 0.0` (transfer not counted)
- `func.high_level_calls = []` → same
- CFG node with Transfer op: also contains `Index: REF_5 → participants[i]` which
  reads state variable `participants` → node typed as CFG_READ (not CFG_CALL)
- `call_target_typed = 1.0` (no raw address calls detected)

The entire "ETH transfer inside unbounded loop" pattern — the core DoS signal — is
invisible to the current feature schema.

### Verified Slither IR for `distribute()`:

```
[EXPRESSION] Index: REF_5(address) -> participants[i]
[EXPRESSION] TypeConversion: TMP_7 = CONVERT REF_5 to address
[EXPRESSION] Transfer: Transfer dest:TMP_7 value:share  ← Transfer op, not LowLevelCall
```

`func.calls_as_expressions` contains the transfer (expression-level) but none of
`high_level_calls`, `low_level_calls` do.

### DoS vs GasException: nearly identical graph topology

```
                     N   loop  ext_calls  CFG_CALL  WRITES_in_loop
distribute() [DoS]  10    1     0.000        0         YES (write pot=0 AFTER loop)
getUserTotal()[Gas] 8     1     0.000        0         NO
```

Both functions have loop=1, ext_calls=0, no CFG_CALL nodes. The model cannot
distinguish them from function-level features. It must learn from surrounding
contract topology (participants array pattern vs Record struct pattern).

### Proposed fix (requires schema bump + re-extraction)

1. Count `Transfer` and `Send` ops in `_compute_external_call_count()`:
   ```python
   from slither.slithir.operations import Transfer, Send
   n += len([op for op in func_ops if isinstance(op, (Transfer, Send))])
   ```

2. In `_cfg_node_type()`, classify nodes with Transfer/Send ops as CFG_CALL (priority
   above WRITE/READ):
   ```python
   if any(isinstance(op, (LowLevelCall, HighLevelCall, Transfer, Send)) for op in node.irs):
       return NodeType.CFG_NODE_CALL
   ```

This would make ETH transfers visible as external calls and give DoS a CFG_CALL
inside the loop — the correct signal.

---

## Summary: Feature Coverage by Vulnerability Class

| Vulnerability | Direct feature signal | Topological signal | Learnable? |
|--------------|----------------------|-------------------|-----------|
| Reentrancy (old .call.value) | `call_target_typed=0.0` | CF: CALL before WRITE | YES |
| Reentrancy (modern callback) | None | CALLS chain → CFG_CALL (3-4 hops) | HARD |
| Reentrancy (safe vs vuln) | Identical function features | CF ordering (2 hops) | POSSIBLE if co-occurrence fixed |
| MishandledException | `ret_ign=1.0` **(after fix)** | — | YES |
| UnusedReturn | `ret_ign=1.0` **(after fix)** | — | YES |
| Timestamp | **NONE** | None | NO (correlations only) |
| DenialOfService | None | `has_loop=1.0` + array access | WEAK |
| GasException | `has_loop=1.0` (for loop type) | Loop + storage reads | WEAK |
| IntegerUO (0.8.x) | `in_unchecked=1.0` | — | YES |
| IntegerUO (0.4.x) | None | Absence of SafeMath CALLS | NO |
| TOD | None | Multiple funcs READS+WRITES same vars | 2-hop |
| CallToUnknown | `call_target_typed=0.0` | — | YES |
| ExternalBug | `ext_calls>0` | Price oracle CALLS chain | MODERATE |

---

## Action Plan

### Immediate (next re-extraction run)

1. **Re-extract all graphs** with the `return_ignored` fix — MishandledException and
   UnusedReturn will have correct `ret_ign` features for the first time.
2. **Bump FEATURE_SCHEMA_VERSION to "v4"** to invalidate old cached graphs.
3. **Rebuild cache** (`create_cache.py`) and retrain.

### Next schema version (v4 feature additions, 2 new features replacing low-signal ones)

Feature candidates to replace or augment (12-dim schema stays locked):
- Replace `pure` (nearly always 0) with `uses_block_globals` (Timestamp signal)
- Modify `ext_calls` to include Transfer/Send ops (DoS signal)
- Modify `_cfg_node_type()` to classify Transfer/Send nodes as CFG_CALL

These three changes together would give direct feature signals to the three
currently-blind vulnerability classes (Timestamp, DoS, and partly GasException).

### Training signal quality (independent of schema)

- **DoS co-occurrence fix**: 99% DoS↔Reentrancy co-occurrence corrupts both labels.
  Root cause is BCCC storing same .sol in multiple dirs. Need dataset-level surgery
  to separate or re-label these samples — cannot be fixed in the model.
- **Data augmentation**: Timestamp (1,493 training samples) and DoS (257 training
  samples) need CEI-style augmentation with clean single-label examples.
