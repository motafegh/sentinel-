# Test Contracts

Minimal Solidity contracts used by the SENTINEL GNN interpretability suite.
Each pair isolates exactly one structural difference so the model's prediction
delta between vulnerable and safe versions is attributable to that difference.

---

## Contract Pairs

### CEI Reentrancy (`reentrancy_*.sol`)
Used in: `exp_l6_counterfactual_contracts.py`, `exp_l9_attention_rollout.py`
Expected class: `Reentrancy` (idx 6)

| File | Difference |
|------|-----------|
| `reentrancy_vulnerable.sol` | `call{value}` **before** `balances[msg.sender] -= amount` (CEI violation) |
| `reentrancy_safe.sol` | State write **before** external call (CEI compliant) |

Structural test: `CFG_NODE_CALL -[CONTROL_FLOW*]-> CFG_NODE_WRITE` path exists
in vulnerable; absent in safe.

---

### Integer Overflow/Underflow (`integer_uo_*.sol`)
Used in: `exp_l6_counterfactual_contracts.py`
Expected class: `IntegerUO` (idx 4)

| File | Difference |
|------|-----------|
| `integer_uo_vulnerable.sol` | `pragma ^0.7.0` — arithmetic silently wraps |
| `integer_uo_safe.sol` | `pragma ^0.8.0` — built-in overflow reverts; explicit `require` guard |

Structural test: `uses_block_globals` is 0 in both; distinction lies in
absence of SafeMath-equivalent check nodes in the vulnerable CFG.

---

### Timestamp Dependence (`timestamp_*.sol`)
Used in: `exp_l6_counterfactual_contracts.py`
Expected class: `Timestamp` (idx 7)

| File | Difference |
|------|-----------|
| `timestamp_vulnerable.sol` | `block.timestamp > deadline` — miner-manipulable |
| `timestamp_safe.sol` | `block.number > deadlineBlock` — block number is not manipulable within same block |

Structural test: node with `uses_block_globals=1.0` exists in vulnerable;
absent in safe.

---

### Unused Return Value (`unused_return_*.sol`)
Used in: `exp_l6_counterfactual_contracts.py`
Expected class: `UnusedReturn` (idx 9)

| File | Difference |
|------|-----------|
| `unused_return_vulnerable.sol` | `target.call(data)` — return tuple not captured |
| `unused_return_safe.sol` | `(bool success,) = target.call(data); require(success)` — return checked |

Structural test: `CFG_NODE_CALL` with `return_ignored=1.0` in vulnerable;
`return_ignored=0.0` in safe.

---

### Transaction Order Dependence (`tod_*.sol`)
Used in: supplementary analysis
Expected class: `TransactionOrderDependence` (idx 8)

| File | Difference |
|------|-----------|
| `tod_vulnerable.sol` | First-come-first-served state update; no front-running protection |
| `tod_safe.sol` | Commit-reveal scheme eliminates ordering advantage |

---

### Inheritance Propagation (`inheritance_*.sol`)
Used in: `exp_l9_attention_rollout.py` (inheritance path traversal)
Expected class: `Reentrancy` (idx 6)

| File | Difference |
|------|-----------|
| `inheritance_propagation.sol` | CEI violation in base contract `ReentrancyBase`; `InheritancePropagation` inherits it |
| `inheritance_safe.sol` | Base contract `SafeBase` uses CEI correctly |

Tests whether Phase 3 (REVERSE_CONTAINS) attention correctly propagates the
vulnerability signal from the base contract CFG up through the inheritance
CONTAINS edge to the derived contract node.

---

## Node Feature Encoding Reference

Feature dim 0 (`type_id_norm`): `float(type_id) / 12.0`
- FUNCTION=1, CFG_NODE_CALL=8, CFG_NODE_WRITE=9

Feature dim 2 (`uses_block_globals`): 1.0 if node reads `block.timestamp`, `block.number` etc.
Feature dim 6 (`return_ignored`): 1.0 if this CALL node's return value is not assigned.
Feature dim 10 (`ext_call_count_raw`): raw count of external calls in this node.
