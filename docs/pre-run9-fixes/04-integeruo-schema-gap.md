# Fix #4 — Add CFG_NODE_ARITH type + `in_unchecked` (Solidity 0.8+) feature

**Status:** ✓ **APPLIED** (commit `eec9323`, 2026-06-06). v9 schema live: `NODE_FEATURE_DIM=12`, `NUM_NODE_TYPES=14`, `FEATURE_SCHEMA_VERSION="v9"`. _compute_in_unchecked uses `node.scope.is_checked`. CFG_NODE_ARITH via `Binary.type` ∈ ARITH_OPS.

**Effort:** 4 hours (re-extract needed)
**Impact:** IntegerUO (the only real schema gap)
**Risk:** High — adds new node type, bumps schema version, invalidates all checkpoints
**Order:** Do this LAST of the data fixes; it's the most invasive.

---

## Problem (Finding F from audit + I context)

**Finding F:** IntegerUO is structurally unlearnable from the current schema. All arithmetic IR
ops (`Add`, `Sub`, `Mul`, `Div`, `Mod`, `Exp`, `BitAnd`, `BitOr`, etc.) collapse into
`CFG_NODE_OTHER` (type 12) in `_cfg_node_type()`. The model sees `a + b` and `some_other_op`
as identical CFG_NODE_OTHER nodes.

**Finding I:** 87.9% of BCCC dataset is pre-0.8 Solidity (0.4-0.7) where:
- `unchecked{}` keyword doesn't exist (introduced in Solidity 0.8.0)
- Integer overflow was implicit (default wrapping behavior)
- BCCC's IntegerUO labels were correct for that era

**Modern Solidity (0.8+):**
- `unchecked { a += b; }` block EXPLICITLY opts out of overflow checks
- This is a true IntegerUO vulnerability pattern
- But our 0.8+ test contracts (`17_integer_simple.sol`, `20_unused_return_minimal.sol`) all
  use `unchecked{}` and the model can't detect them (no feature for it)

**Two-pronged solution needed:**
1. **CFG_NODE_ARITH type** — distinguish arithmetic CFG nodes from synthetic-other nodes
2. **`in_unchecked_block` feature** — detect `unchecked { }` context (Solidity 0.8+ only;
   feature=0.0 for pre-0.8 contracts which were always implicitly unchecked)

---

## Source Code References

### Where CFG node types are assigned

`ml/src/preprocessing/graph_extractor.py:_cfg_node_type` (line 587-652):
```python
def _cfg_node_type(slither_node: Any) -> int:
    # Priority 1: any IR op is an external call
    if any(isinstance(op, (LowLevelCall, HighLevelCall, Transfer, Send)) for op in irs):
        return NODE_TYPES["CFG_NODE_CALL"]
    # Priority 2: node writes a state variable
    sv_written = list(getattr(slither_node, "state_variables_written", None) or [])
    if sv_written or ...:
        return NODE_TYPES["CFG_NODE_WRITE"]
    # Priority 3: node reads a state variable
    sv_read = list(getattr(slither_node, "state_variables_read", None) or [])
    if sv_read or ...:
        return NODE_TYPES["CFG_NODE_READ"]
    # Priority 4: control-flow check node type
    if getattr(slither_node, "type", None) in check_types:
        return NODE_TYPES["CFG_NODE_CHECK"]
    # Priority 5: everything else -> CFG_NODE_OTHER (12)  <- ARITHMETIC OPS GO HERE
    return NODE_TYPES["CFG_NODE_OTHER"]
```

### Node type vocabulary

`ml/src/preprocessing/graph_schema.py:250-269` — NODE_TYPES:
```python
NODE_TYPES: dict[str, int] = {
    "STATE_VAR":   0, "FUNCTION": 1, "MODIFIER": 2, "EVENT": 3,
    "FALLBACK":    4, "RECEIVE":  5, "CONSTRUCTOR": 6, "CONTRACT": 7,
    "CFG_NODE_CALL":   8,
    "CFG_NODE_WRITE":  9,
    "CFG_NODE_READ":  10,
    "CFG_NODE_CHECK": 11,
    "CFG_NODE_OTHER": 12,  # <- arithmetic ops fall here currently
}
```

`ml/src/preprocessing/graph_schema.py:205` — `NUM_NODE_TYPES: int = 13` (must bump to 14).

### Feature vector — slot re-use is NOT possible (critical correction)

`ml/src/preprocessing/graph_schema.py:174` — `NODE_FEATURE_DIM: int = 11` (v8).
`ml/src/preprocessing/graph_schema.py:422-435` — FEATURE_NAMES (11 dims in v8):

```
[0]  type_id              [1]  visibility           [2]  uses_block_globals
[3]  view                 [4]  payable              [5]  complexity
[6]  loc                  [7]  return_ignored       [8]  call_target_typed
[9]  has_loop             [10] external_call_count
```

**Common misconception to avoid:** "re-use the dropped in_unchecked slot at index 9." This
was the case in v6 (BUG-L2 fix from 2026-05-18, see `graph_schema.py:119-129`), but in v7
and v8 the in_unchecked feature was removed ENTIRELY and the indices shifted — `has_loop`
now occupies index 9. There is no dead slot to re-use.

**Correct approach: add a new dimension 11 and bump `NODE_FEATURE_DIM` to 12.** The 11-dim
schema stays intact (no shifting of has_loop / external_call_count), the new
`in_unchecked_block` slot is appended at index 11. Re-validates all graph .pt files
under the v9 schema version.

### `_compute_in_unchecked` is currently `NotImplementedError`

`ml/src/preprocessing/graph_extractor.py:393-403`:
```python
def _compute_in_unchecked(func: Any) -> float:
    raise NotImplementedError(
        "_compute_in_unchecked is deprecated in v7 (BUG-L2). "
        "in_unchecked was dropped from the feature vector — any call site was not updated. "
        "Remove the call or replace it with a schema-correct alternative."
    )
```

**Re-activate this function** with new semantics: detect `unchecked { }` block context.

---

## Fix — Part A: Add CFG_NODE_ARITH type

```python
# ml/src/preprocessing/graph_schema.py:127-141
NODE_TYPES: dict[str, int] = {
    ...
    "CFG_NODE_OTHER": 12,
    "CFG_NODE_ARITH": 13,  # NEW: any IR op is an arithmetic/overflow-prone operation
}
NUM_NODE_TYPES: int = 14

# In graph_extractor.py:_cfg_node_type, add new priority between READ and CHECK:
# Priority 3.5: any IR op is arithmetic (verified against Slither 0.10.0 source)
from slither.slithir.operations import Binary
from slither.slithir.operations.BinaryType import (
    ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, MODULO,
    POWER, LEFT_SHIFT, RIGHT_SHIFT,
)
ARITH_OPS = {
    ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, MODULO,
    POWER, LEFT_SHIFT, RIGHT_SHIFT,
}

# In _cfg_node_type, BEFORE the "Priority 4" check:
if any(
    isinstance(op, Binary) and op.type in ARITH_OPS
    for op in irs
):
    return NODE_TYPES["CFG_NODE_ARITH"]
```

**IMPORTANT — verified against installed Slither 0.10.0:**
- `BinaryType` is at `slither.slithir.operations.BinaryType` (NOT `slither.slithir.variables.binary`)
- Correct member names: `ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, MODULO, POWER, LEFT_SHIFT, RIGHT_SHIFT`
- `Binary.type` is a property (accessed as `op.type` on an instance)
- The doc previously proposed `BinaryType.ADD` etc. — these DO NOT EXIST and would raise `AttributeError` silently

**Priority placement:** between READ (priority 3) and CHECK (priority 4). Rationale: a node
that does `balances[msg.sender] += amount` is primarily a WRITE (state change), not an
arithmetic op. Only assign CFG_NODE_ARITH when the node is PURELY arithmetic with no side
effect — this avoids losing the WRITE signal which is critical for reentrancy detection.

---

## Fix — Part B: Add `in_unchecked_block` feature

```python
# ml/src/preprocessing/graph_extractor.py:392 (replace NotImplementedError)

def _compute_in_unchecked(func: Any) -> float:
    """
    1.0 if any IR op in this function is inside an `unchecked { }` block (Solidity 0.8+).
    0.0 otherwise.

    For pre-0.8 contracts (BCCC 87.9%), all arithmetic was implicitly unchecked.
    The model must learn this from the FEATURE_ABSENT signal — for 0.8+ contracts
    that explicitly opt in, this feature fires.
    """
    try:
        for node in (getattr(func, "nodes", None) or []):
            scope = getattr(node, "scope", None)
            if scope is not None and not getattr(scope, "is_checked", True):
                return 1.0
    except Exception as exc:
        global _in_unchecked_fail_count
        _in_unchecked_fail_count += 1
        logger.debug(
            "[NF-8] _compute_in_unchecked failed for %s: %s",
            getattr(func, "canonical_name", "?"), exc,
        )
    return 0.0
```

**IMPORTANT — verified against installed Slither 0.10.0 source:**

The doc previously proposed checking `getattr(op, "in_unchecked_block", False)` — this
attribute does NOT exist on Slither's Operation class. The verified attributes of Operation
are: `compilation_unit, context, expression, get_variable, lvalue, node, read,
set_expression, set_node, type, type_str, used`. No `in_unchecked_block`.

The CORRECT mechanism (verified via `slither/core/cfg/scope.py`):
- `Scope.__init__` takes `is_checked: bool` parameter
- `Scope.is_checked` is a regular instance attribute
- Unchecked blocks in Solidity 0.8+ set `Scope(is_checked=False)`
- Access path: `node.scope` -> `Scope` -> `scope.is_checked`
- `node.scope` is typed as `Union[Scope, Function]` (see `Node.__init__` line 83)

The doc also previously referenced `NodeType.STARTUNCHECKED` — this enum does NOT exist.
NodeType members (verified): `ASSEMBLY, BREAK, CATCH, CONTINUE, ENDASSEMBLY, ENDIF,
ENDLOOP, ENTRYPOINT, EXPRESSION, IF, IFLOOP, OTHER_ENTRYPOINT, PLACEHOLDER, RETURN,
STARTLOOP, THROW, TRY, VARIABLE`. There is no `STARTUNCHECKED`.

```python
# ml/src/preprocessing/graph_schema.py:422-435 (replace FEATURE_NAMES)
FEATURE_NAMES: tuple[str, ...] = (
    "type_id",              # [0]
    "visibility",           # [1]
    "uses_block_globals",   # [2]
    "view",                 # [3]
    "payable",              # [4]
    "complexity",           # [5]
    "loc",                  # [6]
    "return_ignored",       # [7]
    "call_target_typed",    # [8]
    "has_loop",             # [9]  <- unchanged from v8
    "external_call_count",  # [10] <- unchanged from v8
    "in_unchecked_block",   # [11] <- NEW (was dropped in v7 BUG-L2; re-introduced for 0.8+)
)
NODE_FEATURE_DIM: int = 12
```

```python
# ml/src/preprocessing/graph_extractor.py:_build_node_features (lines 1078-1181)
# Replace the deprecated call with the new function:
uses_unchecked = _compute_in_unchecked(obj)  # was: raise NotImplementedError

# Append at index 11 (KEEP has_loop at [9] and external_call_count at [10] intact):
return [
    float(type_id) / _MAX_TYPE_ID,    # [0]
    visibility,                        # [1]
    uses_block_globals,                # [2]
    view,                              # [3]
    payable,                           # [4]
    complexity,                        # [5]
    loc,                               # [6]
    return_ignored,                    # [7]
    call_target_typed,                 # [8]
    has_loop,                          # [9]  <- unchanged
    external_call_count,               # [10] <- unchanged
    uses_unchecked,                    # [11] <- NEW
]
```

---

## Validation Steps

```bash
# 1. Spot-check modern contract with unchecked block
python -c "
import torch
g = torch.load('ml/data/graphs/<md5_of_17_integer_simple>.pt', weights_only=False)
# feat[11] = in_unchecked_block
unchecked_sum = float(g.x[:, 11].sum())
print(f'feat[11] sum (in_unchecked_block): {unchecked_sum}')  # Expect > 0.5
# type_id normalised: CFG_NODE_ARITH (id 13) -> 13/_MAX_TYPE_ID(13) = 1.0
arith_count = int((g.x[:, 0] >= 0.99).sum())
print(f'CFG_NODE_ARITH nodes: {arith_count}')  # Expect > 0
"

# 2. Full re-extract
source ml/.venv/bin/activate
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --workers 8

# 3. Validate
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py --check-arith-nodes --check-unchecked-feature
# New checks to add to validate_graph_dataset.py
```

---

## Expected Impact

| Class | Before | After |
|-------|--------|-------|
| IntegerUO | 0.598 F1 (test, lucky — only class with any signal) | 0.70+ F1 (real detection of `unchecked` blocks + arithmetic patterns) |
| Manual test 17_integer_simple.sol | predicted Reentrancy 0.408 | predicted IntegerUO 0.6+ (if arith node type + unchecked feature fire) |

**Caveat:** BCCC IntegerUO labels are themselves noisy (45%+ false positives per audit).
Fix #5 (re-derive from Slither detectors) is needed in parallel to fully fix IntegerUO.

---

## Risk Assessment

**HIGH.** This is the most invasive change:
1. New node type (13) requires reinitializing type_embedding
2. New feature dimension (11->12) requires reinitializing input_proj
3. Bumping FEATURE_SCHEMA_VERSION invalidates all v8 caches
4. `_compute_in_unchecked` re-introduction breaks the BUG-L2 "dead signal" assumption — must
   document that the 87.9% pre-0.8 data will have feature=0.0 (not a bug, just a no-op)

**Rollback plan:** git revert this commit. The graph schema has 13 node types and 11 dims
saved as a snapshot before the change. Inference cache invalidates cleanly on version mismatch.

---

## Files Changed

| File | Change |
|------|--------|
| `ml/src/preprocessing/graph_schema.py:205` | Bump `NUM_NODE_TYPES = 14` |
| `ml/src/preprocessing/graph_schema.py:250-269` | Add `CFG_NODE_ARITH = 13` |
| `ml/src/preprocessing/graph_schema.py:174` | Bump `NODE_FEATURE_DIM = 12` |
| `ml/src/preprocessing/graph_schema.py:422-435` | Add `in_unchecked_block` to FEATURE_NAMES (at index 11) |
| `ml/src/preprocessing/graph_extractor.py:393-403` | Re-implement `_compute_in_unchecked` using node.scope.is_checked |
| `ml/src/preprocessing/graph_extractor.py:587-652` | Add CFG_NODE_ARITH priority in `_cfg_node_type` (verify BinaryType import) |
| `ml/src/preprocessing/graph_extractor.py:1078-1181` | Append new feature dim at index 11, update return list |
| `ml/src/preprocessing/graph_schema.py:160` | Bump `FEATURE_SCHEMA_VERSION = "v9"` |
| `ml/scripts/validate_graph_dataset.py` | Add `--check-arith-nodes` and `--check-unchecked-feature` |
| `ml/src/models/gnn_encoder.py` | Update input projection to consume 12-dim features (was 11) |

---

## Appendix: Verified Slither 0.10.0 API

```python
# BinaryType location: slither.slithir.operations.BinaryType
# Members:
ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, MODULO,
POWER, LEFT_SHIFT, RIGHT_SHIFT, AND, OR, CARET, ANDAND, OROR,
EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL

# Binary operation attributes:
compilation_unit, context, expression, get_variable, lvalue,
node, read, set_expression, set_node, type, type_str, used,
variable_left, variable_right

# Scope (unchecked block detection):
class Scope:
    def __init__(self, is_checked: bool, is_yul: bool, parent_scope): ...
    self.is_checked = is_checked  # bool attribute

# Node attributes (relevant subset):
scope, file_scope, irs, function, sons, fathers, type, ...
```
