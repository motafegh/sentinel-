# Fix #2 — Fix _compute_uses_block_globals extraction

**Status:** ✓ **APPLIED** (commit `eec9323`, 2026-06-06). v9 graphs re-extracted.

**Effort:** 30 minutes (re-extract needed)
**Impact:** Timestamp, TOD
**Risk:** Medium — changes feature values, requires full re-extract
**Order:** Do this AFTER Fix #1, BEFORE Fix #3.

---

## Problem (Finding E from audit)

`_compute_uses_block_globals()` at `ml/src/preprocessing/graph_extractor.py:459` only detects
block.timestamp/number/etc. when the IR variable is an instance of `_SolidityVariableComposed`.
This misses:

1. **`now` keyword** — Solidity 0.4.x alias for `block.timestamp`. Slither/solc 0.4.x ABI parses
   this as `SolidityVariable` (not `SolidityVariableComposed`) in some versions.

2. **Library wrapper patterns** — `SafeMath.add(a, b)` and similar library calls that internally
   read block globals but the call site doesn't directly reference them.

3. **Inheritance + virtual calls** — block.timestamp accessed via an inherited contract that is
   not parsed by Slither (interface-only stubs).

**Audit evidence:** `roulette.sol` (pre-0.8, uses `now` keyword) → `feat[2] = 0.0` despite the
contract being labeled Timestamp in BCCC.

---

## Source Code References

### Current implementation (broken)

`ml/src/preprocessing/graph_extractor.py:459-491` — `_compute_uses_block_globals()`:

```python
def _compute_uses_block_globals(func: Any) -> float:
    """
    1.0 if any IR op in this function reads block.timestamp, block.number,
    block.difficulty, or block.basefee.
    """
    try:
        _BLOCK_GLOBALS = {"timestamp", "number", "difficulty", "basefee", "prevrandao"}
        for node in (getattr(func, "nodes", None) or []):
            for op in (getattr(node, "irs", None) or []):
                for rv in (getattr(op, "read", None) or []):
                    if _SolidityVariableComposed is not None and isinstance(rv, _SolidityVariableComposed):
                        name = getattr(rv, "name", "") or ""
                        part = name.split(".")[-1].lower()
                        if part in _BLOCK_GLOBALS:
                            return 1.0
    except Exception as exc:
        global _block_globals_fail_count
        _block_globals_fail_count += 1
        logger.debug(...)
    return 0.0
```

### `_SolidityVariableComposed` import

`ml/src/preprocessing/graph_extractor.py:144-155` — module-level import:
```python
try:
    from slither.core.declarations.solidity_variables import (
        SolidityVariableComposed as _SolidityVariableComposed,
    )
except (ImportError, AttributeError):
    _SolidityVariableComposed = None
    logger.warning(
        "[A9] SolidityVariableComposed not importable from "
        "slither.core.declarations.solidity_variables — "
        "uses_block_globals will always be 0.0. ..."
    )
```

### Per-node variant (C-1 fix)

`ml/src/preprocessing/graph_extractor.py:552-567` — `_node_uses_block_globals()` has the same
bug. **Must be fixed in parallel with the function-level one.**

### Inherited by CFG nodes (BUG-C3)

`ml/src/preprocessing/graph_extractor.py` — `_build_cfg_node_features()` inherits features
[1, 3, 4, 5, 9] from parent FUNCTION. The parent fn's `uses_block_globals` is recomputed per
CFG node via `_node_uses_block_globals` — **so fixing the function-level version is NOT enough**;
must also fix the per-node variant.

### Where the feature is consumed

`ml/src/models/sentinel_model.py:126` — `_PREFIX_NODE_PRIORITY: dict[int, int]` dict defines
prefix-node selection priority. The model consumes feat[2] through `gnn_to_bert_proj` for
prefix injection. No code change needed there for this fix.

---

## Fix

Add a fallback check by name (covers `now` and any future Slither class drift):

```python
# ml/src/preprocessing/graph_extractor.py:459
def _compute_uses_block_globals(func: Any) -> float:
    """
    1.0 if any IR op in this function reads block.timestamp, block.number,
    block.difficulty, block.basefee, block.prevrandao, blockhash, or `now` (pre-0.8 alias).
    """
    try:
        _BLOCK_GLOBALS = {"timestamp", "number", "difficulty", "basefee", "prevrandao"}
        for node in (getattr(func, "nodes", None) or []):
            for op in (getattr(node, "irs", None) or []):
                for rv in (getattr(op, "read", None) or []):
                    # Primary: SolidityVariableComposed (block.X)
                    if _SolidityVariableComposed is not None and isinstance(rv, _SolidityVariableComposed):
                        name = getattr(rv, "name", "") or ""
                        part = name.split(".")[-1].lower()
                        if part in _BLOCK_GLOBALS:
                            return 1.0
                    # Fallback: name-based check (catches `now` in Solidity 0.4.x
                    # and any other Slither class drift)
                    rv_name = (getattr(rv, "name", "") or "").lower()
                    if rv_name in {
                        "now",
                        "block.timestamp", "block.number", "block.difficulty",
                        "block.basefee", "block.prevrandao",
                        "blockhash",
                    }:
                        return 1.0
                    # Library call pattern: SafeMath-style wrappers
                    rv_type = getattr(rv, "type", None)
                    if rv_type is not None and "block" in str(rv_type).lower():
                        return 1.0
    except Exception as exc:
        global _block_globals_fail_count
        _block_globals_fail_count += 1
        logger.debug(...)
    return 0.0
```

Apply the same three-tier check to `_node_uses_block_globals()` at line 545.

---

## Validation Steps

```bash
# 1. Test on roulette.sol (the known broken case)
python -c "
import torch
from pathlib import Path
import sys; sys.path.insert(0, '.')
from ml.src.preprocessing.graph_extractor import GraphExtractionConfig, extract_contract_graph
g = extract_contract_graph(
    Path('BCCC-SCsVul-2024/SourceCodes/Timestamp/roulette.sol'),
    GraphExtractionConfig(solc_version='0.4.26')
)
# feat[2] column = uses_block_globals
block_globals_sum = float(g.x[:, 2].sum())
print(f'feat[2] sum: {block_globals_sum}')  # Expect > 0.5 (was 0.0)
"

# 2. Run full re-extract (41,576 contracts, ~30-60 min on 8 workers)
source ml/.venv/bin/activate
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --workers 8

# 3. Validate
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py --check-block-globals
# Expect: "Block globals (feat[2] fires): >9500/41576" (was 3799, was 27.5% of Timestamp=1)
```

---

## Expected Impact

Before fix:
- Timestamp=1 with feat[2] > 0: ~520/1901 (27.5%)
- Timestamp=0 with feat[2] > 0: ~3,279/39,675 (8.3%)
- Lift: 27.5% / 8.3% = **3.3x**

After fix:
- Timestamp=1 with feat[2] > 0: ~1,200/1,901 (63%)
- Timestamp=0 with feat[2] > 0: ~4,500/39,675 (11.3%)
- Lift: 63% / 11.3% = **5.6x** (vs 3.3x before)

The cleaner Timestamp labels (Fix #1) combined with this fix should make Timestamp learnable
on top of the existing feat[2] signal.

---

## Risk Assessment

**MEDIUM.** Changes feature values for all 41,576 graphs. After re-extraction, the model's
existing weights (Run 7, Run 8) will produce slightly different logits for the same input — but
this is a feature improvement, not a regression. The new feature values are more semantically
correct (catching `now` aliases that were silently dropped).

**Backward compat:** must bump `FEATURE_SCHEMA_VERSION` because `_compute_uses_block_globals`
is part of the v8 schema. Bump to `"v9"` after re-extract.

---

## Files Changed

| File | Change |
|------|--------|
| `ml/src/preprocessing/graph_extractor.py:459-492` | Add `now`/library fallback to `_compute_uses_block_globals` |
| `ml/src/preprocessing/graph_extractor.py:552-567` | Add same fallback to `_node_uses_block_globals` |
| `ml/src/preprocessing/graph_schema.py:160` | Bump `FEATURE_SCHEMA_VERSION = "v9"` (after re-extract) |
| `ml/scripts/reextract_graphs.py` | (no change — uses GraphExtractionConfig defaults) |
| `ml/data/cached_dataset_v10.pkl` | Rebuild with `create_cache.py` after re-extract |
| `ml/data/processed/multilabel_index.csv` | Re-run `build_multilabel_index.py` after re-extract (contract_path may shift due to most_derived heuristic — already v8, no change needed unless heuristic changed) |
