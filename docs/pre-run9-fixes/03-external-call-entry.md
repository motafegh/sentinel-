# Fix #3 — Add CALL_ENTRY/RETURN_TO for external calls

**Effort:** 2 hours (re-extract needed)
**Impact:** DoS, ExternalBug, Reentrancy
**Risk:** High — adds new edges to the graph, must re-validate all GAT message-passing paths
**Order:** Do this AFTER Fix #1 and #2.

---

## Problem (Finding D from audit)

`_add_icfg_edges()` at `ml/src/preprocessing/graph_extractor.py:858-863` only iterates
`node.internal_calls`:

```python
for callee in sorted(
    (getattr(node, "internal_calls", None) or []),
    key=lambda c: getattr(c, "canonical_name", None) or "",
):
```

This means:
- `CALL_ENTRY` (edge type 8) edges only exist for **internal** function calls
- `RETURN_TO` (edge type 9) edges only exist for **internal** function calls
- External calls (HighLevelCall to other contracts, LowLevelCall to raw addresses) get
  NO cross-function CFG edges in the graph

**Audit evidence:** only internal functions of the same contract get connected via CALL_ENTRY.
The 73.7x over-prediction of DoS (Run 8 test set) is partly caused by the model's inability
to distinguish "call in this loop" from "call across contracts" — it sees an isolated CFG_CALL
node with a loop in its parent function and fires DoS with high confidence.

---

## Source Code References

### Current `_add_icfg_edges()` (incomplete)

`ml/src/preprocessing/graph_extractor.py:_add_icfg_edges`:
```python
for func in contract.functions:
    func_key = getattr(func, "canonical_name", None) or func.name
    local_map = func_cfg_maps.get(func_key)
    if local_map is None:
        continue

    for node in (getattr(func, "nodes", None) or []):
        caller_idx = local_map.get(node)
        if caller_idx is None:
            continue

        for callee in sorted(
            (getattr(node, "internal_calls", None) or []),  # ← INTERNAL ONLY
            key=lambda c: getattr(c, "canonical_name", None) or "",
        ):
            ...
```

### Slither API for external calls

`func.high_level_calls` — list of `(destination, function)` tuples for HIGH-LEVEL external calls.
`func.low_level_calls` — list of `(destination, function)` tuples for LOW-LEVEL external calls
(`.call()`, `.delegatecall()`, etc.).

**Both are already used in feature computation:**
- `ml/src/preprocessing/graph_extractor.py:338` `_compute_call_target_typed()` iterates
  `low_level_calls` and `high_level_calls`
- `ml/src/preprocessing/graph_extractor.py:428` `_compute_external_call_count()` counts them

### Edge type vocabulary

`ml/src/preprocessing/graph_schema.py:382-398` — EDGE_TYPES dict:
```python
"CALL_ENTRY":        8,   # calling CFG_NODE → ENTRYPOINT of callee function
"RETURN_TO":         9,   # terminal CFG_NODE of callee → call-site successor
"DEF_USE":           10,
```

The 11 edge types (ids 0-10) are fixed by `NUM_EDGE_TYPES = 11` in
`ml/src/preprocessing/graph_schema.py:208`. Adding `EXTERNAL_CALL` would require bumping to
12 edge types.

### GNN encoder edge-type usage

`ml/src/models/gnn_encoder.py:218-220` — GNNEncoder constructs
`nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)` and feeds the embedding to GATConv.
`ml/src/models/gnn_encoder.py:471-483` — Phase 2 default `cfg_mask` includes types
`[6, 8, 9, 10]` (CONTROL_FLOW + CALL_ENTRY + RETURN_TO + DEF_USE); docstring at lines 182-188
documents the same. After Fix #3, Phase 2 will see external-call edges as well — must
re-validate that the 3-layer phase still has enough receptive field.

---

## Fix

Add an `EXTERNAL_CALL` edge type (id 11) and emit it from `_add_icfg_edges()` for any external
call discovered:

```python
# ml/src/preprocessing/graph_extractor.py
# 1. Add to schema (graph_schema.py:382-398):
EDGE_TYPES: dict[str, int] = {
    ...
    "DEF_USE":               10,
    "EXTERNAL_CALL":         11,  # calling CFG_NODE → external call site marker
}

# 2. Update count (graph_schema.py:208):
NUM_EDGE_TYPES: int = 12

# 3. In _add_icfg_edges, after internal_calls loop:
for node in (getattr(func, "nodes", None) or []):
    caller_idx = local_map.get(node)
    if caller_idx is None:
        continue

    # EXISTING: internal calls (types 8, 9)
    for callee in sorted(...):
        ...

    # NEW: external calls (type 11)
    _EXTERNAL_CALL = EDGE_TYPES["EXTERNAL_CALL"]
    # Use a synthetic "external" target node index = -1 (resolved in GNNEncoder)
    # OR add a dedicated EXTERNAL_CALL_NODE type
    for high_lvl in (getattr(node, "high_level_calls", None) or []):
        if caller_idx not in external_call_seen:
            edges.append([caller_idx, EXTERNAL_TARGET_IDX])  # virtual target
            edge_types.append(_EXTERNAL_CALL)
            external_call_seen.add(caller_idx)

    for low_lvl in (getattr(node, "low_level_calls", None) or []):
        if caller_idx not in external_call_seen:
            edges.append([caller_idx, EXTERNAL_TARGET_IDX])
            edge_types.append(_EXTERNAL_CALL)
            external_call_seen.add(caller_idx)
```

**Design decision: virtual target node.** Use a reserved graph index (e.g., `0` reserved for
CONTRACT node, or add a special "EXTERNAL" target) so the GNN can route external-call
information into Phase 2. Alternative: use self-loops on the calling node (id 11) so the
GAT layer sees "this node called externally" as a structural feature.

**Recommended: self-loop variant** (simpler, no node index conflicts):

```python
# In _add_icfg_edges, after internal loop:
_EXTERNAL_CALL = EDGE_TYPES["EXTERNAL_CALL"]
external_call_sites: set[int] = set()
for node in (getattr(func, "nodes", None) or []):
    caller_idx = local_map.get(node)
    if caller_idx is None:
        continue
    high_lvl = list(getattr(node, "high_level_calls", None) or [])
    low_lvl  = list(getattr(node, "low_level_calls",  None) or [])
    if high_lvl or low_lvl:
        if caller_idx not in external_call_sites:
            # Self-loop with type 11 = "this node makes an external call"
            edges.append([caller_idx, caller_idx])
            edge_types.append(_EXTERNAL_CALL)
            external_call_sites.add(caller_idx)
```

This adds a self-loop edge of type 11 to every CFG node that makes an external call. The GAT
layer's attention will weight it differently from regular CONTROL_FLOW self-loops (which are
absent in Phase 2 because `add_self_loops=False`).

---

## Validation Steps

```bash
# 1. Spot-check: 05_denial_of_service.sol has a loop with external call
python -c "
import torch
g = torch.load('ml/data/graphs/7de45bbc...pt', weights_only=False)  # compute md5 first
edge_attr = g.edge_attr
# Count EXTERNAL_CALL (id 11) edges -- should be >= 1
n_ext = int((edge_attr == 11).sum())
print(f'EXTERNAL_CALL edges: {n_ext}')  # Expect >= 1
"

# 2. Full re-extract
source ml/.venv/bin/activate
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --workers 8

# 3. Validate coverage
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py --check-external-call-edges
# New check to add: count graphs with edge_attr == 11
# Expect: >2,000/41,576 graphs now have an EXTERNAL_CALL edge (was 0)
```

---

## Expected Impact

| Class | Before | After |
|-------|--------|-------|
| DoS | sees loop + CALL in isolation, fires confidently | sees loop + CALL + EXTERNAL_CALL marker, can weight "external call in loop" more precisely |
| ExternalBug | impossible to learn from graph alone | external call destination unknown = EXTERNAL_CALL marker + call_target_typed=0.0 |
| Reentrancy | partial: 4/5 SmartBugs hit | expected 5/5: low-level call always fires EXTERNAL_CALL marker (separate from internal CALL_ENTRY) |

**Caveat:** this is a STRUCTURAL signal, not a semantic one. The model still can't tell
WHICH external contract is being called. To learn that, the transformer (CodeBERT) path
already does the work — this fix just brings the GNN path to parity.

---

## Risk Assessment

**HIGH.** Adding a new edge type (11) requires:
1. Bumping `NUM_EDGE_TYPES` from 11 to 12
2. Re-initializing `nn.Embedding(12, 64)` in GNNEncoder
3. Phase 2 edge mask must now include type 11 — update `--phase2-edge-types` default in train.py
4. All existing checkpoints are invalidated (v8 → v9)

**Rollback plan:** if Run 9 results regress, revert `_add_icfg_edges` to only emit internal
calls. The schema change (NUM_EDGE_TYPES=12) can stay — empty type-11 column in old data
won't hurt.

---

## Files Changed

| File | Change |
|------|--------|
| `ml/src/preprocessing/graph_schema.py:208` | Bump `NUM_EDGE_TYPES = 12` |
| `ml/src/preprocessing/graph_schema.py:382-398` | Add `EXTERNAL_CALL = 11` |
| `ml/src/preprocessing/graph_extractor.py:_add_icfg_edges` (lines 825-888) | Emit type-11 self-loop for high_lvl/low_lvl call sites |
| `ml/src/preprocessing/graph_schema.py:160` | Bump `FEATURE_SCHEMA_VERSION = "v9"` |
| `ml/scripts/train.py:165-166` | Update `--phase2-edge-types` default to include 11 |
| `ml/src/models/gnn_encoder.py:471-483` | Update Phase 2 mask to include 11 |
| `ml/scripts/validate_graph_dataset.py` | Add `--check-external-call-edges` flag |
