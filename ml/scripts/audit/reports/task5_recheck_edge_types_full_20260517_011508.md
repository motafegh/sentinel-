# Task 5: Re-check Edge Types Full Audit

**Files scanned:** 5000 / 44470  
**Total edges examined:** 1,130,191  
**Stale (8-dim) graphs skipped:** 0

## Edge Type Distribution

| Edge Type ID | Name | Count | Percentage |
|-------------|------|-------|------------|
| 0 | CALLS | 60,138 | 5.32% |
| 1 | READS | 88,013 | 7.79% |
| 2 | WRITES | 87,411 | 7.73% |
| 5 | CONTAINS | 481,212 | 42.58% |
| 6 | CONTROL_FLOW | 413,417 | 36.58% |

## EMITS (Edge Type 3) Findings

**No EMITS edges found.** Edge type 3 is absent from all scanned graphs.

This is expected for the v4 schema because:
- The graph_extractor.py DOES create EMITS edges (see code analysis below)
- But EMITS edges require the event node to be in `node_map` (i.e., the event must be a non-dependency declaration in the target contract)
- Many contracts may not emit events, or the events may be in inherited contracts (filtered as dependencies)

## INHERITS (Edge Type 4) Findings

**No INHERITS edges found.** Edge type 4 is absent from all scanned graphs.

This is expected for the v4 schema because:
- The graph_extractor.py DOES create INHERITS edges (see code analysis below)
- But INHERITS edges require the parent contract to be in `node_map`
- Parent contracts from inherited files are filtered as dependencies by `is_from_dependency()`, so they are never added to node_map
- The target contract's own name IS in node_map, but `contract.inheritance` lists contracts that may not be in the current file

## Source Code Analysis

### Does the code explicitly skip EMITS/INHERITS?

#### graph_extractor.py

- **Creates EMITS edges:** Yes
- **Creates INHERITS edges:** Yes

**EMITS references:**

- L940: `_add_edge(fn, evt.canonical_name, EDGE_TYPES["EMITS"])`

**INHERITS references:**

- L946: `_add_edge(contract.name, parent.name, EDGE_TYPES["INHERITS"])`

#### ast_extractor.py

- **Creates EMITS edges:** No
- **Creates INHERITS edges:** No

#### reextract_graphs.py

- **Creates EMITS edges:** No
- **Creates INHERITS edges:** No

## Why EMITS/INHERITS May Be Absent Despite Code Creating Them

The graph_extractor.py contains code that creates both EMITS and INHERITS edges:

```python
# EMITS edge creation (graph_extractor.py ~L937-942)
if hasattr(func, 'events_emitted'):
    for evt in func.events_emitted:
        _add_edge(fn, evt.canonical_name, EDGE_TYPES['EMITS'])

# INHERITS edge creation (graph_extractor.py ~L944-948)
for parent in contract.inheritance:
    _add_edge(contract.name, parent.name, EDGE_TYPES['INHERITS'])
```

However, `_add_edge` only creates an edge if **both** source and target keys exist in `node_map`. The key reasons these edges may be absent:

1. **EMITS**: The event must be declared in the same contract (not a dependency). If the contract only emits events from inherited interfaces, those event nodes are filtered out and won't be in `node_map`.

2. **INHERITS**: The parent contract must be defined in the same file and not be a dependency. If inheritance is from an imported contract, it's filtered by `is_from_dependency()`.

3. **Try/except wrapping**: Both edge creation blocks are wrapped in `try/except`, so any error silently skips the edge.

## Summary

| Metric | Value |
|--------|-------|
| Graphs scanned | 5000 |
| Total edges | 1,130,191 |
| EMITS edges (type 3) | 0 |
| INHERITS edges (type 4) | 0 |
| Graphs with EMITS | 0 |
| Graphs with INHERITS | 0 |

**Result: EMITS and INHERITS edges are completely absent from the dataset.** The code can create them but they never appear in practice due to the dependency filtering and the `_add_edge` guard. These edge types are effectively dead in the dataset.

### Implications

- The `nn.Embedding(8, edge_emb_dim)` table has rows 3 and 4 that are **never trained** — they remain at random initialization
- This is not harmful (unused embeddings don't affect output) but wastes 2 embedding rows
- If the model is later used on contracts that DO have EMITS/INHERITS edges (e.g., during inference on new contracts), those edge types will use untrained random embeddings — potentially degrading inference quality
