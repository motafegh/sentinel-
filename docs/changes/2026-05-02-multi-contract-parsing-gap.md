# 2026-05-02 — Multi-Contract Parsing: Gap Tracked

**Session date:** 2026-05-02  
**Type:** Planning doc addition — tracked gap, no source code changed  
**Status:** Docs updated; implementation deferred to Move 9 (post-M6)  
**Related docs updated:** `docs/ROADMAP.md`, `docs/STATUS.md`, `docs/changes/INDEX.md`

---

## Origin of This Entry

An external reviewer proposed adding multi-contract parsing as a planning item.
Before accepting the proposal, the actual source code was audited. This changelog
records what the audit found, what was corrected in the reviewer's description,
and the final accurate tracking entry that was written.

---

## What the Source Code Audit Found

### Finding 1 — `_select_contract` uses `candidates[0]`, not `contracts[0]`

The reviewer described the fallback as `contracts[0]`. The actual implementation
in `ml/src/preprocessing/graph_extractor.py` is:

```python
candidates = [c for c in sl.contracts if not c.is_from_dependency()]
...
return candidates[0]
```

It first filters out all imported dependency contracts (OpenZeppelin, forge-std,
etc.) and then picks the first of the remaining user-supplied contracts. This is
an intentional design decision, not a naive bug, and it matches the policy applied
during training data construction in `ast_extractor.py`.

### Finding 2 — Multi-contract scaffold already exists (reviewer missed this)

The reviewer wrote: "Allow `_select_contract()` to add a flag" — implying no
infrastructure exists. This is incorrect.

`GraphExtractionConfig` already has:

```python
multi_contract_policy: str = "first"   # or "by_name"
target_contract_name:  str | None = None
```

The `"by_name"` policy is already implemented and tested. Adding multi-contract
support means adding `"all"` as a third value to the **existing** field — not
creating a new flag or a new config class.

### Finding 3 — Affected files list was incomplete

The reviewer listed `preprocess.py`, `predictor.py`, `api.py`, and the dataset
assembler. Missing from their list:

- **`graph_extractor.py`** — the primary implementation point; `_select_contract()`
  and `extract_contract_graph()` are where the `"all"` policy is actually coded.
- **`cache.py`** — current cache key maps to one `(graph, tokens)` pair. A
  multi-contract result is multiple graphs from the same source string — the same
  content hash. Either the cache must store `list[(graph, tokens)]` or use
  per-contract sub-keys. This design decision must be made before touching cache.py.
- **`PredictResponse` schema in `api.py`** — currently returns a flat list of
  `vulnerabilities` with no `contract_name` field. Multi-contract results need
  either a `contracts_analysed: list[str]` field (Option A, backward compatible)
  or a `per_contract: list[ContractResult]` field (Option B, breaking change).

### Finding 4 — Confirmed in ml/README.md Known Limitations

The reviewer correctly identified that the limitation is documented in `ml/README.md`
under Known Limitation #2:

> Single-contract scope — only the first non-dependency contract in a
> multi-contract file is analysed. Multi-file protocol audits are not yet supported.

This confirms the gap is real and documented — it just wasn't tracked in planning docs.

---

## What Was Accepted and What Was Corrected

| Reviewer claim | Decision |
|---|---|
| Gap is real and should be tracked | ✅ Accepted — added to ROADMAP and STATUS |
| Falls back to `contracts[0]` | ❌ Corrected — falls back to `candidates[0]` after filtering dependencies |
| "add a flag to `_select_contract()`" | ❌ Corrected — add `"all"` to existing `multi_contract_policy` enum |
| "not a pre-retrain requirement" | ✅ Accepted — inference-pipeline-only change |
| Affected files: preprocess/predictor/api/dataset | ⚠️ Incomplete — added `graph_extractor.py`, `cache.py`, `PredictResponse` schema |

---

## Where It Was Added

### `docs/ROADMAP.md`
- New section **"Post-M6: Multi-Contract Parsing (Move 9)"** added after the M6 section.
- Covers: what already exists, `"all"` policy approach, per-file affected files table,
  two `PredictResponse` schema options (Option A recommended), cache key design decision.
- New row in Unit Test Plan table: key test cases for the multi-contract pipeline.

### `docs/STATUS.md`
- New row in Module Completion table: `M1 ML — known limitation` with ⚠️ status,
  noting scaffold exists and pointing to Move 9.
- New row in Open Half-Open Loops table: multi-contract gap, scaffold state, and
  the two design decisions (cache key strategy, `PredictResponse` extension) that
  must be made before implementation.

---

## Implementation Notes for When Move 9 Is Executed

Decisions that must be made before writing any code:

1. **`PredictResponse` schema**: Option A (`contracts_analysed: list[str]` + aggregated
   vulnerabilities) vs Option B (`per_contract: list[ContractResult]` breakdown).
   Recommendation: Option A first — backward compatible, can upgrade to B later.

2. **Cache key strategy**: same content hash, multiple contracts — options:
   - Store `list[(graph, tokens)]` under the existing key with a `multi=True` lookup flag
   - Use sub-keys `"{content_md5}_{FEATURE_SCHEMA_VERSION}_{contract_name}"`
   Recommendation: sub-keys — simpler to reason about, avoids implicit ordering
   dependency in the stored list.

3. **`ast_extractor.py` (offline batch)**: does not need to change. The BCCC-SCsVul-2024
   dataset has one contract per `.sol` file; multi-contract is an inference-only concern.
