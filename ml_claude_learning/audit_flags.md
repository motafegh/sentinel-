# Audit Flags — Learning With Claude

All issues found during teaching. Every entry is permanent — never delete, only append.
Entries are added **immediately** when an `[AUDIT]` flag is raised inline during teaching.

Format per entry:
```
## A# — File — Short description
**File:** path
**Location:** function/line
**Issue:** what is wrong and why it matters
**Fix:** concrete fix
**Severity:** Low / Medium / High
**Status:** Open / Noted / Fixed
**Raised:** Session N, Chunk N
```

Severity scale:
- **High** — correctness bug, data loss, silent wrong result, security hole
- **Medium** — design flaw, missing guard, misleading abstraction, perf trap
- **Low** — style issue, suboptimal naming, missed documentation, minor inefficiency

---

## A1 — gnn_encoder.py — phase2_edge_types no bounds validation
**File:** ml/src/models/gnn_encoder.py
**Location:** `GNNEncoder.__init__`, `phase2_edge_types` parameter
**Issue:** Accepts arbitrary integers with no validation. Passing `[99]` silently builds an all-False `cfg_mask` in the forward pass — Phase 2 is completely disabled, no warning is raised, and the model trains on structural signal alone. Misconfiguration is indistinguishable from correct ablation.
**Fix:** Add in `__init__` after the `num_layers` guard: `if phase2_edge_types is not None and any(t < 0 or t >= NUM_EDGE_TYPES for t in phase2_edge_types): raise ValueError(f"phase2_edge_types contains invalid type ID. Valid range: [0, {NUM_EDGE_TYPES-1}].")`
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 2

## A3 — gnn_encoder.py — _param_dtype cache not invalidated on dtype cast
**File:** ml/src/models/gnn_encoder.py
**Location:** `_param_dtype` field / `refresh_dtype_cache()` method (~line 352)
**Issue:** `_param_dtype` is cached once at construction. If the caller does `model.half()` or `model.bfloat16()` and forgets to call `refresh_dtype_cache()`, the forward pass dtype guard compares input dtype against a stale cached value, silently skipping the cast. On CPU: mixed-dtype ops upcast to float32 (wastes half-precision). On CUDA: dtype mismatch runtime error.
**Fix:** Override `.to()` / `.half()` / `.float()` / `.bfloat16()` to auto-invalidate the cache, or use a `@property` with a `_dtype_dirty` flag. Manual `refresh_dtype_cache()` call is error-prone.
**Severity:** Medium
**Status:** Open
**Raised:** Session 3, Chunk 3
**File:** ml/src/models/gnn_encoder.py
**Location:** `GNNEncoder.__init__`, lines computing `_p2_heads` / `_p2_head_dim`
**Issue:** Phase 1 validates `_head_dim * heads != hidden_dim` at construction time, but Phase 2 hardcodes `_p2_heads=4` and computes `_p2_head_dim = hidden_dim // _p2_heads` with no equivalent check. If `hidden_dim` is not divisible by 4, GATConv raises an unhelpful shape error at first forward pass rather than at model construction, making root cause hard to diagnose.
**Fix:** Add immediately after `_p2_heads = 4`: `if hidden_dim % _p2_heads != 0: raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by Phase 2 heads ({_p2_heads}).")`
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 2
