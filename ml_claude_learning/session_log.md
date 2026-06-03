# Session Log — Learning With Claude

Record of every teaching session. Entries are permanent — never delete, only append.
A new entry is added **after each chunk is fully delivered and challenge questions are posted**.

Entry format:
```
## Session N — Phase 5: filename (Chunk N)
**Date:** YYYY-MM-DD
**File:** ml/src/models/filename.py (lines X–Y)
**Concepts taught:** bullet list
**Warm-up recall:** pass/fail per question, gaps noted
**Challenge questions:** answered Y/N, gaps closed
**Audit flags raised:** A# list or "none"
```

---

## Progress Summary

| Phase | File | Chunks | Status |
|-------|------|--------|--------|
| Phase 5 | `gnn_encoder.py` | 5 planned | 🔄 Chunk 1 ✅ |
| Phase 5 | `transformer_encoder.py` | 2 planned | ⬜ not started |
| Phase 5 | `fusion_layer.py` | 1 planned | ⬜ not started |
| Phase 5 | `sentinel_model.py` | 2 planned | ⬜ not started |

---

## Session 1 — Phase 5: gnn_encoder.py (Chunk 1)
**Date:** 2026-06-03
**File:** ml/src/models/gnn_encoder.py (lines 1–60, docstring + architecture overview)
**Concepts taught:**
- The 11 edge types: taxonomy, IDs, disk vs runtime distinction
- Three-phase architecture rationale (why each phase exists, what it solves)
- Phase 1: structural aggregation + IMP-G2 input_proj skip
- Phase 2: execution-order encoding, add_self_loops=False invariant, IMP-G1 layer-specific subsets
- Phase 3: bidirectional CONTAINS propagation, IMP-G3 directional embeddings, zero-message no-op
- JK (Jumping Knowledge) connections: why keep all 3 phase outputs, how attention aggregation works
- Per-Phase LayerNorm: why magnitude normalization is needed before JK softmax

**Warm-up recall:** N/A — first chunk, no prior material

**Challenge questions answered:**
- Q1 (add_self_loops=False in Phase 2): "No idea" → full gap-fill delivered.
  Gap: did not know GAT self-loops dilute directional signal; now understands why direction
  requires excluding self from neighborhood aggregation in execution-order phases.
- Q2 (REVERSE_CONTAINS runtime-only): Partial correct — knew it was derived not stored,
  knew double-edge concern. Gap closed: precise mechanism (flip(0) on contains_mask slice),
  3 reasons not to store on disk (redundancy, same-fact-two-directions, sync risk).

**Audit flags raised:** none
