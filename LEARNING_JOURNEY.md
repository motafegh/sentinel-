# Sentinel ML — Learning Journey Spec

Active preferences and notes for our study of the ML module.
Updated as we progress. Both explicit (user-stated) and observed preferences are recorded here.

---

## Preferences

### P1 — Teach + Audit Simultaneously
When covering any file, section, or concept: **do not blindly accept the existing code or its stated reasoning**.
Actively audit as we go. If something looks off — a questionable design decision, a potential bug, a misleading comment, a missed edge case, a better alternative — flag it immediately, inline with the teaching.

Goals this serves:
- Deep understanding of the code (not just what it does, but whether it's correct)
- Identify real issues in a production ML system
- Build the critical thinking to challenge any codebase, not just this one

Flagging format to use during teaching:
> **[AUDIT]** — description of the concern, why it matters, what a better approach looks like

---

### P2 — Gap-Fill After Challenge Questions
After the user answers challenge questions, identify gaps and misconceptions explicitly.
Do not just say "correct" or "incorrect" — teach back the exact concept they were missing,
with enough depth that the gap is closed before moving to the next phase.

Observed from: Session 1 challenge question review.

---

### P3 — Chunk Large/Complex/Important Files
For any file or section that is: (a) complex, (b) large, (c) important, or (d) necessary to understand deeply —
**do not teach it in one pass**. Break it into logical chunks. After each chunk, post challenge questions
before moving to the next chunk.

This applies unconditionally to: `graph_extractor.py` (1,328 lines), `trainer.py` (1,633 lines),
`sentinel_model.py`, `gnn_encoder.py`, `predictor.py`, and any other file that qualifies on criteria a–d.

Chunk boundaries should follow natural logical units (e.g. one class, one phase, one responsibility),
not arbitrary line counts.

---

### P4 — Data Flow Diagrams per Chunk (where useful)
When teaching a chunk, include a data flow diagram, call graph, or state diagram **if it genuinely aids
understanding** — not mechanically for every chunk. Criteria: use it when the chunk involves multiple
steps, transformations, or interactions that are hard to follow linearly in prose.

Format: ASCII/text diagrams inline in the teaching. Not required for simple utility functions or
single-responsibility helpers where prose is clearer.

---

### P5 — Big Picture First for Every New File
When starting to teach a new file: **always open with a big-picture overview** before diving into
any code. This should cover:
- What problem this file solves
- Its role in the overall system
- Its major sections/responsibilities
- How it fits into the data flow (input → output)

---

### P6 — Cross-File Relationships: Recall or Preview
When the file being taught has dependencies on or relationships with other files:
- **Already learned** → actively recall the relevant concepts from that file to reinforce and
  connect. Do not assume the user remembers — trigger the connection explicitly.
- **Not yet learned** → briefly name the file, state its role in one sentence, and flag that
  we will cover it in depth later. Do not go deep now.

---

### P7 — Teach Alternative Approaches with High Educational Value
If the code in a chunk has a meaningfully different alternative approach that was not implemented —
and that alternative has high educational value (common in the field, better trade-offs, standard
pattern) — teach it alongside the current implementation and compare the two.

Criteria for "high educational value": the alternative is used widely in production ML systems,
reveals a real trade-off, or is the kind of thing that appears in technical interviews or design
discussions.

---

### P8 — Explicitly Highlight Critical Concepts
When a concept, logic, invariant, or code block is **highly important** — something that underpins
everything else, is a common source of bugs, or is non-obvious — mark it explicitly so the user
does not fast-read past it.

Highlighting format:
> ⚠️ **CRITICAL** — [explanation of why this must not be skimmed]

---

## Session Log

### Session 1 — Phase 1: `graph_schema.py` + `hash_utils.py`
- Covered: node types (13), edge types (11), feature vector (11 dims), schema versioning, hash-based file pairing
- Challenge questions: answered, gap-fill teaching delivered (see Session 1 gap analysis)
- Audit flags raised: A1, A2 (see table below)
- Status: complete

### Session 2 — Phase 2: `graph_extractor.py`
- Chunk 1 taught (module structure, exceptions, config, `_MAX_TYPE_ID`)
- New preferences P4–P8 added mid-session
- Audit flags raised: A3 (see table below)
- Status: Chunk 1 questions pending answer → then Chunk 2

---

## Audit Flags (running list)

| # | File | Location | Issue | Severity |
|---|------|----------|-------|----------|
| A1 | `graph_schema.py` | `NODE_FEATURE_DIM` / `type_id` normalization | `type_id` is hardcoded `/12.0` in comments but the actual divisor in `graph_extractor.py` is `max(NODE_TYPES.values())` (dynamic). If a 14th node type is added, all existing type_id normalizations silently shift. No assert guards this. | Medium |
| A2 | `hash_utils.py` | `validate_hash()` | Uses `int(hash_string, 16)` which accepts uppercase hex, but `hexdigest()` always returns lowercase. The validator is silently permissive. `re.match(r'^[0-9a-f]{32}$', ...)` would be more precise. | Low |
| A3 | `graph_extractor.py` | `_MAX_TYPE_ID = float(max(NODE_TYPES.values()))` | Dynamic normalization divisor contradicts `graph_schema.py`'s documented `/12.0`. Adding a new node type silently reshuffles all existing `type_id` normalizations, causing train/inference mismatch for deployed models. Should be hardcoded `12.0` with an explicit comment pointing to `graph_schema.py`. | Medium |

---

*Last updated: Session 2 — Chunk 1 + preferences P4–P8 added*
