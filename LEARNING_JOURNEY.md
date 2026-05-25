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

### P3 — Chunk Large/Complex/Important Files
For any file or section that is: (a) complex, (b) large, (c) important, or (d) necessary to understand deeply —
**do not teach it in one pass**. Break it into logical chunks. After each chunk, post challenge questions
before moving to the next chunk.

This applies unconditionally to: `graph_extractor.py` (1,328 lines), `trainer.py` (1,633 lines),
`sentinel_model.py`, `gnn_encoder.py`, `predictor.py`, and any other file that qualifies on criteria a–d.

Chunk boundaries should follow natural logical units (e.g. one class, one phase, one responsibility),
not arbitrary line counts.

---

### P2 — Gap-Fill After Challenge Questions
After the user answers challenge questions, identify gaps and misconceptions explicitly.
Do not just say "correct" or "incorrect" — teach back the exact concept they were missing,
with enough depth that the gap is closed before moving to the next phase.

Observed from: Session 1 challenge question review.

---

## Session Log

### Session 1 — Phase 1: `graph_schema.py` + `hash_utils.py`
- Covered: node types (13), edge types (11), feature vector (11 dims), schema versioning, hash-based file pairing
- Challenge questions: answered, gap-fill teaching delivered (see Session 1 gap analysis)
- Audit flags raised: A1, A2 (see table below)
- Status: complete

---

## Audit Flags (running list)

| # | File | Location | Issue | Severity |
|---|------|----------|-------|----------|
| A1 | `graph_schema.py` | `NODE_FEATURE_DIM` / `type_id` normalization | `type_id` is hardcoded `/12.0` but the comment says "range 0–12". If a 14th node type is added (id=13), normalization silently exceeds [0,1] without any assert catching it. Should add: `assert max(NODE_TYPES.values()) == NODE_FEATURE_DIM - 1` or normalize by `max(NODE_TYPES.values())` dynamically at extraction time, not at GNNEncoder runtime. | Medium |
| A2 | `hash_utils.py` | `validate_hash()` | Uses `int(hash_string, 16)` which accepts uppercase hex (e.g. `"ABCDEF..."`), but `hashlib.md5().hexdigest()` always returns lowercase. The validator is silently permissive — it accepts hashes the system never produces. A regex `re.match(r'^[0-9a-f]{32}$', hash_string)` would be more precise and more readable. Not a bug, but a misleading contract. | Low |

---

*Last updated: Session 1 — gap-fill review*
