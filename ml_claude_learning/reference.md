# Reference — Learning With Claude (How This Works)

This folder tracks the ongoing deep-dive study of the Sentinel ML module.
It contains 4 files. This file is the entry point — read it first.

---

## The 4 Files

| File | Purpose | When to Update |
|------|---------|----------------|
| `reference.md` | **This file.** How the system works, rules for each file, overall structure. | When meta-rules, current status, or roadmap change. |
| `preferences.md` | All teaching preferences (P1, P2, ...). Controls HOW teaching is delivered. | Immediately when a new preference is stated or observed. Never batch. |
| `audit_flags.md` | All issues found during teaching (A1, A2, ...). Bugs, design problems, missing guards. | Immediately when an `[AUDIT]` flag is raised during teaching. |
| `session_log.md` | Record of what was covered in each session. Progress tracker. | After each chunk is fully delivered and questions posted. |

---

## Spec File Update Protocol

This protocol defines exactly when, how, and what to update in each file.
Claude must follow this on every session — not selectively.

### WHEN to update

| Trigger | File(s) to update | Timing |
|---------|------------------|--------|
| User states a new preference | `preferences.md` | **Immediately** — before continuing teaching |
| Claude observes a new teaching pattern | `preferences.md` | At the end of the current response |
| An `[AUDIT]` flag is raised inline | `audit_flags.md` | **Immediately** — same response that raised it |
| A chunk finishes (teaching delivered, questions posted) | `session_log.md` | End of that response |
| Current status changes (chunk complete, phase done) | `reference.md` (Current Status section) | End of that response |
| A preference is refined or clarified | `preferences.md` | Immediately, with a note on what changed |
| CLAUDE.md project facts change (branch, constraints, etc.) | `CLAUDE.md` | When the change is confirmed |

### HOW to update

- **preferences.md** — Append new `### P#` section at the bottom. Never rewrite existing ones silently; add a "(refined: ...)" note if clarifying.
- **audit_flags.md** — Append new `## A#` entry at the bottom. Full format: File, Location, Issue, Fix, Severity, Status, Raised. Never delete or edit past entries.
- **session_log.md** — Append new `## Session N` block. Include: file/lines, concepts taught, warm-up results, gaps closed, audit flags raised.
- **reference.md** — Update the `Current Status` section inline. Update roadmap phase markers (✅ → 🔄 → pending).

### WHAT must be in each entry

**preferences.md entry minimum:**
```
### P# — Short Title
What the rule is.
When it applies.
Format/example if relevant.
```

**audit_flags.md entry minimum:**
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

**session_log.md entry minimum:**
```
## Session N — Phase X: filename (Chunk N)
**File:** path (lines)
**Concepts taught:** bullet list
**Warm-up recall:** pass/fail per question, gaps noted
**Challenge questions:** answered Y/N, gaps closed
**Audit flags raised:** A# list
```

### Commit rule

After any spec file update: `git add ml_claude_learning/ && git commit && git push`.
Spec files are the persistent memory of this journey — uncommitted updates are lost if the session ends.

---

## Rules

### For Claude

1. **At the start of every teaching session:** read all 4 files to restore full context.
   No assumptions — state of the journey lives here, not in conversation memory.

2. **Preferences are non-negotiable constraints.** Every teaching response must comply
   with ALL active preferences in `preferences.md`. Check before writing.

3. **Audit flags must be raised inline** during teaching using the format:
   > **[AUDIT] A#** — description
   Then immediately add to `audit_flags.md`. Never delay.

4. **Session log updates** happen after each chunk is fully delivered and questions posted —
   not mid-chunk.

5. **Never delete entries** from audit_flags.md or session_log.md. Only append.

6. **Preferences can be updated or refined** — add new ones, clarify existing ones.
   Never silently override an existing preference; add a note if it evolves.

7. **Follow the Spec File Update Protocol above** on every session, every response that triggers
   an update. No exceptions.

### For the User

- State new preferences at any time — Claude will add them to `preferences.md` immediately.
- Challenge question answers trigger gap-fill teaching (P2) and session log update.
- To resume a session after a break: just say "resume" — Claude reads all 4 files first.

---

## Current Status

- **Active phase:** Phase 5 — `ml/src/models/`
- **Current chunk:** gnn_encoder.py Chunk 1 ✅ — ready for Chunk 2 (Phase 1 layer implementation)
- **Preferences active:** P1 through P15
- **Audit flags raised:** None yet
- **Files taught so far:** `gnn_encoder.py` (Chunk 1 of 5)

---

## Teaching Roadmap (full)

```
Phase 1  ⬜  graph_schema.py + hash_utils.py       (graph representation & hashing)
Phase 2  ⬜  graph_extractor.py                     (CFG/ICFG/DEF-USE extraction)
Phase 3  ⬜  data_extraction/                        (ast_extractor.py, tokenizer.py)
Phase 4  ⬜  datasets/                               (dual_path_dataset.py)
Phase 5  🔄  models/                                 ← CURRENT FOCUS
              Chunk 1 ✅  gnn_encoder.py — module docstring, architecture overview,
                           edge type taxonomy, phase design rationale
              Chunk 2 ⬜  gnn_encoder.py — Phase 1 layers (conv1, conv2, input_proj skip)
              Chunk 3 ⬜  gnn_encoder.py — Phase 2 layers (conv3/3b/3c, CFG + ICFG)
              Chunk 4 ⬜  gnn_encoder.py — Phase 3 layers (conv4/4b/4c, reverse-CONTAINS)
              Chunk 5 ⬜  gnn_encoder.py — JK connections, LayerNorm, forward() assembly
              Chunk 6 ⬜  transformer_encoder.py — CodeBERT backbone, LoRA injection
              Chunk 7 ⬜  transformer_encoder.py — WindowAttentionPooler, forward()
              Chunk 8 ⬜  fusion_layer.py — CrossAttentionFusion, bidirectional attention
              Chunk 9 ⬜  sentinel_model.py — three-eye architecture, auxiliary heads
              Chunk 10 ⬜ sentinel_model.py — forward(), pooling strategy, classifier head
Phase 6  ⬜  training/    (focalloss.py, losses.py, trainer.py)
Phase 7  ⬜  inference/   (preprocess.py, predictor.py, cache.py,
                           drift_detector.py, api.py)
```

---

## File Map — ml/src/models/

| File | Lines | Complexity | Teaching approach |
|------|-------|------------|-------------------|
| `gnn_encoder.py` | 623 | Very High | Chunked (5 chunks) — 3-phase 8-layer GAT, JK connections |
| `sentinel_model.py` | 611 | Very High | Chunked (2 chunks) — three-eye classifier, aux heads |
| `transformer_encoder.py` | 388 | High | Chunked (2 chunks) — LoRA on CodeBERT, window pooler |
| `fusion_layer.py` | 281 | High | Single chunk — CrossAttentionFusion, bidirectional attention |
| `__init__.py` | 0 | Trivial | Skip |

**Teaching order rationale:**
`gnn_encoder.py` first — it produces the node embeddings that fusion_layer.py and
sentinel_model.py depend on. transformer_encoder.py next — it produces the token embeddings.
fusion_layer.py third — it combines both. sentinel_model.py last — it orchestrates all three.
