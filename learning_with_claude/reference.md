# Reference — Learning With Claude (How This Works)

This folder tracks the ongoing deep-dive study of the Sentinel ML module.
It contains 4 spec files and a `learning_roadmap/` subfolder. This file is the entry point — read it first.

---

## The 4 Spec Files

| File | Purpose | When to Update |
|------|---------|----------------|
| `reference.md` | **This file.** How the system works, rules for each file, overall structure. | When meta-rules, current status, or roadmap change. |
| `preferences.md` | All teaching preferences (P1, P2, ...). Controls HOW teaching is delivered. | Immediately when a new preference is stated or observed. Never batch. |
| `audit_flags.md` | All issues found during teaching (A1, A2, ...). Bugs, design problems, missing guards. | Immediately when an `[AUDIT]` flag is raised during teaching. |
| `session_log.md` | Record of what was covered in each session. Progress tracker. | After each chunk is fully delivered and questions posted. |

## The `learning_roadmap/` Subfolder

Detailed per-phase teaching plans. Created ahead of teaching the phase — covers session breakdown,
chunk boundaries, new concepts per session, anticipated audit flags, and cross-file dependencies.

| File | Phase | Status |
|------|-------|--------|
| `learning_roadmap/phase5_models.md` | Phase 5 — `models/` | ✅ Ready (6 sessions, 7 chunks) |

**When to create a roadmap file:** Before starting a new phase, after reading all source files in that
folder. Roadmap files are read-only once created — update only if the session plan changes materially.
Do not update them to reflect session progress; `session_log.md` tracks that.

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

After any spec file update: `git add learning_with_claude/ && git commit && git push`.
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

- **Active phase:** Phase 5 — `models/`
- **Current chunk:** Session 11 (`fusion_layer.py`) delivered; challenge answers pending
- **Preferences active:** P1 through P14
- **Audit flags raised:** A1 through A31 (A27 renumbered from A24 to maintain append order)
- **Files taught so far:** `graph_schema.py`, `hash_utils.py`, `graph_extractor.py` (all 5 chunks), `ast_extractor.py`, `gnn_encoder.py` (both chunks), `transformer_encoder.py`, `fusion_layer.py`
- **Skipped (deferred):** `tokenizer.py` (Phase 3) — can return to later
- **Roadmap available:** `learning_roadmap/phase5_models.md`

---

## Teaching Roadmap (full)

```
Phase 1  ✅  graph_schema.py + hash_utils.py
Phase 2  ✅  graph_extractor.py
              Chunk 1 ✅  exceptions, config, _MAX_TYPE_ID
              Chunk 2 ✅  feature computation helpers (_compute_*)
              Chunk 3 ✅  CFG node typing, feature building, control-flow edges
              Chunk 4 ✅  ICFG + DEF_USE + _build_node_features + _select_contract
              Chunk 5 ✅  extract_contract_graph() — main assembly
Phase 3  🔄  data_extraction/
              ast_extractor.py ✅
              tokenizer.py     ⬜
Phase 4  ⬜  datasets/         (dual_path_dataset.py)
Phase 5  🔄  models/           → see learning_roadmap/phase5_models.md
              Session 8  ✅  gnn_encoder.py Chunk 1 (_JKAttention + __init__)
              Session 9  ✅  gnn_encoder.py Chunk 2 (forward pass)
              Session 10 ✅  transformer_encoder.py (full)
              Session 11 ✅  fusion_layer.py (full)
              Session 12 ⬜  sentinel_model.py Chunk 1 (constants + __init__ + select_prefix_nodes)
              Session 13 ⬜  sentinel_model.py Chunk 2 (forward + aux heads)
Phase 6  ⬜  training/         (focalloss.py, losses.py, trainer.py)
Phase 7  ⬜  inference/        (preprocess.py, predictor.py, cache.py,
                                drift_detector.py, api.py)
```
