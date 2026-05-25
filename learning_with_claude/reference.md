# Reference — Learning With Claude (How This Works)

This folder tracks the ongoing deep-dive study of the Sentinel ML module.
It contains 4 files. This file is the entry point — read it first.

---

## The 4 Files

| File | Purpose | When to Update |
|------|---------|----------------|
| `reference.md` | **This file.** How the system works, rules for each file, overall structure. | When the meta-rules or workflow change. |
| `preferences.md` | All teaching preferences (P1, P2, ...). Controls HOW teaching is delivered. | When a new preference is stated or observed. Add immediately — do not batch. |
| `audit_flags.md` | All issues found during teaching (A1, A2, ...). Bugs, design problems, missing guards. | Every time an `[AUDIT]` flag is raised during teaching. |
| `session_log.md` | Record of what was covered in each session. Progress tracker. | At the end of each teaching chunk or session. |

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

### For the User

- State new preferences at any time — Claude will add them to `preferences.md` immediately.
- Challenge question answers trigger gap-fill teaching (P2) and session log update.
- To resume a session after a break: just say "resume" — Claude reads all 4 files first.

---

## Current Status

- **Active phase:** Phase 2 — `graph_extractor.py`
- **Current chunk:** Chunk 1 complete, questions answered, gap-fill done → ready for Chunk 2
- **Preferences active:** P1 through P8
- **Audit flags raised:** A1 through A4
- **Files taught so far:** `graph_schema.py`, `hash_utils.py`, `graph_extractor.py` (Chunk 1)

---

## Teaching Roadmap (full)

```
Phase 1  ✅  graph_schema.py + hash_utils.py
Phase 2  🔄  graph_extractor.py  (Chunk 1 ✅, Chunks 2–5 pending)
Phase 3      data_extraction/  (ast_extractor.py, tokenizer.py)
Phase 4      datasets/         (dual_path_dataset.py)
Phase 5      models/           (gnn_encoder.py, transformer_encoder.py,
                                fusion_layer.py, sentinel_model.py)
Phase 6      training/         (focalloss.py, losses.py, trainer.py)
Phase 7      inference/        (preprocess.py, predictor.py, cache.py,
                                drift_detector.py, api.py)
```
