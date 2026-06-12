# seam_swap_pre_2026-06-12 — Archive

**Why this exists:** Stage 7B seam swap (2026-06-12) consolidated three
scattered local-only backup directories into one place. These files
were NEVER committed to git (they're listed as `??` untracked in
`git status`). The authoritative history of the files they backed up
is in git:

| Archived file | Last committed version in git |
|---|---|
| `ml_preprocessing/graph_schema.py` | Was `ml/src/preprocessing/graph_schema.py` (pre-shim, full v9 schema) — see `git log --follow ml/src/preprocessing/graph_schema.py` |
| ~~`ml_datasets/dual_path_dataset.py`~~ | **(DELETED from archive 2026-06-12)** — was `ml/src/datasets/dual_path_dataset.py` (MD5-keyed, old data dir); see `git log --follow ml/src/datasets/dual_path_dataset.py` for the committed history |
| `ml_datasets/data_extraction/*` | Was `ml/src/data_extraction/*` (ast_extractor, tokenizer, windowed_tokenizer) — ast_extractor + tokenizer were `git rm`'d on 2026-06-12; windowed_tokenizer was restored in-place to `ml/src/data_extraction/windowed_tokenizer.py` |
| `data_module_representation/graph_schema.py` | Was `data_module/sentinel_data/representation/graph_schema.py` (the v1 thin adapter that re-imported from `ml/src/preprocessing/graph_schema.py`) |

**Why these files were kept locally at all:** they were `cp`'d before
the seam swap so we could roll back if the flip broke something. The
flip worked; git is the authoritative source.

**Why now consolidated:** three `_backup_pre_seam_swap_2026-06-12/`
directories scattered across `ml/src/preprocessing/`,
`ml/src/datasets/`, and `data_module/sentinel_data/representation/`
were easy to mistake for "current code." This single location makes
the archive's purpose obvious and prevents future contributors from
`cp`'ing stale code thinking it's live.

**Note (2026-06-12 follow-up):** `ml_datasets/dual_path_dataset.py`
was DELETED from the archive on 2026-06-12 once the seam-swap
cleanup confirmed nothing in `ml/src/` or `ml/tests/` referenced
`DualPathDataset` (after `tune_threshold.py` migration to
`SentinelDataset` + `test_dataset.py` deletion). The git history
still has the file for reference.

**When to delete this directory:** once Stage 7B ships to Run 10
training (the seam swap is exercised end-to-end and the old code paths
are provably dead). Suggested trigger: after first successful Run 10
epoch that loads via `SentinelDataset`.

**Related docs:**
- Live plan: `data_module/temp/live_plans/stage_7b_seam_swap_active.md`
- Handoff: `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_stage7b_handoff.md`
- Memory: `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` (Sentinel v2 Data Module Build section)
