# data_module/temp/

> **Working directory for the data module.** Not for production data — those live in `data_module/data/`. This directory is for plans and working scripts.

## Organization

```
data_module/temp/
├── README.md                          ← this file
├── live_plans/                        ← ACTIVE and ARCHIVED plan markdown files
│   ├── seam_swap_completion_2026-06-13.md     ← CURRENT active plan
│   └── archive/                       ← completed stage 7 plans
│       ├── stage_7a_export_module.md          (2026-06-12)
│       ├── stage_7b_seam_swap.md             (2026-06-12)
│       └── stage_7b_seam_swap_active.md      (2026-06-12)
└── archive/                           ← completed task artifacts (scripts + plans)
    └── 2026-06-13_run12_prep/         ← all artifacts from the "prepare v3 + launch Run 12" effort
        ├── plans/                     ← the pre-run12-fixes + data-source-addition plans
        └── scripts/                   ← 12 one-off scripts used to build the v3 export
```

## What's in each subdirectory

### `live_plans/` (PLANS — markdown files only)

- **`seam_swap_completion_2026-06-13.md`** — **CURRENT** plan: complete the seam swap by moving graph_extractor.py + windowed_tokenizer.py to data_module as canonical implementations (with thin re-export shims in ml/src/)
- `archive/` — Stage 7A/7B plans (completed 2026-06-12), kept for historical reference

### `archive/2026-06-13_run12_prep/` (SCRIPTS + PLANS — completed work)

The full audit trail of "prepare v3 export + launch Run 12" effort that happened 2026-06-13.

- **`plans/pre-run12-fixes-2026-06-13.md`** — the 4-step plan (Step A: DoS, B: L3 dedup, C: SmartBugs, D: 7 gates) that produced the v3 export. This plan is now SUPERSEDED by `live_plans/seam_swap_completion_2026-06-13.md` for the new work
- **`plans/data-source-addition-plan-2026-06-13.md`** — forward-looking plan for adding CGT, HF audit-firms, Kaggle synthetic data sources AFTER Run 12

- **`scripts/patch_dos_v3.py`** — the DoS/Reentrancy co-occurrence patch (zeroed 2,655 DIVE labels)
- **`scripts/resplit_and_reexport.py`** — re-split + re-export orchestration
- **`scripts/verify_v3_loads.py`** — Gate 3 + leak + DoS verification
- **`scripts/smoke_v3.py`** — SentinelDataset smoke test on v3
- **`scripts/l3_scan_and_apply.py`** — L3 text-hash dedup scanner + applier (83/147 groups applied)
- **`scripts/remerge_and_reexport.py`** — re-merge + re-export (debug)
- **`scripts/audit_dos_patch.py`** — DoS patch audit (debug)
- **`scripts/debug_patch_path.py`** — traced DoS patch through pipeline (identified splits JSONL as propagation point)
- **`scripts/check_export_labels.py`** — helper: read export labels
- **`scripts/check_leakage.py`** — helper: check 0% leakage
- **`scripts/check_manifest.py`** — helper: read v3 export manifest
- **`scripts/verify_export.py`** — verify v3 export hash + splits

These scripts are kept for historical reference and can be re-run if needed (e.g., if a v4 export is built from a new data source).

## When to add files here

**Add to `live_plans/`** when:
- You're starting a new substantial task that needs a plan (pre-flight for a code change, a new stage, an audit, etc.)
- The plan is `> 50 lines` and will be referenced from `~/.claude/projects/.../memory/` or commit messages
- Use the naming pattern: `<topic>_<YYYYMMDD>.md` (matches the existing `*_2026-06-13.md` pattern)
- Move to `live_plans/archive/<stage_name>/` when the plan is completed

**Add to `archive/<task_name>/`** when:
- The task is completed and you want to preserve the working scripts
- The scripts are not needed daily but might be re-run for verification or re-application
- The plan is complete and historical

**Do NOT add to `temp/`:**
- Runtime data (graphs, exports, logs) — those go in `data_module/data/` (gitignored)
- Permanent source code — that goes in `data_module/sentinel_data/`
- ML training code — that goes in `ml/src/`
- Documentation that should be permanent — that goes in `data_module/docs/` or `~/.claude/.../memory/`

## Archive policy

- **Completed plans** → move to `live_plans/archive/<name>/`
- **Completed scripts** (one-off, used, no longer actively needed) → move to `archive/<date>_<task_name>/scripts/`
- **Old .py files in `archive/.../scripts/`** → keep indefinitely (they document what was done; safe to delete if disk space is needed)
- **Old .md files in `archive/.../plans/`** → keep indefinitely (they document decisions made)
- **Any file older than 6 months with no recent reference** → safe to delete (git history preserves the audit trail even after deletion)

## Versioning & git

This directory IS tracked by git. The `archive/` structure is preserved in git history. If you need to revert or reference old scripts, `git log --follow <path>` will show the file's history.

For runtime artifacts (Run 12's in-flight logs, checkpoints), those go in `ml/logs/` and `ml/checkpoints/` which are gitignored.

## Current state (2026-06-14)

| Subdirectory | Files | Status |
|---|---|---|
| `live_plans/` | 1 active + 3 archived | Clean |
| `archive/2026-06-13_run12_prep/plans/` | 2 plans | Clean |
| `archive/2026-06-13_run12_prep/scripts/` | 12 scripts | Clean |
| **Total** | 17 files + this README | Organized |

**Next time someone visits `data_module/temp/`:** they should see this README + 2 subdirs (live_plans/, archive/), and the only "live work" is the current active plan in `live_plans/`.
