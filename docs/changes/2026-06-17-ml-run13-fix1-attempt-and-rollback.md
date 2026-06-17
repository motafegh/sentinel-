---
date: 2026-06-17
module: ml
run: Run12
what: changes
descriptor: run13_fix1_attempt_and_rollback
status: ACTIVE
---

# 2026-06-17 — Run 13 Fix 1 (drop GasException) attempted and rolled back

## What happened

Ali asked to "proceed with the next steps" of the project. Following CLAUDE.md's
"next steps" priority, I picked the smallest pending item from the Run 13 plan:
**Fix 1 — drop GasException → NUM_CLASSES=9**.

I made the schema change, updated all 13+ dependent files, and verified the new
9-class tests passed. **However**, the final sanity check on `ml/tests/test_api.py`
surfaced a hard failure: the **Run 12 checkpoint currently serving in Staging is
trained with 10 classes** (including GasException at index 3). Removing GasException
from the schema invalidated the checkpoint, and the API's strict class_names
validation correctly refused to load it.

## Ali's decision

Run 13 is not going to be implemented anytime soon. The current plan is to complete
the other modules first. **Ali asked to keep Run 12 working**, so I rolled back
**every** GasException change to fully restore Run 12.

## Rollback steps (executed in this order)

1. **Schema files** — restored from `.PRE-V4` backups (created before the edit):
   - `data_module/sentinel_data/representation/graph_schema.py` (10 classes restored)
   - `ml/src/training/trainer.py` (CLASS_NAMES has 10 entries again)
   - `data_module/sentinel_data/labeling/schema/taxonomy.yaml` (id=3 GasException restored, ids 4-9 unchanged)

2. **Moved files** — restored from `data_module/.bin/2026-06-17_Run13_drop_GasException/`:
   - `data_module/sentinel_data/verification/patterns/GasException.yaml`
   - `ml/data/processed/multilabel_index.csv` (10-column v3 export)
   - The `.bin/` rollback dir was removed after restoration

3. **Modified test files** — `git checkout` restored:
   - 8 data_module test files (taxonomy, thin_adapter, patterns, splitters, bccc_regression, analysis, semantic_checker, gate)
   - 2 agent test files (test_routing_phase0, test_smoke_e2e)

4. **Modified agent code** — `git checkout` restored:
   - `agents/src/orchestration/routing.py` (3 GasException dict entries restored)
   - `agents/src/mcp/servers/graph_inspector_server.py` (2 dicts restored)
   - `agents/src/mcp/servers/inference_server.py` (10-class mock restored)

5. **conftest.py** — `git checkout` removed the Run 13 note I had added

6. **`.PRE-V4` backups** — removed (served their purpose)

## Verification after rollback

| Test suite | Result |
|---|---|
| `data_module/tests/test_labeling/test_taxonomy.py` + 7 others | 142 passed, 26 skipped |
| `ml/tests/test_api.py` | 18 passed |
| `agents/tests/test_routing_phase0.py` | 46 passed |

**Run 12 is fully working again.** The Staging API serves the Run 12 checkpoint
correctly with the 10-class schema.

## What was NOT reverted (kept from earlier work)

These files were modified in Step 1 of the session (2026-06-17) for the false-positive
API schema bug investigation. They are independent of Run 13 and remain in place:

- `docs/reports/2026-06-16_ml_Run12_mlops_full_state_check/...INDEX.md` — REVISION
  section correcting the false-positive "critical API schema bug" claim
- `ml/audit_docs/ISSUES.md` — BUG-1 filed as closed (false positive)
- `ml/testing_specs/K_inference_api.md` — type corrected to `dict[str, float | list[float]]`
  (this is a real correction; both shapes were always valid)
- `docs/changes/2026-06-17-ml-api-schema-claim-correction.md` — Step 1 changelog
- `docs/learning_sentinel/2026-06-17_step1_api_bug_investigation.md` — Step 1 learning doc
- `~/.claude/projects/.../memory/MEMORY.md` — false P0 claim removed

## Lesson learned (for future Run 13)

**The plan said "Low risk, 30 min" for Fix 1. It was wrong.** The actual risk is
HIGH because:

- **All existing checkpoints become invalid** the moment the class list changes
- **The live API serving those checkpoints breaks immediately**
- The 30-min estimate only counted code edits, not the production impact

**Correct workflow for the next time we do Fix 1:**

1. **DO NOT edit the schema while a 10-class checkpoint is serving**
2. Build the 9-class model FIRST (or run the schema change in a feature branch
   that's not deployed to Staging)
3. Test the new 9-class model against the new schema
4. ONLY THEN promote the new model and update the schema
5. Update the docs and tests in the same change

Alternatively, if schema-first is required, plan for a **maintenance window** where
the API is offline while the schema and checkpoint are swapped together.

## Cross-references

- Run 13 plan: `docs/plans/2026-06-14_Run12_to_Run13_handoff.md`
- Run 13 Fix 1 detail: `docs/plans/2026-06-14_Run13_4_fixes_preparation.md` §Fix 1
- The (now-corrected) "false positive" investigation: `docs/learning_sentinel/2026-06-17_step1_api_bug_investigation.md`
