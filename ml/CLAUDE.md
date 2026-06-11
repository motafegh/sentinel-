# ml/CLAUDE.md

Instructions for Claude Code when working inside the `ml/` module.
Read this file once at session start. Do not re-read during the session.

---

## What This Module Is

SENTINEL ml module: GNN + CodeBERT multi-label classifier for Solidity
vulnerability detection. Dual-path architecture (graph + token sequence),
trained on labelled Solidity contracts, deployed via FastAPI inference server.

---

## Key Source Locations

| What | Path |
|---|---|
| Model architecture | `ml/src/models/sentinel_model.py` |
| Graph schema constants | `ml/src/preprocessing/graph_schema.py` |
| Training entry point | `ml/src/training/trainer.py` |
| Training log field names | `ml/src/training/training_logger.py` |
| Inference API | `ml/src/inference/api.py` |
| Threshold load path | `ml/src/inference/predictor.py` |
| Preprocessing (inference) | `ml/src/inference/preprocess.py` |
| Calibration scripts | `ml/calibration/` |
| One-off scripts | `ml/scripts/` |
| Smoke test suite | `ml/scripts/smoke/` |
| Audit findings | `ml/audit_docs/` |
| Run logs (JSONL) | `ml/logs/<run_name>/` |
| Interpretability results | `ml/interpretability_results/` |
| Current state + checkpoint paths | `MEMORY.md` (repo root) |
| Architecture decisions | `docs/ml/adr/INDEX.md` |

**Read the source file before asserting any constant, field name, or path.
Do not use values from memory or prior conversation.**

---

## Dependency and Environment

- This module has its own `ml/pyproject.toml` and `ml/poetry.lock` — separate
  from the root `pyproject.toml`. Always run `poetry install` from inside `ml/`.
- Python environment: activate with `cd ml && poetry shell`
- Data files are DVC-tracked. Run `dvc pull ml/data/` before any data operation.
- Set `TRANSFORMERS_OFFLINE=1` before any training or evaluation run.
- RTX 3070 8 GB VRAM constraint applies. Run `vram_gate_test.py` before
  launching any training run. VRAM budget: 7500 MB safe / 7900 MB abort threshold.

---

## Before Modifying Any Source File

1. Check `MEMORY.md` for open bugs relevant to the file you are editing
2. Check `ml/audit_docs/` for known failure modes in that component
3. If editing `graph_schema.py`: schema changes have broad downstream impact —
   read `ml/testing_specs/J_schema_migration.md` before making any change
4. If editing `trainer.py` or `training_logger.py`: run the smoke suite
   (`ml/scripts/smoke/run_all.py`) after your change
5. Do not edit files listed in `ml/locked_files.sha256` without explicit instruction

---

## Coding Conventions in This Module

- Type hints required on all function signatures in `ml/src/`
- No hardcoded schema constants in any file outside `graph_schema.py` —
  import from there
- New training config fields: add to the config dataclass in `trainer.py`,
  not as bare `argparse` arguments
- Log new metrics via `StructuredLogger` in `training_logger.py`, not `print()`
  or bare `loguru` calls — field names must follow the Spec §8 schema
- Tests go in `ml/tests/` — mirror the `ml/src/` directory structure
- One script = one responsibility. Do not expand existing scripts in `ml/scripts/`
  to do unrelated work; create a new script instead

---

## Validation and Audit Procedures — IMPORTANT SCOPE NOTE

The `ml/testing_specs/` folder contains a validation and audit spec suite.

**These files are for: validating runs, auditing model behaviour, triaging
alerts, promoting checkpoints, verifying data integrity, and session handoff.**

**These files are NOT for: writing code, building features, refactoring,
adding scripts, or general development work.** Do not load them during
normal coding tasks — they are irrelevant to code authoring and will waste
context budget.

Load the spec suite only when the task explicitly involves one of:
- Running or evaluating a training/benchmark run
- Promoting or comparing checkpoints
- Investigating a model failure or alert
- Validating data, schema, or preprocessing
- Closing a validation session

**Entry point when needed:** `ml/testing_specs/README.md`
It contains the routing table — read it to identify which 1–2 files apply.
Always load `ml/testing_specs/00_rules.md` alongside any spec file.

---

## Before Ending Any Session

If this session involved a training run, a model finding, a bug, or a decision:

- Update `MEMORY.md` Current State if anything changed
- Write any bug or finding to `ml/audit_docs/` (do not leave it only in conversation)
- If a checkpoint was promoted, update `MEMORY.md` Training History
- If schema changed, confirm an ADR entry exists in `docs/ml/adr/`
