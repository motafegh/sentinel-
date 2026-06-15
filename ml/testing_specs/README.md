# ml/testing_specs/ — Validation Spec Suite

> **Entry point for all validation and testing procedures in the ml module.**
> Always load `00_rules.md` alongside any spec file you use.
> The rules in `00_rules.md` apply universally and are not repeated in section files.
>
> **Last revised: 2026-06-14** (post-Run-12 launch). The 12-spec alphabet covers general procedures; session-specific plans (Run 12 → Run 13) live in `data_module/temp/live_plans/`. See routing table for cross-refs.

---

## How to Use This Suite

1. Read the task-to-spec routing table below to identify which file applies
2. Load `00_rules.md` + the relevant section file(s) — nothing else is needed
3. Follow the procedure in order
4. Apply Rule 2 (Validate Your Validation) at every step
5. Write the completion attestation before the session ends

When multiple files are listed for a task, load them in the order shown.
Each file is independent and scoped — do not load files not relevant to
your current task.

---

## File Descriptions

| File | Scope |
|---|---|
| `00_rules.md` | Universal invariants — always loaded |
| `A_benchmark_runs.md` | External benchmark evaluation procedures |
| `B_data_pipeline.md` | Label CSV construction, split generation, data rebuild pipeline |
| `B_contract_deep_dive.md` | Diagnosing individual mispredicted contracts |
| `C_diagnostic_checks.md` | Training log analysis and post-run model behaviour |
| `D_smoke_preflight.md` | Smoke tests, VRAM gate, compile validation |
| `E_preprocessing_consistency.md` | Confirming train/eval preprocessing alignment |
| `F_new_run_checklist.md` | Pre-launch, promotion, and post-run gates |
| `G_ablation_protocol.md` | Controlled ablation experiment design and execution |
| `H_issue_triage.md` | Alert triage, guardrail response, bug filing |
| `I_regression_guard.md` | Promotion gate and regression prevention |
| `J_schema_migration.md` | Safe schema version change protocol |
| `K_inference_api.md` | Validating `api.py` endpoint correctness |
| `L_release_readiness.md` | Reproducibility protocol, release gate, session handoff |

---

## Task-to-Spec Routing

| I need to... | Load these files |
|---|---|
| Run a benchmark on a dataset | `00_rules` + `A` |
| Rebuild the label CSV or splits | `00_rules` + `B_data_pipeline` |
| Investigate a mispredicted contract | `00_rules` + `B_contract_deep_dive` + `C` |
| Analyse training logs after a run | `00_rules` + `C` |
| Run smoke tests or VRAM check | `00_rules` + `D` |
| Verify train/eval preprocessing match | `00_rules` + `E` |
| Launch a new training run | `00_rules` + `D` + `E` + `F` |
| Promote a checkpoint | `00_rules` + `C` + `F` (section F.2) + `I` |
| Investigate a drift alert | `00_rules` + `H` + `B_contract_deep_dive` |
| Run a controlled ablation | `00_rules` + `G` |
| Triage a training alert or guardrail | `00_rules` + `H` |
| Validate the API endpoint | `00_rules` + `K` + `C` |
| Change the graph schema | `00_rules` + `J` + `E` + `F` |
| Verify a result is reproducible | `00_rules` + `L` + `C` |
| Close a session / hand off | `00_rules` + `L` (section L.5) |
| **Plan the post-training workflow** (Run N done → next run) | **CROSS-REF: `data_module/temp/live_plans/post_training_process_2026-06-14.md` + `run_12_to_13_handoff_2026-06-14.md`** (session-specific plans in `live_plans/`) |
| Unsure what applies | Read this README, then `00_rules` |

---

## Integration with CLAUDE.md

This suite is referenced from `ml/CLAUDE.md`. The spec files are for
validation, audit, and training procedures only — not for coding tasks.
Read `ml/CLAUDE.md` for the full scope note.

---

## Adding a New Spec File

1. Name it with the next letter prefix: `M_<scope>.md`
2. Write it according to the four guiding principles in the proposal:
   no hardcoding, no dictating outcomes, procedures not knowledge, dynamic
3. Add it to the File Descriptions table and the routing table above
4. Do not edit any existing spec file to accommodate it
5. Confirm every multi-step procedure in the new file supports Rule 2 attestation
