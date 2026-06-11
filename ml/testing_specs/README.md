# ml/testing_specs/ — Validation Spec Suite

> **Entry point for all validation and testing procedures in the ml module.**
> Always load `00_rules.md` alongside any spec file you use.
> The rules in `00_rules.md` apply universally and are not repeated in section files.

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
| `B_contract_deep_dive.md` | Diagnosing individual mispredicted contracts |
| `C_diagnostic_checks.md` | Training log analysis and post-run model behaviour |
| `D_smoke_preflight.md` | Smoke tests, VRAM gate, compile validation |
| `E_preprocessing_consistency.md` | Confirming train/eval preprocessing alignment |
| `F_new_run_checklist.md` | Pre-launch, promotion, and post-run gates |
| `G_drift_detection.md` | Responding to inference drift signals |
| `H_api_validation.md` | Validating `api.py` response correctness |
| `I_interpretability.md` | Running and recording interpretability experiments |
| `J_schema_migration.md` | Safe schema version change protocol |
| `K_label_validation.md` | Validating a label file before it enters training |
| `L_reproducibility.md` | Confirming results are genuinely reproducible |
| `M_session_handoff.md` | Session close and handoff checklist |

---

## Task-to-Spec Routing

| I need to... | Load these files |
|---|---|
| Run a benchmark on a dataset | `00_rules` + `A` |
| Investigate a mispredicted contract | `00_rules` + `B` + `C` |
| Analyse training logs after a run | `00_rules` + `C` |
| Run smoke tests or VRAM check | `00_rules` + `D` |
| Verify train/eval preprocessing match | `00_rules` + `E` |
| Launch a new training run | `00_rules` + `D` + `E` + `F` |
| Promote a checkpoint | `00_rules` + `C` + `F` (section F.2) |
| Investigate a drift alert | `00_rules` + `G` + `B` |
| Validate the API endpoint | `00_rules` + `H` + `C` |
| Run interpretability experiments | `00_rules` + `I` + `C` |
| Change the graph schema | `00_rules` + `J` + `E` + `F` |
| Validate a new label file | `00_rules` + `K` |
| Verify a result is reproducible | `00_rules` + `L` + `C` |
| Close a session / hand off | `00_rules` + `M` |
| Unsure what applies | Read this README, then `00_rules` |

---

## Integration with CLAUDE.md

Reference this suite in `CLAUDE.md` with a single pointer:

```
For all ml/ validation and testing procedures, read:
ml/testing_specs/README.md
```

Do not inline individual spec sections into `CLAUDE.md`. The routing table
above is the interface — everything else is loaded on demand.

---

## Adding a New Spec File

1. Name it with the next letter prefix: `N_<scope>.md`
2. Write it according to the four guiding principles in the proposal:
   no hardcoding, no dictating outcomes, procedures not knowledge, dynamic
3. Add it to the File Descriptions table and the routing table above
4. Do not edit any existing spec file to accommodate it
5. Confirm every multi-step procedure in the new file supports Rule 2 attestation
