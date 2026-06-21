# ml/testing_specs — Validation and Audit Spec Suite

Validation and audit procedures for model runs, data integrity, and session handoff.

---

## Scope

**These files ARE for:** validating runs, auditing model behaviour, triaging alerts, promoting checkpoints, verifying data integrity, and session handoff.

**These files are NOT for:** writing code, building features, refactoring, adding scripts, or general development work.

---

## Files

| File | Purpose |
|------|---------|
| `00_rules.md` | Spec rules — load alongside any spec file |
| `README.md` | Routing table — identifies which 1-2 spec files apply |
| `QUICKSTART.md` | Quick start guide |
| `MIGRATION.md` | Migration guide |
| `A_benchmark_runs.md` | A: Benchmark run validation |
| `B_contract_deep_dive.md` | B: Contract deep-dive analysis |
| `B_data_pipeline.md` | B: Data pipeline validation |
| `C_diagnostic_checks.md` | C: Diagnostic checks |
| `D_smoke_preflight.md` | D: Smoke preflight checks |
| `E_preprocessing_consistency.md` | E: Preprocessing consistency |
| `F_new_run_checklist.md` | F: New run checklist |
| `G_ablation_protocol.md` | G: Ablation study protocol |
| `H_issue_triage.md` | H: Issue triage procedures |
| `I_regression_guard.md` | I: Regression guard checks |
| `J_schema_migration.md` | J: Schema migration procedures |
| `K_inference_api.md` | K: Inference API validation |
| `L_release_readiness.md` | L: Release readiness checklist |

---

## Testing Framework

**Directory:** `framework/`

| File | Purpose |
|------|---------|
| `cli.py` | CLI entry point: `python -m ml.testing_specs.framework.cli run` |
| `config.py` | Gate configuration and thresholds |
| `gates.py` | Gate implementations (9 gates wired into CLI) |
| `reporters.py` | Report generation |
| `templates/` | Report templates |

**9 gates** validate model state: contamination, diagnostics, inference smoke, staging checks, etc.

---

## Usage

```bash
# Run all gates
python -m ml.testing_specs.framework.cli run

# Run specific spec
cat ml/testing_specs/00_rules.md   # always load alongside any spec
```

---

## Run 12 Validation State

- 6 gates PASS, 3 FAIL, Exit 1 — correct for current state
- Phase 5 closed all loose ends: BUG-6 fixed, 36 unit tests at 91% coverage
