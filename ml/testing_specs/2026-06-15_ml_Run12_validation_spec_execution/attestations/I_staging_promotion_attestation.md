# Procedure Attestation — I_regression_guard — 2026-06-15

**Checkpoint:** ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt  
**Target stage:** Staging (completed 2026-06-14)  
**Prior Production F1:** N/A — no prior Production model

## Metrics

| Metric | Value | Source |
|--------|-------|--------|
| val_f1_macro_tuned (Run 12) | 0.7004 | epoch_summary.jsonl ep50 |
| val_f1_macro_tuned (prior best Run 11) | 0.3384 | MEMORY.md Training History |
| Delta vs Run 11 | +0.3620 | +107% improvement |
| AUC-ROC macro | 0.9109 | epoch_summary.jsonl ep50 |
| AUC-PR macro | 0.7891 | epoch_summary.jsonl ep50 |
| ECE (post-calibration) | 0.035 | calibration/run12/temperatures_run12_stats.json |

## Steps completed

| Step | Result | Detail |
|------|--------|--------|
| I.3.1 Behaviour checks — SmartBugs Curated smoke | UNVERIFIED | Set contaminated (95.8%). Wild FP probe done instead (see C.2.2). |
| I.3.1 Known-positive round-trip | PARTIAL | RocketCash (Reentrancy high-conf) = TP. ShareCrowdsale (Timestamp) = TP. TokenVesting (Timestamp) = TP. |
| I.3.1 Known-negative check | PARTIAL | DAVID Token (p=0.18), ECF Token (p=0.28) — correctly low. |
| I.3.2 Calibration files | PRESENT | `_best_thresholds.json` + `temperatures_run12.json` — both present, dated after checkpoint |
| I.3.3 Contamination check | PASS | `gate_reports/A1_contamination_check.txt` |
| I.3.4 Smoke suite | UNVERIFIED | Smoke suite not re-run this session. Was passed for Run 12 launch (2026-06-13). |
| I.3.5 Drift baseline (Production only) | UNVERIFIED — N/A for Staging | Placeholder file with source="warmup" status="PLACEHOLDER". Not valid for Production. See `gate_reports/I35_drift_baseline_check.txt`. |
| I.3.6 F1 dry-run (Production only) | N/A — Staging only | |

## Promotion result

SUCCESS — MLflow `sentinel-vulnerability-detector` v1 in Staging (2026-06-14)

## Regression signal

NONE for overall F1 (massive improvement vs prior runs).  
FIND-R12-01/02: ExternalBug and TransactionOrderDependence show [9.3.6c] threshold gaming — not a regression, a known class quality issue.

## Rollback required

NO

## Production readiness

BLOCKED — two items must be resolved before Production:
1. **Drift baseline**: Run `compute_drift_baseline.py --source warmup` after API collects real warmup traffic
2. **Smoke suite re-run**: Re-run `ml/scripts/smoke/run_all.py` against current checkpoint

## BUG filed

- FIND-R12-03: GasException drop queued for Run 13 (not a bug, a planned fix)
- FIND-R12-ExternalBug: Class definition mismatch queued for Run 13/14 label quality review

## Written to

`ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/attestations/I_staging_promotion_attestation.md`
