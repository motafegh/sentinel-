# Procedure Attestation — C_diagnostic_checks — 2026-06-15

**Run:** GCB-P1-Run12-v3dospatched-20260613  
**Log path:** ml/logs/GCB-P1-Run12-v3dospatched-20260613/

## Steps completed

| Step | Result | Detail |
|------|--------|--------|
| C.1.1 Log files located | PASS | alerts.jsonl, epoch_summary.jsonl, step_metrics.jsonl — all present |
| C.1.2 KILL alerts checked | PASS — 0 KILL events | No TrainingAbortError; all 51 epochs completed |
| C.1.3 WARN alerts reviewed | PASS (expected) — count=18 | [9.3.6b]×11 + [9.3.6c]×5 + [1.8]+[1.9] info |
| C.1.4 JK entropy range | PASS — min=1.090, max=1.099 | All epochs above 0.5 threshold. Note: locked near ln(3)=1.099 — uniform phase distribution, not collapse |
| C.1.5 GNN share | PASS | step_metrics: 102 entries, range 0.66–0.91. Well above 15% floor. Not in epoch_summary (null there) |
| C.1.6 Per-class F1 convergence | PASS with findings | GasException=0.0 every epoch (0 val positives). DenialOfService=0.30 (low but improving). ExternalBug threshold-gaming signal. |
| C.1.7 AUC/Brier trends | PASS with findings | AUC-ROC=0.911, AUC-PR=0.789 at best. Brier=0.112. ECE=0.183 pre-cal → 0.035 post-cal |
| C.2.1 Smoke inference | UNVERIFIED | No formal curated benchmark run this session. Wild eval used instead. |
| C.2.2 FP probe | PASS | Manual inspection: 9 contracts, 4 TP / 3 FP / 2 borderline. ExternalBug high-conf FP identified. |
| C.2.3 Threshold verification | DONE | Companion `_best_thresholds.json` exists. Per-class thresholds range 0.05–0.50. |

## Steps skipped

- C.2.1 formal SmartBugs Curated smoke inference: **SKIPPED** — set is 95.8% contaminated. Wild eval (47K) used as functional substitute. Marking UNVERIFIED per Rule 2.

## Unverified items

- C.2.1: SmartBugs Curated smoke inference not run. Should be re-run on the v0.1 honest benchmark (66 contracts) before Production promotion.

## New findings

- FIND-R12-01: ExternalBug [9.3.6c] F1-AUC divergence at ep17/21 — threshold gaming consistent with class label quality issue.
- FIND-R12-02: TransactionOrderDependence [9.3.6c] divergence at ep21/31/37 — minority class, model threshold-games a near-random signal.
- FIND-R12-03: GasException F1=0.0 (confirmed). Drop for Run 13, NUM_CLASSES=9.
- FIND-R12-04: JK entropy locked near ln(3) — try lower jk_entropy_reg_lambda in Run 13 to allow phases to specialise.

## Written to

`ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/attestations/C_diagnostic_checks_attestation.md`  
Raw data: `gate_reports/C1_training_log_diagnostics.json`
