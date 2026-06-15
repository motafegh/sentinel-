# Run 12 Spec Execution — 2026-06-15

**Run:** GCB-P1-Run12-v3dospatched-20260613  
**Model stage:** Staging  
**Date:** 2026-06-15

This folder contains all gate assertions, reports, and attestations produced
while executing the `ml/testing_specs/` suite for Run 12.

---

## Folder Layout

```
attestations/     Formal spec attestation documents (markdown)
gate_reports/     Raw gate check output (JSON/text — Rule 2 Layer 1)
scripts/          One-off validation scripts for pending gates
```

---

## Attestation Status

| Spec | File | Status |
|------|------|--------|
| A — Benchmark + Contamination | `attestations/A_benchmark_contamination_attestation.md` | ✅ COMPLETE |
| C — Diagnostic Checks | `attestations/C_diagnostic_checks_attestation.md` | ✅ COMPLETE (C.2.1 UNVERIFIED) |
| F.3 — Post-Run Checklist | `attestations/F3_post_run_attestation.md` | ✅ COMPLETE |
| I — Staging Promotion | `attestations/I_staging_promotion_attestation.md` | ✅ COMPLETE |
| L.5 — Session Handoff | `attestations/L5_session_handoff.md` | ✅ COMPLETE |
| Stage 7B — Extractor Seam Flip | `attestations/Stage7B_extractor_flip_attestation.md` | ✅ COMPLETE (2026-06-15) |
| D — Smoke Preflight | _not yet_ | ⏳ Needed before Run 13 |
| E — Preprocessing Consistency | _not yet_ | ⏳ Needed before Run 13 |
| F.1 — Pre-Launch Gates | _not yet_ | ⏳ Needed before Run 13 |
| K — Inference API | _not yet_ | ❌ Not done — needed before Production |

---

## Gate Reports

| Report | Content | Result |
|--------|---------|--------|
| `gate_reports/A1_contamination_check.txt` | Wild 47K contamination check | PASS |
| `gate_reports/C1_training_log_diagnostics.json` | JK entropy, GNN share, alerts, metrics | PASS + findings |
| `gate_reports/I32_calibration_files_check.txt` | Thresholds + temperatures present | PASS |
| `gate_reports/I35_drift_baseline_check.txt` | Drift baseline is placeholder | UNVERIFIED (Production blocked) |

---

## Open Items (blocking Production)

1. **C.2.1** — Smoke inference on honest v0.1 benchmark (66 contracts). Script: `scripts/run_c21_smoke_inference.py`
2. **K** — Inference API validation. Spec: `../K_inference_api.md`
3. **I.3.5** — Real drift baseline (requires API warmup traffic)
4. **I.3.4** — Re-run smoke suite against current checkpoint

---

## Key Findings Produced

- FIND-R12-01: ExternalBug [9.3.6c] divergence — threshold gaming, class label issue
- FIND-R12-02: TransactionOrderDependence divergence — minority class gaming
- FIND-R12-03: GasException F1=0.0 (0 val positives) — drop for Run 13
- FIND-R12-04: JK entropy locked near ln(3) — ablation candidate for Run 13
- FIND-R12-B01: ExternalBug class definition mismatch (DeFiHackLabs ≠ tool access_control)
- FIND-R12-B02: SENTINEL correctly identifies Timestamp/Reentrancy that tools miss by design
