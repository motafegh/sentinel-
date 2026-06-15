# Procedure Attestation — A_benchmark_runs — 2026-06-15

**Run:** GCB-P1-Run12-v3dospatched-20260613  
**Scope:** A.1 Contamination check + Wild 47K benchmark

## Steps completed

| Step | Result | Source |
|------|--------|--------|
| A.1 Contamination check (v3 train vs Wild) | PASS — 17.37% flagged, 39,165 OOD | `gate_reports/A1_contamination_check.txt` |
| A.1 Contamination check (v3 internal splits) | PASS — 0% train/val/test overlap | `check_contamination_v3.py` output |
| A.2 Wild 47K eval benchmark | PASS — 40,616 successful / 6,782 Slither errors | `docs/reports/.../2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_47K_complete/` |
| A.2 OOD-only analysis | PASS — 32,496 OOD evals, 96.4% trigger rate | `analyze_wild_ood.py` + OOD summary MD |
| A.3 SENTINEL vs tool labels comparison | DONE — strict ≥2 tool agreement analysis | Manual inspection doc |

## Steps skipped

- A.2 SmartBugs Curated formal benchmark run: **SKIPPED** — curated set is 95.8% contaminated in v3 (cannot produce honest numbers). Only 66 honest OOD contracts available in v0.1 benchmark. Benchmark F1=0.8743 on those 66 contracts.

## Unverified items

- None

## New findings

- FIND-R12-B01: 65% S_only rate is not over-prediction for Timestamp/Reentrancy; IS class-definition mismatch for ExternalBug. See `docs/reports/.../sentinel_vs_tools_manual_inspection.md`.
- FIND-R12-B02: Static tools (Slither+SmartCheck) miss genuine timestamp vulnerabilities in vesting/ICO contracts by design — SENTINEL detects these correctly.

## Written to

`ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/attestations/A_benchmark_contamination_attestation.md`
