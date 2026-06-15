# Procedure Attestation — F_new_run_checklist F.3 — 2026-06-15

**Run:** GCB-P1-Run12-v3dospatched-20260613  
**Section applied:** F.3 Post-Run Checklist (run is complete, in Staging)

## Steps completed

| Step | Result | Detail |
|------|--------|--------|
| F.3.1 Diagnostic checks (C spec) | PASS with UNVERIFIED | See C attestation. KILL=0, WARN=18 expected, JK/GNN healthy. C.2.1 smoke unverified. |
| F.3.2 Contamination check before benchmark | PASS | Completed before Wild eval started. 17.37% contamination identified and partitioned. |
| F.3.3 MEMORY.md Training History updated | DONE | Run 12 row: ep50, F1_tuned=0.7004, in Staging. Table updated 2026-06-14. |
| F.3.4 Findings externalised | DONE | All findings in `docs/reports/2026-06-15_ml_Run12_eval_*/` + manual inspection report + MEMORY.md |

## Steps skipped

- F.1 Pre-launch (not applicable — run is complete)
- F.2 Staging promotion (covered by I attestation — promotion already completed)

## Unverified items

- C.2.1 curated smoke inference — see C attestation

## New findings

- See C attestation findings FIND-R12-01 through FIND-R12-04
- FIND-R12-B01/B02 in A attestation (ExternalBug mismatch, Timestamp TP)

## Written to

`ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/attestations/F3_post_run_attestation.md`
