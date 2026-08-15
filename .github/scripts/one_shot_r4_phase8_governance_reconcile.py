from pathlib import Path


index = Path("docs/plan/ml-R4/ARTIFACT_INDEX.md")
text = index.read_text(encoding="utf-8")
marker = "## Phase 8 artifacts — repaired-v2 physical acceptance and no-launch"
if marker not in text:
    section = """## Phase 8 artifacts — repaired-v2 physical acceptance and no-launch

| Artifact ID | Phase | Type | Path/URI | SHA-256 / binding identity | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P8-RUN-001 | 8 | real_data_audit | runs/2026-08-14_PHASE8_real_data_readiness_audit.md | — | 8dc81e865a82+ | New | AVAILABLE_VERIFIED | NO | Full-corpus audit that placed the historical v1 physical lineage on launch hold |
| R4-P8-RUN-002 | 8 | local_gate_reaudit | runs/2026-08-15_PHASE8_local_gate_reaudit_and_corrections.md | — | 433c5cd021b6+ | New | AVAILABLE_VERIFIED | NO | Local fail-closed gate corrections before accepted rebuild |
| R4-P8-RUN-003 | 8 | acceptance_decision | runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md | evidence source `fb31326da442`; binding `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd` | 89059bfb0b9b | New | AVAILABLE_VERIFIED | NO | Accept repaired-v2 physical DATA for bounded research; 100-epoch run NOT AUTHORIZED; G8 OPEN |
| R4-P8-ADR-001 | 8 | ADR | adrs/ADR-R4-008-repaired-v2-data-acceptance-and-phase8-no-launch.md | — | governance reconciliation | New | AVAILABLE_VERIFIED | NO | Formal R4-D-008 acceptance/no-launch authority |
| R4-P8-LED-001 | 8 | repaired_evidence_ledger | local generated `data_module/data/r4-v2-build/evidence_ledger_r4_v2.parquet` or equivalent bound build artifact | `5317aba94b9cdbe900bd90bd9b2fdf22d69c3810ec2b0a08d9be032f21658d6d` | local evidence source `fb31326da442` | New | LOCAL_GENERATED_VERIFIED | YES_LOCAL | 22,540 contracts / 225,400 rows; generated/Git-ignored; hash recorded in acceptance decision |
| R4-P8-BND-001 | 8 | repaired_representation_binding | local generated `sentinel-r4-vnext-v2/representation_binding_report.json` | binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd` | local evidence source `fb31326da442` | New | LOCAL_GENERATED_VERIFIED | YES_LOCAL | 22,540/22,540 contracts; 67,620 files; zero missing/invalid; physical root not committed |
| R4-P8-EVL-001 | 8 | token_coverage_experiment | local generated `data_module/data/r4-v2-build/bounded_window_experiment.json` | — | local evidence source `fb31326da442` | New | LOCAL_GENERATED_VERIFIED | YES_LOCAL | 11,341/11,341 role records; target-aware median target coverage 0.5119 vs 0.2760 control; candidate not promoted |
| R4-P8-SMK-001 | 8 | cuda_micro_smoke | local generated `data_module/data/r4-v2-build/repaired_gpu_smoke.json` | — | local evidence source `fb31326da442` | New | LOCAL_GENERATED_VERIFIED | YES_LOCAL | RTX 3070 Laptop BF16; two optimizer steps; finite; no Run12 weights/checkpoint; full_training_authorized=false |

"""
    anchor = "## Availability\n"
    if anchor not in text:
        raise SystemExit("ARTIFACT_INDEX availability anchor missing")
    text = text.replace(anchor, section + anchor, 1)
    index.write_text(text, encoding="utf-8")


log = Path("docs/plan/ml-R4/EXECUTION_LOG.md")
log_text = log.read_text(encoding="utf-8")
log_marker = "R4-LOG-20260815-021"
if log_marker not in log_text:
    log_text += """

### R4-LOG-20260815-021 — Independent repaired-v2 audit and governance reconciliation

- **Phase:** 8 evidence/governance review
- **Operator:** ChatGPT / GPT-5.6 Sol using the connected GitHub repository
- **Audited repository head:** `89059bfb0b9bf68447d96e0d416a7b4b78964209`
- **Scope:** independently verify the local assistant's repaired-v2 summary against governing documents, source, tests, workflow logs, and committed evidence; correct governance contradictions only
- **Verified physical evidence:** 22,823 raw records; accepted 22,540-contract / 225,400-row repaired-v2 publication; 67,620 bound representation files; zero missing/invalid; 20/20 repaired-lineage checks; binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`
- **Verified repository evidence at audited head:** dedicated Phase-8 repair CI `108 passed` with four warnings; historical frozen G6 validator PASS; diff gate PASS; Handbook `145` static checks and `11` unit tests PASS
- **Verified ML limitation:** 899 effective loss cells, all target `1`; zero confirmed negatives; positive-only model selection; threshold/calibration/untouched acceptance unsupported
- **Verified representation limitations:** 19,451 contracts over four token windows; target-aware selector improves median target coverage but remains unpromoted; 28 compatibility-mode representations; 4,211 multi-component file-union graphs; graph max 16,065 nodes / 166,459 edges
- **Governance defects corrected:** repaired-v2 acceptance registered as R4-D-008 / ADR-R4-008; stale local blockers reconciled; explicit grouping/compatibility/file-union risks added; current-status scheduler horizon corrected; PRE-R4 claim matrix updated; Phase-0 START_HERE and README redirected to the current Phase-8 boundary; Phase-8 artifacts registered here
- **New open risks/blockers:** R4-R022 source-scoped address grouping breadth; R4-R023 compatibility-mode sensitivity; R4-R024 multi-component file-union semantics; R4-B005 positive-only objective/quality gate; R4-B006 token-selector promotion
- **Scheduler correction:** historical v1 88 micro-batches / 11 steps per epoch / 1,100-step horizon is not repaired-v2 authority. With the currently measured 831 active groups and unchanged batch=8/accumulation=8 mechanics, planning arithmetic is 104 micro-batches / 13 optimizer steps per epoch / 1,300 steps over 100 epochs, but no full-run horizon is authorized until objective/selector/population are frozen.
- **Protected/local DATA changed:** NO
- **Model/representation code changed:** NO
- **Training launched:** NO
- **Decision:** repaired-v2 physical DATA acceptance stands; no rollback indicated. Full 100-epoch training remains NOT AUTHORIZED and G8 remains OPEN.
- **Next permitted action:** evidence-honest objective/evaluation design plus versioned selector/grouping/compatibility diagnostics; any changed physical lineage must be rebuilt/rebound locally before training re-authorization.
"""
    log.write_text(log_text, encoding="utf-8")
