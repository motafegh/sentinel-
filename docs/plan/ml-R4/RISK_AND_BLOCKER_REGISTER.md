# R4 Risk and Blocker Register

| ID | Type | Description | Impact | Mitigation | Status | Gate |
|---|---|---|---|---|---|---|
| R4-R001 | Risk | Agent repeats prior contract audits instead of reusing them. | High | Non-duplication policy and mandatory gap IDs. | OPEN | G1-G4 |
| R4-R002 | Risk | Historical zeros are treated as confirmed negatives. | Critical | Explicit label states and masks. | OPEN | G2-G7 |
| R4-R003 | Risk | Tool-correlated evidence is counted as independent confirmation. | High | Independence groups. | OPEN | G3-G4 |
| R4-R004 | Risk | Architecture changes distract from label repair. | High | Architecture freeze. | MITIGATED | G8 |
| R4-R005 | Risk | Threshold/calibration reuse inflates evaluation. | Critical | Separate leakage-safe roles. | OPEN | G6-G9 |
| R4-R006 | Risk | Web3Bugs declared enabled in config but entirely absent (no data/crosswalk/parser). | High | Phase 0 finding F0.4; config-vs-reality contradiction registered. DATA vNext must either acquire Web3Bugs or explicitly exclude it. | OPEN | G5 |
| R4-R007 | Risk | GasException has zero support in the active split (0 positives in train/val/test). | Medium | Phase 0 finding F0.5; class is effectively unsupported. DATA vNext must address or explicitly disable. | OPEN | G5-G9 |
| R4-R008 | Risk | 836 contracts have labels but no representations (cannot be loaded by ML). | Low | Phase 0 finding F0.8; recorded in export manifest n_contracts_with_reps=21657. | OPEN | G7 |
| R4-R009 | Risk | locked_files.sha256 is stale (4/5 source file hashes do not match). | Low | Phase 0 finding F0.6; stale lock is NOT a protected R4 artifact. Current on-disk hashes are the baseline. | OPEN | — |
| R4-R010 | Risk | 2,635-contract discrepancy between Run12 training population (19,858) and current export (22,493). Split version used by Run12 is unknown. | High | Query MLflow for Run12 export hash; determine if export was regenerated after training. UNRESOLVED at G1. | OPEN | G1-G7 |
| R4-R011 | Risk | BCCC v1.4 verified labels exist but are DEFERRED in config; 90%+ label reduction for Reentrancy/CTU/DoS. | High | Config says DEFERRED; no pipeline change needed yet. Evidence exists for Phase 2 when KEEP/DROP decisions are made. | OPEN | G2 |
| R4-R012 | Risk | DIVE EB TP count discrepancy: per-table shows 3 TP / 72 FP, tally claims 4 TP / 71 FP (off by 1). | Low | Verify against scratch file before DROP/KEEP decision. | OPEN | G2 |
| R4-R013 | Risk | Benchmark manifest has 74 entries but documented as 66. Contamination risk if discrepancy indicates duplicate/overlap. | Low | Update documentation to match actual count; verify no duplicates. | OPEN | G7 |
