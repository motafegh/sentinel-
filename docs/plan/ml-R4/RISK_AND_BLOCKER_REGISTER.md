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
| R4-B001 | Blocker | Active local DATA/ML bundle is not yet frozen. | Critical | Execute Phase 0. | CLOSED | G0 |
