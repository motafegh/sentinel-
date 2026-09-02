# R4 Artifact Index

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P0-CHK-001 | 0 | checkpoint | ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt | 6a220c6b085a8e0b6b8ae8f5b7610d22bee931d56721000d17e3e304b2daa6cb | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | DVC-tracked 281MB |
| R4-P0-CHK-002 | 0 | thresholds | ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL_thresholds.json | ea3c762afcd4b820ac0e61d554f1ead3e6b840d2589696c327cd2580cfedd937 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | DVC-tracked per-class F1-tuned |
| R4-P0-CHK-003 | 0 | checkpoint_state | ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.state.json | 6de3216c5ad388fcdececc90b490fd1260c3df839b07f18cac2c6dea8111871d | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | DVC-tracked epoch=51 |
| R4-P0-MLO-001 | 0 | mlops_config | ml/mlops_config.json | 6192953c1af8f592895fbcd0ee973e00597cb1a9281d3401fc3453fe06dd7ee4 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | Inference API source of truth |
| R4-P0-CFG-001 | 0 | data_config | data_module/config.yaml | 543e37cc8ccb42a5f20889014cc3a64ad0b1370f75ed4e4f13ab61a6a9436b3b | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | Sources/pins/pipeline settings |
| R4-P0-EXP-001 | 0 | export_manifest | data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/manifest.json | 142824d72277899f73c6b6797eae9665acea82f52b579113b1300997e2135008 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 22493 contracts 5 shards v9 |
| R4-P0-EXP-002 | 0 | export_labels | data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/labels.parquet | 26e739b5d82ba512e5a1830817d09609216e2184b79cf4ca7ec2d62ef34e32b5 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 22493 rows |
| R4-P0-EXP-003 | 0 | export_metadata | data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/metadata.parquet | ca65aa695cb1f03242d9485c4ade0e6c250d92fd6a714f4fb21c4c53087f3a03 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 22493 rows |
| R4-P0-SPL-001 | 0 | split_manifest | data_module/data/splits/v3/split_manifest.json | 8b89c544871a6fec30a3489e6fbb2fad5c535dc92fb4e66399a5a3bac14e7b2e | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | seed=42 3036 dedup groups |
| R4-P0-SPL-002 | 0 | split_train | data_module/data/splits/v3/train.jsonl | 03f2a2376f630165d89615ef47a796ea01a015375313208b556d921dd7d6409b | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 18596 contracts |
| R4-P0-SPL-003 | 0 | split_val | data_module/data/splits/v3/val.jsonl | cf9a7b45fabbad2e3581282f69d5adf4fa4d09eb88bce3721544956a01b7506f | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 1983 contracts |
| R4-P0-SPL-004 | 0 | split_test | data_module/data/splits/v3/test.jsonl | b9bb4649283cc7ec1d39b6e4cee980140b1752aea1c1df69e4b17a498d6fd20c | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 1914 contracts |
| R4-P0-SCH-001 | 0 | graph_schema | ml/src/preprocessing/graph_schema.py | 7af67eb785cc9538bbefd02aaeadf88f3a0b7815fc356814597164f2ba3b0ea0 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | v9 schema constants |
| R4-P0-MDL-001 | 0 | model_architecture | ml/src/models/sentinel_model.py | 4b13c65ed0d40ae4aa71a1e1e373747e02598046c85948dea2d76956429d76d3 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | Frozen architecture |
| R4-P0-DRF-001 | 0 | drift_baseline | ml/data/drift_baseline_run12.json | 73ad1e8447ca66b36b27ebe65760d1d4af53067a7ae61b8085a9aec5e5d1324d | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 500 synthetic samples |
| R4-P0-XWK-001 | 0 | crosswalk | data_module/sentinel_data/labeling/crosswalks/dive.yaml | f1a8ca8d8135012eb240be6441fb8bb4b80cc8d8d7b6ddf4f101a91038d1da49 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | DIVE 8 DASP -> 10-class |
| R4-P0-XWK-002 | 0 | crosswalk | data_module/sentinel_data/labeling/crosswalks/solidifi.yaml | cd2cbf11e6f1a73fafc435df08a4466e670be7080ded0a8e0e2fc566ab39000e | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | SolidiFI injection types |
| R4-P0-XWK-003 | 0 | crosswalk | data_module/sentinel_data/labeling/crosswalks/smartbugs_curated.yaml | 711506948c553bf532fd4e07ee1ffa9efd60800a34cdabb1a3e0d67cbd300542 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | SmartBugs DASP direct |
| R4-P0-XWK-004 | 0 | crosswalk | data_module/sentinel_data/labeling/crosswalks/web3bugs.yaml | — | 4b5bd333c | — | UNAVAILABLE | NO | Referenced in config but file does not exist |
| R4-P0-LBL-001 | 0 | source_labels | data_module/data/raw_staging/dive_labels/DIVE_Labels.csv | a260946ec7741ca4212648a0568f419a32e7cac96c9c0dca1fe3d812bb43f029 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 22330 rows DIVE original |
| R4-P0-EVD-001 | 0 | dive_evidence | data_module/audit/2026-06-18_dive_crosswalk_sample_validation.md | e2071d7b525db22ba98e10dd08eb422ee0f2cdddaa7959af9e4347a70c62a0d0 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 150-contract manual review |
| R4-P0-EVD-002 | 0 | dive_evidence | data_module/audit/2026-06-18_dive_slither_agreed_subset_validation.md | 32a0a84cc3f2acf120f6ed5a59b142499998143776097368f8883065ac2af0d9 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 175-contract second review |
| R4-P0-EVD-003 | 0 | dive_evidence | data_module/audit/2026-06-18_dive_externalbug_reentrancy_slither_corroboration.json | 01c22e50127b530614f7c6165e7ca97481ead96eafd615a2ae7f992e1a60aa3c | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 2MB tool corroboration |
| R4-P0-EVD-004 | 0 | bccc_evidence | data_module/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/contracts_clean_v1.4.csv | 93ec9ec7011fe0851d67fff10dbd75777f71d05cba090500b12c3df071e2098a | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | BCCC verified labels v1.4 |
| R4-P0-EVD-005 | 0 | bccc_evidence | data_module/docs/legacy/bccc_deep_dive/2026-06-08_bccc_deep_dive_00_overview.md | a0ac01a9b1aa7b80b9ee1c9dd7bc638467360860118cb854818b10fd22ca1816 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | BCCC 5-phase overview |
| R4-P0-EVD-006 | 0 | smartbugs_evidence | data_module/data/verification/smartbugs_curated_recall_test/report.json | 6f97760897fa4c9363e441e505dde102fd18178dd94e345f4f7a059dea50138c | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 94.4% recall gate |
| R4-P0-EVD-007 | 0 | data_audit | data_module/2026-06-13_DATA_MODULE_AUDIT_v2_45pct_leakage_finding.md | a8c1cde25f7928e607e0c83b1f60e2a16df330f83cba3b4cd27c39197fe7c405 | 4b5bd333c | Historical | AVAILABLE_VERIFIED | YES | 45% leakage + DoS patch |

## Phase 1 artifacts

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P1-FND-001 | 1 | findings | findings/02A_dive_previous_evidence_recovery.md | — | 6febb4283 | New | AVAILABLE | NO | DIVE evidence recovery report |
| R4-P1-FND-002 | 1 | findings | findings/02B_bccc_previous_evidence_recovery.md | — | 6febb4283 | New | AVAILABLE | NO | BCCC evidence recovery report |
| R4-P1-FND-003 | 1 | findings | findings/02C_other_sources_and_manual_evidence_recovery.md | — | 6febb4283 | New | AVAILABLE | NO | Other sources evidence recovery |
| R4-P1-FND-004 | 1 | findings | findings/02D_model_run_and_export_lineage.md | — | 6febb4283 | New | AVAILABLE | NO | Run12 lineage investigation |
| R4-P1-FND-005 | 1 | findings | findings/02_previous_evidence_recovery_summary.md | — | 6febb4283 | New | AVAILABLE | NO | Phase 1 summary + G1 report |
| R4-P1-MAN-001 | 1 | evidence_inventory | manifests/evidence_inventory.jsonl | — | 6febb4283 | New | AVAILABLE | NO | 27 structured evidence items |

## Phase 2 artifacts

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P2-PLN-001 | 2 | execution_plan | runs/2026-08-11_PHASE2_label_corruption_reconstruction_execution_plan.md | — | 96037edd8 | New | AVAILABLE | NO | Read-only reconstruction plan |
| R4-P2-FND-001 | 2 | findings | findings/03_source_authority_matrix.md | — | d908cb7f | New | AVAILABLE | NO | Active/deferred source authority by class |
| R4-P2-FND-002 | 2 | findings | findings/03_source_semantics_cards.md | — | 27ce7cf6 | New | AVAILABLE | NO | Native positive/negative semantics by source |
| R4-P2-FND-003 | 2 | findings | findings/03_crosswalk_effect_table.md | — | 180b17ee | New | AVAILABLE | NO | Dropped/lossy/unsupported mappings and DoS patch |
| R4-P2-FND-004 | 2 | findings | findings/03_merger_sensitivity_table.md | — | 96650974 | New | AVAILABLE | NO | Positive precedence, weak zero provenance, verification boundary |
| R4-P2-FND-005 | 2 | findings | findings/03_all_zero_decomposition.md | — | d1a3b99d | New | AVAILABLE | NO | Historical-zero origin taxonomy |
| R4-P2-FND-006 | 2 | findings | findings/03_population_reconciliation.md | — | fe6d8028 | New | AVAILABLE | NO | Run12 2,635-row count reconciliation |
| R4-P2-FND-007 | 2 | findings | findings/03_quantification_matrix.md | — | 3d38ab36 | New | AVAILABLE | NO | Source/class/target/mechanism counts and bounded unavailable cross-tabs |
| R4-P2-FND-008 | 2 | findings | findings/03_label_corruption_reconstruction_summary.md | — | a9a2b1e9 | New | AVAILABLE | NO | G2 PASS assessment and next action |
| R4-P2-MAN-001 | 2 | trace_manifest | manifests/phase2_end_to_end_traces.jsonl | — | 353d68c0 | New | AVAILABLE | NO | Representative source→ML target traces |

## Phase 3 artifacts — validated production ledger (G3 PASS)

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P3-PLN-001 | 3 | execution_plan | runs/2026-08-11_PHASE3_evidence_ledger_execution_plan.md | — | 77794238 | New | AVAILABLE | NO | Full Phase-3 work packages and materialization boundary |
| R4-P3-SCH-001 | 3 | schema | schemas/evidence_ledger_row.v1.schema.json | — | d75b3d8f | New | AVAILABLE | NO | Contract×class ledger schema; explicit unresolved-zero state |
| R4-P3-SCH-002 | 3 | schema | schemas/evidence_item.v1.schema.json | — | 6dcad5d8 | New | AVAILABLE | NO | Evidence scope/independence/provenance schema |
| R4-P3-SCH-003 | 3 | schema | schemas/evidence_ledger_manifest.v1.schema.json | — | fe235ab4 | New | AVAILABLE | NO | Materialization/validation manifest schema |
| R4-P3-SCR-001 | 3 | validator | scripts/p3_validate_evidence_ledger.py | — | 43817dab | New | AVAILABLE | NO | Semantic + population + leakage validator |
| R4-P3-TST-001 | 3 | fixture | fixtures/p3_valid_ledger_fixture.jsonl | — | 8036ac61 | New | AVAILABLE | NO | Complete one-contract ten-class valid ledger fixture |
| R4-P3-TST-002 | 3 | fixture | fixtures/p3_valid_evidence_fixture.jsonl | — | f40c4cb6 | New | AVAILABLE | NO | Injection evidence fixture |
| R4-P3-TST-003 | 3 | fixture | fixtures/p3_valid_manifest_fixture.json | — | 47b31dc1 | New | AVAILABLE | NO | One-contract valid manifest fixture |
| R4-P3-TST-004 | 3 | fixture | fixtures/p3_invalid_ledger_cases.jsonl | — | 734e25e8 | New | AVAILABLE | NO | Ten targeted invalid semantic cases |
| R4-P3-TST-005 | 3 | test | scripts/test_p3_validate_evidence_ledger.py | — | 1b42f2c6 | New | AVAILABLE_VERIFIED | NO | Deterministic unittest harness; exercised in Phase-3 framework CI and local gate |
| R4-P3-FND-001 | 3 | findings | findings/04_phase3_state_mapping.md | — | a9a5d7f9 | New | AVAILABLE | NO | Conservative Phase-2→ledger state initialization |
| R4-P3-FND-002 | 3 | findings | findings/04_evidence_ledger_schema_report.md | — | 7b9ef185 | New | AVAILABLE | NO | Framework coverage and materialization boundary |
| R4-P3-MAN-001 | 3 | evidence_items | manifests/evidence_items_v1.jsonl | f0b2684d1b59272a549e61801287cf381e312b3af429507fcd06e60a3705f36d | b8911daed | New | AVAILABLE_VERIFIED | NO | 10 source/category-scoped evidence items; historical DoS patch represented as transformation evidence |
| R4-P3-MAN-002 | 3 | ledger_manifest_draft | manifests/evidence_ledger_v1.manifest.json | — | 602f6d8b | New | SUPERSEDED | NO | Original remote-only DRAFT manifest; superseded by materialized manifest |
| R4-P3-LED-001 | 3 | evidence_ledger | ledger/evidence_ledger_v1.parquet | 3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7 | 17fa20495 | New | AVAILABLE_VERIFIED | NO | 22,493 contracts × 10 classes = 224,930 unique rows; 2,245,123 bytes |
| R4-P3-MAN-003 | 3 | ledger_manifest | manifests/evidence_ledger_v1.materialized.json | 7fac5025a913e83a157231cd1034fcd555c5247e80e3ffe771d6b9681bf05c3e | 17fa20495 | New | AVAILABLE_VERIFIED | NO | VALIDATED manifest; generation commit b8911daed; binds ledger/evidence/strict report identities |
| R4-P3-FND-003 | 3 | semantic_validation | findings/04_evidence_ledger_validation_report.json | 6b9dc920bc25f6cfe19d395a755c10f6b0e5190fe67e6c4ea40675b5e57ae56f | 17fa20495 | New | AVAILABLE_VERIFIED | NO | 224,930 unique keys; frozen split/labels/source/class counts verified; zero errors/warnings |
| R4-P3-FND-004 | 3 | strict_validation | findings/04_evidence_ledger_strict_validation_report.json | acd54021e8ff614c5517b1dbc0eecbcf20ac076aa43a12d9713837e4a2427b2b | 17fa20495 | New | AVAILABLE_VERIFIED | NO | Strict schema + semantic validation PASS; zero surface/semantic errors |
| R4-P3-FND-005 | 3 | artifact_binding | findings/04_evidence_ledger_artifact_binding_report.json | 0686975ad81df6255a0ae0caa694ff59fb444d3189e108ab67e3c48f16b0152a | 17fa20495 | New | AVAILABLE_VERIFIED | NO | Canonical ledger/evidence/strict-report SHA-256 binding PASS |

## Phase 4 artifacts — targeted gap adjudication (G4 PASS)

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P4-PLN-001 | 4 | execution_plan | runs/2026-08-11_PHASE4_gap_authorization_and_adjudication_plan.md | — | d8b138b1 | New | AVAILABLE | NO | Scope-minimal Phase-4 authorization/adjudication plan |
| R4-P4-AUT-001 | 4 | authorization | authorizations/2026-08-11_R4-GAP-002_authorization.md | — | 0613aeee | New | AVAILABLE | NO | Delegated approval of R4-GAP-002; five mapped DIVE strata only |
| R4-P4-MAN-001 | 4 | population_manifest | manifests/p4_gap002_population_manifest.json | — | 4e5ff9be | New | AVAILABLE_VERIFIED | NO | Phase-3-ledger-bound DIVE positive population counts and group-aware eligibility |
| R4-P4-MAN-002 | 4 | frozen_sample | manifests/p4_gap002_initial_sample.jsonl | 2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9 | 757c368d | New | AVAILABLE_VERIFIED | NO | 100 TRAIN-only contracts; 20 per stratum; no review-group reuse; groups touching val/test excluded |
| R4-P4-BND-001 | 4 | blind_source_bundle | review_bundles/r4_gap002_blind_review_bundle_v1.zip | 2b1ce12fdd96819c89bbb9fe1dfb2d9aa992ec0a05ce32f651c6b834b97ddf38 | 02f254249 | New | AVAILABLE_VERIFIED | NO | Checksum-bound normalized/flattened Solidity for the exact frozen 100-contract sample |
| R4-P4-REV-001 | 4 | semantic_review | reviews/R4-GAP-002/p4_gap002_blind_semantic_review_v1.jsonl | 7d7f0cce287c847df2376ac0f580abf6be05b46f6a2c90b5f00f9b34f8fc5473 | c8f283f5 | New | AVAILABLE_VERIFIED | NO | 100 source-only blind AI primary semantic review rows; no confirmed negatives created |
| R4-P4-FND-001 | 4 | review_report | findings/06_gap002_blind_semantic_review_report.json | — | c8f283f5 | New | AVAILABLE_VERIFIED | NO | Exact per-stratum review counts, descriptive Wilson intervals, and bounded role recommendations |
| R4-P4-FND-002 | 4 | adjudication | findings/06_gap002_blind_semantic_review.md | — | 3f3b6123 | New | AVAILABLE | NO | Source-role interpretation: four DIVE strata masked/excluded; TOD limited to TRAIN_WEAK |

## Phase 5 artifacts — DATA vNext policy and design (G5 PASS)

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P5-PLN-001 | 5 | execution_plan | runs/2026-08-12_PHASE5_data_vnext_policy_design_plan.md | a7a74d85f4c8dfbd8f38193a3fea4be459ba7cf8c40054aed4001c01f351bd88 | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | Design-only Phase-5 execution plan |
| R4-P5-POL-001 | 5 | machine_policy | specs/data_vnext_policy_v1.json | b1cfce9cf85c49e4eea533808005d466e0872e98737d366641e287e2a8cfe094 | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | Accepted DATA vNext source/class/state/role policy v1 |
| R4-P5-SCH-001 | 5 | schema | schemas/data_vnext_label_state_v1.schema.json | 14e414a568f090891cb39b4a9a16b3c710d9d69e2279aace50c310aece98959b | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | Contract×class label/outcome/training-signal schema |
| R4-P5-FND-001 | 5 | specification | findings/07_data_vnext_policy_and_design_specification.md | fdf236a4bf8729a4bf3ee5e3c2c9b0a4dce2efc8666a25fae007204b12a913d4 | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | Implementation-facing semantic specification |
| R4-P5-ADR-001 | 5 | ADR | adrs/ADR-R4-001-label-state-and-training-signal.md | 3aaa724585740572bb5daf912b75df4fe927c370b937ebb93d4b63d081690ff3 | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | Outcome truth separated from training signal |
| R4-P5-ADR-002 | 5 | ADR | adrs/ADR-R4-002-source-class-authority-and-enablement.md | 9a66da37f911981bb38a420969633bb9ca8f26e3ce2ec98aa242094fe578234d | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | First-baseline source/class authority and disabled classes |
| R4-P5-ADR-003 | 5 | ADR | adrs/ADR-R4-003-crosswalk-and-aggregation-semantics.md | 2cfa22fc781635b42f8e2c22d2e7c002227ba185aa7dad2351e49ff0f902d7f5 | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | No-target crosswalk states and evidence aggregation |
| R4-P5-ADR-004 | 5 | ADR | adrs/ADR-R4-004-export-and-ml-consumer-contract.md | 018b26b7ebae0c6332875b41750ee0fa17a3ed26a14e592c761e7b2a9ddc47d8 | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | Explicit v2 export and masked ML consumer contract |
| R4-P5-ADR-005 | 5 | ADR | adrs/ADR-R4-005-lineage-versioning-and-rollback.md | 21cb80d2d9f5912a07d7cab60c369175de609523ebe7c3089286161a95a3c8a1 | 104dd4f6f8a1 | New | AVAILABLE_VERIFIED | NO | Immutable history, versioning, fail-closed publication, rollback |

## Phase 6 artifacts — role partitions and acceptance freeze (G6 PASS)

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P6-PLN-001 | 6 | execution_plan | runs/2026-08-12_PHASE6_partitions_acceptance_freeze_plan.md | 9198a198fbf5ec153b68dd4612d77bbd3dffc5187eaf83fa0030cbbdd673ebeb | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | Phase-6 role/acceptance execution plan |
| R4-P6-INV-001 | 6 | role_support_inventory | manifests/p6_role_support_inventory.json | d6571bb8ede7b358ca2668c4377db92a4895114c97725c4cb4fcebe5d8ef185f | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | Measured Phase-5-authorized strong/weak/unlabeled group support |
| R4-P6-INV-002 | 6 | group_eligibility_inventory | manifests/p6_group_eligibility_inventory.jsonl | e7c380f0a584b4d0bfde15389d51bb73ea3d848b2eda6b3bd18cd534a0e0e6e2 | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | 13,509 leakage-group eligibility rows |
| R4-P6-MAN-001 | 6 | role_group_manifest | manifests/p6_role_group_manifest.jsonl | f3bad495c2273ccec9d7b49b98095e1023681e8c5334e9c8d8be1f0b18dde6bc | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | One frozen role per leakage group |
| R4-P6-MAN-002 | 6 | contract_role_manifest | manifests/p6_contract_role_manifest.jsonl | def09010aafb1681c9866dd346de58409c89473b65d03c6c3adea69b75237174 | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | One role per 22,493 contracts |
| R4-P6-MAN-003 | 6 | role_support_table | manifests/p6_role_support_table.json | be9f0790480aa9a63fb95cdecee706ba9b6a4135e5fe6c32f6752ef2fee80453 | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | Per-role/class support and limitations |
| R4-P6-MAN-004 | 6 | unsupported_roles | manifests/p6_unsupported_roles.json | 23c6206736aa4c04ca947a24c551f73e0a722c1acf8fe816a3b37a66273127f0 | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | Threshold/calibration/acceptance controlled empty roles |
| R4-P6-MAN-005 | 6 | untouched_acceptance | manifests/p6_untouched_acceptance_manifest.json | 499b78e2c59fac8e56362d8ca240e37fb0cda0b12e98e4a2433aeb291ed48318 | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | Frozen empty unsupported untouched-acceptance manifest |
| R4-P6-MAN-006 | 6 | partition_manifest | manifests/p6_partition_manifest.json | f3b9727f5e74e007c0833ccd9001d80d82976894c8365a75b7fb9b63ce12c21d | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | r4-vnext-roles-v1 frozen partition root |
| R4-P6-ADR-001 | 6 | ADR | adrs/ADR-R4-006-role-partition-and-acceptance-freeze.md | 9657278b2dad070a78564a4bf365da23493e020e92a0040cd2dde3934069a132 | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | Leakage-safe partition and empty-acceptance decision |
| R4-P6-FND-001 | 6 | findings | findings/08_phase6_role_partition_and_acceptance_freeze.md | 6c3f9657aa564f8ce16ac36efbb96b4c3545aa087d2542231c73a8c9f76a1070 | 5e981fe96407 | New | AVAILABLE_VERIFIED | NO | Partition/support/exposure interpretation |

## Phase 7 artifacts — DATA vNext implementation and local representation binding (G7 PASS)

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P7-PLN-001 | 7 | execution_plan | docs/plan/ml-R4/runs/2026-08-12_PHASE7_data_vnext_implementation_plan.md | c6b4919cfb6c3e452fbea61a5322d64aefa6235318896a51e78cc715a0dbc135 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Phase-7 implementation plan |
| R4-P7-SCH-001 | 7 | export_schema | data_module/sentinel_data/export/format_schema/v2.yaml | fcd1cbc454c10bc3dcbeae43aabcbf68ef5ba2796c6846d1b2d01c9965d428cc | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Explicit DATA vNext v2 format schema |
| R4-P7-EXP-001 | 7 | manifest | data_module/data/exports/sentinel-r4-vnext-v1/manifest.json | 1fd80ebafa036ccb0065f6a6e9e4ff5309ac6c17661b0ecf45469ecf8e8f0d43 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | G7-passed DATA vNext publication root |
| R4-P7-EXP-002 | 7 | label_states | data_module/data/exports/sentinel-r4-vnext-v1/label_states.parquet | 1a21cf931200c33353111cf5cb6a5f7874f851018bc45dab9db1d888c69be1e5 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | 224,930-row canonical contract×class semantic state |
| R4-P7-EXP-003 | 7 | ml_targets | data_module/data/exports/sentinel-r4-vnext-v1/ml_targets.parquet | dac0673c03767502b3a294d42d8a57d82969c5a612bdfc2a852260576df359b5 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Derived per-contract ten-class target/strength/mask/role projection |
| R4-P7-REG-001 | 7 | source_registry | data_module/data/exports/sentinel-r4-vnext-v1/source_registry.json | f0b640f9396c902dd929584b1b4525b90f023b370c4df38e70ab3399f64f6dcf | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Frozen first-baseline source authority snapshot |
| R4-P7-REG-002 | 7 | crosswalk_registry | data_module/data/exports/sentinel-r4-vnext-v1/crosswalk_registry.json | a384720d47d0ce823c7ef7518921974dd0f396c11144fb0fd865159349de8395 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Frozen vNext crosswalk action snapshot |
| R4-P7-BND-001 | 7 | evidence_snapshot | data_module/data/exports/sentinel-r4-vnext-v1/evidence_snapshot.json | bf7761b42701501c5c256729da149526005dbc6c10dfc6beac0b4db70245bd66 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Ledger/policy/partition evidence bindings |
| R4-P7-BND-002 | 7 | representation_requirements | data_module/data/exports/sentinel-r4-vnext-v1/representation_requirements.json | 04ec986f195a0d81079cbd8414865cd049124529b1e8e76db712b642d8410d39 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Exact non-excluded representation requirement set |
| R4-P7-VAL-001 | 7 | semantic_validation | data_module/data/exports/sentinel-r4-vnext-v1/validation_report.json | f2ebc067fff50d23bd72fdf091d78922bcdef928dab93d243ce3c0a58e54ec5d | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Independent semantic validation report |
| R4-P7-VAL-002 | 7 | representation_binding | data_module/data/exports/sentinel-r4-vnext-v1/representation_binding_report.json | d5bba3b037a0f443c5764bfb4be17da35a3e44201d97f0f671298b9fbf4dc33d | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | 21,657-contract / 64,971-file local physical binding report |
| R4-P7-VAL-003 | 7 | g7_validation | data_module/data/exports/sentinel-r4-vnext-v1/g7_validation_report.json | 9bfd5abca187d9eb8a966e5e914e7203f9c2f70102f899936c46c23ae9aa0d21 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Final representation-required G7 validation report |
| R4-P7-ADR-001 | 7 | ADR | docs/plan/ml-R4/adrs/ADR-R4-007-data-vnext-implementation-and-g7-publication.md | b3e61db0c0e7eb0f132dd35845a3f8f63fb022130877fa5399f21006a8af08aa | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Accepted G7 publication/training-input authority |
| R4-P7-FND-001 | 7 | findings | docs/plan/ml-R4/findings/09_phase7_data_vnext_implementation_and_g7.md | 5e8ea6a0f71284de0d0e3e61d1c270b2c1a05deabfe5cb0241433479aace9e32 | 81d9c547d361 | New | AVAILABLE_VERIFIED | NO | Phase-7 implementation and G7 result |

## Phase 8 artifacts — repaired-v2 acceptance, V10 remediation, and current V2.6 physical gate

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
| R4-P8-ADR-002 | 8 | ADR | adrs/ADR-R4-010-versioned-external-call-representation-correction.md | — | 2026-08-21 decision | New | AVAILABLE_VERIFIED | NO | R4-D-010: preserve v9 history, require versioned V10 call-kind lineage before new full run |
| R4-P8-AUD-001 | 8 | transition_audit_v2 | reviews/R4-GAP-008/v10_transition_audit_v2.json | `5793b059e7e5149424e10a5361a5b0e420b1f86f3630920e36344c5737fd4f9b` | 2026-08-23 tranche | New | AVAILABLE_VERIFIED | NO | 22,540-identity V2.4 transition audit; historical source set for the later exact 20-identity structural investigation |
| R4-P8-REV-001 | 8 | bounded_structural_closure | reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md | — | 150d2ad1fa79 | New | AVAILABLE_VERIFIED | NO | Durable closure: V2.5 bounded 20/20 resolved as 8 index-equivalence + 12 storage-WRITE corrections; zero unexplained drift |
| R4-P8-RUN-004 | 8 | current_restart_checkpoint | runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md | — | a9890332dbca+ | New | AVAILABLE_VERIFIED | NO | Canonical checkpoint updated with passed protected-local Stages A-D and Stage-E full-population blocker; physical acceptance/training false |
| R4-P8-RUN-005 | 8 | full_candidate_staging_protocol | runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md | — | a9890332dbca+ | New | AVAILABLE_VERIFIED | NO | Heterogeneous 22,539 Slither-0.10 + 1 Slither-0.11.5 staged V2.5 protocol executed through Stage E; full gate blocked |
| R4-P8-RUN-006 | 8 | full_population_structural_evidence_plan | runs/2026-08-30_PHASE8_v10_v25_full_population_structural_evidence_plan.md | — | local V2.6 continuation | New | AVAILABLE_VERIFIED | NO | V2.6 continuation: 355/355 drift identities reconciled as 349 WRITE + 6 index-equivalent; later accepted by R4-D-011 |
| R4-P8-RUN-007 | 8 | full_population_structural_analysis | runs/2026-08-30_PHASE8_v10_v25_full_population_structural_analysis.md | — | local V2.6 continuation | New | AVAILABLE_VERIFIED | NO | Preserves historical V2.5 blocker and records the complete bound V2.6 candidate/evidence/audit result |
| R4-P8-RUN-008 | 8 | v10_v26_physical_acceptance_no_launch | runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md | — | 012a33594498 evidence source | New | AVAILABLE_VERIFIED | NO | R4-D-011 acceptance review; exact V2.6 physical root/digest accepted, selector promotion and training unauthorized |
| R4-P8-RUN-009 | 8 | selector_control_equivalence_plan | runs/2026-09-02_PHASE8_selector_control_equivalence_plan.md | — | post-R4-D-011 | New | AVAILABLE_VERIFIED | NO | Read-only full-population prerequisite plan; control reconstruction only, no selector promotion or training authority |
| R4-P8-RUN-010 | 8 | selector_promotion_review | runs/2026-09-02_PHASE8_selector_promotion_review.md | — | b332aa3c6 evidence source | New | AVAILABLE_VERIFIED | NO | Source-first CPU/CUDA/control-equivalence review; R4-D-012 authorizes a new guarded-token candidate only |
| R4-P8-ADR-003 | 8 | ADR | adrs/ADR-R4-011-v10-v26-physical-representation-acceptance.md | — | governance reconciliation | New | AVAILABLE_VERIFIED | NO | Formal exact-root V10 V2.6 physical acceptance and no-launch authority |
| R4-P8-ADR-004 | 8 | ADR | adrs/ADR-R4-012-target-aware-guarded-selector-promotion.md | — | b332aa3c6 evidence source | New | AVAILABLE_VERIFIED | NO | Promotes guarded selector for a new versioned candidate only; physical build/acceptance and training remain separate |
| R4-P8-ACC-001 | 8 | physical_acceptance_manifest | evidence/2026-09-02_v10_v26_physical_acceptance/acceptance.json | `5fc83eff39d4a28db9a5b6b5255a95ad64ee75ca88a948ba99dadb2bc03ee165` | 012a33594498 evidence source | New | AVAILABLE_VERIFIED | NO | Machine-readable R4-D-011 boundary; binds protected-local root/digest, refreshed reports, runtime split, and remaining gates |
| R4-P8-EQV-001 | 8 | selector_control_equivalence | evidence/2026-09-02_selector_control_equivalence/report.json | `636838f376d8991e9ac07d26105aa2f907e535bbf90e4504e11d663f0c656021` | 735eda59dd02 | New | AVAILABLE_VERIFIED | NO | 22,540/22,540 historical-control dynamic token tensors and selected indices equal the R4-D-011 bound payloads; promotion/training false |
| R4-P8-DEC-001 | 8 | selector_promotion_decision | evidence/2026-09-02_selector_promotion/decision.json | `657a4936dc2c3c1beb2850932737f484103409b19189115fa76617bbe640ca1a` | b332aa3c6 evidence source | New | AVAILABLE_VERIFIED | NO | R4-D-012 machine boundary: guarded selector promoted for new candidate construction; R4-D-011 mutation and training false |
| R4-P8-SCR-001 | 8 | full_transition_audit_v3 | scripts/p8_audit_v10_transition_v3.py | — | ff9f4bea4069+ | New | AVAILABLE_VERIFIED | NO | Reuses V2 mechanics and fail-closed re-proves exact bounded 8+12 evidence classes against actual full candidate |
| R4-P8-SCR-002 | 8 | evidence_chain_preflight | scripts/p8_validate_v10_v25_evidence_chain.py | — | cafc3c475dce+ | New | AVAILABLE_VERIFIED | NO | SHA-binds original transition audit, bounded V2.5 report, and merged semantic evidence; protected-local preflight passed |
| R4-P8-SCR-003 | 8 | primary_attempt_driver | scripts/p8_generate_v10_v25_primary_attempt.py | — | 1aa94b7c0351+ | New | AVAILABLE_VERIFIED | NO | Stage A driver; exact 22,539 ordinary primary partition; declared runtime exception never invoked in primary process |
| R4-P8-SCR-004 | 8 | primary_stage_validator | scripts/p8_stage_v10_v25_primary_attempt.py | — | 97617eb23937+ | New | AVAILABLE_VERIFIED | NO | Stage B fail-closed validator/transfer; refuses any failure/population/runtime/token mismatch beyond declared exception set |
| R4-P8-SCR-005 | 8 | full_transition_audit_v4 | scripts/p8_audit_v10_transition_v4.py | — | local V2.6 continuation | New | AVAILABLE_VERIFIED | NO | Reuses complete V2 mechanics and independently re-proves the exact 355-case V2.6 evidence against the bound full candidate |
| R4-P8-SCR-006 | 8 | full_population_write_evidence | scripts/p8_collect_v10_v25_full_population_write_evidence.py | — | local V2.6 continuation | New | AVAILABLE_VERIFIED | NO | Duplicate-safe persistent-storage evidence for all WRITE drift groups; exact multiplicity retained |
| R4-P8-SCR-007 | 8 | structural_repeat_generator | scripts/p8_generate_v10_v25_structural_repeat.py | — | local V2.6 continuation | New | AVAILABLE_VERIFIED | NO | Exact audit-derived primary-runtime repeat generator used for three fresh 355-identity runs |
| R4-P8-SCR-008 | 8 | full_population_probe | scripts/p8_probe_v10_v25_full_population.py | — | local V2.6 continuation | New | AVAILABLE_VERIFIED | NO | All-pairs exact labelled graph proof over reference, candidate, and three repeats after evidence-bounded canonicalization |
| R4-P8-SCR-009 | 8 | selector_control_equivalence_verifier | scripts/p8_verify_v10_bound_token_control_equivalence.py | `426e209be80bf65f69c64859d1db25dc1a9db57911d27f9f6c343f77e22cd707` | 735eda59dd02 | New | AVAILABLE_VERIFIED | NO | Multiprocessing source-to-token verifier for all 22,540 accepted identities; writes compact local evidence and never mutates representations |

### Current Phase-8 V10 protected-local evidence note

The current deterministic V2.5 bounded reproducibility JSON and merged
semantic-evidence JSON are protected-local/generated evidence and are **not
committed repository artifacts**. They are persistent under
`data_module/data/r4-v10-v25-evidence-deterministic-v2/` and their exact
SHA-256 identities are bound by the current restart/staging records:

- bounded V2.5 report SHA-256: `67192b2a81383af74f70ed3ed6e1c0dfbd50d6b9525a9a939a250653e2a53adc`;
- merged semantic evidence SHA-256: `16e264fbed941ab16ead47dacd4e19c7a02511539e0950664e2cdc28373bfa8e`;
- evidence-chain preflight SHA-256: `1d28f9b2f4a597ff04f62052cad95713dafd6169f5d0f97de100fde452e542cb`.

The earlier `/tmp`-bound SHA pair `cffcb74c...` / `483012e3...` is historical
and superseded for current execution because its exact merged semantic input
was not durable and fresh regeneration exposed byte-level nondeterminism in
informational probe output.

The protected-local full-candidate execution is also verified locally but is not
a committed artifact publication:

- Stage-A report SHA-256: `a227a3a6d2340c7f3ab3bb15687fad4002f66c29ebd05d64108f7dce13deeb76`;
- Stage-B report SHA-256: `3f24b4b294340580cf4579cc5b7c230b6c803cabe7234aacc31f6a3d0bf4fdf0`;
- Stage-C report SHA-256: `0994b8921905b1db82f01f5f16868a85252cc432cac615c85f73692107a301d8`;
- Stage-D binding report SHA-256: `3cab4b19d7708b8d706699577dbfcaebf504b6ceb918c60a21956441fa238774`;
- Stage-E audit SHA-256: `b469e63e91e22b75eea1f66432e7cbddf4461289b32c4f77ddf7bba39f82031f`;
- full-population probe SHA-256: `d9c512015d180c67fee6dc8848952992914abed07f373ebdc7845a3398b1b3b4`.

Stages A-D pass, but Stage E is false with 298 unapproved structural-drift
identities. Do not mark these protected-local JSON files `AVAILABLE_VERIFIED`
unless they are deliberately published in a later versioned evidence snapshot.

That V2.5 candidate is historical. The current protected-local V2.6 lineage is:

- Stage-A report SHA-256: `46f63d24ed614a6dfd427c3c6c19512578e9c5092e05300a3fb1445002e753cf`;
- Stage-B report SHA-256: `85286706b189fa09d06e4113aeeb4168bb283a9bd0ce6bdd8e64b862ae4cb41f`;
- Stage-C report SHA-256: `3df8583b0929086b5ef9a7d4135499fa47f2989e1115f7a8d3e0dab2ef1f15bc`;
- Stage-D binding report SHA-256: `93a4d15e0793d7b144fc5cc98dbd29627f0d7372cb56e2431a79f8d02c761311`;
- Stage-D binding digest: `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- 355-case full-population probe SHA-256: `9a1cf96465613b61fae2d10ccaa81def0548663a4c4711ca745841f6354e7a55`;
- final V4 audit SHA-256: `c6ddc61b8005a688d422f4f8de28118fa3e644b9648d070ef53972ec9f2191ce`.

The V2.6 V4 audit passes with zero unexplained drift. R4-D-011 physically
accepts only the exact root and digest above. Selector promotion and training
authorization remain false pending their separate decisions.

## Availability

- `AVAILABLE_VERIFIED`
- `AVAILABLE_UNVERIFIED`
- `UNAVAILABLE`
- `CORRUPT`
- `SUPERSEDED`

## Rules

- Path is not identity; hash the artifact.
- Hash deterministic directory manifests for directories.
- A rerun gets a new artifact ID.
- New reproduction is not historical recovery.
