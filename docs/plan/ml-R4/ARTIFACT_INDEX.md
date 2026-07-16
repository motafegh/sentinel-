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
