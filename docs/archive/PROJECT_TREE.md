# SENTINEL — Active Project Tree

All source files, data, and artifacts currently in use across all modules.
Excludes: `.venv`, `.git`, `.dvc/cache`, `__pycache__`, archived/superseded files, external dataset source files.

---

```
sentinel/
│
├── CLAUDE.md                          # Project guide + milestone tracker
├── README.md
├── pyproject.toml                     # Poetry root config (ml deps)
├── poetry.lock
├── mlruns.db                          # SQLite MLflow backend
├── check_verify.sh                    # Calls verifyProof on-chain (Sepolia)
├── submit_audit.sh                    # Submits audit + proof on-chain (Sepolia)
├── check_status.sh
├── status.sh
│
├── docs/
│   ├── PROJECT_TREE.md               # This file
│   ├── ML_ARCHITECTURE.md            # Model layers, FocalLoss, training loop
│   ├── ML_INFERENCE.md               # Preprocessor, Predictor, FastAPI
│   ├── ML_TRAINING.md                # DualPathDataset, trainer, checkpointing
│   ├── ML_SCRIPTS.md                 # CLI scripts, test scripts, validation
│   ├── ML_DATASET_PIPELINE.md        # Offline data build (extractor, tokenizer, splits)
│   ├── ML_FILE_INVENTORY.md          # Every file: active vs archived
│   ├── CONTRACTS.md                  # AuditRegistry, SentinelToken, UUPS, guards
│   ├── ZKML_PIPELINE.md              # EZKL steps 1–8, proxy model, agreement gate
│   ├── ENCODING_REFERENCE.md         # BN254 field element encoding (little-endian)
│   ├── QUICKSTART.md
│   └── PROJECT_DOCUMENTATION.md
│
├── ml/
│   ├── __init__.py
│   │
│   ├── src/
│   │   ├── models/
│   │   │   ├── sentinel_model.py     # Top-level: GNN + Transformer + Fusion + head
│   │   │   ├── gnn_encoder.py        # 3-layer GAT → global_mean_pool → [B,64]
│   │   │   ├── transformer_encoder.py# Frozen CodeBERT CLS → [B,768]
│   │   │   └── fusion_layer.py       # concat(832) → MLP → [B,64]
│   │   │
│   │   ├── datasets/
│   │   │   └── dual_path_dataset.py  # DualPathDataset + dual_path_collate_fn
│   │   │
│   │   ├── training/
│   │   │   ├── trainer.py            # TrainConfig, train(), evaluate(), resume
│   │   │   └── focalloss.py          # FocalLoss(gamma=2.0, alpha=0.25)
│   │   │
│   │   ├── inference/
│   │   │   ├── preprocess.py         # ContractPreprocessor → (graph, tokens)
│   │   │   ├── predictor.py          # Predictor → score + label
│   │   │   └── api.py                # FastAPI: POST /predict, GET /health
│   │   │
│   │   └── utils/
│   │       └── hash_utils.py         # MD5 hashing — shared by all pipeline stages
│   │
│   ├── scripts/
│   │   │
│   │   │   ── TRAINING ──
│   │   ├── train.py                  # CLI entry point → TrainConfig → train()
│   │   ├── tune_threshold.py         # Val-set threshold sweep → F1-macro criterion
│   │   ├── run_overnight_experiments.py  # 4-experiment sequential launcher
│   │   │
│   │   │   ── OFFLINE DATA PIPELINE (run once — data already built) ──
│   │   ├── ast_extractor_v4_production.py  # Slither → PyG graphs, 11 workers
│   │   ├── tokenizer_v1_production.py      # CodeBERT tokenizer, 11 workers
│   │   ├── create_label_index.py           # Scan graphs → label_index.csv
│   │   ├── create_splits.py                # Stratified 70/15/15 → .npy indices
│   │   │
│   │   │   ── VALIDATION ──
│   │   ├── comprehensive_data_validation.py  # Full data check before training
│   │   ├── analyze_token_stats.py            # Truncation rate + token length stats
│   │   │
│   │   │   ── SMOKE TESTS (manual, not pytest) ──
│   │   ├── test_sentinel_model.py    # End-to-end: DataLoader → full forward pass
│   │   ├── test_dataloader.py        # DataLoader + collate_fn
│   │   ├── test_dataset.py           # DualPathDataset __getitem__
│   │   ├── test_fusion_layer.py      # FusionLayer forward pass
│   │   └── test_gnn_encoder.py       # GNNEncoder forward pass
│   │
│   ├── tests/                        # pytest suite
│   │   ├── conftest.py               # TestClient fixture (session scope)
│   │   └── test_api.py               # /health + /predict integration tests
│   │
│   ├── analysis/
│   │   └── data_quality_validation.py  # DataQualityValidator class
│   │
│   ├── checkpoints/
│   │   ├── run-alpha-tune_best.pt    # ← PRODUCTION  (val F1-macro 0.6686, ep ~26)
│   │   ├── run-more-epochs_best.pt   # (val F1-macro 0.6584, ep 22, killed at ep 25)
│   │   └── sentinel_best.pt          # baseline (val F1-macro 0.6515, ep 16)
│   │
│   ├── data/
│   │   ├── graphs/                   # 68,556 × <md5>.pt  [PyG Data, x=[N,8], y]
│   │   ├── tokens/                   # 68,570 × <md5>.pt  [input_ids[512], mask[512]]
│   │   ├── splits/
│   │   │   ├── train_indices.npy     # 47,988 positions (70%)
│   │   │   ├── val_indices.npy       # 10,283 positions (15%)
│   │   │   └── test_indices.npy      # 10,284 positions (15%) ← never touched
│   │   └── processed/
│   │       ├── contract_labels_correct.csv  # Ground-truth labels (source of truth)
│   │       ├── label_index.csv              # hash → label (lightweight index)
│   │       └── contracts_metadata.parquet   # contract_path, detected_version, success
│   │
│   └── logs/
│       └── overnight.log             # Training run output
│
├── contracts/                        # Foundry project (Module 5 — complete)
│   ├── foundry.toml
│   ├── src/
│   │   ├── AuditRegistry.sol         # UUPS upgradeable, staking, audit submission, pausable
│   │   ├── SentinelToken.sol         # ERC20 + Permit + Votes, 1B SENT supply
│   │   └── IZKMLVerifier.sol         # Interface: verifyProof(proof, publicSignals)
│   ├── test/
│   │   └── SentinelTest.t.sol        # 20 tests: unit + fuzz + invariant
│   └── script/
│       └── Deploy.s.sol              # Deployment script (used for Sepolia deploy)
│
└── zkml/                             # ZK proof pipeline (Module 4 — complete)
    ├── src/
    │   ├── distillation/
    │   │   ├── proxy_model.py        # ProxyModel: 2,625 param MLP (fits EZKL circuit)
    │   │   ├── train_proxy.py        # Knowledge distillation from SentinelModel
    │   │   ├── export_onnx.py        # Export proxy to ONNX (opset 11)
    │   │   └── generate_calibration.py  # Generate calibration data for EZKL
    │   └── ezkl/
    │       ├── setup_circuit.py      # Steps 1–5: gen_settings → calibrate → compile → srs → setup
    │       ├── run_proof.py          # Steps 6–8: gen_witness → prove → verify
    │       └── extract_calldata.py   # proof.json → check_verify.sh + submit_audit.sh
    │
    ├── models/
    │   ├── proxy_best.pt             # Trained proxy weights (2,625 params)
    │   ├── proxy.onnx                # ONNX export for EZKL circuit
    │   └── proxy.onnx.data           # ONNX external data tensor store
    │
    └── ezkl/                         # EZKL circuit artifacts
        ├── settings.json             # Circuit settings (scale=13, input shape)
        ├── calibration.json          # Calibration data
        ├── model.compiled            # Compiled Halo2 circuit
        ├── srs.params                # Structured Reference String (~4 MB)
        ├── proving_key.pk            # Proving key (one-time setup)
        ├── verification_key.vk       # Verification key (one-time setup)
        ├── proof_input.json          # Input to the prover (per audit)
        ├── witness.json              # Generated witness (per audit)
        ├── proof.json                # Final ZK proof → submitted on-chain
        └── verifier_abi.json         # ABI of the on-chain verifier contract
```

---

## Active data flow (end to end)

```
Raw .sol contract
       │
       ▼  ml/src/inference/preprocess.py
  (graph [N,8], tokens [1,512])
       │
       ▼  ml/src/inference/predictor.py
  score ∈ [0,1]  ──────────────────────►  ml/src/inference/api.py
       │                                   POST /predict → JSON response
       │
       ▼  zkml/src/distillation/proxy_model.py
  proxy_score  (2,625-param MLP, ZK-compatible)
       │
       ▼  zkml/src/ezkl/run_proof.py
  proof.json + instances[65]
       │
       ▼  zkml/src/ezkl/extract_calldata.py
  submit_audit.sh  (calldata: proof bytes + publicSignals[])
       │
       ▼  contracts/src/AuditRegistry.sol
  on-chain audit record  (Sepolia testnet)
```

---

## Checkpoint status

| Checkpoint | Val F1-macro | Threshold | Status |
|---|---|---|---|
| `run-alpha-tune_best.pt` | **0.6686** | **0.50** | **Production** |
| `run-more-epochs_best.pt` | 0.6584 | pending sweep | Killed at ep 25/40 |
| `sentinel_best.pt` | 0.6515 | — | Baseline |

Test set (10,284 samples) has **never been evaluated** — reserved for final holdout.

---

## Contracts deployed (Sepolia testnet)

| Contract | Address |
|---|---|
| `SentinelToken` | Deployed — see `contracts/broadcast/Deploy.s.sol/11155111/run-latest.json` |
| `AuditRegistry` (proxy) | Deployed — same broadcast file |
| ZKMLVerifier | Deployed — same broadcast file |

Last successful `submitAudit` tx: block 10595257, `scoreFieldElement=4497` (≈ 0.5490 probability).
