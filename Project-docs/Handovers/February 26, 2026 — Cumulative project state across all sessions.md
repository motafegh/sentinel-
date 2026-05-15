
---

# SENTINEL SESSION HANDOVER

**Generated:** February 26, 2026 — Cumulative project state across all sessions
**Session exchanges:** ~80+ this session alone

***

## POSITION

**Phase 1 — Foundation**

- **Module 1 — ML Core:** M3.3 Training Loop — **🟡 IN PROGRESS** (20-epoch run running on GPU, epoch 9/20 last seen)
- **Module 5 — Solidity Contracts:** `SentinelToken.sol` + `AuditRegistry.sol` + tests — **✅ COMPLETE**

***

## CODEBASE STATE

**Health: 🟡 Yellow**
**Reason:** Training is running and healthy. M3.3 not closed until epoch 20 prints and results are read. Everything else is green.

***

## FULL DISK INVENTORY

### Data Pipeline — 100% complete

```
 ~/projects/sentinel   main ±  tree BCCC-SCsVul-2024 -L 2         
BCCC-SCsVul-2024
├── BCCC-SCsVul-2024.csv
├── BCCC-SCsVul-2024.md5
├── SourceCodes
│   ├── CallToUnknown
│   ├── DenialOfService
│   ├── ExternalBug
│   ├── GasException
│   ├── IntegerUO
│   ├── MishandledException
│   ├── NonVulnerable
│   ├── Reentrancy
│   ├── Timestamp
│   ├── TransactionOrderDependence
│   ├── UnusedReturn
│   └── WeakAccessMod
└── Sourcecodes.md5

14 directories, 3 files



 # current ml folder tree 
 
 ~/projects/sentinel   main ±  tree ml -L 3 -I '*.pt|*.json|*.log|*.sol'
ml
├── __init__.py
├── __pycache__
│   └── __init__.cpython-312.pyc
├── analysis
│   └── data_quality_validation.py
├── checkpoints
├── configs
├── data
│   ├── BCCC-SCsVul-2024_README.md
│   ├── SolidiFI
│   │   ├── Dockerfile
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── __pycache__
│   │   ├── bug_types.conf
│   │   ├── bugs
│   │   ├── code_trans.conf
│   │   ├── contracts
│   │   ├── evaluator.py
│   │   ├── inject_file.py
│   │   ├── inspection.py
│   │   ├── performance.py
│   │   ├── requirements.txt
│   │   ├── sec_methods.conf
│   │   ├── setup.py
│   │   ├── solidifi.egg-info
│   │   └── solidifi.py
│   ├── SolidiFI-benchmark
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── buggy_contracts
│   │   ├── results
│   │   └── scripts
│   ├── SolidiFI-processed
│   ├── archive
│   │   └── old_test_runs
│   ├── graphs
│   ├── graphs_old_backup
│   ├── graphs_old_duplicates
│   ├── graphs_old_stem_naming
│   ├── graphs_v4_test
│   ├── processed
│   │   ├── bccc_full_dataset_results_OLD.json.bak
│   │   ├── contract_labels.csv
│   │   ├── contract_labels_correct.csv
│   │   ├── contracts_metadata.parquet
│   │   ├── contracts_ml_ready_clean.parquet
│   │   ├── contracts_ml_ready_csv.parquet
│   │   └── label_index.csv
│   ├── reports
│   │   └── data_quality_final_report.txt
│   ├── slither_results
│   ├── smartbugs-curated
│   │   ├── ICSE2020_curated_69.txt
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── dataset
│   │   ├── scripts
│   │   └── versions.csv
│   ├── smartbugs-results-master
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── metadata
│   │   ├── plots
│   │   └── results
│   ├── smartbugs-results-master_2.zip
│   ├── smartbugs-wild
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── contracts
│   │   ├── contracts.csv
│   │   ├── contracts.csv.tar.gz
│   │   ├── master.zip
│   │   ├── nb_lines.csv
│   │   ├── script
│   │   └── smartbugs-results
│   ├── splits
│   │   ├── test_indices.npy
│   │   ├── train_indices.npy
│   │   └── val_indices.npy
│   ├── tokens
│   └── tokens_test
├── docker
│   └── Dockerfile.slither
├── logs
│   ├── enrichment.pid
│   └── overnight.pid
├── models
├── notebooks
│   ├── 01_dataset_validation.ipynb
│   └── 02_dual_path
├── poetry.lock
├── pyproject.toml
├── scripts
│   ├── __pycache__
│   │   └── enrich_dataset_with_ast.cpython-312.pyc
│   ├── analyze_token_stats.py
│   ├── ast_extractor_v4_production.py
│   ├── comprehensive_data_validation.py
│   ├── create_label_index.py
│   ├── create_splits.py
│   ├── fix_labels_from_csv.py
│   ├── test_dataloader.py
│   ├── test_dataset.py
│   ├── test_fusion_layer.py
│   ├── test_gnn_encoder.py
│   ├── test_sentinel_model.py
│   └── tokenizer_v1_production.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-312.pyc
│   ├── data
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── bccc_dataset.py
│   │   ├── graphs
│   │   ├── solidifi_dataset.py
│   │   ├── validate_dataset.py
│   │   └── validate_solidifi.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── dual_path_dataset.py
│   ├── inference
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── fusion
│   │   ├── fusion_layer.py
│   │   ├── gnn
│   │   ├── gnn_encoder.py
│   │   ├── sentinel_model.py
│   │   ├── transformer
│   │   └── transformer_encoder.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── slither_wrapper.py
│   │   ├── slither_wrapper_backup_20260206_160828.py
│   │   └── slither_wrapper_turbo.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── focalloss.py
│   │   └── trainer.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── hash_utils.py
│   └── validation
│       ├── __init__.py
│       ├── __pycache__
│       ├── models.py
│       ├── models_v2.py
│       ├── statistical_validation.py
│       ├── test_full_dataset_final.py
│       ├── test_models.py
│       └── test_real_data.py
└── tests
motafeq@ARlenovo  ~/projects/sentinel   main ±  tree ml/src                              
ml/src
├── __init__.py
├── __pycache__
│   └── __init__.cpython-312.pyc
├── data
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── bccc_dataset.cpython-312.pyc
│   │   └── solidifi_dataset.cpython-312.pyc
│   ├── bccc_dataset.py
│   ├── graphs
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   ├── ast_extractor.cpython-312.pyc
│   │   │   ├── ast_extractor_v3.cpython-312.pyc
│   │   │   └── graph_builder.cpython-312.pyc
│   │   ├── ast_extractor.py
│   │   ├── ast_extractor_v2.py
│   │   ├── ast_extractor_v3.py
│   │   └── graph_builder.py
│   ├── solidifi_dataset.py
│   ├── validate_dataset.py
│   └── validate_solidifi.py
├── datasets
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   └── dual_path_dataset.cpython-312.pyc
│   └── dual_path_dataset.py
├── inference
├── models
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── fusion_layer.cpython-312.pyc
│   │   ├── gnn_encoder.cpython-312.pyc
│   │   ├── sentinel_model.cpython-312.pyc
│   │   └── transformer_encoder.cpython-312.pyc
│   ├── fusion
│   │   └── __init__.py
│   ├── fusion_layer.py
│   ├── gnn
│   │   └── __init__.py
│   ├── gnn_encoder.py
│   ├── sentinel_model.py
│   ├── transformer
│   │   └── __init__.py
│   └── transformer_encoder.py
├── tools
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── slither_wrapper.cpython-312.pyc
│   │   └── slither_wrapper_turbo.cpython-312.pyc
│   ├── slither_wrapper.py
│   ├── slither_wrapper_backup_20260206_160828.py
│   └── slither_wrapper_turbo.py
├── training
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── focalloss.cpython-312.pyc
│   │   └── trainer.cpython-312.pyc
│   ├── focalloss.py
│   └── trainer.py
├── utils
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   └── hash_utils.cpython-312.pyc
│   └── hash_utils.py
└── validation
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-312.pyc
    │   ├── models.cpython-312.pyc
    │   └── models_v2.cpython-312.pyc
    ├── models.py
    ├── models_v2.py
    ├── statistical_validation.py
    ├── test_full_dataset_final.py
    ├── test_models.py
    └── test_real_data.py

22 directories, 63 files
 motafeq@ARlenovo  ~/projects/sentinel   main ±  
69 directories, 92 files
```

ml/data/
```

├── graphs/                   68,556 graph .pt files (PyG Data objects)
├── tokens/                   68,570 token .pt files (CodeBERT tokenizations)
├── processed/
│   ├── contract_labels_correct.csv    hash → binary label
│   └── label_index.csv               lightweight split-safe mapping
└── splits/
    ├── train_indices.npy     47,988 samples (70%) — dtype int64, position indices
    ├── val_indices.npy       10,283 samples (15%)
    └── test_indices.npy      10,284 samples (15%)
```

**Pairing:** 68,555 matched graph+token pairs by MD5 hash (filename stem).
**Labels:** NonVulnerable folder = 0 (safe). All 11 other folders = 1 (vulnerable). 60/40 distribution.
**Splits:** Stratified via sklearn. No overlaps. Full coverage range 0–68,554.

***

### Model Architecture — 100% complete, verified

```
ml/src/models/
├── gnn_encoder.py          GNNEncoder — 3×GAT layers → global mean pool → (B, 64)
├── transformer_encoder.py  TransformerEncoder — CodeBERT frozen → CLS token → (B, 768)
├── fusion_layer.py         FusionLayer — concat(64+768=832) → MLP → (B, 64)
└── sentinel_model.py       SentinelModel — GNN + Transformer + Fusion + Linear(64,1) + Sigmoid → (B,)
```

**Exact architecture:**

- `GNNEncoder`: Input `graph.x (N,8)`, `edge_index (2,E)` → Conv1 `GAT(8→8, heads=8, concat=True)` → `(N,64)` → Conv2 `GAT(64→8, heads=8, concat=True)` → `(N,64)` → Conv3 `GAT(64→64, heads=1, concat=False)` → `(N,64)` → `global_mean_pool` → `(B,64)`. Dropout 0.2 between layers.
- `TransformerEncoder`: `microsoft/codebert-base`, ALL params frozen (`requires_grad=False`). Input `(B,512)`. Output `last_hidden_state[:,0,:]` = CLS token `(B,768)`. Wrapped in `torch.no_grad()`.
- `FusionLayer`: Concat `(B,832)` → `Linear(832,256)` → ReLU → Dropout(0.3) → `Linear(256,64)` → ReLU → `(B,64)`
- `SentinelModel`: `Linear(64,1)` → `Sigmoid()` → `.squeeze(1)` → `(B,)` float in. **Already sigmoid-activated.**[^1]
- **Trainable params:** 239,041. **Frozen params:** 124,645,632 (CodeBERT).

**Verified working command:**

```bash
poetry run python ml/scripts/test_sentinel_model.py
# → End-to-end test PASSED
# → output: tensor([0.4887, 0.4958, 0.4987, 0.5086]) on untrained model
```


***

### Dataset + DataLoader — 100% complete

```
ml/src/datasets/dual_path_dataset.py
```

- `DualPathDataset`: lazy loading (files read only at `__getitem__`). Pairing by MD5 hash. Accepts `indices: List[int]`.
- `__getitem__` returns: `graph` (PyG Data), `tokens` dict (`input_ids (512)`, `attention_mask (512)`), `label` (`torch.long` scalar)
- `dual_path_collate_fn`: **MUST be used as `collate_fn`**. Uses `Batch.from_data_list()` for variable-size graphs. Returns **tuple** `(batched_graphs, batched_tokens, batched_labels)`.

**CRITICAL:** Collate returns a **tuple**, not a dict. Unpack as:

```python
graphs, tokens, labels = batch  # correct
batch["graphs"]                  # WRONG — will crash
```

**CRITICAL:** Labels come out as `torch.long`. Must cast before loss:

```python
labels = labels.to(device).float().squeeze()  # squeeze handles [B,1] → [B]
```


***

### Training Module — 100% complete, running

```
ml/src/training/
├── __init__.py          empty package marker
├── focalloss.py         FocalLoss(gamma=2.0, alpha=0.25)
└── trainer.py           TrainConfig + train_one_epoch + evaluate + train
```

**`focalloss.py` — full source:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(predictions, targets, reduction="none")
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
```

**`TrainConfig` defaults:**

```python
epochs=20, batch_size=32, lr=1e-4, weight_decay=1e-2
focal_gamma=2.0, focal_alpha=0.25
checkpoint_dir="ml/checkpoints", checkpoint_name="sentinel_best.pt"
experiment_name="sentinel-training", run_name="baseline"
device=auto (cuda if available)
```

**Run command:**

```bash
cd ~/projects/sentinel
poetry run python -m ml.src.training.trainer
```

**MLflow backend:** SQLite (migrated from file store, deprecated Feb 2026):

```bash
# View UI:
poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db
# Open: http://localhost:5000
```


***

### Solidity Contracts — 100% complete (Phase 1)

```
contracts/src/
├── SentinelToken.sol    ERC20 + ERC20Permit + ERC20Votes. 1B SENT supply.
└── AuditRegistry.sol    UUPS upgradeable. Staking/slashing. submitAudit with ZK verification placeholder.
contracts/test/          Unit + fuzz + invariant tests — COMPLETE
```

**Key design decisions in contracts:**

- `SafeERC20` used for all token transfers — handles non-standard ERC20 variants
- CEI pattern in `stake()`/`unstake()` — reentrancy safe
- `_verifyProof` is a **placeholder** — returns `zkProof.length > 0`. Real verifier comes in Phase 2 (EZKL-generated)
- Storage gap `uint256[^43] private __gap` — correct UUPS pattern
- `ZKMLVerifier.sol` placeholder still needed (not yet created)

***

## TRAINING RESULTS SO FAR

**20-epoch run in progress on CUDA. Last seen: epoch 14/20.**

```
✘ motafeq@ARlenovo  ~/projects/sentinel   main ±  poetry run python -m ml.src.training.trainer
Training on: cuda
Unpaired tokens: 13
Unpaired tokens: 13
Loading weights: 100%|█| 199/199 [00:00<00:00, 4948.49it/s, Materializing param=p
2026-02-26 10:53:31.255 | INFO     | ml.src.models.fusion_layer:__init__:56 - FusionLayer init — input: 832 (64 GNN + 768 Transformer) → output: 64
2026-02-26 10:53:31.257 | INFO     | ml.src.models.sentinel_model:__init__:78 - SentinelModel initialized — GNN + Transformer + Fusion + Head
2026/02/26 10:53:32 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/02/26 10:53:32 INFO mlflow.store.db.utils: Updating database tables
2026/02/26 10:53:33 INFO mlflow.tracking.fluent: Experiment with name 'sentinel-training' does not exist. Creating a new experiment.
Epoch 1/20 | Loss: 0.0707 | Val F1-macro: 0.3151 | Val F1-vuln: 0.0975
  ✓ New best F1: 0.3151 — checkpoint saved
Epoch 2/20 | Loss: 0.0691 | Val F1-macro: 0.6253 | Val F1-vuln: 0.7026
  ✓ New best F1: 0.6253 — checkpoint saved
Epoch 3/20 | Loss: 0.0680 | Val F1-macro: 0.5771 | Val F1-vuln: 0.5749
Epoch 4/20 | Loss: 0.0673 | Val F1-macro: 0.5996 | Val F1-vuln: 0.6035
Epoch 5/20 | Loss: 0.0671 | Val F1-macro: 0.5170 | Val F1-vuln: 0.4556
Epoch 6/20 | Loss: 0.0665 | Val F1-macro: 0.5909 | Val F1-vuln: 0.5841
Epoch 7/20 | Loss: 0.0663 | Val F1-macro: 0.6266 | Val F1-vuln: 0.6509
  ✓ New best F1: 0.6266 — checkpoint saved
Epoch 8/20 | Loss: 0.0655 | Val F1-macro: 0.6492 | Val F1-vuln: 0.7133
  ✓ New best F1: 0.6492 — checkpoint saved
Epoch 9/20 | Loss: 0.0652 | Val F1-macro: 0.6387 | Val F1-vuln: 0.6676
Epoch 10/20 | Loss: 0.0648 | Val F1-macro: 0.6350 | Val F1-vuln: 0.6635
Epoch 11/20 | Loss: 0.0645 | Val F1-macro: 0.6334 | Val F1-vuln: 0.6598
Epoch 12/20 | Loss: 0.0640 | Val F1-macro: 0.6351 | Val F1-vuln: 0.6755
Epoch 13/20 | Loss: 0.0639 | Val F1-macro: 0.6136 | Val F1-vuln: 0.6165
Epoch 14/20 | Loss: 0.0637 | Val F1-macro: 0.6295 | Val F1-vuln: 0.6409

```



***

## ALL DECISIONS MADE (ALL SESSIONS)

| Decision | Chosen | Rejected | Reason |
| :-- | :-- | :-- | :-- |
| Classification type | Binary (0/1) | Multi-class (13) | Baseline first, collapse BCCC folders |
| FusionLayer depth | 2-layer MLP (832→256→64) | 1-layer (832→64) | Non-linear cross-modal combinations need depth |
| Classifier head | Linear(64,1) + Sigmoid | Softmax 2-class | Second softmax neuron always = 1 - first, redundant |
| CodeBERT training | Fully frozen | Fine-tuned | 239K trainable vs 124M frozen; fine-tune after baseline |
| Loss function | Focal Loss γ=2.0, α=0.25 | Plain BCE | Class imbalance 60/40; original paper defaults |
| Optimiser | AdamW lr=1e-4 | Adam | Correct weight decay decoupling; conservative lr for transformer-adjacent |
| Split strategy | Stratified 70/15/15 | Random split | Preserves class distribution across all splits |
| Collate return | Tuple (graphs, tokens, labels) | Dict | Matches actual `dual_path_collate_fn` implementation |
| MLflow backend | SQLite `mlruns.db` | File store | File store deprecated Feb 2026 |
| Config management | `@dataclass` | YAML/Hydra | Single dev, type safety, IDE autocomplete; migrate at MLOps phase |
| Proxy pattern | UUPS | Transparent proxy | Gas efficient; Ali knows the pattern |
| Agent LLM | Ollama local | GPT-4/Claude API | Free, no API cost during development |
| ZK library | EZKL | Custom circuits | Production library, Python bindings, active community |
| Frontend MVP | Streamlit | Next.js | Simpler, unblocks demo |


***

## OPEN DECISIONS

- **Inference threshold:** Currently `0.5` default. Tune on val set per class after baseline run completes. Lower threshold for vulnerable class catches more positives.
- **`ZKMLVerifier.sol` placeholder:** Needs creating before Module 5 is truly complete. Simple interface file, 10 minutes.
- **Config YAML migration:** Migrate `TrainConfig` → Hydra/YAML when entering MLOps Phase 4. Log as ADR then.

***

## ARCHITECTURE LOG (ADRs)

| \# | Decision | Chosen | Rejected | Reason | Revisit if |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 001 | ML MVP path | CodeBERT + GNN dual-path | CodeBERT only | Already building dual-path | Phase 5 if time allows |
| 002 | Proxy pattern | UUPS | Transparent | Gas efficient, Ali knows pattern | Never |
| 003 | Agent LLM | Ollama local | GPT-4/Claude | Free, no API cost | Quality insufficient for demo |
| 004 | ZK library | EZKL | Custom circuits | Production library, Python bindings | EZKL deprecated |
| 005 | Frontend MVP | Streamlit | Next.js | Simpler, unblocks demo | Phase 5 stretch |


***

## CONCEPTS TAUGHT AND LOCKED THIS SESSION

- Binary Cross-Entropy — what it is, why imbalance breaks it
- Focal Loss — `pt` formula, modulating factor `(1-pt)^gamma`, `alpha_t` direction, full formula
- AdamW — vs Adam, weight decay purpose, why filter frozen params (OOM not accuracy)
- `super().__init__()` — parent class setup, mandatory first line in all PyTorch modules
- `torch.no_grad()` — skips computation graph, ~50% memory saving during eval
- `model.train()` vs `model.eval()` — dropout on/off, deterministic predictions
- `@dataclass` — config bag, auto `__init__`, single-field override at call time
- `reduction="none"` in BCE — per-sample loss required before modulating factor multiply
- `torch.where` — vectorised if/else, no Python loop
- Logits vs probabilities — `binary_cross_entropy` vs `binary_cross_entropy_with_logits`
- Dataclass vs YAML — right tool per phase
- Training vs inference — weight updates vs forward-pass-only
- MLflow — experiment/run/param/metric/artifact structure, `step=epoch` for time-series
- Optuna / LoRA / fine-tuning — what they are, when they become relevant
- DVC — what it versions, why needed, when to add
- DAG — in MLOps pipelines and in GNN context
- Binary vs multi-class — current state, migration path when ready
- Multi-experiment patterns — `TrainConfig` overrides, overnight experiment lists

***

## BLOCKERS

None.

***

## PARKED TOPICS

- **Loguru debug verbosity** — model files log at DEBUG, noisy during training. Already fixed with `logger.remove(); logger.add(sys.stderr, level="INFO")` in `trainer.py`
- **`HF_TOKEN` warning** — set env var before long training runs to suppress rate limit warnings
- **Inference threshold tuning** — tune `0.5` on val set after baseline F1 established
- **Head+tail truncation for CodeBERT** — first 256 + last 254 tokens. After baseline.
- **LoRA fine-tuning** — after baseline F1 established
- **GMU (Gated Multimodal Unit)** — replacing FusionLayer. Phase 5 stretch.
- **DVC setup** — `dvc init` + `dvc add ml/data/graphs ml/data/tokens ml/data/splits`. 20 min. Do at start of any session.
- **`ZKMLVerifier.sol` placeholder** — simple interface, 10 min. Before Module 5 is fully closed.
- **Multi-class 13-vulnerability classification** — after binary baseline solid
- **Optuna hyperparameter search** — after baseline F1 established
- **Evidently AI drift detection** — Phase 4
- **Dagster retraining pipeline** — Phase 4
- **CCIP cross-chain / ERC-4337** — Phase 5 stretch

***

## NEXT SESSION — START HERE IN ORDER

**1. Read epoch 20 training results** (may already be done)

```
Paste full output. Looking for:
- Final loss value
- Best val F1-macro (checkpoint epoch)
- Val F1-vulnerable at best checkpoint
```

**2. Open MLflow UI and read the curves**

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db
# http://localhost:5000
# Check: train_loss curve (should decrease monotonically)
# Check: val_f1_macro curve (oscillation narrowing = good)
# Check: val_f1_vulnerable (this is the signal that matters most)
```

**3. DVC setup — 20 minutes, do this before anything else new**

```bash
cd ~/projects/sentinel
poetry run pip install dvc
dvc init
dvc add ml/data/graphs ml/data/tokens ml/data/splits ml/checkpoints
git add .dvc .gitignore
git commit -m "chore(ml): add DVC tracking for data and model artifacts"
```

**4. Build inference API — `ml/src/inference/predictor.py`**

This is the highest-value next build. Takes a `.sol` file → returns `{"risk_score": 0.73, "vulnerable": true}`.

Structure to build:

```
ml/src/inference/
├── __init__.py
├── predictor.py      Predictor class — loads checkpoint, processes one contract, returns score
└── preprocess.py     Single-contract graph extraction + tokenisation (reuses existing pipeline)
```

Key facts for `predictor.py`:

- Load `sentinel_best.pt` via `model.load_state_dict(torch.load(checkpoint_path))`
- `model.eval()` + `torch.no_grad()` always
- Reuse `ASTExtractor` from `ml/src/data/graphs/ast_extractor.py` for graph
- Reuse CodeBERT tokenizer for tokens
- Output: `{"risk_score": float, "vulnerable": bool, "threshold": 0.5}`

**5. FastAPI wrapper — `api/src/routes/audit.py`**

`POST /v1/audit` → accepts `.sol` file → calls `Predictor` → returns JSON result.

***
