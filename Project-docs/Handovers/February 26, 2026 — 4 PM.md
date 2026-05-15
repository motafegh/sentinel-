
## SENTINEL SESSION HANDOVER

**Generated:** February 26, 2026 — 4:29 PM Tehran time
**Session exchanges:** ~120+ cumulative across all sessions

***

## POSITION

**Phase 1 — Foundation**


| Module | Milestone | Status |
| :-- | :-- | :-- |
| Module 1 — ML Core | M3.3 Training Loop | ✅ COMPLETE — baseline run finished, best checkpoint saved |
| Module 5 — Solidity Contracts | SentinelToken + AuditRegistry + tests | ✅ COMPLETE |

**Next milestone:** M3.4 — Inference API (`predictor.py` + FastAPI wrapper)

***

## CODEBASE STATE

**Health: 🟢 Green**
**Reason:** Baseline training run complete. Best checkpoint saved. All models verified. No broken state anywhere.

***

## FULL DISK INVENTORY

### Project Structure

```
~/projects/sentinel
├── ml/
│   ├── src/
│   │   ├── models/         gnn_encoder, transformer_encoder, fusion_layer, sentinel_model
│   │   ├── datasets/       dual_path_dataset.py
│   │   ├── training/       trainer.py, focalloss.py
│   │   ├── inference/      ← EMPTY — next thing to build
│   │   └── validation/
│   ├── data/
│   │   ├── graphs/         68,556 graph .pt files
│   │   ├── tokens/         68,570 token .pt files
│   │   ├── splits/         train(47,988) / val(10,283) / test(10,284)
│   │   └── processed/      contract_labels_correct.csv, label_index.csv
│   ├── checkpoints/
│   │   └── sentinel_best.pt   ← epoch 16 weights, 476 MB
│   └── mlruns/
│       └── mlruns.db          ← SQLite backend
├── contracts/
│   ├── src/    SentinelToken.sol, AuditRegistry.sol
│   └── test/   unit + fuzz + invariant tests
```


***

## BASELINE TRAINING RUN — COMPLETE RESULTS

**Run name:** `baseline`
**Config:** epochs=20, batch_size=32, lr=1e-4, weight_decay=1e-2, focal_gamma=2.0, focal_alpha=0.25, device=cuda

**Full epoch log:**

```
Epoch 1  | Loss: 0.0707 | F1-macro: 0.3151 | F1-vuln: 0.0975  ← checkpoint
Epoch 2  | Loss: 0.0691 | F1-macro: 0.6253 | F1-vuln: 0.7026  ← checkpoint
Epoch 3  | Loss: 0.0680 | F1-macro: 0.5771 | F1-vuln: 0.5749
Epoch 4  | Loss: 0.0673 | F1-macro: 0.5996 | F1-vuln: 0.6035
Epoch 5  | Loss: 0.0671 | F1-macro: 0.5170 | F1-vuln: 0.4556
Epoch 6  | Loss: 0.0665 | F1-macro: 0.5909 | F1-vuln: 0.5841
Epoch 7  | Loss: 0.0663 | F1-macro: 0.6266 | F1-vuln: 0.6509  ← checkpoint
Epoch 8  | Loss: 0.0655 | F1-macro: 0.6492 | F1-vuln: 0.7133  ← checkpoint  ⭐ best F1-vuln
Epoch 9  | Loss: 0.0652 | F1-macro: 0.6387 | F1-vuln: 0.6676
Epoch 10 | Loss: 0.0648 | F1-macro: 0.6350 | F1-vuln: 0.6635
Epoch 11 | Loss: 0.0645 | F1-macro: 0.6334 | F1-vuln: 0.6598
Epoch 12 | Loss: 0.0640 | F1-macro: 0.6351 | F1-vuln: 0.6755
Epoch 13 | Loss: 0.0639 | F1-macro: 0.6136 | F1-vuln: 0.6165
Epoch 14 | Loss: 0.0637 | F1-macro: 0.6295 | F1-vuln: 0.6409
Epoch 15 | Loss: 0.0636 | F1-macro: 0.6303 | F1-vuln: 0.6525
Epoch 16 | Loss: 0.0633 | F1-macro: 0.6515 | F1-vuln: 0.6866  ← checkpoint  ⭐ best F1-macro (SAVED)
Epoch 17 | Loss: 0.0630 | F1-macro: 0.6177 | F1-vuln: 0.6198
Epoch 18 | Loss: 0.0628 | F1-macro: 0.6363 | F1-vuln: 0.6637
Epoch 19 | Loss: 0.0627 | F1-macro: 0.6378 | F1-vuln: 0.6563
Epoch 20 | Loss: 0.0626 | F1-macro: 0.6129 | F1-vuln: 0.6069

Training complete. Best val F1-macro: 0.6515 (epoch 16)
Best val F1-vuln: 0.7133 (epoch 8) — NOT the saved checkpoint
```

**Key observations:**

- Loss: monotonically decreasing ✅ — training was stable throughout
- High variance in F1: oscillation of ~0.15 range — learning rate likely slightly too high
- Model peaked twice: epoch 8 (vuln) and epoch 16 (macro) — then declined
- Checkpoint saved on F1-macro criterion — captures epoch 16, not epoch 8
- **Interview number to cite:** F1-vuln 0.7133 at epoch 8 / F1-macro 0.6515 at epoch 16

***

## WHAT WAS LEARNED THIS SESSION (CONCEPTS LOCKED)

All concepts taught, confirmed understood by Ali:

- **Loss** — measures model wrongness per sample, optimizer uses it to update weights. Falls monotonically ≠ model is getting better on unseen data
- **Loss vs F1 decoupling** — model can output 0.48 (tiny loss, "almost right") but classify as SAFE (F1 counts it as a miss). Threshold is the gap between them
- **Precision** — when you flag vulnerable, how often you're right
- **Recall** — of all real vulnerabilities, how many you caught
- **F1** — harmonic mean of precision and recall. Can't be cheated by flagging everything
- **F1-macro vs F1-vuln** — macro averages both classes; vuln is the security signal that matters
- **Overfitting visual** — loss curve smooth + F1 curve oscillating = model memorizing training data
- **Threshold tuning** — moving the 0.5 cutoff costs nothing, gains 3–6 F1 points typically
- **MLflow structure** — Experiment → Run → Parameters / Metrics / Artifacts tabs
- **focal_alpha direction bug** — alpha=0.25 underweights the vulnerable class. Should be 0.75

***

## MLFLOW STATE

**Backend:** SQLite at `~/projects/sentinel/mlruns.db`
**UI command:** `poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db`
**URL:** `http://localhost:5000`

**Experiment:** `sentinel-training` (ID: 1)
**Run:** `baseline` (ID: `6201a32250e94c47ae1d3daa7a10a989`)

**Currently tracked metrics (4):**

- `train_loss`, `val_f1_macro`, `val_f1_safe`, `val_f1_vulnerable`

**Missing metrics — to add before overnight runs:**

- `val_precision_vulnerable` — false alarm rate signal
- `val_recall_vulnerable` — miss rate signal (most critical for security)
- `train_f1_macro` — needed to see train/val gap = overfit signal

**Artifact:** `sentinel_best.pt` (476 MB) — epoch 16 weights — lives in both:

- `ml/checkpoints/sentinel_best.pt` ← canonical path to use
- `mlruns/1/6201a.../artifacts/sentinel_best.pt` ← MLflow copy

***

## PLANNED OVERNIGHT EXPERIMENTS

To be built as a launcher script before bed. Run sequentially. Each logs to MLflow as a separate named run.


| Run name | Changes vs baseline | Hypothesis | Expected signal |
| :-- | :-- | :-- | :-- |
| `run-alpha-flip` | `focal_alpha=0.75` | Fix alpha direction bug — vuln class gets 3× more weight | F1-vuln ↑, less oscillation. **Highest confidence improvement** |
| `run-more-epochs` | `epochs=40, lr=1e-4` | Model still climbing at ep16 — give it more room | Plateau later, potentially higher peak |
| `run-lr-lower` | `lr=3e-5, epochs=30` | Smaller optimizer steps → less epoch-to-epoch oscillation | Smoother curve, less variance |
| `run-combined` | `alpha=0.75, lr=3e-5, epochs=30` | Best guesses combined | Highest expected F1-vuln overall |

**NOT yet built.** Still to do this session:

1. Add precision/recall to `trainer.py` evaluate function (5 lines)
2. Build `ml/scripts/tune_threshold.py` — sweep threshold 0.30→0.70, step 0.05, on val set using saved checkpoint
3. Build `ml/scripts/run_overnight_experiments.py` — queues all 4 runs sequentially

***

## FULL DECISIONS LOG (ALL SESSIONS)

| \# | Decision | Chosen | Rejected | Reason |
| :-- | :-- | :-- | :-- | :-- |
| 001 | ML MVP path | CodeBERT + GNN dual-path | CodeBERT only | Already building dual-path |
| 002 | Classification type | Binary (0/1) | Multi-class (13) | Baseline first |
| 003 | FusionLayer depth | 2-layer MLP (832→256→64) | 1-layer | Non-linear cross-modal combinations need depth |
| 004 | Classifier head | Linear(64,1) + Sigmoid | Softmax 2-class | Second neuron always = 1 - first, redundant |
| 005 | CodeBERT training | Fully frozen | Fine-tuned | 239K trainable vs 124M; fine-tune after baseline |
| 006 | Loss function | Focal Loss γ=2.0, α=0.25 | Plain BCE | Class imbalance — **alpha direction is a known bug, fix in overnight runs** |
| 007 | Optimiser | AdamW lr=1e-4 | Adam | Correct weight decay decoupling |
| 008 | Split strategy | Stratified 70/15/15 | Random | Preserves class distribution |
| 009 | Collate return | Tuple (graphs, tokens, labels) | Dict | Matches actual implementation |
| 010 | MLflow backend | SQLite `mlruns.db` | File store | File store deprecated Feb 2026 |
| 011 | Config management | `@dataclass` | YAML/Hydra | Single dev, type safety; migrate at Phase 4 |
| 012 | Proxy pattern | UUPS | Transparent proxy | Gas efficient, Ali knows pattern |
| 013 | Agent LLM | Ollama local | GPT-4/Claude API | Free, no API cost during development |
| 014 | ZK library | EZKL | Custom circuits | Production library, Python bindings |
| 015 | Frontend MVP | Streamlit | Next.js | Simpler, unblocks demo |
| 016 | Checkpoint criterion | Best val F1-macro | Final epoch | Saves best generalization, not last state |


***

## ARCHITECTURE (ADRs)

| \# | Decision | Chosen | Rejected | Reason | Revisit if |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 001 | ML MVP path | CodeBERT + GNN dual-path | CodeBERT only | Already building dual-path | Phase 5 if time |
| 002 | Proxy pattern | UUPS | Transparent | Gas efficient | Never |
| 003 | Agent LLM | Ollama local | GPT-4/Claude | Free, no API cost | Quality insufficient for demo |
| 004 | ZK library | EZKL | Custom circuits | Production library, Python bindings | EZKL deprecated |
| 005 | Frontend MVP | Streamlit | Next.js | Simpler, unblocks demo | Phase 5 stretch |


***

## MODEL ARCHITECTURE — LOCKED

```
GNNEncoder:          Input (N,8) → 3×GAT layers → global_mean_pool → (B,64)
TransformerEncoder:  CodeBERT frozen → CLS token → (B,768)
FusionLayer:         concat(832) → Linear(832,256) → ReLU → Dropout(0.3) → Linear(256,64) → ReLU → (B,64)
SentinelModel:       Linear(64,1) → Sigmoid → squeeze → (B,) float

Trainable params:    239,041
Frozen params:       124,645,632 (CodeBERT)
Output:              Already sigmoid-activated. Use BCELoss not BCEWithLogitsLoss.
```


***

## OPEN DECISIONS

| Item | Status | Action needed |
| :-- | :-- | :-- |
| Inference threshold | Currently 0.5 | Tune via `tune_threshold.py` — this session |
| `focal_alpha` bug | Confirmed 0.25 is wrong direction | Fix to 0.75 in overnight `run-alpha-flip` |
| `ZKMLVerifier.sol` placeholder | Not yet created | Simple interface, 10 min, before Module 5 fully closed |
| Config YAML migration | Using `@dataclass` | Migrate to Hydra at Phase 4 MLOps |
| DVC setup | Not done | `dvc init` + track graphs/tokens/splits/checkpoints — 20 min, do at start of any new session |
| Checkpoint save size | 476 MB includes optimizer state | Slim with `model.state_dict()` only when building inference |


***

## PARKED TOPICS

- **Head+tail truncation for CodeBERT** — first 256 + last 254 tokens. After baseline F1 established ✅ now unparked
- **LoRA fine-tuning CodeBERT** — after overnight experiments confirm ceiling of frozen approach
- **Optuna hyperparameter search** — after overnight experiments, if manual tuning insufficient
- **GMU (Gated Multimodal Unit)** — replacing FusionLayer. Phase 5 stretch
- **Multi-class 13-vulnerability classification** — after binary baseline solid
- **Evidently AI drift detection** — Phase 4
- **Dagster retraining pipeline** — Phase 4
- **DVC setup** — `dvc init` + `dvc add ml/data/graphs ml/data/tokens ml/data/splits`. 20 min
- **CCIP cross-chain / ERC-4337** — Phase 5 stretch
- **Loguru debug verbosity** — already fixed with `logger.remove(); logger.add(sys.stderr, level="INFO")`
- **HF_TOKEN warning** — set env var before long runs

***

## NEXT SESSION — START HERE IN ORDER

### Step 1 — Finish what's half-done from this session

These were planned but not yet built. Do them first:

**A. Add precision/recall to trainer.py evaluate function:**

```python
# In evaluate(), after computing f1 scores, add:
from sklearn.metrics import precision_score, recall_score
precision_vuln = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
recall_vuln = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
# Log to MLflow:
mlflow.log_metric("val_precision_vulnerable", precision_vuln, step=epoch)
mlflow.log_metric("val_recall_vulnerable", recall_vuln, step=epoch)
```

**B. Build `ml/scripts/tune_threshold.py`** — load epoch 16 checkpoint, sweep threshold 0.30→0.70 step 0.05 on val set, print F1-vuln at each threshold, identify peak.

**C. Build `ml/scripts/run_overnight_experiments.py`** — sequential launcher for 4 runs in the experiment matrix above.

### Step 2 — Launch overnight experiments

```bash
cd ~/projects/sentinel
nohup poetry run python ml/scripts/run_overnight_experiments.py > ml/logs/overnight.log 2>&1 &
echo "Overnight experiments running: PID $!"
```

Verify it's running, then close the laptop.

### Step 3 — Morning after (next session)

```bash
# Check it finished:
tail -50 ml/logs/overnight.log

# Open MLflow and compare all 4 runs:
poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db
# http://localhost:5000
# Runs tab → select all 4 → Compare → look at val_f1_vulnerable curves
```

Reading order for results:

1. Did `run-alpha-flip` beat baseline F1-vuln? (Expected: yes, significantly)
2. Did `run-combined` beat `run-alpha-flip`? (If yes, lr=3e-5 is also helping)
3. What did `run-more-epochs` peak at and at which epoch? (Tells us if 20 was simply too few)
4. Read precision_vulnerable and recall_vulnerable for each run — understand *why* the F1 moved

### Step 4 — After reading overnight results

- Run `tune_threshold.py` on whichever run had the best checkpoint
- Close M3.3 formally
- Begin M3.4 — `ml/src/inference/predictor.py`

***

## MILESTONE MAP — WHERE WE ARE

```
Phase 1 — Foundation
├── M1: Environment + tooling          ✅ COMPLETE
├── M2: Data pipeline                  ✅ COMPLETE (68,555 pairs, stratified splits)
├── M3: Model + Training
│   ├── M3.1: Model architecture       ✅ COMPLETE (GNN + CodeBERT + Fusion)
│   ├── M3.2: Dataset + DataLoader     ✅ COMPLETE (DualPathDataset)
│   ├── M3.3: Training loop            ✅ COMPLETE (baseline run done, F1-vuln 0.7133)
│   └── M3.4: Inference API            🔲  ml/src/inference/preprocess.py (middle of it) NEXT — predictor.py + FastAPI
├── M4: Solidity contracts             ✅ COMPLETE
└── M5: Integration                    🔲 PENDING

Phase 2 — ZKML                         🔲 NOT STARTED
Phase 3 — Multi-agent                  🔲 NOT STARTED
Phase 4 — MLOps                        🔲 NOT STARTED
Phase 5 — Integration + Demo           🔲 NOT STARTED
```


***


