# SENTINEL ML Scripts — Technical Reference

## Overview

All scripts live in `ml/scripts/`. They are thin entry points that wire CLI arguments to the training and inference modules. All commands must be run from the project root (`~/projects/sentinel`), not from `ml/`.

| Script | Purpose |
|---|---|
| `train.py` | CLI entry point for training runs (new + resume) |
| `tune_threshold.py` | Post-training decision threshold optimiser |
| `run_overnight_experiments.py` | Multi-experiment sequential launcher |

---

## `train.py`

### Purpose

Exposes all `TrainConfig` fields as CLI arguments. Maps parsed args to a `TrainConfig` instance and calls `train(config)`.

### Usage

```bash
# Start a new run with defaults (20 epochs, lr=1e-4, focal_alpha=0.25)
poetry run python ml/scripts/train.py

# Override common hyperparameters
poetry run python ml/scripts/train.py \
    --run-name run-alpha-tune \
    --epochs 30 \
    --lr 3e-4 \
    --batch-size 32 \
    --focal-alpha 0.25 \
    --focal-gamma 2.0

# Resume from a previous checkpoint
poetry run python ml/scripts/train.py \
    --resume ml/checkpoints/run-alpha-tune_best.pt \
    --run-name run-alpha-tune-resumed \
    --epochs 40

# Save checkpoint with a custom name
poetry run python ml/scripts/train.py \
    --run-name run-lr-lower \
    --checkpoint-name run-lr-lower_best.pt
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--run-name` | `baseline` | MLflow run name. Checkpoint saved as `<run-name>_best.pt` unless overridden. |
| `--experiment-name` | `sentinel-training` | MLflow experiment name. |
| `--epochs` | 20 | Number of training epochs. |
| `--batch-size` | 32 | Batch size for train + val loaders. |
| `--lr` | 1e-4 | AdamW learning rate. |
| `--weight-decay` | 1e-2 | AdamW weight decay (L2 regularisation). |
| `--focal-gamma` | 2.0 | FocalLoss gamma. Higher = more focus on hard examples. |
| `--focal-alpha` | 0.25 | FocalLoss alpha (weight for vulnerable/label=1). **Do not change without justification.** |
| `--graphs-dir` | `ml/data/graphs` | Graph `.pt` files directory. |
| `--tokens-dir` | `ml/data/tokens` | Token `.pt` files directory. |
| `--splits-dir` | `ml/data/splits` | Split index `.npy` files directory. |
| `--checkpoint-dir` | `ml/checkpoints` | Directory to save checkpoints. |
| `--checkpoint-name` | `<run-name>_best.pt` | Checkpoint filename override. |
| `--resume` | `None` | Path to a resumable checkpoint (new format only — dict with `"model"` key). |

### Resume limitations

Only checkpoints saved with the new format (April 2026+) can be resumed. New-format checkpoints are dicts with keys `"model"`, `"optimizer"`, `"epoch"`, `"best_f1"`, `"config"`. Old-format checkpoints (plain `state_dict()`) only contain model weights — optimizer state and epoch counter are not available, so training cannot continue from where it left off.

If you attempt to resume from an old checkpoint:
```
ValueError: Cannot resume from ...: this is an old-format checkpoint (plain state dict).
```

---

## `tune_threshold.py`

### Purpose

The training loop uses a fixed 0.5 threshold for checkpoint selection. That threshold is a training tool, not necessarily the optimal inference threshold. This script sweeps thresholds 0.30 → 0.70 on the val set to find the cutoff that maximises F1-macro.

**The model is only run once.** Probabilities are collected in a single forward pass and cached as numpy arrays. The sweep applies different thresholds to the same cached arrays — orders of magnitude faster than re-running the model per threshold.

### Usage

```bash
# Sweep the production checkpoint (default)
poetry run python ml/scripts/tune_threshold.py

# Sweep a different checkpoint
poetry run python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/run-more-epochs_best.pt

# Use a larger batch for faster GPU sweep
poetry run python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/run-alpha-tune_best.pt \
    --batch-size 64
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `ml/checkpoints/run-alpha-tune_best.pt` | Path to checkpoint to sweep. |
| `--batch-size` | `TrainConfig.batch_size` (32) | Val loader batch size. Increase for faster sweeps on GPU. |

### Output

```
Threshold |  F1-vuln | Precision |   Recall | F1-macro
---------------------------------------------------------
     0.30 |   0.8003 |    0.6672 |   0.9997 |   0.4917
     0.35 |   0.8011 |    0.6687 |   0.9991 |   0.4972
     0.40 |   0.8013 |    0.6701 |   0.9962 |   0.5036
     0.45 |   0.7936 |    0.7160 |   0.8901 |   0.6294
     0.50 |   0.7458 |    0.7797 |   0.7147 |   0.6686  ← best
     0.55 |   0.6446 |    0.8543 |   0.5176 |   0.6325
     0.60 |   0.4903 |    0.8888 |   0.3385 |   0.5415
     0.65 |   0.3135 |    0.9340 |   0.1884 |   0.4405
     0.70 |   0.0742 |    0.9922 |   0.0385 |   0.3048

──────────────────────────────────────────────────────────
  Best threshold : 0.50
  F1-vuln        : 0.7458
  Precision-vuln : 0.7797   (of all flagged contracts, this fraction are actually vulnerable)
  Recall-vuln    : 0.7147   (of all truly vulnerable contracts, this fraction were caught)
  F1-macro       : 0.6686
──────────────────────────────────────────────────────────
  → Set INFERENCE_THRESHOLD = 0.50 in predictor.py
```

### Selection criterion: F1-macro, not F1-vuln

F1-vuln is gameable. At threshold 0.30, recall=1.0 and F1-vuln=0.80 look great — but the model is flagging almost everything as vulnerable. Precision collapses to 0.667, which means 1 in 3 flagged contracts are false alarms. F1-macro for this threshold is only 0.49 because F1-safe is near zero.

F1-macro averages F1 over both classes. A model that mass-flags everything gets F1-safe≈0, which pulls F1-macro down even if F1-vuln is high. This prevents selecting a degenerate low-threshold.

### Two-pass algorithm

**Problem:** In a single pass, "← best" would mark every row that beats all previous rows — making it look like multiple thresholds tied.

**Solution:** Pass 1 computes all metrics and finds the true best. Pass 2 prints the table with exactly one marker. The table is always correct and unambiguous.

### Val set independence

The threshold is tuned on the val set — the same split used for checkpoint selection during training. This introduces a small but acceptable selection bias. The test set (10,284 samples) remains completely untouched for the final holdout evaluation.

---

## `run_overnight_experiments.py`

### Purpose

Runs 4 hyperparameter experiments sequentially without manual intervention. Designed to be launched before sleeping and read in the morning.

**Sequential, not parallel** — the GPU can only train one model at a time.

### Usage

```bash
# Launch in background, redirect all output to log file
nohup poetry run python ml/scripts/run_overnight_experiments.py \
    > ml/logs/overnight.log 2>&1 &
echo "Launched with PID $!"

# Resume from experiment 3 (if 1 and 2 already completed)
nohup poetry run python ml/scripts/run_overnight_experiments.py \
    --start-from 3 > ml/logs/overnight_resume.log 2>&1 &

# Check live progress
tail -50 ml/logs/overnight.log
```

### The 4 experiments

Each `TrainConfig` overrides only what differs from the defaults (epochs=20, lr=1e-4, focal_alpha=0.25):

| # | `run_name` | Change | Hypothesis |
|---|---|---|---|
| 1 | `run-alpha-tune` | `focal_alpha=0.35` | Softens class weight ratio from 3× to 1.86× (closer to actual 1.8× imbalance). Might improve F1-safe without hurting F1-vuln. |
| 2 | `run-more-epochs` | `epochs=40` | Baseline peaked at epoch 16 but was still improving. 40 epochs reveals the true plateau. |
| 3 | `run-lr-lower` | `lr=3e-5, epochs=30` | Baseline showed ~0.15 F1 oscillation — classic sign of LR too high. 3× smaller LR should produce a smoother, higher peak. |
| 4 | `run-combined` | `focal_alpha=0.35, lr=3e-5, epochs=30` | If 1 and 3 individually improve, combined should be best. |

### Error isolation

Each experiment is wrapped in `try/except`. A crash in experiment 2 does not abort 3 and 4:

```python
for i, config in experiments_to_run:
    try:
        train(config)
        completed.append(config.run_name)
    except Exception:
        logger.exception(f"Run {i}/{total} FAILED — continuing")
        failed.append(config.run_name)
```

This means waking up with 3 of 4 results is common and expected.

### Final summary (printed after all runs)

```
=======================================================
  Overnight experiments complete — 5.23 hr total

  Completed 2/4: run-alpha-tune, run-more-epochs
  Failed 2/4: run-lr-lower, run-combined
  Check the exception tracebacks above for each failed run.
  To resume: python run_overnight_experiments.py --start-from 3
=======================================================
```

### `--start-from` flag

```bash
--start-from 3
```

Skips experiments 1 and 2, starts from experiment 3 (1-indexed). Used to resume after a crash without re-running completed experiments. The `total` counter stays at 4 so "Run 3/4" and "Run 4/4" logging is consistent whether resuming or not.

### Reading order for morning MLflow review

1. `run-lr-lower` — Did oscillation reduce? Look for a smoother `val_f1_macro` curve.
2. `run-more-epochs` — What epoch did F1-macro peak? Did it plateau before 40?
3. `run-alpha-tune` — Did F1-safe improve vs baseline (0.5856)? Did F1-vuln hold?
4. `run-combined` — Did combining alpha + LR changes beat all singles?
5. For each run: `val_recall_vulnerable` — the real security signal. A model that misses fewer vulnerabilities is strictly better, even if F1-macro is slightly lower.

### Known run results (as of April 2026)

| Run | Checkpoint | Val F1-macro | Status |
|---|---|---|---|
| `baseline` | `sentinel_best.pt` (ep 16) | 0.6515 | complete |
| `run-alpha-tune` | `run-alpha-tune_best.pt` (ep ~26) | **0.6686** ← production | complete |
| `run-more-epochs` | `run-more-epochs_best.pt` (ep 22) | 0.6584 | killed at ep 25/40 |
| `run-lr-lower` | — | — | never ran |
| `run-combined` | — | — | never ran |

`run-alpha-tune` is the production checkpoint. Threshold swept to 0.50 by `tune_threshold.py`.

---

## Common patterns and gotchas

### `sys.path.insert` requirement

All scripts add the project root to `sys.path` before importing `ml.*`:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

`parents[2]` from `ml/scripts/` resolves to the project root (`~/projects/sentinel`). Without this, `from ml.src.training.trainer import ...` fails with `ModuleNotFoundError`.

### Loguru configuration

All scripts configure loguru before any `ml.src.*` imports:

```python
logger.remove()
logger.add(sys.stderr, level="INFO")
```

`logger.remove()` clears loguru's default handler. Without this, HuggingFace's model loading and PyG's internals flood the terminal with DEBUG messages. The `logger.remove()/add()` must come **before** the first `ml.src.*` import — those imports may initialise their own loguru handlers at import time.

### Running from the project root

All path defaults in `TrainConfig` and scripts are relative to the project root:
- `ml/data/graphs` ← relative to `~/projects/sentinel`
- `ml/checkpoints` ← relative to `~/projects/sentinel`

If you run from `ml/`, these paths resolve to `ml/ml/data/graphs` — wrong. Always run from `~/projects/sentinel`.

---

## Data validation scripts

### `ml/scripts/comprehensive_data_validation.py`

Run **before training** to verify the full dataset is intact and correctly formed. Validates:

- Graph file count and shape (`graph.x` must be `[N, 8]`, `graph.y` must exist)
- Token file count and shape (`input_ids [512]`, `attention_mask [512]`)
- Hash pairing between graphs and tokens
- Label distribution check (expected ~64% vulnerable)
- Random sample of 1000 files loaded end-to-end
- Results written to `validation_results.json`

```bash
poetry run python ml/scripts/comprehensive_data_validation.py
```

### `ml/scripts/analyze_token_stats.py`

Reports truncation rate and token length statistics across all 68,570 token files:

```
Total files scanned:   68,570
Truncated (len==512):  14,230  (20.8%)
Not truncated:         54,340  (79.2%)

Of non-truncated contracts:
  Median length: 187 tokens
  Mean length:   201 tokens
  Min length:    12 tokens
```

A contract is considered truncated if all 512 attention mask positions are 1 (no padding). Useful for understanding what fraction of the dataset lost tail code.

```bash
poetry run python ml/scripts/analyze_token_stats.py
```

### `ml/analysis/data_quality_validation.py`

`DataQualityValidator` class — runs structural, statistical, and semantic validation over graph-token pairs. Takes `metadata_path` (parquet) for cross-referencing. More thorough than `comprehensive_data_validation.py`, slower. Used during initial dataset construction to catch extraction bugs before committing to full training runs.

---

## Development test scripts (`ml/scripts/test_*.py`)

These scripts are manual smoke tests — not pytest, no assertions framework. Run them by hand after changes to the corresponding module to verify the pipeline is working.

| Script | Tests | When to run |
|---|---|---|
| `test_gnn_encoder.py` | GNNEncoder forward pass, output shape `[B,64]` | After any GNNEncoder change |
| `test_fusion_layer.py` | FusionLayer forward pass, output shape `[B,64]` | After any FusionLayer change |
| `test_dataset.py` | DualPathDataset `__getitem__` and split loading | After dataset changes |
| `test_dataloader.py` | Full DataLoader batching with `dual_path_collate_fn` | After collate_fn changes |
| `test_sentinel_model.py` | End-to-end batch: loads real data → forward pass through full model | After any model change |

```bash
# End-to-end model test — the most comprehensive smoke test
poetry run python ml/scripts/test_sentinel_model.py
```

`test_sentinel_model.py` verifies:
- No shape errors through the full pipeline
- Output shape `[B]` — one score per contract
- Scores in `[0, 1]` — sigmoid is applied
- Labels shape matches scores — ready for loss computation

---

## Pytest test suite (`ml/tests/`)

### `ml/tests/conftest.py`

Provides the `client` fixture — a FastAPI `TestClient` with `scope="session"`:

```python
@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c
```

`scope="session"` means the model loads **once** for the entire test run (not once per test). The SENTINEL model is ~500 MB and takes ~10s to load. Without session scope, 4 tests would require 40s of loading; with it, the overhead is ~10s total.

`TRANSFORMERS_OFFLINE=1` is set before the app import to prevent HuggingFace from making network calls on startup (fails silently in WSL2 and pollutes output).

### `ml/tests/test_api.py`

Integration tests for the inference API. Uses `TestClient` — no real server, no network calls.

| Test | What it checks |
|---|---|
| `test_health_returns_ok` | `/health` returns 200 with `predictor_loaded=True` |
| `test_predict_valid_contract` | Valid Solidity returns 200, correct response shape (`label`, `confidence`, `threshold`, `truncated`, `num_nodes`, `num_edges`) |
| `test_predict_error_cases` | Non-Solidity input returns 422; empty string returns 422 |
| `test_predict_consistent_on_same_input` | Same contract always returns same score (model is in `eval()` mode, deterministic) |

Tests check **shape and types**, not specific label values. A label assertion like `label == "vulnerable"` would break every time the model is retrained. Shape never changes.

```bash
# Run the full test suite
poetry run pytest ml/tests/ -v
```
