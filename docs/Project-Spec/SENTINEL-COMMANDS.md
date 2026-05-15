
# SENTINEL — Commands Quick Reference

Load when you need a specific CLI command or startup sequence.

**Active checkpoint:** `ml/checkpoints/v5.2-jk-20260515c-r3_best.pt`
**Active architecture:** v5.2 three-eye GNN+CodeBERT fusion, JK attention aggregation
**Python venv:** `source ml/.venv/bin/activate`

---

## Environment Setup

Activate the venv (required before any ml/ command):

```bash
source ml/.venv/bin/activate
```

Set required env vars for offline mode (must be set at shell level):

```bash
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.
```

---

## ML Training & Evaluation

### Validate graph dataset (must pass before any retrain)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python -m ml.scripts.validate_graph_dataset
```

### Resume active v5.2 training run (r4 continues from r3 best checkpoint)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --resume ml/checkpoints/v5.2-jk-20260515c-r3_best.pt \
    --no-resume-model-only \
    --run-name v5.2-jk-20260515c-r4 \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4 \
    --early-stop-patience 20 \
    --eval-threshold 0.35
```

### Fresh full v5.2 training run (replace YYYYMMDD with today's date)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk-YYYYMMDD \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4 \
    --early-stop-patience 20 \
    --eval-threshold 0.35
```

### Resume from any checkpoint (model weights + optimizer state)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --resume ml/checkpoints/<checkpoint>_best.pt \
    --no-resume-model-only \
    --run-name <new-run-name> \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4 \
    --early-stop-patience 20 \
    --eval-threshold 0.35
```

Note: `--no-resume-model-only` restores optimizer + scheduler state in addition to model
weights, enabling true warm-restart. Omit this flag to load model weights only (cold restart
with new optimizer state — use when changing LR or optimizer config).

### Tune per-class thresholds (run after training completes)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/v5.2-jk-20260515c-r3_best.pt
```

Output: `ml/checkpoints/v5.2-jk-20260515c-r3_best_thresholds.json`

Note: tune_threshold.py uses `from torch.utils.data import DataLoader` (fixed in commit
35028f9 — do not revert to the old import path).

### Promote model to MLflow Registry

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/v5.2-jk-20260515c-r3_best.pt \
    --stage Staging --note "v5.2 three-eye JK attention, deduped 44K dataset"
```

---

## Dataset & Cache

### Build deduped RAM cache (uses multilabel_index_deduped.csv by default)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/create_cache.py
# Output: ml/data/cached_dataset_deduped.pkl
# Source CSV: ml/data/processed/multilabel_index_deduped.csv (44,420 rows)
```

Verify cache exists and check size:

```bash
ls -lh ml/data/cached_dataset_deduped.pkl
```

### Check graph and token file counts

```bash
# Graphs on disk (should be 44,420)
find ml/data/graphs/ -name "*.pt" | wc -l

# Token files on disk (~68K — only canonical MD5s loaded by dataset)
find ml/data/tokens/ -name "*.pt" | wc -l
```

---

## Training Monitoring

### Monitor active training run (live log tail)

```bash
tail -f ml/logs/v5.2-jk-20260515c-r3.log
```

### Grep for epoch progress, best F1, and patience counter

```bash
grep "Epoch\|Best\|F1\|patience" ml/logs/v5.2-jk-20260515c-r3.log | tail -20
```

### Check if training process is still running

```bash
ps aux | grep train.py | grep -v grep
# Active run PID: 43784 (v5.2-jk-20260515c-r3)
```

### Check MLflow experiment (v5.2 runs)

```bash
source ml/.venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
# Open http://localhost:5000 — filter by experiment "sentinel-v5.2"
```

Note: use `--backend-store-uri sqlite:///mlruns.db` — NOT `file:///mlruns`. Experiments 1, 2, 3 are corrupt.

---

## Behavioral Tests (primary evaluation — run after training completes)

Behavioral tests are the only honest judge of model quality. Macro val F1 during training
is a noisy proxy — run these before accepting any checkpoint.

### Run behavioral test suite

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/manual_test.py \
    --checkpoint ml/checkpoints/v5.2-jk-20260515c-r3_best.pt
# Test contracts: ml/scripts/test_contracts/
```

### Run all unit/integration tests (excluding API tests — API does not exist yet)

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python -m pytest ml/tests/ --ignore=ml/tests/test_api.py -q
```

---

## ML Inference API (v5.2)

### Start the API server

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=ml/checkpoints/v5.2-jk-20260515c-r3_best.pt \
SENTINEL_THRESHOLDS=ml/checkpoints/v5.2-jk-20260515c-r3_best_thresholds.json \
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001
```

### Quick test (single contract)

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"source_code": "contract C { function f() public {} }"}'
```

---

## Agents / MCP / LangGraph

### Start MCP servers (standalone)

```bash
cd agents
poetry run python -m src.mcp.servers.inference_server   # port 8010
poetry run python -m src.mcp.servers.rag_server         # port 8011
poetry run python -m src.mcp.servers.audit_server       # port 8012
```

### Run LangGraph smoke test

```bash
cd agents
poetry run python scripts/smoke_langgraph.py
```

---

## RAG Index

### Full rebuild

```bash
cd agents
poetry run python -m src.rag.build_index
```

---

## MLOps UIs

### MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

### Dagster

```bash
cd agents
poetry run dagster dev -f src/ingestion/scheduler_dagster.py
# UI on http://localhost:3000
# Schedule: daily_ingestion_schedule runs at 02:00
```

---

## Smart Contracts (Module 5 — never built/run)

```bash
cd contracts
forge install    # run once — contracts/lib/ is currently empty
forge build
forge test -vvv
```

---

## ZKML Pipeline (Module 2 — source complete, never executed)

Awaiting v5.2 checkpoint. Proxy MLP: Linear(128→64→32→10). EZKL scale=8192 little-endian.

```bash
# Step 1–4 (one-time setup)
python zkml/src/ezkl/setup_circuit.py gen-settings
python zkml/src/ezkl/setup_circuit.py calibrate
python zkml/src/ezkl/setup_circuit.py compile
python zkml/src/ezkl/setup_circuit.py setup

# Step 5 (per audit)
python zkml/src/ezkl/run_proof.py
```

---

## Checkpoint Reference

| Checkpoint | Architecture | Val F1 | Status |
|---|---|---|---|
| `v5.2-jk-20260515c-r3_best.pt` | v5.2 three-eye JK | 0.3130 (ep 21) | **ACTIVE — training in progress** |
| `v5-full-60ep_best.pt` | v5.0 three-eye | — | FAILED behavioral (15%/0%) |
| `multilabel-v4-finetune-lr1e4_best.pt` | v4 | F1=0.5422 | **Active fallback** |

v4 fallback per-class floors (v5.2 must exceed floor = v4 F1 − 0.05 to graduate):

```
CallToUnknown 0.397  DoS 0.384       ExternalBug 0.434  GasException 0.507
IntegerUO 0.776      MishandledException 0.459  Reentrancy 0.519  Timestamp 0.478
TOD 0.472            UnusedReturn 0.495
```
