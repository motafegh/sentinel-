
# SENTINEL — Commands Quick Reference

Load when you need a specific CLI command or startup sequence.

---

## ML Training & Evaluation

### Validate graph dataset (must pass before any retrain)
```bash
cd ml
poetry run python -m ml.scripts.validate_graph_dataset
```

Train model (v4 example)

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/train.py \
  --run-name multilabel-v4-focal-lora16 \
  --experiment sentinel-retrain-v4 \
  --epochs 60 \
  --batch-size 32 \
  --patience 10 \
  --loss-fn focal --focal-gamma 2.0 \
  --lora-r 16 --lora-alpha 32
```

Tune per-class thresholds

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python -m ml.scripts.tune_threshold \
  --checkpoint ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
```

Promote model to MLflow Registry

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/promote_model.py \
  --checkpoint ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
  --stage Staging --note "Baseline for v4 comparison"
```

---

ML Inference API

Start the API server

```bash
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
SENTINEL_THRESHOLDS=ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json \
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001
```

Quick test

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"source_code": "contract C { function f() public {} }"}'
```

---

Agents / MCP / LangGraph

Start MCP servers (standalone)

```bash
cd agents
poetry run python -m src.mcp.servers.inference_server   # port 8010
poetry run python -m src.mcp.servers.rag_server         # port 8011
poetry run python -m src.mcp.servers.audit_server       # port 8012
```

Run LangGraph smoke test

```bash
cd agents
poetry run python scripts/smoke_langgraph.py
```

---

RAG Index

Full rebuild

```bash
cd agents
poetry run python -m src.rag.build_index
```

---

MLOps UIs

MLflow

```bash
mlflow ui --port 5000
```

Dagster

```bash
cd agents
poetry run dagster dev -f src/ingestion/scheduler_dagster.py
# UI on http://localhost:3000
```

---

Smart Contracts (Module 5)

```bash
cd contracts
forge install    # run once
forge build
forge test -vvv
```

---

ZKML Pipeline (when environment ready)

```bash
# Step 1–4 (one-time setup)
python zkml/src/ezkl/setup_circuit.py gen-settings
python zkml/src/ezkl/setup_circuit.py calibrate
python zkml/src/ezkl/setup_circuit.py compile
python zkml/src/ezkl/setup_circuit.py setup

# Step 5 (per audit)
python zkml/src/ezkl/run_proof.py
```


