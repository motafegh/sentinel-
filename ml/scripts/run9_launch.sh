#!/usr/bin/env bash
# Run 9 launch — v9 schema, drop-complexity + APPNP + EXTERNAL_CALL + ARITH + in_unchecked
# Pre-conditions verified by this script:
#  - v9 graphs present at ml/data/graphs/ (re-extracted)
#  - v9 cache at ml/data/cached_dataset_v9.pkl
#  - deduped multilabel CSV at ml/data/processed/multilabel_index_deduped.csv
#  - deduped splits at ml/data/splits/deduped/
#
# Output:
#  - ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt
#  - /tmp/run9_v11.log
#  - MLflow run in sentinel-multilabel experiment

cd "$(dirname "$0")/../.." || exit 1
set -e

# Pre-flight
[[ -f ml/data/cached_dataset_v9.pkl ]] || { echo "ERROR: ml/data/cached_dataset_v9.pkl missing"; exit 1; }
[[ -f ml/data/processed/multilabel_index_deduped.csv ]] || { echo "ERROR: deduped CSV missing"; exit 1; }
[[ -d ml/data/splits/deduped ]] || { echo "ERROR: deduped splits dir missing"; exit 1; }

# Verify schema version is v9
SCHEMA=$(PYTHONPATH=. /home/motafeq/projects/sentinel/ml/.venv/bin/python3 -c \
    "from ml.src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION; print(FEATURE_SCHEMA_VERSION)")
[[ "$SCHEMA" == "v9" ]] || { echo "ERROR: schema is $SCHEMA, expected v9"; exit 1; }

# Start watcher BEFORE training so it picks up first epoch
bash ml/scripts/run9_watcher.sh stop 2>/dev/null || true
bash ml/scripts/run9_watcher.sh start

# Run 9 — fresh cold start (cannot resume Run 8 because schema dim changed)
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --run-name    GCB-P1-Run9-v11-20260606 \
  --experiment-name sentinel-multilabel \
  --splits-dir  ml/data/splits/deduped \
  --label-csv   ml/data/processed/multilabel_index_deduped.csv \
  --cache-path  ml/data/cached_dataset_v9.pkl \
  --epochs      100 \
  --batch-size  8 \
  --gradient-accumulation-steps 8 \
  --lr          1e-4 \
  --gnn-lr-multiplier     2.5 \
  --fusion-lr-multiplier  0.3 \
  --gnn-prefix-k         48 \
  --gnn-prefix-warmup-epochs 5 \
  --jk-entropy-reg-lambda  0.0075 \
  --aux-loss-weight       0.30 \
  --aux-phase2-loss-weight 0.20 \
  --threshold-tune-interval 10 \
  --early-stop-patience 30 \
  --drop-complexity-feature \
  --appnp-alpha 0.2 \
  2>&1 | tee /tmp/run9_v11.log
