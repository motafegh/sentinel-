# 2026-05-03 — Training Pipeline Fix & Full Dataset Regeneration

## Problem discovered

After graph re-extraction (see `2026-05-03-graph-reextraction.md`), running
`create_label_index.py` produced:

```
Safe (0): 68523 (100.0%)
Vulnerable (1): 0 (0.0%)
```

Root cause: `ast_extractor.py` hardcodes `label=0` for every contract in its
worker function. The `graph.y` field in every `.pt` file is 0 regardless of
actual vulnerability class. This is correct behaviour for multi-label training
(labels come from `multilabel_index.csv`, not `graph.y`) but makes
`label_index.csv` permanently unusable for stratification.

Consequence: `create_splits.py` was stratifying on all-zero labels, which
produces splits with 0% vulnerable — meaningless stratification.

---

## Fix: `create_splits.py` — derive binary labels from `multilabel_index.csv`

**File:** `ml/scripts/create_splits.py`

Changed STEP 1 from reading `label_index.csv` to reading `multilabel_index.csv`
and deriving a binary label as:

```python
all_labels = (df[class_cols].sum(axis=1) > 0).astype(int).values
```

Any contract with at least one positive vulnerability class is labelled
vulnerable (1); all-zero contracts are safe (0). This correctly reflects the
BCCC dataset ground truth.

`label_index_path` parameter kept for API compatibility but is now unused.

---

## `label_index.csv` / `create_label_index.py` — now obsolete

Neither file is used anywhere in the current training pipeline:

| Consumer | Actual source |
|---|---|
| `trainer.py` | `multilabel_index.csv` |
| `train.py` | `multilabel_index.csv` |
| `tune_threshold.py` | `multilabel_index.csv` |
| `analyse_truncation.py` | `multilabel_index.csv` |
| `create_splits.py` | `multilabel_index.csv` (after this fix) |
| `DualPathDataset` binary mode | `graph.y` directly |

`create_label_index.py` can be kept for diagnostics but must not be used as
a stratification source. `label_index.csv` may be deleted safely.

---

## Full pipeline regeneration (2026-05-03)

All training inputs rebuilt fresh after graph re-extraction:

| Step | Command | Output | Result |
|---|---|---|---|
| 1. Multilabel index | `build_multilabel_index.py` | `multilabel_index.csv` | 68,523 rows, 10 classes |
| 2. Splits | `create_splits.py` | `ml/data/splits/*.npy` | 47,966 / 10,278 / 10,279; **64.3% vulnerable** (stratified) |
| 3. Tokens | `tokenizer.py` | `ml/data/tokens/*.pt` | 68,568 processed, 96.6% truncated at 512 tokens |

Previous splits (based on all-zero labels) have been overwritten. New splits
have correct stratification — 64.3% vulnerable in each partition matches the
BCCC dataset composition.

---

## Retrain launched

```bash
TRANSFORMERS_OFFLINE=1 \
  ml/.venv/bin/python ml/scripts/train.py \
    --run-name multilabel-v2-edge-attr \
    --experiment-name sentinel-retrain-v2 \
    --epochs 40 \
    --batch-size 16 \
    --graphs-dir ml/data/graphs \
    --tokens-dir ml/data/tokens \
    --splits-dir ml/data/splits \
    --checkpoint-dir ml/checkpoints \
    --checkpoint-name multilabel_crossattn_v2_best.pt
```

Key training configuration (from startup log):

| Setting | Value |
|---|---|
| Device | CUDA |
| Classes | 10 |
| Loss | BCEWithLogitsLoss + class-balanced pos_weight |
| LoRA | r=8, alpha=16, modules=query+value; 294,912 trainable / 124M frozen |
| edge_attr | True (P0-B active — edge embeddings via `nn.Embedding(5, 16)`) |
| Fusion | CrossAttentionFusion: node_dim=64, token_dim=768, attn_dim=256, heads=8 |
| MLflow experiment | `sentinel-retrain-v2` |
| Checkpoint | `ml/checkpoints/multilabel_crossattn_v2_best.pt` |

---

## Post-training steps

Once all 40 epochs complete (or early stopping triggers):

1. **Check MLflow** — compare val F1-macro vs baseline 0.4679
   ```bash
   ml/.venv/bin/mlflow ui --port 5000
   ```

2. **Tune thresholds** — find optimal per-class decision thresholds on val split
   ```bash
   TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/tune_threshold.py \
     --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt \
     --label-csv ml/data/processed/multilabel_index.csv \
     --splits-dir ml/data/splits \
     --graphs-dir ml/data/graphs \
     --tokens-dir ml/data/tokens
   ```

3. **Promote checkpoint** if val F1-macro > 0.4679
   ```bash
   ml/.venv/bin/python ml/scripts/promote_model.py \
     --run-id <mlflow-run-id> \
     --stage Staging
   ```

4. **Rollback** if val F1-macro ≤ 0.4679 after 40 epochs — keep
   `multilabel_crossattn_best.pt`; investigate `edge_emb_dim` (try 8 instead
   of 16) before re-running.
