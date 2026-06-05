# SENTINEL ML Scripts

All commands must be run from the **project root** (`~/projects/sentinel`).
Activate the venv first: `source ml/.venv/bin/activate`

---

## Active Pipeline (v10)

### Training

- **train.py** — main training entry point (v8.1, 8-layer GNN, four-eye classifier, GraphCodeBERT+LoRA, Flash Attention 2, GNN prefix injection, AsymmetricLoss, torch.compile)

```bash
# v10 training run (K=48 prefix, warmup=15 epochs)
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. nohup \
    python ml/scripts/train.py \
    --run-name v10-$(date +%Y%m%d) \
    --experiment-name sentinel-v10 \
    --epochs 100 \
    --gradient-accumulation-steps 8 \
    --gnn-prefix-k 48 \
    --gnn-prefix-warmup-epochs 15 \
    --gnn-prefix-proj-lr-mult 5.0 \
    --phase2-edge-types 6 8 9 10 \
    --weighted-sampler positive \
    --cache-path ml/data/cached_dataset_v10.pkl \
    > ml/logs/v10-$(date +%Y%m%d).log 2>&1 &
```

**Key train.py flags:**

| Flag | Default | Notes |
|------|---------|-------|
| `--gnn-prefix-k` | `0` | Set to 48 for GNN prefix injection |
| `--gnn-prefix-warmup-epochs` | `15` | Prefix suppressed until this epoch |
| `--gnn-prefix-proj-lr-mult` | `5.0` | LR multiplier for gnn_to_bert_proj |
| `--phase2-edge-types` | `6` | Space-separated edge type ints for Phase 2 |
| `--weighted-sampler` | `""` | `"positive"` = 3× weight for any-vuln rows |
| `--cache-path` | `ml/data/cached_dataset_v10.pkl` | Path to paired cache |
| `--early-stop-patience` | `30` | Epochs without val improvement before stop |

---

## Data Pipeline (run in order for full re-extraction)

1. **reextract_graphs.py** — re-run Slither extraction → `ml/data/graphs/` (v8 schema, 11-dim, 11 edge types)
2. **retokenize_windowed.py** — windowed GraphCodeBERT tokenization → `ml/data/tokens_windowed/` (shape [4,512], stride=256)
3. **build_multilabel_index.py** — scan graphs/tokens → `ml/data/processed/multilabel_index.csv`
4. **create_cache.py** — build paired dataset cache → `ml/data/cached_dataset_v10.pkl`
5. **create_splits.py** — generate stratified train/val/test splits (only if splits need regeneration; current splits at `ml/data/splits/v10_deduped/` are valid)

**Note on retokenization:** stride=256 with K=48 (code_budget=464) gives 208-token overlap. Retokenization is only needed if K > 256, which would create gaps between windows.

---

## Diagnostics and Audits

- **audit_prefix_node_counts.py** — analyse declaration-node count distribution across the dataset; confirms K=48 covers 95.5% of contracts at P95 → `ml/logs/prefix_node_count_audit.json`
- **validate_graph_dataset.py** — full dataset integrity check (graph shape, edge types, schema version, label coverage)
- **analyse_truncation.py** — measure token truncation blind spot across token files

---

## Post-Training

- **tune_threshold.py** — per-class threshold sweep on val set; prints optimal cutoff per class
  ```bash
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/tune_threshold.py \
      --checkpoint ml/checkpoints/v8-<date>_best.pt
  ```
- **manual_test.py** — behavioral gate (≥80% detection, ≥80% specificity on 20 test contracts in `test_contracts/`)
  ```bash
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/manual_test.py \
      --checkpoint ml/checkpoints/v8-<date>_best.pt
  ```
- **promote_model.py** — move best checkpoint to production path after both gates pass

---

## Utilities

- **compute_drift_baseline.py** — compute prediction distribution baseline for drift monitoring
- **run_augmentation.sh** — DoS augmentation wrapper (calls generate_dos_pairs + inject_augmented)

---

## Archive

- **archive/** — completed audit scripts, superseded utilities, external reviewer scripts
  - `archive/audit/` — all audit task scripts + reports (findings applied to codebase)
  - One-off utilities: verify_splits, compute_locked_hashes, create_label_index, etc.

All findings from archived scripts have been applied to the codebase (see docs/CHANGELOG.md).
