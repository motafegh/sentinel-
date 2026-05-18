# SENTINEL ML Scripts

All commands must be run from the **project root** (`~/projects/sentinel`).
Activate the venv first: `source ml/.venv/bin/activate`

---

## Active Pipeline (v7)

### Training

- **train.py** — main training entry point (v7, AsymmetricLoss, LoRA, torch.compile)
- **monitor.sh** — live training dashboard; run in a second terminal while training

```bash
# Full train command (v7.0)
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v7.0 --experiment-name sentinel-v7 \
    --epochs 100 --gradient-accumulation-steps 8 --compile --num-workers 4
```

---

### Data Pipeline (run in order for re-extraction)

1. **reextract_graphs.py** — re-run Slither extraction → `ml/data/graphs/` (v7 schema, 11-dim)
2. **retokenize_windowed.py** — windowed CodeBERT tokenization → `ml/data/tokens_windowed/` (shape [4,512])
3. **build_multilabel_index.py** — scan graphs/tokens → `ml/data/processed/multilabel_index.csv`
4. **dedup_multilabel_index.py** — content-hash deduplication + Timestamp relabeling
5. **inject_augmented.py** — inject DoS augmented pairs into the CSV
6. **label_cleaner.py** — remove noisy labels; outputs `multilabel_index_cleaned.csv`
7. **create_cache.py** — build paired dataset cache → `ml/data/cached_dataset_deduped.pkl`
8. **create_splits.py** — generate stratified train/val/test splits (only if splits need regeneration)

---

### Post-Training

- **tune_threshold.py** — per-class threshold sweep on val set; prints optimal cutoff per class
- **manual_test.py** — behavioral gate (target: ≥80% detection, ≥80% specificity on test_contracts/)
- **promote_model.py** — move best checkpoint to production path

---

### Utilities

- **analyse_truncation.py** — measure 512-token truncation blind spot across token files
- **validate_graph_dataset.py** — full dataset integrity check (graph shape, edge types, label coverage)
- **patch_graph_features.py** — in-place graph feature patching (useful for future schema fixes without full re-extraction)
- **compute_drift_baseline.py** — compute prediction distribution baseline for drift monitoring
- **run_augmentation.sh** — DoS augmentation wrapper (calls generate_dos_pairs + inject_augmented)

---

## Archive

- **archive/** — completed audit scripts, superseded utilities, external reviewer scripts
  - `archive/audit/` — all 26 audit task scripts + reports (findings applied to codebase)
  - `archive/external-reviewer-scripts/` — external reviewer one-off scripts
  - One-off utilities: verify_splits, compute_locked_hashes, create_label_index, extract_augmented, etc.

All findings from archived scripts have been applied to the codebase (see docs/ACTIVE_BUGS.md).
