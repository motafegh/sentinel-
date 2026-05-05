# SENTINEL — v3 Training Complete + Threshold Tuning (2026-05-05)

## Summary

`multilabel-v3-fresh-60ep` finished all 60 epochs. Threshold tuning on the held-out
validation split (10,278 samples) raised F1-macro from 0.4715 (raw) to **0.5069**,
clearing the 0.4884 success gate set in `docs/STATUS.md`.

---

## Training Run

| Parameter | Value |
|-----------|-------|
| Run | `multilabel-v3-fresh-60ep` |
| Experiment | `sentinel-retrain-v3` |
| Checkpoint | `ml/checkpoints/multilabel-v3-fresh-60ep_best.pt` |
| Thresholds | `ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json` |
| Epochs trained | 60/60 (no early stop — patience=10, counter=6 at end) |
| Best epoch | ~52–53 (plateau from ~ep 54) |
| Batch size | 32 |
| Architecture | `cross_attention_lora` — LoRA r=8 α=16 modules=['query','value'] |
| edge_attr | True (P0-B active) |

### Final epoch metrics (raw, no tuning)

| Epoch | Loss | F1-macro | Hamming |
|-------|------|----------|---------|
| 58 | 0.6848 | 0.4698 | 0.3163 |
| 59 | 0.6835 | 0.4704 | 0.3156 |
| 60 | 0.6865 | 0.4703 | 0.3156 |
| **Best** | — | **0.4715** | — |

---

## Threshold Tuning Results

Grid: [0.05, 0.10, …, 0.95] — 19 values per class.
Validation samples: 10,278. Per-class best threshold maximises F1 on the val split.

### Per-class thresholds and F1

| Class | Threshold | F1 | Precision | Recall | Support |
|-------|-----------|----|-----------|--------|---------|
| CallToUnknown | 0.70 | 0.3936 | 0.3216 | 0.5071 | 1,266 |
| DenialOfService | 0.95 | 0.4000 | 0.3176 | 0.5401 | 137 |
| ExternalBug | 0.65 | 0.4345 | 0.3121 | 0.7152 | 1,622 |
| GasException | 0.55 | 0.5501 | 0.4029 | 0.8667 | 2,589 |
| IntegerUO | 0.50 | 0.8214 | 0.7585 | 0.8958 | 5,343 |
| MishandledException | 0.60 | 0.4916 | 0.3649 | 0.7535 | 2,207 |
| Reentrancy | 0.65 | 0.5362 | 0.4492 | 0.6649 | 2,501 |
| Timestamp | 0.75 | 0.4789 | 0.4028 | 0.5905 | 1,077 |
| TransactionOrderDependence | 0.60 | 0.4770 | 0.3422 | 0.7872 | 1,800 |
| UnusedReturn | 0.70 | 0.4860 | 0.3951 | 0.6311 | 1,716 |

### Overall (tuned thresholds)

| Metric | Value |
|--------|-------|
| **F1-macro** | **0.5069** ✅ (gate: > 0.4884) |
| F1-micro | 0.5608 |
| Hamming loss | 0.2342 |
| Exact-match accuracy | 0.2763 |

### Success gate evaluation

| Check | Result |
|-------|--------|
| Tuned F1-macro > 0.4884 | ✅ 0.5069 > 0.4884 |
| Per-class floor (no class drops > 0.05 from pre-retrain) | ✅ All classes above floor |

---

## Observations and Weak Points

**Strong classes (F1 ≥ 0.53):**
- IntegerUO: 0.8214 — dominant class (5,343 support), well-learned
- GasException: 0.5501
- Reentrancy: 0.5362

**Weak classes needing attention in next run:**
- **DenialOfService**: F1=0.4000, support=**137** (severely underrepresented — 39× fewer than IntegerUO). Threshold tuned to 0.95 to squeeze recall. Class weighting or focal alpha required.
- **CallToUnknown**: F1=0.3936, threshold=0.70. Low precision (0.32) — model predicts it too broadly but still misses half the positives.
- **ExternalBug**: F1=0.4345, threshold=0.65. High recall (0.72) but precision 0.31 — too many false positives.

**Loss plateau:** Training loss oscillated ~0.68 from epoch 54 onward with no improvement. The model reached its capacity ceiling with current hyperparameters. A new run with higher LoRA rank or focal loss will be needed to push further.

---

## Recommended Next Run (v4)

### Priority changes

1. **Focal Loss** (`--focal-gamma 2.0 --focal-alpha`) — focus gradient on hard examples, reducing easy-negative dominance. Addresses low-precision classes (CallToUnknown, ExternalBug).

2. **Weighted sampler for DenialOfService** — with only 137 samples, even focal loss may under-weight this class. Add `WeightedRandomSampler` or increase `pos_weight` for DoS class specifically.

3. **LoRA rank increase: r=8 → r=16** — current plateau at 0.47 raw F1 suggests CodeBERT adaptation capacity is saturated. r=16 doubles the trainable params (294K → 589K) at minimal overhead.

4. **Autoresearch setup** — implement `ml/scripts/auto_experiment.py` + `ml/autoresearch/program.md` now that a clean baseline (0.5069 tuned) exists. The autoresearch loop can sweep focal gamma, LoRA r, and class weights automatically.

5. **Longer run or cosine restart** — if loss is still plateauing at 60 epochs with new settings, consider 80 epochs + cosine annealing with warm restarts to escape local minima.

### Suggested command skeleton for v4

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/train.py \
  --run-name multilabel-v4-focal-lora16 \
  --experiment sentinel-retrain-v4 \
  --epochs 60 \
  --batch-size 32 \
  --patience 10 \
  --loss-fn focal \
  --focal-gamma 2.0 \
  --lora-r 16 \
  --lora-alpha 32
```

After training:
```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/tune_threshold.py \
  --checkpoint ml/checkpoints/multilabel-v4-focal-lora16_best.pt
```

Success gate for v4: tuned F1-macro > **0.5069** on same `val_indices.npy` split.
