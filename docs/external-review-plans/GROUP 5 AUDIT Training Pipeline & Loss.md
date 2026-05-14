

---

# GROUP 5 AUDIT: Training Pipeline & Loss Functions
**Files:** `trainer.py` + `focalloss.py` + `train.py`

---

## 5.1 [CRITICAL] `train.py --aux-loss-weight` default=0.3 but `TrainConfig.aux_loss_weight` default=0.1 — CLI silently overruns dataclass

**Location:** `train.py:193` vs `trainer.py:288`

**Bug:** 
```python
# train.py line 193
p.add_argument("--aux-loss-weight", type=float, default=0.3, ...)

# trainer.py line 288
aux_loss_weight: float = 0.1   # λ in: main + λ*(aux_gnn + aux_tf + aux_fused)
```

When a user runs `python ml/scripts/train.py` without specifying `--aux-loss-weight`, argparse provides `0.3` as the default. This is then passed to `TrainConfig(aux_loss_weight=args.aux_loss_weight)`, overriding the dataclass default of `0.1`. So **CLI runs get 0.3, programmatic TrainConfig() usage gets 0.1**.

This is a three-way contradiction:
1. `train.py` default = 0.3 (the Phase 0 fix value)
2. `TrainConfig` default = 0.1 (the pre-fix value)
3. `SentinelModel` docstring says "λ=0.1" (the pre-fix value)

If anyone constructs `TrainConfig()` programmatically (e.g. Jupyter notebooks, sweep scripts, the predictor), they get `0.1` — the gradient collapse value. The Phase 0 fix (0.3) is only applied when going through the CLI.

**Impact:** The GNN gradient collapse fix is silently absent for any non-CLI usage. This directly contributed to v5.0's 6.7% GNN eye gradient share by epoch 43.

**Fix:** Update `TrainConfig.aux_loss_weight` default to `0.3`. Update the `SentinelModel` docstring to match. Make `train.py`'s default explicitly reference the dataclass default, not hardcode its own.

---

## 5.2 [CRITICAL] `TrainConfig.batch_size` default=16 contradicts `train.py --batch-size` default=16, but Fix #17 says both should be 32

**Location:** `trainer.py:283` vs `train.py:123`

**Bug:**
```python
# trainer.py line 283
batch_size: int = 16

# train.py line 123
p.add_argument("--batch-size", type=int, default=16, ...)
```

Fix #17 (line 114) says: "TrainConfig.batch_size default was 16 while train.py --batch-size defaulted to 32. Both now default to 32." But the actual code shows both at 16, not 32. The fix was either reverted or never applied. This means Fix #17's documentation is lying — the discrepancy was supposedly fixed but the code still has the old values.

**Impact:** The MLflow `batch_size` param logged from `TrainConfig` says 16, but if someone references the Fix #17 comment, they'd expect 32. More importantly, the batch size directly affects gradient noise scale, OneCycleLR step count, and pos_weight calibration — all of which were tuned for a specific batch size.

**Fix:** Decide on a single batch size (16 or 32), set both defaults to it, and update Fix #17's comment to match reality.

---

## 5.3 [HIGH] `FocalLoss.alpha=0.25` is wrong for multi-label — up-weights negative class for rare vulnerability types

**Location:** `focalloss.py:35, 69`

**Bug:**
```python
self.alpha = 0.25  # scalar, same for all classes
alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
```

For `alpha=0.25`:
- When `target=1` (vulnerable): weight = 0.25
- When `target=0` (safe): weight = 0.75

This is correct for binary classification where "vulnerable" is the 64% majority class. But in multi-label (Track 3), this same `alpha=0.25` is applied to **every class independently**. For rare classes like DenialOfService (137/68K = 0.2% positive rate), `target=1` gets weight 0.25 and `target=0` gets weight 0.75 — the rare positive examples are **down-weighted by 3x** relative to the abundant negatives. This is the opposite of what's needed for rare-class detection.

The README (line 80-86) even documents this: "Alpha 0.25 down-weights the vulnerable majority and up-weights the safe minority. This is correct — vulnerable is the majority class (64.33%). Do not change it." — but that's only true for binary mode. In multi-label mode, each class has its own positive rate, and a single scalar alpha cannot be correct for all of them.

The `MultiLabelFocalLoss` class (line 77-131) exists precisely to solve this with per-class alphas, but **it is never used**. The training loop uses `FocalLoss` with scalar alpha for all 10 classes.

**Impact:** For rare classes (DoS, Timestamp, TOD), the focal loss actively suppresses the few positive examples that matter most. This directly contributes to the 0% specificity observed in v5.0 behavioral tests — the model learns to predict "safe" for rare classes because the loss function rewards it.

**Fix:** When `loss_fn="focal"` and `num_classes > 1`, use `MultiLabelFocalLoss` with per-class alphas derived from `pos_weight` (e.g. `alpha[c] = 1 - pos_count[c] / N`). Remove `FocalLoss` from the multi-label training path entirely.

---

## 5.4 [HIGH] `_FocalFromLogits` wrapper doesn't apply `pos_weight` — silently drops class balancing when using focal loss

**Location:** `trainer.py:840-847`

**Bug:**
```python
if config.loss_fn == "focal":
    _focal = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    class _FocalFromLogits(nn.Module):
        def forward(self, logits, targets):
            return _focal(torch.sigmoid(logits.float()), targets)
    loss_fn = _FocalFromLogits()
else:  # "bce"
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

When `loss_fn="bce"`, `pos_weight` (sqrt-scaled class imbalance correction) is passed to `BCEWithLogitsLoss`. When `loss_fn="focal"`, the `pos_weight` is silently discarded. The `_FocalFromLogits` wrapper doesn't use `pos_weight` at all. So switching from BCE to FocalLoss removes the entire class-balancing mechanism that was carefully computed from the training split.

For rare classes, this means focal loss without pos_weight has even less incentive to detect them — the sqrt-scaled pos_weight (e.g. 12.4x for DoS) that was boosting rare-class gradients is completely absent.

**Impact:** Any experiment comparing BCE vs Focal is not a fair comparison — BCE has class balancing, Focal doesn't. If Focal underperforms, it may be entirely due to the missing pos_weight rather than the focal mechanism itself.

**Fix:** Either incorporate pos_weight into `_FocalFromLogits` (multiply the per-element loss by pos_weight before averaging), or use `MultiLabelFocalLoss` which has per-class alpha that serves the same purpose.

---

## 5.5 [HIGH] Training README says "predictions must be post-sigmoid" and "Do not use BCEWithLogitsLoss" — contradicts v5 code

**Location:** `ml/src/training/README.md:96-97`

**Bug:**
```
**Critical:** predictions must be post-sigmoid. `SentinelModel` outputs are already sigmoid-activated.
Do not use `BCEWithLogitsLoss` — that applies sigmoid internally and would double-apply it.
```

This is the binary-v4 documentation. In v5, `SentinelModel` outputs **raw logits** (no sigmoid). The trainer uses `BCEWithLogitsLoss`. The README's instruction is the exact opposite of what the code does. If an engineer follows the README, they would:
1. Add sigmoid to `SentinelModel.forward()` (or use `BCELoss`)
2. Double-apply sigmoid → all predictions compressed toward 0.5 → gradient vanishing

**Impact:** Same as Group 4 Finding 4.2 — the training README is an active hazard for v5.

**Fix:** Rewrite the README or stamp it as v4-only.

---

## 5.6 [HIGH] `train_one_epoch` applies `aux_loss_weight` from `train()` default (0.1) instead of from `TrainConfig`

**Location:** `trainer.py:455` vs `trainer.py:481` vs `trainer.py:1037`

**Bug:** `train_one_epoch` has its own default:
```python
def train_one_epoch(..., aux_loss_weight: float = 0.1) -> float:
```

This default of 0.1 is never used because `train()` always passes it explicitly at line 1037:
```python
aux_loss_weight=config.aux_loss_weight,
```

But the function signature default of 0.1 is misleading — it suggests the old pre-Phase-0 value is the correct default. If `train_one_epoch` is ever called directly (e.g. in a test, a Jupyter cell, or a probing script), it silently uses 0.1 instead of 0.3.

**Impact:** Same class of bug as 5.1 — the function-level default contradicts the intended Phase 0 value.

**Fix:** Change the default to 0.3 or remove the default and require it to be passed.

---

## 5.7 [MEDIUM] `FocalLoss` uses `F.binary_cross_entropy` which is numerically unstable for near-0/1 probabilities

**Location:** `focalloss.py:61`

**Bug:**
```python
bce = F.binary_cross_entropy(predictions, targets, reduction="none")
```

`F.binary_cross_entropy` computes `-log(p)` directly, which is numerically unstable when `p` is very close to 0 or 1. Under BF16/AMP, the explicit `.float()` cast helps, but the fundamental issue remains: `log(p)` where `p` is a post-sigmoid value can produce very large gradients.

The numerically stable alternative is `F.binary_cross_entropy_with_logits`, which uses the log-sum-exp trick internally. But this class receives post-sigmoid values, so it can't use that. The `_FocalFromLogits` wrapper in trainer.py applies sigmoid first, then passes the result to `FocalLoss` — losing the stability benefit.

**Impact:** For high-confidence predictions (p near 0 or 1), the log computation can produce large gradients, especially under AMP. This can cause training instability for well-classified samples.

**Fix:** Refactor `_FocalFromLogits` to compute focal loss from logits directly (like `MultiLabelFocalLoss` does), avoiding the sigmoid→log roundtrip.

---

## 5.8 [MEDIUM] `pos_weight` computed from training split only — not saved in checkpoint, not reproducible across runs

**Location:** `trainer.py:343-391, 1081-1086`

**Bug:** `compute_pos_weight` is called at training time and the resulting `pos_weight` tensor is passed to `BCEWithLogitsLoss`. But:
1. The `pos_weight` tensor itself is NOT saved in the checkpoint config dict (line 1081-1086 only saves `pos_weight_{name}` as rounded floats for MLflow)
2. The checkpoint config stores `pos_weight_{classname}: round(pw, 3)` — only 3 decimal places. For small weight differences between classes, this rounding changes the actual pos_weight used at inference vs training
3. The predictor doesn't reconstruct `pos_weight` from the checkpoint — it's only needed during training, but if someone re-trains from the checkpoint config, the pos_weight values will differ from the original training

**Impact:** Training resumability is compromised. If splits change (e.g. after dedup), the pos_weight changes but there's no record of the original values in the checkpoint. The 3-decimal rounding in MLflow is too coarse for exact reproduction.

**Fix:** Save the full-precision `pos_weight` tensor in the checkpoint (e.g. `ckpt["pos_weight"] = pos_weight`). On resume, compare the saved pos_weight to the recomputed one and warn if they differ.

---

## 5.9 [MEDIUM] `evaluate()` uses fixed threshold=0.5 — no per-class threshold tuning despite severe class imbalance

**Location:** `trainer.py:401, 421`

**Bug:**
```python
def evaluate(model, loader, device, threshold: float = 0.5, ...) -> dict:
    ...
    preds = (probs >= threshold).long()
```

A single threshold of 0.5 is applied to all 10 classes. For a model outputting raw logits passed through sigmoid, the threshold determines the precision/recall tradeoff per class. With severe class imbalance (DoS at 0.2% vs IntegerUO at 7.8%), a single threshold of 0.5 is almost certainly suboptimal — it may yield 0% recall for rare classes (all predictions below 0.5) or 0% precision for common classes (too many false positives).

The README (line 135-136) says "Use `ml/scripts/tune_threshold.py` to find the optimal inference threshold post-training." But the training loop's early-stopping and checkpoint decisions are based on F1-macro computed with threshold=0.5. If the optimal threshold for DoS is 0.05, the model may have been stopped early because F1@0.5 was poor, even though F1@0.05 would have been much better.

**Impact:** The early-stopping criterion uses a suboptimal threshold, potentially stopping training before the model learns rare-class patterns. This is a direct contributor to the 0% specificity in v5.0 behavioral tests.

**Fix:** Either: (a) use per-class thresholds during evaluation (from a calibration step), or (b) use a threshold-free metric like average precision (AUPRC) for early stopping, or (c) at minimum, evaluate at multiple thresholds and log all of them.

---

## 5.10 [MEDIUM] `compute_pos_weight` uses `pd.read_csv(label_csv)` — loads entire label file for every call, ignores existing indices

**Location:** `trainer.py:367-370`

**Bug:**
```python
df = pd.read_csv(label_csv)
class_cols = CLASS_NAMES[:num_classes]
label_matrix = df[class_cols].values
train_labels = label_matrix[train_indices]
```

This loads the **entire** label CSV (68K+ rows) every time `compute_pos_weight` is called. Then it selects only the training rows. The full CSV load is wasteful but not incorrect.

The real issue is that `train_indices` are numpy array indices into `label_matrix`, which assumes the CSV row order matches the graph/token file order. If the CSV is ever sorted or filtered differently, the indices would select wrong rows. There's no integrity check (e.g. comparing md5 hashes or row counts).

**Impact:** If the label CSV is regenerated with a different row order, `pos_weight` would be computed from wrong samples. The model would train with incorrect class weights.

**Fix:** Add a row-count assertion: `assert len(df) == expected_count`. Or use md5-based indexing instead of positional.

---

## 5.11 [MEDIUM] `train.py` missing `--fusion-dropout` and `--lora-target-modules` CLI flags — can't reproduce checkpoint from CLI

**Location:** `train.py` (entire file)

**Bug:** The `train.py` CLI exposes `--gnn-dropout` and `--lora-dropout`, but NOT `--fusion-dropout`. The `TrainConfig.fusion_dropout` is always set to the dataclass default (0.3). If a checkpoint was trained with `fusion_dropout=0.5`, there's no CLI way to reproduce that configuration.

Similarly, `lora_target_modules` has no CLI flag. The default `["query", "value"]` is always used. If a checkpoint was trained with `["query", "key", "value"]`, the CLI can't reproduce it.

These missing flags mean the `train.py` CLI cannot reproduce any training run that used non-default values for these two parameters.

**Impact:** Reproducibility gap — MLflow may show `fusion_dropout=0.5` for a run, but `train.py --resume` would reconstruct with `fusion_dropout=0.3` (the dataclass default).

**Fix:** Add `--fusion-dropout` and `--lora-target-modules` CLI flags.

---

## 5.12 [MEDIUM] `WeightedRandomSampler` with `replacement=True` changes effective dataset size — not accounted for in scheduler/early-stopping

**Location:** `trainer.py:560, 653`

**Bug:**
```python
WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
```

With `replacement=True`, the sampler draws `len(weights)` samples with replacement per epoch. This means:
1. Some samples appear multiple times, some not at all
2. The effective dataset size per epoch is the same, but the composition changes
3. `steps_per_epoch` is calculated from `len(train_dataset)`, which counts unique samples, not the sampler's actual draw count (which is the same, but with repeats)
4. More critically, the weighted sampler interacts badly with `OneCycleLR` — the scheduler's learning rate curve is based on `total_steps = epochs * steps_per_epoch`, but the gradient distribution is different because rare-class samples appear more often

The sampler is documented as an "autoresearch harness" feature, but it's exposed in the CLI with no guardrails.

**Impact:** Using `--weighted-sampler DoS-only` with `--no-resume-model-only` could cause severe optimizer state mismatch because the gradient distribution changes between runs.

**Fix:** Document the interaction with scheduler and optimizer state. Consider resetting the optimizer when switching sampler modes.

---

## 5.13 [LOW] `MultiLabelFocalLoss` is dead code — defined but never imported or used anywhere

**Location:** `focalloss.py:77-131`

**Bug:** `MultiLabelFocalLoss` exists in `focalloss.py` with per-class alpha support (the correct approach for multi-label), but it is never imported or used by any file in the project. The trainer always uses the scalar-alpha `FocalLoss` even for multi-label classification.

**Impact:** Dead code that provides the correct solution to Finding 5.3 but isn't wired up. Confusing for anyone reading the file and wondering why the multi-label version exists but isn't used.

**Fix:** Wire `MultiLabelFocalLoss` into the training loop when `num_classes > 1` and `loss_fn="focal"`.

---

## 5.14 [LOW] `train.py` sets `mp.set_start_method('spawn', force=True)` at module level — affects any import

**Location:** `train.py:73`

**Bug:**
```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

This sets the multiprocessing start method globally at import time. If any other module imports `train.py` (even just to use `parse_args` or `TrainConfig`), it changes the multiprocessing behavior for the entire process. The `force=True` means it overrides any previous setting, which could break other libraries that expect `fork` (the default on Linux).

This is especially problematic because DataLoader workers are affected — `spawn` is slower than `fork` for worker startup because it has to serialize the entire process state.

**Impact:** Minor performance impact on DataLoader worker startup. Could cause unexpected behavior if `train.py` is imported rather than run as a script.

**Fix:** Move the `set_start_method` call inside `if __name__ == "__main__":` block, or use a context manager.

---

## 5.15 [LOW] Training loop logs gradient norms for eye projections only — misses LoRA gradient and GNN conv gradient

**Location:** `trainer.py:494-503`

**Bug:**
```python
gnn_norm = _grad_norm(model.gnn_eye_proj)
tf_norm  = _grad_norm(model.transformer_eye_proj)
fused_norm = _grad_norm(model.fusion)
```

The gradient norm logging only covers:
- `gnn_eye_proj` (Linear projection after pooling)
- `transformer_eye_proj` (Linear projection after CLS extraction)
- `fusion` (CrossAttentionFusion module)

It does NOT log:
- LoRA adapter gradients (the main transformer learning signal)
- GNN conv layer gradients (the structural learning signal)
- Classifier gradients

During the gradient collapse investigation, knowing whether LoRA gradients were flowing would have been critical. The current logging shows that `gnn_eye_proj` gradient was small, but doesn't reveal whether the GNN conv layers themselves had healthy gradients or if the collapse was in the projection.

**Impact:** Diagnostic blind spot. Future gradient collapse investigations will lack the data to distinguish between "GNN convs have zero gradient" vs "GNN pooling loses the signal."

**Fix:** Add gradient norm logging for `model.gnn` (all conv layers), `model.transformer.bert` (LoRA params), and `model.classifier`.

---

## Summary Table

| # | Severity | File | Finding |
|---|----------|------|---------|
| 5.1 | **CRITICAL** | train.py / trainer.py | `--aux-loss-weight` default=0.3 (CLI) vs TrainConfig default=0.1 — Phase 0 gradient collapse fix silently absent for programmatic use |
| 5.2 | **CRITICAL** | trainer.py / train.py | `batch_size` both at 16, but Fix #17 claims both should be 32 — fix never applied or reverted |
| 5.3 | **HIGH** | focalloss.py | Scalar `alpha=0.25` down-weights rare-class positives in multi-label — opposite of needed behavior for DoS/Timestamp |
| 5.4 | **HIGH** | trainer.py | `_FocalFromLogits` discards `pos_weight` — switching to focal silently removes all class balancing |
| 5.5 | **HIGH** | training/README.md | README says "use BCELoss, model outputs sigmoid" — v5 uses BCEWithLogitsLoss on raw logits |
| 5.6 | **HIGH** | trainer.py | `train_one_epoch` aux_loss_weight default=0.1 contradicts Phase 0 fix (0.3) |
| 5.7 | MEDIUM | focalloss.py | `F.binary_cross_entropy` on post-sigmoid values is numerically unstable; `binary_cross_entropy_with_logits` from logits would be better |
| 5.8 | MEDIUM | trainer.py | `pos_weight` not saved in checkpoint — rounding to 3 decimals in MLflow is lossy; not reproducible |
| 5.9 | MEDIUM | trainer.py | `evaluate()` uses threshold=0.5 for all 10 classes — suboptimal for rare classes, affects early stopping |
| 5.10 | MEDIUM | trainer.py | `compute_pos_weight` loads full CSV; no integrity check that row order matches indices |
| 5.11 | MEDIUM | train.py | Missing `--fusion-dropout` and `--lora-target-modules` CLI flags — can't reproduce non-default configs |
| 5.12 | MEDIUM | trainer.py | WeightedRandomSampler with replacement=True not accounted for in scheduler/optimizer state |
| 5.13 | LOW | focalloss.py | `MultiLabelFocalLoss` is dead code — correct solution for 5.3 exists but isn't wired up |
| 5.14 | LOW | train.py | `mp.set_start_method('spawn')` at module level affects any import of the file |
| 5.15 | LOW | trainer.py | Gradient norm logging misses LoRA, GNN conv, and classifier — diagnostic blind spot |

**Cross-references:**
- Finding 5.1 ↔ Group 4 Finding 4.4: Same `aux_loss_weight=0.1` vs `0.3` contradiction, now confirmed to be a CLI vs dataclass default mismatch
- Finding 5.3/5.4 ↔ v5.0's 0% specificity: The combination of wrong alpha direction (5.3) and missing pos_weight (5.4) when using focal loss means rare-class detection is actively suppressed
- Finding 5.5 ↔ Group 4 Finding 4.2: Both READMEs are v4-stale and actively dangerous
Group 5 audit complete — **15 findings** (2 CRITICAL, 4 HIGH, 6 MEDIUM, 3 LOW).

Key takeaways for the training pipeline:

1. **Most damaging to v5.0**: Findings 5.1 + 5.3 + 5.4 form a triple punch against rare-class detection:
   - 5.1: The `aux_loss_weight=0.3` Phase 0 fix is only applied via CLI, not in `TrainConfig` defaults
   - 5.3: FocalLoss `alpha=0.25` *down-weights* rare-class positives in multi-label (DoS gets 0.25 weight for its rare positives)
   - 5.4: Switching to focal loss silently discards the carefully computed `pos_weight`

2. **batch_size mystery (5.2)**: Fix #17 claims both defaults were changed to 32, but the code shows both at 16. Either the fix was never applied, or it was reverted. This matters because gradient noise scale and OneCycleLR calibration are batch-size-sensitive.

3. **Dead code is the solution (5.13)**: `MultiLabelFocalLoss` with per-class alphas exists in the codebase but is never used. It's exactly the right solution for Finding 5.3, but nobody wired it up.

4. **Documentation is actively dangerous (5.5)**: The training README tells engineers to use `BCELoss` and that the model outputs sigmoid. Both are wrong for v5.

Ready for Group 6 when you are.