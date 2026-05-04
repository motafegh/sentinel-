# 2026-05-04 — Post-Training Audit: Fixes #1–#7 + MLflow Focal Params (#9)

Date: 2026-05-04  
Author: motafegh  
Fixes: #1 (dataset), #2 (predictor), #3 (tune_threshold), #4 (predictor warmup), #5 (tune_threshold prefetch), #6 (predictor API schema), #7 (predictor arch mapping), #9 (MLflow focal params)

---

## Summary

A second audit pass over inference and data pipeline files after the training
audit (2026-05-04 earlier today). Eight bugs corrected spanning the dataset
loader, predictor, threshold-tuning script, and MLflow logging. All fixes are
correctness or robustness issues; no new features.

---

## Fix #1 — `ml/data_extraction/dataset.py`: edge_attr shape guard for pre-refactor `.pt` files

**Commit:** 7f52484a01cb2a7a88ea56aef30307297abb05cf

**Problem:**  
Old graph `.pt` files (generated before the 2026-05-03 re-extraction) stored
`edge_attr` as shape `[E, 1]`. `GNNEncoder` passes `edge_attr` into
`nn.Embedding()` which requires a 1-D index tensor of shape `[E]`. Any attempt
to load an old file through the current code path would crash at the embedding
lookup step with a shape error.

**Fix:**  
Added a `squeeze(-1)` guard in `__getitem__` before return:

```python
if graph.edge_attr is not None and graph.edge_attr.dim() == 2:
    graph.edge_attr = graph.edge_attr.squeeze(-1)
```

Safe for correctly-shaped new files (squeeze on a 1-D tensor is a no-op).

---

## Fix #2 — `ml/src/inference/predictor.py`: missing `SentinelModel` args on load

**Commit:** 2348eec2af2f85bc26479b6fe2de6aec4083e1e3

**Problem:**  
`SentinelModel()` was constructed passing only `num_classes` and
`fusion_output_dim` from the saved checkpoint config. Three fields were silently
omitted: `dropout` (fusion_dropout), `gnn_dropout`, and `lora_target_modules`.

For a checkpoint trained with non-default LoRA targets (e.g.
`lora_target_modules=["query","value","key"]`), `load_state_dict()` would crash
on API startup with a key mismatch error because the reconstructed model's
LoRA layer names differ from the checkpoint's.

**Fix:**  
All relevant fields from `saved_cfg` are now forwarded to `SentinelModel()`:
`dropout`, `gnn_dropout`, `lora_r`, `lora_alpha`, `lora_dropout`,
`lora_target_modules`, `use_edge_attr`, `gnn_hidden_dim`, `gnn_heads`,
`gnn_edge_emb_dim`.

---

## Fix #3 — `ml/scripts/tune_threshold.py`: same missing `SentinelModel` args

**Commit:** 15f3db00bdec9861425c1e54f0b930956b88f28a

**Problem:**  
Identical to Fix #2. `load_model_from_checkpoint()` in `tune_threshold.py` had
the same incomplete `SentinelModel()` call — dropout, gnn_dropout, and
lora_target_modules were not passed. Non-default LoRA checkpoints would crash
with a state_dict key mismatch during threshold tuning.

**Fix:**  
Same treatment as Fix #2 — all arch fields forwarded from the saved config.

---

## Fix #4 — `ml/src/inference/predictor.py`: missing `edge_attr` in warmup dummy graph

**Commit:** 2348eec2af2f85bc26479b6fe2de6aec4083e1e3

**Problem:**  
`_warmup()` built a 2-node dummy graph with no `edge_attr`. When
`use_edge_attr=True` is stored in the checkpoint config, `GNNEncoder` calls
`self.edge_emb(edge_attr)` (`nn.Embedding`). The warmup graph's missing
`edge_attr` was never exercised, so:
- If a bug in the embedding path existed it would not surface during warmup
- Any path that gates on `edge_attr is not None` diverges between warmup and
  real inference, defeating the warmup's purpose

**Fix:**  
Added a 1-D long zero tensor of shape `[E]` (2 edges) to the dummy graph when
`use_edge_attr=True` in the saved config:

```python
if self._cfg.get("use_edge_attr", False):
    dummy_graph.edge_attr = torch.zeros(dummy_graph.edge_index.shape[1],
                                        dtype=torch.long)
```

---

## Fix #5 — `ml/scripts/tune_threshold.py`: `prefetch_factor` warning with `num_workers=0`

**Commit:** e9e4baf91d519639fa4c953f0416d152585b6666

**Problem:**  
`build_val_loader()` passed `prefetch_factor=2 if num_workers > 0 else None`
directly as a keyword argument to `DataLoader`. PyTorch 2.x raises a
`UserWarning` when `prefetch_factor` is passed at all (even as `None`) when
`num_workers=0`, because the parameter is meaningless in the single-process
case.

**Fix:**  
DataLoader kwargs are now built conditionally — `prefetch_factor`, `pin_memory`,
and `persistent_workers` are only included in the dict when `num_workers > 0`:

```python
loader_kwargs: dict = {"batch_size": batch_size, "shuffle": False}
if num_workers > 0:
    loader_kwargs.update({
        "num_workers": num_workers,
        "prefetch_factor": 2,
        "pin_memory": True,
        "persistent_workers": True,
    })
return DataLoader(dataset, **loader_kwargs)
```

---

## Fix #6 — `ml/src/inference/predictor.py`: API response schema `threshold` → `thresholds`

**Commit:** e9e4baf91d519639fa4c953f0416d152585b6666

**Problem:**  
`_format_result()` was returning `"threshold": self.threshold` — a single
fallback float — even when per-class thresholds were loaded from a JSON file.
API consumers had no visibility into the actual per-class decision boundaries
being used; all runs looked like they used one uniform threshold.

**Fix:**  
Changed to `"thresholds": self.thresholds.cpu().tolist()` — a list of floats,
one per class, matching the actual values used to produce the predictions.

> ⚠️ **Breaking API change**: the response key is renamed from `"threshold"`
> (singular float) to `"thresholds"` (list of floats). Any downstream consumer
> parsing this field must be updated.

---

## Fix #7 — `ml/src/inference/predictor.py`: `fusion_output_dim` fallback for legacy checkpoints

**Commit:** e9e4baf91d519639fa4c953f0416d152585b6666

**Problem:**  
`fusion_output_dim` lookup used only `_ARCH_TO_FUSION_DIM` (a hardcoded dict).
New checkpoints (from trainer.py post-P0-C) already store `fusion_output_dim`
directly in the checkpoint config, making the hardcoded dict redundant — and a
potential source of conflict if the model was trained with a non-default dim.

**Fix:**  
Lookup now prefers `saved_cfg.get("fusion_output_dim")` first and falls back to
`_ARCH_TO_FUSION_DIM` only for legacy checkpoints that predate the trainer
saving this value:

```python
fusion_output_dim = (
    saved_cfg.get("fusion_output_dim")
    or _ARCH_TO_FUSION_DIM.get(architecture, 64)
)
```

---

## Fix #9 — `ml/src/training/trainer.py`: log `focal_gamma` / `focal_alpha` to MLflow

**Commit:** 13eb71f111735ba9cfe620ddb6e2af43604625c9

**Problem:**  
`focal_gamma` and `focal_alpha` were added to `TrainConfig` in the training
audit (PR #24/#25) but were never included in the `params` dict logged to MLflow
at run start. Every run looked identical from a loss-config perspective in the
MLflow UI, making it impossible to filter or compare Focal Loss sweeps.

**Fix:**  
Both params added unconditionally to the logged params dict:

```python
"focal_gamma": config.focal_gamma,
"focal_alpha": config.focal_alpha,
```

When `loss_fn='bce'` they are present but irrelevant (clearly labeled);
when `loss_fn='focal'` they are the primary hyperparameters under sweep.

---

## Files Changed

| File | Fixes Applied |
|---|---|
| `ml/data_extraction/dataset.py` | Fix #1 (edge_attr squeeze guard) |
| `ml/src/inference/predictor.py` | Fix #2 (SentinelModel args), Fix #4 (warmup edge_attr), Fix #6 (thresholds API key), Fix #7 (fusion_output_dim preference) |
| `ml/scripts/tune_threshold.py` | Fix #3 (SentinelModel args), Fix #5 (prefetch_factor guard) |
| `ml/src/training/trainer.py` | Fix #9 (MLflow focal params) |
