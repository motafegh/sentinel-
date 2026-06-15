# Pre-Run 10 Proposal Audit: Codebase Feasibility Analysis

**Date:** 2026-06-07
**Source:** `docs/pre-run10-fixes/pre_run10_proposal.md`
**Codebase:** Run 9 (v9 schema, v10 data, four-eye v8.1 architecture)
**Author:** Claude Code — code audit against current source tree

---

## TL;DR

| Proposal | Status | Risk | Effort | Dependencies |
|----------|--------|------|--------|--------------|
| **A1** Static Calibrator | New | Zero | 4h | SCsVulLyzer (`secondAnalyzer`) **not installed** |
| **A2** Pre-Filter Agent | New | Zero | 1d | SCsVulLyzer **not installed** |
| **A3** Curriculum Sampler | New | Medium | 2d | None |
| **B1** Temperature Scaling | New (calibration files exist, not wired) | Zero | 30m | None |
| **B2** Optuna Threshold Search | **Replaces** existing grid sweep | Zero | 2h | `optuna` **not in pyproject.toml** |
| **B3** Label Smoothing | **ALREADY IMPLEMENTED** | — | — | — |
| **B4** R-Drop | New | Low | 1d | None |
| **B5** Label Dependency Graph | New | Low | 1d | None |
| **B6** MC Dropout Uncertainty | New | Zero | 4h | None |
| **B7** SWA | New | Low | 1d | None |
| **B8** Self-Paced Learning | New (depends on A3) | Medium | 1d | A3 first |

---

## Context: What Was Verified Against

All findings below are based on reading (not running) the current source:

- `ml/src/models/sentinel_model.py` — four-eye v8.1 architecture, 670 lines
- `ml/src/models/gnn_encoder.py` — three-phase 8-layer GAT, 667 lines
- `ml/src/training/trainer.py` — training loop, v8 config + early stopping, 2059 lines
- `ml/src/training/losses.py` — ASL + BCE implementations, 126 lines
- `ml/src/inference/predictor.py` — inference pipeline, per-class thresholds, 757 lines
- `ml/src/inference/api.py` — FastAPI endpoint with drift detection, 402 lines
- `ml/scripts/train.py` — CLI entry point, 356 lines
- `ml/scripts/tune_threshold.py` — threshold sweep, 625 lines
- `ml/src/datasets/dual_path_dataset.py` — dataset + RAM cache, 403 lines
- `ml/src/preprocessing/graph_schema.py` — v9 schema constants, 504 lines
- `agents/src/mcp/servers/` — existing MCP server pattern
- `ml/calibration/` — Run 7 calibration artifacts
- `ml/pyproject.toml` — dependency registry (verified SCsVulLyzer absent)

---

## Block A: SCsVulLyzer V2.0 Integration

### A1: Post-Inference Hard Constraint Calibrator

**Claim:** Zero risk, 4 hours, no model changes.

#### Verified Against Code

**Integration point:** After `predictor.predict_source()` returns sigmoid probs but before `_format_result()`. The natural hook is in `predictor._score_windowed()` (predictor.py:609-639) — specifically between `probs = torch.sigmoid(logits.float()).squeeze(0)` (line 638) and `self._format_result(graph, probs, ...)` (line 639).

**What would need to change:**
1. `predictor.py` — `_score_windowed()` needs an optional calibrator parameter or the calibrator wraps the predictor externally.
2. A new `ml/src/inference/static_calibrator.py` file.

**Feature name alignment risk:** The proposal's `IMPOSSIBILITY_RULES` reference features like `f["Solidity call_CALL"]` and `f["Bytecode Length and Entropy_bytecode_entropy"]`. These are SCsVulLyzer V2.0 output column names. **These must be verified against the actual `secondAnalyzer.analyze_solidity_contract()` return dict** before implementation — if the actual feature names differ, the calibrator silently becomes a no-op.

**Dependency status:** `secondAnalyzer` is NOT in `ml/pyproject.toml`. SCsVulLyzer would need to be installed (pip or git dependency). Its own dependency chain includes `py-solc-x` and a solc binary — these add failure modes to inference that currently don't exist.

**Performance:** Every inference call re-compiles the .sol file. Solc compilation adds 2-5s per contract on top of the existing ~15s inference. The proposal acknowledges ~200ms for feature extraction but that excludes solc compilation time.

**Verdict:** Architecturally sound. Integration is clean. Blocked by SCsVulLyzer not being in the dependency tree. If SCsVulLyzer is installed, implementation is straightforward.

---

### A2: Pre-Filter Agent (MCP Tool)

**Claim:** Zero risk, 1 day, completely separate from ML module.

#### Verified Against Code

**Pattern exists:** `agents/src/mcp/servers/inference_server.py` (481 lines, SSE transport) shows the exact pattern. A new `agents/src/mcp/servers/static_analyzer_server.py` would follow it.

**Integration challenge:** This is not "0 effort 1 day" if you want it to actually route traffic. The LangGraph orchestration at `agents/src/orchestration/` would need to be updated to:
1. Call the static analyzer first
2. Only route to inference_server if `route_to_sentinel` is true
3. Merge the `impossible_vulns` list into the final report

**Threshold calibration requirement:** The proposal itself notes the 1.5 threshold must be validated against the labeled dataset — "compute the fraction of true-positive contracts that would be incorrectly filtered out. Target: < 1% true positive suppression rate." This is non-trivial verification work.

**Verdict:** Same dependency block as A1. The MCP server structure exists and works. Implementation is plumbing + the SCsVulLyzer dependency.

---

### A3: Training Curriculum via Complexity Score

**Claim:** Medium risk, 2 days + 1 training run.

#### Verified Against Code

**Code integration:**

1. The `compute_complexity_score` function reads from `graph_data` (PyG Data object) — these are all available from the cached dataset `DualPathDataset.cached_data` (a dict of hash → (graph, tokens) pairs).

2. The current DataLoader setup in `trainer.py:1121-1124`:
```python
if _sampler is not None:
    train_loader = DataLoader(train_dataset, sampler=_sampler, shuffle=False, **_loader_kwargs)
else:
    train_loader = DataLoader(train_dataset, shuffle=True, **_loader_kwargs)
```
The `CurriculumSampler` would replace `_sampler` entirely. The existing `_build_weighted_sampler` and its `--weighted-sampler` CLI argument (supporting "positive", "DoS-only", "all-rare", "timestamp-size") would need to coexist with or be replaced by the curriculum sampler.

3. The proposal recommends a mixing floor (15% hard samples) to prevent distribution shift — this is critical because the existing `WeightedRandomSampler` with "timestamp-size" mode is addressing a specific known failure mode (Timestamp size shortcut). Mixing floors would partially override that.

**Risk — label-complexity correlation:** The proposal correctly flags this. You already have `drop_complexity_feature=True` in Run 8 for this exact reason. The complexity score must be validated:
```python
np.corrcoef(scores, label_density) < 0.4
```
This correlation check is not trivial — it requires computing complexity scores for all 29,103 training graphs and correlating with their label densities. If it fails, the decorrelation step (residualizing against label density) must be applied. This adds implementation surface.

**Metadata availability:** Edge diversity (`graph_data.edge_attr.unique().numel()`) is available from the cached graphs. Node type IDs from `graph.x[:, 0]` are always present. The `_FUNC_IDS_CPU` reference from `sentinel_model.py` can be reused (it's a module-level frozenset).

**Verdict:** Medium risk as labeled. The complexity-label decorrelation check is the gate. Without it, this re-introduces the shortcut you removed with `drop_complexity_feature`. Implementation is clean but the correlation gate adds ~1 day of analysis work before any training.

---

## Block B: Advanced Training Techniques

### B1: Temperature Scaling

**Claim:** Zero risk, 30 minutes, no retraining.

#### Verified Against Code

**Existing artifacts:** `ml/calibration/temperatures_run7.json` exists but contains **thresholds** (35-50% per class), not temperature values. The temperature scaler code from the proposal is **not wired into any pipeline**.

**What would need to change:**
1. New `ml/src/inference/temperature_scaler.py` (20 lines + fitting function)
2. Add temperature loading to `predictor.py::__init__` — load T from a JSON sidecar, apply to logits before sigmoid
3. The proposal's `calibrate_temperature` function runs on the full validation set — this takes ~35 min (one epoch) of inference time but requires no training.

**Checkpoint save format:** The checkpoint dict at `trainer.py:2003-2026` already has a `config` sub-dict. Adding `"temperature": T` to this would make temperature a first-class checkpoint attribute. Currently the checkpoint config saves all TrainConfig attrs via `**dataclasses.asdict(config)`.

**Verdict:** Zero risk, trivially implementable. The 30-minute estimate is accurate. Saves alongside existing thresholds JSON.

---

### B2: Optuna Per-Class Threshold Search

**Claim:** Zero risk, 2 hours.

#### Verified Against Code

**Current implementation:** `ml/scripts/tune_threshold.py` — per-class independent grid sweep over 19 candidates (linspace 0.05-0.95, step 0.05). This is exactly what the proposal describes as the limitation. The `evaluate()` function in `trainer.py:555-576` has the same per-class sweep (19 candidates, linspace 0.1-0.9).

**What Optuna changes:**
- Joint search over all 10 dimensions simultaneously (captures class interactions)
- TPE sampler is more efficient than grid (2000 trials in ~2 min CPU)
- The proposal's `objective(trial)` evaluates `macro_f1(preds, val_labels)` globally, not per-class

**Dependency:** `optuna` is NOT in `ml/pyproject.toml`. Would need `pip install optuna`.

**Integration:** The proposal expects pre-computed validation probabilities (`val_probs: [N, 10]`). This data is already collected during `tune_threshold.py:collect_probabilities()` and during `trainer.py::evaluate()`. The Optuna search is a drop-in replacement for the `sweep_one_class()` loop.

**Run 7 baseline:** Current tuned thresholds (from `temperatures_run7.json`) give tuned F1 = 0.3423. The proposal claims +0.03-0.08 F1 improvement potential. This is realistic for a search that captures joint class interactions which the independent per-class sweep cannot.

**Verdict:** Clean win. No model risk. Add optuna dependency, replace the per-class loop. The tuner already exists so this is a modification to an existing script.

---

### B3: Label Smoothing

**Claim:** Low risk, 1 hour.

#### Verified Against Code

**ALREADY FULLY IMPLEMENTED** in the existing codebase. No changes needed.

`TrainConfig` (trainer.py:308-319):
```python
class_label_smoothing: dict = field(default_factory=lambda: {
    "CallToUnknown":               0.10,
    "DenialOfService":             0.18,
    "ExternalBug":                 0.10,
    "GasException":                0.12,
    "IntegerUO":                   0.08,
    "MishandledException":         0.12,
    "Reentrancy":                  0.14,
    "Timestamp":                   0.05,
    "TransactionOrderDependence":  0.10,
    "UnusedReturn":                0.10,
})
```

Applied in `train_one_epoch()` (trainer.py:659-662):
```python
if class_eps is not None:
    labels = labels * (1.0 - class_eps) + 0.5 * class_eps
elif label_smoothing > 0.0:
    labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
```

**Difference from proposal:** The proposal suggests different epsilon values (e.g., IntegerUO=0.03 vs code's 0.08; TOD=0.10 vs code's 0.10 matches). The proposal uses `(1 - labels) * eps` for the smoothing term while the code uses `0.5 * eps`. The difference is:
- Proposal: `smooth = orig * (1-ε) + (1-orig) * ε` — shifts toward 0.5 for both classes
- Code: `smooth = orig * (1-ε) + 0.5 * ε` — also shifts toward 0.5, but asymmetrically

Both achieve the same regularization effect. The proposal's formulation is standard label smoothing; the code's `0.5 * ε` is a slight variant. Neither is wrong.

**Verdict:** The feature exists, is configurable, and is active in all runs. **Skip B3.** If you want to experiment with the proposal's specific epsilon values, that's a hyperparameter change not an implementation change.

---

### B4: R-Drop Regularization

**Claim:** Low risk, 1 day, adds ~40% training time.

#### Verified Against Code

**Integration into trainer.py:**

The proposal's `rdrop_loss` requires:
1. Two model forward passes per batch (instead of one)
2. KL divergence computation between dropout-perturbed probability distributions
3. Modified loss: `task_loss + alpha * KL`

**Current `train_one_epoch()` structure:** One forward pass at line 665:
```python
logits, aux = model(graphs, input_ids, attention_mask, return_aux=True)
```

A second forward pass would duplicate this call. The model's `torch.compile` submodules (enabled by default as `use_compile=True`) may recompile on the second forward pass due to different dropout masks triggering different graph shapes in dynamo. The `cache_size_limit=256` setting helps but the first few batches would be slow.

**Compatibility with existing features:**
- ASL loss: The proposal uses `criterion()` which is currently `AsymmetricLoss()`. R-Drop computes KL on sigmoid probabilities, which is independent of the task loss — this works.
- Auxiliary heads: The model returns `aux` dict when `return_aux=True`. Both forward passes would return aux dicts. The auxiliary loss computation (lines 700-716) would need to be duplicated or averaged over the two passes.
- DoS gradient scaling: Already applied via `_logits_for_loss` (lines 672-680). Would need to apply to both passes.
- JK entropy regularization: Applied at line 732-737, reads `aux["jk_entropy"]`. With two passes, you'd average the JK entropy or compute it on the combined logits.
- Gradient accumulation: Already at 8 accum steps. R-Drop doubles the forward cost per micro-batch.

**Training time impact:** Proposal claims ~40% overhead. Given current ~35 min/epoch (Run 7 baseline), this becomes ~49 min/epoch. This is acceptable.

**Interaction with `torch.compile`:** The KL divergence computation (`F.kl_div`, `torch.sigmoid`) is outside any compiled submodule — it runs in eager mode on the combined outputs. No issue.

**Verdict:** Feasible but the code changes to `train_one_epoch()` are not trivial due to the auxiliary head, DoS weight masking, and JK entropy entanglement. Estimate 2 days for careful implementation + testing, not 1 day.

---

### B5: Label Dependency Graph

**Claim:** Low risk, 1 day.

#### Verified Against Code

**Integration into `sentinel_model.py`:**

1. Add `LabelDependencyLayer` class at the top or in a new file
2. In `SentinelModel.__init__`: accept `label_adj: torch.Tensor | None = None`, create `self.label_dep = LabelDependencyLayer(num_classes, label_adj)` when provided
3. In `SentinelModel.forward()`: apply after classifier, before return:
```python
logits = self.label_dep(logits)
```
This is clean — the layer operates on [B, C] logits, same shape throughout.

**Label adjacency data:** `build_label_adjacency(train_labels)` needs the training label matrix. This is available in `trainer.py::train()` after `DualPathDataset` loads `_label_map`. The labels are used in `compute_pos_weight()` (line 440-444) which accesses `train_dataset._label_map`.

**Initialization safety:** `nn.init.eye_` + 0.1 residual scale ensures identity initialization. The proposal's analysis is correct — this starts as a no-op and only backprop introduces corrections. If signal is noisy, the layer learns to suppress it.

**Forward pass shape:** The proposal's `forward()` uses `torch.mm(logits, self.adj)` then `self.W(neighbor_signal)`. The `self.adj` buffer is [C, C] and `self.W` is Linear(C, C). ZK proof compatibility note: this is 100 linear parameters (10×10 bias=False) — negligible for the ZK circuit.

**Checkpoint impact:** No change. `self.label_dep` parameters are saved as part of `model.state_dict()` automatically.

**Verdict:** Low risk, clean integration. The training label matrix is already in memory. Adding this before the Phase 1 planned training run is practical.

---

### B6: Monte Carlo Dropout Uncertainty

**Claim:** Zero risk, 4 hours.

#### Verified Against Code

**Integration into `predictor.py`:**

The proposal's `predict_with_uncertainty` adds a separate inference mode. This slots into the Predictor class as a new method alongside `predict_source()` and `predict_with_hotspots()`.

**Model.train() at inference:** The proposal sets `model.train()` to activate dropout paths, then runs n=30 forward passes with `torch.no_grad()`. The current `predictor.py` always calls `self.model.eval()` before inference (lines 482, 609, 648). A separate method avoids changing existing behavior.

**Compatibility concerns:**
1. `torch.compile` + dropout: Compiled submodules handle dropout correctly — it's traced as a control flow operation. No issue.
2. GNN prefix injection: At inference, `model._current_epoch = 9999` so prefix is always active. MC dropout with `model.train()` keeps this setting — no change.
3. BatchNorm: The GNN uses LayerNorm (not BatchNorm), and CodeBERT is frozen. No BatchNorm statistics to worry about in train() mode.

**Performance:** 30 forward passes = 30× inference cost. The proposal's claim of 4 hours is for implementation only. At inference time, each request becomes ~30× slower (from ~15s to ~450s on GPU). This is only practical for:
- An optional `/predict/uncertainty` endpoint
- Batch/offline analysis, not real-time API

**Tier integration:** The uncertainty output naturally extends the three-tier system:
- `needs_review` flag → adds a fourth "UNCERTAIN" tier between SUSPICIOUS and CONFIRMED
- The proposal's output format (`"Reentrancy: 0.87 ± 0.03 → HIGH CONFIDENCE"`) could complement the existing probabilities dict

**Verdict:** Zero model risk. Implementation is straightforward. The 30× inference cost makes this practical only as an optional endpoint, not the default path. The existing predictor structure supports this cleanly.

---

### B7: Stochastic Weight Averaging

**Claim:** Low risk, 1 day.

#### Verified Against Code

**Integration into `trainer.py`:**

The proposal wraps the model in `AveragedModel` after initialization and switches to SWALR at 70% of training. This interacts with several existing mechanisms:

1. **`torch.compile`:** The `AveragedModel` wrapper averages parameter weights across epochs. Compiled submodules share parameter storage with the original — `AveragedModel` keeps a shadow copy of parameters that averages them. This should work because PyTorch's `AveragedModel` operates on `.data`, not on compiled graphs. However, this needs testing.

2. **OneCycleLR conflict:** The proposal's `SWALR` replaces the standard `scheduler.step()` after `swa_start`. Currently, `OneCycleLR` is stepped every optimizer step (line 891: `scheduler.step()`). The SWA phase would need to stop stepping OneCycleLR and start stepping SWALR. This requires tracking a `swa_active` flag.

3. **Checkpoint format:** Currently saves `{"model", "optimizer", "scheduler", "epoch", "best_f1", ...}`. SWA would add `{"swa_model": swa_model.state_dict()}`. The existing 2.5 GB checkpoint would grow by the model size (~1.5 GB for the full checkpoint with optimizer state). This is acceptable but worth noting.

4. **`update_bn`:** The proposal correctly calls `update_bn(train_loader, swa_model)` after training. This requires one full pass through the training data (~35 min). This is a one-time cost at the end of training.

**Training time vs value:** SWA adds ~1 epoch of time (35 min for `update_bn`) plus the SWALR phase in the last 30% of epochs. If the model converges at ~ep30 (Run 7 stopped at ep41), SWA starts at ~ep21. The benefit is flatter minima for OOD generalization — this addresses the real problem of test_contracts being massively OOD (MEMORY.md: "median 20 nodes vs 90 training; 100% use 0.8.x").

**Verdict:** Feasible but requires careful scheduler integration. The main challenge is coexisting with OneCycleLR and torch.compile. Estimate closer to 2 days for the scheduler refactoring alone.

---

### B8: Self-Paced Learning

**Claim:** Medium risk, 1 day on top of A3.

#### Verified Against Code

**Dependency on A3:** B8 is an extension of `CurriculumSampler` to `SelfPacedSampler`. Without A3 being implemented first, B8 has no base class to extend.

**Per-sample loss computation:** The proposal computes `model(sample.graphs, sample.input_ids, sample.mask)` for each training sample after each epoch. This iterates through 29,103 training samples one by one — NOT batched. At 34 min/epoch for batched training, iterating sample-by-sample would take significantly longer.

**Practical implementation:** The per-sample loss computation should be batched (reuse the existing DataLoader). The losses per sample can be extracted by computing `F.binary_cross_entropy_with_logits(logits, labels, reduction='none')` and averaging over classes:
```python
per_sample = loss.mean(dim=1)  # [B]
```

This batch-compatible computation avoids the slow per-sample loop in the proposal.

**Threshold λ dynamics:** The proposal's `λ = base_lambda * (1 + epoch * growth_rate)` grows linearly. This is a coarse heuristic — the per-sample loss distribution changes non-linearly as training progresses. A percentile-based threshold (e.g., `λ = np.percentile(losses, 70 + epoch * 2)`) would be more robust.

**Rare class concern (GasException, TOD):** The proposal correctly identifies that self-paced learning initially excludes these high-loss samples, then includes them when λ rises. However, if GasException and TOD samples always have high loss (because the model can never learn them from the current features), they would remain excluded indefinitely. The mixing floor (10% hard samples) is the safety net — it ensures some fraction of hard samples are always included.

**Verdict:** Correct in concept. The proposal's sample-by-sample loss computation is impractical — must be batched. The threshold dynamics need more thought than a linear growth schedule. Requires A3 first.

---

## Summary: Feasibility by Implementation Phase

### Phase 0 (This Week, No Retraining — Zero Risk)

| Task | Verdict | Notes |
|------|---------|-------|
| **B1: Temperature Scaling** | ✅ **Do it** | 30 min, no dependency, clean win |
| **B2: Optuna Threshold Search** | ✅ **Do it** | Needs `pip install optuna`, replaces existing grid |
| **B6: MC Dropout Uncertainty** | ✅ **Do it (optional endpoint)** | 4h impl, 30× inference cost — make it optional |
| **A1: Static Calibrator** | ⏸️ **Blocked** | Needs SCsVulLyzer installed first |

### Phase 1 (Next Training Run)

| Task | Verdict | Notes |
|------|---------|-------|
| **B5: Label Dependency Graph** | ✅ **Do it** | Clean integration, 100 params, zero risk |
| **B4: R-Drop** | ⚠️ **Careful** | Trainer refactor is non-trivial (aux heads, DoS masking, compile). 2d not 1d. |
| **B7: SWA** | ⚠️ **Careful** | Scheduler + compile interaction needs testing. 2d. |
| **B3: Label Smoothing** | ❌ **Already done** | Exists in `class_label_smoothing` — different epsilon values only |

### Phase 2 (Experimental Runs)

| Task | Verdict | Notes |
|------|---------|-------|
| **A3: Curriculum Learning** | ✅ **Sound but gate it** | Mandatory: validate `corrcoef(scores, label_density) < 0.4` first |
| **A2: Pre-Filter Agent** | ⏸️ **Blocked** | Needs SCsVulLyzer + LangGraph orchestration changes |
| **B8: Self-Paced Learning** | ⏸️ **Depends on A3** | Must batch loss computation (not per-sample loop) |

### Critical Dependencies Not in Codebase

| Dependency | Where Needed | Status |
|------------|-------------|--------|
| `secondAnalyzer` (SCsVulLyzer) | A1, A2 | **Not in pyproject.toml** |
| `py-solc-x` + solc binary | A1, A2 | Pulled by SCsVulLyzer |
| `optuna` | B2 | **Not in pyproject.toml** |

---

## Integration-Specific Findings

### `predictor.py` Integration Surface

- `_score_windowed()` (line 609): The natural calibrator hook — currently:
  ```python
  probs = torch.sigmoid(logits.float()).squeeze(0)
  return self._format_result(graph, probs, windows[0], n_real)
  ```
  Calibrator would slot between lines 638 and 639.

- `__init__` loading (line 303): Currently loads thresholds JSON. Temperature + calibrator would be additional sidecar files loaded here.

- MC Dropout (`predict_with_uncertainty`): New method alongside `predict_source()` and `predict_with_hotspots()`. No changes to existing methods.

### `trainer.py` Integration Surface

- Label adjacency: `compute_pos_weight()` (line 427) already iterates over `train_dataset._label_map` for pos_weight computation. The same labels can be used for `build_label_adjacency`.

- Curriculum sampler: The `_build_weighted_sampler` function (line 954) and its CLI integration at lines 1117-1124 would be replaced or wrapped.

- Checkpoint save: `trainer.py:2003-2026` — currently saves model, optimizer, scheduler, RNG states, thresholds. Would need SWA model state and temperature as additions.

### `sentinel_model.py` Integration Surface

- `LabelDependencyLayer`: Add after classifier (line 560: `logits = self.classifier(combined)`) — before the return at line 561. Only active when `label_adj` is passed to `__init__`.

- Forward pass signature: No changes to `forward(graphs, input_ids, attention_mask, return_aux)` — the label dependency layer operates within this unchanged signature.
