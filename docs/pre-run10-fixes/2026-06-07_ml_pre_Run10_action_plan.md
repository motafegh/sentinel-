# Pre-Run 10: Action Plan
**Date:** 2026-06-07
**Source docs:** `pre_run10_proposal.md` (original) + `pre_run10_proposal_audit_2026-06-07.md` (codebase verification)
**Current baseline:** Run 9 v11 best F1-macro = 0.2586 (lost in resume incident; current `.pt` is ep1 restart)
**Target:** validated macro-F1 > 0.35

---

## Phase Overview

| Phase | Who | What | Risk | Timeline |
|-------|-----|------|------|----------|
| **P0** | you | No-retraining fixes: Optuna thresholds, temperature scaling, predictor threshold wiring | Zero | Before Run 10 |
| **P1** | you | Next training run: B5 (label dep), B4 (R-Drop), B7 (SWA). Run 10. | Low | ~3 days |
| **P2** | you | Experimental: A3 (curriculum), B8 (self-paced) after correlation gate | Medium | Next sprint |
| **P3** | blocked | A1 (calibrator), A2 (pre-filter) — need SCsVulLyzer installed first | Zero model | TBD |

**Critical path:** P0 → P1 → P2. P3 runs in parallel but is currently blocked on `secondAnalyzer` dependency.

---

## Phase 0: Fix What's Broken Now (No Retraining)

**Goal:** Achieve the highest F1 possible from the current Run 9 checkpoint without any training.

### P0-A: Wire predictor to use tuned per-class thresholds [BUGFIX]

**What's wrong:** `predictor._format_result()` uses hardcoded `TIER_CONFIRMED_F1_SCORE = 0.55` instead of the per-class thresholds saved by `tune_threshold.py`. The tuned thresholds JSON is saved but never loaded.

**Action:**
1. `predictor.py:__init__` — load `*_thresholds.json` sidecar into `self.thresholds: dict`
2. `predictor.py:_format_result()` — replace `threshold = self.TIER_CONFIRMED_F1_SCORE` with `self.thresholds.get(class_name, 0.55)`
3. `predictor.py:_format_result()` — update three-tier logic to use per-class threshold: `confirmed` = `prob >= threshold`, `suspicious` = `prob >= threshold * 0.7` (or similar)

**Files:** `ml/src/inference/predictor.py`
**Effort:** 30 min
**Verification:** Re-run 20-contract manual test (`ml/scripts/manual_test/`). Compare tier assignments before/after.

---

### P0-B: Replace per-class grid with Optuna joint threshold search [ENHANCEMENT]

**What's wrong:** `tune_threshold.py` sweeps 19 candidates per class independently (linspace 0.05–0.95). No joint optimization across classes — misses class interaction effects.

**Action:**
1. Add `optuna` to `ml/pyproject.toml`: `optuna = "^3.6"`
2. Add `from optuna.samplers import TPESampler` to `tune_threshold.py`
3. Replace the `for vuln_idx, vuln_name in enumerate(VULN_CLASSES):` loop (lines ~226–246) with Optuna study:
   ```python
   study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
   study.optimize(lambda trial: objective(trial, val_probs, val_labels), n_trials=2000)
   ```
4. Keep old grid sweep as `--method grid` fallback
5. Save thresholds JSON in same format (keys: class names → threshold values)

**Files:** `ml/scripts/tune_threshold.py`, `ml/pyproject.toml`
**Effort:** 2h
**Verification:** Run on cached validation probabilities from Run 9. Compare tuned F1 against current independent grid results. Expected gain: +0.01–0.03 macro-F1.

---

### P0-C: Add temperature scaling to inference pipeline [ENHANCEMENT]

**What's wrong:** Raw logits go through `torch.sigmoid` directly. No calibration — overconfident on complex contracts (probs cluster 0.40–0.67 across 6–8 classes simultaneously).

**Action:**
1. New file `ml/src/inference/temperature_scaler.py` with:
   - `TemperatureScaler` class (single learnable param T, forward = logits / T)
   - `calibrate_temperature(model, val_loader, device)` — collects all val logits, optimizes T via L-BFGS on NLL
2. Add `compute_and_save_temperature(checkpoint_path, ...)` that loads checkpoint, runs val forward pass, saves `*_temperature.json`
3. `predictor.py:__init__` — load `*_temperature.json` into `self.temperature: float`
4. `predictor.py:_score_windowed()` — apply T before sigmoid: `probs = torch.sigmoid(logits.float() / self.temperature)`

**Files:** new `ml/src/inference/temperature_scaler.py`, `ml/src/inference/predictor.py`
**Effort:** 30 min compute + 30 min integration
**Verification:** Temperature > 1.0 means model was overconfident (expected). ECE should improve. No change to ranking — only probability magnitudes shift.

---

### P0-D: MC Dropout as optional inference mode [ENHANCEMENT]

**What's wrong:** Single forward pass gives no uncertainty estimate. FP explosion on complex contracts has no diagnostic signal.

**Action:**
1. `predictor.py` — add `predict_with_uncertainty(graphs, input_ids, mask, n_passes=30)` method:
   ```python
   self.model.train()  # enable dropout
   with torch.no_grad():
       for _ in range(n_passes):
           ...
   ```
2. Returns `{"probabilities": mean, "uncertainty": std, "needs_review": bool}`
3. API endpoint: optional `/predict/uncertainty` route (not default — 30× inference cost)
4. Integrate uncertainty flag into three-tier response: if `uncertainty[class_i] > 0.15` and probability above threshold, add `"uncertain": true` to tier output

**Files:** `ml/src/inference/predictor.py`, `ml/src/inference/api.py`
**Effort:** 4h
**Verification:** On safe contracts 12 and 19 (high FP count), uncertainty should be elevated for FP classes (CallToUnknown, Reentrancy). Uncertainty < 0.05 on ground-truth-relevant classes.

---

## Phase 1: Next Training Run (Run 10)

**Goal:** Implement model-level changes that require retraining. Target F1-macro > 0.30.

### P1-A: Label Dependency Graph [ENHANCEMENT]

**What:** Add `LabelDependencyLayer` after classifier. Pre-compute label co-occurrence matrix from training labels, use as fixed adjacency. Learnable transform via Linear(C, C) initialized to identity.

**Action:**
1. New class `LabelDependencyLayer(nn.Module)` in `sentinel_model.py` or new file `ml/src/models/label_dependency.py`
2. `build_label_adjacency(train_labels)` — compute conditional probability matrix from `train_labels: [N, C]`
3. `SentinelModel.__init__` — accept optional `label_adj: Tensor`, create `self.label_dep`
4. `SentinelModel.forward` — apply after classifier: `logits = self.label_dep(logits)`
5. `trainer.py:train()` — compute adjacency from `train_dataset._label_map`, pass to model constructor

**Files:** `ml/src/models/sentinel_model.py`, `ml/src/training/trainer.py`
**Effort:** 1d
**Risk:** Zero at init (identity passthrough). ~100 learnable params.
**Expected F1 gain:** +0.005–0.015 (per proposal)

---

### P1-B: R-Drop Regularization [ENHANCEMENT]

**What:** Two forward passes per batch with different dropout masks. KL divergence penalizes inconsistency.

**Action:**
1. `trainer.py:train_one_epoch()` — duplicate forward pass call (lines ~665–666):
   ```python
   logits_1, aux_1 = model(graphs, input_ids, attention_mask, return_aux=True)
   logits_2, aux_2 = model(graphs, input_ids, attention_mask, return_aux=True)
   ```
2. Compute KL divergence on sigmoid probabilities
3. Merge auxiliary losses (JK entropy, DoS weight masking) over both passes
4. Loss = `(task_loss_1 + task_loss_2) / 2 + alpha * kl_loss`
5. CLI flag: `--rdrop-alpha 0.5`

**Smoke test:** Run 1 epoch with `--rdrop-alpha 0.5` on RTX 3070. Check:
- Compile cache reuse (2nd forward should not recompile)
- No NaN in KL divergence (clamp log to -100)
- Epoch time: should be ~1.4× (not 2×) due to cached computations in compiled graph
- Val F1 not collapsed (within 0.02 of baseline)

**Files:** `ml/src/training/trainer.py`, `ml/scripts/train.py` (CLI argument)
**Effort:** 2d (was estimated 1d, but auxiliary head entanglement adds complexity)
**Risk:** Low — KL divergence is bounded. Start alpha=0.3 if 0.5 causes instability.
**Expected F1 gain:** +0.010–0.020 (per proposal)

---

### P1-C: SWA — Post-hoc Averaging [ENHANCEMENT]

**What:** Average the top-N checkpoints after training. Simpler than in-training SWA (no scheduler conflict with OneCycleLR).

**Action:**
1. New script `ml/scripts/swa_average.py`:
   - Load `GCB-P1-RunX-v10-YYYYMMDD_best.pt` + `_epoch_*.pt` intermediate checkpoints
   - Average top-5 state dicts by val F1
   - Run `update_bn` on full training set
   - Save as `*_swa.pt`
2. `trainer.py` — save intermediate checkpoints at threshold-tuning intervals (every 10 epochs) for SWA pool

**Alternative (in-training):** If compile interaction is clean:
1. `trainer.py:__init__` — wrap model in `AveragedModel`
2. `trainer.py:train()` — at each epoch end: `swa_model.update_parameters(model)`
3. After training: `update_bn(train_loader, swa_model)`, save alongside best checkpoint

**Files:** new `ml/scripts/swa_average.py`, `ml/src/training/trainer.py`
**Effort:** 1d (post-hoc) / 2d (in-training)
**Risk:** Low. Post-hoc approach has zero scheduler interaction.
**Expected F1 gain:** +0.005–0.010 (per proposal)

---

### P1-D: Label Smoothing — Tune Epsilon Values [HYPERPARAMETER]

**What:** Already implemented with different epsilon values and formulation. This is a search over values, not a code change.

**Action:**
1. Compare current epsilon values vs proposal values in a sweep (3 runs):
   - Run A: current codebase values (current baseline)
   - Run B: proposal values (`IntegerUO: 0.03, ...`)
   - Run C: asymmetric formulation (`1 - labels) * eps`)
2. Test on validation F1 after 30 epochs. If no significant difference, keep current values.

**Files:** `ml/src/training/trainer.py` (default dict change only)
**Effort:** 1h code + 3 full training runs (~3 days)
**Priority:** **Low** — the mechanism is already active. Epsilon tuning is diminishing returns until the architecture is stable.

---

### P1-E: Run 10 Configuration

**Proposed Run 10 config:**
```
--run-id GCB-P1-Run10-v11-202606XX
--model four-eye
--vuln-classes 10
--gnn-layers 8 --gnn-hidden-dim 256 --gnn-phase-heatup 3
--jk-type cat --jk-agg-max false
--jk-entropy-reg-lambda 0.005
--prefix-k 48 --gnn-prefix-warmup-epochs 15
--gnn-dropout 0.2
--transformer-name microsoft/graphcodebert-base --lora-r 16 --lora-alpha 32
--fusion-dim 256
--lr 1e-4 --weight-decay 0.01
--epochs 80 --patience 30
--gradient-accumulation-steps 8 --batch-size 8
--loss asl --asl-gamma-negative 4 --asl-gamma-positive 0
--aux-heads --num-aux-categories 4 --aux-ramp-epochs 8 --aux-final-weight 0.30
--label-dependency  (NEW: P1-A)
--rdrop-alpha 0.5  (NEW: P1-B)
--swa-start 0.7    (NEW: P1-C, if in-training)
--weighted-sampler timestamp-size
--drop-complexity-feature true
--threshold-tune-interval 10
```

---

## Phase 2: Experimental Runs (Not Yet Scheduled)

### P2-A: Curriculum Sampler (A3)

**Gate condition:** Compute `np.corrcoef(complexity_scores, label_density)` on training set. Must be `|rho| < 0.4`.

**Action if gate passes:**
1. New file `ml/src/training/curriculum_sampler.py` with:
   - `compute_complexity_score(graph_data)` — structural difficulty measure
   - `CurriculumSampler(DataLoader.Sampler)` — progressive complexity filtering
2. `trainer.py` — replace `WeightedRandomSampler` when `--curriculum-sampler` flag is active
3. Mixing floor: 15% hard samples always included
4. CLI: `--curriculum-sampler --complexity-progress-n-epochs 40`

**Action if gate fails:**
- Residualize complexity scores against label density (LinearRegression)
- Re-check `|rho| < 0.4` with decorrelated scores
- If still fails: do not deploy. Complexity correlates too strongly with labels in this dataset.

**Files:** new `ml/src/training/curriculum_sampler.py`, `ml/src/training/trainer.py`
**Effort:** 2d
**Risk:** Medium — re-introduces the shortcut you removed with `drop_complexity_feature` if correlation gate is not respected.

---

### P2-B: Self-Paced Learning (B8)

**Dependency:** P2-A must be implemented first (extends `CurriculumSampler`).

**Action:**
1. `SelfPacedSampler(CurriculumSampler)` — replaces complexity-based ranking with per-sample loss
2. After each epoch, batch-compute per-sample losses (NOT per-sample loop as in proposal)
3. Threshold λ grows linearly: `λ = 2.0 * (1 + epoch * 0.05)`
4. Mixing floor: 10% highest-loss samples always included

**Risk:** Per-sample loss computation doubles epoch time (~35 min → ~70 min). Consider:
- Compute only every 5 epochs (save 80% cost)
- Or skip until architecture is stable

**Files:** `ml/src/training/curriculum_sampler.py`
**Effort:** 1d on top of A3

---

## Phase 3: SCsVulLyzer Integration (Blocked)

### Blocked on: `secondAnalyzer` dependency

**Current status:** `secondAnalyzer` is NOT in `ml/pyproject.toml` or `agents/pyproject.toml`. No solc binary installed in WSL. Feature name alignment between SCsVulLyzer output and proposal rules is unverified.

**Unblocking checklist:**
1. [ ] Determine `secondAnalyzer` availability (pip package? git repo? internal?)
2. [ ] Install solc binary in WSL: `pip install py-solc-x` then `solcx.install_solc('0.8.28')`
3. [ ] Add `secondAnalyzer` to relevant `pyproject.toml`
4. [ ] Smoke test: run `analyze_solidity_contract("test.sol")` and inspect output dict keys
5. [ ] Map actual keys to proposal's `IMPOSSIBILITY_RULES` feature names — if they differ, update rules
6. [ ] Profile: time breakdown for solc compile vs feature extraction vs full inference with calibrator

### P3-A: Static Calibrator (A1)

**When unblocked:**
1. New file `ml/src/inference/static_calibrator.py`
2. `IMPOSSIBILITY_RULES` dict + `ATTENUATION_RULES` dict (per audit: verify feature names first)
3. `calibrate(contract_path, sentinel_probs, thresholds, vuln_classes) -> Tensor`
4. Wire into `predictor.py` as optional post-processing step
5. CLI flag or env var to enable/disable

### P3-B: Pre-Filter MCP Agent (A2)

**When unblocked:**
1. New file `agents/src/mcp/servers/static_analyzer_server.py` (following `inference_server.py` pattern)
2. `compute_risk_score(features)` + risk threshold calibration
3. Update LangGraph orchestration to route through pre-filter
4. Test with safe contracts: verify `route_to_sentinel = False` for trivially-safe contracts

---

## Dependency Graph

```
P0-A (predictor threshold fix) ─┐
P0-B (Optuna) ──────────────────┤
P0-C (temperature) ─────────────┤── all independent, no ordering
P0-D (MC dropout) ──────────────┘

P1-A (label dep graph) ─────────┐
P1-B (R-Drop) ──────────────────┤── all independent, same training run
P1-C (SWA) ─────────────────────┤
P1-D (label smoothing tune) ────┘

P2-A (curriculum) ──────────────┐── P2-B depends on P2-A
P2-B (self-paced) ──────────────┘

P3-A (calibrator) ──────────────┐── SCsVulLyzer dep. Independent of P0/P1/P2
P3-B (pre-filter agent) ────────┘
```

---

## Run 10 Training Schedule

| Day | Task | Output |
|-----|------|--------|
| Day 1 (P0) | Fix predictor thresholds + Optuna search + temperature | Tuned checkpoint JSONs. Re-run manual test. |
| Day 1 (P1 code) | Implement label dep graph + R-Drop + SWA | Code ready for training |
| Day 2–4 (P1 train) | Launch Run 10 with P1 changes | ~60 epochs = ~35h |
| Day 4 (P1 eval) | Evaluate Run 10 best checkpoint. Run Optuna again. | Final F1 for this phase |
| Day 5+ (P2) | Gate check for curriculum. Launch P2 if pass. | Experimental F1 |

---

## What Success Looks Like

| Metric | Current (Run 9 best) | Run 10 target | Stretch |
|--------|---------------------|---------------|---------|
| Validation macro-F1 (fixed threshold 0.55) | 0.2586 (ep14, lost) | > 0.30 | > 0.32 |
| Validation macro-F1 (tuned thresholds) | 0.2851 (v8 baseline) | > 0.33 | > 0.35 |
| Manual test FP count (20 contracts) | ~6–8 on 09/11/20 | < 4 on worst contract | < 2 |
| GasException recall | 0.0 (not detected) | > 0.10 | > 0.20 |
| TOD recall | 0.0 (not detected) | > 0.10 | > 0.20 |
| Uncertainty: FP classes above threshold | N/A | Uncertainty > 0.15 for FP predictions | Consistent signal |
| Calibration (ECE) | Unknown (no calibration) | ECE < 0.10 | ECE < 0.05 |

---

## Checklist Summary

### Now (P0):
- [ ] `predictor.py`: Load and apply per-class thresholds (not hardcoded 0.55)
- [ ] `tune_threshold.py`: Add Optuna joint search (`pip install optuna`)
- [ ] `temperature_scaler.py`: New file, compute T, wire into predictor
- [ ] `predictor.py`: Add `predict_with_uncertainty()` as optional method
- [ ] Re-run manual test with all P0 changes active

### Next Run (P1):
- [ ] `sentinel_model.py`: Add `LabelDependencyLayer`
- [ ] `trainer.py`: R-Drop dual-forward + KL divergence
- [ ] `trainer.py` or new script: SWA (post-hoc or in-training)
- [ ] Launch Run 10 with P1 config

### Future (P2):
- [ ] Validate complexity-label correlation on training set
- [ ] `curriculum_sampler.py`: Implementation (only if gate passes)
- [ ] `curriculum_sampler.py`: Self-paced extension (only after A3 validated)

### Blocked (P3):
- [ ] Install SCsVulLyzer + solc
- [ ] Verify feature name alignment
- [ ] `static_calibrator.py`: Implementation
- [ ] `agents/static_analyzer_server.py`: Pre-filter MCP agent
