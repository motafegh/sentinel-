# Pre-Run 10: Final Enhancement Proposal (Corrected)
**Date:** 2026-06-07  
**Supersedes:** `pre_run10_proposal.md` (original) and `pre_run10_action_plan.md` (first draft)  
**Source verification:** Full cross-check against `predictor.py`, `trainer.py`, `train.py`, `sentinel_model.py`  
**Peer review corrections:** 8 items flagged and fixed (see §Change Log)  
**Current baseline:** Run 9 v11 (in-flight, PID 3362523, best ep14 F1=0.2586, ~20-27 epochs remaining)  
**Target:** validated macro-F1 > 0.35  

---

## Change Log (vs First Draft Action Plan)

| # | Issue | First draft said | Corrected |
|---|-------|-----------------|-----------|
| 1 | P0-A threshold bug | "thresholds not loaded from JSON" | Thresholds ARE loaded (predictor.py:298-336). Bug is `_format_result()` never uses `self.thresholds[i]` in tiering loop (line 712) — uses global 0.55 for every class. |
| 2 | P1-E flag `--prefix-k` | `--prefix-k 48` | `--gnn-prefix-k 48` (train.py:210) |
| 3 | P1-E flag `--drop-complexity-feature true` | `--drop-complexity-feature true` | `--drop-complexity-feature` (train.py:220, `store_true`) |
| 4 | P1-E non-existent flags | `--swa-start 0.7`, `--label-dependency`, `--rdrop-alpha` | These don't exist yet — they must be implemented in P1 code BEFORE the Run 10 config is valid |
| 5 | Run 9 lifecycle | Launched into Run 10 without deciding Run 9 fate | Run 9 is healthy and running to ep60-65 (~20-27 more epochs, ~16-22h). P0 work slots into this window naturally. Decision: let Run 9 finish; use its final checkpoint as P0-B/P0-C baseline. |
| 6 | BCCC Phase 4 Stage 1 | Not mentioned | 780/~10,000 Slither done, 20/~10,000 Aderyn. Run 10 uses v10 data (explicit decision: validate arch changes first, BCCC for Run 11). |
| 7 | Post-hoc SWA needs intermediate checkpoints | Said "save every 10 epochs" but didn't account for it | Trainer currently only saves best. Intermediate save logic is prerequisite for SWA — adds ~1h to P1-C estimate. |
| 8 | Run 8 flags omitted from Run 10 config | Silent omission | `--appnp-alpha 0.2` and `--fusion-lr-multiplier 0.3` were Run 8 flags. Run 10 uses defaults (0.0 and 0.5) — rationale: Run 8's 0.2851 was below Run 7's 0.3074, so these may have been harmful. State this explicitly. |
| 9 | Curriculum gate analysis | Embedded in 2d estimate | Separate: 2-4h gate analysis (load 29K graphs, compute scores, check correlation) before committing to P2 implementation. |
| 10 | R-Drop + grad accum interaction | Not mentioned | 8 grad accum steps × 2 forward passes = 16 micro-batches per optimizer step. Start `--rdrop-alpha 0.3` (not 0.5), warm up over 5 epochs. |
| 11 | Three-tier redesign needed | Not mentioned | With per-class thresholds for CONFIRMED, the SUSPICIOUS tier boundary is ambiguous. Needs design decision: per-class suspicious thresholds, or decouple tier display from threshold-based binary call? |

---

## §0 — Run 9 Status and Implications

**Run 9 is actively training** (PID 3362523, v11 schema). Trajectory:
- Best: ep14 F1=0.2586 (checkpoint lost in lambda-typo resume incident)
- Current: resumed from ep1 restart, ~ep35-40 now, patience=30 from ep35
- Projected: runs to ep60-65 (~20-27 more epochs, ~16-22h from now, terminating ~2026-06-08 evening)
- Speed: 38→48 min/ep (+25%) over 15+ epochs — likely thermal throttle on RTX 3070

**What Run 9's trajectory tells us:**
- Tuned F1 peaked at ep20 then declined while fixed F1 kept improving → **probability miscalibration drift during training**. Raw predictions improve but optimal thresholds shift epoch-by-epoch. Directly validates:
  - P0-C (temperature scaling) — T > 1.0 expected
  - P0-B (Optuna joint search) — per-class grid can't track interaction effects causing this drift
- Speed drop from 38→48 min/ep means Run 10's 80-epoch target is ~64h, not ~47h on same hardware

**Decision: Let Run 9 finish.** P0 implementation work happens during the remaining ~20h. P0-B and P0-C should target Run 9's final best `.pt` when it terminates.

---

## §1 — Phase 0: No-Retraining Fixes (This Week)

### P0-A: Fix Three-Tier Predictor Thresholding [BUGFIX]

**What the bug actually is:**

Thresholds ARE loaded from JSON at `predictor.py:298-336`:
```python
self.thresholds = torch.tensor(per_class_thresholds, ...)  # [10] per-class values
self.thresholds_loaded = True
```

But `_format_result()` at line 698-715 never uses them:
```python
conf_thr = self.tier_confirmed_threshold   # global 0.55 ← BUG
susp_thr = self.tier_suspicious_threshold  # global 0.25
for cls_name, prob in zip(self._class_names, probs_list):
    if prob >= conf_thr:       # ← should be self.thresholds[i]
        confirmed.append(...)
    elif prob >= susp_thr:     # ← should use per-class suspicious threshold
        suspicious.append(...)
```

**Fix:**
1. In the tiering loop, replace `conf_thr` with `self.thresholds[i]` for the CONFIRMED tier
2. Replace `susp_thr` with `self.thresholds[i] * 0.7` (or add a per-class suspicious threshold multiplier) for SUSPICIOUS tier
3. Add `tier_confirmed_threshold` override to `__init__` defaults (currently hardcoded; make configurable)

**Three-tier redesign required:**
- `self.thresholds[i]` sets CONFIRMED boundary per-class
- SUSPICIOUS needs a rule: `prob >= self.thresholds[i] * suspicious_multiplier` where `suspicious_multiplier` is a new config param (default 0.7)
- Or: add a second per-class thresholds dict for SUSPICIOUS (loaded from JSON if present)
- NOTEWORTHY tier remains static at 0.10 (unchanged)

**Files:** `ml/src/inference/predictor.py`  
**Effort:** 1h  
**Verification:** Re-run 20-contract manual test. Compare tier assignments before/after. Expected: contracts 12 and 19 (safe) stop showing CONFIRMED CallToUnknown/Reentrancy.

---

### P0-B: Optuna Joint Threshold Search [ENHANCEMENT]

**Action:**
1. `pip install optuna` (add to `ml/pyproject.toml`: `optuna = "^3.6"`)
2. `ml/scripts/tune_threshold.py` — add `--method {grid,optuna}` flag, default `optuna`
3. Grid sweep kept as `--method grid` fallback
4. Optuna: 2000 trials, TPESampler, joint 10-dim search over [0.05, 0.95]
5. Output format unchanged (`_thresholds.json` with `{class: threshold}` under `thresholds` key)

**Note:** Run P0-A first. The Optuna search optimizes over the same decision boundary that P0-A fixes — calibrating thresholds before the predictor uses them creates a coherent measurement loop.

**Files:** `ml/scripts/tune_threshold.py`, `ml/pyproject.toml`  
**Effort:** 2h  
**Target:** Run on Run 9's best `.pt` when it terminates.

---

### P0-C: Temperature Scaling [ENHANCEMENT]

**Action:**
1. New file `ml/src/inference/temperature_scaler.py`:
   - `TemperatureScaler(nn.Module)` — single param `T`, forward = `logits / T.clamp(min=0.1)`
   - `calibrate_temperature(model, val_loader, device)` — collect val logits, L-BFGS optimize T on NLL
2. `predictor.py:__init__` — load `*_temperature.json` into `self.temperature`
3. `predictor.py:_score_windowed()` — `probs = sigmoid(logits / self.temperature)`

**Files:** new `ml/src/inference/temperature_scaler.py`, `ml/src/inference/predictor.py`  
**Effort:** 1h implementation + 1h compute (one val forward pass)  
**Expected:** T > 1.0 (model is overconfident). No ranking change, just calibration.

---

### P0-D: MC Dropout Uncertainty [ENHANCEMENT]

**Action:**
1. `predictor.py` — add `predict_with_uncertainty(graphs, input_ids, mask, n_passes=30)`:
   ```python
   self.model.train()
   with torch.no_grad():
       for _ in range(n_passes):
           logits, _ = self.model(batch)
           probs = torch.sigmoid(logits)
   ```
2. Returns `{"probabilities": mean, "uncertainty": std, "needs_review": bool}`
3. API: optional `/predict/uncertainty` endpoint (not default — 30× inference cost)
4. Flag `uncertain: true` where `uncertainty[i] > 0.15` and class is CONFIRMED

**Files:** `ml/src/inference/predictor.py`, `ml/src/inference/api.py`  
**Effort:** 4h  
**Note:** 30× cost means ~450s per contract on RTX 3070. Batch/offline only.

---

## §2 — Three-Tier Redesign (Cross-Cutting)

The P0-A fix forces a design decision about the three-tier system:

**Current tiers (broken):**
```
CONFIRMED:  prob >= 0.55  (global, same for all classes)
SUSPICIOUS: prob >= 0.25  (global)
NOTEWORTHY: prob >= 0.10  (static)
```

**With per-class thresholds, options:**

| Approach | CONFIRMED | SUSPICIOUS | NOTEWORTHY | Complexity |
|----------|-----------|------------|------------|------------|
| A — Scaled | `prob >= T_i` | `prob >= T_i * 0.7` | `prob >= 0.10` | Low — one config param `suspicious_multiplier = 0.7` |
| B — Dual threshold dict | `prob >= T_confirmed[i]` | `prob >= T_suspicious[i]` | `prob >= 0.10` | Medium — second JSON key in thresholds file |
| C — Decouple | `prob >= T_i` (binary call) | Tiers use separate per-class thresholds from JSON | `prob >= 0.10` | Cleanest — but needs schema change in `*_thresholds.json` |

**Recommendation: Approach A** for now (least code, one config addition). B can be implemented later. C is the cleanest but changes the threshold file format — defer until P0-B is ported to output the extended format.

---

## §3 — Phase 1: Next Training Run (Run 10)

### P1-A: Label Dependency Graph

**Action:**
1. `sentinel_model.py` — add `LabelDependencyLayer(nn.Module)`:
   - `register_buffer("adj", label_adj)` — [C, C] conditional probability matrix
   - `self.W = nn.Linear(C, C, bias=False)`, init `eye_`
   - `forward(logits)`: `logits + 0.1 * self.W(logits @ self.adj.T)`
2. `trainer.py:train()` — precompute `build_label_adjacency` from `train_dataset._label_map`, pass to model constructor
3. CLI: `--label-dependency` (new flag, `store_true`, default `False`)

**Files:** `ml/src/models/sentinel_model.py`, `ml/src/training/trainer.py`, `ml/scripts/train.py`  
**Effort:** 1d  
**Risk:** Zero at init (identity passthrough with 0.1 residual). ~100 parameters.

---

### P1-B: R-Drop Regularization

**Action:**
1. `trainer.py:train_one_epoch()` — two forward passes:
   ```python
   logits_1, aux_1 = model(graphs, input_ids, attention_mask, return_aux=True)
   logits_2, aux_2 = model(graphs, input_ids, attention_mask, return_aux=True)
   ```
2. KL divergence on sigmoid probs (symmetric)
3. Merge aux losses (JK entropy, DoS masking) over both passes
4. Loss = `(task_loss_1 + task_loss_2) / 2 + alpha * kl_loss`
5. CLI: `--rdrop-alpha 0.3` (start at 0.3, warm up from 0.0 over 5 epochs)
6. At `--gradient-accumulation-steps 8`: 16 micro-batches per optimizer step. Gradients are noisier early on — alpha=0.5 risks destabilizing. Warmup: `alpha_effective = alpha * min(1, epoch / 5)`.

**Interaction with compile:** Test with `torch.compile` enabled. Two forward passes with different dropout masks should NOT cause recompilation (dropout is dynamic control flow, not a graph shape change). If recompilation occurs, wrap the forward loop:
```python
for _ in range(2):
    logits, aux = model(graphs, ...)
    logits_list.append(logits)
```
This keeps both passes inside the same compiled-graph trace.

**Files:** `ml/src/training/trainer.py`, `ml/scripts/train.py`  
**Effort:** 2d (more than 1d due to aux head entanglement + compile interaction)  
**Smoke test required:** 1 epoch on Run 10 config. Check: no NaN, compile cache reused, epoch time ≤ 1.5× baseline.

---

### P1-C: SWA

**Two approaches:**

**A — Post-hoc (recommended):**
1. Add intermediate checkpoint save at `--threshold-tune-interval` epochs (currently trainer only saves best)
2. New script `ml/scripts/swa_average.py`:
   - Load top-5 intermediate `.pt` checkpoints by val F1
   - Average state dicts
   - `update_bn(train_loader, swa_model)` — one full pass (~35min)
   - Save as `*_swa.pt`

**B — In-training:**
1. `trainer.py:__init__` — `AveragedModel(model)` + `SWALR`
2. Requires scheduler refactoring — OneCycleLR × SWALR coexistence
3. More fragile; higher effort

**Recommendation: Post-hoc.** Avoids scheduler conflict. Only requires adding intermediate checkpoint saving to the trainer, which is reusable for model analysis anyway.

**Intermediate checkpoint save cost:** ~2.5GB per checkpoint × 8 saves (every 10 epochs up to 80) = 20GB disk. Acceptable on a 1TB drive. Add `--save-intermediate` flag, default `True` when `--swa-post-hoc` is set.

**Files:** new `ml/scripts/swa_average.py`, `ml/src/training/trainer.py` (intermediate save logic), `ml/scripts/train.py`  
**Effort:** 1d (post-hoc) / 2d (in-training)

---

### P1-D: Label Smoothing — Tuning Only

Already implemented (trainer.py:659-662). If desired:
- Compare current epsilon values vs proposal values in a sweep
- Current formulation: `labels * (1 - ε) + 0.5 * ε` (bidirectional, push toward 0.5)
- Proposal: `labels * (1 - ε) + (1 - labels) * ε` (asymmetric)
- Run A: current values → Run B: proposal values → compare val F1

**Effort:** 1h code, 3 full training runs (~3 days). **Defer until architecture is stable.**

---

### P1-E: Run 10 Config (Corrected)

```bash
poetry run python ml/scripts/train.py \
    --run-id GCB-P1-Run10-v11-202606XX \
    --model four-eye \
    --vuln-classes 10 \
    --gnn-layers 8 --gnn-hidden-dim 256 --gnn-phase-heatup 3 \
    --jk-type cat --jk-agg-max false \
    --jk-entropy-reg-lambda 0.005 \
    --gnn-prefix-k 48 --gnn-prefix-warmup-epochs 15 \
    --gnn-dropout 0.2 \
    --transformer-name microsoft/graphcodebert-base --lora-r 16 --lora-alpha 32 \
    --fusion-dim 256 \
    --lr 1e-4 --weight-decay 0.01 \
    --epochs 80 --patience 30 \
    --gradient-accumulation-steps 8 --batch-size 8 \
    --loss asl --asl-gamma-negative 4 --asl-gamma-positive 0 \
    --aux-loss-weight 0.3 --aux-phase2-loss-weight 0.20 \
    --num-aux-categories 4 --aux-ramp-epochs 8 --aux-final-weight 0.30 \
    --drop-complexity-feature \
    --weighted-sampler timestamp-size \
    --threshold-tune-interval 10 \
    --compile \
    --appnp-alpha 0.0 \          # Run 8 used 0.2; Run 10 resets to 0.0 (rationale: may have hurt F1)
    --fusion-lr-multiplier 0.5 \  # Run 8 used 0.3; Run 10 uses v7 default 0.5
    # NEW P1 flags (must be implemented first):
    # --label-dependency \
    # --rdrop-alpha 0.3 \
    # --save-intermediate \
```

**Rationale for dropping Run 8's `--appnp-alpha 0.2`:**
- Run 8 tuned F1 (0.2851) was below Run 7 (0.3074) which did NOT use APPNP
- APPNP teleport may have diluted Phase 2 structural signal — consistent with H1 finding ("Phase 2 multi-edge dilution hurts Reentrancy −0.017 F1")
- Reset to 0.0 (disabled). If Run 10 undershoots expectations, re-enable as ablation.

**Rationale for default `--fusion-lr-multiplier 0.5`:**
- Run 8's 0.3 was an experiment. Run 7's 0.5 achieved higher F1.
- No evidence 0.3 improves — revert to working value.

---

## §4 — Phase 2: Experimental (Deferred)

### P2-A: Curriculum Gate Analysis (Do First, Before Any Implementation)

**Independent 2-4h analysis — NOT part of P2 implementation estimate:**

1. Load all 29,103 training graphs from cache (~2 min)
2. For each graph: `compute_complexity_score(graph)` = weighted sum of:
   - `log1p(num_nodes) × 0.4`
   - `log1p(num_edges) × 0.3`
   - `log1p(num_functions) × 0.2`
   - `edge_diversity × 0.1`
3. Compute `label_density = sum(y) / C` per sample (class-agnostic positive fraction)
4. `rho = np.corrcoef(scores, label_density)`

**If `|rho| < 0.4`:**
- Implement `CurriculumSampler` in `ml/src/training/curriculum_sampler.py`
- Mixing floor: 15% hard samples always included
- Add `--curriculum-sampler` flag to train.py
- Run as ablation: Run 10 config ± curriculum → compare val F1

**If `|rho| >= 0.4`:**
- Attempt decorrelation via residualization: `scores_decorr = scores - reg.predict(label_density)`
- Re-check `|rho| < 0.4`. If still fails: do NOT deploy — complexity is a label proxy.
- This scenario means the `drop_complexity_feature` removal of feat[5] was insufficient — complexity still leaks through num_nodes/num_edges.

---

### P2-B: Self-Paced Learning

Depends on P2-A. Do NOT implement before curriculum is validated.
- Extends `CurriculumSampler` to `SelfPacedSampler`
- Per-sample loss computed in **batches** (NOT per-sample loop as in original proposal)
- Computing every 5 epochs (not every epoch) to avoid 2× time cost

---

## §5 — Phase 3: SCsVulLyzer Integration (Blocked)

### Dependency Status
- `secondAnalyzer` NOT in `ml/pyproject.toml` or `agents/pyproject.toml`
- No solc binary in WSL
- Feature name keys from `analyze_solidity_contract()` UNVERIFIED against proposal rules

### Unblocking Checklist
1. [ ] Determine `secondAnalyzer` source (pip/git/internal)
2. [ ] Install solc: `pip install py-solc-x && python -c "import solcx; solcx.install_solc('0.8.28')"`
3. [ ] Add `secondAnalyzer` to relevant `pyproject.toml`
4. [ ] Smoke test: run `analyze_solidity_contract()` on a test .sol, inspect output dict
5. [ ] Map actual keys to `IMPOSSIBILITY_RULES` in A1 — if keys differ, update rules
6. [ ] Profile: solc compile vs feature extraction vs GPU inference

### P3-A: Static Calibrator
New file `ml/src/inference/static_calibrator.py`. Inserted in predictor between sigmoid and tiering.

### P3-B: Pre-Filter MCP Agent
New agent following `inference_server.py` pattern. Requires LangGraph orchestration update.

---

## §6 — BCCC Phase 4 Stage 1 Status

**Current:**
- Slither: 780/~10,000 sampled contracts done (~8%)
- Aderyn: 20/~10,000 done (~0.2%)
- 67,311 total cleaned contracts (D-I-11/12 applied) — NOT ready for training
- 3-way consensus (Slither + Aderyn + manual rules) needed for D-P3-10 decision → Aderyn lag is a bottleneck

**Decision: Run 10 uses v10 data.** Rationale:
- Architecture changes (label dependency, R-Drop, SWA) must be validated on the current data distribution before switching
- BCCC changes both distribution and label schema — conflating architecture change with data change makes ablation impossible
- Run 11 targets BCCC data, using Run 10's best config as starting point

**Risk:** If BCCC Stage 1 finishes during Run 10 training (~3-4 days), there's a temptation to abort Run 10 mid-way and switch. Resist this — let Run 10 complete its 80 epochs so there's a clean baseline for Run 11.

---

## §7 — Missing Items (Not in Original Proposal or First Draft)

### Optuna Hyperparameter Search (beyond thresholds)
The proposal mentions AutoML for Phase 3. But with BCCC data incoming and Run 10 training potentially unstable, an early Optuna sweep over key hyperparameters would provide a stable starting point. Suggested sweep (15-epoch proxy runs on v10 validation split):

| Hyperparameter | Range | Default | Rationale |
|---------------|-------|---------|-----------|
| `lora_r` | [8, 16, 32] | 16 | LoRA rank — 32 may be unnecessary for 124M frozen CodeBERT |
| `gnn_hidden_dim` | [128, 256, 384] | 256 | 384 may cause VRAM OOM on RTX 3070 8GB |
| `asl_gamma_neg` | [2, 4, 6] | 4 | 4 caused all-zeros collapse in Run 3; 6 may be worse |
| `rdrop_alpha` | [0.1, 0.3, 0.5] | 0.3 | Starting range for P1-B |

Each trial = 15 epochs on v10 validation (no holdout split needed for relative comparison). ~3h per trial × 27 combos = ~81h on single GPU. Filter to fractional factorial design: run 9 combos covering main effects plus R-Drop × LoRA interaction. Estimated: ~27h.

**Not blocking.** Run when GPU is idle between Run 10 and Run 11.

### Aderyn Stage 1 Lag
Aderyn is at 20/~10,000 (0.2%) vs Slither at 780/~10,000 (8%). At this rate, Aderyn will be the bottleneck for 3-way consensus. If D-P3-10 requires 3-way consensus before BCCC training:
- 3-way labels available for ~0.2% of BCCC = ~135 contracts
- 2-way consensus (Slither + Aderyn when available, Slither-only fallback) covers ~8% = ~5,400
- If D-P3-10 can accept 2-way consensus with Aderyn prioritized for edge cases, BCCC readiness is determined by Slither, not Aderyn

**Recommendation:** Clarify whether 3-way consensus is required for D-P3-10 or 2-way is acceptable. This determines whether Aderyn speed is a blocking factor.

---

## §8 — Integration Map & Dependency Graph

```
P0 ORDERING:
  P0-A (predictor tier fix) ──────────────── must be first (fixes tiering before 
                                                     any threshold tuning)
  P0-B (Optuna threshold search) ─────────── run after P0-A, target Run 9 best .pt
  P0-C (temperature scaling) ─────────────── independent, any order
  P0-D (MC dropout) ──────────────────────── independent, any order
  Three-tier redesign decision ────────────── after P0-A (needs design conversation)

P1 TRAINING RUN (Run 10):
  P1-A (label dep graph) ────────────────── code: before Run 10 launch
  P1-B (R-Drop) ─────────────────────────── code: before Run 10 launch
  P1-C (SWA) ────────────────────────────── code: intermediate save logic before Run 10
  Run 10 launch ─────────────────────────── after P1 A/B/C code ready
  P1-D (label smoothing tune) ───────────── optional, after Run 10 baseline known

P2 (deferred):
  P2-A gate analysis ────────────────────── 2-4h compute, independent. Do before Run 10 ends.
  P2-A implementation ───────────────────── only if |rho| < 0.4
  P2-B self-paced ───────────────────────── only after P2-A validated

P3 (blocked):
  SCsVulLyzer install + verify ──────────── unblocking prerequisite
  P3-A calibrator ───────────────────────── after dependency resolved
  P3-B pre-filter agent ─────────────────── after P3-A
```

---

## §9 — Timeline (Updated for Run 9 In-Flight)

| Window | Task | Output |
|--------|------|--------|
| **Today (Jun 7)** | Verify all friend's corrections against source | ✅ Done (this document) |
| **Jun 7-8 (Run 9 running)** | P0-A: Fix predictor tier thresholds | Code + re-run manual test |
| | P0-B: Optuna implementation (code, not run) | Code ready, waits for Run 9 .pt |
| | P0-C: Temperature scaler code | Code ready, waits for Run 9 .pt |
| | P0-D: MC dropout code | Code ready |
| | P2-A gate analysis | `compute_complexity_scores` + correlation report |
| | Three-tier redesign decision | Document decision (approach A/B/C) |
| **Jun 8 eve (~Run 9 end)** | P0-B: Run Optuna on Run 9 best .pt | `*_thresholds.json` (Optuna) |
| | P0-C: Compute T on Run 9 best .pt | `*_temperature.json` |
| **Jun 9** | P1-A: Label dependency graph code | Code + PR |
| | P1-B: R-Drop code + smoke test | Code + 1-epoch smoke result |
| | P1-C: Intermediate checkpoint save | Code + PR |
| **Jun 10** | Run 10 launch | ~60-80 epochs, ~47-64h |
| | Run P2-A curriculum ablation in parallel if passes gate | Extra run, separate output |
| **Jun 12-13 (~Run 10 end)** | Evaluate Run 10. Run Optuna + T on Run 10 best. | Final F1 for P1 |
| **After Run 10** | Decide: Run 11 with BCCC, or P2 experimental first | Planning doc |

---

## §10 — Checklist

### Now (P0 code, while Run 9 runs):
- [ ] P0-A: Fix `_format_result()` to use `self.thresholds[i]` per-class (not global 0.55)
- [ ] P0-A: Design three-tier approach with per-class CONFIRMED and scaled SUSPICIOUS
- [ ] P0-B: Add `optuna` to pyproject.toml + implement joint search in `tune_threshold.py`
- [ ] P0-C: `temperature_scaler.py` — `calibrate_temperature()` function
- [ ] P0-C: Wire T into `predictor.py:_score_windowed()`
- [ ] P0-D: `predict_with_uncertainty()` method + optional API endpoint
- [ ] P2-A: Run complexity-label correlation gate analysis on training set

### Next Run (P1 code, before Run 10 launch):
- [ ] P1-A: `LabelDependencyLayer` in `sentinel_model.py` + `--label-dependency` CLI flag
- [ ] P1-B: Dual-forward + KL divergence in `trainer.py` + `--rdrop-alpha` CLI flag
- [ ] P1-C: Intermediate checkpoint save logic (needed for post-hoc SWA)
- [ ] P1-C: `ml/scripts/swa_average.py` post-hoc averaging script
- [ ] Fix P1-E config flags: correct `--gnn-prefix-k`, `--drop-complexity-feature`, remove non-existent flags
- [ ] Add `--appnp-alpha 0.0` and `--fusion-lr-multiplier 0.5` explicitly to Run 10 config (document rationale)
- [ ] P1-B smoke test: 1 epoch with R-Drop, verify compile + no NaN

### Deferred:
- [ ] P1-D: Label smoothing epsilon sweep (only if architecture is stable after Run 10)
- [ ] P2-B: Self-paced learning (only after P2-A validated)
- [ ] P3: SCsVulLyzer dependency setup
- [ ] Optuna hyperparameter sweep over lora_r, gnn_hidden_dim, asl_gamma_neg (GPU idle time)
- [ ] Clarify D-P3-10 requirement: 2-way or 3-way consensus for BCCC labels?

---

## Appendix A: Source Verification Notes

| Claim | File:Lines | Verdict |
|-------|-----------|---------|
| Thresholds ARE loaded from JSON | `predictor.py:298-336` | ✅ True — `self.thresholds` tensor built correctly |
| Thresholds NOT used in tiering | `predictor.py:698-715` | ✅ True — `conf_thr = self.tier_confirmed_threshold` used for all classes |
| `--gnn-prefix-k` not `--prefix-k` | `train.py:210` | ✅ Correct |
| `--drop-complexity-feature` is `store_true` | `train.py:220-222` | ✅ Correct — no value argument |
| No `--label-dependency` CLI flag | `train.py` (full scan) | ✅ Correct — doesn't exist yet |
| No `--rdrop-alpha` CLI flag | `train.py` (full scan) | ✅ Correct — doesn't exist yet |
| No `--swa-start` CLI flag | `train.py` (full scan) | ✅ Correct — doesn't exist yet |
| `--fusion-lr-multiplier` exists | `train.py:146` | ✅ Correct — default 0.5 |
| `--appnp-alpha` exists | `train.py:228-235` | ✅ Correct — default 0.0 |
| Trainer only saves best checkpoint | `trainer.py:1997-2029` | ✅ Correct — `_best.pt` overwritten; no intermediate saves |
| B3 label smoothing exists | `trainer.py:308-319` (config), `trainer.py:659-662` (application) | ✅ Already implemented |
| `predict_with_uncertainty` doesn't exist | `predictor.py` (full scan) | ✅ Correct — new method needed |
