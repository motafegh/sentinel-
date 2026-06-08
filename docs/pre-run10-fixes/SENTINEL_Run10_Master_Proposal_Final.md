# SENTINEL Run 10 — Definitive Master Proposal

> **Supersedes:** `pre_run10_proposal.md`, `pre_run10_final_proposal.md`, `proposal v10.md`
> **Informed by:** `pre_run10_proposal_audit_2026-06-07.md`, `Adversarial Technical Review — Pre-Run 10 Final Proposal.md`, training session analysis 2026-06-08
> **Ground truth:** actual source reads of `sentinel_model.py`, `trainer.py`, `predictor.py`

---

## 0. Current State Snapshot

### Run 9 v11 (live, as of 2026-06-08 00:32 UTC)

| Epoch | F1-macro | Notes |
|-------|----------|-------|
| ep14 | 0.2586 ★ | Checkpoint LOST (resume typo incident, lambda 0.0075 vs 0.005) |
| ep35 | 0.2885 ★ | Recovered best after restart from 0.2395 |
| ep39 | 0.2917 ★ | Current best, checkpoint saved |
| ep40 | 0.2858   | 1/30 patience |
| ep41 | in progress | Started 00:05 UTC, step 400/455 at 00:32 |

**Distance to target:** 0.016 below Run 7 raw best (0.3074), 0.051 below Run 7 tuned (0.3423).

### Per-class performance (ep39 best)
```
IntegerUO=0.666    GasException=0.361   MishandledException=0.307
UnusedReturn=0.228 Timestamp=0.197      DenialOfService=0.124
```
Reentrancy, CallToUnknown, ExternalBug, TOD: not in top/bottom 3 → moderate.

### Persistent training signals
- **ph2 loss ≈ 0.60 every epoch** vs other eyes ≈ 0.34-0.37 — CFG eye is consistently 1.6× harder
- **`gnn_to_bert_proj` weight norm monotonically growing:** ep36=31.94 → ep41=32.81 (no sign of saturation)
- **BFloat16 diagnostic warning every epoch:** zero visibility into prefix attention contribution
- **JK weights:** Phase3 dominant (0.37 ± 0.09) vs Phase1/Phase2 (0.31 ± 0.05 each)

### Dataset facts (train split, v10)
| Class | Train samples | Share |
|-------|--------------|-------|
| IntegerUO | ~9,486 | 32.6% |
| DoS | 246 | 0.85% |
| TOD | ~135 | 0.46% |
| Safe | ~16,850 | 57.9% |
| Total | 29,103 | — |

**Core imbalance ratio:** IntegerUO:DoS = 38.6×

---

## P0: No-Retraining Fixes
*Implement during Run 9 tail. Zero model changes, zero new training runs.*

---

### P0-A: Fix Predictor Tier Threshold Bug
**File:** `ml/src/inference/predictor.py:698-715`
**Priority:** Critical | **Effort:** 1h | **Risk:** None

**The bug (confirmed from source):**
```python
# CURRENT CODE — BUGGY (lines ~698-701):
conf_thr = self.tier_confirmed_threshold   # global 0.55 for ALL classes
susp_thr = self.tier_suspicious_threshold  # global constant
for cls_name, prob in zip(self._class_names, probs_list):
    if prob >= conf_thr:   # ← same threshold for DoS and IntegerUO
```
Per-class thresholds from Optuna are loaded into `self.thresholds` but never used in tiering logic. Every class is evaluated against the single global 0.55 cutoff.

**Fix — recommended approach: Approach A (0.7× scaled suspicious tier):**
```python
# FIXED CODE:
for i, (cls_name, prob) in enumerate(zip(self._class_names, probs_list)):
    p = round(prob, 4)
    cls_conf_thr = self.thresholds.get(cls_name, self.tier_confirmed_threshold)
    cls_susp_thr = cls_conf_thr * 0.7   # suspicious = 70% of confirmed threshold
    if prob >= cls_conf_thr:
        confirmed.append({"vulnerability_class": cls_name, "probability": p, "tier": "CONFIRMED"})
    elif prob >= cls_susp_thr:
        suspicious.append({"vulnerability_class": cls_name, "probability": p, "tier": "SUSPICIOUS"})
```

**Note:** This does not change training F1. It fixes the deployment API so that per-class calibrated thresholds (from P0-E Optuna) are actually applied at inference.

---

### P0-B: Fix BFloat16 Prefix Diagnostic (5-minute fix)
**File:** `ml/src/models/sentinel_model.py` — `compute_prefix_attention_mean()`
**Priority:** Medium | **Effort:** 5 min | **Risk:** None

The WARNING `prefix_attention_mean diagnostic failed: expected scalar type Float but found BFloat16` fires every epoch because AMP produces BFloat16 tensors and the diagnostic doesn't cast before computing the mean. This means you have zero visibility into prefix attention contribution for every epoch of Run 9 (and Run 10 if unfixed).

```python
# In compute_prefix_attention_mean(), around line where result is unpacked:
if isinstance(result, tuple):
    _, prefix_attn_mean = result
    # Cast to float before scalar extraction (BFloat16 → Float32 guard)
    if isinstance(prefix_attn_mean, torch.Tensor):
        prefix_attn_mean = prefix_attn_mean.float().item()
    return float(prefix_attn_mean) if prefix_attn_mean is not None else None
return None
```

---

### P0-C: Per-Class Platt Scaling (NOT single-T temperature)
**File:** New `ml/calibration/platt_scaler.py`
**Priority:** High | **Effort:** 1 day | **Risk:** Low

**Why NOT single-T:** Temperature T is optimized on val NLL. With 57.9% safe + 32.6% IntegerUO dominating, T will be set to compress the majority predictions. DoS/TOD logits (already near-zero) get divided by the same T — their already-weak signal is further compressed. Single-T cannot independently shift DoS logits upward.

**Per-class Platt scaling** fits an independent logistic regression scaler per class (slope + bias, 2 parameters each) on val logits. It can shift near-zero DoS logits upward while compressing overconfident IntegerUO predictions independently.

```python
# ml/calibration/platt_scaler.py
from sklearn.linear_model import LogisticRegression
import numpy as np, json
from pathlib import Path

CLASS_NAMES = [
    "IntegerUO", "Reentrancy", "GasException", "MishandledException",
    "DenialOfService", "Timestamp", "CallToUnknown", "ExternalBug",
    "UnusedReturn", "TransactionOrderDependence",
]

def fit_platt_scalers(val_logits_np, val_labels_np, save_path):
    """
    val_logits_np : [N, 10] numpy float32  (raw model logits, no sigmoid)
    val_labels_np : [N, 10] numpy binary
    Fits one Platt scaler per class, saves coefficients to JSON.
    """
    scalers = {}
    for i, cls in enumerate(CLASS_NAMES):
        logits_i = val_logits_np[:, i].reshape(-1, 1)
        labels_i = val_labels_np[:, i]
        if labels_i.sum() < 5:  # too few positives — skip, use identity
            scalers[cls] = {"slope": 1.0, "bias": 0.0, "calibrated": False}
            continue
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        lr.fit(logits_i, labels_i)
        scalers[cls] = {
            "slope": float(lr.coef_[0][0]),
            "bias":  float(lr.intercept_[0]),
            "calibrated": True,
        }
    Path(save_path).write_text(json.dumps(scalers, indent=2))
    return scalers

def apply_platt_scaling(logits_tensor, scalers):
    """Apply fitted scalers. Returns calibrated logits for downstream sigmoid+threshold."""
    cal = logits_tensor.clone()
    for i, cls in enumerate(CLASS_NAMES):
        s = scalers[cls]
        cal[:, i] = logits_tensor[:, i] * s["slope"] + s["bias"]
    return cal   # still logits — apply sigmoid externally
```

Run after Run 9 finishes: collect val-split logits from best checkpoint, fit, save to
`ml/calibration/platt_scalers_run9.json`.

**Note:** `ml/calibration/temperatures_run7.json` contains per-class *thresholds*, not temperatures.
Do not repurpose it — create a new file.

---

### P0-D: Logit Adjustment at Inference
**File:** `ml/src/inference/predictor.py` — new pre-sigmoid hook
**Priority:** HIGHEST ROI of all P0 items | **Effort:** 1 day | **Risk:** Low

**Why this is the strongest single fix for DoS/TOD recall=0:**
The model's decision function is biased toward high-frequency classes because the loss surface is dominated by them. Logit Adjustment (Menon et al., 2021) exactly reverses this bias: at inference subtract `τ × log(π_y)` where `π_y` is the class marginal frequency in training data. The model must produce a stronger signal to predict IntegerUO than DoS, proportional to their frequency ratio. Theoretically recovers the Bayes-optimal classifier under uniform recall evaluation.

```python
# ml/src/inference/logit_adjustment.py
import torch, math, json
from pathlib import Path

# Exact class priors: compute from training split pkl BEFORE deploying.
# These values are approximate — replace with exact counts from dataset.
_APPROX_PRIORS = {
    "IntegerUO":               9486 / 29103,
    "Reentrancy":              3500 / 29103,
    "GasException":            3392 / 29103,
    "MishandledException":     2100 / 29103,
    "DenialOfService":          246 / 29103,
    "Timestamp":               1500 / 29103,
    "CallToUnknown":           1200 / 29103,
    "ExternalBug":              800 / 29103,
    "UnusedReturn":            2000 / 29103,
    "TransactionOrderDependence": 135 / 29103,
}

def compute_exact_priors_from_dataset(train_pkl_path):
    """
    Load the cached dataset, compute exact class marginals from train split.
    Call once, save to ml/calibration/class_priors_v10.json.
    """
    import pickle
    with open(train_pkl_path, "rb") as f:
        data = pickle.load(f)
    labels = data["train_labels"]   # [N, 10] — adjust key as needed
    priors = {cls: float(labels[:, i].mean()) for i, cls in enumerate(CLASS_NAMES)}
    return priors

def apply_logit_adjustment(logits, class_names, priors, tau=1.0):
    """
    logits : [B, 10] raw model logits (pre-sigmoid)
    tau    : adjustment strength. 1.0 = full theoretical correction.
             Tune on val; reduce to 0.5 if DoS FP rate explodes.
    Returns adjusted logits. Apply Platt scaling AFTER this.
    """
    log_priors = torch.tensor(
        [math.log(max(priors[c], 1e-6)) for c in class_names],
        device=logits.device, dtype=logits.dtype,
    )
    return logits - tau * log_priors   # [B, 10]
```

**Integration order in `_score_windowed()` (predictor.py:638-639):**
```
raw logits → logit_adjustment() → platt_scaling() → sigmoid() → threshold comparison
```

**Tuning τ:** Start at 1.0. Evaluate DoS/TOD recall and IntegerUO precision on val set.
If DoS FP rate exceeds 30% at τ=1.0, reduce to 0.5. τ=0.0 is a no-op (baseline).

**Critical:** Compute exact priors from training pkl before first deployment. The approximate values above are estimates.

---

### P0-E: Optuna Threshold Search with F2 Objective + Precision Floor
**File:** New `ml/scripts/tune_thresholds.py`
**Priority:** High | **Effort:** 1 day | **Risk:** Medium (val overfitting)
**Dependency:** `optuna` must be added to `ml/pyproject.toml` (currently absent)

**Objective change (critical):** Use F2 (β=2), not F1. For a security oracle, missing a vulnerability has greater cost than a false alarm. F2 weights recall 4× over precision.

**Anti-overfitting guard:** The adversarial review identified the DoS co-occurrence trap: Optuna can achieve apparent DoS recall by lowering the Reentrancy threshold (P(Reentrancy|DoS)=0.985), boosting recall without the model actually learning DoS. Add a precision floor to prevent this.

```python
# ml/scripts/tune_thresholds.py
import optuna, torch, numpy as np
from sklearn.metrics import fbeta_score, precision_score

CLASS_NAMES = [
    "IntegerUO", "Reentrancy", "GasException", "MishandledException",
    "DenialOfService", "Timestamp", "CallToUnknown", "ExternalBug",
    "UnusedReturn", "TransactionOrderDependence",
]
PRECISION_FLOOR = 0.15   # minimum acceptable precision per class (if >=20 positives in val)
MIN_VAL_POSITIVES = 20   # don't enforce floor for extremely rare classes

def make_objective(val_probs_np, val_labels_np, test_probs_np, test_labels_np):
    def objective(trial):
        thresholds = np.array([
            trial.suggest_float(f"thr_{cls}", 0.05, 0.90)
            for cls in CLASS_NAMES
        ])
        preds = (val_probs_np > thresholds).astype(int)  # [N, 10]

        # Precision floor guard: penalise trials that tank precision for any class
        per_class_prec = precision_score(val_labels_np, preds, average=None, zero_division=0)
        per_class_pos  = val_labels_np.sum(0)
        for i, (prec, n_pos) in enumerate(zip(per_class_prec, per_class_pos)):
            if n_pos >= MIN_VAL_POSITIVES and prec < PRECISION_FLOOR:
                return 0.0   # invalid trial

        val_f2 = fbeta_score(val_labels_np, preds, beta=2, average="macro", zero_division=0)

        # Track val→test gap (Run 8 gap was 0.054 — flag if >0.04)
        test_preds = (test_probs_np > thresholds).astype(int)
        test_f2    = fbeta_score(test_labels_np, test_preds, beta=2, average="macro", zero_division=0)
        trial.set_user_attr("test_f2",  round(test_f2,  4))
        trial.set_user_attr("val_test_gap", round(val_f2 - test_f2, 4))

        return val_f2

    return objective

def run_threshold_search(val_probs, val_labels, test_probs, test_labels,
                         n_trials=500, output_path="ml/calibration/thresholds_run10.json"):
    import json
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        make_objective(val_probs, val_labels, test_probs, test_labels),
        n_trials=n_trials, show_progress_bar=True,
    )
    best = study.best_params
    thresholds = {cls: best[f"thr_{cls}"] for cls in CLASS_NAMES}
    json.dumps(thresholds, indent=2)  # save

    # Warn if val→test gap is large
    gap = study.best_trial.user_attrs["val_test_gap"]
    if gap > 0.04:
        print(f"WARNING: val→test F2 gap = {gap:.3f}. Consider reducing search space.")

    return thresholds
```

---

### P0-F: MC Dropout Uncertainty Endpoint
**File:** `ml/src/inference/predictor.py` — new method
**Priority:** Medium | **Effort:** 0.5 days | **Risk:** None

```python
def predict_with_uncertainty(self, graphs, input_ids, mask, n_samples=30):
    """
    n_samples forward passes with dropout active. Returns mean prediction,
    per-class std (uncertainty), and a flag for contracts needing human review.

    From adversarial review: the FP explosion on complex contracts is STABLE
    (std < 0.03 across masks). MC dropout correctly identifies these as high-
    confidence errors, not uncertainty — useful for production triage, not for
    fixing training F1.
    """
    self.model.train()   # activate dropout (do NOT call eval() until done)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = self.model(graphs, input_ids, mask, return_aux=False)
            preds.append(torch.sigmoid(logits))
    self.model.eval()

    preds       = torch.stack(preds)       # [30, B, 10]
    mean        = preds.mean(0)            # [B, 10]
    std         = preds.std(0)             # [B, 10]
    needs_review = (std > 0.15).any(dim=1) # [B] — genuinely uncertain contracts

    return mean, std, needs_review
```

---

## P1: Run 10 Architecture Additions
*Implement before Run 10 launch. Each can be toggled independently via config flags.*

---

### P1-A: Label Dependency Layer with IntegerUO Outgoing-Edge Cap
**File:** New `ml/src/models/label_dependency.py` + hook in `sentinel_model.py:558`
**Priority:** High | **Effort:** 1.5 days | **Risk:** Medium

**The IntegerUO dominance problem (adversarial review):**
The co-occurrence matrix gives P(IntegerUO|X) ≈ 0.66-0.99 for 6 of 10 classes. Without correction, the label GCN learns one thing: amplify IntegerUO everywhere. This helps the already-best class (F1=0.666) while providing zero gradient signal for classes with near-zero logits (DoS logit ≈ 0 → neighbor_signal ≈ 0).

**Critical fix:** Zero IntegerUO's outgoing edges before using the matrix.

```python
# ml/src/models/label_dependency.py
import torch
import torch.nn as nn

CLASS_NAMES = [
    "IntegerUO", "Reentrancy", "GasException", "MishandledException",
    "DenialOfService", "Timestamp", "CallToUnknown", "ExternalBug",
    "UnusedReturn", "TransactionOrderDependence",
]
INTEGERUO_IDX = CLASS_NAMES.index("IntegerUO")  # = 0

class LabelDependencyLayer(nn.Module):
    """
    Lightweight label graph refinement. ~100 parameters.
    Initialized as zero-refinement (identity pass-through at training start).

    The adjacency is the empirical co-occurrence P(j|i) from training labels,
    with IntegerUO's outgoing edges zeroed to prevent it dominating neighbor
    aggregation for every other class (adversarial review correction IMP-L1-A).
    """
    def __init__(self, num_classes: int, train_labels: torch.Tensor):
        super().__init__()

        # Co-occurrence matrix from training labels [N, C]
        co  = torch.mm(train_labels.T.float(), train_labels.float())  # [C, C]
        dia = co.diagonal().clamp(min=1.0)
        adj = co / dia.unsqueeze(1)   # P(j | i) — row-normalised

        # CAP IntegerUO outgoing edges (row AND column for symmetry)
        # Without this: P(IntegerUO | *) dominates every row, layer learns
        # only to amplify IntegerUO predictions everywhere.
        adj[INTEGERUO_IDX, :] = 0.0
        adj[:, INTEGERUO_IDX] = 0.0
        adj.fill_diagonal_(0.0)   # no self-loops

        self.register_buffer("adj", adj)   # [C, C] — fixed, not trained

        # Learned transformation on neighbor-aggregated signals.
        # Zero init → identity at t=0; learns incrementally.
        self.W = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.zeros_(self.W.weight)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]
        neighbor = torch.mm(logits, self.adj)   # [B, C]
        refined  = self.W(neighbor)              # [B, C]
        return logits + refined                   # residual — never replaces original

    def get_signal_strength(self) -> float:
        """Monitoring metric: if < 1e-3 at epoch 10, layer has zero signal — disable."""
        return float(self.W.weight.abs().max().item())
```

**Integration in `sentinel_model.py` (after line 558):**
```python
# sentinel_model.py forward(), AFTER classifier:
combined = torch.cat([gnn_eye, transformer_eye, fused_eye, cfg_eye], dim=1)  # [B, 512]
logits   = self.classifier(combined)        # [B, num_classes]  ← existing line 558
logits   = self.label_dep(logits)           # [B, num_classes]  ← NEW LINE
```

**Monitoring:** Log `model.label_dep.get_signal_strength()` each epoch. Kill condition: if still < 1e-3 at ep10, disable and remove from graph.

**Computing the adjacency:** Pass training split labels tensor to `LabelDependencyLayer.__init__()` at model construction time, OR pre-compute and save as `ml/calibration/label_cooccurrence_v10.pt`.

---

### P1-B: GraphMixup — Rare-Class Targeted Only
**File:** `ml/src/training/trainer.py` — inside `train_one_epoch`, new helper function
**Priority:** Medium-High | **Effort:** 2 days | **Risk:** Medium

**Why rare-class targeted is non-negotiable:**
Global GraphMixup (proposal v10) mixes every sample with a random batch partner. Since 57.9% of contracts are safe, DoS positive samples would be mixed with safe contracts 57.9% of the time — further diluting the already minimal 246-sample DoS signal. Targeted mixing ensures rare-class positives only mix with other vulnerable contracts.

**Where to mix:** After `combined = torch.cat([...], dim=1)` in `sentinel_model.py` (line 557), before `self.classifier(combined)`. The [B, 512] combined embedding is the right point: both GNN and transformer paths are already computed, mixing happens in the shared representation space.

**Implementation — trainer-side hook:**
```python
# ml/src/training/trainer.py — add helper function
import numpy as np

RARE_CLASS_NAMES = {"DenialOfService", "TransactionOrderDependence", "ExternalBug"}

def apply_rare_mixup(combined_embs: torch.Tensor,
                     labels: torch.Tensor,
                     class_names: list,
                     alpha: float = 0.4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mix [B, 512] embeddings only for samples with at least one rare-class positive.
    Rare class positives mix only with other positives (not with safe contracts).

    Returns (mixed_combined, mixed_labels) — unchanged originals if <2 rare samples.
    """
    rare_indices = [i for i, c in enumerate(class_names) if c in RARE_CLASS_NAMES]
    rare_mask    = labels[:, rare_indices].any(dim=1)   # [B] bool
    if rare_mask.sum() < 2:
        return combined_embs, labels   # not enough rare samples in this batch

    rare_idx = rare_mask.nonzero(as_tuple=True)[0]
    perm     = rare_idx[torch.randperm(len(rare_idx), device=combined_embs.device)]

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)   # keep lambda > 0.5 — mixed sample stays mostly original

    mixed = combined_embs.clone()
    mixed_labels = labels.clone().float()

    mixed[rare_idx]        = lam * combined_embs[rare_idx] + (1 - lam) * combined_embs[perm]
    mixed_labels[rare_idx] = lam * labels[rare_idx].float() + (1 - lam) * labels[perm].float()

    return mixed, mixed_labels
```

**Model-side: pass `combined` out from forward() for trainer-level mixing** (alternative: add a `mixup_fn` callback to the model). Cleanest approach is a training-mode flag in `SentinelModel` that returns `combined` as an extra output when `return_combined=True`.

**Kill condition (ep10):** If DoS F1 remains 0.0 at epoch 10, disable mixup — rare class signal is absent from individual batches and mixing provides no useful gradient.

---

### P1-C: SWA as Post-Hoc Checkpoint Averaging
**File:** New `ml/scripts/swa_average.py`
**Priority:** Medium | **Effort:** 0.5 days | **Risk:** Low

**Critical clarifications vs proposal v10 and pre_run10_final_proposal:**
1. **Do NOT integrate SWALR into training** — it conflicts with OneCycleLR; the coexistence requires 2 days of scheduler testing.
2. **`update_bn()` is a NO-OP** — SENTINEL uses LayerNorm throughout (not BatchNorm). Skip it entirely.
3. Post-hoc averaging of saved checkpoints achieves the same flat-minima benefit with zero training-time risk.

```python
# ml/scripts/swa_average.py
import torch
from collections import OrderedDict
from pathlib import Path

def swa_average_checkpoints(checkpoint_paths: list[str], output_path: str):
    """
    Average weights of N checkpoints. Load top-N by val F1 from training run,
    spaced >=5 epochs apart to sample different regions of the loss landscape.

    Equivalent to weight-space ensembling / post-hoc SWA.
    """
    state_dicts = []
    for p in checkpoint_paths:
        ckpt = torch.load(p, map_location="cpu")
        state_dicts.append(ckpt["model"])
        print(f"  Loaded {Path(p).name} | best_f1={ckpt.get('best_f1', 'n/a')}")

    avg_state = OrderedDict()
    for key in state_dicts[0].keys():
        # Cast to float for averaging (BFloat16 checkpoints lose precision)
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(0)

    torch.save({
        "model": avg_state,
        "source": "swa_post_hoc",
        "num_averaged": len(checkpoint_paths),
        "checkpoints": checkpoint_paths,
    }, output_path)
    print(f"Saved SWA model to {output_path}")
```

**Requirement:** Run 10 must save intermediate checkpoints, not just best. Add `--save-top-k 5` to the launch command (or save every 10 epochs). Without this, only one checkpoint is available to average.

**Expected gain:** +0.5-1.5 F1 on test set, primarily from better OOD generalisation (contracts from DeFi protocols underrepresented in training).

---

### P1-D: Hierarchical Classification Head (Risk Variant — optional)
**File:** New `ml/src/models/hierarchical_head.py` + replaces `self.classifier` in `sentinel_model.py`
**Priority:** Experimental | **Effort:** 2-3 days | **Risk:** Medium-High

**Correct class families for SENTINEL's actual 10 classes:**
```python
VULNERABILITY_FAMILIES = {
    "Arithmetic":     ["IntegerUO", "GasException"],                            # 2 classes
    "ReentrancyCEI":  ["Reentrancy", "MishandledException", "UnusedReturn"],   # 3 classes
    "ExternalCall":   ["CallToUnknown", "ExternalBug"],                         # 2 classes
    "OrderingState":  ["Timestamp", "TransactionOrderDependence", "DenialOfService"],  # 3 classes
}
FAMILY_SIZES = [2, 3, 2, 3]   # must sum to 10
```

**Rationale:**
- Arithmetic (IntegerUO + GasException): both require numeric property analysis; shared arithmetic-overflow detection capability
- ReentrancyCEI (Reentrancy + MishandledException + UnusedReturn): all relate to CEI pattern violations and return value handling
- ExternalCall (CallToUnknown + ExternalBug): both involve uncontrolled external interactions
- OrderingState (Timestamp + TOD + DoS): all involve ordering/timing/state manipulation — DoS gets indirect gradient from Timestamp (more common, better learned)

```python
# ml/src/models/hierarchical_head.py
import torch
import torch.nn as nn

class HierarchicalHead(nn.Module):
    """
    Drop-in replacement for the flat Linear(512,256)→ReLU→Dropout→Linear(256,10) classifier.
    Level 1: family classification (4 families).
    Level 2: per-family class prediction (class sees embedding + family context).

    Key benefit: DoS (OrderingState family) receives gradient from Timestamp predictions
    even when DoS itself is absent from a batch — indirect supervision for rare classes.
    """
    def __init__(self, eye_dim: int = 512, num_families: int = 4,
                 family_sizes: list = None, dropout: float = 0.1):
        super().__init__()
        if family_sizes is None:
            family_sizes = [2, 3, 2, 3]
        assert sum(family_sizes) == 10, "family_sizes must sum to num_classes"

        self.family_head = nn.Sequential(
            nn.Linear(eye_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_families),
        )
        # Each class head sees: embedding [512] + family logits [4] = [516]
        enriched_dim = eye_dim + num_families
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enriched_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )
            for n_classes in family_sizes
        ])

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        # combined: [B, 512]
        family_logits = self.family_head(combined)          # [B, 4]
        enriched      = torch.cat([combined, family_logits], dim=1)  # [B, 516]

        class_outputs = [head(enriched) for head in self.class_heads]
        return torch.cat(class_outputs, dim=1)              # [B, 10]
```

**Note on class ordering:** Output order must match `CLASS_NAMES` exactly. The family ordering above maps to:
`[IntegerUO, GasException, Reentrancy, MishandledException, UnusedReturn, CallToUnknown, ExternalBug, Timestamp, TransactionOrderDependence, DenialOfService]`
This is a DIFFERENT order than the current `CLASS_NAMES` — the dataset and loss computation must be reordered accordingly, or an index mapping layer must be added. **This is the highest implementation risk in P1-D.**

**Recommendation:** Only implement P1-D if Run 10 with P1-A + P1-B shows DoS F1 still = 0. The class reordering risk is significant.

---

## P2: Run 10 Training Configuration

### P2-A: Full Launch Command

```bash
# Corrections vs pre_run10_final_proposal.md:
# - appnp_alpha 0.0 (Run 8 used 0.2 — regression evidence)
# - fusion_lr_multiplier 0.5 (Run 7 used 0.5; Run 8 used 0.3 — revert)
# - save_top_k 5 (NEW — required for post-hoc SWA in P1-C)

poetry run python ml/scripts/train.py \
    --run-id GCB-P1-Run10-v11-20260610 \
    --model four-eye \
    --gnn-layers 8 --gnn-hidden-dim 256 --gnn-phase-heatup 3 \
    --jk-type cat --jk-agg-max false \
    --jk-entropy-reg-lambda 0.005 \
    --gnn-prefix-k 48 --gnn-prefix-warmup-epochs 15 \
    --gnn-dropout 0.2 \
    --transformer-name microsoft/graphcodebert-base \
    --lora-r 16 --lora-alpha 32 \
    --fusion-dim 256 \
    --lr 1e-4 --weight-decay 0.01 \
    --epochs 80 --patience 30 \
    --gradient-accumulation-steps 8 --batch-size 8 \
    --loss asl --asl-gamma-negative 4 --asl-gamma-positive 0 \
    --aux-loss-weight 0.3 --aux-phase2-loss-weight 0.20 \
    --drop-complexity-feature \
    --weighted-sampler timestamp-size \
    --threshold-tune-interval 10 \
    --compile \
    --appnp-alpha 0.0 \
    --fusion-lr-multiplier 0.5 \
    --save-top-k 5
```

**If P1-A and P1-B are implemented, add:**
```bash
    --label-dependency true \
    --label-dep-integeruo-cap true \
    --mixup-rare-classes true \
    --mixup-alpha 0.4
```

### P2-B: Early Monitoring Checklist for Run 10

Check at these epoch milestones before allowing training to continue:

| Epoch | Check | Kill Condition |
|-------|-------|----------------|
| ep5  | ph2 loss / avg_eyes ratio | If ratio > 2.5 (vs current ~1.7), CFG eye is diverging |
| ep10 | `label_dep.get_signal_strength()` | If < 1e-3, disable LabelDependencyLayer |
| ep10 | DoS F1 | If still 0.0 and Mixup enabled, disable Mixup |
| ep15 | GNN prefix warmup end | Confirm `gnn_to_bert_proj` weight norm starts stabilising |
| ep20 | val F1-macro | If < 0.2907 (Run 9 ep39 best), STOP — run is regressing |
| ep30 | val→test F1 gap | If > 0.05, threshold tuning is overfitting — reduce search interval |

---

## P3: Post-Run 10 (with results in hand)

### P3-A: Decoupled Classifier Rebalancing
**Effort:** 1 day | **Expected gain:** +2-5 F1 on minority classes | **Risk:** Low

After Run 10 completes: freeze the entire backbone, retrain only `self.classifier` for 10-15 epochs on class-balanced data.

**Why this is powerful:** Run 10's backbone will have learned good representations for DoS and TOD (the GNN sees the CEI pattern; the transformer sees the vulnerability-related tokens). The problem is the decision boundary — the classifier head was trained on imbalanced data throughout and its weights heavily favour IntegerUO. Retraining it on balanced data with a frozen backbone directly fixes this without touching the representations.

```python
# Freeze everything except classifier
for name, param in model.named_parameters():
    if "classifier" not in name and "label_dep" not in name:
        param.requires_grad_(False)

# Balanced sampler: oversample all classes to equal representation
n_target = max_class_count   # e.g., equal to IntegerUO count
balanced_weights = compute_balanced_weights(train_labels, n_target)
balanced_sampler = WeightedRandomSampler(balanced_weights, num_samples=len(train_dataset))

# Retrain for 10-15 epochs at low LR
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5
)
```

**Note:** This gives a "free" second chance at a better decision boundary without a full new training run (~6h vs ~60h).

---

### P3-B: Error Analysis Tooling
**Effort:** 2-4h | **Should have been done before Run 9**

Run a systematic forward pass on val/test to answer the fundamental diagnostic question for DoS:

```python
# One-time diagnostic script: ml/scripts/error_analysis.py
model.eval()
dos_idx = CLASS_NAMES.index("DenialOfService")
results = []

with torch.no_grad():
    for batch in val_loader:
        logits = model(batch.graphs, batch.input_ids, batch.mask, return_aux=False)
        probs  = torch.sigmoid(logits)
        for i in range(len(batch)):
            results.append({
                "dos_label":    batch.labels[i, dos_idx].item(),
                "dos_logit":    logits[i, dos_idx].item(),
                "dos_prob":     probs[i, dos_idx].item(),
                "num_nodes":    batch.graphs[i].num_nodes,
                "is_complex":   batch.graphs[i].x[:, 5].max().item() > 0.5,  # complexity feature
            })

import pandas as pd
df = pd.DataFrame(results)
dos_pos = df[df["dos_label"] == 1]
dos_neg = df[df["dos_label"] == 0]

print(f"DoS positives: {len(dos_pos)}")
print(f"  Median logit: {dos_pos['dos_logit'].median():.3f}")
print(f"  Median prob:  {dos_pos['dos_prob'].median():.4f}")
print(f"  % above 0.3:  {(dos_pos['dos_prob'] > 0.3).mean():.1%}")
print(f"DoS negatives:")
print(f"  Median logit: {dos_neg['dos_logit'].median():.3f}")
```

**Interpretation:**
- DoS positive median logit ≈ DoS negative median logit → **feature representation gap** (model has no distinguishing signal) → schema additions needed for Run 11
- DoS positive median logit < DoS negative median logit → **inverted prediction** (model learned inverse correlation) → label quality issue
- DoS positive median logit > DoS negative but both below threshold → **calibration problem** → logit adjustment + Platt scaling sufficient

This determination gates whether P4-B (schema additions) is necessary for Run 11.

---

### P3-C: F2 vs F1 Threshold Comparison
After running P0-E (Optuna with F2) and P3-B (error analysis), compare:
- F1-tuned thresholds vs F2-tuned thresholds on test set
- Document the precision/recall tradeoff explicitly for the ZK oracle deployment

For SENTINEL as a security oracle: a false negative (missed vuln) is worse than a false positive (unnecessary human review). F2 is almost certainly the right operating point, but verify this aligns with the SENTINEL design doc and ZK proof constraints before hardcoding.

---

## P4: Longer Horizon — Run 11

### P4-A: Graph Schema Additions
**Status:** Blocked on re-extraction | **Expected value:** High for DoS and TOD

Two targeted node features for v10 schema:

1. **Loop nesting depth per basic block** (1 feature, float normalised 0-1):
   - Direct signal for DoS via unbounded gas consumption in nested loops
   - Extractable from Slither: `node.depth` in CFG
   - Primarily helps: DenialOfService, GasException

2. **Call sequence index** (1 feature, int normalised by total calls in contract):
   - Position of each CALL/DELEGATECALL node in the execution flow
   - Direct encoding of temporal ordering — the core TOD signal
   - Currently absent from v9 schema; TOD has F1 ≈ 0 without it

**Implementation path:** Increment `NODE_FEATURE_DIM` from 12 → 14, update `graph_schema.py`, re-extract all 41,576 graphs with new extractor. The `_MAX_TYPE_ID` assert in `sentinel_model.py` will catch any schema mismatch.

**Gate:** Run P3-B error analysis first. If DoS positives have clearly lower logits than DoS negatives, loop depth is a critical missing feature. If not, the problem is elsewhere.

### P4-B: BCCC Dataset Expansion
**Status:** Stage 1 in progress (Slither 8%, Aderyn 0.2%)
**Target:** 30k+ additional graphs for Run 11

Continue Stage 1 in parallel with Run 10 training. This is the highest-leverage long-term investment for improving rare-class performance — more real DoS/TOD examples are worth more than any architectural improvement.

### P4-C: Supervised Contrastive Loss Head
**Effort:** 2-3 days | **Risk:** Medium

Add a SupCon auxiliary loss on the [B,512] `combined` embeddings. This pulls same-class embeddings together and pushes different-class embeddings apart, independent of class frequency. DoS embeddings are pulled together regardless of 246 samples.

```python
# In trainer.py, alongside main ASL loss:
from pytorch_metric_learning.losses import SupConLoss
import torch.nn.functional as F

supcon_loss_fn = SupConLoss(temperature=0.07)
# combined: [B, 512] — retrieved as extra output from model
supcon_loss = supcon_loss_fn(F.normalize(combined, dim=1), labels)
total_loss = main_loss + 0.1 * supcon_loss   # start at 0.1, tune
```

**Note:** Use supervised contrastive loss (labels available), NOT unsupervised contrastive pre-training (requires 100k unlabeled graphs which don't exist yet).

### P4-D: Logit Adjustment Refinement
After BCCC data arrives, recompute class priors from the expanded dataset. If BCCC meaningfully changes the class distribution (especially for DoS/TOD which should have more samples), update the logit adjustment prior vector accordingly.

### P4-E: GNNExplainer for Audit Trails
**Priority:** Production | **Effort:** 3-5 days

After F1 reaches production quality, add `torch_geometric.explain.GNNExplainer` to generate edge-level explanations for each vulnerability prediction. Auditors require "which specific function/line triggered this flag" — a model that outputs `reentrancy: 0.87` with no explanation will not be adopted.

```python
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=sentinel_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(mode="multiclass_classification",
                      task_level="graph",
                      return_type="log_probs"),
)
```

Expected output: edge masks identifying `CHECK→CALL→WRITE` pattern for reentrancy, `CALL loop → GAS_LIMIT` for DoS, etc. This is the difference between a research model and a commercial product.

---

## Do Not Implement — With Reasons

| Item | Reason | Source |
|------|--------|--------|
| **R-Drop** | The FP explosion is STABLE across dropout masks (std < 0.03). R-Drop penalises inconsistency — the opposite problem. Near-zero expected gradient against the actual failure mode. Also doubles training time. | Adversarial review |
| **Self-Paced Learning** | Logic is inverted for rare classes: ALL DoS positives have high loss throughout training (not just early), so the λ schedule perpetually excludes them. Adversarial review gate will likely fail (57.9% safe contracts = low complexity). | Adversarial review + proposal analysis |
| **Single-T Temperature Scaling** | T dominated by 57.9% safe + 32.6% IntegerUO. Cannot shift DoS logits upward while compressing IntegerUO. Replaced by per-class Platt scaling (P0-C). | Adversarial review |
| **Global GraphMixup** | Mixing DoS with random batch elements hits safe contracts 57.9% of the time, diluting the already-sparse DoS signal. Must be rare-class targeted (P1-B). | Analysis |
| **Curriculum Learning** | Run gate analysis before any implementation. 57.9% safe contracts cluster at low complexity → curriculum = mostly safe warmup. | Adversarial review |
| **Pseudo-Labeling** | No class currently has F1 > 0.75 (IntegerUO at 0.666 is the best). Model's systematic errors would be amplified. Valid in Run 12+ if a class exceeds threshold. | proposal v10 + analysis |
| **Unsupervised Contrastive Pre-training** | Blocked on 100k+ unlabeled contract graphs. BCCC is 8% done. Use supervised contrastive loss (P4-C) instead — no additional data needed. | proposal v10 + analysis |
| **Adversarial Token Training (FGSM)** | Requires `forward_from_embeddings()` hook (doesn't exist), doubles training step time, has BFloat16 gradient issues with current AMP setup. Valid for Run 12+ after infrastructure work. | proposal v10 analysis |
| **Label Smoothing (add)** | Already implemented. `trainer.py:308-319` (per-class ε config) and `trainer.py:658-662` (application). Per-class ε values already set (DoS=0.18, Reentrancy=0.14, IntegerUO=0.08, etc). No further action needed. | Code audit |

---

## Implementation Timeline

```
NOW (Run 9 tail, today):
├── P0-A  predictor.py threshold bug fix                  1h
└── P0-B  BFloat16 diagnostic cast fix                    5 min

BEFORE RUN 10 LAUNCH (~Jun 10):
├── P0-C  Per-class Platt scaling (fit on Run 9 val)      1 day
├── P0-D  Logit Adjustment (compute exact priors)         1 day
├── P0-E  Optuna F2 + precision floor (add optuna to pyproject.toml)  1 day
├── P0-F  MC Dropout uncertainty endpoint                 0.5 day
├── P1-A  LabelDependencyLayer + IntegerUO cap            1.5 days
├── P1-B  Rare-class GraphMixup                           2 days
└── P1-C  SWA checkpoint averaging script                 0.5 day

RUN 10 (~Jun 10 launch, ~80 epochs × 45 min ≈ 60h):
├── Monitor ep5:  ph2/avg loss ratio
├── Monitor ep10: LabelDep signal strength + Mixup kill
├── Monitor ep15: prefix warmup end, weight norm trend
├── Monitor ep20: val F1 kill condition (< 0.2907)
└── Save top-5 checkpoints for post-hoc SWA

POST-RUN 10:
├── P3-B  Error analysis tooling (DoS diagnosis)         2-4h
├── P3-A  Decoupled classifier rebalancing (if DoS still 0)  1 day
└── P3-C  F2 vs F1 threshold comparison

RUN 11 PREPARATION (parallel to all above):
├── P4-A  BCCC Stage 1 completion
├── P4-B  Schema additions after P3-B diagnosis confirms feature gap
└── P4-C  Supervised contrastive loss head
```

---

## Known Open Questions

1. **Does DoS fail due to feature representation or decision boundary?** → P3-B error analysis answers this and gates P4-B (schema additions) vs P4-C (contrastive loss).

2. **Does logit adjustment with τ=1.0 produce acceptable DoS FP rate?** → Tune on val before deploying. Expected: DoS recall increases substantially, IntegerUO precision decreases slightly, net macro F2 improves.

3. **What is the val→test gap for F2-tuned thresholds?** → Track in Optuna trial attributes. If gap > 0.04, reduce n_trials or narrow search space.

4. **Does `gnn_to_bert_proj` weight norm stabilise or continue growing?** → If it exceeds 40.0 during Run 10, add specific weight decay to this parameter group or add a gradient clip per-parameter.

5. **Will LabelDependencyLayer have any signal after IntegerUO cap?** → The remaining meaningful co-occurrences are DoS↔Reentrancy (0.985), GasException↔MishandledException, CallToUnknown↔ExternalBug. If these are present, the layer should learn. If not (cap removes all signal), disable.
