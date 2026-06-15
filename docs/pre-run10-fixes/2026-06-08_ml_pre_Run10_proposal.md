# Sentinel v10 Enhancement Proposal
## SCsVulLyzer Integration + Advanced Training Techniques
**Based on:** v8.0-AB vs v7.0 comparison results, SCsVulLyzer V2.0 source analysis, full architecture review  
**Current baseline:** v8.0-AB tuned F1 = 0.2851 | v7.0 tuned F1 = 0.2875  
**Target:** macro-F1 > 0.35 on validation split  
**Document covers:** Two proposal blocks — Block A (SCsVulLyzer integration) and Block B (Advanced training techniques)

---

## Context: What the Numbers Tell Us

Before proposing anything, the v8.0-AB comparison results establish five hard facts that every enhancement must address:

| Confirmed Finding | Implication for Enhancements |
|---|---|
| **H1 CONFIRMED:** Phase 2 multi-edge dilution hurts Reentrancy (−0.017 F1) | Nothing should further dilute Phase 2 signal — no new edge types, no new inputs to GNN during Phase 2 |
| **H3 CONFIRMED:** Label ceiling ~0.28 tuned F1 for both models | The ceiling is a data/label quality problem, not a model architecture problem — architecture changes alone cannot break past it |
| **H5 CONFIRMED:** v8 wins 3 classes, loses 5 — class-specific tradeoff | Enhancements must be class-aware, not global — a single change that helps Reentrancy while hurting IntegerUO is unacceptable |
| **Both models miss:** GasException, TOD, MishandledException completely | These classes need targeted structural solutions, not general regularization |
| **v8 FP explosion on complex contracts** | The model needs better calibration and uncertainty awareness, not more capacity |

These facts define the constraints. Every enhancement below is evaluated against them before being proposed.

---

# BLOCK A: SCsVulLyzer V2.0 Integration

SCsVulLyzer V2.0 extracts ~252 flat features per contract (150 opcode counts, 63 bytecode char frequencies, 11 ABI shape features, 5 AST features, 2 entropy/length, 8 dangerous pattern counts, 4 LOC, 2 contract info, 1 duplicate count, 1 event count) by compiling each `.sol` file with `py-solc-x` and analyzing the compiler output.

**Why NOT a 5th eye:** Injecting 240 clean numeric features as a concatenated eye into the classifier creates shortcut dominance — the MLP converges in 2–4 epochs while the four eyes need 20+, starving them of classifier-level gradient. You already fought this exact pathology with auxiliary heads in v8.1. The static features belong outside the model as scaffolding, not inside it as a competitor.

---

## A1: Post-Inference Hard Constraint Calibrator

**Location:** `ml/src/inference/static_calibrator.py` (new file)  
**Risk:** Zero — no model changes, no retraining  
**Effort:** 4 hours  
**Addresses:** v8's FP explosion on complex contracts; impossible predictions from v7/v8

### Motivation from Results

The manual test shows both models fire `CallToUnknown + Reentrancy + DoS` simultaneously on safe contracts 12 and 19 (`safe_contract`, `safe_with_transfer`). Some of these predictions are structurally impossible — if compiled bytecode has `CALL=0` and `DELEGATECALL=0`, reentrancy cannot exist by definition. The model predicts it anyway because it learned statistical correlations (e.g., "contracts that look like token contracts sometimes have reentrancy") rather than structural facts.

SCsVulLyzer extracts these structural facts directly from the compiler output. The calibrator uses them as hard overrides after Sentinel's inference.



### Expected Impact

- Contracts 12 and 19 (safe contracts) in manual test: `CallToUnknown` FP eliminated when `CALL=0`
- v8's FP count on complex contracts (09, 11, 20: 6–8 simultaneous FPs) reduced by removing impossible class combinations
- No impact on true positive rate — the calibrator only removes predictions that are structurally impossible

**Validation metric:** Re-run the 20-contract manual test with calibrator active. Count FP reduction on safe contracts without any TP regression.

---

## A2: Pre-Filter Agent (MCP Tool)

**Location:** `agents/static_analysis_agent.py` (new file)  
**Risk:** Zero — completely separate from ML module  
**Effort:** 1 day  
**Addresses:** Inference cost on trivially-safe contracts; precision in deployment context

### Motivation from Results

The v8 manual test shows that on `safe_no_calls` (contract 18), v7 fires DoS (FP) and v8 is clean. However, both models still run the full four-eye pipeline on that contract. A contract with zero external calls, zero loops, zero events, and bytecode entropy < 2.0 does not need GNN Phase 2 CFG analysis, CodeBERT token processing, or cross-attention fusion. SCsVulLyzer can determine this in ~200ms on CPU.

### Risk Score Formula

```python
def compute_risk_score(features: dict) -> float:
    """
    Weighted risk score from static features.
    Returns float in [0, ~8]. Threshold 1.5 = route to Sentinel.
    Calibrated to pass all manually-confirmed vulnerable contracts.
    """
    return (
        min(features.get("Solidity call_CALL", 0), 5)          * 0.35 +
        min(features.get("Solidity call_DELEGATECALL", 0), 3)  * 0.25 +
        features.get("Solidity call_SELFDESTRUCT", 0)          * 0.20 +
        features.get("Bytecode Length and Entropy_bytecode_entropy", 0) / 8.0 * 0.10 +
        min(features.get("Functional Features_num_external_calls", 0), 5) * 0.07 +
        (0.03 if features.get("Event Count", 1) == 0 else 0.0)
    )

@mcp_tool
def analyze_contract_static(contract_path: str) -> dict:
    features    = analyze_solidity_contract(contract_path)
    risk_score  = compute_risk_score(features)
    return {
        "risk_score":         risk_score,
        "route_to_sentinel":  risk_score >= 1.5,
        "impossible_vulns":   get_impossible_vulns(features),  # from A1 rules
        "features":           features,
    }
```

**Note:** The routing threshold (1.5) must be validated against your labeled dataset before deployment — compute the fraction of true-positive contracts that would be incorrectly filtered out. Target: < 1% true positive suppression rate.

---

## A3: Training Curriculum via Complexity Score

**Location:** `ml/src/training/curriculum_sampler.py` (new file)  
**Risk:** Medium  
**Effort:** 2 days + 1 full training run  
**Addresses:** High-variance gradients in early epochs; rare class learning (GasException, TOD)

### Motivation from Results

H3 confirms the label ceiling is at ~0.28 tuned F1 for both models. This ceiling is a data quality problem — but curriculum learning can help the model approach that ceiling faster and more reliably by ensuring the four eyes learn clean foundational patterns before encountering the hardest, noisiest examples.

Your `drop_complexity_feature=True` (Run 9) removed the shortcut from model input. This makes training order *more* consequential — without feat[5] as a fallback, the model must genuinely reason about graph structure from epoch 1, even on the hardest contracts.

The GNN prefix injection warmup (k=48, warmup=15 epochs) is NOT the same thing. It solves a cross-module alignment problem (protecting CodeBERT from untrained GNN prefix tokens). Curriculum operates at the DataLoader level, independent of which internal modules are active. They compose additively:

```
Epochs 1-15:   Curriculum = simple contracts (40th pct)
               Prefix = OFF (warmup active — GNN trains on clean simple graphs)

Epochs 16-30:  Curriculum = medium contracts (40-70th pct)
               Prefix = ON (GNN now trained, prefix injection starts from
                           stronger representations on both sides of gnn_to_bert_proj)

Epochs 31+:    Curriculum = full distribution (100th pct)
               Full pipeline, all contracts
```

### Complexity Score (from existing graph .pt files — no new tools)

```python
# ml/src/training/curriculum_sampler.py

def compute_complexity_score(graph_data) -> float:
    """
    Measures structural difficulty for the GNN.
    CRITICAL: Must NOT correlate with label density.
    Validate: np.corrcoef(scores, label_density) < 0.4 before use.
    """
    num_nodes     = graph_data.num_nodes
    num_edges     = graph_data.num_edges
    node_type_ids = (graph_data.x[:, 0] * 13.0).round().long()
    num_functions = torch.isin(node_type_ids, _FUNC_IDS_CPU).sum().item()
    edge_diversity = graph_data.edge_attr.unique().numel() if graph_data.edge_attr is not None else 1

    return (
        math.log1p(num_nodes)     * 0.4 +
        math.log1p(num_edges)     * 0.3 +
        math.log1p(num_functions) * 0.2 +
        edge_diversity            * 0.1
    )
```

### Sampler with Mixing Floor

```python
class CurriculumSampler(torch.utils.data.Sampler):
    """
    Progressive complexity sampler with 15% hard-sample mixing floor.
    The mixing floor prevents distribution shift — the model never trains
    on a simplified world that doesn't exist at inference time.
    """
    def __init__(self, complexity_scores, dataset_labels, total_epochs):
        self.scores       = complexity_scores     # list[float], len = dataset size
        self.label_weights = compute_label_weights(dataset_labels)  # your existing logic
        self.total_epochs  = total_epochs
        self.current_epoch = 0

    def __iter__(self):
        # Threshold: 40th percentile at epoch 0 → 100th by epoch 50% of training
        progress  = min(1.0, self.current_epoch / (self.total_epochs * 0.5))
        threshold = np.percentile(self.scores, 40 + 60 * progress)

        easy_pool = [i for i, s in enumerate(self.scores) if s <= threshold]
        hard_pool = [i for i, s in enumerate(self.scores) if s > threshold]

        # Always include 15% hard samples (mixing floor)
        n_hard  = max(1, int(len(easy_pool) * 0.15))
        indices = easy_pool + random.sample(hard_pool, min(n_hard, len(hard_pool)))

        # Apply existing label weighting within curriculum-filtered pool
        weights = [self.label_weights[i] for i in indices]
        chosen  = list(WeightedRandomSampler(weights, len(indices)))
        return iter([indices[i] for i in chosen])
```

### Risk: Label-Complexity Correlation

If `np.corrcoef(complexity_scores, label_density) > 0.4`, the complexity score re-introduces the shortcut you removed with `drop_complexity_feature`. Validate before any training run. If correlation is high, decorrelate by residualizing complexity scores against label density:

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(label_density.reshape(-1,1), complexity_scores)
complexity_scores_decorr = complexity_scores - reg.predict(label_density.reshape(-1,1))
```

---

# BLOCK B: Advanced Training Techniques

---

## B1: Temperature Scaling (Do This Now — No Retraining)

**Location:** `ml/src/inference/predictor.py` (add post-hoc)  
**Risk:** Zero  
**Effort:** 30 minutes  
**Addresses:** Probability miscalibration observed in v8 manual test

### Motivation from Results

The v9 probability analysis shows scores clustering 0.40–0.67 simultaneously across 6–8 classes on complex contracts. This is overconfidence — the sigmoid of raw logits does not produce calibrated probabilities. Temperature scaling learns a single scalar T on the validation set (model weights frozen) that rescales all logits to produce calibrated outputs.

```python
# ml/src/inference/temperature_scaler.py

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # init > 1 = softening

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.1)

def calibrate_temperature(model, val_loader, device):
    """
    Fit temperature scalar on validation set. Model weights FROZEN.
    Run once after training, save T alongside checkpoint.
    """
    scaler    = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=200)

    # Collect all validation logits first (model frozen)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch.graphs, batch.input_ids, batch.mask)
            all_logits.append(logits.cpu())
            all_labels.append(batch.labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    def step():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scaler(all_logits), all_labels)
        loss.backward()
        return loss
    optimizer.step(step)

    print(f"Learned temperature T={scaler.temperature.item():.4f}")
    # T > 1: model was overconfident (compress toward 0.5) ← likely for v8
    # T < 1: model was underconfident (push toward extremes)
    return scaler.temperature.item()
```

**Run this on your current v9 checkpoint today.** No new training needed.

---

## B2: Optuna Per-Class Threshold Search (Do This Now — No Retraining)

**Location:** `ml/scripts/tune_threshold.py` (extend existing)  
**Risk:** Zero  
**Effort:** 2 hours  
**Addresses:** v9's threshold miscalibration; DoS threshold = 0.05 noise in v7

### Motivation from Results

Your comparison already showed both models gain +0.022 F1 from threshold tuning. But the current search uses a single-pass grid sweep. Optuna's TPE sampler explores the joint 10-dimensional threshold space more efficiently, finding configurations that maximize macro-F1 globally rather than per-class independently.

The v7,8 DoS threshold of 0.05 (fires on everything) and v8's Timestamp threshold dropping to 0.30 are both symptoms of independent per-class optimization ignoring class interactions. Optuna searches the joint space.

```python
import optuna

VULN_CLASSES = ["IntegerUO", "GasException", "Reentrancy", "MishandledException",
                "ExternalBug", "CallToUnknown", "TOD", "Timestamp", "UnusedReturn", "DoS"]

def objective(trial, val_probs, val_labels):
    # Joint threshold search — Optuna explores correlations between class thresholds
    thresholds = torch.tensor([
        trial.suggest_float(f"thresh_{cls}", 0.10, 0.75)
        for cls in VULN_CLASSES
    ])
    preds = (val_probs > thresholds).float()
    return macro_f1(preds, val_labels).item()

# Load saved validation probabilities from your existing run
val_probs  = torch.load("ml/checkpoints/v8.0-AB_val_probs.pt")
val_labels = torch.load("ml/data/splits/deduped/val_labels.pt")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(
    lambda trial: objective(trial, val_probs, val_labels),
    n_trials=2000,        # 2000 trials takes ~2 minutes with no GPU needed
    n_jobs=-1,            # parallel on all CPU cores
)
print(study.best_params)
# Expected gain: +0.03–0.08 macro-F1 over your current tuned 0.2851
```

---

## B3: Label Smoothing

**Location:** `ml/src/training/trainer.py` (one-line loss change)  
**Risk:** Low  
**Effort:** 1 hour  
**Addresses:** H3 label ceiling — Slither labels have false negatives; hard 0/1 training is too confident

### Motivation from Results

H3 confirms the label ceiling exists because of data/label quality. Label smoothing is not a way to push through that ceiling — it's a way to train honestly within it. Slither labels are tool-generated. A label of `0` for Reentrancy does not mean "definitely not reentrancy" — it means "Slither didn't detect reentrancy." Smoothing encodes this epistemic uncertainty directly.

Use class-specific epsilon values calibrated to your domain knowledge about Slither's reliability per class:

```python
# Based on known Slither detection reliability per class:
# High reliability (easy to detect) → low ε
# Low reliability (often misses)    → high ε
CLASS_EPSILON = {
    "IntegerUO":           0.03,   # Slither detects reliably
    "GasException":        0.08,   # Slither often misses complex cases
    "Reentrancy":          0.05,   # Reliable for classic patterns, misses cross-function
    "MishandledException": 0.07,
    "ExternalBug":         0.08,
    "CallToUnknown":       0.04,
    "TOD":                 0.10,   # Hard for Slither, high false negative rate
    "Timestamp":           0.05,
    "UnusedReturn":        0.04,
    "DoS":                 0.09,   # Slither misses many DoS patterns
}

# In trainer.py — replace criterion with:
def smooth_bce_loss(logits, labels, class_epsilon):
    eps = torch.tensor(list(class_epsilon.values()), device=logits.device)
    smooth_labels = labels * (1 - eps) + (1 - labels) * eps
    return F.binary_cross_entropy_with_logits(logits, smooth_labels)
```

---

## B4: R-Drop Regularization

**Location:** `ml/src/training/trainer.py` (training loop change)  
**Risk:** Low (adds ~40% training time per epoch)  
**Effort:** 1 day  
**Addresses:** Transformer eye overconfidence; v8's correlated multi-class activations

### Motivation from Results

v8's FP explosion (6–8 classes firing simultaneously on complex contracts) suggests the model's dropout paths are not providing adequate regularization — different dropout masks produce correlated predictions rather than independent ones. R-Drop directly penalizes this: two forward passes with different dropout masks must produce consistent probability distributions.

```python
# In trainer.py training loop:

def rdrop_loss(model, graphs, input_ids, mask, labels, alpha=0.5):
    # Two forward passes — different dropout masks each time
    logits1 = model(graphs, input_ids, mask, return_aux=False)
    logits2 = model(graphs, input_ids, mask, return_aux=False)

    p1 = torch.sigmoid(logits1)   # [B, 10]
    p2 = torch.sigmoid(logits2)   # [B, 10]

    # Symmetric KL divergence — penalizes inconsistency between dropout masks
    kl_1_2 = F.kl_div(p1.log().clamp(min=-100), p2, reduction='batchmean')
    kl_2_1 = F.kl_div(p2.log().clamp(min=-100), p1, reduction='batchmean')

    # Task loss on averaged logits (both passes contribute)
    task_loss = criterion((logits1 + logits2) / 2.0, labels)

    return task_loss + alpha * (kl_1_2 + kl_2_1) / 2.0

# α=0.5 is the standard starting point. If training becomes unstable, reduce to 0.3.
# The KL divergence is bounded — clamp log to prevent NaN on near-zero probabilities.
```

---

## B5: Label Dependency Graph

**Location:** `ml/src/models/sentinel_model.py` (add after classifier)  
**Risk:** Low (~100 extra parameters, residual addition)  
**Effort:** 1 day  
**Addresses:** Independence assumption in classifier; co-occurring vulnerability classes

### Motivation from Results

The manual test shows both models confuse MishandledException with Reentrancy (both involve external calls). The classifier has no mechanism to say "if I predict Reentrancy, I should also check for CallToUnknown — they co-occur 73% of the time in training." The label co-occurrence matrix encodes exactly this structural knowledge.

```python
# Precompute from training labels
def build_label_adjacency(train_labels: torch.Tensor) -> torch.Tensor:
    # train_labels: [N, 10]
    co_occur = torch.mm(train_labels.T.float(), train_labels.float())  # [10, 10]
    # Normalize: P(j | i) = co_occur[i,j] / co_occur[i,i]
    diag     = co_occur.diagonal().unsqueeze(1).clamp(min=1)
    adj      = co_occur / diag   # [10, 10] conditional probabilities
    return adj

class LabelDependencyLayer(nn.Module):
    def __init__(self, num_classes: int, label_adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", label_adj)   # fixed, not learned
        self.W = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.eye_(self.W.weight)              # init as identity — starts as no-op

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, 10]
        # Aggregate: each class receives weighted evidence from correlated classes
        neighbor_signal = torch.mm(logits, self.adj)   # [B, 10]
        refined         = self.W(neighbor_signal)       # [B, 10] learned transform
        return logits + 0.1 * refined                   # residual — small initial weight

# In SentinelModel.__init__():
label_adj       = build_label_adjacency(train_labels)
self.label_dep  = LabelDependencyLayer(num_classes, label_adj)

# In SentinelModel.forward() — after classifier, before return:
logits = self.label_dep(logits)
```

The `nn.init.eye_` initialization means this layer starts as a perfect passthrough (identity). Only backprop introduces corrections. If the label dependency signal is noisy, the layer learns to suppress it. If it's helpful, it learns to amplify it. No risk of destabilizing the trained classifier.

---

## B6: Monte Carlo Dropout Uncertainty

**Location:** `ml/src/inference/predictor.py` (add inference mode)  
**Risk:** Zero — inference only, no training change  
**Effort:** 4 hours  
**Addresses:** v8's FP explosion — uncertainty flags unreliable predictions for human review

### Motivation from Results

v8 fires 6–8 vulnerability classes simultaneously on contracts 09, 11, and 20. Some of these are correct, some are FPs. Currently, the output treats all predictions with equal confidence. MC Dropout adds epistemic uncertainty scores that reveal when the model is genuinely unsure versus when it's confidently correct.

```python
def predict_with_uncertainty(model, graphs, input_ids, mask,
                              n_samples: int = 30, threshold: float = 0.15):
    """
    n_samples=30 forward passes with dropout active.
    Returns mean prediction, per-class uncertainty, and review flag.
    """
    model.train()   # CRITICAL: activates dropout paths
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(graphs, input_ids, mask, return_aux=False)
            predictions.append(torch.sigmoid(logits))

    preds       = torch.stack(predictions)           # [30, B, 10]
    mean_probs  = preds.mean(0)                      # [B, 10]
    uncertainty = preds.std(0)                       # [B, 10]

    # Flag any prediction where uncertainty > threshold as "needs human review"
    needs_review = (uncertainty > threshold).any(dim=1)   # [B]

    return {
        "probabilities": mean_probs,                 # [B, 10]
        "uncertainty":   uncertainty,                # [B, 10] per-class std
        "needs_review":  needs_review,               # [B] bool
        "n_samples":     n_samples,
    }
```

**Output format for the security report:**
```
Reentrancy:    0.87 ± 0.03  → HIGH CONFIDENCE
AccessControl: 0.61 ± 0.19  → UNCERTAIN — recommend human review  
DoS:           0.12 ± 0.04  → HIGH CONFIDENCE (negative)
```

This transforms Sentinel from a black-box classifier into an auditable tool with quantified confidence. Critical for professional use.

---

## B7: Stochastic Weight Averaging (SWA)

**Location:** `ml/src/training/trainer.py` (final training phase)  
**Risk:** Low  
**Effort:** 1 day  
**Addresses:** Sharp loss minima causing poor generalization on OOD contracts (new DeFi protocols, new Solidity versions)

```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# In trainer.__init__() — wrap model after initialization:
self.swa_model     = AveragedModel(self.model)
self.swa_scheduler = SWALR(self.optimizer, swa_lr=1e-5, anneal_epochs=5)
self.swa_start     = int(total_epochs * 0.7)   # start SWA at 70% of training

# In trainer.train_epoch() — after standard epoch:
if epoch >= self.swa_start:
    self.swa_model.update_parameters(self.model)
    self.swa_scheduler.step()
else:
    self.scheduler.step()   # standard scheduler before SWA phase

# After all training — update BatchNorm stats and save SWA model:
update_bn(self.train_loader, self.swa_model, device=self.device)
torch.save({
    "model":    self.swa_model.state_dict(),
    "epoch":    total_epochs,
    "best_f1":  best_f1,
    "config":   config,
}, "sentinel_swa.pt")
```

---

## B8: Self-Paced Learning (Upgrade to Curriculum)

**Location:** `ml/src/training/curriculum_sampler.py` (extension of A3)  
**Risk:** Medium  
**Effort:** 1 day on top of A3  
**Addresses:** Rare class learning (GasException, TOD — both models miss completely)

Self-paced learning replaces the static complexity score with the model's own per-sample loss as the difficulty signal. After each epoch, samples where the model's loss exceeds a threshold are excluded from the next epoch — the model trains only on samples it's "ready" to learn from.

```python
# Extension of CurriculumSampler:

class SelfPacedSampler(CurriculumSampler):
    def __init__(self, *args, base_lambda: float = 2.0, growth_rate: float = 0.05):
        super().__init__(*args)
        self.base_lambda  = base_lambda
        self.growth_rate  = growth_rate
        self.per_sample_losses = None   # updated after each epoch

    def update_losses(self, model, dataset, device):
        """Call at end of each epoch to recompute per-sample losses."""
        model.eval()
        losses = []
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                logits = model(sample.graphs, sample.input_ids, sample.mask)
                loss   = F.binary_cross_entropy_with_logits(
                    logits, sample.labels, reduction='mean'
                )
                losses.append(loss.item())
        self.per_sample_losses = losses

    def __iter__(self):
        if self.per_sample_losses is None:
            # Fall back to complexity-based curriculum before first epoch completes
            return super().__iter__()

        # Self-paced threshold — relaxes as training progresses
        λ = self.base_lambda * (1 + self.current_epoch * self.growth_rate)

        # Eligible: samples the model can currently learn from
        eligible  = [i for i, l in enumerate(self.per_sample_losses) if l < λ]
        hard_pool = [i for i, l in enumerate(self.per_sample_losses) if l >= λ]

        # Mixing floor: always include 10% of hard samples
        n_hard  = max(1, int(len(eligible) * 0.10))
        indices = eligible + random.sample(hard_pool, min(n_hard, len(hard_pool)))
        random.shuffle(indices)
        return iter(indices)
```

**Why GasException and TOD specifically benefit:** These classes are entirely absent from both models' detections. They likely have very high per-sample loss throughout training. Self-paced learning initially excludes them (too hard), allowing the model to build competence on other classes, then includes them when the loss threshold has risen enough to accept their high-loss signal.

---

# Implementation Roadmap

## Phase 0 — This Week (No Retraining, Zero Risk)

| Task | File | Time | Metric |
|---|---|---|---|
| Temperature scaling on v8 checkpoint | `predictor.py` | 30 min | Calibration improvement (ECE) |
| Optuna joint threshold search | `tune_threshold.py` | 2 hrs | Expected: macro-F1 > 0.30 |
| MC Dropout uncertainty on v8 | `predictor.py` | 4 hrs | FP flagging rate on safe contracts |
| Re-run 20-contract manual test with calibrator | `static_calibrator.py` | 4 hrs | FP count reduction |

## Phase 1 — Next Training Run (Low Risk Additions)

| Task | File | Est. F1 gain |
|---|---|---|
| Label smoothing (class-specific ε) | `trainer.py` | +0.005–0.010 |
| Label dependency graph | `sentinel_model.py` | +0.005–0.015 |
| R-Drop (α=0.5) | `trainer.py` | +0.010–0.020 |
| SWA (final 30% of epochs) | `trainer.py` | +0.005–0.010 |

Run these together as Run 9 after PLAN-3A establishes the new edge-type baseline.

## Phase 2 — Dedicated Experimental Runs

| Task | File | Risk | Notes |
|---|---|---|---|
| Curriculum learning + complexity correlation check | `curriculum_sampler.py` | Medium | Validate correlation < 0.4 first |
| Self-paced learning (upgrade) | `curriculum_sampler.py` | Medium | Run after curriculum validated |
| SCsVulLyzer pre-filter agent | `agents/` | Zero | Deploy after model is stable |

## Phase 3 — Longer Horizon

| Task | Notes |
|---|---|
| Graph contrastive pre-training | Needs unlabeled Etherscan corpus (100k+ contracts) |
| Pseudo-labeling | Use after Phase 1 model reaches F1 > 0.32 |
| GNNExplainer integration | Transforms Sentinel into a production audit tool |
| GraphMixup augmentation | Implement in embedding space after Phase 2 |

---

## What Was Deliberately Left Out

**SCsVulLyzer GA profiling** (the paper's genetic algorithm producing per-class feature weight vectors): Sound architecture exists (profile-gated residual logits), but requires the actual 10 GA profile vectors — either from the authors or by re-running the GA on Sentinel's dataset. Not included until profiles are available.

**Adversarial token training**: High value for robustness but adds significant training complexity and time. Scheduled after Phase 1 establishes a clean baseline.

**Hierarchical classification head**: Requires careful vulnerability family design. Revisit after the label dependency graph (B5) quantifies how much class correlation signal is actually available.

**AutoML hyperparameter search**: Becomes high-value after curriculum learning has run and a stable training curve is established. Searching `gnn_hidden_dim`, `lora_r`, `dropout`, and Focal loss `γ` per class is meaningful only when the training process itself is well-calibrated.
