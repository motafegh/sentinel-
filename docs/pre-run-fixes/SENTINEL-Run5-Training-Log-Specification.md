# SENTINEL Run 5 — Training Log Specification

> **Purpose**: Exhaustive checklist of everything to log during Run 5 training to guarantee full observability, early anomaly detection, and post-mortem capability. Every item here maps to a known bug, RC, or risk identified in the unified preflight proposal.
>
> **Architectural Context**: The ML module serves as a **probabilistic input signal** for the downstream agent module. The agent consumes vulnerability **probabilities**, not hard classifications. This means probability quality (ranking, calibration, discriminative power) is MORE important than threshold-dependent metrics like F1. AUC metrics and calibration measures directly reflect how useful the ML output will be to the agent.

---

## 1. Data Integrity & Label Health

These logs are **non-negotiable** given A20 (label=0 poisoning) and the archive/data-integrity requirement.

| # | Log Item | What to Track | Frequency | Rationale |
|---|----------|---------------|-----------|-----------|
| 1.1 | Label distribution per batch | Count of each class label in every batch | Every step | Catches A20-style poisoning — if any batch is 100% label=0, alarm immediately |
| 1.2 | Label distribution delta | Compare current epoch's label distribution against baseline from v9 extraction | Every epoch | Detects data drift or corrupted loaders after archive swap |
| 1.3 | Feature matrix sanity | `graphs.x.min()`, `graphs.x.max()`, `graphs.x.mean()`, `graphs.x.std()` per batch | Every step | Catches NaN/Inf infiltration into features before model sees them |
| 1.4 | Edge index validity | `edge_index.max() < num_nodes`, no self-loops unless expected, no duplicate edges | Every step | Catches graph corruption from re-extraction (v8→v9) |
| 1.5 | NaN/Inf scan (pre-forward) | Scan `x`, `edge_index`, `edge_attr`, `batch`, `y` before each forward pass | Every step | A38 prevention — catch corruption at the source before it enters the model |
| 1.6 | Graph size distribution | `num_nodes`, `num_edges` per graph in batch (min/mean/max/p95) | Every step | Detects outlier graphs that could blow VRAM; relates to Gate 5.3 |
| 1.7 | Feature dimension check | Assert `graphs.x.shape[1] == expected_dim` | Every N steps (e.g., every 50) | Catches silent dimension mismatch from re-extraction or def_map changes |
| 1.8 | Data loader integrity hash | Hash or checksum of dataset files used at training start | Once at startup | Confirms we are training on the correct v9 data, not stale v8 or cached data |
| 1.9 | Archive verification log | Confirm archive directory contains all expected v8 artifacts before v9 becomes active | Once at startup | Fulfills archive/data-integrity requirement — ensures nothing was lost in migration |

---

## 2. NaN/Inf & Gradient Health

Given A38 (NaN corrupts Adam state permanently), this section is **kill-run critical**.

| # | Log Item | What to Track | Frequency | Rationale |
|---|----------|---------------|-----------|-----------|
| 2.1 | Loss NaN/Inf check | Check `loss.item()` for NaN/Inf after `loss.backward()` | Every step | First line of defense for A38 — if loss is NaN, everything downstream is suspect |
| 2.2 | Parameter NaN/Inf scan | Scan ALL `param.data` for NaN/Inf | Every N steps (e.g., every 100) | If Adam state is corrupted, parameters go NaN — early detection before it cascades |
| 2.3 | Gradient norm per layer | `param.grad.norm()` for each named parameter | Every N steps (e.g., every 50) | Spot which layer first produces NaN/Inf gradients — root cause identification |
| 2.4 | Gradient norm total | `total_grad_norm = sqrt(sum(p.grad.norm()**2 for p in params))` | Every step | Global gradient health — sudden spikes signal instability before NaN appears |
| 2.5 | Gradient zero count | Count of parameters with `grad.norm() == 0` | Every N steps (e.g., every 50) | Detects dead gradients — A6/A10/A18 silent failure territory |
| 2.6 | Adam state monitoring | Check `exp_avg` and `exp_avg_sq` for NaN/Inf in optimizer state dict | Every N steps (e.g., every 200) | A38 specifically — if Adam's running averages are corrupted, you MUST restart the run |
| 2.7 | Loss spike counter | Track loss values exceeding `median_loss * spike_threshold` (e.g., 5x) | Every step | Sudden spikes often precede NaN blowups; logging spike frequency gives early warning |
| 2.8 | Gradient norm history (rolling) | Rolling mean and std of total gradient norm over last 100 steps | Every step | Context for spike detection — a "spike" is only meaningful relative to recent history |

---

## 3. Training Dynamics

| # | Log Item | What to Track | Frequency | Rationale |
|---|----------|---------------|-----------|-----------|
| 3.1 | Loss components separately | `main_loss`, `aux_phase2_loss`, `cei_loss` (if applicable), `total_loss` | Every step | If aux_phase2 dominates or vanishes, total loss alone will not reveal it |
| 3.2 | Loss ratio tracking | `aux_loss / main_loss`, `cei_loss / main_loss` | Every step | Catches loss imbalance that total loss masks; aux_head collapse detection |
| 3.3 | Learning rate (actual) | Log `lr` from scheduler | Every step | NF-4 style CLI mismatches — verify the CLI arg actually took effect at runtime |
| 3.4 | Learning rate schedule confirmation | Log scheduler state (milestone, warmup progress) | Every milestone step | Ensures warmup/decay is happening as configured, not silently overridden |
| 3.5 | Per-class metrics | Precision, Recall, F1 per class | Every epoch | Some classes may silently degrade while macro average looks fine |
| 3.6 | Confusion matrix snapshots | Full confusion matrix | Every N epochs (e.g., every 5) | Reveals systematic misclassification patterns that per-class metrics miss |
| 3.7 | AUROC per label | AUROC for each individual label in multilabel setup | Every epoch | Threshold-independent ranking quality per label — the most important metric for agent consumption (see Section 3B) |
| 3.8 | Prediction entropy | Mean entropy of `softmax(logits)` per batch | Every step | Collapsed predictions (all same class) = low entropy = model not learning; also indicates probability quality degradation |
| 3.9 | Confidence calibration error (ECE) | Expected Calibration Error | Every epoch | Directly tied to temperature scaling — if ECE is not improving, calibration is broken; critical for agent probability trust |
| 3.10 | Training speed | Steps/sec, samples/sec | Every step | Performance regression detection — if speed drops, something is wrong (OOM fallback, dataloader stall) |
| 3.11 | Epoch duration | Wall-clock time per epoch | Every epoch | Baseline for performance comparison; sudden increases signal problems |

---

## 3B. AUC & Probability Quality Metrics (Critical for Agent Module Input)

**Why this section exists as a separate category**: The ML module outputs vulnerability **probabilities** that the agent module consumes as input signals. The agent does not make hard 0/1 vulnerability calls from ML — it reasons over the probability gradient. This makes **threshold-independent** and **calibration** metrics far more important than threshold-dependent metrics like F1. A model with mediocre F1 at threshold=0.5 but excellent probability ranking is significantly more useful to the agent than the reverse. These metrics directly answer: "Can the agent trust these probability values?"

| # | Log Item | What to Track | Frequency | Rationale |
|---|----------|---------------|-----------|-----------|
| 3B.1 | AUC-ROC per label | Area Under the ROC Curve for each individual vulnerability label | Every epoch | Core ranking quality metric — measures how well the model ranks vulnerable above non-vulnerable contracts per label, independent of threshold. The agent needs good ranking to prioritize which contracts to analyze deeply |
| 3B.2 | AUC-PR per label | Area Under the Precision-Recall Curve for each individual label | Every epoch | **More important than AUC-ROC for imbalanced labels.** Vulnerability detection is typically class-imbalanced (far more safe than vulnerable). AUC-ROC can look misleadingly good on imbalanced data — a model that always predicts "safe" gets high ROC-AUC. AUC-PR punishes models that ignore the minority (vulnerable) class |
| 3B.3 | Macro-averaged AUC-ROC | Mean of per-label AUC-ROC values, treating each label equally | Every epoch | Single-number summary — detects if AUC is good on average but terrible on specific labels. A label-specific collapse would be hidden by micro-average |
| 3B.4 | Macro-averaged AUC-PR | Mean of per-label AUC-PR values | Every epoch | Same as 3B.3 but for precision-recall — more honest about performance on rare vulnerability types |
| 3B.5 | Micro-averaged AUC-ROC | Aggregate all label predictions into one pool, then compute AUC-ROC | Every epoch | Overall system ranking quality — better for measuring total agent-input quality when some labels are very rare |
| 3B.6 | Micro-averaged AUC-PR | Aggregate all label predictions into one pool, then compute AUC-PR | Every epoch | Harsh but honest overall metric — if this is low, the agent cannot trust the probabilities across the board |
| 3B.7 | Brier Score per label | Mean squared difference between predicted probability and actual label, per label | Every epoch | Directly measures **probability calibration per label** — if the model says "70% vulnerable," Brier score checks if those contracts are actually vulnerable ~70% of the time. This is what the agent actually needs: calibrated probabilities it can trust as confidence signals |
| 3B.8 | Brier Score (overall) | Mean Brier Score across all labels | Every epoch | Single-number calibration quality — if this is high, the agent is receiving miscalibrated probability signals and may over/under-trust them |
| 3B.9 | Brier Score decomposition | Decompose into reliability, resolution, and uncertainty components | Every N epochs (e.g., every 5) | Reliability = calibration quality (lower is better). Resolution = discriminative power (higher is better). Uncertainty = inherent label uncertainty. Tells you whether bad Brier Score is from poor calibration or poor discrimination — very different problems with very different fixes |
| 3B.10 | Probability distribution stats | Min, max, mean, std, p5, p50, p95 of raw probability outputs per label | Every epoch | Detects probability collapse — if all outputs cluster near 0.5, the model is uninformative. If they cluster near 0 or 1, the model is overconfident. The agent needs a meaningful probability gradient |
| 3B.11 | Reliability diagram data | Bin predictions by probability, compare predicted vs actual frequency per bin | Every N epochs (e.g., every 5) | Visual calibration diagnostic — perfect calibration means the diagonal line. Essential for understanding where the model is over/under-confident |
| 3B.12 | AUC-ROC delta (epoch over epoch) | Change in AUC-ROC per label compared to previous epoch | Every epoch | If AUC is degrading for any label, detect it immediately — don't wait for the full training run to discover regression |
| 3B.13 | AUC-PR delta (epoch over epoch) | Change in AUC-PR per label compared to previous epoch | Every epoch | Same as 3B.12 but for PR — more sensitive to minority-class regression |
| 3B.14 | Partial AUC (pAUC) at low FPR | AUC in the low false-positive-rate region (e.g., FPR 0–0.1) | Every N epochs | If the agent only acts on high-confidence vulnerability predictions (low FPR regime), this is the metric that matters most — full AUC can be dominated by the uninteresting high-FPR region |
| 3B.15 | F1 vs AUC divergence | Flag when F1 improves but AUC degrades (or vice versa) | Every epoch | If F1 improves at a specific threshold but AUC degrades, the model is getting better at one threshold but worse at ranking overall — dangerous for agent consumption because the agent uses all thresholds |
| 3B.16 | Per-label positive rate vs prediction rate | Compare actual label frequency with model's mean predicted probability per label | Every epoch | If a label appears in 5% of samples but the model outputs mean 50% probability for it, the model is systematically miscalibrated for that label — the agent will over-react to it |

### Why AUC-PR Can Save Your Run

Consider this scenario: You have 1000 contracts, 50 are actually vulnerable (5% prevalence).

- **Model A**: Predicts 0.51 probability for vulnerable contracts, 0.49 for safe ones. AUC-ROC = 0.72 (looks okay). AUC-PR = 0.08 (terrible — the model barely separates). F1 at 0.5 threshold = 0.0 (predicts everything as vulnerable).
- **Model B**: Predicts 0.8 for vulnerable, 0.2 for safe. AUC-ROC = 0.95. AUC-PR = 0.62. F1 at 0.5 threshold = 0.73.

AUC-ROC alone would say Model A is "okay" — but the agent would receive near-random probability signals. AUC-PR immediately reveals that Model A is useless for the agent. **Always track both.**

### Why Brier Score Matters for Agent Trust

The agent module reasons over probability values. If the model says "90% vulnerable" but only 40% of those contracts are actually vulnerable, the agent will over-trust the ML signal and waste resources chasing false positives. Brier score directly measures this mismatch. Combined with temperature calibration (Section 5), you have a complete pipeline for ensuring the agent receives trustworthy probability signals.

---

## 4. Model-Specific Logs (GNN + Transformer Fusion)

| # | Log Item | What to Track | Frequency | Rationale |
|---|----------|---------------|-----------|-----------|
| 4.1 | JK weight monitoring | Jumping Knowledge attention weight values per layer | Every N steps (e.g., every 50) | If JK weights collapse to a single layer, the fusion mechanism is broken |
| 4.2 | JK weight entropy | Shannon entropy of JK attention weights | Every N steps | Single number summary — low entropy = JK not blending layers, just picking one |
| 4.3 | `aux_head_phase2.weight.norm()` | L2 norm of aux head weight matrix | Every N steps (e.g., every 50) | Suggestion 11 — if this decays to near zero, aux head is disconnected from learning |
| 4.4 | `aux_head_phase2.bias.norm()` | L2 norm of aux head bias vector | Every N steps (e.g., every 50) | Companion to weight norm — bias collapse is also a signal of disconnection |
| 4.5 | Per-layer GNN output norms | `norm()` of each GNN layer's output representation | Every N steps (e.g., every 100) | Detects which layer first produces degenerate representations; identifies layer collapse |
| 4.6 | Transformer attention entropy | Mean attention entropy across all heads per layer | Every N steps (e.g., every 100) | If attention becomes uniform or concentrated on a single position, something is wrong |
| 4.7 | Fusion gate values | Distribution of gate values if fusion uses gating | Every N steps (e.g., every 100) | Ensures both GNN and transformer paths contribute meaningfully |
| 4.8 | Embedding drift | L2 distance between current and initial (epoch 0) embeddings at key layers | Every epoch | Detects representation collapse or stagnation over training |
| 4.9 | Weight norm per layer | `param.data.norm()` for all key layers | Every N steps (e.g., every 100) | Exploding or vanishing weights — early warning before NaN appears |
| 4.10 | Weight update ratio | `param.grad.norm() / param.data.norm()` per layer | Every N steps (e.g., every 100) | If ratio is consistently near zero, learning has stalled for that layer |

---

## 5. Temperature Calibration Logs

Temperature must NOT be reused across runs — these logs confirm fresh calibration.

| # | Log Item | What to Track | Frequency | Rationale |
|---|----------|---------------|-----------|-----------|
| 5.1 | Temperature value | Log `T` value at calibration step | At calibration time | Must confirm T is freshly computed, not carried over from previous run |
| 5.2 | Pre-calibration ECE | ECE before temperature scaling | At calibration time | Baseline for comparison — without this you cannot measure improvement |
| 5.3 | Post-calibration ECE | ECE after temperature scaling | At calibration time | Should decrease from pre-calibration; if it increases, calibration is broken |
| 5.4 | Temperature stability | If running calibration multiple times, log T variance | At calibration time | If T varies wildly across runs, calibration data or validation set is unstable |
| 5.5 | NLL before/after | Negative log-likelihood before and after T scaling | At calibration time | Alternative calibration quality metric; should decrease |
| 5.6 | Calibration dataset hash | Hash of the validation split used for calibration | Once at calibration time | Ensures calibration is not accidentally done on training data or wrong split |
| 5.7 | Temperature not loaded from checkpoint | Explicit boolean confirming T was computed, not loaded | At calibration time | Prevents accidental reuse of temperature from a previous run's checkpoint |

---

## 6. Resource & VRAM Logs (Critical for RTX 3070 8GB)

These directly support Gate 5.3 and your max_nodes=2048 VRAM constraint.

| # | Log Item | What to Track | Frequency | Rationale |
|---|----------|---------------|-----------|-----------|
| 6.1 | VRAM allocated | `torch.cuda.memory_allocated()` in MB | Every N steps (e.g., every 10) | Track against 8GB ceiling; identify which training phase consumes most memory |
| 6.2 | VRAM reserved | `torch.cuda.memory_reserved()` in MB | Every N steps | PyTorch caching allocator can hide fragmentation — reserved but unused VRAM matters |
| 6.3 | Peak VRAM per epoch | `torch.cuda.max_memory_allocated()` reset each epoch | Every epoch | Worst-case VRAM per epoch — the single most important number for OOM risk |
| 6.4 | VRAM per component estimate | Model params, optimizer states, gradient buffers, activation memory estimates | Once at startup + every epoch | Dissect which component is eating VRAM when approaching limit |
| 6.5 | GPU utilization % | Via `pynvml` or `nvidia-smi` | Every N steps (e.g., every 50) | Low utilization + high VRAM = bottleneck (likely dataloader, not compute) |
| 6.6 | GPU temperature | Hardware GPU temperature in Celsius | Every N steps | Thermal throttling can cause silent performance degradation |
| 6.7 | CPU RAM usage | `psutil.Process().memory_info().rss` | Every epoch | DataLoader workers + cached graphs can blow system RAM too |
| 6.8 | DataLoader queue depth | Number of prefetched batches in queue | Every N steps (e.g., every 100) | If queue is always empty, dataloader is bottleneck and GPU is starved |
| 6.9 | DataLoader worker status | Number of active workers, any worker crashes | Every epoch | Worker crashes cause silent data loss or hanging |

---

## 7. Graph-Specific Logs

These are specific to your PyG Data objects, CEI labeling, and the v8→v9 re-extraction.

| # | Log Item | What to Track | Frequency | Rationale |
|---|----------|---------------|-----------|-----------|
| 7.1 | Graph-level feature stats | Per-graph mean/std of node features | Every epoch | Catches individual graphs with degenerate features that batch stats might average out |
| 7.2 | Node degree distribution | Histogram of node degrees per batch | Every epoch | A14 (RETURN_TO edges) — unusual degree spikes could indicate edge construction bugs |
| 7.3 | select_prefix_nodes output stats | Number of nodes selected per graph, which indices chosen | Every N steps (e.g., every 100) | A34 access path — verify prefix selection is working correctly and not selecting degenerate subsets |
| 7.4 | def_map construction audit | `def_map` size, key collision count, duplicate function name count | Once at startup + every epoch if rebuilt | NF-10 synthetic key — verify no silent overwrites from duplicate function names |
| 7.5 | CEI label distribution | Per-contract CEI label counts (Checks/Effects/Interactions) | Every epoch | Concern 4 — must verify CEI labels computed on v9 graphs are correct and not stale |
| 7.6 | Edge type distribution | Count per edge type (including RETURN_TO) | Every epoch | A14 — verify RETURN_TO edges are present, correct, and in expected proportions |
| 7.7 | Number of graphs per contract | Track distribution of graphs per contract address | Every epoch | Anomalous counts (e.g., one contract with 1000 graphs) could indicate a data bug |
| 7.8 | Re-extraction label comparison | Compare label distribution in v9 vs archived v8 baseline | Once after re-extraction | Suggestion 9 — confirms re-extraction did not silently alter labels |

---

## 8. Epoch-Level Summary

At the end of every epoch, emit a **structured summary** capturing all key indicators in one record.

| # | Field | Type | Rationale |
|---|-------|------|-----------|
| 8.1 | `epoch` | int | Identifier |
| 8.2 | `train_loss` | float | Overall training loss |
| 8.3 | `val_loss` | float | Overall validation loss |
| 8.4 | `main_loss` | float | Main task loss component |
| 8.5 | `aux_loss` | float | aux_phase2 loss component |
| 8.6 | `total_loss` | float | Combined weighted loss |
| 8.7 | `lr` | float | Current learning rate |
| 8.8 | `grad_norm_total` | float | Global gradient norm |
| 8.9 | `grad_norm_max_layer` | str, float | Name and norm of layer with highest gradient norm |
| 8.10 | `param_nan_count` | int | Number of parameters containing NaN/Inf |
| 8.11 | `grad_nan_count` | int | Number of gradients containing NaN/Inf |
| 8.12 | `vram_peak_mb` | float | Peak VRAM this epoch |
| 8.13 | `vram_current_mb` | float | VRAM at end of epoch |
| 8.14 | `label_dist_train` | dict | Label distribution in training set this epoch |
| 8.15 | `label_dist_val` | dict | Label distribution in validation set this epoch |
| 8.16 | `aux_weight_norm` | float | `aux_head_phase2.weight.norm()` |
| 8.17 | `aux_bias_norm` | float | `aux_head_phase2.bias.norm()` |
| 8.18 | `jk_weight_entropy` | float | Entropy of JK attention weights |
| 8.19 | `prediction_entropy_mean` | float | Mean prediction entropy across validation set |
| 8.20 | `per_class_f1` | dict | F1 score per class |
| 8.21 | `auc_roc_per_label` | dict | AUC-ROC per vulnerability label — core ranking quality for agent consumption |
| 8.22 | `auc_pr_per_label` | dict | AUC-PR per vulnerability label — more honest for imbalanced labels |
| 8.23 | `auc_roc_macro` | float | Macro-averaged AUC-ROC across all labels |
| 8.24 | `auc_pr_macro` | float | Macro-averaged AUC-PR across all labels |
| 8.25 | `brier_score_per_label` | dict | Brier Score per label — probability calibration quality |
| 8.26 | `brier_score_overall` | float | Overall Brier Score — aggregate calibration quality |
| 8.27 | `ece` | float | Expected Calibration Error |
| 8.28 | `temperature` | float | Current temperature scaling value |
| 8.29 | `prob_dist_stats` | dict | Min/max/mean/std/p5/p50/p95 of raw probability outputs per label |
| 8.30 | `auc_roc_delta` | dict | Epoch-over-epoch change in AUC-ROC per label — detect regression early |
| 8.31 | `auc_pr_delta` | dict | Epoch-over-epoch change in AUC-PR per label |
| 8.32 | `f1_auc_divergence` | bool | True if any label shows F1 improving while AUC degrades (or vice versa) |
| 8.33 | `epoch_duration_sec` | float | Wall-clock time for this epoch |
| 8.34 | `steps_per_epoch` | int | Number of optimization steps this epoch |
| 8.35 | `gpu_util_mean_pct` | float | Mean GPU utilization this epoch |
| 8.36 | `loss_spike_count` | int | Number of loss spikes detected this epoch |
| 8.37 | `grad_zero_count` | int | Number of parameters with zero gradients |

---

## 9. Alert-Grade Anomalies

These conditions trigger **immediate action** — not just logging.

### 9.1 KILL RUN (Immediate Abort)

| # | Condition | Rationale | Action |
|---|-----------|-----------|--------|
| 9.1.1 | `loss == NaN or Inf` | Cannot recover — A38 | Kill run, log state, archive checkpoint for post-mortem |
| 9.1.2 | Any `param.data` contains NaN/Inf | Adam state likely corrupted | Kill run, do NOT save this checkpoint |
| 9.1.3 | Any optimizer state (`exp_avg`, `exp_avg_sq`) contains NaN/Inf | Confirmed A38 — corrupted Adam state is permanent | Kill run, must restart from last clean checkpoint |

### 9.2 WARN + SKIP BATCH (Continue Training)

| # | Condition | Rationale | Action |
|---|-----------|-----------|--------|
| 9.2.1 | Batch label distribution is 100% single class | Possible A20 poisoning | Skip batch, log batch index and label stats, continue |
| 9.2.2 | Batch contains NaN/Inf in input data | Corrupted data sample | Skip batch, log batch index, continue |

### 9.3 WARN (Continue Training, Flag for Review)

| # | Condition | Rationale | Action |
|---|-----------|-----------|--------|
| 9.3.1 | `vram_peak > 7500 MB` (on 8GB GPU) | Approaching OOM | Log warning, consider reducing batch_size or max_nodes |
| 9.3.2 | `grad_norm_total > 100x rolling_mean` | Gradient explosion imminent | Log warning, consider reducing lr or clipping |
| 9.3.3 | `aux_head weight norm < 1e-6` | Aux head is disconnected from learning | Log warning, check loss weighting |
| 9.3.4 | `jk_weight entropy < 0.5` | JK fusion collapsed to single layer | Log warning, check GNN layer outputs |
| 9.3.5 | `prediction entropy < threshold` | Model predicting same class for everything | Log warning, check learning rate and data |
| 9.3.6 | `per-class F1 == 0` for any class | Class being completely ignored | Log warning, check class weights and label distribution |
| 9.3.6b | `AUC-PR < 0.1` for any label | Model cannot rank this label better than random | Log warning, label may be too rare or model is failing on it — agent will receive useless signal for this vulnerability type |
| 9.3.6c | `F1 improves but AUC-ROC degrades` for any label | Model improving at one threshold but worsening at ranking overall | Log warning, dangerous for agent consumption — the model is getting worse at the probabilities the agent actually uses |
| 9.3.6d | `Brier Score > 0.4` for any label (with binary labels) | Severe miscalibration — probabilities do not match reality | Log warning, agent cannot trust probability values for this label |
| 9.3.7 | `lr != expected_lr` at any point | NF-4 style mismatch may have occurred | Log warning, verify scheduler and CLI args |
| 9.3.8 | `edge_index.max() >= num_nodes` | Graph corruption detected | Log warning, check re-extraction pipeline |
| 9.3.9 | `graphs.x.shape[1] != expected_dim` | Feature dimension mismatch | Log warning, check def_map and feature pipeline |

---

## 10. Log Format & Infrastructure

### 10.1 Structured JSON Logging

Use three separate log streams for different granularity levels:

**File 1: `step_metrics.jsonl`** — Per-step granular data
```json
{"step": 1000, "epoch": 2, "total_loss": 0.342, "main_loss": 0.281, "aux_loss": 0.061, "lr": 0.0003, "grad_norm_total": 1.23, "vram_mb": 6234.5, "label_dist_batch": {"0": 12, "1": 8, "2": 5}, "prediction_entropy": 1.82, "duration_ms": 45}
```

**File 2: `epoch_summary.jsonl`** — One line per epoch (see Section 8 for full schema)
```json
{"epoch": 2, "train_loss": 0.342, "val_loss": 0.389, "main_loss": 0.281, "aux_loss": 0.061, "lr": 0.0003, ...}
```

**File 3: `alerts.jsonl`** — All warnings and kill events
```json
{"timestamp": "2026-06-02T14:32:01", "level": "WARN", "message": "VRAM peak exceeded 7500 MB", "vram_peak_mb": 7612.3, "epoch": 5, "step": 4200}
```

### 10.2 Logger Implementation Skeleton

```python
import json
from datetime import datetime
from pathlib import Path

class TrainingAbortError(Exception):
    """Raised when a KILL-level alert is triggered."""
    pass

class StructuredLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_log = open(self.log_dir / "step_metrics.jsonl", "a")
        self.epoch_log = open(self.log_dir / "epoch_summary.jsonl", "a")
        self.alert_log = open(self.log_dir / "alerts.jsonl", "a")

    def log_step(self, metrics: dict):
        self.step_log.write(json.dumps(metrics) + "\n")
        self.step_log.flush()

    def log_epoch(self, summary: dict):
        self.epoch_log.write(json.dumps(summary) + "\n")
        self.epoch_log.flush()

    def alert(self, level: str, message: str, data: dict = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        }
        if data:
            entry.update(data)
        self.alert_log.write(json.dumps(entry) + "\n")
        self.alert_log.flush()
        if level == "KILL":
            raise TrainingAbortError(f"KILL: {message}")

    def close(self):
        self.step_log.close()
        self.epoch_log.close()
        self.alert_log.close()
```

### 10.3 Sampling Strategy

Not every metric needs to be logged every step to avoid I/O overhead and massive log files:

| Granularity | Items | Example Frequency |
|---|---|---|
| **Every step** | Loss components, lr, grad_norm_total, vram, label_dist_batch, prediction_entropy | Every step |
| **Every N steps** | Per-layer grad norms, weight norms, aux head norms, JK weights, attention entropy | Every 50–100 steps |
| **Every epoch** | Per-class F1, AUC-ROC per label, AUC-PR per label, Brier Score, ECE, confusion matrix, probability distribution stats, AUC deltas, epoch summary | End of epoch |
| **Every N epochs** | Brier Score decomposition, reliability diagram data, partial AUC, confusion matrix | Every 5 epochs |
| **Once at startup** | Data integrity hash, archive verification, VRAM component estimates, def_map audit | Before training starts |
| **At calibration time** | Temperature, ECE pre/post, NLL pre/post, Brier Score pre/post | After calibration step |

### 10.4 Post-Run Analysis Checklist

After the run completes (or aborts), these logs should enable you to answer:

1. Was the correct v9 data used? (Check 1.8, 1.9)
2. Did any batch have poisoned labels? (Check 1.1, 9.2.1)
3. Did NaN/Inf ever appear? (Check 2.1–2.6, 9.1.1–9.1.3)
4. Was aux_phase2 learning? (Check 4.3, 4.4, 3.2)
5. Was JK fusion working? (Check 4.1, 4.2)
6. Did VRAM approach the limit? (Check 6.1–6.3, 9.3.1)
7. Was the learning rate correct? (Check 3.3, 3.4, 9.3.7)
8. Were CEI labels correct on v9 graphs? (Check 7.5, 7.8)
9. Was temperature freshly calibrated? (Check 5.1, 5.7)
10. Were all classes being learned? (Check 3.5, 3.7, 3.8, 9.3.6)
11. Are probability outputs trustworthy for the agent? (Check 3B.1–3B.16, 8.21–8.32)
12. Is calibration good enough for agent consumption? (Check 3B.7–3B.9, 3B.11, 5.1–5.7)
13. Is AUC improving or regressing? (Check 3B.12, 3B.13, 8.30, 8.31)
14. Is there F1-AUC divergence? (Check 3B.15, 8.32, 9.3.6c)

---

## 11. Cross-Reference: Log Items to Proposal Concerns

| Proposal Concern | Relevant Log Items |
|---|---|
| A20 (label=0 poisoning) | 1.1, 1.2, 9.2.1 |
| A38 (NaN corrupts Adam) | 2.1, 2.2, 2.6, 9.1.1, 9.1.2, 9.1.3 |
| A14 (RETURN_TO edges) | 7.2, 7.6 |
| A15 (scope collision) | 7.4 |
| A34 (select_prefix_nodes access path) | 7.3 |
| A6/A10/A18 (silent failures) | 2.5, 3.8 |
| NF-4 (CLI default mismatch) | 3.3, 3.4, 9.3.7 |
| NF-8 (empty-batch guard) | 1.6, 9.2.2 |
| NF-10 (synthetic key for duplicates) | 7.4 |
| CEI labeling on v9 graphs (Concern 4) | 7.5, 7.8 |
| VRAM Gate 5.3 (Concern 5) | 6.1–6.4, 9.3.1 |
| JK weight monitoring (Suggestion 11) | 4.1, 4.2 |
| Aux head monitoring (Suggestion 11) | 4.3, 4.4 |
| Pre/post re-extraction gate (Suggestion 9) | 7.8, 1.2 |
| Archive/data-integrity | 1.8, 1.9, 7.8 |
| Agent probability quality (AUC + calibration) | 3B.1–3B.16, 8.21–8.32, 9.3.6b–9.3.6d |
| Temperature calibration | 5.1–5.7, 3B.7–3B.9 |
