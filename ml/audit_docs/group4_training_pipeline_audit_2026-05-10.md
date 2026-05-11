# 🛡️ ML Code Audit Report - Group 4 (Training Pipeline)

**Audit Date:** 2026-05-10  
**Auditor:** AI Code Auditor  
**Group:** Training Pipeline  
**Status:** COMPLETE

---

## Executive Summary

**Modules Audited:**
- `src/training/trainer.py` (1,057 lines) - Core training loop with AMP, early stopping, MLflow integration
- `scripts/train.py` (274 lines) - CLI entry point for training
- `scripts/auto_experiment.py` (571 lines) - Autoresearch single-run wrapper with threshold tuning
- `scripts/run_overnight_experiments.py` (242 lines) - Sequential hyperparameter experiment launcher
- `src/training/focalloss.py` (73 lines) - Focal Loss implementation

**Overall Assessment:** ⚠️ **MODERATE-HIGH RISK** — Extensive audit fixes already applied, but several critical architectural issues remain around checkpoint compatibility, scheduler state management, and experiment tracking.

---

## Detailed Analysis

### 1. `src/training/trainer.py` (1,057 lines)

#### ✅ STRENGTHS IDENTIFIED

1. **Extensive Audit Trail**: 25+ documented audit fixes (Fix #1-25) with clear timestamps and reasoning
2. **Performance Optimizations**: AMP, TF32 matmuls, num_workers=2, pin_memory, zero_grad(set_to_none=True)
3. **Robust Resume Logic**: Handles model-only vs full resume, batch size mismatch detection, optimizer state validation
4. **Early Stopping**: Patience counter persistence via sidecar JSON files
5. **MLflow Integration**: Comprehensive parameter and metric logging
6. **Safety Checks**: Architecture validation, loss function validation, gradient clipping on trainable params only

#### 🔴 CRITICAL ISSUES

##### **Issue #1: Checkpoint Format Lock-in**
- **Location:** Lines 450-550 (resume logic)
- **Problem:** Checkpoint dict structure is rigid - any change to keys breaks backward compatibility
- **Risk:** Future refactors will break all existing checkpoints
- **Evidence:** Multiple guards like `if "optimizer" not in ckpt:` suggest fragile format
- **Fix Required:** Implement versioned checkpoint schema with migration functions

```python
# Current fragile approach:
if "optimizer" in ckpt:
    optimizer.load_state_dict(ckpt["optimizer"])

# Better approach:
CHECKPOINT_VERSION = 2
def migrate_checkpoint(old_ckpt):
    version = old_ckpt.get("checkpoint_version", 1)
    if version == 1:
        # Add missing keys with defaults
        old_ckpt["patience_counter"] = 0
        old_ckpt["checkpoint_version"] = 2
    return old_ckpt
```

##### **Issue #2: Sidecar File Race Condition**
- **Location:** Lines 145-148 (Fix #23 description)
- **Problem:** JSON sidecar file written after every epoch without atomic write or locking
- **Risk:** Corruption during crash mid-write, race condition with resume reading stale data
- **Fix:** Use atomic rename pattern (write to `.tmp`, then `os.rename()`)

##### **Issue #3: Scheduler State Mismatch Silent Degradation**
- **Location:** Lines 556-560
- **Problem:** When `total_steps` doesn't match, scheduler state is skipped with warning but training continues
- **Risk:** User may not notice suboptimal learning rate schedule for entire run
- **Fix:** Add `--strict-scheduler` flag that fails fast on mismatch, or auto-recalculate scheduler state

##### **Issue #4: Hardcoded Class Names in Module**
- **Location:** Lines 218-229
- **Problem:** CLASS_NAMES defined here AND potentially in other modules (graph_schema.py, tune_threshold.py)
- **Risk:** Inconsistency when adding/removing classes causes silent mislabeling
- **Fix:** Import from single source of truth (`ml.src.preprocessing.graph_schema`)

##### **Issue #5: Memory Leak Risk with RAM Cache**
- **Location:** DualPathDataset integration (lines 600-650)
- **Problem:** `cache_ram` dict grows unbounded across multiple train() calls in same process
- **Risk:** OOM when running multiple experiments sequentially (e.g., auto_experiment.py)
- **Fix:** Add cache eviction policy or explicit cache.clear() between runs

##### **Issue #6: Weighted Sampler Implementation Gap**
- **Location:** Lines 334-335 (use_weighted_sampler config)
- **Problem:** Config supports "DoS-only" and "all-rare" but actual sampler implementation not visible in audited code
- **Risk:** Silent fallback to uniform sampling if sampler creation fails
- **Fix:** Add assertion that weighted sampler was actually created when requested, log class weights used

#### ⚠️ BAD APPROACHES & IMPROVEMENTS

##### **Issue #7: Magic Numbers in Config Defaults**
- **Location:** TrainConfig dataclass (lines 250-336)
- **Examples:** 
  - `fusion_output_dim: int = 128` - Why 128?
  - `gnn_hidden_dim: int = 64` - Why 64?
  - `early_stop_patience: int = 7` - Arbitrary
- **Fix:** Add comments explaining rationale or derive from data characteristics

##### **Issue #8: No Validation of Config Consistency**
- **Location:** TrainConfig initialization
- **Problem:** Nothing prevents invalid combinations like:
  - `lora_r=64` with `lora_alpha=8` (alpha should typically be ≥ r)
  - `warmup_pct=0.5` with `epochs=5` (warmup longer than training)
- **Fix:** Add `__post_init__` validation method to TrainConfig

```python
def __post_init__(self):
    if self.warmup_pct * self.epochs < 1:
        raise ValueError(f"warmup_pct={self.warmup_pct} too high for epochs={self.epochs}")
    if self.lora_alpha < self.lora_r:
        logger.warning(f"lora_alpha ({self.lora_alpha}) < lora_r ({self.lora_r}) - unusual configuration")
```

##### **Issue #9: MLflow Experiment Name Collision**
- **Location:** Line 312 (`experiment_name: str = "sentinel-multilabel"`)
- **Problem:** Default experiment name shared across binary and multi-label modes
- **Risk:** Metrics from different model types mixed in same experiment
- **Fix:** Auto-generate experiment name based on num_classes or require explicit name

##### **Issue #10: Inconsistent Error Handling**
- **Location:** Throughout trainer.py
- **Problem:** Mix of `raise ValueError`, `logger.warning`, and silent fallbacks
- **Examples:**
  - Unknown loss_fn → raises ValueError (good)
  - Missing optimizer state → logs warning (okay)
  - Empty DataLoader → returns 0.0 with warning (Fix #18, okay but inconsistent)
- **Fix:** Define error handling policy: what must fail fast vs what can warn

##### **Issue #11: No Support for Gradient Accumulation**
- **Location:** Training loop (lines 700-800)
- **Problem:** Cannot simulate larger batch sizes on memory-constrained GPUs
- **Use Case:** RTX 3070 (8GB) cannot fit batch_size=64 but could do grad_accum=2 with batch_size=32
- **Fix:** Add `grad_accum_steps` parameter to TrainConfig

```python
# Pseudo-code addition:
for batch_idx, batch in enumerate(train_loader):
    loss = compute_loss(batch) / grad_accum_steps
    scaler.scale(loss).backward()
    
    if (batch_idx + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

##### **Issue #12: Threshold Hardcoded in Training**
- **Location:** Line 285 (`threshold: float = 0.5`)
- **Problem:** Training uses 0.5 threshold for F1 computation, but optimal threshold may differ
- **Risk:** Early stopping based on suboptimal threshold may discard better models
- **Fix:** Either (a) use tuned thresholds from previous run, or (b) use AUC/mAP for early stopping instead of F1@0.5

#### ✅ GOOD PRACTICES NOTED

1. **Comprehensive Docstrings**: Every major function has detailed docstring with args/returns
2. **Type Hints**: Full type annotations throughout
3. **Logging**: Structured logging with appropriate levels (INFO/WARNING/ERROR)
4. **Progress Bars**: tqdm integration for monitoring
5. **AMP Best Practices**: GradScaler usage, autocast context managers
6. **Resume Guide**: Clear documentation in train.py for different resume scenarios

---

### 2. `scripts/train.py` (274 lines)

#### ✅ STRENGTHS

1. **Excellent CLI Documentation**: Usage examples for all common scenarios
2. **Resume Mode Clarity**: Explicit flags for model-only vs full resume vs force-reset
3. **Thin Wrapper**: Minimal logic, delegates to trainer module

#### 🔴 CRITICAL ISSUES

##### **Issue #13: No Pre-flight Validation**
- **Location:** main() function (lines 234-274)
- **Problem:** Starts training without checking:
  - Disk space availability
  - GPU memory sufficiency
  - Data directory existence
  - Label CSV validity
- **Fix:** Add pre-flight checks before calling train()

```python
def preflight_checks(config: TrainConfig):
    # Check directories exist
    for dir_path in [config.graphs_dir, config.tokens_dir, config.splits_dir]:
        if not Path(dir_path).exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    # Check label CSV if multi-label mode
    if config.label_csv and not Path(config.label_csv).exists():
        raise FileNotFoundError(f"Label CSV not found: {config.label_csv}")
    
    # Check disk space (need ~2x checkpoint size free)
    # Check GPU memory (dry-run forward pass)
```

##### **Issue #14: No Dry-Run Mode**
- **Problem:** Cannot test configuration without committing to full training run
- **Use Case:** Validate new hyperparameters with 1 batch before overnight run
- **Fix:** Add `--dry-run` flag that processes 1 batch and exits

##### **Issue #15: Multiprocessing Start Method Hardcoded**
- **Location:** Line 74 (`mp.set_start_method('spawn', force=True)`)
- **Problem:** 'spawn' required for CUDA but may cause issues on some systems
- **Risk:** Silent performance degradation if spawn fails and falls back to fork
- **Fix:** Add try/except with clear error message, or make configurable

---

### 3. `scripts/auto_experiment.py` (571 lines)

#### ✅ STRENGTHS

1. **Clear Contract**: stdout format for agent parsing (SENTINEL_SCORE=, PEAK_VRAM_MB=)
2. **Exit Code Semantics**: 0=success, 1=preflight fail, 2=OOM, 3=tuning fail
3. **Two Regimes**: Smoke (fast validation) vs Confirm (full evaluation)
4. **Hash Locking**: Integration with compute_locked_hashes for reproducibility

#### 🔴 CRITICAL ISSUES

##### **Issue #16: Threshold Tuning Dependency**
- **Location:** Lines 58-67 (imports from tune_threshold.py)
- **Problem:** Entire experiment fails if threshold tuning crashes (exit code 3)
- **Risk:** Good model discarded due to tuning bug, not model quality
- **Fix:** Make threshold tuning optional, emit score with default 0.5 if tuning fails

##### **Issue #17: VRAM Tracking Accuracy**
- **Location:** Line 13 (PEAK_VRAM_MB output)
- **Problem:** Peak VRAM measurement timing unclear - during training or tuning?
- **Risk:** Inaccurate memory profiling leads to wrong hardware requirements
- **Fix:** Track peak separately for training phase and tuning phase

##### **Issue #18: Stratified Subsample Implementation**
- **Location:** Line 93 (`SMOKE_SUBSAMPLE = 0.10`)
- **Problem:** Unclear if 10% subsample maintains class distribution
- **Risk:** Smoke results misleading if rare classes excluded
- **Fix:** Verify stratified sampling preserves minimum samples per class

##### **Issue #19: Hardcoded Promotion Threshold**
- **Location:** Line 100 (`SMOKE_PROMOTE_THRESHOLD = 0.42`)
- **Problem:** Single threshold for all classes ignores per-class variance
- **Risk:** Promotes models good at easy classes, bad at rare ones
- **Fix:** Use per-class minimum thresholds or weighted macro-F1

##### **Issue #20: No Checkpoint Reuse Between Regimes**
- **Problem:** Smoke regime trains from scratch, Confirm retrains from scratch
- **Inefficiency:** Could use smoke checkpoint as starting point for confirm
- **Fix:** Add option to continue from smoke checkpoint for confirm regime

#### ⚠️ IMPROVEMENTS NEEDED

##### **Issue #21: MLflow Run Naming Collision**
- **Location:** --run-name argument
- **Problem:** Auto-generated names like "auto-001" don't encode hyperparameters
- **Fix:** Auto-append key params to run name: `auto-focal-g2-a0.25-lr3e-4`

##### **Issue #22: No Timeout Protection**
- **Problem:** Confirm regime can run indefinitely if model doesn't converge
- **Risk:** Wastes GPU hours on doomed experiments
- **Fix:** Add `--max-runtime-hours` flag with graceful termination

---

### 4. `scripts/run_overnight_experiments.py` (242 lines)

#### ✅ STRENGTHS

1. **Error Isolation**: Each experiment wrapped in try/except, one failure doesn't abort all
2. **Resume Support**: --start-from flag for partial completion
3. **Clear Logging**: Final summary with completed/failed runs
4. **Documentation**: Excellent usage examples and hypothesis explanations

#### 🔴 CRITICAL ISSUES

##### **Issue #23: Sequential Execution Bottleneck**
- **Location:** Line 8 comment ("Sequential not parallel")
- **Problem:** Only one experiment runs at a time even if multiple GPUs available
- **Fix:** Detect available GPUs and parallelize across them

```python
import torch
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    # Parallelize experiments across GPUs
    torch.multiprocessing.spawn(run_experiment_on_gpu, args=(experiments, num_gpus))
```

##### **Issue #24: No Resource Monitoring**
- **Problem:** Doesn't check GPU temperature, power limits, or thermal throttling
- **Risk:** Overnight runs may throttle performance or damage hardware
- **Fix:** Add periodic resource monitoring with alerts

##### **Issue #25: Hardcoded Experiment Matrix**
- **Location:** EXPERIMENTS list (lines 55-97)
- **Problem:** Changing experiments requires code modification
- **Fix:** Load experiment matrix from YAML/JSON config file

```yaml
experiments:
  - name: run-alpha-tune
    focal_alpha: 0.35
    epochs: 20
  - name: run-lr-sweep
    lr: [1e-4, 3e-5, 1e-5]
    epochs: 30
```

##### **Issue #26: No Notification System**
- **Problem:** User must manually check logs in morning
- **Fix:** Add optional Slack/Discord/email notifications on completion or failure

#### ⚠️ IMPROVEMENTS NEEDED

##### **Issue #27: Baseline Drift**
- **Location:** Line 53 comment ("Baseline defaults: epochs=20, lr=1e-4...")
- **Problem:** Baseline hardcoded in comments, not enforced
- **Risk:** Experiments diverge from comparable baseline over time
- **Fix:** Define baseline as TrainConfig instance, experiments specify deltas

---

### 5. `src/training/focalloss.py` (73 lines)

#### ✅ STRENGTHS

1. **Correct Implementation**: Follows original Focal Loss paper formula
2. **BF16 Safety**: Explicit .float() cast prevents underflow (Fix #6)
3. **Multi-label Support**: Element-wise operations handle both binary and multi-label
4. **Clear Documentation**: Warns against passing raw logits (must be post-sigmoid)

#### 🔴 CRITICAL ISSUES

##### **Issue #28: Alpha Imbalance Calculation Incorrect**
- **Location:** Line 67 (`alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)`)
- **Problem:** Assumes binary case where alpha for class 0 is (1-alpha)
- **Risk:** For multi-label, each class may need independent alpha weight
- **Current Behavior:** All negative labels get same weight (1 - 0.25 = 0.75)
- **Expected:** Per-class alpha based on frequency

```python
# Better approach for multi-label:
def __init__(self, gamma=2.0, alpha=0.25, pos_weights=None):
    # pos_weights: tensor of shape (num_classes,) with inverse frequency
    self.pos_weights = pos_weights

def forward(self, predictions, targets):
    # Use per-class weights instead of scalar alpha
    alpha_t = torch.where(
        targets == 1,
        self.pos_weights if self.pos_weights is not None else self.alpha,
        1.0  # or separate neg_weights
    )
```

##### **Issue #29: No Gradient Checking**
- **Problem:** Doesn't verify gradients are flowing (no NaN/Inf check)
- **Risk:** Silent failure if loss becomes constant
- **Fix:** Add optional gradient norm logging for debugging

##### **Issue #30: Reduction Mode Hardcoded**
- **Location:** Line 59 (`reduction="none"` then manual .mean())
- **Problem:** Always averages over all elements, no option for sum or per-sample loss
- **Use Case:** Per-sample loss needed for uncertainty estimation
- **Fix:** Add `reduction` parameter supporting "mean", "sum", "none"

---

## 📊 Summary Table

| Priority | Module | Issue | Severity | Impact | Effort |
|----------|--------|-------|----------|--------|--------|
| **P0** | trainer.py | Checkpoint format lock-in | 🔴 Critical | Future breakage | Medium |
| **P0** | trainer.py | Sidecar file race condition | 🔴 Critical | Data corruption | Low |
| **P0** | auto_experiment.py | Threshold tuning dependency | 🔴 Critical | False negatives | Low |
| **P0** | focalloss.py | Alpha imbalance incorrect | 🔴 Critical | Suboptimal weighting | Medium |
| **P1** | trainer.py | Scheduler state silent degradation | ⚠️ High | Poor convergence | Low |
| **P1** | trainer.py | Hardcoded CLASS_NAMES | ⚠️ High | Mislabeling risk | Low |
| **P1** | train.py | No pre-flight validation | ⚠️ High | Wasted compute | Medium |
| **P1** | overnight_experiments.py | Sequential execution bottleneck | ⚠️ High | Slow iteration | High |
| **P1** | trainer.py | No gradient accumulation | ⚠️ High | Hardware limits | Medium |
| **P2** | trainer.py | RAM cache memory leak | 🟡 Medium | OOM risk | Low |
| **P2** | auto_experiment.py | Stratified subsample verification | 🟡 Medium | Misleading smoke | Low |
| **P2** | train.py | No dry-run mode | 🟡 Medium | UX issue | Low |
| **P2** | focalloss.py | No per-class alpha | 🟡 Medium | Class imbalance | Medium |
| **P2** | overnight_experiments.py | Hardcoded experiment matrix | 🟡 Medium | Maintenance burden | Low |

---

## 🚀 Recommended Actions (Prioritized)

### Immediate (Before Next Training Run)
1. **Fix Checkpoint Versioning** (trainer.py) - Add version field and migration function
2. **Atomic Sidecar Writes** (trainer.py) - Use tmp file + rename pattern
3. **Make Threshold Tuning Optional** (auto_experiment.py) - Don't fail entire run if tuning crashes
4. **Add Preflight Checks** (train.py) - Validate paths, disk space, GPU before training

### Short-Term (Before Publication)
5. **Fix Focal Loss Alpha** (focalloss.py) - Implement per-class weighting for multi-label
6. **Centralize CLASS_NAMES** (trainer.py) - Import from graph_schema.py
7. **Add Gradient Accumulation** (trainer.py) - Enable larger effective batch sizes
8. **Parallelize Overnight Experiments** (overnight_experiments.py) - Multi-GPU support

### Long-Term (Architecture Improvements)
9. **Config Validation** (trainer.py) - Add __post_init__ checks for TrainConfig
10. **Experiment Config Files** (overnight_experiments.py) - YAML-based experiment definitions
11. **Resource Monitoring** (all scripts) - GPU temp, power, memory alerts
12. **Notification System** (overnight_experiments.py) - Slack/email on completion

---

## 📁 Files Requiring Changes

| File | Lines to Change | Estimated Effort |
|------|----------------|------------------|
| `src/training/trainer.py` | ~150 lines (versioning, validation, grad accum) | 6 hours |
| `scripts/train.py` | ~50 lines (preflight checks, dry-run) | 2 hours |
| `scripts/auto_experiment.py` | ~30 lines (optional tuning, better error handling) | 2 hours |
| `scripts/run_overnight_experiments.py` | ~80 lines (parallel execution, YAML config) | 4 hours |
| `src/training/focalloss.py` | ~40 lines (per-class alpha, reduction modes) | 2 hours |

**Total Estimated Effort:** ~16 hours

---

## 🔍 Testing Recommendations

1. **Checkpoint Compatibility Test**: Train with old checkpoint format, verify migration works
2. **Resume Stress Test**: Interrupt training at random epochs, resume 10+ times, verify patience counter accuracy
3. **OOM Test**: Run with batch_size near GPU limit, verify graceful error vs crash
4. **Multi-GPU Test**: Run overnight experiments on 2+ GPU system, verify parallelization
5. **Class Imbalance Test**: Synthetic dataset with extreme imbalance (99:1), verify Focal Loss weighting

---

## 📝 Notes for Next Audit Group

**Observations:**
- Training pipeline is well-documented with extensive audit trail
- Most critical issues are around edge cases and long-term maintainability
- Strong foundation exists; improvements are refinements rather than rewrites
- Consider auditing `tune_threshold.py` next as it's tightly coupled with training evaluation

**Dependencies on Other Groups:**
- **Group 2 (Datasets)**: RAM cache interaction needs verification
- **Group 3 (Models)**: LoRA parameter validation should align with model architecture
- **Group 6 (Inference)**: Threshold tuning results must be compatible with predictor

---

**Next Group:** Group 5 (Model Evaluation & Threshold Tuning) or Group 6 (Inference & API)
