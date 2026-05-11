# 🛡️ Audit Report: Group 5 (Model Evaluation & Threshold Tuning)

**Audit Date:** 2026-05-10  
**Auditor:** AI Code Assistant  
**Status:** ✅ COMPLETE

---

## Executive Summary

**Modules Audited:** 3 scripts totaling 935 lines
- `scripts/tune_threshold.py` (564 lines) - Per-class threshold optimization
- `scripts/promote_model.py` (192 lines) - MLflow model registry promotion
- `scripts/compute_drift_baseline.py` (179 lines) - Drift detection baseline computation

**Overall Assessment:** ⚠️ **MODERATE RISK** — Well-documented with several critical security and reliability issues identified.

**Key Findings:**
- ✅ **Strengths:** Excellent documentation, proper multi-label handling, good CLI design, dry-run support
- 🔴 **Critical Issues:** Unsafe deserialization (`weights_only=False`) in all 3 files, missing input validation
- ⚠️ **High Priority:** No class imbalance warnings, silent fallback behaviors, MLflow error handling gaps

---

## 1. `scripts/tune_threshold.py` (564 lines)

### 🔴 CRITICAL ISSUES

#### **Issue #1: Security Vulnerability - Unsafe Deserialization**
- **Location:** Line 188 - `torch.load(checkpoint_path, map_location=device, weights_only=False)`
- **Severity:** 🔴 **CRITICAL** - Remote Code Execution (RCE) vulnerability
- **Risk:** Malicious checkpoint files can execute arbitrary code during loading
- **Impact:** Complete system compromise if attacker controls checkpoint file
- **Fix Required:** Change to `weights_only=True`
- **Code Example:**
```python
# VULNERABLE (Line 188)
raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

# FIXED
raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
```

#### **Issue #2: Missing Class Imbalance Validation**
- **Location:** Lines 325-328 (support logging only)
- **Problem:** No warning when classes have <10 positive samples in validation set
- **Risk:** Threshold tuning unreliable for rare classes, may produce meaningless thresholds
- **Impact:** Poor model performance on underrepresented vulnerability types
- **Fix:** Add minimum support check with warning
- **Code Example:**
```python
# Add after line 328
MIN_SUPPORT = 10
for name, sup in zip(CLASS_NAMES, supports):
    if sup < MIN_SUPPORT:
        logger.warning(
            f"Class '{name}' has only {sup} positive samples (min={MIN_SUPPORT}). "
            f"Threshold tuning may be unreliable."
        )
```

#### **Issue #3: No Validation of Probability Distribution**
- **Location:** Lines 307-323
- **Problem:** Doesn't check for NaN/Inf in probabilities before threshold sweep
- **Risk:** Silent propagation of numerical errors, invalid threshold selection
- **Fix:** Add probability validation after collection
- **Code Example:**
```python
# Add after line 312 (after concatenation)
if not np.all(np.isfinite(probs_np)):
    nan_count = np.sum(~np.isfinite(probs_np))
    raise RuntimeError(
        f"Collected probabilities contain {nan_count} NaN/Inf values. "
        "Check model output for numerical instability."
    )
```

#### **Issue #4: Hardcoded Threshold Grid Without Adaptation**
- **Location:** Lines 130-146 (CLI args), Lines 333-375 (sweep logic)
- **Problem:** Fixed step size (0.05) may miss optimal thresholds for imbalanced classes
- **Risk:** Suboptimal threshold selection, especially for rare classes needing fine-grained tuning
- **Improvement:** Implement adaptive grid refinement around high-F1 regions
- **Alternative:** Add `--fine-grain` flag for 0.01 step size in second pass

#### **Issue #5: Silent Degradation on Zero-Division**
- **Location:** Lines 354-356 - `zero_division=0` parameter
- **Problem:** Returns 0 for precision/recall/F1 when no positives predicted, without warning
- **Risk:** May hide issues with classes where model predicts all negatives
- **Improvement:** Log warning when zero_division occurs for any class
- **Code Example:**
```python
# In sweep_one_class(), add after line 356
if row.precision == 0 and labels.sum() > 0:
    logger.debug(
        f"Class {class_name}: zero precision at threshold {threshold} "
        f"(actual positives: {labels.sum()})"
    )
```

### ⚠️ BAD APPROACHES & IMPROVEMENTS

#### **Issue #6: No Checkpoint Metadata Validation**
- **Location:** Lines 188-222 (`load_model_from_checkpoint`)
- **Problem:** Doesn't verify checkpoint config matches expected architecture
- **Risk:** Silent use of wrong architecture parameters, subtle bugs
- **Improvement:** Add explicit validation of required config keys
- **Code Example:**
```python
# Add after line 190
required_keys = ["architecture", "num_classes", "gnn_hidden_dim"]
missing_keys = [k for k in required_keys if k not in ckpt_config]
if missing_keys:
    logger.warning(
        f"Checkpoint config missing keys: {missing_keys}. "
        f"Using defaults which may not match training configuration."
    )
```

#### **Issue #7: Memory Inefficiency in Probability Collection**
- **Location:** Lines 280-313 (`collect_probabilities`)
- **Problem:** Stores ALL probabilities in memory before processing
- **Risk:** OOM errors on large validation sets (>100k samples)
- **Improvement:** Process classes incrementally or use memory-mapped arrays
- **Alternative:** Add streaming mode that processes one class at a time

#### **Issue #8: Tie-Breaking Logic May Not Match Business Needs**
- **Location:** Lines 361-364 - tie-breaking prefers lower threshold
- **Problem:** Always prefers catching more positives, may increase false positives unnecessarily
- **Risk:** High false positive rate in production, alert fatigue
- **Improvement:** Make tie-breaking strategy configurable
- **Code Example:**
```python
# Add CLI argument
parser.add_argument(
    "--tie-break",
    choices=["recall", "precision", "f1"],
    default="recall",
    help="Tie-breaking strategy when multiple thresholds have same F1"
)

# Update line 363
if args.tie_break == "precision":
    key_func = lambda row: (round(row.f1, 8), round(row.precision, 8), -row.threshold)
elif args.tie_break == "recall":
    key_func = lambda row: (round(row.f1, 8), round(row.recall, 8), -row.threshold)
else:  # f1
    key_func = lambda row: (round(row.f1, 8), -row.threshold)
```

### ✅ GOOD PRACTICES NOTED
- Comprehensive docstring explaining why script exists
- Proper multi-label handling (per-class thresholds)
- Shape validation for logits and labels (Lines 292-304)
- BF16/FP16 safety with `.float()` cast before sigmoid (Line 307)
- Detailed per-class reporting with sweep tables
- JSON output for programmatic consumption
- Inclusive threshold grid with proper rounding (Lines 161-163)

---

## 2. `scripts/promote_model.py` (192 lines)

### 🔴 CRITICAL ISSUES

#### **Issue #9: Security Vulnerability - Unsafe Deserialization**
- **Location:** Line 67 - `torch.load(checkpoint, map_location="cpu", weights_only=False)`
- **Severity:** 🔴 **CRITICAL** - Remote Code Execution (RCE) vulnerability
- **Risk:** Same as Issue #1
- **Fix Required:** Change to `weights_only=True`

#### **Issue #10: No MLflow Connection Error Handling**
- **Location:** Lines 106-137 (`promote` function)
- **Problem:** If MLflow server is unavailable, script crashes with unhelpful traceback
- **Risk:** Operators cannot determine if promotion succeeded or failed
- **Impact:** Potential for duplicate model versions or inconsistent registry state
- **Fix:** Add try-except with clear error messages and rollback logic
- **Code Example:**
```python
# Wrap lines 106-137
try:
    mlflow.set_experiment(experiment_name)
    # ... rest of promotion logic
except mlflow.exceptions.MlflowException as e:
    print(f"ERROR: MLflow operation failed: {e}", file=sys.stderr)
    print("Check MLflow tracking server connectivity and permissions.", file=sys.stderr)
    return 1
except Exception as e:
    print(f"ERROR: Unexpected error during promotion: {e}", file=sys.stderr)
    # Attempt rollback if model version was created
    return 1
```

#### **Issue #11: No Validation of Checkpoint Integrity**
- **Location:** Lines 66-75 (`_load_checkpoint_meta`)
- **Problem:** Doesn't verify checkpoint contains valid model state_dict
- **Risk:** Promoting corrupted or incomplete checkpoints to registry
- **Impact:** Production deployment of broken models
- **Fix:** Add basic integrity checks
- **Code Example:**
```python
def _load_checkpoint_meta(checkpoint: Path) -> dict:
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")
    
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint must be a dictionary")
    
    if "model" not in raw and not any(k.endswith(".weight") for k in raw.keys()):
        raise ValueError("Checkpoint missing model state_dict")
    
    cfg = raw.get("config", {})
    return {
        "architecture": cfg.get("architecture", "unknown"),
        "num_classes":  cfg.get("num_classes", "unknown"),
        "epoch":        raw.get("epoch", "unknown"),
    }
```

#### **Issue #12: Hardcoded Model Name Without Validation**
- **Location:** Line 53 - `MODEL_NAME = "sentinel-vulnerability-detector"`
- **Problem:** No way to override model name for different experiments
- **Risk:** Accidental overwriting of different model versions
- **Improvement:** Make model name configurable via CLI argument
- **Code Example:**
```python
# Add CLI argument
parser.add_argument(
    "--model-name",
    default=MODEL_NAME,
    help=f"MLflow model registry name (default: {MODEL_NAME})"
)

# Use args.model_name instead of hardcoded MODEL_NAME
```

#### **Issue #13: No Gate Validation for Production Promotion**
- **Location:** Lines 140-188 (`main` function)
- **Problem:** No minimum F1 threshold enforced for Production stage
- **Risk:** Promoting underperforming models to production
- **Impact:** Degraded service quality
- **Fix:** Add optional `--min-f1` gate with stage-specific defaults
- **Code Example:**
```python
# Add CLI arguments
parser.add_argument(
    "--min-f1",
    type=float,
    default=None,
    help="Minimum F1-macro required for promotion. Stage-specific defaults: Staging=0.3, Production=0.4"
)

# Add validation in promote()
MIN_F1_GATES = {"Staging": 0.3, "Production": 0.4}
min_f1 = args.min_f1 if args.min_f1 is not None else MIN_F1_GATES.get(stage, 0.0)
if val_f1_macro < min_f1:
    print(
        f"ERROR: val_f1_macro ({val_f1_macro:.4f}) below minimum gate ({min_f1:.4f}) "
        f"for {stage} stage.",
        file=sys.stderr
    )
    return 1
```

### ⚠️ BAD APPROACHES & IMPROVEMENTS

#### **Issue #14: Git Commit Fallback to "unknown"**
- **Location:** Lines 57-63 (`_git_commit`)
- **Problem:** Silently returns "unknown" if git command fails
- **Risk:** Lost audit trail in production environments without git
- **Improvement:** Log warning and attempt alternative metadata sources
- **Code Example:**
```python
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception as e:
        logger.warning(f"Failed to get git commit: {e}")
        # Try environment variable as fallback
        import os
        if sha := os.getenv("GIT_SHA"):
            return sha[:8]
        return "unknown-no-git"
```

#### **Issue #15: No Dry-Run for MLflow Registration**
- **Location:** Lines 126-133 (registration logic)
- **Problem:** Dry-run only skips artifact logging, still registers model
- **Risk:** Accidental registry pollution during testing
- **Fix:** Extend dry-run to skip registration entirely
- **Code Example:**
```python
# Update lines 126-133
if not dry_run:
    client = MlflowClient()
    mv = mlflow.register_model(model_uri=artifact_uri, name=MODEL_NAME)
    client.transition_model_version_stage(...)
    print(f"Registered '{MODEL_NAME}' version {mv.version} → {stage}")
else:
    print("DRY RUN: Would register model but skipped due to --dry-run flag")
```

### ✅ GOOD PRACTICES NOTED
- Clear documentation with usage examples
- Dry-run support for safe testing
- Atomic operation design (all-or-nothing promotion)
- Proper experiment tracking with MLflow
- Archive existing versions on Production promotion (Line 132)
- Git commit tracking for audit trail
- Exit codes for scripting integration

---

## 3. `scripts/compute_drift_baseline.py` (179 lines)

### 🔴 CRITICAL ISSUES

#### **Issue #16: Security Vulnerability - Unsafe Deserialization**
- **Location:** Line 56 - `torch.load(path, map_location="cpu", weights_only=False)`
- **Severity:** 🔴 **CRITICAL** - Remote Code Execution (RCE) vulnerability
- **Risk:** Same as Issue #1 and #9
- **Fix Required:** Change to `weights_only=True`

#### **Issue #17: Insufficient Warmup Sample Size Validation**
- **Location:** Lines 83-89 - requires only 30 records
- **Problem:** 30 samples statistically insufficient for KS-test baseline
- **Risk:** High false positive/negative drift alerts
- **Impact:** Alert fatigue or missed drift detection
- **Fix:** Increase minimum to 100-500 samples with statistical justification
- **Code Example:**
```python
# Update line 83
MIN_WARMUP_SAMPLES = 100  # Increased from 30 for statistical significance

if len(records) < MIN_WARMUP_SAMPLES:
    print(
        f"ERROR: warmup log has only {len(records)} records — need at least "
        f"{MIN_WARMUP_SAMPLES} for statistically reliable KS-test baseline. "
        f"Collect more requests first.",
        file=sys.stderr,
    )
    return 1
```

#### **Issue #18: No Validation of Stat Distribution**
- **Location:** Lines 91-95 (baseline accumulation)
- **Problem:** Doesn't check for extreme outliers or bimodal distributions
- **Risk:** Mean/std baseline skewed by outliers, poor drift detection
- **Impact:** Missed drift or false alerts
- **Fix:** Add outlier detection and robust statistics (median, IQR)
- **Code Example:**
```python
# Add after line 94
import numpy as np

for k, values in baseline.items():
    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    outliers = np.sum((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr))
    if outliers > len(arr) * 0.05:  # >5% outliers
        logger.warning(
            f"Stat '{k}' has {outliers} outliers ({100*outliers/len(arr):.1f}%). "
            f"Consider cleaning data or using robust statistics."
        )
```

#### **Issue #19: Training Data Warning Insufficient**
- **Location:** Lines 107-112 - prints warning but continues
- **Problem:** Warning alone doesn't prevent dangerous behavior
- **Risk:** Operators accidentally use training data, causing constant false alerts
- **Fix:** Add `--force` flag to require explicit confirmation for training source
- **Code Example:**
```python
# Add CLI argument
parser.add_argument(
    "--force",
    action="store_true",
    help="Override safety check when using --source training (NOT RECOMMENDED)"
)

# Update from_training() call
if args.source == "training" and not args.force:
    print(
        "ERROR: Using --source training is strongly discouraged.\n"
        "This will cause false drift alerts on modern contracts.\n"
        "If you understand the risks, add --force to proceed.",
        file=sys.stderr
    )
    return 1
```

### ⚠️ BAD APPROACHES & IMPROVEMENTS

#### **Issue #20: Limited Stat Types (Only Nodes/Edges)**
- **Location:** Line 51 - `STAT_NAMES = ["num_nodes", "num_edges"]`
- **Problem:** Only tracks graph size, ignores feature distribution
- **Risk:** Misses drift in node features, edge types, or label distribution
- **Impact:** False sense of security when feature drift occurs
- **Improvement:** Add feature statistics (mean, std per feature dimension)
- **Code Example:**
```python
# Expand _extract_stats_from_graph()
def _extract_stats_from_graph(path: Path) -> dict[str, float] | None:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"  SKIP {path.name}: {exc}", file=sys.stderr)
        return None
    
    stats = {
        "num_nodes": float(data.num_nodes or 0),
        "num_edges": float(data.num_edges or 0),
    }
    
    # Add feature statistics (sampled to avoid memory issues)
    if hasattr(data, 'x') and data.x is not None:
        x = data.x.cpu().numpy()
        stats["node_feature_mean"] = float(np.mean(x))
        stats["node_feature_std"] = float(np.std(x))
        stats["node_feature_max"] = float(np.max(x))
    
    return stats
```

#### **Issue #21: No Output File Validation**
- **Location:** Lines 96-103 (output writing)
- **Problem:** Doesn't verify output file was written correctly
- **Risk:** Silent corruption if disk full or permission denied
- **Fix:** Add post-write validation
- **Code Example:**
```python
# Add after line 98
with open(output, "w") as f:
    json.dump(dict(baseline), f, indent=2)

# Validate write
if not output.exists() or output.stat().st_size == 0:
    print(f"ERROR: Failed to write baseline file or file is empty: {output}", file=sys.stderr)
    return 1

print(f"Baseline written — {output} ({output.stat().st_size} bytes)")
```

#### **Issue #22: Hardcoded Minimum Record Count**
- **Location:** Line 83 - hardcoded `30`
- **Problem:** Magic number without explanation or configurability
- **Fix:** Make configurable via CLI argument with documented rationale
- **Code Example:**
```python
parser.add_argument(
    "--min-samples",
    type=int,
    default=100,
    help="Minimum warmup samples for statistical significance (default: 100)"
)
```

### ✅ GOOD PRACTICES NOTED
- Strong warning against using training data (Lines 5-10, 107-112)
- Clear documentation with recommended workflow
- Graceful handling of corrupted files (Lines 55-59)
- Line number reporting for JSONL parsing errors (Line 81)
- Support for both warmup and training sources (flexibility)
- Proper exit codes for scripting

---

## 📊 Summary Table

| Priority | Module | Issue | Severity | Impact | Effort |
|----------|--------|-------|----------|--------|--------|
| **P0** | ALL 3 FILES | Unsafe `torch.load(weights_only=False)` | 🔴 Critical | RCE Vulnerability | Low (15 min) |
| **P0** | `tune_threshold.py` | No class imbalance validation | ⚠️ High | Unreliable thresholds | Medium (1 hour) |
| **P0** | `compute_drift_baseline.py` | Insufficient warmup samples (30) | ⚠️ High | False drift alerts | Low (30 min) |
| **P1** | `promote_model.py` | No MLflow error handling | ⚠️ High | Registry inconsistency | Medium (1.5 hours) |
| **P1** | `promote_model.py` | No production F1 gate | ⚠️ High | Deploy bad models | Low (30 min) |
| **P1** | `tune_threshold.py` | No probability validation (NaN/Inf) | ⚠️ High | Invalid thresholds | Low (30 min) |
| **P1** | `compute_drift_baseline.py` | No --force flag for training source | ⚠️ High | Accidental misuse | Low (20 min) |
| **P2** | `tune_threshold.py` | Hardcoded threshold grid | ⚠️ Medium | Suboptimal thresholds | Medium (2 hours) |
| **P2** | `promote_model.py` | Hardcoded model name | ⚠️ Medium | Registry conflicts | Low (20 min) |
| **P2** | `compute_drift_baseline.py` | Limited stat types | ⚠️ Medium | Missed feature drift | Medium (1.5 hours) |
| **P2** | `tune_threshold.py` | Memory inefficiency | ⚠️ Medium | OOM on large datasets | Medium (2 hours) |
| **P3** | `promote_model.py` | Git fallback to "unknown" | ℹ️ Low | Lost audit trail | Low (20 min) |
| **P3** | `tune_threshold.py` | Zero-division silence | ℹ️ Low | Hidden issues | Low (30 min) |

---

## 🚀 Recommended Actions (Prioritized)

### Immediate (Before Next Use)
1. **Fix ALL `torch.load` calls** → Change to `weights_only=True` across all 3 files
   - Files: `tune_threshold.py:188`, `promote_model.py:67`, `compute_drift_baseline.py:56`
   - Estimated effort: 15 minutes

2. **Increase warmup sample requirement** in `compute_drift_baseline.py`
   - Change: 30 → 100 samples minimum
   - Add statistical justification comment
   - Estimated effort: 30 minutes

3. **Add class imbalance warnings** in `tune_threshold.py`
   - Warn when support < 10 for any class
   - Estimated effort: 1 hour

### Short-Term (Before Production Deployment)
4. **Add MLflow error handling** in `promote_model.py`
   - Try-except blocks with clear error messages
   - Rollback logic for partial failures
   - Estimated effort: 1.5 hours

5. **Add production F1 gate** in `promote_model.py`
   - Default gates: Staging=0.3, Production=0.4
   - Configurable via `--min-f1` flag
   - Estimated effort: 30 minutes

6. **Add probability validation** in `tune_threshold.py`
   - Check for NaN/Inf after collection
   - Fail fast with clear error message
   - Estimated effort: 30 minutes

7. **Add --force flag** in `compute_drift_baseline.py`
   - Require explicit confirmation for training source
   - Estimated effort: 20 minutes

### Long-Term (Enhancement Phase)
8. **Implement adaptive threshold grid** in `tune_threshold.py`
   - Two-pass approach: coarse then fine refinement
   - Estimated effort: 2 hours

9. **Expand drift statistics** in `compute_drift_baseline.py`
   - Add node feature mean/std/max
   - Add edge type distribution
   - Estimated effort: 1.5 hours

10. **Add streaming mode** in `tune_threshold.py`
    - Process classes incrementally to reduce memory
    - Estimated effort: 2 hours

---

## 📁 Files Requiring Changes

| File | Lines to Change | Critical Fixes | Total Estimated Effort |
|------|----------------|----------------|----------------------|
| `scripts/tune_threshold.py` | 8 locations | 3 | 4.5 hours |
| `scripts/promote_model.py` | 5 locations | 2 | 2.5 hours |
| `scripts/compute_drift_baseline.py` | 4 locations | 2 | 2 hours |

**Total Estimated Effort:** ~9 hours

---

## 🧪 Testing Recommendations

### Unit Tests Needed
1. **`tune_threshold.py`:**
   - Test with imbalanced classes (<10 samples)
   - Test with NaN/Inf probabilities
   - Test tie-breaking strategies
   - Test memory efficiency with large datasets

2. **`promote_model.py`:**
   - Test MLflow connection failure scenarios
   - Test F1 gate enforcement
   - Test checkpoint integrity validation
   - Test dry-run completeness

3. **`compute_drift_baseline.py`:**
   - Test with insufficient warmup samples
   - Test outlier detection
   - Test --force flag for training source
   - Test feature statistics extraction

### Integration Tests
- End-to-end threshold tuning pipeline
- Model promotion with rollback scenarios
- Drift baseline creation and consumption

---

## 📋 Dependencies on Other Groups

| Dependency | Group | Status | Impact |
|------------|-------|--------|--------|
| Checkpoint format | Group 4 (Training) | ✅ Stable | None |
| Dataset splits | Group 2 (Datasets) | ⚠️ Data leakage issue | May invalidate thresholds |
| Model architecture | Group 3 (Models) | ✅ Stable | None |
| Drift detector | Group 6 (Inference) | ✅ Compatible | None |

---

## ✅ Conclusion

Group 5 modules are **well-designed and well-documented** but share the **critical security vulnerability** of unsafe deserialization found throughout the codebase. The threshold tuning logic is sophisticated and appropriate for multi-label classification, but needs better validation and error handling.

**Priority Focus:**
1. Fix all `weights_only=False` instances immediately (security)
2. Add validation layers (class imbalance, probability quality, sample size)
3. Improve error handling and user feedback

**Estimated Time to Production-Ready:** 1-2 days for critical fixes, 1 week for full enhancements.

---

**Next Group:** Group 6 (Inference & API) - covering `src/inference/api.py`, `predictor.py`, `preprocess.py`, `cache.py`, `drift_detector.py`, and `src/utils/hash_utils.py`
