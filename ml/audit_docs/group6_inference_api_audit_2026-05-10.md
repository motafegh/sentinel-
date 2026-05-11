# 🛡️ Audit Report: Group 6 (Inference & API)

**Audit Date:** 2026-05-10  
**Auditor:** AI Code Auditor  
**Status:** COMPLETE

---

## Executive Summary

**Modules Audited:** 6 files totaling 1,596 lines
- `api.py` (254 lines) - FastAPI endpoints (/predict, /health, /metrics)
- `predictor.py` (463 lines) - Prediction logic with threshold application
- `preprocess.py` (532 lines) - ContractPreprocessor (online inference)
- `cache.py` (154 lines) - InferenceCache (MD5 + schema version)
- `drift_detector.py` (193 lines) - Rolling KS-test drift detection
- `__init__.py` (0 lines) - Package marker

**Overall Assessment:** ⚠️ **MODERATE-HIGH RISK** — Production-ready architecture with excellent documentation, but critical security vulnerabilities and operational gaps identified.

**Key Strengths:**
- ✅ Excellent documentation with clear "WHY THIS EXISTS" sections
- ✅ Proper exception translation for HTTP boundaries
- ✅ Atomic file operations for cache writes
- ✅ Comprehensive warmup testing at startup
- ✅ Content-addressed caching with schema versioning
- ✅ Sliding window support for long contracts
- ✅ SIGKILL-safe temp file management

**Critical Concerns:**
- 🔴 Unsafe `torch.load(weights_only=False)` in cache and predictor
- 🔴 Hardcoded thresholds without runtime override
- 🔴 No input validation for graph feature dimensions in cache
- 🔴 Drift detector baseline strategy unclear for production

---

## Detailed Analysis

### 1. `api.py` (254 lines)

**Role:** FastAPI HTTP interface for vulnerability prediction

#### ✅ GOOD PRACTICES
1. **Lifespan Pattern** (Lines 81-100): Predictor loaded once at startup, not per-request
2. **Proper Error Handling**: Specific HTTP status codes (400, 413, 500, 503, 504)
3. **Timeout Control**: `SENTINEL_PREDICT_TIMEOUT` env var (default 60s)
4. **Size Validation**: Enforces `MAX_SOURCE_BYTES` before preprocessing
5. **Prometheus Integration**: GPU memory gauge, model loaded gauge
6. **Bug Fixes Documented**: Lines 19-31 clearly document fixes from previous audits

#### 🔴 CRITICAL ISSUES

**Issue #1: Missing Input Validation on Graph Dimensions**
- **Location:** Line 193 (`predictor.predict_source`)
- **Problem:** No validation that returned graph has correct feature dimensions
- **Risk:** Silent failures if preprocess.py returns wrong shape
- **Fix:** Add assertion after prediction:
```python
assert result["num_nodes"] >= 0, "Invalid num_nodes in prediction result"
assert result["num_edges"] >= 0, "Invalid num_edges in prediction result"
```

**Issue #2: No Rate Limiting**
- **Location:** `/predict` endpoint (Line 176)
- **Problem:** No rate limiting or request throttling
- **Risk:** DoS vulnerability, resource exhaustion
- **Fix:** Add slowapi or custom rate limiter:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, ...):
```

**Issue #3: Hardcoded Checkpoint Path**
- **Location:** Line 64-67
- **Problem:** Default checkpoint path hardcoded, only overridable via env var
- **Risk:** Deployment inflexibility, accidental use of wrong checkpoint
- **Fix:** Add CLI argument or config file support with validation

#### ⚠️ HIGH-PRIORITY IMPROVEMENTS

**Issue #4: No Request/Response Logging**
- **Location:** Line 191
- **Problem:** Only logs character count, no structured logging
- **Fix:** Add structured logging with correlation IDs:
```python
import uuid
correlation_id = str(uuid.uuid4())
logger.info(f"Inference request", extra={
    "correlation_id": correlation_id,
    "source_chars": len(body.source_code),
    "source_hash": hashlib.md5(body.source_code.encode()).hexdigest()[:16]
})
```

**Issue #5: Missing Model Version in Response**
- **Location:** `PredictResponse` schema (Lines 140-148)
- **Problem:** Response doesn't include model version/checkpoint hash
- **Risk:** Cannot trace which model version produced predictions
- **Fix:** Add `model_version: str` field to `PredictResponse`

**Issue #6: No Health Check Depth Control**
- **Location:** `/health` endpoint (Lines 154-172)
- **Problem:** Always reports full health, no "shallow" mode for load balancers
- **Fix:** Add optional `?depth=shallow` parameter to skip heavy checks

---

### 2. `predictor.py` (463 lines)

**Role:** Core prediction engine loading checkpoint and applying thresholds

#### ✅ GOOD PRACTICES
1. **Architecture Registry Pattern** (Lines 78-82): `_ARCH_TO_FUSION_DIM` dict for extensibility
2. **Strict Metadata Validation** (Lines 148-159): Cross-checks class_names order
3. **Per-Class Thresholds** (Lines 221-263): Loads from companion JSON file
4. **Warmup Forward Pass** (Lines 277-330): Catches CUDA issues at startup
5. **Comprehensive Documentation**: Lines 1-56 explain all changes and fixes

#### 🔴 CRITICAL ISSUES

**Issue #7: Unsafe Deserialization in Checkpoint Loading**
- **Location:** Line 124
- **Code:** `torch.load(checkpoint, map_location=self.device, weights_only=False)`
- **Risk:** **RCE VULNERABILITY** - Malicious checkpoint can execute arbitrary code
- **Severity:** **CRITICAL** for any system accepting external checkpoints
- **Fix:** 
```python
# Option 1: Use weights_only=True (breaks LoRA checkpoints)
raw = torch.load(checkpoint, map_location=self.device, weights_only=True)

# Option 2: Validate checkpoint hash before loading
import hashlib
expected_hash = os.getenv("SENTINEL_CHECKPOINT_HASH")
if expected_hash:
    with open(checkpoint, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"Checkpoint hash mismatch!")
```

**Issue #8: Silent Fallback to Uniform Threshold**
- **Location:** Lines 255-263
- **Problem:** If threshold JSON missing, falls back to uniform 0.5 without alerting
- **Risk:** Production system may run with suboptimal thresholds unknowingly
- **Fix:** Raise warning/error in production mode:
```python
if not thresholds_path.exists():
    if os.getenv("SENTINEL_STRICT_THRESHOLDS", "false").lower() == "true":
        raise FileNotFoundError(
            f"Threshold file required in strict mode: {thresholds_path}"
        )
    logger.warning(...)  # existing warning
```

**Issue #9: No Threshold Validation Range**
- **Location:** Lines 247-249
- **Problem:** Doesn't validate thresholds are in (0, 1) range
- **Risk:** Invalid thresholds could cause all predictions to be 0 or 1
- **Fix:** Add validation:
```python
for cls_name, thresh in class_thresholds_dict.items():
    if not (0.0 < thresh < 1.0):
        raise ValueError(f"Invalid threshold for {cls_name}: {thresh}")
```

#### ⚠️ HIGH-PRIORITY IMPROVEMENTS

**Issue #10: Hardcoded CLASS_NAMES Dependency**
- **Location:** Line 69, 129, 182
- **Problem:** Tightly coupled to `trainer.CLASS_NAMES`, breaking if order changes
- **Risk:** Checkpoint becomes unusable if trainer.py is refactored
- **Fix:** Store class_names entirely in checkpoint config, remove trainer dependency

**Issue #11: No Model Performance Metrics**
- **Location:** Throughout
- **Problem:** Doesn't track inference latency, throughput, or error rates
- **Fix:** Add Prometheus histograms:
```python
from prometheus_client import Histogram
_inference_latency = Histogram(
    'sentinel_inference_latency_seconds',
    'Time spent running inference',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
```

**Issue #12: Warmup Graph Too Simple**
- **Location:** Lines 303-318
- **Problem:** 2-node 1-edge graph may not exercise all edge cases
- **Risk:** Shape bugs in complex graphs only appear in production
- **Fix:** Add second warmup with more complex graph (5+ nodes, multiple edge types)

---

### 3. `preprocess.py` (532 lines)

**Role:** Online preprocessing converting Solidity to (graph, tokens)

#### ✅ EXCELLENT PRACTICES
1. **Thin Wrapper Architecture** (Lines 26-34): Delegates to shared preprocessing module
2. **SIGKILL-Safe Temp Management** (Lines 78-113): Handles orphaned temp files
3. **Exception Translation** (Lines 367-389): Maps internal exceptions to HTTP-appropriate errors
4. **Sliding Window Support** (Lines 269-293, 453-509): Handles long contracts gracefully
5. **Shape Contracts Documented** (Lines 44-51): Clear specification of expected tensor shapes

#### 🔴 CRITICAL ISSUES

**Issue #13: Unsafe Deserialization in Cache Load**
- **Location:** Lines 88-89
- **Code:** `torch.load(graph_path, map_location="cpu", weights_only=False)`
- **Risk:** **RCE VULNERABILITY** - Cache poisoning attack vector
- **Severity:** **CRITICAL** - Cache files less trusted than checkpoints
- **Fix:** Same as Issue #7 - validate hash or use safe serialization format

**Issue #14: No Validation of Cached Graph Schema**
- **Location:** Lines 92-101
- **Problem:** Only validates `x.shape[1] == 8`, misses other schema fields
- **Risk:** Stale cached graphs with wrong edge_index or missing fields pass validation
- **Fix:** Comprehensive schema validation:
```python
required_attrs = ["x", "edge_index", "contract_hash"]
for attr in required_attrs:
    if not hasattr(graph, attr):
        raise ValueError(f"Cached graph missing required attribute: {attr}")
if graph.edge_index.shape[0] != 2:
    raise ValueError(f"edge_index must have shape [2, E], got {graph.edge_index.shape}")
```

**Issue #15: MD5 Hash Collision Risk**
- **Location:** Lines 234-235 (via `hash_utils.get_contract_hash_from_content`)
- **Problem:** MD5 is cryptographically broken, collisions possible
- **Risk:** Two different contracts could produce same cache key
- **Fix:** Upgrade to SHA-256:
```python
# In hash_utils.py
import hashlib
hashlib.sha256(content.encode()).hexdigest()[:32]  # still 32 chars like MD5
```

#### ⚠️ HIGH-PRIORITY IMPROVEMENTS

**Issue #16: No Cache Size Limit**
- **Location:** Throughout cache.py
- **Problem:** Cache grows indefinitely, no LRU eviction
- **Risk:** Disk space exhaustion in long-running services
- **Fix:** Add max_size parameter and LRU eviction:
```python
def __init__(self, ..., max_entries: int = 10000):
    self.max_entries = max_entries
    
def put(self, ...):
    if len(self._list_keys()) >= self.max_entries:
        self._evict_oldest()
```

**Issue #17: TTL Check Uses File Mtime**
- **Location:** Lines 82-87
- **Problem:** Uses filesystem mtime, which can be manipulated
- **Risk:** Attacker could extend cache entry lifetime
- **Fix:** Store timestamp in file metadata or separate manifest:
```python
# Store timestamp in filename or sidecar file
# Format: {key}_{timestamp}_graph.pt
```

**Issue #18: No Cache Hit/Miss Metrics**
- **Location:** cache.py
- **Problem:** Doesn't expose cache performance metrics
- **Fix:** Add Prometheus counters:
```python
_cache_hits = Counter('sentinel_cache_hits_total', 'Cache hit count')
_cache_misses = Counter('sentinel_cache_misses_total', 'Cache miss count')
```

**Issue #19: Tokenizer Download on First Request**
- **Location:** Line 142
- **Problem:** `AutoTokenizer.from_pretrained()` downloads on first call
- **Risk:** First request latency spike (5-30s depending on network)
- **Fix:** Pre-download in container build or add explicit warmup method:
```python
def warmup_tokenizer(self):
    logger.info("Pre-downloading tokenizer...")
    _ = self.tokenizer("pragma solidity ^0.8.0; contract Test {}")
```

---

### 4. `cache.py` (154 lines)

**Role:** Disk-backed content-addressed cache for inference results

#### ✅ GOOD PRACTICES
1. **Atomic Writes** (Lines 127-131): tmp file + rename pattern
2. **Schema Versioning** (Lines 16-19): Cache invalidation on schema change
3. **Thread Safety** (Lines 27-32): Clear documentation of thread safety guarantees
4. **Defensive Exception Handling** (Lines 105-108): Treats any load error as cache miss

#### 🔴 CRITICAL ISSUES

**Issue #20: Duplicate of Issue #13** - Unsafe `torch.load` (Lines 88-89)

**Issue #21: No Cache Integrity Verification**
- **Location:** Lines 88-103
- **Problem:** No checksum verification of cached files
- **Risk:** Corrupted cache entries silently return wrong data
- **Fix:** Store and verify checksums:
```python
# On write:
checksum = hashlib.sha256(pickle.dumps(obj)).hexdigest()
metadata = {"checksum": checksum, "created_at": time.time()}
torch.save({"data": obj, "meta": metadata}, tmp)

# On read:
loaded = torch.load(path)
expected = loaded["meta"]["checksum"]
actual = hashlib.sha256(pickle.dumps(loaded["data"])).hexdigest()
if expected != actual:
    raise ValueError("Cache integrity check failed")
```

#### ⚠️ IMPROVEMENTS

**Issue #22: Fixed TTL for All Entries**
- **Location:** Line 60
- **Problem:** All entries have same TTL (24h default)
- **Risk:** Frequently accessed entries expire unnecessarily
- **Fix:** Implement adaptive TTL based on access frequency

---

### 5. `drift_detector.py` (193 lines)

**Role:** KS-test based drift detection for production monitoring

#### ✅ EXCELLENT PRACTICES
1. **Clear Baseline Strategy** (Lines 22-32): Warns against using training data
2. **Two-Phase Operation** (Lines 24-27): Warm-up phase prevents false positives
3. **Prometheus Integration** (Lines 40-44): Proper counter with labels
4. **Lazy Import** (Line 77): scipy imported only when needed

#### 🔴 CRITICAL ISSUES

**Issue #23: Insufficient Warmup Samples**
- **Location:** Line 51 (`MIN_SAMPLES_FOR_KS = 30`)
- **Problem:** 30 samples too few for reliable KS test
- **Risk:** High false positive rate in early production
- **Fix:** Increase to 100-200 samples:
```python
MIN_SAMPLES_FOR_KS = 100  # Statistical reliability
```

**Issue #24: No Baseline Validation**
- **Location:** Lines 85-96
- **Problem:** Doesn't validate baseline has sufficient samples
- **Risk:** Baseline with <30 samples produces unreliable KS tests
- **Fix:** Add validation on load:
```python
for stat_name, values in self._baseline.items():
    if len(values) < 100:
        logger.warning(f"Baseline for {stat_name} has only {len(values)} samples (recommended: 100+)")
```

**Issue #25: Hardcoded p-value Threshold**
- **Location:** Line 47 (`KS_ALPHA = 0.05`)
- **Problem:** No way to adjust sensitivity without code change
- **Risk:** May need tuning for specific production environment
- **Fix:** Make configurable:
```python
KS_ALPHA = float(os.getenv("SENTINEL_DRIFT_ALPHA", "0.05"))
```

#### ⚠️ IMPROVEMENTS

**Issue #26: No Drift Severity Levels**
- **Location:** Lines 133-140
- **Problem:** Binary alert (drift/no-drift), no severity gradation
- **Fix:** Add severity levels based on p-value:
```python
if p_value < 0.001:
    severity = "CRITICAL"
elif p_value < 0.01:
    severity = "HIGH"
elif p_value < 0.05:
    severity = "MEDIUM"
```

**Issue #27: No Baseline Auto-Update**
- **Location:** Throughout
- **Problem:** Baseline never updates, may become stale over months
- **Risk:** Legitimate distribution shifts trigger constant alerts
- **Fix:** Add optional exponential moving average update:
```python
if os.getenv("SENTINEL_DRIFT_ADAPTIVE", "false") == "true":
    # Blend 10% current window into baseline
    self._update_baseline(current_values, alpha=0.1)
```

---

### 6. `__init__.py` (0 lines)

**Assessment:** Empty package marker - acceptable

**Recommendation:** Add package metadata and exports:
```python
"""SENTINEL Inference Module"""
from .api import app
from .predictor import Predictor
from .preprocess import ContractPreprocessor
from .cache import InferenceCache
from .drift_detector import DriftDetector

__all__ = ["app", "Predictor", "ContractPreprocessor", "InferenceCache", "DriftDetector"]
__version__ = "3.0.0"
```

---

## 📊 Summary Table

| Priority | Module | Issue | Severity | Impact | Effort |
|----------|--------|-------|----------|--------|--------|
| **P0** | ALL | Unsafe `torch.load(weights_only=False)` | 🔴 Critical | RCE Vulnerability | Low |
| **P0** | `cache.py` | No cache integrity verification | 🔴 Critical | Data corruption | Medium |
| **P0** | `preprocess.py` | MD5 hash collision risk | 🔴 Critical | Cache poisoning | Low |
| **P0** | `api.py` | No rate limiting | 🔴 Critical | DoS vulnerability | Medium |
| **P1** | `predictor.py` | Silent fallback to uniform threshold | ⚠️ High | Suboptimal predictions | Low |
| **P1** | `drift_detector.py` | Insufficient warmup samples (30) | ⚠️ High | False positives | Low |
| **P1** | `cache.py` | No cache size limit | ⚠️ High | Disk exhaustion | Medium |
| **P1** | `api.py` | No model version in response | ⚠️ High | Traceability gap | Low |
| **P2** | `predictor.py` | No inference latency metrics | 🟡 Medium | Observability gap | Low |
| **P2** | `preprocess.py` | Tokenizer download on first request | 🟡 Medium | Latency spike | Low |
| **P2** | `drift_detector.py` | No drift severity levels | 🟡 Medium | Alert fatigue | Low |
| **P2** | `api.py` | No request/response structured logging | 🟡 Medium | Debugging difficulty | Low |

---

## 🚀 Recommended Actions (Prioritized)

### Immediate (Before Production Deployment)
1. **Fix ALL `torch.load` calls** → Change to `weights_only=True` where possible, add hash validation where not
2. **Add cache integrity verification** → SHA-256 checksums for all cached files
3. **Upgrade from MD5 to SHA-256** → In `hash_utils.py` for contract hashing
4. **Add rate limiting** → Implement slowapi or similar in `api.py`
5. **Increase MIN_SAMPLES_FOR_KS** → From 30 to 100 in `drift_detector.py`

### Short-Term (Before Scale-Up)
6. **Add cache size limits** → LRU eviction with configurable max_entries
7. **Add model version to response** → Include checkpoint hash in `PredictResponse`
8. **Add inference metrics** → Prometheus histograms for latency, throughput
9. **Validate threshold ranges** → Ensure all thresholds in (0, 1)
10. **Add structured logging** → Correlation IDs, source hashes

### Long-Term (Continuous Improvement)
11. **Implement adaptive TTL** → Based on access frequency
12. **Add drift severity levels** → CRITICAL/HIGH/MEDIUM based on p-value
13. **Consider baseline auto-update** → Optional EMA blending for long-term deployments
14. **Remove CLASS_NAMES coupling** → Store entirely in checkpoint config
15. **Add comprehensive warmup** → Multiple graph complexities at startup

---

## 📁 Files Requiring Changes

| File | Lines to Change | Estimated Effort |
|------|----------------|------------------|
| `api.py` | 50 lines (rate limiting, logging, validation) | 3 hours |
| `predictor.py` | 40 lines (threshold validation, metrics) | 2 hours |
| `preprocess.py` | 60 lines (hash upgrade, validation, metrics) | 3 hours |
| `cache.py` | 80 lines (integrity checks, LRU, metrics) | 4 hours |
| `drift_detector.py` | 30 lines (configurable alpha, validation) | 2 hours |
| `../utils/hash_utils.py` | 20 lines (SHA-256 upgrade) | 1 hour |

**Total Estimated Effort:** ~15 hours

---

## 🔒 Security Considerations

### Attack Vectors Identified
1. **Checkpoint Poisoning** - Malicious `.pt` file executes arbitrary code
2. **Cache Poisoning** - Corrupted cache entries cause wrong predictions
3. **Hash Collision** - MD5 collisions allow cache bypass
4. **DoS via Unlimited Requests** - No rate limiting on `/predict`
5. **Disk Exhaustion** - Unbounded cache growth

### Mitigation Priority
1. **P0**: Fix unsafe deserialization (Issues #7, #13, #20)
2. **P0**: Add cache integrity verification (Issue #21)
3. **P0**: Upgrade to SHA-256 (Issue #15)
4. **P0**: Add rate limiting (Issue #2)

---

## 🧪 Testing Recommendations

### Unit Tests Needed
- [ ] Cache integrity verification (valid/corrupted files)
- [ ] Threshold validation (edge cases: 0, 1, negative, >1)
- [ ] Drift detector with various sample sizes
- [ ] Rate limiting behavior
- [ ] Hash function collision resistance

### Integration Tests Needed
- [ ] End-to-end prediction with corrupted cache
- [ ] Model version tracking across deployments
- [ ] Drift detection with synthetic distribution shift
- [ ] Cache eviction under memory pressure

### Load Tests Needed
- [ ] Sustained high-throughput inference (100+ req/min)
- [ ] Cache hit/miss ratio under realistic workload
- [ ] Memory leak detection over 24h runtime

---

## 📋 Next Steps

**Group 7 (Utilities & Analysis Scripts)** is the next logical group to audit, followed by **Group 8 (Testing Suite)** to ensure adequate test coverage for all identified issues.

**Recommended Order:**
1. Fix P0 security issues in Group 6 (4-6 hours)
2. Audit Group 7 (Utilities) - 2 hours
3. Audit Group 8 (Tests) - 3 hours
4. Implement remaining P1/P2 improvements (10-12 hours)

---

**Audit Completed:** 2026-05-10  
**Next Group:** Group 7 (Utilities & Analysis Scripts)
