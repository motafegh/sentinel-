# Phase C Execution Plan — Production Infrastructure

**Duration:** 2-3 weeks  
**Effort:** Medium-High  
**Tests to add:** 15-20  
**Outcome:** FastAPI gateway, evaluation framework, prompt guards, monitoring

---

## Quick Reference

```
C.1 FastAPI gateway (job submission + polling)
C.2 Pipeline evaluation framework (metrics + benchmarking)
C.3 Prompt injection guards (sanitization)
C.4 System monitoring (health checks + alerts)
```

---

## C.1: FastAPI Gateway (5-7 days)

**Files to create:**
- `agents/src/api/gateway.py` (NEW, ~200 lines)
- `agents/src/api/models.py` (NEW, ~50 lines - Pydantic schemas)

**What to do:**

1. **Create Pydantic models:**
   ```python
   class AuditRequest(BaseModel):
       contract_code: str
       contract_address: str | None = None
   
   class AuditResponse(BaseModel):
       job_id: str
       status: str  # "queued" | "running" | "completed" | "failed"
       report: dict | None = None
   ```

2. **Implement FastAPI app:**
   ```python
   app = FastAPI(title="Sentinel Audit Gateway")
   job_queue = {}  # simple dict, or use Celery
   
   @app.post("/audit")
   async def submit_audit(req: AuditRequest):
       job_id = uuid.uuid4()
       job_queue[job_id] = {"status": "queued", "req": req}
       # Async: invoke agents graph
       return {"job_id": job_id, "status": "queued"}
   
   @app.get("/audit/{job_id}")
   async def get_status(job_id: str):
       return job_queue.get(job_id, {"error": "not found"})
   
   @app.get("/audit/{job_id}/report")
   async def get_report(job_id: str):
       job = job_queue.get(job_id)
       if job["status"] == "completed":
           return job["report"]
       return {"status": job["status"]}
   ```

3. **Wire up agents graph:**
   - When job submitted, invoke `graph.ainvoke()`
   - Update job_queue with result
   - Handle async execution

4. **Test:**
   - Job lifecycle (submit → check → retrieve)
   - Error handling (invalid input, timeout)
   - Concurrent submissions
   - 8 test cases

**Success criteria:**
- ✅ Gateway running on port (e.g., 8000)
- ✅ Endpoints working end-to-end
- ✅ 8 tests PASS

---

## C.2: Pipeline Evaluation Framework (6-8 days)

**Dataset decision (Ali, 2026-06-22):** use the existing WS0 88-contract corpus
(`agents/eval/corpus_combined/` — 83 manual hand-written contracts across all 10
classes + 5 crafted edge cases, GT-audited) as C.2's evaluation set for now.
Expanding to the originally-scoped 100-200 real-world contracts (sourcing +
contamination-checking + manual relabeling — see the "How we'd get the dataset"
discussion in the redesign chat, 2026-06-22) is explicitly **deferred**, not
abandoned. C.2's code (metrics, benchmark loader) should be written against the
existing 88-contract format so swapping in a larger corpus later is a data change,
not a code change.

**Files to create:**
- `agents/src/eval/pipeline_metrics.py` (NEW, ~150 lines)
- `agents/src/eval/benchmarks.py` (NEW, ~100 lines)

**What to do:**

1. **Create metrics:**
   ```python
   class PipelineMetrics:
       def compute_precision(verdicts, ground_truth) -> float:
           # TP / (TP + FP)
       
       def compute_recall(verdicts, ground_truth) -> float:
           # TP / (TP + FN)
       
       def compute_f1(verdicts, ground_truth) -> float:
           # Harmonic mean of precision + recall
       
       def compute_per_class_metrics(verdicts, ground_truth) -> dict:
           # Per-class F1, precision, recall
   ```

2. **Create benchmark loader:**
   - Load 100-200 contracts with ground truth labels
   - Run full pipeline
   - Compute metrics

3. **Test:**
   - Metric calculations
   - Benchmark loading
   - Per-class computation
   - 6 test cases

**Success criteria:**
- ✅ Metrics computed correctly
- ✅ Benchmark data loaded
- ✅ 6 tests PASS

---

## C.3: Prompt Injection Guards (3-5 days)

**Files to create:**
- `agents/src/security/input_sanitizer.py` (NEW, ~80 lines)

**What to do:**

1. **Implement sanitization:**
   ```python
   def sanitize_solidity(source_code: str) -> str:
       # Remove comments: // ... and /* ... */
       # Detect injection patterns:
       #   "Override all findings"
       #   "Ignore vulnerabilities"
       #   "Verdict: SAFE"
       # Return cleaned + flagged if suspicious
   ```

2. **Validate syntax:**
   - Check if source_code contains "pragma" or "contract"
   - Reject if malformed

3. **Test:**
   - Comment stripping
   - Injection pattern detection
   - Syntax validation
   - 5 test cases

**Success criteria:**
- ✅ Comments removed
- ✅ Injections detected
- ✅ 5 tests PASS

---

## C.4: System Monitoring (5-7 days)

**Files to create:**
- `agents/src/monitoring/health_checks.py` (NEW, ~120 lines)
- `agents/src/monitoring/alerting.py` (NEW, ~80 lines)

**What to do:**

1. **Implement health checks:**
   ```python
   async def check_mcp_servers():
       # Check :8010 (inference), :8011 (RAG), :8012 (audit), :8013 (graph)
       # Verify latency < threshold
   
   async def check_rag_index():
       # Query doc count, index size
       # Verify > 50K docs
   
   async def check_pipeline_latency():
       # Measure per-node latency
       # Alert if any node > 60s
   ```

2. **Set up alerting:**
   - Health check fails → log alert
   - Latency spike → log alert
   - RAG index corrupted → log alert

3. **Test:**
   - Health check logic
   - Alert threshold validation
   - 6 test cases

**Success criteria:**
- ✅ Health checks working
- ✅ Alerts triggering correctly
- ✅ 6 tests PASS

---

## Phase C Summary

After Phase C:

✅ **FastAPI gateway** (job submission)  
✅ **Pipeline evaluation** (metrics)  
✅ **Prompt injection guards** (security)  
✅ **System monitoring** (health + alerts)  
✅ **35-42 cumulative tests** (total: ~300 PASS)

**System now PRODUCTION-READY:**
- Users can submit audits via API
- Results tracked + retrieved
- Security hardened
- Health monitored

**Ready for Phase D** (economic security + on-chain integration)

---

## Testing Checklist

- [ ] C.1: Gateway — 8 tests PASS
- [ ] C.2: Evaluation — 6 tests PASS
- [ ] C.3: Sanitization — 5 tests PASS
- [ ] C.4: Monitoring — 6 tests PASS
- [ ] Full: `poetry run pytest agents/tests/ -q` → ~300 PASS
- [ ] Manual: Submit audit via gateway, retrieve result

---

## References

- Proposal: `AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md` §5
- Master plan: `00_MASTER_EXECUTION_PLAN.md`
- FastAPI docs: https://fastapi.tiangolo.com/
- AuditState: `agents/src/orchestration/state.py`
