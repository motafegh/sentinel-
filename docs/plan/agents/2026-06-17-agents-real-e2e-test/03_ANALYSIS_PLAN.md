# Analysis Plan — Understanding Results & Debugging Issues

**Duration:** 2-3 hours (after running all 3 tests)  
**Scope:** Analyze JSON reports, debug issues, document findings, establish baselines  

---

## Create Scratch File

Before analyzing, create a working file for notes:

```bash
touch ~/.claude/scratch/agents_e2e_test_analysis_20260617.md
```

Use this to record:
- Each test's timing breakdown
- Quality observations (LLM reasoning, RAG relevance)
- Issues found + fixes applied
- Baseline performance numbers
- Recommendations for Phase A

---

## Step 1: Load & Review JSON Reports (30 minutes)

After running all 3 audits, you'll have reports in:

```
agents/test_audit_reports/
├─ erc20_safe_report.json
├─ vulnerable_reentrant_report.json
└─ real_contract_report.json (if test 3)
```

### Parse Each Report

Open each JSON and record:

```bash
cd ~/projects/sentinel/agents

# Contract 1: Safe
cat test_audit_reports/erc20_safe_report.json | jq '.report' | head -50

# Contract 2: Vulnerable
cat test_audit_reports/vulnerable_reentrant_report.json | jq '.report' | head -50

# Contract 3: Real (if done)
cat test_audit_reports/real_contract_report.json | jq '.report' | head -50
```

### What to Extract

For each report, document in your scratch file:

```markdown
## Test 1: erc20_safe.sol

### Performance
- Total time: ___s
- Memory peak: ___MB
- timestamp: ___

### Verdict
- Overall label: SAFE / LIKELY_VULNERABLE / CONFIRMED_VULNERABLE
- Risk probability: ___
- Top vulnerability: ___

### Evidence Sources
- ML assessment: P=___ Label=___
- Slither findings: [list]
- Aderyn findings: [list]
- RAG docs: [top 3 retrieved]

### LLM Reasoning (cross_validator)
- Prosecutor argument: [summarize]
- Defender argument: [summarize]
- Judge verdict: [verdict + reasoning]

### Quality Observations
- [ ] Prosecutor/Defender arguments coherent?
- [ ] Judge reasoning makes sense?
- [ ] Verdict aligns with evidence?
- [ ] Any hallucinations or wrong info?
- [ ] RAG docs actually relevant?
```

---

## Step 2: Extract Performance Metrics (30 minutes)

### Timing Breakdown

For each contract, create a timing table. Edit your scratch file to capture per-node latency:

```bash
# Extract from json (if node timings were captured)
jq '.report | keys' test_audit_reports/erc20_safe_report.json
```

**Expected fields in report:**
- `ml_assessment_time`
- `quick_screen_time`
- `rag_research_time`
- `static_analysis_time`
- `cross_validator_time`
- `synthesizer_time`
- `total_time`

### Create Timing Table

In your scratch file:

```markdown
## Performance Summary

| Contract | ML | QS | RAG | Static | XVal | Synth | Total |
|----------|----|----|-----|--------|------|-------|-------|
| erc20_safe | 15s | 8s | 3s | 12s | 20s | 2s | 60s |
| vuln_reent | 16s | 9s | 4s | 10s | 22s | 2s | 63s |
| real_dapp | (if done) | ... |

- Expected total: 45-120s
- OK: <120s
- Slow: 120-180s
- Broken: >180s
```

### Memory Analysis

```bash
# During tests, you should have recorded peak memory
# Create a table:

## Memory Usage

| Contract | Peak RAM | Notes |
|----------|----------|-------|
| erc20_safe | ___MB | Graph only |
| vuln_reent | ___MB | Graph + RAG |
| real_dapp | ___MB | (if done) |

- Expected: < 4GB total
- Acceptable: 4-6GB
- High: 6-8GB
- Danger: > 8GB
```

---

## Step 3: Analyze Verdict Quality (45 minutes)

### Coherence Check

For each contract, review the LLM reasoning:

```markdown
## Quality Analysis — erc20_safe.sol

### Prosecutor (accusation)
Text: "..."
Issues:
- [ ] Makes logical sense?
- [ ] References evidence correctly?
- [ ] Any hallucinations (false evidence)?
- [ ] Tone professional?

### Defender (defense)
Text: "..."
Issues:
- [ ] Rebuts prosecutor fairly?
- [ ] Adds new evidence?
- [ ] Any strawman arguments?

### Judge (verdict)
Text: "..."
Issues:
- [ ] Verdict matches evidence weight?
- [ ] Confidence justified?
- [ ] Explains reasoning?
- [ ] Acknowledges uncertainty?
```

### Verdict Alignment

Check if verdict matches what we know:

```markdown
## Verdict Correctness

| Contract | Expected | Got | Match |
|----------|----------|-----|-------|
| erc20_safe | SAFE | ? | Yes/No |
| vuln_reent | VULNERABLE | ? | Yes/No |
| real_dapp | (depends) | ? | Yes/No |

- Count correct: __/3
- Count incorrect: __/3
- Accuracy: __%
```

### False Positive Analysis

If safe contract flagged as vulnerable:

```markdown
## False Positives

Contract: erc20_safe.sol
Verdict: LIKELY_VULNERABLE (WRONG)
Claimed vulnerability: Reentrancy

**Analysis:**
- Where did ML go wrong? [check probabilities]
- Did Slither agree? [check slither output]
- Was RAG misleading? [check retrieved docs]
- What pattern did it misidentify? [analyze code]

**Root cause:** [hypothesis]

**Lesson for Phase A:** [what to fix]
```

### False Negative Analysis

If vulnerable contract marked as safe:

```markdown
## False Negatives

Contract: vulnerable_reentrant.sol
Verdict: SAFE (WRONG — should be VULNERABLE)
Missed vulnerability: Reentrancy in withdraw()

**Analysis:**
- Why didn't ML detect? [check ML probabilities]
- Why didn't Slither find it? [check slither config]
- Did RAG help? [check retrieved docs]
- Did LLM misinterpret evidence? [check judge reasoning]

**Root cause:** [hypothesis]

**Lesson for Phase A:** [what to fix]
```

---

## Step 4: RAG Quality Assessment (30 minutes)

### Document Retrieval Scoring

For each contract, evaluate RAG quality:

```bash
# Extract top RAG documents from report
jq '.report.rag_documents | .[0:5]' test_audit_reports/erc20_safe_report.json
```

### Relevance Scoring

Score each retrieved document as: **relevant (1)** or **irrelevant (0)**

```markdown
## RAG Quality — erc20_safe.sol

| Doc # | Title | Vulnerability Type | Relevant | Why |
|-------|-------|-------------------|----------|-----|
| 1 | "Reentrancy in tokens" | Reentrancy | 1 | Mentions transfer pattern |
| 2 | "Integer overflow bug" | IntegerUO | 0 | Contract has no math ops |
| 3 | "Access control" | AccessControl | 1 | Contract has public functions |

Precision: 2/3 = 67%
Expected: > 70%
Status: OK / NEEDS_WORK
```

### RAG Impact

Did RAG help the verdict?

```markdown
## RAG Helpfulness

Contract: erc20_safe.sol
Prosecutor's RAG usage: [quote where prosecutor cited RAG]
Judge's RAG consideration: [did judge mention RAG?]

Assessment:
- [ ] RAG docs changed verdict?
- [ ] RAG docs reinforced correct verdict?
- [ ] RAG docs were distracting?
- [ ] RAG docs were ignored?

Recommendation: [keep/expand/deprioritize RAG]
```

---

## Step 5: Identify Bugs & Issues (30 minutes)

### Execution Issues

Document any crashes, timeouts, errors:

```markdown
## Bugs Found

### Bug #1: MCP Server Timeout
**When:** Contract 1, during rag_research
**Error:** "Connection timeout after 30s"
**Frequency:** Once out of 3 tests
**Severity:** Medium (recoverable)
**Root cause:** RAG query took > 30s
**Fix:** Increase timeout in rag_server.py line XXX
**Status:** DOCUMENTED / FIXED / DEFERRED

### Bug #2: LLM Hallucination
**When:** Contract 2, cross_validator step
**Error:** Judge mentioned "SafeMath" library but code uses native math
**Frequency:** 1 hallucination in 3 tests
**Severity:** Low (non-critical)
**Root cause:** LLM model confusion
**Fix:** Add prompt instruction to not hallucinate (Phase C)
**Status:** DOCUMENTED / ACCEPTED_RISK / DEFERRED

### Bug #3: Memory Leak
**When:** After 3rd contract, memory didn't drop
**Error:** Peak memory 6.2GB, dropped to 4.1GB (no full GC)
**Frequency:** Consistent
**Severity:** Medium (limits batch processing)
**Root cause:** Graph not releasing references
**Fix:** Add explicit cleanup in orchestrator (Phase A)
**Status:** DOCUMENTED / DEFERRED_TO_PHASE_A
```

### Integration Issues

Any integration problems between nodes?

```markdown
## Integration Issues

### State Serialization
- [ ] AuditState properly passed between nodes?
- [ ] All fields populated?
- [ ] No data loss between steps?

### Concurrent Execution
- [ ] Parallel nodes (rag_research + static_analysis) conflict?
- [ ] Results properly merged?
- [ ] No race conditions?

### Error Handling
- [ ] One failing node stops graph?
- [ ] Graceful degradation?
- [ ] Error messages clear?
```

---

## Step 6: Document Baselines (45 minutes)

Create a "BASELINE" section in your scratch file. This is the reference point for Phase A:

```markdown
# BASELINE PERFORMANCE (Pre-Phase A)

## Summary Stats
- Total audits: 3
- Success rate: 100% / 67% / 33%
- Avg end-to-end time: ___s
- Avg memory peak: ___MB
- Verdict accuracy: __% (for known contracts)

## Timing Baselines

| Node | P50 | P95 | Max |
|------|-----|-----|-----|
| ml_assessment | 15s | 18s | 25s |
| quick_screen | 8s | 10s | 14s |
| rag_research | 3s | 5s | 10s |
| static_analysis | 12s | 14s | 18s |
| cross_validator | 20s | 28s | 35s |
| synthesizer | 2s | 2s | 3s |
| **Total** | **60s** | **77s** | **105s** |

## Quality Baselines

| Metric | Value | Target |
|--------|-------|--------|
| LLM coherence score | 4/5 | 4.5/5 |
| RAG precision | 67% | 75% |
| Verdict accuracy | 67% | 95% |
| False positive rate | 0/1 | < 10% |
| False negative rate | 1/1 | < 5% |
| Node reliability | 99% | 99.5% |

## Stability Baselines

- Crashes: 0
- Timeouts: 0 (or list)
- OOM events: 0
- Memory leak: Yes/No
- Error recovery: Good/Fair/Poor

## Critical Findings (if any)

1. __
2. __
3. __

## Recommendations for Phase A

1. [ ] Fix: __
2. [ ] Improve: __
3. [ ] Monitor: __
4. [ ] Defer: __
```

---

## Step 7: Debugging (as needed)

### If Test Hung or Timed Out

```bash
# Check which node is stuck
ps aux | grep -E 'python.*orchestration|inference|rag_server'

# Check if MCP servers are responding
for port in 8010 8011 8012 8013; do
  timeout 2 curl -s http://localhost:$port/health || echo "Port $port: DOWN"
done

# Check logs
tail -f ~/projects/sentinel/agents/logs/*.log &
tail -f ~/projects/sentinel/ml/logs/*.log &

# If stuck on rag_research: check index
ls -lh ~/projects/sentinel/agents/data/index/

# If stuck on ml_assessment: check GPU
nvidia-smi

# If stuck on cross_validator: check LM Studio
curl -s http://localhost:1234/v1/models
```

### If Test Returned Wrong Verdict

```markdown
## Debug: Wrong Verdict

Contract: vulnerable_reentrant.sol
Expected: VULNERABLE
Got: SAFE

### 1. Check ML prediction
- Get ML API output directly:
  ```
  curl -X POST http://localhost:8001/predict \
    -d '{"source_code":"..."}'
  ```
- Look for reentrancy probability
- Should be > 0.7 for this contract

### 2. Check Slither
- Run directly:
  ```bash
  slither test_contracts/vulnerable_reentrant.sol
  ```
- Did it find reentrancy?
- If yes: slither_server might have filtering issue
- If no: slither config issue

### 3. Check RAG
- What docs were retrieved?
- Were any about reentrancy patterns?
- Did they help or confuse?

### 4. Check LLM reasoning
- What did prosecutor say?
- What did defender say?
- Did judge make right call?
- If judge ignored evidence, that's LLM issue → Phase C prompt tuning

### 5. Root Cause
- [ ] ML model is weak (Phase A: retrain)
- [ ] Slither misconfigured (fix config.yaml)
- [ ] LLM was confused (Phase C: prompt tuning)
- [ ] RAG was wrong (Phase A: expand corpus)
```

### If Memory Spiked

```bash
# Check what's using memory
ps aux --sort=-%mem | head -10

# Check GC behavior
python -c "import gc; gc.collect(); print('GC done')"

# Check if ML model staying in memory
curl http://localhost:8001/health  # Should show memory usage

# Recommended fix: Add explicit cleanup
# In graph.py, after each contract audit:
# - release_resources() method
# - Clear cache/indices
# - Run garbage collection
```

---

## Step 8: Write Summary Report (30 minutes)

Create a final report document:

```bash
cat > agents/test_audit_reports/ANALYSIS_SUMMARY.md << 'EOF'
# E2E Test Analysis Summary

**Date:** 2026-06-17  
**Tester:** [Your name]  
**Duration:** [hours spent]  

## Executive Summary

Tested agents module with 3 real contracts. Results: [summary].

## Key Findings

1. **Performance:** [Good/OK/Slow] — avg ___s per audit
2. **Quality:** [High/Medium/Low] — __% accuracy on known vulns
3. **Stability:** [Stable/Good/Issues] — [list of issues]
4. **RAG:** [Helpful/Neutral/Distracting] — 67% relevance

## Metrics

See timing breakdown and quality scores above.

## Issues Found

1. [Issue + severity + fix]
2. [Issue + severity + fix]

## Blockers for Phase A

- [ ] None, proceed immediately
- [ ] Fix Issue #X first
- [ ] Consider tuning Issue #Y

## Recommendations

1. [Action]
2. [Action]

## Next Steps

- [ ] Review this report with team
- [ ] Proceed to Phase A
- [ ] Or fix identified issues first
EOF
```

---

## Checklist for Analysis

- [ ] Created scratch file (`~/.claude/scratch/agents_e2e_test_analysis_20260617.md`)
- [ ] Extracted JSON reports from all 3 contracts
- [ ] Recorded verdict + confidence for each
- [ ] Timing breakdown captured (all 6 nodes)
- [ ] Memory peak recorded for each test
- [ ] LLM reasoning quality assessed (Prosecutor/Defender/Judge)
- [ ] RAG document relevance scored
- [ ] Bugs identified + documented
- [ ] Baselines extracted (timing, quality, stability)
- [ ] Summary report written
- [ ] Recommendations documented

---

## Key Metrics Summary Table

Create this table in your scratch file:

```markdown
## Final Test Summary

| Metric | Test 1 | Test 2 | Test 3 | Baseline Target | Status |
|--------|--------|--------|--------|-----------------|--------|
| Total Time | ___s | ___s | ___s | <120s | OK/SLOW |
| Verdict Correct | Yes/No | Yes/No | Yes/No | 100% | OK/FAIL |
| RAG Precision | __% | __% | __% | 70% | OK/POOR |
| Memory Peak | __MB | __MB | __MB | <4GB | OK/HIGH |
| Crashes | 0 | 0 | 0 | 0 | ✅ |
| Timeouts | 0 | 0 | 0 | 0 | ✅ |
| LLM Quality | 4/5 | 4/5 | 4/5 | 4.5/5 | OK/GOOD |

**Overall Status:** READY_FOR_PHASE_A / NEEDS_FIXES / NEEDS_REDESIGN
```

---

## Common Findings & Interpretations

### Timing

- **Total < 60s:** Excellent, ready for real use
- **60-120s:** Good, acceptable for async API
- **120-180s:** Slow, consider optimization (Phase C)
- **> 180s:** Bottleneck detected, fix before Phase A

### Quality

- **Verdict accuracy 95%+:** Model is reliable
- **Verdict accuracy 70-90%:** Model is OK, consider retraining
- **Verdict accuracy < 70%:** Model needs work (Phase A)

### RAG

- **Relevance > 75%:** Corpus is good, expand in Phase A
- **Relevance 50-75%:** Corpus is OK, needs tuning
- **Relevance < 50%:** Corpus is bad, major expansion needed

### Stability

- **0 crashes, 0 timeouts:** Great, proceed
- **1-2 minor issues:** Document, proceed with caution
- **> 2 issues or 1 critical:** Fix before Phase A

---

## Next Steps After Analysis

✅ Complete analysis (you are here)  
→ Update scratch file with all findings  
→ Review summary report  
→ Make go/no-go decision for Phase A:
   - **GO:** Proceed to Phase A immediately
   - **CAUTION:** Fix identified issues first
   - **NO-GO:** Major redesign needed

---

## Files to Preserve

After analysis, save these to project memory:

1. **Scratch file:** `~/.claude/scratch/agents_e2e_test_analysis_20260617.md`
   - Raw findings, timing data, observations
   
2. **Test reports:** `agents/test_audit_reports/`
   - JSON reports from each contract
   - Analysis summary
   
3. **Baselines:** Add to MEMORY.md
   - Timing P50/P95/Max
   - Accuracy baseline
   - Stability status

---

**Duration:** 2-3 hours for full analysis

**When done:** Update MEMORY.md and create implementation notes for Phase A

**Success:** You have a clear picture of the current agents system + confidence in the architecture
