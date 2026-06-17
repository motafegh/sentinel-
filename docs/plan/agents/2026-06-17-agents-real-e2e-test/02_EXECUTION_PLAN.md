# Execution Plan — Running Real Audits

**Duration:** 1-2 hours per contract (3 contracts = 3-6 hours total)  
**Scope:** Run the agents graph on real contracts and capture output  

---

## Prerequisites

Before starting, verify:
- ✅ LM Studio running at :1234
- ✅ ML API running at :8001 (MLOps standard)
- ✅ All 4 MCP servers running (:8010, :8011, :8012, :8013)
- ✅ Connectivity check passed
- ✅ Test contracts created

(If not done yet, complete `01_SETUP_PLAN.md` first)

---

## Create Test Harness Script

Create `agents/scripts/run_real_audit.py`:

```python
import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Add agents to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.orchestration.graph import build_graph
from src.orchestration.state import AuditState


async def run_audit(contract_path: str, contract_address: str = None) -> dict:
    """Run a real audit on a contract file."""
    
    # Read contract
    with open(contract_path) as f:
        contract_code = f.read()
    
    if not contract_address:
        contract_address = f"0x{Path(contract_path).stem}"
    
    print(f"\n{'='*60}")
    print(f"AUDITING: {Path(contract_path).name}")
    print(f"Address: {contract_address}")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    # Build graph (this will use real MCP servers)
    graph = build_graph(use_checkpointer=False)
    
    # Prepare initial state
    initial_state = {
        "contract_code": contract_code,
        "contract_address": contract_address,
    }
    
    # Track timing
    timings = {}
    
    # Run the graph
    try:
        start = time.time()
        result = await graph.ainvoke(initial_state)
        total_time = time.time() - start
        
        # Extract results
        final_report = result.get("final_report", {})
        
        # Print summary
        print(f"\n{'='*60}")
        print("AUDIT COMPLETE")
        print(f"{'='*60}")
        print(f"Overall verdict: {final_report.get('overall_label', 'N/A')}")
        print(f"Total time: {total_time:.2f}s")
        print(f"\nVerdicts by class:")
        
        verdicts = final_report.get('vulnerability_verdicts', {})
        for cls, verdict in verdicts.items():
            print(f"  {cls:25} {verdict}")
        
        print(f"\nTop vulnerability: {final_report.get('top_vulnerability', 'N/A')}")
        print(f"Risk probability: {final_report.get('risk_probability', 'N/A'):.3f}")
        
        # Save full report
        report_file = Path(f"test_audit_reports/{Path(contract_path).stem}_report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump({
                "contract": Path(contract_path).name,
                "address": contract_address,
                "total_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "report": final_report,
            }, f, indent=2)
        
        print(f"\nReport saved: {report_file}")
        
        return {
            "success": True,
            "contract": Path(contract_path).name,
            "total_time": total_time,
            "verdict": final_report.get("overall_label"),
        }
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "contract": Path(contract_path).name,
            "error": str(e),
        }


async def main():
    contracts = [
        "test_contracts/erc20_safe.sol",
        "test_contracts/vulnerable_reentrant.sol",
    ]
    
    results = []
    for contract in contracts:
        result = await run_audit(contract)
        results.append(result)
        print("\n\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{status}: {r['contract']:40} {r.get('total_time', 'error'):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Test 1: Safe Contract (Baseline)

**Run:**

```bash
cd ~/projects/sentinel/agents
poetry run python scripts/run_real_audit.py test_contracts/erc20_safe.sol

# Expected output:
# ============================================================
# AUDITING: erc20_safe.sol
# ...
# AUDIT COMPLETE
# ============================================================
# Overall verdict: SAFE
# Total time: 45.5s
# Verdicts by class:
#   Reentrancy         SAFE
#   IntegerUO          SAFE
#   ExternalBug        LIKELY
#   ...
```

**What to observe:**
- Does it complete without error?
- What's the total time?
- Does verdict make sense? (should be mostly SAFE)
- Any MCP server errors in the background terminals?

**Measurements to record:**
- [ ] Total time: ___s
- [ ] Memory peak: ___MB
- [ ] Verdict: ________
- [ ] Any errors: Yes/No

---

## Test 2: Vulnerable Contract

**Run:**

```bash
cd ~/projects/sentinel/agents
poetry run python scripts/run_real_audit.py test_contracts/vulnerable_reentrant.sol

# Expected output:
# ============================================================
# AUDITING: vulnerable_reentrant.sol
# ...
# AUDIT COMPLETE
# ============================================================
# Overall verdict: CONFIRMED_VULNERABLE
# Total time: 48.2s
# Verdicts by class:
#   Reentrancy         CONFIRMED
#   ...
```

**What to observe:**
- Does it detect the reentrancy bug?
- Confidence level?
- Cross_validator reasoning (did prosecutor/defender explain it well)?

**Measurements to record:**
- [ ] Total time: ___s
- [ ] Memory peak: ___MB
- [ ] Verdict: ________
- [ ] Reentrancy detected: Yes/No
- [ ] Confidence: ___%

---

## Test 3: Real Contract (Optional)

If you have a real contract from Etherscan:

```bash
cd ~/projects/sentinel/agents
poetry run python scripts/run_real_audit.py test_contracts/real_contract.sol

# Will take 1-2 hours if contract is large
# Don't do this test if time-constrained
```

---

## What to Monitor During Execution

**In a separate terminal, run:**

```bash
# Monitor memory every 5 seconds
watch -n 5 'ps aux | grep python | grep -v grep | sort -k6 -rn | head -10'

# Or use top
top -u $(whoami) -b
```

**Record:**
- Peak memory usage
- CPU usage patterns
- Any OOM errors

---

## Interpreting Results

### Timing Expectations

| Component | Expected | OK | Slow | Broken |
|-----------|----------|---|------|--------|
| ml_assessment | 10-30s | <40s | 40-60s | >60s |
| static_analysis | 5-20s | <30s | 30-45s | >45s |
| rag_research | 2-5s | <10s | 10-20s | >20s |
| cross_validator | 15-30s | <45s | 45-60s | >60s |
| **Total** | **45-60s** | **<120s** | **120-180s** | **>180s** |

### Verdict Quality

Check the saved JSON report for:

```json
{
  "verdict": "CONFIRMED_VULNERABLE",
  "confidence": 0.92,
  "evidence": [
    "ml:0.87",
    "slither:reentrancy-eth",
    "rag:0.81"
  ]
}
```

**Good signs:**
- Evidence sources are diverse (ML + Slither + RAG)
- Confidence is calibrated (0.8-0.95 for CONFIRMED)
- Verdicts match contract behavior

**Red flags:**
- All evidence from one source (over-reliant)
- Confidence misaligned with evidence
- Prosecutor/Defender disagreed badly

---

## Execution Checklist

**Before each test:**
- [ ] All 5 services still running (spot-check :8010-:8013)
- [ ] Sufficient free RAM (> 3GB)
- [ ] Contract file exists

**During each test:**
- [ ] Monitor memory
- [ ] Note any error messages in service terminals
- [ ] Record timing from script output

**After each test:**
- [ ] Report file saved at `test_audit_reports/`
- [ ] Timing recorded
- [ ] Memory peak recorded
- [ ] Verdict reviewed

---

## Common Issues & Recovery

| Issue | Solution |
|-------|----------|
| MCP server crash | Restart that service, re-run audit |
| LLM timeout | Increase timeout in cross_validator, restart ML API |
| OutOfMemory | Close other apps, restart ML API |
| "No such file" | Verify contract path in script |
| Empty report | Check JSON file, might be parsing error |

---

## Success Criteria

✅ All 3 contracts run without crashes  
✅ Total time < 2 minutes per contract  
✅ Verdicts are coherent and defensible  
✅ RAG retrieval returns relevant docs  
✅ No MCP errors in background terminals  

---

## Output Files

After running audits, you'll have:

```
agents/test_audit_reports/
  ├─ erc20_safe_report.json
  ├─ vulnerable_reentrant_report.json
  └─ real_contract_report.json (if test 3)
```

These JSON files contain:
- Full audit verdict
- All evidence sources
- Cross_validator reasoning
- Timing metrics
- Contract details

---

## Next Steps

When audits are complete:
1. ✓ Review JSON reports
2. → Open `03_ANALYSIS_PLAN.md`
3. → Analyze results + document findings

---

**Estimated total execution time:** 1-2 hours for all 3 contracts

**Next:** `03_ANALYSIS_PLAN.md` →
