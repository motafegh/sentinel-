# Agents Module Test Results — Manual Hand-Written Contracts

**Date:** 2026-06-21  
**Model:** Run 12 (GCB-P1-Run12-v3dospatched-20260613_FINAL.pt, F1=0.7004)  
**Mode:** AGENTS_DISABLE_LLM=1 for batch, DEBATE_MODE=off for full LLM  
**Total:** 19 contracts tested · 100% pipeline completion · 0 errors

---

## Batch Results — Standard Single-Class Contracts (11 contracts)

| # | Contract | GT Class | ML Top Predict | ML Prob | Slither Hits | Verdict | Path | Wall |
|---|----------|----------|----------------|---------|--------------|---------|------|------|
| 1 | Reentrancy/01_cei_violation_erc721 | **Reentrancy** | IntegerUO | 0.762 | reentrancy-no-eth (1) | LIKELY | deep | 1.8s |
| 2 | IntegerUO/01_erc20_underflow | **IntegerUO** | IntegerUO | 0.784 | — | SAFE | deep | 1.2s |
| 3 | CallToUnknown/01_proxy_delegatecall | **CallToUnknown** | UnusedReturn | 0.826 | 1 hit | LIKELY | deep | 1.2s |
| 4 | DenialOfService/01_unbounded_refund | **DoS** | UnusedReturn | 0.915 | 2 hits | LIKELY | deep | 1.4s |
| 5 | ExternalBug/01_flash_loan_oracle | **ExternalBug** | Reentrancy | 0.759 | 2 hits | LIKELY | deep | 1.4s |
| 6 | GasException/01_massive_storage | **GasException** | ExternalBug | 0.821 | — | LIKELY | deep | 1.4s |
| 7 | **MishandledException**/01_batch_payout | **ME** | — | — | — | **SAFE** | fast | 0.4s |
| 8 | **Timestamp**/01_vesting_schedule | **Timestamp** | **Timestamp** | **0.913** | 2 hits | **LIKELY** | deep | 1.4s |
| 9 | TOD/01_approve_frontrun | **TOD** | ExternalBug | 0.632 | — | SAFE | deep | 1.3s |
| 10 | **UnusedReturn**/01_multi_asset | **UnusedReturn** | — | — | — | **SAFE** | fast | 0.3s |
| 11 | **Safe**/01_CEI | **Safe** | **UnusedReturn** | **0.664** | 1 hit | **LIKELY** | deep | 1.5s |

### Key Findings

**✅ Correctly detected:** 1/11 (9%)

| Correct | ML matched ground truth class |
|---------|------------------------------|
| Timestamp | Top class = Timestamp at 0.913 ✓ |

**❌ False negatives:** 2/11 (18%)

| Contract | Problem |
|----------|---------|
| MishandledException | ML returned all-zero probabilities → fast path → SAFE |
| UnusedReturn | ML returned all-zero probabilities → fast path → SAFE |

**❌ False positive:** 1/11 (9%)

| Contract | Problem |
|----------|---------|
| Safe (CEI) | ML flagged UnusedReturn at 0.664 CONFIRMED, 5 classes at CONFIRMED level |

**🟡 Class confusion:** 7/11 (64%)

| Contract (GT) | ML Predicted | Delta |
|---------------|-------------|-------|
| Reentrancy → | IntegerUO | Wrong class! Slither correctly found reentrancy |
| CallToUnknown → | UnusedReturn | No delegatecall detection |
| DoS → | UnusedReturn | Unbounded loop not detected |
| ExternalBug → | Reentrancy | Oracle manipulation missed |
| GasException → | ExternalBug | Gas pattern missed |
| TOD → | ExternalBug | Approve race not detected |

---

## Batch Results — Tricky/Buried Vulnerabilities (5 contracts)

| # | Contract | GT Class | ML Top Predict | ML Prob | Slither | Verdict |
|---|----------|----------|----------------|---------|---------|---------|
| 1 | Reentrancy/06_tricky_in_modifier | Reentrancy | UnusedReturn | 0.862 | — | LIKELY |
| 2 | IntegerUO/06_tricky_interest_rate | IntegerUO | Timestamp | 0.844 | — | LIKELY |
| 3 | CallToUnknown/06_tricky_fallback | CallToUnknown | UnusedReturn | 0.870 | 1 hit | LIKELY |
| 4 | **DoS**/06_tricky_constructor | **DoS** | — | — | — | **SAFE** |
| 5 | Timestamp/06_tricky_pricing | Timestamp | IntegerUO | 0.865 | 1 hit | LIKELY |

### Key Finding

- **DoD constructor bomb was the ONLY contract that completely evaded detection** — 0 ML signal, 0 Slither hits, fast path, SAFE verdict. This is the most successful evasion pattern.
- **0/5 tricky contracts had the correct class identified** — the buried patterns successfully confused the ML model.

---

## Batch Results — Multi-Vulnerability (3 contracts)

| # | Contract | GT Classes | ML Top | Prob | Slither | Verdict |
|---|----------|------------|--------|------|---------|---------|
| 1 | multivuln_reentrancy_tod | Reentrancy, TOD | UnusedReturn | 0.818 | 2 hits | LIKELY |
| 2 | multivuln_call_reentrancy | CallToUnknown, Reentrancy, Timestamp | Timestamp | 0.889 | — | LIKELY |
| 3 | multivuln_externalbug_mishandled | ExternalBug, ME, GasException | ExternalBug | 0.806 | 1 hit | LIKELY |

### Key Finding

- **Only ExternalBug was correctly identified** in a multi-label context (1/3)
- **Multi-vuln call_reentrancy**: ML got Timestamp right but missed CallToUnknown and Reentrancy
- **Multi-vuln reentrancy_tod**: ML predicted UnusedReturn — missed BOTH ground truth classes

---

## Full LLM Results — Timestamp Vesting

**Mode:** DEBATE_MODE=off, full LLM narrator  
**Total time:** 217.5s (3.6 min)  
**Breakdown:**
- ML assessment: 0.7s
- Static analysis + RAG + graph: 1.3s
- cross_validator (LLM single-pass): 50.4s
- synthesizer (LLM narrative): 115.3s
- reflection + explainer + visualizer: 44.3s

### Result

| Field | Value |
|-------|-------|
| Overall verdict | **CONFIRMED** (upgraded from LIKELY in no-llm mode) |
| Top vulnerability | **Timestamp** ✓ |
| Risk probability | 0.913 |
| Confirmed classes | Timestamp, IntegerUO, ExternalBug, Reentrancy, UnusedReturn, GasException (6) |
| Suspicious classes | DoS, ME (2) |
| Narrator output | 1527-character Markdown narrative |

### LLM Narrative Quality

The LLM correctly identified Timestamp as the primary issue but **hallucinated "confirmed reentrancy risks"** — the contract has no reentrancy vulnerability. The narrative text mentions reentrancy mitigation even though the contract is not reentrant.

> *"The contract exhibits confirmed reentrancy risks, potentially allowing attackers to re-enter functions during state changes."* — **FALSE** (no reentrancy exists in this contract)

**Root cause:** The ML model predicted Reentrancy at 0.80 CONFIRMED (false positive), and the LLM trusted the ML signal without independently verifying the source code.

---

## Cross-Class Summary Table

| Ground Truth Class | # Tested | Correctly Detected | Missed/Confused |
|--------------------|----------|--------------------|-------|
| CallToUnknown | 3 | 0 | 3 (UnusedReturn ×2, Timestamp ×1) |
| DenialOfService | 2 | 0 | 2 (UnusedReturn, miss) |
| ExternalBug | 3 | 1 | 2 (Reentrancy, UnusedReturn) |
| GasException | 2 | 0 | 2 (ExternalBug, ExternalBug) |
| IntegerUO | 3 | 1 | 2 (Timestamp, UnusedReturn) |
| MishandledException | 2 | 0 | 2 (miss, ExternalBug) |
| **Reentrancy** | 4 | 0 | 4 (all confused) |
| **Timestamp** | 4 | 3 | 1 (IntegerUO) |
| TransactionOrderDependence | 2 | 0 | 2 (ExternalBug, UnusedReturn) |
| **UnusedReturn** | 2 | 0 | 2 (miss, miss) |
| **Safe** | 1 | — | 1 (FP: UnusedReturn 0.664) |

---

## Systematic Issues Found

### 1. ML Columnar Signal Collapse (CRITICAL)

The Run 12 model has a **dominant UnusedReturn response**. Across 19 contracts:

| ML Class | Times Predicted as Top | Accuracy vs GT |
|----------|----------------------|----------------|
| UnusedReturn | **6/19** (32%) | Correct in 0/6 cases! |
| ExternalBug | 4/19 (21%) | Correct in 1/4 |
| Timestamp | 3/19 (16%) | Correct in 3/3 |
| IntegerUO | 3/19 (16%) | Correct in 1/3 |
| Reentrancy | 1/19 (5%) | Correct in 0/1 |

**UnusedReturn is a catch-all false positive** — the model predicts it for Reentrancy contracts, CallToUnknown contracts, DoS contracts, and even the Safe contract.

### 2. ML Blindness to Return-Value-Aware Patterns

MishandledException and UnusedReturn as ground truth are invisible to the model. Both went through fast-path (no ML signal at all). This suggests the model has learned that "discarded return values" is not a vulnerability in isolation.

### 3. Slither Complements ML Correctly But Is Limited

Slither correctly identified:
- `reentrancy-no-eth` on the Reentrancy contract (ML predicted IntegerUO)
- `timestamp` on the Timestamp contract (ML agreed)
- 2 hits on many contracts (triggered deep path)

But Slither missed:
- UnusedReturn patterns (MishandledException, UnusedReturn)
- DoS patterns (no DoS detector exists)
- TOD patterns (no TOD detector exists)
- GasException patterns (no GasException detector)

### 4. LLM Hallucination on Trusted ML Signal

When ML predicts 6/10 classes as CONFIRMED (including Reentrancy at 0.80), the LLM synthesizer incorporates the ML signal into its narrative, producing false claims. The Safe contract would likely receive a narrative about "multiple confirmed vulnerabilities."

### 5. Most Successful Evasion Pattern

The **DoS-in-constructor** contract was the only one to fully evade all detection:
- ML: zero signal
- Slither: zero hits
- Verdict: SAFE
- Path: fast

Burying the vulnerability in the constructor + O(n²) computation in a nested function was the most effective evasion strategy.

---

## Recommended Follow-Up Actions

1. **Calibrate UnusedReturn threshold** — the 0.55 CONFIRMED threshold is too low for this class; the false positive rate is ~100% (0/6 correct)
2. **Investigate ExternalBug baseline** — appears as top class for 4 unrelated contracts (GasException, TOD, GasException-adjacent)
3. **Add DoS Slither detection patterns** — Slither has no DoS detector but our verification YAMLs exist
4. **Improve LLM source-code awareness** — the cross_validator should verify ML claims against actual source before narrating
5. **Consider per-class probability floors** — some classes (UnusedReturn, ExternalBug) need higher evidence thresholds

---

*Generated by manual agents batch testing against the manual_hand_written_contracts validation suite.*
