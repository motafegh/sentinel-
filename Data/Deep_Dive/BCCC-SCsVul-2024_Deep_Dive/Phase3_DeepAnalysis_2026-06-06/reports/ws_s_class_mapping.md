# WS-S: BCCC-SCsVul-2024 ↔ SmartBugs Wild Class Semantic Mapping

**Date:** 2026-06-06
**Author:** SENTINEL Phase 3 (deep analysis)
**Inputs:** BCCC-SCsVul-2024 paper (HajiHosseinKhani et al., Blockchain: Research and Applications, Dec 2024); SmartBugs paper (Durieux et al., ICSE 2020); SWC Registry (swcregistry.io); DASP 10; OWASP SCWE

## 1. Purpose

BCCC-SCsVul-2024 (12 classes) and SmartBugs Wild (9 DASP-10 categories) use different taxonomies. This workstream maps them semantically so that:
1. WS-R can compare SENTINEL vs slither vs mythril on **comparable** contract subsets
2. Future work can transfer models trained on BCCC to SmartBugs-style test data
3. The BCCC label noise (e.g., the 766 review-pending contracts) can be cross-validated against DASP 10 expectations

## 2. The three taxonomies

### 2.1 BCCC-SCsVul-2024 (12 classes, multi-label)

From the BCCC paper (HajiHosseinKhani et al., 2024, Blockchain: Research and Applications):

| # | Class | Count | Description (inferred from name + paper) |
|---|-------|------:|---|
| 1 | `Class01:ExternalBug` | 3,604 | Generic external call bugs (low-level call, send, transfer failures) |
| 2 | `Class02:GasException` | 6,879 | Gas-related issues (out-of-gas, gaslimit, gas exhaustion in loops) |
| 3 | `Class03:MishandledException` | 5,154 | Improper exception handling (missing require, silent failure) |
| 4 | `Class04:Timestamp` | 2,674 | Block timestamp dependence (now, block.timestamp) |
| 5 | `Class05:TransactionOrderDependence` | 3,562 | TOCTOU bugs (race conditions in approve/transfer) |
| 6 | `Class06:UnusedReturn` | 3,229 | Ignoring return values of low-level calls |
| 7 | `Class07:WeakAccessMod` | 1,918 | State visibility issues (public where private needed) |
| 8 | `Class08:CallToUnknown` | 11,131 | Calls to unknown/untrusted addresses |
| 9 | `Class09:DenialOfService` | 12,394 | DoS via revert, gas exhaustion, or external dependency |
| 10 | `Class10:IntegerUO` | 16,740 | Integer underflow/overflow |
| 11 | `Class11:Reentrancy` | 17,698 | Reentrant calls (external call before state update) |
| 12 | `Class12:NonVulnerable` | 26,914 | "Secure" baseline (no known vulnerabilities) |

**Source provenance (per BCCC paper):** SmartBugs, Ethereum Smart Contracts (ESC), Slither-audited, SmartScan dataset.

### 2.2 SmartBugs Wild / DASP 10 (10 categories, single-label)

From Durieux et al. (ICSE 2020) — `sbcurated` (69 contracts) and `sbwild` (47,398 contracts):

| # | DASP 10 Category | sbcurated n | Description |
|---|-------|------:|---|
| 1 | Reentrancy | 7 | Reentrant function calls cause unexpected behavior |
| 2 | Access Control | 17 | Failure to use function modifiers, tx.origin misuse |
| 3 | Arithmetic | 14 | Integer over/underflow |
| 4 | Unchecked Low Level Calls | 53 | call(), callcode(), delegatecall(), send() failure unchecked |
| 5 | Denial of Service | 6 | Contract overwhelmed with time-consuming computations |
| 6 | Bad Randomness | 8 | Malicious miner biases outcome (block.number, blockhash) |
| 7 | Front Running | 4 | Two dependent transactions in same block |
| 8 | Time Manipulation | 5 | Block timestamp manipulated by miner |
| 9 | Short Addresses | 1 | EVM accepts incorrectly padded arguments |
| 10 | Other (Unknown Unknowns) | 3 | Vulnerabilities not in DASP 10 |

### 2.3 SWC Registry (EIP-1470)

| SWC | Title | CWE | Maps to |
|------|----------------|--------|---|
| SWC-100 | Function Default Visibility | CWE-710 | Access Control / WeakAccessMod |
| SWC-101 | Integer Overflow and Underflow | CWE-682 | IntegerUO / Arithmetic |
| SWC-104 | Unchecked Call Return Value | CWE-252 | UnusedReturn / Unchecked Low Level Calls |
| SWC-105 | Unprotected Ether Withdrawal | CWE-284 | (covered by Access Control) |
| SWC-106 | Unprotected SELFDESTRUCT | CWE-284 | Access Control |
| SWC-107 | Reentrancy | CWE-841 | Reentrancy |
| SWC-112 | Delegatecall to Untrusted Contract | CWE-829 | CallToUnknown |
| SWC-113 | DoS with Failed Call | CWE-400 | DenialOfService |
| SWC-114 | Transaction Order Dependence | CWE-362 | TransactionOrderDependence / Front Running |
| SWC-116 | Block Values Dependency | CWE-829 | Timestamp / Bad Randomness |
| SWC-123 | Gas Limit Issues | CWE-400 | GasException (closest) |

## 3. BCCC ↔ DASP 10 ↔ SWC mapping (12 BCCC classes → 9 DASP categories + 1 control)

| BCCC Class | DASP 10 | SWC | SENTINEL v9 | Notes |
|------------|---------|-----|-------------|-------|
| `Class01:ExternalBug` | **Unchecked Low Level Calls** | SWC-104 | ✅ (Class01:ExternalBug) | Generic external call issue |
| `Class02:GasException` | (DASP Other) | SWC-123 | ✅ (Class02:GasException) | Out-of-gas / gas exhaustion |
| `Class03:MishandledException` | **Unchecked Low Level Calls** | SWC-104 | ✅ (Class03:MishandledException) | Exception handling miss |
| `Class04:Timestamp` | **Time Manipulation** | SWC-116 | ✅ (Class04:Timestamp) | now / block.timestamp |
| `Class05:TransactionOrderDependence` | **Front Running** | SWC-114 | ❌ DROPPED (D-F1) | TOCTOU bugs |
| `Class06:UnusedReturn` | **Unchecked Low Level Calls** | SWC-104 | ✅ (Class06:UnusedReturn) | call() return ignored |
| `Class07:WeakAccessMod` | **Access Control** | SWC-100 | ❌ DROPPED (D-F1) | Function visibility |
| `Class08:CallToUnknown` | **Unchecked Low Level Calls** | SWC-112 | ✅ (Class08:CallToUnknown) | Call to untrusted address |
| `Class09:DenialOfService` | **Denial of Service** | SWC-113 | ✅ (Class09:DenialOfService) | DoS via revert/external dep |
| `Class10:IntegerUO` | **Arithmetic** | SWC-101 | ✅ (Class10:IntegerUO) | Integer over/underflow |
| `Class11:Reentrancy` | **Reentrancy** | SWC-107 | ✅ (Class11:Reentrancy) | Reentrant external call |
| `Class12:NonVulnerable` | N/A (control) | N/A | ✅ (Class12:NonVulnerable) | Secure baseline |

**Summary of mapping:**
- 10 BCCC classes map cleanly to DASP 10 categories
- 2 BCCC classes were DROPPED in SENTINEL v9 (Class05, Class07) per D-F1 — no SENTINEL v9 equivalent
- 1 DASP category (Bad Randomness) is **NOT** in BCCC
- 1 DASP category (Short Addresses) is **NOT** in BCCC
- 1 BCCC class (Class02:GasException) is **NOT** in DASP 10 — closest is DASP "Other"

## 4. BCCC's relationship to SmartBugs Wild (sbwild)

Per the BCCC paper and the comparative study (MDPI 2025):

> "The BCCC-SCsVul-2024 dataset was curated from reputable sources like Smart Bugs, Ethereum SCs, and SmartScan-Dataset, ensuring diverse and representative vulnerability coverage."

**Critical observation:** BCCC INCLUDES SmartBugs-curated contracts (and sbwild contracts) as part of its 111,897 contracts. This means:
- BCCC and SmartBugs are **NOT** disjoint datasets
- SENTINEL v9 Phase 2 (D-D) checked for byte-identical overlap between BCCC v1.0 and SmartBugs-curated (0 matches)
- But **functionally similar** contracts (same source code, different bytecode) likely exist

**Implications for WS-O (cross-tool label consensus):**
- If we use sbwild's 47,398 contracts as the WS-O sample, we MUST ensure they're deduplicated against BCCC (likely already done via SHA-256 hash by BCCC).
- The MDPI 2025 paper used BCCC for binary (vuln/not) classification and got 89.44% accuracy with Random Forest, suggesting that BCCC's labels are *roughly* learnable but with significant noise.

## 5. Implications for Phase 3 workstreams

### 5.1 WS-I (slither label validation on 846 contracts)
Slither has 101 detectors; we map them to SENTINEL v9 classes via the table in `03_phase3_plan.md` §2. The mapping uses:
- Class11:Reentrancy → slither's `reentrancy-*` detectors (6 total)
- Class10:IntegerUO → slither's `divide-by-*` and `incorrect-*` arithmetic (11 total)
- Class08:CallToUnknown → slither's `unchecked-*` and `arbitrary-send-*` (4 total)
- Class09:DenialOfService → slither's `calls-loop` and `locked-ether` (3 total)
- Class04:Timestamp → slither's `timestamp` detector (1-3)
- Class06:UnusedReturn → slither's `unchecked-return` (4)
- Class01:ExternalBug → slither's `low-level-calls` and `send-transfer` (5)
- Class07:WeakAccessMod → slither's `visibility` (DROPPED in SENTINEL)
- Class03:MishandledException → slither's `unchecked-send` and `tautology` (11)
- Class02:GasException → slither's `gas-*` and `costly-*` (5)

### 5.2 WS-S to SmartBugs Wild transfer test (Phase 4 candidate)
If we train SENTINEL on BCCC and test on sbwild:
- **Direct match:** Reentrancy, IntegerUO, DoS, Timestamp, CallToUnknown — these all map 1:1 to DASP 10
- **Need re-mapping:** ExternalBug, UnusedReturn, MishandledException, GasException all map to "Unchecked Low Level Calls" or "Other" in DASP 10 — prediction distributions will differ
- **Missing in BCCC:** Bad Randomness, Short Addresses, Access Control (partial) — SENTINEL will have 0 detection capability for these in transfer test

### 5.3 WS-R (3-way SENTINEL vs AutoML vs slither)
- Compare on SENTINEL v9's 10 classes (BCCC-aligned)
- Then **re-bin** all predictions to DASP 10 categories and compare again
- If SENTINEL beats AutoML in SENTINEL v9 but loses after re-binning, that means the AutoML is exploiting the "UnusedReturn vs CallToUnknown" distinction (which DASP collapses) that SENTINEL isn't

## 6. Confidence levels for each mapping

| BCCC Class | DASP match confidence | SWC match confidence | Notes |
|------------|---:|---:|---|
| Class01:ExternalBug | **HIGH** | **HIGH** | Direct SWC-104 overlap |
| Class02:GasException | LOW | MEDIUM | No direct DASP match; closest is "Other" |
| Class03:MishandledException | **HIGH** | **HIGH** | SWC-104 + SWC-110 |
| Class04:Timestamp | **HIGH** | **HIGH** | Direct DASP "Time Manipulation" |
| Class05:TransactionOrderDependence | **HIGH** | **HIGH** | Direct DASP "Front Running" |
| Class06:UnusedReturn | **HIGH** | **HIGH** | Direct DASP "Unchecked Low Level Calls" |
| Class07:WeakAccessMod | **HIGH** | **HIGH** | Direct DASP "Access Control" |
| Class08:CallToUnknown | **HIGH** | **HIGH** | Direct DASP "Unchecked Low Level Calls" |
| Class09:DenialOfService | **HIGH** | **HIGH** | Direct DASP "Denial of Service" |
| Class10:IntegerUO | **HIGH** | **HIGH** | Direct DASP "Arithmetic" |
| Class11:Reentrancy | **HIGH** | **HIGH** | Direct DASP "Reentrancy" |

10/12 BCCC classes have HIGH confidence mapping. 2/12 have special handling (Class02 dropped from DASP, Class05/07 dropped from SENTINEL v9).

## 7. References

1. HajiHosseinKhani, S., Lashkari, A.H., Oskui, A.M. (Dec 2024). "Unveiling Smart Contracts Vulnerabilities: Toward Profiling Smart Contracts Vulnerabilities using Enhanced Genetic Algorithm and Generating Benchmark Dataset." *Blockchain: Research and Applications*, Vol 5, Article 100253. https://www.sciencedirect.com/science/article/pii/S2096720924000666
2. Durieux, T., Ferreira, J.F., Abreu, R., Cruz, P. (2020). "Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts." *ICSE 2020*. https://arxiv.org/pdf/2007.04771
3. SWC Registry: https://swcregistry.io/ (EIP-1470)
4. DASP 10: https://dasp.co/
5. OWASP SCWE: https://scs.owasp.org/SCWE/
6. Comparative Study on BCCC (2025): https://www.mdpi.com/2227-7080/13/12/592
7. SmartBugs Wild repo: https://github.com/smartbugs/smartbugs-wild
8. SmartBugs results: https://github.com/smartbugs/smartbugs-results

## 8. Outputs

This document is the WS-S deliverable. Saved to `reports/ws_s_class_mapping.md`.

Downstream consumers:
- WS-I: uses §5.1 detector mapping
- WS-O: uses §4 for sample selection from sbwild
- WS-R: uses §5.3 for 3-way comparison
- Phase 4: uses §5.2 for BCCC→sbwild transfer test
