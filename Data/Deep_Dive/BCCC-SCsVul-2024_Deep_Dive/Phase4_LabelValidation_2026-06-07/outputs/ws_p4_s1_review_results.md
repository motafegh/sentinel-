# Manual Review Results: 43 High-Uncertainty Contracts

**Date:** 2026-06-08

## Summary

| Decision | Count | Percentage |
|----------|-------|------------|
| KEEP (label correct) | 5 | 12% |
| DROP (label incorrect) | 22 | 51% |
| UNCERTAIN | 16 | 37% |
| **Total** | **43** | |

## Per-Class Breakdown

| Class | KEEP | DROP | UNCERTAIN | Total |
|-------|------|------|-----------|-------|
| Class01:ExternalBug | 0 | 1 | 0 | 1 |
| Class02:GasException | 0 | 1 | 3 | 4 |
| Class03:MishandledException | 0 | 0 | 2 | 2 |
| Class04:Timestamp | 0 | 1 | 0 | 1 |
| Class06:UnusedReturn | 0 | 1 | 3 | 4 |
| Class08:CallToUnknown | 0 | 6 | 0 | 6 |
| Class09:DenialOfService | 0 | 1 | 0 | 1 |
| Class10:IntegerUO | 3 | 0 | 8 | 11 |
| Class11:Reentrancy | 2 | 11 | 0 | 13 |

## Detailed Results

### fae5624b6a384099 (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 201
- **Findings:** transfer call(1)

### 2c181f48e7141661 (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 109
- **Findings:** transfer call(12)

### 0a7a47e9290ed384 (Class11:Reentrancy)
- **Decision:** KEEP
- **Reason:** State change before external call (reentrancy pattern)
- **LOC:** 148
- **Findings:** low-level call(1)

### 6da017c2465a3757 (Class11:Reentrancy)
- **Decision:** KEEP
- **Reason:** State change after external call (reentrancy pattern)
- **LOC:** 11
- **Findings:** external call with value transfer (old syntax)(1)

### 60c3189a471809ec (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 380
- **Findings:** transfer call(2)

### 57904673f62ab2d8 (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 57

### 3645b3a49eb67358 (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 123
- **Findings:** transfer call(1)

### 28d28a7245ccea11 (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 579
- **Findings:** transfer call(1)

### d415ba0a60ff3632 (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 227
- **Findings:** transfer call(2)

### 221ee40fe539b90c (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 814

### e5df9eb444fb470e (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 248

### a5987a43180e0433 (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 262

### 0cc01185e3eb6a7e (Class11:Reentrancy)
- **Decision:** DROP
- **Reason:** No external call found
- **LOC:** 62
- **Findings:** send call(1)

### 3211900d19fd22f0 (Class10:IntegerUO)
- **Decision:** UNCERTAIN
- **Reason:** No clear arithmetic pattern found
- **LOC:** 1097
- **Findings:** addition without SafeMath(1), subtraction without SafeMath(1), multiplication without SafeMath(1)

### 0a04b7966715ae71 (Class10:IntegerUO)
- **Decision:** UNCERTAIN
- **Reason:** No clear arithmetic pattern found
- **LOC:** 497
- **Findings:** addition without SafeMath(1), subtraction without SafeMath(1), multiplication without SafeMath(1), division without SafeMath(1)

### 7a144ba43261c763 (Class10:IntegerUO)
- **Decision:** KEEP
- **Reason:** Arithmetic without SafeMath
- **LOC:** 1123
- **Findings:** value subtraction(1)

### 6e1c529e8b915147 (Class10:IntegerUO)
- **Decision:** UNCERTAIN
- **Reason:** No clear arithmetic pattern found
- **LOC:** 918

### 6233829661cf360e (Class10:IntegerUO)
- **Decision:** KEEP
- **Reason:** Arithmetic without SafeMath
- **LOC:** 1717
- **Findings:** addition without SafeMath(2), subtraction without SafeMath(1)

### 3dfb34b0c2904921 (Class10:IntegerUO)
- **Decision:** UNCERTAIN
- **Reason:** No clear arithmetic pattern found
- **LOC:** 124

### 2e04a1243fcbabf9 (Class10:IntegerUO)
- **Decision:** UNCERTAIN
- **Reason:** Arithmetic with SafeMath (may be safe)
- **LOC:** 224
- **Findings:** addition without SafeMath(1), subtraction without SafeMath(1), multiplication without SafeMath(1), division without SafeMath(1)

### f31393f9cff440df (Class10:IntegerUO)
- **Decision:** UNCERTAIN
- **Reason:** No clear arithmetic pattern found
- **LOC:** 638
- **Findings:** addition without SafeMath(1), subtraction without SafeMath(1), multiplication without SafeMath(1), division without SafeMath(1)

### 28363a3df6294062 (Class10:IntegerUO)
- **Decision:** KEEP
- **Reason:** Arithmetic without SafeMath
- **LOC:** 574
- **Findings:** value addition(1)

### be006469bb2e7ae7 (Class10:IntegerUO)
- **Decision:** UNCERTAIN
- **Reason:** No clear arithmetic pattern found
- **LOC:** 474
- **Findings:** addition without SafeMath(1), subtraction without SafeMath(1), multiplication without SafeMath(1), division without SafeMath(1)

### 0e363473de1be614 (Class10:IntegerUO)
- **Decision:** UNCERTAIN
- **Reason:** Arithmetic with SafeMath (may be safe)
- **LOC:** 311
- **Findings:** addition without SafeMath(1), subtraction without SafeMath(1), multiplication without SafeMath(1), division without SafeMath(1), totalSupply addition(1), amount multiplication(1)

### b5f07eac33c5f726 (Class09:DenialOfService)
- **Decision:** DROP
- **Reason:** No loop found
- **LOC:** 52

### 2be5c39959fc3e9e (Class08:CallToUnknown)
- **Decision:** DROP
- **Reason:** No call to unknown address
- **LOC:** 182

### 0c4b244a4391b1b4 (Class08:CallToUnknown)
- **Decision:** DROP
- **Reason:** No call to unknown address
- **LOC:** 270
- **Findings:** address cast(2)

### 7300cf995dd3b67d (Class08:CallToUnknown)
- **Decision:** DROP
- **Reason:** No call to unknown address
- **LOC:** 913
- **Findings:** address cast(7)

### 6654483dbdfe6fac (Class08:CallToUnknown)
- **Decision:** DROP
- **Reason:** No call to unknown address
- **LOC:** 100
- **Findings:** address cast(4)

### 5e0096dbcc1d4f8d (Class08:CallToUnknown)
- **Decision:** DROP
- **Reason:** No call to unknown address
- **LOC:** 300
- **Findings:** address cast(6)

### 36427ce69f44a76d (Class08:CallToUnknown)
- **Decision:** DROP
- **Reason:** No call to unknown address
- **LOC:** 577

### afc20d2fcb52f714 (Class02:GasException)
- **Decision:** UNCERTAIN
- **Reason:** Loop found (may have gas issues)
- **LOC:** 1141
- **Findings:** call in potential loop(1)

### 217dfa4632f1c8d8 (Class02:GasException)
- **Decision:** UNCERTAIN
- **Reason:** Loop found (may have gas issues)
- **LOC:** 540

### 07aa3c486b42f189 (Class02:GasException)
- **Decision:** UNCERTAIN
- **Reason:** Loop found (may have gas issues)
- **LOC:** 902
- **Findings:** while loop(1)

### c6bd49b4aec50e0e (Class02:GasException)
- **Decision:** DROP
- **Reason:** No loop found
- **LOC:** 330

### da081796d0c57c5f (Class06:UnusedReturn)
- **Decision:** UNCERTAIN
- **Reason:** Possible unchecked return
- **LOC:** 463
- **Findings:** transfer (always reverts on failure)(3)

### 230347c96df3515f (Class06:UnusedReturn)
- **Decision:** UNCERTAIN
- **Reason:** Possible unchecked return
- **LOC:** 287
- **Findings:** transfer (always reverts on failure)(3)

### 0829214881ec3423 (Class06:UnusedReturn)
- **Decision:** UNCERTAIN
- **Reason:** Possible unchecked return
- **LOC:** 524
- **Findings:** transfer (always reverts on failure)(5)

### fd1df8d6413c2f20 (Class06:UnusedReturn)
- **Decision:** DROP
- **Reason:** No unchecked return pattern
- **LOC:** 636

### e97c9438d1928e75 (Class03:MishandledException)
- **Decision:** UNCERTAIN
- **Reason:** No clear mishandled exception pattern
- **LOC:** 117

### 2ad7bbb4d3e569a4 (Class03:MishandledException)
- **Decision:** UNCERTAIN
- **Reason:** No clear mishandled exception pattern
- **LOC:** 245
- **Findings:** transfer (reverts on failure)(1)

### b1982e1e1108812d (Class04:Timestamp)
- **Decision:** DROP
- **Reason:** No timestamp usage
- **LOC:** 359

### b64aaff28af406fa (Class01:ExternalBug)
- **Decision:** DROP
- **Reason:** No external bug pattern
- **LOC:** 146
