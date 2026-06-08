# Expanded Review: 200 Contracts

**Date:** 2026-06-08

## Summary

| Decision | Count | % |
|----------|-------|---|
| KEEP | 17 | 9% |
| DROP | 99 | 50% |
| UNCERTAIN | 83 | 42% |

## Noise Estimates (DROP rate)

| Class | FN Noise | TP Noise | Interpretation |
|-------|----------|----------|----------------|
| Class11:Reentrancy | 80% (20/25) | 94% (17/18) | HIGH noise |
| Class10:IntegerUO | 0% (0/23) | 0% (0/16) | LOW noise |
| Class06:UnusedReturn | 0% (0/7) | 0% (0/3) | LOW noise |
| Class01:ExternalBug | 100% (1/1) | 94% (15/16) | HIGH noise |
| Class08:CallToUnknown | 91% (10/11) | 85% (17/20) | HIGH noise |
| Class03:MishandledException | 0% (0/3) | 0% (0/17) | LOW noise |
| Class02:GasException | 67% (6/9) | 0% (0/3) | HIGH noise |
| Class09:DenialOfService | 56% (10/18) | 0% (0/1) | HIGH noise |
| Class04:Timestamp | 50% (1/2) | 33% (2/6) | MODERATE noise |