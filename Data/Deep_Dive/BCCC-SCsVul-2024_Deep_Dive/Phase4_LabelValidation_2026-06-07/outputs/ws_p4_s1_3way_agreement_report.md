# 3-Way Agreement: BCCC vs Slither vs Aderyn

**Date:** 2026-06-08

**Sample:** 67311 contracts
**Slither OK:** 7186
**Aderyn OK:** 6192

## Per-Class F1

| Class | Slither F1 | Aderyn F1 | Majority F1 | Support |
|-------|-----------|----------|------------|---------|
| Class11:Reentrancy | 0.229 | 0.169 | 0.165 | 1030 |
| Class10:IntegerUO | 0.236 | 0.000 | 0.000 | 2611 |
| Class06:UnusedReturn | 0.118 | 0.000 | 0.000 | 800 |
| Class01:ExternalBug | 0.101 | 0.145 | 0.117 | 508 |
| Class08:CallToUnknown | 0.245 | 0.140 | 0.168 | 595 |
| Class03:MishandledException | 0.158 | 0.173 | 0.133 | 712 |
| Class02:GasException | 0.131 | 0.000 | 0.000 | 851 |
| Class09:DenialOfService | 0.013 | 0.000 | 0.000 | 60 |
| Class04:Timestamp | 0.129 | 0.000 | 0.000 | 352 |
| Class12:NonVulnerable | 0.000 | 0.000 | 0.000 | 3686 |

**Median F1:** Slither=0.131  Aderyn=0.000  Majority=0.000

**Slither gate:** FAIL

**Aderyn gate:** FAIL

**Majority gate:** FAIL

