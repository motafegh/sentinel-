# Stage 1 Agreement Report: BCCC vs Slither

**Date:** 2026-06-07

**Sample size:** 67311 contracts (Stage 1 sample)

**Slither OK results:** 7186

## Per-Class Results

| Class | TP | FP | FN | TN | Precision | Recall | F1 | Support |
|-------|----|----|----|----|-----------|--------|----|---------|
| Class11:Reentrancy | 434 | 2332 | 596 | 3824 | 0.157 | 0.421 | 0.229 | 1030 |
| Class10:IntegerUO | 415 | 493 | 2196 | 4082 | 0.457 | 0.159 | 0.236 | 2611 |
| Class06:UnusedReturn | 78 | 442 | 722 | 5944 | 0.150 | 0.098 | 0.118 | 800 |
| Class01:ExternalBug | 105 | 1475 | 403 | 5203 | 0.066 | 0.207 | 0.101 | 508 |
| Class08:CallToUnknown | 200 | 837 | 395 | 5754 | 0.193 | 0.336 | 0.245 | 595 |
| Class03:MishandledException | 305 | 2844 | 407 | 3630 | 0.097 | 0.428 | 0.158 | 712 |
| Class02:GasException | 83 | 338 | 768 | 5997 | 0.197 | 0.098 | 0.131 | 851 |
| Class09:DenialOfService | 11 | 1678 | 49 | 5448 | 0.007 | 0.183 | 0.013 | 60 |
| Class04:Timestamp | 150 | 1825 | 202 | 5009 | 0.076 | 0.426 | 0.129 | 352 |
| Class12:NonVulnerable | 0 | 0 | 3686 | 3500 | 0.000 | 0.000 | 0.000 | 3686 |

**Median F1 (vuln classes):** 0.131

**Gate:** FAIL

**Reason:** Median F1 < 0.5: Need escalation to 30%/50% sampling

## Escalation Decision

Escalate to Stage 2: 30% sampling of disagreeing classes.
If median F1 still < 0.5 after Stage 2 → Stage 3: 50% sampling.
