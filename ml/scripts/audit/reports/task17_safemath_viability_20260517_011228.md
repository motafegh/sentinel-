# Task 17: SafeMath Viability Audit
**IntegerUO=1 sampled:** 200  **NonVulnerable sampled:** 200  **Paths resolved:** 400  **Graph subsample for verification:** 59

## SafeMath Category Distribution
| Category | IntegerUO | NonVulnerable |
|----------|-----------|---------------|
| using_for_uint256 | 98 | 83 |
| direct_calls | 5 | 4 |
| inheritance | 7 | 3 |
| none | 90 | 110 |

## Confusion Matrix: SafeMath Presence vs IntegerUO Label
| | IntegerUO=1 | NonVulnerable |
|---|------------|---------------|
| SafeMath present | 110 | 90 |
| SafeMath absent | 90 | 110 |

**IntegerUO without SafeMath:** 45.0% (90/200)
**NonVulnerable with SafeMath:** 45.0% (90/200)

## Graph-Based SafeMath Verification (Subsample)
| | Graph has SafeMath | Graph no SafeMath |
|---|--------------------|-------------------|
| Source has SafeMath | 17 | 22 |
| Source no SafeMath | 3 | 17 |

**Graph SafeMath recall** (source→graph): 43.6% (17/39)

## Pragma Version Analysis (Feature Viability)
| Version | IntegerUO=1 | Not IntegerUO | Iuo Rate |
|---------|-------------|---------------|----------|
| <0.8.0 | 337 | 663 | 33.7% |
| >=0.8.0 | 0 | 0 | N/A |

## Pragma Version Distribution by Label Group
| Version | IntegerUO | NonVulnerable |
|---------|-----------|---------------|
| 0.4 | 179 | 200 |
| 0.5 | 21 | 0 |

## Recommendation
SafeMath absence is a **weak signal** for IntegerUO (only 45.0% of IntegerUO contracts lack SafeMath). 

**Consider adding `pragma_version` as a graph-level feature** (currently not in the 12-dim node feature vector). SafeMath presence could also be encoded as a contract-level feature, but the graph already captures it via CALLS edges to SafeMath functions (graph recall: 17/39).