# Task 23: .send() Unchecked Prevalence Audit

**MishandledException sample:** 500  
**DenialOfService sample:** 200  
**.sol files resolved:** 700/700

## 1. .send() Prevalence in MishandledException Contracts

| Metric | Count | Rate |
|--------|-------|------|
| Contracts with .send() | 32 | 6.4% |
| Contracts with unchecked .send() | 4 | 12.5% |
| Contracts with no .sol file | 0 | — |
| Contracts with no graph file | 0 | — |

### Aggregate .send() Statistics

| Metric | Count |
|--------|-------|
| Total .send() calls found | 67 |
| Checked .send() calls | 59 |
| Unchecked .send() calls | 8 |
| Unchecked rate | 11.9% |

## 2. Cross-Tabulation: .send() Unchecked vs return_ignored

| Source \ Graph | return_ignored=1 | return_ignored=0 | Total |
|----------------|-------------------|-------------------|-------|
| .send() unchecked | 1 | 3 | 4 |
| .send() checked | 10 | 18 | 28 |
| No .send() | 171 | 297 | 468 |
| Incomplete data | — | — | 0 |

**Agreement rate** (source and graph agree): 63.2%

**Missed by graph:** 3 contracts with unchecked .send() in source but no return_ignored in graph

## 3. .send() in Loops (DenialOfService Contracts)

| Metric | Count | Rate |
|--------|-------|------|
| DoS contracts with .send() | 10 | 5.0% |
| .send() in loops | 3 | 30.0% |

## 4. .call() vs .send() Detection Comparison

| Metric | Count | Rate |
|--------|-------|------|
| ME contracts with .call() | 34 | 6.8% |
| ME contracts with both .call() and .send() | 1 | 2.9% of .call() contracts |

## 5. Conclusions & Recommendations

- Most .send() calls in ME contracts are checked, suggesting the vulnerability comes from other patterns (e.g., .call(), .transfer()).
- **3 contracts** have unchecked .send() in source but no return_ignored in the graph. Consider improving the AST extractor to catch these.
- **3 DoS contracts** have .send() inside loops — a classic DoS pattern. Verify the graph captures loop + external_call correctly.
