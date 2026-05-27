# Task 22: Graph Size Confound Audit

**Sample size:** 5000  
**Skipped:** 0

## 1. Per-Class Graph Size Statistics

### num_nodes

| Class | Pos Count | Neg Count | Pos Mean | Pos Median | Pos P25 | Pos P75 | Pos P95 | Neg Mean | Neg Median | Neg P25 | Neg P75 | Neg P95 |
|-------|-----------|-----------|----------|------------|---------|---------|---------|----------|------------|---------|---------|----------|
| CallToUnknown | 382 | 4618 | 145.9 | 96 | 70 | 184 | 345 | 131.8 | 95 | 63 | 163 | 342 |
| DenialOfService | 55 | 4945 | 137.2 | 95 | 66 | 177 | 379 | 132.8 | 96 | 63 | 166 | 342 |
| ExternalBug | 386 | 4614 | 152.5 | 109 | 72 | 201 | 404 | 131.2 | 95 | 63 | 161 | 336 |
| GasException | 612 | 4388 | 163.3 | 122 | 76 | 209 | 444 | 128.6 | 92 | 63 | 158 | 326 |
| IntegerUO | 1727 | 3273 | 153.2 | 117 | 74 | 194 | 373 | 122.1 | 83 | 59 | 144 | 313 |
| MishandledException | 553 | 4447 | 156.2 | 124 | 74 | 193 | 409 | 129.9 | 93 | 63 | 159 | 333 |
| Reentrancy | 574 | 4426 | 151.3 | 102 | 70 | 195 | 391 | 130.5 | 94 | 63 | 160 | 336 |
| Timestamp | 251 | 4749 | 181.1 | 139 | 81 | 218 | 541 | 130.3 | 94 | 63 | 160 | 333 |
| TransactionOrderDependence | 371 | 4629 | 163.0 | 121 | 78 | 209 | 418 | 130.4 | 94 | 63 | 160 | 336 |
| UnusedReturn | 343 | 4657 | 149.6 | 117 | 73 | 200 | 370 | 131.6 | 94 | 63 | 162 | 336 |

### num_edges

| Class | Pos Count | Neg Count | Pos Mean | Pos Median | Pos P25 | Pos P75 | Pos P95 | Neg Mean | Neg Median | Neg P25 | Neg P75 | Neg P95 |
|-------|-----------|-----------|----------|------------|---------|---------|---------|----------|------------|---------|---------|----------|
| CallToUnknown | 382 | 4618 | 248.5 | 155 | 91 | 312 | 636 | 224.2 | 137 | 91 | 273 | 616 |
| DenialOfService | 55 | 4945 | 224.8 | 138 | 100 | 268 | 758 | 226.1 | 137 | 91 | 279 | 619 |
| ExternalBug | 386 | 4614 | 262.3 | 166 | 106 | 348 | 774 | 223.0 | 136 | 91 | 270 | 608 |
| GasException | 612 | 4388 | 281.4 | 192 | 112 | 364 | 839 | 218.3 | 134 | 91 | 264 | 597 |
| IntegerUO | 1727 | 3273 | 262.2 | 182 | 110 | 326 | 713 | 206.9 | 128 | 87 | 236 | 557 |
| MishandledException | 553 | 4447 | 269.0 | 190 | 110 | 325 | 760 | 220.7 | 134 | 91 | 266 | 608 |
| Reentrancy | 574 | 4426 | 259.1 | 164 | 91 | 334 | 762 | 221.7 | 136 | 91 | 267 | 605 |
| Timestamp | 251 | 4749 | 318.8 | 227 | 115 | 382 | 1140 | 221.1 | 134 | 91 | 269 | 604 |
| TransactionOrderDependence | 371 | 4629 | 281.1 | 191 | 112 | 362 | 798 | 221.6 | 134 | 91 | 268 | 609 |
| UnusedReturn | 343 | 4657 | 256.8 | 186 | 110 | 344 | 676 | 223.8 | 135 | 91 | 271 | 612 |

### num_functions

| Class | Pos Count | Neg Count | Pos Mean | Pos Median | Pos P25 | Pos P75 | Pos P95 | Neg Mean | Neg Median | Neg P25 | Neg P75 | Neg P95 |
|-------|-----------|-----------|----------|------------|---------|---------|---------|----------|------------|---------|---------|----------|
| CallToUnknown | 382 | 4618 | 19.8 | 16 | 13 | 26 | 44 | 18.1 | 15 | 8 | 23 | 42 |
| DenialOfService | 55 | 4945 | 21.4 | 16 | 10 | 28 | 47 | 18.2 | 15 | 9 | 23 | 42 |
| ExternalBug | 386 | 4614 | 21.0 | 19 | 11 | 26 | 51 | 18.0 | 15 | 8 | 23 | 42 |
| GasException | 612 | 4388 | 22.3 | 20 | 13 | 28 | 48 | 17.7 | 14 | 8 | 22 | 42 |
| IntegerUO | 1727 | 3273 | 21.4 | 20 | 12 | 27 | 46 | 16.6 | 13 | 8 | 21 | 40 |
| MishandledException | 553 | 4447 | 21.4 | 19 | 12 | 27 | 45 | 17.9 | 15 | 8 | 23 | 42 |
| Reentrancy | 574 | 4426 | 20.8 | 17 | 12 | 26 | 48 | 17.9 | 15 | 8 | 23 | 42 |
| Timestamp | 251 | 4749 | 23.9 | 21 | 14 | 32 | 48 | 18.0 | 15 | 8 | 23 | 42 |
| TransactionOrderDependence | 371 | 4629 | 22.5 | 20 | 12 | 28 | 50 | 17.9 | 15 | 8 | 23 | 42 |
| UnusedReturn | 343 | 4657 | 20.8 | 20 | 11 | 27 | 46 | 18.1 | 15 | 8 | 23 | 42 |

## 2. Logistic Regression AUC-ROC (Size Features Only)

Using [num_nodes, num_edges, num_functions] → predict each class.

| Class | AUC-ROC | Confounded? |
|-------|---------|-------------|
| CallToUnknown | 0.5673 | Borderline |
| DenialOfService | 0.5759 | Borderline |
| ExternalBug | 0.5646 | Borderline |
| GasException | 0.6086 | Borderline |
| IntegerUO | 0.6180 | Borderline |
| MishandledException | 0.5862 | Borderline |
| Reentrancy | 0.5551 | Borderline |
| Timestamp | 0.6370 | Borderline |
| TransactionOrderDependence | 0.5998 | Borderline |
| UnusedReturn | 0.5706 | Borderline |

## 3. Mann-Whitney U Test: Size vs Label (Positive vs Negative)

| Class | Metric | Pos Median | Neg Median | U-stat | p-value | Significant? |
|-------|--------|------------|------------|--------|---------|---------------|
| CallToUnknown | num_nodes | 96 | 95 | 956938 | 0.0057 | ⚠️ YES |
| CallToUnknown | num_edges | 155 | 137 | 933806 | 0.0562 | No |
| CallToUnknown | num_functions | 16 | 15 | 966456 | 0.0018 | ⚠️ YES |
| DenialOfService | num_nodes | 95 | 96 | 138354 | 0.8241 | No |
| DenialOfService | num_edges | 138 | 137 | 135995 | 0.9995 | No |
| DenialOfService | num_functions | 16 | 15 | 144156 | 0.4424 | No |
| ExternalBug | num_nodes | 109 | 95 | 1010217 | 0.0000 | ⚠️ YES |
| ExternalBug | num_edges | 166 | 136 | 1005731 | 0.0000 | ⚠️ YES |
| ExternalBug | num_functions | 19 | 15 | 1012405 | 0.0000 | ⚠️ YES |
| GasException | num_nodes | 122 | 92 | 1612889 | 0.0000 | ⚠️ YES |
| GasException | num_edges | 192 | 134 | 1594458 | 0.0000 | ⚠️ YES |
| GasException | num_functions | 20 | 14 | 1635440 | 0.0000 | ⚠️ YES |
| IntegerUO | num_nodes | 117 | 83 | 3453014 | 0.0000 | ⚠️ YES |
| IntegerUO | num_edges | 182 | 128 | 3417992 | 0.0000 | ⚠️ YES |
| IntegerUO | num_functions | 20 | 13 | 3521724 | 0.0000 | ⚠️ YES |
| MishandledException | num_nodes | 124 | 93 | 1435640 | 0.0000 | ⚠️ YES |
| MishandledException | num_edges | 190 | 134 | 1425582 | 0.0000 | ⚠️ YES |
| MishandledException | num_functions | 19 | 15 | 1446778 | 0.0000 | ⚠️ YES |
| Reentrancy | num_nodes | 102 | 94 | 1409426 | 0.0000 | ⚠️ YES |
| Reentrancy | num_edges | 164 | 136 | 1387921 | 0.0003 | ⚠️ YES |
| Reentrancy | num_functions | 17 | 15 | 1414783 | 0.0000 | ⚠️ YES |
| Timestamp | num_nodes | 139 | 94 | 751576 | 0.0000 | ⚠️ YES |
| Timestamp | num_edges | 227 | 134 | 745490 | 0.0000 | ⚠️ YES |
| Timestamp | num_functions | 21 | 15 | 762168 | 0.0000 | ⚠️ YES |
| TransactionOrderDependence | num_nodes | 121 | 94 | 1025744 | 0.0000 | ⚠️ YES |
| TransactionOrderDependence | num_edges | 191 | 134 | 1019682 | 0.0000 | ⚠️ YES |
| TransactionOrderDependence | num_functions | 20 | 15 | 1034221 | 0.0000 | ⚠️ YES |
| UnusedReturn | num_nodes | 117 | 94 | 908790 | 0.0000 | ⚠️ YES |
| UnusedReturn | num_edges | 186 | 135 | 907337 | 0.0000 | ⚠️ YES |
| UnusedReturn | num_functions | 20 | 15 | 915605 | 0.0000 | ⚠️ YES |

## 4. Label Distribution: Top-25% vs Bottom-25% by Size

Size metric: num_nodes + num_edges.  
Bottom-25% threshold: ≤161 (n=1437)  
Top-25% threshold: ≥445 (n=1255)

| Class | Bottom-25% Pos Rate | Top-25% Pos Rate | Diff | Concern? |
|-------|---------------------|-------------------|------|----------|
| CallToUnknown | 0.0765 | 0.0956 | +0.0191 | No |
| DenialOfService | 0.0084 | 0.0127 | +0.0044 | No |
| ExternalBug | 0.0508 | 0.1044 | +0.0536 | ⚠️ YES |
| GasException | 0.0717 | 0.1745 | +0.1028 | ⚠️ YES |
| IntegerUO | 0.2109 | 0.4637 | +0.2529 | ⚠️ YES |
| MishandledException | 0.0731 | 0.1522 | +0.0791 | ⚠️ YES |
| Reentrancy | 0.1009 | 0.1538 | +0.0529 | ⚠️ YES |
| Timestamp | 0.0251 | 0.0869 | +0.0618 | ⚠️ YES |
| TransactionOrderDependence | 0.0383 | 0.1131 | +0.0749 | ⚠️ YES |
| UnusedReturn | 0.0480 | 0.0932 | +0.0452 | No |

## 5. Summary & Recommendations

### Confounded Classes (AUC > 0.65 from size alone)

No classes strongly confounded by size.

### Significant Size Differences (Mann-Whitney p < 0.01)

- **CallToUnknown** — num_nodes (p=0.0057)
- **CallToUnknown** — num_functions (p=0.0018)
- **ExternalBug** — num_nodes (p=0.0000)
- **ExternalBug** — num_edges (p=0.0000)
- **ExternalBug** — num_functions (p=0.0000)
- **GasException** — num_nodes (p=0.0000)
- **GasException** — num_edges (p=0.0000)
- **GasException** — num_functions (p=0.0000)
- **IntegerUO** — num_nodes (p=0.0000)
- **IntegerUO** — num_edges (p=0.0000)
- **IntegerUO** — num_functions (p=0.0000)
- **MishandledException** — num_nodes (p=0.0000)
- **MishandledException** — num_edges (p=0.0000)
- **MishandledException** — num_functions (p=0.0000)
- **Reentrancy** — num_nodes (p=0.0000)
- **Reentrancy** — num_edges (p=0.0003)
- **Reentrancy** — num_functions (p=0.0000)
- **Timestamp** — num_nodes (p=0.0000)
- **Timestamp** — num_edges (p=0.0000)
- **Timestamp** — num_functions (p=0.0000)
- **TransactionOrderDependence** — num_nodes (p=0.0000)
- **TransactionOrderDependence** — num_edges (p=0.0000)
- **TransactionOrderDependence** — num_functions (p=0.0000)
- **UnusedReturn** — num_nodes (p=0.0000)
- **UnusedReturn** — num_edges (p=0.0000)
- **UnusedReturn** — num_functions (p=0.0000)

### Recommendations

1. Graph size does not appear to be a major confound for most classes.
2. Continue monitoring as new data is added.
4. **For classes with significant Mann-Whitney results**, verify that model predictions are not proxies for graph size.
