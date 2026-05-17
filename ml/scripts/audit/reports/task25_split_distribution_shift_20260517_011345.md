# Task 25: Split Distribution Shift Audit

**CSV rows:** 44470  
**train split:** 31142 samples  
**val split:** 6661 samples  
**test split:** 6667 samples  

## 1. Solidity Version Distribution per Split

| Version | train | val | test |
|---------|-------|-------|-------|
| 0.4 | 475 (95.0%) | 484 (96.8%) | 473 (94.6%) |
| 0.5 | 25 (5.0%) | 14 (2.8%) | 26 (5.2%) |
| unknown | 0 (0.0%) | 2 (0.4%) | 1 (0.2%) |

## 2. Mean Graph Size per Split

| Split | Loaded | Mean Nodes | Median Nodes | Mean Edges | Median Edges |
|-------|--------|------------|--------------|------------|---------------|
| train | 200 | 126.4 | 87 | 215.0 | 134 |
| val | 200 | 134.4 | 84 | 229.7 | 128 |
| test | 200 | 142.1 | 94 | 249.2 | 134 |

## 3. Per-Class Positive Rate per Split

| Class | train Rate | val Rate | test Rate | train Count | val Count | test Count |
|-------|-------|-------|-------|-------|-------|-------|
| CallToUnknown | 0.0803 | 0.0808 | 0.0855 | 2502/31142 | 538/6661 | 570/6667 |
| DenialOfService | 0.0083 | 0.0078 | 0.0102 | 257/31142 | 52/6661 | 68/6667 |
| ExternalBug | 0.0778 | 0.0757 | 0.0717 | 2422/31142 | 504/6661 | 478/6667 |
| GasException | 0.1259 | 0.1231 | 0.1284 | 3921/31142 | 820/6661 | 856/6667 |
| IntegerUO | 0.3496 | 0.3463 | 0.3504 | 10886/31142 | 2307/6661 | 2336/6667 |
| MishandledException | 0.1072 | 0.1046 | 0.1012 | 3337/31142 | 697/6661 | 675/6667 |
| Reentrancy | 0.1118 | 0.1179 | 0.1135 | 3483/31142 | 785/6661 | 757/6667 |
| Timestamp | 0.0479 | 0.0522 | 0.0525 | 1493/31142 | 348/6661 | 350/6667 |
| TransactionOrderDependence | 0.0760 | 0.0758 | 0.0780 | 2366/31142 | 505/6661 | 520/6667 |
| UnusedReturn | 0.0680 | 0.0727 | 0.0652 | 2118/31142 | 484/6661 | 435/6667 |

## 4. Feature Distributions & KS Test

### loc

| Split | Mean | Std | Median | Min | Max |
|-------|------|-----|--------|-----|-----|
| train | 1.8035 | 0.9794 | 1.4406 | 0.1445 | 8.4615 |
| val | 1.7800 | 1.5184 | 1.4693 | 0.1205 | 20.5730 |
| test | 1.7535 | 0.7735 | 1.5173 | 0.1294 | 6.4615 |

**KS Test Results:**

| Comparison | Statistic | p-value | Significant? |
|------------|-----------|---------|--------------|
| train_vs_val | 0.0800 | 0.5453 | No |
| train_vs_test | 0.1200 | 0.1123 | No |
| val_vs_test | 0.0950 | 0.3281 | No |

### complexity

| Split | Mean | Std | Median | Min | Max |
|-------|------|-----|--------|-----|-----|
| train | 0.7234 | 0.0949 | 0.7226 | 0.0294 | 1.0046 |
| val | 0.7052 | 0.1391 | 0.7099 | 0.0000 | 1.4035 |
| test | 0.7064 | 0.1472 | 0.7143 | 0.0000 | 0.9787 |

**KS Test Results:**

| Comparison | Statistic | p-value | Significant? |
|------------|-----------|---------|--------------|
| train_vs_val | 0.0800 | 0.5453 | No |
| train_vs_test | 0.0600 | 0.8655 | No |
| val_vs_test | 0.0800 | 0.5453 | No |

### external_call_count

| Split | Mean | Std | Median | Min | Max |
|-------|------|-----|--------|-----|-----|
| train | 0.0095 | 0.0072 | 0.0096 | 0.0000 | 0.0444 |
| val | 0.0082 | 0.0071 | 0.0062 | 0.0000 | 0.0391 |
| test | 0.0087 | 0.0074 | 0.0080 | 0.0000 | 0.0568 |

**KS Test Results:**

| Comparison | Statistic | p-value | Significant? |
|------------|-----------|---------|--------------|
| train_vs_val | 0.1000 | 0.2705 | No |
| train_vs_test | 0.0800 | 0.5453 | No |
| val_vs_test | 0.0800 | 0.5453 | No |

## 5. Token Window Distribution per Split

| Windows | train | val | test |
|---------|-------|-------|-------|
| 1 | 6 (3.0%) | 6 (3.0%) | 6 (3.0%) |
| 2 | 3 (1.5%) | 6 (3.0%) | 8 (4.0%) |
| 3 | 11 (5.5%) | 14 (7.0%) | 16 (8.0%) |
| 4 | 180 (90.0%) | 174 (87.0%) | 170 (85.0%) |

## 6. Special: Pure DoS Contracts & 0.8.x Distribution

### Pure DenialOfService Contracts (n=7)

| Split | Count |
|-------|-------|
| train | 3 |
| val | 1 |
| test | 3 |
| not_in_split | 0 |

**Individual pure-DoS contracts:**

- `1f39978ecb7edec3189c6819e0bd4a9f` → test
- `59a86f6d07e149e022ce2c6a9919bdc3` → train
- `94513bf8fb7bac45dcb4e904ebc05511` → test
- `95fc8e424291e1bc6bea4edc8a0a6ca3` → train
- `ae8313104cef07a8782330fb682cd39e` → train
- `d7c71805cb6c20c4fe276367bbddb2b4` → test
- `dd3f8807cfccdc8a784c147b4d34856e` → val

### Solidity 0.8.x Distribution

| Split | 0.8.x Count | Resolved | 0.8.x Rate | Sampled |
|-------|-------------|----------|------------|--------|
| train | 0 | 300 | 0.0% | 300 |
| val | 0 | 300 | 0.0% | 300 |
| test | 0 | 300 | 0.0% | 300 |

## 7. Summary & Distribution Shift Concerns

### No Significant Distribution Shifts

Feature distributions appear consistent across splits.

