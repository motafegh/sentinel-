# Task 21: Feature Correlation & Redundancy Audit

**Sample size:** 1000  
**Total nodes:** 132,471  
**Skipped files:** 0

## 1. Pearson Correlation Matrix

```
            type_id  visibili  uses_blo      view   payable  complexi       loc  return_i  call_tar  in_unche  has_loop  external
 type_id      1.000    -0.412    -0.078    -0.336    -0.100    -0.427     0.094    -0.110     0.051       nan    -0.108    -0.269
visibili     -0.412     1.000     0.081     0.080    -0.016     0.275    -0.001     0.045    -0.008       nan     0.084     0.144
uses_blo     -0.078     0.081     1.000     0.102     0.071     0.184     0.003     0.040     0.002       nan     0.104     0.096
    view     -0.336     0.080     0.102     1.000    -0.016     0.156    -0.031    -0.010     0.007       nan     0.102     0.085
 payable     -0.100    -0.016     0.071    -0.016     1.000     0.199    -0.008     0.066    -0.037       nan     0.012     0.173
complexi     -0.427     0.275     0.184     0.156     0.199     1.000     0.006     0.211    -0.083       nan     0.413     0.540
     loc      0.094    -0.001     0.003    -0.031    -0.008     0.006     1.000     0.000     0.006       nan     0.005    -0.005
return_i     -0.110     0.045     0.040    -0.010     0.066     0.211     0.000     1.000    -0.015       nan     0.076     0.389
call_tar      0.051    -0.008     0.002     0.007    -0.037    -0.083     0.006    -0.015     1.000       nan    -0.005    -0.142
in_unche        nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan
has_loop     -0.108     0.084     0.104     0.102     0.012     0.413     0.005     0.076    -0.005       nan     1.000     0.116
external     -0.269     0.144     0.096     0.085     0.173     0.540    -0.005     0.389    -0.142       nan     0.116     1.000
```

## 2. Spearman Rank Correlation Matrix

```
            type_id  visibili  uses_blo      view   payable  complexi       loc  return_i  call_tar  in_unche  has_loop  external
 type_id      1.000    -0.353    -0.063    -0.265    -0.084    -0.494     0.741    -0.087     0.040       N/A    -0.086    -0.225
visibili     -0.353     1.000     0.085     0.079    -0.017     0.363    -0.288     0.044    -0.003       N/A     0.083     0.124
uses_blo     -0.063     0.085     1.000     0.102     0.071     0.130    -0.016     0.040     0.002       N/A     0.104     0.083
    view     -0.265     0.079     0.102     1.000    -0.016     0.355    -0.241    -0.010     0.007       N/A     0.102     0.088
 payable     -0.084    -0.017     0.071    -0.016     1.000     0.186    -0.080     0.066    -0.037       N/A     0.012     0.156
complexi     -0.494     0.363     0.130     0.355     0.186     1.000    -0.405     0.181    -0.085       N/A     0.189     0.467
     loc      0.741    -0.288    -0.016    -0.241    -0.080    -0.405     1.000    -0.057     0.034       N/A    -0.048    -0.152
return_i     -0.087     0.044     0.040    -0.010     0.066     0.181    -0.057     1.000    -0.015       N/A     0.076     0.387
call_tar      0.040    -0.003     0.002     0.007    -0.037    -0.085     0.034    -0.015     1.000       N/A    -0.005    -0.177
in_unche        N/A       N/A       N/A       N/A       N/A       N/A       N/A       N/A       N/A       N/A       N/A       N/A
has_loop     -0.086     0.083     0.104     0.102     0.012     0.189    -0.048     0.076    -0.005       N/A     1.000     0.099
external     -0.225     0.124     0.083     0.088     0.156     0.467    -0.152     0.387    -0.177       N/A     0.099     1.000
```

## 3. Highly Correlated Feature Pairs (|r| > 0.5)

| Feature A | Feature B | Pearson | Spearman |
|-----------|-----------|---------|----------|
| type_id | loc | 0.094 | 0.741 |
| complexity | external_call_count | 0.540 | 0.467 |

## 4. Mutual Information: Feature ↔ Label

| Feature | CallTo | Denial | Extern | GasExc | Intege | Mishan | Reentr | Timest | Transa | Unused | Total |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| type_id | 0.0037 | 0.0000 | 0.0000 | 0.0164 | 0.0395 | 0.0035 | 0.0015 | 0.0058 | 0.0033 | 0.0039 | 0.0775 |
| visibility | 0.0096 | 0.0000 | 0.0116 | 0.0205 | 0.0567 | 0.0119 | 0.0064 | 0.0093 | 0.0121 | 0.0022 | 0.1404 |
| uses_block_globals | 0.0120 | 0.0000 | 0.0048 | 0.0000 | 0.0153 | 0.0028 | 0.0000 | 0.0000 | 0.0056 | 0.0098 | 0.0502 |
| view | 0.0000 | 0.0718 | 0.0017 | 0.0000 | 0.0000 | 0.0066 | 0.0027 | 0.0000 | 0.0054 | 0.0000 | 0.0882 |
| payable | 0.0029 | 0.0580 | 0.0000 | 0.0066 | 0.0105 | 0.0066 | 0.0000 | 0.0020 | 0.0000 | 0.0012 | 0.0879 |
| complexity | 0.0127 | 0.0000 | 0.0000 | 0.0117 | 0.0513 | 0.0163 | 0.0017 | 0.0000 | 0.0010 | 0.0025 | 0.0970 |
| loc | 0.0088 | 0.0000 | 0.0133 | 0.0180 | 0.0602 | 0.0249 | 0.0069 | 0.0019 | 0.0090 | 0.0057 | 0.1485 |
| return_ignored | 0.0093 | 0.0000 | 0.0078 | 0.0001 | 0.0000 | 0.0072 | 0.0092 | 0.0000 | 0.0000 | 0.0013 | 0.0350 |
| call_target_typed | 0.0079 | 0.0744 | 0.0035 | 0.0000 | 0.0108 | 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0062 | 0.1037 |
| in_unchecked | 0.0038 | 0.0000 | 0.0044 | 0.0000 | 0.0205 | 0.0117 | 0.0000 | 0.0050 | 0.0054 | 0.0000 | 0.0508 |
| has_loop | 0.0055 | 0.0006 | 0.0000 | 0.0001 | 0.0000 | 0.0037 | 0.0000 | 0.0000 | 0.0128 | 0.0165 | 0.0393 |
| external_call_count | 0.0022 | 0.0000 | 0.0016 | 0.0138 | 0.0743 | 0.0141 | 0.0183 | 0.0000 | 0.0000 | 0.0101 | 0.1343 |

## 5. Features with Zero MI (all labels)

No features with zero total MI.

## 6. Unique Information per Feature (1 - R²)

| Feature | Unique Info | Interpretation |
|---------|-------------|----------------|
| type_id | 0.7133 | High unique info |
| visibility | 0.7953 | High unique info |
| uses_block_globals | 0.8396 | High unique info |
| view | 0.8548 | High unique info |
| payable | 0.8296 | High unique info |
| complexity | 0.5074 | High unique info |
| loc | 0.8476 | High unique info |
| return_ignored | 0.8741 | High unique info |
| call_target_typed | 0.9456 | High unique info |
| in_unchecked | 0.0000 | ⚠️ Nearly fully redundant |
| has_loop | 0.7009 | High unique info |
| external_call_count | 0.6515 | High unique info |

## 7. PCA Variance Explained

| Variance Threshold | Components Needed |
|--------------------|--------------------|
| 90% | 9 |
| 95% | 10 |
| 99% | 11 |

### Individual Component Variance

| Component | Variance Explained | Cumulative |
|-----------|--------------------|-----------|
| PC1 | 0.2258 | 0.2258 |
| PC2 | 0.1170 | 0.3428 |
| PC3 | 0.0982 | 0.4410 |
| PC4 | 0.0936 | 0.5346 |
| PC5 | 0.0891 | 0.6237 |
| PC6 | 0.0877 | 0.7114 |
| PC7 | 0.0819 | 0.7933 |
| PC8 | 0.0781 | 0.8714 |
| PC9 | 0.0550 | 0.9264 |
| PC10 | 0.0438 | 0.9702 |
| PC11 | 0.0298 | 1.0000 |
| PC12 | 0.0000 | 1.0000 |

## 8. Feature Drop Recommendation

Top 3 candidates for removal (highest drop score):

| Rank | Feature | Drop Score | Unique Info | Max |Corr| | Total MI |
|------|---------|------------|-------------|--------------|----------|
| 1 | complexity | 0.2426 | 0.5074 | 0.5403 | 0.0970 |
| 2 | type_id | 0.1135 | 0.7133 | 0.4265 | 0.0775 |
| 3 | visibility | 0.0740 | 0.7953 | 0.4123 | 0.1404 |

**Recommendation:** Consider dropping the above features as they contribute the least unique information and have the highest redundancy with other features.

## Summary

- **Highly correlated pairs (|r|>0.5):** 2
- **Features with zero MI:** 0
- **PCA components for 95% variance:** 10
- **Recommended to drop:** complexity, type_id, visibility
