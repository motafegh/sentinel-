# Task 1: Re-check Activation Split (Declaration vs CFG)

**Sample per class:** 20 pure-label contracts  
**Declaration threshold:** type_id < 0.6667 (8/12)  
**Focus features:** uses_block_globals, return_ignored, in_unchecked, has_loop, external_call_count

## Per-Class Feature Activation Rates

### CallToUnknown

**Contracts loaded:** 12  
**Nodes:** 838 total, 310 declaration, 528 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.8711 | 0.6516 | 1.0000 | 0.3484 |
| visibility | 0.0752 | 0.2032 | 0.0000 | 0.2032 |
| uses_block_globals | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| view | 0.0406 | 0.1097 | 0.0000 | 0.1097 |
| payable | 0.0084 | 0.0226 | 0.0000 | 0.0226 |
| complexity | 0.1527 | 0.4129 | 0.0000 | 0.4129 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0024 | 0.0065 | 0.0000 | 0.0065 |
| call_target_typed | 0.9988 | 0.9968 | 1.0000 | 0.0032 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0024 | 0.0065 | 0.0000 | 0.0065 |
| external_call_count | 0.0418 | 0.1129 | 0.0000 | 0.1129 ** |

### DenialOfService

**Contracts loaded:** 7  
**Nodes:** 143 total, 43 declaration, 100 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9091 | 0.6977 | 1.0000 | 0.3023 |
| visibility | 0.0979 | 0.3256 | 0.0000 | 0.3256 |
| uses_block_globals | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| view | 0.0420 | 0.1395 | 0.0000 | 0.1395 |
| payable | 0.0070 | 0.0233 | 0.0000 | 0.0233 |
| complexity | 0.1608 | 0.5349 | 0.0000 | 0.5349 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| call_target_typed | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0420 | 0.1395 | 0.0000 | 0.1395 ** |
| external_call_count | 0.0070 | 0.0233 | 0.0000 | 0.0233 |

### ExternalBug

**Contracts loaded:** 20  
**Nodes:** 3054 total, 859 declaration, 2195 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9181 | 0.7090 | 1.0000 | 0.2910 |
| visibility | 0.0557 | 0.1979 | 0.0000 | 0.1979 |
| uses_block_globals | 0.0010 | 0.0035 | 0.0000 | 0.0035 |
| view | 0.0439 | 0.1560 | 0.0000 | 0.1560 |
| payable | 0.0098 | 0.0349 | 0.0000 | 0.0349 |
| complexity | 0.1467 | 0.5215 | 0.0000 | 0.5215 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0052 | 0.0186 | 0.0000 | 0.0186 |
| call_target_typed | 0.9987 | 0.9953 | 1.0000 | 0.0047 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0043 | 0.0151 | 0.0000 | 0.0151 |
| external_call_count | 0.0390 | 0.1385 | 0.0000 | 0.1385 ** |

### GasException

**Contracts loaded:** 20  
**Nodes:** 2951 total, 835 declaration, 2116 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9221 | 0.7246 | 1.0000 | 0.2754 |
| visibility | 0.0695 | 0.2455 | 0.0000 | 0.2455 |
| uses_block_globals | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| view | 0.0441 | 0.1557 | 0.0000 | 0.1557 |
| payable | 0.0064 | 0.0228 | 0.0000 | 0.0228 |
| complexity | 0.1494 | 0.5281 | 0.0000 | 0.5281 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0024 | 0.0084 | 0.0000 | 0.0084 |
| call_target_typed | 0.9993 | 0.9976 | 1.0000 | 0.0024 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0075 | 0.0263 | 0.0000 | 0.0263 |
| external_call_count | 0.0302 | 0.1066 | 0.0000 | 0.1066 ** |

### IntegerUO

**Contracts loaded:** 20  
**Nodes:** 3123 total, 802 declaration, 2321 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9334 | 0.7406 | 1.0000 | 0.2594 |
| visibility | 0.0637 | 0.2481 | 0.0000 | 0.2481 |
| uses_block_globals | 0.0042 | 0.0162 | 0.0000 | 0.0162 |
| view | 0.0458 | 0.1783 | 0.0000 | 0.1783 |
| payable | 0.0064 | 0.0249 | 0.0000 | 0.0249 |
| complexity | 0.1377 | 0.5362 | 0.0000 | 0.5362 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0035 | 0.0137 | 0.0000 | 0.0137 |
| call_target_typed | 0.9994 | 0.9975 | 1.0000 | 0.0025 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0061 | 0.0237 | 0.0000 | 0.0237 |
| external_call_count | 0.0243 | 0.0948 | 0.0000 | 0.0948 |

### MishandledException

**Contracts loaded:** 20  
**Nodes:** 1836 total, 521 declaration, 1315 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9227 | 0.7274 | 1.0000 | 0.2726 |
| visibility | 0.0615 | 0.2169 | 0.0000 | 0.2169 |
| uses_block_globals | 0.0011 | 0.0038 | 0.0000 | 0.0038 |
| view | 0.0349 | 0.1228 | 0.0000 | 0.1228 |
| payable | 0.0136 | 0.0480 | 0.0000 | 0.0480 |
| complexity | 0.1498 | 0.5278 | 0.0000 | 0.5278 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0065 | 0.0230 | 0.0000 | 0.0230 |
| call_target_typed | 0.9951 | 0.9827 | 1.0000 | 0.0173 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0038 | 0.0134 | 0.0000 | 0.0134 |
| external_call_count | 0.0349 | 0.1228 | 0.0000 | 0.1228 ** |

### Reentrancy

**Contracts loaded:** 20  
**Nodes:** 1599 total, 496 declaration, 1103 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9156 | 0.7278 | 1.0000 | 0.2722 |
| visibility | 0.0575 | 0.1855 | 0.0000 | 0.1855 |
| uses_block_globals | 0.0075 | 0.0242 | 0.0000 | 0.0242 |
| view | 0.0400 | 0.1290 | 0.0000 | 0.1290 |
| payable | 0.0225 | 0.0726 | 0.0000 | 0.0726 |
| complexity | 0.1770 | 0.5706 | 0.0000 | 0.5706 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0106 | 0.0343 | 0.0000 | 0.0343 |
| call_target_typed | 0.9881 | 0.9617 | 1.0000 | 0.0383 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0050 | 0.0161 | 0.0000 | 0.0161 |
| external_call_count | 0.0350 | 0.1129 | 0.0000 | 0.1129 ** |

### Timestamp

**Contracts loaded:** 20  
**Nodes:** 5443 total, 1194 declaration, 4249 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9421 | 0.7362 | 1.0000 | 0.2638 |
| visibility | 0.0524 | 0.2387 | 0.0000 | 0.2387 |
| uses_block_globals | 0.0072 | 0.0327 | 0.0000 | 0.0327 |
| view | 0.0347 | 0.1583 | 0.0000 | 0.1583 |
| payable | 0.0059 | 0.0268 | 0.0000 | 0.0268 |
| complexity | 0.1168 | 0.5327 | 0.0000 | 0.5327 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0022 | 0.0101 | 0.0000 | 0.0101 |
| call_target_typed | 0.9987 | 0.9941 | 1.0000 | 0.0059 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0088 | 0.0402 | 0.0000 | 0.0402 |
| external_call_count | 0.0314 | 0.1432 | 0.0000 | 0.1432 ** |

### TransactionOrderDependence

**Contracts loaded:** 20  
**Nodes:** 2914 total, 810 declaration, 2104 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9317 | 0.7543 | 1.0000 | 0.2457 |
| visibility | 0.0796 | 0.2864 | 0.0000 | 0.2864 |
| uses_block_globals | 0.0017 | 0.0062 | 0.0000 | 0.0062 |
| view | 0.0484 | 0.1741 | 0.0000 | 0.1741 |
| payable | 0.0106 | 0.0383 | 0.0000 | 0.0383 |
| complexity | 0.1520 | 0.5469 | 0.0000 | 0.5469 |
| loc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| return_ignored | 0.0038 | 0.0136 | 0.0000 | 0.0136 |
| call_target_typed | 0.9986 | 0.9951 | 1.0000 | 0.0049 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0045 | 0.0160 | 0.0000 | 0.0160 |
| external_call_count | 0.0292 | 0.1049 | 0.0000 | 0.1049 ** |

### UnusedReturn

**Contracts loaded:** 20  
**Nodes:** 2445 total, 697 declaration, 1748 CFG

| Feature | Overall | Declaration | CFG | Decl-CFG Diff |
|---------|---------|-------------|-----|---------------|
| type_id | 0.9243 | 0.7346 | 1.0000 | 0.2654 |
| visibility | 0.0703 | 0.2468 | 0.0000 | 0.2468 |
| uses_block_globals | 0.0033 | 0.0115 | 0.0000 | 0.0115 |
| view | 0.0413 | 0.1449 | 0.0000 | 0.1449 |
| payable | 0.0070 | 0.0244 | 0.0000 | 0.0244 |
| complexity | 0.1456 | 0.5108 | 0.0000 | 0.5108 |
| loc | 0.9787 | 0.9555 | 0.9880 | 0.0325 |
| return_ignored | 0.0061 | 0.0215 | 0.0000 | 0.0215 |
| call_target_typed | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_loop | 0.0070 | 0.0244 | 0.0000 | 0.0244 |
| external_call_count | 0.0368 | 0.1291 | 0.0000 | 0.1291 ** |

## Focus Feature Summary Across All Classes

| Class | Feature | Overall | Declaration | CFG | Diff | Note |
|-------|---------|---------|-------------|-----|------|------|
| CallToUnknown | uses_block_globals | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| CallToUnknown | return_ignored | 0.0024 | 0.0065 | 0.0000 | 0.0065 | DECL-only |
| CallToUnknown | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| CallToUnknown | has_loop | 0.0024 | 0.0065 | 0.0000 | 0.0065 | DECL-only |
| CallToUnknown | external_call_count | 0.0418 | 0.1129 | 0.0000 | 0.1129 | DECL-only |
| DenialOfService | uses_block_globals | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| DenialOfService | return_ignored | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| DenialOfService | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| DenialOfService | has_loop | 0.0420 | 0.1395 | 0.0000 | 0.1395 | DECL-only |
| DenialOfService | external_call_count | 0.0070 | 0.0233 | 0.0000 | 0.0233 | DECL-only |
| ExternalBug | uses_block_globals | 0.0010 | 0.0035 | 0.0000 | 0.0035 | DECL-only |
| ExternalBug | return_ignored | 0.0052 | 0.0186 | 0.0000 | 0.0186 | DECL-only |
| ExternalBug | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| ExternalBug | has_loop | 0.0043 | 0.0151 | 0.0000 | 0.0151 | DECL-only |
| ExternalBug | external_call_count | 0.0390 | 0.1385 | 0.0000 | 0.1385 | DECL-only |
| GasException | uses_block_globals | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| GasException | return_ignored | 0.0024 | 0.0084 | 0.0000 | 0.0084 | DECL-only |
| GasException | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| GasException | has_loop | 0.0075 | 0.0263 | 0.0000 | 0.0263 | DECL-only |
| GasException | external_call_count | 0.0302 | 0.1066 | 0.0000 | 0.1066 | DECL-only |
| IntegerUO | uses_block_globals | 0.0042 | 0.0162 | 0.0000 | 0.0162 | DECL-only |
| IntegerUO | return_ignored | 0.0035 | 0.0137 | 0.0000 | 0.0137 | DECL-only |
| IntegerUO | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| IntegerUO | has_loop | 0.0061 | 0.0237 | 0.0000 | 0.0237 | DECL-only |
| IntegerUO | external_call_count | 0.0243 | 0.0948 | 0.0000 | 0.0948 | DECL-only |
| MishandledException | uses_block_globals | 0.0011 | 0.0038 | 0.0000 | 0.0038 | DECL-only |
| MishandledException | return_ignored | 0.0065 | 0.0230 | 0.0000 | 0.0230 | DECL-only |
| MishandledException | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| MishandledException | has_loop | 0.0038 | 0.0134 | 0.0000 | 0.0134 | DECL-only |
| MishandledException | external_call_count | 0.0349 | 0.1228 | 0.0000 | 0.1228 | DECL-only |
| Reentrancy | uses_block_globals | 0.0075 | 0.0242 | 0.0000 | 0.0242 | DECL-only |
| Reentrancy | return_ignored | 0.0106 | 0.0343 | 0.0000 | 0.0343 | DECL-only |
| Reentrancy | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| Reentrancy | has_loop | 0.0050 | 0.0161 | 0.0000 | 0.0161 | DECL-only |
| Reentrancy | external_call_count | 0.0350 | 0.1129 | 0.0000 | 0.1129 | DECL-only |
| Timestamp | uses_block_globals | 0.0072 | 0.0327 | 0.0000 | 0.0327 | DECL-only |
| Timestamp | return_ignored | 0.0022 | 0.0101 | 0.0000 | 0.0101 | DECL-only |
| Timestamp | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| Timestamp | has_loop | 0.0088 | 0.0402 | 0.0000 | 0.0402 | DECL-only |
| Timestamp | external_call_count | 0.0314 | 0.1432 | 0.0000 | 0.1432 | DECL-only |
| TransactionOrderDependence | uses_block_globals | 0.0017 | 0.0062 | 0.0000 | 0.0062 | DECL-only |
| TransactionOrderDependence | return_ignored | 0.0038 | 0.0136 | 0.0000 | 0.0136 | DECL-only |
| TransactionOrderDependence | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| TransactionOrderDependence | has_loop | 0.0045 | 0.0160 | 0.0000 | 0.0160 | DECL-only |
| TransactionOrderDependence | external_call_count | 0.0292 | 0.1049 | 0.0000 | 0.1049 | DECL-only |
| UnusedReturn | uses_block_globals | 0.0033 | 0.0115 | 0.0000 | 0.0115 | DECL-only |
| UnusedReturn | return_ignored | 0.0061 | 0.0215 | 0.0000 | 0.0215 | DECL-only |
| UnusedReturn | in_unchecked | 0.0000 | 0.0000 | 0.0000 | 0.0000 | DEAD feature |
| UnusedReturn | has_loop | 0.0070 | 0.0244 | 0.0000 | 0.0244 | DECL-only |
| UnusedReturn | external_call_count | 0.0368 | 0.1291 | 0.0000 | 0.1291 | DECL-only |

## Key Findings

### Dead Features (zero activation across all classes)

- **in_unchecked**: Never activated in any class. The feature is wasted and provides no discriminative signal.

### Declaration-Only Activation

These features are only activated on declaration nodes (type_id < 8/12), not CFG nodes:
- **external_call_count**
- **has_loop**
- **return_ignored**
- **uses_block_globals**

*Expected*: Features like `in_unchecked`, `has_loop`, `external_call_count` are function-level and should only activate on declaration (FUNCTION) nodes. CFG nodes get 0.0 for these by design.

## Comparison with Prior Audit (Task 09)

Task 09 computed overall activation rates without splitting by node type. This audit provides a more granular view:

- **Declaration-only rates** show feature activation on structural nodes (CONTRACT, FUNCTION, STATE_VAR, etc.)
- **CFG-only rates** show feature activation on statement-level nodes (CFG_NODE_CALL, CFG_NODE_WRITE, etc.)
- The split reveals that most semantic features (has_loop, in_unchecked, external_call_count) are **function-level signals** that only activate on declaration nodes — CFG nodes always get 0.0 for these features
- This is **by design** (see graph_extractor.py `_build_cfg_node_features`) — the GNN propagates function-level signals to CFG nodes via CONTAINS edges

