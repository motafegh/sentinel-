# Task 09: Feature Range Audit
**Sample size:** 500 graph files  **Files loaded:** 500  **Total nodes:** 66288 (declaration: 18113, CFG: 48175)
## Feature Statistics
```
       Feature       | Category |  min   |   max    |   p5   |  p50   |  p95   | count_gt1 | count_lt_neg1 | nan_count | inf_count 
-----------------------------------------------------------------------------------------------------------------------------------
       type_id       |   ALL    | 0.0000 |  1.0000  | 0.0000 | 0.8333 | 1.0000 |     0     |       0       |     0     |     0     
       type_id       |   DECL   | 0.0000 |  0.5833  | 0.0000 | 0.0833 | 0.5000 |     0     |       0       |     0     |     0     
       type_id       |   CFG    | 0.6667 |  1.0000  | 0.6667 | 1.0000 | 1.0000 |     0     |       0       |     0     |     0     
      visibility     |   ALL    | 0.0000 |  2.0000  | 0.0000 | 0.0000 | 1.0000 |    358    |       0       |     0     |     0     
      visibility     |   DECL   | 0.0000 |  2.0000  | 0.0000 | 0.0000 | 1.0000 |    358    |       0       |     0     |     0     
      visibility     |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
  uses_block_globals |   ALL    | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
  uses_block_globals |   DECL   | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
  uses_block_globals |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
         view        |   ALL    | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
         view        |   DECL   | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 1.0000 |     0     |       0       |     0     |     0     
         view        |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
       payable       |   ALL    | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
       payable       |   DECL   | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
       payable       |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
      complexity     |   ALL    | 0.0000 | 67.0000  | 0.0000 | 0.0000 | 5.0000 |    9046   |       0       |     0     |     0     
      complexity     |   DECL   | 0.0000 | 67.0000  | 0.0000 | 1.0000 | 9.4000 |    9046   |       0       |     0     |     0     
      complexity     |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
         loc         |   ALL    | 0.1003 | 946.0000 | 0.1003 | 1.0000 | 7.0000 |   11507   |       0       |     0     |     0     
         loc         |   DECL   | 0.1003 | 946.0000 | 0.1003 | 0.2007 | 1.0000 |    806    |       0       |     0     |     0     
         loc         |   CFG    | 1.0000 | 129.0000 | 1.0000 | 1.0000 | 9.0000 |   10701   |       0       |     0     |     0     
    return_ignored   |   ALL    | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
    return_ignored   |   DECL   | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
    return_ignored   |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
  call_target_typed  |   ALL    | 0.0000 |  1.0000  | 1.0000 | 1.0000 | 1.0000 |     0     |       0       |     0     |     0     
  call_target_typed  |   DECL   | 0.0000 |  1.0000  | 1.0000 | 1.0000 | 1.0000 |     0     |       0       |     0     |     0     
  call_target_typed  |   CFG    | 1.0000 |  1.0000  | 1.0000 | 1.0000 | 1.0000 |     0     |       0       |     0     |     0     
     in_unchecked    |   ALL    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
     in_unchecked    |   DECL   | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
     in_unchecked    |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
       has_loop      |   ALL    | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
       has_loop      |   DECL   | 0.0000 |  1.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
       has_loop      |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
 external_call_count |   ALL    | 0.0000 |  0.9494  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
 external_call_count |   DECL   | 0.0000 |  0.9494  | 0.0000 | 0.0000 | 0.3608 |     0     |       0       |     0     |     0     
 external_call_count |   CFG    | 0.0000 |  0.0000  | 0.0000 | 0.0000 | 0.0000 |     0     |       0       |     0     |     0     
```
## Known Bugs (suppressed)
- BUG-1: CFG loc can exceed 1.0 (raw line count on CFG nodes)
- BUG-2: complexity can exceed 1.0 (raw block count on CFG nodes)
- BUG-3: visibility=2 (private) is valid ordinal encoding
## New Flags
- **NEW FLAG**: type_id — DECL min (0.0000) vs CFG min (0.6667) differ by >0.5
- **NEW FLAG**: visibility — DECL max = 2.0000 > 1.0 (expected [0,1])
- **NEW FLAG**: visibility — DECL max (2.0000) vs CFG max (0.0000) differ by >0.5
- **NEW FLAG**: uses_block_globals — DECL max (1.0000) vs CFG max (0.0000) differ by >0.5
- **NEW FLAG**: view — DECL max (1.0000) vs CFG max (0.0000) differ by >0.5
- **NEW FLAG**: payable — DECL max (1.0000) vs CFG max (0.0000) differ by >0.5
- **NEW FLAG**: complexity — DECL max = 67.0000 > 1.0 (expected [0,1])
- **NEW FLAG**: complexity — DECL max (67.0000) vs CFG max (0.0000) differ by >0.5
- **NEW FLAG**: loc — DECL max = 946.0000 > 1.0 (expected [0,1])
- **NEW FLAG**: loc — DECL min (0.1003) vs CFG min (1.0000) differ by >0.5
- **NEW FLAG**: loc — DECL max (946.0000) vs CFG max (129.0000) differ by >0.5
- **NEW FLAG**: return_ignored — DECL max (1.0000) vs CFG max (0.0000) differ by >0.5
- **NEW FLAG**: call_target_typed — DECL min (0.0000) vs CFG min (1.0000) differ by >0.5
- **NEW FLAG**: has_loop — DECL max (1.0000) vs CFG max (0.0000) differ by >0.5
- **NEW FLAG**: external_call_count — DECL max (0.9494) vs CFG max (0.0000) differ by >0.5
