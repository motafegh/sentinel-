# Stage 5.1 Evidence Coverage Report

**Generated:** 2026-06-08

**Total contracts:** 67,311

**Tool-sampled contracts:** 10,693 (15% stratified sample from Phase 4)


## Per-Class Summary

class,n_positive_bccc,n_pos_sampled,n_slither_pos,n_aderyn_pos,n_regex_pos,n_manual,n_manual_keep,n_manual_drop,pct_tool_agree_sampled_pos,pct_tool_reject_sampled_pos,stage51_gate
Class01:ExternalBug,3604,535,63,3821,9076,17,1,16,0.6,91.2,PROCEED_TO_5.2
Class02:GasException,6879,1022,413,0,N/A,12,0,6,0.6,76.7,PROCEED_TO_5.2
Class03:MishandledException,5154,782,1073,803,11969,20,0,0,7.0,86.6,VERIFIED
Class04:Timestamp,2674,394,2049,33,25698,8,5,3,11.2,54.6,PROCEED_TO_5.2
Class06:UnusedReturn,3229,846,520,1844,43212,10,0,0,8.6,69.7,VERIFIED
Class08:CallToUnknown,11131,1688,1037,3839,6742,31,4,27,12.1,83.3,PROCEED_TO_5.2
Class09:DenialOfService,12394,1858,1689,0,N/A,19,0,10,0.0,3.1,PROCEED_TO_5.2
Class10:IntegerUO,16740,2780,1056,93,63035,39,3,0,6.3,82.3,VERIFIED
Class11:Reentrancy,17698,3110,2061,1474,514,43,4,37,2.5,90.4,PROCEED_TO_5.2



## Gate Results

**VERIFIED at Stage 5.1** (3): Class03:MishandledException, Class06:UnusedReturn, Class10:IntegerUO


**Proceed to Stage 5.2** (6): Class01:ExternalBug, Class02:GasException, Class04:Timestamp, Class08:CallToUnknown, Class09:DenialOfService, Class11:Reentrancy


## Notes

- High-confidence threshold = 0.75 (weighted average from M3+M4+M9 weights in Stage 5.0)

- Gate: ≥80% of BCCC-positive contracts have high-confidence verdict from existing evidence

- Regex features are present for ALL 67,311 contracts; slither/aderyn for 10,693 (15%) only

- Manual reviews from ws_p4_s1_review_200.csv (199 contracts) used as M9 evidence
