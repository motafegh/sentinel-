# Phase 5 Verification Report
**Generated:** 2026-06-08 (Session 2)
**Stage 5.5 status:** DEFERRED (Run 9 training active — VRAM unavailable)
**Dataset version:** v1.3 (pre-Stage 5.5)

## Summary
- Input: 67,311 contracts (v1.1+12, post D-I-11/12)
- Output: 67,311 contracts with verified labels
- Labels dropped: 46,977
- Labels kept: 7,403
- Newly reclassified to NonVulnerable: 18,751

## Per-Class Gate Results

class,n_total,n_keep,pct_keep,n_drop,pct_drop,n_uncertain,pct_uncertain,pct_high_conf,gate
Class11:Reentrancy,17698,1699,9.6,15999,90.4,0,0.0,99.8,VERIFIED ✅
Class08:CallToUnknown,11131,239,2.1,10892,97.9,0,0.0,87.9,PROVISIONAL ✅ → Stage 5.5
Class04:Timestamp,2674,1075,40.2,1599,59.8,0,0.0,52.6,BEST-EFFORT → structural patterns applied
Class01:ExternalBug,3604,344,9.5,3260,90.5,0,0.0,93.1,PROVISIONAL ✅ → Stage 5.5
Class02:GasException,6879,2794,40.6,4085,59.4,0,0.0,80.8,PROVISIONAL ✅ → Stage 5.5
Class09:DenialOfService,12394,1252,10.1,11142,89.9,0,0.0,64.5,BEST-EFFORT → structural patterns applied

## What was NOT verified (Stage 5.5 pending)
- GraphCodeBERT embedding + HDBSCAN cluster-based propagation
- This would improve confidence for Timestamp (52.6%) and DoS (64.5%) classes
- PREREQUISITE: `ps aux | grep train.py` shows no active training process

## Files
- `contracts_clean_v1.3.csv` — full dataset with all metadata
- `contracts_clean_v1.3_compact.csv` — id + labels + verdicts only
- `p5_s4_final_verdict.csv` — per-contract automated verdicts
- `p5_s4_gate_results.csv` — per-class gate summary
- `review_batches/` — ~40 contracts per class for manual QA
