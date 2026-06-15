# Comparison: SENTINEL Comprehensive Benchmark vs Prior "Benchmark"

> **Date:** 2026-06-14
> **Author:** Claude (post contamination audit)
> **Context:** Replaces the inflated benchmark approach that tested on 80-95% training data.

## TL;DR

The prior benchmark approach (`benchmark_run9_smartbugs.py` + `benchmark_run9_solidifi.py`) was **measuring training accuracy, not OOD performance**. The new `data_module/benchmarks/` approach is properly OOD-verified, multi-source, and versioned.

## Side-by-side comparison

| Aspect | Prior (`benchmark_run9_*`) | New (`data_module/benchmarks/`) |
|---|---|---|
| **Sources** | SmartBugs Curated + SolidiFI (both heavily contaminated) | 5 tiers: existing OOD + DeFiHackLabs + BCCC 2-tool + mutation + safe |
| **Total contracts (current state)** | 493 raw, 66 honest OOD | 66 (Tier A) + 100 (Tier E) = 166 in v0.1; ~3,000-4,500 in v1.0 |
| **Contamination ratio (verified)** | 80-95% in v3 training | 0% (HARD GATE per build) |
| **OOD verification** | None — assumed clean | SHA-256 audit per build + per evaluation run |
| **Label quality** | SmartBugs = DASP-10, SolidiFI = synthetic injection (clean for honest OOD; contaminated for the rest) | 2-tool consensus (Tier C) + manual verification (Tier B DeFiHackLabs CVE) + hand-crafted (Tier D) + 2-tool clean (Tier E) |
| **Class coverage (v0.1 quickstart)** | 6 classes (CtU, ME, Reentrancy, ToD, Timestamp, NonVulnerable) — IntegerUO, DoS, ExternalBug, UnusedReturn have 0 honest OOD | Same as prior for v0.1; v1.0 will fill gaps via Tier C (BCCC 2-tool) + Tier D (mutation) |
| **Version mix** | Pre-0.6 (SolidiFI) + 0.4-0.6 (SmartBugs) | 0.3-0.8 (full range) in v1.0 |
| **Negative examples** | Implicit (NonVulnerable subset) | Explicit (Tier E: 100 contracts 2-tool verified clean) |
| **Reproducible** | Script + path | Script + version tag + manifest + contamination_check.json |
| **Re-evaluable** | Yes (run scripts) | Yes (re-run `evaluate.py`) |
| **Comparable across runs** | Hard (no version tag) | Easy (version tag + manifest) |
| **Statistically meaningful** | No (6 contracts per class honest OOD) | Yes in v1.0 (30+ target per class) |

## What changed in the post-training process

`data_module/temp/live_plans/post_training_process_complete_2026-06-14.md` §4 was updated to:
1. Add Phase 3.0 (HARD GATE): v3-aware contamination audit
2. Make the comprehensive benchmark the primary evaluation method
3. Document that prior benchmark numbers (Run 9 tier 0.2965 / tuned 0.3081) were inflated
4. Provide Option A (honest OOD only) for Run 12 reporting

## Honest comparison of prior numbers

| Run | Benchmark F1 reported | Actual honest OOD F1 (estimated) |
|---|---|---|
| Run 9 | tier 0.2965, tuned 0.3081 | Unknown — needs re-evaluation on 6 SB + 60 SF honest OOD |
| Run 10 | (not reported; F1=0.683 was v1 leakage, not benchmark) | n/a (overfit to leakage) |
| Run 11 | (paused at ep1) | n/a |
| Run 12 (current) | to be reported | to be reported |

**Action item:** for honest historical comparison, re-evaluate Run 9 best on the v0.1 quickstart benchmark (66 contracts) and report both numbers (full contaminated + honest OOD).

## Migration timeline

- **2026-06-14:** v0.1 quickstart built (Tier A + E skeleton, 66 contracts, 0% contamination verified)
- **2026-06-14 to 2026-06-15:** Tier E (NonVulnerable safe) — 4-6 hours
- **2026-06-15 to 2026-06-16:** Tier B (DeFiHackLabs held-out) — 4-6 hours
- **2026-06-16 to 2026-06-20:** Tier C (BCCC 2-tool consensus) — 3-5 days
- **2026-06-20 to 2026-06-22:** Tier D (mutation-based for rare classes) — 2-3 days
- **2026-06-22:** v1.0 released (~3,000-4,500 contracts, all OOD, all 8-9 classes)
- **2026-06-25+:** evaluate.py fully implemented, integrated with post-training process

## What Run 12 can use RIGHT NOW

- **v0.1 quickstart (66 contracts)** — usable for Run 12 final validation
  - Run contamination_check.py first (HARD GATE)
  - Run evaluate.py with the Run 12 best checkpoint
  - Report the 6-class F1 (others will be 0 because the benchmark doesn't have them yet)
- **Run 9 re-evaluation** — also possible, for historical comparison

## What Run 12 CANNOT use

- SmartBugs + SolidiFI full sets (80-95% contaminated) — these inflate F1 by memorization
- The old `ml/scripts/benchmark_run9_*.py` scripts — they evaluate on contaminated data
- Any F1 claim that doesn't include the contamination disclosure
