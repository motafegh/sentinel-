# Run 9 — SolidiFI OOD Benchmark (2026-06-10)

## Dataset
- **SolidiFI**: 350 standalone `.sol` files, 50 per category × 7 categories
- **Contamination**: 341/350 clean (9 Unchecked-Send near-dups excluded)
- **Pragma**: all `>=0.4.22 <0.6.0` → compiled with solc 0.5.17
- **Graph size**: median 250 nodes vs training median 90 (2.8× OOD)
- **Errors**: 0/341 (all compiled and processed successfully)

### Bug fixed during this session
`_detect_solc_version()` in `ml/src/inference/preprocess.py` used to pick `0.4.26`
for range pragmas like `>=0.4.22 <0.6.0` (extracting only the lower bound). SolidiFI
injects `address payable` (Solidity 0.5.0+ syntax), causing Slither to fail.
Fix: detect the exclusive upper bound (`<0.6.0`) and select `0.5.17` instead.

## Results

### Raw numbers (misleading — see below)

| Category | SENTINEL class | N | Tier R | Tuned R |
|---|---|---|---|---|
| Overflow-Underflow | IntegerUO | 50 | 1.00 | 1.00 |
| Re-entrancy | Reentrancy | 50 | 1.00 | 1.00 |
| TOD | TransactionOrderDependence | 50 | 1.00 | 0.98 |
| Timestamp-Dependency | Timestamp | 50 | 1.00 | 0.98 |
| Unchecked-Send | CallToUnknown | 41 | 1.00 | 0.95 |
| Unhandled-Exceptions | MishandledException | 50 | 1.00 | 0.98 |
| **TOTAL** | | **291** | **1.00** | **0.98** |

**Why these numbers are meaningless**: the model outputs `confirmed=1, suspicious=9` on
virtually every contract — all 10 classes are above the 0.25 suspicious threshold
simultaneously. The FP probe confirms: 100% of `tx.origin` contracts (no SENTINEL
equivalent) also pass both tier and tuned thresholds. Recall=1.0 is trivially achieved.

### Honest metric: Rank-based analysis

Does the model at least *rank* the correct vulnerability class highest?

| Category | SENTINEL class | N | Top-1% | Top-2% | Top-3% | AvgP(correct) | AvgP(other) |
|---|---|---|---|---|---|---|---|
| Overflow-Underflow | IntegerUO | 50 | **100%** | 100% | 100% | 0.7233 | 0.4261 |
| Timestamp-Dependency | Timestamp | 50 | 48% | **74%** | 82% | 0.6980 | 0.5238 |
| Re-entrancy | Reentrancy | 50 | 36% | **90%** | 98% | 0.5673 | 0.4083 |
| TOD | TransactionOrderDependence | 50 | 0% | 0% | 2% | 0.3875 | 0.3925 |
| Unchecked-Send | CallToUnknown | 41 | 0% | 0% | 5% | 0.3887 | **0.4166** |
| Unhandled-Exceptions | MishandledException | 50 | 0% | 6% | 20% | 0.4124 | 0.4015 |

## Interpretation

**Genuinely learned:**
- **IntegerUO** (100% Top-1): Perfect discrimination. BCCC Phase 5 confirmed IntegerUO
  labels were `VERIFIED (clean)` — this translates directly to strong OOD performance.

**Partially learned:**
- **Timestamp** (48% Top-1, 74% Top-2): Model assigns meaningfully higher probability
  to the correct class (0.70 vs 0.52). BCCC had 40.2% label retention at BEST-EFFORT gate.
- **Reentrancy** (36% Top-1, 90% Top-2): Weak discrimination in top slot but correct
  class almost always top-2. Despite 89% FP rate in BCCC training, the 11% clean
  signal carried through.

**Not learned (noise-level discrimination):**
- **TOD**: AvgP(correct)=0.3875 ≈ AvgP(other)=0.3925 — random.
- **CallToUnknown**: AvgP(correct) *lower* than AvgP(other) — BCCC 86.9% FP confirmed.
- **MishandledException**: Marginally above noise (6% Top-2 — likely random).

## Key takeaway for v2 dataset

This benchmark directly validates the Phase 5 BCCC label-quality audit predictions:
- Classes with verified/clean BCCC labels → model learned them (IntegerUO)
- Classes with moderate label quality → partial learning (Timestamp, Reentrancy)  
- Classes with near-random BCCC labels → zero learning (TOD, CallToUnknown)

Improving label quality in v2 (multi-source verified pipeline) is the primary lever
for performance improvement. Run 9 is hitting the label-quality ceiling, not an
architectural ceiling.

## Run configuration
- Checkpoint: `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` (ep52)
- Thresholds: `ml/calibration/GCB-P1-Run9-v11-20260606_thresholds.json`
- Tier: 0.55 confirmed / 0.25 suspicious
- Tuned: 0.300–0.375 per-class (CallToUnknown=0.325, Reentrancy=0.350, TOD=0.300, etc.)
