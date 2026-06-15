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

## Training data (from checkpoint config)

Run 9 was trained on the **original BCCC labels** — NOT Phase 5 cleaned data. Phase 5
was a retrospective audit done after Run 9 completed to explain its performance.

Actual files used (verified from checkpoint config):
- `ml/data/processed/multilabel_index_deduped.csv` — original BCCC labels, deduplicated
- `ml/data/splits/deduped/` — train=29,101 / val=6,234 / test=6,241
- `ml/data/cached_dataset_v9.pkl` — v9 schema graphs

Actual train-split label counts:
| Class | Train positives |
|---|---|
| IntegerUO | 9,486 |
| Reentrancy | 3,100 |
| CallToUnknown | 2,237 |
| MishandledException | 2,874 |
| GasException | 3,392 |
| Timestamp | 678 |
| TransactionOrderDep. | 2,048 |

Phase 5 retrospectively estimated ~89% FP rate for Reentrancy and ~87% for CallToUnknown
in the full raw BCCC corpus — the same underlying issues affect `multilabel_index_deduped.csv`,
but the Phase 5 percentages were computed on a larger, different slice of the data.

## Interpretation

**Genuinely learned:**
- **IntegerUO** (100% Top-1): IntegerUO had 9,486 training examples — the largest class
  — and the token-level text patterns (arithmetic keywords, ERC-20 SafeMath) are strong
  and consistent. The Transformer + Fused eyes avg P=0.81/0.82 on SolidiFI contracts.

**Partially learned:**
- **Timestamp** (48% Top-1, 74% Top-2): 678 training examples (smallest positive class).
  Sparse but relatively clean signal; Fused eye avg P=0.76.
- **Reentrancy** (36% Top-1, 90% Top-2): 3,100 training examples but heavily noisy (Phase 5
  estimates ~89% of full BCCC Reentrancy labels are FP). The ~11% genuine signal carried
  through — Fused eye avg P=0.55.

**Not learned (noise-level discrimination):**
- **TOD**: All four eyes below 0.18. Noisy labels + vulnerability requires multi-tx reasoning.
- **CallToUnknown**: All eyes below 0.18. Noisy labels + category mismatch with SolidiFI.
- **MishandledException**: All eyes below 0.33. 2,874 training examples but syntax era
  mismatch (Solidity 0.5 `.call.value()` vs 0.8 `(bool,) = .call{value:}("")` in training).

## Key takeaway for v2 dataset

Label quality in training directly impacts OOD detection:
- High training volume + consistent text signal → learned (IntegerUO)
- Sparse but relatively clean labels → partial learning (Timestamp)
- Noisy labels → zero or near-zero learning (TOD, CallToUnknown)
- Clean labels but syntax era mismatch → zero learning (MishandledException)

Improving label quality AND including Solidity 0.5 examples in v2 are both required
levers for performance improvement.

## Run configuration
- Checkpoint: `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` (ep52)
- Thresholds: `ml/calibration/GCB-P1-Run9-v11-20260606_thresholds.json`
- Tier: 0.55 confirmed / 0.25 suspicious
- Tuned: 0.300–0.375 per-class (CallToUnknown=0.325, Reentrancy=0.350, TOD=0.300, etc.)
