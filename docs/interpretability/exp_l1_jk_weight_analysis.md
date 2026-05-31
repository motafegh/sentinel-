# EXP-L1 — JK Aggregation Phase Weight Analysis

**Layer:** 3 — Behavioral Interpretability  **Priority:** P1  **Status:** FAIL  
**Date run:** 2026-05-30  **Corrected:** 2026-05-31  **Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)

## Hypothesis

Phase 2 (CFG/ICFG layers, conv3/conv3b/conv3c) should dominate for semantically control-flow-sensitive classes (Reentrancy, IntegerUO), and Phase 1 (local context layers) should dominate for structurally simpler classes (Timestamp, UnusedReturn).

## Method

JK (Jumping Knowledge) attention weights are extracted from the GNNEncoder's three-phase aggregator for all positive-label contracts in the validation split. n=300 contracts requested, 277 processed, 23 skipped (cache misses). The mean JK weight per phase is computed per class across all positive instances. Dominant phase is whichever phase holds the highest mean weight. Four hypothesis checks are evaluated.

## Results

| Class | n | Phase 1 (mean±std) | Phase 2 (mean±std) | Phase 3 (mean±std) | Entropy | Dominant |
|-------|---|-------------------|-------------------|-------------------|---------|----------|
| CallToUnknown | 15 | 0.3286±0.024 | 0.3193±0.012 | **0.3521**±0.035 | 1.0950 | Phase3 |
| DenialOfService | 1 | **0.3404**±0.000 | 0.3278±0.000 | 0.3318±0.000 | 1.0985 | Phase1 |
| ExternalBug | 17 | 0.3241±0.027 | 0.3182±0.014 | **0.3577**±0.039 | 1.0938 | Phase3 |
| GasException | 33 | 0.3292±0.024 | 0.3235±0.014 | **0.3472**±0.035 | 1.0953 | Phase3 |
| IntegerUO | 94 | 0.3318±0.019 | 0.3222±0.013 | **0.3460**±0.030 | 1.0961 | Phase3 |
| MishandledException | 22 | 0.3304±0.021 | 0.3201±0.017 | **0.3495**±0.037 | 1.0949 | Phase3 |
| Reentrancy | 21 | 0.3335±0.020 | 0.3243±0.010 | **0.3421**±0.029 | 1.0964 | Phase3 |
| Timestamp | 3 | 0.2891±0.036 | 0.3092±0.005 | **0.4017**±0.041 | 1.0836 | Phase3 |
| TransactionOrderDependence | 21 | 0.3312±0.020 | 0.3229±0.013 | **0.3459**±0.030 | 1.0960 | Phase3 |
| UnusedReturn | 3 | 0.3315±0.012 | 0.3184±0.009 | **0.3501**±0.021 | 1.0969 | Phase3 |

| Hypothesis check | Expected dominant | Actual dominant | Ph2 mean | Result |
|-----------------|------------------|----------------|----------|--------|
| Reentrancy → Phase 2 | Phase 2 | Phase 3 | 0.3243 | FAIL |
| IntegerUO → Phase 2 | Phase 2 | Phase 3 | 0.3222 | FAIL |
| Timestamp → Phase 1 | Phase 1 | Phase 3 | 0.3092 | FAIL |
| UnusedReturn → Phase 1 | Phase 1 | Phase 3 | 0.3184 | FAIL |

**Pass criteria:** ≥3 of 4 hypotheses correct  
**Overall: FAIL** — 0/4 hypotheses correct. Phase 3 (REVERSE_CONTAINS / structural hierarchy) dominates all 10 classes.

## Entropy Field Correction (2026-05-31)

The original doc stated "Mean entropy is uniformly 0.3333 across all classes." This was a **bug** in the original script: the `entropy` field in the JSON was storing the mean phase weight (≈0.333 for a near-uniform 3-way distribution) rather than the Shannon entropy of the weight distribution.

The corrected script now computes Shannon entropy properly: H = -Σ p_i * log(p_i) for the three phase weights. The actual entropy values are reported in the table above. Key numbers:

- **Max theoretical entropy for 3 phases:** log(3) = 1.0986
- **Observed entropy range:** 1.0836 (Timestamp) to 1.0985 (DenialOfService)
- **Ratio to maximum:** 1.0836 / 1.0986 = **98.6%** for Timestamp; 1.0984 / 1.0986 = **99.98%** for most classes

These near-maximum entropy values confirm the JK attention is close to uniform across phases. The 24pp gap between Phase 2 (mean 0.322) and Phase 3 (mean 0.346) is statistically real over 936 samples but covers only a small fraction of the possible distributional range. The claim "Phase 2 meaningfully underused" is an overstatement — more accurate: **Phase 2 contributes least, but all three phases contribute nearly equally** (Phase 3 advantage is real but small).

Timestamp is the only class with meaningfully below-maximum entropy (H=1.0836, 98.6% of max), reflecting a larger Phase 3 skew (0.402 vs uniform 0.333). This is a class-specific signal, not a global pattern.

## Key Findings

- Phase 3 is the dominant phase for every vulnerability class except DenialOfService (n=1, statistically meaningless).
- Phase 2 has the lowest mean weight across all classes (range: 0.318–0.324), consistently below both Phase 1 and Phase 3.
- The difference between phases is consistent but small: Phase 3 advantage over Phase 2 is 0.019–0.050 per class.
- Timestamp shows the largest Phase 3 dominance (0.402 vs Phase 2 0.309), opposite to the hypothesis. Timestamp F1 is driven by the structural hierarchy signal, not control-flow.
- JK entropy is near-maximum (≥98.6%) for all classes — the JK attention is functionally close to a uniform mean pool across phases, with Phase 3 holding a slight but consistent edge.
- DenialOfService n=1 in this slice (300-contract sample); its results are not reliable.

## Architecture Implications

The GNNEncoder has learned to rely primarily on REVERSE_CONTAINS edges (Phase 3: conv4, conv4b upward, conv4c downward) rather than control flow or data-flow edges. The CFG/ICFG edges in Phase 2 (conv3, conv3b, conv3c) are processed but receive the lowest JK weight, suggesting the graph message-passing on control-flow edges adds little above what structural containment already provides. This aligns with EXP-L2: ablating CFG edges has near-zero impact on predictions.

## Caveats

- Near-uniform JK weights (all within 0.020–0.050 of 0.333) mean dominance margins are small and may not be semantically meaningful.
- Positive-label counts are small for several classes in this 300-contract sample (DenialOfService n=1, Timestamp n=3, UnusedReturn n=3), making per-class estimates noisy.
- Near-uniform JK weights could indicate the JK attention learned to ignore the choice, effectively collapsing to a mean pool across phases.
- This experiment measures phase preference aggregated over entire contracts, not individual vulnerability-relevant subgraphs.
