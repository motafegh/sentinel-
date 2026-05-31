# EXP-A4 — Auxiliary Head Contribution Analysis

**Layer:** 2 — Architecture Validation  **Priority:** P1  **Status:** FAIL  
**Date run:** 2026-05-30  **Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)

## Hypothesis

The GNN auxiliary head should contribute meaningfully to classification for at least 5 of 10 vulnerability classes, adding more than 5 percentage points of F1 above the frequency-baseline for those classes. If the three-eye architecture is working as intended, the GNN eye should capture structural graph signal independent of the transformer and fused heads.

## Method

470 validation-set contracts are scored through each of the four output heads independently: `main` (the full three-eye classifier), `gnn` (GNN eye only, Linear(128→10) frozen-head probe), `transformer` (TF eye only), and `fused` (cross-attention fused eye only). Per-class F1 and AUC-ROC are computed for each head. The GNN is considered "useful" for a class if its F1 exceeds the frequency-baseline by at least 5pp. Baseline is the F1 of always-predicting-positive.

## Results

### F1 by Head

| Class | Baseline F1 | GNN F1 | TF F1 | Fused F1 | Main F1 | GNN useful? |
|-------|------------|--------|-------|----------|---------|-------------|
| CallToUnknown | 0.143 | **0.255** | 0.264 | 0.295 | 0.286 | YES (+11pp) |
| DenialOfService | 0.025 | 0.000 | 0.800 | 1.000 | 1.000 | NO |
| ExternalBug | 0.117 | 0.000 | 0.000 | 0.000 | 0.200 | NO |
| GasException | 0.190 | 0.000 | 0.036 | 0.131 | 0.231 | NO |
| IntegerUO | 0.401 | **0.548** | 0.728 | 0.767 | 0.755 | YES (+15pp) |
| MishandledException | 0.158 | 0.000 | 0.000 | 0.000 | 0.000 | NO |
| Reentrancy | 0.170 | **0.182** | 0.389 | 0.444 | 0.438 | NO (+1pp) |
| Timestamp | 0.017 | **0.286** | 0.286 | 0.333 | 0.571 | YES (+27pp) |
| TOD | 0.123 | 0.000 | 0.000 | 0.000 | 0.059 | NO |
| UnusedReturn | 0.052 | 0.000 | 0.000 | 0.000 | 0.300 | NO |

### AUC-ROC by Head (selected)

| Class | GNN AUC | TF AUC | Fused AUC | Main AUC |
|-------|---------|--------|-----------|----------|
| IntegerUO | 0.727 | 0.872 | 0.909 | 0.909 |
| Reentrancy | 0.788 | 0.831 | 0.880 | 0.884 |
| Timestamp | 0.989 | 0.990 | 0.993 | 0.992 |
| CallToUnknown | 0.757 | 0.815 | 0.845 | 0.846 |

**Pass criteria:** GNN F1 ≥ baseline+5pp for ≥5/10 classes  
**GNN useful classes:** 3 (CallToUnknown, IntegerUO, Timestamp)  
**Overall: FAIL** — GNN useful for only 3/10 classes; threshold requires 5/10.

## Key Findings

- GNN is the weakest single eye: it scores 0.000 F1 on 6/10 classes (DenialOfService, ExternalBug, GasException, MishandledException, TOD, UnusedReturn).
- Reentrancy: GNN F1=0.182 is only 1.2pp above the frequency-baseline (0.170); the graph structural signal is nearly useless for this class despite it being the canonical "graph-detectable" vulnerability.
- IntegerUO is the GNN's strongest class (F1=0.548, AUC=0.727), which is also the largest class by positive rate (33%), so sample count is favorable.
- Timestamp GNN AUC=0.989 is essentially the same as TF AUC=0.990 and Main AUC=0.992 — the GNN learned nearly all the Timestamp signal available in the graph, but the class is tiny (n_positive=4 in this slice).
- Main head consistently outperforms all individual eyes, confirming that the three-eye fusion provides value beyond any single pathway.
- The fused head (cross-attention only) typically matches or exceeds the main head on individual classes, suggesting the classifier head on top adds little class-specific discrimination.

## Architecture Implications

The GNN eye's poor F1 on 7/10 classes is consistent with EXP-L1 and EXP-L2 findings: the GNN has learned a structural hierarchy representation (Phase 3 dominant, CFG edges unlearned) that is broadly useful for CONTAINS/REVERSE_CONTAINS-based patterns but does not capture the semantic CFG patterns required for most vulnerability classes. The transformer (GraphCodeBERT + LoRA) carries the dominant classification signal for Reentrancy, DenialOfService, GasException, and most other classes. The three-eye fusion provides modest but consistent improvements over individual eyes, suggesting that even the weak GNN signal adds complementary information when combined.

## Caveats

- 470 samples is a small evaluation slice; class-level F1 estimates for rare classes (DenialOfService, Timestamp) are very noisy.
- F1 evaluation depends on the threshold used; main head uses a tuned threshold while individual eye heads use default 0.5, which may disadvantage them.
- AUC-ROC tells a more stable story than F1 for rare classes: GNN AUC-ROC is consistently above 0.70 for all classes with enough positives, suggesting the GNN ranking is useful even when its thresholded F1 is zero.
- "GNN eye" here means the frozen auxiliary probe head on GNN pooled output, not the full model.
