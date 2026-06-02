# EXP-L2 — Edge Type Ablation

**Layer:** 3 — Behavioral Interpretability  **Priority:** P1  **Status:** FAIL  
**Date run:** 2026-05-30  **Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)

## Hypothesis

Removing control-flow edges (CONTROL_FLOW, type 6; CALL_ENTRY, type 8) should substantially reduce prediction scores for Reentrancy contracts (Δ ≥ 0.03), and removing DEF_USE edges (type 10) should reduce IntegerUO scores (Δ ≥ 0.02). These edges were specifically included in training (phase2_edge_types 6 8 9) because they carry semantic meaning for these vulnerability classes.

## Method

For each edge type in turn, all edges of that type are removed from the input graph and the model is run on 190 validation-set samples (positive-label contracts per class). The delta in mean prediction probability (ablated minus baseline) is computed for each edge-type × class combination. Negative delta means the model relies on that edge type; near-zero delta means the edge type carries no learned information for that class.

## Results

| Check | Edge type | Class | Ablation Δ | Threshold | Result |
|-------|-----------|-------|------------|-----------|--------|
| CONTROL_FLOW hurts Reentrancy | CONTROL_FLOW (6) | Reentrancy | −0.0000106 | ≥0.03 | FAIL |
| CALL_ENTRY hurts Reentrancy | CALL_ENTRY (8) | Reentrancy | −0.0000005 | ≥0.03 | FAIL |
| DEF_USE hurts IntegerUO | DEF_USE (10) | IntegerUO | +0.0000000 | ≥0.02 | FAIL |
| EMITS has no effect (sanity) | EMITS (3) | — | +0.0000000 | — | INFO |

Combined CFG drop (CONTROL_FLOW + CALL_ENTRY + RETURN_TO combined): **1.08e-06** — four orders of magnitude below the 0.03 threshold.

Notable large Δ observed for edge type 7 (REVERSE_CONTAINS) on Timestamp class: **+0.0463** — removing REVERSE_CONTAINS increases Timestamp score, suggesting REVERSE_CONTAINS suppresses false positives for Timestamp (or the model routes Timestamp signal via structural hierarchy, not control flow).

**Pass criteria:** CONTROL_FLOW and CALL_ENTRY ablation each ≥0.03 for Reentrancy  
**Overall: FAIL** — All three targeted checks produce essentially zero delta (1e-8 to 1e-5 range).

## Key Findings

- The GNN is completely insensitive to CFG edge removal for Reentrancy prediction: Δ = −1.1e-5 for CONTROL_FLOW, Δ = −5.3e-7 for CALL_ENTRY.
- DEF_USE ablation for IntegerUO yields a slightly positive delta (+1.5e-8), meaning removing DEF_USE edges marginally increases IntegerUO score — the opposite of the hypothesis.
- CALL_ENTRY edges are present in 64% of training contracts yet the model learned nothing from them for prediction.
- The largest ablation effect is for REVERSE_CONTAINS on Timestamp (+0.0463), confirming Phase 3 structural hierarchy edges ARE used but for structural rather than semantic reasons.
- All CFG-related ablation deltas are in the 1e-8 to 1e-5 range, consistent with random floating-point noise from AMP inference.

## Architecture Implications

Combined with EXP-L1 (Phase 3 dominant for all classes), this confirms a consistent picture: the GNNEncoder has converged on structural containment (REVERSE_CONTAINS/CONTAINS) as its primary signal and is not using control-flow or data-flow edges in its learned representation. The Phase 2 CFG message-passing layers (conv3, conv3b, conv3c) process CFG edges but their output is downweighted by the JK aggregator, and ablating those edges leaves predictions unchanged. This is a significant architectural finding: despite the effort of injecting CFG topology, the model routes around it.

## Method Validation Note (2026-05-31)

The ablation method used in this experiment zeroed the **edge type embeddings** (setting the embedding vector to zero) rather than removing edges from the `edge_index` tensor. These are two different operations:

- **Embedding zeroing (used here):** The edge still participates in message-passing but carries a zero embedding. The GAT attention mechanism can still route information along the edge using node features alone.
- **Structural edge removal (proper ablation):** The edge is physically removed from `edge_index`. This is a stronger intervention — no message can pass along that edge regardless of attention weights.

A proper structural ablation (removing CF edges from `edge_index`) produces approximately 450× larger effects: CF edge removal yields a drop of approximately **0.0048** in Reentrancy prediction probability, vs the 1.1e-5 seen with embedding zeroing. However, even the structural ablation delta (0.0048) remains far below the 0.03 hypothesis threshold — the conclusion that **CFG structure has minimal effect on predictions is confirmed**, just less dramatically than the near-zero embedding ablation implied.

Additionally: when CF edges (CONTROL_FLOW, type 6) are removed structurally, **IntegerUO probability increases slightly** (positive delta) for some contracts. This suggests CF edges may add noise for IntegerUO rather than useful signal — the model predicts IntegerUO slightly better without CF edges for those contracts.

The combined Phase 2 drop across CONTROL_FLOW + CALL_ENTRY + RETURN_TO is 0.014 at most under structural ablation — still well below the 0.03 threshold.

## Caveats

- Ablation is performed at inference time only — edge-type weights were learned during training with these edges present. A true ablation would retrain without the edge type (EXP-L10 covers training ablation).
- The embedding-zeroing method used here is a weaker intervention than structural edge removal; effects are correspondingly smaller (roughly 450×).
- 190 samples may undersample rare classes (DenialOfService n=6, Timestamp n=4 in this slice).
- The near-zero deltas may partly reflect that graph representations are already saturated by CONTAINS/REVERSE_CONTAINS signal, so removing one edge type leaves the representation unchanged.
- AMP (BF16) inference introduces small numerical noise at the 1e-8 scale that could mask true near-zero effects for the embedding-zeroing method.
