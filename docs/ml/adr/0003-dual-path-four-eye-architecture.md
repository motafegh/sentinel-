# ADR-0003: Dual-path GNN+CodeBERT Four-Eye architecture

**Status:** Accepted
**Date:** 2026-06-06
**Deciders:** Ali Motafegh, Claude (assistant)
**Supersedes:** v7 single-eye design (deprecated 2026-05-11, see
  `docs/changes/2026-05-11-v5-implementation-record.md`)
**Superseded by:** None (current)

## Context

SENTINEL classifies Solidity smart contracts for vulnerability patterns. Each contract
has two natural representations that capture different information:

- **A graph:** Slither's CFG + call graph + declaration graph (per-contract, structural,
  hand-crafted features, ~50–500 nodes). Captures *how* the code is structured —
  call patterns, state variables, control flow, scope.
- **A sequence of tokens:** raw Solidity source run through GraphCodeBERT (a 124M-param
  RoBERTa pretrained on code). Captures *what* the code looks like — naming, idioms,
  comments, the syntactic surface.

Early experiments (v3, v4) used a single encoder — either GNN-only (PLAN-3A) or
CodeBERT-only (GCB-P0). Results plateaued at F1=0.27–0.29 because each encoder alone
missed information the other would catch. The GNN missed semantic similarity (e.g. a
reentrancy variant using a slightly different pattern), and CodeBERT missed structural
properties (e.g. whether a function is called from inside an `unchecked{}` block, which
requires CFG scope information).

The natural design is a **dual-path** model: both encoders run in parallel, and their
outputs are fused before classification. The question is then how many "eyes" (=
classification heads) to use, and what each eye should see.

v5 tried a **three-eye** design (GNN, Transformer, Fused). The F1 plateaued at 0.26
with a recurring issue: the CFG-derived features (specifically `complexity` = CFG node
count) dominated the GNN eye's signal, masking class-specific features. Run 7's
interpretability audit confirmed this — `complexity` was the top feature for all 10
classes at 34–36% importance, and per-class features (`return_ignored`,
`external_call_count`, `uses_block_globals`) showed zero per-class elevation.

The fix was to add a **fourth eye** that consumes the Phase 2 GNN output (CFG subgraph
embeddings, post-control-flow-message-passing) directly, bypassing the complexity
dominance that came from pooling all GNN nodes together.

## Decision

We use a **dual-path, four-eye classifier**:

### Path 1: GNN encoder
- **3-phase, 8-layer GAT** on the contract graph (see ADR-0004)
- **Input:** `[N, NODE_FEATURE_DIM]` (v9: 12) + node type embedding (16-dim)
- **Output:** per-node embeddings, then per-phase outputs
- **JK-attention aggregation** over the 3 phases to produce a single graph-level
  embedding per contract

### Path 2: GraphCodeBERT
- **Frozen** GraphCodeBERT base (124M params)
- **LoRA fine-tuning** with `r=16, α=32` on `query` and `value` projections across all
  12 layers. LoRA is a hard requirement — without it, CodeBERT has 0 trainable
  parameters in our scope and cannot adapt to vulnerability semantics
  (`ml/src/models/transformer_encoder.py:66-72` raises `RuntimeError` if `peft` is
  missing).
- **Windowed tokenization:** contracts are split into overlapping 512-token windows
  (510 content + `[CLS]` + `[SEP]`), stride 256, max 4 windows, linspace subsample
- **LayerNorm(768)** on the `[CLS]` token output before fusion

### Fusion
- `CrossAttentionFusion`: bidirectional attention between GNN `[B, hidden_dim]` and
  CodeBERT `[B, 768]` embeddings, output dim 128 (locked, do not change)
- `_gnn_to_bert_proj`: Linear projection from GNN dim to 768 before cross-attention
- This lets the GNN attend to token positions and vice versa, instead of just being
  concatenated

### Four eyes
| Eye | Input | Purpose |
|-----|-------|---------|
| **GNN eye** | GNN graph-level embedding (post-JK) | Structural signal (call graph, state vars) |
| **TF eye** | CodeBERT `[CLS]` embedding | Semantic/syntactic signal (naming, idioms) |
| **Fused eye** | CrossAttentionFusion output | Joint structural+semantic signal |
| **CFG eye** | Phase 2 GNN output (`_phase2_x`) | CFG-subgraph signal, bypasses complexity dominance |

Each eye produces a `[B, 10]` logit tensor. They are **summed** (not concatenated) for
the final loss and per-class sigmoid. This is the "four-eye" formulation.

A small linear classifier `[512 → 256 → 10]` projects each eye's pooled embedding.

The choice of **4 eyes** is deliberate: the GNN eye captures graph-level structure,
the TF eye captures surface code, the Fused eye captures their interaction, and the
CFG eye captures *intra-function* structure that the GNN eye washes out via
mean-pooling. Each eye has a different failure mode; summing their logits averages
the failures.

## Consequences

### Positive

- **Interpretability.** Each eye can be evaluated in isolation. The Run 7
  interpretability suite (`docs/interpretability/`) confirmed which eye contributes
  which signal: GNN eye is useful solo for IntegerUO, TF eye for the high-resource
  classes, Fused eye best for Reentrancy, CFG eye for the structural patterns.
- **Graceful degradation.** If one eye is miscalibrated, the other three compensate.
  The four-eye F1 is consistently higher than the best three-eye ablation.
- **Disables complexity proxy.** The CFG eye sees the Phase 2 output *before* the JK
  aggregation that was washing out the class-specific features. This was the L4 fix
  in `RUN8-ULTRACODE.md`.
- **Independent training signals.** Each eye can be ablated independently in
  interpretability experiments. The B1/B2/B3 experiments in the Phase 2
  interpretability suite use this.

### Negative

- **Computational cost.** 4 eyes × `[512 → 256 → 10]` linear = 1.3M extra parameters
  per eye, ~5.2M total. The GNN is the dominant cost. At 35 min/epoch on RTX 3070 8GB,
  this is the ceiling.
- **Aux loss is required.** Without an aux loss on each eye, the GNN eye collapses
  early (GNN share → 0% by ep8 in some runs). The aux BCE pathway supervision
  (`aux_phase2=0.20` weight, 0→0.30 warmup over 8 epochs) forces the GNN eye to
  stay engaged.
- **Per-eye calibration drift.** Each eye has its own ECE and per-class threshold
  behavior. Tuning is per-eye, not global. The predictor's 3-tier output (CONFIRMED /
  SUSPICIOUS / NOTEWORTHY) must account for which eye drove the high score.

### Neutral

- The model file is `ml/src/models/sentinel_model.py`. It exports `SentinelModel`
  which encapsulates the four-eye design. The 3-eye design is preserved in git
  history (commit `bf57069` and earlier v5/v5.1 work).

## Alternatives Considered

**1. Single-path GNN only (v3, PLAN-3A).**

Rejected at v4. Plateaued at F1=0.27. The 8% gap from the four-eye design
(Run 7: 0.3423 tuned) is consistent across runs. The GNN alone misses semantic
patterns that CodeBERT picks up trivially.

**2. Single-path CodeBERT only (v4 GCB-P0).**

Rejected. Plateaued at F1=0.22. Without the structural signal, CodeBERT cannot
distinguish "uses block.timestamp for randomness" (Timestamp) from "uses
block.timestamp for time-locked rewards" (not a vulnerability). Both are semantically
similar; only the structural context (no source of randomness elsewhere) tells them
apart.

**3. Three-eye (GNN + TF + Fused), no CFG eye.**

This is v5/v5.1. Plateaued at F1=0.2651 (tuned). The complexity proxy dominated the
GNN eye, and the F1 ceiling was clear from the B1/B2/B3 interpretability results.
The CFG eye was added in v7/v8 specifically to bypass this.

**4. Concat instead of sum for the four eyes.**

Considered. Rejected because concat would require the classifier head to learn the
weight per-eye from scratch, which on this dataset size (29K training contracts) is
underdetermined. Sum uses the natural class probability average.

**5. Cross-attention only, no separate eyes.**

Considered. Would give a single `[B, 10]` from the fused output. Rejected because
the per-eye interpretability and per-eye calibration is valuable for the agents
module's 3-tier output.

## References

- Source: `ml/src/models/sentinel_model.py:308-470` (Four-Eye classifier)
- Source: `ml/src/models/gnn_encoder.py:160-220` (GNN input projection, JK attention)
- Source: `ml/src/models/transformer_encoder.py:1-388` (CodeBERT + LoRA, peft hard req)
- Source: `ml/src/models/fusion_layer.py` (CrossAttentionFusion)
- Source: `ml/src/training/trainer.py:1389-1672` (per-eye LR groups, aux loss warmup)
- Run history: `docs/changes/2026-05-11-v5-implementation-record.md` (v5/v5.1 three-eye)
- Run history: `docs/changes/2026-05-14-v5.2-pre-training-implementation.md` (JK
  attention added, four-eye finalized)
- Audit: `docs/pre-run8-fixes/RUN8-ULTRACODE.md` (L4 fix: CFG eye as complexity bypass)
- Interpretability: `docs/interpretability/SENTINEL-Understanding-Run7.md` (per-eye
  analysis)
- Related: ADR-0004 (three-phase GAT routing, defines what each eye sees)
- Related: ADR-0006 (loss formulation, aux loss is the four-eye training signal)
