# ADR-0004: Three-phase GAT routing (Phase 1 / Phase 2 / Phase 3)

**Status:** Accepted
**Date:** 2026-06-06
**Deciders:** Ali Motafegh, Claude (assistant)
**Supersedes:** None (current)
**Superseded by:** None (current)

## Context

SENTINEL's GNN encoder is built on the observation that **different edge types in the
contract graph carry different semantic roles**, and a single message-passing scheme
that treats all edges uniformly loses information.

The contract graph has these edge types (post-v9, see `graph_schema.py:382-398`):

- **Structural (ids 0–4):** `CALLS`, `READS`, `WRITES`, `EMITS`, `INHERITS` — what
  declarations reference what other declarations.
- **Intra-function control flow (ids 5–6):** `CONTAINS` (function → its CFG nodes),
  `CONTROL_FLOW` (CFG node → successor CFG node).
- **Reverse of CONTAINS (id 7):** `REVERSE_CONTAINS` — runtime-only, built inside
  GNNEncoder Phase 3 by reversing `CONTAINS(5)`. Never written to disk.
- **Cross-function (ids 8–10):** `CALL_ENTRY` (CFG node → callee entrypoint),
  `RETURN_TO` (callee terminal → call-site successor), `DEF_USE` (defining CFG node
  → reading CFG node).
- **External calls (id 11, v9):** `EXTERNAL_CALL` — self-loop on CFG nodes that make
  cross-contract calls.

Two competing pressures exist:

1. **Direction matters for control flow.** `CONTROL_FLOW` is a directed edge. Reversing
   it during message passing breaks the "execute the function in order" semantics.
2. **`CONTAINS` is bidirectional in spirit.** The function contains the CFG node, but
   the CFG node is part of the function. A round-trip (CFG→function→CFG) lets the
   function's embedding inform each CFG node and vice versa.

A naive GAT that runs all 8 layers in one phase, treating all edges uniformly, has
two failure modes:

- **Phase confusion:** a CONTROL_FLOW message and a CALL_ENTRY message are weighted
  identically. In practice, CONTROL_FLOW is the strongest signal for control-flow-
  sensitive bugs (Reentrancy, CEI), and CALL_ENTRY is the strongest signal for
  cross-function bugs (ExternalBug, delegatecall). Conflating them washes out the
  signal.
- **Pooling loss:** the v3 design mean-pooled all GNN nodes into a single graph
  embedding. Per-class features at the CFG-node level were lost. This was the L4
  finding in the Run 7 interpretability audit.

## Decision

We use a **three-phase, 8-layer GAT** with explicit edge-type routing per phase:

```
Layer 1-2: Phase 1 (structural + CONTAINS down)
  Edge types: 0,1,2,3,4 (structural) + 5 (CONTAINS function→CFG)
  Purpose: build function-level embeddings, populate CFG nodes with function context
  Heads: 1

Layer 3-5: Phase 2 (CF + ICFG-Lite + DEF_USE + EXTERNAL_CALL)
  Edge types: 6 (CONTROL_FLOW) + 8 (CALL_ENTRY) + 9 (RETURN_TO) + 10 (DEF_USE) + 11
    (EXTERNAL_CALL, v9 Fix #3)
  Sub-routing:
    conv3a (layer 3): CONTROL_FLOW only — local control flow
    conv3b (layer 4): CALL_ENTRY + RETURN_TO + EXTERNAL_CALL — cross-function
    conv3c (layer 5): CONTROL_FLOW + CALL_ENTRY + RETURN_TO + EXTERNAL_CALL +
      DEF_USE — joint CF+ICFG
  Heads: 4 (IMP-R7-1 — was 1, raised to 4 for multi-head attention on edge types)
  Output: `_phase2_x` is the post-Phase-2 CFG node embedding, exposed to the
    CFG eye of the four-eye classifier (ADR-0003)

Layer 6-8: Phase 3 (REVERSE_CONTAINS up + CONTAINS down)
  Edge types: 7 (REVERSE_CONTAINS, runtime-built) + 5 (CONTAINS)
  Purpose: round-trip aggregation — let function embeddings absorb CFG details,
    then let CFG embeddings absorb refined function context
  Heads: 1
  Output: final per-node embeddings
```

**JK-attention aggregation** over the 3 phases: each phase contributes a learned
weight, and the graph-level embedding is a softmax-weighted sum of the per-phase
outputs. The weights are logged to MLflow as `jk_phase{1,2,3}_attn`.

**Hard requirements:**
- `num_layers=8` is enforced. `GNNEncoder.__init__` raises `ValueError` on any
  other value.
- Phase 2 sub-routing is implemented in `gnn_encoder.py:516-540`. EXTERNAL_CALL is
  added to the ICFG sub-mask (v9 Fix #3 sub-routing fix — was previously only in
  the joint conv3c, which diluted the signal).

The phase split is a **runtime architectural constraint**, not a data choice.
`REVERSE_CONTAINS` edges are not stored on disk; they are constructed in-memory
inside Phase 3 by reversing `CONTAINS(5)` edges. This avoids a 2x cache size
increase.

## Consequences

### Positive

- **Per-phase interpretability.** The `jk_phase{1,2,3}_attn` weights show what the
  model uses. Run 7 Phase 2 was 0.328 (vs 0.182 in v3) — the structural signal is
  engaged. Drift in these weights is a leading indicator of training failure.
- **Clear ablation targets.** The `--phase2-edge-types` flag (added in v8) lets
  experiments disable specific Phase 2 edges. The v8-AB run (ep26, F1=0.2651)
  enabled the full Phase 2 with measurable gains.
- **Sub-routing in Phase 2 is non-obvious.** Putting EXTERNAL_CALL only in conv3c
  (the joint layer) diluted the signal because the joint layer's 5-way attention
  averages weights across 5 edge types. Adding it to conv3b (ICFG-only) raised
  the per-edge weight to ~25% of attention capacity.
- **JK aggregation is gradient-friendly.** All three phases carry gradient to the
  input projection, so no phase "disconnects" from the loss. The 0.005 entropy
  regularizer on JK weights (raised to 0.0075 in Run 8) prevents collapse to
  one phase.

### Negative

- **8-layer constraint is rigid.** Cannot experiment with 6-layer or 10-layer
  architectures without code changes. The constraint exists because the sub-routing
  is hard-coded to layers 1-2, 3-5, 6-8.
- **REVERSE_CONTAINS must be rebuilt every forward pass.** It's not cached. For a
  41K-graph dataset, this adds ~5% to forward time.
- **Phase 2 sub-routing is fragile.** When new edge types are added (e.g. v9's
  EXTERNAL_CALL), the sub-routing masks must be updated by hand. A bug in
  `gnn_encoder.py:529-532` would route EXTERNAL_CALL to the wrong sub-layer
  silently.
- **JK weight drift was a real failure mode (A3 in interpretability).** Run 7's JK
  Phase 3 weight drifted from ep1 and never corrected. The 0.005 entropy
  regularizer was added in v5.2 to combat this; Run 8 raised it to 0.0075.

### Neutral

- The three phases are not equal in capacity: 2+3+3 = 8 layers but Phase 2 has
  4× the heads (4 vs 1) and the most edge types. This is by design — control
  flow is the most semantically loaded signal.

## Alternatives Considered

**1. Single-phase 8-layer GAT, all edges equal.**

Rejected (v3 baseline). F1 plateaued at 0.27. Phase 1 vs Phase 2 vs Phase 3
attention was conflated; Reentrancy signal was diluted by the structural
function→state-variable edges.

**2. Two-phase: structural (1-4) + control flow (5-8), no REVERSE_CONTAINS round-trip.**

Considered in v4. Rejected because the function-level embedding built in Phase 1
was too coarse — by the time Phase 2 ran, the function embedding had no awareness
of the specific CFG nodes inside it. Run 4 confirmed this empirically
(F1=0.3362 ceiling at ep32).

**3. Heterogeneous GAT with separate parameters per edge type.**

Considered. Would mean ~12 separate `GATConv` modules, one per edge type, with
no shared parameters. Rejected because: (a) the parameter count explodes
(~4M extra params, exceeds VRAM budget on RTX 3070 8GB), (b) many edge types have
too few examples per batch to train independently (DEF_USE is sparse in
short contracts), (c) interpretability gets harder — which GAT learned what?

**4. Transformer over the graph (Graphormer-style).**

Considered in v5 design. Rejected because the contract graph is small (~50–500
nodes) and dense, which is the regime where GAT outperforms Graphormer. The
attention matrix for 500 nodes × 500 nodes is 1M elements; at 32 GB of attention
per layer, we cannot afford full self-attention. GAT with edge-type embeddings
gives us 95% of the expressivity at 10% of the cost.

**5. Phase 1 + Phase 2 only (no Phase 3 round-trip).**

Considered. Rejected because Phase 3 REVERSE_CONTAINS is what allows the
function-level embedding to be informed by CFG details. Without it, the GNN eye
sees the function's "average" CFG node, not its specific patterns. The
interpretability L1 finding (JK weights near-uniform) suggests this might not
matter in practice, but removing Phase 3 cost us 0.03 F1 in early ablations.

## References

- Source: `ml/src/models/gnn_encoder.py:160-220` (3-phase architecture, 8-layer
  constraint)
- Source: `ml/src/models/gnn_encoder.py:471-533` (Phase 2 sub-routing + EXTERNAL_CALL
  in conv3b)
- Source: `ml/src/preprocessing/graph_schema.py:382-398` (EDGE_TYPES vocabulary,
  FEATURE_SCHEMA_VERSION v9)
- Source: `ml/src/models/sentinel_model.py:456-461` (`_MAX_TYPE_ID` recovery from
  `x[:,0] * _MAX_TYPE_ID` round-trip)
- Source: `ml/scripts/train.py:165-166` (`--phase2-edge-types` ablation flag)
- Audit: `docs/pre-run8-fixes/PHASE2-RECEPTIVE-FIELD-ANALYSIS.md` (receptive
  field analysis that motivated the 3-phase split)
- Interpretability: `docs/interpretability/SENTINEL-Understanding-Run7.md` (JK
  weight analysis, A3 drift finding)
- Related: ADR-0001 (schema versioning, v9 added EXTERNAL_CALL)
- Related: ADR-0003 (Four-Eye architecture, the CFG eye consumes `_phase2_x`)
