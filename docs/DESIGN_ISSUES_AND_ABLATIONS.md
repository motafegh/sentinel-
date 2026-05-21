# SENTINEL ML — Design Issues, Risks, and Ablation Candidates

A living document capturing every design assumption, known risk, architecture
limitation, and experiment worth running — collected during deep code review
and learning sessions. Items are not bugs (those live in ACTIVE_BUGS.md) but
rather open questions about whether design decisions are correct, and things
worth testing or improving.

**Sections:**
1. [Ablation Candidates](#1-ablation-candidates) — controlled experiments to validate assumptions
2. [Design Risks](#2-design-risks) — assumptions baked into the architecture that might be wrong
3. [Architecture Limitations](#3-architecture-limitations) — fundamental constraints of the current design
4. [Data Quality Issues](#4-data-quality-issues) — data problems that distort what the model learns
5. [Potential Improvements](#5-potential-improvements) — concrete changes worth exploring

---

## 1. Ablation Candidates

Things to test by changing one thing at a time and comparing F1.
The point is not to improve the model — it is to confirm whether each design
decision actually contributes or just adds complexity.

---

### A1 — JK Connections: Do They Actually Prevent Oversmoothing?

**Assumption:** JK connections prevent node embeddings from converging to similar
vectors at depth, by giving the model access to all phase outputs not just the last.

**Risk:** JK adds parameters and complexity. With 41K training examples, the
additional capacity may add noise rather than signal.

**Ablation:** train identical model without JK (replace with last-layer output only).
If F1 drops: JK is earning its cost. If F1 is equal or higher: JK is adding
parameters without benefit and should be simplified.

---

### A2 — Phase Separation: Do Isolated Phases Matter?

**Assumption:** processing CONTROL_FLOW edges in isolation (Phase 2) lets the
model develop execution-order semantics without structural noise from CALLS/READS/WRITES.

**Risk:** phases may not actually isolate signal. JK connection means all phases
influence the final embedding regardless of separation.

**Ablation:** train with all edge types processed together in a single phase
(no masking). Compare F1 to the three-phase model. If similar F1: phase
separation is architectural theatre. If lower F1: isolation is doing real work.

---

### A3 — Phase 2 Depth: Is 3 Layers Better Than 2?

**Assumption:** CONTROL_FLOW reasoning requires more depth than structural
reasoning — hence 3 layers in Phase 2 vs 2 in Phase 1.

**Risk:** extra depth in Phase 2 may cause oversmoothing within the CFG subgraph
despite self-loops being off.

**Ablation:** Phase 2 with 2 layers vs 3 layers. Also 4 layers. Map F1 vs depth.
Does the middle (3 layers) actually outperform both extremes?

---

### A4 — Visibility Encoding: Scalar vs Two-Binary

**Current design:** `{public: 0.0, external: 0.0, internal: 0.5, private: 1.0}`

**Problem:** public and external collapse to the same value. Public functions are
callable from inside AND outside; external functions are callable from outside
ONLY. This distinction matters for internal call chain analysis but is invisible
to the GNN with the current scalar.

**Alternative:** two binary features per node:
`is_externally_callable` and `is_internally_callable`

```
public:   [1, 1]
external: [1, 0]
internal: [0, 1]
private:  [0, 0]
```

**Ablation:** retrain with two-binary encoding. Compare F1 especially on
access-control vulnerability class where internal vs external matters most.

---

### A5 — LoRA Rank: Is 16 Enough for 10 Classes?

**Assumption:** 16 directions of adaptation from general code semantics to
Solidity vulnerability semantics is sufficient.

**Risk:** 10 vulnerability classes compete for 16 directions. Rare classes
(DoS: 7 samples, Timestamp: mislabeled) may get effectively zero adaptation
direction because high-sample classes dominate gradient updates.

**Ablation:** train with r=8, r=16, r=32, r=64. Plot per-class F1 vs rank.
If rare classes improve with higher rank while common classes stay flat: rank
is the constraint. If all classes move together: data is the constraint, not rank.

---

### A6 — LoRA Target Modules: Query+Value vs All Projections

**Current design:** LoRA injected into query and value projections only.
Key and output projections are frozen.

**Risk:** for Solidity-specific vulnerability patterns, the key projection
(how tokens present themselves) may also need adaptation. A token like
`msg.value` might need to present itself differently than it does in
general Python/Java code.

**Ablation:** add LoRA to key+output projections with same rank. Compare
parameter efficiency (more params, same data) vs potential signal gain.

---

### A7 — Pooling Strategy: Max+Mean vs Alternatives

**Current design:** max pooling and mean pooling concatenated → [2×256] → Linear → [128]

**Alternatives to test:**
- Max only → [256] → Linear → [128]
- Mean only → [256] → Linear → [128]
- Sum → [256] → Linear → [128]
- Attention pooling (learned query over function nodes)

**Question:** does concatenating max+mean actually give the classifier more
information than either alone? The Linear layer could learn to ignore one half.

**Ablation:** ablate each pooling variant, compare F1 and look at whether the
Linear layer weights collapse toward one side (indicating the other is unused).

---

### A8 — CFG Feature Inheritance: Function-Level vs Per-Statement Features

**Current design (BUG-C3 fix):** every CFG node inherits its parent FUNCTION's
features (visibility, payable, complexity, has_loop, external_call_count).

**Problem:** all CFG statements within the same function have identical feature
vectors (except type_id). Phase 2 attention cannot differentiate between "the
`.call()` statement" and "the balance-update statement" in the same function.
Intra-function ordering relies entirely on graph topology, not features.

**Alternative:** add per-statement binary features:
- `stmt_has_call`: this statement contains an external call
- `stmt_has_write`: this statement writes a state variable
- `stmt_has_read`: this statement reads a state variable
- `stmt_is_conditional`: this statement is a branch condition

This requires changes to both graph_schema.py (new features) and
graph_extractor.py (per-statement analysis via Slither IR).

**Ablation:** retrain with per-statement features. Expected improvement in
reentrancy detection specifically — the model can now see "call at position X,
write at position Y" rather than just topology.

---

### A9 — Transformer Eye: Is the CLS Path Adding Value?

**Two CodeBERT output paths:**
- Path A (fusion): all tokens → CrossAttentionFusion → ordering preserved
- Path B (transformer eye): window CLS → pooled → ordering discarded

**Risk:** Path B discards exactly the information (execution ordering) that
the GNN was extended to capture. Its value is pretrained global semantics —
but the fused eye (Path A) also carries those semantics, anchored to graph structure.

**Ablation:** remove Path B entirely. Train with GNN eye + fused eye only
(no standalone transformer eye). If F1 is equal or better: Path B is redundant
and the architecture simplifies. If F1 drops: Path B's global semantic summary
contributes something the fused eye cannot replicate alone.

---

### A10 — v8 Graph Extensions (Planned in ACTIVE_PLAN.md)

Three separate ablations when v8 extraction is ready:
- **v8-A only:** ICFG-Lite edges (CALL_ENTRY + RETURN_TO) — does cross-function CFG help?
- **v8-B only:** DEF_USE edges — does value flow tracking improve precision?
- **v8-AB:** both together — do they interact constructively or add noise?

Each ablation requires full re-extraction of ~41K contracts at the new schema version.

---

## 2. Design Risks

Assumptions baked into the architecture before training. Each is grounded in
theory or empirical results from other domains, but has not been validated
for this specific task and dataset.

---

### R1 — "Topology Alone Is Sufficient for Intra-Function Ordering"

**The assumption:** within Phase 2, CFG nodes of the same function have
identical feature vectors. The model must distinguish "call before write" from
"write before call" using CONTROL_FLOW topology alone, without any feature
signal about what kind of statement each node is.

**Why it might be wrong:** GAT attention computes edge weights from feature
similarity. If source and target have identical features, attention weights
are identical — effectively uniform aggregation. The topology signal exists
but the attention mechanism cannot amplify it for specific edge types.

**Evidence needed:** GradCAM analysis on confirmed reentrancy predictions.
Do CFG nodes receive high attribution? Or does the model rely only on the
FUNCTION-level `external_call_count` feature and ignore CFG topology?

---

### R2 — "Phase 3 Dominating Means Containment Is the Right Signal"

**Observation:** JK phase weights at epoch 11: Phase1=0.096, Phase2=0.33, Phase3=0.57.
Phase 3 (REVERSE_CONTAINS) carries 57% of the model's attention.

**The assumption:** this means the model has learned that bottom-up information
flow from CFG nodes to FUNCTION nodes is the most informative signal.

**Why it might be wrong:** the model could be using Phase 3 dominance as a
shortcut for contract size/complexity. Contracts with more CFG nodes (larger,
more complex) send more Phase 3 messages to their parent FUNCTION nodes. More
messages → more varied FUNCTION embeddings → easier classification. The model
might be learning "complex contracts have more vulnerabilities" rather than
reasoning about which specific patterns cause them.

**Evidence needed:** ablation removing Phase 3 entirely. If F1 stays similar,
Phase 3 is not contributing genuine signal. Also: compare JK weights on
contracts where the model is correct vs incorrect.

---

### R3 — "16 LoRA Adaptation Directions Are Sufficient"

**The assumption:** the mapping from general code understanding (Python/Java/Go)
to Solidity vulnerability detection can be expressed in 16 linearly independent
directions of change in the query and value projections.

**Why it might be wrong:** 10 different vulnerability classes, each requiring
different attention patterns, competing for 16 shared directions. Classes with
few training examples may receive effectively zero adaptation direction.
DoS has 7 training samples — its gradient contribution is noise-level.

**Evidence needed:** per-class F1 breakdown as a function of LoRA rank.
If DoS F1 improves significantly at r=64 while Reentrancy stays flat:
rank is the constraint for rare classes.

---

### R4 — "WRITES Edge Direction Is Correct for Vulnerability Detection"

**Current design:** WRITES edges go FUNCTION → STATE_VAR. Message passing
carries information from function to state variable. STATE_VAR learns about
its writers.

**The assumption:** state-variable-centric information flow is useful for
detecting access control vulnerabilities (who can write this variable?).
Function-centric knowledge (what variables does this function write?) is
already captured in node features (external_call_count, return_ignored).

**Why it might be wrong:** the combination of "which state variable" + "when
it is written relative to external calls" requires the FUNCTION to know about
its specific write targets, not just a count. Scalar `external_call_count`
doesn't say WHICH state variable is written. A WRITES edge in the wrong
direction means the FUNCTION never learns the identity of its write targets.

**Alternative:** add reverse WRITES edges (STATE_VAR → FUNCTION) in Phase 1,
or store them on disk as a new edge type in v8.

---

### R5 — "most_derived Contract Selection Is ~92% Accurate"

**The assumption:** selecting the contract with the deepest inheritance chain
correctly identifies the deployed contract in ~92% of cases.

**Why it might be wrong:** the 92% figure was measured on a sample, not the
full 41K dataset. Files with multiple independent inheritance hierarchies
(two unrelated contract families in one file) may have higher error rates
than the sample suggested.

**Evidence needed:** audit contract selection accuracy on 500 randomly sampled
training contracts, manually verify which contract the vulnerability label
actually describes, compute accuracy. If significantly below 92%: re-evaluate
the selection heuristic.

---

### R6 — "Ghost Graph Handling Is Correct (Return Zeros)"

**Current design (BUG-H2 fix):** contracts with no function nodes produce a
zero GNN embedding. The model receives all-zeros for the GNN eye.

**Why it might be wrong:** a zero vector is a specific signal — it's the origin
of the embedding space. The model might learn "GNN eye near zero = specific
vulnerability class" if that class happens to correlate with sparse contracts.
A learned "ghost" embedding (registered parameter) would be more expressive.

**Alternative:** add a learnable `ghost_embedding` parameter, use it instead
of zeros. Costs 256 parameters, removes the zero-vector ambiguity.

---

## 3. Architecture Limitations

Fundamental constraints of the current design that cannot be fixed by
retraining — they require architectural changes.

---

### L1 — Single-Contract Analysis: Cross-Deployment Blind Spot

**What it is:** the model analyzes one contract in isolation. External calls
to separately deployed contracts (different addresses) are represented only
as edge endpoints, not as analyzed subgraphs.

**What is missed:** vulnerabilities that require understanding the called
contract's behavior — oracle manipulation, flash loan attack surfaces,
cross-protocol reentrancy.

**Partial fix available (v8):** ICFG-Lite edges for cross-function flow
within a single contract's scope. Does not address cross-deployment.

**Full fix:** multi-contract workspace in the Agents module — analyze
each contract individually, then reason about interactions at the agent level.

---

### L2 — No Value Flow Tracking

**What it is:** the graph knows that function F reads state variable X
(READS edge). It does not know that the VALUE read from X flows into the
condition of a branch, which governs whether an external call happens.

**What is missed:** taint-style analysis — tracking that user-controlled
input reaches a sensitive operation. Integer overflow where a computed
value is used unsafely downstream.

**Fix:** DEF_USE edges (Extension B, v8/v9 proposal). Adds definition-to-use
edges tracking value flow across CFG statements.

---

### L3 — Intra-Function Statement Differentiation

**What it is:** all CFG statements within the same function have identical
feature vectors (BUG-C3 inherited features). Phase 2 cannot distinguish
"the `.call()` statement" from "the balance-update statement" via features.
Ordering signal comes only from graph topology.

**What is missed:** direct feature-level evidence that a specific statement
contains an external call or a state write. The model must infer this from
topology alone.

**Fix:** per-statement binary features (stmt_has_call, stmt_has_write) — see
Ablation A8. Requires schema version bump and re-extraction.

---

### L4 — REVERSE_CONTAINS Only (No Reverse READS/WRITES/CALLS)

**What it is:** Phase 3 adds reverse edges only for CONTAINS. Functions learn
from their CFG children. But functions do NOT learn from state variables they
write (no reverse WRITES), and called functions do NOT learn about their
callers (no reverse CALLS beyond what GATConv already does).

**What is missed:** "who calls me" information at function level, "what I write
to" information at function level — both require reverse edges not present.

**Fix:** add runtime-generated reverse READS, reverse WRITES, reverse CALLS
edges. Process them in a Phase 4. Or store them on disk as explicit edge types.

---

### L5 — Fixed NUM_CLASSES = 10 (ZKML Constraint)

**What it is:** the output layer is locked at 10 classes because the ZKML
proxy MLP circuit is compiled for exactly Linear(128→64→32→10). Changing
the number of vulnerability classes requires recompiling the ZK circuit.

**What is missed:** new vulnerability classes discovered after circuit
compilation cannot be added without a full ZKML re-design.

**Design note:** this is a deliberate constraint, not an oversight. Document
it here as a reminder that NUM_CLASSES is NOT a hyperparameter to tune.

---

## 4. Data Quality Issues

Problems in training data that teach the model wrong patterns.
Source: ACTIVE_BUGS.md and discoveries during learning sessions.

---

### D1 — Timestamp Labels 48.2% Mislabeled (BUG-H4)

**What it is:** Slither's Timestamp detector flags any use of `block.timestamp`
in a conditional, regardless of whether the timing can actually affect
security-sensitive behavior. Almost half the Timestamp training labels are
contracts where `block.timestamp` is used safely.

**Effect on training:** the model learns "contracts with block.timestamp in
conditionals" not "contracts where timing manipulation affects outcomes."
Timestamp F1 will be low regardless of model sophistication.

**Fix required:** manual re-labeling of Timestamp class, or a better Slither
heuristic. Active Learning approach: surface the highest-uncertainty Timestamp
predictions for manual human review. ~500 carefully selected examples
would provide more signal than all 48.2% wrong labels combined.

---

### D2 — DoS Has 7 Pure Training Samples

**What it is:** the DoS via unbounded loops vulnerability class has effectively
no training data. 7 samples cannot train any model component — LoRA, GNN,
or classifier — to detect this pattern.

**Current handling:** `dos_loss_weight=0.0` — DoS is excluded from the loss
entirely. The model never tries to learn DoS.

**Effect:** SENTINEL cannot detect DoS. Confirmed by near-zero DoS F1.
ZKML constraint prevents removing the class from the output.

**Fix required:** data collection. Identify Solidity contracts with confirmed
DoS vulnerabilities, add them to the training set. No architectural change
needed — just more labeled data for this class.

---

### D3 — 14% Reentrancy Contracts Have No External Calls (BUG-H5)

**What it is:** 14% of contracts labeled as reentrancy-vulnerable have zero
external calls in their extracted graph. Reentrancy requires an external call
by definition — a contract with no external calls cannot have reentrancy.

**Possible causes:**
- Contract selection chose the wrong contract (most_derived fix may not be 100%)
- Slither did not detect the external call (Slither limitation)
- Labels are wrong for these contracts

**Effect:** the model receives training examples that say "this structure =
reentrancy" when the structure has no mechanism for reentrancy. Adds noise
to the reentrancy embedding.

**Fix required:** audit these 14% manually. Separate cause from effect before
deciding whether to drop or relabel.

---

### D4 — EMITS/INHERITS Edges Never Appear in Training

**What it is:** both edge types are defined in graph_schema.py with reserved
type numbers, but graph_extractor.py has no code to extract them. Zero edges
of these types exist in any training graph.

**Effect on model:** `nn.Embedding` rows for these edge types are randomly
initialized and never updated. If extraction code is ever added without
retraining, these edge types will inject random noise into message passing.

**Fix required:** either implement extraction (add to extractor, bump schema
version, re-extract, retrain) OR remove from schema if not planned. Do not
leave named-but-empty slots indefinitely.

---

### D5 — 8.5% Graphs Have Empty contract_path (BUG-M7)

**What it is:** 8.5% of saved graph files have no contract_path metadata.
The source Solidity file cannot be traced for these graphs.

**Effect:** cannot audit these graphs manually, cannot verify contract
selection was correct, cannot re-extract if the schema changes.

**Fix required:** re-run extraction on the original dataset, ensure contract_path
is written. If source files are unavailable: flag these graphs and consider
dropping them from training.

---

## 5. Potential Improvements

Concrete changes that would likely improve the system, ordered by estimated
impact-to-effort ratio.

---

### I1 — Active Learning for Label Quality (High impact, Medium effort)

**What it is:** instead of random sampling for manual review, use model
uncertainty to surface the most valuable contracts to label.

**Why now:** Timestamp labels are 48.2% wrong. Manual review of all Timestamp
contracts is infeasible. But reviewing the 500 highest-uncertainty Timestamp
predictions would fix the labels that matter most for model improvement.

**How:** after each training epoch, collect contracts where:
- Timestamp probability is in the ambiguous range (0.35–0.65)
- GNN eye and Transformer eye strongly disagree on Timestamp score
These are the labels the model is most uncertain about — the ones human
review would provide the highest information gain.

---

### I2 — Per-Statement CFG Features (Medium impact, High effort)

**What it is:** instead of inheriting function features, give each CFG node
binary indicators of what that specific statement does.

**Why:** solves Limitation L3 — Phase 2 can distinguish the `.call()` statement
from the balance-update statement using features, not only topology.

**What it requires:** changes to graph_schema.py (new feature columns),
graph_extractor.py (per-statement Slither IR analysis), schema version bump,
full re-extraction, full retraining.

---

### I3 — Learnable Ghost Embedding (Low impact, Low effort)

**What it is:** replace the zero-vector fallback for contracts with no
function nodes (BUG-H2) with a learned `nn.Parameter` vector.

**Why:** zero vector is a specific point in embedding space that the model
may learn to associate with specific classes. A learned ghost embedding
is less ambiguous.

**What it requires:** one-line change in sentinel_model.py, retraining.
No schema changes, no re-extraction.

---

### I4 — Reverse WRITES/READS Edges in Phase 4 (Medium impact, Medium effort)

**What it is:** add a Phase 4 that processes reverse WRITES and READS edges
(STATE_VAR → FUNCTION direction), giving FUNCTION nodes direct access to
which specific state variables they modify.

**Why:** solves part of Limitation L4 and Design Risk R4. Functions learn
"I write to the balances variable" rather than inferring it from scalar counts.

**What it requires:** new edge types in schema (v8+), extraction code,
new Phase 4 in gnn_encoder.py, new LayerNorm entry, JK update to include
Phase 4 output. Significant but contained change.

---

### I5 — Assert Guards Audit (Low impact, Low effort)

**What it is:** graph_schema.py has assert statements at import time that
check consistency invariants. These have never been systematically documented
or reviewed for completeness.

**Why:** a missing assert means a schema inconsistency can silently propagate
into training. An over-broad assert means a valid schema extension fails to
import.

**What it requires:** read graph_schema.py assertions, document what each
checks, identify any gaps.

---

## Progress

| Item | Status |
|------|--------|
| A1 JK ablation | ☐ Not run |
| A2 Phase separation ablation | ☐ Not run |
| A3 Phase 2 depth ablation | ☐ Not run |
| A4 Visibility encoding ablation | ☐ Not run |
| A5 LoRA rank ablation | ☐ Not run |
| A6 LoRA target modules ablation | ☐ Not run |
| A7 Pooling strategy ablation | ☐ Not run |
| A8 CFG feature inheritance ablation | ☐ Not run |
| A9 Transformer eye ablation | ☐ Not run |
| A10 v8 extension ablations | ☐ Blocked on v8 extraction |
| D1 Timestamp relabeling | ☐ Open |
| D2 DoS data collection | ☐ Open |
| D3 Reentrancy no-external-calls audit | ☐ Open |
| D4 EMITS/INHERITS decision | ☐ Open |
| D5 Empty contract_path fix | ☐ Open |
| I1 Active learning setup | ☐ Not started |
| I2 Per-statement CFG features | ☐ Blocked on schema decision |
| I3 Learnable ghost embedding | ☐ Ready to implement |
| I4 Reverse WRITES/READS phase | ☐ Requires schema change |
| I5 Assert guards audit | ☐ Not started |
