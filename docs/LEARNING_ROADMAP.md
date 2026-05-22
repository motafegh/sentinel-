# SENTINEL ML — Learning Roadmap

Active learning plan for understanding the ML module from first principles.
Each item represents a concept, design decision, or file section that must be
understood deeply — not just what it does, but why it exists, what assumption
it makes, and what would break if it were wrong.

**Status markers:**
- ✅ Covered and understood
- ☐ Not yet covered

---

## Teaching Approach

**Code:** every concept from a source file is shown with the relevant code
snippet first, then explained, then questioned. You do not need to keep files
open — relevant code is brought here.

**Questions:** answer from your understanding of what was just explained.
Do not look up the code to answer — the questions test understanding, not
memory or search skill.

**If you don't get the question itself:** ask for a hint.
**If you partially don't know:** ask for a hint.
**If you have no idea at all:** say so and the full explanation follows.

---

## How To Use This

Work through each section in order. Code is shown alongside each concept.
After an item is covered and understood, it gets checked.

---

## Phase 1 — Graph Schema (`ml/src/preprocessing/graph_schema.py`)

The single source of truth for all graph constants. Everything downstream
depends on decisions made here. Understand this file completely before
anything else.

### Why Graph Representation At All

- ✅ The locality problem — why text distance is semantically meaningless in code
- ✅ Cross-function visibility — why CodeBERT alone cannot see reentrancy patterns
- ✅ Graph vs raw text vs AST vs CFG vs PDG — the alternatives landscape and tradeoffs
- ✅ Why SENTINEL uses a purpose-built hybrid graph instead of a standard representation

### Node Types

- ✅ Declaration-level nodes (CONTRACT, STATE_VAR, FUNCTION, MODIFIER, EVENT) — the big picture layer
- ✅ CFG-level nodes (CFG_ENTRY, CFG_EXIT, CFG_STATEMENT, CFG_IF, CFG_LOOP) — the execution layer
- ✅ Why both layers are needed simultaneously — what reentrancy requires from each
- ✅ The CONTAINS edge as the connector between the two layers
- ✅ Design tension: 72% of nodes are CFG nodes — the pooling imbalance problem
- ✅ Three alternatives to handle the tension: separate GNNs, hierarchical GNN, HGT
- ✅ Why SENTINEL chose unified graph with pooling surgery over theoretical cleanliness

### Node Features (11-vector per node)

- ✅ `type_id` [0] — encoding node type as a number, why not one-hot
- ✅ `visibility` [1] — public/external=0.0, internal=0.5, private=1.0; encoding internality not attack surface
- ✅ Why public and external map to the same value — attack surface equivalence
- ✅ What the scalar encoding loses vs a two-binary encoding (is_externally_callable, is_internally_callable)
- ✅ `uses_block_globals` [2] — block.timestamp/number/difficulty as direct Timestamp vuln signal
- ✅ `view` [3] — declared no state changes; false views as suspicious
- ✅ `payable` [4] — direct attack surface indicator
- ✅ `complexity` [5] — cyclomatic complexity, why log-normalized
- ✅ `loc` [6] — lines of code, why log-normalized
- ✅ Heavy-tailed distributions — why linear normalization creates gradient dominance
- ✅ Multiplicative relationships — why log captures the right semantics
- ✅ `return_ignored` [7] — unchecked call return value; the silent failure vulnerability pattern
- ✅ `call_target_typed` [8] — typed interface vs raw address; compile-time guarantee vs arbitrary execution
- ✅ `has_loop` [9] — DoS gas exhaustion signal
- ✅ `external_call_count` [10] — reentrancy surface area as a scalar
- ✅ Three feature categories: structural, scale, behavioral
- ✅ Why behavioral features are direct signals vs structural features as proxies

### Edge Types

- ✅ `CONTAINS` (0) — parent→child relationships; CONTRACT→FUNCTION, FUNCTION→CFG_ENTRY
- ✅ `CALLS` (1) — function A calls function B; cross-function reachability
- ✅ `READS` (2) — function reads state variable; data dependency
- ✅ `WRITES` (3) — function writes state variable; direction problem + compensation via features
- ✅ `MODIFIES` (4) — function applies modifier; access control chain
- ✅ `HAS_MODIFIER` (5) — modifier definition linkage
- ✅ `CONTROL_FLOW` (6) — CFG statement A executes before B; execution ordering
- ✅ `REVERSE_CONTAINS` (7) — runtime-only: disk=contract semantics, runtime=GNN architecture
- ✅ Why these 8 and not more — each edge describes a real semantic relationship in Solidity
- ✅ EMITS and INHERITS — placeholder slots; untrained embeddings = random noise if ever used
- ✅ What the current edge set cannot express — cross-function CFG, value flow (v8/v9)
- ✅ Schema versioning: v7→v8→v9 coupled system (graph files + embedding table + weights)
- ☐ Assert guards at import time — what consistency invariants they enforce and why at import

---

## Phase 2 — Graph Extractor (`ml/src/preprocessing/graph_extractor.py`)

The algorithm that converts raw Solidity source into the graph schema defined above.
Every bug fixed here represents a real data quality problem that corrupted training.

### Contract Selection

- ✅ BUG-6: why `most_funcs` was wrong — 47.4% picked base contract, not the deployed derived one
- ✅ The impact on training: identical corruption in train+val = undetectable from loss curve
- ✅ What "most derived" means in Solidity inheritance — deepest inheritance chain = final deployed contract
- ✅ Why ~92% accuracy and not 100% — flat files with multiple unrelated inheritance trees are ambiguous
- ✅ Single-contract analysis scope: inherited parents ARE included (Slither merges them); unrelated siblings correctly excluded; cross-deployment calls are the real blind spot

### Graph Construction Algorithm

- ✅ Node insertion order: CONTRACT → parent CONTRACTs → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs
- ✅ Why insertion order matters for graph_idx assignment — position = permanent identity
- ✅ `graph_idx = len(x_list)` vs `len(node_index_map)` — parent CONTRACT node inserted in x_list but not always in node_index_map → index divergence → silent edge corruption
- ✅ Slither integration — three layers: contract/function objects → CFG nodes → IR operations; extractor translates hierarchical objects into flat tensors; single shared module prevents silent feature divergence between training and inference

### CFG Construction

- ✅ Two-pass CFG building — forward edges need target indices that don't exist yet in one pass
- ✅ Pass 1: assign node indices; Pass 2: build edges — why this order
- ✅ CFG node sorting: source_line order for determinism across machines and Slither versions
- ✅ BUG-C3: CFG nodes had all-zero features except type_id — Phase 2 attention was uniform (useless)
- ✅ BUG-C3: inheriting parent FUNCTION features (visibility, payable, complexity, has_loop) gives inter-function differentiation
- ✅ The implication: intra-function CFG statements still identical in feature space — ordering from topology only
- ✅ cfg_node_map scoped per-function vs global — local works now; v8 ICFG needs global so cross-function CFG entry indices are accessible; nested per-function maps would also work but require function-context lookup

### Feature Computation

- ✅ `_compute_return_ignored` — walks IR ops, checks lvalue is None or never read; BUG-9 added Send alongside LowLevelCall/HighLevelCall
- ✅ `_compute_uses_block_globals` — walks IR ops, finds SolidityVariableComposed reads matching block.* names; uses type(rv).__name__ defensively
- ☐ BUG-1: loc log-normalization fix — was raw line count per CFG node, violated [0,1] range
- ☐ BUG-2: complexity log-normalization fix — raw CFG block count could be 100+, dominated dot products
- ☐ Feature range validation at extraction time — why [-1, 1], what the sentinel value -1.0 means

---

## Phase 3 — GNN Encoder (`ml/src/models/gnn_encoder.py`)

The graph neural network that converts the extracted graph into a 128-dimensional
vulnerability embedding. Every architectural decision here is a hypothesis.

### Message Passing Fundamentals

- ✅ The actual mathematical operation — one layer = one hop; aggregate(transform(neighbors))
- ✅ Simultaneous update vs sequential traversal — all nodes read current embeddings, write at once
- ✅ Aggregation functions: sum, mean, max — what each preserves; max catches rare extremes mean washes out
- ✅ Softmax normalization — sums to 1; dense neighborhoods dilute individual attention
- ✅ Phase 2 two compounding problems — identical features + dense neighborhoods → uniform attention
- ☐ The "maybe" problem: capacity vs guarantee — why architecture creates conditions not certainty
- ☐ What the model might actually be learning vs what we designed it to learn

### GAT — Graph Attention Networks

- ✅ Why GAT over GCN — GCN uses fixed degree-based weights; GAT uses learned feature-based weights
- ✅ Attention weight computation — e_ij = LeakyReLU(a · concat(W·h_i, W·h_j, W_e·edge_attr)); softmax → α_ij
- ☐ Multi-head attention in GAT — 8 heads × 32 dims = 256 hidden dim
- ☐ Why multiple heads — what each head can specialize to detect
- ☐ What attention weights do NOT tell you — the faithfulness problem

### Three-Phase Architecture

- ☐ Phase 1: 2 layers, structural edges (types 0–5), self-loops ON
- ☐ Why self-loops are ON in Phase 1 — what it means for a node to include itself
- ☐ Phase 2: 3 layers, CONTROL_FLOW only (type 6), self-loops OFF — why critical
- ☐ Why no self-loops in Phase 2 — what self-loops would destroy for CFG learning
- ☐ Why Phase 2 has 3 layers instead of 2 — what requires the extra depth
- ☐ Phase 3: 2 layers, REVERSE_CONTAINS (type 7), self-loops OFF
- ☐ Why REVERSE_CONTAINS runs last — what information flows bottom-up
- ☐ Why each phase is isolated — what bleeds together without isolation
- ☐ Phase separation as computation-level separation in a unified graph

### JK — Jumping Knowledge Connections

- ☐ Oversmoothing — what it is, why it happens at depth, why it destroys node distinctiveness
- ☐ What oversmoothing looks like in practice — all nodes converging to the same embedding
- ☐ JK design — learned attention over all 3 phase outputs, not just final layer
- ☐ `_JKAttention`: Linear(channels, 1), softmax over phases
- ☐ `register_buffer("last_weights")` — why JK weights are logged as a diagnostic
- ☐ JK weights as signal: Phase3=0.57 dominating — what it tells you and what it doesn't
- ☐ The _live list (without .detach()) vs _intermediates dict (with .detach()) — gradient flow reason

### Layer Normalization

- ☐ What LayerNorm does — normalizing across feature dimensions
- ☐ Per-phase LayerNorm — why between phases, not once at the end
- ☐ What happens without it — Phase 1 magnitude dominating JK softmax
- ☐ The interaction between LayerNorm and JK — why both are needed together

### Pooling

- ☐ Max pooling — what information it preserves (extremes, rare signals)
- ☐ Mean pooling — what information it preserves (average character)
- ☐ Why both — what each misses alone, what concatenation gives
- ☐ Why function-level nodes only — revisited with the actual code
- ☐ `_FUNC_TYPE_IDS` frozenset — FUNCTION, MODIFIER, FALLBACK, RECEIVE, CONSTRUCTOR
- ☐ Ghost graph fix (BUG-H2) — contracts with no function nodes, why zero not fallback

### Architecture Parameters

- ☐ Hidden dim 256 (was 128) — what doubled capacity enables vs costs
- ☐ Edge embedding dim 64 — why embed edge types as learned vectors
- ☐ `nn.Embedding` for edge types — what this means vs one-hot encoding
- ☐ ~2.4M parameters vs original ~91K — what changed and whether it's justified

---

## Phase 4 — Transformer Encoder (`ml/src/models/transformer_encoder.py`)

### Covered

- ✅ Why CodeBERT — pretrained semantics from 6M+ code files
- ✅ Domain gap — CodeBERT trained on Python/Java/Go, not Solidity
- ✅ Catastrophic forgetting — the gradient overwriting mechanism
- ✅ LoRA — freeze 125M, train 590K; requires_grad=False as the physical barrier
- ✅ Why query+value projections specifically — redirect what model looks for and extracts
- ✅ Parameter efficiency ratio — 590K trainable / 44K contracts vs 125M / 44K
- ✅ LoRA rank-16 constraint — 16 directions of adaptation, class competition problem
- ✅ Multi-window design — contracts avg 1,737 tokens, 82% need multiple 512-token windows
- ✅ WindowAttentionPooler — CLS per window → learned attention → [768]
- ✅ Two CodeBERT output paths: all tokens (fusion) vs CLS (transformer eye)
- ✅ Transformer eye value: global semantic fingerprint, not ordering

### Still Needed

- ☐ Flash Attention 2 vs SDPA — what they are, why the fallback exists
- ☐ bfloat16 vs float32 — numerical precision tradeoff in CodeBERT
- ☐ alpha/r scaling (32/16 = 2.0) — what it controls at initialization and during training
- ☐ B is zero at init, A is random — why this matters for stable LoRA training start

---

## Phase 5 — Fusion Layer (`ml/src/models/fusion_layer.py`)

- ☐ Cross-attention mechanism — how it differs from self-attention
- ☐ Node-to-token direction: GNN nodes as queries, CodeBERT tokens as keys/values
- ☐ Token-to-node direction: CodeBERT tokens as queries, GNN nodes as keys/values
- ☐ What each direction captures — anchoring semantics to structure vs context
- ☐ Why bidirectional — what one direction alone cannot express
- ☐ Why cross-attention instead of concatenation or addition — the information bottleneck argument
- ☐ `need_weights=False` on both MHA calls — training speed vs interpretability tradeoff
- ☐ Output [B, 128] — how two different-sized representations collapse to one vector
- ☐ The fused eye as canary metric — why fused loss > individual eye losses is a warning sign
- ☐ Why CFG-enriched GNN queries changed what the fusion layer could find

---

## Phase 6 — SENTINEL Model (`ml/src/models/sentinel_model.py`)

- ☐ Three-eye architecture: GNN eye + Transformer eye + Fused eye → [384] → classifier
- ☐ GNN eye projection: Linear(2×256, 128) — why 2×256 (max+mean concatenation from pooling)
- ☐ Transformer eye projection: Linear(768, 128) — the dimension reduction
- ☐ Eye concatenation [B, 384] → classifier — why concatenate not sum
- ☐ Classifier: Linear(384→192) → ReLU → Dropout → Linear(192→10)
- ☐ Why 10 outputs — one per vulnerability class, why sigmoid not softmax
- ☐ Multi-label classification vs multi-class — what the difference means for training
- ☐ Auxiliary heads (aux_gnn, aux_transformer, aux_fused) — what they are and why
- ☐ `return_aux=False` at inference — why auxiliary heads are training-only
- ☐ `_FUNC_IDS_CPU` prebuilt tensor — BUG-L1 performance fix, what was wrong before

---

## Phase 7 — Loss Functions (`ml/src/training/losses.py`)

- ☐ BCE (Binary Cross-Entropy) baseline — why inadequate for 85% label=0 matrices
- ☐ The easy negative problem — why most gradient signal comes from non-vulnerable cases
- ☐ Focal Loss — gamma parameter, how it downweights easy examples
- ☐ Why Focal Loss is still symmetric — same gamma for positives and negatives
- ☐ AsymmetricLoss (ASL) — why asymmetry is the right design for this problem
- ☐ `gamma_neg=2.0` vs `gamma_pos=1.0` — what different gammas achieve per class
- ☐ `clip=0.01` — zeroing easy negatives below threshold, the hard boundary effect
- ☐ `prob_neg = (prob - self.clip).clamp(min=0.0)` — why clamp not soft threshold
- ☐ BF16 guard — why `.float()` cast is needed at the start of forward()
- ☐ Label smoothing — what it does to the target distribution
- ☐ Per-class label smoothing — why Reentrancy=0.14, Timestamp=0.05, DoS=0.18 are different

---

## Phase 8 — Trainer (`ml/src/training/trainer.py`)

- ☐ AdamW optimizer — why AdamW not Adam; weight decay as implicit regularization
- ☐ Parameter groups — why different learning rates for different parts
- ☐ `lora_lr_multiplier=0.3` — why LoRA trains slower than the rest
- ☐ `fusion_lr_multiplier=0.5` — why fusion layer gets its own rate
- ☐ Weighted sampler — 3× weight on any-vulnerable contracts; what this does to batch composition
- ☐ `eval_threshold=0.35` vs naive 0.5 — why 0.5 is wrong for imbalanced multi-label
- ☐ Threshold calibration — how to find the right threshold and what F1 measures
- ☐ `pos_weight_min_samples=3000` — the Reentrancy 2.82× over-amplification fix
- ☐ `dos_loss_weight=0.0` — why DoS is excluded from loss, and the cost of that decision
- ☐ Patience=30 and early stopping — what patience measures and what the trap is
- ☐ Auxiliary loss weighting — how aux_gnn, aux_transformer, aux_fused contribute to total loss
- ☐ MLflow tracking — what gets logged and how to use it for post-hoc debugging

---

## Cross-Cutting Concepts

Concepts that appear across multiple files. Track separately because they are
the transferable skills — the things that apply beyond this project.

### Design and Architecture

- ✅ Inductive bias — assumptions baked into architecture before training
- ✅ Design assumptions vs guarantees — the "maybe" problem
- ✅ Ablation studies — changing one thing at a time to test assumptions
- ✅ Signal dilution through mixing — what happens when order-aware and order-blind signals combine
- ✅ Representation mixing — how query shapes determine what cross-attention finds

### Training Dynamics

- ✅ Shortcut learning — model finding valid-but-wrong correlates
- ✅ Catastrophic forgetting — gradient overwriting of pretrained knowledge
- ✅ Label leakage / target leakage — answer given as input feature
- ✅ Data contamination — entity-level vs row-level deduplication
- ✅ Heavy-tailed distributions and their effect on gradient signal
- ☐ Oversmoothing — mechanics in depth beyond the JK motivation
- ☐ Gradient flow — how requires_grad=False stops backward computation
- ☐ Class imbalance — the full landscape of solutions (resampling, reweighting, loss design)
- ☐ Evaluation metrics — precision, recall, F1 for multi-label; why accuracy is useless here
- ☐ Threshold calibration — why the optimal threshold is not 0.5 for imbalanced problems

### Architecture Patterns

- ✅ Transfer learning — pretrain then adapt, why it works, data efficiency argument
- ✅ Low-rank adaptation (LoRA) — the parameter efficiency argument
- ☐ Attention mechanism from first principles — dot product attention formula
- ☐ Multi-head attention — why multiple heads, what each head can specialize for
- ☐ Residual connections — why skip connections prevent gradient vanishing
- ☐ Hierarchical GNN — the two-level alternative to unified graph
- ☐ HGT (Heterogeneous Graph Transformer) — different weights per edge type

### Interpretability

- ☐ GradCAM on graphs — gradient of output w.r.t. node features as importance scores
- ☐ Three explanation families: gradient-based, perturbation-based, attention-based
- ☐ The faithfulness problem — why explanations ≠ understanding
- ☐ JK phase weights as diagnostic — what high Phase 3 weight tells and doesn't tell

---

## Phase 9 — Graph Extensions (v8/v9 Proposal)

Cover after all current files are understood. These are the next design decisions.

- ☐ Extension A: ICFG-Lite — CALL_ENTRY(8) and RETURN_TO(9) edges crossing function boundaries
- ☐ Why global_cfg_node_map is needed for ICFG (vs per-function map in v7)
- ☐ Extension B: DEF_USE(10) — value flow from definition site to use sites
- ☐ Three DEF_USE categories: call return values, arithmetic results, state variable reads
- ☐ Extension C: Control-Dep(11) — direct edge from CHECK to governed statements
- ☐ Schema evolution: v7(8) → v8(11) → v9(12) edge types
- ☐ Risk analysis: graph explosion, recursive cycles, Phase 2 heterogeneity
- ☐ Ablation plan: v8-A, v8-B, v8-AB — how to isolate which extension actually helps

---

## Progress Summary

| Phase | File | Status |
|-------|------|--------|
| 1 | `graph_schema.py` — node features | ✅ Complete |
| 1 | `graph_schema.py` — edge types | ✅ Complete (assert guards remain) |
| 2 | `graph_extractor.py` | 🔄 Nearly complete — 3 feature computation items remain |
| 3 | `gnn_encoder.py` | 🔄 In progress — message passing fundamentals done, mid-GAT |
| 4 | `transformer_encoder.py` | ✅ Mostly complete (4 items remain) |
| 5 | `fusion_layer.py` | ☐ Not started |
| 6 | `sentinel_model.py` | ☐ Not started |
| 7 | `losses.py` | ☐ Not started |
| 8 | `trainer.py` | ☐ Not started |
| 9 | v8/v9 extensions | ☐ Not started |

**Next up:** Phase 2 — finish extractor (cfg_node_map scope, Slither integration, feature computation). Then Phase 3 GNN encoder.
