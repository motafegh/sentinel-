# SENTINEL ML — Learning Roadmap

Active learning plan for understanding the ML module from first principles.
Each item represents a concept, design decision, or file section that must be
understood deeply — not just what it does, but why it exists, what assumption
it makes, and what would break if it were wrong.

**Status markers:**
- ✅ Covered and understood
- ☐ Not yet covered
- ⚠ Covered but needs reinforcement (recall session flagged)

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

**Experimental grounding:** wherever a design decision has been tested by an
actual ablation run (v7, v8-AB, PLAN-3A), the results are brought in alongside
the theory. Knowing what happened in practice is as important as knowing why
the architecture was designed a certain way.

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

### Typed Aliases and Prefix Constants (Phase 3.6 additions)

- ☐ `NodeType` IntEnum — typed aliases for NODE_TYPES integer IDs; derived from NODE_TYPES at module load, cannot drift; why IntEnum over plain int constants
- ☐ `STRUCTURAL_PREFIX_TYPES` frozenset — 5 declaration node types eligible for GNN prefix injection; why CFG nodes excluded (Phase 3 already aggregates them into FUNCTION nodes)
- ☐ PRE-4 audit result — mean=20.3 declaration nodes per graph, P50=16, P95=47 → K=48 covers 95.5%; why K=48 not K=32 or K=64
- ☐ `_PREFIX_NODE_PRIORITY` ordering — CONSTRUCTOR > FALLBACK > RECEIVE > MODIFIER > FUNCTION; why entry-point nodes get highest priority

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

- ☐ Node insertion order: CONTRACT → parent CONTRACTs → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs — why this order is fixed
- ☐ Why insertion order matters for graph_idx assignment — position = permanent identity; changing order silently corrupts all edges
- ☐ `x_list` (global, all nodes) vs `node_index_map` (local, CFG nodes only) — why `graph_idx = len(x_list)` not `len(node_index_map)`; what bug the wrong choice creates
- ☐ Slither integration — three abstraction layers: contract/function objects → CFG nodes → IR operations; single shared module prevents silent feature divergence

### CFG Construction

- ☐ Two-pass CFG building — why forward edges cannot be built in one pass; what forward-reference problem arises
- ☐ Pass 1: assign indices + build feature vectors; Pass 2: build CONTROL_FLOW edges — why this exact split
- ☐ CFG node sorting: source_line order for determinism across Slither versions and machines
- ☐ BUG-C3: CFG nodes had all-zero features except type_id — Phase 2 attention was completely uniform; what was useless about it
- ☐ BUG-C3 fix: inheriting parent FUNCTION features (visibility, payable, complexity, has_loop) — what this gives the model that zeros don't
- ☐ The remaining implication: intra-function CFG statements still identical in feature space — ordering comes from topology only
- ☐ `cfg_node_map` scoped per-function vs global — why local works for CONTROL_FLOW; why v8 ICFG needs global

### Feature Computation

- ☐ `_compute_return_ignored` — why `op.lvalue is never None`; what the actual check is; BUG-9 added Send
- ☐ `_compute_uses_block_globals` — why block.timestamp creates no READS edge; the SolidityVariableComposed distinction
- ☐ BUG-1: loc log-normalization fix — was raw line count per CFG node, violated [0,1] range
- ☐ BUG-2: complexity log-normalization fix — raw CFG block count could be 100+, dominated dot products
- ☐ Feature range validation at extraction time — why [-1, 1], what the OOR warning does

### v8 Edge Construction (ICFG + DEF_USE)

- ☐ `_add_icfg_edges` — func_entry_map, func_terminal_map, func_cfg_maps — what each stores and why all three are needed
- ☐ CALL_ENTRY: calling CFG node → callee ENTRYPOINT; RETURN_TO: callee terminals → call-site successors
- ☐ `_add_def_use_edges` — two-pass: def_map build → use scan; why only LocalVariable tracked (not StateVariable, not TemporaryVariable)
- ☐ Deduplication of (def_node, use_node) pairs — why multi-IR reads on the same node produce one edge

---

## Phase 3 — GNN Encoder (`ml/src/models/gnn_encoder.py`)

The graph neural network that converts the extracted graph into a 256-dimensional
node embedding (then pooled to 128 for the classifier). Every architectural
decision here is a hypothesis. Currently at v8 — v7 added JK+LayerNorm,
v8 added ICFG-Lite (CALL_ENTRY, RETURN_TO) and DEF_USE edges to Phase 2.

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
- ✅ Multi-head attention in GAT — 8 heads × 32 dims = 256 hidden dim; each head learns independently
- ✅ Why multiple heads — different heads specialize for different edge type relationships
- ✅ Why Phase 2 uses heads=1 — all Phase 2 edge types express the same semantic (execution ordering)
- ✅ v8 heads=1 open question — 3→4 edge types in Phase 2; same-purpose argument still holds but weaker
- ✅ The faithfulness problem — softmax always sums to 1 regardless of actual importance; high attention ≠ causal importance; local per-layer weights don't reflect 7-layer global prediction influence

### Three-Phase Architecture (v8)

- ✅ Phase 1: 2 layers, structural edges (types 0–5), self-loops ON
- ✅ Why self-loops are ON in Phase 1 — node participates in its own attention; own features influence neighbor weighting
- ✅ Self-loop vs residual — self-loop is inside attention computation; residual is added after
- ✅ Phase 2: 3 layers, CONTROL_FLOW(6) + CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10), self-loops OFF
- ✅ Why no self-loops in Phase 2 — self-loops inject undirected Phase 1 context into directed CFG flow; dilutes ordering signal at every hop
- ✅ Why Phase 2 has 3 layers — CEI hop count: ENTRY→CHECK→CALL→TMP→WRITE needs 3 hops
- ✅ `phase2_edge_types` parameter — runtime ablation switch; None=all 4, [6,8,9]=ICFG-only, [6,10]=DFG-only
- ✅ Phase 3: 2 layers, REVERSE_CONTAINS (type 7), self-loops OFF
- ✅ Why REVERSE_CONTAINS runs last — CFG nodes must be enriched by Phase 2 before lifting to FUNCTION
- ✅ `.flip(0)` to reverse edges at runtime — no re-extraction needed; type-7 embedding distinct from type-5
- ✅ Residual connections — gradient vanishing problem; identity path ensures gradient ≥ 1 at every layer
- ✅ Three phases as a pipeline — Phase 1 output feeds Phase 2; Phase 2 output feeds Phase 3
- ✅ Why each phase uses isolated edge sets — mixing edge types in one pass prevents per-relationship specialization
- ✅ Docstring inconsistency — header says 3 Phase 2 edge types; code (line 413–418) processes 4 (DEF_USE included); PARAMETERS section still says "v7 defaults"

### JK — Jumping Knowledge Connections

- ✅ Oversmoothing — repeated averaging at depth; node embeddings converge to global average; loses per-node distinctiveness
- ✅ What oversmoothing looks like in practice — FUNCTION and CFG_ENTRY end up with near-identical embeddings after 7 layers
- ✅ JK design — learned attention over all 3 phase outputs; early phases = sharp local; late phases = smooth global
- ✅ `_JKAttention`: Linear(channels, 1) scores each phase; softmax normalizes; weighted sum combines
- ✅ `register_buffer("last_weights")` — buffers survive .to(device), state_dict save/load, DDP; contrast with plain attribute
- ✅ `last_weight_stds` buffer — per-phase std; std < 0.05 = global constant (collapse); std > 0.10 = genuine per-node routing
- ✅ `last_node_weights` — stored in eval mode only; [N, K] size varies per batch (can't be buffer); zero cost in training
- ✅ eval vs training mode — `if not self.training` gate; per-node weights only needed by diagnostic scripts in eval
- ✅ The _live list (no .detach()) vs _intermediates dict (.detach().clone()) — gradients must flow through _live into all conv layers; _intermediates are disconnected copies for inspection only
- ✅ JK collapse confirmed — v7 and v8 both: 99.99% of nodes Phase 3 dominant; global constant not per-node routing
- ✅ JK trajectory during training — Phase 2 starts high (new edges novel), collapses to Phase 3 dominance by training end
- ✅ PLAN-3D fix — switch jk_mode from "attention" to "cat"; concatenation cannot collapse; downstream linear decides per class
- ☐ `Linear(256, 1)` shared across all node types — what this means for per-node specialization limits

### Layer Normalization

- ☐ What LayerNorm does — normalizing across feature dimensions per node
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
- ☐ Edge embedding dim 64 — why embed edge types as learned vectors not one-hot
- ☐ `nn.Embedding` for edge types — lookup table; row per type; learned during training
- ☐ ~2.4M parameters vs original ~91K — what changed and whether it's justified

---

## Phase 4 — Transformer Encoder (`ml/src/models/transformer_encoder.py`)

**Current model: GraphCodeBERT** (`microsoft/graphcodebert-base`), not CodeBERT. Pretrained on code + data-flow graphs. All LoRA mechanics unchanged.

### Covered

- ✅ Why pretrained transformer — pretrained semantics from 6M+ code files
- ✅ Domain gap — pretrained on Python/Java/Go, not Solidity
- ✅ Catastrophic forgetting — the gradient overwriting mechanism
- ✅ LoRA — freeze 125M, train 590K; requires_grad=False as the physical barrier
- ✅ Why query+value projections specifically — redirect what model looks for and extracts
- ✅ Parameter efficiency ratio — 590K trainable / 44K contracts vs 125M / 44K
- ✅ LoRA rank-16 constraint — 16 directions of adaptation, class competition problem
- ✅ Multi-window design — contracts avg 1,737 tokens, 82% need multiple 512-token windows
- ✅ WindowAttentionPooler — CLS per window → learned attention → [768]
- ✅ Two output paths: all tokens (fusion) vs CLS (transformer eye)
- ✅ Transformer eye value: global semantic fingerprint, not ordering

### Still Needed — Core

- ☐ Flash Attention 2 vs SDPA — what they are, why the fallback exists
- ☐ bfloat16 vs float32 — numerical precision tradeoff
- ☐ alpha/r scaling (32/16 = 2.0) — what it controls at initialization and during training
- ☐ B is zero at init, A is random — why this matters for stable LoRA training start

### Still Needed — GraphCodeBERT Upgrade (Phase 3.6)

- ☐ Why GraphCodeBERT over CodeBERT — pretrained with DFG signal; what the additional pretraining teaches the model about data flow
- ☐ `_word_embeddings` property — why direct access to the embedding layer is needed for prefix injection
- ☐ GNN prefix injection path in `forward()` — `inputs_embeds` instead of `input_ids`; how K prefix embeddings are prepended before code tokens
- ☐ Position IDs with prefix — prefix tokens at `position_id=1` (RoBERTa padding slot); code tokens at positions 3..3+(L-K-1); why position 1 not 0 or 2
- ☐ Code budget: `code_budget = L - K` — window is still exactly L tokens; K prefix tokens displace K code tokens from the right end; tradeoff
- ☐ Multi-window prefix expansion — `[B, K, 768]` expanded to `[B*W, K, 768]` via `.expand(); same K nodes shared across all windows; why shared not per-window
- ☐ `WindowAttentionPooler.prefix_k` — CLS shifts from position 0 to position K when prefix active; single-window returns `token_embs[:, prefix_k, :]`

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

### Core Three-Eye Architecture

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

### GNN Prefix Injection (Phase 3.6 — active)

- ☐ `gnn_prefix_k` parameter — 0=disabled (backward compat); 48=Phase 1 setting; why 48
- ☐ `gnn_prefix_warmup_epochs=15` — prefix suppressed during warmup; why train GNN first
- ☐ `_current_epoch` — set by trainer each epoch; how the warmup gate works: `if epoch >= warmup_epochs`
- ☐ `gnn_to_bert_proj` — Linear(256, 768); projects GNN node embeddings into transformer embedding space; why this projection needs its own LR group
- ☐ `prefix_type_embedding` — Embedding(5, 768); type-specific bias per declaration node type; why the transformer needs to know node type not just embedding value
- ☐ `select_prefix_nodes()` — priority sort → truncate to K → project → add type bias → zero-pad; the zero-pad choice vs attention mask
- ☐ Priority order: CONSTRUCTOR(6) > FALLBACK(4) > RECEIVE(5) > MODIFIER(2) > FUNCTION(1) — why entry-point nodes first
- ☐ Warmup design: projection starts from random init at epoch 15 with a well-trained GNN — why this is better than training both from scratch simultaneously
- ☐ Backward compat: `gnn_prefix_k=0` → identical output to original model; all existing checkpoints load unchanged

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

### Optimization

- ☐ AdamW optimizer — why AdamW not Adam; weight decay as implicit regularization
- ☐ Parameter groups — why different learning rates for different parts
- ☐ `gnn_lr_multiplier=2.5` — GNN collapsed to ~10% gradient share by ep8; boosting LR counteracts dominance
- ☐ `lora_lr_multiplier=0.3` — why LoRA trains slower; catastrophic forgetting risk at higher LR
- ☐ `fusion_lr_multiplier=0.5` — fusion had 4-5× higher grad norm than GNN; 0.5× lets GNN catch up
- ☐ `gnn_prefix_proj_lr_mult=1.0` — PrefixProj gets its own LR group; why full LR (not reduced) for projection

### Data and Sampling

- ☐ Weighted sampler — 3× weight on any-vulnerable contracts; what this does to batch composition
- ☐ `label_csv = "multilabel_index_cleaned.csv"` — model now trains on cleaned labels (3,859 removed); what changed

### Loss and Thresholds

- ☐ `eval_threshold=0.35` vs naive 0.5 — why 0.5 is wrong for imbalanced multi-label; patience trap at 0.5
- ☐ Threshold calibration — how to find the right threshold and what F1 measures
- ☐ `pos_weight_min_samples=3000` — the Reentrancy 2.82× over-amplification fix
- ☐ `dos_loss_weight=0.5` — changed from 0.0; fractional gradient scaling mechanism: `w*logit + (1-w)*logit.detach()`; why detach not mask
- ☐ `aux_loss_warmup_epochs=8` — aux weight ramps 0→0.3 over first 8 epochs; why aux heads hurt early training
- ☐ Patience=30 and early stopping — what patience measures and what the trap is
- ☐ Auxiliary loss weighting — how aux_gnn, aux_transformer, aux_fused contribute to total loss

### Prefix Injection Training

- ☐ `gnn_prefix_k=0` default; `48` for Phase 1 run — how trainer passes this to model and how model uses `_current_epoch`
- ☐ Per-epoch prefix status logging — `prefix_active` flag + `prefix_proj_weight_norm` logged to MLflow
- ☐ Weight norm trajectory of `gnn_to_bert_proj` during warmup vs post-warmup — what drift means

### Observability

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

- ✅ Shortcut learning — model finding valid-but-wrong correlates; confirmed in SENTINEL via Phase 3 dominance
- ✅ Catastrophic forgetting — gradient overwriting of pretrained knowledge
- ✅ Label leakage / target leakage — answer given as input feature
- ✅ Data contamination — entity-level vs row-level deduplication
- ✅ Heavy-tailed distributions and their effect on gradient signal
- ✅ Oversmoothing — repeated averaging at depth; all nodes converge to global average; destroys distinctiveness
- ⚠ Gradient flow / backpropagation — chain rule, gradient vanishing at depth; covered but needs reinforcement
- ✅ Residual connections fix gradient vanishing — identity path; gradient always ≥ 1 along skip connection
- ✅ Data ceiling — when label quality caps F1 regardless of architecture; confirmed by v7/v8-AB/PLAN-3A
- ✅ Edge presence ≠ signal quality direction — PLAN-3A: 92.5% DEF_USE coverage on Timestamp, yet DEF_USE hurt Timestamp
- ☐ Class imbalance — full landscape of solutions (resampling, reweighting, loss design)
- ☐ Evaluation metrics — precision, recall, F1 for multi-label; why accuracy is useless here
- ☐ Threshold calibration — why optimal threshold is not 0.5; v7 gained +0.022 F1 from tuning alone

### Architecture Patterns

- ✅ Transfer learning — pretrain then adapt, why it works, data efficiency argument
- ✅ Low-rank adaptation (LoRA) — the parameter efficiency argument
- ✅ Multi-head attention — why multiple heads, what each head can specialize for
- ✅ Residual connections — skip connections prevent gradient vanishing (covered in Phase 3)
- ☐ Attention mechanism from first principles — dot product attention formula (vs GAT's additive attention)
- ☐ Hierarchical GNN — the two-level alternative to unified graph
- ☐ HGT (Heterogeneous Graph Transformer) — different weights per edge type

### Interpretability

- ✅ The faithfulness problem — attention weights ≠ causal importance; softmax distributes over irrelevant neighbors
- ✅ JK phase weights as diagnostic — global constant behavior, not per-node routing; what std tells you
- ☐ GradCAM on graphs — gradient of output w.r.t. node features as importance scores
- ☐ Three explanation families: gradient-based, perturbation-based, attention-based

---

## Phase 10 — Training Dynamics (Experimental Evidence)

Understanding what the training logs and ablation results actually teach us.
This phase is grounded entirely in docs/ml/ — real runs, real numbers.

### Convergence Patterns

- ☐ What a healthy loss curve looks like — steady descent, matched train/val
- ☐ Plateau-then-burst pattern — PLAN-3A: plateau ep17–35, then ep36→38→41 rapid improvement; what causes it
- ☐ Why F1 breakthroughs correlate with fused gradient spikes — the fusion layer as the main learner
- ☐ Patience=30 in practice — why PLAN-3A spent 26 epochs after ep41 without improvement; auto-expire mechanics
- ☐ Aux warmup pattern ep1–8 — what auxiliary heads do early; why the main classifier lags behind
- ☐ GNN share dropping below 55% after ep25 — what this says about which eye dominates post-ep25

### JK Attention Trajectory During Training

- ☐ Phase 2 weight starts high (ep1: ~0.329 in v8-AB) — why new edge types get explored early
- ☐ Collapse over training — Phase 2 falls from 0.329→0.204; Phase 3 rises 0.486→0.744
- ☐ PLAN-3A ICFG-only trajectory — Phase 2 std=0.152 at ep3 (higher than v8-AB's 0.078); why ICFG-only gave cleaner early signal
- ☐ Phase 2 collapse vs Phase 2 std — mean tells you where weight landed; std tells you if it's routing or fixed
- ☐ Collapse alert threshold — all phase stds < 0.05 after ep3; what would trigger it in practice

### Prediction vs Reality

- ☐ Pre-PLAN-3A predictions scorecard — 4/7 correct direction, 3/7 wrong or opposite
- ☐ Why edge coverage rates don't predict signal quality direction — DEF_USE 92.5% on Timestamp yet hurt Timestamp
- ☐ Timestamp surprise — DEF_USE was amplifying label noise, not adding signal; ICFG better captured guard patterns
- ☐ ExternalBug surprise — CALL_ENTRY preserved but DEF_USE removal still hurt; both dimensions needed together
- ☐ What this teaches about designing ablations — one-at-a-time changes; predictions force you to make your assumptions explicit

### Macro F1 Ceiling

- ☐ v7: 0.2875 / v8-AB: 0.2851 / PLAN-3A: 0.2877 — all three converge to same ceiling
- ☐ Why different architectures converge to the same ceiling — the data bottleneck argument
- ☐ v8.0-B result (GATE-GCB-0): F1=0.2460 at ep10-11, killed; cleaned labels did NOT break the ceiling
- ☐ H5 REFUTED — label cleaning alone cannot break the 0.287 ceiling; ceiling is architectural
- ☐ What this means for strategy — ceiling is architecture not data; GraphCodeBERT + prefix injection is the path forward, data cleaning is secondary

---

## Phase 11 — Label Quality & Data Engineering

The confirmed primary bottleneck. Label noise directly caps F1 regardless of architecture.
All items here are actionable before v9 training.

### Documented Label Problems and Root Cause

- ☐ OR-labeling root cause — BCCC uses folder-level labeling; every contract in "reentrancy/" folder gets Reentrancy=1 regardless of whether that specific contract is vulnerable; systematic false positives
- ☐ Tier 1 mislabeling — structural impossibility: contract labeled vulnerable when graph literally cannot exhibit the vulnerability (has no external calls → cannot reenter)
- ☐ Tier 2 mislabeling — semantic: structure permits vulnerability but specific contract is safe (has external call + state write, but nonReentrant modifier or safe CEI order)
- ☐ D1: Timestamp — before cleaning: 48.2% had uses_block_globals=0 (structural impossibility); after cleaning: structural cases removed but semantic noise (~20–35%) remains
- ☐ D3: Reentrancy — 14% of Reentrancy=1 contracts have no external calls; cleaned -611 labels
- ☐ D2: DoS — was 7 training samples (excluded); now ~243 positives; dos_loss_weight=0.5

### label_cleaner.py (`ml/scripts/label_cleaner.py`)

- ☐ Conservative design — can only REMOVE positive labels, never CREATE new ones; audit JSON logs every change
- ☐ Per-class structural precondition functions — what each class requires to be structurally possible
- ☐ Actual cleaning results (2026-05-23): −611 Reentrancy, −568 Timestamp, −1665 UnusedReturn, −383 CallToUnknown, −632 MishandledException = 3,859 total from 44,524 rows
- ☐ v8.0-B used cleaned labels — F1=0.2460 at ep10-11, did not break 0.287 ceiling → Tier 1 cleaning was necessary but not sufficient; Tier 2 and architecture are the real limiters

### Re-Extraction Pipeline

- ☐ Why 45 minutes matters — fast re-extraction makes data experiments as cheap as architecture experiments
- ☐ Version coupling: graph files + embedding table + weights must stay synchronized
- ☐ What happens if you train with v8 graphs + v9 embeddings — silent feature mismatch; no error at load time
- ☐ The re-extraction checklist — schema version bump, graph_schema.py constants, assert guards, embedding table dim

### Current Status and Pending Experiments

- ☐ PLAN-3B (DFG-only: CF + DEF_USE) — ablation matrix still incomplete; pending
- ☐ PLAN-3D (JK concatenation mode) — switch jk_mode="cat"; pending; superseded by GraphCodeBERT priority
- ☐ Tier 2 noise remains — architecture upgrade (GraphCodeBERT + prefix) is now the primary path

---

## Recall Sessions Needed

Items flagged ⚠ require a focused re-teaching session before we advance past
the phase that depends on them. These are not gaps — they were covered — but
the understanding was incomplete or a misconception was found.

| Topic | Why flagged | Blocking phase |
|-------|-------------|----------------|
| Backpropagation / gradient flow | User said "no idea" during residuals explanation; misconception about 0.4 gradients; chain rule not yet internalized | Phase 3 LayerNorm, Phase 8 Trainer |
| Softmax sums to 1 | Initial misconception: "softmax doesn't make sum=1." Corrected but benefit of re-doing from scratch | Phase 3 GAT, Phase 7 Losses |

### Recall Session Format

Each recall session:
1. **Code first** — show the actual line where the concept appears in the codebase
2. **From scratch** — re-derive the concept without assuming prior explanation held
3. **Check question** — one question with the answer closed; see if it sticks this time
4. **Mark ✅** when the user answers without a hint

---

## Phase 12 — GraphCodeBERT + GNN Prefix Injection (`docs/proposal/`)

**Status: ACTIVE EXPERIMENT.** GATE-GCB-2 passed (Phase 0 verified); GATE-GCB-3 smoke running; P1-TRAIN scheduled.
This is the primary architectural upgrade path. Learn alongside the other phases — execution plan at `docs/proposal/EXECUTION_PLAN.md`.

### Why This Architecture Change

- ☐ Why the 0.287 ceiling is architectural not data — v8.0-B confirmed H5 REFUTED
- ☐ What GraphCodeBERT knows that CodeBERT doesn't — DFG-aware pretraining; what data flow in pretraining teaches vs LoRA adaptation
- ☐ The core problem being solved — GNN and transformer run in parallel paths; neither knows what the other found; cross-attention fusion is one-directional signal transfer

### GNN Prefix Injection Concept

- ☐ The key idea — inject K GNN node embeddings as prefix tokens into the transformer's input; transformer can attend to structural context while processing code tokens
- ☐ Why prefix (not postfix or fusion) — prefix tokens are attended to by all subsequent code tokens via self-attention; equivalent to giving the transformer "context notes" before reading code
- ☐ `gnn_to_bert_proj` Linear(256, 768) — why a projection is needed; the embedding spaces of GNN and BERT are not aligned
- ☐ `prefix_type_embedding` Embedding(5, 768) — type-specific bias added to proj output; why the transformer needs to know node role, not just embedding value
- ☐ Warmup design — GNN trains 15 epochs independently; at ep15 projection starts from random init with well-trained GNN; why this ordering matters

### Execution Plan Gates

- ☐ GATE-GCB-0 — v8.0-B result gate: CLOSED; F1=0.2460, H5 REFUTED → ceiling is architectural → accelerate
- ☐ GATE-GCB-1 — Prerequisites (PRE-1 to PRE-5): GraphCodeBERT download, tokenizer update, audit_prefix_node_counts.py (K=48 covers 95.5%)
- ☐ GATE-GCB-2 — Phase 0 go/no-go: GraphCodeBERT drop-in smoke test → PASSED
- ☐ GATE-GCB-3 — Implementation verification smoke test (K=48 prefix): running now
- ☐ GATE-GCB-4 — Phase 1 full training results: 60–80 GPU hours; go/no-go for Phase 2
- ☐ GATE-GCB-5 — Phase 2 results: Option C (shared DFG)

### Expected Outcomes and What to Watch

- ☐ Primary signal: does prefix injection improve Reentrancy? (CALL→WRITE pattern visible to transformer)
- ☐ `prefix_proj_weight_norm` trajectory — constant during warmup, drifts post-ep15; fast drift = projection learning fast
- ☐ Risk: prefix tokens dominate CLS → transformer ignores code; monitor transformer eye F1 vs GNN eye F1 balance

---

## Phase 13 — Scripts (`ml/scripts/`)

Supporting scripts for diagnostics, data preparation, and model validation. Each script is a learning artifact — they reveal what problems were discovered and how they were diagnosed.

### Data Quality Scripts

- ☐ `label_cleaner.py` — structural precondition filter; per-class checks; audit JSON output (covered in Phase 11)
- ☐ `audit_prefix_node_counts.py` — PRE-4 gate; counts declaration-level nodes across 41,576 graphs; output: mean=20.3, P50=16, P95=47; why this audit was needed before choosing K=48
- ☐ `validate_graph_dataset.py` — pre-training validation; checks shape contract [N,11] and [E] (not [E,1])
- ☐ `patch_graph_features.py` — in-place patch for BUG-3 visibility fix; why in-place patching instead of re-extraction

### Diagnostic Scripts

- ☐ `jk_weight_hist.py` — full per-node JK weight distribution on val set; used to confirm JK collapse (Phase 3 dominant 99.99%); run on any checkpoint post-training
- ☐ `complexity_correlation.py` — correlates complexity feature with vulnerability labels; was complexity discriminative?
- ☐ `edge_activation.py` — per-edge-type attention weight analysis; which edge types actually get attended to

### Training Support

- ☐ `tune_threshold.py` — per-class threshold calibration; v7 gained +0.022 F1 from tuning alone; how grid search works over threshold space
- ☐ `create_splits.py` and `dedup_multilabel_index.py` — entity-level deduplication; why contract-level not row-level
- ☐ `retokenize_windowed.py` — re-tokenizes dataset when window strategy changes

---

## Phase 9 — Graph Extensions (v8 implemented; v9 pending)

v8 is fully implemented and ablations have been run (see git log and docs/).
Cover the design decisions and results after all other files are understood.

### v8 — Implemented (cover the design reasoning)

- ☐ Extension A: ICFG-Lite — CALL_ENTRY(8) and RETURN_TO(9): what cross-function CFG enables
- ☐ Why global_cfg_node_map was needed for ICFG (vs per-function map in v7)
- ☐ Extension B: DEF_USE(10) — value flow from definition site to use sites
- ☐ Three DEF_USE categories: call return values, arithmetic results, state variable reads
- ☐ `phase2_edge_types` ablation param — how it isolates v8-A vs v8-B vs v8-AB
- ☐ v8-AB ablation results — what actually improved and what didn't (results in docs/)
- ☐ Schema evolution: v7(8 edge types) → v8(11 edge types) — what changed and why

### v9 — Not yet implemented (future)

- ☐ Extension C: Control-Dep(11) — direct edge from CHECK node to governed statements
- ☐ Why Control-Dep is different from CONTROL_FLOW — semantic vs structural dependency
- ☐ Schema evolution: v8(11) → v9(12) edge types
- ☐ Risk analysis: graph explosion on recursive contracts, Phase 2 heterogeneity with 5 edge types

---

## Progress Summary

| Phase | File / Topic | Status | Next item |
|-------|-------------|--------|-----------|
| 1 | `graph_schema.py` — node types + features | ✅ Complete | — |
| 1 | `graph_schema.py` — edge types | 🔄 Nearly complete | assert guards ☐ + NodeType IntEnum ☐ + STRUCTURAL_PREFIX_TYPES ☐ |
| 2 | `graph_extractor.py` | 🔄 Nearly complete | BUG-1/BUG-2 log-norm ☐, feature range validation ☐ |
| 3 | `gnn_encoder.py` — message passing | ✅ Complete | — |
| 3 | `gnn_encoder.py` — GAT | ✅ Complete | — |
| 3 | `gnn_encoder.py` — Three-Phase Architecture | ✅ Complete | — |
| 3 | `gnn_encoder.py` — JK Connections | 🔄 Nearly complete | `Linear(256,1)` shared weights ☐ |
| 3 | `gnn_encoder.py` — LayerNorm | ☐ Not started | — |
| 3 | `gnn_encoder.py` — Pooling | ☐ Not started | — |
| 3 | `gnn_encoder.py` — Architecture Parameters | ☐ Not started | — |
| 4 | `transformer_encoder.py` — core LoRA | 🔄 Mostly complete | Flash Attn ☐, bfloat16 ☐, LoRA alpha/r ☐, B=0 init ☐ |
| 4 | `transformer_encoder.py` — GraphCodeBERT + prefix | ☐ Not started | Why GCB, prefix path, position IDs, multi-window expand |
| 5 | `fusion_layer.py` | ☐ Not started | — |
| 6 | `sentinel_model.py` — three-eye | ☐ Not started | — |
| 6 | `sentinel_model.py` — GNN prefix injection | ☐ Not started | gnn_prefix_k, warmup gate, select_prefix_nodes |
| 7 | `losses.py` | ☐ Not started | — |
| 8 | `trainer.py` | ☐ Not started | — |
| 9 | v8 extensions (implemented) | ☐ Design reasoning not yet covered | — |
| 9 | v9 extensions (pending) | ☐ Not yet implemented | — |
| 10 | Training Dynamics (experimental) | ☐ Not started | — |
| 11 | Label Quality & Data Engineering | ☐ Not started | — |
| 12 | GraphCodeBERT + GNN Prefix Injection | ☐ Not started | Active experiment — learn alongside other phases |
| 13 | Scripts (`ml/scripts/`) | ☐ Not started | label_cleaner ☐, audit_prefix_node_counts ☐, jk_weight_hist ☐ |
| — | Recall: Backpropagation | ⚠ Needs reinforcement | Chain rule re-derivation |
| — | Recall: Softmax sums to 1 | ⚠ Needs reinforcement | From-scratch re-teaching |

**Current position:** Recall sessions for Phase 1+2 (in progress). Mid-recall on graph_extractor.py Round 1 — 4 questions pending (Q1–Q4 from last session).
**Recommended next step:** Complete recall sessions → finish Phase 3 remaining items → Phase 10 (training dynamics) and Phase 12 (GCB+prefix) can be taught in parallel as an active experiment.

### Key Findings Integrated Into Roadmap

| Finding | Source | Impact on what we're learning |
|---------|--------|-------------------------------|
| JK collapse confirmed: 99.99% Phase 3 dominant | jk-attention-collapse-findings.md | JK is a fixed global weighting, not routing; PLAN-3D needed |
| v7/v8-AB/PLAN-3A all hit 0.2877 ceiling | plan-3a-results.md + v8-vs-v7 | Architecture is not the bottleneck — or so it seemed |
| DEF_USE hurt Timestamp (+0.038 when dropped) | plan-3a-results.md | Edge presence ≠ signal quality; label noise interacts with edge types |
| Reentrancy: only +0.005 with ICFG preserved | plan-3a-results.md | Label noise was believed to be ceiling |
| v8.0-B: F1=0.2460 with cleaned labels, ceiling holds | GATE-GCB-0 | H5 REFUTED — ceiling is architectural, not data; pivot to GCB |
| Fusion layer drives all F1 breakthroughs | v8-AB-training-analysis.md | Fused grad spikes = mechanism of learning |
| K=48 covers 95.5% of declaration nodes | audit_prefix_node_counts.py | PRE-4 gate passed; K=48 is the right budget |
| GATE-GCB-2 passed (Phase 0 smoke) | EXECUTION_PLAN.md | GraphCodeBERT drop-in works; P1-IMPL complete; P1-TRAIN imminent |
