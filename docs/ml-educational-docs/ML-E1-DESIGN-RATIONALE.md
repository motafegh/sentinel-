# SENTINEL ML Module — Design Rationale

## How to Use This Document

This document explains the *why* behind every major architectural decision in the SENTINEL ML module. It is written for a new team member who already understands machine learning fundamentals and wants to understand not just what SENTINEL does, but why it does it that way — and what happens when it goes wrong.

Each section follows a consistent narrative arc: what problem existed, what alternatives were considered, why the chosen approach was selected, and what failure modes the decision prevents. Where possible, real failure stories from SENTINEL's development are included. These are not hypothetical — they describe bugs, regressions, and training collapses that actually occurred during the v5.0 and v5.1 development cycle. Understanding these failures is as instructive as understanding the successes.

Reading this document alongside `ML-T1-ARCHITECTURE.md` (the reference guide for configuration values and module interfaces) and `ML-T2-TRAINING-PIPELINE.md` (the operational guide for running training jobs) will give you a complete picture of the system. This document deliberately avoids duplicating reference content; it focuses on narrative and reasoning, not configuration values. When you read about a constant like `NODE_FEATURE_DIM=12` here, the intent is to explain why it exists and why changing it is dangerous — not to tell you where to find it in the code.

The sections are ordered from high-level architectural decisions (Why GNN + CodeBERT) to low-level implementation decisions (Why No Sigmoid). The final two sections — "Failure Modes and Their Root Causes" and "How the Architecture Evolved" — are intended as reference material once you understand the architecture, giving you the diagnostic vocabulary to debug training problems and the historical context to understand why the current design is shaped the way it is. You do not need to read them on first pass through the document.

---

## The Core Problem: Why Smart Contract Vulnerability Detection Is Hard

Automated smart contract security analysis has been attempted many ways. Rule-based static analysis tools like Slither and MythX encode known vulnerability patterns as code checks — they look for specific syntactic signatures and flag them. Symbolic execution tools like Manticore explore program paths by treating inputs as symbolic variables, checking whether any reachable state violates a security property. Formal verification tools encode contract behavior as logical formulas and use theorem provers to check correctness properties.

Each approach has a fundamental ceiling. Rule-based tools are excellent at shallow, syntactic patterns — a missing access modifier, a `delegatecall` to a user-controlled address, an integer operation without an overflow check. They fail at patterns that require understanding program flow: the same code can be vulnerable or safe depending on what happens before and after it executes. Symbolic execution is theoretically complete but practically limited by path explosion — real contracts have thousands of paths, and most symbolic execution engines time out or require aggressive pruning that misses real bugs. Formal verification requires writing a specification of correct behavior, which is often harder than writing the contract itself.

The core challenge is that smart contract vulnerabilities are fundamentally about the intersection of *structural* and *semantic* properties. A reentrancy vulnerability requires both a structural property (there exists a path in the call graph from contract A through an external call back into contract A before a state update) and a semantic property (the external call target is untrusted, the state being updated is security-relevant). Neither property alone is sufficient to identify the vulnerability — you need both simultaneously.

SENTINEL's 10 vulnerability classes span a range along this structural-semantic axis. Integer overflow is primarily structural: find arithmetic operations that receive untrusted input via READS edges, operating on types that can overflow. The structure of the data flow is the vulnerability, and a careful human auditor can often spot it from the call graph alone. Timestamp dependence is more mixed: the structural part is "the contract reads `block.timestamp` and uses it to make a branching decision," and the semantic part is "the branching decision is security-relevant (affects who can call a function, who receives funds) rather than merely informational." Unused return values straddle the line: it is structurally observable (a READS edge from a function call result that has no subsequent use), but recognizing that the return value *matters* requires understanding what the called function does — a semantic judgment.

Understanding this spectrum helps predict which vulnerability classes will be hardest for the model. Classes that are purely structural should be learnable from the graph alone, and CodeBERT provides confirmation. Classes that require semantic judgment about context and intent are fundamentally harder — they require the model to learn what makes a particular code pattern dangerous versus merely unusual.

This dual nature of the problem is what drives SENTINEL's dual-path architecture. No single representation captures both structural flow and semantic content at the level of abstraction needed for reliable vulnerability detection across diverse, real-world contracts.

---

## Why Dual-Path: GNN + CodeBERT

### The Case Against Pure Text Models

A text-only model like CodeBERT operating on raw Solidity source faces two fundamental limitations that become acute for vulnerability detection.

The first limitation is the 512-token context window. CodeBERT was pretrained with a maximum sequence length of 512 tokens. Real Solidity contracts — especially production contracts from DeFi protocols — routinely exceed this limit. When a contract is truncated to 512 tokens, the beginning of the file (import statements, contract declaration, state variable declarations) consumes tokens that could have been used to represent the function bodies where vulnerabilities actually live. The model never sees the vulnerable code at all for long contracts.

The second limitation is that structural relationships are implicit in token sequences. Consider reentrancy: the vulnerability requires understanding that an external call happens before a state update, that the external call recipient can call back into the original contract, and that this callback will execute against pre-update state. In the token sequence, the external call and the state update might be separated by fifty tokens of argument formatting, event emission, and validation logic. The attention mechanism does attend over all pairwise token positions, but attention decays with distance and is not guaranteed to capture multi-hop structural relationships. CodeBERT's pretraining objective (masked language modeling on code) does not reward learning call graph topology — it rewards predicting masked tokens, which correlates with local context, not cross-function call chains.

Worse, different developers write the same structural pattern differently. One developer writes `msg.sender.call{value: amount}("")` directly; another writes `_execute(msg.sender, amount)` where the implementation is in a base contract. A text model that learned the first form may not recognize the second as the same structural pattern.

### The Case Against Pure Graph Models

A GNN operating only on the AST/CFG graph representation has the opposite problems. The graph captures structural topology with precision. CALLS edges represent actual function invocations. CONTROL_FLOW edges represent actual execution paths. READS and WRITES edges represent actual data dependencies — which state variables are read and written, in which functions, triggered by which execution paths. These are exactly the relationships needed for structural vulnerability reasoning.

But the GNN's node features are necessarily limited. A node feature vector of dimension 12 can encode type information, position information, and basic structural statistics, but it cannot encode the *content* of the code — the actual variable names, function names, expression patterns, and library identifiers that carry semantic meaning.

Consider two contracts: one that implements a custom token transfer with a hand-rolled reentrancy guard, and one that imports OpenZeppelin's `ReentrancyGuard` and inherits from it. Their graph structures may look similar (both have a transfer function, both have a lock state variable, both have the appropriate call pattern). But the token-level difference — the presence of the OpenZeppelin import and inheritance — is a strong signal that the developer was aware of the reentrancy risk and applied a standard mitigation. A pure GNN cannot see this difference; it sees two structurally similar graphs and must predict identical vulnerability scores.

Similarly, a GNN cannot distinguish `transfer(address, uint256)` from `transferFrom(address, address, uint256)`. Both are FUNCTION nodes with similar structural profiles. The semantic distinction between them — one requires prior approval, one does not — is encoded in their names, not their graph structure. This semantic distinction matters for detecting certain authorization-bypass vulnerabilities.

### Why the Combination Works

The dual-path design exploits a natural division of labor that matches how human auditors actually work. An experienced auditor does two things simultaneously: they read the code's semantic content (names, patterns, library choices, comments) to form initial hypotheses about what might be wrong, and they trace control flow and data flow to verify whether those hypotheses are correct.

CodeBERT models the first activity. It reads the Solidity source and develops representations that encode semantic patterns — which library patterns are present, which naming conventions are used, which expression patterns appear in which contexts. The GNN models the second activity. It traces the graph structure to identify structural properties — call chains, data flow paths, control flow orderings — that correspond to vulnerability patterns.

For reentrancy, the GNN identifies the structural cycle in the call graph (CALLS edges creating a path back to the original contract) and the CONTROL_FLOW ordering (external call before state write). CodeBERT identifies the semantic context: the `call()` or `send()` idiom that makes an external call, the `balance` or other state variable that is updated after. Neither path's evidence alone is conclusive; together they provide mutually reinforcing signals.

The CrossAttentionFusion module is where this complementary evidence is synthesized. Rather than simply concatenating the two representations (which would allow no cross-modal interaction), CrossAttentionFusion allows the graph representation to query the token sequence, asking "for this node in the graph, what does the token sequence say about it?" The result is a fused representation that is more than the sum of its parts.

There is also a mutual reinforcement dynamic that makes the dual path more robust than either path alone. If the GNN produces a strong signal (e.g., a clear reentrancy cycle in the call graph), the fused representation has strong evidence independent of whether CodeBERT's representation is well-calibrated. If CodeBERT produces a strong signal (e.g., recognizing the SafeMath import that mitigates integer overflow), the fused representation has evidence independent of whether the GNN has traced the data flow correctly. When both paths agree, confidence is high. When they disagree, the fused representation should be uncertain — which is the correct behavior for a security tool. Expressing calibrated uncertainty rather than false confidence is as important as detecting true positives.

---

## Why Three-Phase GNN

### The Failure of a Monolithic GNN

SENTINEL v4 used a single-phase, four-layer GAT (Graph Attention Network) that processed all edge types together. Every edge in the graph — whether it was a CALLS edge, a CONTROL_FLOW edge, a READS edge, or a CONTAINS edge — was fed into the same attention mechanism with the same hyperparameters: the same number of attention heads, the same `add_self_loops` setting, the same dropout rate.

This design failed because different edge types have fundamentally different semantics that require different propagation mechanics. The most critical problem involves CONTROL_FLOW edges and the `add_self_loops` setting.

Self-loops in a GAT add an edge from each node to itself, ensuring that a node's own features are included as one of its "neighbors" when computing its updated representation. This is appropriate for many edge types — a function node should include its own features when aggregating messages from its called functions. But for CONTROL_FLOW edges, a self-loop implies that a CFG node's execution proceeds to itself — that is, an infinite loop or a node that is its own successor in the control flow graph. This is semantically wrong for the vast majority of CFG nodes.

When `add_self_loops=True` is used on CONTROL_FLOW edges in a GAT, the attention mechanism computes attention weights over both the self-loop and the actual successors. The self-loop introduces a spurious attention target that did not exist in the real program. Because the self-loop feature vector is identical to the current node's feature vector (it is the node itself), it creates a shortcut: the easiest way for the attention mechanism to achieve low loss is to primarily attend to the self (preserving the current representation) rather than learning to propagate information across the actual control flow edges. The structural edges — where the real signal lives — are down-weighted by an attention mechanism that found it easier to attend to the self-loop.

The result in v4 was a GNN that learned surprisingly little about control flow and primarily encoded structural connectivity from the denser CALLS/READS/WRITES edges.

### Why Node Type Normalization Matters

Before discussing the three phases, a low-level design decision deserves explanation: the normalization of node type IDs in feature vectors.

SENTINEL's graph nodes have 13 types (CONTRACT, FUNCTION, MODIFIER, EVENT, FALLBACK, RECEIVE, CONSTRUCTOR, STATE_VAR, and five CFG node subtypes), assigned integer IDs from 0 to 12. These IDs are included as one of the 12 node features, encoded as a float. The other features in the vector encode position ratios, degree statistics, and other structural properties, all normalized to the `[0.0, 1.0]` range. If the raw type integer is used directly (`feature[0] = float(type_id)`), then CFG_NODE_RETURN (type 12) would have a feature value of 12.0 — far outside the `[0, 1]` range of the other features.

In the dot-product attention of a GAT layer, features with larger magnitude dominate the similarity computation. A raw type ID of 12.0 would contribute as much to the similarity as all other 11 features combined. The GAT would learn to classify nodes almost entirely by type identity, ignoring the position, degree, and structural properties that differentiate nodes of the same type from each other. Two CFG_NODE_RETURN nodes in very different graph positions would appear identical to the attention mechanism.

The fix is to normalize: `feature[0] = float(type_id) / 12.0`, bringing the type ID into `[0.0, 1.0]` consistent with other features. This is a one-line change in the graph schema, but removing this normalization would cause the GNN to regress to a type-dominated representation and would not be detectable from loss curves alone — the model would still converge, just to a weaker solution. If the GNN shows signs of over-relying on node type at the expense of structural position (all nodes of the same type behave identically in learned representations), this normalization should be verified.

### Phase 1: Structural Connectivity

Phase 1 processes the structural and containment subgraph: CALLS edges (function calls between functions), READS edges (functions reading state variables), WRITES edges (functions writing state variables), EMITS edges (functions emitting events), INHERITS edges (contracts inheriting from other contracts), and CONTAINS edges (contracts/functions containing their children).

These edge types share common properties. They are semantically bidirectional in effect — knowing that function A calls function B is structurally informative for both A (what it does) and B (when it is called). Self-loops are appropriate because a node's own features should participate in its updated representation. Two GAT layers are used because these structural relationships require multi-hop propagation: a reentrancy vulnerability might require following a chain of CALLS edges (A calls B calls C writes state-variable-owned-by-A) to identify the cross-contract write that enables the attack.

Eight attention heads in Phase 1 allow the GAT to simultaneously specialize different heads on different edge types. In practice, learned attention patterns show different heads specializing on CALLS versus INHERITS versus READS/WRITES relationships, though the model learns these specializations implicitly rather than having them explicitly assigned.

The output of Phase 1 is a 128-dimensional node embedding for every node in the graph, encoding its structural position in the contract.

### Phase 2: Control Flow

Phase 2 processes only the CONTROL_FLOW subgraph using a single GAT layer with one attention head and `add_self_loops=False`. This configuration is derived from the semantics of control flow.

Control flow is directed: execution proceeds from a CFG node to its successors, not the other way. The direction matters for vulnerability detection. A reentrancy check (the `require(!locked)` guard) must come *before* the external call in the control flow; if it comes after, it provides no protection. Timestamp manipulation requires understanding the order in which the timestamp is read and the state is updated. These orderings are encoded in the direction of CONTROL_FLOW edges, and the GNN must learn to respect this direction.

One attention head is used because control flow has a single semantic meaning: execution order. Unlike the multi-faceted structural subgraph where different heads can specialize on different edge types, the control flow subgraph has only one type of edge and one interpretation. Multiple heads would not provide additional specialization — they would simply add parameters without a corresponding increase in representational capacity.

`add_self_loops=False` is an absolute constraint, not a preference. If this is accidentally changed to `True` in any refactoring, the GNN's control-flow representations will be corrupted. The corruption is subtle enough that it may not immediately manifest as a crash or obvious failure — the model will still train and produce outputs — but the vulnerability classes that depend on control-flow reasoning (reentrancy, timestamp dependence, denial-of-service) will degrade. This degradation may be incorrectly attributed to other causes during debugging.

### Phase 3: Reverse Containment

Phase 3 addresses a specific information aggregation gap. After Phase 2 has run, every CFG node has an embedding that encodes its position in the control flow graph, what execution paths lead to it, and what execution paths follow from it. This CFG-level information is exactly what is needed for control-flow vulnerability reasoning.

But the final classifier needs a contract-level representation, which is obtained by pooling over FUNCTION nodes (not CFG nodes — see the pooling section). Function nodes in the original graph do not have direct access to their CFG children's Phase 2 representations unless there is an edge allowing function nodes to aggregate from CFG nodes.

The original graph has CONTAINS edges going from FUNCTION nodes to their CFG children. Message passing along these edges propagates from CFG children up to FUNCTION parents — which is what we want. But in PyTorch Geometric, the convention is that edges in a `(source, target)` format propagate messages from source to target. CONTAINS edges are stored as (FUNCTION, CFG_NODE) — from parent to child. Message passing along these edges propagates from FUNCTION to CFG_NODE (parent to child), which is the *wrong* direction for aggregating CFG information into FUNCTION nodes.

Phase 3 generates REVERSE_CONTAINS edges at runtime: for every CONTAINS edge (FUNCTION, CFG_NODE), a reversed edge (CFG_NODE, FUNCTION) is added with type-7. Message passing along these reversed edges propagates from CFG_NODE to FUNCTION — the correct direction for function nodes to aggregate from their CFG children. After Phase 3, each FUNCTION node's representation incorporates both the structural information from Phase 1 and the control-flow information from Phase 2.

---

## Why REVERSE_CONTAINS Is a Runtime Concept

When the need for Phase 3's reversed containment edges was identified, the straightforward implementation would have been to modify the graph extraction pipeline: add type-7 edges to every graph `.pt` file during extraction, re-run the extractor on all 44,420 contracts, and update every file on disk. This would have required several days of compute time, and re-extraction always carries the risk of introducing new extraction bugs that corrupt the dataset.

The alternative — generating reversed edges at runtime inside the GNN's `forward()` method — requires no re-extraction. The GNN identifies every type-5 (CONTAINS) edge in the incoming graph, creates a reversed copy with type-7, and includes these reversed edges in Phase 3's edge computation. The cost is a few milliseconds per forward pass for the reversal operation, which is negligible compared to the GAT computation itself.

This design choice has an important consequence for the embedding table. The edge embedding table (`edge_emb`) is an `Embedding(NUM_EDGE_TYPES, 32)` module that maps edge type indices to 32-dimensional embedding vectors. When the reversed containment edges were first introduced, the initial implementation reused the type-5 (CONTAINS) embedding for the reversed edges: type-7 edges would look up index 5 in the embedding table. This saved one embedding row but created an ambiguity — the GNN received the same embedding vector for both parent-to-child containment (type 5) and child-to-parent reverse containment (type 7), and could not learn the directional distinction.

The correct implementation assigns type-7 to reversed edges and expands the embedding table to `Embedding(8, 32)`. The new row (index 7) is initialized randomly and trained from scratch, allowing the GNN to learn a distinct representation for reverse containment. The operational consequence is `NUM_EDGE_TYPES = 8` everywhere in the configuration, including `graph_schema.py` and any code that constructs or validates edge attribute tensors.

Critically: type-7 edges are never stored in graph `.pt` files. If you inspect a graph file on disk, you will never see a type-7 edge. They exist only during a forward pass. Any code that reads graph files and validates that all edge types are in `[0, NUM_EDGE_TYPES-1]` must account for the fact that files on disk only contain types 0–6, while the GNN internally works with types 0–7.

There is a related implementation detail about edge attribute tensor shape that caused a separate class of bugs during development. The edge attributes in graph files are stored as a 1-D integer tensor of shape `[E]`, where `E` is the number of edges. The `nn.Embedding` lookup expects a 1-D integer index tensor. If the edge attribute tensor were stored as shape `[E, 1]` (a 2-D tensor with one column), the embedding lookup would fail with a shape error — `nn.Embedding` does not accept 2-D index tensors. The graph schema enforces `[E]` shape explicitly, and any code that adds edges at runtime (such as the REVERSE_CONTAINS generation in Phase 3) must produce the same 1-D shape. A 2-D tensor `torch.tensor([[7], [7], [7]])` and a 1-D tensor `torch.tensor([7, 7, 7])` have the same values but different shapes, and only the 1-D version works with `nn.Embedding` without additional manipulation.

---

## Why Jumping Knowledge (JK) Connections

### The Gradient Collapse That Motivated JK

In the v5.1-fix28 training run, per-module gradient norm monitoring was added to the training loop to track how much each module was contributing to weight updates. The results revealed a catastrophic pattern: the GNN's share of the total gradient norm started at approximately 65% in epoch 1 — meaning GNN updates were dominating early training — and monotonically dropped to approximately 10% by epoch 8. For all subsequent epochs, the GNN contribution remained near 10%, while the transformer path dominated.

A model where the GNN contributes 10% of gradient updates is, for practical purposes, ignoring the graph. The weights in the GAT layers are barely moving. The structural information in the graph is not being learned. The final model is essentially a text-only model with a dormant graph path attached.

The cause was architectural. In the v5.1-fix28 configuration, only Phase 3's output was passed to the classifier. The gradient path from the loss to, say, Phase 1's parameters in layer 1 required backpropagating through: the classifier linear layers, the fused representation, the JK combination (if any), the Phase 3 GAT layer, the Phase 2 GAT layer, Phase 1's second GAT layer, and finally Phase 1's first GAT layer. Each GAT layer applies softmax normalization and attention-weighted aggregation — operations whose gradients can be very small when attention weights are concentrated or when representations have similar values across neighbors. By the time gradient signal passed through three-to-four GAT layers, it had attenuated to near zero for Phase 1's early parameters.

### What Jumping Knowledge Provides

Jumping Knowledge (JK) networks address gradient attenuation through direct connections: rather than only using the final layer's output, the classifier receives learned combinations of all intermediate outputs. In SENTINEL's three-phase GNN, this means Phase 1's output, Phase 2's output, and Phase 3's output are all collected and combined through a learned attention mechanism before reaching the classifier.

The gradient path for Phase 1's parameters is now: loss → classifier → JK attention → Phase 1 output → Phase 1 GAT layer 2 → Phase 1 GAT layer 1. This is a two-layer path (within Phase 1) rather than a four-layer path. The gradient reaches Phase 1's parameters with much less attenuation. In practice, after implementing JK connections, Phase 1's gradient norm share stabilized at 30–40% throughout training — a healthy contribution that indicates the GNN is actively learning.

The JK attention mechanism (`_JKAttention`) is a small learned module that takes the three phase outputs and produces a single combined output. It computes per-phase attention weights via a linear projection and softmax, then computes the weighted sum of phase outputs. Critically, the weights are global (same for all nodes in a given forward pass) rather than per-node, which keeps the mechanism simple and avoids adding complexity that could itself become unstable.

### The Critical Implementation Detail: Live Tensors

When the JK mechanism was being implemented, the codebase already contained infrastructure for diagnostic collection of intermediate outputs: the `return_intermediates=True` flag caused the GNN to collect Phase 1, Phase 2, and Phase 3 outputs during the forward pass for logging and visualization. These were collected using `.detach().clone()` — creating copies of the tensors disconnected from the computational graph, safe for logging without affecting gradient computation.

The obvious implementation mistake would be to build JK on top of this existing infrastructure — to have JK consume the same list that `return_intermediates` populates. If this were done, JK would receive detached tensors. The `_JKAttention` module's parameters would receive no gradients (their input has no gradient history). More critically, because the Phase 1, 2, and 3 outputs would be detached before entering JK, the gradient would stop there entirely. Phase 1's parameters would see zero gradient through the JK path, and the only path for Phase 1's gradient would again be through Phase 3 — the exact long chain we were trying to bypass. The fix would have no effect.

The implementation uses a separate internal list, `_live`, that collects phase outputs during the forward pass without any `.detach()`. The `return_intermediates` path continues to exist and continues to use `.detach().clone()` for its diagnostic purpose. These two lists serve different purposes and must never be conflated. The `_live` list is local to the forward method — it is not stored as an attribute, it cannot be accidentally reused, and it goes out of scope after the forward pass returns. This isolation prevents future code from accidentally using the detached version.

---

## Why Per-Phase LayerNorm

### The Magnitude Imbalance Problem

When JK connections were first implemented without per-phase normalization, monitoring of the JK attention weights revealed an unexpected problem: Phase 1 was consistently receiving attention weights above 85%, with Phases 2 and 3 receiving less than 8% each, across all training steps. The model had learned to rely almost entirely on Phase 1's output — not because Phase 1's content was most relevant, but because Phase 1's output had a higher L2 norm.

The magnitude difference arises from depth. Phase 1 runs two GAT layers with a residual connection: the output is `LayerNorm(GAT2(GAT1(x)) + x)`. With a skip connection, the output magnitude is approximately the sum of the input magnitude and the transformation magnitude. Phase 2 and Phase 3 each run only one GAT layer: their output magnitude is lower because they have processed fewer layers.

The JK attention mechanism computes softmax weights over the three phase outputs. Softmax is sensitive to magnitude: a vector with twice the L2 norm of another will tend to receive much higher softmax weight unless the projections in the attention mechanism are carefully calibrated to cancel the magnitude difference. In practice, learned projections do not reliably cancel magnitude differences — the optimization converges to the easier solution of attending mostly to the highest-magnitude vector.

The effect is that JK with magnitude imbalance is functionally equivalent to no JK at all — only Phase 1's output matters, and Phases 2 and 3 are decorative. All the gradient benefits of JK are still present (Phase 1 still has a short gradient path), but the control-flow signal from Phase 2 and the containment-aggregation signal from Phase 3 are nearly ignored.

### The Fix

Applying `LayerNorm(128)` after each phase's residual connection, before appending to the `_live` list, normalizes all three outputs to comparable magnitude without destroying their directional information. LayerNorm subtracts the mean and divides by the standard deviation across the 128 feature dimensions for each node independently, then applies a learnable affine transformation (scale and shift).

After per-phase LayerNorm, the JK attention weights are governed by learned content relevance rather than magnitude accidents. The model can learn, for example, that for reentrancy detection, Phase 2's control-flow signal should receive high weight (because reentrancy is a control-flow ordering vulnerability), while for integer overflow, Phase 1's READS/WRITES signal should receive high weight (because overflow depends on what values flow into arithmetic operations). These are genuinely different informational requirements, and JK with per-phase normalization can represent them.

The three LayerNorm modules are separate `nn.Module` instances with separate learnable scale and shift parameters. This allows each phase's normalization to have independent affine parameters, which is important because the three phases produce representations with structurally different content (structural connectivity vs. control flow vs. containment aggregation). A single shared LayerNorm would impose the same learned transformation on all three, which would be unnecessarily restrictive.

---

## Why Function-Level Pooling

### The CFG_NODE_RETURN Flood

Global mean pooling is the natural way to aggregate a set of variable-size node embeddings into a fixed-size graph representation. Take all node embeddings, compute their mean, and pass the result to the classifier. This is what SENTINEL v5.0 used.

After v5.0 completed training and achieved a validation F1 of 0.5828 — above the passing gate of 0.50 — behavioral tests were run on 10 known-vulnerable real-world contracts. The results were catastrophic: overall detection rate was 15%, with several vulnerability classes showing 0% detection. The model was predicting all-negative for most contracts despite the positive validation numbers.

Diagnosing this failure required analyzing the node type distribution in the dataset. The finding: across the 44,420 contracts in the dataset, approximately 77% of all graph nodes are `CFG_NODE_RETURN` nodes. Every function, every conditional branch, every loop body ends with a CFG return node. A four-function contract with moderate complexity might have 200 total graph nodes, of which 155 are CFG_NODE_RETURN.

Global mean pooling over these 200 nodes produces a representation that is weighted 77% by CFG_NODE_RETURN embeddings. The FUNCTION nodes, which encode the contract's actual logic — what functions exist, what they call, what state they read and write — account for perhaps 2% of the pooled representation. The classifier was trying to detect reentrancy from a representation where 77% of the signal was "this contract has many return statements" and 2% was "this contract has these functions with these call patterns."

The model still achieved positive validation F1 because the training and validation data came from the same distribution. When the same CFG_NODE_RETURN-dominated representations appeared in validation, the model had learned to detect whatever weak signal remained. But real-world contracts from different codebases had different CFG structures that broke the learned pattern.

### The Fix: Semantic Node Pooling

The fix is to pool only over nodes that carry semantic content relevant to vulnerability reasoning: FUNCTION, FALLBACK, RECEIVE, CONSTRUCTOR, and MODIFIER. These five node types represent the callable entry points of a contract — the places where execution can begin, and the places that contain all the security-relevant logic. CFG nodes are structural scaffolding that represents the flow *within* functions; the vulnerability-relevant information from CFG analysis has already been propagated up to the FUNCTION nodes via Phase 3's reverse containment aggregation.

Semantic node pooling reduces the effective pooling set from hundreds of nodes to typically 5–20 nodes per contract. The resulting representation is dominated by the contract's function-level structure rather than its CFG scaffolding. A reentrancy-vulnerable contract's representation is shaped by the embedding of its transfer function (which has CALLS edges to an external call and WRITES edges to a balance state variable in the wrong order) rather than by the embeddings of the 150 return nodes that exist purely because every function must eventually return.

An important edge case must be handled: library contracts, abstract contracts, and interface contracts sometimes have no FUNCTION/FALLBACK/RECEIVE/CONSTRUCTOR/MODIFIER nodes in the graph (they may consist only of declaration stubs or pure CFG structure). In this case, the semantic pooling set is empty, and attempting to pool over it would produce an undefined or NaN representation. The implementation falls back to global mean pooling over all nodes when no semantic nodes are present. This fallback is a correctness requirement — without it, inference on library files would crash.

---

## Why Separate Learning Rate Groups

### The Catastrophic Forgetting / Under-Training Tradeoff

Early SENTINEL training used a single AdamW learning rate for all model parameters. This creates an irreconcilable tension between two parts of the model that have fundamentally different training requirements.

CodeBERT's 124 million parameters were pretrained on massive code corpora using substantial compute. They encode general code understanding: syntax patterns, idiom recognition, code structure correlations. This pretraining is SENTINEL's foundation — without it, the text path would be a random-feature extractor. A learning rate appropriate for training from scratch (10⁻³ or higher) will destroy this pretraining within the first few epochs through catastrophic forgetting. The optimizer will update the weights by large amounts to fit the vulnerability detection task, overwriting the general code patterns with task-specific patterns that are too narrow and too brittle.

The GNN parameters are randomly initialized. They start with no knowledge of graph structure, edge type semantics, or vulnerability patterns. They need aggressive weight updates to escape the random initialization and converge to useful representations within a manageable number of training epochs. A conservative learning rate appropriate for CodeBERT fine-tuning (10⁻⁵ or lower) would leave the GNN effectively frozen for the first dozen epochs, severely under-trained even by epoch 60.

With a single learning rate, one of two bad outcomes is inevitable: set it low enough for CodeBERT safety, and the GNN never properly trains; set it high enough for GNN convergence, and CodeBERT forgets its pretraining within a few epochs.

### The Three-Group Solution

Three parameter groups are defined, each with a learning rate multiplier applied to the base learning rate:

The GNN parameter group (`name.startswith("gnn.")`) receives a 2.5× multiplier. With a base learning rate of 2×10⁻⁴, GNN parameters train at 5×10⁻⁴. This multiplier was determined empirically as the minimum value that prevents GNN gradient collapse (GNN gradient norm share dropping below 15%) during the first ten training epochs. Higher multipliers converge faster but risk GNN over-fitting the structural patterns before the text path has converged.

The LoRA adapter group (`"lora_" in name`) receives a 0.5× multiplier, training at 1×10⁻⁴. LoRA adapters inject low-rank update matrices into CodeBERT's attention layers. They need to update faster than frozen CodeBERT parameters (which are not in any optimizer group — they receive zero gradient by design) but more conservatively than the GNN. If LoRA adapters update too aggressively, they will over-fit their low-rank subspace early in training, producing CodeBERT representations optimized for the training distribution before the GNN has converged enough to provide balanced gradient signal.

All remaining parameters — the CrossAttentionFusion module, the Three-Eye Classifier's output heads, the JK attention module, the per-phase LayerNorm modules, and the auxiliary loss heads — receive the base learning rate of 2×10⁻⁴. These modules are trained from scratch but have simpler optimization landscapes than the GNN (which must learn edge type semantics from random init) and benefit from a moderately conservative learning rate that matches the fused gradient signal.

---

## Why Auxiliary Loss Heads (Three-Eye Classifier)

### The Dead Path Problem

The Three-Eye Classifier receives a concatenated vector built from three sources: the GNN eye (pooled FUNCTION-node embeddings, 128 dimensions), the TF eye (CodeBERT CLS token, 128 dimensions), and the Fused eye (CrossAttentionFusion output, 128 dimensions). These are concatenated to form a 384-dimensional vector, which is passed to a linear layer that produces the 10 vulnerability logits.

Without auxiliary supervision, gradient flow to the GNN eye and TF eye depends entirely on the learned weights in the final linear layer. The linear layer assigns weights to each of the 384 input dimensions; dimensions that are informative for the classification task will receive larger weights. The critical problem is that this weight assignment is circular: a dimension receives large weights only if it is informative, but a dimension can only become informative if it receives adequate gradient to train the upstream components.

Early in training, the CrossAttentionFusion output (the Fused eye) has an advantage: it is produced by a module that explicitly combines information from both paths, creating smooth interpolations between graph and text signals even before either path has converged. This makes the Fused eye's dimensions statistically more predictive earlier in training, causing the linear layer to assign them larger weights. The GNN and TF eyes' dimensions receive smaller weights, generating smaller gradients for the upstream GNN and CodeBERT-LoRA parameters. These paths train more slowly, their representations become less distinctive, and the linear layer reinforces its decision to down-weight them. This is a positive feedback loop toward a dead-path equilibrium.

In the v5.0 run, the auxiliary loss weight was `λ=0.1` — too weak to prevent this. Post-training analysis showed the TF eye contributing nearly nothing to predictions for several vulnerability classes, and the GNN eye exhibiting the gradient collapse described in the JK section.

### The Fix: Per-Eye Supervision

Auxiliary loss heads add separate output layers applied to the GNN eye alone and the TF eye alone. The total loss is computed as:

```
L_total = L_main + λ × (L_gnn_aux + L_tf_aux)
```

where `L_main` is the focal loss from the final fused classification head and `λ = 0.3` scales the auxiliary contributions.

With auxiliary heads, the GNN eye receives gradient directly from `L_gnn_aux`: the gradient path is loss → GNN aux head → GNN eye representation → GNN pooling → GNN parameters. This path is independent of the fused classifier's weight allocation. Even if the fused linear layer assigns small weights to GNN-eye dimensions early in training, the auxiliary path provides a direct gradient signal that keeps the GNN training.

The weight `λ = 0.3` reflects a lesson from v5.0's `λ = 0.1` setting. The auxiliary gradient needs to be large enough to compete with the fused path's gradient — not dominate it, but remain competitive. At `λ = 0.1`, the fused path's gradient was typically 5–10× larger than the auxiliary path's, which was insufficient to prevent the dead-path equilibrium. At `λ = 0.3`, auxiliary and main gradients are in the same order of magnitude, providing genuine competition.

The auxiliary heads are removed at inference time (only the main fused head's output is used). The auxiliary heads are only meaningful during training as gradient conduits — at inference time, they would produce lower-quality predictions than the fused head and are discarded.

---

## Why CrossAttentionFusion

### The Limitation of Concatenation

SENTINEL v4 fused the GNN and text path outputs through simple concatenation: the GNN graph embedding and the CodeBERT text embedding were concatenated and passed directly to the classifier. This is the simplest possible fusion strategy, and it has a fundamental limitation.

Concatenation allows no cross-modal interaction. The classifier receives a vector where the first portion encodes graph structure and the second portion encodes text semantics, but there is no mechanism for the model to learn correspondences between specific aspects of the graph and specific aspects of the text. The model cannot learn "the suspicious CALLS edge in the graph corresponds to the `msg.sender.call{value:...}()` pattern in the token sequence" — it must learn this correspondence implicitly through the downstream classification loss, applied to a 768-dimensional concatenated vector. This is a very weak learning signal for learning cross-modal correspondences.

In practice, concatenation with a shared classifier tends to produce a model that primarily relies on whichever modality is more informative early in training. The cross-modal correspondences that would benefit from explicit attention — the alignment between graph nodes and token positions — are never properly learned.

### What CrossAttentionFusion Does

CrossAttentionFusion implements a cross-attention mechanism adapted to SENTINEL's asymmetric modalities. The GNN produces per-node embeddings (after Phase 3 and JK combination, a 128-dimensional vector per node). CodeBERT produces per-token embeddings (a 768-dimensional vector per token position, sequence length 512). These representations have different sizes and different numbers of elements.

CrossAttentionFusion uses the GNN's contract-level representation (after pooling over FUNCTION nodes) as queries and the CodeBERT last-hidden-state as keys and values. For each query position, the attention mechanism computes a weighted sum over all 512 token positions, with weights determined by the relevance of each token position to the graph query. The output is a single 128-dimensional vector that encodes which parts of the token sequence were most relevant to the graph structure.

This allows the fused representation to encode specific cross-modal alignments: for a contract with a suspicious external call pattern in the graph, the attention will focus on the token positions that describe how that call is constructed. The fused representation is then better positioned to detect the combination of structural and semantic properties that characterize vulnerabilities.

### Why the Output Is Locked at 128

The fusion output dimension is 128, and this value is an architectural constant that cannot be changed without significant downstream disruption. The ZKML module (Module 2, responsible for generating zero-knowledge proofs of inference) builds a proxy MLP with its input dimension hardcoded to 128. The proxy MLP architecture is `Linear(128, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 10)`, and this architecture is baked into the EZKL ZK circuit.

ZK circuits for neural network inference are compiled artifacts. Changing the input dimension of the proxy MLP requires recompiling the circuit with the new architecture, regenerating the trusted setup (a cryptographic ceremony that is computationally expensive and should ideally involve multiple independent parties), and redeploying the on-chain verifier contract. This is a multi-day process with significant coordination overhead.

The practical implication: any proposed redesign of the CrossAttentionFusion module must preserve the 128-dimensional output regardless of what internal changes are made. If there is ever a good reason to increase fusion representational capacity, the increase must happen in the internal `attn_dim` (which is 256 and is free to change) rather than in the output projection.

---

## Why LoRA

### The VRAM Constraint

CodeBERT has approximately 124 million parameters. Full fine-tuning of this model — meaning every parameter receives gradient updates every backward pass — requires storing: the parameter values themselves, the gradients for each parameter, and the optimizer state (two momentum vectors per parameter for AdamW). For 124M parameters in float32, this is approximately 1.5 GB just for parameters, and 3× that (4.5 GB total) when gradients and optimizer state are included. Combined with the GNN, fusion layer, activations for the forward pass, and the batch of graphs and token tensors, full fine-tuning exceeds the 8 GB VRAM available on the RTX 3070.

Even with gradient checkpointing (which trades VRAM for recomputation time), full fine-tuning at any useful batch size is impossible. A batch size of 1–2 samples would produce extremely noisy gradient estimates and make training impractically slow.

### What LoRA Does

LoRA (Low-Rank Adaptation) addresses the VRAM problem by freezing all CodeBERT parameters and injecting trainable low-rank matrices into specific weight matrices. Instead of fine-tuning the full `W` matrix of a projection layer, LoRA learns two small matrices `A` (shape `hidden_dim × r`) and `B` (shape `r × hidden_dim`) such that the effective update is `W + BA`. For rank `r = 16` applied to Q and V projections in all 12 transformer layers, the total number of trainable LoRA parameters is approximately 590,000 — about 0.5% of CodeBERT's size.

Frozen CodeBERT parameters require no gradient storage and no optimizer state. The VRAM savings are enormous: only the 590K LoRA parameters need gradient and optimizer state, plus the GNN and other components. This brings total VRAM usage within the 8 GB budget at a batch size of 8–16.

The LoRA adapters are applied specifically to Q (query) and V (value) projection matrices, not K (key). The query determines what the attention mechanism looks for; the value determines what information is retrieved. Adapting these allows LoRA to change both the attention queries and the retrieved content, which is sufficient to align CodeBERT's representations with the vulnerability detection task. The key projection, which determines how queries match against positions, generalizes well from code pretraining and does not benefit from task-specific adaptation. Applying LoRA to K would add parameters without proportional benefit.

The rank `r = 16` with scaling `α = 32` (effective scaling factor = α/r = 2.0) are established defaults from the LoRA literature that provide a good balance of representational capacity and regularization for fine-tuning tasks. Lower rank reduces capacity but also reduces overfitting risk on the 44K training samples. The current settings have not been ablated for SENTINEL specifically — if the LoRA path appears to underfit (TF-eye performance plateaus early), increasing r to 32 is a reasonable first experiment.

---

## Why Deduplication Was Critical to the Dataset

### The BCCC Duplication Problem

The BCCC-SCsVul-2024 dataset organizes vulnerable contracts by vulnerability category. Contracts are grouped into directories named after their vulnerability class: `reentrancy/`, `integer_overflow/`, `timestamp_dependence/`, and so forth. Contracts with multiple vulnerabilities appear as separate copies in multiple directories — a contract vulnerable to both reentrancy and timestamp dependence appears in both `reentrancy/` and `timestamp_dependence/` with identical source content but different file paths.

SENTINEL's original data pipeline assigned each contract a unique identifier based on the MD5 hash of its *file path*. Two copies of the same source file at different paths received different hashes and were treated as separate training samples. The dataset construction pipeline then assigned these to train/validation/test splits using stratified random sampling without any check for content duplication.

The consequence: 34.9% of content groups — 7,630 unique contract sources — had identical copies assigned to multiple splits. A contract in the training set appeared in the validation set as well. When the model trained on a contract and then validated on a copy of the same contract, the "validation" was measuring how well the model had memorized training data, not how well it generalized.

The inflation of validation F1 was not trivial. Contracts are labeled with their vulnerability classes, and the model that memorizes a training contract will correctly predict all its labels when it encounters the same contract in validation. For a 10-class multilabel problem, this translates to a significant F1 inflation — especially for rare classes (like DenialOfService with only 377 examples) where memorization of duplicated samples can artificially double the effective training set for that class.

### The Full Scope of the Damage

Because all pre-dedup F1 metrics are inflated by data leakage, there is no reliable quantitative comparison available between v4 and any v5.x checkpoint trained on the leaky dataset. The architectural comparisons that justified v5's three-phase design, the ablation studies on JK connections, the threshold-tuning analysis — all of these were conducted on a leaky dataset where validation F1 included memorized training samples.

This does not mean v5's architectural decisions were wrong. The behavioral test failures (15% detection on real contracts) confirm that the underlying problem was real and that better architecture was needed. But the quantitative improvement claims from leaky training runs are not trustworthy baselines. The only honest comparison available is between architectures trained on the deduplicated dataset and evaluated on behavioral tests.

All training from v5.1 onward uses the content-deduplicated dataset at `ml/data/processed/multilabel_index_deduped.csv`, which reduces 68,523 rows to 44,420 rows by assigning identifiers based on source content MD5. The splits in `ml/data/splits/deduped/` are stratified by label and verified to have zero cross-split content overlap. Any new F1 metrics from these runs are the first reliable baselines for the project.

### The DenialOfService Data Starvation Problem

Deduplication revealed a secondary problem that was invisible in the leaky dataset: the DenialOfService (DoS) vulnerability class has only 377 examples in the deduplicated set, with approximately 257 in the training split. This is an extreme class imbalance — IntegerUO has 15,529 examples, more than 60× as many. A model trained naively on this distribution will almost certainly never predict DoS as positive, because predicting negative on every DoS contract still achieves 99.1% accuracy for that class.

Positive class weights (BCEWithLogitsLoss `pos_weight` parameter) partially address class imbalance by scaling the positive-class loss by a weight proportional to the negative-to-positive ratio. For DoS with 257 positives in a training set of 31,092 samples, the positive weight is approximately (31,092 − 257) / 257 ≈ 120. This means the model receives 120× the gradient signal when it misclassifies a DoS-positive contract as negative, compared to a non-DoS contract. In practice, this helps but is insufficient — with only 257 positive examples, the model has very few DoS contracts to learn from, and high positive weights amplify noise as well as signal.

Phase 3 of the v5.1 preparation plan addresses this through data augmentation for DoS: generating or collecting additional DoS-positive examples to expand the training set. The target is to bring the DoS training set to at least 1,000 examples before training v5.2. Without this augmentation, DoS will likely remain a weak class even with correct architecture and class weights.

The deduplication also changed the effective label distribution for all classes, not just DoS. Before deduplication, classes with contracts that appeared in many category directories were over-represented, because each directory copy counted as a separate positive example. After deduplication, these classes' counts reflect the actual number of unique vulnerable contracts in the dataset. Classes that were previously inflated by cross-directory copies see a larger reduction. The positive weights in `pos_weights_v5.1.pt` were recomputed from the deduplicated training split and are specific to this distribution. They must not be mixed with checkpoints or metrics from pre-dedup training runs.

---

## Why Focal Loss and Positive Class Weights

### The Class Imbalance Challenge

SENTINEL's 10 vulnerability classes are severely imbalanced. IntegerUO appears in 15,529 of 44,420 contracts (35%); DenialOfService appears in 377 contracts (0.85%). This 60:1 ratio is not unusual for vulnerability detection datasets — rare vulnerabilities are rare by definition. The imbalance creates a systematic incentive for the model to predict negative on all classes, since predicting negative is correct the vast majority of the time for most classes.

Standard binary cross-entropy loss handles imbalance poorly because it weights each example equally. In a training batch of 32 contracts, perhaps 27 are DoS-negative. Predicting negative for all 32 contracts achieves 84% accuracy on DoS and produces a low loss signal — but the model has learned nothing. The 5 DoS-positive contracts in the batch contribute 5/32 of the DoS-class loss, which may be insufficient to overcome the 27/32 contribution from easy negatives.

### Positive Class Weights

The first mitigation is positive class weights, applied as the `pos_weight` parameter in BCEWithLogitsLoss. For each class, the positive weight is computed as the ratio of negative to positive examples in the training set. A class with 1,000 positives and 30,000 negatives gets a positive weight of 30. This means each positive example contributes 30× as much to the loss as each negative example, approximately equalizing the gradient signal from the two groups.

Positive weights are recomputed from the deduplicated training split whenever the dataset changes. The saved tensor at `ml/data/processed/pos_weights_v5.1.pt` is specific to the v5.1 deduplicated training split. If the training split changes (for example, due to re-stratification or augmentation), the positive weights must be recomputed to match. Using positive weights computed from a different split or from the pre-dedup dataset would produce incorrect class weighting.

### Focal Loss

Positive weights address the between-class imbalance (DoS vs. IntegerUO), but do not address within-class imbalance between easy and hard examples. Within each class, there are easy negatives (contracts that look nothing like the vulnerable class and are correctly predicted with high confidence) and hard examples (contracts that share many patterns with the vulnerable class but differ in the specific detail that determines vulnerability). Standard cross-entropy treats all examples equally; easy negatives dominate the gradient because there are many more of them.

Focal Loss adds a modulating factor `(1 - p)^γ` to the loss, where `p` is the model's predicted probability and `γ` is a focusing parameter (typically 2). For an easy negative that the model correctly predicts with probability 0.95, the factor is `(1 - 0.05)^2 = 0.9025` — the loss from this example is reduced by 90%. For a hard example that the model is uncertain about (predicted probability 0.5), the factor is `(1 - 0.5)^2 = 0.25` — the loss is reduced less severely. Hard examples receive relatively more gradient signal, and the model's training is focused on the decisions that matter most.

In practice, Focal Loss and positive weights are complementary: positive weights balance the class representation, and Focal Loss focuses training on the hard examples within each class. Used together, they significantly improve detection of rare and hard-to-detect vulnerability classes compared to standard cross-entropy alone.

---

## Why No Sigmoid Inside the Model

SENTINEL's `forward()` method returns raw logits — 10 floating-point values per sample with no range constraint. Sigmoid activation is applied externally when computing predictions. This is a deliberate design choice required by three independent constraints, any one of which would be sufficient to mandate it. Having all three constraints pointing in the same direction is fortunate; it means the design is robust to changes in any one of the three downstream modules.

The first constraint is numerical stability of the loss function. Both BCEWithLogitsLoss and the Focal Loss variant used in SENTINEL apply sigmoid internally as part of their computation, using the numerically stable log-sum-exp form `max(x, 0) - x * y + log(1 + exp(-|x|))`. This formulation avoids the catastrophic numerical failure that occurs when computing `log(sigmoid(large_positive_x))` directly. For a large positive logit like `x = 50`, `sigmoid(x)` rounds to 1.0 in float32, and `log(1.0) = 0` — but the mathematically correct value of `log(sigmoid(50))` is approximately −1.9 × 10⁻²². If sigmoid is applied inside the model before the loss function sees the output, the loss function receives a value of exactly 0 where it should see a very small negative number. The gradient computation goes wrong in the same way. By accepting raw logits, the loss function can use its numerically stable form throughout.

The Focal Loss variant is particularly sensitive to this issue. Focal Loss adds a modulating factor `(1 - p)^γ` to down-weight easy examples, where `p = sigmoid(x)` is the predicted probability. For a correctly-classified easy example with large logit `x`, `p ≈ 1.0` and `(1 - p)^γ ≈ 0`, which is correct — easy examples should receive near-zero loss weight. But if sigmoid is applied before the loss, the float32 rounding error in `(1 - 1.0)^γ = 0.0^γ` makes the computation exact zero, which is numerically correct in this case but creates a gradient computation that no longer flows through to the logit itself. The numerically stable implementation avoids all of these edge cases by keeping everything in log-space until the final result is computed.

The second constraint is ZKML compatibility. The ZKML module generates a zero-knowledge proof that the model inference was performed correctly. ZK circuits for neural networks use fixed-point arithmetic with limited precision — the EZKL framework uses scale-8192 little-endian representation, effectively a fixed-point encoding of real values. The sigmoid function maps all real numbers to the interval (0, 1), producing output values that are very close to 0 or very close to 1 for extreme logits. Representing numbers very close to 0 or 1 in this fixed-point encoding is imprecise relative to representing numbers with larger absolute values. A logit of +10 maps cleanly to a fixed-point value; the corresponding sigmoid output of 0.9999546 requires more precision bits to represent accurately, and accumulated fixed-point errors across the proxy MLP could push the output to the wrong side of the decision boundary. Passing logits to the ZK circuit allows fixed-point arithmetic to represent them with acceptable precision using the chosen scale factor.

The third constraint is per-class threshold flexibility. Because different vulnerability classes have dramatically different base rates in the training data — IntegerUO appears in 15,529 of 44,420 contracts while DenialOfService appears in only 377 — and because the consequences of false positives and false negatives differ by class (a false negative for reentrancy is catastrophic; a false positive for an informational class is merely annoying), each class needs its own prediction threshold. The standard practice of using 0.5 as the classification threshold is wrong for any class that is not approximately balanced.

Per-class threshold tuning works as follows: after training is complete, the model's sigmoid-transformed output probabilities are swept across a range of thresholds for each class independently, and the threshold that maximizes that class's F1 on the validation set is selected. This tuning is done in a post-processing step, using probabilities (sigmoid of logits), after the model checkpoint is saved. Keeping sigmoid external means the threshold tuning has access to the full probability range for each class, and thresholds can be re-tuned at any time without touching the model weights — for example, if the sensitivity/specificity tradeoff needs to be adjusted for production based on user feedback.

---

## Locked Constants and Why They Are Locked

Several constants in SENTINEL's configuration are designated as "locked." This designation means that changing them requires re-extracting all graph files from source contracts, re-tokenizing all contracts, and retraining from scratch. For a 44,420-contract dataset with complex extraction and multi-day training, a locked-constant change is a multi-week project. The lock is not a bureaucratic policy — it reflects the fact that these values are baked into binary artifacts that take hours to regenerate at scale.

`NODE_FEATURE_DIM = 12` is locked because it determines the shape of the node feature tensor in every graph `.pt` file on disk. Each file stores a tensor of shape `[N_nodes, 12]`. The 12 features encode node type (as a normalized float), position-in-parent, depth in the AST hierarchy, edge degree statistics, and other structural properties. If NODE_FEATURE_DIM is changed to 13 — for example, to add a new structural feature — every file on disk has the wrong feature dimension and must be regenerated by re-running the extraction pipeline on all source contracts. The extraction pipeline also involves compiling each contract with the appropriate Solidity compiler version and parsing the AST with the extractor; errors in either step can silently corrupt individual graphs. A feature dimension change therefore carries a significant quality risk in addition to the compute cost.

`NUM_CLASSES = 10` is locked because it propagates through every artifact in the data pipeline. The multilabel CSV has 10 label columns. The train/validation/test split files preserve this schema. The positive weight tensor (`pos_weights.pt`) has exactly 10 values, one per class, reflecting the class imbalance in the training set. Every model checkpoint's final linear layer has output dimension 10. Every threshold JSON file has 10 entries. Changing NUM_CLASSES requires rebuilding every one of these artifacts and invalidates every saved checkpoint. The policy that NUM_CLASSES is append-only (new classes can be added by retraining, but existing class indices cannot be reordered or removed) exists so that threshold files and checkpoint files remain interpretable without version conversion.

`MAX_TOKEN_LENGTH = 512` is locked because every token `.pt` file was pre-tokenized to exactly 512 tokens using CodeBERT's tokenizer with padding and truncation to 512. The token files store `input_ids` and `attention_mask` tensors of shape `[512]`. A change to 1024 (for example, to capture longer contracts more completely) requires re-running tokenization across the full dataset. Tokenization is fast but there are 44,420 files to process, and the output files occupy significant disk space. More importantly, changing MAX_TOKEN_LENGTH while keeping the same CodeBERT backbone requires careful consideration of whether CodeBERT's positional embeddings (trained to length 512) generalize to longer sequences — which they do not without modification.

`fusion_output_dim = 128` is locked for ZK-circuit reasons described in the CrossAttentionFusion section. Changing this dimension requires recompiling the ZK circuit, regenerating the trusted setup, and redeploying the on-chain verifier contract. This is a multi-day process that cannot be run on developer hardware — it requires a trusted setup ceremony.

`NUM_EDGE_TYPES = 8` and `REVERSE_CONTAINS = 7` are *not* locked in this sense. Type-7 edges are generated at runtime and the type-7 embedding is a new row learned from scratch each training run. No graph files contain type-7 edges. The number 8 is an in-memory configuration value that tells the GNN's embedding table how many rows to allocate. Changing it affects only code and checkpoints, not files on disk. This distinction — between constants that are embedded in binary files and constants that exist only in runtime configuration — is the key to understanding what "locked" actually means.

The practical workflow implication: any proposed change should be evaluated for locked-constant impact before implementation. A change to `NODE_FEATURE_DIM` requires scheduling extraction compute before training can begin. A change to `fusion_output_dim` requires ZK circuit recompilation and on-chain redeployment before the new checkpoint can be put into production. A change to a runtime constant like the GNN's hidden dimension or the number of edge types only requires code changes and a new training run. Understanding which changes fall into which category prevents wasted compute and avoids the situation where a training run completes but cannot be deployed because of a downstream locked-constant incompatibility.

### Why Graph Loading Uses `weights_only=True` But Checkpoint Loading Does Not

PyTorch's `torch.load()` function accepts a `weights_only` parameter that controls whether deserialization can execute arbitrary Python code embedded in the file. When `weights_only=False`, loading a `.pt` file is equivalent to running arbitrary serialized Python — a malicious or accidentally corrupted file could execute harmful operations at load time. When `weights_only=True`, only tensor data is loaded, and any attempt to deserialize non-tensor objects raises an error.

Graph `.pt` files contain `torch_geometric.data.Data` objects with edge attributes, node features, and metadata. These are PyG data classes, not bare tensors. Loading them with `weights_only=True` requires explicitly allowing these classes via the `safe_globals` parameter. SENTINEL's policy explicitly allows `[Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]` — the minimal set needed for graph data objects. Any PyG object not in this list raises an error on load, preventing arbitrary code execution while still allowing the legitimate data classes.

Checkpoint `.pt` files containing LoRA-fine-tuned model state dicts cannot be loaded with `weights_only=True`. LoRA adapter objects from the PEFT library are serialized as Python objects, and `weights_only=True` refuses to deserialize them. SENTINEL's policy is `weights_only=False` for checkpoints, accepting this tradeoff in exchange for the ability to load LoRA state dicts. The practical mitigation: only load checkpoints from the project's own training runs (paths under `ml/checkpoints/`). Never load checkpoints from external, unverified sources with `weights_only=False` without first inspecting the file's source and provenance.

---

## How the Architecture Evolved: v4 to v5.2

Understanding the history of architectural decisions helps explain why the current design has the shape it does. Each major version introduced changes in direct response to observed failure modes.

### v4: The Baseline That Established the Problem

SENTINEL v4 used a single-phase four-layer GAT over all edge types, with global mean pooling over all nodes, and simple concatenation fusion. It used full CodeBERT fine-tuning within the VRAM budget by reducing batch size to the minimum. It used a single learning rate for all parameters. There were no auxiliary loss heads, no JK connections, and no per-class node type filtering in the pooling step.

v4 achieved a tuned F1-macro of 0.5422 on the (leaky, pre-dedup) validation set. Behavioral tests showed essentially the same failure pattern as v5.0: detection rates that were acceptable on benchmark contracts but failed on real-world contracts. The v4 fallback checkpoint is still used as a baseline for per-class F1 floors (each class must achieve v4_F1 − 0.05 to be considered a regression-free improvement).

The lessons from v4 were: global mean pooling is wrong (node type imbalance problem), single-phase GNN mixes incompatible edge types, single LR causes CodeBERT to forget, concatenation fusion misses cross-modal correspondences, and the validation dataset is leaky.

### v5.0: Introducing the Three-Eye Architecture and CrossAttentionFusion

v5.0 introduced the three-phase GNN (though without JK connections — only Phase 3 output fed the classifier), CrossAttentionFusion to replace concatenation, the Three-Eye classifier (though with λ=0.1 auxiliary weight, too weak), and LoRA to replace full fine-tuning. The dataset was still the leaky pre-dedup 68K.

The v5.0 training run achieved tuned F1-macro 0.5828 on the leaky validation set — an apparent improvement over v4. But behavioral tests showed the same 15% detection rate. Two root causes were identified: first, the leaky dataset inflated the F1 numbers, obscuring how little had actually improved; second, all-node mean pooling was still in use, and the CFG_NODE_RETURN flooding problem described earlier was causing the classifier to operate on noise-dominated representations.

### v5.1: Fixing the Known Failures Systematically

v5.1 (currently in progress) addresses each identified failure systematically. Phase 1 introduced function-level pooling (FUNCTION/FALLBACK/RECEIVE/CONSTRUCTOR/MODIFIER nodes only). The dataset was deduplicated from 68K to 44K samples. JK connections were added with live tensors. Per-phase LayerNorm was added. Separate LR groups were introduced. Auxiliary loss weight was increased to λ=0.3. The GNN LR multiplier was set to 2.5×. NUM_EDGE_TYPES was extended to 8 with a dedicated REVERSE_CONTAINS embedding.

The fix28 run — which was the first attempt at v5.1 training — discovered the gradient collapse problem (GNN contribution dropped to 10% by epoch 8) and the JK live-tensor issue. The fix28 run was declared invalid and all its F1 metrics were discarded. The Phase 0+1+2 code fixes that followed these discoveries have now all passed testing (11/11 GNN encoder tests pass, including the non-negotiable `test_jk_gradient_flow` test).

### v5.2: The Target State

v5.2 is the planned stable training target. It incorporates all Phase 0–3 improvements: the architectural fixes from Phase 0+1+2, CEI (Check-Effects-Interaction) pair augmentation for reentrancy detection (Phase 3), DenialOfService augmentation for the severely data-starved DoS class (377 examples), and recomputed positive weights from the deduplicated dataset. A smoke run (Phase 4) will verify that gradient norms, loss curves, and per-class detection rates all behave correctly before committing to the full 60-epoch v5.2 run (Phase 5).

The key model version string `"v5.2"` is stored in checkpoints. This identifier is how downstream systems (the ZKML module, the inference API, the audit agents) know which architecture they are running. When a v5.2 checkpoint is loaded, the loading code verifies this version string and rejects checkpoints from earlier architectures. This version gating prevents the silent failure mode where a v5.0 checkpoint (with incorrect pooling) is accidentally loaded into a v5.2 model definition (with correct pooling), producing a model that runs but produces wrong results because the weights were trained with a different architecture.

### What Changes Between Versions Require Re-Extraction vs. Retraining

A practical question that comes up during development: "I want to change X. Do I need to re-extract all graphs?" The answer depends on where X lives in the data pipeline:

Changes that require re-extraction of all 44K graphs: any change to the graph extractor that affects the set of nodes, the set of edges, the node feature values, or the node feature dimension. Examples: adding a new edge type that must be stored in `.pt` files, changing NODE_FEATURE_DIM, changing the type normalization formula, adding new node type categories.

Changes that require retraining but not re-extraction: any change to the GNN architecture that does not affect the graph schema. Examples: adding a new GNN phase (as long as it uses edge types already in the files or generates them at runtime), changing attention head counts, changing hidden dimensions, enabling or disabling JK, changing pooling strategy.

Changes that require neither re-extraction nor retraining: changes to the training procedure (learning rate groups, loss function weights, augmentation), changes to the inference API, changes to threshold JSON files, changes to the ZKML proxy MLP (though these require ZK circuit recompilation).

---

## Why Behavioral Tests Are the Real Judge

### The Disconnect Between Benchmark F1 and Operational Performance

After v5.0 training completed and achieved a tuned F1-macro of 0.5828 on the held-out validation split — exceeding the project's stated passing gate of 0.50 — behavioral tests were run on 10 contracts sourced from real deployed protocols. The detection rate was 15%, with several vulnerability classes showing exactly 0% detection on contracts that a skilled human auditor would identify as clearly vulnerable.

This result was not an anomaly. It reflects a systematic gap between what benchmark F1 measures and what operational security performance requires.

Benchmark F1 on a held-out split measures whether the model has learned the same patterns that are present in the held-out examples as were present in the training examples. When training and validation both come from the same source distribution (BCCC contracts, with the same coding style, the same vulnerability patterns, the same Solidity version distribution), the held-out F1 measures distribution memorization quality. A model that memorizes BCCC's vulnerability patterns will score well on BCCC validation samples.

Operational performance requires generalization to contracts outside the training distribution. Real deployed contracts come from different teams, different coding cultures, different libraries, different Solidity versions, and different design patterns. A reentrancy vulnerability in a Uniswap V2 fork looks structurally different from a reentrancy vulnerability in a BCCC benchmark contract — the function names are different, the state variable names are different, the surrounding code patterns are different. A model that learned "this structural + semantic pattern = reentrancy in BCCC contracts" may not recognize the same vulnerability when it appears in a different codebase.

### What Behavioral Tests Actually Measure

The behavioral tests in `ml/scripts/manual_test.py` run known-vulnerable contracts from outside the training distribution through the model. The test contracts in `ml/scripts/test_contracts/` are either from real deployed protocols known to have been exploited, or carefully crafted synthetic contracts that represent the clearest possible examples of each vulnerability class.

A behavioral test passing does not mean the model is production-ready. It means the model can detect the most obvious examples of each vulnerability class in code that looks different from its training data. This is the minimum bar for a useful security tool. Behavioral test failure means the model has not learned generalizable vulnerability patterns — it has only learned dataset artifacts. A detection rate of 0% for any class is not acceptable regardless of the validation F1 for that class.

The rule of thumb: validation F1 is a development metric that helps detect training problems (if it drops, something broke). Behavioral test results are the product metric that determines whether the model is ready for deployment. Decisions about promoting checkpoints, changing architectures, or ending training runs should be made on behavioral test results, not on validation F1 alone.

There is an important asymmetry to be aware of. Good behavioral test results do not guarantee good validation F1 (a model could detect the obvious cases while being miscalibrated on the harder cases in the validation distribution). But good validation F1 definitely does not guarantee good behavioral test results, as v5.0 demonstrated. Validation F1 is necessary but not sufficient; behavioral test detection is the binding constraint. When the two metrics disagree — validation looks good but behavioral tests fail — trust the behavioral tests.

### The v4 Per-Class Floor Policy

Because all pre-dedup F1 metrics are inflated, absolute F1 thresholds from those runs cannot be used as deployment gates. Instead, SENTINEL uses relative thresholds: each class must achieve at least v4_F1 − 0.05 on the deduplicated validation set to be considered a non-regression. The v4 per-class floors are:

CallToUnknown 0.397, DenialOfService 0.384, ExternalBug 0.434, GasException 0.507, IntegerUO 0.776, MishandledException 0.459, Reentrancy 0.519, Timestamp 0.478, TOD 0.472, UnusedReturn 0.495.

These floors are conservative — v4 was trained on leaky data, so its true generalization performance was likely lower than these numbers. But the floors serve as a sanity check: if v5.x training produces a model that is worse than v4 on several classes according to the behavioral tests, something has gone wrong with the new architecture.

---

## Failure Modes and Their Root Causes

This section maps observed failure patterns to their architectural explanations. Understanding these failures is important for debugging future training runs and for designing experimental changes without inadvertently reintroducing known problems. Each entry describes the symptom (what you observe), the root cause (why it happens architecturally), and the diagnostic check (how to confirm and fix it).

### GNN Gradient Collapse

Symptom: Monitoring shows the GNN's share of total gradient norm dropping from 50–65% in early epochs to below 15% by epoch 8–10, remaining suppressed for all subsequent epochs. Per-class validation performance on structural vulnerability classes (reentrancy, TOD, external call vulnerabilities) degrades relative to semantic classes (unused return, integer overflow from simple expressions). The GNN eye's auxiliary loss contribution also drops sharply.

Root cause: Only Phase 3's output connects to the classifier without JK connections, requiring gradient to propagate through four GAT layers before reaching Phase 1's parameters. Over-smoothing and attention concentration in later layers attenuate the gradient to near-zero. Alternatively, JK is enabled but consuming detached tensors (from `_intermediates` instead of `_live`), silently preventing gradient flow to all phase parameters through the JK path.

Diagnostic check: (a) Confirm JK is enabled (`use_jk=True`) and that `_live` (not `_intermediates`) is used as input to `_JKAttention`. A quick way to verify: add a print statement in the forward method that checks `_live[0].requires_grad` — it must be `True`. (b) Confirm per-phase LayerNorm is present and registered as `nn.Module` attributes (check `model.gnn.named_modules()`). (c) Confirm the GNN learning rate multiplier is 2.5× in the optimizer group definitions. (d) Run a single backward pass on a small batch and check gradient norms for `gnn.phase1_conv1.weight` — they should be non-zero and of comparable order of magnitude to classifier gradients. If gradient collapse recurs despite these checks, increase the GNN LR multiplier to 3.0× and observe whether the collapse is merely delayed or resolved.

### All-Negative Predictions on Real Contracts

Symptom: Behavioral tests show detection rates near zero across multiple vulnerability classes. The model outputs probabilities below threshold for every class on known-vulnerable contracts. This can coexist with acceptable validation F1 if validation data comes from the same distribution as training data. A distinguishing sign: if validation F1 is positive but behavioral test detection is near zero for the same class, the model has memorized dataset patterns rather than learned vulnerability semantics.

Root cause: Global mean pooling over all nodes produces a representation dominated by CFG_NODE_RETURN nodes (77% of all nodes in a typical contract graph). The FUNCTION-level signal that carries vulnerability information is diluted below the detection threshold. The classifier sees a representation that is almost entirely "this contract has many return nodes" and cannot distinguish vulnerable from non-vulnerable contracts.

Diagnostic check: Confirm function-node pooling is active. In the GNN encoder's pooling step, log the node type distribution of nodes included in the pool for the first few batches — you should see only type IDs corresponding to FUNCTION (1), MODIFIER (2), EVENT (3), FALLBACK (4), RECEIVE (5), and CONSTRUCTOR (6), never CFG node types (8–12). Verify that the fallback to all-node pooling is only triggering when the semantic node mask is empty (add an explicit log message when the fallback activates). If thresholds have been carried over from a leaky training run (where the model was over-confident), re-tune them on the deduplicated validation set.

### Inflated Validation Metrics That Don't Transfer to Real Contracts

Symptom: Validation F1 substantially exceeds behavioral test detection rates. Specifically, a gap of more than 0.15 between validation F1-macro and the proportion of behavioral test contracts correctly detected should trigger investigation. In the extreme case (v5.0), validation F1 was 0.5828 while behavioral detection was 15%.

Root cause: Content leakage between train and validation splits. The BCCC dataset copies contracts into multiple category directories; path-based hashing treats each copy as a separate sample. Random split assignment places some copies in training and others in validation. The model memorizes training contracts and "generalizes" to validation copies of the same contracts. This is not generalization — it is retrieval.

Diagnostic check: Verify training is using `multilabel_index_deduped.csv` (44,420 rows) with splits from `ml/data/splits/deduped/`. Run a content-overlap check: extract the source MD5 for 200 random validation contracts and confirm none appear in the training set. If MD5s are not stored in the split files, compute them on the fly from the source contract text. Any overlap is a data leakage bug that invalidates the run.

### Phase 2 Control-Flow Signal Loss

Symptom: Reentrancy and timestamp dependence detection rates degrade in behavioral tests while integer overflow detection remains stable. Integer overflow depends primarily on READS/WRITES edges (Phase 1) to identify untrusted values flowing into arithmetic — it does not require CFG ordering. Reentrancy and timestamp dependence critically depend on the relative ordering of operations in the control flow graph (Phase 2). If Phase 2's signal is lost, the model becomes effectively blind to ordering-dependent vulnerabilities.

Root cause: The most common cause is `add_self_loops=True` being set for Phase 2, which corrupts control-flow propagation by introducing spurious self-attention. The secondary cause is Phase 3's REVERSE_CONTAINS edge generation being broken — if type-7 edges are not being generated, or if they are generated with the wrong direction, FUNCTION nodes cannot aggregate CFG information and Phase 2's output is wasted.

Diagnostic check: Confirm Phase 2 GAT is initialized with `add_self_loops=False`. During a debug forward pass with `return_intermediates=True`, inspect the edge index passed to Phase 3's GAT call and verify: (a) type-7 edges are present (count of edges with type 7 should be non-zero), (b) the direction is CFG_NODE → FUNCTION (source node type is a CFG type, target node type is FUNCTION/MODIFIER/etc.), not the reverse. Also check that `NUM_EDGE_TYPES=8` is set — if the edge embedding table has only 7 rows and a type-7 edge appears, an index-out-of-bounds error will crash the forward pass or silently produce the wrong embedding via modular wraparound depending on implementation.

### JK Attention Weight Degeneration

Symptom: JK attention weight logging shows one phase consistently receiving above 80% weight while others receive below 10%, across all batches and all epochs. The disparity does not diminish with training. The GNN-eye gradient norms for Phases 2 and 3 are near-zero relative to Phase 1, even though JK is enabled. Overall model performance is similar to having only Phase 1 (i.e., similar to the v4 architecture).

Root cause: Phase magnitude imbalance is overwhelming the learned attention content. Phase 1 processes two GAT layers plus residual connection, producing higher L2-norm outputs than Phases 2 and 3 (one layer each). The JK softmax computes attention weights that are dominated by magnitude rather than content direction. The model cannot learn to balance phases because the magnitude signal always wins.

Diagnostic check: Log the L2 norms of Phase 1, Phase 2, and Phase 3 outputs before they enter JK (i.e., after per-phase LayerNorm). These norms should be within a factor of 2 of each other. If Phase 1's norm is 5× or more larger than Phases 2 or 3, the LayerNorm is either missing, not learnable, or not applied before the `_live` append. Check `model.gnn.named_modules()` to confirm three separate `LayerNorm(128)` instances with `elementwise_affine=True` are registered. If LayerNorm is confirmed correct but Phase 2 or 3 norms are near zero, the corresponding phase may be producing near-zero outputs due to a different problem — check the edge masks being applied for each phase to confirm they are selecting the correct edge types.

### CodeBERT Catastrophic Forgetting

Symptom: Vulnerability classes that depend on semantic content — naming patterns, library usage, expression forms — degrade over training epochs. Classes like UnusedReturn (which requires recognizing the pattern of calling a function and not checking its return value) or ExternalBug (which requires recognizing unsafe external interaction patterns in the source text) show decreasing validation F1 as training progresses beyond the first few epochs. The TF-eye auxiliary loss value increases after initially decreasing.

Root cause: The LoRA learning rate is too high, causing the LoRA adapters to overfit their low-rank subspace to the training distribution and overwrite CodeBERT's general code representations. Alternatively, if the base CodeBERT parameters are accidentally included in an optimizer group (which should never happen — they must be frozen), they will be updated and CodeBERT's pretraining will be erased within a few epochs.

Diagnostic check: Confirm that base CodeBERT parameters (all parameters that are not LoRA adapters) have `requires_grad=False`. In PyTorch, this can be checked with `sum(p.requires_grad for p in model.transformer.parameters())` — only LoRA parameters should be True. Confirm the LoRA learning rate multiplier is 0.5× (not 1.0× or higher). If the problem persists with correct settings, the LoRA rank `r=16` may be too high for the available training data, causing the adapters to memorize training contracts. Try reducing to `r=8`.

### Zero-Knowledge Proof Failure

Symptom: The ZKML module fails to generate a valid EZKL proof, or the Groth16 on-chain verifier rejects the proof with a constraint violation error. The model inference itself succeeds (predictions are produced) but the proof generation or verification fails.

Root cause: The ZK circuit was compiled with `fusion_output_dim = 128` and the proxy MLP architecture `Linear(128, 64) → Linear(64, 32) → Linear(32, 10)`. Any mismatch between the live model's fusion output and these hardcoded circuit dimensions will cause the circuit's constraint system to be unsatisfied. The most common cause is a change to `CrossAttentionFusion`'s output projection that produces a tensor with a different size. A secondary cause is sigmoid being applied inside the model — the circuit receives probabilities where it expects logits, and the proxy MLP's expected input distribution is calibrated to logit-scale values.

Diagnostic check: Assert that `CrossAttentionFusion.forward()` produces an output of shape `[batch_size, 128]` by adding a shape assertion to the forward method during debugging. Confirm that the proxy MLP module's `fc1` layer has `in_features = 128`. Confirm that `model.forward()` returns raw logits (check that the maximum absolute value across a batch is not constrained to (0, 1), which would indicate sigmoid has been applied). If all checks pass and proof generation still fails, the likely cause is that the circuit was compiled against a different checkpoint and needs to be recompiled against the current architecture.

---

## Summary: Principles Behind the Architecture

Every decision described in this document traces back to a small set of principles that emerged from real failures:

**Measure what you care about.** Validation F1 measures distribution matching. Behavioral tests measure vulnerability detection. Only the latter matters for a security tool. Every training run must clear both gates — validation F1 as a sanity check, behavioral tests as the real verdict.

**Preserve gradient pathways.** The dual-path architecture is only effective if both paths receive gradient signal. JK connections, auxiliary loss, separate LR groups, and function-level pooling all exist to ensure neither path goes dormant. Any architectural change should be evaluated against the question: "does this risk cutting off gradient flow to part of the model?"

**Separate what is different.** Edge types with different semantic relationships require different propagation rules. Learning rates for pre-trained and randomly-initialised components should be different. Structural and semantic modalities should be allowed to specialise. When the original architecture mixed these, performance was poor and the failures were opaque.

**Make failures visible.** NaN loss counter, GNN collapse streak alert, JK phase dominance warning, per-epoch JK weight logging — these diagnostics exist because silent failures are the hardest to debug. A model that silently ignores the graph and achieves mediocre F1 using only CodeBERT is far more dangerous than one that crashes with an error.

**Lock what is locked.** The constants labeled "locked" are physical constraints, not conventions. Changing them without rebuilding the data pipeline produces misaligned shapes at runtime. Treat locked constants as you would treat an API contract: they can be changed, but changing them has a known, non-trivial cost that must be planned for.
