FM2.2.
2
Near-empty CFG node
feature vectors
Of the 12 dimensions, only 3 carry signal on CFG nodes:
type_id [0], loc [6], and call_target_typed [8] (always 1.0). The
remaining 9 features are hardcoded to 0.0. 48,175 CFG out of
66,288 total nodes (72%) carry almost no discriminative
information. The GNN must learn everything about CFG nodes
from structural context alone.
FM2.2.
3
id(lval) identity
comparison fragility
_compute_return_ignored() uses id(lval) for identity
comparison. This relies on CPython's object identity semantics.
If Slither ever caches or interns IR variable objects across
functions, id() comparison could produce false negatives. A
more robust approach would compare by .name attribute.
FM2.2.
4
Regex misses
.call{value:...} syntax
_compute_call_target_typed() source-scan fallback regex does
not match address(addr).call{value: 1 ether}("") (the
.call{value:...} syntax with curly braces before parentheses).
Such calls would be missed, returning 1.0 (typed) instead of
0.0 (raw address).
FM2.2.
5
most_derived heuristic
tiebreak bias
When two contracts have the same derivation score, the
tiebreak is non_iface.index(c) which picks the one defined
earlier in the file. This is actually worse than "last" for some
BCCC patterns where the vulnerable contract is the LAST
defined. The fallback to non_iface[-1] only triggers when
derivation_score is 0 for ALL candidat


5.1 gnn_encoder.py

Failure Modes:
ID Failure Mode Description
FM5.1.
1
Phase 2 operates on
near-featureless CFG
nodes
After Phase 1, declaration nodes have rich 12-dim features but
CFG nodes have only type_id, loc, and call_target_typed=1.0.
The GAT attention mechanism computes query/key dot
products — when keys are near-identical across CFG nodes,
attention weights become near-uniform, and message passing
degrades to simple averaging. Phase 2's directional signal
(call→write ordering for CEI detection) is washed out.
FM5.1.
2
JK attention learns to
down-weight Phase 2
If Phase 2 produces near-identical embeddings (because CFG
nodes are featureless), the JK attention scores for Phase 2 will
be low. The model effectively learns to ignore the
CONTROL_FLOW signal — the exact signal it was designed to
capture. This creates a vicious cycle where Phase 2 gets less
gradient, stays weaker, and gets further down-weighted.
FM5.1.
3
2-layer-per-phase
limits hop reach
Each phase sees only 2 hops of its edge type. Phase 2 can
reach nodes 2 hops away via CONTROL_FLOW edges. But the
typical reentrancy pattern
(ENTRY→CHECK→CALL→TMP→WRITE→RETURN) spans 5 nodes
— 4 hops. Phase 2 can only reach WRITE from CALL if there are
exactly 2 intermediate nodes. With 3 intermediate nodes
(CALL→TMP→ASSIGN→WRITE), the signal doesn't propagate.


5.3 fusion_layer.py

Failure Modes:
ID Failure Mode Description
FM5.3.
1
Key Norm Dominance
problem
Node projection: Linear(256, 256) maps GNN output which has
been through LayerNorm (norm ~1). Token projection:
Linear(768, 256) maps CodeBERT output which has norm
~10-15 (not normalized). Token keys dominate attention
scores, and node→token attention mostly attends to the
highest-norm tokens rather than the most semantically
relevant ones.

FM5.3.
2
Fusion eye dominance
risk
The fused output is one of three eyes — but
CrossAttentionFusion output_dim=128 while GNN and
Transformer eyes are also 128. If fusion learns faster (which it
does, as noted in RC1 fix), it can dominate the classifier before
the other eyes catch up. The aux loss with warmup mitigates
this but doesn't eliminate it entirely.


5.4 sentinel_model.py


Failure Modes:
ID Failure Mode Description
FM5.4.
1
O(N×K) pooling mask
computation
The function-level pooling mask uses torch.isin() which is O(N
× K) where N is total nodes and K=5 function type IDs. For
large graphs (N>1000), this is slower than a simple range
check: (node_type_ids >= 1) & (node_type_ids <= 6). This is a
minor performance issue but adds up across thousands of
training iterations.
FM5.4.
2
Ghost graph fallback
semantic error
Ghost graph fallback includes ALL nodes from graphs without
function nodes. This means interface-only contracts (which
have only CONTRACT and STATE_VAR nodes) are pooled over
state variables, producing a graph embedding dominated by
variable-type features rather than function-level behavior. An
interface contract should produce a near-zero embedding, not
a state-variable embedding.
FM5.4.
3
Dead neuron risk in
classifier
The classifier has a hidden layer of 192 with ReLU activation.
Combined with 0.3 dropout and ASL loss, this can cause the
"dead neuron" problem. If many neurons die during early
training (likely with the all-zeros collapse), the effective
capacity of the classifier drops, making it harder to recover.



6.1 trainer.py


Failure Modes:
ID Failure Mode Description
FM6.1.
1
ASL gradient
starvation loop
60% of training samples have zero labels. ASL with
gamma_neg=4 down-weights easy negatives by (1-p)^4. For a
near-zero prediction p≈0.05, the negative weight is (0.05)^4
= 0.00000625 — effectively zero gradient. The model quickly
learns all-zeros (p≈0.01) because this minimizes loss on clean
contracts with almost no gradient signal. Once in the all-zeros
basin, positive gradients from 40% vulnerable contracts are
insufficient to push predictions above the 0.35 threshold.
FM6.1.
2
Default loss_fn is BCE
not ASL
The TrainConfig has loss_fn="bce" as default, meaning anyone
running train.py without --loss-fn=asl will use
BCEWithLogitsLoss with pos_weight. This is a regression from
the v6 plan which called for ASL as the primary loss function.
FM6.1.
3
pos_weight_min_sampl
es disabled by default
pos_weight_min_samples defaults to 0 (disabled). The v6 plan
recommended 3000 to prevent Reentrancy's 2.82× FN penalty
from causing behavioral collapse. Without this, Reentrancy
gets amplified and dominates the gradient, suppressing
learning for other classes.
FM6.1.
4
eval_threshold=0.35 is
a band-aid
It lowers the F1 measurement threshold to avoid noise from
minority classes near 0.5, but doesn't help the model actually
learn those classes. It just makes the early stopping patience
less noisy without addressing the underlying learning problem.
FM6.1.
5
Insufficient warmup
steps
Gradient accumulation steps=1 means effective batch=16.
With 31,142 training samples, that is ~1,946 steps per epoch.
The OneCycleLR scheduler with warmup_pct=0.10 gives only
~195 warmup steps, which may be insufficient for the
2.4M-parameter GNN to find a stable initial direction.


FM6.1.
6
Aux loss warmup too
short
The aux loss warmup of 3 epochs is too short. The main loss
dominates for only 3 epochs before aux loss kicks in at full
weight (0.3). Given 3 auxiliary classification heads producing
independent logits, the combined aux gradient magnitude can
exceed the main loss gradient, especially early in training
when the main classifier hasn't learned anything useful yet.


6.2 losses.py (AsymmetricLoss)

Failure Modes:
ID Failure Mode Description
FM6.2.
1
CLIP threshold creates
hard gradient
boundary
The CLIP mechanism (p<0.05 → zero gradient) creates a hard
boundary. When the model predicts p≈0.03 for a class (slightly
below the clip threshold), the negative gradient is zeroed
entirely. This creates oscillation at the boundary rather than
smooth discrimination. Predictions below 0.05 get no negative
gradient and predictions above 0.05 get full negative gradient.
FM6.2.
2
No class-wise gamma
or clip parameters
All 10 classes share the same gamma_neg and clip. But DoS (3
training samples) needs much more aggressive positive
mining than Reentrancy (3,500 samples). A class-adaptive ASL
with per-class gamma and clip parameters would be more
appropriate for the extreme imbalance profile of this dataset.



8.1 The Patch-and-Pray Anti-Pattern
The three feature bugs (BUG-1/2/3) were fixed via patch_graph_features.py — an in-place .pt file
modifier. This approach has a fundamental problem: it treats the symptom (wrong values on
disk) without fixing the cause (no write-time validation in the extraction pipeline). The next time
graph_extractor.py is modified and re-extraction is run, any new bugs will again produce corrupt
.pt files that must be patched post-hoc. There is no structural guarantee of feature correctness.
This anti-pattern is particularly dangerous because it creates a false sense of security: the
existing files are patched and correct, but the pipeline that produced them is still capable of
generating incorrect data. Every code change to the extraction pipeline must be accompanied
by a manual verification step, and there is no automated mechanism to catch regressions. The
patch script itself is also a maintenance burden — it must be updated every time the feature
schema changes, and it operates on a specific file format that could become incompatible with
future versions of PyTorch's serialization format.
8.2 The 72% Featureless Node Problem
48,175 out of 66,288 nodes (72%) are CFG nodes with 9 of 12 features hardcoded to 0.0. This
means the GNN's 12-dim input is effectively 3-dim for most nodes. The GAT attention
mechanism computes query/key dot products over 12 dimensions, but 9 of those dimensions
are constant zero — they contribute nothing to the attention score and waste representational
capacity. This is the single biggest architectural bottleneck in the system. The impact extends
beyond just the attention computation: the learnable linear projections in each GAT layer
allocate parameters for all 12 input dimensions, but 75% of those parameters receive zero
gradient for CFG nodes, creating an inefficient parameter utilization pattern. The model's
effective capacity for understanding control flow — the core capability it was designed to
provide — is severely limited by this feature sparsity. Propagating parent function features to
CFG nodes would address this fundamentally, turning the 9 dead dimensions into meaningful
signals about loop structure, external call patterns, and return value handling.
8.3 The DoS Class Unlearnability
Only 7 pure-DoS contracts exist in the entire dataset. 98.1% of DoS=1 samples are also
Reentrancy=1. The model cannot learn to distinguish DoS from Reentrancy with only 7
contradictory examples. This is not a model architecture problem or a loss function problem — it
is a data problem that no amount of hyperparameter tuning can fix. The DoS class as currently
defined is effectively a subset of the Reentrancy class in the training data, and the model has no
way to learn the distinguishing features that separate genuine DoS vulnerabilities from
reentrancy patterns that also cause denial-of-service effects. The options are stark: either
merge DoS into Reentrancy (losing the class distinction entirely), remove DoS as a separate
class and focus the model's capacity on the 9 remaining classes, or invest in acquiring
significantly more pure-DoS training data (at least 300-500 samples) to make the class
learnable. Any training run that includes DoS as a separate class is wasting gradient signal on an unlearnable target, which can suppress learning for the other classes and lead to suboptimal overall performance.



8.4 The Timestamp Label Noise
48.2% of Timestamp=1 labels have no supporting evidence (no block global access in source,
no uses_block_globals feature activation). These are likely mislabelled in the BCCC dataset.
Training on noisy labels actively harms the model by teaching it to associate "no timestamp
signal" with "timestamp vulnerability," which is the opposite of the correct behavior. The model
learns to predict Timestamp vulnerabilities for contracts that have no timestamp-related code,
which will produce massive false positive rates in production. This is a particularly insidious form
of label noise because it doesn't just add random error — it systematically inverts the signal for
nearly half of the positive examples. The correct approach is to either remove the 1,056
mislabelled contracts from the training set or re-label them based on actual source code
evidence. Training on the current labels is worse than not training on Timestamp at all, because
the model actively learns the wrong association.
8.5 Hash-Based Pairing Fragility
The graph-token pairing relies on MD5 hashing of file paths. If the source directory structure
changes (e.g., BCCC-SCsVul-2024 is reorganized), all hashes change and the entire dataset
must be rebuilt. Content-based hashing (MD5 of source code) would be more robust but would
cause duplicate contracts (same source in different BCCC categories) to collide, which could
create label conflicts for multi-label training. The current path-based approach is deterministic
and collision-free by construction, but its fragility to directory restructuring creates a significant
operational risk. In practice, this means that any change to the source dataset layout — even a
simple rename of a parent directory — invalidates the entire cache and requires a full
re-extraction, which takes hours with Slither processing. A hybrid approach using both path and
content hashes would provide robustness against directory restructuring while maintaining the
ability to detect and handle duplicate contracts.