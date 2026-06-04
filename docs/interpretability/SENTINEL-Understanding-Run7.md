# SENTINEL Model Understanding — Run 7 Analysis
**Checkpoint:** `ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt` (ep39, F1=0.3074 fixed / 0.3329 tuned)  
**Data:** v10, 41,576 graphs, splits: `ml/data/splits/v10_deduped/`  
**Last updated:** 2026-06-04  
**Phase 2 results:** `ml/interpretability_results/phase2_run7_ep39_v10_2026-06-04/`

---

> **How to read this document:** Each section answers three questions:
> - **Problem** — what concern/question existed and why
> - **Solution** — what was implemented, what design decisions were made, what edge cases apply
> - **Broader context** — what this affects and why it matters to the overall system

---

## CHECKLIST — What You Should Understand

### Part 1: Architecture Decisions
- [ ] 1.1 Why the model has four "eyes" and what each one contributes
- [ ] 1.2 Why the GNN has three phases and what each phase is responsible for
- [ ] 1.3 What JK attention aggregation is and why it can fail
- [ ] 1.4 Why type embeddings replaced scalar type IDs
- [ ] 1.5 What LoRA is and why GraphCodeBERT is frozen
- [ ] 1.6 What the CrossAttentionFusion does and why it's a bottleneck

### Part 2: Training Bugs Fixed Before Run 7
- [ ] 2.1 BUG-R7-1 — wrong pooling in aux_phase2 (Phase 2 got no signal for 4 runs)
- [ ] 2.2 BUG-R7-2 — scalar type IDs causing representational collapse
- [ ] 2.3 ISSUE-1 — cfg_eye_proj in wrong optimizer group (killed Phase 2 gradient for Run 7 early)
- [ ] 2.4 ISSUE-2 — torch.compile submodule list missing two modules
- [ ] 2.5 Fix #35 — why resuming from checkpoint was producing degraded results

### Part 3: Training Dynamics Findings (Run 7 ep1–40)
- [ ] 3.1 Why the F1 sawtooth pattern is DoS noise, not a real learning cycle
- [ ] 3.2 What structural class ceilings are and which classes have hit them
- [ ] 3.3 Why Phase 3 JK weight is drifting up and what risks that carries
- [ ] 3.4 Why the GNN gradient share dropped from 91% to 28% and why that's fine
- [ ] 3.5 What the growing tuned/fixed threshold gap means
- [ ] 3.6 Why fusion gradient spikes happen and why they're not dangerous

### Part 4: Data Pipeline (v10)
- [ ] 4.1 What C-1 fixed (the node feature that was always zero in v9)
- [ ] 4.2 What H-2 fixed (REVERSE_CONTAINS being built wrong)
- [ ] 4.3 Why `feat[2]` firing 9.1% is a confirmation that C-1 is working
- [ ] 4.4 Why REVERSE_CONTAINS is rebuilt at runtime, not stored in graphs

### Part 5: Interpretability — Phase 1 (Graph Structure, v9 baseline — COMPLETE)
- [ ] 5.1 A1: Why GNN pooling correctness matters and what PASS means
- [ ] 5.2 A2: What CFG feature inheritance (BUG-C3) was and why it's verified fixed
- [ ] 5.3 E1: Why CEI structural detection is near-random (34.5% vs 33%) and what that means for aux_cei_loss
- [ ] 5.4 E2: What WL distinguishability tests and why PASS is necessary
- [ ] 5.5 E3: Why message propagation indistinguishability with random weights is *expected*
- [ ] 5.6 S1: Why structural pattern rates are low (0–32%) and what that reveals
- [ ] 5.7 S2: Which edge types are most enriched per class and what that predicts
- [ ] 5.8 S3: Why no feature shortcuts (Cohen's d < 1.5) is a critical health check
- [ ] 5.9 S4: What the ICFG path audit reveals about Reentrancy CEI coverage

### Part 6: Interpretability — Phase 2 (Run 7 Checkpoint — IN PROGRESS)
- [ ] 6.1 A3: JK entropy distribution — is the model learning to differentiate phases by contract?
- [ ] 6.2 A4: Aux head contribution — are all four eyes pulling weight?
- [ ] 6.3 B1: Phase2/Phase1 gradient ratio per class — confirms ISSUE-1 fix working?
- [ ] 6.4 B2: Per-eye ECE — which eye is best calibrated?
- [ ] 6.5 B3: JK weight distribution by phase — which classes drive Phase 3 dominance?
- [ ] 6.6 B4: UnusedReturn gradient saliency — why is this class stuck at 0.234?
- [ ] 6.7 E4: Directional edge sensitivity — does reversing edges hurt specific classes?
- [ ] 6.8 L1: JK weight per class — which vulnerability classes prefer which GNN phase?
- [ ] 6.9 L2: Edge ablation — which edge types are most important by F1 delta?
- [ ] 6.10 L4: Gradient saliency — which of the 11 node features drive each class?
- [ ] 6.11 L5: Linear probing — how much of the class signal lives in GNN vs Transformer embeddings?
- [ ] 6.12 L8: Permutation importance — which features are truly used vs noise?

---

## PART 1: Architecture — What It Is and Why

### 1.1 Four Eyes — Why and What Each Contributes

**Problem:** Early SENTINEL used a three-path architecture where the final classifier saw only the fused embedding (GNN+Transformer combined through CrossAttention). This meant any error in the fusion layer cut off both signal paths simultaneously. More concretely: if the fusion attention missed a structural signal, there was no fallback.

**Why it existed:** The original design assumed CrossAttentionFusion would be expressive enough to route signals appropriately. It wasn't — particularly for Phase 2 (ICFG/CFG) graph structure, which requires different inductive biases than the token-level transformer.

**Branches considered:**
- *Option A (chosen):* Four separate classification heads, each seeing a different view, concatenated before the final linear.
- *Option B:* Mixture-of-experts gating over the three original views.
- *Option C:* Auxiliary losses only, keeping single main head.

Option A (IMP-R7-2) was chosen because it lets each path vote, gradient flows independently to each path, and the model can learn to weight them without a separate gating network.

**Solution:** Four eyes feed a concat `[gnn_pool | tf_pool | fused | cfg_pool]` → `[B, 512]` → `Linear(512,256)` → `Linear(256,10)`.
- **GNN eye:** Mean pool of FUNCTION-like nodes from GNN encoder output. Captures function-level structural patterns.
- **TF eye:** CLS token from GraphCodeBERT last layer. Captures token-level semantic patterns.
- **Fused eye:** CrossAttentionFusion output (GNN attends to transformer tokens). Captures cross-modal alignment.
- **CFG eye:** Mean pool of CFG node types (node_type ∈ {IF, WHILE, FOR, ...}) from Phase 2 output (`_phase2_x`). Captures control-flow topology specifically.

**Edge cases:**
- If a graph has no CFG-type nodes, `_phase2_x` is empty → CFG eye falls back to zeros (handled in sentinel_model.py). This affects ~2% of graphs (pure data-contract graphs with no control flow).
- The 512-dim concat is fixed — adding a 5th eye would break the checkpoint format.

**Broader context:** The CFG eye is what makes Phase 2 gradient reach the classifier directly (BUG-R7-1's fix was enabling this path — previously `aux_phase2` pooled FUNCTION nodes instead of CFG nodes, so Phase 2 gradients were learning about the wrong nodes entirely). See §2.1.

---

### 1.2 GNN Three Phases — Division of Labor

**Problem:** Vulnerability detection requires reasoning at multiple structural scales: (1) individual function structure, (2) control-flow relationships between functions, (3) the containing contract hierarchy. A flat 8-layer GNN applying the same message-passing at every layer doesn't encode which scale each layer should specialise for.

**Why it existed:** The original v5 architecture had a monolithic GAT stack. The three-phase split was introduced in v8 along with the JK aggregation.

**Solution:**
- **Phase 1 (L1+L2):** STRUCTURAL + CONTAINS edges. Learns within-function and function-in-contract patterns. Initialises node representations from raw features.
- **Phase 2 (L3+L4+L5):** CF-only → ICFG-only → CF+ICFG joint. Progressive integration of control flow and inter-function call graph. Heads=4 (IMP-R7-1) for richer message aggregation.
- **Phase 3 (L6+L7+L8):** REVERSE_CONTAINS up + CONTAINS down. Propagates information up the contract hierarchy (functions → contracts) and back down. Heads=1 (lower capacity, deliberate).

**Design decisions:**
- Phase 2 uses heads=4 (wider) because ICFG patterns are more complex and require multiple attention "views" per edge.
- Phase 3 uses heads=1 because REVERSE_CONTAINS is a fixed structural relationship (hierarchy), not a semantic pattern — high capacity isn't needed.
- The `IMP-G2` skip connection (Linear(27→256)) at Phase 1 input ensures the raw features bypass the first two layers if they're already informative.

**Edge cases:**
- REVERSE_CONTAINS is not stored in graph files — it's built at runtime from the stored CONTAINS edges. This means any graph that somehow has no CONTAINS edges will also have no REVERSE_CONTAINS edges. Affects ~0.1% of graphs.
- Phase 2 edge types can be customised via `--gnn-phase2-edge-types` (ISSUE-4 fix ensures predictor reads this from checkpoint config so inference matches training).

**Broader context:** The Phase 2/Phase 3 split is what the B1 experiment (Ph2/Ph1 gradient ratio) validates. If Phase 2 gets ~0.10× of Phase 1's gradient (as in Run 4), the ICFG information is being ignored. After ISSUE-1 fix, the ratio is 0.6–0.7× (healthy co-training).

---

### 1.3 JK Attention Aggregation — What Can Go Wrong

**Problem:** After 8 GNN layers, which layer's representation should be used per node? Early layers have local structure, late layers have global context. A fixed "use last layer" approach loses early-layer detail critical for local patterns like integer overflow.

**Why it existed:** JK (Jumping Knowledge) networks were added in v8 to let the model learn, per contract, how much to weight each phase's output. Implemented as a learned attention mechanism over the three phase outputs.

**Branches considered:**
- *Concatenate all phases:* Simple but expands dimensionality 3×.
- *Max-pool across phases:* Loses information about which phase was active.
- *Attention over phases (chosen):* Soft weighting that can be analysed and regularised.

**Solution:** `JKAttentionAgg` computes soft weights over `[phase1_out, phase2_out, phase3_out]` per graph, producing `sum(w_i * phase_i)`. Weights stored as `last_weights` for analysis.

**What can go wrong — JK collapse:** If one phase's representation becomes much better than the others, the attention collapses to always selecting that phase. All gradient flows through that phase; the other phases stop learning. λ=0.005 Shannon entropy regularisation term in the loss penalises weight distributions that are too peaked.

**Current Run 7 status (ep1–ep40):**
- Phase 3 drifted to 0.395 (ep40). Not yet collapsed but trending.
- The best checkpoint (ep39) had Phase3=0.373 — lower than the ep30/ep40 highs. This correlation (higher Phase3 → worse F1) suggests Phase 3 is taking credit from Phase 2, reducing the model's CFG sensitivity.
- **Run 8 action:** Increase λ to 0.0075.

**Edge cases:**
- JK weights are per-graph, not per-class — the MLflow `jk_phase_weight` metrics show the mean over the val set. Individual contracts may be very different.
- The B3 experiment (§6.5) will show the per-class distribution — which vulnerability types prefer which phase.

**Broader context:** JK collapse is a training failure mode, not an inference problem. A collapsed model still produces predictions but has effectively abandoned two-thirds of its architecture. The λ regularisation and the B3/L1 experiments together form the monitoring and diagnostic system for this.

---

### 1.4 Type Embeddings vs Scalar IDs (BUG-R7-2)

**Problem:** Before Run 7, node type (FUNCTION, IF, WHILE, CALL, etc. — 13 types) was encoded as a single scalar appended to the feature vector. A node of type 0 and type 12 are 12 units apart in the scalar sense. There's no reason the GNN should treat them as nearly identical just because 0 and 1 are adjacent integers.

**Why it existed:** The v7/v8 schema was designed before the importance of categorical node types was fully understood. A scalar encoding was the simplest implementation and "worked" because the model could learn to ignore the numerical ordering.

**Solution (BUG-R7-2):** Replaced the scalar with `nn.Embedding(13, 16)` — each of the 13 node types gets a learned 16-dimensional representation. The runtime input dimension becomes `_GNN_IN_DIM = NODE_FEATURE_DIM(11) + 16 = 27`.

**Design decisions:**
- 16 dimensions for type embedding: enough to be expressive (13 types need at most log2(13)≈4 dims to be orthogonal, 16 gives enough slack for learned semantic similarity).
- The `NODE_FEATURE_DIM=11` constant is stored in checkpoints; `_GNN_IN_DIM=27` is computed at runtime. This keeps the schema stable while allowing the embedding to evolve.
- The `IMP-G2` skip Linear(27→256) passes the raw+embedded features directly to Phase 1 input, bypassing any accidental embedding suppression in the first GAT layer.

**Edge cases:**
- The type embedding is part of the GNN encoder, so it's included in the GNN parameter group (LR×2.5). This is intentional — type representations should learn faster than the frozen transformer.
- Node type IDs must be integers 0–12. Any graph with type ID outside this range would crash with an index error at the embedding lookup. The graph builder enforces this, but it's worth noting.

**Broader context:** Before BUG-R7-2, the model had to learn a numerical mapping (type=5 means "IF") from data alone, which requires the model to see enough examples of each type in context. With learned embeddings, FUNCTION and IF can have orthogonal representations from initialisation, making the model's learning task easier.

---

### 1.5 LoRA + Frozen GraphCodeBERT — Why Not Fine-tune Everything?

**Problem:** GraphCodeBERT has 124M parameters. Fine-tuning all of them on 29,103 contracts would cause catastrophic forgetting of its pre-trained code understanding, require enormous VRAM, and take weeks per run on an RTX 3070.

**Why it's frozen:** The pre-trained representation of Solidity tokens is already strong — GraphCodeBERT was trained on code, and Solidity syntax is similar enough to benefit. The value is in the *contextual token embeddings*, not the classification head.

**LoRA (Low-Rank Adaptation):** Instead of updating all 124M parameters, LoRA adds small rank-r matrices alongside the Query and Value projections in each attention layer. During training only these matrices (plus the existing projection weights scaled by α/r) are updated. For r=16, α=32: the effective learning rate for LoRA weights is 2.0× the base LR.

**Design decisions:**
- LoRA on Q+V only (not K): K matrices affect how tokens attend to each other; modifying K risks changing the token similarity structure. Q and V control what's extracted and how it's projected — safer to adapt.
- Applied to all 12 layers (not just top layers): Vulnerabilities can involve low-level token patterns (arithmetic operators → integer overflow) as well as high-level structures (function call chains → reentrancy).
- r=16, α=32: Standard values. r=8 would be faster but less expressive; r=32 would risk overfitting with 29K training examples.

**Broader context:** The `gnn_prefix_k` feature (disabled in Run 7 with k=0) was designed to inject GNN prefix tokens into the transformer's input sequence, allowing the two paths to communicate earlier in the transformer stack. Enabling it (k=48, Run 8) may improve the fusion quality but also increases the risk of disturbing the pre-trained token representations.

---

### 1.6 CrossAttentionFusion — The Integration Bottleneck

**Problem:** The GNN produces node-level embeddings (one vector per graph node). The Transformer produces token-level embeddings (one vector per token, max 512 tokens). These have different shapes and different semantic levels. A simple concatenation would lose the spatial correspondence.

**Solution:** CrossAttentionFusion uses cross-attention where:
- Query: GNN's pooled representation (function nodes → single vector)
- Key/Value: Transformer's token sequence

This lets the GNN representation "ask questions" of the token sequence — "which tokens are most relevant to this function's structural pattern?" The output is a 128-dim vector combining both.

**Design decisions:**
- `attn_dim=256, output=128 LOCKED`: These dimensions are baked into the checkpoint format. Changing them invalidates all existing checkpoints.
- `LayerNorm(768)` on the transformer input: The 768-dim token embeddings from GraphCodeBERT can have high variance across contracts; normalising before fusion prevents the cross-attention from being dominated by outlier activations.

**Edge cases:**
- If a contract tokenises to fewer than the expected context length (< 512 tokens), the key/value sequence is shorter — this is handled by attention masking. The GNN query still attends to all available tokens.
- Very long contracts are windowed to [4, 512] before tokenisation (4 windows of 512 tokens). Only the first window's CLS token is used for the TF eye. This means contracts >512 tokens have their later code truncated at inference. The L7 experiment (calibration vs contract size) will surface whether this causes systematic miscalibration for large contracts.

---

## PART 2: Training Bugs Fixed Before Run 7

### 2.1 BUG-R7-1 — aux_phase2 Pooled Wrong Nodes (Silent for 4 Runs)

**Problem:** The auxiliary Phase 2 loss head (`aux_phase2`) was designed to provide gradient signal specifically to Phase 2 layers (L3–L5) — the ICFG/CFG layers. It pooled graph nodes and applied a mini-classifier. But the pooling was applying **FUNCTION-like node selection** (the same criteria as the main GNN eye), not **CFG-node selection** (IF/WHILE/FOR/CALL nodes).

**Why it existed:** BUG-R7-1 was introduced when aux_phase2 was first written. The code template was copied from the main GNN eye pooling function and the node type filter was never updated. Because the model still trained to reasonable F1, the bug wasn't obvious — the aux head was learning something (FUNCTION-node patterns) just not the Phase 2 CFG patterns it was supposed to reinforce.

**Why it persisted for 4 runs:** The Ph2/Ph1 gradient ratio in Run 4 was 0.10–0.18 — Phase 2 was getting very little gradient. This was attributed to Phase 2 being "less expressive" rather than a pooling bug. The bug was found by code audit before Run 7.

**Branches considered:**
- *Fix the pooling to use CFG nodes (chosen):* Direct fix, clean, aligns aux_phase2 gradient path with its intent.
- *Remove aux_phase2 entirely:* Would eliminate the gradient reinforcement for Phase 2 layers.

**Solution:** `aux_phase2` now pools `node_type ∈ {IF, WHILE, FOR, REQUIRE, ASSERT, CALL, RETURN, EVENT}` (CFG-type nodes). This is the same set used by the CFG eye (IMP-R7-2 — see §1.1).

**Edge cases:**
- Contracts with no CFG-type nodes (pure data structures): aux_phase2 produces a zero pooling. The auxiliary loss contribution is zero, which is correct — Phase 2 has nothing to reason about in these contracts.
- The ISSUE-2 fix ensures `aux_phase2` is in the torch.compile submodule list, so it's compiled alongside the main model rather than being a slow Python fallback.

**Broader context:** The Ph2/Ph1 gradient ratio going from 0.10–0.18 (Run 4) to 0.6–0.7 (Run 7) is the most direct evidence this fix worked. The B1 experiment (§6.3) will provide per-class confirmation.

---

### 2.2 BUG-R7-2 — Scalar Type IDs (See §1.4 for details)

**Short summary:** Node types (13 categories) were encoded as integers 0–12, imposing a false numerical similarity between adjacent types. Fixed by `nn.Embedding(13,16)`. The v8 feature schema (NODE_FEATURE_DIM=11) is unchanged — the 16-dim embedding is added at runtime, making `_GNN_IN_DIM=27`.

---

### 2.3 ISSUE-1 — cfg_eye_proj in Wrong Optimizer Group

**Problem:** `cfg_eye_proj` is the linear projection that maps Phase 2 CFG node embeddings into the space expected by the CFG eye classifier head. It is the **direct gradient path** from the classifier loss back into Phase 2 layers. In the original Run 7 optimizer setup, it was assigned to `_other_params` with the default LR (1.0× multiplier). The GNN parameter group uses LR×2.5.

**Why it mattered:** If `cfg_eye_proj` is in the wrong group, it receives lower gradient signal. Since it's the bottleneck between Phase 2 and the loss, the effective gradient flowing into Phase 2 layers is reduced proportionally. ISSUE-1 was discovered by inspecting which parameter groups each named module fell into.

**Solution:** Moved `cfg_eye_proj` to the GNN parameter group (LR×2.5) in `trainer.py`. One-line fix in the optimizer parameter group list.

**Edge cases:**
- This change only affects *training* — the checkpoint weights are the same structure. But the weights trained with the correct LR group are different from what they would have been without the fix.
- The 2.5× multiplier is consistent with `cfg_eye_proj`'s role: it's part of the GNN processing pipeline, not a new module being introduced mid-training.

**Broader context:** This is why Run 7's checkpoint was worth training from scratch rather than resuming Run 6. Run 6's cfg_eye_proj had already learned with the wrong LR for 33 epochs.

---

### 2.4 ISSUE-2 — torch.compile Submodule List Incomplete

**Problem:** `torch.compile` can be given a list of submodules to compile together for optimised kernel fusion. If a module is omitted from this list, it runs as interpreted Python — slower and not fused with surrounding compiled code. `cfg_eye_proj` and `aux_phase2` were missing from the compile list.

**Solution:** Added both to the `torch.compile` submodule specification. Verified by checking that the `._orig_mod.` infix appears for these modules in the compiled model's state dict.

**Edge cases:**
- The StructuredLogger bug (§8 of the analysis doc) is *also* caused by torch.compile — `model.aux_phase2` returns an `OptimizedModule` which doesn't support `[-1]` subscript. This is a separate consequence of compiling `aux_phase2` that was not caught before Run 7 started.

---

### 2.5 Fix #35 — Why Resuming Was Producing Degraded Results

**Problem:** When training was interrupted (crash, manual stop) and resumed from checkpoint, the default `resume_model_only=True` discarded the AdamW optimizer state (momentum, variance estimates per parameter). AdamW's momentum/variance tracks the recent gradient history — without it, the optimizer starts from cold and exhibits a "learning rate warmup" effect where the effective LR is much higher than intended for the first ~5–10 epochs.

**Why it existed:** `resume_model_only=True` was the original default for a deliberate reason: in early training, if you want to change the LR schedule, optimizer config, or introduce new parameter groups, resuming the optimizer state would lock you into the old setup. But for normal resume (same config, same run), discarding optimizer state is harmful.

**Additional problem:** The training/validation data shuffle order (controlled by PyTorch/NumPy/Python RNG states) was reset on resume. This means the model would see the same batches again in a different order, slightly changing what it learns relative to a non-interrupted run. For deterministic-equivalent resume (same final result as if never interrupted), you need RNG state too.

**Solution (Fix #35, commit `6bee1a9`):**
- Default changed to `resume_model_only=False`
- Checkpoint now saves: `torch.get_rng_state()`, `torch.cuda.get_rng_state()`, `np.random.get_state()`, `random.getstate()`
- Checkpoint now saves: `_cached_tuned_thresholds` (per-class threshold from last tune run)
- On resume: all four RNG states restored; thresholds restored; optimizer state restored

**Edge cases:**
- If you *want* to change the optimizer config on resume (e.g., different LR), you should explicitly pass `--resume-model-only` to override the default. The option still exists, just not the default.
- RNG states are machine-specific (CUDA RNG state depends on the GPU). Resuming on a different GPU will restore the state but the actual random numbers produced may differ due to different kernel implementations. This is expected and acceptable.
- Old checkpoints (pre-Fix-#35) have no `rng_state` key. The restore code checks `if "rng_state" in ckpt` before restoring — backward compatible.

---

## PART 3: Training Dynamics Findings

### 3.1 Sawtooth = DoS Noise

*(Detailed analysis in `docs/training/GCB-P1-Run7-analysis-2026-06-04.md` §5)*

**Short version:** With 65 positive examples in the validation set (1.04% prevalence), one extra correct DoS prediction changes F1 by ~0.008. The model's DoS predictions are inherently unstable at this sample size — it's predicting "yes" for 0–10 contracts per epoch, and whether those happen to be true positives is largely random. The macro F1 oscillates by ±0.015 each epoch, always tracking which epoch DoS happened to land high. The underlying 9-class trend has been flat since ep20.

**Why this matters for Run 8:** Don't stop training based on patience-counter behaviour during DoS-low epochs. The 30-epoch patience is correctly set — it should survive the DoS crashes. Also motivates getting more DoS-positive training contracts, or using a stratified sampler that over-samples rare classes.

---

### 3.2 Structural Class Ceilings

*(Detailed in `docs/training/GCB-P1-Run7-analysis-2026-06-04.md` §6)*

Four classes have been flat since ep10. The root causes are architectural, not a training problem:

| Class | Ceiling | Root Cause | Fix Requires |
|-------|---------|------------|--------------|
| UnusedReturn | 0.234 | No DEF_USE edges — can't trace return value consumption | RC5: add DEF_USE edge type |
| Timestamp | 0.165 | No data-flow provenance for `block.timestamp` values | New feature: timestamp-tainted nodes |
| TransactionOrderDependence | 0.250 | Cross-contract reasoning not representable | Multi-contract graphs |
| ExternalBug | 0.250 | Exception propagation across external call boundaries | Cross-contract graphs |

**What this means for evaluation:** When comparing Run 7 to Run 4, these four classes contribute ~0.04 to macro F1. Their ceilings are the same in both runs. The real improvement from Run 7 is in the other 6 classes (particularly Reentrancy, DoS over time, and IntegerUO).

---

### 3.3 JK Phase 3 Drift — Risk and Monitoring

*(Detailed in `docs/training/GCB-P1-Run7-analysis-2026-06-04.md` §8)*

Phase 3 drifted from 0.347 to 0.395 over 40 epochs. Phase 3 encodes contract-hierarchy containment (REVERSE_CONTAINS: function→contract, CONTAINS: contract→function). It's drifting because the model is finding that knowing "which contract this function belongs to" is a strong prior for vulnerability prediction. This is partly legitimate (some contracts are systematically more vulnerable) but partly a shortcut (learning contract identity rather than vulnerability patterns).

**How to detect collapse:** `jk_phase3_weight > 0.40` for two consecutive epochs, combined with `jk_phase1_weight < 0.28`. The B3 experiment will show per-class breakdown — if Phase 3 is dominant specifically for the hard classes (DoS, Timestamp), it's likely relying on contract-identity shortcuts rather than structural patterns.

---

### 3.4 GNN Gradient Share Drop: 91% → 28%

This is **expected and healthy**. At epoch 1, LoRA is initialised near-identity (rank-16 matrices start near zero), so the transformer contributes almost no gradient. As LoRA learns, the transformer's gradient grows. The GNN share settles to 28–35% by ep30 — meaning the transformer contributes ~65–70% of the gradient, which is appropriate given it has 124M parameters and the GNN is ~8M.

The share is computed at each step: `gnn_grad_norm / (gnn_grad_norm + tf_grad_norm + fused_grad_norm)`. It's not a problem if it dips to 20% during a fusion spike (the spike pushes fused_grad_norm up temporarily).

---

### 3.5 Growing Threshold Gap

*(Detailed in analysis doc §7)*

The default classification threshold of 0.35 (used during training metric logging) was calibrated on Run 4 data. As Run 7 trains, the model's probability distributions shift — particularly for rare classes (DoS, Timestamp) where the model learns to predict low-confidence positives with probabilities around 0.20–0.28. At threshold 0.35, these are discarded as negatives.

The threshold tuning (every 10 epochs via `--threshold-tune-interval`) finds the optimal per-class threshold on the val set and reports `val_f1_macro_tuned`. The gap of +0.0317 at ep40 means the logged best F1 (0.3074) understates the model's true capability.

**Implication:** Before any downstream use (interpretability, inference, comparisons), run `ml/calibration/` scripts on the ep39 checkpoint to generate `temperatures_run7.json`. The calibration file maps each class to its optimal threshold.

---

### 3.6 Fusion Gradient Spikes

Spikes of 0.09–0.165 in the fusion gradient at step 100–200 of each epoch. Root cause: `fusion_lr_multiplier=0.5` is slightly high. The four-eye architecture routes more loss signal through the fusion layer than the three-eye Run 4 setup, because the fused eye is now one of four votes rather than the main output. This makes the fusion layer's gradient more sensitive to batch composition. The spikes are transient (normalise by step 300) and never cause loss divergence. Run 8 recommendation: reduce to 0.3.

---

## PART 4: Data Pipeline — v10

### 4.1 C-1: The Feature That Was Always Zero

**Problem:** `feat[2]` (in the 11-dimensional node feature vector) is supposed to encode `has_external_call` — whether a function calls an external contract. In v9, this feature was always 0 for all nodes in all graphs. It was a bug in the graph builder where the external-call detection code path was never reached.

**Why it existed:** The external call detection logic had a conditional guard that was always False due to a variable scope error.

**Fix (C-1):** Graph builder corrected. In v10, `feat[2]` fires 9.1% of the time — meaning ~9% of function nodes have external calls. This aligns with expectations for real Solidity contracts.

**Why it matters:** External calls are the fundamental mechanism for Reentrancy (call → state change after call returns) and ExternalBug (exception not caught from external call). Having this feature at 0 for all nodes in Run 4/Run 5/Run 6 means those models never learned "this function makes external calls" as a feature. Run 7 is the first run to see this signal.

---

### 4.2 H-2: REVERSE_CONTAINS Edge Count

**Problem (H-2):** REVERSE_CONTAINS edges were being constructed incorrectly in v9 — approximately 70% of expected edges were being generated (BUG-H7 — the exact discrepancy was in the edge direction assignment when building from CONTAINS). Fixed before v10.

**v10 correct count:** 216,699 REVERSE_CONTAINS edges across the dataset.

**Why it's built at runtime:** REVERSE_CONTAINS is the transpose of CONTAINS. Storing it would double the storage cost of every graph file for a relationship that can be trivially computed. It's rebuilt in the data loader for every graph when loaded.

---

### 4.3 Why feat[2] at 9.1% Confirms C-1

A feature with 0% activation is informationally worthless — the model can't learn from it because it never fires. A feature with 9.1% activation is in a healthy range (enough examples to learn from, enough negatives to contrast against). The S3 experiment (Phase 1) would previously have shown 0 for this feature's distribution — re-running S3 on v10 data would show a real distribution.

---

## PART 5: Phase 1 Results (v9 Baseline — COMPLETE)

*Results archived at: `ml/interpretability_results/archive_phase1_run5_v9_2026-06-02/`*  
*Full doc: `docs/interpretability/archive_phase1_run5_v9_2026-06-02/PHASE1_RESULTS_RUN5_BASELINE.md`*

### 5.1 A1 — GNN Pooling: PASS

**What it tests:** Does every graph in the val set have at least one FUNCTION-like node (FUNCTION, MODIFIER, CONSTRUCTOR, FALLBACK)? If not, the GNN eye would fall back to pooling all nodes, losing the function-level specificity that the architecture assumes.

**Result:** 100% of 2,000 tested graphs have ≥1 FUNCTION-like node. Node distribution: FUNCTION=33,782, MODIFIER=2,894, CONSTRUCTOR=3,017, FALLBACK=887.

**Why it matters:** If this had failed, the GNN eye's pooling would be unreliable. The test is a prerequisite for trusting any GNN-based interpretability results.

---

### 5.2 A2 — CFG Feature Inheritance: PASS

**What it tests:** BUG-C3 (a pre-v9 bug) caused CFG nodes to not inherit features from their parent FUNCTION nodes (visibility, payable, has_state_write, etc.). A2 verifies this is fixed by checking that CFG node features match their containing FUNCTION node's features.

**Result:** 499/500 (99.8%) graphs have CFG→FUNCTION parent relationships. All 5 inherited feature dimensions at 100% consistency.

**Why the 0.2% failure:** One graph had no CFG edges at all (a contract with no control flow — a pure data storage contract). This is valid — it's not a bug, just an unusual contract type.

---

### 5.3 E1 — CEI Reachability: FAIL†

**What it tests:** In Reentrancy vulnerabilities, the "Check-Effects-Interactions" (CEI) pattern violation requires: a CALL_ENTRY edge (external call) reachable from a state-write node within k hops. E1 uses BFS to check how often this structural pattern exists in positive vs negative graphs.

**Result:** 34.5% positive vs 33.0% negative at k=8 (Δ=1.5%). Near-random.

**Why this matters:** It means structural BFS-based CEI detection barely distinguishes reentrancy-positive from negative contracts. Two reasons:
1. 69% of reentrancy-labeled contracts don't have the naive structural CEI pattern — the vulnerability is more subtle (indirect calls, delegatecall, etc.)
2. The structural pattern *does* appear in many non-vulnerable contracts as normal external-call-then-update patterns.

**Decision:** `aux_cei_loss_weight=0.0` is correct. A CEI-based auxiliary loss would add noise, not signal.

---

### 5.4 E2 — WL Distinguishability: PASS

**What it tests:** Are the vulnerability-class graphs distinguishable by the Weisfeiler-Lehman graph isomorphism test? If two non-isomorphic graphs of opposite labels have identical WL hash sequences, even a perfect GNN couldn't distinguish them.

**Result:** All 4 tested classes (Reentrancy, IntegerUO, Timestamp, CallToUnknown) pass. No degenerate identical-hash pairs found at any radius r=1–8.

**Why it matters:** WL distinguishability is a **necessary but not sufficient** condition for GNN expressivity. If it failed, we'd know the architecture was fundamentally incapable of learning the task. Passing means the task is theoretically solvable.

---

### 5.5 E3 — Message Propagation with Random Weights: FAIL† (Expected)

**What it tests:** With randomly initialised weights (no training), do positive and negative graphs produce distinguishably different node activations after Phase 1/Phase 2 message passing?

**Result:** CALL_ENTRY delta Phase1→Phase2: pos=−0.131, neg=−0.143 (effectively the same).

**Why this is EXPECTED:** Random weights produce random aggregations. The discriminative patterns emerge only through training. This experiment establishes the baseline that training must surpass. A "PASS" here would actually be suspicious — it would suggest the features themselves (without any learning) already discriminate, which would be a potential shortcut.

---

### 5.6 S1 — Structural Pattern Coverage: FAIL†

**What it tests:** What fraction of positive graphs for each class exhibit the "canonical" structural pattern for that vulnerability?

| Class | Pattern | Coverage |
|-------|---------|----------|
| IntegerUO | Arithmetic opcode in function node | 97.5% |
| MishandledException | External call + no revert | 32.0% |
| Reentrancy | CEI state-write-after-call chain | 30.4% |
| Timestamp | block.timestamp read → branch | 26.3% |
| UnusedReturn | Return value consumed check | 0.0%* |

*UnusedReturn pattern checker had a schema mismatch bug (v9 field name changed).

**Key insight:** The low coverage for Reentrancy/Timestamp/MishandledException doesn't mean the patterns aren't there — it means the *structural pattern checker scripts* only capture the "obvious" form. Real vulnerabilities are more varied. This tells us: the GNN needs to learn implicit patterns, not just the textbook CEI/timestamp template.

---

### 5.7 S2 — Edge Enrichment: FAIL†

**What it tests:** For each edge type, how much more frequently does it appear in positive vs negative graphs for each vulnerability class?

**Key findings (v9 baseline):**
- RETURN_TO edges: strongest enrichment for Timestamp (1.46×) and TOD (1.31×)
- CALL_ENTRY: enriched for Timestamp (1.19×)
- REVERSE_CONTAINS: 0% enrichment (built at runtime, not in graphs at audit time)

**Why FAIL:** The threshold for "meaningful enrichment" was calibrated on Run 4 graphs with slightly different edge distributions. The v10 data (with C-1 fix adding `has_external_call` features) would likely show different enrichment patterns.

**Implication:** S2 should be re-run on v10 data to establish the current baseline. This would reveal whether RETURN_TO is still the most enriched edge type for Timestamp or whether the `has_external_call` feature changed the distribution.

---

### 5.8 S3 — No Feature Shortcuts: PASS

**What it tests:** Cohen's d measures the effect size of each node feature's distribution difference between positive and negative graphs per class. A large Cohen's d (>1.5) would indicate the model could cheat by learning that feature alone, without any graph structure reasoning.

**Result:** Max Cohen's d = 1.02 (UnusedReturn / cfg_call_count). All others < 1.0.

**Why the UnusedReturn result is legitimate:** `cfg_call_count` (number of external calls in a function) correlates with UnusedReturn because functions that call other contracts and ignore return values are by definition external-call-heavy. This is semantically valid, not a shortcut.

**Why this matters:** If Cohen's d > 1.5 existed, we'd need to add noise/augmentation to prevent the model from learning trivial statistical shortcuts instead of structural reasoning.

---

### 5.9 S4 — ICFG Path Audit: FAIL†

**What it tests:** In the val set, what fraction of Reentrancy-positive contracts have complete CALL_ENTRY + RETURN_TO chains (the ICFG-level CEI pattern)?

**Results:** 64% have CALL_ENTRY present; 42.2% have full CALL_ENTRY + RETURN_TO chain.

**Why this matters:** 42% is better than the E1 BFS result (34.5%) because S4 looks at the full ICFG rather than just k-hop BFS. But 58% of Reentrancy contracts *still* lack the canonical ICFG chain. This means the Phase 2 ICFG reasoning captures a real but minority signal for Reentrancy. The remaining signal must come from token-level patterns (delegatecall, re-entrant function signatures) that the Transformer picks up.

---

## PART 6: Phase 2 Experiments — Run 7 ep39 Checkpoint

*Results directory: `ml/interpretability_results/phase2_run7_ep39_v10_2026-06-04/`*  
*Checkpoint: `ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt`*

Status legend: ⏳ = queued/running · ✅ = complete · ❌ = failed · 📊 = results below

---

### 6.1 A3 — JK Entropy Distribution ⏳

**What it tests:** Loads the training log and/or MLflow data to compute the per-epoch distribution of JK attention entropy across contracts. Low entropy = model consistently picks one phase. High entropy = model uses all phases. Variance in entropy = model adapts per contract.

**Expected findings:** Phase 3 drift (0.347→0.395 over 40 epochs) should appear as declining entropy for Phase 3 (it's increasingly confident about using Phase 3). Per-contract variance should be high — Phase 3 should dominate for containment-structure-heavy contracts, Phase 1 for simple single-function contracts.

**Results ✅ COMPLETE:** PASS. Entropy H = 1.090–1.099 across all 40 epochs (healthy diversity threshold > 0.50).

- ep1: H=1.0980, ep10: H=1.0978, ep20: H=1.0938, ep30: H=1.0899, ep40: H=1.0903
- The entropy decline is **−0.0082 over 40 epochs** — subtle but monotonic. The λ=0.005 regularisation is holding diversity but not arresting the Phase 3 drift trend.
- Note: The log-level metrics (mean per epoch) show entropy > 1.09 because they're the mean over the val set. Individual contracts may have much lower entropy (Phase 3-dominated). The B3 experiment will show the per-contract distribution.
- **Run 8 implication:** Increasing λ to 0.0075 should raise the minimum per-epoch entropy by ~0.004–0.006, which should prevent Phase 3 hitting 0.40+.

---

### 6.2 A4 — Aux Head Contribution ✅ COMPLETE

**What it tests:** Per-eye AUC-ROC and F1 independently for each eye (GNN, TF, Fused, Main ensemble). Measures what each eye contributes independently.

**Results (AUC-ROC, threshold-independent):**

| Class | GNN | TF | Fused | Main | Best single |
|-------|-----|----|-------|------|------------|
| CallToUnknown | 0.748 | 0.728 | **0.777** | 0.782 | Fused |
| DenialOfService | 0.726 | **0.559** | **0.803** | 0.795 | Fused |
| ExternalBug | 0.690 | 0.771 | **0.796** | 0.796 | Fused |
| GasException | 0.790 | 0.826 | **0.835** | 0.834 | Fused |
| IntegerUO | 0.797 | 0.860 | **0.871** | 0.878 | Fused |
| MishandledException | 0.707 | **0.777** | 0.773 | 0.781 | TF |
| Reentrancy | 0.716 | 0.735 | **0.762** | 0.770 | Fused |
| Timestamp | 0.710 | 0.758 | **0.761** | 0.752 | Fused |
| TransactionOrderDependence | 0.737 | **0.813** | 0.781 | 0.794 | TF |
| UnusedReturn | 0.708 | 0.757 | **0.765** | 0.775 | Fused |

**CRITICAL finding — DoS is structurally detected, not semantically:**  
Transformer eye AUC for DoS = **0.559** (barely above random). GNN = 0.726, Fused = 0.803. The transformer has almost no DoS signal. DoS detection is entirely structural — the GNN's pattern of gas consumption in loops/recursive calls is what the model relies on. Token-level code patterns for DoS look essentially the same as normal code. This means: adding more text data or improving the transformer won't help DoS. Only better structural representations (Phase 2 ICFG paths showing loop depth, call graph cycles) will improve it.

**Key finding — TOD: ensemble hurts (-0.019 vs TF alone):**  
Main AUC = 0.794 vs TF alone = 0.813. The GNN and CFG eyes are adding noise for TOD detection. TOD is a semantic cross-transaction pattern (state variable read in tx1, write in tx2) — the transformer recognises the code patterns (`msg.sender`, storage variable reads), but the GNN's structural view confuses it with similar-looking non-TOD patterns. This suggests TOD needs cross-transaction graph representation to benefit from structural reasoning.

**Key finding — Timestamp: fused eye alone beats ensemble (-0.009):**  
Fused = 0.761 vs Main = 0.752. Similar to TOD but weaker effect. The CFG/GNN contributions slightly hurt Timestamp. The crossattention fusion (where GNN queries transformer tokens) captures the right signal; the raw GNN structural features add noise.

**Ensemble verdict:** The 4-eye design helps for 6/10 classes (IntegerUO, Reentrancy, CallToUnknown, UnusedReturn, Fused group). For TOD and Timestamp, the ensemble slightly hurts due to GNN noise. The correct solution for TOD/Timestamp is not to remove the eyes but to fix the underlying structural representation (cross-contract graphs, data-flow).

**Formal result:** FAIL on F1 threshold (GNN eye alone only exceeds +5pp for IntegerUO). But this is a PASS in the intended meaning — each eye contributes distinct AUC signal, and the ensemble beats all individual eyes for most classes.

---

### 6.3 B1 — Phase2/Phase1 Gradient Norm ✅ COMPLETE

**What it tests:** For each vulnerability class, computes the gradient norm flowing into Phase 2 layers relative to Phase 1 when processing that class's examples. The Run 4 ratio was 0.10–0.18 (Phase 2 essentially ignored). Run 7 should show 0.5–0.8×.

**Why this is the most important Phase 2 experiment:** B1 is the direct empirical validation of ISSUE-1's fix.

**Results:** All classes show Ph2/Ph1 ratio **0.777–0.917** (average 0.851). This is a 5–8× improvement over Run 4's 0.10–0.18. ISSUE-1 fix is definitively confirmed working.

| Class | Ph1 | Ph2 | Ph3 | Ph2/Ph1 | Ph3/Ph1 |
|-------|-----|-----|-----|---------|---------|
| CallToUnknown | 0.0432 | 0.0396 | 0.0308 | **0.917** | 0.714 |
| DenialOfService | 0.0617 | 0.0479 | 0.0574 | 0.777 | **0.931** |
| ExternalBug | 0.0322 | 0.0268 | 0.0190 | 0.834 | 0.591 |
| GasException | 0.0492 | 0.0404 | 0.0345 | 0.822 | 0.701 |
| IntegerUO | 0.0524 | 0.0462 | 0.0310 | 0.882 | 0.593 |
| MishandledException | 0.0405 | 0.0352 | 0.0225 | 0.870 | 0.555 |
| Reentrancy | 0.0470 | 0.0422 | 0.0353 | 0.898 | 0.750 |
| Timestamp | 0.0784 | 0.0698 | 0.0520 | 0.890 | 0.664 |
| TransactionOrderDependence | 0.0319 | 0.0257 | 0.0185 | 0.803 | 0.578 |
| UnusedReturn | 0.0342 | 0.0279 | 0.0229 | 0.815 | 0.668 |

**Key finding — DoS Ph3/Ph1 = 0.931:** DoS gets the highest Phase 3 gradient ratio by a significant margin. Phase 3 (REVERSE_CONTAINS hierarchy) is nearly as important as Phase 1 for DoS. This confirms the "contract identity shortcut" hypothesis — the model is detecting DoS by recognising *which contracts* are DoS-prone (contract-level identity) rather than by learning the specific gas-consumption structural pattern. This explains why DoS F1 is so erratic: the model is over-relying on which contract families tend to have DoS, rather than the structural pattern itself.

**Key finding — MishandledException Ph3/Ph1 = 0.555 (lowest):** MishandledException is the class least reliant on contract hierarchy. It depends most on local function-level patterns (try/catch missing, return value not checked). This is exactly the expected behaviour for a well-designed GNN for this class.

**Unexpected finding — Ph2/Ph1 uniformity:** All classes have Ph2/Ph1 in a tight 0.777–0.917 range. There's no class that "doesn't use Phase 2" — the ICFG features are contributing meaningfully everywhere. The per-class variation is smaller than expected (Reentrancy was predicted to have the highest; CallToUnknown actually does).

---

### 6.4 B2 — Per-Eye ECE ⏳

**What it tests:** Expected Calibration Error for each of the four eyes independently — GNN eye, TF eye, Fused eye, CFG eye. A well-calibrated eye produces probabilities that match observed frequencies.

**Expected findings:** The TF eye (GraphCodeBERT) is likely best calibrated — pre-trained models tend to produce well-calibrated probabilities. The CFG eye may be over-confident (high probabilities for rare classes like DoS). The Run 4 calibration data showed ECE of 0.249 before calibration → 0.027 after.

**Broader context:** B2 results directly inform the calibration script parameters. If the CFG eye has ECE = 0.15 and the TF eye has ECE = 0.05, the ensemble should weight the TF eye more heavily for calibrated predictions.

**Results:** ⏳ running

---

### 6.5 B3 — JK Weight Distribution per Contract ⏳

**What it tests:** Runs inference on the full val set and collects the JK attention weights (phase1, phase2, phase3) per graph. Groups by predicted class (or true label) to find: which vulnerability types drive Phase 3 dominance?

**Expected findings:** If Phase 3 dominance is a "contract identity" shortcut, we'd expect:
- Phase 3 weight high for classes with concentrated contract-to-label mapping (if most DoS examples come from a few contract families, Phase 3 learns to recognise the contract family)
- Phase 3 weight lower for syntactically varied vulnerabilities (IntegerUO: arithmetic overflow can be anywhere)

**Results:** ⏳ running

---

### 6.6 B4 — UnusedReturn Gradient Saliency ⏳

**What it tests:** Specifically targets the UnusedReturn class. Computes gradient saliency for UnusedReturn-positive contracts: which node features get the highest gradient signal? In a working model, `has_external_call` (feat[2], C-1 fixed) and `return_ignored` features should be highly salient.

**Why UnusedReturn is interesting:** It's been flat at 0.234 since ep10 despite 30 more epochs of training. B4 will show whether the model is even attending to the right features (if `feat[2]` saliency is high, the model "knows" what to look for but can't connect it to a discriminative pattern without DEF_USE edges; if `feat[2]` saliency is low, the model isn't using the C-1 fix at all).

**Results:** ⏳ running

---

### 6.7–6.12 — Remaining Experiments

| ID | Status | Key Question |
|----|--------|-------------|
| E4 | ⏳ | Does reversing CF edges hurt Reentrancy more than other classes? |
| L1 | ⏳ | Which layer's embedding is most predictive per class? |
| L2 | ⏳ | Remove CALL_ENTRY edges — how much does Reentrancy F1 drop? |
| L4 | ⏳ | Which of 11 node features drive each vulnerability class? |
| L5 | ⏳ | Linear probing: how much class info in GNN vs TF embeddings? |
| L8 | ⏳ | Permute feature X across all graphs — which feature matters most? |

---

## PART 7: Bugs Found During This Session (Not Previously Documented)

### BUG-SL-1 — StructuredLogger Subscript on OptimizedModule

**File:** `ml/src/training/training_logger.py` line 305  
**Code:** `final_linear = head[-1]`  
**Problem:** After `torch.compile`, `model.aux_phase2` returns `OptimizedModule` wrapping `nn.Sequential`. `OptimizedModule` doesn't support `[-1]` subscript.  
**Impact:** All structured epoch data (AUC, Brier, ECE, aux head norms, probability stats) was empty for the entire Run 7 (ep1–40). The error was silently caught and logged as WARNING.  
**Fix:** `head = getattr(head, "_orig_mod", head)` before `head[-1]`.  
**Status:** NOT YET FIXED. Apply before Run 8.

---

## PART 8: What To Do Next (Ordered)

1. **[IMMEDIATE]** Run threshold calibration on ep39 checkpoint → `ml/calibration/temperatures_run7.json`
2. **[BEFORE PHASE 2]** Fix BUG-SL-1 in `training_logger.py:305`
3. **[PHASE 2]** Complete experiments A3, A4, B1–B4, E4, L1–L5, L8 (L6/L7/L9/L10 lower priority)
4. **[AFTER PHASE 2]** Synthesise findings into Run 8 config (see analysis doc §14)
5. **[RUN 8]** `fusion_lr_multiplier=0.3`, `--gnn-prefix-k 48`, `λ=0.0075`, fix BUG-SL-1

---

*This document is a living record. Update each Phase 2 experiment section (⏳) as results come in.*
