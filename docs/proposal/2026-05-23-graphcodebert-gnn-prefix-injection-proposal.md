# SENTINEL — GraphCodeBERT + GNN Prefix Injection Architecture Proposal

**Project:** Sentinel ML Module — Smart Contract Vulnerability Detection  
**Document type:** Full Architectural Upgrade Proposal  
**Authors:** Ali Rajabi  
**Date:** 2026-05-23  
**Status:** Draft — Pending Implementation Decision  
**Baseline:** v8 schema, PLAN-3A checkpoint (`v8.0-A-20260521_best.pt`, tuned F1-macro 0.2877)  
**Supersedes:** Aspects of `SENTINEL-v7-Comprehensive-Improvement-Proposal.md` related to transformer path

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement — Why the Current Architecture Has a Ceiling](#2-problem-statement)
3. [Evidence From Training History](#3-evidence-from-training-history)
4. [Proposed Architecture — Overview](#4-proposed-architecture-overview)
5. [Component 1 — GraphCodeBERT Replacing CodeBERT](#5-component-1-graphcodebert-replacing-codebert)
6. [Component 2 — GNN Prefix Injection](#6-component-2-gnn-prefix-injection)
7. [Component 3 — Three Options for DFG Integration](#7-component-3-three-options-for-dfg-integration)
8. [Multi-Window Strategy for Long Contracts](#8-multi-window-strategy-for-long-contracts)
9. [What Stays Unchanged](#9-what-stays-unchanged)
10. [Implementation Phases](#10-implementation-phases)
11. [Assumptions](#11-assumptions)
12. [Expected Outcomes and Predictions](#12-expected-outcomes-and-predictions)
13. [Risk Analysis](#13-risk-analysis)
14. [ZKML Compatibility](#14-zkml-compatibility)
15. [Decision Matrix — Which Option to Start With](#15-decision-matrix)
16. [Appendix A — Current Architecture Reference](#appendix-a)
17. [Appendix B — GraphCodeBERT Technical Reference](#appendix-b)
18. [Appendix C — Token Budget Calculations](#appendix-c)

---

## 1. Executive Summary

The current SENTINEL system has converged to a stable F1-macro ceiling of approximately **0.287–0.288** across three independent training runs (v7, v8-AB, PLAN-3A) despite significant architectural investment, schema improvements, data cleaning, and hyperparameter tuning. The evidence strongly indicates this ceiling is not primarily a data problem or a training problem — it is an **architectural ceiling** imposed by a fundamental property of the current design: the CodeBERT transformer encodes source code tokens in complete isolation from the contract's structural graph, receiving graph information only at the fusion stage, after its 12 attention layers have already finished computation.

This proposal defines a new architecture that eliminates this limitation by:

1. **Replacing CodeBERT with GraphCodeBERT** — pre-trained with Data Flow Graph awareness baked into its attention mechanism, giving the transformer path native structural knowledge.

2. **Injecting GNN node embeddings as prefix tokens into the transformer input** — making CFG ordering, cross-function call chains, and multi-hop structural patterns visible to every transformer attention layer at encoding time, not just at the final fusion step.

3. **Preserving the full three-eye architecture** — GNN Eye, Transformer Eye, and Fused Eye (CrossAttentionFusion) all remain, but now operate on fundamentally richer representations.

Three implementation options are defined for the DFG integration, ordered by complexity and expected quality. The recommendation is to start with **Option B** (pragmatic: GNN prefix + GraphCodeBERT weights, flat mask) and progress to **Option A** (full per-window DFG masking) if Option B plateaus.

**Expected improvement over current baseline:** +8–18% F1-macro improvement in the transformer-sensitive vulnerability classes (Reentrancy, UnusedReturn, ExternalBug, Timestamp), with the full system expected to break above 0.35 F1-macro for the first time.

---

## 2. Problem Statement — Why the Current Architecture Has a Ceiling

### 2.1 The Late-Binding Problem

The current three-eye architecture processes its two input modalities completely independently before they interact:

```
Graph (Slither)                 Source Code (Solidity)
      │                                  │
      ▼                                  ▼
GNNEncoder (7-layer GAT)       CodeBERT (12 layers)
  node_embs [N, 256]             token_embs [B, 512, 768]
      │                                  │
      │       CrossAttentionFusion ◄──────┘
      └──────────────► (bidirectional cross-attention)
                              │
                        Fused Eye [B, 128]
```

CodeBERT's 12 transformer layers process tokens with **zero knowledge of the contract's graph structure**. The attention weights inside CodeBERT are determined purely by token-to-token semantic similarity. By the time CodeBERT finishes, its internal representations are fixed. The CrossAttentionFusion then attempts to reconcile those already-fixed representations with the GNN output — a post-hoc correction, not an integral encoding.

This means CodeBERT cannot learn: *"emit a different embedding for `call.value()` depending on whether a state write occurs before or after this call across function boundaries."* That ordering is a CFG-temporal signal. It is invisible to a transformer operating on a flat token sequence.

### 2.2 Vulnerability Classes That Require Graph-at-Encoding

The following vulnerability classes provably require graph structure at encoding time to detect correctly:

**Reentrancy (cross-function CEI violation):**
The Check-Effects-Interactions (CEI) pattern violation is: external call happens, then state is written. The temporal ordering of these events is encoded in the CFG via `CONTROL_FLOW`, `CALL_ENTRY`, and `RETURN_TO` edges. The token sequence of a vulnerable contract and a safe contract can look nearly identical — the difference is the CFG ordering. CodeBERT cannot distinguish these.

**UnusedReturn (def-use chain):**
A function returns a value, the caller ignores it. This is a def-use relationship between the return point of a callee and the call site in the caller. It spans function boundaries. The data-flow graph encodes this directly; a token sequence does not.

**ExternalBug (cross-function data flow):**
An external call's return value flows into a state-modifying operation. PLAN-3A results confirmed that removing `DEF_USE` edges hurt ExternalBug by −0.015, and removing `CALL_ENTRY`/`RETURN_TO` hurt it further. Both edge types are needed simultaneously — this class requires cross-function structural awareness at the encoding stage.

**Timestamp (contextual conditional):**
`block.timestamp` appears as a token in many contracts (safe and vulnerable). Its danger depends entirely on how it flows into guard conditions or storage writes — a data flow question. The PLAN-3A result where removing DEF_USE *improved* Timestamp (+0.032) indicates the current DEF_USE edges at the GNN level are introducing noise, not signal. This suggests a finer-grained, token-level data flow representation (as in GraphCodeBERT's DFG) would be more precise.

### 2.3 What the CrossAttentionFusion Does and Does Not Solve

The CrossAttentionFusion does provide partial relief — it is not without value:
- GNN nodes can **attend to relevant tokens** after encoding (a `CFG_CALL` node finds `call.value` tokens)
- Tokens can **attend to structural context** after encoding (a `transfer` token finds its `CALLS` edge node)

What it cannot solve:
- The transformer's internal 12-layer representations are already fixed before fusion runs
- CodeBERT cannot learn **context-dependent token embeddings** based on graph topology
- Fusion helps the classifier, but the transformer eye's independent representation remains graph-blind
- The transformer eye pooled output (used as one of three eyes) carries no structural information

The architectural constraint is: fusion provides graph context for the *classifier*, but not for the *encoder*. Both are needed.

---

## 3. Evidence From Training History

The case for architectural change is supported by three lines of evidence from the training record.

### 3.1 The F1-Macro Plateau (v4 through PLAN-3A)

| Run | Key change | Tuned F1-macro | Δ vs prev |
|-----|-----------|----------------|-----------|
| v3 | Baseline multi-label | 0.507 | — |
| v4 | Focal loss fix, LoRA r=16 | 0.542 | +0.035 |
| v5.0 | 3-phase GNN, 3-eye, aux loss | 0.583 (val) | +0.041 |
| v5.1 | Dedup fix (34.9% leakage removed) | — | (reset) |
| v6 | Schema rebuild, windowed tokens, ASL | — | (rebuild) |
| v7.0 | 27 bugs fixed, v7 schema, 11-dim | 0.2875 | baseline |
| v8-AB | ICFG + DEF_USE edges (v8 schema) | 0.2851 | −0.0024 |
| PLAN-3A | ICFG-only Phase 2 | 0.2877 | +0.0002 |

After the data quality reset (v5.1 dedup), three independent runs across different graph schemas and edge type combinations all converge to **0.287–0.288**. The model is not learning from architectural changes at this scale. The plateau is real.

### 3.2 Behavioral Test Results (Most Diagnostic Metric)

| Run | Behavioral pass (19 contracts) |
|-----|-------------------------------|
| v7.0 | 7/19 (37%) |
| v8-AB | 8/19 (42%) |
| PLAN-3A | Comparable to v8 (not independently retested) |

42% behavioral pass rate means the model fails to detect known vulnerable patterns in 58% of hand-curated test cases. The Peculiar paper (GraphCodeBERT for smart contracts) achieved 91.8% precision / 92.4% recall on reentrancy alone — a class where SENTINEL currently achieves F1=0.291. This gap is too large to be explained by dataset differences; the architecture is the primary constraint.

### 3.3 PLAN-3A Hypothesis Autopsy

PLAN-3A was designed to test whether ICFG-only Phase 2 would recover Reentrancy. The hypothesis was partially refuted:
- Reentrancy: +0.005 over v8-AB but still −0.012 below v7 — **structural edge changes within the GNN are not the bottleneck**
- Timestamp: +0.032 surprise improvement — suggests token-level data flow (DFG) is the right representation for Timestamp, not coarser node-level DEF_USE
- F1 net: +0.0002 over v7 — within noise

**Interpretation:** the GNN architecture has been heavily optimized (7 layers, 3 phases, JK attention, per-phase LayerNorm, REVERSE_CONTAINS, per-group LR). Further tuning of the GNN alone is unlikely to produce meaningful gains. The bottleneck has shifted to the transformer path, which has never been made structurally aware.

---

## 4. Proposed Architecture — Overview

The proposed system keeps the three-eye design intact but fundamentally changes how the transformer path encodes contracts.

### 4.1 Current Architecture

```
Graph ──► GNN ──────────────────────────────────► GNN Eye [B,128]
                │                                        │
                └──────────────────────────────► CrossAttentionFusion
                                                         │
Code ──► CodeBERT (graph-blind) ──────────────► Transformer Eye [B,128]
                                    │                    │
                                    └──────────► CrossAttentionFusion
                                                         │
                                               Fused Eye [B,128]
                                                         │
                                    GNN Eye + Transformer Eye + Fused Eye
                                              cat [B,384] → [B,10]
```

### 4.2 Proposed Architecture

```
Graph ──► GNN ──────────────────────────────────────────────────► GNN Eye [B,128]
                │                                                        │
                ├── select K structural nodes                            │
                │   project [K, 256→768]                                 │
                │   ↓                                                    │
                │   GNN prefix tokens [B, K, 768] ─────────────┐        │
                │                                               │        │
Code ──► Tokenizer (windowed)                                   │        │
                │                                               ▼        │
                │   per window:                                 │        │
                │   [CLS][K GNN prefix][SEP][code tokens]       │        │
                │           ↑ injected at input embeds ─────────┘        │
                │           (same prefix in every window)                │
                ▼                                                        │
         GraphCodeBERT (frozen + LoRA on Q+V)                           │
         (graph-aware pre-training from DFG)                            │
         layer 1 through 12: code tokens attend to GNN prefix           │
                │                                                        │
                ├──► Transformer Eye [B,128] (now graph-aware!)         │
                │                                                        │
                └──► CrossAttentionFusion ◄──────────────────────────────┘
                          (node_embs ↔ token_embs — both now richer)
                                  │
                         Fused Eye [B,128]
                                  │
                 GNN Eye + Transformer Eye + Fused Eye
                           cat [B,384] → [B,10]
```

**Key differences from current:**
1. GraphCodeBERT replaces CodeBERT — structural pre-training built in
2. GNN node embeddings are injected as prefix tokens before code tokens — graph visible at layer 1
3. Every window of a multi-window contract sees the same global structural context
4. CrossAttentionFusion receives richer inputs from both sides

---

## 5. Component 1 — GraphCodeBERT Replacing CodeBERT

### 5.1 What GraphCodeBERT Is

GraphCodeBERT (Guo et al., ICLR 2021, `microsoft/graphcodebert-base`) is pre-trained on the same RoBERTa-base architecture as CodeBERT — **identical parameter count (125M), identical hidden dimension (768), identical 12-layer structure**. The difference is the pre-training objective and input format.

During pre-training, GraphCodeBERT receives:
```
[CLS] code_tokens [SEP] dfg_variable_nodes
```
where DFG variable nodes are appended token positions representing variable occurrences in the code. The attention mask is a structured matrix (not a flat vector) that enforces:
- Code tokens attend freely to all code tokens
- Code token T attends to its corresponding DFG variable node (if T is a variable)
- DFG variable node V attends to its code token(s) and to adjacent DFG nodes (def-use edges)
- All other pairs: masked to −∞ (cannot attend)

This trains the model to produce token representations that are intrinsically aware of variable def-use relationships — **from layer 1, not at a post-hoc fusion stage**.

### 5.2 Why GraphCodeBERT Over CodeBERT

| Property | CodeBERT | GraphCodeBERT |
|---|---|---|
| Backbone | RoBERTa-base | RoBERTa-base (identical) |
| Parameters | 125M | 125M |
| Pre-training task | MLM + Replaced Token Detection | MLM + Edge Prediction + Node Alignment |
| Graph awareness | None | DFG (def-use variable chains) |
| HuggingFace ID | `microsoft/codebert-base` | `microsoft/graphcodebert-base` |
| LoRA compatibility | Full | Full (same module names: `query`, `value`) |
| Drop-in replacement | — | Yes — same `AutoModel.from_pretrained()` API |
| Flash Attention 2 | Yes | Yes (same architecture) |
| Inference cost | Baseline | Same (no additional params) |

The case for switching is that it is a direct improvement at **zero parameter cost** and **zero architectural change** — just a different set of pre-trained weights in the same model slot.

### 5.3 LoRA Configuration on GraphCodeBERT

The current LoRA configuration (`r=16, alpha=32, target_modules=["query", "value"]`) applies unchanged. GraphCodeBERT uses the same RoBERTa attention module naming convention. The PEFT library does not interact with the DFG attention mask — LoRA modifies `W_Q` and `W_V` parameter matrices; the mask is applied at forward-pass time as a separate operation. They are fully orthogonal.

Trainable parameters post-LoRA: ~590K (same as current). All 125M GraphCodeBERT base weights remain frozen.

### 5.4 Impact Without GNN Prefix

Even used as a pure drop-in replacement (no prefix, flat attention mask), GraphCodeBERT should provide:
- Better code token representations (DFG-aware pre-training generalizes to vulnerability patterns)
- Improved UnusedReturn detection (def-use chains from pre-training directly relevant)
- Improved Timestamp detection (token-level data flow more precise than coarse node-level DEF_USE)

The Peculiar paper (ISSRE 2021) achieved 91.8%/92.4% precision/recall on Reentrancy using GraphCodeBERT on Solidity contracts — vs SENTINEL's current F1=0.291. Even without GNN prefix injection, the pre-trained model is a meaningful upgrade.

---

## 6. Component 2 — GNN Prefix Injection

### 6.1 The Core Idea

The GNN already encodes the full contract graph through 7 layers of message passing — producing node embeddings that carry structural information (CFG ordering, ICFG cross-function call chains, typed edge relationships, multi-hop patterns). Currently these embeddings are used in two places: as the GNN Eye (after function-level pooling) and as input to CrossAttentionFusion.

The proposal adds a third use: **project K selected node embeddings from the GNN into the transformer's embedding space and prepend them to every window's input sequence**. This makes the graph structure visible to every one of GraphCodeBERT's 12 attention layers, not just at the final fusion step.

### 6.2 Node Selection — Which K Nodes

Not all N graph nodes are equally informative. The following node types carry the vulnerability-relevant structural signals:

```python
STRUCTURAL_PREFIX_TYPES = {
    NodeType.FUNCTION,         # function definitions — entry points
    NodeType.MODIFIER,         # access control guards
    NodeType.CONSTRUCTOR,      # initialization patterns
    NodeType.FALLBACK,         # low-level call receivers
    NodeType.RECEIVE,          # ETH receive handlers
    NodeType.CFG_NODE_CALL,    # external call sites — reentrancy, gas griefing
    NodeType.CFG_NODE_WRITE,   # state variable writes — CEI violations
    NodeType.CFG_NODE_CHECK,   # conditional guards — access control
}
```

Contracts typically have 10–25 nodes of these types. The upper bound K=32 covers ~95% of contracts. Contracts with fewer than K eligible nodes pad the prefix with zero embeddings (which the attention mask marks as invalid, contributing nothing).

Nodes NOT included: `STATE_VAR`, `CFG_NODE_READ`, `CFG_NODE_OTHER`, `CONTRACT`, `EVENT` — these carry less direct vulnerability signal and their information is already aggregated into FUNCTION/CFG_CALL/CFG_WRITE nodes after Phase 3 (REVERSE_CONTAINS).

### 6.3 Projection Layer

A learned linear projection converts GNN hidden dimension to BERT hidden dimension:

```python
self.gnn_to_bert_proj = nn.Linear(gnn_hidden_dim, bert_hidden_dim)
# gnn_hidden_dim = 256 (GNN output)
# bert_hidden_dim = 768 (GraphCodeBERT)
# Parameters: 256 × 768 = 196,608 ≈ 197K parameters
```

This is trainable from scratch. It learns to map the GNN's structural representation space to the BERT semantic space. Weight initialization: Kaiming uniform (default PyTorch).

### 6.4 Input Sequence Construction Per Window

Each window is constructed as follows:

```
Position 0:      [CLS] — standard BERT CLS token (from embedding table)
Positions 1..K:  GNN prefix — K projected node embeddings [K, 768]
Position K+1:    [SEP] — separator (from embedding table)
Positions K+2..: code tokens — source code chunk for this window
Position end:    [SEP] — end separator (from embedding table)
Padding:         [PAD] tokens up to sequence length limit

Total sequence length: 1 + K + 1 + code_chunk + 1 ≤ MAX_LENGTH
```

**Important:** The GNN prefix tokens are continuous projected vectors, not vocabulary token IDs. They are passed via `inputs_embeds` instead of `input_ids`. The full embedding sequence is assembled manually:

```python
# Conceptual forward pass (simplified)
cls_emb   = bert.embeddings.word_embeddings(cls_id)   # [B, 1, 768]
sep_emb   = bert.embeddings.word_embeddings(sep_id)   # [B, 1, 768]
gnn_emb   = self.gnn_to_bert_proj(selected_nodes)     # [B, K, 768]
code_emb  = bert.embeddings.word_embeddings(input_ids) # [B, L_code, 768]

full_emb  = cat([cls_emb, gnn_emb, sep_emb, code_emb, sep_emb, pad_emb], dim=1)
# Pass: model(inputs_embeds=full_emb, attention_mask=full_mask)
```

### 6.5 Attention Mask for the Prefix Region

GNN prefix tokens are allowed to attend to everything (all code tokens, all other prefix tokens, DFG nodes if present). Code tokens are allowed to attend to prefix tokens bidirectionally. This open attention policy allows the transformer to:
- Let `call.value()` code tokens attend to the `CFG_CALL` prefix node that represents the same call site
- Let `balances[msg.sender]` code tokens attend to the `CFG_WRITE` prefix node
- Let the `CFG_CALL` prefix attend to `call.value()` and vice versa — encoding their co-occurrence

```
Attention mask (1 = can attend, 0 = blocked):

              CLS  GNN_1..GNN_K  SEP  code_tokens  PAD
CLS:         [  1      1          1       1          0  ]
GNN_1..GNN_K:[  1      1          1       1          0  ]  ← open attention
SEP:         [  1      1          1       1          0  ]
code_tokens: [  1      1          1       1          0  ]  ← code can attend to prefix
PAD:         [  0      0          0       0          0  ]  ← padding: fully masked
```

### 6.6 Why the Same Prefix in Every Window

For a multi-window contract (W=4 windows), the GNN prefix tokens are **identical across all windows**. This is the correct design because:

1. The GNN processes the **entire contract graph** — it has no concept of windows or token positions. Its node embeddings represent the whole contract's structure.
2. A window covering function `deposit()` benefits from knowing that function `withdraw()` has a `CFG_CALL` node with a `CONTROL_FLOW` edge to a `CFG_WRITE` — even though `withdraw()` is in a different window.
3. The alternative (per-window node selection) would require alignment between token positions and graph nodes, which is a non-trivial engineering problem and introduces a new source of alignment errors.

**Consequence:** Every window effectively has a "global structural summary" as its prefix. This directly solves the window isolation problem — currently each window is an island with no knowledge of what other windows contain. With the shared GNN prefix, every window knows the contract's structural skeleton.

### 6.7 The WindowAttentionPooler Under the New Design

The `WindowAttentionPooler` currently extracts the CLS token from position 0 of each window. This is unchanged — CLS remains at position 0. However, the CLS embedding is now substantially richer: by layer 12 of GraphCodeBERT, CLS has attended to K GNN structural tokens (positions 1..K), the code tokens, and the DFG variable nodes (if Option A or C). The pooler's learned attention over window-CLS embeddings now has a better signal for selecting the most vulnerability-relevant window.

---

## 7. Component 3 — Three Options for DFG Integration

GraphCodeBERT's key advantage over CodeBERT is its DFG-guided attention masking during pre-training. To fully exploit this in inference and fine-tuning, we need to supply the DFG structure as an attention mask. Three options are defined, ordered by implementation complexity and expected quality.

---

### Option A — Full Per-Window DFG Masking (Highest Quality)

**What it does:** For each window, extract the Solidity Data Flow Graph for the variables appearing in that window's code tokens. Append the DFG variable nodes to the sequence and construct the full GraphCodeBERT-style attention mask matrix.

**Input sequence per window:**
```
[CLS][K GNN prefix][SEP][code_tokens_chunk_i][SEP][DFG_var_nodes for chunk_i]
```

**Attention mask:** Full `[seq_len × seq_len]` matrix encoding:
- GNN prefix → everything: open (as in §6.5)
- code token T → DFG node V: allowed if T is a use/def of variable V
- DFG node V → code token T: allowed if T is a use/def of variable V
- DFG node V → DFG node V': allowed if (V, V') is a def-use edge
- Everything → padding: blocked

**How to extract Solidity DFG:** Three approaches:
1. Reuse SENTINEL's existing DEF_USE edges (v8 schema) — map node-level DEF_USE to token positions via `node_metadata` stored in graph files (the `source_mapping` Slither provides). This is the most integrated approach.
2. Use tree-sitter with a Solidity grammar to extract variable def-use chains at token granularity — independent of the GNN pipeline, more accurate, more engineering.
3. Use Slither's `ssa_variables` and `variables_written`/`variables_read` per IR node — already available in the graph extraction pass.

**Token budget per window (K=16 GNN prefix, M=32 DFG nodes):**
```
1 [CLS] + 16 [GNN] + 1 [SEP] + code + 1 [SEP] + 32 [DFG] ≤ 512
→ code budget: 461 tokens per window
→ 4 windows × stride 230: covers ~950 unique content tokens
```

**Pros:**
- Fully exploits GraphCodeBERT's pre-training — the model was trained with exactly this input format
- Token-level DFG is more precise than node-level DEF_USE (PLAN-3A showed Timestamp improved when node-level DEF_USE was removed — token-level DFG may capture cleaner signal)
- The Peculiar paper used this approach and achieved 91.8%/92.4% on Reentrancy

**Cons:**
- Significant preprocessing engineering: per-window variable identification, token-to-variable alignment, attention matrix construction
- Dynamic attention mask shapes (different number of DFG nodes per window) require careful batching
- The DFG masks must be stored or recomputed at training time — adds to preprocessing cost
- Alignment between Slither's node-level representation and token-level positions is non-trivial

**Implementation estimate:** 3–4 weeks (DFG extractor + mask builder + training integration)

---

### Option B — GNN Prefix as DFG Proxy, Flat Mask (Pragmatic, Recommended Start)

**What it does:** Use GraphCodeBERT weights (loaded via `from_pretrained("microsoft/graphcodebert-base")`), inject GNN prefix tokens as described in §6, but use a **flat attention mask** (same format as current CodeBERT usage). No per-window DFG matrix is constructed.

**Input sequence per window:**
```
[CLS][K GNN prefix][SEP][code_tokens_chunk_i][SEP][PAD...]
```

**Attention mask:** Standard flat vector: `[1, 1, ..., 1, 0, 0, ...]` — 1 for real tokens (including GNN prefix), 0 for padding. Same format as current code.

**Why this works:** The GNN's 7-layer message passing already incorporated DEF_USE edges (edge type 10 in v8 schema) along with all other structural information. The GNN prefix tokens carry this aggregated information. So the transformer does see def-use information — it arrives via the GNN prefix rather than via a DFG attention mask. The difference from Option A is granularity: GNN DEF_USE is at AST-node level, GraphCodeBERT's DFG is at variable-token level.

**Additionally:** GraphCodeBERT's pre-trained weights still provide better code understanding than CodeBERT's — the DFG-aware pre-training improves the model's ability to relate code tokens to structural patterns, even when the DFG mask is not explicitly provided at inference time.

**Pros:**
- No per-window DFG extraction or matrix construction
- Token files keep their current `[W, 512]` format — no data pipeline changes
- Only model code changes: swap `codebert-base` → `graphcodebert-base`, add `gnn_to_bert_proj`, modify `forward()` to inject prefix via `inputs_embeds`
- LoRA configuration unchanged
- WindowAttentionPooler unchanged (CLS at position 0)
- Can be implemented and first-trained within 1 week

**Cons:**
- Does not use GraphCodeBERT's DFG masking — loses some pre-training alignment
- Node-level DEF_USE (from GNN prefix) is coarser than token-level DFG
- Fine-grained variable tracking (same variable written in line 10, read in line 47) requires Option A to be fully exploited

**Implementation estimate:** 5–7 days

---

### Option C — Contract-Level Shared DFG, No Per-Window Complexity (Middle Ground)

**What it does:** Build the DFG once for the entire contract (not per-window). Append ALL contract DFG variable nodes as a shared suffix in every window — similar to how GNN prefix tokens are shared. The attention mask is semi-structured: DFG nodes are globally connected to the code tokens that reference the same variable across any window.

**Input sequence per window:**
```
[CLS][K GNN prefix][SEP][code_tokens_chunk_i][SEP][ALL DFG var nodes for whole contract]
```

**Attention mask:** Flat for most connections, but DFG nodes → code tokens connections follow GraphCodeBERT's rules (DFG node V attends to code token T if T is a def/use of V in this window, else 0).

**Why shared DFG:** A variable `balance` that is written in window 0 and read in window 2 has a def-use chain that spans windows. With contract-level DFG, both windows include the `VAR:balance` DFG node. The attention mask for window 0 connects `VAR:balance` to the write token; the mask for window 2 connects it to the read token. The model can learn cross-window data flow patterns.

**Token budget per window (K=16 GNN prefix, M_total=64 DFG nodes for whole contract):**
```
1 + 16 + 1 + code + 1 + 64 ≤ 512
→ code budget: 429 tokens per window
```

**Pros:**
- No per-window variable identification — DFG built once per contract
- Enables cross-window def-use tracking (a key advantage over Option B)
- More aligned with GraphCodeBERT's pre-training than Option B
- DFG extraction can reuse Slither's existing ssa_variables data

**Cons:**
- Larger DFG overhead per window (64 tokens fixed vs variable M per window in Option A)
- For contracts with many variables (>64), DFG must be pruned — most-connected variables selected
- More engineering than Option B but less than Option A

**Implementation estimate:** 1.5–2 weeks

---

### Option Summary

| | Option A | Option B | Option C |
|---|---|---|---|
| DFG masking | Per-window, full matrix | None (flat mask) | Contract-level, shared |
| GNN prefix | Yes | Yes | Yes |
| GraphCodeBERT weights | Yes | Yes | Yes |
| Code changes | High | Low | Medium |
| Data pipeline changes | High | None | Medium |
| Token budget impact | −51 per window (K=16, M=32) | −18 per window (K=16) | −82 per window (K=16, M=64) |
| Expected quality | Highest | Good | Better than B |
| Implementation weeks | 3–4 | 1 | 1.5–2 |
| Recommended order | 3rd | 1st | 2nd |

---

## 8. Multi-Window Strategy for Long Contracts

### 8.1 Current Windowing (Unchanged Infrastructure)

The existing windowing system remains fully compatible with all three options. Current parameters:

```
WINDOW_SIZE  = 512    # CodeBERT/GraphCodeBERT max sequence length
STRIDE       = 256    # 50% overlap between consecutive windows
MAX_WINDOWS  = 4      # cap for very long contracts (linspace sub-sampling)
```

The `transformer_encoder.py` flattening trick (`view(B*W, L)`) is model-agnostic and works identically with GraphCodeBERT.

### 8.2 Window Isolation — The Problem GNN Prefix Solves

Current windowing is limited by window isolation: window 2 covering `withdraw()` has no knowledge of what window 0's `deposit()` structure looks like. Each window is independently encoded by CodeBERT with zero cross-window structural awareness.

With GNN prefix injection, **all windows share the same structural prefix** derived from the full contract graph. Window 2's encoding of `withdraw()` now attends to:
- `CFG_CALL` prefix node: "there is an external call site in this contract"
- `CFG_WRITE` prefix node: "there is a state write node in this contract"
- `FUNCTION:deposit` prefix node: "there is a deposit function"

The window does not need to see these things in its own token slice — they arrive via the shared GNN prefix. This effectively gives every window a global contract summary.

### 8.3 Adjusted Token Budget Per Window

With GNN prefix (all options include this):

| Option | CLS | GNN prefix | SEP tokens | DFG nodes | Code budget | Effective coverage (4 windows, stride adjusted) |
|---|---|---|---|---|---|---|
| Current (CodeBERT) | 1 | 0 | 2 | 0 | 509 | ~1280 unique code tokens |
| Option B (K=16) | 1 | 16 | 2 | 0 | 493 | ~1240 unique code tokens |
| Option C (K=16, M=64) | 1 | 16 | 2 | 64 | 429 | ~1080 unique code tokens |
| Option A (K=16, M=32) | 1 | 16 | 2 | 32 | 461 | ~1160 unique code tokens |

Coverage loss is 3–16% of code tokens per window. This is acceptable given that each token now has structural context (worth more information per token processed).

### 8.4 WindowAttentionPooler — No Changes Required

The pooler extracts CLS embeddings from positions `[0, K+L, 2*(K+L), ...]` — i.e., position 0 of each window. Since CLS always occupies position 0 in the sequence, the pooler logic is unchanged. However, the CLS embedding is now significantly richer: by layer 12, it has attended to K GNN structural tokens and (in Options A/C) DFG variable nodes, yielding a better per-window summary vector for the pooler's learned attention weighting.

### 8.5 Inference Windowing (Online)

The `preprocess.py` `process_source_windowed()` method generates windows with proper `[CLS]...[SEP]` framing for each window. The GNN prefix injection happens at model forward-pass time (not at preprocessing time), so the preprocessor output format is unchanged. The predictor loads the graph, runs the GNN, selects prefix nodes, projects them, and injects them during the transformer forward pass.

---

## 9. What Stays Unchanged

The following components are preserved exactly:

| Component | Status | Reason |
|---|---|---|
| GNNEncoder (7-layer, 3-phase GAT) | Unchanged | Best-performing GNN configuration found in ablations |
| v8 graph schema (11 edge types, 11-dim features) | Unchanged | PLAN-3A confirmed ICFG-only is best current schema |
| CrossAttentionFusion | Unchanged | Still provides fused eye; now receives richer inputs |
| Three-eye classifier | Unchanged | Architecture preserved; eyes are now richer |
| `fusion_output_dim = 128` | LOCKED | ZKML proxy MLP hardcoded to this dimension |
| `NUM_CLASSES = 10` | LOCKED | ZKML circuit hardcoded; classifier output locked |
| `CLASS_NAMES` order | LOCKED | Append-only rule remains |
| AsymmetricLoss (γ⁻=2.0, γ⁺=1.0) | Unchanged | Proven effective for this class imbalance |
| Per-group learning rates (GNN ×2.5, LoRA ×0.3) | Preserved | Add gnn_to_bert_proj group at rate ×1.0 |
| Auxiliary heads (λ=0.3) | Unchanged | Eye dominance prevention still needed |
| WindowAttentionPooler | Unchanged | Works identically with new input structure |
| Windowed tokenization (W=4, stride=256) | Unchanged | Infrastructure compatible with all options |
| MLflow tracking | Unchanged | Add new config keys for architecture version |
| DVC data versioning | Unchanged | Token file format unchanged in Options B/C |
| Inference cache | Unchanged | Schema version bump invalidates stale caches |
| Drift detector | Unchanged | KS test on graph statistics still valid |
| `FEATURE_SCHEMA_VERSION` | Bump to `v9` | New architecture version — invalidates inference caches |

---

## 10. Implementation Phases

### Phase 0 — Prerequisite Validation (3–5 days)

Before any model code changes:

1. **Verify GraphCodeBERT loads in the current environment:**
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/graphcodebert-base')"
   ```
   Confirm Flash Attention 2 or SDPA falls back correctly.

2. **Confirm LoRA applies correctly to GraphCodeBERT:**
   ```python
   model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
   lora_model = get_peft_model(model, lora_config)
   lora_model.print_trainable_parameters()
   # Expected: trainable params ~590,000, all others frozen
   ```

3. **Verify node selection produces K≤32 nodes for >95% of training contracts:**
   ```bash
   python scripts/audit_prefix_node_counts.py  # new script
   # Report: distribution of eligible node counts across 41,522 graphs
   ```

4. **Baseline: run GraphCodeBERT as pure drop-in (no prefix) for 5 epochs:**
   - Replace `codebert-base` → `graphcodebert-base` in `transformer_encoder.py`
   - Zero other changes
   - Compare 5-epoch training curve to v7/PLAN-3A baselines
   - Expected: same or slightly improved loss trajectory
   - This separates the effect of better pre-training from the GNN prefix injection

### Phase 1 — Option B Implementation (5–7 days)

**Files to modify:**

| File | Change |
|---|---|
| `ml/src/models/transformer_encoder.py` | Load `graphcodebert-base`; add `inputs_embeds` path; add `gnn_prefix_nodes` argument to `forward()` |
| `ml/src/models/sentinel_model.py` | Add `gnn_to_bert_proj: nn.Linear(256, 768)`; select K structural nodes; call `transformer_encoder.forward(gnn_prefix_nodes=...)` |
| `ml/src/training/trainer.py` | Add `gnn_to_bert_proj` parameter group (LR ×1.0) |
| `ml/scripts/train.py` | Add CLI flags: `--gnn-prefix-k`, `--graphcodebert` |
| `ml/src/inference/predictor.py` | Update warmup forward pass for new architecture; load `gnn_prefix_k` from checkpoint config |
| `ml/src/preprocessing/graph_schema.py` | Bump `FEATURE_SCHEMA_VERSION` to `v9` |

**New checkpoint config keys to add:**
```python
{
    "architecture": "three_eye_graphcodebert_v1",
    "transformer_model": "microsoft/graphcodebert-base",
    "gnn_prefix_k": 16,
    "gnn_prefix_node_types": [...],  # list of included NodeType values
    "gnn_to_bert_proj_dim": [256, 768],
}
```

**Training run: `graphcodebert-v1-prefix16`**
- All training hyperparameters unchanged from PLAN-3A
- `gnn_prefix_k = 16`
- `max_epochs = 100`, `early_stop_patience = 30`
- Gate metric: tuned F1-macro > 0.2877 (beat PLAN-3A) AND ≥ 1 behavioral test improvement
- Full run expected: ~60–80 hours on RTX 3070

### Phase 2 — Option C Implementation (1.5–2 weeks)

After Phase 1 completes and results are analyzed:

1. Implement Solidity DFG extractor using Slither's `ssa_variables` or tree-sitter
2. Add contract-level DFG variable nodes to preprocessing pipeline
3. Modify token file format to include `dfg_var_ids: [M, 512]` suffix tokens and `dfg_attention_mask` per-contract
4. Modify transformer forward pass to handle the structured attention mask
5. Run comparative training: `graphcodebert-v2-shared-dfg`

**Gate metric:** tuned F1-macro > Phase 1 result AND Timestamp F1 > 0.265

### Phase 3 — Option A Implementation (3–4 weeks)

If Phase 2 shows clear improvement over Phase 1:

1. Per-window DFG variable identification (which variables appear in token positions start_i:end_i)
2. Per-window attention mask matrix `[seq_len_i × seq_len_i]` construction
3. Batching with dynamic sequence lengths (per-window DFG node count varies)
4. Full training run with complete GraphCodeBERT DFG masking

**Gate metric:** tuned F1-macro > Phase 2 result AND Reentrancy F1 > 0.350

---

## 11. Assumptions

The following assumptions underpin the proposal. Each should be validated during Phase 0.

**A1 — GraphCodeBERT LoRA works identically:**
The PEFT library applies LoRA to `query` and `value` modules by name. GraphCodeBERT uses the same module naming as CodeBERT (`roberta.encoder.layer.N.attention.self.query/value`). Assumption: LoRA applies without modification. *Validation: Phase 0 step 2.*

**A2 — GNN prefix node count is bounded by K=32 for most contracts:**
Contracts in the BCCC dataset have a typical function count of 5–15 and CFG_CALL/CFG_WRITE counts of 5–20. Together these sum to 10–35 structural nodes. Assumption: K=32 covers ≥95% without overflow. *Validation: Phase 0 step 3.*

**A3 — The F1 plateau is architectural, not purely a data problem:**
The convergence of v7, v8-AB, and PLAN-3A to ~0.287 across different graph schemas is taken as evidence of an architectural ceiling, not purely a data ceiling. However, label noise (BUG-H5: ~14% Reentrancy mislabeling) is a contributing factor. Assumption: fixing the architectural bottleneck will unlock improvement even with current label quality. If this assumption is wrong (i.e., F1 does not improve even with GraphCodeBERT+prefix), the next step is label cleaning before any further architectural work.

**A4 — GraphCodeBERT pre-training transfers to Solidity:**
GraphCodeBERT was pre-trained on 6 programming languages (Python, Java, JavaScript, PHP, Ruby, Go) — Solidity is not included. Assumption: the DFG-aware pre-training generalizes to Solidity's def-use patterns via LoRA fine-tuning, similar to how CodeBERT generalizes despite Solidity not being in its training data. *Evidence: Peculiar paper used GraphCodeBERT on Solidity and achieved strong results.*

**A5 — Shared GNN prefix is better than no prefix, even for windows whose code doesn't match the structural nodes:**
A window covering a pure utility function (e.g., SafeMath library code) will also receive the `CFG_CALL` and `CFG_WRITE` prefix nodes that relate to `withdraw()`. Assumption: the attention mechanism learns to down-weight mismatched prefix-token pairs (low attention scores) without being confused by them. This is a reasonable assumption given how masked attention works in transformers.

**A6 — ZKML compatibility is preserved:**
The `fusion_output_dim=128` constraint is not violated by any proposed change. The ZKML proxy MLP interfaces only with the fusion layer output. Assumption: all changes upstream of the fusion layer are transparent to the ZKML pipeline. *Validation: confirm `fusion_output_dim` appears unchanged in all new checkpoint configs.*

---

## 12. Expected Outcomes and Predictions

### 12.1 Phase 0 (Baseline GraphCodeBERT Drop-In, No Prefix)

| Metric | Prediction | Reasoning |
|---|---|---|
| F1-macro at 5 epochs | ≥ v7 at same epoch | Better pre-training even without DFG mask |
| UnusedReturn F1 | +0.010–0.030 vs PLAN-3A | DFG pre-training directly encodes def-use chains |
| Timestamp F1 | +0.010–0.025 vs PLAN-3A | Token-level DFG more precise than node-level DEF_USE |
| Reentrancy F1 | +0.000–0.010 vs PLAN-3A | CFG ordering not captured by DFG alone |
| IntegerUO F1 | ±0.005 | Robust class, less sensitive to architecture |

### 12.2 Phase 1 (Option B: GraphCodeBERT + GNN Prefix K=16)

| Metric | Prediction | Reasoning |
|---|---|---|
| Tuned F1-macro | 0.31–0.35 | Transformer now sees CFG ordering (structural ceiling lifted) |
| Reentrancy F1 | 0.33–0.42 | CFG_CALL + CFG_WRITE prefix nodes directly encode CEI pattern |
| ExternalBug F1 | 0.28–0.35 | Cross-function data flow visible via GNN prefix + GraphCodeBERT DFG |
| Timestamp F1 | 0.26–0.32 | Token-level DFG (GraphCodeBERT) + structural context (GNN prefix) |
| Behavioral test | 12–15/19 | Major improvement over current 8/19 |
| DoS F1 | 0.02–0.05 | Data problem dominates — architecture helps little |
| Training time | +15–20% | GNN prefix projection is ~197K params, minimal overhead |
| VRAM | +200–400 MB | K=16 extra tokens per window × 4 windows × batch size 8 |

### 12.3 Phase 2 (Option C: + Shared Contract-Level DFG)

| Metric | Prediction | Reasoning |
|---|---|---|
| Δ Tuned F1-macro vs Phase 1 | +0.02–0.05 | Cross-window def-use chains now trackable |
| Δ UnusedReturn vs Phase 1 | +0.015–0.040 | Shared DFG directly encodes the return-ignored pattern |
| Δ Timestamp vs Phase 1 | +0.010–0.025 | block.timestamp DFG chains captured at token level |

### 12.4 Phase 3 (Option A: Full Per-Window DFG Masking)

| Metric | Prediction | Reasoning |
|---|---|---|
| Δ Tuned F1-macro vs Phase 2 | +0.01–0.03 | Marginal gain over shared DFG; per-window precision vs contract-level coverage tradeoff |
| Reentrancy F1 ceiling | ~0.50 | Limited by label noise (BUG-H5: ~14% mislabeling) |
| Behavioral test ceiling | 15–17/19 | Remaining failures likely mislabeled or ambiguous contracts |

### 12.5 What Would Falsify These Predictions

- If Phase 0 (pure GraphCodeBERT drop-in) shows **no improvement** over CodeBERT at 5 epochs: the pre-training transfer assumption (A4) is wrong; the approach should not proceed to Phase 1.
- If Phase 1 F1-macro stays ≤ 0.295: the GNN prefix is not providing useful signal; investigate whether node selection K=16 is insufficient or the projection initialization is problematic. Try K=32 before abandoning.
- If Phase 1 Reentrancy does not improve from 0.291: the CEI signal is primarily a label noise problem (BUG-H5), not an architecture problem. Label cleaning should precede Phase 2.

---

## 13. Risk Analysis

### 13.1 Risk Table

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| GraphCodeBERT pre-training doesn't transfer to Solidity | Low | High | Phase 0 validation before committing; fallback to CodeBERT + prefix |
| GNN prefix confuses transformer on contracts with many unrelated structural nodes | Medium | Medium | Ablate K=8, 16, 32; try masking prefix-to-code attention if needed |
| BF16 dtype issues during training (documented in past runs) | Medium | Low | `model.float()` after load_state_dict — already a known fix |
| VRAM overflow with K=32 prefix + 4 windows | Low | Medium | Reduce K to 16 or reduce max_windows to 3 |
| Windowing code path breaks with `inputs_embeds` vs `input_ids` | Medium | Medium | Careful unit test of shapes before full training run |
| `_orig_mod.` prefix on compiled model state dict | Known | Low | Already fixed in predictor.py and tune_threshold.py |
| Label noise caps Reentrancy regardless of architecture | High | Medium | Accept — fix label_cleaner.py (BUG-H5) in parallel as independent track |

### 13.2 Fallback Plan

If Phase 1 (Option B) does not beat PLAN-3A after full training:
1. Investigate per-class breakdown — if 3+ classes improve, proceed to Phase 2 regardless of macro
2. Try increasing K to 32 (more structural context)
3. Try including `STATE_VAR` nodes in prefix (def-use chains explicitly represented)
4. If macro still flat after K=32: the architectural ceiling is label noise, not graph blindness — shift to label cleaning track

---

## 14. ZKML Compatibility

The ZKML pipeline interfaces with SENTINEL through the proxy MLP that takes the `fusion_output_dim=128` fused representation as input. All proposed changes are upstream of this interface.

**Preservation guarantees:**
- `fusion_output_dim = 128` — LOCKED. The projection layer inside CrossAttentionFusion outputs exactly 128 dimensions. Nothing in this proposal changes the CrossAttentionFusion.
- `NUM_CLASSES = 10` — LOCKED. Classifier output unchanged.
- The main classifier (`Linear(384, 192)` → `Linear(192, 10)`) is unchanged.
- The fused eye output (`[B, 128]`) is unchanged.

**Impact:** None. The ZKML pipeline does not need to be modified or recompiled for any of the three options.

---

## 15. Decision Matrix — Which Option to Start With

```
START HERE:
  Phase 0 — Pure GraphCodeBERT drop-in (3 days, diagnostic)
    │
    ├── Improvement seen at 5 epochs? ──NO──► Investigate transfer; consider stopping
    │
    └──YES──►
         Phase 1 — Option B (GraphCodeBERT + GNN prefix K=16, flat mask)
           │
           ├── Tuned F1 > 0.30 AND behavioral > 10/19? ──NO──► Debug K, node types, LR
           │                                                   Try K=32 before stopping
           └──YES──►
                Phase 2 — Option C (+ shared contract-level DFG)
                  │
                  ├── Δ F1 > 0.02 vs Phase 1? ──NO──► Option C marginal; skip to analysis
                  │
                  └──YES──►
                       Phase 3 — Option A (full per-window DFG masking)
                         │
                         └── Full ablation complete; document and decide next steps
```

The strong recommendation is to execute Phase 0 and Phase 1 before any decision on Phase 2 or 3. Phase 0 is cheap (3 days, no data pipeline changes) and provides a clear signal on whether the GraphCodeBERT transfer works. Phase 1 is the main investment and the expected primary source of improvement.

---

## Appendix A — Current Architecture Reference

### A.1 Model Parameters

| Component | Parameters | Trainable |
|---|---|---|
| GNNEncoder (7-layer, 3-phase GAT) | ~2.4M | All (~2.4M) |
| CodeBERT base | 125M | 0 (frozen) |
| LoRA matrices (r=16, Q+V, 12 layers) | ~590K | All (~590K) |
| CrossAttentionFusion | ~800K | All |
| Main + Aux classifiers | ~200K | All |
| **Total trainable** | **~4.0M** | — |

### A.2 With Proposed Changes (Option B)

| Component | Parameters | Trainable | Change |
|---|---|---|---|
| GNNEncoder | ~2.4M | All | Unchanged |
| GraphCodeBERT base | 125M | 0 (frozen) | Weights replaced, count same |
| LoRA matrices | ~590K | All | Unchanged |
| **gnn_to_bert_proj** | **~197K** | **All** | **NEW** |
| CrossAttentionFusion | ~800K | All | Unchanged |
| Main + Aux classifiers | ~200K | All | Unchanged |
| **Total trainable** | **~4.2M** | — | +197K (+5%) |

### A.3 Locked Constants (Do Not Change)

```python
fusion_output_dim = 128     # ZKML proxy MLP input dim
NUM_CLASSES       = 10      # ZKML circuit output dim
CLASS_NAMES       = [...]   # append-only, no reordering
NODE_FEATURE_DIM  = 11      # GNN in_channels
NUM_EDGE_TYPES    = 11      # GNN edge_embedding size (v8 schema)
```

---

## Appendix B — GraphCodeBERT Technical Reference

### B.1 Input Format Comparison

```
CodeBERT:
  input_ids:      [B, L]       token IDs, L ≤ 512
  attention_mask: [B, L]       flat: 1=real, 0=pad

GraphCodeBERT (full DFG mode):
  input_ids:      [B, L+M]     token IDs + DFG node IDs, L+M ≤ 512
  attention_mask: [B, L+M, L+M] structured matrix (Option A)

GraphCodeBERT (Option B — our pragmatic use):
  inputs_embeds:  [B, 1+K+1+L+1, 768]  CLS + GNN prefix + SEP + code + SEP
  attention_mask: [B, 1+K+1+L+1]        flat: 1=real, 0=pad
```

### B.2 Pre-training Tasks

| Task | Description | Benefit for Vulnerability Detection |
|---|---|---|
| MLM | Masked Language Modeling | General code token semantics |
| Edge Prediction | Predict DFG edge existence | Learns variable relationship patterns |
| Node Alignment | Predict code token ↔ DFG node mapping | Learns token-to-structure correspondence |

### B.3 Key Paper Citation

Guo, D., Ren, S., Lu, S., Feng, Z., Tang, D., Liu, S., ... & Zhou, M. (2021). GraphCodeBERT: Pre-training Code Representations with Data Flow. *ICLR 2021*. https://arxiv.org/abs/2009.08366

Peculiar (most relevant applied work): Wang, S., et al. (2021). Peculiar: Smart Contract Vulnerability Detection Based on Crucial Data Flow Graph. *ISSRE 2021*. Achieved 91.8%/92.4% precision/recall on Reentrancy using GraphCodeBERT on Solidity.

DeepDFA (architectural inspiration): Steenhoek, B., et al. (2024). Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection. *ICSE 2024*. F1=96.46 using graph-guided GNN embeddings injected into UniXcoder.

---

## Appendix C — Token Budget Calculations

### C.1 Coverage Comparison (4 windows, stride adjusted)

Current windowing with CodeBERT (STRIDE=256):
```
Window 0: tokens 0   → 509    (510 content tokens)
Window 1: tokens 256 → 765
Window 2: tokens 512 → 1021
Window 3: tokens 768 → 1277
Coverage: tokens 0–1277 → ~1278 unique content tokens from beginning
```

Option B with K=16 GNN prefix (adjusted STRIDE to maintain coverage):
```
Per window code budget: 512 - 1(CLS) - 16(GNN) - 2(SEP) = 493 content tokens
STRIDE = 246 (keep ~50% overlap)
Window 0: code tokens 0   → 492
Window 1: code tokens 246 → 738
Window 2: code tokens 492 → 984
Window 3: code tokens 738 → 1230
Coverage: tokens 0–1230 → ~1230 unique content tokens
```

Net coverage loss: ~48 unique code tokens (3.8%). Acceptable.

### C.2 Very Long Contract Handling

Contracts with >4 windows worth of content use linspace sub-sampling (current behavior). The GNN prefix is contract-level and does not change for sub-sampled windows. For a 10,000-token contract sampled to 4 windows:
- Window 0: code tokens ~0–492 (start of contract) + GNN prefix of whole contract
- Window 1: code tokens ~2502–2994 (middle) + same GNN prefix
- Window 2: code tokens ~4994–5486 (middle-end) + same GNN prefix
- Window 3: code tokens ~9507–9999 (end) + same GNN prefix

Each sampled window still has full structural context of the entire contract — a significant improvement over current windowing where each sampled window is completely isolated.

---

*Document ends. Status: Draft — requires engineering review and Phase 0 validation before implementation begins.*
