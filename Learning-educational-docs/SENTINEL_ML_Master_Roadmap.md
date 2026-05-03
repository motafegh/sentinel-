# SENTINEL ML Module — Master Learning Roadmap

> Smart contract vulnerability detection · GNN + CodeBERT hybrid · Multi-label · Production ML  
> **The single source of truth. Supersedes all previous roadmap versions.**

---

## How to Use This Document

This plan answers three things for every topic:

- **What** to study and in what order
- **How deep** to go (depth signal on every topic)
- **Why** it matters — for the codebase, for interviews, or for the AI era

Work through phases in order. Do not skip the teach-back exercises. Do not skip Phase 0 — it is the foundation every other phase is built on.

---

## SMART Objectives

Before you study a single file, write these down and keep them visible.

| Objective | Specific | Measurable | Achievable | Relevant | Time-bound |
|-----------|----------|------------|------------|----------|------------|
| Own the architecture | Explain SENTINEL end-to-end, draw it on a whiteboard from memory | Can answer all 25 interview questions without notes | 8 weeks of structured study | Core to Senior ML Engineer role | Week 8 |
| Own the data pipeline | Trace a raw `.sol` file to model input, name every function and file | Teach-back exercises passed without notes | 2 weeks of pipeline focus | Data pipeline literacy separates ML engineers from researchers | Week 4 |
| Own the production ML layer | Describe monitoring, drift, rollback, threshold tuning, model promotion | Can diagnose any of the 5 production failure modes cold | 2 weeks of inference focus | Production ownership is the highest-value senior skill | Week 7 |
| Own AI-generated code | Narrate 13+ audit fixes as your own code review | Can recite each fix with "the original did X, which caused Y, I changed it to Z" | Ongoing across all phases | The 2026 meta-skill. Companies test this directly. | Week 6 |
| Pass senior interviews | Answer whiteboard + system design questions without hesitation | Mock answers timed under 2 minutes per question | Realistic with this plan | Direct career goal | Week 8 |

---

## Depth Legend

Every file and topic carries one of three depth signals:

| Signal | Meaning | Time budget |
|--------|---------|-------------|
| 🔴 **Master** | Core concept. Companies test this directly. Explain cold, draw on a whiteboard, defend every decision. | 2–8 hours |
| 🟡 **Understand** | Know what it does, why it exists, and how it connects to 🔴 topics. Can explain in 2–3 minutes. | 1–2 hours |
| 🟢 **Survey** | Know it exists, its purpose, and when you'd reach for it. No need to memorise internals. | 15–30 min |

---

## What Senior ML Companies Actually Test in 2026

Before you start, understand what you are preparing for. Map every phase back to these five dimensions:

**1. System design thinking**  
Can you describe an ML system end-to-end? Where are the bottlenecks? What breaks at scale? What would you do differently? Tested in every senior ML systems interview.

**2. Architectural tradeoff reasoning**  
Why this approach, not the obvious alternative? Why GAT not GCN? Why cross-attention not concatenation? Why LoRA not full fine-tune? Why BCE+pos_weight not FocalLoss by default? You need a 30-second answer for every major design choice.

**3. Production ML ownership**  
Do you understand monitoring, drift detection, threshold tuning, model versioning, and rollback? Can you describe what happens when the model degrades in production?

**4. Owning AI-generated code (the 2026 meta-skill)**  
Most of this codebase was written by an AI assistant. Companies know this is the new reality. What they test is: can you *own* it? A candidate who says "AI wrote it so I don't fully understand it" fails. A candidate who narrates the audit fixes as "I reviewed this code and found these 5 bugs" succeeds.

**5. Data pipeline literacy**  
Can you trace a raw artifact from source to model input? Do you understand data versioning, feature contracts, schema versioning, and what breaks when you skip validation? This separates ML engineers from ML researchers.

---

## Priority Intelligence — What Companies Are Hiring For

Before starting, understand which parts of SENTINEL map to the highest hiring demand. Allocate your deepest study time accordingly.

### 🔥 Highest demand (go deepest here)
- **GNN for code analysis** (`gnn_encoder.py`) — PyG, message passing, GAT, edge attributes, graph pooling
- **CodeBERT fine-tuning + LoRA** (`transformer_encoder.py`) — PEFT, parameter-efficient adaptation, LoRA math
- **Multi-modal fusion** (`fusion_layer.py`) — combining graph + sequence representations, cross-attention
- **Drift detection** (`drift_detector.py`) — statistical monitoring, KS test, production alert design
- **MLOps model lifecycle** (`promote_model.py`, DVC, MLflow) — experiment to production pipeline

### ✅ Solid differentiators
- Per-class threshold tuning (`tune_threshold.py`) — beyond the default 0.5
- Focal loss for imbalanced multi-label (`focalloss.py`) — derivation + intuition
- PyG Batch collation (`dual_path_dataset.py`) — variable-size graph batching
- FastAPI ML serving (`api.py`) — production inference API design
- Smart contract security domain knowledge — the 10 vulnerability classes

### ⏩ Fast-forward confidently
- `poetry.lock` / `pyproject.toml` — know every dependency and its purpose; skip lock file internals
- `docker/Dockerfile.slither` — know Slither exists and runs in Docker; skip Dockerfile syntax
- All `__init__.py` files — quick scan for namespace exports only
- `ml/scripts/run_overnight_experiments.py` — read as a template for experiment management; no new concepts
- `ml/scripts/create_label_index.py` — documented obsolete in STATUS.md; skip

---

## The Senior's Angle — Framework for Every File

Since the code is AI-written, your job is not to memorise what it does but to **critique and explain the design decisions**. For every major file, run through these five questions before moving on:

1. **Why this architecture and not the alternative?** (GATConv vs GCNConv? Cross-attention vs concatenation? FocalLoss vs BCE+pos_weight?)
2. **What are the input and output shapes?** Write them down — tensor shape fluency is directly tested.
3. **What would break if this changed?** (e.g., changing `NODE_FEATURE_DIM` without rebuilding graphs — trace the failure chain)
4. **What is this component protecting against?** (Focal loss → class imbalance; drift detector → model staleness; atomic cache write → partial file corruption)
5. **How does this connect to the file I read before?** (pos_weight in `build_multilabel_index.py` → `focalloss.py`; content hash in `hash_utils.py` → cache key in `cache.py`)

Being able to answer these "why" questions confidently is what separates a senior from a junior — and precisely what interviewers probe for.

---

## Day 0 — Environment Setup & Project Orientation

**Do this before any studying. If you cannot run the code, the study is half as effective.**  
**Time: 2–3 hours**

### Setup checklist

```bash
# 1. Clone and enter project
git clone <repo>
cd sentinel

# 2. Install dependencies (Poetry)
poetry install
poetry shell

# 3. Verify CUDA (if GPU available)
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 4. Pull data artifacts
dvc pull  # requires remote configured at /mnt/d/sentinel-dvc-remote

# 5. Run the test suite to confirm setup
pytest ml/tests/ -v --tb=short

# 6. Verify MLflow tracking
mlflow ui --port 5000  # open http://localhost:5000

# 7. Start the inference API
uvicorn ml.src.inference.api:app --reload
# Confirm: curl http://localhost:8000/health
```

### Orientation questions (answer before Phase 0)
- How many `.pt` graph files are in `ml/data/graphs/`? (should be ~68,523)
- How many vulnerability classes are in `CLASS_NAMES`? Are they the same as `multilabel_index.csv` columns?
- What does `dvc status` show? Are your data artifacts in sync with the git commit?
- What is the current `FEATURE_SCHEMA_VERSION` in `graph_schema.py`?
- Run one test: `pytest ml/tests/test_gnn_encoder.py -v`. All passing?

---

## Phase 0 — Conceptual Foundations

**Theme:** The "why this architecture" questions that interviewers open with.  
**Goal:** Answer any "why X instead of Y" question for the six major design choices — without looking at any file.  
**Time:** 4–5 hours total

---

### Topic A — Why GNNs for smart contract analysis 🔴

**Interview question:** "Why not just feed the source code to a second transformer?"

- Solidity vulnerability patterns are fundamentally **structural**, not textual. A `withdraw()` function called by a function called by an external-facing function three hops away in the call graph is *adjacent in the graph* but 400 lines apart in the source text.
- GNNs propagate information along edges — "this node reads this state variable" is a READS edge that a text model cannot directly encode.
- The five SENTINEL edge types (`CALLS`, `READS`, `WRITES`, `EMITS`, `INHERITS`) encode the semantic relationships that define reentrancy, integer overflow, and access control vulnerabilities.
- **Why GAT specifically over GCN:** GCN uses fixed normalised adjacency weights. GAT learns which neighbours to attend to. A `withdraw()` node should attend more strongly to the state variable it reads than to an unrelated helper it calls. GAT makes this learnable.

---

### Topic B — Why cross-attention fusion, not concatenation 🔴

**Interview question:** "Why not just concatenate the GNN output and transformer output?"

- Concatenation produces `[B, 64+768]` — a flat vector that loses all node-token relationships. It cannot represent "node 7 (the withdraw function) is most relevant to tokens 340–380 (the balance check)."
- Cross-attention lets each node **query** the token sequence and vice versa — the fusion is position-aware and bidirectional.
- SENTINEL's fusion is bidirectional: node→token (structural queries semantic) AND token→node (semantic queries structural). Concatenation only *combines* — it does not *relate*.

---

### Topic C — LoRA mathematical foundation 🔴

**Interview question:** "Explain how LoRA works mathematically."

Own this on a whiteboard:

- Full fine-tuning updates `W ∈ ℝ^{d×k}` — all `d×k` parameters change.
- LoRA freezes W and learns two low-rank matrices: `B ∈ ℝ^{d×r}` and `A ∈ ℝ^{r×k}` where `r ≪ min(d,k)`.
- Effective weight: `W' = W + (α/r) × BA` where `α` is `lora_alpha`.
- B is initialised to **zero**, A to **random Gaussian** → at init, `BA = 0`, so the model starts identical to the frozen base. No disruption at the start of fine-tuning.
- With `r=8` on Q and V of 12 CodeBERT layers: `2 × 12 × (768×8 + 8×768) = 294,912` trainable params vs 124M frozen.
- **Why `α/r` scaling?** It decouples the learning rate from rank — changing `r` without re-tuning `lr` works because `α/r` stays constant.
- **Why Q and V, not K?** K controls what gets attended *to* (structural). Q and V control how much and what is retrieved (semantic). LoRA on Q+V gives enough expressiveness for domain adaptation with minimal params.
- **Rank trade-off:** `r=1` is a rank-1 update (very constrained). `r=768` is full fine-tune. `r=8` is the practical sweet spot: enough capacity to learn Solidity-specific semantics without overfitting on 68K contracts.

---

### Topic D — Why CodeBERT not a general LLM 🟡

- CodeBERT was pre-trained on code-comment pairs from GitHub (6 programming languages). Identifiers like `transfer`, `require`, `msg.sender` have richer representations than in a general language model.
- 125M params — small enough to fine-tune with LoRA on a single GPU; large enough to capture code structure.
- Newer models (GPT-4, Codex) require far more VRAM and inference latency makes per-request use expensive. CodeBERT fits in 8 GB VRAM alongside the GNN.
- The 512-token limit is a deliberate constraint, handled by sliding windows for long contracts.

---

### Topic E — Multi-label vs multi-class 🔴

**Interview question:** "Why sigmoid not softmax?"

- **Multi-class (softmax):** Classes are mutually exclusive. One correct answer. `CrossEntropyLoss`. Example: digit classification (0–9). Softmax forces all probabilities to sum to 1.
- **Multi-label (sigmoid):** Classes are *independent*. A contract CAN simultaneously have Reentrancy AND IntegerUO AND Timestamp. Each class has its own sigmoid — probability in [0,1] independent of others.
- `BCEWithLogitsLoss` applies sigmoid internally and computes binary cross-entropy per class per sample. Numerically more stable than `sigmoid` then `BCELoss`.
- `pos_weight`: for class `c`, weight = `neg_count / pos_count`. Contracts with rare vulnerability (DenialOfService: pos_weight=68) get 68× loss contribution on positive examples — rebalances gradients without oversampling or undersampling.

---

### Topic F — The 10 Vulnerability Classes (Domain Knowledge) 🔴

**This is what SENTINEL detects. You must know each class: what it is, what vulnerable Solidity looks like, and why it is dangerous. Interviewers open with this.**

| Class | What it is | Why dangerous |
|-------|-----------|--------------|
| **Reentrancy** | External call before state update allows attacker to re-enter and drain funds | The DAO hack ($60M). Most famous Solidity bug. |
| **IntegerUO** | Integer overflow/underflow (pre-Solidity 0.8) wraps arithmetic | Attacker mints unlimited tokens or bypasses balance checks |
| **Timestamp** | `block.timestamp` used for randomness or critical logic | Miners can manipulate ~15 seconds — gameable for lottery/auction contracts |
| **CallToUnknown** | `call()` or `delegatecall()` to an unknown/user-supplied address | Arbitrary code execution, wallet draining |
| **DenialOfService** | Gas exhaustion or reverts that permanently lock contract functionality | Attacker makes the contract permanently unusable |
| **ExternalBug** | Relying on external contract that itself has a vulnerability | Trust chain failure — your contract is secure but your dependency is not |
| **GasException** | Out-of-gas in a loop or unbounded operation | Transaction reverts, state partially updated |
| **MishandledException** | Return value of `send()`/`call()` not checked | Silent failure — funds lost with no indication |
| **TransactionOrderDependence** | Outcome depends on transaction ordering (front-running) | MEV exploitation — sandwich attacks, front-running DEX trades |
| **UnusedReturn** | Return value from a function call is ignored | Logic errors — function signals failure but caller proceeds anyway |

**Study exercise:** For each class, write 5 lines of vulnerable Solidity. You should be able to spot the pattern when reading unfamiliar code.

---

### Topic G — EVM and Blockchain Primer 🟡

*(Added to give context for why vulnerability classes exist and why graph structure matters)*

- **EVM (Ethereum Virtual Machine):** All Solidity contracts compile to EVM bytecode. The EVM is stack-based, stateful, and gas-metered.
- **Gas:** Every operation costs gas. This creates the attack surface for DoS (running a contract out of gas) and GasException.
- **`msg.sender`:** The address calling the current function. Central to access control — most AccessControl bugs come from incorrectly trusting `msg.sender`.
- **`delegatecall`:** Executes code from another contract *in the context of the current contract*. Storage is the caller's. Extremely powerful and extremely dangerous.
- **State vs memory:** State variables persist on-chain. Memory is ephemeral per call. The READS/WRITES edges in SENTINEL's graph schema directly encode state variable access.
- **Why this matters for SENTINEL:** The graph's edge types (`CALLS`, `READS`, `WRITES`, `EMITS`, `INHERITS`) are not arbitrary — they map directly to the EVM execution model. You cannot explain why the graph schema is structured this way without understanding EVM semantics.

---

### Phase 0 Teach-Back Exercise

Answer all seven questions without looking at any file:
1. Why does SENTINEL need a GNN path at all? (30 seconds)
2. What would you lose by replacing CrossAttentionFusion with concatenation? (30 seconds)
3. Write the LoRA weight update formula. What does r=8 buy you vs r=64? (1 minute)
4. Why BCEWithLogitsLoss? What does pos_weight=68 for DenialOfService mean? (1 minute)
5. A colleague proposes switching CodeBERT to GPT-4o. What are the tradeoffs? (1 minute)
6. Name all 10 vulnerability classes. For three of them, describe what vulnerable Solidity looks like. (2 minutes)
7. What is `delegatecall` and why does it appear as its own edge type in the graph schema? (30 seconds)

---

## Phase 1 — The Contract: Locked Invariants

**Theme:** Before touching a model file, understand why the schema is immutable.  
**Goal:** Explain the shape contract, every file it spans, and the exact consequences of violating it.  
**Time:** 2–3 hours

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/preprocessing/graph_schema.py` | 🔴 Master | The single source of truth for all feature contracts |
| `ml/src/utils/hash_utils.py` | 🟡 Understand | The two hash systems and why they must never be mixed |
| `ml/pyproject.toml` | 🟡 Understand | Every dependency and why it is there — the architectural story in 60 lines |

### Read `pyproject.toml` first (15 minutes)

Before any code: read every dependency and ask why it is here.
- `torch-geometric` = GNN support
- `transformers` + `peft` = CodeBERT + LoRA
- `mlflow` + `wandb` + `dvc` = MLOps stack (experiment tracking + data versioning)
- `fastapi` = production inference API
- `scipy` = drift detection (KS test)
- `slither-analyzer` = Solidity static analysis (graph extraction)

This is the architecture described in 60 lines of TOML. Reading it first gives you the system map.

### Questions to answer

**graph_schema.py:**
- Why does this file exist instead of defining constants in `graph_extractor.py`? What silent failure did it make structurally impossible?
- `VISIBILITY_MAP` uses ordinal encoding (0, 1, 2) not one-hot. When would the ordinal assumption be wrong?
- The `assert len(FEATURE_NAMES) == NODE_FEATURE_DIM` fires at *import time*, not in a test. Why is import-time the right boundary for this assertion?

**hash_utils.py:**
- `get_contract_hash()` hashes the file *path*; `get_contract_hash_from_content()` hashes the *content*. Which pipeline uses each? What breaks if they're swapped?
- Why is collision resistance irrelevant here but *stability* (same input → same output, forever) is essential?

### Cross-cutting: The Two Hash Systems (never mix)

| | Path hash | Content hash |
|---|---|---|
| Function | `get_contract_hash()` | `get_contract_hash_from_content()` |
| Input | File path string | Source code text |
| Used by | `ast_extractor.py`, `tokenizer.py` | `preprocess.py`, `cache.py` |
| Behaviour | Two identical files at different paths → different hashes | Same source string always → same hash |
| Purpose | `.pt` filename (pairing key between graphs/ and tokens/) | Cache key prefix |

Cache key = `"{content_md5}_{FEATURE_SCHEMA_VERSION}"` — content hash alone is not enough.

### The 4-Step Schema Change Policy (memorise this)

Any change to `NODE_TYPES`, `VISIBILITY_MAP`, `EDGE_TYPES`, `FEATURE_NAMES`, or `_build_node_features()`:

1. Rebuild all 68K graph `.pt` files (`ast_extractor.py`)
2. Rebuild all token `.pt` files (`tokenizer.py`)
3. Retrain from scratch
4. Increment `FEATURE_SCHEMA_VERSION` in `graph_schema.py`

Skipping step 4 is the most dangerous omission: the inference cache serves old graphs to the new model silently. The `graph.x.shape[1] == 8` check in `InferenceCache.get()` catches some mismatches but NOT a reordering of features that keeps `NODE_FEATURE_DIM=8`.

### Teach-Back Exercise

"If I add a new node feature, what are the exact 4 steps, and what happens at every level of the system if I skip step 4?" Walk through: dataset rebuild, retrain, inference cache, the cache key format, and the validation check in `InferenceCache.get()`.

---

## Phase 2 — From Solidity to Tensors

**Theme:** Trace a `.sol` file from bytes on disk to the two tensors `SentinelModel.forward()` expects.  
**Strategy:** Trace ONE contract (e.g., a Reentrancy contract from the dataset) through every step. If you can narrate that journey end-to-end, you understand the whole data pipeline.  
**Goal:** Describe every transformation step, the exception hierarchy, and why two hash functions exist.  
**Time:** 3–4 hours

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/preprocessing/graph_extractor.py` | 🔴 Master | Solidity → PyG graph; typed exceptions; the replication constraint |
| `ml/data_extraction/ast_extractor.py` | 🔴 Master | Batch orchestration; multiprocessing; checkpoint/resume; how Slither produces AST |
| `ml/data_extraction/tokenizer.py` | 🟡 Understand | CodeBERT tokenisation; truncation detection; worker init |
| `ml/src/inference/preprocess.py` | 🟡 Understand | Online path (contrast with offline); temp file; cache integration |
| `ml/scripts/build_multilabel_index.py` | 🔴 Master | SHA256→multi-hot; GROUP BY logic; WeakAccessMod exclusion; pos_weight calculation |
| `ml/scripts/create_splits.py` | 🟡 Understand | Iterative stratification for multi-label; split ratio; determinism via seed |
| `ml/src/datasets/dual_path_dataset.py` | 🔴 Master | Paired loading; binary vs multi-label mode; RAM cache; collate_fn |
| `ml/scripts/validate_graph_dataset.py` | 🟢 Survey | Pre-retrain data validation gate |
| `ml/scripts/analyse_truncation.py` | 🟡 Understand | Read as engineering document — why 96.6% truncation motivates sliding window |

### Questions to answer

**graph_extractor.py:**
- The comment says "REPLICATION CONSTRAINT — DO NOT CHANGE WITHOUT RETRAINING." What was the concrete silent failure before this function existed?
- Node insertion order: `CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs`. What breaks if you reverse two of them? Trace it to a tensor shape error inside GATConv.
- `SolcCompilationError` → HTTP 400; `SlitherParseError` → HTTP 500. Why is that distinction meaningful to a caller?

**ast_extractor.py:**
- It's "orchestration only" after the refactor. What 3 things does it still own exclusively?
- Batch error policy: return `None` on `GraphExtractionError`, re-raise on `RuntimeError`. How does `contract_to_pyg()` implement this without modifying the shared function?
- Why does `init_worker()` load CodeBERT once per worker process? Why can't the main process pass the tokenizer directly?

**tokenizer.py:**
- Truncation detection uses a separate `encode()` call without truncation. Why can't you just check `attention_mask.sum() == 512`?

**build_multilabel_index.py:**
- `GROUP BY SHA256 then max()` = OR-reduction for multi-label. Why OR and not AND? What real-world labelling scenario does this handle?
- `WeakAccessMod` is excluded with zero training examples. What would happen at training time if it were included?
- pos_weight per class is output from this script. Trace how it flows to `trainer.py` and ultimately to `BCEWithLogitsLoss`.

**dual_path_dataset.py:**
- What happens in `__getitem__` when the graph `.pt` exists but the matching token `.pt` is missing?
- How does `collate_fn` handle variable-size graphs via `Batch.from_data_list`? What does the `batch` tensor contain?
- The RAM cache: what is the cache key, and what integrity check prevents stale data from being served after a schema change? (Audit #11)

### analyse_truncation.py — read as a decision document

This script exists because 96.6% of contracts exceed 512 tokens. The output it produces is the engineering justification for the sliding window. Read the *output section* and ask: what would have happened if the team had ignored this and used CLS-only on truncated inputs? Which vulnerability classes would have suffered most (hint: which vulnerabilities appear at the end of a contract)?

### Teach-Back Exercise

Trace the full path of a source string submitted to the API. Start at `process_source()` in `preprocess.py`, end at `graph.x` entering `GNNEncoder.forward()`. Name every function, every file written/read (including the temp file and why it exists), and the exact cache key at the point it is constructed.

---

## Phase 3 — The Two Encoders

**Theme:** Each encoder in isolation — what it consumes, produces, and why it was designed this way.  
**Goal:** Explain GAT with edge attributes, LoRA parameter math from first principles, and what would be lost if either encoder were simplified.  
**Time:** 3–4 hours

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/models/gnn_encoder.py` | 🔴 Master | 3-layer GAT + edge embeddings; no pooling decision; graceful degradation |
| `ml/src/models/transformer_encoder.py` | 🔴 Master | CodeBERT + LoRA; the no_grad trap; why all token positions |

### Deep study: PyG Batch mechanics 🔴

This is not in any single file — it is the mechanism underlying the entire model. **Draw this on paper before reading `fusion_layer.py`.**

**The problem:** PyG graphs have variable numbers of nodes. `graph1` has 5 nodes, `graph2` has 3 nodes — `torch.stack()` fails.

**The solution — `Batch.from_data_list([g1, g2, g3])`:**
- Merges all graphs into one large disconnected graph with `N1 + N2 + N3` total nodes
- Adds `batch` tensor `[N_total]` where `batch[i]` is the graph index node `i` belongs to
- `edge_index` values are offset by cumulative node counts so each graph's edges stay internal
- Result: a single PyG `Batch` object that passes through GATConv as one large graph

**`to_dense_batch(x, batch)` (used in `fusion_layer.py`):**
- Reverses the above: produces padded tensor `[B, N_max, F]` and mask `[B, N_max]`
- Graphs with fewer nodes than `N_max` get zero-padded rows; mask marks real vs padded nodes
- This is what enables `CrossAttentionFusion` to process all graphs in a batch simultaneously

### Questions to answer

**gnn_encoder.py:**
- Pooling was removed from `forward()`. What information is preserved by NOT pooling? Give a concrete example using a `withdraw()` function.
- Layer 3 uses `heads=1, concat=False`; layers 1–2 use `heads=8, concat=True`. Why does layer 3 collapse heads? What would the output shape be if layer 3 also used `concat=True`?
- `edge_attr=None` triggers graceful degradation to zero vectors (no error). What would an operator observe in production metrics before realising edge embeddings weren't being used?

**transformer_encoder.py:**
- "Never wrap `self.bert()` in `torch.no_grad()`." Why would this silently kill LoRA training even though CodeBERT weights have `requires_grad=False`?
- Returning `[B, 512, 768]` (all positions) not `[B, 768]` (CLS). What would reentrancy detection specifically lose if you reverted to CLS-only?
- With `r=8` and `lora_alpha=16`: what is the scale factor applied to `BA`? Why does this decoupling of alpha and r matter?

### Teach-Back Exercise

"Why does reentrancy detection specifically benefit from keeping node-level embeddings unfused AND returning all 512 token positions instead of just CLS?" Use the `withdraw()` example. Walk through which attention weights in Phase 4 let the vulnerability signal survive to the classifier.

---

## Phase 4 — Fusion: Where Structure Meets Semantics

**Theme:** `CrossAttentionFusion` completely — bidirectional design, masking, the 8 audit fixes.  
**Goal:** Walk through the forward pass on a whiteboard, explain each fix, describe masked mean pooling.  
**Time:** 3–4 hours  
**Note:** Read the test file `test_fusion_layer.py` BEFORE reading `fusion_layer.py` — the tests are the spec.

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/models/fusion_layer.py` | 🔴 Master | The most complex component; 8 audit fixes; masking decisions |
| `ml/src/models/sentinel_model.py` | 🔴 Master | Composition; num_classes fix; attention_mask threading |

### Questions to answer

**fusion_layer.py:**
- Fix #2 threads `token_padding_mask` into node→token cross-attention as `key_padding_mask`. Before this fix, what happened to a node's enriched representation? Error, wrong results, or both?
- Fix #7 replaces a Python for-loop with `to_dense_batch()`. What was the loop doing and why is the Python version a GPU utilisation problem specifically?
- Fix #8 zeros padded node positions after node→token attention, even though pooling already excludes them via `node_real_mask`. Under what future code change would omitting Fix #8 silently reintroduce the bug?

**sentinel_model.py:**
- Fix #3 changed `num_classes` default from 1 → 10. What would a developer observe loading a 10-class checkpoint into `SentinelModel()` with no args under the old default? Python error, shape error, or wrong predictions?
- Trace `attention_mask` from `SentinelModel.forward()` through `CrossAttentionFusion`. List the 3 distinct places it is used and what breaks at each if it is missing.

### Teach-Back Exercise

On a whiteboard: draw tensor shapes at each step of `CrossAttentionFusion.forward()` for B=2 contracts where contract 1 has 5 nodes and contract 2 has 3 nodes. Label:
`nodes_proj [N=8, 256]` → `to_dense_batch` → `padded_nodes [2, 5, 256]` + `node_real_mask [2, 5]` → both cross-attention outputs → final `[2, 128]`.

---

## Phase 5 — Training: Loss, Optimisation, Stability

**Theme:** Every training decision and why it was made.  
**Goal:** Explain class imbalance handling from scratch; configure a TrainConfig for a new run; diagnose a resume bug; describe what AMP does to the gradient graph.  
**Time:** 3–4 hours

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/training/focalloss.py` | 🟡 Understand | FocalLoss mechanics; BF16 underflow bug and fix |
| `ml/src/training/trainer.py` | 🔴 Master | TrainConfig; CLASS_NAMES; the 6 speed fixes; resume correctness; WandB + MLflow logging |
| `ml/scripts/tune_threshold.py` | 🔴 Master | Per-class threshold tuning; why not on test split |
| `ml/scripts/create_splits.py` | 🟡 Understand | Stratified splitting; why multilabel_index.csv not label_index.csv |
| `ml/scripts/build_multilabel_index.py` | 🟡 Understand | SHA256 → multi-hot; the two hash system bridge |
| `ml/scripts/run_overnight_experiments.py` | 🟡 Understand | Hyperparameter sweep orchestration; use as experiment management template |
| `ml/scripts/train.py` | 🟢 Survey | CLI wrapper; know the flags |

### Deep study: AMP and GradScaler 🔴

**What `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` does:**
- Casts specific ops (matmul, conv, attention) to BF16 automatically during the forward pass
- Keeps accumulation, batch norm, and loss in FP32
- **BF16 vs FP16:** same bit width (16 bits) but BF16 has 8 exponent bits (same as FP32) vs FP16's 5. BF16 has the same *dynamic range* as FP32 but lower *precision* — much safer for gradient accumulation. FP16 overflows on large values; BF16 does not.

**Why BF16 on Ampere (RTX 3070+):** Native BF16 tensor cores — hardware-accelerated to same speed as FP16 but without overflow risk.

**`GradScaler` in BF16 context:** GradScaler was designed for FP16 (scales loss up before backward, unscales before optimizer step, skips step if inf/nan). In BF16 this is essentially a no-op — BF16 doesn't have the same underflow problem. It stays for API consistency and future-proofing if FP16 is ever preferred.

**The FocalLoss BF16 underflow bug (real example from this codebase):**
- `sigmoid(-10.0)` in FP32 → `4.5e-5` (small but nonzero)
- `sigmoid(-10.0)` in BF16 → `0.0` (underflows — subnormals flushed to zero on some hardware)
- `log(0.0)` → `-inf` → loss becomes `nan` → training diverges
- **Fix:** `logits.float()` before the sigmoid in FocalLoss — forces FP32 computation for the numerically sensitive part, even inside an autocast region.

### Deep study: FocalLoss vs BCE+pos_weight 🔴

`FL(p) = -α(1-p)^γ · log(p)`

The `(1-p)^γ` term down-weights easy negatives. In SENTINEL: safe contracts (all-zero labels) vastly outnumber vulnerable ones — Focal Loss prevents "always predict safe." The γ parameter controls how aggressively easy examples are down-weighted.

**When to switch from BCE+pos_weight to FocalLoss:** When you observe the model assigning uniformly high confidence to negative predictions (high precision, collapsed recall), and pos_weight alone isn't enough. Focal Loss directly modulates gradient flow; pos_weight only scales the loss magnitude.

### Questions to answer

**focalloss.py + trainer.py:**
- What is `pos_weight` doing mathematically? Under what training metric pattern would you switch from `bce` to `focal`?
- Audit Fix #8: `OneCycleLR(epochs=remaining_epochs)` not `config.epochs`. Draw the LR curve for resuming at epoch 20/40 with the old code. Why does this hurt convergence?
- Audit Fix #7: filter `trainable_params` before `clip_grad_norm_`. What extra computation was happening, and why was it *wrong* not just slow?
- MLflow vs WandB: both are used. What does each track that the other doesn't? When would you choose to look at WandB vs MLflow during a training run?

**tune_threshold.py:**
- Thresholds are tuned on the validation split. Why the same split used during training, not the test split?
- Tie-break rule: higher F1 → higher recall → lower threshold. Why does "prefer lower threshold" make sense for a vulnerability detector vs a spam filter?

### Teach-Back Exercise

Explain the full training loop to a colleague who knows image classifiers but not multi-label. Cover: sigmoid not softmax; `BCEWithLogitsLoss` not `CrossEntropyLoss`; what `pos_weight` does; why training-time threshold (0.5) is replaced by per-class tuned values; what "early stopping on F1-macro" means in a 10-class multi-label setting; what AMP buys; and how MLflow and WandB logging work together.

---

## Phase 6 — Inference: API, Cache, Production Hardening

**Theme:** Full inference path from HTTP request to JSON response.  
**Goal:** Describe the complete path from memory; explain every HTTP status code; describe the sliding window algorithm and when it activates.  
**Time:** 3–4 hours

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/inference/predictor.py` | 🔴 Master | Checkpoint loading; warmup; sliding window; per-class thresholds |
| `ml/src/inference/cache.py` | 🔴 Master | Atomic writes; TTL; schema-version invalidation |
| `ml/src/inference/api.py` | 🔴 Master | FastAPI lifespan; async threading; error codes; Prometheus |
| `ml/src/inference/preprocess.py` | 🟡 Understand | Online extraction; temp file lifecycle; cache integration |
| `ml/src/datasets/dual_path_dataset.py` | 🟡 Understand | Paired loading; binary vs multi-label mode; RAM cache |
| `docker/Dockerfile.slither` | 🟢 Survey | Slither runs in Docker for graph extraction; skip Dockerfile syntax |

### Questions to answer

**predictor.py:**
- `_warmup()` uses a 2-node 1-edge graph (Audit Fix #5). What would NOT be exercised by the 0-edge warmup? What bug type would only surface at the first real request?
- If `{checkpoint.stem}_thresholds.json` is missing, all classes fall back to 0.5. What is the concrete production risk given DenialOfService pos_weight=68?
- Sliding window uses `max` aggregation. With a 1200-token contract with `withdraw()` at token 800+: why is `max` correct and `mean` wrong?

**cache.py:**
- `tmp.rename(dest)` vs `torch.save(obj, dest)` directly. What failure mode does atomic rename prevent?
- `get()` validates `graph.x.shape[1] == 8` on every hit. Under what sequence of events could the correct schema-versioned key still contain a graph with the wrong feature dimension?
- What is the TTL and what production scenario does it protect against?

**api.py:**
- `/predict` uses `asyncio.wait_for(asyncio.to_thread(...))`. Why must `to_thread()` be used? What happens to the event loop if inference runs synchronously?
- The `must_look_like_solidity` validator checks for `pragma` or `contract`. Security gate or UX convenience? Where is the real security boundary?
- Prometheus metrics: what gauges/counters are exposed, and what alert rule would you write for each?

### Teach-Back Exercise

Walk through one API request for a 1500-token contract: Pydantic validation → size check → `predict_source()` → `process_source_windowed()` → graph extraction (including temp file) → sliding window tokenisation (stride=256, max_windows=8) → three forward passes → `max` aggregation → per-class threshold application → JSON response → drift detector update → Prometheus gauge update.

---

## Phase 7 — MLOps: Drift, Registry, Data Versioning

**Theme:** The operational layer that keeps the model honest after deployment.  
**Goal:** Explain the warm-up drift strategy, walk through model promotion, understand DVC data versioning.  
**Time:** 3–4 hours

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/inference/drift_detector.py` | 🔴 Master | KS test; warm-up suppression; rolling buffer; Prometheus |
| `ml/scripts/compute_drift_baseline.py` | 🟡 Understand | Warm-up vs training source; why 30-record minimum |
| `ml/scripts/promote_model.py` | 🟡 Understand | MLflow registry; archive-on-Production; checkpoint validation |
| `ml/scripts/validate_graph_dataset.py` | 🟡 Understand | Pre-retrain gate; edge_attr shape check |
| `ml/pyproject.toml` | 🟢 Survey | Review dependency list with Phase 1 in mind |

### Deep study: DVC data versioning 🔴

This is not in any Python source file — it is infrastructure. You must understand it.

**What DVC solves:**  
Git cannot version 68K `.pt` files (each ~50KB = ~3.4 GB total). DVC tracks large binary artifacts by storing a small `.dvc` pointer file in git and the actual data in remote storage.

**How it works in this project:**
- `dvc pull` — downloads graph `.pt`, token `.pt`, checkpoints, and splits from `/mnt/d/sentinel-dvc-remote`
- `dvc push` — uploads new artifacts after retraining
- `.dvc` files in the repo contain a hash of the directory tree
- `git commit` the `.dvc` files → reproducible data version tied to every code commit

**Why not git LFS?** Cost, performance at 68K files, no content-addressing. DVC is purpose-built for ML data versioning.

**Commands to know:**
- `dvc status` — are your local files in sync with the `.dvc` pointers in git?
- `dvc repro` — replay the data pipeline from the DVC pipeline file
- `dvc pull` — sync data to match current git commit
- After a retrain: `git add ml/data/*.dvc ml/checkpoints/*.dvc && git commit -m "update data version"`

### Questions to answer

**drift_detector.py:**
- Using BCCC-2024 training corpus as the KS baseline causes `sentinel_drift_alerts_total` to fire constantly on 2026 DeFi contracts. Explain precisely why, and what that makes the alerting.
- KS test fires at `p < 0.05`. What does the p-value represent here, and why does a low p-value NOT tell you which direction the distribution shifted?
- The rolling buffer has a fixed size. What happens when it fills up? What data is evicted and why is FIFO the right eviction policy here?

**promote_model.py:**
- `archive_existing_versions=(stage == "Production")` archives old Production but not Staging. What production incident does this prevent?
- If a developer promotes a `legacy_binary` checkpoint (num_classes=1), trace the failure to the specific check in `predictor.py` that surfaces it.

**DVC:**
- A teammate regenerates 500 graph files with a new feature engineering change and runs `dvc push`. What must happen in git to make this reproducible for everyone else?
- You roll back to a previous git commit. What DVC command makes the data match that commit?

### Teach-Back Exercise

Describe the complete MLOps cycle for a retrain:  
`validate_graph_dataset.py` → training with MLflow + WandB → `tune_threshold.py` → `promote_model.py --stage Staging` → warm-up period (500 real requests) → `compute_drift_baseline.py --source warmup` → `promote_model.py --stage Production`.  
Explain `--dry-run` and when you use it. Explain what `dvc push` does after the checkpoint is saved.

---

## Phase 8 — The Test Suite as a Learning Tool

**Theme:** Tests encode the intended contract for each component. Read each test file BEFORE the source it tests — tests are the spec. For AI-written code, tests are often the best documentation.  
**Goal:** Understand the testing strategy well enough to write a new test for any component.  
**Time:** 2–3 hours total (spread across phases — read each test alongside its phase)

### Files

| File | Depth | Read with phase | What it confirms |
|------|-------|-----------------|-----------------|
| `ml/tests/conftest.py` | 🔴 Master | Phase 1 | Shared fixtures; how synthetic data is built; fastest way to learn input/output contracts |
| `ml/tests/test_gnn_encoder.py` | 🔴 Master | Phase 3 | edge_attr shapes; graceful degradation; head-divisibility |
| `ml/tests/test_fusion_layer.py` | 🔴 Master | Phase 4 | Masked pooling correctness; attn_dim divisibility; device detection |
| `ml/tests/test_model.py` | 🟡 Understand | Phase 4 | Full forward pass with stub TransformerEncoder (avoids 500MB download in CI) |
| `ml/tests/test_preprocessing.py` | 🟡 Understand | Phase 2 | Mocked Slither + CodeBERT; temp file handling |
| `ml/tests/test_dataset.py` | 🟡 Understand | Phase 2 | Pairing; split indices; collate function; binary vs multi-label label shapes |
| `ml/tests/test_trainer.py` | 🟡 Understand | Phase 5 | pos_weight; evaluate(); FocalLoss BF16 fix |
| `ml/tests/test_api.py` | 🔴 Master | Phase 6 | /health + /predict schema; determinism; error codes |
| `ml/tests/test_cache.py` | 🟡 Understand | Phase 6 | miss/hit/TTL/schema-version |
| `ml/tests/test_drift_detector.py` | 🟡 Understand | Phase 7 | Warm-up suppression; KS fires on drift; rolling buffer eviction |
| `ml/tests/test_promote_model.py` | 🟢 Survey | Phase 7 | Dry-run; stage validation; MLflow tag stubs |

### Key testing patterns to master

**Stub TransformerEncoder (`test_model.py`):** The model test doesn't load actual CodeBERT — it is 500MB. The stub returns random tensors of the correct shape `[B, 512, 768]`. This lets you test GNN, fusion, and classifier without any HuggingFace dependency. This pattern — replacing expensive components with shape-correct stubs — is standard in production ML testing.

**Mocking Slither (`test_preprocessing.py`):** Slither requires a full compiler + analysis pass (3–5 seconds per contract). Tests mock `subprocess.run` or `SlitherWrapper` to return pre-built PyG objects. This isolates the unit being tested (feature extraction logic) from the tool dependency.

**Parametrised device testing:** Several tests check `cpu` and `cuda` (if available). The pattern `pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")` lets the CI suite pass on CPU-only machines.

**Coverage check:** Run `pytest --cov=ml/src --cov-report=term-missing`. Check which lines in `trainer.py` have no coverage — those are the complex branching paths to manually trace. Any branch not covered by tests is a risk to reason about manually.

---

## Cross-Cutting Concerns

Own these without being prompted which phase they belong to.

### 1. The Masked Mean Pooling Pattern (appears in 3 places)

Naive `.mean(dim=...)` includes PAD positions, diluting signal with zeros.

| Location | Mask source | What it excludes |
|---|---|---|
| `CrossAttentionFusion` node pooling | `to_dense_batch` mask | Padded node positions |
| `CrossAttentionFusion` token pooling | `attention_mask` from tokenizer | PAD tokens ([PAD]=0) |
| `DualPathDataset` collate | — | Not pooling, but same precision awareness |

When reading any new pooling code: ask "what is the mask and is it applied?"

### 2. The `weights_only` Split

| Where | `weights_only` | Reason |
|-------|---------------|--------|
| Checkpoint loading (predictor, trainer, tune_threshold, promote_model) | `False` | LoRA peft classes cannot be loaded with `weights_only=True` |
| Graph/token `.pt` files (dataset `__getitem__`) | `True` | Safe — PyG classes are registered via `add_safe_globals()` at module import |

### 3. The Three Locked Architecture Contracts

Changing any of these requires a full retrain:
- `in_channels=8` in `GNNEncoder` — locked by `NODE_FEATURE_DIM`
- `token_dim=768` in `CrossAttentionFusion` — locked by CodeBERT hidden size
- `num_classes=10` in `SentinelModel.classifier` — locked by `CLASS_NAMES`

`_ARCH_TO_FUSION_DIM = {"cross_attention_lora": 128, "legacy": 64}` in `predictor.py` lets old and new checkpoints coexist — adding a new architecture = one dict entry.

### 4. CLASS_NAMES — the append-only registry

```
CLASS_NAMES = [
  "CallToUnknown",          # index 0
  "DenialOfService",        # index 1
  ...
  "UnusedReturn",           # index 9
]
```

Never insert in the middle. Adding class 10 at the end is safe: existing model outputs for indices 0–9 are unchanged. Inserting at index 3 silently maps "GasException" predictions to "ExternalBug" in all existing checkpoints.

### 5. Owning AI-Generated Code (the 2026 meta-skill)

**Narrate the audit fixes as your own code review.** Practice explaining each one as:  
*"The original implementation did X, which would cause Y under Z conditions. I identified it and changed it to W."*

**Key audit fixes to memorise:**

| Fix | File | Original problem | Consequence |
|-----|------|-----------------|-------------|
| Fix #3 | `sentinel_model.py` | `num_classes` default 1→10 | Silent wrong-shape on checkpoint load |
| Fix #5 | `predictor.py` | warmup used 0-edge graph | GAT propagate never called on 0-edge |
| Fix #6 | `fusion_layer.py` | token PAD mask not applied in pooling | Naive mean diluted with PAD zeros |
| Fix #7 | `fusion_layer.py` | Python loop → `to_dense_batch` | Per-sample loop cannot batch on GPU |
| Fix #8 | `trainer.py` | `remaining_epochs` not `config.epochs` for resume | LR curve restarted from epoch 1 |
| Audit #3 | `dual_path_dataset.py` | `weights_only=True` for graph files | Pickle security |
| Audit #11 | `dual_path_dataset.py` | RAM cache integrity check missing | Stale cache silently served wrong data |

**Be able to extend the system correctly:**
- Add vulnerability class 10 → append to `CLASS_NAMES`, rebuild `multilabel_index.csv`, retrain
- Add edge type 5 → update `EDGE_TYPES` in `graph_schema.py`, bump `FEATURE_SCHEMA_VERSION`, rebuild graphs, retrain
- Change fusion output dim → update `_ARCH_TO_FUSION_DIM`, create new architecture key, retrain
- Add per-class precision/recall to the API response → extend `PredictResponse`, thread thresholds through

### 6. Evaluation Metrics Deep Study 🔴

Multi-label metrics are not the same as multi-class metrics. Own this table:

| Metric | How computed | When it misleads |
|--------|-------------|-----------------|
| **F1-macro** | F1 per class, then unweighted average | Gives rare classes equal weight — may look good when DenialOfService (rare) is poorly predicted |
| **F1-micro** | Aggregate TP, FP, FN across all classes | Dominated by frequent classes — hides failures on rare vulnerabilities |
| **Per-class F1** | F1 for each of the 10 classes independently | The most informative; reveals which classes the model actually handles well |
| **Precision-Recall trade-off** | Controlled by per-class threshold | Lower threshold → higher recall, lower precision. For a security tool, prefer recall. |

Early stopping on **F1-macro** is a deliberate choice: it prevents overfitting to the majority class and forces the model to maintain performance across rare vulnerability types.

---

## What to Skip or Fast-Forward

| File | Decision | Reason |
|------|----------|--------|
| `ml/scripts/run_overnight_experiments.py` | 🟡 Understand (template only) | Convenience wrapper; use as experiment management template |
| `ml/scripts/create_label_index.py` | 🟢 Skip | Documented obsolete in STATUS.md; `graph.y=0` for all contracts |
| `ml/scripts/analyse_truncation.py` | 🟡 Understand (output only) | Read output section only; confirms 96.6% truncation → why sliding window exists |
| `ml/tests/test_promote_model.py` | 🟢 Survey | MLflow stubs; understand the pattern, not the implementation |
| `ml/tests/conftest.py` | 🟡 Understand | Skim fixtures; refer back when a test is confusing |
| `docker/Dockerfile.slither` | 🟢 Survey | Know Slither exists and runs in Docker; skip Dockerfile syntax |
| All `__init__.py` files | 🟢 Survey | Quick scan for namespace exports only |
| `poetry.lock` | 🟢 Survey | Know what Poetry does; skip lock file internals |

---

## Interview Question Bank (25 Questions)

For each question: the answer lives in the anchored file(s) noted. Practice answering out loud, timed under 2 minutes.

| # | Question | Anchor |
|---|----------|--------|
| Q1 | Walk me through SENTINEL architecture end-to-end. | `sentinel_model.py` — all 4 components |
| Q2 | Why does the GNN path not pool before fusion? What specifically would you lose? | `gnn_encoder.py` docstring + `fusion_layer.py` |
| Q3 | Explain LoRA mathematically. Why r=8 and not r=64? | Phase 0 Topic C + `transformer_encoder.py` |
| Q4 | Why BCEWithLogitsLoss and not CrossEntropyLoss? | Phase 0 Topic E + `trainer.py` |
| Q5 | There's a production bug: inference scores are worse than val metrics. Same checkpoint. Where do you look first? | `graph_schema.py` CHANGE POLICY + `cache.py` validation + two hash systems |
| Q6 | What is pos_weight and when would you switch to FocalLoss? | `trainer.py` compute_pos_weight() + `focalloss.py` |
| Q7 | How does the inference cache work, and what prevents it serving stale data after a feature change? | `cache.py` + `preprocess.py` — content hash + schema version |
| Q8 | Your model has Dropout. How do you ensure inference is deterministic? | `predictor.py` `model.eval()` + `test_api.py` determinism test |
| Q9 | A user submits a 3000-token contract. Walk me through exactly what happens. | `predictor.py` `predict_source()` + `preprocess.py` windowing |
| Q10 | We want to add an 11th vulnerability class. Minimum change required, and what breaks? | `trainer.py` CLASS_NAMES + `predictor.py` Bug 5 fix |
| Q11 | Why is the KS drift baseline built from warm-up requests rather than training data? | `drift_detector.py` + `compute_drift_baseline.py` |
| Q12 | Explain the GAT edge attribute mechanism. What does SENTINEL's GATConv see that a GCN doesn't? | `gnn_encoder.py` + `graph_schema.py` EDGE_TYPES |
| Q13 | An engineer re-extracts a subset of contracts but skips `validate_graph_dataset.py`. What happens at training time? | `validate_graph_dataset.py` + `gnn_encoder.py` graceful degradation |
| Q14 | `OneCycleLR` was resuming incorrectly after a checkpoint. Describe the bug and fix. | `trainer.py` Audit Fix #8 |
| Q15 | What does the startup warmup protect against, and why did the 0-edge warmup fail? | `predictor.py` `_warmup()` Audit Fix #5 |
| Q16 | How do you version 68K binary `.pt` files alongside your code? | DVC section in Phase 7 |
| Q17 | Why did you choose BF16 over FP16 for mixed-precision training on this hardware? | Phase 5 AMP section + `trainer.py` |
| Q18 | Explain `Batch.from_data_list()`. Why can't you just `torch.stack()` PyG graphs? | Phase 3 PyG Batch section |
| Q19 | The original code had `SentinelModel(num_classes=1)` as the default. Walk me through the failure this caused. | `sentinel_model.py` Fix #3 |
| Q20 | I see this codebase was AI-assisted. Walk me through the bugs you found and fixed in the fusion layer. | Phase 8 audit fixes narration — the meta-skill |
| Q21 | Why is F1-macro the early stopping metric and not accuracy or F1-micro? | Phase Cross-Cutting §6 Evaluation Metrics |
| Q22 | Explain what `build_multilabel_index.py` does. Why GROUP BY SHA256 with max()? What does WeakAccessMod exclusion tell you about the dataset? | Phase 2 + `build_multilabel_index.py` |
| Q23 | Walk me through what happens to a Reentrancy contract from raw Solidity to the classifier's sigmoid output. Use concrete node/edge/token examples. | Phases 0 + 2 + 3 + 4 |
| Q24 | The model is deployed. In 6 months, DeFi contracts look very different from your training set. How do you detect and respond to this? | `drift_detector.py` + `promote_model.py` + Phase 7 |
| Q25 | Describe SENTINEL's per-class threshold tuning. Why 0.5 is wrong for every class, and how the tie-breaking rule reflects the purpose of the system. | `tune_threshold.py` + Phase 5 |

---

## Master Timeline — 8 Weeks

| Week | Phases | Focus | Deliverable |
|------|--------|-------|-------------|
| **Week 1** | Day 0 + Phase 0 | Setup, foundations, all 10 vulnerability classes, EVM primer | Phase 0 teach-back passed without notes |
| **Week 2** | Phase 1 + Phase 2 (first half) | Locked contracts, graph_schema, hash systems, graph extraction | Trace one contract from `.sol` to `graph.x` verbally |
| **Week 3** | Phase 2 (second half) | `build_multilabel_index`, `dual_path_dataset`, truncation analysis, create_splits | Full data pipeline narration end-to-end |
| **Week 4** | Phase 3 | Both encoders. Draw PyG Batch mechanics on paper first. | Tensor shape diagram for the full forward pass |
| **Week 5** | Phase 4 + Phase 5 | Fusion (most complex) and training. Read `fusion_layer.py` with a notepad for each fix. | Fusion whiteboard teach-back + training loop explanation |
| **Week 6** | Phase 6 | Inference pipeline. Actually run the API locally and call `/predict`. | Live API demo + walk through sliding window out loud |
| **Week 7** | Phase 7 + Phase 8 | MLOps and tests. Run `dvc status`. Run `pytest --cov`. | Full MLOps cycle teach-back |
| **Week 8** | Review + Interview prep | Answer all 25 interview questions without notes. Timed. Out loud. | Mock interview: 25 questions under 2 minutes each |

---

## Final Verification Checklist

Before considering this study plan complete, verify each item:

- [ ] Can explain SENTINEL end-to-end in under 3 minutes without notes
- [ ] Can write LoRA formula from scratch on a whiteboard
- [ ] Can name all 10 vulnerability classes and describe one Solidity example each
- [ ] Can trace a `.sol` file to model input naming every function and file
- [ ] Can draw PyG Batch mechanics (Batch.from_data_list + to_dense_batch) from memory
- [ ] Can explain all 7 key audit fixes using "original did X, caused Y, I changed it to Z"
- [ ] Can narrate the full MLOps cycle (validate → train → tune → stage → baseline → promote)
- [ ] Can describe the DVC model: what it solves, how `.dvc` files work, what `dvc pull` does
- [ ] Passed all 25 interview questions out loud, timed, without looking at notes
- [ ] `pytest ml/tests/ -v` passes cleanly on your machine
- [ ] `dvc status` shows clean (data in sync with git)
- [ ] Can explain why F1-macro is the right early stopping metric for this task

---

*This document was synthesised from two prior roadmap versions and extended with domain knowledge, EVM primer, evaluation metrics, WandB/MLflow integration notes, iterative stratification details, Docker/infrastructure context, the Senior's Angle framework, SMART objectives, and a 25-question interview bank. It supersedes all previous versions.*
