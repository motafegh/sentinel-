# SENTINEL ML — Roadmap 1 of 3: Foundations & Data Pipeline

> **Covers:** Meta framework · Environment setup · Conceptual foundations · Schema contract · Data pipeline (Solidity → tensors)
> **Weeks:** 1–3 of the 8-week plan
> **Next:** → [Roadmap 2: Architecture & Training](SENTINEL_Roadmap_2_Architecture_and_Training.md)
> **Then:** → [Roadmap 3: Production, MLOps & Interview Prep](SENTINEL_Roadmap_3_Production_and_Interview.md)

---

## How to Use This Document

This plan answers four things for every topic:

- **What** to study and in what order
- **How deep** to go (depth signal on every topic)
- **Why** it matters — for the codebase, for interviews, or for the AI era
- **How** to study it — not just what to read, but how to actively learn it using the project's own code as the teaching material

**Three rules before you start:**

1. Work through phases in order. Do not skip Phase 0.
2. Do not skip the teach-back exercises. They are where learning becomes ownership.
3. When you hit a concept you do not understand, do not push past it. Stop, open an AI session, resolve it using the project file you are currently reading as the anchor, then continue.

---

## SMART Objectives

Before you study a single file, write these down and keep them visible.

| Objective | Specific | Measurable | Achievable | Relevant | Time-bound |
|-----------|----------|------------|------------|----------|------------|
| Own the architecture | Explain SENTINEL end-to-end, draw it on a whiteboard from memory | Can answer all 25 interview questions without notes | 8 weeks of structured study | Core to Senior ML Engineer role | Week 8 |
| Own the data pipeline | Trace a raw `.sol` file to model input, name every function and file | Teach-back exercises passed without notes | 2 weeks of pipeline focus | Data pipeline literacy separates ML engineers from researchers | Week 4 |
| Own the production ML layer | Describe monitoring, drift, rollback, threshold tuning, model promotion | Can diagnose any of the 5 production failure modes cold | 2 weeks of inference focus | Production ownership is the highest-value senior skill | Week 7 |
| Own AI-generated code | Narrate audit fixes as your own code review | Can recite each fix with "the original did X, which caused Y, I changed it to Z" | Ongoing across all phases | The 2026 meta-skill. Companies test this directly. | Week 6 |
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

## Reading Method for 🔴 Master Files

Every file marked 🔴 gets this two-pass treatment. This replaces plain reading.

**Pass 1 — Structural scan (10 minutes max)**
Read only the `__init__` method and the `forward()` or main function signature. Write down in your own words:
- What does this component take as input? (shapes)
- What does it produce as output? (shapes)
- What major sub-components does it own?

Do not proceed to Pass 2 until you can answer all three without looking.

**Pass 2 — Annotated read (main time budget)**
Go through the file block by block. Above every non-trivial block, write a comment in your own words explaining what it does AND why it has to be there. When you cannot write the comment, that is the stop signal — open an AI session, resolve it using the file as context, write the comment, then continue. Do not proceed past a block you cannot annotate.

The questions listed in each phase are then your verification step after the annotated read — not a substitute for it.

---

## What Senior ML Companies Actually Test in 2026

Before you start, understand what you are preparing for. Map every phase back to these six dimensions:

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

**6. Model interpretability and explainability**
For a security tool, "this contract is flagged" is not a complete answer. Can you describe how a developer would understand *why* the model flagged it? Attention weight visualization, SHAP values, and confidence calibration are increasingly tested at senior levels. SENTINEL does not implement these — but you should be able to describe how you would add them.

---

## Priority Intelligence — What Companies Are Hiring For

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

Since the code is AI-written, your job is not to memorise what it does but to critique and explain the design decisions. For every major file, run through these five questions before moving on:

1. **Why this architecture and not the alternative?** (GATConv vs GCNConv? Cross-attention vs concatenation? FocalLoss vs BCE+pos_weight?)
2. **What are the input and output shapes?** Write them down — tensor shape fluency is directly tested.
3. **What would break if this changed?** (e.g., changing `NODE_FEATURE_DIM` without rebuilding graphs — trace the failure chain)
4. **What is this component protecting against?** (Focal loss → class imbalance; drift detector → model staleness; atomic cache write → partial file corruption)
5. **How does this connect to the file I read before?** (pos_weight in `build_multilabel_index.py` → `focalloss.py`; content hash in `hash_utils.py` → cache key in `cache.py`)

---

## ⚡ Principle vs Project-Specific — How to Think About This Codebase

**This is the most important meta-skill for using this roadmap correctly.**

Throughout these phases, you will encounter specific values: r=8, pos_weight=68, 96.6% truncation, 68K graphs, 512 tokens, 10 classes, `FEATURE_SCHEMA_VERSION`. You must learn ALL of them — but you must learn them at two levels simultaneously:

| Level | What to learn | How it transfers |
|-------|---------------|-----------------|
| **Project-specific** | The exact value and why it was chosen for SENTINEL | Lets you narrate the project fluently in interviews |
| **Generalizable principle** | The reasoning process for choosing that value in ANY project | Lets you answer "how would you approach this from scratch?" |

**Example:** The roadmap tells you `r=8` for LoRA. The project-specific knowledge is: r=8 was chosen for CodeBERT on SENTINEL because it gives ~295K trainable params vs 124M frozen, which is sufficient expressiveness for smart contract domain adaptation. The generalizable principle is: LoRA rank should be the minimum that captures task-specific variation without overfitting — start at r=4 for simple domain shift, r=8–16 for complex task transfer, validate by watching validation loss stability. If the project ever changes rank, the principle stays true.

Every time you encounter a hardcoded value, ask both levels. The roadmap will flag these moments with ⚡.

---

## When You Find Something Unexpected

This roadmap does not cover every line of every file. When you read something not explicitly discussed:

1. Apply the Senior's Angle five questions immediately.
2. Ask: "Is this a pattern I've seen elsewhere in the codebase, or a unique design?"
3. Ask: "What would break if this were removed or changed?"
4. If you cannot answer those in 5 minutes, open an AI session with that specific code block as context and work through it.

Do not skip unexplained code. The goal is zero unexplained blocks by the end of Week 8.

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
**Goal:** Answer any "why X instead of Y" question for the six major design choices without looking at any file.
**Time:** 4–5 hours total

> There are no code files to open in this phase. These are the conceptual foundations you need before touching any source file. Interviewers ask these questions before asking anything about the code.

---

### Topic A — Why GNNs for smart contract analysis 🔴

**Interview question:** "Why not just feed the source code to a second transformer?"

- Solidity vulnerability patterns are fundamentally **structural**, not textual. A `withdraw()` function called by a function called by an external-facing function three hops away in the call graph is adjacent in the graph but 400 lines apart in the source text.
- GNNs propagate information along edges — "this node reads this state variable" is a READS edge that a text model cannot directly encode.
- The five SENTINEL edge types (`CALLS`, `READS`, `WRITES`, `EMITS`, `INHERITS`) encode the semantic relationships that define reentrancy, integer overflow, and access control vulnerabilities.
- **Why GAT specifically over GCN:** GCN uses fixed normalised adjacency weights. GAT learns which neighbours to attend to. A `withdraw()` node should attend more strongly to the state variable it reads than to an unrelated helper it calls. GAT makes this learnable.

**⚡ Broader GNN landscape — know these exist:**
GNNs for code/graph analysis are an active research area. You should be able to place GAT in the landscape:

| Architecture | Key property | When to prefer over GAT |
|---|---|---|
| **GCN** | Fixed normalised weights | Simpler baseline; faster on homogeneous graphs |
| **GraphSAGE** | Samples a fixed neighbourhood | Scalability to very large graphs (millions of nodes) |
| **GIN (Graph Isomorphism Network)** | Maximum discriminative power (Weisfeiler-Lehman test) | When graph structure differences are subtle |
| **Graph Transformer** | Global attention between all node pairs | When local neighbourhood is insufficient; at the cost of O(N²) complexity |
| **GAT (SENTINEL's choice)** | Learned attention weights per neighbour | When neighbour importance varies and must be learned — reentrancy, access control |

The generalizable principle: **choose the GNN architecture based on whether importance of neighbourhood relationships is fixed (GCN), needs sampling (GraphSAGE), needs maximum power (GIN), or needs to be learned per-neighbour (GAT).**

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

**⚡ Project-specific vs generalizable:**
The project uses `r=8`. The generalizable principle for rank selection:
- `r=1`: rank-1 update, very constrained, useful only for the simplest domain shifts
- `r=4`: appropriate for simple domain adaptation (e.g., same language, different domain vocabulary)
- `r=8–16`: practical sweet spot for complex domain transfer (e.g., general code → smart contract analysis)
- `r=64+`: approaching full fine-tune; use only when r=16 clearly underfits on validation set
- `r=d` (e.g., r=768): equivalent to full fine-tune with extra parameters — avoid

The right process: start at r=8, validate F1 stability on the validation set across epochs, increase only if you observe consistent underfitting. Never increase rank without re-tuning learning rate.

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

**⚡ Project-specific vs generalizable:**
`pos_weight=68` for DenialOfService is specific to the class distribution in the BCCC-2024 dataset. The generalizable principle: pos_weight should be computed fresh any time the training data changes. The formula is always `neg_count / pos_count` per class. A pos_weight above ~100 is a signal that a class may be too rare for reliable learning — consider either getting more positive examples or removing that class from training.

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

**Note on Slither as the label source:** SENTINEL's labels come from Slither's static analysis. Slither has its own false positive rate — some contracts labelled as vulnerable may not be, and some vulnerabilities may be missed. When asked in an interview about label quality, be able to say: "Labels are Slither-derived, which means they inherit Slither's precision/recall characteristics. A more rigorous approach would layer human audit annotations on top of static analysis labels."

---

### Topic G — EVM and Blockchain Primer 🟡

- **EVM (Ethereum Virtual Machine):** All Solidity contracts compile to EVM bytecode. Stack-based, stateful, gas-metered.
- **Gas:** Every operation costs gas. This creates the attack surface for DoS and GasException.
- **`msg.sender`:** The address calling the current function. Central to access control vulnerabilities.
- **`delegatecall`:** Executes code from another contract *in the context of the current contract*. Storage is the caller's. Extremely powerful and extremely dangerous.
- **State vs memory:** State variables persist on-chain. Memory is ephemeral per call. The READS/WRITES edges in SENTINEL's graph schema directly encode state variable access.
- **Why this matters for SENTINEL:** The graph's edge types (`CALLS`, `READS`, `WRITES`, `EMITS`, `INHERITS`) map directly to the EVM execution model. You cannot explain why the graph schema is structured this way without understanding EVM semantics.

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

---

### Concept Injection — Before Opening Any File

**Python `assert` at module import time**
When a statement like `assert len(FEATURE_NAMES) == NODE_FEATURE_DIM` appears at the top level of a module (not inside a function or class), Python executes it the moment the module is imported — before any user code runs. This means the check fires at startup, making an invalid schema structurally impossible to load. Open `graph_schema.py` and find this line. Ask yourself: what would happen if this were inside `graph_extractor.py`'s `contract_to_pyg()` function instead? It would only fire on first use, potentially after 10,000 graphs had been processed. Import-time assertion is a deliberate design decision to make the failure immediate and unmissable.

**`hashlib.md5()` and `hashlib.sha256()` in Python**
Both take bytes and produce a fixed-length hex digest. The pattern in this codebase is:
```python
hashlib.md5(path_string.encode()).hexdigest()      # path hash
hashlib.sha256(content_bytes).hexdigest()          # content hash
```
The key property that matters here is *stability*: same input → same output, always, across machines and Python versions. Collision resistance is irrelevant for this use case — the hashes are used as identifiers, not as security primitives. Open `hash_utils.py` immediately after reading this, find both functions, and confirm you can read them without any friction.

---

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
- The `assert len(FEATURE_NAMES) == NODE_FEATURE_DIM` fires at import time, not in a test. Why is import-time the right boundary for this assertion?

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

Skipping step 4 is the most dangerous omission: the inference cache serves old graphs to the new model silently.

### Code Directing Exercise

Write the prompt you would give an AI to generate a `graph_schema.py` equivalent for a new project. Your prompt must specify: why the assert must be at module level, why ordinal encoding is used for visibility, what `FEATURE_SCHEMA_VERSION` protects against, and why the edge types are the specific five they are. If you can write this prompt precisely, you own the schema design.

### Teach-Back Exercise

"If I add a new node feature, what are the exact 4 steps, and what happens at every level of the system if I skip step 4?" Walk through: dataset rebuild, retrain, inference cache, the cache key format, and the validation check in `InferenceCache.get()`.

---

## Phase 2 — From Solidity to Tensors

**Theme:** Trace a `.sol` file from bytes on disk to the two tensors `SentinelModel.forward()` expects.
**Strategy:** Trace ONE contract — a Reentrancy contract from the dataset — through every step. If you can narrate that journey end-to-end, you understand the whole data pipeline.
**Goal:** Describe every transformation step, the exception hierarchy, and why two hash functions exist.
**Time:** 3–4 hours

---

### Concept Injection — Before Opening Any File

**The PyG `Data` object — SENTINEL-shaped**
Before reading `graph_extractor.py`, manually construct the smallest possible SENTINEL-shaped graph in Python. Open a Python shell and run:

```python
import torch
from torch_geometric.data import Data

# A 3-node SENTINEL graph (CONTRACT + 1 STATE_VAR + 1 FUNCTION)
# NODE_FEATURE_DIM = 8, so each node needs 8 features
x = torch.zeros(3, 8)            # [num_nodes, NODE_FEATURE_DIM]

# One CALLS edge: node 2 (function) calls node 0 (contract)
edge_index = torch.tensor([[2], [0]], dtype=torch.long)  # [2, num_edges]

# Edge type as one-hot or index — SENTINEL uses edge type index
edge_attr = torch.tensor([[1]], dtype=torch.float)       # [num_edges, edge_feature_dim]

# Multi-label target: 10 classes, this contract has Reentrancy
y = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)

graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
print(graph)
print(graph.x.shape, graph.edge_index.shape)
```

Run this before reading `graph_extractor.py`. Then when you read how the extractor builds graphs, you already know what it is building towards.

**Python `multiprocessing` — the one concept that matters here**
`ast_extractor.py` uses multiprocessing for batch extraction. The key constraint: you cannot pass a CodeBERT tokenizer from the main process to a worker process because complex objects with C extensions often fail to pickle. The solution is `init_worker()` — a function that each worker process calls at startup to load its own tokenizer instance. Open `ast_extractor.py`, find `init_worker()`, and read it with this context in mind.

---

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/preprocessing/graph_extractor.py` | 🔴 Master | Solidity → PyG graph; typed exceptions; the replication constraint |
| `ml/data_extraction/ast_extractor.py` | 🔴 Master | Batch orchestration; multiprocessing; checkpoint/resume |
| `ml/data_extraction/tokenizer.py` | 🟡 Understand | CodeBERT tokenisation; truncation detection; worker init |
| `ml/src/inference/preprocess.py` | 🟡 Understand | Online path vs offline; temp file; cache integration |
| `ml/scripts/build_multilabel_index.py` | 🔴 Master | SHA256→multi-hot; GROUP BY logic; WeakAccessMod exclusion; pos_weight |
| `ml/scripts/create_splits.py` | 🟡 Understand | Iterative stratification for multi-label; split ratio; determinism via seed |
| `ml/src/datasets/dual_path_dataset.py` | 🔴 Master | Paired loading; binary vs multi-label mode; RAM cache; collate_fn |
| `ml/scripts/validate_graph_dataset.py` | 🟢 Survey | Pre-retrain data validation gate |
| `ml/scripts/analyse_truncation.py` | 🟡 Understand | Read as engineering decision document — why 96.6% truncation motivates sliding window |

### Questions to answer

**graph_extractor.py:**
- The comment says "REPLICATION CONSTRAINT — DO NOT CHANGE WITHOUT RETRAINING." What was the concrete silent failure before this function existed?
- Node insertion order: `CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs`. What breaks if you reverse two of them? Trace it to a tensor shape error inside GATConv.
- `SolcCompilationError` → HTTP 400; `SlitherParseError` → HTTP 500. Why is that distinction meaningful to a caller?

**ast_extractor.py:**
- It's "orchestration only" after the refactor. What 3 things does it still own exclusively?
- Batch error policy: return `None` on `GraphExtractionError`, re-raise on `RuntimeError`. How does `contract_to_pyg()` implement this without modifying the shared function?

**tokenizer.py:**
- Truncation detection uses a separate `encode()` call without truncation. Why can't you just check `attention_mask.sum() == 512`?

**build_multilabel_index.py:**
- `GROUP BY SHA256 then max()` = OR-reduction for multi-label. Why OR and not AND?
- `WeakAccessMod` is excluded with zero training examples. What would happen at training time if it were included?
- Trace how `pos_weight` per class flows from this script to `trainer.py` and ultimately to `BCEWithLogitsLoss`.

**dual_path_dataset.py:**
- What happens in `__getitem__` when the graph `.pt` exists but the matching token `.pt` is missing?
- How does `collate_fn` handle variable-size graphs via `Batch.from_data_list`?
- The RAM cache: what is the cache key, and what integrity check prevents stale data after a schema change? (Audit #11)

### analyse_truncation.py — read as a decision document

This script exists because 96.6% of contracts exceed 512 tokens. Read the *output section* and ask: what would have happened if the team had ignored this and used CLS-only on truncated inputs? Which vulnerability classes would have suffered most — specifically those whose vulnerable code appears late in a contract?

**⚡ Project-specific vs generalizable:**
96.6% is the truncation rate for this specific dataset. The generalizable principle: always run a truncation analysis before committing to a context window strategy. For any new code/text classification project, compute the distribution of sequence lengths and ask: "What percentage of inputs are truncated by my model's context window?" If it's above ~30%, CLS-only is likely insufficient and a windowing or hierarchical approach is warranted.

### Code Directing Exercise

Write the prompt you would give an AI to generate `dual_path_dataset.py`. Your prompt must specify: the paired loading contract (graph + token must correspond to the same contract), what happens on a missing pair, why `Batch.from_data_list` is used in `collate_fn` and not `torch.stack`, and the binary vs multi-label label shape difference. If your prompt is precise enough that the AI output would be functionally correct, you own the dataset design.

### Teach-Back Exercise

Trace the full path of a source string submitted to the API. Start at `process_source()` in `preprocess.py`, end at `graph.x` entering `GNNEncoder.forward()`. Name every function, every file written/read (including the temp file and why it exists), and the exact cache key at the point it is constructed.

---

## Phase 8 — Test Files for Phases 0–2

> Read these test files alongside the phase they cover — not all at once at the end.

### Concept Injection — Before Reading Any Test File

**Read `conftest.py` first — it is the fastest way to learn input/output contracts**
`conftest.py` contains shared fixtures that define valid input shapes for every component. Before reading any test, read `conftest.py`. The synthetic graph it generates tells you exactly what `GNNEncoder` expects. The synthetic token tensors tell you what `TransformerEncoder` expects. This is the fastest path to shape fluency in the entire codebase.

**Run `pytest --cov=ml/src --cov-report=term-missing` after each phase**
Coverage reports show which lines in source files have no test coverage — those are the complex branching paths to trace manually.

### Test files for this roadmap

| File | Depth | Read with phase | What it confirms |
|------|-------|-----------------|-----------------|
| `ml/tests/conftest.py` | 🔴 Master | Phase 1 | Shared fixtures; fastest way to learn input/output contracts |
| `ml/tests/test_preprocessing.py` | 🟡 Understand | Phase 2 | Mocked Slither + CodeBERT; temp file handling |
| `ml/tests/test_dataset.py` | 🟡 Understand | Phase 2 | Pairing; split indices; collate function; binary vs multi-label label shapes |

---

## Weeks 1–3 Timeline

| Week | Phases | Focus | Deliverable |
|------|--------|-------|-------------|
| **Week 1** | Day 0 + Phase 0 | Setup, all 7 conceptual foundations, 10 vulnerability classes, EVM primer | Phase 0 teach-back passed without notes |
| **Week 2** | Phase 1 + Phase 2 first half | Locked contracts, graph_schema, hash systems, graph extraction, AST extractor | Trace one contract from `.sol` to `graph.x` verbally |
| **Week 3** | Phase 2 second half | `build_multilabel_index`, `dual_path_dataset`, truncation analysis, splits | Full data pipeline narration end-to-end |

**Continue with → [Roadmap 2: Architecture & Training](SENTINEL_Roadmap_2_Architecture_and_Training.md)**
