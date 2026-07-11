# 06. RAG: Hybrid FAISS+BM25 Retrieval with Reciprocal Rank Fusion

> **Prerequisites:** [01. The Audit Pipeline] — `rag_research` is a deep-path node. [02. Evidence Model & Fuse()] — RAG emits `Evidence(kind=SEMANTIC)`. [05. MCP Architecture] — RAG server runs on port 8011, lazy-loaded.
> **Next:** [07. Gateway Production] covers the HTTP boundary and health monitoring that probes the RAG server.
> **Cross-ref:** [08. Evaluation Framework] covers the reliability fitting that produces RAG's per-class reliability weight. [04. Reproducibility] covers why RAG is skipped in `SENTINEL_DETERMINISTIC` mode.
> **Scope:** This doc covers the complete RAG pipeline: chunking → embedding → FAISS+BM25 hybrid retrieval → Reciprocal Rank Fusion → metadata filtering → cross-encoder reranking. It also covers the P7 zero-match diagnosis and the query-construction fixes. It does NOT cover the MCP transport (see Doc 05) or how RAG evidence flows into `fuse()` (see Doc 02).
> **TL;DR:** SENTINEL's RAG system retrieves relevant DeFi exploit post-mortems from a 752-chunk corpus (DeFiHackLabs) using hybrid retrieval: FAISS for semantic similarity ("flash loan attack" finds "price oracle manipulation") + BM25 for keyword matching (exact CVE numbers, protocol names, tx hashes). Reciprocal Rank Fusion (RRF) combines both rankings without needing score calibration: `score(chunk) = Σ 1/(rank + 60)`. The embedding model is `nomic-embed-text-v1.5` (768-dim, runs via LM Studio). P7 fixed a zero-match issue where 26.5% of audits got empty RAG results — root cause: the query contained Solidity code (confusing the text embedder) and ML class names ("IntegerUO") that didn't match the corpus vocabulary ("integer overflow"). The fix: map class names to descriptive keywords, remove code from queries, and add a fallback query on zero results.

---

## The Problem: One Retrieval Method Is Not Enough

### Why not just semantic search?

FAISS semantic search finds chunks with similar *meaning*. Query "flash loan attack on lending protocol" finds chunks about "price oracle manipulation" — different words, same concept. This is powerful. But it has a blind spot:

**Teaching: the semantic blind spot.** FAISS can't find exact identifiers. If you search for "Euler Finance 0xc310a0af" (a specific tx hash from a real exploit), FAISS's embedding of that string is a 768-dim vector that may or may not be close to the chunk containing that hash. Embedding models encode *meaning*, not *strings*. A CVE number like "CVE-2023-9999" has no semantic meaning — it's an arbitrary identifier. The embedding of "CVE-2023-9999" is essentially noise.

**The test:** search your corpus with a protocol name ("Wormhole"), then with a tx hash ("0x0a6ef..."). If semantic search finds the protocol but misses the tx hash, you need keyword search too.

### Why not just keyword search?

BM25 keyword search finds chunks with matching *words*. Query "Euler Finance" finds chunks containing "Euler Finance" — exact match. But it has a different blind spot:

**Teaching: the keyword blind spot.** BM25 can't find synonyms. If you search "flash loan attack," BM25 won't find a chunk that says "instantaneous borrowing exploit" — the words don't match, even though the concept is identical. BM25 is a bag-of-words model: it counts word overlap, not meaning.

### The solution: hybrid retrieval with RRF

You need *both*: semantic search for conceptual similarity, keyword search for exact matches. But how do you combine two ranking systems with different score scales? FAISS returns L2 distances (lower = better); BM25 returns TF-IDF scores (higher = better). You can't add them directly.

**Reciprocal Rank Fusion (RRF)** solves this by operating on *ranks*, not scores:

```
RRF_score(chunk) = Σ  1 / (rank_i + K)

where:
  rank_i = the chunk's position in retrieval system i's ranked list (0-based)
  K      = 60 (empirically tuned constant — controls rank-1 dominance)
```

A chunk ranked #1 in FAISS and #3 in BM25 gets: `1/(0+60) + 1/(2+60) = 0.0167 + 0.0161 = 0.0328`. A chunk ranked #1 in *both* gets: `1/60 + 1/60 = 0.0333`. RRF doesn't care about score magnitudes — it only cares about *where* each system ranks the chunk.

**Teaching: why K=60?** K controls how much rank-1 dominates. With K=1, rank-1 gets `1/1 = 1.0` and rank-2 gets `1/2 = 0.5` — rank-1 is 2× rank-2, so rank-1 dominates. With K=60, rank-1 gets `1/60 = 0.0167` and rank-2 gets `1/61 = 0.0164` — nearly equal. K=60 makes the fusion smooth: being ranked #1 in one system is only slightly better than being ranked #2. The value 60 is from the original RRF paper and works well for this corpus size.

---

## How We Arrived at This Design

> **How to read this section:** Each step shows the question, *how to reason about it*, and the chain of logic connecting the answer to the design.

### Step 1 — Identify the invariant

**The question:** What must always be true about RAG evidence?

**Applying the "useless or dangerous" test:**

| Candidate property | If violated → | Verdict |
|---|---|---|
| RAG adds evidence only when relevant chunks exist | Irrelevant chunks → noise in `evidence_list` → biased `fuse()` | **Invariant** |
| RAG query uses the same embedding model as the index | Different vector space → garbage results → wrong evidence | **Invariant** |
| RAG skips gracefully when no classes are flagged | Query "unknown vulnerability" → garbage embeddings → noise | **Invariant** |

**The reasoning chain:** RAG evidence feeds into `fuse()` as `Evidence(source="rag", kind=SEMANTIC)`. If the retrieved chunks are irrelevant (returned because the query was bad, not because the chunks match), the evidence is noise — it contributes `reliability × strength` to the confidence sum, potentially pushing a class from SAFE to DISPUTED on false premises. So: RAG must only add evidence when the retrieval is *meaningful* — i.e., when the query is well-constructed and the results are relevant.

### Step 2 — Identify the constraints

**Constraint A: The embedding model is a text model, not a code model.**
- *Why:* `nomic-embed-text-v1.5` is trained on natural language text. Solidity code (`mapping(address => uint256) balances`) is not natural language — the embedding of code-like text is semantically meaningless.
- *What this forces:* The RAG query must be *natural language*, not Solidity code. The `rag_research` node constructs queries from vulnerability class keywords, not from the contract source.

**Constraint B: ML class names don't match corpus vocabulary.**
- *Why:* The ML model outputs class names like "IntegerUO" and "TransactionOrderDependence". The corpus uses phrases like "integer overflow" and "front-running". These don't match — the embedding of "IntegerUO" is not close to the embedding of "integer overflow" because they're different strings with different tokenizations.
- *What this forces:* A mapping from ML class names to RAG-friendly keywords: `_VULN_CLASS_TO_RAG_KEYWORDS = {"IntegerUO": "integer overflow underflow arithmetic", ...}`.

**Constraint C: The embedding model must be the same at index time and query time.**
- *Why:* FAISS indexes vectors in a specific embedding space. If the index was built with `nomic-embed-text-v1.5` and the query is embedded with `text-embedding-3-small`, the query vector is in a different space — FAISS L2 distances are meaningless.
- *What this forces:* The `Embedder` class uses `get_embedding_model()` which returns the same model configuration at index time and query time. The model name is stored in `index_metadata.json` for verification.

### Step 3 — Eliminate alternatives

| Approach | How it breaks | When it breaks | Eliminate? |
|---|---|---|---|
| **Dense-only** (FAISS semantic) | Misses exact identifiers (CVE numbers, tx hashes, protocol names) | When the query contains specific identifiers | **Yes** |
| **Sparse-only** (BM25 keyword) | Misses synonyms and conceptual similarity | When the query uses different words than the corpus | **Yes** |
| **Learned fusion** (neural ranker) | Needs labeled retrieval data (relevance judgments). We have none. | Always — no training data | **Yes** |
| **RRF hybrid** (FAISS + BM25 + RRF) | Assumes both systems are independent (they're not — both use the same text). No score calibration. | When you need fine-grained score comparison | **No** — survives. |

**The reasoning:** Dense-only fails on exact identifiers (FAISS can't find "0xc310a0af" meaningfully). Sparse-only fails on synonyms (BM25 can't match "flash loan" to "instantaneous borrowing"). Learned fusion needs relevance judgment data we don't have. RRF hybrid combines both rankings without needing score calibration or training data — its only assumption (independence) is approximately true for a text embedder and a keyword matcher.

### Step 4 — Stress-test (the P7 zero-match diagnosis)

**The test:** Run the pipeline on 83 contracts. Count how many get zero RAG results.

**The result:** 22/83 (26.5%) got zero RAG results. This is a stress-test failure — over a quarter of audits get no exploit precedent.

**Root cause analysis (three issues found):**

1. **Query was "unknown vulnerability":** 22 contracts had `topic="unknown"` (ML returned no flagged classes). The query was literally `"smart contract unknown vulnerability exploit attack pattern"` — the embedding of this is near-random noise. Fix: skip RAG when no classes are flagged (nothing to search for).

2. **Solidity code in the query:** The old query included `contract_code[:200]` — 200 chars of Solidity source. The text embedder can't handle `mapping(address => uint) balances` — it produces a garbage embedding. Fix: remove code from the query, use mapped keywords instead.

3. **Class name mismatch:** ML outputs "IntegerUO" but the corpus says "integer overflow". The embedding of "IntegerUO" doesn't match the embedding of "integer overflow" because they're different strings. Fix: `_VULN_CLASS_TO_RAG_KEYWORDS` maps all 10 ML class names to descriptive phrases.

### Step 5 — Measure

**Before P7 fix:** 26.5% of reports had 0 RAG results (22/83).
**After P7 fix:** 0% zero-match on the same corpus (all flagged contracts get results). The fix didn't improve *relevance* (that would need relevance judgments) — it fixed *recall* (the query now actually reaches the corpus meaningfully).

> **The method, summarized:** (1) Find invariants — RAG must only add relevant evidence. (2) Find constraints — text embedder ≠ code embedder, ML names ≠ corpus vocabulary. (3) Eliminate single-method approaches by finding their blind spots. (4) Stress-test with real data — the 26.5% zero-match rate was a measurable failure. (5) Fix root causes, not symptoms — don't tune BM25 params when the query itself is garbage.

---

## The Solution: The Complete RAG Pipeline

### Architecture overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INDEX BUILD TIME (one-time, ~3 min)                                    │
│                                                                         │
│  DeFiHackLabs repo (726 files)                                          │
│       │                                                                 │
│       ▼                                                                 │
│  Fetcher → Document objects (750 docs)                                   │
│       │                                                                 │
│       ▼                                                                 │
│  Chunker (RecursiveCharacterTextSplitter, 1536 chars, 128 overlap)      │
│       │                                                                 │
│       ▼                                                                 │
│  Embedder (nomic-embed-text-v1.5 via LM Studio, 768-dim, batch=32)      │
│       │                                                                 │
│       ▼                                                                 │
│  FAISS IndexFlatL2 (776 vectors) + BM25Okapi + chunks.pkl + metadata     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  QUERY TIME (per audit, ~100ms)                                         │
│                                                                         │
│  rag_research node                                                      │
│       │                                                                 │
│       ├─ ML result → top vulnerability class (e.g., "Reentrancy")       │
│       ├─ _VULN_CLASS_TO_RAG_KEYWORDS["Reentrancy"]                     │
│       │    → "reentrancy reentrant call"                                │
│       ├─ Query: "smart contract reentrancy reentrant call               │
│       │          vulnerability exploit attack pattern"                  │
│       │                                                                 │
│       ▼                                                                 │
│  _call_mcp_tool("search", {query, k=5})                                │
│       │                                                                 │
│       ▼                                                                 │
│  RAG MCP Server (port 8011)                                             │
│       │                                                                 │
│       ▼                                                                 │
│  HybridRetriever.search(query, k=5)                                     │
│       │                                                                 │
│       ├─ Step 1: Embedder.embed_query(query) → 768-dim vector           │
│       ├─ Step 2: FAISS.search(vector, 20 candidates) → ranked list A    │
│       ├─ Step 3: BM25.get_scores(query_tokens) → ranked list B          │
│       ├─ Step 4: RRF fusion: score = Σ 1/(rank + 60) from both lists   │
│       ├─ Step 5: _apply_filters (vuln_type, date_gte, loss_gte, ...)    │
│       └─ Step 6: (optional) _rerank with cross-encoder                 │
│       │                                                                 │
│       ▼                                                                 │
│  Top-k Chunk objects (content + metadata + RRF score)                   │
│       │                                                                 │
│       ▼                                                                 │
│  emit_rag_evidence() → Evidence(source="rag", kind=SEMANTIC)            │
│       │                                                                 │
│       ▼                                                                 │
│  state["evidence_list"] (append-reducer) → fuse()                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Worked example: tracing a Reentrancy query

**Input:** ML flags Reentrancy at probability 0.85.

**Step 1: Query construction** (`rag_research.py:87-111`):
```python
topic = "Reentrancy"
rag_keywords = _VULN_CLASS_TO_RAG_KEYWORDS["Reentrancy"]  # → "reentrancy reentrant call"
query = f"smart contract {rag_keywords} vulnerability exploit attack pattern"
# → "smart contract reentrancy reentrant call vulnerability exploit attack pattern"
```

**Step 2: Embedding** (`embedder.py:166-198`): The query is sent to LM Studio's `/v1/embeddings` endpoint. The `nomic-embed-text-v1.5` model produces a 768-dim vector. If LM Studio is temporarily unavailable, `_embed_batch_with_retry` retries 3 times with exponential backoff (1s, 2s, 4s).

**Step 3: FAISS semantic search** (`retriever.py:153-163`):
```python
query_vector = self.embedder.embed_query(query)      # 768-dim
query_np = np.array([query_vector], dtype=np.float32)
_, indices = self.faiss_index.search(query_np, 20)   # 20 candidates
# indices[0] = [42, 17, 89, 3, ...] — chunk indices ranked by L2 distance
```

FAISS returns the 20 nearest chunks by L2 distance. The distances are discarded — RRF only uses the *rank*, not the distance.

**Step 4: BM25 keyword search** (`retriever.py:165-175`):
```python
query_tokens = query.lower().split()
# → ["smart", "contract", "reentrancy", "reentrant", "call", ...]
bm25_scores = self.bm25.get_scores(query_tokens)
# → [0.0, 2.3, 0.0, 5.1, ...] — BM25 score per chunk
bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
# → [3, 17, 42, 89, ...] — chunk indices ranked by BM25 score
```

BM25 tokenizes the query (lowercase + whitespace split) and scores each chunk by TF-IDF. The top 20 chunks are selected.

**Step 5: RRF fusion** (`retriever.py:177-187`):
```python
rrf_scores = {}
for chunk_idx, rank in faiss_ranked:                    # FAISS ranks
    rrf_scores[chunk_idx] += 1.0 / (rank + 60)         # rank 0 → 1/60
for chunk_idx, rank in bm25_ranked:                     # BM25 ranks
    rrf_scores[chunk_idx] += 1.0 / (rank + 60)         # rank 0 → 1/60
# Chunk 42: FAISS rank 0 + BM25 rank 2 → 1/60 + 1/62 = 0.0167 + 0.0161 = 0.0328
# Chunk 17: FAISS rank 1 + BM25 rank 1 → 1/61 + 1/61 = 0.0164 + 0.0164 = 0.0328
# Chunk 3:  FAISS rank 3 + BM25 rank 0 → 1/63 + 1/60 = 0.0159 + 0.0167 = 0.0326
```

A chunk that ranks high in *both* systems gets a higher RRF score. Chunk 42 (rank 0 in FAISS, rank 2 in BM25) and chunk 17 (rank 1 in both) get similar scores — both are strong candidates.

**Step 6: Metadata filtering** (`retriever.py:243-299`): If filters are provided (e.g., `vuln_type="reentrancy"`), chunks are filtered by metadata. The `_apply_filters` method checks each chunk's metadata fields. If filtering produces zero results, a warning is logged (FIX-12 — previously silent).

**Step 7: Return top-k** (`retriever.py:212`): `results[:k]` returns the top-k chunks, each carrying its RRF score.

### The fallback query (P7 fix)

If the first query returns zero results (`rag_research.py:130-140`):
```python
if not chunks:
    fallback_query = f"{topic} vulnerability exploit"
    # → "Reentrancy vulnerability exploit" — shorter, more likely to match
    fallback_result = await _h._call_mcp_tool(...)
    chunks = fallback_result.get("results", [])
```

**Teaching: why a fallback query?** The first query is descriptive ("smart contract reentrancy reentrant call vulnerability exploit attack pattern") — 10 words. The embedding of a long query is an average of all words, which can dilute the signal. The fallback query is shorter ("Reentrancy vulnerability exploit") — 3 words, more focused. If the long query's embedding is too diffuse to match any chunk, the short query's embedding is more concentrated and more likely to find a match.

## Key Code

### The HybridRetriever class

The core retrieval engine — loads FAISS, BM25, and chunks at startup, then searches on demand:

```python
# retriever.py:59-124
class HybridRetriever:
    def __init__(self):
        self.faiss_index = faiss.read_index(str(FAISS_PATH))
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        # FIX-10: Validate FAISS ↔ chunks sync
        if self.faiss_index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index corruption: FAISS has {self.faiss_index.ntotal} vectors "
                f"but chunks has {len(self.chunks)} entries."
            )

        self.embedder = Embedder()
```

Why this matters: the FAISS↔chunks sync check (FIX-10) is a **data integrity guard**. FAISS position N maps to `self.chunks[N]`. If the index is corrupted (pipeline crash between writing `faiss.index` and `chunks.pkl`), every retrieval result is potentially wrong — FAISS returns chunk index 42, but `self.chunks[42]` is a different chunk than the one that was embedded. The check catches this at startup, not silently at retrieval time. Without it, the pipeline would produce wrong RAG evidence — and the reliability matrix would be biased, just like the Aderyn silent-skip (Rule 5C).

### The RRF fusion formula

The heart of hybrid retrieval — combines rankings without score calibration:

```python
# retriever.py:177-187
rrf_scores: dict[int, float] = {}

for chunk_idx, rank in faiss_ranked:
    rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (rank + RRF_K)

for chunk_idx, rank in bm25_ranked:
    rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (rank + RRF_K)

sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
```

Why this matters: RRF operates on *ranks*, not scores. FAISS returns L2 distances (lower is better); BM25 returns TF-IDF scores (higher is better). You can't add them — they're on different scales. RRF converts each system's output to a rank (0, 1, 2, ...) and then combines ranks with `1/(rank + K)`. This works because ranks are comparable across systems — rank 0 means "this system's top pick," regardless of the score scale.

**Teaching: the RRF formula in detail.**

```
RRF_score(chunk) = Σ_i  1 / (rank_i + K)

i ranges over all retrieval systems (FAISS, BM25)
rank_i is 0-indexed (rank 0 = first result)
K = 60 (the smoothing constant)
```

- A chunk ranked #1 in both systems: `1/60 + 1/60 = 0.0333`
- A chunk ranked #1 in FAISS only (not in BM25's top-20): `1/60 + 0 = 0.0167`
- A chunk ranked #5 in both: `1/65 + 1/65 = 0.0308`

The key insight: a chunk must be found by *both* systems to get a high score. A chunk that's #1 in FAISS but absent from BM25's top-20 gets only half the score of a chunk that's top-5 in both. This is the "both systems agree" signal — cross-system corroboration.

### The Embedder with retry logic

The embedding model client — converts text to 768-dim vectors via LM Studio, with retry on transient failures:

```python
# embedder.py:120-164
def _embed_batch_with_retry(self, batch_texts, batch_num, total_batches):
    last_error = None
    for attempt in range(MAX_EMBED_RETRIES):           # 3 attempts
        try:
            response = self.embedding_model.client.create(
                input=batch_texts,
                model=self.embedding_model.model,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            last_error = e
            if attempt < MAX_EMBED_RETRIES - 1:
                wait = EMBED_RETRY_BASE_SEC * (2 ** attempt)  # 1s, 2s, 4s
                logger.warning(f"Batch {batch_num} failed (attempt {attempt+1}): {e} — retry in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Embedding batch {batch_num} failed after {MAX_EMBED_RETRIES} attempts: {last_error}")
```

Why this matters: three production bugs are prevented here:

1. **FIX-13 (assert → RuntimeError):** The old code used `assert len(vectors) == len(chunks)`. Python's `-O` flag strips `assert` statements — in production (where `-O` is common), a vector/chunk count mismatch would silently corrupt the FAISS index. The fix: `raise RuntimeError(...)` which is never disabled. **Teaching: never use `assert` for data integrity in production code. `assert` is for development-time invariant checking; `raise` is for runtime validation.**

2. **FIX-14 (retry logic):** The old code made a single HTTP call per batch. A transient LM Studio blip (model reload, GPU spike) at batch 40 of 42 killed the entire run — discarding 40 seconds of embedding work. The fix: 3 retries with exponential backoff (1s, 2s, 4s). **Teaching: external service calls must have retry logic. The retry should be per-batch (not per-chunk), so a failure in one batch doesn't discard work done in other batches.**

3. **Direct API call (not LangChain wrapper):** The code calls `self.embedding_model.client.create()` directly, not `self.embedding_model.embed_documents()`. LangChain's `embed_documents()` adds extra formatting that LM Studio doesn't support (causes 400 BadRequestError). **Teaching: when a library wrapper adds formatting your backend doesn't support, call the underlying API directly. The wrapper is convenience; the raw API is correctness.**

### The query construction (P7 fix)

The `_VULN_CLASS_TO_RAG_KEYWORDS` mapping — the fix for the class-name mismatch:

```python
# rag_research.py:30-41
_VULN_CLASS_TO_RAG_KEYWORDS: dict[str, str] = {
    "Reentrancy":              "reentrancy reentrant call",
    "IntegerUO":               "integer overflow underflow arithmetic",
    "GasException":            "gas limit denial of service",
    "Timestamp":               "timestamp manipulation time dependence",
    "TransactionOrderDependence": "transaction ordering front running MEV",
    "ExternalBug":             "external call oracle price manipulation",
    "CallToUnknown":           "unknown external call untrusted contract",
    "MishandledException":     "exception handling error propagation",
    "UnusedReturn":            "unchecked return value ignored",
    "DenialOfService":         "denial of service DoS griefing",
}
```

Why this matters: ML class names are *internal labels* ("IntegerUO", "TransactionOrderDependence"). The RAG corpus uses *natural language* ("integer overflow", "front-running"). Without the mapping, the query "IntegerUO vulnerability exploit" embeds to a vector that doesn't match any chunk — because no chunk contains "IntegerUO." The mapping translates the internal label to the vocabulary the corpus actually uses. **Teaching: know your embedding model's domain. A text embedder encodes natural language; it doesn't encode internal identifiers. When your query contains internal labels, translate them to natural language before embedding.**

### The Chunk dataclass and Chunker

The chunking strategy — how exploit documents are split for embedding:

```python
# chunker.py:33-34, 78-94
DEFAULT_CHUNK_SIZE    = 1536   # characters, not tokens
DEFAULT_CHUNK_OVERLAP = 128

self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],  # priority: paragraph > line > sentence > word > char
)
```

Why this matters: three design decisions:

1. **1536 chars, not 512:** FIX-26 increased the chunk size from 512 to 1536. At 512 chars, most DeFiHackLabs descriptions (800-1500 chars) were split unnecessarily — the protocol header landed in chunk 0 and the attack steps were cut off in chunk 1. At 1536, most descriptions fit in one chunk, preserving semantic coherence. **Teaching: chunk size should match your document size distribution. Too small → semantic fragmentation. Too large → embedding quality degrades (the model averages too much text).**

2. **128-char overlap:** Overlap prevents information loss at chunk boundaries. If a sentence is split at "The attacker called `withdraw()` and|then drained the pool," chunk 1 ends at "called" and chunk 2 starts at "then." With 128-char overlap, chunk 2 starts at "The attacker called `withdraw()` and" — the full sentence appears in both chunks. **Teaching: overlap is insurance against boundary information loss. The overlap should be large enough to contain a full sentence (typically 50-150 chars).**

3. **RecursiveCharacterTextSplitter priority:** The splitter tries to break at paragraph boundaries (`\n\n`) first, then line breaks (`\n`), then sentence ends (`. `), then word boundaries (` `), then characters (last resort). This means: we never split mid-sentence if we can avoid it. **Teaching: the split hierarchy matters. Paragraph > sentence > word > character. Splitting at a higher level preserves more meaning per chunk.**

### The metadata filtering

The `_apply_filters` method — filter chunks by metadata fields:

```python
# retriever.py:243-299
def _apply_filters(self, chunks, filters, query=""):
    filtered = []
    for chunk in chunks:
        meta = chunk.metadata
        if "vuln_type" in filters and meta.get("vuln_type") != filters["vuln_type"]:
            continue
        if "date_gte" in filters and meta.get("date", "") < filters["date_gte"]:
            continue
        if "loss_gte" in filters and (meta.get("loss_usd") or 0) < filters["loss_gte"]:
            continue
        if "source" in filters and meta.get("source") != filters["source"]:
            continue
        if "has_summary" in filters and meta.get("has_summary") != filters["has_summary"]:
            continue
        filtered.append(chunk)

    # FIX-12: Warn when filters eliminate all candidates
    if not filtered and chunks:
        logger.warning(f"_apply_filters returned 0 results from {len(chunks)} candidates.")
    return filtered
```

Why this matters: two things:

1. **Five filter types:** `vuln_type` (exact match), `date_gte` (ISO date string comparison), `loss_gte` (USD threshold), `source` (data source name), `has_summary` (boolean). These let the pipeline narrow results — e.g., "only show reentrancy exploits from 2023+ with losses over $1M." **Teaching: metadata filters operate AFTER RRF fusion, not before. This means filtering can eliminate all candidates if the filters are too strict. The warning (FIX-12) surfaces this — previously silent.**

2. **FIX-12 (zero-result warning):** Without the warning, a filter that eliminates all chunks returns `[]` — indistinguishable from "the query found nothing." The caller has no way to know whether the query was bad or the filter was too strict. The warning logs both the query and the filters, so the developer can diagnose. **Teaching: when a filter produces zero results, log it. Silence hides the difference between "no data matches" and "filter is wrong." This is the same principle as Rule 5C — silent empty returns are ambiguous.**

### The cross-encoder reranking (optional)

The `_rerank` method — re-score candidates with a cross-encoder for higher precision:

```python
# retriever.py:216-241
def _rerank(self, query, chunks):
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c.content) for c in chunks]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]
    except ImportError:
        logger.warning("rerank=True but sentence-transformers not installed — falling back to RRF order")
        return chunks
```

Why this matters: RRF is a *bi-encoder* approach — the query and each chunk are embedded separately, then compared by distance. A *cross-encoder* reads the query and chunk together (bidirectional attention) — more accurate but slower. RRF is for recall (find candidates); cross-encoder is for precision (re-rank the top candidates). **Teaching: bi-encoder (fast, recall) → cross-encoder (slow, precision) is the standard two-stage retrieval pattern. Use bi-encoder to get 20 candidates, then cross-encoder to pick the best 5.**

The `except ImportError` fallback is important: if `sentence-transformers` isn't installed, the method returns the RRF-ranked chunks unchanged. The pipeline still works — just with slightly lower precision. **Teaching: optional dependencies should degrade gracefully. If the reranker is missing, the pipeline produces results (RRF-ranked) — it doesn't crash.**

### The deterministic mode skip

```python
# rag_research.py:74-76
if os.getenv("SENTINEL_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
    logger.info("rag_research | skipped (SENTINEL_DETERMINISTIC mode)")
    return {"rag_results": []}
```

Why this matters: RAG uses an embedding model (nomic-embed-text-v1.5) which is non-deterministic — same input, different embedding vectors across runs (CUDA non-determinism, LM Studio batching). In `SENTINEL_DETERMINISTIC` mode (for ZK proofs), RAG is skipped entirely — no embedding computation, no RAG evidence. This ensures `verdict_provable` is reproducible. See Doc 04 for the full reasoning.

**Teaching: the RAG skip is a *constraint*, not a preference.** The embedding model's non-determinism means RAG evidence can't be part of the ZK-provable tier. In normal mode, RAG evidence enriches `verdict_full` (advisory). In deterministic mode, RAG is absent — `verdict_provable` doesn't include it. This is the dual-tier architecture (Doc 02 + Doc 04) in action.

### The ExternalBug special-case query

```python
# rag_research.py:96-110
ext_calls = state.get("external_call_summary", [])
if topic == "ExternalBug" and ext_calls:
    call_str = "; ".join(
        f"{c['caller_function']}→{c['callee_contract']}.{c['callee_function']}"
        for c in ext_calls[:6]
    )
    query = (
        f"smart contract external dependency oracle manipulation "
        f"price feed vulnerability: {call_str}"
    )
```

Why this matters: "ExternalBug" is a broad class — it covers oracle manipulation, external call to unknown contract, price feed manipulation, and more. The generic keyword mapping ("external call oracle price manipulation") is too diffuse. When the graph inspector has found external calls (e.g., `getPrice→Chainlink.latestRoundData`), the query includes those specific function names — making the retrieval more precise. **Teaching: when a vulnerability class is broad, use additional context (external call summaries) to narrow the query. The function names (`getPrice`, `latestRoundData`) are meaningful keywords that BM25 can match exactly.**

## Design Decision: FAISS+BM25+RRF vs Dense-Only vs Sparse-Only vs Learned Fusion

> **How to read this section:** The table shows the options. The *elimination reasoning* shows how to think about the choice.

### The elimination process

**Dense-only (FAISS) — steel-man:** "Semantic search is the modern standard. FAISS is fast (GPU-accelerated), handles synonyms, and works on meaning. Why add the complexity of a second retrieval system?"

**Why it fails:** FAISS can't find exact identifiers. A query "Wormhole 0x0a6ef" (protocol name + tx hash) produces an embedding that's a *blend* of "Wormhole" (a meaningful word) and "0x0a6ef" (noise). The embedding is dominated by "Wormhole" and the tx hash is lost. BM25, on the other hand, matches "0x0a6ef" exactly — if the chunk contains that string, BM25 finds it. **The blind spot of semantic search is exact string matching.**

**Sparse-only (BM25) — steel-man:** "BM25 is the classic information retrieval algorithm. It's simple, fast, and handles exact matches perfectly. Why add the overhead of an embedding model?"

**Why it fails:** BM25 can't find synonyms. A query "flash loan attack" won't find a chunk that says "instantaneous borrowing exploit" — zero word overlap, zero BM25 score. FAISS finds it because the embeddings of "flash loan attack" and "instantaneous borrowing exploit" are close in vector space. **The blind spot of keyword search is semantic similarity.**

**Learned fusion (neural ranker) — steel-man:** "A neural ranker can learn the optimal combination of FAISS and BM25 scores. It would outperform any hand-tuned fusion formula."

**Why it fails:** A neural ranker needs labeled relevance judgments (query-chunk pairs with human-annotated relevance scores). We have 752 chunks and 0 relevance judgments. Training a ranker without labels is unsupervised — and unsupervised rankers are not better than RRF. **The blind spot of learned fusion is the data requirement.**

**RRF hybrid — why it survives:** It needs zero training data (just the rank formula). It covers both blind spots (FAISS for semantics, BM25 for keywords). Its assumption (independence between systems) is approximately true — FAISS and BM25 use fundamentally different scoring mechanisms. Its cost (two retrieval systems + a simple formula) is manageable.

**The reasoning principle:** "When combining two retrieval systems with different score scales, use rank-based fusion (RRF) rather than score-based fusion. RRF doesn't need score calibration, training data, or normalization. Its only assumption (independence) is approximately true when the systems use different scoring mechanisms (vector distance vs TF-IDF)."

### When this decision would be wrong

**The reversal condition:** If you have thousands of labeled relevance judgments, a learned ranker would outperform RRF. The learned ranker can capture non-linear interactions (e.g., "FAISS score matters more when the query is short") that RRF can't. The trigger: when you have >1000 labeled query-chunk pairs and the eval shows RRF is the bottleneck (precision@k is low despite good recall). Until then, RRF is the right tradeoff.

## Technology Choice: nomic-embed-text-v1.5

**The 5-question framework:**

1. **What category?** Text embedding model for semantic retrieval.
2. **What alternatives?** (a) `nomic-embed-text-v1.5` (local, 768-dim, via LM Studio), (b) `text-embedding-3-small` (OpenAI API, 1536-dim), (c) `voyage-code-2` (Voyage AI, code-specific), (d) `all-MiniLM-L6-v2` (local, 384-dim, via sentence-transformers).
3. **Why this?** Local (no API cost, no data leaving the machine), 768-dim (good balance of quality and memory), runs on LM Studio (same inference server as the LLM — no additional infrastructure), context length 8192 tokens (handles our 1536-char chunks easily).
4. **When is voyage-code-2 better?** When the corpus is primarily code (not natural language). DeFiHackLabs post-mortems are natural language (English descriptions of attacks), not code. If we were embedding Solidity source for code-to-code search, a code-specific embedder would be better.
5. **Migration trigger:** If retrieval quality is low (relevance@k is poor) and the eval shows the embedding is the bottleneck, try `text-embedding-3-small` (higher dimension, better quality, but API cost). The migration is a re-index (re-embed all chunks) — ~3 minutes for 752 chunks.

**Teaching: the embedding model must match the corpus domain.** nomic-embed-text is a *text* model — it encodes natural language well. It would encode Solidity code poorly (code is not natural language). That's why the query construction (P7 fix) removes Solidity code from the query — the embedder can't handle it. If you need to embed code, use a code-specific embedder (voyage-code-2, codebert). Don't use a text embedder on code.

## Anti-Patterns

### ❌ Dense-only retrieval — "semantic is enough"
**What it looks like:** Use only FAISS for retrieval. "BM25 is old technology. FAISS handles synonyms, which is the hard part."
**Why someone would build this:** FAISS is the modern standard. It's GPU-accelerated, handles conceptual similarity, and produces a single ranked list (no fusion needed).
**Why it's wrong:**
1. *Misses exact identifiers* — CVE numbers, tx hashes, contract addresses. The embedding of "0xdeadbeef" is noise.
2. *Embedding dilution* — a query with both meaningful words ("reentrancy") and noise ("0xdeadbeef") produces an embedding dominated by the meaningful word. The identifier is lost.
**The right approach:** Hybrid FAISS + BM25 with RRF. FAISS finds semantics; BM25 finds exact strings. Both contribute to the final ranking.

### ❌ Solidity code in the query — "more context = better retrieval"
**What it looks like:** Include `contract_code[:200]` in the RAG query. "The code gives the embedder more context about what the contract does."
**Why someone would build this:** It sounds logical — more context should help retrieval. The contract source is the most detailed description of the vulnerability.
**Why it's wrong:**
1. *Text embedder can't handle code* — `nomic-embed-text-v1.5` is trained on natural language. `mapping(address => uint256) balances` is not natural language. The embedding is garbage.
2. *Query dilution* — 200 chars of code in a 100-char query means 2/3 of the embedding is noise. The meaningful keywords ("reentrancy") are drowned by code tokens.
**The right approach:** Use mapped keywords only. `_VULN_CLASS_TO_RAG_KEYWORDS["Reentrancy"] → "reentrancy reentrant call"`. No Solidity code in the query. The embedder handles natural language; let it.

## Mistakes & Fixes

### Mistake: 26.5% of reports had zero RAG results
**What happened:** 22/83 contracts got zero RAG chunks. The RAG evidence was empty — `Evidence(source="rag")` was never emitted. The synthesizer's report had no exploit precedent for these contracts.
**Why it happened:** Three root causes:
1. 22 contracts had `topic="unknown"` (ML returned no flagged classes). The query was `"smart contract unknown vulnerability exploit attack pattern"` — the embedding is noise.
2. The query included `contract_code[:200]` — 200 chars of Solidity source that confused the text embedder.
3. ML class names ("IntegerUO") didn't match the corpus vocabulary ("integer overflow").
**How we found it:** P7 diagnosis — counted zero-result reports, inspected the queries, found garbage embeddings.
**The fix:** Three changes in `rag_research.py`:
1. Skip RAG when no classes are flagged (rag_research.py:83-85).
2. Remove `contract_code` from the query; use `_VULN_CLASS_TO_RAG_KEYWORDS` instead (rag_research.py:94, 107-110).
3. Add a fallback query on zero results (rag_research.py:130-140).
**The lesson:** When a retrieval system returns zero results, diagnose the *query* before tuning the *retriever*. The query is the input; if the input is garbage, no retriever can produce good output. "Garbage in, garbage out" applies to embeddings too — a text embedder can't produce a meaningful vector from Solidity code.

### Mistake: `assert` in production code (FIX-13)
**What happened:** The embedder used `assert len(vectors) == len(chunks)` to verify embedding count. Python's `-O` flag (common in production) strips `assert` statements. A vector/chunk count mismatch would silently corrupt the FAISS index — every retrieval would return wrong chunks.
**Why it happened:** `assert` is the natural Python idiom for "this should never happen." But `-O` disables it.
**The fix:** Replace `assert` with `if ... raise RuntimeError(...)` (embedder.py:111-116). `RuntimeError` is never disabled by `-O`.
**The lesson:** Never use `assert` for data integrity in production code. `assert` is for development-time invariant checking (test code, debug builds). `raise` is for runtime validation (production code). The difference: `assert` can be disabled; `raise` cannot.

### Mistake: No retry on embedding API calls (FIX-14)
**What happened:** A single transient LM Studio failure (model reload, GPU spike) at batch 40 of 42 killed the entire embedding run. 40 seconds of successful embedding work was discarded.
**Why it happened:** The old code made a bare `httpx.post()` call per batch with no retry.
**The fix:** `_embed_batch_with_retry` — 3 attempts per batch with exponential backoff (1s, 2s, 4s). Only raises after all retries are exhausted.
**The lesson:** External service calls must have retry logic with exponential backoff. The retry should be per-batch (not per-run), so a failure in one batch doesn't discard work done in other batches. The backoff should be exponential (1s, 2s, 4s) — linear backoff (1s, 1s, 1s) doesn't give the service time to recover.

### Mistake: FAISS↔chunks sync not validated (FIX-10)
**What happened:** If the index build pipeline crashed between writing `faiss.index` and `chunks.pkl` (e.g., power failure, OOM), the FAISS index would have N vectors but `chunks.pkl` would have M chunks (N ≠ M). FAISS position 42 would map to the wrong chunk — silent wrong retrieval results.
**Why it happened:** The index build wrote files sequentially without a sync check.
**The fix:** Validate at startup: `if self.faiss_index.ntotal != len(self.chunks): raise RuntimeError(...)` (retriever.py:104-109).
**The lesson:** When two files must be in sync (index + metadata), validate the sync at load time. A mismatch is data corruption — fail loud, fail early. Silent wrong results are worse than a crash, because the pipeline continues producing wrong evidence.

## What Would Break If You Removed This?

**Remove FAISS (semantic search):** BM25-only retrieval misses synonyms. "Flash loan attack" won't find "instantaneous borrowing exploit." Retrieval quality drops on conceptual queries.

**Remove BM25 (keyword search):** FAISS-only retrieval misses exact identifiers. "0xc310a0af" (tx hash) produces a meaningless embedding. Retrieval quality drops on identifier-based queries.

**Remove RRF:** you need score calibration between FAISS (L2 distance, lower=better) and BM25 (TF-IDF, higher=better). This requires normalization (min-max, z-score) or training data (learned weights). RRF avoids both by operating on ranks.

**Remove `_VULN_CLASS_TO_RAG_KEYWORDS`:** back to the zero-match problem. "IntegerUO" doesn't match "integer overflow" in the corpus. 26.5% of reports get zero RAG results again.

**Remove the fallback query:** if the first query returns zero results, the pipeline gives up. With the fallback, a shorter query ("Reentrancy vulnerability exploit") gets a second chance.

**Remove the FAISS↔chunks sync check:** a corrupted index produces wrong retrieval results silently. The pipeline reports wrong exploit precedent. Rule 5C violation.

**Remove retry logic on embeddings:** a single LM Studio blip kills the entire index build. No RAG index → no RAG evidence → degraded verdicts.

## At Scale

*Scale metric: corpus size (current: 752 chunks).*

| Scale | What works | What breaks | Migration path |
|-------|-----------|-------------|----------------|
| 752 chunks (current) | FAISS FlatL2 (brute-force), ~100ms/query | — | — |
| 7,520 chunks (10x) | Still FlatL2, ~1s/query | Latency noticeable | Switch to IVF (approximate) |
| 75,200 chunks (100x) | IVF works, ~100ms/query | Index rebuild takes hours | Incremental indexing |
| 752,000 chunks (1000x) | Need sharding | Single FAISS index too large | Sharded FAISS across multiple servers |

FAISS `IndexFlatL2` is brute-force — it computes L2 distance to every vector. At 752 vectors, this is <1ms. At 75,200, it's ~100ms (still fast). At 752,000, it's ~1s — and you'd switch to `IndexIVFFlat` (approximate nearest neighbor, partitions the space) for sub-100ms queries. The migration is a config change in the index builder — the retriever code doesn't change.

## Try It Yourself

> TRY IT: `cd agents && python -c "from src.rag.retriever import HybridRetriever; r = HybridRetriever(); results = r.search('reentrancy exploit attack pattern', k=3); [print(f'Score: {c.score:.4f} | {c.metadata.get(\"protocol\", \"?\")} | {c.content[:80]}...') for c in results]"` — see RRF scores and retrieved chunks.

> TRY IT: `cd agents && python -c "from src.orchestration.nodes.rag_research import _VULN_CLASS_TO_RAG_KEYWORDS; [print(f'{k}: {v}') for k,v in _VULN_CLASS_TO_RAG_KEYWORDS.items()]"` — see the 10 class-name → keyword mappings.

> TRY IT: `cd agents && pytest tests/test_rag_query.py -v` — runs all 9 RAG query tests (skip when no classes, keyword mapping, no code in query, fallback, ExternalBug call summary, all classes mapped).

## Limitations & What's Missing

- **97.9% of chunks have `vuln_type="other"`.** The DeFiHackLabs corpus has almost no labeled vulnerability types. The `vuln_type` filter is nearly useless — filtering by "reentrancy" returns only the 2.1% of chunks that happen to be labeled. This is a data quality problem, not a retrieval problem. Future: better labeling pipeline (manual or LLM-assisted).

- **No code-specific embedder.** `nomic-embed-text-v1.5` is a text model. It can't embed Solidity code meaningfully. If we wanted code-to-code search (find similar contracts), we'd need a code-specific embedder (voyage-code-2, codebert). Currently, we embed *natural language queries* and search against *natural language post-mortems* — text embedder is correct for this.

- **No incremental indexing.** Adding new exploit reports requires a full rebuild (~3 minutes for 752 chunks). At 75,000 chunks, a full rebuild would take hours. FAISS supports incremental insertion (`index.add()`), but the BM25 index needs a full rebuild (BM25Okapi doesn't support incremental updates).

- **BM25 not tuned.** The BM25Okapi uses default parameters (k1=1.5, b=0.75). These are tuned for English news text, not DeFi exploit post-mortems. Tuning k1 and b on the corpus could improve keyword retrieval quality.

- **No query expansion.** The query is a fixed string constructed from the vulnerability class. No synonym expansion ("flash loan" → "flash loan, instantaneous borrowing, atomic arbitrage"), no query rewriting, no pseudo-relevance feedback.

- **English-only corpus.** DeFiHackLabs is English-language. Non-English exploit reports are not indexed. A multilingual embedder (like multilingual-e5) would be needed for multilingual retrieval.

## Transferable Patterns

1. **Hybrid retrieval — semantic + keyword, neither alone is enough** — FAISS for meaning, BM25 for exact matches, RRF to combine.
   - *Interview story:* "SENTINEL's RAG retrieves DeFi exploit post-mortems to ground ML verdicts in historical evidence. Semantic search (FAISS) finds conceptually similar attacks — 'flash loan' matches 'price oracle manipulation.' Keyword search (BM25) finds exact identifiers — tx hashes, CVE numbers, protocol names. Neither alone is enough: FAISS misses '0xc310a0af'; BM25 misses 'instantaneous borrowing.' We combine both with Reciprocal Rank Fusion — a rank-based formula that doesn't need score calibration or training data."
   - *When this pattern is WRONG:* when your corpus is homogeneous (all the same type of document, all in the same vocabulary). If every document uses the same terminology, semantic search alone is sufficient — BM25 adds complexity without coverage. Use hybrid when your query vocabulary differs from your corpus vocabulary.

2. **Reciprocal Rank Fusion — combine rankings without score calibration** — `score = Σ 1/(rank + 60)`.
   - *Interview story:* "We needed to combine FAISS L2 distances (lower=better) and BM25 TF-IDF scores (higher=better). The score scales are incompatible — you can't add them. RRF solves this by operating on ranks, not scores: each system's top result gets 1/60, second gets 1/61, etc. A chunk ranked high in both systems gets a higher score. No calibration, no normalization, no training data — just the rank formula."
   - *When this pattern is WRONG:* when you need fine-grained score comparison (e.g., "this chunk is 2.3× more relevant than that one"). RRF gives a ranking, not a calibrated score. If downstream code needs confidence values (not just rankings), use a learned ranker or a calibrated fusion method.

3. **Query engineering — know your embedding model's domain** — text embedder ≠ code embedder.
   - *Interview story:* "Our RAG queries were returning zero results for 26.5% of audits. The root cause: the query contained Solidity code (`mapping(address => uint) balances`), and our embedding model (nomic-embed-text-v1.5) is a text model — it can't embed code meaningfully. The embedding was noise. We fixed it by removing code from queries and mapping ML class names ('IntegerUO') to descriptive keywords ('integer overflow underflow arithmetic') that the text embedder can handle. The lesson: know your embedding model's domain. If your input isn't in the model's domain, translate it before embedding."
   - *When this pattern is WRONG:* when you have a code-specific embedder (voyage-code-2, codebert) that *can* embed code. Then including code in the query is correct — the embedder handles it. The pattern is "translate input to the embedder's domain." If the embedder's domain includes code, no translation is needed.

4. **Data integrity guards at load time** — FAISS↔chunks sync check (FIX-10).
   - *Interview story:* "Our RAG index is two files: a FAISS index (vectors) and a chunks.pkl (text). FAISS position N maps to chunks[N]. If the index build crashed between writing these files, the mapping is broken — FAISS returns chunk 42, but chunks[42] is a different chunk. Every retrieval result is wrong, silently. We added a sync check at startup: if `faiss.ntotal != len(chunks)`, raise immediately. The pipeline fails fast instead of producing wrong evidence."
   - *When this pattern is WRONG:* when the two files are written atomically (e.g., in a single transaction). Then the sync check is unnecessary overhead. But file writes are rarely atomic across multiple files — the check is cheap insurance.

---

**Source files verified:**
- `agents/src/rag/retriever.py:54-56, 59-124, 126-214, 216-241, 243-299` — RRF_K, HybridRetriever, search(), _rerank(), _apply_filters()
- `agents/src/rag/embedder.py:32-33, 36-49, 51-118, 120-164, 166-198` — retry constants, Embedder class, embed_chunks(), _embed_batch_with_retry(), embed_query()
- `agents/src/rag/chunker.py:33-34, 37-61, 64-94, 101-134` — chunk size/overlap, Chunk dataclass, Chunker, chunk_document()
- `agents/src/orchestration/nodes/rag_research.py:30-41, 74-142` — keyword mapping, deterministic skip, query construction, fallback query
- `agents/src/mcp/servers/rag_server.py:70-111` — lazy loading, port 8011
- `agents/data/index/index_metadata.json` — 752 chunks, 768-dim, nomic-embed-text-v1.5
- `agents/tests/test_rag_query.py:1-179` — 9 tests (skip, keywords, no-code, fallback, ExternalBug, all-mapped)

**Verified against commit hash:** `c47898ea5`
