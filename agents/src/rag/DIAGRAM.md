# RAG — Retrieval-Augmented Generation

> **Scope:** `agents/src/rag/` — hybrid FAISS + BM25 retriever over DeFi
> exploit history. Source-of-truth: the code. Last verified: 2026-06-21.

---

## 1. One-Page Overview

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  RAG knowledge base — grounds ML risk scores in historical precedent    │
  │                                                                         │
  │  "Has this vulnerability pattern been exploited before?"                 │
  │  Answer: search 752 chunks from 726 DeFiHackLabs PoCs + 5 audit-firm   │
  │  corpora (Phase A) using hybrid semantic + keyword retrieval.           │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─── 6 FETCHERS (rag/fetchers/) ──────────────────────────────────────────┐
  │  base_fetcher.py        abstract BaseFetcher + Document dataclass       │
  │                                                                         │
  │  github_fetcher.py      DeFiHackLabs .sol PoCs (3 comment formats)    │
  │  json_corpus_fetcher.py shared base for curated JSON corpora (A.5)    │
  │    └─ code4rena_fetcher.py    C4 contest findings                      │
  │    └─ sherlock_fetcher.py     Sherlock contest findings                │
  │    └─ solodit_fetcher.py      Solodit aggregated findings              │
  │    └─ immunefi_fetcher.py     Immunefi bounty disclosures              │
  │    └─ swc_registry_fetcher.py SWC weakness-classification registry     │
  └─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  list[Document]
  ┌─── CHUNKER (rag/chunker.py) ───────────────────────────────────────────┐
  │  RecursiveCharacterTextSplitter                                        │
  │  Chunk size 1536 chars, overlap 128 chars                              │
  │  Split priority: \n\n → \n → ". " → " " → ""                          │
  └─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  list[Chunk]
  ┌─── EMBEDDER (rag/embedder.py) ─────────────────────────────────────────┐
  │  LM Studio: text-embedding-nomic-embed-text-v1.5 (768-dim)            │
  │  Retry: 3 attempts with exponential backoff (1s, 2s)                  │
  └─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  list[list[float]]
  ┌─── BUILD / UPDATE INDEX (rag/build_index.py + ingestion/pipeline.py) ──┐
  │  FAISS IndexFlatL2 + BM25Okapi + chunks.pkl + seen_hashes.json        │
  │  Atomic writes with FileLock + rollback snapshot                      │
  └─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─── INDEX (data/index/) ────────────────────────────────────────────────┐
  │  faiss.index             768-dim vectors, IndexFlatL2                  │
  │  bm25.pkl                BM25Okapi (pickled)                           │
  │  chunks.pkl              list[Chunk] (pickled)                         │
  │  index_metadata.json     build_id, config_hash, artifact SHA256        │
  │  seen_hashes.json        source file SHA256 (dedup)                    │
  │  .index.lock             FileLock (concurrent write protection)        │
  │  backups/                rollback snapshots per build_id               │
  └─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─── RETRIEVER (rag/retriever.py) ────────────────────────────────────────┐
  │  HybridRetriever()                                                     │
  │    search(query, k, filters) → list[Chunk] (RRF-fused top-k)          │
  └─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  Consumed by:
   • rag_server :8011 (MCP tool)            • rag_research node (graph)
```

---

## 2. Build Pipeline — End-to-End

```
  ┌───────────────┐
  │  6 fetchers   │  github_fetcher + 5 json_corpus fetchers
  │  (rag/        │  (Phase A — A.5, 2026-06-21)
  │   fetchers/)  │
  └───────┬───────┘
          │ fetch()  → list[Document]
          │              (content, source, doc_id, metadata)
          ▼
  ┌──────────────────┐
  │  Chunker         │  RecursiveCharacterTextSplitter
  │  (chunker.py)    │  1536-char chunks, 128-char overlap
  │                  │  Priority: ¶ → ↵ → sentence → word → char
  └───────┬──────────┘
          │ chunk_documents(docs) → list[Chunk]
          │     (content, doc_id, chunk_id, total_chunks, metadata, score)
          ▼
  ┌──────────────────┐
  │  Embedder        │  httpx → LM Studio → /v1/embeddings
  │  (embedder.py)   │  model: text-embedding-nomic-embed-text-v1.5
  │                  │  dim: 768
  │                  │  retry: 3 attempts, exp backoff (1s, 2s)
  │                  │  count validation: vectors == chunks (or raise)
  └───────┬──────────┘
          │ embed_chunks(chunks) → list[list[float]]
          ▼
  ┌──────────────────────────────────────────────────┐
  │  build_index.py  (full rebuild)                  │
  │  OR                                               │
  │  ingestion/pipeline.py (incremental)             │
  │  ─────────────────────────────────────────       │
  │  FileLock (data/index/.index.lock, 300s timeout) │
  │  Atomic write: each artifact → .tmp → replace()  │
  │  Rollback snapshot: backups/{build_id}/ before   │
  │  SHA256 checksums recorded in metadata            │
  │  config_hash: chunk_size, overlap, model, faiss   │
  └───────┬──────────────────────────────────────────┘
          │ writes 5 files atomically
          ▼
  agents/data/index/
    faiss.index              FAISS IndexFlatL2
    bm25.pkl                 BM25Okapi
    chunks.pkl               list[Chunk]
    index_metadata.json      build_id, config_hash, SHA256s
    seen_hashes.json         SHA256 of source files (dedup)
    .index.lock              FileLock
    backups/{build_id}/      rollback snapshots
```

---

## 3. HybridRetriever — The Retrieval Engine

### 3.1 Search Algorithm

```python
def search(query, k=5, filters=None, faiss_candidates=20, rerank=False):
    # ① FAISS semantic — embed query, top-20 by L2 distance
    query_vector = self.embedder.embed_query(query)
    _, indices = self.faiss_index.search(np.array([query_vector]), 20)
    faiss_ranked = [(idx, rank) for rank, idx in enumerate(indices[0]) if idx != -1]

    # ② BM25 keyword — tokenize query, top-20 by BM25 score
    query_tokens = query.lower().split()
    bm25_scores  = self.bm25.get_scores(query_tokens)
    bm25_ranked  = [(idx, rank) for rank, idx in enumerate(
                       sorted(range(len(bm25_scores)),
                              key=lambda i: bm25_scores[i], reverse=True)[:20])]

    # ③ Reciprocal Rank Fusion (RRF)
    #    score(chunk) = Σ 1/(rank + RRF_K)   with RRF_K=60
    rrf_scores = {}
    for idx, rank in faiss_ranked:
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + 60)
    for idx, rank in bm25_ranked:
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + 60)
    sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

    # ④ Optional metadata filter
    results = [Chunk(...) for i in sorted_indices]
    if filters: results = self._apply_filters(results, filters, query)

    # ⑤ Optional cross-encoder rerank (off by default)
    if rerank: results = self._rerank(query, results)

    # ⑥ Slice to top-k
    return results[:k]
```

### 3.2 Why Hybrid?

| System | Good at | Bad at |
|--------|---------|--------|
| **FAISS (semantic)** | "borrowing exploit" finds "flash loan attack" (synonyms) | Exact terms: CVE numbers, tx hashes, addresses |
| **BM25 (keyword)** | "Euler Finance 0xc310a0af" exact match | Synonyms, paraphrases, conceptual similarity |
| **RRF fusion** | Chunks ranking high in EITHER system get boosted | — |

### 3.3 RRF Constant

```python
RRF_K = 60    # empirically tuned for this corpus
              # higher = smaller rank differences (smoother fusion)
              # lower  = rank-1 dominates more
```

### 3.4 Metadata Filters

| Filter | Type | Example | Behaviour |
|--------|------|---------|-----------|
| `vuln_type` | str | `"Reentrancy"` | exact match on `metadata.vuln_type` |
| `date_gte` | str (ISO) | `"2023-01-01"` | exclude docs with `metadata.date < filter` |
| `loss_gte` | int (USD) | `1_000_000` | minimum `metadata.loss_usd` |
| `source` | str | `"DeFiHackLabs"` | exact match on `metadata.source` |
| `has_summary` | bool | `True` | only chunks from `@Summary` documents |

**FIX-12 (2026-04-11):** When filters return 0 results, the retriever now
logs a warning instead of failing silently — callers can distinguish
"no matches" from "filters too aggressive".

---

## 4. Init Validation — Index Integrity

```python
# retriever.py:71-124
class HybridRetriever:
    def __init__(self):
        # 1. Load all 5 index artifacts
        self.faiss_index = faiss.read_index(str(FAISS_PATH))   # 768-dim vectors
        with open(BM25_PATH, "rb") as f: self.bm25 = pickle.load(f)
        with open(CHUNKS_PATH, "rb") as f: self.chunks = pickle.load(f)
        with open(METADATA_PATH) as f: self.metadata = json.load(f)
        self.embedder = Embedder()

        # 2. FIX-10: Validate FAISS ↔ chunks sync
        if self.faiss_index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index corruption: FAISS has {faiss_index.ntotal} vectors "
                f"but chunks has {len(self.chunks)}. Re-run build_index."
            )

        # 3. FIX-5: tolerate both 'built_at' (build_index) and 'last_run' (pipeline)
        built = self.metadata.get("built_at") or self.metadata.get("last_run", "unknown")
```

**Why this matters:** a crash mid-write could leave FAISS with N vectors
but chunks with M<N — silent corruption where index position N maps to the
wrong chunk. We fail loud at startup, not during the first search.

---

## 5. Build vs Ingest — Two Pipelines, One Index

```
  ┌────────────────────────┐         ┌────────────────────────┐
  │  rag/build_index.py    │         │  ingestion/pipeline.py │
  │  (full rebuild)        │         │  (incremental update)  │
  │                        │         │                        │
  │  • all 6 fetchers      │         │  • DeFiHackLabs only   │
  │  • re-chunk everything │         │  • only new docs since │
  │  • re-embed everything │         │    last seen_hashes    │
  │  • new build_id        │         │  • no new build_id     │
  │  • used after schema   │         │  • used by cron /      │
  │    changes / recovery  │         │    Dagster schedulers  │
  └───────────┬────────────┘         └────────┬───────────────┘
              │                              │
              │   same write safety:         │
              │   FileLock + atomic +         │
              │   rollback + SHA256          │
              │                              │
              └──────────────┬───────────────┘
                             ▼
                  agents/data/index/  (5 files)
```

### When to use which?

| Use | When |
|-----|------|
| `build_index.py` | First-time setup, schema changes, recovery |
| `ingestion/pipeline.py` | Scheduled incremental updates (cron / Dagster) |

---

## 6. Chunker — Recursive Split

```
  DEFAULT_CHUNK_SIZE    = 1536   chars (~300 tokens for nomic-embed)
  DEFAULT_CHUNK_OVERLAP =  128   chars

  Split priority (RecursiveCharacterTextSplitter):
    1. \n\n    (paragraph break)    ← preferred
    2. \n      (line break)
    3. ". "    (sentence end)
    4. " "     (word boundary)
    5. ""      (character — last resort)


  @dataclass
  class Chunk:
      content:      str         # text to embed
      doc_id:       str         # parent document ID
      chunk_id:     int         # position within parent (0-indexed)
      total_chunks: int         # total chunks from parent
      metadata:     dict        # inherited from parent + chunk_id, total_chunks
      score:        float = 0.0 # RRF retrieval score (populated by search)
```

**Why overlap 128 chars?** So a sentence/paragraph that straddles a chunk
boundary is still fully present in at least one chunk.

---

## 7. Embedder — LM Studio with Retry

```
  Embedder.embed_chunks(chunks) → list[list[float]]
       │
       │  Calls LM Studio /v1/embeddings directly
       │  (not LangChain's embed_documents — LangChain adds formatting
       │   LM Studio doesn't support)
       │
       │  Batches chunks → httpx POST → list[float]
       │
       │  Retry on transient failure:
       │    attempt 1: immediate
       │    attempt 2: wait 1s, retry
       │    attempt 3: wait 2s, retry
       │
       │  After embedding: validate count
       │    if len(vectors) != len(chunks): raise RuntimeError
       │    (not assert — assert is silently disabled by `python -O`)
       │
       ▼
  list[list[float]]  (768-dim, one vector per chunk)
```

---

## 8. The 6 Fetchers

### 8.1 Common Pattern — Strategy + Document

```python
# fetchers/base_fetcher.py

@dataclass
class Document:
    content:  str
    source:   str
    doc_id:   str
    metadata: dict = field(default_factory=dict)
    # metadata keys: protocol, date, vuln_type, severity, loss_usd, chain, url

class BaseFetcher(ABC):
    @abstractmethod
    def fetch(self) -> list[Document]:                # for full rebuild
        pass

    @abstractmethod
    def fetch_since(self, since: datetime) -> list[Document]:  # for incremental
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:                     # for metadata
        pass

    def health_check(self) -> bool:                    # default True
        return True
```

### 8.2 Fetcher Inventory

| Fetcher | Source | Format | Notes |
|---------|--------|--------|-------|
| `github_fetcher` (DeFiHackLabs) | DeFiHackLabs GitHub | .sol PoC files | 3 comment formats: @Summary / @KeyInfo / free-form. 726 files. |
| `code4rena_fetcher` | Code4rena contests | JSON (curated) | Phase A (A.5, 2026-06-21) |
| `sherlock_fetcher` | Sherlock contests | JSON (curated) | Phase A (A.5) |
| `solodit_fetcher` | Solodit aggregated | JSON (curated) | Phase A (A.5) |
| `immunefi_fetcher` | Immunefi bounties | JSON (curated) | Phase A (A.5) |
| `swc_registry_fetcher` | SWC weakness registry | JSON (curated) | Phase A (A.5) |

The 5 Phase A fetchers share `json_corpus_fetcher.py` as their base — they
read curated JSON files from `data/knowledge/` and return `[]` gracefully
if a file is missing (degraded to DeFiHackLabs-only).

### 8.3 DeFiHackLabsFetcher — The Heavy Lifter

```
  Input: DeFiHackLabs/src/test/*.sol and DeFiHackLabs/src/past/*.sol
  Output: list[Document]  (one per .sol file)

  Per-file extraction (github_fetcher.py):
    1. _extract_date()                → "2023-03-01" (from dir name)
    2. _extract_summary_block()       → attack narrative (Format A)
    3. _extract_keyinfo_block()       → loss + addresses (Format B)
    4. _extract_all_analysis_urls()   → post-mortem links
    5. _extract_first(PATTERN_TX)     → transaction URL
    6. _extract_root_cause()          → root cause string
    7. _extract_loss()                → "$197M" → 197000000
    8. _infer_vuln_type()             → one of 11 categories

  Three comment formats:
    Format A  @Summary    ~25 files   step-by-step attack narrative
    Format B  @KeyInfo   ~473 files   loss + addresses + @Analysis URLs
    Format C  free-form  ~159 files   plain text / bare URLs (older)

  _infer_vuln_type() pattern-matches on root_cause + summary + keyinfo
  (NOT on raw file content — first 1000 chars are always SPDX/pragma/imports).
  Returns one of: reentrancy, flash_loan, oracle_manipulation, access_control,
  integer_overflow, front_running, logic_error, timestamp_dependence,
  delegatecall, denial_of_service, other.
```

---

## 9. Index Write Safety

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Every write goes through these safety nets:                             │
  │                                                                          │
  │  1. FileLock (data/index/.index.lock)                                    │
  │     Prevents concurrent writes (300s timeout)                            │
  │     Shared by build_index.py AND ingestion/pipeline.py                   │
  │                                                                          │
  │  2. Atomic writes                                                        │
  │     Each artifact: write to .tmp sibling → Path.replace()               │
  │     POSIX rename is atomic — readers never see a half-written file      │
  │                                                                          │
  │  3. Rollback snapshot                                                    │
  │     Before replacing, copy current artifacts to backups/{build_id}/     │
  │     If the new build fails mid-write, manual recovery from backup       │
  │                                                                          │
  │  4. SHA256 checksums                                                     │
  │     Each artifact's hash recorded in index_metadata.json                │
  │     Enables corruption detection on next startup                         │
  │                                                                          │
  │  5. config_hash                                                          │
  │     Hash of: chunk_size, overlap, embedding_model, faiss_type           │
  │     If config changes, _index_is_current() flags the index as stale     │
  │     Forces a rebuild                                                     │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Build vs Ingest — Metadata Schema

`index_metadata.json` is the canonical record of how the index was built.

```json
{
  "build_id":   "uuid-v4",
  "built_at":   "2026-06-15T12:34:56Z",
  "schema_version": 1,
  "config_hash": "sha256...",
  "source_file_count": 726,
  "chunk_size": 1536,
  "chunk_overlap": 128,
  "embedding_model": "text-embedding-nomic-embed-text-v1.5",
  "faiss_type": "IndexFlatL2",
  "artifacts": {
    "faiss.index":   { "sha256": "...", "size_bytes": 1234567 },
    "bm25.pkl":      { "sha256": "...", "size_bytes": 2345678 },
    "chunks.pkl":    { "sha256": "...", "size_bytes": 3456789 },
    "seen_hashes.json": { "sha256": "...", "size_bytes": 45678 }
  }
}
```

(ingestion/pipeline.py writes "last_run" instead of "built_at" — FIX-5
in retriever.py:118 makes the reader tolerate both keys.)

---

## 11. Cross-Encoder Reranking (Optional)

```python
# retriever.py:216-241
def _rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c.content) for c in chunks]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]
    except ImportError:
        logger.warning("rerank=True but sentence-transformers not installed — fallback to RRF")
        return chunks
    except Exception as exc:
        logger.warning(f"Cross-encoder reranking failed: {exc} — fallback to RRF")
        return chunks
```

**Why optional:** bi-encoder recall (FAISS) is fast but lossy. Cross-encoder
reads query+chunk jointly (bidirectional attention) → more accurate but
slower. Off by default to keep search < 100ms; enable for high-precision
queries (e.g. "Euler Finance 0xc310a0af specific transaction").

---

## 12. Consumers — Who Calls the Retriever

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. rag_research node (orchestration/nodes.py) — primary consumer   │
  │     Deep path only. Called via MCP :8011.                           │
  │     Query: "{ML top class} exploit pattern in {contract snippet}"   │
  │     Filters: {vuln_type: top_class, loss_gte: 100_000}              │
  │     k: AUDIT_RAG_K (default 5)                                       │
  │     Output → state.rag_results → consumed by cross_validator,       │
  │                                     synthesizer, explainer          │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────┐
  │  2. rag_server :8011 (mcp/servers/rag_server.py)                    │
  │     Exposes HybridRetriever.search() as MCP tool `search`.          │
  │     Lazy-loaded HybridRetriever in _on_startup() (Bug 10 fix).      │
  │     Same search() method, same results, different transport.        │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────┐
  │  3. retriever.py __main__ — dev/test                                │
  │     Three hard-coded test queries for smoke testing the retriever.  │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 13. Configuration

| Env var | Default | Used by | Effect |
|---------|---------|---------|--------|
| `LM_STUDIO_BASE_URL` | `http://<wsl-gateway>:4567/v1` | `embedder.py` | Where to fetch embeddings |
| `LM_STUDIO_TIMEOUT` | `60` | `embedder.py` | Per-request timeout |
| `MCP_RAG_PORT` | `8011` | `rag_server.py` | Port the MCP server listens on |
| `RAG_DEFAULT_K` | `5` | `rag_server.py` | Default k if client doesn't specify |
| `RAG_MAX_K` (hardcoded) | `20` | `rag_server.py` | Server-side cap on k |
| `AUDIT_RAG_K` | `5` | `rag_research` node | k for in-graph RAG queries |
| `MODULE1_MOCK` | `false` | `embedder.py` indirectly | If true, no LM Studio needed |

---

## 14. File Map

```
  agents/src/rag/
  │
  ├── retriever.py            334 lines  HybridRetriever (FAISS + BM25 + RRF)
  │                                  FIX-5  .get() with fallback (built_at/last_run)
  │                                  FIX-10 FAISS↔chunks sync validation
  │                                  FIX-12 empty-filter warning
  │                                  FIX-23 __file__-anchored paths
  │
  ├── chunker.py              199 lines  RecursiveCharacterTextSplitter wrapper
  │                                       DEFAULT_CHUNK_SIZE=1536, OVERLAP=128
  │
  ├── embedder.py             228 lines  LM Studio embeddings + 3-attempt retry
  │                                       count validation (RuntimeError, not assert)
  │
  ├── build_index.py          604 lines  Full rebuild (FileLock + atomic + rollback)
  │                                       Calls all 6 fetchers (Phase A)
  │
  └── fetchers/
      ├── base_fetcher.py      95 lines  Abstract BaseFetcher + Document dataclass
      ├── github_fetcher.py   478 lines  DeFiHackLabsFetcher (3 comment formats)
      ├── json_corpus_fetcher.py        Shared base for 5 curated JSON corpora (A.5)
      ├── code4rena_fetcher.py
      ├── sherlock_fetcher.py
      ├── solodit_fetcher.py
      ├── immunefi_fetcher.py
      └── swc_registry_fetcher.py
```

---

## 15. Failure Modes

```
  ┌────────────────────────────────────────┬──────────────────────────────────────┐
  │ Failure                               │ Behaviour                             │
  ├────────────────────────────────────────┼──────────────────────────────────────┤
  │ Index files missing                   │ HybridRetriever.__init__ raises       │
  │                                        │ FileNotFoundError                     │
  │                                        │ "Run poetry run python -m             │
  │                                        │  src.rag.build_index"                 │
  │                                        │                                      │
  │ FAISS count ≠ chunks count            │ RuntimeError "Index corruption"       │
  │ (e.g. crash mid-write)                │ "Re-run build_index.py"               │
  │                                        │                                      │
  │ concurrent build_index + pipeline     │ FileLock timeout (300s) → second     │
  │                                        │ call blocks until first finishes     │
  │                                        │                                      │
  │ Write fails mid-build                 │ Rollback snapshot in backups/{id}/   │
  │                                        │ still holds the old working index.   │
  │                                        │ Manual recovery or retry.            │
  │                                        │                                      │
  │ LM Studio down                        │ Embedder retries 3 times, then       │
  │                                        │ raises. Whole build fails.           │
  │                                        │                                      │
  │ Embedder returns wrong vector count   │ RuntimeError "vector count mismatch" │
  │                                        │ (not assert — survives `python -O`)  │
  │                                        │                                      │
  │ Filters too aggressive                │ Returns []. Logs warning (FIX-12).   │
  │                                        │ Caller can distinguish from           │
  │                                        │ "no exploits exist".                 │
  │                                        │                                      │
  │ cross-encoder rerank enabled but      │ Silent fallback to RRF order.        │
  │ sentence-transformers not installed   │ Logs warning.                        │
  │                                        │                                      │
  │ Curated JSON corpus file missing      │ _extra_fetchers() returns []          │
  │ (Phase A fetcher)                      │ gracefully. Build continues with     │
  │                                        │ DeFiHackLabs only.                   │
  └────────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 16. Quick Reference

| Concept | File:Line |
|---------|-----------|
| `HybridRetriever.__init__` | `agents/src/rag/retriever.py:71-124` |
| FAISS↔chunks validation (FIX-10) | `retriever.py:104-109` |
| Built/last_run tolerance (FIX-5) | `retriever.py:118` |
| `search()` (RRF algorithm) | `retriever.py:126-214` |
| RRF_K constant | `retriever.py:56` |
| `_rerank()` (cross-encoder) | `retriever.py:216-241` |
| `_apply_filters()` | `retriever.py:243-299` |
| Empty-filter warning (FIX-12) | `retriever.py:291-296` |
| `Chunk` dataclass | `agents/src/rag/chunker.py` |
| DEFAULT_CHUNK_SIZE / OVERLAP | `chunker.py:14-15` |
| Embedder retry logic | `agents/src/rag/embedder.py` |
| Vector count validation (RuntimeError) | `embedder.py` (search for RuntimeError) |
| `build_index.py` orchestration | `agents/src/rag/build_index.py` |
| FileLock | `build_index.py` (search for FileLock) |
| Atomic write pattern | `build_index.py` (search for `.tmp` + `replace()`) |
| 6 fetchers imported | `build_index.py:60-68` |
| Phase A fetchers helper | `build_index.py:87-100` |
| `BaseFetcher` abstract | `agents/src/rag/fetchers/base_fetcher.py` |
| `Document` dataclass | `fetchers/base_fetcher.py:19-39` |
| `DeFiHackLabsFetcher` | `fetchers/github_fetcher.py` |
| `_infer_vuln_type()` | `fetchers/github_fetcher.py` |
| RAG server (consumer MCP) | `agents/src/mcp/servers/rag_server.py` |
| Lazy retriever init (Bug 10 fix) | `mcp/servers/rag_server.py:97-111` |
| `rag_research` node (consumer) | `agents/src/orchestration/nodes.py:421-...` |
| Index directory | `agents/data/index/` |

---

## 17. See Also

- `~/projects/sentinel/agents/DIAGRAM.md` — top-level module diagram
- `~/projects/sentinel/agents/src/rag/README.md` — text companion
- `~/projects/sentinel/agents/src/mcp/servers/DIAGRAM.md` — RAG server
- `~/projects/sentinel/agents/src/orchestration/DIAGRAM.md` — orchestration (consumers)
- `~/projects/sentinel/agents/src/ingestion/DIAGRAM.md` — incremental pipeline
- `docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md` (Phase A.5)
