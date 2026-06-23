# RAG — Retrieval-Augmented Generation

Hybrid FAISS + BM25 retriever over DeFi exploit history. The knowledge base that grounds ML risk scores in historical precedent — enabling the audit graph to answer "has this pattern been exploited before?"

## Architecture

```
DeFiHackLabs (.sol files)
    │
    ▼
BaseFetcher.fetch()          → list[Document]
    │
    ▼
Chunker.chunk_documents()    → list[Chunk]       (1536 chars, 128 overlap)
    │
    ▼
Embedder.embed_chunks()      → list[list[float]]  (768-dim via nomic-embed-text)
    │
    ▼
build_index / pipeline       → FAISS + BM25 + chunks.pkl + seen_hashes.json
    │
    ▼
HybridRetriever.search()     → list[Chunk]        (RRF-fused top-k)
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `retriever.py` | 334 | `HybridRetriever` — FAISS semantic + BM25 keyword + Reciprocal Rank Fusion |
| `chunker.py` | 199 | `Chunker` — RecursiveCharacterTextSplitter wrapper |
| `embedder.py` | 228 | `Embedder` — LM Studio embedding with retry logic |
| `build_index.py` | 661 | Full index rebuild with atomic writes and rollback |
| `fetchers/base_fetcher.py` | 94 | Abstract `BaseFetcher` + `Document` dataclass |
| `fetchers/github_fetcher.py` | 478 | `DeFiHackLabsFetcher` — .sol exploit PoC parser |

## Knowledge Base

| Item | Value |
|------|-------|
| Source | DeFiHackLabs GitHub (`src/test/` + `past/` directories) |
| `.sol` files | 726 |
| Chunks | ~752 |
| Chunk size | 1536 characters, 128 overlap |
| Embedding model | `text-embedding-nomic-embed-text-v1.5` via LM Studio |
| Vector dimension | 768 |
| Vector index | FAISS `IndexFlatL2` |
| Keyword index | `BM25Okapi` |
| Fusion | Reciprocal Rank Fusion (RRF_K = 60) |

## `retriever.py` — HybridRetriever

### Search Algorithm

```
1. FAISS semantic search   → top-20 by L2 distance
2. BM25 keyword search     → top-20 by BM25 score
3. Reciprocal Rank Fusion  → score[chunk] = Σ 1/(60 + rank)
4. Optional metadata filter
5. Optional cross-encoder rerank
6. Return top-k
```

### Why Hybrid

| System | Good at | Bad at |
|--------|---------|--------|
| FAISS (semantic) | "borrowing exploit" finds "flash loan attack" | Exact terms: CVE numbers, tx hashes |
| BM25 (keyword) | "Euler Finance 0xc310a0af" exact match | Synonyms, paraphrases |
| **RRF fusion** | **Chunks ranking high in EITHER system get boosted** | — |

### Usage

```python
from src.rag.retriever import HybridRetriever

retriever = HybridRetriever()
results = retriever.search(
    query="flash loan attack on lending protocol",
    k=5,
    filters={"vuln_type": "Reentrancy", "loss_gte": 1_000_000},
)
for chunk in results:
    print(chunk.metadata["protocol"], chunk.score)
```

### Metadata Filters

| Key | Type | Example |
|-----|------|---------|
| `vuln_type` | `str` | `"Reentrancy"` |
| `date_gte` | `str` | `"2023-01-01"` |
| `loss_gte` | `int` | `1_000_000` |
| `source` | `str` | `"DeFiHackLabs"` |
| `has_summary` | `bool` | `True` |

### Cross-Encoder Reranking

Optional second-pass reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Reads query + chunk content jointly (bidirectional attention) — more accurate than bi-encoder recall but slower. Off by default. Requires `sentence-transformers`.

### Index Validation

On startup, `HybridRetriever.__init__()` validates that FAISS vector count matches the chunks list length. If the index is corrupted (e.g. pipeline crash mid-write), it raises `RuntimeError` immediately instead of returning wrong results.

## `chunker.py` — Chunker

### Configuration

```python
DEFAULT_CHUNK_SIZE    = 1536   # characters (~300 tokens for nomic-embed)
DEFAULT_CHUNK_OVERLAP = 128    # characters
```

### Split Priority

`RecursiveCharacterTextSplitter` splits in this order:
1. `\n\n` (paragraph break)
2. `\n` (line break)
3. `. ` (sentence end)
4. ` ` (word boundary)
5. `` (character — last resort)

### Chunk Dataclass

```python
@dataclass
class Chunk:
    content:      str           # text to embed
    doc_id:       str           # parent document ID
    chunk_id:     int           # position within parent (0-indexed)
    total_chunks: int           # total chunks from parent
    metadata:     dict          # inherited from parent + chunk_id, total_chunks
    score:        float = 0.0   # RRF retrieval score (populated by search)
```

Every chunk inherits parent metadata — when you retrieve chunk 2 of Euler Finance, you still know protocol, date, vuln_type, loss_usd.

## `embedder.py` — Embedder

### LM Studio Integration

Calls `client.create()` directly (not LangChain's `embed_documents()`) because LangChain adds extra formatting that LM Studio doesn't support.

### Retry Logic

Each batch is retried up to 3 times with exponential backoff:
- Attempt 1: immediate
- Attempt 2: wait 1s
- Attempt 3: wait 2s

A transient LM Studio blip (model reload, GPU spike) doesn't discard all work done in previous batches.

### Count Validation

After embedding, validates that vector count matches chunk count using `RuntimeError` (not `assert` — assert is silently disabled by Python's `-O` flag).

## `build_index.py` — Full Index Rebuild

### When to Use

- First-time RAG setup
- After chunking/embedding model changes
- Recovery from corrupted index artifacts

### Pipeline

```
Step 1: Fetch     → DeFiHackLabsFetcher.fetch()
Step 2: Chunk     → Chunker.chunk_documents()
Step 3: Embed     → Embedder.embed_chunks()
Step 4: FAISS     → faiss.IndexFlatL2 + add vectors
Step 5: BM25      → BM25Okapi(corpus)
Step 6: Save      → atomic writes with FileLock + rollback snapshot
```

### Write Safety

1. **FileLock** — `data/index/.index.lock` prevents concurrent writes (300s timeout)
2. **Atomic writes** — each artifact written to `.tmp` sibling first, then `Path.replace()`
3. **Rollback snapshot** — existing artifacts backed up to `data/index/backups/{build_id}/` before replacement
4. **Artifact checksums** — SHA256 of each file recorded in `index_metadata.json`
5. **Config hash** — index-affecting config (chunk_size, overlap, embedding_model, faiss_type) hashed; stale indexes trigger rebuild

### Staleness Detection

`_index_is_current()` checks:
1. All 5 required artifacts exist (`faiss.index`, `bm25.pkl`, `chunks.pkl`, `index_metadata.json`, `seen_hashes.json`)
2. `config_hash` in metadata matches current code config
3. Source file count in DeFiHackLabs matches `source_file_count` in metadata

## `fetchers/` — Data Fetchers

> **⚠ WS2 (2026-06-22):** Only `DeFiHackLabsFetcher` is active. The 5 Phase A.5
> corpus fetchers (Code4rena/Sherlock/Solodit/Immunefi/SWC) are **disabled** in
> `build_index.py:_extra_fetchers()` — their seed corpora were synthetic hand-written
> placeholders, and one caused a hallucinated verdict (Finding #2). Re-enable with
> real data per `02_RAG_BUILD_PLAN.md`.

### BaseFetcher (Abstract)

```python
class BaseFetcher(ABC):
    def fetch(self) -> list[Document]: ...
    def fetch_since(self, since: datetime) -> list[Document]: ...
    def source_name(self) -> str: ...
    def health_check(self) -> bool: ...
```

Strategy pattern: swap data sources without changing pipeline code.

### Document Dataclass

```python
@dataclass
class Document:
    content:   str    # raw text to embed
    source:    str    # where it came from
    doc_id:    str    # unique identifier for dedup
    metadata:  dict   # protocol, date, vuln_type, severity, loss_usd, chain, url
```

### DeFiHackLabsFetcher

Parses Solidity PoC files from the DeFiHackLabs repository. Three comment formats:

| Format | Marker | Files | Content |
|--------|--------|-------|---------|
| A (`@Summary`) | `// @Summary` | ~25 | Step-by-step attack narrative |
| B (`@KeyInfo`) | `// @KeyInfo` | ~473 | Loss + addresses + `@Analysis` URLs |
| C (free-form) | — | ~159 | Plain text / bare URLs (older files) |

### Extraction Pipeline per `.sol` File

```
1. _extract_date()           → "2023-03-01" (from directory name)
2. _extract_summary_block()  → attack narrative (Format A)
3. _extract_keyinfo_block()  → loss + addresses (Format B)
4. _extract_all_analysis_urls() → post-mortem links
5. _extract_first(PATTERN_TX)   → transaction URL
6. _extract_root_cause()     → root cause string
7. _extract_loss()           → $197M → 197000000
8. _infer_vuln_type()        → "reentrancy", "flash_loan", etc.
```

### Vulnerability Type Inference

`_infer_vuln_type()` pattern-matches on root_cause + summary_block + keyinfo (not raw file content — the first 1000 chars of any .sol file are always SPDX/pragma/imports). Returns one of: `reentrancy`, `flash_loan`, `oracle_manipulation`, `access_control`, `integer_overflow`, `front_running`, `logic_error`, `timestamp_dependence`, `delegatecall`, `denial_of_service`, `other`.

## Index Artifact Layout

```
agents/data/index/
  faiss.index           FAISS IndexFlatL2 (752 × 768-dim)
  bm25.pkl              BM25Okapi model (pickle)
  chunks.pkl            752 Chunk dataclass instances (pickle)
  index_metadata.json   build_id, config_hash, artifact SHA256 checksums
  seen_hashes.json      726 source file SHA256 hashes (deduplication)
  .index.lock           FileLock (concurrent write protection)
  backups/              Rollback snapshots per build_id
```
