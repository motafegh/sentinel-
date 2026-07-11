# RAG — Retrieval-Augmented Generation

Hybrid FAISS + BM25 retriever over DeFi exploit history. The knowledge base that grounds
ML risk scores in historical precedent — enabling the audit graph to answer "has this
pattern been exploited before?" (P7 zero-match fix included.)

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
build_index/ package         → FAISS + BM25 + chunks.pkl + seen_hashes.json
    │
    ▼
HybridRetriever.search()     → list[Chunk]        (RRF-fused top-k)
```

## Files

| File/Package | Purpose |
|------|---------|
| `retriever.py` | `HybridRetriever` — FAISS semantic + BM25 keyword + Reciprocal Rank Fusion |
| `chunker.py` | `Chunker` — RecursiveCharacterTextSplitter wrapper |
| `embedder.py` | `Embedder` — LM Studio embedding with retry logic |
| `build_index/` | Full index rebuild with atomic writes and rollback (was `build_index.py`, split P2.5) |
| `fetchers/base_fetcher.py` | Abstract `BaseFetcher` + `Document` dataclass |
| `fetchers/github_fetcher.py` | `DeFiHackLabsFetcher` — .sol exploit PoC parser |

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

### P7 Zero-Match Fix

Prior to P7, queries with no semantic match (very novel attack patterns) returned empty
results, which the `rag_research` node silently treated as "no history." P7 adds minimum
fallback retrieval: if RRF returns 0 results above the score floor, a keyword-only BM25
pass is performed with relaxed thresholds. This closes the "model says vulnerable, RAG
says nothing" gap.

### Usage

```python
from src.rag.retriever import HybridRetriever

retriever = HybridRetriever()
results = retriever.search(
    query="flash loan attack on lending protocol",
    k=5,
    filters={"vuln_type": "Reentrancy"},
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

### Index Validation

On startup, `HybridRetriever.__init__()` validates that FAISS vector count matches the
chunks list length. If the index is corrupted (pipeline crash mid-write), it raises
`RuntimeError` immediately instead of returning wrong results.

## `chunker.py` — Chunker

```python
DEFAULT_CHUNK_SIZE    = 1536   # characters (~300 tokens for nomic-embed)
DEFAULT_CHUNK_OVERLAP = 128    # characters
```

`RecursiveCharacterTextSplitter` splits in order: `\n\n` → `\n` → `. ` → ` ` → `""`.

Every chunk inherits parent metadata — when you retrieve chunk 2 of Euler Finance, you
still know protocol, date, vuln_type, loss_usd.

## `embedder.py` — Embedder

Calls `client.create()` directly (not LangChain's `embed_documents()`) because LangChain
adds extra formatting LM Studio doesn't support. Each batch retried up to 3 times with
exponential backoff. Count validated after embedding with `RuntimeError` (not `assert`).

## `build_index/` — Full Index Rebuild Package

Was the monolithic `build_index.py` (661 LOC). Split into a package in P2.5
(2026-06-25) per Rule A.

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point (`python -m src.rag.build_index`) |
| `_orchestrator.py` | Top-level pipeline coordinator |
| `_pipeline.py` | Fetch → chunk → embed → FAISS + BM25 pipeline steps |
| `_io.py` | Atomic write helpers + FileLock |
| `_metadata.py` | `index_metadata.json` build/read/staleness detection |
| `_paths.py` | Centralized path constants |

### When to Use

- First-time RAG setup
- After changing chunk size, embedding model, or FAISS type
- Recovery from a corrupted index

```bash
cd agents
poetry run python -m src.rag.build_index
```

### Write Safety

1. **FileLock** — `data/index/.index.lock` (300s timeout) prevents concurrent writes
2. **Atomic writes** — each artifact written to `.tmp` sibling first, then `Path.replace()`
3. **Rollback snapshot** — existing artifacts backed up to `data/index/backups/{build_id}/`
4. **Artifact checksums** — SHA256 recorded in `index_metadata.json`
5. **Config hash** — stale index (chunk_size/overlap/model changed) triggers forced rebuild

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
