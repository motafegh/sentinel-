# Ingestion — Incremental RAG Pipeline + On-Chain Feedback Loop

> **Scope:** `agents/src/ingestion/` — keeps the RAG knowledge base fresh
> by fetching new DeFiHackLabs exploits and feeding high-confidence audit
> findings back from Sepolia's AuditRegistry. Source-of-truth: the code.
> Last verified: 2026-06-23.

---

## 1. One-Page Overview

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Two data-in paths into the RAG knowledge base                          │
  │                                                                         │
  │  Path A: IngestionPipeline    DeFiHackLabs → dedup → embed → RAG      │
  │  Path B: FeedbackLoop         Sepolia AuditRegistry → embed → RAG      │
  │                                                                         │
  │  Three schedulers can trigger Path A:                                   │
  │    • scheduler_cron.py         (host cron, 0 2 * * *)                   │
  │    • scheduler_dagster.py      (Dagster asset + daily 02:00 UTC)        │
  │    • GitHub Actions            (CI, optional)                           │
  │                                                                         │
  │  Path B runs as a continuous daemon (feedback_loop.py main).            │
  └─────────────────────────────────────────────────────────────────────────┘

                              ┌────────────────────┐
                              │   agents/data/     │
                              │   index/           │
                              │   (FAISS+BM25+...) │
                              │   + reports/       │
                              │   + feedback_state │
                              └────────────────────┘
```

---

## 2. The Two Paths in Detail

### 2.1 Path A — IngestionPipeline (DeFiHackLabs)

```
  ┌──────────────┐
  │ DeFiHackLabs │  GitHub: sunWeb3Sec/DeFiHackLabs
  │  .sol files  │  src/test/*.sol  +  src/past/*.sol
  │  (726 files) │
  └──────┬───────┘
         │ DeFiHackLabsFetcher.fetch()
         │  parses 3 comment formats (@Summary, @KeyInfo, free-form)
         │  → list[Document]
         ▼
  ┌──────────────────┐
  │ Deduplicator     │  SHA256(file_path)[:16] in seen_hashes.json
  │ filter_new(docs) │  returns only docs not yet indexed
  └──────┬───────────┘
         │ list[Document]  (new only)
         ▼
  ┌──────────────────┐
  │ Chunker          │  RecursiveCharacterTextSplitter
  │ chunk_documents()│  1536-char chunks, 128 overlap
  └──────┬───────────┘
         │ list[Chunk]
         ▼
  ┌──────────────────┐
  │ Embedder         │  LM Studio text-embedding-nomic-embed-text-v1.5
  │ embed_chunks()   │  768-dim, 3-attempt retry
  └──────┬───────────┘
         │ list[list[float]]
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Write under FileLock (data/index/.index.lock, 300s)     │
  │                                                          │
  │  1. Append to existing FAISS (faiss.index)               │
  │  2. Append to chunks.pkl                                 │
  │  3. FULL rebuild of BM25 (bm25.pkl) — needed because     │
  │     BM25Okapi doesn't support incremental append          │
  │  4. Mark new hashes in seen_hashes.json                  │
  │  5. Update index_metadata.json (last_run, counts)         │
  │                                                          │
  │  All writes use _atomic_write_binary() helper:            │
  │     write to .tmp sibling → tmp.replace(path)            │
  │     POSIX-atomic on Linux/WSL2, safe on Windows          │
  └──────────────────────────────────────────────────────────┘
         │
         ▼
   data/index/   (5 files updated)
```

### 2.2 Path B — FeedbackLoop (On-Chain AuditRegistry)

```
  ┌──────────────────┐
  │  Sepolia         │  AuditRegistry proxy
  │  AuditRegistry   │  0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf
  │  (UUPS proxy)    │  AuditSubmitted event
  └──────┬───────────┘
         │ OnChainListener polls every 30s
         │   • eth_getLogs (chunks of 1999 blocks, FIX-4)
         │   • exponential backoff on RPC failure (FIX-5)
         │   • state in data/feedback_state.json
         ▼
  ┌──────────────────────────────────────────┐
  │  Score threshold: scoreFieldElement ≥ 5734│  (= 0.70 * 8192 EZKL scale)
  │                                          │  Only high-confidence findings
  │  Recover vulnerability_class:            │  enter the knowledge base
  │    read data/reports/{address}.json      │
  │    (written by synthesizer, BRIDGE fix)  │
  │    fallback to "unknown" for legacy      │
  └──────┬───────────────────────────────────┘
         │ FeedbackIngester
         │   construct Document from event payload
         │   metadata: {vuln_type, date, severity, source: "OnChainAudit"}
         ▼
  ┌──────────────────┐
  │ Deduplicator     │  dedupe by contract_address + tx_hash
  │ (single instance)│  (FIX-2: was 2 instances, in-memory state diverged)
  └──────┬───────────┘
         │ list[Document]  (new only)
         ▼
  ┌──────────────────┐
  │ Chunker          │  ~1-2 chunks per finding (short)
  │ + Embedder       │  via LM Studio
  └──────┬───────────┘
         │ vectors
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Write under FileLock (same lock as Path A)              │
  │  Same atomic write pattern                               │
  │  BM25 FULLY REBUILT (FIX-1: on-chain findings were       │
  │  previously invisible to keyword search forever)         │
  └──────────────────────────────────────────────────────────┘
         │
         ▼
   data/index/   (5 files updated)
```

### 2.3 The REPORTS_DIR Bridge

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Issue #1 (2026-04-29) — BRIDGE between orchestrator and feedback loop  │
  │                                                                         │
  │  synthesizer (orchestration/nodes/synthesizer.py)                      │
  │       │                                                                 │
  │       │  write final_report JSON                                        │
  │       ▼                                                                 │
  │  data/reports/{contract_address}.json                                   │
  │       │                                                                 │
  │       │  read by contract_address                                       │
  │       ▼                                                                 │
  │  FeedbackIngester (feedback_loop.py)                                    │
  │       │                                                                 │
  │       │  recover vulnerability_class (was hardcoded "unknown")          │
  │       ▼                                                                 │
  │  document metadata gets real Track 3 class name                         │
  │                                                                         │
  │  This is why pipeline.py defines REPORTS_DIR — it's imported by both:  │
  │     • nodes/synthesizer.py: REPORTS_DIR from src.ingestion.pipeline    │
  │     • feedback_loop.py: REPORTS_DIR from .pipeline (for reads)          │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Inventory

```
  agents/src/ingestion/
  ├── pipeline.py             313 lines  IngestionPipeline
  │                                  FileLock + atomic writes
  │                                  Imports REPORTS_DIR (shared with feedback_loop)
  │                                  FIX-7  atomic write via .tmp + .replace()
  │                                  FIX-8  FileLock guards all index writes
  │                                  FIX-9  BM25Okapi import moved to module level
  │                                  FIX-23 paths anchored to __file__
  │                                  FIX-BugA .rename() → .replace() for Windows compat
  │
  ├── deduplicator.py         ~136 lines  Deduplicator
  │                                  SHA256-hash-based seen_hashes.json
  │                                  filter_new() / mark_seen() / seen()
  │                                  FIX-25 type hints tightened
  │
  ├── feedback_loop.py        470 lines  OnChainListener + FeedbackIngester
  │                                  AuditRegistry event polling
  │                                  FIX-1  BM25 rebuilt inside process_event()
  │                                  FIX-2  single Deduplicator instance
  │                                  FIX-3  removed dead self.pipeline = IngestionPipeline()
  │                                  FIX-4  block-range chunking (1999 max)
  │                                  FIX-5  exponential backoff on RPC failure
  │                                  FIX-6  hardcoded paths → imported constants
  │                                  BRIDGE reads data/reports/{address}.json
  │
  ├── scheduler_cron.py       200 lines   cron install/remove/status/run-now
  │                                   0 2 * * * → pipeline.py
  │
  └── scheduler_dagster.py    140 lines   Dagster asset + schedule
                                     rag_index asset, daily_ingestion_schedule
                                     cron 0 2 * * *  (02:00 UTC)
```

---

## 4. Deduplicator — How Dedup Works

```python
# deduplicator.py
class Deduplicator:
    """
    Tracks indexed document IDs in a JSON file.
    
    Interface:
      seen(doc_id)        → bool
      filter_new(docs)    → new docs only       (list[Document] → list[Document])
      mark_seen(doc_ids)  → None                (list[str] → None)
    
    doc_id is SHA256(file_path)[:16] from BaseFetcher.
    Stable across content edits, deterministic across runs.
    """
```

```
  First pipeline run:
    seen_hashes.json does not exist
    Deduplicator._load() returns {}
    filter_new(726 docs) → all 726 returned
    mark_seen([id1, id2, ...])
    726 documents chunked, embedded, indexed
    seen_hashes.json now has 726 entries
    
  Subsequent runs:
    seen_hashes.json has 726 entries
    DeFiHackLabs has 4 new .sol files
    filter_new(730 docs) → 4 new returned
    mark_seen([4 new ids])
    4 documents chunked, embedded, appended
    seen_hashes.json now has 730 entries
    ~0.5s embedding time (vs 12s for full re-embed)
```

**Why JSON not DB?** Simple, human-readable, git-trackable. For 100K+
documents, upgrade to SQLite (same interface, different _load/_save).

---

## 5. FileLock + Atomic Writes — The Safety Pattern

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Every write to data/index/* goes through:                               │
  │                                                                          │
  │  Step 1: Acquire FileLock(data/index/.index.lock, timeout=300s)          │
  │          → blocks if another process is mid-write                        │
  │                                                                          │
  │  Step 2: _atomic_write_binary(path, write_fn)                            │
  │          → write to .tmp sibling                                         │
  │          → tmp.replace(path)    [POSIX-atomic on Linux]                 │
  │          → on exception, tmp.unlink(missing_ok=True)                     │
  │                                                                          │
  │  Step 3: Release FileLock (context manager exit)                        │
  │                                                                          │
  │  Why this is safe:                                                      │
  │    • Concurrent build_index + pipeline → FileLock blocks, no corruption│
  │    • Crash mid-write → only .tmp is orphaned, real file is intact       │
  │    • POSIX-atomic rename → readers never see half-written file          │
  │    • Cross-platform → .replace() works on both Linux and Windows       │
  └──────────────────────────────────────────────────────────────────────────┘
```

This is the **same** pattern used by:
- `rag/build_index.py` (full rebuild)
- `ingestion/pipeline.py` (incremental)
- `ingestion/feedback_loop.py` (on-chain ingest)

All three share the FileLock — only one can write at a time.

---

## 6. OnChainListener — Event Polling Details

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  on_new_event (continuous daemon, default 30s poll)                     │
  │                                                                          │
  │  1. get_new_events()                                                    │
  │     • eth_getLogs(from_block=last_block+1, to_block=latest)            │
  │     • CHUNKED into 1999-block batches (FIX-4)                           │
  │     • Why: most RPC providers cap getLogs at 2000 blocks                │
  │     • Long offline periods → many batches, recovers correctly           │
  │                                                                          │
  │  2. for each AuditSubmitted event:                                      │
  │     • decode (contractAddress, scoreFieldElement, proofHash,           │
  │       timestamp, agent, verified)                                       │
  │     • if scoreFieldElement ≥ 5734 (0.70 * 8192):                        │
  │         call FeedbackIngester.process_event(event)                      │
  │     • else: skip (low confidence)                                       │
  │                                                                          │
  │  3. update data/feedback_state.json (last_block, last_run_at)           │
  │                                                                          │
  │  4. exponential backoff on RPC failure (FIX-5)                           │
  │     • 30s, 60s, 120s, ..., up to MAX_BACKOFF_SECONDS=300s               │
  │     • resets on first successful poll                                   │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. FeedbackIngester — Event → RAG Document

```
  OnChainListener          FeedbackIngester.process_event(event)
  ┌──────────────┐          ┌─────────────────────────────────────────────┐
  │              │          │  1. check score ≥ 5734 (else skip)          │
  │  AuditSubmit │ ───────► │  2. read data/reports/{address}.json        │
  │  ted event   │          │     to recover vulnerability_class          │
  │              │          │     (BRIDGE — was "unknown" before)         │
  └──────────────┘          │  3. construct Document:                      │
                            │       content: "SENTINEL Audit Finding       │
                            │                 Contract: {address}           │
                            │                 Risk Score: {score/8192}      │
                            │                 Confidence: HIGH              │
                            │                 Verified on-chain: YES         │
                            │                 Transaction: {tx_hash}        │
                            │                 Block: {block_number}         │
                            │                 Agent: {agent}"               │
                            │       doc_id: f"{address}:{tx_hash}"        │
                            │       metadata: {                            │
                            │         vuln_type: <recovered class>,        │
                            │         date: <event timestamp>,             │
                            │         severity: "high" (score≥0.7),         │
                            │         source: "OnChainAudit",              │
                            │         loss_usd: null,                       │
                            │         chain: "ethereum",                   │
                            │         url: <etherscan tx link>             │
                            │       }                                       │
                            │  4. dedup by doc_id (Deduplicator)          │
                            │  5. chunk (1-2 short chunks)                 │
                            │  6. embed (LM Studio)                        │
                            │  7. atomic write to FAISS + chunks + BM25    │
                            │  8. mark_seen + update metadata              │
                            └─────────────────────────────────────────────┘
```

---

## 8. Schedulers

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Three ways to schedule Path A (ingestion):                              │
  │                                                                          │
  │  ① scheduler_cron.py (~30 lines)                                        │
  │     Host-level cron entry                                               │
  │     Usage:                                                               │
  │       0 2 * * * cd /path/to/agents &&                                    │
  │           poetry run python -m src.ingestion.pipeline                    │
  │                                                                          │
  │  ② scheduler_dagster.py (~80 lines)                                     │
  │     Dagster asset: rag_index                                            │
  │     Dagster schedule: daily_ingestion_schedule                          │
  │       cron: 0 2 * * *  (02:00 UTC daily)                                │
  │     Usage:                                                               │
  │       DAGSTER_HOME=agents/.dagster \                                    │
  │         poetry run dagster dev -f src/ingestion/scheduler_dagster.py    │
  │       → http://localhost:3000 (Dagster UI)                              │
  │                                                                          │
  │  ③ GitHub Actions  (CI-based)                                            │
  │     .github/workflows/ (not in agents/ — top-level config)              │
  │                                                                          │
  │  Path B (feedback loop) is NOT scheduled — it runs as a daemon:        │
  │     SEPOLIA_RPC=<rpc> poetry run python -m src.ingestion.feedback_loop  │
  │     (runs continuously until Ctrl+C)                                    │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Configuration

| Env var | Default | Used by | Effect |
|---------|---------|---------|--------|
| `SEPOLIA_RPC` | `""` | `feedback_loop.py` | RPC endpoint for event polling |
| `SEPOLIA_RPC_URL` | `""` | `mcp/servers/audit_server.py` | Note: different env name for MCP |
| `AUDIT_REGISTRY` | `0x14E5...fAf` | `feedback_loop.py` | Contract address |
| `AUDIT_REGISTRY_ADDRESS` | `0x14E5...fAf` | `mcp/servers/audit_server.py` | Note: different env name for MCP |
| `LM_STUDIO_BASE_URL` | (gateway:4567) | `embedder.py` (via pipeline) | Embedding endpoint |
| `SCORE_THRESHOLD` (hardcoded) | `5734` (=0.70×8192) | `feedback_loop.py` | Min score to ingest |
| `POLL_INTERVAL_SECONDS` (hardcoded) | `30` | `feedback_loop.py` | Event poll frequency |
| `MAX_BLOCK_RANGE` (hardcoded) | `1999` | `feedback_loop.py` | eth_getLogs chunk size (FIX-4) |
| `MAX_BACKOFF_SECONDS` (hardcoded) | `300` | `feedback_loop.py` | RPC backoff cap (FIX-5) |
| `INDEX_LOCK_TIMEOUT` (hardcoded) | `300` | both | FileLock timeout in seconds |

---

## 10. State Files

```
  agents/data/
  ├── index/                         (written by all 3 paths)
  │   ├── faiss.index                FAISS vector index (append in pipeline, replace in build_index)
  │   ├── bm25.pkl                   BM25 keyword index (FULL REBUILD each time)
  │   ├── chunks.pkl                 Chunk list (append in pipeline)
  │   ├── seen_hashes.json           {doc_id: timestamp} — Deduplicator state
  │   ├── index_metadata.json        build_id / last_run, config_hash, SHA256s
  │   ├── .index.lock                FileLock (FileLock('data/index/.index.lock'))
  │   └── backups/                   rollback snapshots per build_id
  │
  ├── reports/                       (written by orchestrator's synthesizer)
  │   └── {contract_address}.json    final_report — read by FeedbackIngester (BRIDGE)
  │
  └── feedback_state.json            (written by OnChainListener)
                                      {last_block, last_run_at}
                                      Survives restarts — resume polling from here
```

---

## 11. Pipeline Return Value

```python
# pipeline.py — IngestionPipeline.run() returns
{
    "fetched":      726,    # total documents fetched from DeFiHackLabs
    "new_docs":     12,     # new documents (not in seen_hashes)
    "new_chunks":   15,     # chunks created from new docs
    "new_vectors":  15,     # embedding vectors generated
    "skipped":      714,    # documents already indexed
    "errors":       [],     # list of error strings (empty on success)
    "duration_sec": 12.3,   # total pipeline runtime
}
```

Same shape used by all 3 schedulers. CI/observability scripts can parse
this to track pipeline health.

---

## 12. Failure Modes

```
  ┌────────────────────────────────────────┬──────────────────────────────────────┐
  │ Failure                               │ Behaviour                             │
  ├────────────────────────────────────────┼──────────────────────────────────────┤
  │ concurrent build_index + pipeline     │ FileLock timeout (300s) → second     │
  │                                        │ call blocks until first finishes     │
  │                                        │                                      │
  │ Write fails mid-pipeline              │ _atomic_write_binary leaves .tmp      │
  │                                        │ orphaned, real file intact.           │
  │                                        │ Pipeline returns errors=[...].        │
  │                                        │ Next run sees last successful state. │
  │                                        │                                      │
  │ LM Studio down                        │ Embedder retries 3 times then        │
  │                                        │ raises. Whole pipeline fails.        │
  │                                        │                                      │
  │ DeFiHackLabs GitHub unreachable       │ DeFiHackLabsFetcher.fetch raises.    │
  │                                        │ Pipeline fails.                      │
  │                                        │                                      │
  │ BM25 rebuild fails                    │ bm25.pkl left as old (atomic write). │
  │                                        │ FAISS + chunks updated, but BM25     │
  │                                        │ missing the new chunks until next   │
  │                                        │ run. Retriever will return BM25-only │
  │                                        │ results (semantic still works).      │
  │                                        │                                      │
  │ Sepolia RPC down                      │ OnChainListener exponential backoff   │
  │                                        │ (30s → 300s max). Resumes when RPC   │
  │                                        │ recovers. No events missed (uses     │
  │                                        │ last_block from feedback_state.json).│
  │                                        │                                      │
  │ AuditRegistry event but no report     │ FeedbackIngester falls back to        │
  │ file (synthesizer didn't run)         │ vuln_type="unknown". Document is     │
  │                                        │ still indexed, but metadata has     │
  │                                        │ degraded value.                       │
  │                                        │                                      │
  │ Long offline period (>2000 blocks)    │ get_new_events() chunks into 1999    │
  │                                        │ blocks (FIX-4), processes each.     │
  │                                        │ May be slow but correct.             │
  │                                        │                                      │
  │ seen_hashes.json corrupted            │ _load() returns {} + warning.        │
  │                                        │ All docs treated as new.             │
  │                                        │ Next pipeline run: re-embed all.     │
  │                                        │ Idempotent (no harm, just slow).     │
  │                                        │                                      │
  │ Path A and Path B both trying to write│ FileLock serializes them.            │
  │ at the same time                      │ No data corruption.                   │
  └────────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 13. Quick Reference

| Concept | File:Line |
|---------|-----------|
| `IngestionPipeline` class | `agents/src/ingestion/pipeline.py:96-...` |
| `_atomic_write_binary()` helper | `pipeline.py:77-93` |
| REPORTS_DIR constant (BRIDGE) | `pipeline.py:71` |
| FileLock + timeout constants | `pipeline.py:73-74` |
| `Deduplicator` class | `agents/src/ingestion/deduplicator.py:34-...` |
| `seen()` / `filter_new()` / `mark_seen()` | `deduplicator.py:73-...` |
| `OnChainListener` class | `agents/src/ingestion/feedback_loop.py:...` |
| `FeedbackIngester` class | `feedback_loop.py:...` |
| `get_new_events()` block-chunked | `feedback_loop.py` (search for MAX_BLOCK_RANGE) |
| Exponential backoff | `feedback_loop.py` (search for MAX_BACKOFF_SECONDS) |
| `SCORE_THRESHOLD = 5734` (0.70×8192) | `feedback_loop.py:85` |
| `REPORTS_DIR` import (BRIDGE) | `feedback_loop.py:65-74` |
| Cron entry pattern | `scheduler_cron.py` |
| Dagster asset + schedule | `scheduler_dagster.py` |
| DeFiHackLabsFetcher (source) | `agents/src/rag/fetchers/github_fetcher.py` |
| Document dataclass | `agents/src/rag/fetchers/base_fetcher.py:19-39` |
| HybridRetriever (consumer) | `agents/src/rag/retriever.py` |
| Build_index (sister pipeline) | `agents/src/rag/build_index/` |
| rag_research node (consumer) | `agents/src/orchestration/nodes/rag_research.py` |

---

## 14. See Also

- `~/projects/sentinel/agents/DIAGRAM.md` — top-level module diagram
- `~/projects/sentinel/agents/src/ingestion/README.md` — text companion
- `~/projects/sentinel/agents/src/rag/DIAGRAM.md` — RAG index (consumed by this)
- `~/projects/sentinel/agents/src/orchestration/DIAGRAM.md` — orchestrator (writes reports/)
- `~/projects/sentinel/agents/src/mcp/servers/DIAGRAM.md` — audit_server (Web3 layer)
- `~/projects/sentinel/agents/src/mcp/servers/audit_server.py` — AuditRegistry contract details
