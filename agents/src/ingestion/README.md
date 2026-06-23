# Ingestion

Incremental RAG pipeline and on-chain feedback loop. Keeps the knowledge base fresh by fetching new DeFiHackLabs exploits and feeding high-confidence audit findings back from the Sepolia AuditRegistry.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Ingestion Pipeline        │
DeFiHackLabs ──────▶│ fetch → dedup → chunk → embed   │──▶ index/
                    │              ▲         │         │
                    │              │    atomic write    │
                    │              └─────────┘         │
                    └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │          Feedback Loop            │
AuditRegistry ─────▶│ poll events → read report.json  │──▶ index/
(Sepolia)           │              │                   │
                    │         chunk → embed             │
                    │         atomic write              │
                    └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │        Schedulers                │
                    │  Dagster (daily 02:00 UTC)       │
                    │  Cron (0 2 * * *)                │
                    │  GitHub Actions                   │
                    └─────────────────────────────────┘
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline.py` | 313 | `IngestionPipeline` — incremental update with deduplication |
| `deduplicator.py` | 136 | SHA256 hash-based document deduplication |
| `feedback_loop.py` | 470 | `OnChainListener` + `FeedbackIngester` — AuditRegistry polling |
| `scheduler_cron.py` | 200 | Cron crontab manager (install/remove/status/run-now) |
| `scheduler_dagster.py` | 140 | Dagster asset + daily schedule (02:00 UTC) |

## `pipeline.py` — IngestionPipeline

### Pipeline vs build_index.py

| | `build_index.py` | `pipeline.py` |
|---|---|---|
| Use case | First-time setup, forced rebuild | Scheduled incremental updates |
| Embedding | Re-embeds everything | Embeds only new documents |
| Deduplication | None (full rebuild) | SHA256 `seen_hashes.json` |
| Triggered by | Manual CLI | Cron / Dagster / GitHub Actions |

### Pipeline Steps

```
Step 1: Fetch       → DeFiHackLabsFetcher.fetch()
Step 2: Deduplicate  → Deduplicator.filter_new()
Step 3: Chunk        → Chunker.chunk_documents()
Step 4: Embed        → Embedder.embed_chunks()
Step 5: Write FAISS  → append to existing index (atomic)
Step 6: Write Chunks → append to chunks.pkl (atomic)
Step 7: Rebuild BM25 → full rebuild from all chunks (atomic)
Step 8: Mark Seen    → update seen_hashes.json
Step 9: Update Metadata → index_metadata.json
```

### Write Safety

All index writes are protected by:
1. **FileLock** — `data/index/.index.lock` (300s timeout)
2. **Atomic writes** — `.tmp` sibling + `Path.replace()` (POSIX-atomic, cross-platform safe)
3. **`_atomic_write_binary()`** — shared helper used by pipeline, feedback_loop, and build_index

### Usage

```bash
cd agents
poetry run python -m src.ingestion.pipeline
```

### Return Value

```python
{
    "fetched":      726,    # total documents fetched
    "new_docs":     12,     # new documents (not in seen_hashes)
    "new_chunks":   15,     # chunks created from new docs
    "new_vectors":  15,     # embedding vectors generated
    "skipped":      714,    # documents already indexed
    "errors":       [],     # list of error strings
    "duration_sec": 12.3,   # total pipeline runtime
}
```

### REPORTS_DIR Bridge

`pipeline.py` defines `REPORTS_DIR = agents/data/reports/`. The orchestrator's synthesizer node writes `final_report` JSON here before returning. The feedback loop reads it to recover `vulnerability_class` for on-chain RAG indexing.

## `deduplicator.py` — Deduplicator

### How It Works

1. On init: loads `seen_hashes.json` (dict of `{doc_id: timestamp}`)
2. `filter_new(documents)`: returns only documents whose `doc_id` is not in `seen_hashes`
3. `mark_seen(doc_ids)`: adds new entries to `seen_hashes` and persists to disk

### SHA256 Deduplication

`doc_id` is computed by each fetcher. For DeFiHackLabs, it's `sha256(str(sol_file))[:16]` — deterministic per file path.同一文件不会被索引两次，即使 pipeline 多次运行。

## `feedback_loop.py` — On-Chain Feedback Loop

### Purpose

Closes the full SENTINEL loop:
1. ML model detects vulnerabilities → audit submitted on-chain
2. Feedback loop picks up `AuditSubmitted` events
3. High-confidence findings are ingested back into the RAG knowledge base
4. Future audits can reference these verified findings

### Components

#### OnChainListener

Polls `AuditRegistry` on Sepolia for `AuditSubmitted` events.

- **Block-range chunking**: Most RPC providers cap `eth_getLogs` at 2000 blocks. `get_new_events()` chunks requests into 1999-block batches.
- **Exponential backoff**: On RPC failure, backs off exponentially (30s, 60s, 120s, ...) up to 300s max.
- **State persistence**: `data/feedback_state.json` stores `last_block` — survives restarts.

#### FeedbackIngester

Converts audit events into RAG documents and ingests them.

- **Score threshold**: Only findings with `score >= 5734` (field element for 0.70 human-readable) enter the knowledge base.
- **BRIDGE (Issue #1)**: Reads `data/reports/{contract_address}.json` (written by synthesizer) to recover `vulnerability_class`. On-chain findings are now indexed with the actual class name instead of the hardcoded `"unknown"` placeholder.
- **BM25 rebuild**: After each ingestion, BM25 is fully rebuilt so on-chain findings become keyword-searchable immediately.

### Document Schema

```
SENTINEL Audit Finding
Contract: {address}
Risk Score: {score/8192} ({score} field element)
Confidence: HIGH/MEDIUM
Verified on-chain: YES (ZK proof verified by Halo2Verifier)
Transaction: {tx_hash}
Block: {block_number}
Agent: {agent}
```

### Usage

```bash
cd agents
SEPOLIA_RPC=<your-rpc> poetry run python -m src.ingestion.feedback_loop
# Runs continuously until Ctrl+C
```

### Configuration (`.env`)

```bash
SEPOLIA_RPC=<rpc-url>
AUDIT_REGISTRY=0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf
```

## Schedulers

### Dagster

```bash
cd agents
DAGSTER_HOME=agents/.dagster \
poetry run dagster dev -f src/ingestion/scheduler_dagster.py
# → http://localhost:3000 (Dagster UI)
```

Asset: `rag_index` — full pipeline.
Schedule: `daily_ingestion_schedule` — cron `0 2 * * *` (02:00 UTC daily).

### Cron

```bash
0 2 * * * cd /path/to/agents && poetry run python -m src.ingestion.pipeline
```

## Data Layout

```
agents/data/
  index/
    faiss.index           FAISS vector index
    bm25.pkl              BM25 keyword index
    chunks.pkl            Chunk objects
    index_metadata.json   Build metadata + checksums
    seen_hashes.json      Document deduplication hashes
    .index.lock           Concurrent write lock
    backups/              Rollback snapshots
  reports/                Final audit report JSON per contract_address
  feedback_state.json     Last processed Sepolia block number
```
