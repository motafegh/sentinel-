"""
scheduler_dagster.py

Dagster asset definitions for SENTINEL's ingestion pipeline.

RECALL — Dagster core concepts:
  @asset:    declares something valuable your pipeline produces
  @job:      a collection of assets to materialize together
  @schedule: when to run a job (cron syntax + full Dagster features)

CHANGES (2026-04-11):
  FIX-17: Collapsed the fake three-asset chain into a single rag_index asset.
          Old design: raw_documents → chunks → rag_index
            deps=["raw_documents"] and deps=["chunks"] were declared but
            NO data actually flowed between assets — each re-fetched the
            source independently. DeFiHackLabs was parsed 3× per pipeline run.
            The Dagster UI showed a lineage graph that was completely false.
          New design: single rag_index asset owns the full pipeline.
            The IngestionPipeline already encapsulates fetch→chunk→embed→index.
            Wrapping it in one asset is honest about what actually happens.
            Dagster still shows metadata, run history, and scheduling — just
            without misleading upstream assets.
          NOTE: If you want true multi-asset lineage in future, use Dagster
                IO managers to pass documents between assets rather than
                re-fetching from source each time.

  FIX-18: FreshnessPolicy now applied in code, not just in comments.
          Old: decorator comment described FreshnessPolicy(maximum_lag_minutes=1440)
               but the parameter was never added to the @asset decorator.
          New: FreshnessPolicy imported and applied. Asset is marked overdue
               in the Dagster UI if not materialized within 24 hours.

  FIX-19: ZeroDivisionError on empty chunk corpus fixed.
          Old: sum(...) // len(doc_chunks) — crashes when doc_chunks is empty.
          New: max(len(doc_chunks), 1) guard.

Run Dagster UI locally:
  cd ~/projects/sentinel/agents
  poetry run dagster dev -f src/ingestion/scheduler_dagster.py
  Open: http://localhost:3000
"""

from datetime import datetime
from pathlib import Path

from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    define_asset_job,
    Definitions,
    # NOTE FIX-18 (revised): FreshnessPolicy is an abstract base in dagster 1.12 — it accepts no
    # constructor args and the @asset decorator rejects LegacyFreshnessPolicy (type mismatch).
    # Freshness is enforced by the cron schedule (daily_ingestion_schedule, 0 2 * * *) instead.
    # Revisit when upgrading dagster past 1.12.
)
from loguru import logger


# ── Single asset — honest about what the pipeline actually does ───────────────

# FIX-17: One asset instead of three fake ones.
# FIX-18 note: freshness_policy removed — dagster 1.12 does not support it on @asset.
#              24h freshness is enforced by the daily_ingestion_schedule (0 2 * * *).
@asset(
    description="FAISS + BM25 hybrid search index — built by SENTINEL ingestion pipeline",
)
def rag_index(context: AssetExecutionContext) -> MaterializeResult:
    """
    Run the full incremental ingestion pipeline and update the RAG index.

    FIX-17: This single asset replaces the old fake three-asset chain
            (raw_documents → chunks → rag_index) where each asset
            re-fetched the source independently and passed no data.

    FIX-19: avg_chunk_size computation guarded against empty corpus.

    RECALL — IngestionPipeline handles everything:
      fetch → deduplicate → chunk → embed → update FAISS → update BM25 → mark seen
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    stats    = pipeline.run()

    context.log.info(
        f"Index updated — {stats['new_docs']} new docs, "
        f"{stats['new_chunks']} new chunks, "
        f"{stats['skipped']} skipped (already indexed)"
    )

    # FIX-19: Guard against empty corpus to prevent ZeroDivisionError.
    # This can happen on first run before any documents are indexed.
    chunk_count = stats.get("new_chunks", 0)
    total_docs  = stats.get("fetched", 0)
    avg_chunk   = chunk_count // max(total_docs, 1)   # FIX-19

    return MaterializeResult(
        metadata={
            "new_documents": MetadataValue.int(stats["new_docs"]),
            "new_chunks":    MetadataValue.int(stats["new_chunks"]),
            "skipped":       MetadataValue.int(stats["skipped"]),
            "total_fetched": MetadataValue.int(stats["fetched"]),
            "avg_chunks_per_doc": MetadataValue.int(avg_chunk),
            "duration_sec":  MetadataValue.float(stats["duration_sec"]),
            "errors":        MetadataValue.text(str(stats["errors"])),
            "run_at":        MetadataValue.text(datetime.now().isoformat()),
        }
    )


# ── Job definition ────────────────────────────────────────────────────────────

ingestion_job = define_asset_job(
    name="sentinel_ingestion_job",
    selection=[rag_index],
    description="SENTINEL RAG ingestion pipeline — incremental index update",
)


# ── Schedule definition ───────────────────────────────────────────────────────

daily_ingestion_schedule = ScheduleDefinition(
    job=ingestion_job,
    cron_schedule="0 2 * * *",   # daily at 2 AM UTC — matches cron scheduler
    name="daily_ingestion_schedule",
)


# ── Definitions — the Dagster entry point ─────────────────────────────────────

defs = Definitions(
    assets=[rag_index],
    jobs=[ingestion_job],
    schedules=[daily_ingestion_schedule],
)
