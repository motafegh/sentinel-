"""
feedback_loop.py

Listens for SENTINEL audit events on-chain and feeds high-confidence
findings back into the RAG knowledge base.

RECALL — Why this closes the full SENTINEL loop:
  Module 1: ML model detects vulnerabilities
  Module 2: ZK proof verifies the detection honestly
  Module 5: AuditRegistry stores the result on-chain
  Module 4: This module feeds those results back as RAG knowledge

CHANGES (2026-04-11):
  FIX-1:  BM25 now rebuilt inside process_event() — on-chain findings
          were previously invisible to keyword search forever.
  FIX-2:  Single Deduplicator instance in FeedbackIngester. Old code
          created two instances pointing at the same file; in-memory
          state diverged after the first mark_seen() call, allowing
          double-ingestion on the next run.
  FIX-3:  Removed self.pipeline = IngestionPipeline() — it was never
          used; process_event() did everything manually. Dead object
          consumed memory and hid the real code path.
  FIX-4:  Block-range chunking in get_new_events(). Unbounded get_logs()
          calls fail silently on most RPC providers (2 000-block cap).
          Long offline periods now recover correctly instead of losing events.
  FIX-5:  Exponential backoff on RPC failures. Old code hammered the
          provider every 30s regardless of error state.
  FIX-6:  Hardcoded paths replaced with module-level constants imported
          from pipeline.py. A path change propagates everywhere automatically.

CHANGES (2026-04-29):
  FIX-BugA: tmp_faiss.rename(FAISS_PATH) → tmp_faiss.replace(FAISS_PATH).
            .rename() raises FileExistsError on Windows/WSL2 when the
            destination already exists. .replace() is POSIX-atomic on Linux
            AND silently overwrites on Windows — correct on both platforms.
            Same fix applied to pipeline.py (cross-project issue).

Run from agents/ directory:
  poetry run python -m src.ingestion.feedback_loop
  (runs continuously until Ctrl+C)
"""

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv

# FIX-6: Import canonical path constants from pipeline.py.
from .pipeline import (
    FAISS_PATH,
    BM25_PATH,
    CHUNKS_PATH,
    SEEN_HASHES_PATH,
    INDEX_LOCK_PATH,
    INDEX_LOCK_TIMEOUT,
    _atomic_write_binary,
)

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────

SEPOLIA_RPC            = os.getenv("SEPOLIA_RPC", "")
AUDIT_REGISTRY_ADDRESS = os.getenv("AUDIT_REGISTRY", "0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf")

# Score threshold: only high-confidence findings enter the knowledge base.
# 0.7 = field element ~5734  (0.7 * 8192 EZKL scale factor)
SCORE_THRESHOLD = 5734

POLL_INTERVAL_SECONDS = 30

# FIX-4: Most RPC providers cap eth_getLogs at 2000 blocks per request.
MAX_BLOCK_RANGE = 1999

# FIX-5: Exponential backoff caps
MAX_BACKOFF_SECONDS = 300   # 5 minutes maximum between retries

# AuditRegistry ABI — only the parts we need
AUDIT_REGISTRY_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "name": "contractAddress", "type": "address"},
            {"indexed": False, "name": "proofHash",       "type": "bytes32"},
            {"indexed": True,  "name": "agent",           "type": "address"},
            {"indexed": False, "name": "score",           "type": "uint256"},
        ],
        "name": "AuditSubmitted",
        "type": "event",
    },
    {
        "inputs": [{"name": "contractAddress", "type": "address"}],
        "name": "getLatestAudit",
        "outputs": [
            {"name": "score",      "type": "uint256"},
            {"name": "proofHash",  "type": "bytes32"},
            {"name": "timestamp",  "type": "uint256"},
            {"name": "agent",      "type": "address"},
            {"name": "verified",   "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


class OnChainListener:
    """
    Polls AuditRegistry for new AuditSubmitted events.

    FIX-4: Block range is chunked into MAX_BLOCK_RANGE (1999) batches.
    FIX-5: get_new_events() returns None on RPC error (vs [] for no events).
    """

    def __init__(self):
        try:
            from web3 import Web3
            self.w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC))
            if not self.w3.is_connected():
                raise ConnectionError(f"Cannot connect to RPC: {SEPOLIA_RPC}")
            logger.info(f"Connected to Sepolia — block {self.w3.eth.block_number}")
        except ImportError:
            raise ImportError("web3 not installed. Run: poetry add web3")

        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(AUDIT_REGISTRY_ADDRESS),
            abi=AUDIT_REGISTRY_ABI,
        )

        self.state_path = Path(__file__).parent.parent.parent / "data" / "feedback_state.json"
        self.last_block = self._load_last_block()
        logger.info(f"OnChainListener ready — watching from block {self.last_block}")

    def _load_last_block(self) -> int:
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f).get("last_block", 0)
        return self.w3.eth.block_number

    def _save_last_block(self, block_number: int) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump({
                "last_block": block_number,
                "updated_at": datetime.now().isoformat(),
            }, f)

    def get_new_events(self) -> list[dict] | None:
        """
        Fetch new AuditSubmitted events since last check.

        FIX-4: Chunked block-range queries.
        FIX-5: Returns None on RPC error, [] on success-with-no-events.
        """
        try:
            current_block = self.w3.eth.block_number
        except Exception as e:
            logger.error(f"Could not fetch current block number: {e}")
            return None

        if current_block <= self.last_block:
            return []

        all_events: list = []
        from_block = self.last_block + 1

        while from_block <= current_block:
            to_block = min(from_block + MAX_BLOCK_RANGE - 1, current_block)
            try:
                batch = self.contract.events.AuditSubmitted.get_logs(
                    from_block=from_block,
                    to_block=to_block,
                )
                all_events.extend(batch)
                from_block = to_block + 1

            except Exception as e:
                logger.error(
                    f"Error fetching blocks {from_block}–{to_block}: {e}. "
                    f"Will retry from block {from_block} next poll."
                )
                if all_events:
                    logger.info(f"Returning {len(all_events)} partial events before failure")
                    return [self._format_event(e) for e in all_events]
                return None

        self._save_last_block(current_block)
        self.last_block = current_block

        if all_events:
            logger.info(f"Found {len(all_events)} new AuditSubmitted events")

        return [self._format_event(e) for e in all_events]

    @staticmethod
    def _format_event(event) -> dict:
        return {
            "contract_address": event["args"]["contractAddress"],
            "score":            event["args"]["score"],
            "proof_hash":       event["args"]["proofHash"].hex(),
            "agent":            event["args"]["agent"],
            "block_number":     event["blockNumber"],
            "tx_hash":          event["transactionHash"].hex(),
        }


class FeedbackIngester:
    """
    Converts on-chain audit findings into RAG documents and ingests them.

    FIX-2 + FIX-3:
      Removed dead self.pipeline object. One Deduplicator only.
    """

    def __init__(self):
        from .deduplicator import Deduplicator
        from ..rag.chunker import Chunker
        from ..rag.embedder import Embedder

        self.deduplicator = Deduplicator(SEEN_HASHES_PATH)
        self.chunker      = Chunker()
        self.embedder     = Embedder()

    def process_event(self, event: dict) -> bool:
        """
        Convert an audit event to a RAG document and ingest it.

        FIX-1: BM25 rebuilt after each ingestion.
        FIX-6: Uses path constants from pipeline.py.

        Returns:
            True if ingested, False if skipped (low score or already seen)
        """
        from ..rag.fetchers.base_fetcher import Document

        contract_address = event["contract_address"]
        score            = event["score"]
        tx_hash          = event["tx_hash"]

        if score < SCORE_THRESHOLD:
            logger.debug(
                f"Skipping low-confidence finding: "
                f"contract={contract_address[:10]}... score={score}"
            )
            return False

        human_score = round(score / 8192, 3)
        doc_id      = f"onchain_{tx_hash[:16]}"

        if self.deduplicator.seen(doc_id):
            logger.debug(f"Already ingested: {doc_id}")
            return False

        content = f"""SENTINEL Audit Finding
Contract: {contract_address}
Risk Score: {human_score} ({score} field element)
Confidence: {"HIGH" if human_score > 0.85 else "MEDIUM"}
Verified on-chain: YES (ZK proof verified by Halo2Verifier)
Transaction: {tx_hash}
Block: {event['block_number']}
Agent: {event['agent']}

This contract was audited by SENTINEL's ML model and found to have
a risk score of {human_score:.1%}. The finding was verified on-chain
via a zero-knowledge proof — the model computation is cryptographically
guaranteed to be honest. This pattern should be considered when auditing
similar contracts."""

        # TODO-B (design gap — tracked in GitHub Issue #<TBD>):
        # vuln_type is hardcoded "unknown" because AuditRegistry.sol's
        # AuditSubmitted event and getLatestAudit() carry only:
        #   score, proofHash, agent, timestamp, verified
        # The vulnerability_class produced by the ML orchestrator (nodes.py)
        # never reaches this loop. On-chain documents are indexed without a
        # class label, making all vuln_type filters in retriever.py return
        # 0 results for on-chain docs.
        #
        # Three bridge options are documented in GitHub Issue — a design
        # decision is required before implementing any of them:
        #   Option 1: Contract upgrade — add vulnerabilityClass to AuditResult struct
        #   Option 2: Shared store   — orchestrator writes final_report to
        #             data/reports/{contract_address}.json; this loop reads it
        #   Option 3: Re-query inference server — call sentinel-inference with
        #             the contract address (adds ML dependency to this module)
        #
        # RECOMMENDATION: Option 2 (shared store). Lowest coupling, no contract
        # changes, no re-inference. Orchestrator already has final_report;
        # writing it to disk before on-chain submission costs one json.dump().
        doc = Document(
            content=content,
            source="SENTINEL_ONCHAIN",
            doc_id=doc_id,
            metadata={
                "contract_address": contract_address,
                "score":            score,
                "human_score":      human_score,
                "tx_hash":          tx_hash,
                "block_number":     event["block_number"],
                "agent":            event["agent"],
                "date":             datetime.now().strftime("%Y-%m-%d"),
                "source":           "SENTINEL_ONCHAIN",
                "vuln_type":        "unknown",   # TODO-B: see comment above
                "verified_onchain": True,
            }
        )

        chunks  = self.chunker.chunk_document(doc)
        vectors = self.embedder.embed_chunks(chunks)
        vecs_np = np.array(vectors, dtype=np.float32)

        from filelock import FileLock, Timeout
        try:
            with FileLock(str(INDEX_LOCK_PATH), timeout=INDEX_LOCK_TIMEOUT):
                self._write_to_index(chunks, vecs_np)
        except Timeout:
            logger.error(
                f"Could not acquire index lock — skipping {doc_id}. "
                f"Pipeline may be running. Will retry on next event."
            )
            return False

        self.deduplicator.mark_seen([doc_id])

        logger.info(
            f"Ingested on-chain finding: "
            f"contract={contract_address[:10]}... "
            f"score={human_score:.1%} chunks={len(chunks)}"
        )
        return True

    def _write_to_index(self, chunks, vecs_np: np.ndarray) -> None:
        """
        Append new chunks to FAISS, chunks.pkl, and BM25.

        FIX-1:    BM25 rebuilt here.
        FIX-7:    Uses _atomic_write_binary from pipeline.py.
        FIX-BugA: tmp_faiss.replace() instead of .rename() — cross-platform safe.
                  .rename() raises FileExistsError on Windows/WSL2 when the
                  destination file already exists. .replace() is atomic on
                  POSIX and silently overwrites on Windows.
        """
        # FAISS — atomic write
        index = faiss.read_index(str(FAISS_PATH))
        index.add(vecs_np)
        tmp_faiss = FAISS_PATH.with_suffix(".tmp")
        faiss.write_index(index, str(tmp_faiss))
        tmp_faiss.replace(FAISS_PATH)   # FIX-BugA: was .rename(), fails on Windows

        # Chunks — atomic write via helper
        with open(CHUNKS_PATH, "rb") as f:
            all_chunks = pickle.load(f)
        all_chunks.extend(chunks)

        def _write_chunks(tmp: Path) -> None:
            with open(tmp, "wb") as f:
                pickle.dump(all_chunks, f)

        _atomic_write_binary(CHUNKS_PATH, _write_chunks)

        # FIX-1: BM25 rebuild — on-chain findings now keyword-searchable
        corpus = [chunk.content.lower().split() for chunk in all_chunks]
        bm25   = BM25Okapi(corpus)

        def _write_bm25(tmp: Path) -> None:
            with open(tmp, "wb") as f:
                pickle.dump(bm25, f)

        _atomic_write_binary(BM25_PATH, _write_bm25)


def run_feedback_loop() -> None:
    """
    Run the continuous feedback loop until Ctrl+C.

    FIX-5: Exponential backoff on RPC failures.
    """
    if not SEPOLIA_RPC:
        logger.error("SEPOLIA_RPC not set in .env — feedback loop cannot start")
        return

    logger.info("=" * 60)
    logger.info("SENTINEL FEEDBACK LOOP — STARTING")
    logger.info(f"Polling interval:  {POLL_INTERVAL_SECONDS}s")
    logger.info(f"Score threshold:   {SCORE_THRESHOLD} ({SCORE_THRESHOLD / 8192:.1%})")
    logger.info(f"Max block range:   {MAX_BLOCK_RANGE} blocks per RPC call")
    logger.info(f"Contract:          {AUDIT_REGISTRY_ADDRESS}")
    logger.info("=" * 60)

    try:
        listener = OnChainListener()
        ingester = FeedbackIngester()
    except Exception as e:
        logger.error(f"Failed to initialise feedback loop: {e}")
        return

    ingested_total     = 0
    consecutive_errors = 0

    try:
        while True:
            events = listener.get_new_events()

            if events is None:
                consecutive_errors += 1
                backoff = min(
                    POLL_INTERVAL_SECONDS * (2 ** consecutive_errors),
                    MAX_BACKOFF_SECONDS,
                )
                logger.warning(
                    f"RPC failure #{consecutive_errors} — "
                    f"backing off {backoff}s (cap: {MAX_BACKOFF_SECONDS}s)"
                )
                time.sleep(backoff)
                continue

            consecutive_errors = 0

            for event in events:
                if ingester.process_event(event):
                    ingested_total += 1
                    logger.info(f"Total ingested this session: {ingested_total}")

            if not events:
                logger.debug(f"No new events — sleeping {POLL_INTERVAL_SECONDS}s")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info(f"Feedback loop stopped — {ingested_total} findings ingested this session")


if __name__ == "__main__":
    run_feedback_loop()
