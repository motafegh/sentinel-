"""Version-aware feedback runtime boundary.

This module composes chain observation and feedback eligibility without
pretending V3 promotion policy exists. V3 events are durably journaled as
policy-pending observations and never passed into the legacy scalar RAG
ingester.

The historical V1 ingester is loaded lazily only for V1 observations that pass
the unchanged legacy compatibility policy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from .audit_observation import PROTOCOL_V1, PROTOCOL_V3
from .feedback_observer import OnChainFeedbackObserver
from .feedback_policy import evaluate_feedback_eligibility

DEFAULT_PENDING_JOURNAL = (
    Path(__file__).parent.parent.parent / "data" / "v3_feedback_policy_pending.jsonl"
)
DEFAULT_POLL_SECONDS = 30
MAX_BACKOFF_SECONDS = 300


class PendingObservationJournal:
    """Append-only durable journal for observations awaiting future policy."""

    def __init__(self, path: Path = DEFAULT_PENDING_JOURNAL) -> None:
        self.path = path
        self._seen = self._load_seen()

    def _load_seen(self) -> set[str]:
        if not self.path.exists():
            return set()
        seen: set[str] = set()
        with self.path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    identity = str(row["observation"]["event_identity"])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"invalid pending observation journal {self.path}:{line_number}: {exc}"
                    ) from exc
                if identity in seen:
                    raise ValueError(
                        f"duplicate event_identity in pending journal: {identity}"
                    )
                seen.add(identity)
        return seen

    def contains(self, event_identity: str) -> bool:
        return event_identity in self._seen

    def append(self, observation: Mapping[str, Any], decision: Mapping[str, Any]) -> bool:
        identity = str(observation.get("event_identity") or "")
        if not identity:
            raise ValueError("observation event_identity is required for durable journal")
        if identity in self._seen:
            return False
        row = {
            "schema_version": 1,
            "observation": dict(observation),
            "feedback_decision": dict(decision),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        self._seen.add(identity)
        return True


class FeedbackRuntime:
    """Route versioned observations without crossing unapproved policy boundaries."""

    def __init__(
        self,
        *,
        journal: PendingObservationJournal | None = None,
        legacy_ingester: Any | None = None,
    ) -> None:
        self.journal = journal or PendingObservationJournal()
        self._legacy_ingester = legacy_ingester

    def _get_legacy_ingester(self) -> Any:
        if self._legacy_ingester is None:
            # Heavy FAISS/RAG dependencies remain isolated from V3 observation.
            from .feedback_loop import FeedbackIngester

            self._legacy_ingester = FeedbackIngester()
        return self._legacy_ingester

    @staticmethod
    def _legacy_event(observation: Mapping[str, Any]) -> dict[str, Any]:
        """Adapt canonical V1 observation to the historical ingester shape."""
        return {
            "contract_address": observation["contract_address"],
            "score": int(observation["score_field_element"]),
            "proof_hash": observation["proof_hash"],
            "agent": observation["agent"],
            "block_number": int(observation["block_number"]),
            "tx_hash": observation["tx_hash"],
        }

    def process_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        decision = evaluate_feedback_eligibility(observation)
        protocol = str(observation.get("protocol_version") or "")
        identity = str(observation.get("event_identity") or "")

        if protocol == PROTOCOL_V3:
            if decision["eligible"]:
                raise RuntimeError("V3 policy boundary violated: V3 cannot be auto-eligible")
            written = self.journal.append(observation, decision)
            return {
                "status": "recorded_policy_pending" if written else "already_recorded",
                "protocol_version": PROTOCOL_V3,
                "event_identity": identity,
                "ingested_to_rag": False,
                "feedback_decision": decision,
            }

        if protocol == PROTOCOL_V1:
            if not decision["eligible"]:
                return {
                    "status": "ineligible",
                    "protocol_version": PROTOCOL_V1,
                    "event_identity": identity,
                    "ingested_to_rag": False,
                    "feedback_decision": decision,
                }
            ingested = bool(
                self._get_legacy_ingester().process_event(self._legacy_event(observation))
            )
            return {
                "status": "ingested" if ingested else "legacy_ingester_not_ingested",
                "protocol_version": PROTOCOL_V1,
                "event_identity": identity,
                "ingested_to_rag": ingested,
                "feedback_decision": decision,
                "reason_code": None if ingested else "legacy_ingester_returned_false",
            }

        raise ValueError(f"unsupported observation protocol: {protocol!r}")


def run_feedback_runtime(
    *,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    observer: OnChainFeedbackObserver | None = None,
    runtime: FeedbackRuntime | None = None,
) -> None:
    """Continuously observe V1/V3 events and route them through explicit policy."""
    if poll_seconds < 1:
        raise ValueError("poll_seconds must be >= 1")
    observer = observer or OnChainFeedbackObserver()
    runtime = runtime or FeedbackRuntime()
    consecutive_errors = 0

    while True:
        batch = observer.get_new_observations()
        status = batch["status"]

        for observation in batch.get("observations", []):
            result = runtime.process_observation(observation)
            if result["protocol_version"] == PROTOCOL_V3:
                logger.warning("V3 feedback observation retained without promotion: {}", result)
            else:
                logger.info("V1 feedback result: {}", result)

        if status in {"unavailable", "partial"}:
            consecutive_errors += 1
            backoff = min(poll_seconds * (2**consecutive_errors), MAX_BACKOFF_SECONDS)
            logger.warning("Feedback observer batch degraded: {} | retry in {}s", batch, backoff)
            time.sleep(backoff)
            continue

        consecutive_errors = 0
        time.sleep(poll_seconds)


__all__ = [
    "DEFAULT_PENDING_JOURNAL",
    "FeedbackRuntime",
    "PendingObservationJournal",
    "run_feedback_runtime",
]


if __name__ == "__main__":
    run_feedback_runtime()
