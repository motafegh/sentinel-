"""Read-only V1/V3 AuditRegistry event observer.

This module observes chain truth only. It has no RAG mutation, feedback
threshold, private key, transaction construction, signing, or broadcast
capability.

A block range is checkpointed only after *both* historical V1 and V3 event
queries succeed for that chunk. RPC failure therefore cannot manufacture a
clean/empty observation range.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from .audit_observation import (
    AUDIT_EVENT_ABI,
    format_v1_event,
    format_v3_event,
    observation_sort_key,
)

MAX_BLOCK_RANGE = 1999
DEFAULT_STATE_PATH = Path(__file__).parent.parent.parent / "data" / "feedback_observer_state.json"


class OnChainFeedbackObserver:
    """Poll AuditRegistry V1/V3 events with explicit partial/unavailable state."""

    def __init__(
        self,
        *,
        rpc_url: str | None = None,
        registry_address: str | None = None,
        state_path: Path | None = None,
        start_block: int | None = None,
        max_block_range: int = MAX_BLOCK_RANGE,
        web3: Any | None = None,
    ) -> None:
        self.rpc_url = rpc_url if rpc_url is not None else os.getenv("SEPOLIA_RPC", "")
        self.registry_address = (
            registry_address
            if registry_address is not None
            else os.getenv("AUDIT_REGISTRY", "")
        )
        if not self.registry_address:
            raise ValueError("AuditRegistry address is required")
        if max_block_range < 1:
            raise ValueError("max_block_range must be >= 1")
        self.max_block_range = max_block_range
        self.state_path = state_path or DEFAULT_STATE_PATH

        if web3 is None:
            if not self.rpc_url:
                raise ValueError("RPC URL is required")
            try:
                from web3 import Web3
            except ImportError as exc:
                raise ImportError("web3 not installed") from exc
            web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            if not web3.is_connected():
                raise ConnectionError(f"Cannot connect to RPC: {self.rpc_url}")

        self.w3 = web3
        checksum = self.w3.to_checksum_address(self.registry_address)
        self.contract = self.w3.eth.contract(address=checksum, abi=AUDIT_EVENT_ABI)
        self.last_block = self._load_last_block(start_block=start_block)

    def _current_block(self) -> int:
        value = int(self.w3.eth.block_number)
        if value < 0:
            raise ValueError("chain block number must be non-negative")
        return value

    def _load_last_block(self, *, start_block: int | None) -> int:
        if start_block is not None:
            start = int(start_block)
            if start < 0:
                raise ValueError("start_block must be non-negative")
            return start
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                block = int(payload["last_block"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid feedback observer state: {self.state_path}: {exc}") from exc
            if block < 0:
                raise ValueError("feedback observer last_block must be non-negative")
            return block
        return self._current_block()

    def _save_last_block(self, block_number: int) -> None:
        if block_number < 0:
            raise ValueError("block_number must be non-negative")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "last_block": block_number,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "registry_address": self.registry_address,
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.state_path)

    def _fetch_chunk(self, from_block: int, to_block: int) -> list[dict[str, Any]]:
        """Fetch both supported event versions for one complete block chunk."""
        v1_raw = self.contract.events.AuditSubmitted.get_logs(
            from_block=from_block,
            to_block=to_block,
        )
        v3_raw = self.contract.events.AuditSubmittedV3.get_logs(
            from_block=from_block,
            to_block=to_block,
        )
        observations = [format_v1_event(event) for event in v1_raw]
        observations.extend(format_v3_event(event) for event in v3_raw)
        observations.sort(key=observation_sort_key)
        return observations

    def get_new_observations(self) -> dict[str, Any]:
        """Fetch new observations with explicit success/partial/failure truth.

        Returns a structured batch:

        - ``idle``: chain has not advanced;
        - ``success``: every requested chunk completed;
        - ``partial``: one or more complete chunks were checkpointed, then a
          later chunk failed;
        - ``unavailable``: no new chunk could be completed.

        A failed V3 query cannot be represented as a clean range containing only
        V1 events because checkpointing happens only after both queries succeed.
        """
        try:
            current_block = self._current_block()
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason_code": "block_number_unavailable",
                "reason": str(exc),
                "observations": [],
                "last_completed_block": self.last_block,
                "attempted": True,
            }

        if current_block <= self.last_block:
            return {
                "status": "idle",
                "observations": [],
                "last_completed_block": self.last_block,
                "chain_head": current_block,
                "attempted": False,
            }

        observations: list[dict[str, Any]] = []
        from_block = self.last_block + 1
        initial_last_block = self.last_block

        while from_block <= current_block:
            to_block = min(from_block + self.max_block_range - 1, current_block)
            try:
                chunk = self._fetch_chunk(from_block, to_block)
            except Exception as exc:
                status = "partial" if self.last_block > initial_last_block else "unavailable"
                logger.error(
                    "feedback observation failed for blocks {}-{}: {}",
                    from_block,
                    to_block,
                    exc,
                )
                return {
                    "status": status,
                    "reason_code": "registry_event_query_failed",
                    "reason": str(exc),
                    "observations": observations,
                    "failed_range": {"from_block": from_block, "to_block": to_block},
                    "last_completed_block": self.last_block,
                    "chain_head": current_block,
                    "attempted": True,
                }

            observations.extend(chunk)
            # Persist only after the V1 + V3 queries and all decoders for this
            # chunk have succeeded.
            self._save_last_block(to_block)
            self.last_block = to_block
            from_block = to_block + 1

        observations.sort(key=observation_sort_key)
        return {
            "status": "success",
            "observations": observations,
            "last_completed_block": self.last_block,
            "chain_head": current_block,
            "attempted": True,
        }


__all__ = ["DEFAULT_STATE_PATH", "MAX_BLOCK_RANGE", "OnChainFeedbackObserver"]
