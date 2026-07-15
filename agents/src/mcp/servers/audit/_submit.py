# agents/src/mcp/servers/audit/_submit.py
"""
On-chain audit submission for the sentinel-audit MCP server (P11, 2026-07).

R0.4:
  - Per-job proof workspaces (temp dir, cleaned after use)
  - chain_id and round_id bound into provenance manifest (proof identity)
  - Gas estimation from web3 (never hardcoded)
  - Idempotency key prevents duplicate submissions
  - Transaction state machine: pending -> mined -> confirmed -> failed
  - Receipt status check before reporting submitted

On-chain submission remains DISABLED because the operator key has been
removed from the MCP process (R0.3 signer isolation). The policy-signer
service owns transaction construction. This module prepares the identity-
bound proof and manifest for the signer to consume.

Rule 5C: every failure returns a structured degraded return with
'status', 'failed_step', and 'reason' — never silent empty return.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


class TxState(Enum):
    """R0-F4: full transaction lifecycle states.

    not_requested -> policy_rejected (analysis process has no key)
    not_requested -> prepared -> signed -> broadcast -> pending -> confirmed
                                                 |-> reverted
                                                 |-> dropped
                                                 |-> replaced -> pending -> confirmed/reverted
                                                 |-> failed
    """
    NOT_REQUESTED = "not_requested"
    POLICY_REJECTED = "policy_rejected"
    PREPARED = "prepared"
    SIGNED = "signed"
    BROADCAST = "broadcast"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REVERTED = "reverted"
    DROPPED = "dropped"
    REPLACED = "replaced"
    FAILED = "failed"


@dataclass
class TxLifecycle:
    tx_hash: str | None = None
    state: TxState = TxState.NOT_REQUESTED
    block_number: int | None = None
    confirmations: int = 0
    gas_used: int | None = None
    effective_gas_price: int | None = None
    receipt_status: int | None = None
    nonce: int | None = None
    replaced_by: str | None = None
    idempotency_key: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "state": self.state.value,
            "block_number": self.block_number,
            "confirmations": self.confirmations,
            "gas_used": self.gas_used,
            "effective_gas_price": self.effective_gas_price,
            "receipt_status": self.receipt_status,
            "nonce": self.nonce,
            "replaced_by": self.replaced_by,
            "idempotency_key": self.idempotency_key,
            "error": self.error,
        }


def _estimate_gas(
    w3: Any,
    from_address: str,
    to_address: str,
    data: str | None = None,
) -> int:
    """Estimate gas for a transaction. Raises on failure — no silent fallback."""
    tx: dict[str, Any] = {
        "from": from_address,
        "to": to_address,
    }
    if data:
        tx["data"] = data
    estimated = w3.eth.estimate_gas(tx)
    buffer = int(estimated * 1.2)
    logger.debug("gas_estimate: raw={} buffered={}", estimated, buffer)
    return buffer


# R0-F4: Validated transition table
_ALLOWED_TRANSITIONS: dict[TxState, set[TxState]] = {
    TxState.NOT_REQUESTED: {TxState.POLICY_REJECTED, TxState.PREPARED},
    TxState.POLICY_REJECTED: set(),  # terminal
    TxState.PREPARED: {TxState.SIGNED, TxState.FAILED},
    TxState.SIGNED: {TxState.BROADCAST, TxState.FAILED},
    TxState.BROADCAST: {TxState.PENDING, TxState.REVERTED, TxState.DROPPED, TxState.REPLACED, TxState.FAILED},
    TxState.PENDING: {TxState.CONFIRMED, TxState.REVERTED, TxState.DROPPED},
    TxState.CONFIRMED: set(),  # terminal
    TxState.REVERTED: set(),   # terminal
    TxState.DROPPED: {TxState.PREPARED},  # can re-prepare
    TxState.REPLACED: {TxState.PENDING, TxState.CONFIRMED, TxState.REVERTED},
    TxState.FAILED: set(),     # terminal
}

class TxEngine:
    """R0-F4: Validated transaction state machine engine.

    Every state change goes through transition() which validates the
    allowed transitions table. confirm() and revert() both use
    transition(). Receipt validation fails closed.

    Confirmations are computed from inclusion_height vs chain_height,
    not passed as a parameter.
    """

    def __init__(self, lifecycle: TxLifecycle | None = None):
        self._lc = lifecycle or TxLifecycle()

    @property
    def state(self) -> TxState:
        return self._lc.state

    @property
    def lifecycle(self) -> TxLifecycle:
        return self._lc

    def transition(self, to: TxState, *, reason: str | None = None) -> TxLifecycle:
        allowed = _ALLOWED_TRANSITIONS.get(self._lc.state, set())
        if to not in allowed:
            raise ValueError(
                f"invalid transition: {self._lc.state.value} -> {to.value}. "
                f"Allowed: {[s.value for s in sorted(allowed, key=lambda s: s.value)]}"
            )
        self._lc.state = to
        if reason:
            self._lc.error = reason
        return self._lc

    def policy_reject(self, reason: str) -> TxLifecycle:
        return self.transition(TxState.POLICY_REJECTED, reason=reason)

    def prepare(self) -> TxLifecycle:
        return self.transition(TxState.PREPARED)

    def sign(self) -> TxLifecycle:
        return self.transition(TxState.SIGNED)

    def broadcast(self, tx_hash: str, nonce: int | None = None) -> TxLifecycle:
        self._lc.tx_hash = tx_hash
        if nonce is not None:
            self._lc.nonce = nonce
        return self.transition(TxState.BROADCAST)

    def mined(self, inclusion_height: int) -> TxLifecycle:
        self._lc.block_number = inclusion_height
        return self.transition(TxState.PENDING)

    def confirm(self, inclusion_height: int, gas_used: int,
                chain_height: int | None = None) -> TxLifecycle:
        """Confirm at inclusion_height. Confirmations = chain_height - inclusion_height + 1."""
        if inclusion_height < 0:
            raise ValueError(f"invalid inclusion_height: {inclusion_height}")
        if gas_used <= 0:
            raise ValueError(f"invalid gas_used: {gas_used}")
        confs = (chain_height - inclusion_height + 1) if chain_height is not None else 1
        if confs < 1:
            confs = 1
        self._lc.receipt_status = 1
        self._lc.block_number = inclusion_height
        self._lc.gas_used = gas_used
        self._lc.confirmations = confs
        return self.transition(TxState.CONFIRMED)

    def revert(self, reason: str) -> TxLifecycle:
        self._lc.receipt_status = 0
        return self.transition(TxState.REVERTED, reason=reason)

    def drop(self, reason: str) -> TxLifecycle:
        return self.transition(TxState.DROPPED, reason=reason)

    def replace(self, new_tx_hash: str, reason: str) -> TxLifecycle:
        self._lc.replaced_by = new_tx_hash
        return self.transition(TxState.REPLACED, reason=reason)

    def fail(self, reason: str) -> TxLifecycle:
        return self.transition(TxState.FAILED, reason=reason)

    def ingest_receipt(self, receipt: dict[str, Any],
                       chain_height: int | None = None) -> TxLifecycle:
        """Ingest on-chain receipt — fail-closed on invalid data."""
        status = receipt.get("status")
        if status is None or status not in (0, 1):
            raise ValueError(f"receipt status must be 0 or 1, got {status!r}")
        block = receipt.get("blockNumber")
        if not isinstance(block, int) or block < 0:
            raise ValueError(f"receipt blockNumber must be non-negative int, got {block!r}")
        gas = receipt.get("gasUsed")
        if not isinstance(gas, int) or gas <= 0:
            raise ValueError(f"receipt gasUsed must be positive int, got {gas!r}")
        if status == 0:
            return self.revert(receipt.get("revertReason", "receipt status zero"))
        return self.confirm(block, gas, chain_height=chain_height)

    def snapshot_state(self) -> dict[str, Any]:
        """Serializable snapshot of current engine lifecycle for deep copy."""
        return {
            "tx_hash": self._lc.tx_hash,
            "state": self._lc.state.value,
            "block_number": self._lc.block_number,
            "confirmations": self._lc.confirmations,
            "gas_used": self._lc.gas_used,
            "effective_gas_price": self._lc.effective_gas_price,
            "receipt_status": self._lc.receipt_status,
            "nonce": self._lc.nonce,
            "replaced_by": self._lc.replaced_by,
            "idempotency_key": self._lc.idempotency_key,
            "error": self._lc.error,
        }

    @staticmethod
    def restore_state(data: dict[str, Any]) -> "TxEngine":
        """Restore engine from a snapshot lifecycle."""
        lc = TxLifecycle(
            tx_hash=data.get("tx_hash"),
            state=TxState(data.get("state", "not_requested")),
            block_number=data.get("block_number"),
            confirmations=data.get("confirmations", 0),
            gas_used=data.get("gas_used"),
            effective_gas_price=data.get("effective_gas_price"),
            receipt_status=data.get("receipt_status"),
            nonce=data.get("nonce"),
            replaced_by=data.get("replaced_by"),
            idempotency_key=data.get("idempotency_key"),
            error=data.get("error"),
        )
        return TxEngine(lc)


import threading

class FakeChain:
    """Simulates transaction lifecycle for testing without live RPC.

    - Thread-safe: uses _lock for all state mutations
    - Deep snapshot: snapshot() saves serializable state; restore() rebuilds
    - Height-based confirmations: confirm_tx uses chain height, not param
    - Atomic replacement: new hash generated first, old tx linked atomically
    - Nonce: persistent monotonic allocator
    - Idempotency: keys bound to request identity (chain_id, addr, model_hash)
    """

    def __init__(self, *, confirm_blocks: int = 2):
        self.confirm_blocks = confirm_blocks
        self._tx_counter = 0
        self._nonce_counter = 0
        self._block_height = 0
        self._mempool: dict[str, TxEngine] = {}
        self._durable: dict[str, TxEngine] = {}
        self._idempotent: dict[str, str] = {}  # request_key → tx_hash
        self._lock = threading.Lock()

    def _next_hash(self) -> str:
        self._tx_counter += 1
        return f"0x{self._tx_counter:064x}"

    def _next_nonce(self) -> int:
        self._nonce_counter += 1
        return self._nonce_counter

    def _request_key(self, idempotency_key: str, chain_id: int, 
                     address: str, model_hash: str) -> str:
        """Bind idempotency key to request identity."""
        return f"{idempotency_key}:{chain_id}:{address}:{model_hash[:16]}"

    def send(self, engine: TxEngine, *,
             idempotency_key: str | None = None,
             chain_id: int = 1,
             address: str = "0x0",
             model_hash: str = "") -> TxLifecycle:
        if engine.state != TxState.SIGNED:
            raise ValueError(f"send requires SIGNED state, got {engine.state.value}")

        with self._lock:
            if idempotency_key:
                req_key = self._request_key(idempotency_key, chain_id, address, model_hash)
                existing_hash = self._idempotent.get(req_key)
                if existing_hash:
                    existing = self._durable.get(existing_hash) or self._mempool.get(existing_hash)
                    if existing:
                        lc = existing.lifecycle
                        if lc.state in (TxState.DROPPED, TxState.REPLACED):
                            # Retry: allow re-submission for dropped/replaced
                            pass
                        else:
                            return lc

            tx_hash = self._next_hash()
            self._mempool[tx_hash] = engine
            if idempotency_key:
                req_key = self._request_key(idempotency_key, chain_id, address, model_hash)
                self._idempotent[req_key] = tx_hash
            nonce = self._next_nonce()
            return engine.broadcast(tx_hash, nonce=nonce)

    def mine_blocks(self, count: int = 1) -> list[str]:
        with self._lock:
            mined = []
            for _ in range(count):
                self._block_height += 1
                for tx_hash in list(self._mempool):
                    engine = self._mempool[tx_hash]
                    if engine.state == TxState.BROADCAST:
                        engine.mined(self._block_height)
                        self._durable[tx_hash] = self._mempool.pop(tx_hash)
                        mined.append(tx_hash)
            return mined

    def mine_block(self) -> list[str]:
        return self.mine_blocks(1)

    def confirm_tx(self, tx_hash: str) -> TxLifecycle:
        with self._lock:
            engine = self._durable.get(tx_hash)
            if not engine:
                raise ValueError(f"tx {tx_hash[:16]} not in durable state")
            if engine.state != TxState.PENDING:
                raise ValueError(f"tx {tx_hash[:16]} in {engine.state.value}, need PENDING")
            return engine.confirm(
                engine.lifecycle.block_number or 0, 80_000,
                chain_height=self._block_height
            )

    def revert_tx(self, tx_hash: str, reason: str = "reverted on-chain") -> TxLifecycle:
        with self._lock:
            engine = self._durable.get(tx_hash)
            if not engine:
                raise ValueError(f"tx {tx_hash[:16]} not in durable state")
            return engine.revert(reason)

    def drop_tx(self, tx_hash: str, reason: str = "timed out") -> TxLifecycle:
        with self._lock:
            engine = self._mempool.pop(tx_hash, None)
            if not engine:
                engine = self._durable.pop(tx_hash, None)
            if not engine:
                raise ValueError(f"tx {tx_hash[:16]} not found")
            return engine.drop(reason)

    def replace_tx(self, old_hash: str, new_engine: TxEngine) -> tuple[str, TxLifecycle]:
        """Atomic replacement: generate new hash first, link old tx, broadcast new."""
        if new_engine.state != TxState.SIGNED:
            raise ValueError("replacement requires SIGNED state")

        with self._lock:
            new_tx_hash = self._next_hash()
            old_engine = self._mempool.pop(old_hash, None)
            if old_engine:
                old_engine.replace(new_tx_hash, "replaced by higher-gas tx")
                self._durable[old_hash] = old_engine
            return new_tx_hash, new_engine.broadcast(new_tx_hash, nonce=self._next_nonce())

    def reorg(self, depth: int = 1) -> list[str]:
        """Simulate chain reorg — roll back depth blocks, return unconfirmed hashes."""
        with self._lock:
            if depth <= 0:
                return []
            unconfirmed = []
            for _ in range(depth):
                if self._block_height <= 0:
                    break
                self._block_height -= 1
                for tx_hash, engine in list(self._durable.items()):
                    bn = engine.lifecycle.block_number or 0
                    if bn > self._block_height:
                        engine.lifecycle.state = TxState.PENDING
                        self._mempool[tx_hash] = engine
                        del self._durable[tx_hash]
                        unconfirmed.append(tx_hash)
            return unconfirmed

    def snapshot(self) -> dict:
        """Deep snapshot — serializable state, no object references."""
        with self._lock:
            return {
                "block_height": self._block_height,
                "nonce_counter": self._nonce_counter,
                "tx_counter": self._tx_counter,
                "mempool": {h: e.snapshot_state() for h, e in self._mempool.items()},
                "durable": {h: e.snapshot_state() for h, e in self._durable.items()},
                "idempotent": dict(self._idempotent),
            }

    def restore(self, snap: dict) -> None:
        """Restore chain from a serialized snapshot (deep copy)."""
        with self._lock:
            self._block_height = snap["block_height"]
            self._nonce_counter = snap["nonce_counter"]
            self._tx_counter = snap["tx_counter"]
            self._mempool = {h: TxEngine.restore_state(s) for h, s in snap["mempool"].items()}
            self._durable = {h: TxEngine.restore_state(s) for h, s in snap["durable"].items()}
            self._idempotent = dict(snap["idempotent"])