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
    TxState.CONFIRMED: {TxState.SIGNED},   # reorg: return to pre-broadcast to re-mine
    TxState.REVERTED: set(),   # terminal
    TxState.DROPPED: {TxState.PREPARED},  # can re-prepare
    TxState.REPLACED: {TxState.PENDING, TxState.CONFIRMED, TxState.REVERTED},
    TxState.FAILED: set(),     # terminal
}

class TxEngine:
    """R0-F4: Validated transaction state machine engine.

    Every state change goes through transition() which validates the
    allowed transitions table. confirm() and revert() use transition().
    Receipt validation fails closed.
    Confirmation depth = chain_height - inclusion_height + 1.
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
                chain_height: int) -> TxLifecycle:
        """Confirm at inclusion_height with height-based confirmation depth.

        Rejects chain_height < inclusion_height (future inclusion).
        Confirmation depth = chain_height - inclusion_height + 1.
        Requires depth >= 1 (chain_height >= inclusion_height).
        """
        if chain_height < inclusion_height:
            raise ValueError(
                f"chain_height ({chain_height}) < inclusion_height ({inclusion_height})"
            )
        if inclusion_height < 0:
            raise ValueError(f"invalid inclusion_height: {inclusion_height}")
        if gas_used <= 0:
            raise ValueError(f"invalid gas_used: {gas_used}")
        depth = chain_height - inclusion_height + 1
        if depth < 1:
            raise ValueError(f"invalid confirmation depth: {depth}")
        self._lc.receipt_status = 1
        self._lc.block_number = inclusion_height
        self._lc.gas_used = gas_used
        self._lc.confirmations = depth
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

    def reorg_rollback(self) -> TxLifecycle:
        """R0-F4: reorg event — return to SIGNED for re-broadcast via validated transition."""
        return self.transition(TxState.SIGNED, reason="chain reorganisation rollback")

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
        ch = chain_height
        if ch is not None and ch < block:
            raise ValueError(f"chain_height ({ch}) < inclusion_height ({block})")
        if status == 0:
            return self.revert(receipt.get("revertReason", "receipt status zero"))
        if ch is None:
            raise ValueError("chain_height required for successful receipt")
        return self.confirm(block, gas, chain_height=ch)

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


import hashlib as _hl_chain
import threading


class FakeChain:
    """Simulates transaction lifecycle for testing without live RPC.

    - Thread-safe: _lock on all mutations
    - Confirmation threshold: confirm_tx enforces confirm_blocks depth
    - Height-based confirmations: depth = chain_height - inclusion_height + 1
    - Atomic replacement: hash first, link old, insert atomically
    - Idempotency: SHA-256 of canonical request identity
    - Deep snapshot/restore with serializable state
    - Reorg via reorg_pending() validated transition (no direct state mutation)
    """

    def __init__(self, *, confirm_blocks: int = 2):
        if confirm_blocks < 1:
            raise ValueError("confirm_blocks must be >= 1")
        self.confirm_blocks = confirm_blocks
        self._tx_counter = 0
        self._nonce_counter = 0
        self._block_height = 0
        self._mempool: dict[str, TxEngine] = {}
        self._durable: dict[str, TxEngine] = {}
        self._idempotent: dict[str, str] = {}
        self._lock = threading.Lock()

    def _next_hash(self) -> str:
        self._tx_counter += 1
        return f"0x{self._tx_counter:064x}"

    def _next_nonce(self) -> int:
        self._nonce_counter += 1
        return self._nonce_counter

    def _request_identity(self, idempotency_key: str, chain_id: int,
                          address: str, model_hash: str) -> str:
        """Cryptographic identity digest binding key to request."""
        address = address.lower()
        if not address.startswith("0x"):
            address = "0x" + address
        payload = json.dumps({
            "ik": idempotency_key,
            "chain_id": chain_id,
            "address": address,
            "model_hash": model_hash,
        }, sort_keys=True, separators=(",", ":"))
        return _hl_chain.sha256(payload.encode()).hexdigest()

    def send(self, engine: TxEngine, *,
             idempotency_key: str | None = None,
             chain_id: int = 1,
             address: str = "0x0",
             model_hash: str = "") -> TxLifecycle:
        if engine.state != TxState.SIGNED:
            raise ValueError(f"send requires SIGNED state, got {engine.state.value}")

        with self._lock:
            if idempotency_key:
                req_id = self._request_identity(idempotency_key, chain_id, address, model_hash)
                existing_hash = self._idempotent.get(req_id)
                if existing_hash:
                    existing = self._durable.get(existing_hash) or self._mempool.get(existing_hash)
                    if existing:
                        st = existing.lifecycle.state
                        if st in (TxState.DROPPED, TxState.REPLACED):
                            pass  # retry allowed
                        else:
                            return existing.lifecycle

            tx_hash = self._next_hash()
            self._mempool[tx_hash] = engine
            if idempotency_key:
                req_id = self._request_identity(idempotency_key, chain_id, address, model_hash)
                self._idempotent[req_id] = tx_hash
            return engine.broadcast(tx_hash, nonce=self._next_nonce())

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
            inc = engine.lifecycle.block_number or 0
            depth = self._block_height - inc + 1
            if depth < self.confirm_blocks:
                raise ValueError(
                    f"insufficient confirmations: {depth}/{self.confirm_blocks} "
                    f"(height={self._block_height}, inclusion={inc})"
                )
            return engine.confirm(inc, 80_000, chain_height=self._block_height)

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
        """Atomic replacement: generate hash first, link old, insert atomically."""
        if new_engine.state != TxState.SIGNED:
            raise ValueError("replacement requires SIGNED state")
        with self._lock:
            new_hash = self._next_hash()
            old_engine = self._mempool.pop(old_hash, None)
            if old_engine:
                old_engine.replace(new_hash, "replaced by higher-gas tx")
                self._durable[old_hash] = old_engine
                # Update idempotency: find key pointing to old_hash and update
                for k, v in list(self._idempotent.items()):
                    if v == old_hash:
                        self._idempotent[k] = new_hash
            new_engine.broadcast(new_hash, nonce=self._next_nonce())
            self._mempool[new_hash] = new_engine
            return new_hash, new_engine.lifecycle

    def reorg(self, depth: int = 1) -> list[str]:
        """Simulate chain reorg via validated reorg_rollback() transition.

        Confirmed transactions above the reorg depth are rolled back to SIGNED
        and re-broadcast (inserted into mempool). Mempool engines can then
        be mined again by mine_blocks().
        """
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
                        engine.reorg_rollback()  # CONFIRMED → SIGNED (validated)
                        engine.broadcast(tx_hash)  # SIGNED → BROADCAST
                        self._mempool[tx_hash] = engine
                        del self._durable[tx_hash]
                        unconfirmed.append(tx_hash)
            return unconfirmed

    def snapshot(self) -> dict:
        """Deep snapshot — serializable state, no object references."""
        with self._lock:
            return {
                "v": 1,
                "block_height": self._block_height,
                "nonce_counter": self._nonce_counter,
                "tx_counter": self._tx_counter,
                "mempool": {h: e.snapshot_state() for h, e in self._mempool.items()},
                "durable": {h: e.snapshot_state() for h, e in self._durable.items()},
                "idempotent": dict(self._idempotent),
            }

    def restore(self, snap: dict) -> None:
        """Restore chain from a serialized snapshot (deep copy)."""
        if snap.get("v") != 1:
            raise ValueError(f"unsupported snapshot version: {snap.get('v')}")
        with self._lock:
            self._block_height = snap["block_height"]
            self._nonce_counter = snap["nonce_counter"]
            self._tx_counter = snap["tx_counter"]
            self._mempool = {h: TxEngine.restore_state(s) for h, s in snap["mempool"].items()}
            self._durable = {h: TxEngine.restore_state(s) for h, s in snap["durable"].items()}
            self._idempotent = dict(snap["idempotent"])

def _run_submit(
    source_code: str,
    contract_address: str,
    model_hash: str,
    chain_id: int = 1,
    round_id: int = 0,
    idempotency_key: str | None = None,
    target_data_version: str | None = None,
) -> dict[str, Any]:
    """
    Execute the full submit-audit pipeline and return structured result.

    Args:
        source_code:        Raw Solidity source of the audited contract.
        contract_address:   0x-prefixed on-chain address of the deployed contract.
        model_hash:         SHA-256 of the teacher checkpoint (64 hex chars).
        chain_id:           Target chain ID for identity binding.
        round_id:           Submission round ID for identity binding.
        idempotency_key:    Client-supplied key to prevent duplicate submission.
        target_data_version: DATA training-set version bound into the proveance manifest.

    Returns:
        { status, tx_hash, class_scores, class_score_felts, proof_hash,
          model_hash, failed_step, reason, tx_lifecycle, idempotency_key,
          chain_id, round_id }
    """
    from ._config import (
        _ABI_V2,
        _EZKL_RUN_PROOF,
        _ML_API_URL,
        _PROXY_CHECKPOINT,
        _REGISTRY_ADDRESS,
        _SUBMIT_CONFIRM_BLOCKS,
        _w3,
    )

    # ── Per-job proof workspace ─────────────────────────────────────────
    proof_workspace = Path(tempfile.mkdtemp(prefix="sentinel_proof_"))
    try:
        return _run_submit_inner(
            source_code=source_code,
            contract_address=contract_address,
            model_hash=model_hash,
            chain_id=chain_id,
            round_id=round_id,
            idempotency_key=idempotency_key,
            target_data_version=target_data_version,
            proof_workspace=proof_workspace,
            config=(_ABI_V2, _EZKL_RUN_PROOF, _ML_API_URL,
                    _PROXY_CHECKPOINT, _REGISTRY_ADDRESS, _SUBMIT_CONFIRM_BLOCKS, _w3),
        )
    finally:
        import shutil
        if proof_workspace.exists():
            shutil.rmtree(proof_workspace, ignore_errors=True)


def _run_submit_inner(
    source_code: str,
    contract_address: str,
    model_hash: str,
    chain_id: int,
    round_id: int,
    idempotency_key: str | None,
    target_data_version: str | None,
    proof_workspace: Path,
    config: tuple,
) -> dict[str, Any]:
    _ABI_V2, _EZKL_RUN_PROOF, _ML_API_URL, _PROXY_CHECKPOINT, _REGISTRY_ADDRESS, _SUBMIT_CONFIRM_BLOCKS, _w3 = config

    result: dict[str, Any] = {
        "status": "failed",
        "tx_hash": None,
        "class_scores": None,
        "class_score_felts": None,
        "proof_hash": None,
        "model_hash": model_hash,
        "failed_step": None,
        "reason": None,
        "chain_id": chain_id,
        "round_id": round_id,
        "idempotency_key": idempotency_key,
        "target_data_version": target_data_version,
        "tx_lifecycle": None,
        "proof_scope": "none",
        "verified_audit_eligible": False,
        "finality_ineligible_reason": "proof_scope_not_identity_bound",
    }

    # ── Step 1: call /fusion-embedding ─────────────────────────────────
    try:
        import requests
        from src.contracts.execution import require_eligible_payload

        resp = requests.post(
            f"{_ML_API_URL}/fusion-embedding",
            json={"source_code": source_code},
            timeout=120,
        )
        resp.raise_for_status()
        ml_result = resp.json()
        try:
            require_eligible_payload(
                ml_result,
                purpose="proof/submission",
                input_payload={"source_code": source_code},
            )
        except (TypeError, ValueError) as exc:
            result["failed_step"] = "ml_provenance"
            result["reason"] = f"fusion embedding is ineligible: {exc}"
            logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
            return result
        live_model_hash = ml_result.get("model_hash")
        if (
            not isinstance(live_model_hash, str)
            or len(live_model_hash) != 64
            or any(char not in "0123456789abcdef" for char in live_model_hash)
            or live_model_hash != model_hash
        ):
            result["failed_step"] = "ml_provenance"
            result["reason"] = "fusion model hash does not match the requested model identity"
            logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
            return result
        model_hash = live_model_hash
        fusion_embedding_raw = ml_result["fusion_embedding"]
        result["model_hash"] = model_hash
    except Exception as exc:
        result["failed_step"] = "ml_api"
        result["reason"] = f"/fusion-embedding failed: {exc}"
        logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
        return result

    # ── Step 1b: record proof scope (R0-F3) ──
    # V2 proofs are proxy inference over supplied fusion inputs. They do not
    # bind chain/round/contract/model identity. Mark as legacy_proxy_only_unbound
    # so the policy signer, gateway, report, and finality checks can reject them.
    result["proof_scope"] = "legacy_proxy_only_unbound"

    # Pass fusion embedding through unchanged — no feature perturbation
    fusion_embedding = list(fusion_embedding_raw)

    # ── Step 2: run proxy model locally → 10 class scores ─────────────
    try:
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
        from zkml.src.distillation.proxy_model import ProxyModel

        proxy = ProxyModel()
        state = torch.load(_PROXY_CHECKPOINT, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        proxy.load_state_dict(state)
        proxy.eval()

        features = torch.tensor([fusion_embedding])
        with torch.no_grad():
            logits = proxy(features)
        scores = torch.sigmoid(logits).squeeze(0).tolist()
        felts = [round(s * 8192) for s in scores]
        result["class_scores"] = scores
        result["class_score_felts"] = felts
    except Exception as exc:
        result["failed_step"] = "proxy_inference"
        result["reason"] = f"Proxy model failed: {exc}"
        logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
        return result

    # ── Step 3: generate EZKL proof in per-job workspace ──────────────
    COMPILED = Path(__file__).resolve().parents[5] / "zkml/ezkl/model.compiled"
    SETTINGS = Path(__file__).resolve().parents[5] / "zkml/ezkl/settings.json"
    SRS = Path(__file__).resolve().parents[5] / "zkml/ezkl/srs.params"
    PROVING_KEY = Path(__file__).resolve().parents[5] / "zkml/ezkl/proving_key.pk"
    VERIFY_KEY = Path(__file__).resolve().parents[5] / "zkml/ezkl/verification_key.vk"

    for f in (COMPILED, SETTINGS, SRS, PROVING_KEY, VERIFY_KEY):
        if not f.exists():
            result["failed_step"] = "proof_generation"
            result["reason"] = f"EZKL artifact missing: {f.name}. Run setup_circuit.py first."
            logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
            return result

    try:
        import ezkl

        proof_input_path = proof_workspace / "proof_input.json"
        witness_path = proof_workspace / "witness.json"
        proof_path = proof_workspace / "proof.json"

        proof_input = {"input_data": [fusion_embedding]}
        proof_input_path.write_text(json.dumps(proof_input))

        witness = ezkl.gen_witness(
            data=str(proof_input_path),
            model=str(COMPILED),
            output=str(witness_path),
        )

        outputs = witness["outputs"][0]
        public_signals_decoded = []
        for hex_str in outputs:
            felt = int.from_bytes(bytes.fromhex(hex_str), byteorder="little")
            public_signals_decoded.append(felt)

        if len(public_signals_decoded) != 10:
            raise RuntimeError(
                f"Expected 10 output felts, got {len(public_signals_decoded)}"
            )

        ezkl.prove(
            witness=str(witness_path),
            model=str(COMPILED),
            pk_path=str(PROVING_KEY),
            proof_path=str(proof_path),
            srs_path=str(SRS),
        )

        valid = ezkl.verify(
            proof_path=str(proof_path),
            settings_path=str(SETTINGS),
            vk_path=str(VERIFY_KEY),
            srs_path=str(SRS),
        )
        if not valid:
            raise RuntimeError("Off-chain proof verification failed")

        proof_data = json.loads(proof_path.read_text())
        hex_proof = proof_data["hex_proof"]
        _INPUT_OFFSET = 128
        _NUM_CLASSES = 10

        instances = proof_data["instances"][0]
        all_public_signals = [
            int.from_bytes(bytes.fromhex(h), byteorder="little") for h in instances
        ]
        if len(all_public_signals) != _INPUT_OFFSET + _NUM_CLASSES:
            raise RuntimeError(
                f"Expected {_INPUT_OFFSET + _NUM_CLASSES} publicSignals, "
                f"got {len(all_public_signals)}"
            )

        result["class_score_felts"] = all_public_signals[_INPUT_OFFSET:]
        result["proof_hex"] = hex_proof
        result["proof_bytes"] = bytes.fromhex(hex_proof[2:] if hex_proof.startswith("0x") else hex_proof)
        result["public_signals"] = all_public_signals

        result["proof_hash"] = "0x" + hashlib.sha256(result["proof_bytes"]).hexdigest()

    except Exception as exc:
        result["failed_step"] = "proof_generation"
        result["reason"] = f"Proof generation failed: {type(exc).__name__}: {str(exc)[:300]}"
        logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
        return result

    # ── Step 3b: build identity-bound provenance manifest ──────────
    try:
        proxy_hash = hashlib.sha256(_PROXY_CHECKPOINT.read_bytes()).hexdigest()
        provenance = build_provenance_manifest(
            teacher_model_hash=result["model_hash"],
            proxy_checkpoint_hash=proxy_hash,
            fusion_embedding=fusion_embedding_raw,
            class_scores=result["class_scores"],
            operator_address="",
            chain_id=chain_id,
            round_id=round_id,
            contract_address=contract_address,
            idempotency_key=idempotency_key,
            target_data_version=target_data_version,
            proof_scope=result.get("proof_scope"),
        )
        result["provenance"] = provenance
    except Exception as exc:
        result["provenance"] = None
        logger.warning(f"submit_audit: provenance manifest skipped — {exc}")

    # ── Step 4: proof-scope eligibility check (R0-F3) ────────────────
    # V2 proofs are legacy_proxy_only_unbound. The analysis process has no
    # signing key (R0.3 signer isolation) and no raw transaction construction.
    # The proof and manifest are prepared for the policy-signer service, which
    # will reject any request with proof_scope != typed_identity_bound_v3.
    result["status"] = "policy_rejected"
    result["failed_step"] = "transaction"
    result["reason"] = (
        f"V2 proof scope '{result.get('proof_scope', 'none')}' is ineligible "
        f"for on-chain submission. The policy-signer rejects legacy/unbound "
        f"proofs. Full typed identity binding requires R3 V3 protocol work."
    )
    result["verified_audit_eligible"] = False
    result["finality_ineligible_reason"] = (
        "proof_scope_not_identity_bound"
        if result.get("proof_scope") == "legacy_proxy_only_unbound"
        else "no_proof_scope"
    )

    # R0-F3: evaluate against policy-signer boundary
    from src.security.policy_signer import evaluate_submission, PolicyDecision
    policy = evaluate_submission(
        proof_scope=result.get("proof_scope", "none"),
        contract_address=contract_address,
        chain_id=chain_id,
        round_id=round_id,
        model_hash=result.get("model_hash", ""),
    )
    result["policy_decision"] = policy.decision.value
    result["policy_reason"] = policy.reason

    return result


def build_provenance_manifest(
    teacher_model_hash: str,
    proxy_checkpoint_hash: str,
    fusion_embedding: list[float],
    class_scores: list[float],
    operator_address: str,
    chain_id: int | None = None,
    round_id: int | None = None,
    contract_address: str | None = None,
    idempotency_key: str | None = None,
    target_data_version: str | None = None,
    proof_scope: str | None = None,
) -> dict[str, Any]:
    """
    Build a provenance manifest binding ML model metadata to the proof.

    R0-F3: includes proof_scope — the trust scope of the proof generation.
    V2 proofs are 'legacy_proxy_only_unbound' (not identity-bound, not
    eligible for verified audit finality). R3 V3 protocol work will
    introduce 'typed_identity_bound_v3'.
    """
    fusion_hash = hashlib.sha256(json.dumps(fusion_embedding, sort_keys=True).encode()).hexdigest()

    manifest: dict[str, Any] = {
        "teacher_model_hash": teacher_model_hash,
        "proxy_checkpoint_hash": proxy_checkpoint_hash,
        "fusion_embedding_hash": fusion_hash,
        "class_scores": [round(s, 6) for s in class_scores],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operator_address": operator_address,
    }

    if chain_id is not None:
        manifest["chain_id"] = chain_id
    if round_id is not None:
        manifest["round_id"] = round_id
    if contract_address is not None:
        manifest["contract_address"] = contract_address
    if idempotency_key is not None:
        manifest["idempotency_key"] = idempotency_key
    if target_data_version is not None:
        manifest["target_data_version"] = target_data_version
    if proof_scope is not None:
        manifest["proof_scope"] = proof_scope

    # R0-F3: No raw signing key in the analysis/MCP process.
    # The manifest is unsigned; the policy-signer service owns
    # key management and signature production (R4).
    manifest["signature"] = None
    manifest["signature_reason"] = (
        "R0-F3: analysis process has no key. "
        "The policy-signer service (agents/src/security/policy_signer.py) "
        "validates and signs submissions in a separate security domain."
    )

    return manifest


__all__ = [
    "FakeChain",
    "TxEngine",
    "TxLifecycle",
    "TxState",
    "_ALLOWED_TRANSITIONS",
    "_estimate_gas",
    "_run_submit",
    "build_provenance_manifest",
]
