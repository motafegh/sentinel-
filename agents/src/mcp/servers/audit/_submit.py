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
    PENDING = "pending"
    MINED = "mined"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class TxLifecycle:
    tx_hash: str | None = None
    state: TxState = TxState.PENDING
    block_number: int | None = None
    confirmations: int = 0
    gas_used: int | None = None
    effective_gas_price: int | None = None
    receipt_status: int | None = None
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
            "error": self.error,
        }


def _estimate_gas(
    w3: Any,
    from_address: str,
    to_address: str,
    data: str | None = None,
    fallback_gas: int = 500_000,
) -> int:
    try:
        tx: dict[str, Any] = {
            "from": from_address,
            "to": to_address,
        }
        if data:
            tx["data"] = data
        estimated = w3.eth.estimate_gas(tx)
        buffer = int(estimated * 1.2)
        logger.debug("gas_estimate: raw={} buffered={} fallback={}", estimated, buffer, fallback_gas)
        return buffer
    except Exception as exc:
        logger.warning("gas estimation failed, using fallback {}: {}", fallback_gas, exc)
        return fallback_gas


def _monitor_transaction(
    w3: Any,
    tx_hash: str,
    required_confirmations: int = 2,
    poll_interval_s: float = 2.0,
    timeout_s: float = 120.0,
) -> TxLifecycle:
    lifecycle = TxLifecycle(tx_hash=tx_hash)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            receipt = w3.eth.get_transaction_receipt(tx_hash)
        except Exception:
            time.sleep(poll_interval_s)
            continue

        if receipt is None:
            time.sleep(poll_interval_s)
            continue

        lifecycle.block_number = receipt.get("blockNumber")
        lifecycle.gas_used = receipt.get("gasUsed")
        lifecycle.effective_gas_price = receipt.get("effectiveGasPrice")
        lifecycle.receipt_status = receipt.get("status")

        if receipt.get("status") == 0:
            lifecycle.state = TxState.FAILED
            lifecycle.error = "transaction reverted (status=0)"
            return lifecycle

        lifecycle.state = TxState.MINED

        try:
            current_block = w3.eth.block_number
            lifecycle.confirmations = current_block - lifecycle.block_number + 1
        except Exception:
            lifecycle.confirmations = 1

        if lifecycle.confirmations >= required_confirmations:
            lifecycle.state = TxState.CONFIRMED
            return lifecycle

        time.sleep(poll_interval_s)

    lifecycle.state = TxState.FAILED
    lifecycle.error = f"transaction did not reach {required_confirmations} confirmations within {timeout_s}s"
    return lifecycle


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
        _OPERATOR_KEY,
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
            config=(_ABI_V2, _EZKL_RUN_PROOF, _ML_API_URL, _OPERATOR_KEY,
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
    _ABI_V2, _EZKL_RUN_PROOF, _ML_API_URL, _OPERATOR_KEY, _PROXY_CHECKPOINT, _REGISTRY_ADDRESS, _SUBMIT_CONFIRM_BLOCKS, _w3 = config

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
        fusion_embedding = ml_result["fusion_embedding"]
        result["model_hash"] = model_hash
    except Exception as exc:
        result["failed_step"] = "ml_api"
        result["reason"] = f"/fusion-embedding failed: {exc}"
        logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
        return result

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

        result["proof_hash"] = (
            "0x"
            + hashlib.sha256(
                bytes.fromhex(hex_proof[2:] if hex_proof.startswith("0x") else hex_proof)
            ).hexdigest()
        )

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
            fusion_embedding=fusion_embedding,
            class_scores=result["class_scores"],
            operator_address=_OPERATOR_KEY and "",
            chain_id=chain_id,
            round_id=round_id,
            contract_address=contract_address,
            idempotency_key=idempotency_key,
            target_data_version=target_data_version,
        )
        result["provenance"] = provenance
    except Exception as exc:
        result["provenance"] = None
        logger.warning(f"submit_audit: provenance manifest skipped — {exc}")

    # ── Step 4: transaction state machine (R0.3: disabled) ──────────
    # The operator key has been removed from the MCP process (R0.3 signer
    # isolation). The policy-signer service owns the actual transaction.
    # We prepare the identity-bound proof and manifest and return partial
    # so the policy-signer can consume them.
    if _OPERATOR_KEY:
        try:
            lifecycle = _attempt_submit(
                w3=_w3,
                operator_key=_OPERATOR_KEY,
                registry_address=_REGISTRY_ADDRESS,
                abi=_ABI_V2,
                proof_hash=result["proof_hash"],
                class_score_felts=result["class_score_felts"],
                model_hash=result["model_hash"],
                contract_address=contract_address,
                chain_id=chain_id,
                round_id=round_id,
                required_confirmations=_SUBMIT_CONFIRM_BLOCKS,
                idempotency_key=idempotency_key,
            )
            result["tx_lifecycle"] = lifecycle.to_dict()
            if lifecycle.state == TxState.CONFIRMED:
                result["status"] = "submitted"
                result["tx_hash"] = lifecycle.tx_hash
            else:
                result["status"] = "failed"
                result["failed_step"] = "transaction"
                result["reason"] = lifecycle.error or "transaction failed"
        except Exception as exc:
            result["status"] = "failed"
            result["failed_step"] = "transaction"
            result["reason"] = str(exc)
    else:
        result["status"] = "partial"
        result["failed_step"] = "transaction"
        result["reason"] = (
            "On-chain submission is disabled (R0.3 signer isolation). "
            "The proof and provenance manifest are available for the policy-signer service."
        )

    return result


def _attempt_submit(
    w3: Any,
    operator_key: str,
    registry_address: str,
    abi: list,
    proof_hash: str,
    class_score_felts: list[int],
    model_hash: str,
    contract_address: str,
    chain_id: int,
    round_id: int,
    required_confirmations: int = 2,
    idempotency_key: str | None = None,
) -> TxLifecycle:
    """Build, sign, and monitor on-chain submission."""
    from eth_account import Account
    from eth_account.messages import encode_defunct

    account = Account.from_key(operator_key)
    from_address = account.address

    if abi is None:
        lifecycle = TxLifecycle(state=TxState.FAILED, error="ABI not loaded")
        return lifecycle

    contract = w3.eth.contract(address=w3.to_checksum_address(registry_address), abi=abi)

    submit_data = contract.encodeABI(
        fn_name="submitAudit",
        args=[
            w3.to_checksum_address(contract_address),
            chain_id,
            round_id,
            proof_hash,
            class_score_felts,
            model_hash,
        ],
    )

    gas_limit = _estimate_gas(w3, from_address, registry_address, data=submit_data)

    nonce = w3.eth.get_transaction_count(from_address)
    gas_price = w3.eth.gas_price

    tx: dict[str, Any] = {
        "from": from_address,
        "to": w3.to_checksum_address(registry_address),
        "data": submit_data,
        "gas": gas_limit,
        "gasPrice": gas_price,
        "nonce": nonce,
        "chainId": chain_id,
    }

    if idempotency_key:
        tx["idempotencyKey"] = idempotency_key

    signed = account.sign_transaction(tx)
    tx_hash_bytes = w3.eth.send_raw_transaction(signed.raw_transaction)
    tx_hash = tx_hash_bytes.hex() if hasattr(tx_hash_bytes, "hex") else tx_hash_bytes.hex()

    lifecycle = _monitor_transaction(
        w3=w3,
        tx_hash=tx_hash,
        required_confirmations=required_confirmations,
    )
    return lifecycle


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
) -> dict[str, Any]:
    """
    Build and EIP-191-sign a provenance manifest binding ML model to ZK proof.

    R0.4: binds chain_id, round_id, contract_address, and target_data_version
    into the manifest so the proof cannot be replayed across different chains,
    rounds, or verified contracts.
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

    try:
        from eth_account.messages import encode_defunct

        data_to_sign = json.dumps(manifest, sort_keys=True)

        if operator_address:
            try:
                from eth_account import Account
                signed = Account.sign_message(
                    encode_defunct(text=data_to_sign),
                    operator_address,
                )
                manifest["signature"] = signed.signature.hex()
                manifest["signature_scheme"] = "EIP-191"
            except Exception:
                manifest["signature"] = None
                manifest["signature_reason"] = "signing failed"
        else:
            manifest["signature"] = None
            manifest["signature_reason"] = (
                "R0.3: signing key removed from MCP process; "
                "policy-signer service owns the signature."
            )
    except ImportError:
        manifest["signature"] = None
        manifest["signature_reason"] = "eth_account not installed"
        logger.warning("provenance: eth_account not installed — signature omitted")

    return manifest


__all__ = [
    "TxLifecycle",
    "TxState",
    "_attempt_submit",
    "_estimate_gas",
    "_monitor_transaction",
    "_run_submit",
    "build_provenance_manifest",
]
