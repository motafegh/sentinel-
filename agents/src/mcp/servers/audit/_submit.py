# agents/src/mcp/servers/audit/_submit.py
"""
On-chain audit submission for the sentinel-audit MCP server (P11, 2026-07).

Implements the submit_audit MCP tool: fetches the 128-dim fusion embedding
from the ML inference API, runs the proxy model to get 10 class scores,
generates a ZK proof via EZKL, and submits the audit result on-chain via
AuditRegistry.submitAuditV2().

Rule 5C: every subprocess/web3 failure returns a structured degraded return
with 'status', 'failed_step', and 'reason' — never silent empty return.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


def _run_submit(
    source_code: str,
    contract_address: str,
    model_hash: str,
) -> dict[str, Any]:
    """
    Execute the full submit-audit pipeline and return structured result.

    Args:
        source_code:      Raw Solidity source of the audited contract.
        contract_address: 0x-prefixed on-chain address of the deployed contract.
        model_hash:        SHA-256 of the teacher checkpoint (64 hex chars).

    Returns:
        {
            "status":            "submitted" | "partial" | "failed",
            "tx_hash":           str | None,
            "class_scores":      [float × 10] | None,
            "class_score_felts": [int × 10] | None,
            "proof_hash":        str | None,
            "model_hash":        str,
            "failed_step":       str | None,
            "reason":            str | None,
        }

        On partial: proof generated but chain submit blocked (no key / no RPC).
        On failed:  an earlier step (ML API, proxy, proof) failed.
    """
    from ._config import (
        _ML_API_URL, _OPERATOR_KEY, _PROXY_CHECKPOINT,
        _EZKL_RUN_PROOF, _SUBMIT_CONFIRM_BLOCKS,
        _w3, _REGISTRY_ADDRESS, _ABI_V2,
    )

    result: dict[str, Any] = {
        "status": "failed",
        "tx_hash": None,
        "class_scores": None,
        "class_score_felts": None,
        "proof_hash": None,
        "model_hash": model_hash,
        "failed_step": None,
        "reason": None,
    }

    # ── Step 1: call /fusion-embedding ─────────────────────────────────
    try:
        import requests
        resp = requests.post(
            f"{_ML_API_URL}/fusion-embedding",
            json={"source_code": source_code},
            timeout=120,
        )
        resp.raise_for_status()
        ml_result = resp.json()
        fusion_embedding = ml_result["fusion_embedding"]
        result["model_hash"] = ml_result.get("model_hash", model_hash)
    except Exception as exc:
        result["failed_step"] = "ml_api"
        result["reason"] = f"/fusion-embedding failed: {exc}"
        logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
        return result

    # ── Step 2: run proxy model locally → 10 class scores ─────────────
    try:
        import torch
        # Lazy import — only needed when submit tool is actually called
        sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
        from zkml.src.distillation.proxy_model import ProxyModel

        proxy = ProxyModel()
        state = torch.load(_PROXY_CHECKPOINT, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        proxy.load_state_dict(state)
        proxy.eval()

        features = torch.tensor([fusion_embedding])  # [1, 128]
        with torch.no_grad():
            logits = proxy(features)
        scores = torch.sigmoid(logits).squeeze(0).tolist()  # [10]
        felts = [round(s * 8192) for s in scores]
        result["class_scores"] = scores
        result["class_score_felts"] = felts
    except Exception as exc:
        result["failed_step"] = "proxy_inference"
        result["reason"] = f"Proxy model failed: {exc}"
        logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
        return result

    # ── Step 3: generate EZKL proof (inline, not subprocess) ──────────
    # The proof is generated for the ACTUAL contract's fusion embedding —
    # NOT from a corpus contract (unlike run_proof.py CLI mode).
    COMPILED      = Path(__file__).resolve().parents[5] / "zkml/ezkl/model.compiled"
    SETTINGS      = Path(__file__).resolve().parents[5] / "zkml/ezkl/settings.json"
    SRS           = Path(__file__).resolve().parents[5] / "zkml/ezkl/srs.params"
    PROVING_KEY   = Path(__file__).resolve().parents[5] / "zkml/ezkl/proving_key.pk"
    VERIFY_KEY    = Path(__file__).resolve().parents[5] / "zkml/ezkl/verification_key.vk"
    PROOF_INPUT   = Path(__file__).resolve().parents[5] / "zkml/ezkl/proof_input.json"
    WITNESS       = Path(__file__).resolve().parents[5] / "zkml/ezkl/witness.json"
    PROOF         = Path(__file__).resolve().parents[5] / "zkml/ezkl/proof.json"
    _INPUT_OFFSET = 128
    _NUM_CLASSES  = 10

    for f in (COMPILED, SETTINGS, SRS, PROVING_KEY, VERIFY_KEY):
        if not f.exists():
            result["failed_step"] = "proof_generation"
            result["reason"] = f"EZKL artifact missing: {f.name}. Run setup_circuit.py first."
            logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
            return result

    try:
        import ezkl

        # Write the real contract's fusion embedding as proof input
        proof_input = {"input_data": [fusion_embedding]}  # [[128 floats]]
        PROOF_INPUT.write_text(json.dumps(proof_input))

        # Step 6: gen_witness from the REAL fusion embedding
        witness = ezkl.gen_witness(
            data=str(PROOF_INPUT),
            model=str(COMPILED),
            output=str(WITNESS),
        )

        # Decode output field elements (the 10 class scores from the proof)
        outputs = witness["outputs"][0]
        public_signals_decoded = []
        for hex_str in outputs:
            felt = int.from_bytes(bytes.fromhex(hex_str), byteorder='little')
            public_signals_decoded.append(felt)

        if len(public_signals_decoded) != _NUM_CLASSES:
            raise RuntimeError(
                f"Expected {_NUM_CLASSES} output felts, got {len(public_signals_decoded)}"
            )

        # Step 7: prove
        ezkl.prove(
            witness=str(WITNESS),
            model=str(COMPILED),
            pk_path=str(PROVING_KEY),
            proof_path=str(PROOF),
            srs_path=str(SRS),
        )

        # Step 8: verify (off-chain)
        valid = ezkl.verify(
            proof_path=str(PROOF),
            settings_path=str(SETTINGS),
            vk_path=str(VERIFY_KEY),
            srs_path=str(SRS),
        )
        if not valid:
            raise RuntimeError("Off-chain proof verification failed")

        proof_data = json.loads(PROOF.read_text())
        hex_proof = proof_data["hex_proof"]

        # Parse ALL publicSignals from proof.json instances
        instances = proof_data["instances"][0]
        all_public_signals = [
            int.from_bytes(bytes.fromhex(h), byteorder='little')
            for h in instances
        ]
        if len(all_public_signals) != _INPUT_OFFSET + _NUM_CLASSES:
            raise RuntimeError(
                f"Expected {_INPUT_OFFSET + _NUM_CLASSES} publicSignals, "
                f"got {len(all_public_signals)}"
            )

        # Overwrite class_score_felts with the proof's ACTUAL quantized outputs.
        # EZKL's fixed-point sigmoid (lookup table, scale=13) can differ from
        # PyTorch's float32 sigmoid after rounding. Using the proof's values
        # guarantees classScores[i] == publicSignals[128+i] on every class.
        result["class_score_felts"] = all_public_signals[_INPUT_OFFSET:]

        result["proof_hash"] = "0x" + __import__("hashlib").sha256(
            bytes.fromhex(hex_proof[2:] if hex_proof.startswith("0x") else hex_proof)
        ).hexdigest()

        # Clean up temp files
        for tmp in (PROOF_INPUT, WITNESS, PROOF):
            if tmp.exists():
                tmp.unlink()

    except Exception as exc:
        for tmp in (PROOF_INPUT, WITNESS, PROOF):
            if tmp.exists():
                tmp.unlink()
        result["failed_step"] = "proof_generation"
        result["reason"] = f"Proof generation failed: {type(exc).__name__}: {str(exc)[:300]}"
        logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")
        return result

    # ── Step 3b: build provenance manifest ───────────────────────────
    try:
        import hashlib
        proxy_hash = hashlib.sha256(_PROXY_CHECKPOINT.read_bytes()).hexdigest()
        provenance = build_provenance_manifest(
            teacher_model_hash=result["model_hash"],
            proxy_checkpoint_hash=proxy_hash,
            fusion_embedding=fusion_embedding,
            class_scores=result["class_scores"],
            operator_address=_OPERATOR_KEY and "",
        )
        result["provenance"] = provenance
    except Exception as exc:
        result["provenance"] = None
        logger.warning(f"submit_audit: provenance manifest skipped — {exc}")

    # ── Step 4: submit on-chain ───────────────────────────────────────
    if not _OPERATOR_KEY:
        result["status"] = "partial"
        result["failed_step"] = "transaction"
        result["reason"] = "SENTINEL_OPERATOR_KEY not set — proof generated but not submitted"
        return result

    if _w3 is None or _REGISTRY_ADDRESS is None:
        result["status"] = "partial"
        result["failed_step"] = "transaction"
        result["reason"] = "Web3 not initialised or registry address not configured"
        return result

    try:
        account = _w3.eth.account.from_key(_OPERATOR_KEY)
        operator_address = account.address

        registry = _w3.eth.contract(
            address=_REGISTRY_ADDRESS,
            abi=_ABI_V2,
        )

        # Build classScores as Soldity uint256[10] array
        class_score_felts = result["class_score_felts"]

        tx = registry.functions.submitAuditV2(
            contract_address,
            class_score_felts,
            bytes.fromhex(hex_proof[2:]) if hex_proof.startswith("0x") else bytes.fromhex(hex_proof),
            all_public_signals,
            bytes.fromhex(model_hash) if len(model_hash) == 64 else bytes(32),
        ).build_transaction({
            "from": operator_address,
            "nonce": _w3.eth.get_transaction_count(operator_address),
            "gas": 1_000_000,
            "gasPrice": _w3.eth.gas_price,
        })

        signed = account.sign_transaction(tx)
        tx_hash = _w3.eth.send_raw_transaction(signed.raw_transaction)

        # Wait for confirmations
        receipt = _w3.eth.wait_for_transaction_receipt(
            tx_hash,
            timeout=120,
            poll_latency=2.0,
        )

        result["status"] = "submitted"
        result["tx_hash"] = "0x" + tx_hash.hex()

        logger.info(
            f"submit_audit OK — tx: {result['tx_hash'][:18]}... "
            f"block: {receipt.get('blockNumber', '?')}"
        )
    except Exception as exc:
        result["status"] = "partial" if result["proof_hash"] else "failed"
        result["failed_step"] = "transaction"
        result["reason"] = f"Transaction failed: {type(exc).__name__}: {str(exc)[:300]}"
        logger.error(f"submit_audit [{result['failed_step']}]: {result['reason']}")

    return result


def build_provenance_manifest(
    teacher_model_hash: str,
    proxy_checkpoint_hash: str,
    fusion_embedding: list[float],
    class_scores: list[float],
    operator_address: str,
) -> dict[str, Any]:
    """
    Build and EIP-191-sign a provenance manifest binding ML model to ZK proof.

    The ZK proof proves proxy(fusion_128) = class_scores[10], but does NOT
    prove that fusion_128 came from this specific teacher checkpoint. This
    manifest cryptographically binds the teacher, proxy, fusion, and scores
    under the operator's signature — providing an off-chain auditable trail.
    """
    import hashlib
    from datetime import datetime, timezone

    fusion_hash = hashlib.sha256(
        json.dumps(fusion_embedding, sort_keys=True).encode()
    ).hexdigest()

    manifest: dict[str, Any] = {
        "teacher_model_hash": teacher_model_hash,
        "proxy_checkpoint_hash": proxy_checkpoint_hash,
        "fusion_embedding_hash": fusion_hash,
        "class_scores": [round(s, 6) for s in class_scores],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operator_address": operator_address,
    }

    try:
        from eth_account.messages import encode_defunct
        from ._config import _OPERATOR_KEY
        from web3.auto import w3

        if _OPERATOR_KEY:
            message = encode_defunct(text=json.dumps(manifest, sort_keys=True))
            signed = w3.eth.account.sign_message(message, private_key=_OPERATOR_KEY)
            manifest["signature"] = "0x" + signed.signature.hex()
        else:
            manifest["signature"] = None
    except ImportError:
        manifest["signature"] = None
        logger.warning("provenance: eth_account not installed — signature omitted")

    return manifest
