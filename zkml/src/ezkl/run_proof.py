"""
run_proof.py — EZKL Pipeline Steps 6-8 (per-audit proof generation)

RECALL — the full EZKL pipeline has two phases:
    ONE-TIME (setup_circuit.py — already complete):
        Steps 1-5: gen_settings → calibrate → compile → get_srs → setup
        Output: proving_key.pk + verification_key.vk

    PER AUDIT (this file):
        Step 6: gen_witness → encode real inputs as field elements
        Step 7: prove       → generate cryptographic proof π (~2KB)
        Step 8: verify      → confirm proof is valid

RECALL — what a proof actually proves:
    "I know private weights W such that when I run the circuit
     defined by model.compiled on these specific public inputs
     (the 64 contract features), I get this specific public output
     (the risk score) — and I can prove this without showing W."

    The proof is ~2KB. It contains no weight information.
    Anyone with verification_key.vk can verify it in milliseconds.
    On-chain: ZKMLVerifier.verifyProof(proof, publicSignals) → bool

RECALL — what publicSignals contains:
    [features[0..63], risk_score]
    Index 0-63:  the 64 input features (public)
    Index 64:    the risk score output (public)
    AuditRegistry checks: publicSignals[64] == scoreFieldElement
    (full uint256 comparison — NOT uint8 truncation)

RECALL — BN254 field element encoding (CRITICAL):
    EZKL stores all field elements as 32-byte little-endian hex strings.
    proof.json instances[i] = 32-byte little-endian hex
    To get the Solidity uint256:
        CORRECT:  int.from_bytes(bytes.fromhex(instances[i]), byteorder='little')
        WRONG:    int(instances[i], 16)  ← treats as big-endian, huge garbage value
    Example: instances[64] = "9111000000...00"
        little-endian → 0x1191 = 4497  (correct: 4497/8192 = 0.5490 score)
        big-endian    → 65615399...    (wrong: not a valid field element)

RECALL — why gen_witness exists as a separate step:
    The witness encodes floating point values as BN254 field elements
    using the scale factor from calibration (2^13 = 8192).
    Example: 0.131 × 8192 = 1073 → stored as field element.
    EZKL needs this intermediate representation before proving
    because the circuit operates on field elements, not floats.

Usage:
    cd ~/projects/sentinel
    poetry run python zkml/src/ezkl/run_proof.py

    Generates proof for first contract in val set.
    Verifies proof off-chain.
    Prints publicSignals — what gets submitted to AuditRegistry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from ml.src.models.sentinel_model import SentinelModel
from torch_geometric.loader import DataLoader
from zkml.src.distillation.proxy_model import CIRCUIT_VERSION, ProxyModel

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

TEACHER_CHECKPOINT = "ml/checkpoints/run-alpha-tune_best.pt"
PROXY_CHECKPOINT   = "zkml/models/proxy_best.pt"
ONNX_MODEL         = "zkml/models/proxy.onnx"
COMPILED           = "zkml/ezkl/model.compiled"
SETTINGS           = "zkml/ezkl/settings.json"
SRS                = "zkml/ezkl/srs.params"
PROVING_KEY        = "zkml/ezkl/proving_key.pk"
VERIFICATION_KEY   = "zkml/ezkl/verification_key.vk"

GRAPHS_DIR = "ml/data/graphs"
TOKENS_DIR = "ml/data/tokens"
SPLITS_DIR = "ml/data/splits"

# Output paths for this proof
PROOF_INPUT  = "zkml/ezkl/proof_input.json"
WITNESS      = "zkml/ezkl/witness.json"
PROOF        = "zkml/ezkl/proof.json"


def check_prerequisites() -> None:
    """Verify EZKL setup artifacts exist before attempting proof."""
    required = {
        "Compiled circuit":  COMPILED,
        "Settings":          SETTINGS,
        "SRS":               SRS,
        "Proving key":       PROVING_KEY,
        "Verification key":  VERIFICATION_KEY,
    }
    for name, path in required.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} not found: {path}\n"
                f"Run setup_circuit.py first."
            )
    logger.info("Prerequisites check passed — EZKL setup artifacts present")


@torch.no_grad()
def extract_single_contract_features(
    device: str,
) -> tuple[list[float], float, float]:
    """
    Extract 64-dim features and scores for one real contract.

    Returns:
        features:      list of 64 floats — proxy model input
        teacher_score: float — teacher's risk score
        proxy_score:   float — proxy's risk score (should match teacher)

    Raises:
        ValueError: if teacher and proxy disagree on the binary classification.
                    A proof generated under disagreement is cryptographically
                    valid but semantically incorrect — it would attest a score
                    that contradicts the full teacher model. Do not submit it.
    """
    # ── Load teacher ────────────────────────────────────────────────────
    # CHECKPOINT FORMAT NOTE (A-12 fix):
    # Old format (pre-April 2026):  torch.save(model.state_dict(), path)
    #   → torch.load returns an OrderedDict of parameter tensors directly.
    # New format (trainer.py):      torch.save({"model": ..., "optimizer": ...,
    #                                            "epoch": N, "best_f1": f, "config": cfg}, path)
    #   → torch.load returns a plain dict; state_dict is at ckpt["model"].
    #
    # We use weights_only=False because new-format checkpoints contain non-tensor
    # objects (TrainConfig dataclass, optimizer state) that pickle cannot load
    # safely with weights_only=True.  This file only loads checkpoints from
    # our own ml/checkpoints/ directory — the security tradeoff is acceptable.
    teacher = SentinelModel().to(device)
    _ckpt = torch.load(TEACHER_CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(_ckpt, dict) and "model" in _ckpt:
        # New format — extract just the model weights
        _state_dict = _ckpt["model"]
        logger.debug(
            f"Teacher checkpoint — new format detected "
            f"(epoch {_ckpt.get('epoch', '?')}, best_f1 {_ckpt.get('best_f1', '?'):.4f})"
        )
    else:
        # Old format — the loaded object IS the state_dict
        _state_dict = _ckpt
        logger.debug("Teacher checkpoint — legacy format (raw state_dict)")
    teacher.load_state_dict(_state_dict)
    teacher.eval()

    # ── Load proxy ──────────────────────────────────────────────────────
    # Same format detection logic as teacher above.
    proxy = ProxyModel().to(device)
    _ckpt = torch.load(PROXY_CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(_ckpt, dict) and "model" in _ckpt:
        _state_dict = _ckpt["model"]
        logger.debug(
            f"Proxy checkpoint — new format detected "
            f"(epoch {_ckpt.get('epoch', '?')}, best_f1 {_ckpt.get('best_f1', '?'):.4f})"
        )
    else:
        _state_dict = _ckpt
        logger.debug("Proxy checkpoint — legacy format (raw state_dict)")
    proxy.load_state_dict(_state_dict)
    proxy.eval()

    # Load one contract from val set
    val_indices = np.load(f"{SPLITS_DIR}/val_indices.npy")
    dataset = DualPathDataset(
        graphs_dir=GRAPHS_DIR,
        tokens_dir=TOKENS_DIR,
        indices=[val_indices[0]],  # single contract
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dual_path_collate_fn,
    )

    graphs, tokens, label = next(iter(loader))
    graphs         = graphs.to(device)
    input_ids      = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Extract FusionLayer features — proxy input
    gnn_out         = teacher.gnn(graphs.x, graphs.edge_index, graphs.batch)
    transformer_out = teacher.transformer(input_ids, attention_mask)
    features        = teacher.fusion(gnn_out, transformer_out)  # [1, 64]

    # Teacher final score
    teacher_score = teacher.classifier(features).squeeze(1).item()

    # Proxy score — what gets proved
    proxy_score = proxy(features).item()

    teacher_vulnerable = teacher_score >= 0.5
    proxy_vulnerable   = proxy_score   >= 0.5

    logger.info(
        f"Contract loaded — "
        f"label: {label.item()} | "
        f"teacher: {teacher_score:.4f} ({'VULN' if teacher_vulnerable else 'SAFE'}) | "
        f"proxy:   {proxy_score:.4f} ({'VULN' if proxy_vulnerable else 'SAFE'}) | "
        f"agreement: {'YES ✓' if teacher_vulnerable == proxy_vulnerable else 'NO ✗'}"
    )

    # CRITICAL — reject if teacher and proxy classify differently.
    # A proof generated under disagreement is cryptographically valid —
    # the ZK proof confirms the proxy's output — but it contradicts the
    # full teacher model that the proxy is meant to approximate.
    # Submitting such a proof would register a misleading audit result
    # on-chain. Hard-reject here so the issue surfaces immediately.
    if teacher_vulnerable != proxy_vulnerable:
        raise ValueError(
            f"Teacher/proxy disagreement — proof generation rejected.\n"
            f"  Teacher: {teacher_score:.4f} → {'VULNERABLE' if teacher_vulnerable else 'SAFE'}\n"
            f"  Proxy:   {proxy_score:.4f}   → {'VULNERABLE' if proxy_vulnerable else 'SAFE'}\n"
            f"This contract is near the decision boundary.\n"
            f"Options:\n"
            f"  1. Use a different contract with a more decisive score.\n"
            f"  2. Investigate whether the proxy needs retraining.\n"
            f"  3. If proxy agreement is systematically low, check train_proxy.py."
        )

    return features.squeeze(0).cpu().tolist(), teacher_score, proxy_score


def generate_proof(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> bool:
    """
    Full per-audit proof pipeline: witness → prove → verify.

    This is what runs for every audit request in production.
    In Module 6, this logic moves into a Celery task called
    asynchronously from the FastAPI endpoint.

    Returns:
        True if proof is valid and ready for on-chain submission.

    Raises:
        FileNotFoundError: if setup artifacts are missing (run setup_circuit.py)
        ValueError:        if teacher/proxy disagree (contract near boundary)
        RuntimeError:      if proof generation or verification fails

    On any exception, partially written WITNESS and PROOF files are removed
    so the next run starts from a clean state.
    """
    check_prerequisites()
    logger.info(f"Generating proof on: {device}")
    logger.info(f"Circuit version: {CIRCUIT_VERSION}")

    # Partial artifacts to clean up if anything goes wrong after they are created
    _partial_artifacts = [WITNESS, PROOF]

    try:
        # ------------------------------------------------------------------
        # Step 6a — Extract real contract features
        # ------------------------------------------------------------------
        # RECALL — the proxy never sees raw Solidity.
        # The full teacher pipeline runs first:
        #   raw .sol → AST → GNN + CodeBERT → FusionLayer → 64 features
        # The proxy maps those 64 features → risk score.
        # The proof proves: "I ran the proxy on these 64 features
        #                    and got this risk score."
        # ValueError is raised here if teacher/proxy disagree.
        features, teacher_score, proxy_score = extract_single_contract_features(device)

        # ------------------------------------------------------------------
        # Step 6b — Format input for EZKL
        # ------------------------------------------------------------------
        # EZKL proof_input format:
        #   {"input_data": [[f1, f2, ..., f64]]}
        # Single contract = single inner list of 64 floats.
        proof_input = {"input_data": [features]}
        with open(PROOF_INPUT, "w") as f:
            json.dump(proof_input, f)
        logger.info(f"Proof input saved: {PROOF_INPUT}")

        # ------------------------------------------------------------------
        # Step 6c — gen_witness
        # ------------------------------------------------------------------
        # RECALL — what gen_witness does:
        #   Encodes floating point inputs as BN254 field elements
        #   using the scale factor from calibration (2^13).
        #   Example: 0.131 × 8192 = 1073 → hex field element.
        #   The circuit operates on these field elements, not raw floats.
        #   This intermediate step is required before proving.
        import ezkl
        logger.info("Step 6/8 — gen_witness")

        witness = ezkl.gen_witness(
            data=PROOF_INPUT,
            model=COMPILED,
            output=WITNESS,
        )
        logger.info(f"Witness generated: {WITNESS}")

        # Extract the output field element — the encoded risk score.
        # RECALL — witness["outputs"][0][0] is a little-endian hex string.
        # It will become publicSignals[64] on-chain after correct decoding.
        output_felt = witness["outputs"][0][0]
        score_field_element = int.from_bytes(bytes.fromhex(output_felt), byteorder='little')
        logger.info(
            f"Output field element: {score_field_element} "
            f"(human: {score_field_element / 8192:.4f})"
        )

        # ------------------------------------------------------------------
        # Step 7 — prove
        # ------------------------------------------------------------------
        # RECALL — what prove does:
        #   Takes private weights (via proving_key) + public inputs (witness)
        #   + circuit constraints (compiled model).
        #   Runs Halo2 proving protocol over BN254 curve.
        #   Produces proof π — a ~2KB cryptographic object.
        #   Anyone with verification_key.vk can verify this proof.
        #   The proof contains NO weight information — weights stay private.
        logger.info("Step 7/8 — prove (this may take 30-60 seconds)")

        proof_result = ezkl.prove(
            witness=WITNESS,
            model=COMPILED,
            pk_path=PROVING_KEY,
            proof_path=PROOF,
            srs_path=SRS,
        )

        proof_size_kb = Path(PROOF).stat().st_size / 1024
        logger.info(f"Proof generated: {PROOF} ({proof_size_kb:.1f} KB)")

        # ------------------------------------------------------------------
        # Step 8 — verify (off-chain)
        # ------------------------------------------------------------------
        # RECALL — what verify does:
        #   Checks proof π against publicSignals using verification_key.
        #   No weights needed — only the circuit fingerprint (vk).
        #   This is the off-chain version.
        #   On-chain version: ZKMLVerifier.verifyProof(proof, publicSignals) → bool
        #   Both should return true for a valid proof.
        logger.info("Step 8/8 — verify")

        valid = ezkl.verify(
            proof_path=PROOF,
            settings_path=SETTINGS,
            vk_path=VERIFICATION_KEY,
            srs_path=SRS,
        )

        if not valid:
            raise RuntimeError(
                "Off-chain verification failed — proof is cryptographically invalid.\n"
                "Possible causes:\n"
                "  - Proving key does not match the compiled circuit\n"
                "  - Witness was generated with different inputs than the proof\n"
                "  - EZKL version mismatch between setup and prove steps\n"
                "Run: poetry run python zkml/src/ezkl/setup_circuit.py"
            )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Proof pipeline complete")
        logger.info(f"  Teacher score:        {teacher_score:.4f}")
        logger.info(f"  Proxy score:          {proxy_score:.4f}")
        logger.info(f"  Classification:       {'VULNERABLE' if proxy_score >= 0.5 else 'SAFE'}")
        logger.info(f"  Score field element:  {score_field_element}  (= {score_field_element}/8192 = {score_field_element/8192:.4f})")
        logger.info(f"  Proof size:           {proof_size_kb:.1f} KB")
        logger.info(f"  Off-chain valid:      {valid} ✓")
        logger.info("  Ready for AuditRegistry.submitAudit()")
        logger.info("  Run: poetry run python zkml/src/ezkl/extract_calldata.py")
        logger.info("=" * 60)

        return True

    except Exception:
        # Clean up any partially written artifacts so the next run starts fresh.
        # PROOF_INPUT is intentionally kept — it's just input features, harmless.
        for path in _partial_artifacts:
            if Path(path).exists():
                Path(path).unlink()
                logger.warning(f"Cleaned up partial artifact: {path}")
        raise


if __name__ == "__main__":
    generate_proof()
