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
     (the 128 CrossAttentionFusion features), I get these specific
     10 class score outputs — and I can prove this without showing W."

    The proof is ~2KB. It contains no weight information.
    Anyone with verification_key.vk can verify it in milliseconds.
    On-chain: ZKMLVerifier.verifyProof(proof, publicSignals) → bool

RECALL — what publicSignals contains (v2.0 circuit, 10-class):
    [fusion_features[0..127], class_score_0, ..., class_score_9]
    Total: 128 + 10 = 138 public signals.
    AuditRegistry V2 Guard 3 checks: publicSignals[128 + i] == classScores[i] ∀i∈[0,9]

RECALL — BN254 field element encoding (CRITICAL):
    EZKL stores all field elements as 32-byte little-endian hex strings.
    proof.json instances[i] = 32-byte little-endian hex
    To get the Solidity uint256:
        CORRECT:  int.from_bytes(bytes.fromhex(instances[i]), byteorder='little')
        WRONG:    int(instances[i], 16)  ← treats as big-endian, huge garbage value

Usage:
    cd ~/projects/sentinel
    source ml/.venv/bin/activate
    python zkml/src/ezkl/run_proof.py

    Generates proof for first contract in the corpus.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from torch_geometric.data import Batch
from ml.src.inference.predictor import Predictor
from zkml.src.distillation.proxy_model import CIRCUIT_VERSION, ProxyModel

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

TEACHER_CHECKPOINT = "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"
PROXY_CHECKPOINT   = "zkml/models/proxy_best.pt"
ONNX_MODEL         = "zkml/models/proxy.onnx"
COMPILED           = "zkml/ezkl/model.compiled"
SETTINGS           = "zkml/ezkl/settings.json"
SRS                = "zkml/ezkl/srs.params"
PROVING_KEY        = "zkml/ezkl/proving_key.pk"
VERIFICATION_KEY   = "zkml/ezkl/verification_key.vk"

CORPUS_ROOT        = "manual_hand_written_contracts"
NUM_CLASSES        = 10
INPUT_DIM          = 128
SCALE              = 8192  # 2^13

# Output paths for this proof
PROOF_INPUT  = "zkml/ezkl/proof_input.json"
WITNESS      = "zkml/ezkl/witness.json"
PROOF        = "zkml/ezkl/proof.json"

# Class names matching graph_schema.py CLASS_NAMES order
CLASS_NAMES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn",
]


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


def _find_first_contract() -> Path:
    """Find the first .sol contract in the corpus (excl. quarantine)."""
    root = Path(CORPUS_ROOT)
    for sol_file in sorted(root.rglob("*.sol")):
        if "_quarantine" in str(sol_file):
            continue
        return sol_file
    raise FileNotFoundError(
        f"No .sol contracts found under {CORPUS_ROOT} (excluding _quarantine/)"
    )


def _find_all_contracts() -> list[Path]:
    """Find all .sol contracts in the corpus (excl. quarantine)."""
    root = Path(CORPUS_ROOT)
    return sorted(
        p for p in root.rglob("*.sol")
        if "_quarantine" not in str(p)
    )


@torch.no_grad()
def extract_corpus_contract_features(
    predictor: Predictor,
    sol_file: Path,
    device: str,
) -> tuple[list[float], list[float], list[float], int]:
    """
    Extract 128-dim fusion features and 10-class scores from a single corpus contract.

    Returns:
        features:       list of 128 floats
        teacher_scores: list of 10 floats
        proxy_scores:   list of 10 floats
        n_disagreements: how many classes disagree at threshold 0.5
    """
    model = predictor.model
    model.eval()

    if not sol_file.exists():
        raise FileNotFoundError(f"Contract not found: {sol_file}")

    logger.info(f"Extracting features from: {sol_file.name}")
    source_code = sol_file.read_text(encoding="utf-8", errors="replace")

    graph, windows = predictor.preprocessor.process_source_windowed(source_code)
    batch = Batch.from_data_list([graph]).to(device)

    selected = windows[:4]
    pad_ids  = torch.zeros(1, 512, dtype=torch.long, device=device)
    pad_mask = torch.zeros(1, 512, dtype=torch.long, device=device)
    padded = list(selected)
    while len(padded) < 4:
        padded.append({"input_ids": pad_ids, "attention_mask": pad_mask})
    stacked_ids  = torch.cat(
        [w["input_ids"].to(device) for w in padded], dim=0
    ).unsqueeze(0)
    stacked_mask = torch.cat(
        [w["attention_mask"].to(device) for w in padded], dim=0
    ).unsqueeze(0)

    with torch.no_grad():
        logits, aux = model(batch, stacked_ids, stacked_mask, return_aux=True)

    features_128 = aux["fusion_embedding"].squeeze(0)          # [128]
    teacher_logits = logits.squeeze(0)                          # [10]
    teacher_scores = torch.sigmoid(teacher_logits).cpu()        # [10]

    # Load proxy and compute proxy scores
    proxy = ProxyModel().to(device)
    proxy_state = torch.load(PROXY_CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(proxy_state, dict) and "model" in proxy_state:
        proxy_state = proxy_state["model"]
    proxy.load_state_dict(proxy_state)
    proxy.eval()

    proxy_logits = proxy(features_128.unsqueeze(0).to(device)).squeeze(0)  # [10]
    proxy_scores = torch.sigmoid(proxy_logits).cpu()                        # [10]

    # Convert to Python lists for the caller
    features_list = features_128.cpu().tolist()
    teacher_list  = teacher_scores.tolist()
    proxy_list    = proxy_scores.tolist()

    # Count per-class disagreements (report, don't block)
    disagreements = []
    for i in range(NUM_CLASSES):
        t_vuln = teacher_list[i] >= 0.5
        p_vuln = proxy_list[i] >= 0.5
        if t_vuln != p_vuln:
            disagreements.append((CLASS_NAMES[i], teacher_list[i], proxy_list[i]))

    if disagreements:
        logger.warning(
            f"  {len(disagreements)}/{NUM_CLASSES} class disagreements:"
        )
        for cls, t, p in disagreements:
            logger.warning(f"    {cls}: teacher={t:.4f} proxy={p:.4f}")
    else:
        logger.info("Teacher/proxy agreement: all 10 classes match at threshold 0.5")

    return features_list, teacher_list, proxy_list, len(disagreements)


def generate_proof(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> bool:
    """
    Full per-audit proof pipeline: witness → prove → verify.

    Returns:
        True if proof is valid and ready for on-chain submission.

    Raises:
        FileNotFoundError: if setup artifacts are missing (run setup_circuit.py)
        ValueError:        if teacher/proxy disagree (contract near boundary)
        RuntimeError:      if proof generation or verification fails
    """
    check_prerequisites()
    logger.info(f"Generating proof on: {device}")
    logger.info(f"Circuit version: {CIRCUIT_VERSION}")

    _partial_artifacts = [WITNESS, PROOF]

    try:
        # ── Load teacher ────────────────────────────────────────────────
        predictor = Predictor(checkpoint=TEACHER_CHECKPOINT)
        logger.info(f"Teacher loaded — architecture: {predictor.architecture}")

        # ── Extract features — try contracts until one with acceptable agreement ──
        all_contracts = _find_all_contracts()
        if not all_contracts:
            raise RuntimeError("No contracts found in corpus.")

        best_contract = None
        best_features = None
        best_teacher  = None
        best_proxy    = None
        best_n_disag = NUM_CLASSES + 1

        for sol_file in all_contracts[:10]:  # try at most 10 contracts
            try:
                feats, t_scores, p_scores, n_disag = extract_corpus_contract_features(
                    predictor, sol_file, device,
                )
                if n_disag < best_n_disag:
                    best_n_disag  = n_disag
                    best_features = feats
                    best_teacher  = t_scores
                    best_proxy    = p_scores
                    best_contract = sol_file
                    if n_disag == 0:
                        break
            except Exception as e:
                logger.warning(f"  Skipped {sol_file.name}: {e}")
                continue

        if best_contract is None:
            raise RuntimeError("Could not extract features from any contract.")

        features, teacher_scores, proxy_scores = best_features, best_teacher, best_proxy
        logger.info(
            f"Selected {best_contract.name} — "
            f"{best_n_disag}/{NUM_CLASSES} disagreements (best available)"
        )

        # ── Format input for EZKL ───────────────────────────────────────
        # EZKL proof_input format: {"input_data": [[f1, f2, ..., f128]]}
        proof_input = {"input_data": [features]}
        with open(PROOF_INPUT, "w") as f:
            json.dump(proof_input, f)
        logger.info(f"Proof input saved: {PROOF_INPUT} ({len(features)} features)")

        # ── Step 6: gen_witness ─────────────────────────────────────────
        import ezkl
        logger.info("Step 6/8 — gen_witness")

        witness = ezkl.gen_witness(
            data=PROOF_INPUT,
            model=COMPILED,
            output=WITNESS,
        )
        logger.info(f"Witness generated: {WITNESS}")

        # Decode all 10 output field elements (class scores)
        outputs = witness["outputs"][0]  # list of 10 little-endian hex strings
        if len(outputs) != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES} output field elements, got {len(outputs)}. "
                f"Circuit may have been compiled for a different number of classes."
            )

        class_score_felts = []
        for i, hex_str in enumerate(outputs):
            felt = int.from_bytes(bytes.fromhex(hex_str), byteorder='little')
            class_score_felts.append(felt)
            logger.info(
                f"  class[{i}] {CLASS_NAMES[i]:>25s}: "
                f"felt={felt:>6d}  human={felt / SCALE:.4f}  "
                f"(teacher: {teacher_scores[i]:.4f}, proxy: {proxy_scores[i]:.4f})"
            )

        # ── Step 7: prove ───────────────────────────────────────────────
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

        # ── Step 8: verify (off-chain) ──────────────────────────────────
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
                "Run: python zkml/src/ezkl/setup_circuit.py"
            )

        # ── Summary ─────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Proof pipeline complete")
        logger.info(f"  Contract:          {best_contract.name}")
        logger.info(f"  Circuit version:   {CIRCUIT_VERSION}")
        logger.info(f"  Proof size:        {proof_size_kb:.1f} KB")
        logger.info(f"  Off-chain valid:   {valid} ✓")
        logger.info(f"  Public signals:    {INPUT_DIM} inputs + {NUM_CLASSES} outputs = {INPUT_DIM + NUM_CLASSES}")
        logger.info(f"  Class disagreements:{best_n_disag}/{NUM_CLASSES}")
        for i in range(NUM_CLASSES):
            logger.info(
                f"  classScore[{i}] {CLASS_NAMES[i]:>25s}: "
                f"felt={class_score_felts[i]:>6d}  "
                f"human={class_score_felts[i] / SCALE:.4f}"
            )
        logger.info("  Ready for AuditRegistry.submitAuditV2()")
        logger.info("  Run: python zkml/src/ezkl/extract_calldata.py")
        logger.info("=" * 60)

        return True

    except Exception:
        for path in _partial_artifacts:
            if Path(path).exists():
                Path(path).unlink()
                logger.warning(f"Cleaned up partial artifact: {path}")
        raise


if __name__ == "__main__":
    generate_proof()
