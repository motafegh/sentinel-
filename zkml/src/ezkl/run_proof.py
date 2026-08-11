"""Generate and verify a legacy SENTINEL V2 EZKL proxy proof.

The V2 circuit proves only the student computation over a 128-dimensional
fusion embedding. It does **not** bind contract/chain/round/teacher identity and
is therefore not eligible for verified-audit finality under the R0 policy.

Score semantics are locked to the trained artifact: ``ProxyModel.forward()`` is
fitted directly to ``sigmoid(teacher_logits)`` and is used as-is. No second
sigmoid is applied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.src.inference.predictor import Predictor
from zkml.src.distillation.proxy_model import (
    CIRCUIT_VERSION,
    OUTPUT_SEMANTICS,
    ProxyModel,
)

TEACHER_CHECKPOINT = Path(
    "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"
)
PROXY_CHECKPOINT = Path("zkml/models/proxy_best.pt")
COMPILED = Path("zkml/ezkl/model.compiled")
SETTINGS = Path("zkml/ezkl/settings.json")
SRS = Path("zkml/ezkl/srs.params")
PROVING_KEY = Path("zkml/ezkl/proving_key.pk")
VERIFICATION_KEY = Path("zkml/ezkl/verification_key.vk")

CORPUS_ROOT = Path("manual_hand_written_contracts")
PROOF_INPUT = Path("zkml/ezkl/proof_input.json")
WITNESS = Path("zkml/ezkl/witness.json")
PROOF = Path("zkml/ezkl/proof.json")

INPUT_DIM = 128
NUM_CLASSES = 10
TOTAL_SIGNALS = INPUT_DIM + NUM_CLASSES
SCALE = 8192
PROOF_SCOPE = "legacy_proxy_only_unbound"

CLASS_NAMES = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]


def _load_proxy(device: str) -> ProxyModel:
    if not PROXY_CHECKPOINT.exists():
        raise FileNotFoundError(f"proxy checkpoint missing: {PROXY_CHECKPOINT}")
    proxy = ProxyModel().to(device)
    state: Any = torch.load(PROXY_CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    proxy.load_state_dict(state)
    proxy.eval()
    return proxy


def check_prerequisites() -> None:
    required = {
        "teacher checkpoint": TEACHER_CHECKPOINT,
        "proxy checkpoint": PROXY_CHECKPOINT,
        "compiled circuit": COMPILED,
        "settings": SETTINGS,
        "SRS": SRS,
        "proving key": PROVING_KEY,
        "verification key": VERIFICATION_KEY,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "required V2 proof artifacts are missing:\n  - " + "\n  - ".join(missing)
        )


def _default_contract() -> Path:
    for path in sorted(CORPUS_ROOT.rglob("*.sol")):
        if "_quarantine" not in path.parts:
            return path
    raise FileNotFoundError(f"no Solidity contract found under {CORPUS_ROOT}")


@torch.no_grad()
def extract_contract_features(
    predictor: Predictor,
    proxy: ProxyModel,
    sol_file: Path,
    device: str,
) -> tuple[list[float], list[float], list[float]]:
    """Return fusion[128], teacher probabilities[10], student scores[10]."""
    if not sol_file.exists():
        raise FileNotFoundError(f"contract not found: {sol_file}")

    source_code = sol_file.read_text(encoding="utf-8", errors="strict")
    graph, windows = predictor.preprocessor.process_source_windowed(source_code)
    batch = Batch.from_data_list([graph]).to(device)

    selected = list(windows[:4])
    while len(selected) < 4:
        selected.append(
            {
                "input_ids": torch.zeros(1, 512, dtype=torch.long),
                "attention_mask": torch.zeros(1, 512, dtype=torch.long),
            }
        )
    input_ids = torch.cat([w["input_ids"].to(device) for w in selected], dim=0).unsqueeze(0)
    attention_mask = torch.cat(
        [w["attention_mask"].to(device) for w in selected], dim=0
    ).unsqueeze(0)

    model = predictor.model
    model.eval()
    teacher_logits, aux = model(batch, input_ids, attention_mask, return_aux=True)
    fusion = aux["fusion_embedding"]
    if tuple(fusion.shape) != (1, INPUT_DIM):
        raise RuntimeError(f"fusion shape must be [1,{INPUT_DIM}], got {tuple(fusion.shape)}")

    teacher_probabilities = torch.sigmoid(teacher_logits.float()).squeeze(0).cpu()
    student_scores = proxy(fusion.to(device)).squeeze(0).float().cpu()
    if teacher_probabilities.numel() != NUM_CLASSES or student_scores.numel() != NUM_CLASSES:
        raise RuntimeError("teacher/proxy class dimension must be 10")
    if not torch.isfinite(student_scores).all():
        raise RuntimeError("proxy produced non-finite student score(s)")

    return (
        fusion.squeeze(0).float().cpu().tolist(),
        teacher_probabilities.tolist(),
        student_scores.tolist(),
    )


def _decode_felt(hex_str: str) -> int:
    raw = bytes.fromhex(hex_str)
    if len(raw) != 32:
        raise ValueError(f"EZKL field element must be 32 bytes, got {len(raw)}")
    return int.from_bytes(raw, byteorder="little")


def generate_proof(
    sol_file: Path | None = None,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Generate + verify one explicit V2 proof and return its diagnostics."""
    check_prerequisites()
    selected_contract = sol_file or _default_contract()

    predictor = Predictor(checkpoint=str(TEACHER_CHECKPOINT))
    proxy = _load_proxy(device)
    features, teacher_probabilities, student_scores = extract_contract_features(
        predictor, proxy, selected_contract, device
    )

    disagreements = [
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
        if (teacher_probabilities[i] >= 0.5) != (student_scores[i] >= 0.5)
    ]

    logger.info(
        "V2 proof input contract={} circuit={} output_semantics={} disagreements={}",
        selected_contract,
        CIRCUIT_VERSION,
        OUTPUT_SEMANTICS,
        disagreements,
    )

    PROOF_INPUT.parent.mkdir(parents=True, exist_ok=True)
    PROOF_INPUT.write_text(
        json.dumps({"input_data": [features]}), encoding="utf-8"
    )

    # Partial proof artifacts are removed on failure so a stale proof cannot be
    # mistaken for the current request.
    for path in (WITNESS, PROOF):
        if path.exists():
            path.unlink()

    try:
        import ezkl

        witness = ezkl.gen_witness(
            data=str(PROOF_INPUT),
            model=str(COMPILED),
            output=str(WITNESS),
        )
        witness_outputs = witness.get("outputs", [[]])[0]
        if len(witness_outputs) != NUM_CLASSES:
            raise RuntimeError(
                f"expected {NUM_CLASSES} witness outputs, got {len(witness_outputs)}"
            )

        ezkl.prove(
            witness=str(WITNESS),
            model=str(COMPILED),
            pk_path=str(PROVING_KEY),
            proof_path=str(PROOF),
            srs_path=str(SRS),
        )
        if not PROOF.exists():
            raise RuntimeError("ezkl.prove returned without writing proof.json")

        valid = ezkl.verify(
            proof_path=str(PROOF),
            settings_path=str(SETTINGS),
            vk_path=str(VERIFICATION_KEY),
            srs_path=str(SRS),
        )
        if not valid:
            raise RuntimeError("off-chain EZKL proof verification failed")

        proof_data = json.loads(PROOF.read_text(encoding="utf-8"))
        instances = proof_data.get("instances", [[]])[0]
        if len(instances) != TOTAL_SIGNALS:
            raise RuntimeError(
                f"expected exactly {TOTAL_SIGNALS} public signals, got {len(instances)}"
            )
        public_signals = [_decode_felt(item) for item in instances]
        output_felts = public_signals[INPUT_DIM:]

        result = {
            "status": "verified_off_chain",
            "proof_scope": PROOF_SCOPE,
            "submission_eligible": False,
            "submission_ineligible_reason": "proof_scope_not_identity_bound",
            "contract_path": str(selected_contract),
            "circuit_version": CIRCUIT_VERSION,
            "output_semantics": OUTPUT_SEMANTICS,
            "teacher_probabilities": teacher_probabilities,
            "proxy_scores": student_scores,
            "threshold_disagreements": disagreements,
            "public_signal_count": len(public_signals),
            "proxy_output_felts": output_felts,
            "proxy_outputs_approx": [value / SCALE for value in output_felts],
            "proof_path": str(PROOF),
        }
        logger.info(
            "V2 proof verified off-chain; policy eligibility remains false ({})",
            result["submission_ineligible_reason"],
        )
        return result
    except Exception:
        for path in (WITNESS, PROOF):
            if path.exists():
                path.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = generate_proof(args.contract, device=args.device)
    except Exception as exc:
        logger.exception("V2 proof generation failed: {}", exc)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
