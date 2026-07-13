"""End-to-end integration test: ML → ZKML → proof verification.

Tests the full pipeline without an HTTP server:
  1. Load teacher Predictor (Run 12 checkpoint)
  2. Extract 128-dim fusion embedding from a real corpus contract
  3. Run proxy model → 10 class logits
  4. Generate EZKL witness → prove → verify
  5. Assert correct signal layout (138 publicSignals)
  6. Assert class_score_felts match proof outputs
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Skip markers ─────────────────────────────────────────────────────────

requires_teacher = pytest.mark.skipif(
    not (Path("ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt").exists()),
    reason="Teacher checkpoint not available (DVC pull needed)",
)

requires_ezkl_artifacts = pytest.mark.skipif(
    not (Path("zkml/ezkl/model.compiled").exists()
         and Path("zkml/ezkl/proving_key.pk").exists()),
    reason="EZKL artifacts not generated (run setup_circuit.py)",
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def teacher():
    """Load the teacher predictor once per test module."""
    from ml.src.inference.predictor import Predictor
    return Predictor(checkpoint="ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt")


@pytest.fixture(scope="module")
def proxy():
    """Load the proxy model once per test module."""
    from zkml.src.distillation.proxy_model import ProxyModel
    p = ProxyModel()
    state = torch.load("zkml/models/proxy_best.pt", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    p.load_state_dict(state)
    p.eval()
    return p


# ── Tests ─────────────────────────────────────────────────────────────────

@requires_teacher
@requires_gpu
def test_teacher_loads_and_predicts(teacher):
    """Teacher loads and produces fusion embeddings."""
    sol = Path("manual_hand_written_contracts/Reentrancy/01_cei_violation_erc721.sol")
    if not sol.exists():
        pytest.skip("test contract not found")
    source = sol.read_text()

    result = teacher.predict_fusion_embedding(source)
    assert len(result["fusion_embedding"]) == 128
    assert result["num_nodes"] > 0
    assert result["num_edges"] > 0
    assert len(result["model_hash"]) == 64


@requires_teacher
@requires_gpu
def test_proxy_on_teacher_fusion(teacher, proxy):
    """Proxy forward on a real teacher fusion embedding returns [1, 10]."""
    sol = Path("manual_hand_written_contracts/Reentrancy/01_cei_violation_erc721.sol")
    if not sol.exists():
        pytest.skip("test contract not found")
    source = sol.read_text()

    result = teacher.predict_fusion_embedding(source)
    features = torch.tensor([result["fusion_embedding"]])

    with torch.no_grad():
        logits = proxy(features)

    assert logits.shape == (1, 10)
    scores = torch.sigmoid(logits)
    for s in scores.squeeze(0):
        assert 0.0 <= s <= 1.0


@requires_teacher
@requires_ezkl_artifacts
@requires_gpu
def test_full_pipeline_witness_generation(teacher, proxy):
    """Full pipeline: teacher fusion → proxy → EZKL witness → verify."""
    import ezkl

    sol = Path("manual_hand_written_contracts/Reentrancy/01_cei_violation_erc721.sol")
    if not sol.exists():
        pytest.skip("test contract not found")
    source = sol.read_text()

    # 1. Teacher → fusion embedding
    result = teacher.predict_fusion_embedding(source)
    features = result["fusion_embedding"]
    assert len(features) == 128

    # 2. Proxy → class logits
    with torch.no_grad():
        logits = proxy(torch.tensor([features]))
    scores = torch.sigmoid(logits).squeeze(0).tolist()
    assert len(scores) == 10

    # 3. EZKL witness
    proof_input_file = Path("/tmp/sentinel_e2e_proof_input.json")
    witness_file = Path("/tmp/sentinel_e2e_witness.json")
    proof_file = Path("/tmp/sentinel_e2e_proof.json")

    proof_input_file.write_text(json.dumps({"input_data": [features]}))

    try:
        witness = ezkl.gen_witness(
            data=str(proof_input_file),
            model="zkml/ezkl/model.compiled",
            output=str(witness_file),
        )

        # 4. Decode output field elements
        outputs = witness["outputs"][0]
        assert len(outputs) == 10, f"Expected 10 class scores, got {len(outputs)}"

        class_score_felts = []
        for hex_str in outputs:
            felt = int.from_bytes(bytes.fromhex(hex_str), byteorder='little')
            class_score_felts.append(felt)

        # 5. Prove
        ezkl.prove(
            witness=str(witness_file),
            model="zkml/ezkl/model.compiled",
            pk_path="zkml/ezkl/proving_key.pk",
            proof_path=str(proof_file),
            srs_path="zkml/ezkl/srs.params",
        )

        # 6. Verify
        valid = ezkl.verify(
            proof_path=str(proof_file),
            settings_path="zkml/ezkl/settings.json",
            vk_path="zkml/ezkl/verification_key.vk",
            srs_path="zkml/ezkl/srs.params",
        )
        assert valid, "Off-chain proof verification failed"

        # 7. Parse proof.json → publicSignals
        proof_data = json.loads(proof_file.read_text())
        instances = proof_data["instances"][0]
        assert len(instances) == 138, f"Expected 138 publicSignals, got {len(instances)}"

        all_signals = [
            int.from_bytes(bytes.fromhex(h), byteorder='little')
            for h in instances
        ]
        # Verify class scores at positions 128..137 match witness outputs
        proof_class_scores = all_signals[128:]
        assert proof_class_scores == class_score_felts, (
            "Proof publicSignals mismatch witness outputs"
        )

    finally:
        for f in (proof_input_file, witness_file, proof_file):
            if f.exists():
                f.unlink()


def test_proxy_output_range(proxy):
    """Proxy logits are in a reasonable range (not NaN, not Inf)."""
    x = torch.randn(8, 128)
    with torch.no_grad():
        logits = proxy(x)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert logits.abs().max() < 100  # logits shouldn't be extreme


def test_num_classes_constant():
    """NUM_CLASSES = 10 matches proxy output dimension."""
    from zkml.src.distillation.proxy_model import ProxyModel
    p = ProxyModel()
    with torch.no_grad():
        out = p(torch.randn(1, 128))
    assert out.shape[1] == 10
