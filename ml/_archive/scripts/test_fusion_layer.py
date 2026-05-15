# ml/scripts/test_fusion_layer.py

import torch
import sys
from pathlib import Path

# Make sure Python can find our ml/src modules
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from models.fusion_layer import FusionLayer

def test_fusion_layer():
    """
    Verify FusionLayer output shape with realistic fake inputs.
    We don't need real data — shape testing only needs correct dimensions.
    """
    BATCH_SIZE = 4

    # Fake GNN output — same shape GNNEncoder produces
    gnn_out = torch.randn(BATCH_SIZE, 64)

    # Fake Transformer output — same shape TransformerEncoder produces
    transformer_out = torch.randn(BATCH_SIZE, 768)

    model = FusionLayer()
    output = model(gnn_out, transformer_out)

    print(f"gnn_out shape:         {gnn_out.shape}")
    print(f"transformer_out shape: {transformer_out.shape}")
    print(f"fusion output shape:   {output.shape}")

    # The assertion that matters
    assert output.shape == (BATCH_SIZE, 64), \
        f"Expected [4, 64], got {output.shape}"

    print("\n✅ FusionLayer verified — output [B, 64] confirmed")

if __name__ == "__main__":
    test_fusion_layer()
