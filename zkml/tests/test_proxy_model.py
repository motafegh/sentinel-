"""Tests for ProxyModel — architecture freeze guards, forward shapes, ONNX export."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zkml.src.distillation.proxy_model import CIRCUIT_VERSION, EZKL_PARAM_LIMIT, ProxyModel


def test_default_initialisation():
    """ProxyModel(defaults) must succeed with expected architecture."""
    p = ProxyModel()
    assert p.parameter_count() > 8000
    assert p.parameter_count() < EZKL_PARAM_LIMIT
    assert p.circuit_version() == CIRCUIT_VERSION


def test_freeze_guard_input_dim():
    """ProxyModel(input_dim=64) must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="input_dim must be 128"):
        ProxyModel(input_dim=64)


def test_freeze_guard_hidden1():
    """ProxyModel(hidden1=32) must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="hidden1 must be 64"):
        ProxyModel(hidden1=32)


def test_freeze_guard_hidden2():
    """ProxyModel(hidden2=16) must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="hidden2 must be 32"):
        ProxyModel(hidden2=16)


def test_freeze_guard_num_classes():
    """ProxyModel(num_classes=9) must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="num_classes must be 10"):
        ProxyModel(num_classes=9)


def test_forward_shape_single():
    """Forward pass: [1, 128] → [1, 10]."""
    p = ProxyModel()
    p.eval()
    x = torch.randn(1, 128)
    with torch.no_grad():
        out = p(x)
    assert out.shape == (1, 10)


def test_forward_shape_batch():
    """Forward pass: [4, 128] → [4, 10]."""
    p = ProxyModel()
    p.eval()
    x = torch.randn(4, 128)
    with torch.no_grad():
        out = p(x)
    assert out.shape == (4, 10)


def test_forward_deterministic():
    """Same input → same output (no dropout)."""
    p = ProxyModel()
    p.eval()
    x = torch.randn(1, 128)
    with torch.no_grad():
        out1 = p(x)
        out2 = p(x)
    assert torch.allclose(out1, out2)


def test_parameter_count_exact():
    """Architecture Linear(128→64→32→10) = exactly 10,666 params."""
    p = ProxyModel()
    assert p.parameter_count() == 10_666


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_forward_cuda():
    """Forward pass works on GPU."""
    p = ProxyModel().to("cuda")
    p.eval()
    x = torch.randn(2, 128).to("cuda")
    with torch.no_grad():
        out = p(x)
    assert out.shape == (2, 10)
    assert out.device.type == "cuda"
