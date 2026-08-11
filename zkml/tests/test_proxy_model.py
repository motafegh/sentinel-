"""ProxyModel architecture and score-contract tests."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zkml.src.distillation.proxy_model import (
    CIRCUIT_VERSION,
    EZKL_PARAM_LIMIT,
    OUTPUT_SEMANTICS,
    ProxyModel,
)


def test_default_initialisation():
    p = ProxyModel()
    assert p.parameter_count() == 10_666
    assert p.parameter_count() < EZKL_PARAM_LIMIT
    assert p.circuit_version() == CIRCUIT_VERSION
    assert p.output_semantics() == OUTPUT_SEMANTICS
    assert OUTPUT_SEMANTICS == "teacher_probability_regression_v1"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_dim": 64}, "input_dim must be 128"),
        ({"hidden1": 32}, "hidden1 must be 64"),
        ({"hidden2": 16}, "hidden2 must be 32"),
        ({"num_classes": 9}, "num_classes must be 10"),
    ],
)
def test_architecture_freeze_guards(kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        ProxyModel(**kwargs)


def test_forward_shapes_and_determinism():
    p = ProxyModel().eval()
    x = torch.randn(4, 128)
    with torch.no_grad():
        out1 = p(x)
        out2 = p(x)
    assert out1.shape == (4, 10)
    assert torch.equal(out1, out2)


def test_forward_is_direct_student_score_not_sigmoid():
    """A consumer must use forward() directly; a second sigmoid changes V2 semantics."""
    p = ProxyModel().eval()
    # Make the network deterministic and force a known final output of zero.
    with torch.no_grad():
        for param in p.parameters():
            param.zero_()
    x = torch.zeros(1, 128)
    with torch.no_grad():
        score = p(x)
    assert torch.equal(score, torch.zeros(1, 10))
    # A mistaken second sigmoid would turn every circuit score into 0.5.
    assert torch.equal(torch.sigmoid(score), torch.full((1, 10), 0.5))
    assert not torch.equal(score, torch.sigmoid(score))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_forward_cuda():
    p = ProxyModel().to("cuda").eval()
    x = torch.randn(2, 128, device="cuda")
    with torch.no_grad():
        out = p(x)
    assert out.shape == (2, 10)
    assert out.device.type == "cuda"
