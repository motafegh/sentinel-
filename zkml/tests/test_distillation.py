"""Tests for distillation logic — agreement metric, per-logit target."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Inline compute_agreement to avoid importing train_proxy
# (which pulls in ml.src.datasets → sentinel_data → full ML stack)
THRESHOLD = 0.5


def _compute_agreement(proxy_scores, teacher_scores, threshold=THRESHOLD):
    proxy_labels = (proxy_scores >= threshold).long()
    teacher_labels = (teacher_scores >= threshold).long()
    matches = (proxy_labels == teacher_labels).float()
    return matches.mean().item()


def test_compute_agreement_perfect():
    """Identical scores → 1.0 agreement."""
    proxy = torch.tensor([[0.9, 0.1, 0.8], [0.3, 0.7, 0.2]])
    teacher = torch.tensor([[0.9, 0.1, 0.8], [0.3, 0.7, 0.2]])
    assert _compute_agreement(proxy, teacher) == 1.0


def test_compute_agreement_zero():
    """Completely opposite scores → 0.0 agreement."""
    proxy = torch.tensor([[0.9, 0.1, 0.8], [0.3, 0.7, 0.2]])
    teacher = torch.tensor([[0.1, 0.9, 0.2], [0.7, 0.3, 0.8]])
    assert _compute_agreement(proxy, teacher) == 0.0


def test_compute_agreement_partial():
    """3-of-6 pairs agree → 0.5 agreement."""
    proxy = torch.tensor([[0.9, 0.9, 0.9], [0.4, 0.4, 0.4]])
    teacher = torch.tensor([[0.9, 0.1, 0.1], [0.6, 0.4, 0.4]])
    result = _compute_agreement(proxy, teacher)
    assert result == 0.5


def test_compute_agreement_boundary():
    """Scores exactly at threshold → >= threshold is vulnerable."""
    proxy = torch.tensor([[0.5, 0.49], [0.51, 0.5]])
    teacher = torch.tensor([[0.5, 0.49], [0.5, 0.51]])
    assert _compute_agreement(proxy, teacher) == 1.0


def test_compute_agreement_shape():
    """Agreement works with [B, 10] tensors."""
    B, C = 32, 10
    proxy = torch.rand(B, C)
    teacher = torch.rand(B, C)
    result = _compute_agreement(proxy, teacher, threshold=0.5)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
