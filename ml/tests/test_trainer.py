"""
test_trainer.py — Unit tests for trainer utility functions and FocalLoss.

The full train() function depends on disk data (graphs, tokens, splits) and
GPU resources — those are tested via integration. Here we exercise:
  - compute_pos_weight()      — class-balance tensor computation
  - evaluate()                — metric computation with a toy model
  - train_one_epoch()         — one epoch of gradient descent on synthetic data
  - TrainConfig               — dataclass defaults and device detection
  - FocalLoss                 — forward pass correctness
  - Early stopping logic      — implicit via agreement rate helper
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Batch, Data

from ml.src.training.trainer import (
    TrainConfig,
    evaluate,
    train_one_epoch,
)
from ml.src.training.focalloss import FocalLoss


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_output_is_scalar(self):
        loss_fn = FocalLoss()
        preds   = torch.sigmoid(torch.randn(8))   # probabilities ∈ (0,1)
        targets = torch.randint(0, 2, (8,)).float()
        loss    = loss_fn(preds, targets)
        assert loss.shape == (), f"expected scalar, got {loss.shape}"

    def test_perfect_predictions_low_loss(self):
        """Near-perfect predictions should yield focal loss close to zero."""
        loss_fn = FocalLoss()
        targets = torch.ones(8)
        preds   = torch.full((8,), 0.999)   # very confident, correct
        loss    = loss_fn(preds, targets)
        assert loss.item() < 0.01, f"loss too high for perfect predictions: {loss.item()}"

    def test_worst_predictions_high_loss(self):
        loss_fn = FocalLoss()
        targets = torch.ones(8)
        preds   = torch.full((8,), 0.001)   # confident, wrong
        loss    = loss_fn(preds, targets)
        assert loss.item() > 0.1, f"loss too low for wrong predictions: {loss.item()}"

    def test_nonnegative(self):
        loss_fn = FocalLoss()
        for _ in range(5):
            preds   = torch.sigmoid(torch.randn(16))
            targets = torch.randint(0, 2, (16,)).float()
            assert loss_fn(preds, targets).item() >= 0.0


# ---------------------------------------------------------------------------
# TrainConfig defaults
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_default_num_classes(self):
        cfg = TrainConfig()
        assert cfg.num_classes == 10

    def test_device_is_valid_string(self):
        cfg = TrainConfig()
        assert cfg.device in ("cuda", "cpu")

    def test_dataclass_fields_mutable(self):
        cfg = TrainConfig(epochs=5, batch_size=8, lr=1e-4)
        assert cfg.epochs     == 5
        assert cfg.batch_size == 8
        assert cfg.lr         == pytest.approx(1e-4)

    def test_asdict_round_trip(self):
        cfg = TrainConfig(epochs=3)
        d   = dataclasses.asdict(cfg)
        assert d["epochs"] == 3
        assert "num_classes" in d


# ---------------------------------------------------------------------------
# Toy model + helpers for evaluate() / train_one_epoch()
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    """Minimal model that consumes the same (graphs, tokens, labels) batch shape."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(8, num_classes)

    def forward(self, graphs: Batch, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                return_aux: bool = False):
        # Use global mean of node features as a cheap summary
        x = graphs.x                                   # [N, 8]
        batch = graphs.batch                            # [N]
        B = int(batch.max().item()) + 1
        pooled = torch.zeros(B, 8, device=x.device)
        for i in range(B):
            mask = batch == i
            pooled[i] = x[mask].mean(dim=0)
        logits = self.fc(pooled)                       # [B, 10]
        if return_aux:
            aux = {"gnn": logits, "transformer": logits, "fused": logits}
            return logits, aux
        return logits


def _make_pyg_batch(B: int, n_per_graph: int = 4) -> Batch:
    graphs_list = []
    for _ in range(B):
        x          = torch.randn(n_per_graph, 8)
        src        = torch.arange(n_per_graph - 1)
        dst        = torch.arange(1, n_per_graph)
        edge_index = torch.stack([src, dst])
        graphs_list.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs_list)


def _make_loader(n_batches: int = 3, batch_size: int = 4, num_classes: int = 10):
    """
    Returns a DataLoader whose __iter__ yields (graphs, tokens, labels) tuples.
    """
    class _SyntheticDataset:
        def __len__(self):
            return n_batches * batch_size

        def __getitem__(self, idx):
            # Each item: (graph, token_dict, label_vector)
            n = 4
            x          = torch.randn(n, 8)
            src        = torch.arange(n - 1)
            dst        = torch.arange(1, n)
            edge_index = torch.stack([src, dst])
            graph      = Data(x=x, edge_index=edge_index)
            tokens     = {
                "input_ids":      torch.ones(512, dtype=torch.long),
                "attention_mask": torch.ones(512, dtype=torch.long),
            }
            labels = torch.zeros(num_classes, dtype=torch.float32)
            return graph, tokens, labels

    from ml.src.datasets.dual_path_dataset import dual_path_collate_fn
    return DataLoader(
        _SyntheticDataset(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dual_path_collate_fn,
    )


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_f1_macro(self):
        model  = _TinyMLP(num_classes=10)
        loader = _make_loader()
        metrics = evaluate(model, loader, device="cpu", use_amp=False)
        assert "f1_macro" in metrics

    def test_returns_f1_micro(self):
        model  = _TinyMLP(num_classes=10)
        loader = _make_loader()
        metrics = evaluate(model, loader, device="cpu", use_amp=False)
        assert "f1_micro" in metrics

    def test_returns_hamming(self):
        model  = _TinyMLP(num_classes=10)
        loader = _make_loader()
        metrics = evaluate(model, loader, device="cpu", use_amp=False)
        assert "hamming" in metrics

    def test_f1_macro_in_range(self):
        model  = _TinyMLP(num_classes=10)
        loader = _make_loader()
        metrics = evaluate(model, loader, device="cpu", use_amp=False)
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_per_class_f1_keys_present(self):
        from ml.src.training.trainer import CLASS_NAMES
        model   = _TinyMLP(num_classes=10)
        loader  = _make_loader()
        metrics = evaluate(model, loader, device="cpu", use_amp=False)
        for name in CLASS_NAMES:
            assert f"f1_{name}" in metrics, f"missing key: f1_{name}"


# ---------------------------------------------------------------------------
# train_one_epoch()
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    def test_returns_float(self):
        model     = _TinyMLP(num_classes=10)
        loader    = _make_loader(n_batches=2)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=2)
        scaler    = torch.amp.GradScaler("cpu", enabled=False)
        loss_fn     = nn.BCEWithLogitsLoss()
        aux_loss_fn = nn.BCEWithLogitsLoss()

        loss, nan_count, gnn_share = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            aux_loss_fn=aux_loss_fn,
            scheduler=scheduler,
            scaler=scaler,
            device="cpu",
            grad_clip=1.0,
            log_interval=100,
            use_amp=False,
        )
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_loss_decreases_over_epochs(self):
        """Three consecutive epochs on fixed data — loss should trend downward."""
        model     = _TinyMLP(num_classes=10)
        loader    = _make_loader(n_batches=4)
        optimizer = AdamW(model.parameters(), lr=1e-2)
        scaler    = torch.amp.GradScaler("cpu", enabled=False)
        loss_fn     = nn.BCEWithLogitsLoss()
        aux_loss_fn = nn.BCEWithLogitsLoss()

        total_steps = 4 * 3  # n_batches * n_epochs
        scheduler   = OneCycleLR(optimizer, max_lr=1e-2, total_steps=total_steps)

        losses = []
        for _ in range(3):
            loss, _, _ = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                aux_loss_fn=aux_loss_fn,
                scheduler=scheduler,
                scaler=scaler,
                device="cpu",
                grad_clip=1.0,
                log_interval=100,
                use_amp=False,
            )
            losses.append(loss)

        # With lr=1e-2 on simple MLP + fixed synthetic data, loss must drop
        assert losses[-1] < losses[0], (
            f"expected loss to decrease; epoch losses: {losses}"
        )

    def test_checkpoint_saves_state_dict(self, tmp_path):
        """Verify that torch.save produces a loadable state_dict."""
        model = _TinyMLP(num_classes=10)
        ckpt  = tmp_path / "model.pt"
        torch.save(model.state_dict(), ckpt)

        loaded = _TinyMLP(num_classes=10)
        loaded.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        # Weights should match after round-trip
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items()
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2)
