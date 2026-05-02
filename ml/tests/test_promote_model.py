"""
test_promote_model.py — Unit tests for promote_model.py (T2-C).

All tests stub out MLflow and torch.load so no real checkpoint or
MLflow server is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Helpers — synthetic checkpoint on disk
# ---------------------------------------------------------------------------

def _write_checkpoint(path: Path) -> None:
    payload = {
        "model":  {},
        "epoch":  7,
        "config": {
            "architecture":    "cross_attention_lora",
            "num_classes":     10,
            "fusion_output_dim": 128,
        },
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ckpt(tmp_path) -> Path:
    p = tmp_path / "test_model.pt"
    _write_checkpoint(p)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cli_rejects_unknown_stage(ckpt, capsys):
    """argparse choices enforcement raises SystemExit for unknown stage."""
    from ml.scripts import promote_model  # noqa: F401 — ensure importable

    with pytest.raises(SystemExit) as exc_info:
        sys.argv = [
            "promote_model.py",
            "--checkpoint", str(ckpt),
            "--stage", "InvalidStage",
            "--val-f1-macro", "0.47",
        ]
        from ml.scripts.promote_model import main
        main()

    assert exc_info.value.code != 0


def test_dry_run_does_not_register(ckpt, capsys):
    """--dry-run prints summary and exits 0 without touching MLflow."""
    with patch("ml.scripts.promote_model.mlflow") as mock_mlflow:
        from ml.scripts.promote_model import promote

        rc = promote(
            checkpoint    = ckpt,
            stage         = "Staging",
            val_f1_macro  = 0.4679,
            note          = "test note",
            experiment_name = "test-exp",
            dry_run       = True,
        )

    assert rc == 0
    mock_mlflow.set_experiment.assert_not_called()
    mock_mlflow.start_run.assert_not_called()
    mock_mlflow.register_model.assert_not_called()

    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out


def test_mlflow_tags_are_written(ckpt):
    """promote() calls set_tags with architecture, git_commit, and val_f1_macro."""
    fake_run_info = SimpleNamespace(run_id="run-abc123")
    fake_run      = SimpleNamespace(info=fake_run_info)
    cm            = MagicMock()
    cm.__enter__  = MagicMock(return_value=fake_run)
    cm.__exit__   = MagicMock(return_value=False)

    fake_mv = SimpleNamespace(version="3")

    with (
        patch("ml.scripts.promote_model.mlflow") as mock_mlflow,
        patch("ml.scripts.promote_model.MlflowClient") as mock_client_cls,
    ):
        mock_mlflow.start_run.return_value = cm
        mock_mlflow.register_model.return_value = fake_mv
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from ml.scripts.promote_model import promote

        rc = promote(
            checkpoint    = ckpt,
            stage         = "Staging",
            val_f1_macro  = 0.4679,
            note          = "integration test",
            experiment_name = "sentinel-retrain-v2",
            dry_run       = False,
        )

    assert rc == 0

    # set_tags must include architecture and git_commit
    call_kwargs = mock_mlflow.set_tags.call_args[0][0]
    assert "architecture" in call_kwargs
    assert "git_commit"   in call_kwargs

    # val_f1_macro logged as metric
    mock_mlflow.log_metric.assert_called_once_with("val_f1_macro", 0.4679)

    # register_model called with model name
    mock_mlflow.register_model.assert_called_once()
    reg_kwargs = mock_mlflow.register_model.call_args
    assert "sentinel-vulnerability-detector" in str(reg_kwargs)
