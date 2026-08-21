"""Checkpoint edge-vocabulary compatibility tests."""

import pytest

from ml.src.inference.predictor import _requires_legacy_edge_embedding_resize


def test_v10_checkpoint_edge_vocabulary_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="automatic resizing is forbidden"):
        _requires_legacy_edge_embedding_resize(
            graph_schema_version="v10",
            checkpoint_rows=12,
            expected_rows=17,
        )


def test_v9_historical_checkpoint_resize_path_is_preserved() -> None:
    assert _requires_legacy_edge_embedding_resize(
        graph_schema_version="v9",
        checkpoint_rows=11,
        expected_rows=12,
    )
    assert not _requires_legacy_edge_embedding_resize(
        graph_schema_version="v9",
        checkpoint_rows=12,
        expected_rows=12,
    )
