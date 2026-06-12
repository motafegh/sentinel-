"""Stage 7A — token_writer: writes sharded token .pt files.

Per format_schema/v1.yaml: each shard is a torch.Tensor of shape
[N, 4, 512] (N contracts in the shard, 4 windows, 512 tokens each),
written as `tokens-{shard:05d}.pt`.

The contract order mirrors graph_writer exactly (split JSONL order:
train → val → test). Contracts with no .tokens.pt are skipped.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _load_split_jsonl(splits_dir: Path) -> list[tuple[str, str, str]]:
    """Return ordered [(sha256, source, split)] from all 3 JSONL files."""
    rows: list[tuple[str, str, str]] = []
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Split JSONL not found: {path}")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            rows.append((c["sha256"], c.get("source", "unknown"), split_name))
    return rows


def write_tokens_shards(
    rep_root: Path,
    splits_dir: Path,
    output_dir: Path,
    shard_size: int = 5000,
) -> tuple[list[Path], dict[str, int]]:
    """Write sharded token .pt files ([N, 4, 512] per shard).

    Returns:
        (shard_paths, shard_index) — shard_index maps sha256 → shard number.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts = _load_split_jsonl(splits_dir)

    shard_index: dict[str, int] = {}
    shard_paths: list[Path] = []

    current_tensors: list[torch.Tensor] = []
    current_ids: list[str] = []
    shard_num = 0
    skipped = 0

    def _flush() -> None:
        nonlocal shard_num
        if not current_tensors:
            return
        batch = torch.stack(current_tensors, dim=0)  # [N, 4, 512]
        path = output_dir / f"tokens-{shard_num:05d}.pt"
        torch.save(batch, path)
        shard_paths.append(path)
        for sha in current_ids:
            shard_index[sha] = shard_num
        shard_num += 1
        current_tensors.clear()
        current_ids.clear()

    for sha, source, _ in contracts:
        tok_path = rep_root / source / f"{sha}.tokens.pt"
        if not tok_path.exists():
            skipped += 1
            continue
        tok: torch.Tensor = torch.load(tok_path, weights_only=True)
        current_tensors.append(tok)
        current_ids.append(sha)
        if len(current_tensors) >= shard_size:
            _flush()

    _flush()  # final partial shard

    if skipped:
        logger.warning("token_writer: skipped %d contracts with no .tokens.pt file", skipped)

    return shard_paths, shard_index


__all__ = ["write_tokens_shards"]
