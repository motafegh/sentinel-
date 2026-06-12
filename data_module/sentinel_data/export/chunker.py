"""Stage 7A — chunker: orchestrates all 4 writers into one export artifact.

The entry point is `chunk_export()`. It runs the 4 writers in order,
computes the artifact_hash over the data files, then writes manifest.json
LAST (Fix A — avoids circular hash).

Output layout:
  <output_dir>/
    labels.parquet
    metadata.parquet
    graphs/
      graphs-{shard:05d}.pt
      _shard_index.json
    tokens/
      tokens-{shard:05d}.pt
      _shard_index.json
    manifest.json   ← written last
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from sentinel_data.labeling.schema import class_names
from sentinel_data.export.label_writer import write_labels_parquet
from sentinel_data.export.metadata_writer import write_metadata_parquet
from sentinel_data.export.graph_writer import write_graphs_shards
from sentinel_data.export.token_writer import write_tokens_shards


SCHEMA_VERSION = "v1"


@dataclass
class SkippedSource:
    name: str
    reason: str


@dataclass
class ExportManifest:
    schema_version: str
    graph_schema_version: str
    artifact_hash: str
    hash_algorithm: str
    shard_size: int
    n_contracts: int
    n_contracts_with_reps: int
    n_shards: int
    splits: dict[str, list[str]]
    shard_index: dict[str, dict]   # {sha: {shard: int, pos_in_shard: int}}
    source_set: list[str]
    skipped_sources: list[dict]
    preprocessing_config_hash: str
    label_class_columns: list[str]
    created_at: str


def _hash_export_data(export_dir: Path) -> str:
    """SHA-256 over the 4 data file types (excludes manifest.json — Fix A).

    File order is sorted by relative path for determinism.
    """
    candidate_files = sorted(
        p for p in export_dir.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )
    h = hashlib.sha256()
    for p in candidate_files:
        rel = str(p.relative_to(export_dir))
        h.update(rel.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _config_hash(config_path: Path) -> str:
    if not config_path.exists():
        return "unknown"
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _load_split_ids(splits_dir: Path) -> tuple[dict[str, list[str]], int]:
    """Return (splits_dict, total_count).  splits_dict = {split: [sha256...]}."""
    splits: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Split JSONL not found: {path}")
        ids = []
        for line in path.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                ids.append(c["sha256"])
        splits[name] = ids
    total = sum(len(v) for v in splits.values())
    return splits, total


def _build_shard_index(
    shard_index_from_writer: dict[str, int],
    rep_root: Path,
    splits_dir: Path,
) -> dict[str, dict]:
    """Convert {sha: shard_num} → {sha: {shard, pos_in_shard}}."""
    # Rebuild per-shard position ordering from the canonical JSONL order.
    shard_positions: dict[int, int] = {}  # {shard_num: next_pos}
    full_index: dict[str, dict] = {}
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            sha = c["sha256"]
            if sha not in shard_index_from_writer:
                continue  # no rep
            shard_num = shard_index_from_writer[sha]
            pos = shard_positions.get(shard_num, 0)
            full_index[sha] = {"shard": shard_num, "pos_in_shard": pos}
            shard_positions[shard_num] = pos + 1
    return full_index


def chunk_export(
    rep_root: Path,
    preproc_root: Path,
    splits_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    shard_size: int = 5000,
    source_set: list[str] | None = None,
    skipped_sources: list[dict] | None = None,
    graph_schema_version: str = "v9",
) -> ExportManifest:
    """Orchestrate all 4 writers; write manifest.json last (Fix A).

    Args:
        rep_root:    data/representations/ (per-source subdirs)
        preproc_root: data/preprocessed/ (per-source subdirs)
        splits_dir:  data/splits/v{N}/
        output_dir:  where to write the export artifact
        config_path: path to config.yaml (for preprocessing_config_hash)
        shard_size:  contracts per shard (default 5000)
        source_set:  list of source names actually exported (for manifest)
        skipped_sources: sources enabled but skipped (for manifest)
        graph_schema_version: the v9 schema version pin (default "v9")

    Returns:
        ExportManifest (also written to output_dir/manifest.json)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    tokens_dir = output_dir / "tokens"

    # ── 1. labels.parquet ────────────────────────────────────────────────
    write_labels_parquet(splits_dir, output_dir / "labels.parquet")

    # ── 2. metadata.parquet ──────────────────────────────────────────────
    write_metadata_parquet(splits_dir, rep_root, preproc_root, output_dir / "metadata.parquet")

    # ── 3. graph shards ──────────────────────────────────────────────────
    _, graph_shard_map = write_graphs_shards(rep_root, splits_dir, graphs_dir, shard_size)

    # ── 4. token shards ──────────────────────────────────────────────────
    _, token_shard_map = write_tokens_shards(rep_root, splits_dir, tokens_dir, shard_size)

    # ── 5. shard index files ─────────────────────────────────────────────
    full_shard_index = _build_shard_index(graph_shard_map, rep_root, splits_dir)
    shard_index_json = json.dumps(full_shard_index, sort_keys=True)
    (graphs_dir / "_shard_index.json").write_text(shard_index_json)
    (tokens_dir / "_shard_index.json").write_text(shard_index_json)

    # ── 6. compute artifact_hash (Fix A — before manifest.json exists) ──
    artifact_hash = _hash_export_data(output_dir)

    # ── 7. build manifest ────────────────────────────────────────────────
    splits_dict, n_contracts = _load_split_ids(splits_dir)
    n_with_reps = len(full_shard_index)
    n_shards = len({v for v in graph_shard_map.values()}) if graph_shard_map else 0

    manifest = ExportManifest(
        schema_version=SCHEMA_VERSION,
        graph_schema_version=graph_schema_version,
        artifact_hash=artifact_hash,
        hash_algorithm="sha256",
        shard_size=shard_size,
        n_contracts=n_contracts,
        n_contracts_with_reps=n_with_reps,
        n_shards=n_shards,
        splits=splits_dict,
        shard_index=full_shard_index,
        source_set=source_set or [],
        skipped_sources=skipped_sources or [],
        preprocessing_config_hash=_config_hash(config_path) if config_path else "unknown",
        label_class_columns=class_names(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # ── 8. write manifest.json LAST (Fix A) ──────────────────────────────
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True)
    )

    return manifest


__all__ = ["ExportManifest", "chunk_export", "_hash_export_data"]
