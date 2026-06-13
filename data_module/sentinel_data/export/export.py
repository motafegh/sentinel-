"""Stage 7A — SentinelDatasetExport: consumer-facing API for the export artifact.

This class wraps an export directory produced by `chunk_export()` and
provides:
  - manifest loading + verification
  - artifact hash verification (verify_artifact_hash)
  - split-aware contract ID lookup

The ML-side `SentinelDataset` (built in Stage 7B) wraps this class to
implement __len__ and __getitem__ for PyTorch training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sentinel_data.export.chunker import ExportManifest, _hash_export_data, _write_hash_cache, _HASH_EXCLUDED


class SentinelDatasetExport:
    """Read-only view of a `chunk_export()` output directory.

    Args:
        export_dir: Path to the directory written by `chunk_export()`.

    Raises:
        FileNotFoundError: if `export_dir/manifest.json` is missing.
        ValueError: if the manifest is missing required fields.
    """

    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)
        self._manifest_raw: dict = self._load_manifest()
        self.manifest: ExportManifest = self._parse_manifest(self._manifest_raw)

    # ── paths ──────────────────────────────────────────────────────────────

    @property
    def graphs_dir(self) -> Path:
        return self.export_dir / "graphs"

    @property
    def tokens_dir(self) -> Path:
        return self.export_dir / "tokens"

    @property
    def labels_path(self) -> Path:
        return self.export_dir / "labels.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.export_dir / "metadata.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.export_dir / "manifest.json"

    @property
    def shard_index(self) -> dict[str, dict]:
        return self.manifest.shard_index

    # ── verification ───────────────────────────────────────────────────────

    def verify_artifact_hash(self) -> bool:
        """Compare artifact hash to manifest.artifact_hash.

        Warm path (cache hit): stats each data file, compares mtime+size to
        .hash_cache.json. If all match, trusts the cached hash — no disk reads
        of shard data. Returns in milliseconds.

        Cold path (cache miss or stale): falls back to full SHA-256 over all
        data files, then writes a fresh .hash_cache.json for the next call.

        Returns True if hashes match; False if any data file was tampered with.
        Manifest.json and .hash_cache.json are excluded from the hash (Fix A).
        """
        cache_path = self.export_dir / ".hash_cache.json"
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
                cached_hash = cache.get("artifact_hash", "")
                cached_files: dict = cache.get("files", {})
                all_match = True
                for p in self.export_dir.rglob("*"):
                    if not p.is_file() or p.name in _HASH_EXCLUDED:
                        continue
                    rel = str(p.relative_to(self.export_dir))
                    entry = cached_files.get(rel)
                    if entry is None:
                        all_match = False
                        break
                    stat = p.stat()
                    if stat.st_mtime != entry["mtime"] or stat.st_size != entry["size"]:
                        all_match = False
                        break
                if all_match and cached_hash:
                    return cached_hash == self.manifest.artifact_hash
            except Exception:
                pass  # corrupt cache — fall through to full recompute

        actual = _hash_export_data(self.export_dir)
        _write_hash_cache(self.export_dir, actual)
        return actual == self.manifest.artifact_hash

    # ── split helpers ──────────────────────────────────────────────────────

    def get_split_contract_ids(self, split: str) -> list[str]:
        """Return the ordered list of contract sha256s for a split.

        Args:
            split: "train", "val", or "test"

        Raises:
            KeyError: if split is not in the manifest.
        """
        if split not in self.manifest.splits:
            raise KeyError(f"Split '{split}' not in manifest. Available: {list(self.manifest.splits)}")
        return self.manifest.splits[split]

    # ── internals ──────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        path = self.export_dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self.export_dir}")
        return json.loads(path.read_text())

    @staticmethod
    def _parse_manifest(raw: dict) -> ExportManifest:
        required = (
            "schema_version", "graph_schema_version", "artifact_hash",
            "hash_algorithm", "shard_size", "n_contracts",
            "n_contracts_with_reps", "n_shards", "splits", "shard_index",
            "source_set", "skipped_sources", "preprocessing_config_hash",
            "label_class_columns", "created_at",
        )
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"manifest.json missing fields: {missing}")
        return ExportManifest(
            schema_version=raw["schema_version"],
            graph_schema_version=raw["graph_schema_version"],
            artifact_hash=raw["artifact_hash"],
            hash_algorithm=raw["hash_algorithm"],
            shard_size=raw["shard_size"],
            n_contracts=raw["n_contracts"],
            n_contracts_with_reps=raw["n_contracts_with_reps"],
            n_shards=raw["n_shards"],
            splits=raw["splits"],
            shard_index=raw["shard_index"],
            source_set=raw["source_set"],
            skipped_sources=raw["skipped_sources"],
            preprocessing_config_hash=raw["preprocessing_config_hash"],
            label_class_columns=raw["label_class_columns"],
            created_at=raw["created_at"],
        )

    def __repr__(self) -> str:
        m = self.manifest
        return (
            f"SentinelDatasetExport("
            f"n_contracts={m.n_contracts}, "
            f"n_with_reps={m.n_contracts_with_reps}, "
            f"n_shards={m.n_shards}, "
            f"schema_version={m.schema_version!r}, "
            f"graph_schema_version={m.graph_schema_version!r}, "
            f"sources={m.source_set}"
            f")"
        )


__all__ = ["SentinelDatasetExport"]
