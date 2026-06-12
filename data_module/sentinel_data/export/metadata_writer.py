"""Stage 7A — metadata_writer: writes metadata.parquet (per-contract enrichment).

Per format_schema/v1.yaml: one row per labeled contract, enriched from
the .rep.json + .meta.json + .sol sidecars. 14 columns:
  - contract_id, source, split, solc_version, version_bucket, loc,
    n_functions, n_pos, primary_class, node_count, edge_count,
    has_unchecked_block, dedup_group_id, confidence_tier

Per 7A Fix #3: `loc` and `n_functions` are COMPUTED from .sol source
via _loc() and _function_count() from feature_dist.py — the split
JSONL's loc=0 is a default and is NOT the real LoC.

Reads the split JSONL for the base fields (sha256, source, primary_class,
n_pos, split, tier), then enriches each row with the sidecar data.
Missing sidecars → nulls for those fields (per-contract graceful skip,
not crash).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sentinel_data.analysis.feature_dist import _function_count, _loc


def _build_schema() -> pa.Schema:
    return pa.schema([
        pa.field("contract_id",       pa.string(),  nullable=False),
        pa.field("source",            pa.string(),  nullable=False),
        pa.field("split",             pa.string(),  nullable=False),
        pa.field("solc_version",      pa.string(),  nullable=True),
        pa.field("version_bucket",    pa.string(),  nullable=True),
        pa.field("loc",               pa.int32(),   nullable=True),
        pa.field("n_functions",       pa.int32(),   nullable=True),
        pa.field("n_pos",             pa.int8(),    nullable=False),
        pa.field("primary_class",     pa.string(),  nullable=False),
        pa.field("node_count",        pa.int32(),   nullable=True),
        pa.field("edge_count",        pa.int32(),   nullable=True),
        pa.field("has_unchecked_block", pa.bool_(), nullable=True),
        pa.field("dedup_group_id",    pa.string(),  nullable=True),
        pa.field("confidence_tier",   pa.string(),  nullable=True),
    ])


@dataclass
class MetadataParquetRow:
    """One row of the metadata.parquet table. Field order matches the schema."""
    contract_id: str
    source: str
    split: str
    solc_version: str | None
    version_bucket: str | None
    loc: int | None
    n_functions: int | None
    n_pos: int
    primary_class: str
    node_count: int | None
    edge_count: int | None
    has_unchecked_block: bool | None
    dedup_group_id: str | None
    confidence_tier: str | None


def _load_split_jsonl(splits_dir: Path) -> list[tuple[str, dict]]:
    """Same as label_writer._load_split_jsonl; duplicated to keep modules decoupled."""
    rows: list[tuple[str, dict]] = []
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Split JSONL not found: {path}")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append((split_name, json.loads(line)))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSONL in {path}: {e}") from e
    return rows


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None on any error. Per-contract graceful skip."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_sol_text(path: Path) -> str | None:
    """Read a .sol file, returning None on any error. Per-contract graceful skip."""
    if not path.exists():
        return None
    try:
        return path.read_text()
    except OSError:
        return None


def _enrich_row(
    split: str, contract: dict, rep_root: Path, preproc_root: Path,
) -> MetadataParquetRow:
    """Build one metadata row by joining the split row with the sidecars.

    The source determines which subdir to look under; a contract from
    "dive" looks at rep_root/dive/, preproc_root/dive/. Missing sidecars
    are nulls (no crash).
    """
    sha = contract["sha256"]
    source = contract.get("source", "unknown")
    n_pos = int(contract.get("n_pos") or 0)
    tier = contract.get("tier")
    confidence_tier: str | None = tier if n_pos > 0 else None

    # .rep.json — node_count, edge_count, solc_version
    rep = _read_json(rep_root / source / f"{sha}.rep.json")
    node_count: int | None = int(rep["node_count"]) if rep and rep.get("node_count") is not None else None
    edge_count: int | None = int(rep["edge_count"]) if rep and rep.get("edge_count") is not None else None
    solc_version: str | None = rep.get("solc_version") if rep else None

    # .meta.json — version_bucket, has_unchecked_block, dedup_group_id
    meta = _read_json(preproc_root / source / f"{sha}.meta.json")
    version_bucket: str | None = meta.get("version_bucket") if meta else None
    has_unchecked_block: bool | None = (
        bool(meta.get("has_unchecked_block")) if meta and "has_unchecked_block" in meta else None
    )
    dedup_group_id: str | None = meta.get("dedup_group_id") if meta else None

    # .sol — loc and n_functions (per 7A Fix #3: computed here, not from the split JSONL)
    sol_text = _read_sol_text(preproc_root / source / f"{sha}.sol")
    loc: int | None = _loc(sol_text) if sol_text is not None else None
    n_functions: int | None = _function_count(sol_text) if sol_text is not None else None

    return MetadataParquetRow(
        contract_id=sha,
        source=source,
        split=split,
        solc_version=solc_version,
        version_bucket=version_bucket,
        loc=loc,
        n_functions=n_functions,
        n_pos=n_pos,
        primary_class=contract.get("primary_class", "NonVulnerable"),
        node_count=node_count,
        edge_count=edge_count,
        has_unchecked_block=has_unchecked_block,
        dedup_group_id=dedup_group_id,
        confidence_tier=confidence_tier,
    )


def write_metadata_parquet(
    splits_dir: Path, rep_root: Path, preproc_root: Path, output_path: Path,
) -> Path:
    """Build the metadata.parquet from the split JSONL + per-contract sidecars.

    Args:
        splits_dir: Path to `data/splits/v{N}/` (same as label_writer).
        rep_root: Path to `data/representations/` (the .rep.json sidecars live here).
        preproc_root: Path to `data/preprocessed/` (the .sol + .meta.json sidecars).
        output_path: Where to write `metadata.parquet` (snappy-compressed).

    Returns:
        The output_path (for chaining).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_rows = _load_split_jsonl(splits_dir)
    rows = [_enrich_row(split, c, rep_root, preproc_root) for split, c in split_rows]

    schema = _build_schema()
    arrays: list[pa.Array] = [
        pa.array([r.contract_id for r in rows], type=pa.string()),
        pa.array([r.source for r in rows], type=pa.string()),
        pa.array([r.split for r in rows], type=pa.string()),
        pa.array([r.solc_version for r in rows], type=pa.string()),
        pa.array([r.version_bucket for r in rows], type=pa.string()),
        pa.array([r.loc for r in rows], type=pa.int32()),
        pa.array([r.n_functions for r in rows], type=pa.int32()),
        pa.array([r.n_pos for r in rows], type=pa.int8()),
        pa.array([r.primary_class for r in rows], type=pa.string()),
        pa.array([r.node_count for r in rows], type=pa.int32()),
        pa.array([r.edge_count for r in rows], type=pa.int32()),
        pa.array([r.has_unchecked_block for r in rows], type=pa.bool_()),
        pa.array([r.dedup_group_id for r in rows], type=pa.string()),
        pa.array([r.confidence_tier for r in rows], type=pa.string()),
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, output_path, compression="snappy")
    return output_path


__all__ = [
    "MetadataParquetRow",
    "write_metadata_parquet",
]
