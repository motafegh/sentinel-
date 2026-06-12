"""Tests for sentinel_data.export.metadata_writer."""
import json
import pytest
import pyarrow.parquet as pq
from pathlib import Path

from sentinel_data.export.metadata_writer import write_metadata_parquet


def _make_splits(tmp_path: Path, rows: list[dict]) -> Path:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    for r in rows:
        split = r.pop("split", "train")
        buckets[split].append(r)
    for name, items in buckets.items():
        (splits_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(i) for i in items)
        )
    return splits_dir


def _write_sidecars(rep_root: Path, preproc_root: Path, sha: str, source: str,
                    node_count: int = 42, edge_count: int = 60,
                    solc_version: str = "0.8.4",
                    version_bucket: str = "modern",
                    has_unchecked: bool = True,
                    sol_lines: int = 50,
                    n_functions: int = 3) -> None:
    """Write synthetic .rep.json, .meta.json, .sol sidecars."""
    (rep_root / source).mkdir(parents=True, exist_ok=True)
    (preproc_root / source).mkdir(parents=True, exist_ok=True)

    (rep_root / source / f"{sha}.rep.json").write_text(json.dumps({
        "sha256": sha, "source": source, "node_count": node_count,
        "edge_count": edge_count, "solc_version": solc_version,
        "schema_version": "v9", "extractor_version": "v2.1",
    }))
    (preproc_root / source / f"{sha}.meta.json").write_text(json.dumps({
        "sha256": sha, "version_bucket": version_bucket,
        "has_unchecked_block": has_unchecked, "dedup_group_id": "grp-1",
    }))
    # Synthetic .sol with n_functions `function` keywords and sol_lines lines
    func_lines = "\n".join(f"function f{i}() public {{}}" for i in range(n_functions))
    padding = "\n" * max(0, sol_lines - n_functions - 2)
    (preproc_root / source / f"{sha}.sol").write_text(
        f"pragma solidity ^0.8.0;\n{func_lines}{padding}"
    )


def test_metadata_writer_columns(tmp_path):
    sha = "a" * 64
    rows = [{"sha256": sha, "source": "solidifi", "split": "train",
              "n_pos": 0, "tier": "T0", "primary_class": "NonVulnerable",
              "classes": {}, "loc": 0}]
    splits_dir = _make_splits(tmp_path, rows)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    _write_sidecars(rep_root, preproc_root, sha, "solidifi")

    out = tmp_path / "metadata.parquet"
    write_metadata_parquet(splits_dir, rep_root, preproc_root, out)
    table = pq.read_table(out)
    expected = [
        "contract_id", "source", "split", "solc_version", "version_bucket",
        "loc", "n_functions", "n_pos", "primary_class",
        "node_count", "edge_count", "has_unchecked_block", "dedup_group_id",
        "confidence_tier",
    ]
    assert table.schema.names == expected


def test_metadata_writer_loc_computed_from_sol(tmp_path):
    """loc must be computed from .sol, not from the split JSONL's loc=0 (Fix #3)."""
    sha = "b" * 64
    rows = [{"sha256": sha, "source": "solidifi", "split": "train",
              "n_pos": 0, "tier": "T0", "primary_class": "NonVulnerable",
              "classes": {}, "loc": 0}]  # loc=0 in JSONL — must be overridden
    splits_dir = _make_splits(tmp_path, rows)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    _write_sidecars(rep_root, preproc_root, sha, "solidifi", sol_lines=50, n_functions=3)

    out = tmp_path / "metadata.parquet"
    write_metadata_parquet(splits_dir, rep_root, preproc_root, out)
    table = pq.read_table(out)
    loc = table.column("loc")[0].as_py()
    assert loc is not None and loc > 0, f"loc should be computed from .sol, got {loc}"
    n_functions = table.column("n_functions")[0].as_py()
    assert n_functions == 3


def test_metadata_writer_missing_rep_is_null(tmp_path):
    """Contracts missing .rep.json get null node_count/edge_count, not a crash."""
    sha = "c" * 64
    rows = [{"sha256": sha, "source": "solidifi", "split": "train",
              "n_pos": 0, "tier": "T0", "primary_class": "NonVulnerable",
              "classes": {}, "loc": 0}]
    splits_dir = _make_splits(tmp_path, rows)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    # write only meta + sol, no .rep.json
    (preproc_root / "solidifi").mkdir(parents=True, exist_ok=True)
    (rep_root / "solidifi").mkdir(parents=True, exist_ok=True)
    (preproc_root / "solidifi" / f"{sha}.meta.json").write_text(json.dumps({
        "sha256": sha, "version_bucket": "modern",
        "has_unchecked_block": False, "dedup_group_id": None,
    }))
    (preproc_root / "solidifi" / f"{sha}.sol").write_text("pragma solidity ^0.8.0;\n")

    out = tmp_path / "metadata.parquet"
    write_metadata_parquet(splits_dir, rep_root, preproc_root, out)
    table = pq.read_table(out)
    assert table.column("node_count")[0].as_py() is None
    assert table.column("edge_count")[0].as_py() is None


def test_metadata_writer_confidence_tier(tmp_path):
    sha_vuln = "d" * 64
    sha_neg = "e" * 64
    rows = [
        {"sha256": sha_vuln, "source": "solidifi", "split": "train",
         "n_pos": 1, "tier": "T1", "primary_class": "Reentrancy", "classes": {}, "loc": 0},
        {"sha256": sha_neg, "source": "solidifi", "split": "train",
         "n_pos": 0, "tier": "T0", "primary_class": "NonVulnerable", "classes": {}, "loc": 0},
    ]
    splits_dir = _make_splits(tmp_path, rows)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    for sha in (sha_vuln, sha_neg):
        _write_sidecars(rep_root, preproc_root, sha, "solidifi")

    out = tmp_path / "metadata.parquet"
    write_metadata_parquet(splits_dir, rep_root, preproc_root, out)
    table = pq.read_table(out)
    tiers = {r["contract_id"]: r["confidence_tier"]
             for r in table.to_pylist()}
    assert tiers[sha_vuln] == "T1"
    assert tiers[sha_neg] is None
