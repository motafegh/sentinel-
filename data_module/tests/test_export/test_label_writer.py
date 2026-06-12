"""Tests for sentinel_data.export.label_writer."""
import json
import pytest
import pyarrow.parquet as pq
from pathlib import Path

from sentinel_data.export.label_writer import write_labels_parquet
from sentinel_data.labeling.schema import class_names


def _make_splits(tmp_path: Path, rows: list[dict]) -> Path:
    """Write synthetic split JSONL files. Rows must have a 'split' key."""
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


def _synthetic_row(sha: str, split: str = "train", n_pos: int = 0,
                   tier: str = "T0", classes: dict | None = None) -> dict:
    names = class_names()
    c = classes or {n: 0 for n in names}
    return {
        "sha256": sha,
        "source": "solidifi",
        "split": split,
        "n_pos": n_pos,
        "tier": tier,
        "classes": c,
        "primary_class": "NonVulnerable" if n_pos == 0 else next(k for k, v in c.items() if v == 1),
        "loc": 0,
    }


def test_label_writer_column_names(tmp_path):
    rows = [_synthetic_row(f"a{i:063d}", "train") for i in range(5)]
    splits_dir = _make_splits(tmp_path, rows)
    out = tmp_path / "labels.parquet"
    write_labels_parquet(splits_dir, out)
    table = pq.read_table(out)
    expected = ["contract_id", "source", "split"] + [f"class_{i}" for i in range(10)] + ["confidence_tier"]
    assert table.schema.names == expected


def test_label_writer_row_count(tmp_path):
    rows = (
        [_synthetic_row(f"t{i:063d}", "train") for i in range(10)] +
        [_synthetic_row(f"v{i:063d}", "val") for i in range(3)] +
        [_synthetic_row(f"e{i:063d}", "test") for i in range(2)]
    )
    splits_dir = _make_splits(tmp_path, rows)
    out = tmp_path / "labels.parquet"
    write_labels_parquet(splits_dir, out)
    table = pq.read_table(out)
    assert len(table) == 15


def test_label_writer_confidence_tier_nonvuln_is_null(tmp_path):
    """NonVulnerable contracts (n_pos=0) must have confidence_tier=None (Fix #2)."""
    names = class_names()
    vuln_classes = {n: 0 for n in names}
    vuln_classes["Reentrancy"] = 1
    rows = [
        _synthetic_row("0" * 64, "train", n_pos=0, tier="T0"),          # NonVuln → None
        _synthetic_row("1" * 64, "train", n_pos=1, tier="T2", classes=vuln_classes),  # vuln → "T2"
    ]
    splits_dir = _make_splits(tmp_path, rows)
    out = tmp_path / "labels.parquet"
    write_labels_parquet(splits_dir, out)

    import pyarrow as pa
    table = pq.read_table(out)
    tiers = table.column("confidence_tier").to_pylist()
    assert tiers[0] is None, "NonVulnerable contract should have confidence_tier=None"
    assert tiers[1] == "T2", "Vulnerable contract should carry its tier"


def test_label_writer_class_values(tmp_path):
    names = class_names()
    vuln_classes = {n: 0 for n in names}
    vuln_classes["Reentrancy"] = 1  # class index 6
    rows = [_synthetic_row("a" * 64, "train", n_pos=1, tier="T1", classes=vuln_classes)]
    splits_dir = _make_splits(tmp_path, rows)
    out = tmp_path / "labels.parquet"
    write_labels_parquet(splits_dir, out)
    table = pq.read_table(out)
    row = {c: table.column(c)[0].as_py() for c in table.schema.names}
    assert row["class_6"] == 1   # Reentrancy is column 6
    assert row["class_0"] == 0   # CallToUnknown is 0


def test_label_writer_missing_split_raises(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "train.jsonl").write_text("")
    # val.jsonl missing
    with pytest.raises(FileNotFoundError, match="val.jsonl"):
        write_labels_parquet(splits_dir, tmp_path / "labels.parquet")
