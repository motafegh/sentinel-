"""Stage 7A — label_writer: writes labels.parquet from the split JSONL.

Per format_schema/v1.yaml: one row per labeled contract (all 22,356
labeled contracts appear here, even those without representations).
14 columns total:
  - contract_id (str), source (str), split (str)
  - class_0..class_9 (int8) in locked taxonomy order
  - confidence_tier (str, nullable)

Per 7A Fix #1: reads from the split JSONL only (no re-read of merged
label files). The split JSONL is the canonical, deterministic input.
Per 7A Fix #2: confidence_tier is the split's tier field when n_pos > 0,
else None (pyarrow null) — the splitter defaults tier='T0' for
NonVulnerable contracts; we override to None at the parquet level so
the consumer can filter on null vs. tier values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sentinel_data.labeling.schema import class_names


# Field order is locked at the class_names() order (which is the
# taxonomy.yaml order). We pin it here so the parquet column order is
# deterministic across runs and Python versions.
CLASS_FIELD_ORDER: tuple[str, ...] = tuple(f"class_{i}" for i in range(len(class_names())))


@dataclass
class LabelsParquetRow:
    """One row of the labels.parquet table. Field order matches the schema."""
    contract_id: str
    source: str
    split: str
    class_0: int
    class_1: int
    class_2: int
    class_3: int
    class_4: int
    class_5: int
    class_6: int
    class_7: int
    class_8: int
    class_9: int
    confidence_tier: str | None


def _build_schema() -> pa.Schema:
    """The pyarrow schema for labels.parquet. Field order is locked."""
    return pa.schema([
        pa.field("contract_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        *(pa.field(f"class_{i}", pa.int8(), nullable=False) for i in range(len(class_names()))),
        pa.field("confidence_tier", pa.string(), nullable=True),
    ])


def _load_split_jsonl(splits_dir: Path) -> list[tuple[str, dict]]:
    """Read all 3 split JSONL files. Returns a list of (split_name, contract_dict)."""
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


def _contract_to_row(split: str, contract: dict, names: list[str]) -> LabelsParquetRow:
    """Convert one split row to a LabelsParquetRow.

    The split JSONL stores the `classes` field as a dict from class name
    to 0/1. We project it into the 10-column class_0..class_9 layout in
    the locked names order.
    """
    classes: dict[str, int] = contract.get("classes") or {}
    n_pos = int(contract.get("n_pos") or 0)
    tier = contract.get("tier")

    # Per 7A Fix #2: confidence_tier is the split's tier field when n_pos > 0,
    # else None. The splitter's `tier` field is "T0" by default for
    # NonVulnerable contracts; we override to None at the parquet level.
    confidence_tier: str | None = tier if n_pos > 0 else None

    return LabelsParquetRow(
        contract_id=contract["sha256"],
        source=contract.get("source", "unknown"),
        split=split,
        class_0=int(classes.get(names[0], 0)),
        class_1=int(classes.get(names[1], 0)),
        class_2=int(classes.get(names[2], 0)),
        class_3=int(classes.get(names[3], 0)),
        class_4=int(classes.get(names[4], 0)),
        class_5=int(classes.get(names[5], 0)),
        class_6=int(classes.get(names[6], 0)),
        class_7=int(classes.get(names[7], 0)),
        class_8=int(classes.get(names[8], 0)),
        class_9=int(classes.get(names[9], 0)),
        confidence_tier=confidence_tier,
    )


def write_labels_parquet(splits_dir: Path, output_path: Path) -> Path:
    """Build the labels.parquet from the split JSONL files.

    Args:
        splits_dir: Path to `data/splits/v{N}/` containing
            `{train,val,test}.jsonl` (one Contract per line, asdict-serialized).
        output_path: Where to write `labels.parquet` (snappy-compressed).

    Returns:
        The output_path (for chaining).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = class_names()
    split_rows = _load_split_jsonl(splits_dir)
    rows = [_contract_to_row(split, c, names) for split, c in split_rows]

    # Build the pyarrow table from explicit column lists (preserves order).
    schema = _build_schema()
    arrays: list[pa.Array] = [
        pa.array([r.contract_id for r in rows], type=pa.string()),
        pa.array([r.source for r in rows], type=pa.string()),
        pa.array([r.split for r in rows], type=pa.string()),
        *(pa.array([getattr(r, f"class_{i}") for r in rows], type=pa.int8()) for i in range(len(names))),
        pa.array([r.confidence_tier for r in rows], type=pa.string()),
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, output_path, compression="snappy")
    return output_path


__all__ = [
    "CLASS_FIELD_ORDER",
    "LabelsParquetRow",
    "write_labels_parquet",
]
