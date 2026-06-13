"""SmartBugs Curated label parser — Task 3.5.

Reads Stage 1 preprocessed meta.json files for the smartbugs_curated source,
extracts the category folder from original_path, maps it through the
crosswalk YAML, and writes one .labels.json per contract.

Label source: original_path field in meta.json
  e.g. repo/reentrancy/reentrancy_1.sol  →  folder = reentrancy
       folder → crosswalk → canonical class

The SmartBugs dataset/ directory is symlinked as data/raw/smartbugs_curated/repo/,
so original_path takes the form repo/<category>/<contract>.sol, making
parts[1] the category name (one level shallower than SolidiFI's
repo/buggy_contracts/<FOLDER>/buggy_N.sol layout).

Output: data/labels/smartbugs_curated/<sha256>.labels.json
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from sentinel_data.labeling.schema import class_names

log = logging.getLogger("sentinel_data.labeling.smartbugs_curated")

_CROSSWALK_PATH = Path(__file__).parents[1] / "crosswalks" / "smartbugs_curated.yaml"


@dataclass
class LabelResult:
    """Aggregated statistics from a SmartBugs Curated labeling run."""

    source: str = "smartbugs_curated"
    contracts_seen: int = 0
    labels_written: int = 0
    labels_cached: int = 0
    labels_failed: int = 0
    duration_s: float = 0.0


def _load_crosswalk() -> dict:
    with open(_CROSSWALK_PATH) as f:
        return yaml.safe_load(f)


def _extract_folder(original_path: str) -> str | None:
    """Extract the category folder name from original_path.

    original_path = "repo/<CATEGORY>/<contract>.sol"
    Returns <CATEGORY>, or None if the path doesn't match the expected structure.
    """
    parts = Path(original_path).parts
    if len(parts) >= 3 and parts[0] == "repo":
        return parts[1]
    return None


def _make_labels_entry(value: int, tier: str | None) -> dict:
    return {"value": value, "tier": tier}


def _build_labels_json(
    sha256: str,
    folder: str,
    canonical_class: str,
    tier: str,
) -> dict:
    all_classes = class_names()
    classes = {
        cls: _make_labels_entry(1 if cls == canonical_class else 0,
                                tier if cls == canonical_class else None)
        for cls in all_classes
    }
    n_pos = 0 if canonical_class == "NonVulnerable" else 1
    return {
        "sha256": sha256,
        "source": "smartbugs_curated",
        "category_folder": folder,
        "classes": classes,
        "n_pos": n_pos,
    }


def label_source(
    data_dir: Path,
    *,
    force: bool = False,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> LabelResult:
    """Run the SmartBugs Curated labeling parser.

    Reads data/preprocessed/smartbugs_curated/*.meta.json, maps each contract's
    category folder to a canonical class via the crosswalk, and writes
    data/labels/smartbugs_curated/<sha256>.labels.json.

    Args:
        data_dir: Path to data/ (parent of preprocessed/, labels/).
        force: Overwrite existing .labels.json files.
        limit: Process only the first N contracts.
        output_dir: Override output directory (default: data_dir/labels/smartbugs_curated).
    """
    crosswalk = _load_crosswalk()
    class_map: dict[str, str] = crosswalk["class_map"]
    tier: str = crosswalk["confidence_tier"]

    prep_dir = data_dir / "preprocessed" / "smartbugs_curated"
    out_dir = output_dir if output_dir is not None else data_dir / "labels" / "smartbugs_curated"

    if not prep_dir.exists():
        raise FileNotFoundError(
            f"{prep_dir} not found. Run `sentinel-data preprocess --source smartbugs_curated` first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(prep_dir.glob("*.meta.json"))
    if limit:
        meta_paths = meta_paths[:limit]

    result = LabelResult()
    t0 = time.monotonic()

    for meta_path in meta_paths:
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Cannot read {meta_path.name}: {e}")
            result.labels_failed += 1
            continue

        sha256 = meta["sha256"]
        original_path = meta.get("original_path", "")
        result.contracts_seen += 1

        label_path = out_dir / f"{sha256}.labels.json"

        if not force and label_path.exists():
            result.labels_cached += 1
            continue

        folder = _extract_folder(original_path)
        if folder is None:
            log.warning(f"Cannot extract folder from original_path={original_path!r}")
            result.labels_failed += 1
            continue

        canonical_class = class_map.get(folder)
        if canonical_class is None:
            log.warning(f"Folder {folder!r} not in crosswalk class_map — skipping")
            result.labels_failed += 1
            continue

        labels = _build_labels_json(sha256, folder, canonical_class, tier)
        label_path.write_text(json.dumps(labels, indent=2))
        result.labels_written += 1

    result.duration_s = time.monotonic() - t0
    return result
