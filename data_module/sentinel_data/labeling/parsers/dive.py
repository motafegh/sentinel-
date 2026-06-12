"""DIVE label parser — Task 3.6.

Reads Stage 1 preprocessed meta.json files for the dive source,
resolves folder membership from the raw repo (multi-label), maps folders
through the crosswalk YAML, and writes one .labels.json per contract.

Label mechanism:
  DIVE stores all contracts in repo/__source__/<N>.sol.
  The same file also appears in vulnerability folders
  (e.g. repo/Reentrancy/<N>.sol, repo/Arithmetic/<N>.sol).
  Folder membership = positive labels for those classes.
  No folder membership = NonVulnerable.
  Bad Randomness is dropped (no canonical equivalent).

Output: data/labels/dive/<sha256>.labels.json
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from sentinel_data.labeling.schema import class_names

log = logging.getLogger("sentinel_data.labeling.dive")

_CROSSWALK_PATH = Path(__file__).parents[1] / "crosswalks" / "dive.yaml"


@dataclass
class LabelResult:
    """Aggregated statistics from a DIVE labeling run."""

    source: str = "dive"
    contracts_seen: int = 0
    labels_written: int = 0
    labels_cached: int = 0
    labels_failed: int = 0
    nonvulnerable_written: int = 0
    duration_s: float = 0.0


def _load_crosswalk() -> dict:
    """Load the DIVE crosswalk YAML mapping folder names to canonical classes."""
    with open(_CROSSWALK_PATH) as f:
        return yaml.safe_load(f)


def _build_folder_index(raw_repo: Path, class_map: dict[str, str]) -> dict[str, frozenset[str]]:
    """Build filename → frozenset of canonical class names.

    Scans every mapped folder under raw_repo for .sol files and records
    which canonical classes each filename belongs to.
    Bad Randomness and any unmapped folders are silently skipped.
    """
    index: dict[str, set[str]] = {}
    for folder_name, canonical_class in class_map.items():
        folder = raw_repo / folder_name
        if not folder.is_dir():
            log.warning(f"DIVE folder not found: {folder}")
            continue
        for sol in folder.glob("*.sol"):
            index.setdefault(sol.name, set()).add(canonical_class)
    return {k: frozenset(v) for k, v in index.items()}


def _build_labels_json(
    sha256: str,
    filename: str,
    canonical_classes: frozenset[str],
    tier: str,
) -> dict:
    """Build the full .labels.json dict for one DIVE contract."""
    all_classes = class_names()
    classes = {
        cls: {"value": 1 if cls in canonical_classes else 0,
              "tier": tier if cls in canonical_classes else None}
        for cls in all_classes
    }
    return {
        "sha256": sha256,
        "source": "dive",
        "source_filename": filename,
        "classes": classes,
        "n_pos": len(canonical_classes),
    }


def label_source(
    data_dir: Path,
    *,
    force: bool = False,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> LabelResult:
    """Run the DIVE labeling parser.

    Args:
        data_dir: Path to data/ (parent of raw/, preprocessed/, labels/).
        force: Overwrite existing .labels.json files.
        limit: Process only the first N contracts.
        output_dir: Override output directory (default: data_dir/labels/dive).
    """
    crosswalk = _load_crosswalk()
    class_map: dict[str, str] = crosswalk["class_map"]
    tier: str = crosswalk["confidence_tier"]

    raw_repo = data_dir / "raw" / "dive" / "repo"
    prep_dir = data_dir / "preprocessed" / "dive"
    out_dir = output_dir if output_dir is not None else data_dir / "labels" / "dive"

    if not prep_dir.exists():
        raise FileNotFoundError(
            f"{prep_dir} not found. Run `sentinel-data preprocess --source dive` first."
        )
    if not raw_repo.exists():
        raise FileNotFoundError(
            f"{raw_repo} not found. Raw DIVE repo required for label lookup."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the folder membership index once
    folder_index = _build_folder_index(raw_repo, class_map)

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

        # Extract filename from original_path: "repo/__source__/<N>.sol" → "<N>.sol"
        filename = Path(original_path).name
        canonical_classes = folder_index.get(filename, frozenset())

        labels = _build_labels_json(sha256, filename, canonical_classes, tier)
        label_path.write_text(json.dumps(labels, indent=2))
        result.labels_written += 1
        if len(canonical_classes) == 0:
            result.nonvulnerable_written += 1

    result.duration_s = time.monotonic() - t0
    return result
