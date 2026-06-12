"""Lineage tracker + artifact hasher — Stage 5 Task 5.6.

Lineage is the audit trail (per D-5.5): every artifact in the registry
has a DAG of transformations (which ingestion connector, which
preprocessing step, which labeling parser, which verification component,
which splitter, which export writer produced it). Stored as a JSON
field on the artifact.

The hasher is just a thin wrapper around `catalog.compute_hash` for
streaming hash of large artifacts.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from sentinel_data.registry.catalog import compute_hash, compute_dict_hash

log = logging.getLogger("sentinel_data.registry.lineage_tracker")


def record_lineage_step(lineage: dict, step: str, **details) -> dict:
    """Append a step to a lineage dict. Returns the updated dict.

    lineage is a dict: {steps: [{step, ts, ...}, ...], parents: [sha, ...]}
    """
    from datetime import datetime, timezone
    if "steps" not in lineage:
        lineage["steps"] = []
    if "parents" not in lineage:
        lineage["parents"] = []
    entry = {
        "step": step,
        "ts": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    lineage["steps"].append(entry)
    return lineage


def lineage_to_dot(lineage: dict) -> str:
    """Render lineage as Graphviz DOT for visualization."""
    lines = ["digraph lineage {"]
    for i, step in enumerate(lineage.get("steps", [])):
        label = step.get("step", "unknown")
        lines.append(f'  step{i} [label="{label}"];')
    for i in range(len(lineage.get("steps", [])) - 1):
        lines.append(f"  step{i} -> step{i+1};")
    lines.append("}")
    return "\n".join(lines)


def hash_artifact(path: Path) -> str:
    """Compute SHA-256 of a file. Streaming for large files."""
    return compute_hash(path)


def hash_lineage(lineage: dict) -> str:
    """Stable hash of a lineage dict (for content addressing)."""
    return compute_dict_hash(lineage)


def verify_artifact(path: Path, expected_hash: str) -> bool:
    """Verify a file's hash matches the expected value.

    The load-time gate (per D-5.6). The ML module's `SentinelDataset.__init__`
    calls this before loading.
    """
    if not path.exists():
        log.error(f"Artifact not found: {path}")
        return False
    actual = hash_artifact(path)
    if actual != expected_hash:
        log.error(f"Hash mismatch for {path}: expected {expected_hash[:12]}..., got {actual[:12]}...")
        return False
    return True
