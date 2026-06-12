"""Ingestion manifest — per-source pull record with SHA-256 per-file verification.

Every `sentinel-data ingest --source <name>` run writes one manifest.
Manifests are append-only: past entries are never deleted, they are versioned.
A re-ingest re-validates every SHA-256; any changed file fails loud.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileRecord:
    """SHA-256 fingerprint for a single .sol file within an ingested source."""
    path: str           # relative to the source raw dir
    sha256: str
    size_bytes: int


@dataclass
class IngestionManifest:
    """Per-source pull record capturing what was fetched and verified.

    One manifest is written per ``sentinel-data ingest --source <name>``
    invocation. Manifests are append-only; re-ingests re-validate every
    SHA-256 and fail loudly on any changed file.
    """
    source: str
    connector: str
    url: str
    pin: str                           # commit / version / record — empty = HEAD at fetch time
    resolved_pin: str                  # the actual commit/hash resolved at fetch time
    fetched_at: str                    # ISO-8601
    duration_s: float
    contract_count: int
    files: list[FileRecord] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialise the manifest to JSON, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "IngestionManifest":
        """Deserialise a manifest from its JSON file."""
        with open(path) as f:
            raw = json.load(f)
        files = [FileRecord(**r) for r in raw.pop("files", [])]
        return cls(**raw, files=files)

    # ── Verification ──────────────────────────────────────────────────────────

    def verify(self, raw_dir: Path) -> tuple[bool, list[str]]:
        """Re-check every SHA-256 against files on disk.

        Returns (ok, list_of_errors). ok=True means all hashes match.
        """
        errors: list[str] = []
        for rec in self.files:
            fpath = raw_dir / rec.path
            if not fpath.exists():
                errors.append(f"MISSING  {rec.path}")
                continue
            actual = _sha256(fpath)
            if actual != rec.sha256:
                errors.append(f"CHANGED  {rec.path}  expected={rec.sha256[:12]}  got={actual[:12]}")
        return (len(errors) == 0), errors


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_file_records(sol_files: list[Path], base_dir: Path) -> list[FileRecord]:
    """Create FileRecord entries with SHA-256 for each .sol file under base_dir."""
    records = []
    for p in sol_files:
        records.append(FileRecord(
            path=str(p.relative_to(base_dir)),
            sha256=_sha256(p),
            size_bytes=p.stat().st_size,
        ))
    return records


def load_manifest(raw_dir: Path, source: str) -> IngestionManifest:
    """Load the ingestion manifest for *source* from the standard location."""
    return IngestionManifest.load(raw_dir / source / "ingestion_manifest.json")


def verify_manifest(raw_dir: Path, source: str) -> tuple[bool, list[str]]:
    """Load and verify a source's manifest against files on disk."""
    m = load_manifest(raw_dir, source)
    return m.verify(raw_dir / source)
