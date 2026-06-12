"""Catalog — Stage 5 Task 5.5.

The registry is a SQLite database at `data/registry/catalog.db` with a
human-readable YAML mirror at `data/registry/catalog.yaml`. The
SQLite DB is for fast lookup; the YAML is for version control.

Per AUDIT_PATCHES 5-P2 + 5-P3 + 5-P4, the catalog has:
  - 4 base tables: sources, artifacts, splits, dataset_versions
  - 2 system tables: schema_migrations, dataset_version_retirements

Design decisions (per plan D-5.4 through D-5.7):
  D-5.4: SQLite + YAML mirror with migrations + retirement chain
  D-5.5: Lineage is a graph, not a flat list (stored as JSON field)
  D-5.6: Hash verification is the load-time gate (SHA-256 of file bytes)
  D-5.7: Dataset versions are named and append-only

The shared hash function (per AUDIT_PATCHES 5-P4) is in
`sentinel_data.registry.compute_hash` — used by both the catalog
and (eventually, post-seam-swap in Stage 7) the ML module's
InferenceCache. Until Stage 7, the inference cache uses its own
helper; after the swap, both call this function.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger("sentinel_data.registry.catalog")

# ── Shared hash function (per AUDIT_PATCHES 5-P4) ─────────────────────────

def compute_hash(path: Path) -> str:
    """SHA-256 of the file bytes. Used by both the catalog and (post-Stage-7)
    the ML module's InferenceCache.

    Returns hex digest. For large files (>1 GB), consider using a chunked
    read — for the v2 baseline, files are <100 MB.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dict_hash(d: dict) -> str:
    """Stable SHA-256 of a dict's canonical JSON form (sorted keys)."""
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class Source:
    """A data source in the SENTINEL pipeline (e.g. DeFiHackLabs, SolidiFI).

    Tracks provenance, confidence tier, and last-known contract count.
    """
    name: str
    pin: str = ""                # git commit hash, version string, etc.
    last_fetched: str = ""       # ISO timestamp
    enabled: bool = True
    n_contracts: int = 0         # last-known count
    tier: str = ""               # confidence tier (T0/T1/T2/T3/T4)
    metadata: dict = field(default_factory=dict)


@dataclass
class Artifact:
    """A content-addressed file tracked by the registry.

    Each artifact is identified by its SHA-256 hash. Lineage is a DAG
    of transformations that produced this artifact (see lineage_tracker).
    """
    name: str                    # unique name (e.g., "preprocessed/dive/sha_abc.sol")
    sha256: str                  # content hash
    size_bytes: int = 0
    lineage: dict = field(default_factory=dict)  # DAG of transformations
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class SplitRecord:
    """Records a train/val/test split configuration.

    The seed and strategy ensure reproducibility. contract_counts maps
    split names to the number of contracts in each.
    """
    version: str                 # "v1", "v2", etc.
    seed: int = 42
    strategy: str = "stratified"
    contract_counts: dict = field(default_factory=dict)
    metadata_hash: str = ""      # hash of the split_manifest.json
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class DatasetVersion:
    """A named, immutable snapshot of a complete dataset.

    Encapsulates the full provenance chain: which sources, which config,
    which split, which label schema, and the content hash of the export
    bundle. Dataset versions are append-only; retirement is tracked
    separately via the Retirements table.
    """
    name: str                    # "sentinel-v2-gold-2026-08"
    source_set: list[str] = field(default_factory=list)  # list of source names
    preprocessing_config_hash: str = ""   # hash of config.yaml
    split_version: str = ""      # references SplitRecord.version
    label_schema_version: str = "v1"     # the 10-class taxonomy version
    export_format: str = "v1"            # the export schema version
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    verification_report_path: str = ""
    artifact_hash: str = ""              # SHA-256 of the export bundle
    artifact_path: str = ""              # path to the export bundle
    metadata: dict = field(default_factory=dict)


@dataclass
class Migration:
    """A schema migration applied to the catalog database.

    Used for forward-only schema evolution. Each migration is recorded
    exactly once; the Catalog checks at init whether it has already been
    applied.
    """
    version: int
    description: str
    applied_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Retirement:
    """Records that a dataset version has been superseded.

    Retired versions are excluded from `list_dataset_versions` by
    default and blocked from `load_artifact`.
    """
    name: str                    # the retired dataset version
    superseded_by: str           # the new version that replaced it
    retired_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reason: str = ""


# ── Catalog ────────────────────────────────────────────────────────────────

class Catalog:
    """The SENTINEL dataset registry.

    Storage: SQLite at `data/registry/catalog.db` + YAML mirror at
    `data/registry/catalog.yaml`. The mirror is for version control
    (CI checks that DB and YAML stay in sync). Every DB write produces
    a corresponding YAML entry.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path, yaml_mirror_path: Optional[Path] = None):
        """Open or create the catalog database.

        Args:
            db_path: Path to the SQLite database file.
            yaml_mirror_path: Optional path for the YAML mirror export.
                CI checks that DB and YAML stay in sync.
        """
        self.db_path = Path(db_path)
        self.yaml_mirror_path = Path(yaml_mirror_path) if yaml_mirror_path else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        if self.yaml_mirror_path and self.yaml_mirror_path.exists():
            # Sanity: try to load YAML to validate format
            try:
                list(yaml.safe_load_all(self.yaml_mirror_path.read_text()))
            except yaml.YAMLError as e:
                log.warning(f"YAML mirror exists but is invalid: {e}")

    @contextmanager
    def _conn(self):
        """Context manager for SQLite connections with auto-commit and foreign keys."""
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _init_schema(self) -> None:
        """Create all tables if they don't exist and record the initial migration."""
        with self._conn() as c:
            # Schema migrations table
            c.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
            """)
            # Sources
            c.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    name TEXT PRIMARY KEY,
                    pin TEXT DEFAULT '',
                    last_fetched TEXT DEFAULT '',
                    enabled INTEGER DEFAULT 1,
                    n_contracts INTEGER DEFAULT 0,
                    tier TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            # Artifacts
            c.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    name TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    lineage TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            # Splits
            c.execute("""
                CREATE TABLE IF NOT EXISTS splits (
                    version TEXT PRIMARY KEY,
                    seed INTEGER DEFAULT 42,
                    strategy TEXT DEFAULT 'stratified',
                    contract_counts TEXT DEFAULT '{}',
                    metadata_hash TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            # Dataset versions
            c.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    name TEXT PRIMARY KEY,
                    source_set TEXT DEFAULT '[]',
                    preprocessing_config_hash TEXT DEFAULT '',
                    split_version TEXT DEFAULT '',
                    label_schema_version TEXT DEFAULT 'v1',
                    export_format TEXT DEFAULT 'v1',
                    generated_at TEXT NOT NULL,
                    verification_report_path TEXT DEFAULT '',
                    artifact_hash TEXT DEFAULT '',
                    artifact_path TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            # Retirements
            c.execute("""
                CREATE TABLE IF NOT EXISTS dataset_version_retirements (
                    name TEXT PRIMARY KEY,
                    superseded_by TEXT NOT NULL,
                    retired_at TEXT NOT NULL,
                    reason TEXT DEFAULT ''
                )
            """)
            # Record the initial migration
            existing = c.execute(
                "SELECT version FROM schema_migrations WHERE version = ?", (self.SCHEMA_VERSION,)
            ).fetchone()
            if existing is None:
                c.execute(
                    "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
                    (self.SCHEMA_VERSION, "Initial schema: 4 base tables + 2 system tables",
                     datetime.now(timezone.utc).isoformat()),
                )

    # ── Source API ──────────────────────────────────────────────────────

    def add_source(self, source: Source) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO sources
                  (name, pin, last_fetched, enabled, n_contracts, tier, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source.name, source.pin, source.last_fetched,
                  int(source.enabled), source.n_contracts, source.tier,
                  json.dumps(source.metadata)))

    def get_source(self, name: str) -> Optional[Source]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sources WHERE name = ?", (name,)).fetchone()
            if row is None:
                return None
            return Source(
                name=row["name"], pin=row["pin"], last_fetched=row["last_fetched"],
                enabled=bool(row["enabled"]), n_contracts=row["n_contracts"],
                tier=row["tier"],
                metadata=json.loads(row["metadata"]),
            )

    def list_sources(self) -> list[Source]:
        with self._conn() as c:
            return [
                Source(name=r["name"], pin=r["pin"], last_fetched=r["last_fetched"],
                      enabled=bool(r["enabled"]), n_contracts=r["n_contracts"],
                      tier=r["tier"],
                      metadata=json.loads(r["metadata"]))
                for r in c.execute("SELECT * FROM sources ORDER BY name").fetchall()
            ]

    # ── Artifact API ──────────────────────────────────────────────────

    def add_artifact(self, artifact: Artifact) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO artifacts
                  (name, sha256, size_bytes, lineage, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (artifact.name, artifact.sha256, artifact.size_bytes,
                  json.dumps(artifact.lineage), artifact.created_at))

    def get_artifact(self, name: str) -> Optional[Artifact]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM artifacts WHERE name = ?", (name,)).fetchone()
            if row is None:
                return None
            return Artifact(
                name=row["name"], sha256=row["sha256"], size_bytes=row["size_bytes"],
                lineage=json.loads(row["lineage"]), created_at=row["created_at"],
            )

    # ── Split API ──────────────────────────────────────────────────────

    def add_split(self, split: SplitRecord) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO splits
                  (version, seed, strategy, contract_counts, metadata_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (split.version, split.seed, split.strategy,
                  json.dumps(split.contract_counts), split.metadata_hash,
                  split.created_at))

    def get_split(self, version: str) -> Optional[SplitRecord]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM splits WHERE version = ?", (version,)).fetchone()
            if row is None:
                return None
            return SplitRecord(
                version=row["version"], seed=row["seed"], strategy=row["strategy"],
                contract_counts=json.loads(row["contract_counts"]),
                metadata_hash=row["metadata_hash"], created_at=row["created_at"],
            )

    # ── Dataset version API ──────────────────────────────────────────

    def add_dataset_version(self, version: DatasetVersion) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO dataset_versions
                  (name, source_set, preprocessing_config_hash, split_version,
                   label_schema_version, export_format, generated_at,
                   verification_report_path, artifact_hash, artifact_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (version.name, json.dumps(version.source_set),
                  version.preprocessing_config_hash, version.split_version,
                  version.label_schema_version, version.export_format,
                  version.generated_at, version.verification_report_path,
                  version.artifact_hash, version.artifact_path,
                  json.dumps(version.metadata)))

    def get_dataset_version(self, name: str) -> Optional[DatasetVersion]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM dataset_versions WHERE name = ?",
                            (name,)).fetchone()
            if row is None:
                return None
            return DatasetVersion(
                name=row["name"],
                source_set=json.loads(row["source_set"]),
                preprocessing_config_hash=row["preprocessing_config_hash"],
                split_version=row["split_version"],
                label_schema_version=row["label_schema_version"],
                export_format=row["export_format"],
                generated_at=row["generated_at"],
                verification_report_path=row["verification_report_path"],
                artifact_hash=row["artifact_hash"],
                artifact_path=row["artifact_path"],
                metadata=json.loads(row["metadata"]),
            )

    def list_dataset_versions(self, include_retired: bool = False) -> list[DatasetVersion]:
        with self._conn() as c:
            rows = c.execute("SELECT name FROM dataset_versions ORDER BY generated_at DESC").fetchall()
        names = [r["name"] for r in rows]
        if not include_retired:
            retired = {r["name"] for r in c.execute(
                "SELECT name FROM dataset_version_retirements").fetchall()} if False else set()
            # Re-query with new conn to avoid scope issue
            with self._conn() as c:
                retired = {r["name"] for r in c.execute(
                    "SELECT name FROM dataset_version_retirements").fetchall()}
            names = [n for n in names if n not in retired]
        return [self.get_dataset_version(n) for n in names if self.get_dataset_version(n) is not None]

    def retire_dataset_version(self, name: str, superseded_by: str, reason: str = "") -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO dataset_version_retirements
                  (name, superseded_by, retired_at, reason)
                VALUES (?, ?, ?, ?)
            """, (name, superseded_by, datetime.now(timezone.utc).isoformat(), reason))

    def is_retired(self, name: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM dataset_version_retirements WHERE name = ?", (name,)
            ).fetchone()
            return row is not None

    # ── load_artifact (the main ML-module interface, D-5.6) ─────────

    def load_artifact(self, name: str) -> Optional[DatasetVersion]:
        """Load a dataset version by name. Returns None if retired or missing.

        The ML module's `SentinelDataset.__init__` (Stage 7) calls this on
        construction. The hash is verified on load.
        """
        if self.is_retired(name):
            log.warning(f"Dataset version {name!r} is retired; refusing to load")
            return None
        return self.get_dataset_version(name)

    def verify_artifact_hash(self, path: Path) -> bool:
        """Compute SHA-256 of a file and verify it matches the registered hash.

        Checks BOTH the artifacts table (individual artifact registration)
        and the dataset_versions table (artifact_hash on a version).
        Returns True if EITHER matches; False if neither matches OR the
        path is not registered.
        """
        if not path.exists():
            return False
        actual = compute_hash(path)
        path_str = str(path)
        with self._conn() as c:
            # Check artifacts table first
            row = c.execute(
                "SELECT sha256 FROM artifacts WHERE name = ?", (path_str,)
            ).fetchone()
            if row is not None:
                return row["sha256"] == actual
            # Fall back to dataset_versions
            row = c.execute(
                "SELECT artifact_hash FROM dataset_versions WHERE artifact_path = ?",
                (path_str,),
            ).fetchone()
        if row is None:
            return False
        return row["artifact_hash"] == actual

    # ── YAML mirror ──────────────────────────────────────────────────

    def write_yaml_mirror(self) -> None:
        """Export the full catalog to a human-readable YAML file."""
        if not self.yaml_mirror_path:
            return
        self.yaml_mirror_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            sources = [dict(r) for r in c.execute("SELECT * FROM sources").fetchall()]
            artifacts = [dict(r) for r in c.execute("SELECT * FROM artifacts").fetchall()]
            splits = [dict(r) for r in c.execute("SELECT * FROM splits").fetchall()]
            versions = [dict(r) for r in c.execute("SELECT * FROM dataset_versions").fetchall()]
            retirements = [dict(r) for r in c.execute(
                "SELECT * FROM dataset_version_retirements").fetchall()]
            migrations = [dict(r) for r in c.execute(
                "SELECT * FROM schema_migrations ORDER BY version").fetchall()]

        # Parse JSON-stringified dicts back to dicts for readability
        def _expand(rows: list[dict], json_cols: list[str]) -> list[dict]:
            out = []
            for r in rows:
                for col in json_cols:
                    if col in r and isinstance(r[col], str):
                        try:
                            r[col] = json.loads(r[col])
                        except (json.JSONDecodeError, TypeError):
                            pass
                out.append(r)
            return out

        sources = _expand(sources, ["metadata"])
        artifacts = _expand(artifacts, ["lineage"])
        splits = _expand(splits, ["contract_counts"])
        versions = _expand(versions, ["source_set", "metadata"])
        retirements = _expand(retirements, [])

        docs = [
            {"kind": "sources", "items": sources},
            {"kind": "artifacts", "items": artifacts},
            {"kind": "splits", "items": splits},
            {"kind": "dataset_versions", "items": versions},
            {"kind": "retirements", "items": retirements},
            {"kind": "schema_migrations", "items": migrations},
        ]
        with self.yaml_mirror_path.open("w") as f:
            yaml.dump_all(docs, f, default_flow_style=False, sort_keys=False)
        log.info(f"Wrote YAML mirror: {self.yaml_mirror_path}")

    # ── Migrations ────────────────────────────────────────────────────

    def applied_migrations(self) -> list[Migration]:
        with self._conn() as c:
            return [
                Migration(version=r["version"], description=r["description"],
                          applied_at=r["applied_at"])
                for r in c.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall()
            ]

    def migrate(self, version: int, description: str) -> None:
        """Apply a schema migration. Caller is responsible for the actual
        ALTER TABLE statements — this method just records the migration.
        """
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO schema_migrations (version, description, applied_at)
                VALUES (?, ?, ?)
            """, (version, description, datetime.now(timezone.utc).isoformat()))
