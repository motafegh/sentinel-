"""Tests for the registry module (Stage 5 Tasks 5.5, 5.6, 5.7)."""
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from sentinel_data.registry import (
    Artifact, Catalog, DatasetVersion, Migration, PerClassMetric,
    Retirement, Source, SplitRecord, compute_dict_hash, compute_hash,
    diff_dataset_versions, hash_artifact, hash_lineage,
    lineage_to_dot, record_lineage_step,
    update_changelog, verify_artifact,
)


@pytest.fixture
def tmp_catalog(tmp_path):
    db = tmp_path / "catalog.db"
    yaml_p = tmp_path / "catalog.yaml"
    return Catalog(db, yaml_p), yaml_p


@pytest.fixture
def test_file(tmp_path):
    p = tmp_path / "test.sol"
    p.write_text("// contract Foo { uint x; }")
    return p


# ── compute_hash ───────────────────────────────────────────────────────────

class TestComputeHash:
    def test_compute_hash_file(self, test_file):
        h = compute_hash(test_file)
        assert len(h) == 64  # SHA-256 hex
        # Verify by recomputing
        import hashlib
        expected = hashlib.sha256(test_file.read_bytes()).hexdigest()
        assert h == expected

    def test_compute_dict_hash(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}  # same dict, different order
        assert compute_dict_hash(d1) == compute_dict_hash(d2)
        # Different content → different hash
        d3 = {"a": 1, "b": 3}
        assert compute_dict_hash(d1) != compute_dict_hash(d3)

    def test_hash_file_changes_when_content_changes(self, tmp_path):
        f1 = tmp_path / "a.sol"
        f1.write_text("contract A {}")
        h1 = compute_hash(f1)
        f1.write_text("contract A { uint x; }")
        h2 = compute_hash(f1)
        assert h1 != h2


# ── Catalog: init + schema ───────────────────────────────────────────────

class TestCatalogInit:
    def test_init_creates_db(self, tmp_catalog):
        cat, yaml_path = tmp_catalog
        # yaml_path is tmp_path / "catalog.yaml" — its parent contains the DB
        assert yaml_path.parent.exists()
        # The DB file should also be in the same directory
        db = yaml_path.parent / "catalog.db"
        assert db.exists()

    def test_init_records_initial_migration(self, tmp_catalog):
        cat, _ = tmp_catalog
        migrations = cat.applied_migrations()
        assert len(migrations) >= 1
        assert migrations[0].version == 1
        assert "Initial schema" in migrations[0].description

    def test_init_creates_4_base_tables(self, tmp_catalog):
        cat, _ = tmp_catalog
        with cat._conn() as c:
            tables = [r["name"] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()]
        for table in ("sources", "artifacts", "splits", "dataset_versions",
                      "schema_migrations", "dataset_version_retirements"):
            assert table in tables, f"Missing table: {table}"


# ── Sources ────────────────────────────────────────────────────────────────

class TestSources:
    def test_add_and_get_source(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_source(Source(name="solidifi", pin="abc123", n_contracts=283, tier="T0"))
        s = cat.get_source("solidifi")
        assert s is not None
        assert s.name == "solidifi"
        assert s.pin == "abc123"
        assert s.n_contracts == 283
        assert s.tier == "T0"
        assert s.enabled is True

    def test_get_missing_source_returns_none(self, tmp_catalog):
        cat, _ = tmp_catalog
        assert cat.get_source("nonexistent") is None

    def test_add_source_overwrites(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_source(Source(name="dive", pin="v1", n_contracts=100))
        cat.add_source(Source(name="dive", pin="v2", n_contracts=200))
        s = cat.get_source("dive")
        assert s.pin == "v2"
        assert s.n_contracts == 200

    def test_list_sources(self, tmp_catalog):
        cat, _ = tmp_catalog
        for n in ("alpha", "beta", "gamma"):
            cat.add_source(Source(name=n))
        sources = cat.list_sources()
        assert [s.name for s in sources] == ["alpha", "beta", "gamma"]


# ── Artifacts ─────────────────────────────────────────────────────────────

class TestArtifacts:
    def test_add_and_get_artifact(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_artifact(Artifact(name="foo.sol", sha256="abc123", size_bytes=100))
        a = cat.get_artifact("foo.sol")
        assert a is not None
        assert a.sha256 == "abc123"
        assert a.size_bytes == 100
        assert a.lineage == {}

    def test_artifact_with_lineage(self, tmp_catalog):
        cat, _ = tmp_catalog
        lineage = {
            "steps": [{"step": "ingest", "ts": "2026-07-01"}],
            "parents": ["raw/foo.sol"],
        }
        cat.add_artifact(Artifact(name="foo.sol", sha256="abc", lineage=lineage))
        a = cat.get_artifact("foo.sol")
        assert a.lineage == lineage


# ── Splits ─────────────────────────────────────────────────────────────────

class TestSplits:
    def test_add_and_get_split(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_split(SplitRecord(
            version="v1", seed=42, strategy="stratified",
            contract_counts={"train": 100, "val": 30, "test": 30},
        ))
        s = cat.get_split("v1")
        assert s is not None
        assert s.seed == 42
        assert s.strategy == "stratified"
        assert s.contract_counts == {"train": 100, "val": 30, "test": 30}


# ── Dataset versions ────────────────────────────────────────────────────

class TestDatasetVersions:
    def test_add_and_get_version(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_dataset_version(DatasetVersion(
            name="sentinel-v2-gold-2026-08",
            source_set=["solidifi", "dive"],
            split_version="v1",
            artifact_hash="abc",
        ))
        v = cat.get_dataset_version("sentinel-v2-gold-2026-08")
        assert v is not None
        assert v.source_set == ["solidifi", "dive"]
        assert v.split_version == "v1"
        assert v.label_schema_version == "v1"
        assert v.export_format == "v1"

    def test_retire_dataset_version(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_dataset_version(DatasetVersion(
            name="v1-old", source_set=["a"], split_version="v1",
        ))
        cat.retire_dataset_version("v1-old", superseded_by="v2-new", reason="old")
        assert cat.is_retired("v1-old")

    def test_load_artifact_returns_none_for_retired(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_dataset_version(DatasetVersion(name="v1", source_set=[], split_version="v1"))
        cat.retire_dataset_version("v1", superseded_by="v2")
        assert cat.load_artifact("v1") is None

    def test_load_artifact_returns_version_for_active(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_dataset_version(DatasetVersion(name="v1", source_set=[], split_version="v1"))
        v = cat.load_artifact("v1")
        assert v is not None
        assert v.name == "v1"

    def test_list_excludes_retired(self, tmp_catalog):
        cat, _ = tmp_catalog
        cat.add_dataset_version(DatasetVersion(name="v1", source_set=[], split_version="v1"))
        cat.add_dataset_version(DatasetVersion(name="v2", source_set=[], split_version="v1"))
        cat.retire_dataset_version("v1", superseded_by="v2")
        active = cat.list_dataset_versions(include_retired=False)
        assert [v.name for v in active] == ["v2"]


# ── verify_artifact_hash ─────────────────────────────────────────────────

class TestVerifyArtifactHash:
    def test_verify_artifact_via_artifacts_table(self, tmp_catalog, test_file):
        cat, _ = tmp_catalog
        h = compute_hash(test_file)
        cat.add_artifact(Artifact(name=str(test_file), sha256=h))
        assert cat.verify_artifact_hash(test_file) is True

    def test_verify_artifact_via_dataset_versions_table(self, tmp_catalog, test_file):
        cat, _ = tmp_catalog
        h = compute_hash(test_file)
        cat.add_dataset_version(DatasetVersion(
            name="v1", source_set=[], split_version="v1",
            artifact_hash=h, artifact_path=str(test_file),
        ))
        assert cat.verify_artifact_hash(test_file) is True

    def test_verify_artifact_tampered_returns_false(self, tmp_catalog, test_file):
        cat, _ = tmp_catalog
        h = compute_hash(test_file)
        cat.add_artifact(Artifact(name=str(test_file), sha256=h))
        # Tamper with the file
        test_file.write_text("// modified content")
        assert cat.verify_artifact_hash(test_file) is False

    def test_verify_unregistered_returns_false(self, tmp_catalog, test_file):
        cat, _ = tmp_catalog
        assert cat.verify_artifact_hash(test_file) is False


# ── YAML mirror ──────────────────────────────────────────────────────────

class TestYamlMirror:
    def test_write_yaml_mirror(self, tmp_catalog):
        cat, yp = tmp_catalog
        cat.add_source(Source(name="dive", pin="v1"))
        cat.write_yaml_mirror()
        assert yp.exists()
        # Parse the YAML
        docs = list(yaml.safe_load_all(yp.read_text()))
        assert any(d["kind"] == "sources" for d in docs)

    def test_yaml_mirror_roundtrip(self, tmp_catalog):
        cat, yp = tmp_catalog
        cat.add_source(Source(name="solidifi", pin="abc", n_contracts=283, tier="T0"))
        cat.add_artifact(Artifact(name="foo.sol", sha256="hash1", size_bytes=100))
        cat.add_split(SplitRecord(version="v1", seed=42))
        cat.add_dataset_version(DatasetVersion(
            name="v1", source_set=["solidifi"], split_version="v1",
            artifact_hash="hash1", artifact_path="foo.sol",
        ))
        cat.write_yaml_mirror()
        # Re-load from YAML
        docs = list(yaml.safe_load_all(yp.read_text()))
        by_kind = {d["kind"]: d for d in docs}
        assert "sources" in by_kind
        assert any(s["name"] == "solidifi" for s in by_kind["sources"]["items"])


# ── Lineage tracker ─────────────────────────────────────────────────────

class TestLineageTracker:
    def test_record_lineage_step(self):
        lineage: dict = {}
        record_lineage_step(lineage, "ingest", source="dive")
        record_lineage_step(lineage, "preprocess", dedup_threshold=0.85)
        assert len(lineage["steps"]) == 2
        assert lineage["steps"][0]["step"] == "ingest"
        assert lineage["steps"][0]["source"] == "dive"
        assert lineage["steps"][1]["dedup_threshold"] == 0.85

    def test_lineage_to_dot(self):
        lineage = {"steps": [{"step": "ingest"}, {"step": "preprocess"}]}
        dot = lineage_to_dot(lineage)
        assert "digraph lineage" in dot
        assert "step0" in dot
        assert "step1" in dot
        assert "step0 -> step1" in dot

    def test_hash_lineage_stable(self):
        l1 = {"steps": [{"step": "a"}, {"step": "b"}]}
        l2 = {"steps": [{"step": "a"}, {"step": "b"}]}
        assert hash_lineage(l1) == hash_lineage(l2)
        l3 = {"steps": [{"step": "a"}, {"step": "c"}]}
        assert hash_lineage(l1) != hash_lineage(l3)

    def test_verify_artifact_function(self, tmp_path):
        f = tmp_path / "x.sol"
        f.write_text("// hello")
        h = hash_artifact(f)
        assert verify_artifact(f, h) is True
        # Tamper
        f.write_text("// modified")
        assert verify_artifact(f, h) is False


# ── Dataset diff ────────────────────────────────────────────────────────

class TestDatasetDiff:
    def test_diff_added_removed(self):
        old = {"contract_labels": {"sha1": {"Reentrancy": 1}}}
        new = {"contract_labels": {"sha1": {"Reentrancy": 1}, "sha2": {"IntegerUO": 1}}}
        diff = diff_dataset_versions(old, new, "old", "new")
        assert diff.added_contracts == ["sha2"]
        assert diff.removed_contracts == []
        assert "sha1" in diff.common_contracts

    def test_diff_removed_contracts(self):
        old = {"contract_labels": {"sha1": {"R": 1}, "sha2": {"R": 1}}}
        new = {"contract_labels": {"sha1": {"R": 1}}}
        diff = diff_dataset_versions(old, new, "old", "new")
        assert diff.added_contracts == []
        assert diff.removed_contracts == ["sha2"]

    def test_diff_label_changes(self):
        old = {"contract_labels": {"sha1": {"R": 1, "E": 0}}}
        new = {"contract_labels": {"sha1": {"R": 0, "E": 1}}}
        diff = diff_dataset_versions(old, new, "old", "new")
        assert len(diff.label_changes) == 1
        assert diff.label_changes[0]["sha256"] == "sha1"
        assert diff.label_changes[0]["changes"]["R"] == (1, 0)

    def test_diff_per_class_metrics(self):
        old = {"contract_labels": {"sha1": {"R": 1}, "sha2": {"R": 1}}}
        new = {"contract_labels": {"sha1": {"R": 1}, "sha2": {"R": 1}, "sha3": {"R": 1}}}
        diff = diff_dataset_versions(old, new, "old", "new")
        re_metric = next(m for m in diff.per_class if m.class_name == "R")
        assert re_metric.count_old == 2
        assert re_metric.count_new == 3
        assert re_metric.delta_count == 1
        assert re_metric.delta_pct == 50.0

    def test_update_changelog(self, tmp_path):
        p = tmp_path / "changelog.md"
        update_changelog(p, "v1", "Initial release",
                        {"R": {"count_old": 10, "count_new": 15, "delta_count": 5, "delta_pct": 50.0}})
        assert p.exists()
        content = p.read_text()
        assert "v1" in content
        assert "Initial release" in content
        assert "Reentrancy" in content or "R" in content
        # Append a second entry
        update_changelog(p, "v2", "Added more contracts", {})
        content2 = p.read_text()
        assert "v1" in content2 and "v2" in content2
