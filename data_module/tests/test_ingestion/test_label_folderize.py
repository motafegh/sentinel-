"""Tests for sentinel_data.ingestion.label_folderize."""
import csv
import json
from pathlib import Path

import pytest

from sentinel_data.ingestion.label_folderize import folderize_by_labels


@pytest.fixture
def dive_like_setup(tmp_path):
    """Create a fake DIVE-style layout: flat source files + labels CSV."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "__source__").mkdir()
    # Source files: 1.sol, 2.sol, 3.sol, 4.sol, 5.sol
    for i in range(1, 6):
        (repo / "__source__" / f"{i}.sol").write_text(f"// contract {i}")
    # Labels CSV
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contractID", "Reentrancy", "DoS", "Arithmetic"])
        w.writerow(["1", "1", "0", "0"])           # Reentrancy only
        w.writerow(["2", "1", "1", "0"])           # Reentrancy + DoS
        w.writerow(["3", "0", "0", "1"])           # Arithmetic only
        w.writerow(["4", "0", "0", "0"])           # no labels
        w.writerow(["5", "1", "0", "1"])           # Reentrancy + Arithmetic
        w.writerow(["6", "1", "0", "0"])           # ID 6 has no source file
    return repo, labels


class TestFolderizeByLabels:
    def test_creates_per_class_symlinks(self, dive_like_setup):
        repo, labels = dive_like_setup
        r = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy", "DoS", "Arithmetic"],
        )
        assert r.symlinks_created == 6  # 1(Re), 2(Re), 2(DoS), 3(Ar), 5(Re), 5(Ar)
        assert sorted(r.classes_present) == ["Arithmetic", "DoS", "Reentrancy"]
        assert (repo / "Reentrancy" / "1.sol").is_symlink()
        assert (repo / "DoS" / "2.sol").is_symlink()
        assert (repo / "Arithmetic" / "3.sol").is_symlink()
        # Multi-label contract: 2.sol is in both Reentrancy and DoS
        assert (repo / "Reentrancy" / "2.sol").is_symlink()
        assert (repo / "DoS" / "2.sol").is_symlink()

    def test_multi_label_counted(self, dive_like_setup):
        repo, labels = dive_like_setup
        r = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy", "DoS", "Arithmetic"],
        )
        # 2 (Re+DoS) and 5 (Re+Ar) are multi-label → 2 multi-label
        assert r.multi_label == 2
        assert r.contracts_seen == 6  # 6 rows in labels CSV

    def test_symlink_target_is_source_file(self, dive_like_setup):
        repo, labels = dive_like_setup
        folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy"],
        )
        # The symlink must point to ../../__source__/<id>.sol (relative)
        link = repo / "Reentrancy" / "1.sol"
        assert link.is_symlink()
        target = link.readlink()
        # The symlink target should resolve to the real file
        resolved = (link.parent / target).resolve()
        assert resolved.exists()
        assert resolved.name == "1.sol"
        assert resolved.read_text() == "// contract 1"

    def test_idempotent(self, dive_like_setup):
        """Running twice produces the same result, no duplicate symlinks."""
        repo, labels = dive_like_setup
        r1 = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy", "DoS"],
        )
        r2 = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy", "DoS"],
        )
        # First run creates the symlinks; second run is a no-op (idempotent)
        assert r1.symlinks_created > 0
        assert r2.symlinks_created == 0
        # The actual filesystem state is the same
        r1_files = sorted((repo / "Reentrancy").glob("*.sol"))
        r2_files = sorted((repo / "Reentrancy").glob("*.sol"))
        assert r1_files == r2_files

    def test_no_label_rows_skipped(self, dive_like_setup):
        """Contracts with no positive class get no symlinks created."""
        repo, labels = dive_like_setup
        folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy", "DoS", "Arithmetic"],
        )
        # contractID 4 has all zeros — should not appear in any class dir
        for cls in ("Reentrancy", "DoS", "Arithmetic"):
            assert not (repo / cls / "4.sol").exists()

    def test_missing_source_file_skipped(self, dive_like_setup):
        """Labels may reference IDs with no source file (sample count > label count)."""
        repo, labels = dive_like_setup
        # contractID 6 has label=Reentrancy but no source file
        r = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy"],
        )
        assert not (repo / "Reentrancy" / "6.sol").exists()
        # symlinks_created counts the ones that actually got created
        assert r.symlinks_created < r.contracts_seen

    def test_missing_source_dir_raises(self, tmp_path):
        repo = tmp_path / "nonexistent"
        labels = tmp_path / "labels.csv"
        labels.write_text("contractID,Reentrancy\n1,1\n")
        with pytest.raises(FileNotFoundError, match="Source dir"):
            folderize_by_labels(
                repo, labels,
                id_column="contractID",
                class_columns=["Reentrancy"],
            )

    def test_empty_class_columns(self, dive_like_setup):
        repo, labels = dive_like_setup
        r = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=[],
        )
        assert r.symlinks_created == 0
        assert r.classes_present == []

    def test_real_dive_csv_sample(self, tmp_path):
        """Smoke test against the actual DIVE_Labels.csv header + first 10 rows."""
        real_csv = Path("/home/motafeq/projects/sentinel/data_module/data/raw_staging/dive_labels/DIVE_Labels.csv")
        if not real_csv.exists():
            pytest.skip("DIVE_Labels.csv not extracted; skipping real-data test")
        repo = tmp_path / "repo"
        (repo / "__source__").mkdir(parents=True)
        # Create source files for the first 10 IDs
        for i in range(1, 11):
            (repo / "__source__" / f"{i}.sol").write_text(f"// {i}")
        r = folderize_by_labels(
            repo, real_csv,
            id_column="contractID",
            class_columns=["Reentrancy", "Access Control", "Arithmetic",
                          "Unchecked Return Values", "DoS", "Bad Randomness",
                          "Front Running", "Time manipulation"],
        )
        # All 10 contracts should produce at least one symlink (DIVE has
        # multi-label = 100% in our sample, with 0 zero-label contracts).
        assert r.symlinks_created >= 10
        assert r.contracts_seen >= 10
        assert len(r.classes_present) >= 5  # at least 5 classes hit in first 10


class TestFolderizeFlatSourceMove:
    """When source files are at repo_dir root (flat layout like DIVE),
    folderize should MOVE them into repo_dir/__source__/ first, then
    create symlinks pointing to the new canonical location.

    This keeps the root of repo/ clean (only contains __source__/ + per-class
    subdirs) so downstream pipelines can scan per-class directories without
    seeing duplicates.
    """

    def test_flat_files_moved_into_source_subdir(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        # 5 flat .sol files at the root
        for i in range(1, 6):
            (repo / f"{i}.sol").write_text(f"// contract {i}")
        # Labels CSV
        labels = tmp_path / "labels.csv"
        with open(labels, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contractID", "Reentrancy", "DoS"])
            w.writerow(["1", "1", "0"])
            w.writerow(["2", "0", "1"])
        r = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy", "DoS"],
        )
        # All 5 flat files should be moved
        assert r.files_moved == 5
        # No flat .sol files at the root anymore
        assert list(repo.glob("*.sol")) == []
        # All 5 files are in __source__/
        src_files = sorted((repo / "__source__").glob("*.sol"))
        assert len(src_files) == 5
        # Symlinks created in class subdirs
        assert (repo / "Reentrancy" / "1.sol").is_symlink()
        assert (repo / "DoS" / "2.sol").is_symlink()
        # The symlink should resolve to the canonical file
        assert (repo / "Reentrancy" / "1.sol").resolve().read_text() == "// contract 1"

    def test_root_clean_after_folderize(self, tmp_path):
        """After folderize, repo/ should contain only __source__/ + per-class subdirs."""
        repo = tmp_path / "repo"
        repo.mkdir()
        for i in range(1, 4):
            (repo / f"{i}.sol").write_text(f"// {i}")
        labels = tmp_path / "labels.csv"
        with open(labels, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contractID", "ClassA", "ClassB"])
            w.writerow(["1", "1", "0"])
            w.writerow(["2", "0", "1"])
            w.writerow(["3", "1", "1"])
        folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["ClassA", "ClassB"],
        )
        # Root should have: __source__/ + ClassA/ + ClassB/ — no .sol files
        root_entries = list(repo.iterdir())
        names = {e.name for e in root_entries}
        assert names == {"__source__", "ClassA", "ClassB"}
        for e in root_entries:
            if e.suffix == ".sol":
                raise AssertionError(f"flat .sol found at repo root: {e}")

    def test_idempotent_no_removals(self, tmp_path):
        """Second run on a folderized repo should be a no-op (no moves)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        for i in range(1, 4):
            (repo / f"{i}.sol").write_text(f"// {i}")
        labels = tmp_path / "labels.csv"
        with open(labels, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contractID", "Reentrancy"])
            w.writerow(["1", "1"])
            w.writerow(["2", "1"])
        r1 = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy"],
        )
        assert r1.files_moved == 3
        r2 = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy"],
        )
        # No new files to move
        assert r2.files_moved == 0

    def test_stragglers_picked_up(self, tmp_path):
        """If a re-ingest adds new flat .sol files, the second folderize moves them too."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "1.sol").write_text("// 1")
        labels = tmp_path / "labels.csv"
        with open(labels, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contractID", "Reentrancy"])
            w.writerow(["1", "1"])
        # First run: moves 1.sol
        r1 = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy"],
        )
        assert r1.files_moved == 1
        # Simulate re-ingest: add new flat files
        (repo / "2.sol").write_text("// 2")
        (repo / "3.sol").write_text("// 3")
        # Add labels for them
        with open(labels, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["2", "1"])
            w.writerow(["3", "1"])
        r2 = folderize_by_labels(
            repo, labels,
            id_column="contractID",
            class_columns=["Reentrancy"],
        )
        # The 2 new stragglers should be moved
        assert r2.files_moved == 2
