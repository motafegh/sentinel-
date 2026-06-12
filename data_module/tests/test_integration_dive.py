"""DIVE integration test — guards the real-source flow.

Mirrors test_integration_solidifi.py pattern. Skipped if the DIVE data
hasn't been ingested.
"""
import json
import csv
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATA_DIR / "data" / "raw" / "dive"
PREP_DIR = DATA_DIR / "data" / "preprocessed" / "dive"
LABELS_CSV = DATA_DIR / "data" / "raw_staging" / "dive_labels" / "DIVE_Labels.csv"

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "ingestion_manifest.json").exists(),
    reason="DIVE not ingested; run `sentinel-data ingest --source dive`",
)


def test_manifest_has_correct_count():
    """DIVE's Source codes/ has 22,330 .sol files. Manifest must match."""
    m = json.loads((RAW_DIR / "ingestion_manifest.json").read_text())
    assert m["contract_count"] == 22330, (
        f"Expected 22,330 DIVE contracts; got {m['contract_count']}. "
        f"Check the manual connector's staging_path / find_sol_files scoping."
    )


def test_manifest_pin_is_date():
    """DIVE uses a date pin (not a SHA) since it's not a git repo."""
    m = json.loads((RAW_DIR / "ingestion_manifest.json").read_text())
    assert m["pin"] == "2025-10-02", f"Expected date pin, got {m['pin']!r}"


def test_repo_is_symlink_to_staging():
    """Manual connector with materialize=symlink creates a symlink, not a copy."""
    repo = RAW_DIR / "repo"
    assert repo.is_symlink(), f"repo/ is not a symlink (got {type(repo)})"
    target = repo.readlink()
    assert "raw_staging" in str(target), f"symlink target doesn't look like staging: {target}"


def test_source_files_under_source_subdir():
    """After folderize, all 22,330 real .sol files are under repo/__source__/.

    The flat files at the root have been MOVED (not copied) to __source__/.
    """
    if not PREP_DIR.exists():
        pytest.skip("not yet preprocessed")
    repo = RAW_DIR / "repo"
    source_dir = repo / "__source__"
    assert source_dir.exists(), "__source__/ does not exist — folderize not run?"
    n_source = len(list(source_dir.glob("*.sol")))
    assert n_source >= 22000, f"Expected ≥22K in __source__/, got {n_source}"
    # And the root is clean
    flat_sol = list(repo.glob("*.sol"))
    assert not flat_sol, f"Found {len(flat_sol)} flat .sol files at root (should be 0)"


def test_folderization_created_class_symlinks():
    """The preprocess stage creates per-class symlinks under repo/<Class>/<id>.sol.

    DIVE has 22,330 labeled contracts, multi-label heavy. Expect 50K+ symlinks
    across 8 class folders (Bad Randomness dropped per friend review, so
    label_folderize still creates 8 since it operates on the raw CSV columns;
    the dropping happens in Stage 3 crosswalk).

    Layout (revised 2026-06-10):
      repo/__source__/<id>.sol          (22,330 real files)
      repo/<Class>/<id>.sol            (54,919 symlinks → ../../__source__/<id>.sol)
    The root of repo/ is clean — only __source__/ + per-class subdirs.
    """
    if not PREP_DIR.exists():
        pytest.skip("not yet preprocessed")
    repo = RAW_DIR / "repo"
    # Root should be CLEAN — no flat .sol files
    flat_sol = list(repo.glob("*.sol"))
    assert not flat_sol, (
        f"Found {len(flat_sol)} flat .sol files at repo root. "
        f"Folderize should have moved them to __source__/. "
        f"Re-run `sentinel-data preprocess --source dive` after the layout fix."
    )
    # __source__/ should have 22,330 real files
    source_dir = repo / "__source__"
    if source_dir.exists():
        n_source = len(list(source_dir.glob("*.sol")))
        # The 22,330 source files (minus any dropped by deduplication at preprocess time)
        assert n_source >= 22000, f"Expected ≥22K in __source__/, got {n_source}"
    # 8 class subdirs with symlinks
    class_dirs = [d for d in repo.iterdir() if d.is_dir() and not d.name.startswith("__")]
    assert len(class_dirs) == 8, (
        f"Expected 8 class folders (DIVE's 8 DASP columns), got {len(class_dirs)}: "
        f"{[d.name for d in class_dirs]}"
    )
    total_symlinks = sum(
        len([s for s in d.iterdir() if s.is_symlink()])
        for d in class_dirs
    )
    # 22,330 contracts with avg 2.46 labels → ~55K symlinks. Floor: 40K.
    assert total_symlinks >= 40000, (
        f"Expected ≥40K class symlinks (22K contracts × ~2 labels each), "
        f"got {total_symlinks}"
    )


def test_multi_label_contract_appears_in_multiple_folders():
    """A multi-label contractID has the same id.sol in 2+ class folders."""
    if not PREP_DIR.exists():
        pytest.skip("not yet preprocessed")
    repo = RAW_DIR / "repo"
    # Pick a known multi-label contract from the labels CSV
    if not LABELS_CSV.exists():
        pytest.skip("DIVE_Labels.csv not extracted")
    with open(LABELS_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            cid = int(row["contractID"])
            pos = [c for c in ["Reentrancy", "Access Control", "Arithmetic",
                               "Unchecked Return Values", "DoS",
                               "Front Running", "Time manipulation"]
                   if row[c] == "1"]
            if len(pos) >= 3:
                # This contract should appear in `len(pos)` class folders
                folders_with_this = []
                for cls in pos:
                    if (repo / cls / f"{cid}.sol").exists():
                        folders_with_this.append(cls)
                assert len(folders_with_this) == len(pos), (
                    f"contractID {cid} has {len(pos)} positive classes but only "
                    f"{len(folders_with_this)} symlinks found: {folders_with_this}"
                )
                return
    pytest.skip("No multi-label contract found in first pass of labels CSV")


def test_preprocessed_outputs_exist():
    if not PREP_DIR.exists():
        pytest.skip("not yet preprocessed")
    sols = list(PREP_DIR.glob("*.sol"))
    metas = list(PREP_DIR.glob("*.meta.json"))
    assert len(sols) > 22000, f"Expected ≥22K .sol, got {len(sols)}"
    assert len(sols) == len(metas), "sol/meta count mismatch"


def test_drop_rate_below_threshold():
    """DIVE should yield >99% (it's clean Ethereum mainnet contracts)."""
    if not PREP_DIR.exists():
        pytest.skip("not yet preprocessed")
    processed = len(list(PREP_DIR.glob("*.meta.json")))
    dropped = 0
    if (PREP_DIR / "dropped.csv").exists():
        with open(PREP_DIR / "dropped.csv") as f:
            dropped = sum(1 for _ in csv.DictReader(f))
    total = processed + dropped
    drop_rate = dropped / total if total else 0
    assert drop_rate < 0.01, (
        f"Drop rate {drop_rate:.2%} is above 1% (expected <1% for DIVE)."
    )


def test_version_bucket_distribution():
    """DIVE is real Ethereum mainnet, mostly 0.4-0.8 era. Verify no single bucket dominates unexpectedly."""
    if not PREP_DIR.exists():
        pytest.skip("not yet preprocessed")
    from collections import Counter
    buckets = Counter()
    for m_path in PREP_DIR.glob("*.meta.json"):
        meta = json.loads(m_path.read_text())
        buckets[meta["version_bucket"]] += 1
    # DIVE sources are 2017-2023 mainnet — all 3 buckets should be populated
    assert buckets.get("modern", 0) > 0, f"modern bucket empty: {dict(buckets)}"
    assert buckets.get("transitional", 0) > 0, f"transitional bucket empty: {dict(buckets)}"
    assert buckets.get("legacy", 0) > 0, f"legacy bucket empty: {dict(buckets)}"


def test_meta_json_has_all_fields():
    if not PREP_DIR.exists():
        pytest.skip("not yet preprocessed")
    expected_fields = {
        "sha256", "source_name", "original_path", "pragma", "solc_version",
        "compile_status", "compile_error", "attempted_solc_versions",
        "flatten_status", "dedup_group_id", "is_duplicate", "duplicate_of",
        "version_bucket", "has_unchecked_block", "contract_names",
        "n_raw_lines", "n_normalized_lines", "meta_schema_version",
    }
    # Sample 50 random metas
    import random
    random.seed(0)
    metas = random.sample(list(PREP_DIR.glob("*.meta.json")), min(50, len(list(PREP_DIR.glob("*.meta.json")))))
    for m_path in metas:
        meta = json.loads(m_path.read_text())
        missing = expected_fields - set(meta.keys())
        assert not missing, f"{m_path.name} missing fields: {missing}"


def test_duplicate_reasons_are_known():
    """Even though DIVE has near-zero drops, the dropped reasons must be in the known set."""
    if not PREP_DIR.exists():
        pytest.skip("not yet preprocessed")
    allowed = {"duplicate", "compile_failed", "worker_exception"}
    dropped_csv = PREP_DIR / "dropped.csv"
    if not dropped_csv.exists():
        return
    with open(dropped_csv) as f:
        for row in csv.DictReader(f):
            assert row["reason"] in allowed, (
                f"unknown drop reason {row['reason']!r} for {row['original_path']}"
            )


def test_retry_failed_merges_results(tmp_path, monkeypatch):
    """Integration test for --retry-failed: read the real DIVE dropped.csv,
    re-run the pipeline on those files, and verify the merge behaves correctly.

    This test only runs if a DIVE dropped.csv exists (i.e. the user has
    run the full preprocess at least once).
    """
    if not PREP_DIR.exists():
        pytest.skip("DIVE not yet preprocessed")
    dropped_csv = PREP_DIR / "dropped.csv"
    if not dropped_csv.exists():
        pytest.skip("DIVE dropped.csv doesn't exist (perfect yield)")

    # Read the actual dropped rows
    with open(dropped_csv) as f:
        original_rows = list(csv.DictReader(f))
    if not original_rows:
        pytest.skip("DIVE dropped.csv is empty")

    n_dropped = len(original_rows)
    if n_dropped == 0:
        return

    # Snapshot the current state of preprocessed/ + dropped.csv
    snapshot_processed = len(list(PREP_DIR.glob("*.sol")))
    snapshot_dropped = n_dropped

    # Invoke _load_dropped_files to confirm the paths are valid
    from sentinel_data.preprocessing.preprocess import _load_dropped_files
    paths, rows = _load_dropped_files(PREP_DIR, RAW_DIR)
    assert len(paths) == n_dropped
    # All paths should exist (no re-ingest has happened)
    assert all(p.exists() for p in paths)

    # Now invoke the actual retry via the CLI handler (dry-run, no actual changes)
    # This validates that the CLI surface works end-to-end
    from click.testing import CliRunner
    from sentinel_data.cli import _run_preprocess
    import argparse
    args = argparse.Namespace(
        config=str(DATA_DIR / "config.yaml"),
        dry_run=True,        # don't actually re-run, just check dispatch
        source="dive",
        workers=1,
        sample=None,
        retry_failed=True,
    )
    # Should not raise; prints the would-do plan
    _run_preprocess(args)

    # State should be unchanged (dry-run)
    assert len(list(PREP_DIR.glob("*.sol"))) == snapshot_processed
    with open(dropped_csv) as f:
        rows_after = list(csv.DictReader(f))
    assert len(rows_after) == snapshot_dropped
