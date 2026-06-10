"""Tests for the retry-failed merge logic in preprocess.py.

Retry-failed is a build-system-style incremental: when a previous preprocess
run left some files in dropped.csv, retry-failed re-runs ONLY those files
and merges the results — files that now succeed get added to preprocessed/,
files that still fail stay in dropped.csv with their (possibly updated)
error message.

These tests use a temp dir with a fake raw/ + preprocessed/ tree and
exercise the helpers directly so they're fast and don't require real
solc installs.
"""
import csv
import json
from pathlib import Path

import pytest

from sentinel_data.preprocessing.preprocess import (
    _load_dropped_files,
    _merge_retry_results,
)


@pytest.fixture
def dive_like_state(tmp_path):
    """Create raw/, preprocessed/, and dropped.csv in tmp_path.

    Mimics the state of a real source after a partial preprocess:
      - raw/ has 5 source files (1-5)
      - preprocessed/ has 2 successful outputs (1.sol, 2.sol)
      - dropped.csv has 3 failures (3, 4, 5) with reason + error
    """
    raw_dir = tmp_path / "raw" / "dive"
    raw_dir.mkdir(parents=True)
    (raw_dir / "repo").mkdir()
    for i in range(1, 6):
        (raw_dir / "repo" / f"{i}.sol").write_text(f"// contract {i}")

    prep_dir = tmp_path / "preprocessed" / "dive"
    prep_dir.mkdir(parents=True)
    (prep_dir / "1.sol").write_text("processed 1")
    (prep_dir / "1.meta.json").write_text(json.dumps({"sha256": "x", "source_name": "dive"}))
    (prep_dir / "2.sol").write_text("processed 2")
    (prep_dir / "2.meta.json").write_text(json.dumps({"sha256": "y", "source_name": "dive"}))

    dropped_csv = prep_dir / "dropped.csv"
    with open(dropped_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "source", "original_path", "pragma", "reason", "error", "attempted_solc_versions"
        ])
        w.writeheader()
        w.writerow({
            "source": "dive", "original_path": f"repo/{3}.sol",
            "pragma": "0.7.4", "reason": "compile_failed",
            "error": "solc 0.7.4 not installed", "attempted_solc_versions": "",
        })
        w.writerow({
            "source": "dive", "original_path": f"repo/{4}.sol",
            "pragma": "^0.4.0", "reason": "compile_failed",
            "error": "syntax error", "attempted_solc_versions": "",
        })
        w.writerow({
            "source": "dive", "original_path": f"repo/{5}.sol",
            "pragma": "0.8.0", "reason": "compile_failed",
            "error": "0.8.0 not installed", "attempted_solc_versions": "",
        })
    return raw_dir, prep_dir


class TestLoadDroppedFiles:
    def test_loads_paths_and_rows(self, dive_like_state):
        raw_dir, prep_dir = dive_like_state
        paths, rows = _load_dropped_files(prep_dir, raw_dir)
        assert len(paths) == 3
        assert len(rows) == 3
        # Paths should be absolute and point to the right files
        assert all(p.exists() for p in paths)
        assert {p.name for p in paths} == {"3.sol", "4.sol", "5.sol"}

    def test_skips_paths_that_no_longer_exist(self, dive_like_state):
        raw_dir, prep_dir = dive_like_state
        # Remove file 4 from disk (e.g. re-ingest removed it)
        (raw_dir / "repo" / "4.sol").unlink()
        paths, rows = _load_dropped_files(prep_dir, raw_dir)
        # 4.sol is missing → not loaded
        assert {p.name for p in paths} == {"3.sol", "5.sol"}
        # But the row is still in `rows` (the merge function will keep it)
        assert len(rows) == 3

    def test_empty_dropped_csv(self, tmp_path):
        raw_dir = tmp_path / "raw" / "x"
        prep_dir = tmp_path / "preprocessed" / "x"
        raw_dir.mkdir(parents=True)
        prep_dir.mkdir(parents=True)
        paths, rows = _load_dropped_files(prep_dir, raw_dir)
        assert paths == []
        assert rows == []

    def test_no_dropped_csv(self, tmp_path):
        raw_dir = tmp_path / "raw" / "x"
        prep_dir = tmp_path / "preprocessed" / "x"
        raw_dir.mkdir(parents=True)
        prep_dir.mkdir(parents=True)
        # No dropped.csv exists
        paths, rows = _load_dropped_files(prep_dir, raw_dir)
        assert paths == []


class TestMergeRetryResults:
    """Test the merge logic. We construct fake PipelineResult objects.

    The merge reads the .meta.json of each processed file to discover its
    `original_path`. So our test fixtures write meta.json files alongside
    the simulated processed .sol files.
    """

    def _simulate_processed(self, prep_dir, ids):
        """Write .sol + .meta.json for each id, return list of processed paths."""
        paths = []
        for i in ids:
            sol_path = prep_dir / f"{i}.sol"
            sol_path.write_text(f"processed {i}")
            meta_path = prep_dir / f"{i}.meta.json"
            meta_path.write_text(json.dumps({
                "sha256": f"hash{i}",
                "source_name": "dive",
                "original_path": f"repo/{i}.sol",
            }))
            paths.append(sol_path)
        return paths

    def test_now_succeeding_files_removed_from_dropped(self, dive_like_state, tmp_path):
        raw_dir, prep_dir = dive_like_state
        from sentinel_data.preprocessing.pipeline import PipelineResult

        # Simulate: files 3 and 5 now succeed (write their .sol + .meta.json)
        # File 4 still fails.
        processed = self._simulate_processed(prep_dir, [3, 5])
        dropped = [{
            "source": "dive", "original_path": "repo/4.sol",
            "pragma": "^0.4.0", "reason": "compile_failed",
            "error": "new error: still broken", "attempted_solc_versions": "",
        }]
        result = PipelineResult(processed=processed, dropped=dropped, duration_s=1.0)
        _, previously_dropped = _load_dropped_files(prep_dir, raw_dir)

        _merge_retry_results(prep_dir, result, previously_dropped)

        # dropped.csv should now have only 4.sol (not 3 or 5)
        with open(prep_dir / "dropped.csv") as f:
            remaining = list(csv.DictReader(f))
        assert len(remaining) == 1
        assert remaining[0]["original_path"] == "repo/4.sol"
        # The error message should be the NEW one (not the old one)
        assert "new error: still broken" in remaining[0]["error"]

    def test_all_retries_succeed_deletes_dropped_csv(self, dive_like_state):
        raw_dir, prep_dir = dive_like_state
        from sentinel_data.preprocessing.pipeline import PipelineResult

        # All 3 retries now succeed
        processed = self._simulate_processed(prep_dir, [3, 4, 5])
        result = PipelineResult(processed=processed, dropped=[], duration_s=1.0)
        _, previously_dropped = _load_dropped_files(prep_dir, raw_dir)

        _merge_retry_results(prep_dir, result, previously_dropped)

        # dropped.csv should be DELETED
        assert not (prep_dir / "dropped.csv").exists()

    def test_none_retried_keeps_dropped_csv_unchanged(self, dive_like_state):
        """If no files were retried (e.g. all paths gone from disk), keep old rows."""
        raw_dir, prep_dir = dive_like_state
        from sentinel_data.preprocessing.pipeline import PipelineResult

        # All 3 source files removed from disk
        for i in range(3, 6):
            (raw_dir / "repo" / f"{i}.sol").unlink()
        # And the pipeline ran but processed nothing (since no paths were given)
        result = PipelineResult(processed=[], dropped=[], duration_s=0.0)
        _, previously_dropped = _load_dropped_files(prep_dir, raw_dir)

        _merge_retry_results(prep_dir, result, previously_dropped)

        # dropped.csv still exists with all 3 old rows (preserved)
        with open(prep_dir / "dropped.csv") as f:
            remaining = list(csv.DictReader(f))
        assert len(remaining) == 3
        # Old error messages preserved
        assert "solc 0.7.4 not installed" in remaining[0]["error"]
        assert "syntax error" in remaining[1]["error"]
        assert "0.8.0 not installed" in remaining[2]["error"]

    def test_new_error_overwrites_old(self, dive_like_state):
        """When a file still fails, the new error message wins."""
        raw_dir, prep_dir = dive_like_state
        from sentinel_data.preprocessing.pipeline import PipelineResult

        # Files 3, 4, 5 all retried. 3, 5 succeed; 4 still fails with NEW error
        processed = self._simulate_processed(prep_dir, [3, 5])
        dropped = [{
            "source": "dive", "original_path": "repo/4.sol",
            "pragma": "^0.4.0", "reason": "compile_failed",
            "error": "DIFFERENT NEW ERROR", "attempted_solc_versions": "",
        }]
        result = PipelineResult(processed=processed, dropped=dropped, duration_s=1.0)
        _, previously_dropped = _load_dropped_files(prep_dir, raw_dir)

        _merge_retry_results(prep_dir, result, previously_dropped)

        with open(prep_dir / "dropped.csv") as f:
            remaining = list(csv.DictReader(f))
        assert len(remaining) == 1
        assert remaining[0]["error"] == "DIFFERENT NEW ERROR"

    def test_processed_without_meta_skipped_safely(self, dive_like_state):
        """If a .sol exists in out_dir without a .meta.json (corrupt state), don't crash."""
        raw_dir, prep_dir = dive_like_state
        from sentinel_data.preprocessing.pipeline import PipelineResult

        # Write a .sol with no meta.json (corrupt)
        (prep_dir / "3.sol").write_text("orphan .sol")
        # The merge must not crash, must just skip the orphan
        result = PipelineResult(processed=[prep_dir / "3.sol"], dropped=[], duration_s=0.0)
        _, previously_dropped = _load_dropped_files(prep_dir, raw_dir)

        # Should not raise
        _merge_retry_results(prep_dir, result, previously_dropped)

        # 3.sol wasn't recognized as a success (no meta), so it stays in dropped
        with open(prep_dir / "dropped.csv") as f:
            remaining = list(csv.DictReader(f))
        assert any(r["original_path"] == "repo/3.sol" for r in remaining)
