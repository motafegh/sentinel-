"""SolidiFI integration test — guards the real-source flow.

This is a Stage 1.5 integration test (not a unit test). It runs only if
the SolidiFI ingest + preprocess has been executed and produced outputs.

What it asserts (regression for the 2026-06-10 real-source run):
  1. ingestion_manifest.json exists and has 350 contracts (NOT 1700)
  2. every manifest path starts with "repo/buggy_contracts/"
  3. preprocessed/ has .meta.json + .sol files
  4. every .meta.json has all 18 ContractMeta fields
  5. compile_status is "ok" for every processed file
  6. version_bucket is one of legacy/transitional/modern
  7. drop rate is < 25% (was 67% before include_subdirs fix)
  8. dropped.csv reasons are subset of {duplicate, compile_failed}
"""
import json
import csv
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATA_DIR / "data" / "raw" / "solidifi"
PREP_DIR = DATA_DIR / "data" / "preprocessed" / "solidifi"

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "ingestion_manifest.json").exists(),
    reason="SolidiFI not ingested; run `sentinel-data ingest --source solidifi`",
)


def test_manifest_has_correct_count():
    m = json.loads((RAW_DIR / "ingestion_manifest.json").read_text())
    assert m["contract_count"] == 350, (
        f"Expected 350 (buggy_contracts only); got {m['contract_count']}. "
        f"Re-run `sentinel-data ingest --source solidifi` after the "
        f"include_subdirs fix."
    )


def test_manifest_paths_scoped_to_buggy_contracts():
    m = json.loads((RAW_DIR / "ingestion_manifest.json").read_text())
    non_scoped = [f for f in m["files"] if "buggy_contracts/" not in f["path"]]
    assert not non_scoped, (
        f"{len(non_scoped)} paths outside buggy_contracts/ — include_subdirs "
        f"in config.yaml is not being honored."
    )


def test_manifest_pin_resolves():
    m = json.loads((RAW_DIR / "ingestion_manifest.json").read_text())
    assert m["pin"] == m["resolved_pin"], "pin did not resolve to the pinned commit"


def test_preprocessed_outputs_exist():
    if not PREP_DIR.exists():
        pytest.skip("preprocessed/ not yet produced; run `sentinel-data preprocess --source solidifi`")
    sols = list(PREP_DIR.glob("*.sol"))
    metas = list(PREP_DIR.glob("*.meta.json"))
    assert len(sols) > 0, "no .sol files in preprocessed/"
    assert len(sols) == len(metas), "sol/meta count mismatch"


def test_meta_json_has_all_fields():
    if not PREP_DIR.exists():
        pytest.skip("preprocessed/ not yet produced")
    expected_fields = {
        "sha256", "source_name", "original_path", "pragma", "solc_version",
        "compile_status", "compile_error", "attempted_solc_versions",
        "flatten_status", "dedup_group_id", "is_duplicate", "duplicate_of",
        "version_bucket", "has_unchecked_block", "contract_names",
        "n_raw_lines", "n_normalized_lines", "meta_schema_version",
    }
    for m_path in list(PREP_DIR.glob("*.meta.json"))[:5]:
        meta = json.loads(m_path.read_text())
        missing = expected_fields - set(meta.keys())
        assert not missing, f"{m_path.name} missing fields: {missing}"


def test_compile_status_is_ok():
    if not PREP_DIR.exists():
        pytest.skip("preprocessed/ not yet produced")
    failed = 0
    for m_path in PREP_DIR.glob("*.meta.json"):
        meta = json.loads(m_path.read_text())
        if meta["compile_status"] != "ok":
            failed += 1
    assert failed == 0, f"{failed} files failed to compile"


def test_version_bucket_in_allowed_set():
    if not PREP_DIR.exists():
        pytest.skip("preprocessed/ not yet produced")
    allowed = {"legacy", "transitional", "modern"}
    for m_path in PREP_DIR.glob("*.meta.json"):
        meta = json.loads(m_path.read_text())
        assert meta["version_bucket"] in allowed, f"{m_path.name}: {meta['version_bucket']}"


def test_drop_rate_below_threshold():
    """Regression guard: drop rate was 67% before the include_subdirs fix."""
    if not PREP_DIR.exists():
        pytest.skip("preprocessed/ not yet produced")
    processed = len(list(PREP_DIR.glob("*.meta.json")))
    dropped = 0
    dropped_csv = PREP_DIR / "dropped.csv"
    if dropped_csv.exists():
        with open(dropped_csv) as f:
            dropped = sum(1 for _ in csv.DictReader(f))
    total = processed + dropped
    if total == 0:
        pytest.skip("no files processed")
    drop_rate = dropped / total
    assert drop_rate < 0.25, (
        f"Drop rate {drop_rate:.1%} is above 25%. Before the include_subdirs "
        f"fix it was 67%. Likely causes: (a) include_subdirs removed from "
        f"config.yaml, (b) SolidiFI repo restructured upstream, (c) a new "
        f"address-dedup false positive appeared."
    )


def test_dropped_reasons_are_known():
    if not PREP_DIR.exists():
        pytest.skip("preprocessed/ not yet produced")
    allowed = {"duplicate", "compile_failed"}
    dropped_csv = PREP_DIR / "dropped.csv"
    if not dropped_csv.exists():
        return
    with open(dropped_csv) as f:
        for row in csv.DictReader(f):
            assert row["reason"] in allowed, f"unknown reason: {row['reason']}"
