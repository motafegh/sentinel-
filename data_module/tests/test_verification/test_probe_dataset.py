"""Tests for probe_dataset (Stage 4 Task 4.6)."""
import csv
import json
from pathlib import Path

import pytest

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.probe_dataset import (
    ClassProbeBucket, ProbeDataset, ProbeEntry,
    build_probe_dataset, DEFAULT_N_PER_CLASS,
)
from sentinel_data.verification.probe_trivials import (
    TRIVIAL_POSITIVES, TRIVIAL_NEGATIVE, bccc_class_to_sentinel,
)


# Locate the BCCC review_batches from the repo root
_BCCC_REVIEW_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs" / "legacy" / "bccc_deep_dive"
    / "Phase5_LabelVerification_2026-06-08"
    / "outputs" / "review_batches"
)


def _skip_if_no_bccc():
    if not _BCCC_REVIEW_DIR.exists() or not any(_BCCC_REVIEW_DIR.glob("*.csv")):
        pytest.skip(f"BCCC review_batches not found at {_BCCC_REVIEW_DIR}")


class TestProbeTrivials:
    def test_all_10_classes_have_trivial_positive(self):
        assert set(TRIVIAL_POSITIVES.keys()) == set(class_names()), (
            f"Missing: {set(class_names()) - set(TRIVIAL_POSITIVES.keys())}"
        )

    def test_trivial_positives_contain_keyword(self):
        """Each trivial positive should be syntactically plausible Solidity."""
        for cls, src in TRIVIAL_POSITIVES.items():
            assert "pragma solidity" in src, f"{cls}: missing pragma"
            assert "contract " in src, f"{cls}: missing contract declaration"

    def test_trivial_negative_contains_safe_patterns(self):
        """The trivial negative should look like a safe ERC20."""
        assert "pragma solidity" in TRIVIAL_NEGATIVE
        assert "transfer" in TRIVIAL_NEGATIVE
        # Should NOT contain any of the dangerous patterns
        assert ".call(" not in TRIVIAL_NEGATIVE, "trivial negative must not have raw .call"
        assert "tx.origin" not in TRIVIAL_NEGATIVE, "trivial negative must not have tx.origin"
        assert "block.timestamp" not in TRIVIAL_NEGATIVE, "trivial negative must not have block.timestamp"
        assert "unchecked" not in TRIVIAL_NEGATIVE, "trivial negative must not have unchecked"

    def test_bccc_class_to_sentinel_mapping(self):
        assert bccc_class_to_sentinel("reentrancy") == "Reentrancy"
        assert bccc_class_to_sentinel("calltounknown") == "CallToUnknown"
        assert bccc_class_to_sentinel("denialofservice") == "DenialOfService"
        assert bccc_class_to_sentinel("timestamp") == "Timestamp"
        assert bccc_class_to_sentinel("externalbug") == "ExternalBug"
        assert bccc_class_to_sentinel("gasexception") == "GasException"
        assert bccc_class_to_sentinel("integeruo") == "IntegerUO"
        assert bccc_class_to_sentinel("mishandledexception") == "MishandledException"
        assert bccc_class_to_sentinel("unusedreturn") == "UnusedReturn"
        assert bccc_class_to_sentinel("transactionorderdependence") == "TransactionOrderDependence"
        # case insensitive
        assert bccc_class_to_sentinel("REENTRANCY") == "Reentrancy"
        # unknown returns None
        assert bccc_class_to_sentinel("nonexistent") is None


class TestBuildProbeDataset:
    def test_builds_with_bccc_only(self, tmp_path):
        """Build with BCCC review_batches (no DIVE fallback) — 6 of 10 classes have data."""
        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None,
            n_per_class=40,
            output_dir=out,
            bccc_review_dir=_BCCC_REVIEW_DIR,
        )
        assert isinstance(ds, ProbeDataset)
        # 10 class directories
        for cls in class_names():
            assert (out / cls.lower()).exists(), f"missing {cls.lower()}/"
            assert (out / cls.lower() / "trivial_positive.sol").exists()
            assert (out / cls.lower() / "trivial_negative.sol").exists()

    def test_manifest_json_valid(self, tmp_path):
        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None, n_per_class=40,
            output_dir=out, bccc_review_dir=_BCCC_REVIEW_DIR,
        )
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["schema_version"] == "1"
        assert manifest["n_per_class_target"] == 40
        assert set(manifest["classes"].keys()) == set(class_names())
        for cls, info in manifest["classes"].items():
            assert info["trivial_positive"].endswith("trivial_positive.sol")
            assert info["trivial_negative"].endswith("trivial_negative.sol")

    def test_bccc_reentrancy_has_real_contracts(self, tmp_path):
        """BCCC review_class11_reentrancy.csv has 30 KEEPs — all should land in the probe."""
        _skip_if_no_bccc()
        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None, n_per_class=40,
            output_dir=out, bccc_review_dir=_BCCC_REVIEW_DIR,
        )
        reen_bucket = ds.by_class["Reentrancy"]
        assert reen_bucket.real_entries
        # All real entries should be from BCCC
        assert all(e.source == "bccc" for e in reen_bucket.real_entries)
        assert all(e.verdict == "KEEP" for e in reen_bucket.real_entries)
        # Real files exist
        for e in reen_bucket.real_entries:
            assert e.contract_path.exists()
            content = e.contract_path.read_text()
            assert len(content) > 0

    def test_caps_at_n_per_class(self, tmp_path):
        """If a class has > N KEEPs, only N are taken."""
        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None, n_per_class=5,    # cap at 5
            output_dir=out, bccc_review_dir=_BCCC_REVIEW_DIR,
        )
        for cls, bucket in ds.by_class.items():
            if bucket.real_entries:
                assert len(bucket.real_entries) <= 5

    def test_falls_back_to_dive_for_missing_classes(self, tmp_path):
        """4 classes absent from BCCC (IntegerUO, MishandledException, UnusedReturn,
        TransactionOrderDependence) should fall back to DIVE.

        This test requires real DIVE labels/preprocessed data. Skips otherwise.
        """
        data_dir = Path("data_module/data")
        if not (data_dir / "labels" / "merged").exists() \
                or not (data_dir / "preprocessed" / "dive").exists():
            pytest.skip("DIVE labels/preprocessed not on disk")

        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=data_dir, n_per_class=10,
            output_dir=out, bccc_review_dir=_BCCC_REVIEW_DIR,
        )
        # The 4 DIVE-only classes should have some real entries (DIVE positives)
        for cls in ("IntegerUO", "MishandledException", "UnusedReturn",
                    "TransactionOrderDependence"):
            if not ds.by_class[cls].real_entries:
                continue
            # If they have entries, they should be from DIVE
            assert all(e.source == "dive" for e in ds.by_class[cls].real_entries)

    def test_add_trivial_false_skips_trivials(self, tmp_path):
        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None, n_per_class=5,
            output_dir=out, bccc_review_dir=_BCCC_REVIEW_DIR,
            add_trivial=False,
        )
        for cls, bucket in ds.by_class.items():
            assert bucket.trivial_positive is None
            assert bucket.trivial_negative is None

    def test_dataset_output_dir_default(self, tmp_path, monkeypatch):
        """Default output_dir is data/probe_dataset when no data_dir."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "data" / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None, n_per_class=2,
            bccc_review_dir=_BCCC_REVIEW_DIR,
        )
        assert ds.output_dir == out.resolve()


class TestBuildProbeDatasetUnit:
    """Tests that don't require BCCC review_batches (use tmp_path + fake data)."""

    def test_fake_bccc_review_dir_builds(self, tmp_path):
        """Build a fake BCCC review_batches and verify it works."""
        bccc = tmp_path / "review_batches"
        bccc.mkdir()
        # Write 3 fake KEEP rows for reentrancy
        rows = [
            {
                "id": f"sha_{i:064x}",
                "verdict_s4": "KEEP",
                "confidence": "0.9",
                "notes": "",
                "file": f"reentrancy/sha_{i:064x}.sol",
                "source_snippet": (
                    f"// SPDX-License-Identifier: MIT\n"
                    f"pragma solidity ^0.8.0;\n"
                    f"contract Reentrancy_{i} {{ uint x; }}\n"
                ),
            }
            for i in range(3)
        ]
        with (bccc / "review_class11_reentrancy.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None, n_per_class=40,
            output_dir=out, bccc_review_dir=bccc,
        )
        assert len(ds.by_class["Reentrancy"].real_entries) == 3
        for e in ds.by_class["Reentrancy"].real_entries:
            content = e.contract_path.read_text()
            assert "contract Reentrancy_" in content

    def test_empty_bccc_dir(self, tmp_path):
        """Empty BCCC review_batches dir → all classes get trivials, 0 real."""
        bccc = tmp_path / "empty_review"
        bccc.mkdir()
        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None, n_per_class=40,
            output_dir=out, bccc_review_dir=bccc,
        )
        for cls, bucket in ds.by_class.items():
            assert bucket.real_entries == []
            # Trivials still present
            assert bucket.trivial_positive is not None
            assert bucket.trivial_negative is not None

    def test_n_per_class_zero(self, tmp_path):
        bccc = tmp_path / "review_batches"
        bccc.mkdir()
        out = tmp_path / "probe_dataset"
        ds = build_probe_dataset(
            data_dir=None, n_per_class=0,
            output_dir=out, bccc_review_dir=bccc,
        )
        for bucket in ds.by_class.values():
            assert bucket.real_entries == []


class TestProbeEntry:
    def test_kind_field(self):
        e = ProbeEntry(sha256="x", class_name="Reentrancy", kind="real", source="bccc")
        assert e.kind == "real"

    def test_optional_fields_default_none(self):
        e = ProbeEntry(sha256="x", class_name="Reentrancy", kind="trivial_positive", source="t")
        assert e.tier is None
        assert e.verdict is None
        assert e.confidence is None
