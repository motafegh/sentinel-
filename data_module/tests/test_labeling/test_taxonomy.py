"""Tests for the canonical 10-class taxonomy (Task 3.1)."""
import pytest
import yaml
from pathlib import Path

from sentinel_data.labeling.schema import load_taxonomy, class_names, class_index

# The authoritative order from ml/src/training/trainer.py:CLASS_NAMES
_TRAINER_CLASS_NAMES = [
    "CallToUnknown",              # 0
    "DenialOfService",            # 1
    "ExternalBug",                # 2
    "GasException",               # 3
    "IntegerUO",                  # 4
    "MishandledException",        # 5
    "Reentrancy",                 # 6
    "Timestamp",                  # 7
    "TransactionOrderDependence", # 8
    "UnusedReturn",               # 9
]


class TestTaxonomySchema:
    def test_loads_without_error(self):
        tax = load_taxonomy()
        assert isinstance(tax, dict)

    def test_exactly_10_classes(self):
        tax = load_taxonomy()
        assert tax["num_classes"] == 10
        assert len(tax["classes"]) == 10

    def test_locked_flag_set(self):
        assert load_taxonomy()["locked"] is True

    def test_schema_version_present(self):
        assert "schema_version" in load_taxonomy()

    def test_each_class_has_required_fields(self):
        for cls in load_taxonomy()["classes"]:
            assert "id" in cls, f"Missing 'id' in {cls}"
            assert "name" in cls, f"Missing 'name' in {cls}"
            assert "description" in cls, f"Missing 'description' in {cls}"
            assert "severity" in cls, f"Missing 'severity' in {cls}"

    def test_ids_are_sequential_0_to_9(self):
        ids = [c["id"] for c in load_taxonomy()["classes"]]
        assert ids == list(range(10))


class TestClassOrder:
    def test_class_names_match_trainer_exactly(self):
        """The order must match ml/src/training/trainer.py:CLASS_NAMES — LOCKED."""
        assert class_names() == _TRAINER_CLASS_NAMES

    def test_class_index_matches_trainer_position(self):
        for expected_idx, name in enumerate(_TRAINER_CLASS_NAMES):
            assert class_index(name) == expected_idx, (
                f"{name} should be index {expected_idx}, got {class_index(name)}"
            )

    def test_unknown_class_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown class"):
            class_index("NonExistent")

    def test_order_matches_multilabel_csv_columns(self):
        """Cross-check against the actual CSV column order used in training."""
        csv_path = Path(__file__).parents[3] / "ml/data/processed/multilabel_index.csv"
        if not csv_path.exists():
            pytest.skip("multilabel_index.csv not found — skipping CSV cross-check")
        header = csv_path.read_text().split("\n")[0]
        csv_cols = [c.strip() for c in header.split(",")]
        # First column is md5_stem, rest are class names
        csv_class_cols = csv_cols[1:]
        assert csv_class_cols == _TRAINER_CLASS_NAMES, (
            f"CSV column order differs from taxonomy:\n"
            f"  CSV:      {csv_class_cols}\n"
            f"  Taxonomy: {_TRAINER_CLASS_NAMES}"
        )
