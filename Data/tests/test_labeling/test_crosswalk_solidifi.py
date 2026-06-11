"""Structural validation of the SolidiFI crosswalk YAML (Task 3.2)."""
import yaml
from pathlib import Path

from sentinel_data.labeling.schema import class_names

_CROSSWALK_PATH = (
    Path(__file__).parents[2]
    / "sentinel_data/labeling/crosswalks/solidifi.yaml"
)

_SOLIDIFI_FOLDERS = {
    "Re-entrancy",
    "Timestamp-Dependency",
    "Unhandled-Exceptions",
    "TOD",
    "Overflow-Underflow",
    "Unchecked-Send",
    "tx.origin",
}


def _load():
    with open(_CROSSWALK_PATH) as f:
        return yaml.safe_load(f)


class TestSolidiFICrosswalk:
    def test_loads_without_error(self):
        assert _load() is not None

    def test_required_top_level_fields(self):
        cw = _load()
        for field in ("schema_version", "source", "taxonomy_version",
                      "label_field", "confidence_tier", "class_map"):
            assert field in cw, f"Missing field: {field}"

    def test_source_is_solidifi(self):
        assert _load()["source"] == "solidifi"

    def test_confidence_tier_is_T0(self):
        assert _load()["confidence_tier"] == "T0"

    def test_label_field_is_original_path(self):
        assert _load()["label_field"] == "original_path"

    def test_all_seven_folders_are_mapped(self):
        cw = _load()
        mapped = set(cw["class_map"].keys())
        assert mapped == _SOLIDIFI_FOLDERS, (
            f"Missing folders: {_SOLIDIFI_FOLDERS - mapped}\n"
            f"Extra folders:   {mapped - _SOLIDIFI_FOLDERS}"
        )

    def test_all_target_classes_exist_in_taxonomy(self):
        cw = _load()
        valid = set(class_names())
        for folder, target in cw["class_map"].items():
            assert target in valid, (
                f"Folder '{folder}' maps to '{target}' which is not in taxonomy. "
                f"Valid: {sorted(valid)}"
            )

    def test_tx_origin_maps_to_external_bug(self):
        assert _load()["class_map"]["tx.origin"] == "ExternalBug"

    def test_unchecked_send_maps_to_call_to_unknown(self):
        assert _load()["class_map"]["Unchecked-Send"] == "CallToUnknown"

    def test_notes_field_present(self):
        cw = _load()
        assert "notes" in cw
        assert len(cw["notes"]) >= 1
