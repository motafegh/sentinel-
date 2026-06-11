"""Structural validation of the DIVE crosswalk YAML (Task 3.4 — DIVE)."""
import yaml
from pathlib import Path

from sentinel_data.labeling.schema import class_names

_CROSSWALK_PATH = (
    Path(__file__).parents[2]
    / "sentinel_data/labeling/crosswalks/dive.yaml"
)

_DIVE_FOLDERS = {
    "Reentrancy",
    "DoS",
    "Arithmetic",
    "Time manipulation",
    "Front Running",
    "Access Control",
    "Unchecked Return Values",
    # "Bad Randomness" is intentionally absent — DROPPED
}


def _load():
    with open(_CROSSWALK_PATH) as f:
        return yaml.safe_load(f)


class TestDiveCrosswalk:
    def test_loads_without_error(self):
        assert _load() is not None

    def test_required_top_level_fields(self):
        cw = _load()
        for field in ("schema_version", "source", "taxonomy_version",
                      "label_field", "confidence_tier", "class_map"):
            assert field in cw, f"Missing field: {field}"

    def test_source_is_dive(self):
        assert _load()["source"] == "dive"

    def test_confidence_tier_is_T2(self):
        assert _load()["confidence_tier"] == "T2"

    def test_label_field_is_original_path(self):
        assert _load()["label_field"] == "original_path"

    def test_all_seven_mapped_folders_present(self):
        cw = _load()
        mapped = set(cw["class_map"].keys())
        assert mapped == _DIVE_FOLDERS, (
            f"Missing folders: {_DIVE_FOLDERS - mapped}\n"
            f"Extra folders:   {mapped - _DIVE_FOLDERS}"
        )

    def test_bad_randomness_not_in_class_map(self):
        """Bad Randomness must be dropped — not silently mapped."""
        cw = _load()
        assert "Bad Randomness" not in cw["class_map"], (
            "Bad Randomness must not appear in class_map — it is DROPPED (no canonical class)"
        )

    def test_all_target_classes_exist_in_taxonomy(self):
        cw = _load()
        valid = set(class_names())
        for folder, target in cw["class_map"].items():
            assert target in valid, (
                f"Folder '{folder}' maps to '{target}' which is not in taxonomy."
            )

    def test_unchecked_return_values_maps_to_unused_return(self):
        """DIVE Unchecked Return Values → UnusedReturn, not CallToUnknown."""
        assert _load()["class_map"]["Unchecked Return Values"] == "UnusedReturn"

    def test_access_control_maps_to_external_bug(self):
        assert _load()["class_map"]["Access Control"] == "ExternalBug"

    def test_bad_randomness_drop_documented_in_notes(self):
        notes = " ".join(_load().get("notes", []))
        assert "Bad Randomness" in notes and "DROPPED" in notes

    def test_notes_field_present(self):
        cw = _load()
        assert "notes" in cw and len(cw["notes"]) >= 1
