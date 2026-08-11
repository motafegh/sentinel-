"""Dependency-light tests for EZKL setup lineage validation."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "zkml/src/ezkl/setup_circuit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("setup_circuit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path):
    mod = _load_module()
    onnx = tmp_path / "zkml/models/proxy.onnx"
    calibration = tmp_path / "zkml/ezkl/calibration.json"
    onnx_manifest = tmp_path / "zkml/models/proxy.onnx.manifest.json"
    calibration_manifest = tmp_path / "zkml/ezkl/calibration.manifest.json"
    onnx.parent.mkdir(parents=True, exist_ok=True)
    calibration.parent.mkdir(parents=True, exist_ok=True)
    onnx.write_bytes(b"onnx-fixture")
    calibration.write_text('{"input_data":[[0.0]]}', encoding="utf-8")

    teacher = "11" * 32
    export = "22" * 32
    _write_json(
        onnx_manifest,
        {
            "circuit_version": "v2.0",
            "output_semantics": "teacher_probability_regression_v1",
            "input_dim": 128,
            "num_classes": 10,
            "checkpoint": {
                "metadata": {
                    "teacher_checkpoint_sha256": teacher,
                    "export_manifest_sha256": export,
                }
            },
            "onnx": {"sha256": mod.sha256_file(onnx)},
            "onnx_external_data": None,
        },
    )
    _write_json(
        calibration_manifest,
        {
            "input_dim": 128,
            "teacher_checkpoint": {"sha256": teacher},
            "data_export": {"manifest_sha256": export},
            "calibration": {"sha256": mod.sha256_file(calibration)},
        },
    )
    return mod, onnx, calibration, onnx_manifest, calibration_manifest


def _validate(mod, tmp_path, onnx, calibration, onnx_manifest, calibration_manifest):
    return mod.validate_lineage(
        root=tmp_path,
        onnx_path=onnx.relative_to(tmp_path),
        onnx_manifest_path=onnx_manifest.relative_to(tmp_path),
        calibration_path=calibration.relative_to(tmp_path),
        calibration_manifest_path=calibration_manifest.relative_to(tmp_path),
    )


def test_matching_lineage_passes(tmp_path):
    mod, onnx, calibration, onnx_manifest, calibration_manifest = _fixture(tmp_path)
    report = _validate(
        mod, tmp_path, onnx, calibration, onnx_manifest, calibration_manifest
    )
    assert report["teacher_checkpoint_sha256"] == "11" * 32
    assert report["data_export_manifest_sha256"] == "22" * 32
    assert report["onnx"]["sha256"] == mod.sha256_file(onnx)
    assert report["calibration"]["sha256"] == mod.sha256_file(calibration)


def test_onnx_byte_drift_is_rejected(tmp_path):
    mod, onnx, calibration, onnx_manifest, calibration_manifest = _fixture(tmp_path)
    onnx.write_bytes(b"changed-after-manifest")
    with pytest.raises(RuntimeError, match="ONNX file hash"):
        _validate(mod, tmp_path, onnx, calibration, onnx_manifest, calibration_manifest)


def test_calibration_byte_drift_is_rejected(tmp_path):
    mod, onnx, calibration, onnx_manifest, calibration_manifest = _fixture(tmp_path)
    calibration.write_text('{"input_data":[[1.0]]}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="calibration file hash"):
        _validate(mod, tmp_path, onnx, calibration, onnx_manifest, calibration_manifest)


def test_teacher_lineage_mismatch_is_rejected(tmp_path):
    mod, onnx, calibration, onnx_manifest, calibration_manifest = _fixture(tmp_path)
    value = json.loads(calibration_manifest.read_text())
    value["teacher_checkpoint"]["sha256"] = "33" * 32
    _write_json(calibration_manifest, value)
    with pytest.raises(RuntimeError, match="teacher lineage mismatch"):
        _validate(mod, tmp_path, onnx, calibration, onnx_manifest, calibration_manifest)


def test_data_export_lineage_mismatch_is_rejected(tmp_path):
    mod, onnx, calibration, onnx_manifest, calibration_manifest = _fixture(tmp_path)
    value = json.loads(calibration_manifest.read_text())
    value["data_export"]["manifest_sha256"] = "44" * 32
    _write_json(calibration_manifest, value)
    with pytest.raises(RuntimeError, match="DATA export lineage mismatch"):
        _validate(mod, tmp_path, onnx, calibration, onnx_manifest, calibration_manifest)


def test_anonymous_legacy_checkpoint_metadata_is_rejected(tmp_path):
    mod, onnx, calibration, onnx_manifest, calibration_manifest = _fixture(tmp_path)
    value = json.loads(onnx_manifest.read_text())
    value["checkpoint"]["metadata"] = {"schema": "legacy_raw_state_dict"}
    _write_json(onnx_manifest, value)
    with pytest.raises(ValueError, match="teacher_checkpoint_sha256"):
        _validate(mod, tmp_path, onnx, calibration, onnx_manifest, calibration_manifest)
