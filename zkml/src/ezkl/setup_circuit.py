"""Build a versioned EZKL circuit bundle from explicitly bound artifacts.

This setup path is fail-closed on lineage. It will not generate settings,
compiled circuit, keys, or a future Solidity verifier from anonymous files that
merely happen to live under ``zkml/``.

Required inputs:
- ``proxy.onnx`` + ``proxy.onnx.manifest.json``;
- ``calibration.json`` + ``calibration.manifest.json``;
- matching teacher-checkpoint and DATA-export identities across both manifests.

Important protocol rule: this script makes **no assumption** that a verification
key or compiled circuit remains valid after proxy retraining/weight changes.
Any newly exported ONNX artifact is treated as a new setup input identity and
must be regenerated/validated as a versioned bundle unless an explicitly
approved compatibility procedure proves otherwise.

The legacy tracked V2 artifacts remain historical evidence and are inspected by
``validate_bundle.py``. This script is for producing a newly bound setup bundle.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[3]

ONNX_MODEL = Path("zkml/models/proxy.onnx")
ONNX_MANIFEST = Path("zkml/models/proxy.onnx.manifest.json")
CALIBRATION = Path("zkml/ezkl/calibration.json")
CALIBRATION_MANIFEST = Path("zkml/ezkl/calibration.manifest.json")
SETTINGS = Path("zkml/ezkl/settings.json")
COMPILED = Path("zkml/ezkl/model.compiled")
SRS = Path("zkml/ezkl/srs.params")
PROVING_KEY = Path("zkml/ezkl/proving_key.pk")
VERIFICATION_KEY = Path("zkml/ezkl/verification_key.vk")
SETUP_MANIFEST = Path("zkml/ezkl/setup.manifest.json")

EXPECTED_INPUT_DIM = 128
EXPECTED_NUM_CLASSES = 10
EXPECTED_OUTPUT_SEMANTICS = "teacher_probability_regression_v1"
EXPECTED_CIRCUIT_VERSION = "v2.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _require_hash_entry(
    manifest: dict[str, Any],
    path: list[str],
    *,
    field_name: str,
) -> str:
    value: Any = manifest
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{field_name} missing from manifest")
        value = value[key]
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a 64-hex SHA-256 string")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be valid hex") from exc
    return value.lower()


def validate_lineage(
    *,
    root: Path = REPO_ROOT,
    onnx_path: Path = ONNX_MODEL,
    onnx_manifest_path: Path = ONNX_MANIFEST,
    calibration_path: Path = CALIBRATION,
    calibration_manifest_path: Path = CALIBRATION_MANIFEST,
) -> dict[str, Any]:
    """Validate that ONNX + calibration describe one teacher/DATA lineage."""
    absolute_onnx = root / onnx_path
    absolute_onnx_manifest = root / onnx_manifest_path
    absolute_calibration = root / calibration_path
    absolute_calibration_manifest = root / calibration_manifest_path

    for label, path in {
        "ONNX model": absolute_onnx,
        "ONNX manifest": absolute_onnx_manifest,
        "calibration": absolute_calibration,
        "calibration manifest": absolute_calibration_manifest,
    }.items():
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")

    onnx_manifest = load_json(absolute_onnx_manifest)
    calibration_manifest = load_json(absolute_calibration_manifest)

    if onnx_manifest.get("circuit_version") != EXPECTED_CIRCUIT_VERSION:
        raise ValueError(
            f"ONNX circuit_version={onnx_manifest.get('circuit_version')!r}; "
            f"expected {EXPECTED_CIRCUIT_VERSION!r}"
        )
    if onnx_manifest.get("output_semantics") != EXPECTED_OUTPUT_SEMANTICS:
        raise ValueError(
            f"ONNX output_semantics={onnx_manifest.get('output_semantics')!r}; "
            f"expected {EXPECTED_OUTPUT_SEMANTICS!r}"
        )
    if onnx_manifest.get("input_dim") != EXPECTED_INPUT_DIM:
        raise ValueError("ONNX manifest input_dim must be 128")
    if onnx_manifest.get("num_classes") != EXPECTED_NUM_CLASSES:
        raise ValueError("ONNX manifest num_classes must be 10")
    if calibration_manifest.get("input_dim") != EXPECTED_INPUT_DIM:
        raise ValueError("calibration manifest input_dim must be 128")

    onnx_sha = _require_hash_entry(
        onnx_manifest, ["onnx", "sha256"], field_name="onnx.sha256"
    )
    if sha256_file(absolute_onnx) != onnx_sha:
        raise RuntimeError("ONNX file hash does not match ONNX manifest")

    calibration_sha = _require_hash_entry(
        calibration_manifest,
        ["calibration", "sha256"],
        field_name="calibration.sha256",
    )
    if sha256_file(absolute_calibration) != calibration_sha:
        raise RuntimeError(
            "calibration file hash does not match calibration manifest"
        )

    checkpoint_metadata = (
        onnx_manifest.get("checkpoint", {}).get("metadata", {})
        if isinstance(onnx_manifest.get("checkpoint"), dict)
        else {}
    )
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("ONNX checkpoint.metadata must be an object")

    teacher_from_onnx = checkpoint_metadata.get("teacher_checkpoint_sha256")
    export_from_onnx = checkpoint_metadata.get("export_manifest_sha256")
    if not isinstance(teacher_from_onnx, str) or len(teacher_from_onnx) != 64:
        raise ValueError(
            "ONNX checkpoint metadata lacks bound teacher_checkpoint_sha256; "
            "legacy anonymous checkpoints must be re-exported through the "
            "versioned distillation workflow before circuit setup"
        )
    if not isinstance(export_from_onnx, str) or len(export_from_onnx) != 64:
        raise ValueError(
            "ONNX checkpoint metadata lacks bound export_manifest_sha256"
        )

    teacher_from_calibration = _require_hash_entry(
        calibration_manifest,
        ["teacher_checkpoint", "sha256"],
        field_name="calibration.teacher_checkpoint.sha256",
    )
    export_from_calibration = _require_hash_entry(
        calibration_manifest,
        ["data_export", "manifest_sha256"],
        field_name="calibration.data_export.manifest_sha256",
    )

    if teacher_from_onnx.lower() != teacher_from_calibration:
        raise RuntimeError(
            "teacher lineage mismatch between ONNX and calibration manifests"
        )
    if export_from_onnx.lower() != export_from_calibration:
        raise RuntimeError(
            "DATA export lineage mismatch between ONNX and calibration manifests"
        )

    external = onnx_manifest.get("onnx_external_data")
    external_identity: dict[str, Any] | None = None
    if external is not None:
        if not isinstance(external, dict):
            raise ValueError("onnx_external_data must be null or an object")
        external_path_raw = external.get("path")
        external_sha_raw = external.get("sha256")
        if not isinstance(external_path_raw, str) or not external_path_raw:
            raise ValueError("onnx_external_data.path missing")
        if not isinstance(external_sha_raw, str) or len(external_sha_raw) != 64:
            raise ValueError("onnx_external_data.sha256 invalid")
        external_path = root / Path(external_path_raw)
        if not external_path.exists():
            raise FileNotFoundError(f"ONNX external-data file missing: {external_path}")
        if sha256_file(external_path) != external_sha_raw.lower():
            raise RuntimeError("ONNX external-data hash mismatch")
        external_identity = {
            "path": Path(external_path_raw).as_posix(),
            "sha256": external_sha_raw.lower(),
        }

    return {
        "schema": "sentinel-zkml-setup-input-lineage-v1",
        "circuit_version": EXPECTED_CIRCUIT_VERSION,
        "output_semantics": EXPECTED_OUTPUT_SEMANTICS,
        "teacher_checkpoint_sha256": teacher_from_calibration,
        "data_export_manifest_sha256": export_from_calibration,
        "onnx": {"path": onnx_path.as_posix(), "sha256": onnx_sha},
        "onnx_external_data": external_identity,
        "calibration": {
            "path": calibration_path.as_posix(),
            "sha256": calibration_sha,
        },
        "onnx_manifest": {
            "path": onnx_manifest_path.as_posix(),
            "sha256": sha256_file(absolute_onnx_manifest),
        },
        "calibration_manifest": {
            "path": calibration_manifest_path.as_posix(),
            "sha256": sha256_file(absolute_calibration_manifest),
        },
    }


async def _download_srs(ezkl: Any, *, settings: Path, srs: Path) -> None:
    await ezkl.get_srs(settings_path=str(settings), srs_path=str(srs))


def _artifact_identity(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"expected generated artifact missing/empty: {path}")
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_pipeline(
    *,
    root: Path = REPO_ROOT,
    setup_manifest_path: Path = SETUP_MANIFEST,
) -> dict[str, Any]:
    """Generate settings/circuit/SRS/keys and return their bound manifest."""
    lineage = validate_lineage(root=root)

    try:
        import ezkl  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "EZKL Python package is required for circuit setup; use the pinned "
            "SENTINEL ZKML environment"
        ) from exc

    onnx = root / ONNX_MODEL
    calibration = root / CALIBRATION
    settings = root / SETTINGS
    compiled = root / COMPILED
    srs = root / SRS
    proving_key = root / PROVING_KEY
    verification_key = root / VERIFICATION_KEY
    output_manifest = root / setup_manifest_path

    settings.parent.mkdir(parents=True, exist_ok=True)

    logger.info("EZKL setup input lineage verified: {}", lineage)

    result = ezkl.gen_settings(model=str(onnx), output=str(settings))
    if not result:
        raise RuntimeError("ezkl.gen_settings failed")

    result = ezkl.calibrate_settings(
        data=str(calibration),
        model=str(onnx),
        settings=str(settings),
        target="resources",
    )
    if result is not None and not result:
        raise RuntimeError("ezkl.calibrate_settings failed")
    _artifact_identity(settings)

    result = ezkl.compile_circuit(
        model=str(onnx),
        compiled_circuit=str(compiled),
        settings_path=str(settings),
    )
    if not result:
        raise RuntimeError("ezkl.compile_circuit failed")
    _artifact_identity(compiled)

    asyncio.run(_download_srs(ezkl, settings=settings, srs=srs))
    _artifact_identity(srs)

    result = ezkl.setup(
        model=str(compiled),
        vk_path=str(verification_key),
        pk_path=str(proving_key),
        srs_path=str(srs),
    )
    if not result:
        raise RuntimeError("ezkl.setup failed")

    setup_manifest: dict[str, Any] = {
        "schema": "sentinel-zkml-setup-bundle-v1",
        "circuit_version": EXPECTED_CIRCUIT_VERSION,
        "output_semantics": EXPECTED_OUTPUT_SEMANTICS,
        "input_lineage": lineage,
        "ezkl_version_from_settings": None,
        "artifacts": {
            "settings": _artifact_identity(settings),
            "compiled_circuit": _artifact_identity(compiled),
            "srs": _artifact_identity(srs),
            "proving_key": _artifact_identity(proving_key),
            "verification_key": _artifact_identity(verification_key),
        },
        "compatibility_policy": (
            "Any new ONNX identity requires a newly generated and validated "
            "setup bundle unless an explicitly approved compatibility procedure "
            "demonstrates artifact reuse."
        ),
    }

    settings_value = load_json(settings)
    version = settings_value.get("version")
    if isinstance(version, str):
        setup_manifest["ezkl_version_from_settings"] = version

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(setup_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Versioned EZKL setup bundle written: {}", output_manifest)
    return setup_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=SETUP_MANIFEST)
    parser.add_argument(
        "--validate-lineage-only",
        action="store_true",
        help="Validate ONNX/calibration lineage without importing or running EZKL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.validate_lineage_only:
        report = validate_lineage(root=root)
    else:
        report = run_pipeline(root=root, setup_manifest_path=args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
