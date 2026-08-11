"""Validate the tracked SENTINEL legacy V2 ZKML artifact bundle.

This is an identity/integrity validator, not a cryptographic proof test. It
checks that the tracked files agree on the executable protocol shape and emits
SHA-256 identities for later local/CI evidence.

The current V2 bundle is deliberately *not* production eligible because:
- proof scope is ``legacy_proxy_only_unbound``; and
- the tracked EZKL settings use ``check_mode=UNSAFE``.

A structurally coherent historical bundle may therefore validate successfully
while still reporting ``production_eligible=false``. Use
``--require-production-eligible`` when a release gate is supposed to reject
that historical state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIM = 128
NUM_CLASSES = 10
TOTAL_SIGNALS = INPUT_DIM + NUM_CLASSES
CIRCUIT_VERSION = "v2.0"
PROOF_SCOPE = "legacy_proxy_only_unbound"
OUTPUT_SEMANTICS = "teacher_probability_regression_v1"

ARTIFACTS = {
    "proxy_checkpoint": Path("zkml/models/proxy_best.pt"),
    "onnx_model": Path("zkml/models/proxy.onnx"),
    "onnx_external_data": Path("zkml/models/proxy.onnx.data"),
    "calibration": Path("zkml/ezkl/calibration.json"),
    "settings": Path("zkml/ezkl/settings.json"),
    "compiled_circuit": Path("zkml/ezkl/model.compiled"),
    "verification_key": Path("zkml/ezkl/verification_key.vk"),
    "verifier_abi": Path("zkml/ezkl/verifier_abi.json"),
    "solidity_verifier": Path("contracts/src/ZKMLVerifier.sol"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def validate_settings(settings: dict[str, Any], errors: list[str], blockers: list[str]) -> None:
    run_args = settings.get("run_args")
    if not isinstance(run_args, dict):
        errors.append("settings.run_args is missing/not an object")
        return

    shapes = settings.get("model_instance_shapes")
    if shapes != [[1, INPUT_DIM], [1, NUM_CLASSES]]:
        errors.append(
            f"settings model_instance_shapes={shapes!r}; expected [[1,{INPUT_DIM}],[1,{NUM_CLASSES}]]"
        )

    if run_args.get("input_visibility") != "Public":
        errors.append("V2 input_visibility must be Public")
    if run_args.get("output_visibility") != "Public":
        errors.append("V2 output_visibility must be Public")
    if run_args.get("param_visibility") != "Fixed":
        errors.append("V2 param_visibility must be Fixed")

    input_scale = run_args.get("input_scale")
    output_scales = settings.get("model_output_scales")
    if input_scale != 13 or output_scales != [13]:
        errors.append(
            f"V2 fixed-point scale mismatch: input_scale={input_scale!r}, "
            f"model_output_scales={output_scales!r}; expected 13/[13]"
        )

    check_mode = run_args.get("check_mode", settings.get("check_mode"))
    if check_mode == "UNSAFE":
        blockers.append("ezkl_check_mode_unsafe")
    elif not isinstance(check_mode, str) or not check_mode:
        errors.append("EZKL check_mode missing")


def validate_verifier_abi(abi: Any, errors: list[str]) -> None:
    if not isinstance(abi, list):
        errors.append("verifier ABI must be an array")
        return
    candidates = [
        item
        for item in abi
        if isinstance(item, dict)
        and item.get("type") == "function"
        and item.get("name") == "verifyProof"
    ]
    if len(candidates) != 1:
        errors.append(f"expected exactly one verifyProof ABI entry, got {len(candidates)}")
        return
    fn = candidates[0]
    inputs = [entry.get("type") for entry in fn.get("inputs", [])]
    outputs = [entry.get("type") for entry in fn.get("outputs", [])]
    if inputs != ["bytes", "uint256[]"] or outputs != ["bool"]:
        errors.append(
            f"verifyProof ABI mismatch: inputs={inputs!r}, outputs={outputs!r}"
        )


def validate_single_verifier_source(root: Path, errors: list[str]) -> None:
    candidates = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "contracts").rglob("ZKMLVerifier.sol")
        if "/lib/" not in f"/{path.as_posix()}/"
    )
    if candidates != ["contracts/src/ZKMLVerifier.sol"]:
        errors.append(
            "canonical verifier ambiguity: expected only contracts/src/ZKMLVerifier.sol, "
            f"found {candidates}"
        )


def validate_bundle(root: Path = REPO_ROOT) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    artifacts: dict[str, Any] = {}

    for name, relative in ARTIFACTS.items():
        path = root / relative
        if not path.exists():
            errors.append(f"missing artifact: {relative.as_posix()}")
            continue
        if not path.is_file():
            errors.append(f"artifact is not a file: {relative.as_posix()}")
            continue
        artifacts[name] = {
            "path": relative.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    settings_path = root / ARTIFACTS["settings"]
    if settings_path.exists():
        value = _load_json(settings_path)
        if isinstance(value, dict):
            validate_settings(value, errors, blockers)
        else:
            errors.append("settings root must be a JSON object")

    abi_path = root / ARTIFACTS["verifier_abi"]
    if abi_path.exists():
        validate_verifier_abi(_load_json(abi_path), errors)

    validate_single_verifier_source(root, errors)

    # Production blockers are explicit even when historical structure is valid.
    blockers.append("proof_scope_not_identity_bound")

    return {
        "schema": "sentinel-zkml-artifact-bundle-v1",
        "protocol": "sentinel-zkml-v2",
        "circuit_version": CIRCUIT_VERSION,
        "proof_scope": PROOF_SCOPE,
        "output_semantics": OUTPUT_SEMANTICS,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
        "total_public_signals": TOTAL_SIGNALS,
        "structurally_valid": not errors,
        "production_eligible": not errors and not blockers,
        "errors": errors,
        "production_blockers": sorted(set(blockers)),
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-production-eligible", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_bundle(args.root.resolve())
    except (OSError, ValueError) as exc:
        print(f"BUNDLE VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if not report["structurally_valid"]:
        return 1
    if args.require_production_eligible and not report["production_eligible"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
