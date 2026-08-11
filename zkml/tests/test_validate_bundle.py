"""Tests for the tracked V2 ZKML artifact-bundle validator."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "zkml/src/ezkl/validate_bundle.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_bundle", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _good_settings():
    return {
        "run_args": {
            "input_scale": 13,
            "input_visibility": "Public",
            "output_visibility": "Public",
            "param_visibility": "Fixed",
            "check_mode": "UNSAFE",
        },
        "model_instance_shapes": [[1, 128], [1, 10]],
        "model_output_scales": [13],
    }


def test_settings_shape_and_visibility_contract():
    mod = _load_module()
    errors, blockers = [], []
    mod.validate_settings(_good_settings(), errors, blockers)
    assert errors == []
    assert "ezkl_check_mode_unsafe" in blockers


def test_wrong_instance_shape_fails():
    mod = _load_module()
    settings = _good_settings()
    settings["model_instance_shapes"] = [[1, 64], [1, 1]]
    errors, blockers = [], []
    mod.validate_settings(settings, errors, blockers)
    assert any("model_instance_shapes" in error for error in errors)


def test_verifier_abi_is_exact():
    mod = _load_module()
    good = [
        {
            "type": "function",
            "name": "verifyProof",
            "inputs": [{"type": "bytes"}, {"type": "uint256[]"}],
            "outputs": [{"type": "bool"}],
        }
    ]
    errors = []
    mod.validate_verifier_abi(good, errors)
    assert errors == []

    bad = [
        {
            "type": "function",
            "name": "verifyProof",
            "inputs": [{"type": "bytes"}, {"type": "bytes32[]"}],
            "outputs": [{"type": "bool"}],
        }
    ]
    errors = []
    mod.validate_verifier_abi(bad, errors)
    assert errors


def test_live_repository_has_one_canonical_verifier_source():
    mod = _load_module()
    errors = []
    mod.validate_single_verifier_source(ROOT, errors)
    assert errors == []


def test_v2_bundle_can_be_structurally_valid_but_never_production_eligible():
    mod = _load_module()
    report = mod.validate_bundle(ROOT)
    # This test intentionally does not force structural validity: local clones
    # may omit large/private setup artifacts. The policy conclusion is invariant.
    assert report["proof_scope"] == "legacy_proxy_only_unbound"
    assert "proof_scope_not_identity_bound" in report["production_blockers"]
    assert report["production_eligible"] is False
