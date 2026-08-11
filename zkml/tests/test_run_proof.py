"""Tests for V2 proof decoding, signal layout, and signer isolation."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTRACT_PATH = ROOT / "zkml/src/ezkl/extract_calldata.py"


def _load_extract_module():
    spec = importlib.util.spec_from_file_location("extract_calldata", EXTRACT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ── Field element encoding / decoding ────────────────────────────────────

def _decode_field_element(hex_str: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_str), byteorder="little")


def test_little_endian_decode_known_vector():
    """Known fixed-point value: 4497 / 8192 ≈ 0.549."""
    hex_str = "9111000000000000000000000000000000000000000000000000000000000000"
    felt = _decode_field_element(hex_str)
    assert felt == 4497
    assert abs(felt / 8192.0 - 0.549) < 0.001


def test_little_endian_decode_zero():
    hex_str = "00" * 32
    assert _decode_field_element(hex_str) == 0


def test_big_endian_would_be_wrong():
    hex_str = "9111000000000000000000000000000000000000000000000000000000000000"
    assert int(hex_str, 16) != _decode_field_element(hex_str)


# ── Public-signal protocol ───────────────────────────────────────────────

INPUT_OFFSET = 128
NUM_CLASSES = 10
TOTAL_SIGNALS = INPUT_OFFSET + NUM_CLASSES


def test_public_signals_layout():
    signals = [0] * TOTAL_SIGNALS
    for i in range(NUM_CLASSES):
        signals[INPUT_OFFSET + i] = (i + 1) * 1000
    proxy_scores = signals[INPUT_OFFSET:]
    assert len(proxy_scores) == NUM_CLASSES
    assert proxy_scores[0] == 1000
    assert proxy_scores[9] == 10000


def test_total_signals_count():
    assert TOTAL_SIGNALS == 138


def test_proof_input_format():
    features = [0.5] * INPUT_OFFSET
    proof_input = {"input_data": [features]}
    assert len(json.loads(json.dumps(proof_input))["input_data"][0]) == INPUT_OFFSET


# ── Read-only calldata bundle containment ────────────────────────────────

def test_extract_calldata_constants_and_policy_scope():
    mod = _load_extract_module()
    assert mod.NUM_CLASSES == 10
    assert mod.INPUT_OFFSET == 128
    assert mod.TOTAL_SIGNALS == 138
    assert mod.SCALE == 8192
    assert mod.PROOF_SCOPE == "legacy_proxy_only_unbound"
    assert mod.SUBMISSION_ELIGIBLE is False
    assert mod.SUBMISSION_INELIGIBLE_REASON == "proof_scope_not_identity_bound"


def test_extract_calldata_decode_strict_length():
    mod = _load_extract_module()
    good = "9111000000000000000000000000000000000000000000000000000000000000"
    assert mod._decode_field_element(good) == 4497

    try:
        mod._decode_field_element("91")
    except ValueError as exc:
        assert "32 bytes" in str(exc)
    else:
        raise AssertionError("short field element must fail closed")


def test_extract_helper_cannot_generate_direct_signing_path():
    """R0 signer isolation: this helper must remain incapable of raw-key writes."""
    source = EXTRACT_PATH.read_text(encoding="utf-8")
    forbidden = [
        "cast send",
        "DEPLOYER_PRIVATE_KEY",
        "SENTINEL_OPERATOR_KEY",
        "--private-key",
        "send_raw_transaction",
        "sign_transaction",
    ]
    for token in forbidden:
        assert token not in source, f"legacy direct-write capability reintroduced: {token}"


def test_build_bundle_is_explicitly_ineligible(tmp_path):
    mod = _load_extract_module()
    # 138 zero-valued 32-byte field elements and a syntactically valid proof.
    proof = {
        "hex_proof": "0xdeadbeef",
        "instances": [["00" * 32 for _ in range(TOTAL_SIGNALS)]],
    }
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof), encoding="utf-8")

    bundle = mod.build_bundle(path)
    assert bundle["submission_eligible"] is False
    assert bundle["proof_scope"] == "legacy_proxy_only_unbound"
    assert bundle["submission_ineligible_reason"] == "proof_scope_not_identity_bound"
    assert bundle["output_semantics"] == "proxy_score_fixed_point"
    assert len(bundle["public_signals"]) == TOTAL_SIGNALS
    assert len(bundle["proxy_score_felts"]) == NUM_CLASSES
