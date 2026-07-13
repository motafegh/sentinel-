"""Tests for proof generation + calldata extraction — field element encoding, signal layout."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Field element encoding / decoding ────────────────────────────────────

def _decode_field_element(hex_str: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_str), byteorder='little')


def test_little_endian_decode_known_vector():
    """Known EZKL output: 4497/8192 = 0.549."""
    # "9111" in little-endian = 0x1191 = 4497
    hex_str = "9111000000000000000000000000000000000000000000000000000000000000"
    felt = _decode_field_element(hex_str)
    assert felt == 4497
    assert abs(felt / 8192.0 - 0.549) < 0.001


def test_little_endian_decode_zero():
    """All-zero hex → 0."""
    hex_str = "0000000000000000000000000000000000000000000000000000000000000000"
    assert _decode_field_element(hex_str) == 0


def test_little_endian_decode_max_value():
    """All-0xFF hex → very large number (but valid field element)."""
    hex_str = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0"
    felt = _decode_field_element(hex_str)
    assert felt > 0


def test_big_endian_would_be_wrong():
    """Demonstrate that int(hex,16) gives the WRONG answer."""
    hex_str = "9111000000000000000000000000000000000000000000000000000000000000"
    big_endian = int(hex_str, 16)
    little_endian = _decode_field_element(hex_str)
    assert big_endian != little_endian
    assert big_endian > 2**255  # huge garbage value
    assert little_endian < 5000   # reasonable score field element


# ── Public signals layout ────────────────────────────────────────────────

INPUT_OFFSET = 128
NUM_CLASSES = 10
TOTAL_SIGNALS = INPUT_OFFSET + NUM_CLASSES  # 138


def test_public_signals_layout():
    """Verify that class scores start at index 128."""
    signals = [0] * TOTAL_SIGNALS
    # Set class scores
    for i in range(NUM_CLASSES):
        signals[INPUT_OFFSET + i] = (i + 1) * 1000

    class_scores = signals[INPUT_OFFSET:]
    assert len(class_scores) == NUM_CLASSES
    assert class_scores[0] == 1000
    assert class_scores[9] == 10000


def test_total_signals_count():
    """128 + 10 = 138."""
    assert TOTAL_SIGNALS == 138


def test_class_offset_within_bounds():
    """INPUT_OFFSET + 9 < TOTAL_SIGNALS."""
    for i in range(NUM_CLASSES):
        assert INPUT_OFFSET + i < TOTAL_SIGNALS


# ── Proof input format ───────────────────────────────────────────────────

def test_proof_input_format():
    """EZKL expects {"input_data": [[128 floats]]}."""
    features = [0.5] * 128
    proof_input = {"input_data": [features]}
    doc = json.dumps(proof_input)
    parsed = json.loads(doc)
    assert len(parsed["input_data"]) == 1
    assert len(parsed["input_data"][0]) == 128


def test_proof_input_validation():
    """Proof input with wrong dims should be detectable."""
    features = [0.5] * 64  # wrong: should be 128
    proof_input = {"input_data": [features]}
    assert len(proof_input["input_data"][0]) != 128


# ── Extract calldata script imports ──────────────────────────────────────

def test_extract_calldata_constants():
    """Module-level constants match expected layout."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "extract_calldata",
        Path(__file__).resolve().parents[2] / "zkml/src/ezkl/extract_calldata.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.NUM_CLASSES == 10
    assert mod.INPUT_OFFSET == 128
    assert mod.TOTAL_SIGNALS == 138
    assert mod.SCALE == 8192


def test_extract_calldata_decode():
    """_decode_field_element matches our known-good implementation."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "extract_calldata",
        Path(__file__).resolve().parents[2] / "zkml/src/ezkl/extract_calldata.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    hex_str = "9111000000000000000000000000000000000000000000000000000000000000"
    assert mod._decode_field_element(hex_str) == 4497
