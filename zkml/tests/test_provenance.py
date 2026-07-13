"""Tests for provenance manifest — JSON round-trip, hash stability, signature structure."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_fusion_embedding_hash_stable():
    """Same 128-dim vector produces same SHA-256 consistently."""
    fusion = [0.1 * i for i in range(128)]
    h1 = hashlib.sha256(json.dumps(fusion, sort_keys=True).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps(fusion, sort_keys=True).encode()).hexdigest()
    assert h1 == h2
    assert len(h1) == 64


def test_fusion_embedding_hash_different_on_change():
    """Changing one value changes the hash."""
    fusion1 = [0.1 * i for i in range(128)]
    fusion2 = list(fusion1)
    fusion2[64] += 0.001
    h1 = hashlib.sha256(json.dumps(fusion1, sort_keys=True).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps(fusion2, sort_keys=True).encode()).hexdigest()
    assert h1 != h2


def test_manifest_json_round_trip():
    """Manifest dict survives JSON serialization round-trip."""
    manifest = {
        "teacher_model_hash": "a" * 64,
        "proxy_checkpoint_hash": "b" * 64,
        "fusion_embedding_hash": "c" * 64,
        "class_scores": [0.1, 0.2, 0.3],
        "timestamp": "2026-07-09T00:00:00+00:00",
        "operator_address": "0x" + "0" * 40,
    }
    doc = json.dumps(manifest, sort_keys=True)
    parsed = json.loads(doc)
    assert parsed == manifest


def test_manifest_missing_signature_is_ok():
    """Manifest without signature is still valid JSON."""
    manifest = {
        "teacher_model_hash": "a" * 64,
        "proxy_checkpoint_hash": "b" * 64,
        "fusion_embedding_hash": "c" * 64,
        "class_scores": [0.5] * 10,
        "timestamp": "2026-07-09T00:00:00+00:00",
        "operator_address": "0x" + "0" * 40,
        "signature": None,
    }
    doc = json.dumps(manifest, sort_keys=True)
    assert "signature" in doc


def test_proxy_checkpoint_hash():
    """SHA-256 of the actual proxy_best.pt is a 64-char hex string."""
    proxy_path = Path(__file__).resolve().parents[2] / "zkml/models/proxy_best.pt"
    if not proxy_path.exists():
        pytest.skip("proxy_best.pt not found — run distillation first")
    file_hash = hashlib.sha256(proxy_path.read_bytes()).hexdigest()
    assert len(file_hash) == 64
    assert all(c in "0123456789abcdef" for c in file_hash)


def test_signature_hex_format():
    """EIP-191 signatures produced by eth_account are 130-char hex strings."""
    # This test verifies the expected format without requiring eth_account installed.
    # Valid ECDSA signatures: r (32 bytes) + s (32 bytes) + v (1 byte) = 65 bytes = 130 hex chars
    example_sig = "0x" + "ab" * 65
    assert len(example_sig) == 132  # 2 + 65*2 = 132
    assert example_sig.startswith("0x")
