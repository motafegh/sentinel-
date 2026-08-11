"""Decode a legacy V2 EZKL proof into a read-only submission bundle.

This helper intentionally performs **no transaction construction and no signing**.
R0 signer isolation requires all writes to cross the policy-signer boundary; the
current V2 proof scope (``legacy_proxy_only_unbound``) is not eligible for an
on-chain verified-audit submission.

The output is useful for:
- inspecting the 138 public signals;
- checking the 128-input / 10-output layout;
- preserving a deterministic proof/calldata evidence bundle;
- later feeding a policy-signer-compatible V3 implementation.

It must never emit ``cast send`` or accept/read a private key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROOF_PATH = Path("zkml/ezkl/proof.json")
BUNDLE_PATH = Path("zkml/ezkl/calldata_bundle_v2.json")

NUM_CLASSES = 10
INPUT_OFFSET = 128
TOTAL_SIGNALS = INPUT_OFFSET + NUM_CLASSES
SCALE = 8192  # EZKL fixed-point scale 2^13; circuit outputs are student scores.
PROOF_SCOPE = "legacy_proxy_only_unbound"
SUBMISSION_ELIGIBLE = False
SUBMISSION_INELIGIBLE_REASON = "proof_scope_not_identity_bound"


def _decode_field_element(hex_str: str) -> int:
    """Decode EZKL's 32-byte little-endian field-element representation."""
    if not isinstance(hex_str, str) or not hex_str:
        raise ValueError("field element must be a non-empty hex string")
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError as exc:
        raise ValueError("field element is not valid hex") from exc
    if len(raw) != 32:
        raise ValueError(f"field element must be 32 bytes, got {len(raw)}")
    return int.from_bytes(raw, byteorder="little")


def _load_proof(path: Path) -> tuple[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"proof.json not found: {path}")
    try:
        proof_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"proof JSON is invalid: {exc}") from exc

    try:
        hex_proof = proof_data["hex_proof"]
        instances = proof_data["instances"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "proof JSON must contain hex_proof and instances[0]"
        ) from exc

    if not isinstance(hex_proof, str) or not hex_proof.startswith("0x"):
        raise ValueError("hex_proof must be a 0x-prefixed hex string")
    try:
        bytes.fromhex(hex_proof[2:])
    except ValueError as exc:
        raise ValueError("hex_proof is not valid hex") from exc

    if not isinstance(instances, list):
        raise ValueError("instances[0] must be a list")
    if len(instances) != TOTAL_SIGNALS:
        raise ValueError(
            f"expected exactly {TOTAL_SIGNALS} public signals "
            f"({INPUT_OFFSET} fusion inputs + {NUM_CLASSES} proxy-score outputs), "
            f"got {len(instances)}"
        )
    return hex_proof, instances


def build_bundle(proof_path: Path = PROOF_PATH) -> dict[str, Any]:
    """Return a deterministic, non-signing representation of a V2 proof."""
    hex_proof, encoded_instances = _load_proof(proof_path)
    public_signals = [_decode_field_element(item) for item in encoded_instances]
    fusion_features = public_signals[:INPUT_OFFSET]
    proxy_score_felts = public_signals[INPUT_OFFSET:]
    proof_bytes = bytes.fromhex(hex_proof[2:])

    return {
        "protocol": "sentinel-zkml-v2",
        "proof_scope": PROOF_SCOPE,
        "submission_eligible": SUBMISSION_ELIGIBLE,
        "submission_ineligible_reason": SUBMISSION_INELIGIBLE_REASON,
        "input_offset": INPUT_OFFSET,
        "num_classes": NUM_CLASSES,
        "total_public_signals": TOTAL_SIGNALS,
        "fixed_point_scale": SCALE,
        "output_semantics": "proxy_score_fixed_point",
        "proof_hex": hex_proof,
        "proof_sha256": hashlib.sha256(proof_bytes).hexdigest(),
        "public_signals": public_signals,
        "fusion_feature_felts": fusion_features,
        "proxy_score_felts": proxy_score_felts,
        "proxy_scores_approx": [value / SCALE for value in proxy_score_felts],
        "warning": (
            "Legacy V2 proves proxy computation only. It does not bind contract, "
            "chain, round, or teacher-model identity and must not be submitted "
            "outside the policy-signer boundary."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, default=PROOF_PATH)
    parser.add_argument("--output", type=Path, default=BUNDLE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        bundle = build_bundle(args.proof)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 68)
    print("SENTINEL legacy V2 proof bundle (READ ONLY)")
    print(f"  public signals : {bundle['total_public_signals']}")
    print(f"  fusion inputs  : {bundle['input_offset']}")
    print(f"  proxy outputs  : {bundle['num_classes']}")
    print(f"  proof scope    : {bundle['proof_scope']}")
    print(f"  eligible       : {bundle['submission_eligible']}")
    print(f"  reason         : {bundle['submission_ineligible_reason']}")
    print(f"  output         : {args.output}")
    print("No transaction or signing script was generated.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
