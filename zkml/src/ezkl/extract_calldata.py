"""
extract_calldata.py — Extract on-chain calldata from proof.json

Reads zkml/ezkl/proof.json and outputs the exact arguments needed
to call AuditRegistry.submitAuditV2() on-chain via cast.

Usage:
    cd ~/projects/sentinel
    source ml/.venv/bin/activate
    python zkml/src/ezkl/extract_calldata.py

Output:
    - Prints human-readable summary
    - Writes check_verify.sh   (cast call → verifyProof on ZKMLVerifier)
    - Writes submit_audit.sh   (cast send → submitAuditV2 on AuditRegistry)

ENCODING REFERENCE — BN254 field elements:
    EZKL stores every public signal as a 32-byte little-endian hex string
    in proof.json["instances"][0].
    Little-endian LEAST significant byte first → must convert explicitly.
    CORRECT: int.from_bytes(bytes.fromhex(hex_str), byteorder='little')
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

PROOF_PATH = Path("zkml/ezkl/proof.json")

# Sepolia deployment addresses (update after each re-deploy)
ZKML_VERIFIER = "0xB7093Be4958dd95438D6f53Ff7DF8659451CbD97"
AUDIT_REGISTRY = "0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf"
RPC_URL = "https://sepolia.infura.io/v3/31876fad90e24857ab6751fb214da7b9"

AUDIT_TARGET = "0x000000000000000000000000000000000000dEaD"
MODEL_HASH   = "0x0000000000000000000000000000000000000000000000000000000000000000"

NUM_CLASSES   = 10
INPUT_OFFSET  = 128   # first 128 publicSignals are fusion features
TOTAL_SIGNALS = INPUT_OFFSET + NUM_CLASSES  # 138
SCALE         = 8192  # 2^13


def _decode_field_element(hex_str: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_str), byteorder='little')


def main() -> None:
    if not PROOF_PATH.exists():
        print(f"ERROR: proof.json not found at {PROOF_PATH}", file=sys.stderr)
        print("Run: python zkml/src/ezkl/run_proof.py", file=sys.stderr)
        sys.exit(1)

    try:
        proof_data = json.loads(PROOF_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: proof.json is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        hex_proof = proof_data["hex_proof"]
        instances = proof_data["instances"][0]
    except (KeyError, IndexError, TypeError) as e:
        print(
            f"ERROR: proof.json missing expected structure: {e}\n"
            f"Expected keys: hex_proof, instances[0]\n"
            f"Regenerate the proof with run_proof.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(instances) != TOTAL_SIGNALS:
        print(
            f"ERROR: expected {TOTAL_SIGNALS} public signals "
            f"({INPUT_OFFSET} features + {NUM_CLASSES} scores), "
            f"got {len(instances)}.\n"
            f"The proof may have been generated with a different circuit version.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Decode all field elements
    public_signals = [_decode_field_element(h) for h in instances]

    # Split: [0..127] = fusion features, [128..137] = class scores
    fusion_features = public_signals[:INPUT_OFFSET]
    class_scores    = public_signals[INPUT_OFFSET:]

    print("=" * 60)
    print(f"CALLDATA FOR AuditRegistry.submitAuditV2()")
    print(f"  Circuit signals: {len(public_signals)} total")
    print(f"  Fusion features: {INPUT_OFFSET}-dim (indices 0-{INPUT_OFFSET - 1})")
    print(f"  Class scores:    {NUM_CLASSES} outputs (indices {INPUT_OFFSET}-{TOTAL_SIGNALS - 1})")
    print("=" * 60)
    print(f"  hex_proof (first 20 chars): {hex_proof[:20]}...")
    print(f"  Class scores (felt → human):")
    for i in range(NUM_CLASSES):
        print(f"    [{i}] felt={class_scores[i]:>6d}  human={class_scores[i] / SCALE:.4f}")
    print("=" * 60)

    # Build cast arguments
    signals_str     = "[" + ",".join(str(s) for s in public_signals) + "]"
    class_scores_str = "[" + ",".join(str(s) for s in class_scores) + "]"

    # check_verify.sh
    verify_lines = [
        "cast call \\",
        f"  {ZKML_VERIFIER} \\",
        "  'verifyProof(bytes,uint256[])(bool)' \\",
        f"  {hex_proof} \\",
        f"  '{signals_str}' \\",
        f"  --rpc-url {RPC_URL}",
    ]
    Path("check_verify.sh").write_text("\n".join(verify_lines))
    print("check_verify.sh written  — run to confirm proof is valid on-chain")

    # submit_audit.sh  (V2)
    submit_lines = [
        "cast send \\",
        "  --private-key $DEPLOYER_PRIVATE_KEY \\",
        f"  --rpc-url {RPC_URL} \\",
        f"  {AUDIT_REGISTRY} \\",
        "  'submitAuditV2(address,uint256[10],bytes,uint256[],bytes32)' \\",
        f"  {AUDIT_TARGET} \\",
        f"  '{class_scores_str}' \\",
        f"  {hex_proof} \\",
        f"  '{signals_str}' \\",
        f"  {MODEL_HASH}",
    ]
    Path("submit_audit.sh").write_text("\n".join(submit_lines))
    print("submit_audit.sh written   — run after check_verify.sh returns true")

    print()
    print("Summary:")
    for i in range(NUM_CLASSES):
        print(f"  classScore[{i}] = {class_scores[i]:>6d} ({class_scores[i] / SCALE:.4f})")


if __name__ == "__main__":
    main()
