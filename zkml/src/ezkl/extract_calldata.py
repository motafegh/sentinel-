"""
extract_calldata.py — Extract on-chain calldata from proof.json

Reads zkml/ezkl/proof.json and outputs the exact arguments needed
to call AuditRegistry.submitAudit() on-chain via cast.

Usage:
    cd ~/projects/sentinel
    poetry run python zkml/src/ezkl/extract_calldata.py

Output:
    - Prints human-readable summary
    - Writes check_verify.sh   (cast call → verifyProof on ZKMLVerifier)
    - Writes submit_audit.sh   (cast send → submitAudit on AuditRegistry)

ENCODING REFERENCE — BN254 field elements (read this before touching signals):
    EZKL stores every public signal (input features + output score) as a
    32-byte little-endian hex string in proof.json["instances"][0].

    "Little-endian" means the LEAST significant byte comes first in the hex.
    Ethereum's uint256 is big-endian (most significant byte first).
    They are OPPOSITE byte orders — you MUST convert explicitly.

    CORRECT conversion (Python):
        signal_uint256 = int.from_bytes(bytes.fromhex(hex_str), byteorder='little')

    WRONG conversion (do NOT use):
        int(hex_str, 16)  ← treats the string as big-endian → garbage value

    Concrete example:
        instances[64] = "9111000000000000000000000000000000000000000000000000000000000000"
        big-endian  → 65615399444674858847734919285089764922285269...  (wrong, not a field element)
        little-endian → 4497                                           (correct: 4497/8192 = 0.5490)

    The score that AuditRegistry.submitAudit() expects is the little-endian
    interpretation — 4497, not the big-endian garbage number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Config — update VERIFIER and REGISTRY to match your deployment
# ------------------------------------------------------------------

PROOF_PATH = Path("zkml/ezkl/proof.json")

# Sepolia deployment addresses (update after each re-deploy)
ZKML_VERIFIER = "0xB7093Be4958dd95438D6f53Ff7DF8659451CbD97"
AUDIT_REGISTRY = "0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf"
RPC_URL = "https://sepolia.infura.io/v3/31876fad90e24857ab6751fb214da7b9"

# Target contract being audited (update per audit)
AUDIT_TARGET = "0x000000000000000000000000000000000000dEaD"


def _decode_field_element(hex_str: str) -> int:
    """
    Convert a BN254 field element from proof.json to a Solidity uint256.

    EZKL serialises field elements as 32-byte little-endian hex strings.
    Solidity / EVM expects uint256 values in big-endian order.
    This function handles the conversion.

    Args:
        hex_str: 64-character hex string (32 bytes, no 0x prefix), little-endian.

    Returns:
        Integer value as a Python int, suitable for passing to Solidity.
    """
    return int.from_bytes(bytes.fromhex(hex_str), byteorder='little')


def main() -> None:
    # ------------------------------------------------------------------
    # Load and validate proof.json
    # ------------------------------------------------------------------
    if not PROOF_PATH.exists():
        print(f"ERROR: proof.json not found at {PROOF_PATH}", file=sys.stderr)
        print("Run: poetry run python zkml/src/ezkl/run_proof.py", file=sys.stderr)
        sys.exit(1)

    try:
        proof_data = json.loads(PROOF_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: proof.json is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        hex_proof = proof_data["hex_proof"]
        instances = proof_data["instances"][0]  # list of 65 little-endian hex strings
    except (KeyError, IndexError, TypeError) as e:
        print(
            f"ERROR: proof.json missing expected structure: {e}\n"
            f"Expected keys: hex_proof, instances[0]\n"
            f"Regenerate the proof with run_proof.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(instances) != 65:
        print(
            f"ERROR: expected 65 public signals (64 features + 1 score), "
            f"got {len(instances)}.\n"
            f"The proof may have been generated with a different circuit version.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Decode field elements — LITTLE-ENDIAN conversion required
    # ------------------------------------------------------------------
    # RECALL — each instances[i] is a 32-byte little-endian hex string.
    # int.from_bytes(..., byteorder='little') gives the correct uint256.
    # Indices 0-63: input feature field elements.
    # Index 64:     model output — the risk score field element.
    public_signals = [_decode_field_element(h) for h in instances]
    score_field_element = public_signals[64]

    # Human-readable probability: divide field element by 2^scale
    # Scale=13 (from EZKL calibration), so divisor = 2^13 = 8192
    score_human = score_field_element / 8192.0

    # ------------------------------------------------------------------
    # Build cast command strings
    # ------------------------------------------------------------------
    signals_str = "[" + ",".join(str(s) for s in public_signals) + "]"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("CALLDATA FOR AuditRegistry.submitAudit()")
    print("=" * 60)
    print(f"hex_proof (first 20 chars): {hex_proof[:20]}...")
    print(f"Public signals count:       {len(public_signals)}")
    print(f"Score field element:        {score_field_element}  (publicSignals[64])")
    print(f"Score human-readable:       {score_human:.4f}  ({score_field_element} / 8192)")
    print(f"Classification:             {'VULNERABLE' if score_human >= 0.5 else 'SAFE'}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Write check_verify.sh — test the ZK proof against the on-chain verifier
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Write submit_audit.sh — submit the audit to AuditRegistry
    # ------------------------------------------------------------------
    # IMPORTANT — replace DEPLOYER_PRIVATE_KEY with your actual key,
    # or set it as an environment variable and use $DEPLOYER_PRIVATE_KEY.
    submit_lines = [
        "cast send \\",
        "  --private-key $DEPLOYER_PRIVATE_KEY \\",
        f"  --rpc-url {RPC_URL} \\",
        f"  {AUDIT_REGISTRY} \\",
        "  'submitAudit(address,uint256,bytes,uint256[])' \\",
        f"  {AUDIT_TARGET} \\",
        f"  {score_field_element} \\",
        f"  {hex_proof} \\",
        f"  '{signals_str}'",
    ]
    Path("submit_audit.sh").write_text("\n".join(submit_lines))
    print("submit_audit.sh written  — run after check_verify.sh returns true")
    print()
    print(f"score_field_element = {score_field_element}")
    print(f"(divide by 8192 to get probability: {score_human:.4f})")


if __name__ == "__main__":
    main()
