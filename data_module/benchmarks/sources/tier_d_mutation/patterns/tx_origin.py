"""tx_origin.py — Mutation pattern: inject TransactionOrderDependence vulnerability.

SWC-115: Use of tx.origin for authentication.

Pattern: replace `msg.sender` with `tx.origin` in an `onlyOwner` modifier or
similar access control check. The mutated contract now has the tx.origin
vulnerability (msg.sender-based callers can be spoofed via intermediate calls).

Original (clean):
    modifier onlyOwner { require(msg.sender == owner); _; }
    function withdraw() public onlyOwner { ... }

Mutated (vulnerable):
    modifier onlyOwner { require(tx.origin == owner); _; }
    function withdraw() public onlyOwner { ... }
"""
import re
from typing import Tuple


def apply(source: str) -> Tuple[str, str]:
    """Apply the tx.origin mutation to a clean contract.

    Returns: (mutated_source, vulnerability_class)
    """
    # Find the onlyOwner-like modifier
    if "modifier onlyOwner" not in source and "modifier only_owner" not in source:
        # If no access control modifier, inject one
        mutated = source.replace(
            "msg.sender",
            "tx.origin"
        )
    else:
        # Replace msg.sender with tx.origin in modifier context
        # First, find modifiers
        mutated = re.sub(
            r"(modifier\s+\w+\s*\{[^}]*?)msg\.sender",
            r"\1tx.origin",
            source,
            flags=re.DOTALL,
        )
        # If no modifier had msg.sender, do a global replace in access control
        if "tx.origin" not in mutated:
            mutated = source.replace("msg.sender == owner", "tx.origin == owner")
        if "tx.origin" not in mutated:
            # Fallback: just do a global replace
            mutated = source.replace("msg.sender", "tx.origin")

    return mutated, "TransactionOrderDependence"


def verify_mutated(source: str) -> bool:
    """Verify the mutation actually introduced tx.origin (sanity check)."""
    return "tx.origin" in source and "msg.sender" not in source.split("modifier")[1] if "modifier" in source else "tx.origin" in source
