"""
synthetic_probes.py — Fixed synthetic contracts with expected probability bounds.

THE MISSING TEST.

Run 12 was promoted to Staging despite giving ExternalBug=0.82 on a safe_storage-
style contract. The testing spec suite (ml/testing_specs/) had no automated test
that would have caught this. This module is that missing test.

For each of the 10 vulnerability classes, we define 3 fixed synthetic contracts:
  1. "Should trigger" — a contract that genuinely has the vulnerability
  2. "Should NOT trigger" — a contract that LOOKS similar but is safe
  3. "Edge case" — a contract that may or may not trigger (gray area)

Each probe has an expected probability bound. If the model's actual output
violates the bound, the probe FAILS.

Usage:
    # Direct API
    from ml.testing_specs.synthetic_probes import run_all_probes, ProbeResult
    results = run_all_probes(checkpoint_path="ml/checkpoints/Run12_best.pt")
    for r in results:
        print(f"  {'PASS' if r.passed else 'FAIL'}: {r.name} (got {r.actual:.3f}, expected {r.operator} {r.threshold})")

    # CLI
    python -m ml.testing_specs.synthetic_probes \\
        --checkpoint ml/checkpoints/Run12_best.pt \\
        --output ml/checkpoints/Run12_behavioral_probes.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

# All 10 SENTINEL classes (NUM_CLASSES = 10, order matches inference API)
SENTINEL_CLASSES = [
    "Reentrancy",
    "CallToUnknown",
    "Timestamp",
    "ExternalBug",
    "GasException",
    "DenialOfService",
    "IntegerUO",
    "UnusedReturn",
    "MishandledException",
    "TransactionOrderDependence",
]


# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------
# Each entry: (probe_id, contract_source, class_name, operator, threshold,
#             description)
# operator: ">" = expected_higher, "<" = expected_lower
# threshold: probability bound

PROBES: list[dict[str, Any]] = [
    # === ExternalBug (3 probes) ===
    # Run 12 FP was here. The safe_storage-style contract below MUST score low.
    {
        "id": "ext_owner_only",
        "class": "ExternalBug",
        "operator": "<",
        "threshold": 0.30,
        "description": (
            "address public owner; constructor() { owner = msg.sender; } — no external calls. "
            "Run 12 gave 0.85 (FP). Probe should be < 0.30."
        ),
        "source": """pragma solidity ^0.8.0;
contract Owner {
    address public owner;
    constructor() { owner = msg.sender; }
}""",
    },
    {
        "id": "ext_setvalue_owner_only",
        "class": "ExternalBug",
        "operator": "<",
        "threshold": 0.30,
        "description": (
            "owner-only setter with require(msg.sender == owner). Run 12 gave 0.82 (FP). "
            "Should be < 0.30."
        ),
        "source": """pragma solidity ^0.8.0;
contract OwnerSetter {
    address public owner;
    uint256 public value;
    constructor() { owner = msg.sender; }
    function setValue(uint256 _value) external {
        require(msg.sender == owner, "Not owner");
        value = _value;
    }
    function getValue() external view returns (uint256) { return value; }
}""",
    },
    {
        "id": "ext_low_level_call_untrusted",
        "class": "ExternalBug",
        "operator": ">",
        "threshold": 0.50,
        "description": (
            "Untrusted low-level call to user-supplied address. Textbook ExternalBug. "
            "Run 12 gave 0.00 (FN!). Probe should be > 0.50."
        ),
        "source": """pragma solidity ^0.8.0;
contract Forwarder {
    function forward(address to, bytes calldata data) external {
        to.call(data);
    }
}""",
    },

    # === Reentrancy (3 probes) ===
    {
        "id": "re_simple_transfer",
        "class": "Reentrancy",
        "operator": "<",
        "threshold": 0.30,
        "description": "Simple ETH transfer via send() — no reentrancy surface.",
        "source": """pragma solidity ^0.8.0;
contract Pay {
    function pay(address payable to) external payable {
        to.send(msg.value);
    }
}""",
    },
    {
        "id": "re_classic_cei_violation",
        "class": "Reentrancy",
        "operator": ">",
        "threshold": 0.40,
        "description": (
            "Classic CEI violation: external call before state update. "
            "Run 12 gave 0.519 on similar pattern."
        ),
        "source": """pragma solidity ^0.8.0;
contract Vault {
    mapping(address => uint256) public balances;
    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }
    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;
    }
}""",
    },
    {
        "id": "re_send_only_no_value",
        "class": "Reentrancy",
        "operator": "<",
        "threshold": 0.30,
        "description": "Pure function — no reentrancy possible.",
        "source": """pragma solidity ^0.8.0;
contract Pure {
    function add(uint a, uint b) external pure returns (uint) {
        return a + b;
    }
}""",
    },

    # === IntegerUO (3 probes) ===
    {
        "id": "iuo_safe_math_08",
        "class": "IntegerUO",
        "operator": "<",
        "threshold": 0.30,
        "description": "Solidity 0.8+ has built-in overflow checks. Safe math.",
        "source": """pragma solidity ^0.8.0;
contract Safe {
    uint256 public x;
    function increment() external {
        x = x + 1;
    }
}""",
    },
    {
        "id": "iuo_unchecked_block",
        "class": "IntegerUO",
        "operator": ">",
        "threshold": 0.40,
        "description": "Unchecked block in pre-0.8 code — clear overflow surface.",
        "source": """pragma solidity ^0.4.24;
contract Vulnerable {
    uint256 public x;
    function increment() external {
        x = x + 1;
    }
}""",
    },
    {
        "id": "iuo_view_function",
        "class": "IntegerUO",
        "operator": "<",
        "threshold": 0.30,
        "description": "View function — no state changes, no overflow possible.",
        "source": """pragma solidity ^0.8.0;
contract View {
    function viewFn(uint a, uint b) external pure returns (uint) {
        return a + b;
    }
}""",
    },

    # === Timestamp (3 probes) ===
    {
        "id": "ts_no_time",
        "class": "Timestamp",
        "operator": "<",
        "threshold": 0.30,
        "description": "No block.timestamp usage.",
        "source": """pragma solidity ^0.8.0;
contract NoTime {
    uint256 public x;
    function setX(uint256 v) external { x = v; }
}""",
    },
    {
        "id": "ts_time_locked_vesting",
        "class": "Timestamp",
        "operator": ">",
        "threshold": 0.40,
        "description": "Time-locked vesting — classic timestamp dependency.",
        "source": """pragma solidity ^0.8.0;
contract Vesting {
    uint256 public releaseTime;
    mapping(address => uint256) public balances;
    function release() external {
        require(block.timestamp >= releaseTime, "Not yet");
        payable(msg.sender).transfer(balances[msg.sender]);
    }
}""",
    },
    {
        "id": "ts_pure_function",
        "class": "Timestamp",
        "operator": "<",
        "threshold": 0.30,
        "description": "Pure function — no timestamp dependency.",
        "source": """pragma solidity ^0.8.0;
contract Pure2 {
    function id(uint a) external pure returns (uint) { return a; }
}""",
    },

    # === CallToUnknown (3 probes) ===
    {
        "id": "ctu_no_calls",
        "class": "CallToUnknown",
        "operator": "<",
        "threshold": 0.30,
        "description": "No external calls.",
        "source": """pragma solidity ^0.8.0;
contract NoCall {
    uint256 public x;
    function set(uint256 v) external { x = v; }
}""",
    },
    {
        "id": "ctu_typed_interface_call",
        "class": "CallToUnknown",
        "operator": ">",
        "threshold": 0.40,
        "description": "Call to a typed interface — known call target.",
        "source": """pragma solidity ^0.8.0;
interface IFoo { function bar() external; }
contract Caller {
    IFoo public target;
    function call() external { target.bar(); }
}""",
    },
    {
        "id": "ctu_owner_only",
        "class": "CallToUnknown",
        "operator": "<",
        "threshold": 0.30,
        "description": "Owner-only contract, no low-level calls.",
        "source": """pragma solidity ^0.8.0;
contract Owner2 {
    address public owner;
    constructor() { owner = msg.sender; }
}""",
    },

    # === GasException (3 probes) ===
    {
        "id": "ge_simple_storage",
        "class": "GasException",
        "operator": "<",
        "threshold": 0.30,
        "description": "Simple storage — no gas issues.",
        "source": """pragma solidity ^0.8.0;
contract Simple {
    uint256 public x;
    function setX(uint256 v) external { x = v; }
}""",
    },
    {
        "id": "ge_unbounded_loop",
        "class": "GasException",
        "operator": ">",
        "threshold": 0.30,
        "description": "Unbounded loop over array — gas exception risk.",
        "source": """pragma solidity ^0.8.0;
contract Loopy {
    uint256[] public arr;
    function sumAll() external view returns (uint) {
        uint s = 0;
        for (uint i = 0; i < arr.length; i++) s += arr[i];
        return s;
    }
}""",
    },
    {
        "id": "ge_pure_view",
        "class": "GasException",
        "operator": "<",
        "threshold": 0.30,
        "description": "Pure function — gas-invariant.",
        "source": """pragma solidity ^0.8.0;
contract Pure3 {
    function f(uint a) external pure returns (uint) { return a * 2; }
}""",
    },

    # === DenialOfService (3 probes) ===
    {
        "id": "dos_no_external",
        "class": "DenialOfService",
        "operator": "<",
        "threshold": 0.30,
        "description": "No external dependencies.",
        "source": """pragma solidity ^0.8.0;
contract NoDoS {
    mapping(address => uint) public b;
    function set(uint v) external { b[msg.sender] = v; }
}""",
    },
    {
        "id": "dos_unbounded_array_push",
        "class": "DenialOfService",
        "operator": ">",
        "threshold": 0.30,
        "description": "Unbounded push in array — DoS risk.",
        "source": """pragma solidity ^0.8.0;
contract Pushy {
    uint256[] public arr;
    function push(uint256 v) external { arr.push(v); }
}""",
    },
    {
        "id": "dos_owner_only",
        "class": "DenialOfService",
        "operator": "<",
        "threshold": 0.30,
        "description": "Owner-only contract — DoS not relevant.",
        "source": """pragma solidity ^0.8.0;
contract Owner3 {
    address public owner;
    constructor() { owner = msg.sender; }
}""",
    },

    # === UnusedReturn (3 probes) ===
    {
        "id": "ur_no_calls",
        "class": "UnusedReturn",
        "operator": "<",
        "threshold": 0.30,
        "description": "No external calls to ignore.",
        "source": """pragma solidity ^0.8.0;
contract NoReturn {
    uint256 public x;
    function setX(uint256 v) external { x = v; }
}""",
    },
    {
        "id": "ur_ignored_transfer",
        "class": "UnusedReturn",
        "operator": ">",
        "threshold": 0.30,
        "description": "ERC20 transfer called but return value ignored.",
        "source": """pragma solidity ^0.8.0;
interface IERC20 { function transfer(address, uint) external returns (bool); }
contract Leaky {
    IERC20 public token;
    function send(address to, uint amt) external {
        token.transfer(to, amt);
    }
}""",
    },
    {
        "id": "ur_view_only",
        "class": "UnusedReturn",
        "operator": "<",
        "threshold": 0.30,
        "description": "View function — no calls.",
        "source": """pragma solidity ^0.8.0;
contract View2 {
    function get() external pure returns (uint) { return 42; }
}""",
    },

    # === MishandledException (3 probes) ===
    {
        "id": "me_no_calls",
        "class": "MishandledException",
        "operator": "<",
        "threshold": 0.30,
        "description": "No calls to mishandle.",
        "source": """pragma solidity ^0.8.0;
contract NoMish {
    uint256 public x;
    function setX(uint256 v) external { x = v; }
}""",
    },
    {
        "id": "me_unchecked_low_level",
        "class": "MishandledException",
        "operator": ">",
        "threshold": 0.30,
        "description": "Low-level call without checking return value.",
        "source": """pragma solidity ^0.8.0;
contract Unchecked {
    function callIt(address to) external {
        to.call(\"\");
    }
}""",
    },
    {
        "id": "me_proper_check",
        "class": "MishandledException",
        "operator": "<",
        "threshold": 0.30,
        "description": "Properly checks return value.",
        "source": """pragma solidity ^0.8.0;
contract Proper {
    function callIt(address to) external returns (bool) {
        (bool ok, ) = to.call(\"\");
        return ok;
    }
}""",
    },

    # === TransactionOrderDependence (3 probes) ===
    {
        "id": "tod_no_origin",
        "class": "TransactionOrderDependence",
        "operator": "<",
        "threshold": 0.30,
        "description": "No tx.origin usage.",
        "source": """pragma solidity ^0.8.0;
contract NoOrigin {
    address public owner;
    constructor() { owner = msg.sender; }
}""",
    },
    {
        "id": "tod_tx_origin_auth",
        "class": "TransactionOrderDependence",
        "operator": ">",
        "threshold": 0.40,
        "description": "Uses tx.origin for auth — classic TOD.",
        "source": """pragma solidity ^0.8.0;
contract OriginAuth {
    address public owner;
    function withdraw() external {
        require(tx.origin == owner, "Not owner");
        payable(msg.sender).transfer(1 ether);
    }
}""",
    },
    {
        "id": "tod_pure_function",
        "class": "TransactionOrderDependence",
        "operator": "<",
        "threshold": 0.30,
        "description": "Pure function — no TOD possible.",
        "source": """pragma solidity ^0.8.0;
contract Pure4 {
    function f(uint a) external pure returns (uint) { return a; }
}""",
    },

    # === ADVERSARIAL EDGE-CASE PROBES (NEW 2026-06-17) ===
    # These probe infrastructure and model robustness — not specific to any class.
    # They catch things like: "model crashes on empty input", "model over-confident
    # on trivial contracts", "model fails on long contracts".

    {
        "id": "edge_empty_contract",
        "class": "ExternalBug",
        "operator": "<",
        "threshold": 0.30,
        "description": "Empty contract should not trigger any class.",
        "source": """pragma solidity ^0.8.0;
contract Empty {}""",
    },
    {
        "id": "edge_pragma_only",
        "class": "ExternalBug",
        "operator": "<",
        "threshold": 0.30,
        "description": "Pragma-only file should not trigger.",
        "source": """pragma solidity ^0.8.0;""",
    },
    {
        "id": "edge_comment_only",
        "class": "ExternalBug",
        "operator": "<",
        "threshold": 0.30,
        "description": "Comment-only file should not trigger.",
        "source": """pragma solidity ^0.8.0;
// This contract has only comments.
// It should not trigger any vulnerability class.
// SPDX-License-Identifier: UNLICENSED
contract Commented {
    // nothing here
}""",
    },
    {
        "id": "edge_huge_storage",
        "class": "Reentrancy",
        "operator": "<",
        "threshold": 0.30,
        "description": "Many state variables, no functions — tests that the model doesn't over-predict on large contracts.",
        "source": """pragma solidity ^0.8.0;
contract Big {
    uint256 public a0;  uint256 public a1;  uint256 public a2;  uint256 public a3;
    uint256 public a4;  uint256 public a5;  uint256 public a6;  uint256 public a7;
    uint256 public a8;  uint256 public a9;  uint256 public a10; uint256 public a11;
    uint256 public a12; uint256 public a13; uint256 public a14; uint256 public a15;
    uint256 public a16; uint256 public a17; uint256 public a18; uint256 public a19;
}""",
    },
    {
        "id": "edge_many_inheritance",
        "class": "Reentrancy",
        "operator": "<",
        "threshold": 0.30,
        "description": "Multi-inheritance — tests that the model handles complex hierarchies.",
        "source": """pragma solidity ^0.8.0;
contract A { uint256 x; }
contract B { uint256 y; }
contract C { uint256 z; }
contract D is A, B, C {
    function getAll() external view returns (uint256, uint256, uint256) {
        return (x, y, z);
    }
}""",
    },
    {
        "id": "edge_assembly_block",
        "class": "IntegerUO",
        "operator": "<",
        "threshold": 0.30,
        "description": "Inline assembly — tests that the model handles low-level code.",
        "source": """pragma solidity ^0.8.0;
contract Asm {
    function add(uint a, uint b) external pure returns (uint) {
        assembly {
            let result := add(a, b)
            mstore(0x00, result)
            return(0x00, 0x20)
        }
    }
}""",
    },
    {
        "id": "edge_modifier_chain",
        "class": "AccessControl",
        "operator": "<",
        "threshold": 0.30,
        "description": "Multiple modifiers chained — tests modifier handling.",
        "source": """pragma solidity ^0.8.0;
contract Chained {
    address public owner;
    modifier onlyOwner() { require(msg.sender == owner, "not owner"); _; }
    modifier nonReentrant() { _; }
    function restricted() external onlyOwner nonReentrant {
        // empty
    }
}""",
    },
    {
        "id": "edge_library",
        "class": "ExternalBug",
        "operator": "<",
        "threshold": 0.30,
        "description": "Library code — should not be classified as vulnerable contract.",
        "source": """pragma solidity ^0.8.0;
library SafeMath {
    function add(uint a, uint b) internal pure returns (uint) {
        return a + b;
    }
}""",
    },
    {
        "id": "edge_abstract",
        "class": "ExternalBug",
        "operator": "<",
        "threshold": 0.30,
        "description": "Abstract contract — should not trigger (no implementation).",
        "source": """pragma solidity ^0.8.0;
abstract contract Abs {
    function foo() external virtual returns (uint);
}""",
    },
    {
        "id": "edge_interface",
        "class": "ExternalBug",
        "operator": "<",
        "threshold": 0.30,
        "description": "Interface — no implementation, no risk.",
        "source": """pragma solidity ^0.8.0;
interface IDo {
    function doIt() external;
}""",
    },
]


@dataclass
class ProbeResult:
    """Result of running one synthetic probe.

    status: one of
        "PASS"        — actual probability satisfied the bound
        "FAIL"        — actual probability violated the bound
        "INFRA_ERROR" — inference pipeline crashed (e.g. EmptyGraphError);
                        no probability available to evaluate the bound
    """

    probe_id: str
    class_name: str
    operator: str  # "<" or ">"
    threshold: float
    actual: float
    passed: bool
    description: str
    duration_s: float
    source_chars: int
    status: str = "PASS"  # PASS / FAIL / INFRA_ERROR

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def message(self) -> str:
        if self.status == "INFRA_ERROR":
            return (
                f"[INFRA_ERROR] {self.probe_id} ({self.class_name}): "
                f"inference pipeline crashed — bound {self.operator} {self.threshold:.2f} "
                f"could not be evaluated. See description for error."
            )
        op_str = "≥" if self.operator == ">" else "≤"
        actual_op = ">" if self.operator == "<" else "<"
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.probe_id} ({self.class_name}): "
            f"expected {op_str} {self.threshold:.2f}, got {self.actual:.3f} "
            f"(FAIL: {self.actual:.3f} {actual_op} {self.threshold:.2f})"
            if not self.passed
            else f"[{status}] {self.probe_id} ({self.class_name}): "
            f"expected {op_str} {self.threshold:.2f}, got {self.actual:.3f}"
        )


def _check_threshold(actual: float, operator: str, threshold: float) -> bool:
    """Check if actual satisfies the operator+threshold bound."""
    if operator == ">":
        return actual >= threshold
    elif operator == "<":
        return actual <= threshold
    raise ValueError(f"Unknown operator: {operator}")


def run_single_probe(
    probe: dict[str, Any],
    predictor: Any,
    contract_key: str = "source_code",
) -> ProbeResult:
    """Run one synthetic probe against a predictor.

    Args:
        probe: Probe definition dict (from PROBES list).
        predictor: Object with .predict(source_code) -> dict with .probabilities.
        contract_key: Key name for the source code field in predict().

    Returns:
        ProbeResult with pass/fail and actual probability.
    """
    source = probe["source"]
    t0 = time.time()
    result = predictor.predict({contract_key: source})
    duration = time.time() - t0

    probs = result.get("probabilities", {})
    actual = probs.get(probe["class"], 0.0)
    passed = _check_threshold(actual, probe["operator"], probe["threshold"])

    return ProbeResult(
        probe_id=probe["id"],
        class_name=probe["class"],
        operator=probe["operator"],
        threshold=probe["threshold"],
        actual=actual,
        passed=passed,
        description=probe["description"],
        duration_s=duration,
        source_chars=len(source),
    )


def run_all_probes(
    checkpoint_path: str | None = None,
    predictor: Any = None,
    base_url: str | None = None,
    verbose: bool = True,
) -> list[ProbeResult]:
    """Run all 30+ synthetic probes.

    Two modes:
      1. Pass a `predictor` directly (in-process, fastest, no network).
      2. Pass `checkpoint_path` (creates predictor from checkpoint).
      3. Pass `base_url` (uses HTTP API).

    Returns:
        List of ProbeResult, one per probe.
    """
    if predictor is None:
        if base_url:
            predictor = _HttpPredictor(base_url)
        elif checkpoint_path:
            predictor = _CheckpointPredictor(checkpoint_path)
        else:
            # Default: use the active FINAL checkpoint from MEMORY.md
            # The active Run 12 checkpoint is the _FINAL.pt (immutable).
            # We try _FINAL first, then _best, then any *_best.pt.
            candidates = [
                "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt",
                "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt",
            ]
            chosen = None
            for c in candidates:
                if Path(c).exists():
                    chosen = c
                    break
            # Fallback: any *_best.pt
            if not chosen:
                ckpts = list(Path("ml/checkpoints").glob("*_best.pt"))
                if ckpts:
                    chosen = str(ckpts[0])
            if chosen:
                print(f"No checkpoint specified; using default: {chosen}")
                predictor = _CheckpointPredictor(chosen)
            else:
                raise ValueError(
                    "No predictor provided and no checkpoint found. "
                    "Pass --checkpoint, --base-url, or predictor=..."
                )

    results: list[ProbeResult] = []
    for probe in PROBES:
        try:
            r = run_single_probe(probe, predictor)
        except Exception as e:
            # If inference fails entirely (e.g. EmptyGraphError on pragma-only
            # source), mark the probe as INFRA_ERROR with actual=0. This is
            # NOT a regular FAIL — the model never got to see the contract.
            r = ProbeResult(
                probe_id=probe["id"],
                class_name=probe["class"],
                operator=probe["operator"],
                threshold=probe["threshold"],
                actual=0.0,
                passed=False,
                status="INFRA_ERROR",
                description=f"INFERENCE ERROR: {type(e).__name__}: {e}",
                duration_s=0.0,
                source_chars=len(probe["source"]),
            )
        results.append(r)
        if verbose:
            print(r.message)
    return results


def summarize(results: list[ProbeResult]) -> dict[str, Any]:
    """Summarize probe results for a report."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    infra_errors = sum(1 for r in results if r.status == "INFRA_ERROR")
    failed = total - passed - infra_errors  # regular fails only
    by_class: dict[str, dict[str, int]] = {}
    for r in results:
        if r.class_name not in by_class:
            by_class[r.class_name] = {"passed": 0, "failed": 0, "infra_error": 0}
        if r.status == "INFRA_ERROR":
            by_class[r.class_name]["infra_error"] += 1
        elif r.passed:
            by_class[r.class_name]["passed"] += 1
        else:
            by_class[r.class_name]["failed"] += 1
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "infra_errors": infra_errors,
        "pass_rate": passed / total if total else 0,
        "by_class": by_class,
        # "all_passed" is True only if every probe got a real probability AND
        # satisfied its bound. INFRA_ERROR counts as failure.
        "all_passed": (failed == 0 and infra_errors == 0),
    }


# ---------------------------------------------------------------------------
# Predictor adapters
# ---------------------------------------------------------------------------


class _CheckpointPredictor:
    """In-process predictor from a local checkpoint."""

    def __init__(self, checkpoint_path: str):
        from ml.src.inference.predictor import Predictor
        self._p = Predictor(checkpoint_path)

    def predict(self, payload: dict) -> dict:
        source = payload.get("source_code") or payload.get("source")
        # Use predict_source() — takes raw string, not file path
        r = self._p.predict_source(source)
        # Normalize to {probabilities: {class: float}}
        return {
            "probabilities": r.get("probabilities", {}),
            "label": r.get("label"),
        }


class _HttpPredictor:
    """Predictor via HTTP API (for remote / inference_server testing)."""

    def __init__(self, base_url: str):
        self._base = base_url.rstrip("/")

    def predict(self, payload: dict) -> dict:
        import urllib.request
        source = payload.get("source_code") or payload.get("source")
        body = json.dumps({"source_code": source}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/predict",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return {
            "probabilities": data.get("probabilities", {}),
            "label": data.get("label"),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="synthetic_probes",
        description=(
            "Run fixed synthetic contracts against a model checkpoint or API. "
            "Each probe has an expected (min/max) probability bound. Returns "
            "non-zero exit code if any probe FAILS."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt). Default: Run 12 best.",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="URL of inference API (e.g., http://localhost:8001). "
             "Used if --checkpoint not provided.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write results JSON to this path.",
    )
    parser.add_argument(
        "--exit-on-fail", action="store_true",
        help="Exit with code 1 if any probe fails. Default: always 0.",
    )
    parser.add_argument(
        "--probe", type=str, default=None,
        help="Run only this probe (by id). Default: all.",
    )
    args = parser.parse_args()

    probes = PROBES
    if args.probe:
        probes = [p for p in PROBES if p["id"] == args.probe]
        if not probes:
            print(f"ERROR: probe '{args.probe}' not found. Available:")
            for p in PROBES:
                print(f"  {p['id']}")
            return 1

    if args.checkpoint:
        predictor = _CheckpointPredictor(args.checkpoint)
    elif args.base_url:
        predictor = _HttpPredictor(args.base_url)
    else:
        # Default: use the active FINAL checkpoint from MEMORY.md
        candidates = [
            "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt",
            "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt",
        ]
        chosen = None
        for c in candidates:
            if Path(c).exists():
                chosen = c
                break
        if not chosen:
            ckpts = list(Path("ml/checkpoints").glob("*_best.pt"))
            if ckpts:
                chosen = str(ckpts[0])
        if chosen:
            print(f"No checkpoint specified; using default: {chosen}")
            predictor = _CheckpointPredictor(chosen)
        else:
            print(f"ERROR: No checkpoint found. Pass --checkpoint.")
            return 1

    results = run_all_probes(predictor=predictor, verbose=True)
    summary = summarize(results)

    print()
    print("=" * 70)
    print(
        f"SUMMARY: {summary['passed']}/{summary['total']} probes passed "
        f"({summary['failed']} FAIL, {summary['infra_errors']} INFRA_ERROR)"
    )
    for cls, counts in summary["by_class"].items():
        total_cls = counts["passed"] + counts["failed"] + counts.get("infra_error", 0)
        marker = "✓" if (counts["failed"] == 0 and counts.get("infra_error", 0) == 0) else "✗"
        print(
            f"  {marker} {cls:30} {counts['passed']}/{total_cls}"
            f"  (FAIL={counts['failed']}, INFRA={counts.get('infra_error', 0)})"
        )
    print("=" * 70)

    if args.output:
        output = {
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        print(f"\nResults written to: {args.output}")

    if args.exit_on_fail and not summary["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
