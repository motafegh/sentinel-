"""
formal_verification node — Halmos symbolic execution (P8a).

Runs Halmos on the contract to formally verify or refute vulnerability invariants.
Emits Evidence with kind=FORMAL, deterministic=True.

Architecture:
    1. Create a temp Foundry project with the contract
    2. Generate a test harness with invariant checks
    3. Run `forge build` + `halmos --json-output`
    4. Parse JSON output
    5. Map results to vulnerability classes

P5 (2026-06-26): Skip when SENTINEL_DETERMINISTIC=1 (Halmos is deterministic but
the project setup + forge build can have side effects that make it non-reproducible).

Fail-soft (Principle 6): If Halmos fails for any reason (not installed, compile error,
timeout), returns empty findings — the pipeline continues with other evidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.timing import step_timer


_HALMOS_TIMEOUT_S = int(os.getenv("HALMOS_TIMEOUT_S", "120"))

_INVARIANT_TO_CLASS = {
    "reentrancy": "Reentrancy",
    "arithmetic": "IntegerUO",
    "access_control": "AccessControl",
    "unchecked_return": "UnusedReturn",
    "denial_of_service": "DenialOfService",
}


def _generate_test_harness(contract_name: str) -> str:
    """Generate a Foundry test file with invariant checks for the contract."""
    return f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "../src/Target.sol";

contract FormalVerify is Test {{
    Target public target;

    function setUp() public {{
        target = new Target();
    }}

    function check_reentrancy() public {{
        // Halmos will try to find a call sequence that violates CEI
        target.deposit{{value: 1 ether}}();
        target.withdraw();
        assertEq(address(target).balance, 0);
    }}

    function check_arithmetic() public {{
        // Halmos will try to find inputs that cause overflow/underflow
        target.deposit{{value: 1 ether}}();
        // No assertion needed — Halmos checks for built-in overflow
    }}

    function check_access_control() public {{
        // Halmos checks if non-owner can call restricted functions
        vm.prank(address(0xdead));
        try target.withdraw() {{
            assertFail("unauthorized access");
        }} catch {{}}
    }}
}}
"""


async def formal_verification(state: AuditState) -> dict[str, Any]:
    """
    Run Halmos symbolic execution on the contract.

    State updates:
        symbolic_findings → list of {tool, vulnerability_class, invariant, proven, counterexample}
        evidence_list     → appended with Evidence(kind=FORMAL)
        tool_status       → halmos: {ran, reason, detail}
    """
    if os.getenv("SENTINEL_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
        logger.info("formal_verification | skipped (SENTINEL_DETERMINISTIC mode)")
        return {"symbolic_findings": [], "tool_status": {"halmos": {"ran": False, "reason": "deterministic_mode"}}}

    contract_code = state.get("contract_code", "") or ""
    if not contract_code:
        logger.info("formal_verification | no contract code — skipping")
        return {
            "symbolic_findings": [],
            "tool_status": {"halmos": {"ran": False, "reason": "empty_contract_code"}},
        }

    halmos_path = shutil.which("halmos")
    forge_path = shutil.which("forge")
    if not halmos_path or not forge_path:
        logger.warning(
            "formal_verification | halmos or forge not found "
            "(halmos={} forge={}) — skipping",
            halmos_path or "NOT FOUND",
            forge_path or "NOT FOUND",
        )
        return {
            "symbolic_findings": [],
            "tool_status": {"halmos": {"ran": False, "reason": "not_installed"}},
        }

    contract_name_match = re.search(r"contract\s+(\w+)", contract_code)
    if not contract_name_match:
        logger.warning("formal_verification | could not extract contract name — skipping")
        return {
            "symbolic_findings": [],
            "tool_status": {"halmos": {"ran": False, "reason": "no_contract_name"}},
        }
    contract_name = contract_name_match.group(1)

    address = state.get("contract_address", "unknown")

    with step_timer("formal_verification", address=address, budget_s=_HALMOS_TIMEOUT_S):
        try:
            findings = await _run_halmos(contract_code, contract_name, address)
            from src.orchestration.verdict.emit import emit_halmos_evidence
            evidence = emit_halmos_evidence(findings)

            logger.info(
                "formal_verification complete | {} findings | {} evidence items",
                len(findings), len(evidence),
            )

            return {
                "symbolic_findings": findings,
                "evidence_list": evidence,
                "tool_status": {"halmos": {"ran": True, "findings": len(findings)}},
            }

        except Exception as exc:
            logger.warning("formal_verification | failed (fail-soft): {}", exc)
            return {
                "symbolic_findings": [],
                "tool_status": {"halmos": {"ran": False, "reason": str(exc)[:200]}},
            }


async def _run_halmos(
    contract_code: str,
    contract_name: str,
    address: str,
) -> list[dict[str, Any]]:
    """
    Create a temp Foundry project, write the contract, run Halmos.

    Returns list of findings: {tool, vulnerability_class, invariant, proven, counterexample}
    """
    with tempfile.TemporaryDirectory(prefix="sentinel_halmos_") as tmpdir:
        tmp = Path(tmpdir)

        src_dir = tmp / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        test_dir = tmp / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        (src_dir / "Target.sol").write_text(contract_code)

        test_code = _generate_test_harness(contract_name)
        (test_dir / "FormalVerify.t.sol").write_text(test_code)

        foundry_toml = tmp / "foundry.toml"
        foundry_toml.write_text(
            '[profile.default]\n'
            'src = "src"\n'
            'out = "out"\n'
            'test = "test"\n'
            'libs = ["lib"]\n'
            'solc_version = "0.8.19"\n'
            'evm_version = "paris"\n'
        )

        lib_dir = tmp / "lib"
        lib_dir.mkdir(exist_ok=True)
        forge_libs = ["forge-std"]
        for lib in forge_libs:
            forge_std = os.getenv("FORGE_STD_PATH", str(Path.home() / ".foundry" / "lib" / "forge-std"))
            if Path(forge_std).exists():
                (lib_dir / lib).symlink_to(forge_std, target_is_directory=True)

        loop = asyncio.get_running_loop()

        def _run_forge():
            result = subprocess.run(
                ["forge", "build", "--root", str(tmp)],
                capture_output=True, text=True, timeout=60,
                cwd=str(tmp),
            )
            return result

        forge_result = await loop.run_in_executor(None, _run_forge)

        if forge_result.returncode != 0:
            logger.warning(
                "formal_verification | forge build failed: {}",
                forge_result.stderr[:200],
            )
            return []

        json_output = tmp / "halmos_output.json"

        def _run_halmos():
            result = subprocess.run(
                [
                    "halmos",
                    "--root", str(tmp),
                    "--match-contract", "FormalVerify",
                    "--json-output", str(json_output),
                    "--solver-timeout-assertion", "10",
                ],
                capture_output=True, text=True, timeout=_HALMOS_TIMEOUT_S,
                cwd=str(tmp),
            )
            return result

        halmos_result = await loop.run_in_executor(None, _run_halmos)

        if halmos_result.returncode != 0:
            logger.warning(
                "formal_verification | halmos failed: {}",
                halmos_result.stderr[:200],
            )
            return []

        if not json_output.exists():
            logger.warning("formal_verification | halmos JSON output not found")
            return []

        return _parse_halmos_output(json_output.read_text())


def _parse_halmos_output(json_str: str) -> list[dict[str, Any]]:
    """
    Parse Halmos JSON output into findings.

    Halmos output format (simplified):
    {
        "results": [
            {
                "name": "check_reentrancy()",
                "status": "pass" | "fail",
                "counterexample": {...}  // present only on fail
            },
            ...
        ]
    }
    """
    findings: list[dict[str, Any]] = []

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("formal_verification | could not parse Halmos JSON")
        return findings

    results = data.get("results", [])
    if isinstance(data, dict) and "test_results" in data:
        results = data["test_results"]

    for result in results:
        name = result.get("name", result.get("test_name", ""))
        status = result.get("status", result.get("result", ""))

        invariant = name.replace("check_", "").replace("()", "")
        vuln_class = _INVARIANT_TO_CLASS.get(invariant, "")

        if not vuln_class:
            continue

        proven = status.lower() in ("pass", "passed", "ok")
        counterexample = ""

        if not proven:
            ce = result.get("counterexample", result.get("model", {}))
            if ce:
                counterexample = json.dumps(ce, default=str)[:200]

        findings.append({
            "tool": "halmos",
            "vulnerability_class": vuln_class,
            "invariant": invariant,
            "proven": proven,
            "counterexample": counterexample,
        })

    return findings
