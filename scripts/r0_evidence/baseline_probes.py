"""Isolated probes that reproduce the eight known-failing R0 baseline rows."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def _assertion(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _result(invariant_passed: bool, assertions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pass" if invariant_passed else "fail",
        "invariant_passed": invariant_passed,
        "assertions": assertions,
    }


def probe_ml_outage(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "agents"))
    import httpx
    from src.mcp.servers import inference_server as server

    class UnreachableClient:
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            raise httpx.ConnectError("r0-baseline-connect-failure")

    previous_mode, previous_client = server._MOCK_MODE, server._http_client
    try:
        server._MOCK_MODE = False
        server._http_client = UnreachableClient()
        response = asyncio.run(server._call_inference_api("contract C {}"))
    finally:
        server._MOCK_MODE, server._http_client = previous_mode, previous_client

    explicit_failure = "error" in response or response.get("ran") is False
    no_evidence = not response.get("probabilities") and not response.get("vulnerabilities")
    invariant_passed = explicit_failure and no_evidence
    return _result(
        invariant_passed,
        [
            _assertion("outage_is_explicit", explicit_failure, f"keys={sorted(response)}"),
            _assertion(
                "outage_emits_no_prediction_evidence",
                no_evidence,
                (
                    f"label={response.get('label')!r}; "
                    f"probabilities={len(response.get('probabilities', {}))}"
                ),
            ),
            _assertion(
                "model_identity_is_not_mock_disguised_as_live",
                not str(response.get("model_hash", "")).startswith("mock_model_hash_"),
                f"model_hash={response.get('model_hash')!r}",
            ),
        ],
    )


def probe_gateway_auth(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "agents"))
    from fastapi.testclient import TestClient
    from src.api.gateway import create_app
    from src.api.job_store import JobStore

    class Graph:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            return {"final_report": {"overall_label": "safe"}}

    app = create_app(
        store=JobStore(),
        graph_factory=lambda: Graph(),
        skip_service_probes=True,
    )
    with TestClient(app) as client:
        response = client.post(
            "/audit",
            json={"contract_code": "pragma solidity ^0.8.0; contract C {}"},
        )

    requires_auth = response.status_code in {401, 403}
    challenge_present = bool(response.headers.get("www-authenticate"))
    return _result(
        requires_auth and challenge_present,
        [
            _assertion(
                "unauthenticated_mutation_rejected", requires_auth, f"HTTP {response.status_code}"
            ),
            _assertion(
                "authentication_challenge_present",
                challenge_present,
                f"WWW-Authenticate={response.headers.get('www-authenticate')!r}",
            ),
        ],
    )


def probe_zip_containment(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "data_module"))
    from sentinel_data.ingestion.connectors.manual_connector import _extract_zip

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "bad.zip"
        destination = root / "repo"
        destination.mkdir()
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../repo_evil/pwned.txt", "escaped")
        exception_name = None
        try:
            _extract_zip(archive, destination, "r0-baseline")
        except Exception as exc:  # The type is evidence and is serialized below.
            exception_name = type(exc).__name__
        escaped = (root / "repo_evil" / "pwned.txt").exists()

    rejected = exception_name is not None
    return _result(
        rejected and not escaped,
        [
            _assertion("traversal_rejected", rejected, f"exception={exception_name!r}"),
            _assertion("no_outside_write", not escaped, f"escaped={escaped}"),
        ],
    )


def _source_probe(workspace: Path, probe_name: str) -> dict[str, Any]:
    if probe_name == "report-path":
        source = (workspace / "agents/src/orchestration/nodes/synthesizer.py").read_text()
        unsafe = 'REPORTS_DIR / f"{contract_address}.json"' in source
        return _result(
            not unsafe,
            [
                _assertion(
                    "logical_address_not_used_as_filename",
                    not unsafe,
                    f"unsafe_expression={unsafe}",
                )
            ],
        )

    if probe_name == "data-release":
        chunker = (workspace / "data_module/sentinel_data/export/chunker.py").read_text()
        export = (workspace / "data_module/sentinel_data/export/export.py").read_text()
        manifest_excluded = (
            '"manifest.json"' in chunker.partition("_HASH_EXCLUDED")[2].split("\n", 1)[0]
        )
        no_expected_set_equality = "set(cached_files)" not in export
        passed = not manifest_excluded and not no_expected_set_equality
        return _result(
            passed,
            [
                _assertion(
                    "semantic_manifest_is_committed",
                    not manifest_excluded,
                    f"excluded={manifest_excluded}",
                ),
                _assertion(
                    "warm_cache_checks_exact_file_set",
                    not no_expected_set_equality,
                    f"exact_set_check_present={not no_expected_set_equality}",
                ),
            ],
        )

    if probe_name == "signer-boundary":
        config = (workspace / "agents/src/mcp/servers/audit/_config.py").read_text()
        submit = (workspace / "agents/src/mcp/servers/audit/_submit.py").read_text()
        handler = (workspace / "agents/src/mcp/servers/audit/_handlers.py").read_text()
        raw_key_in_mcp = "SENTINEL_OPERATOR_KEY" in config and "from_key(_OPERATOR_KEY)" in submit
        mutation_advertised = 'name="submit_audit"' in handler
        return _result(
            not raw_key_in_mcp and not mutation_advertised,
            [
                _assertion(
                    "analysis_process_has_no_raw_key",
                    not raw_key_in_mcp,
                    f"raw_key_path={raw_key_in_mcp}",
                ),
                _assertion(
                    "mcp_does_not_advertise_signing",
                    not mutation_advertised,
                    f"advertised={mutation_advertised}",
                ),
            ],
        )

    if probe_name == "proof-identity":
        source = (workspace / "agents/src/mcp/servers/audit/_submit.py").read_text()
        has_v2_submit = "submitAuditV2(" in source
        has_chain_binding = '"chain_id"' in source or "chainId" in source
        has_round_binding = '"round_id"' in source or "roundId" in source
        passed = not has_v2_submit or (has_chain_binding and has_round_binding)
        return _result(
            passed,
            [
                _assertion(
                    "proof_binds_chain", has_chain_binding, f"chain_binding={has_chain_binding}"
                ),
                _assertion(
                    "proof_binds_round", has_round_binding, f"round_binding={has_round_binding}"
                ),
            ],
        )

    if probe_name == "transaction-truth":
        source = (workspace / "agents/src/mcp/servers/audit/_submit.py").read_text()
        fixed_gas = '"gas": 1_000_000' in source
        receipt_checked = (
            'receipt["status"]' in source
            or "receipt['status']" in source
            or 'receipt.get("status")' in source
            or "receipt.get('status')" in source
        )
        passed = not fixed_gas and receipt_checked
        return _result(
            passed,
            [
                _assertion("transaction_gas_is_estimated", not fixed_gas, f"fixed_gas={fixed_gas}"),
                _assertion(
                    "receipt_status_is_required", receipt_checked, f"checked={receipt_checked}"
                ),
            ],
        )

    raise ValueError(f"Unknown source probe: {probe_name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "probe",
        choices=(
            "ml-outage",
            "gateway-auth",
            "zip-containment",
            "report-path",
            "data-release",
            "signer-boundary",
            "proof-identity",
            "transaction-truth",
        ),
    )
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()

    if args.probe == "ml-outage":
        result = probe_ml_outage(workspace)
    elif args.probe == "gateway-auth":
        result = probe_gateway_auth(workspace)
    elif args.probe == "zip-containment":
        result = probe_zip_containment(workspace)
    else:
        result = _source_probe(workspace, args.probe)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
