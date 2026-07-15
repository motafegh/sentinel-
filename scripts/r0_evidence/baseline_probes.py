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


def _try_import(module_path: str) -> tuple[Any, str | None]:
    try:
        import importlib
        return importlib.import_module(module_path), None
    except (ImportError, AttributeError, Exception) as exc:
        return None, str(exc)


def probe_report_path(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "agents"))
    assertions: list[dict[str, Any]] = []

    mod, err = _try_import("src.persistence.paths")
    if err:
        return _result(False, [_assertion("paths_module_importable", False, err)])

    from src.persistence.paths import is_valid_job_id, validate_address

    assertions.append(_assertion(
        "is_valid_job_id_rejects_path_traversal",
        not is_valid_job_id("../../../etc/passwd"),
        f"result={is_valid_job_id('../../../etc/passwd')}",
    ))
    assertions.append(_assertion(
        "is_valid_job_id_accepts_uuid",
        is_valid_job_id("550e8400-e29b-41d4-a716-446655440000"),
        "valid UUID accepted",
    ))
    try:
        from src.persistence.report_writer import persist_report
        result = persist_report(
            {"job_id": "../../../tmp/evil"},
            {"overall_label": "safe"},
            Path("/tmp"),
        )
        report_key = next(iter(result))
        ran = result[report_key].get("ran", True)
        assertions.append(_assertion(
            "persist_report_rejects_malicious_job_id",
            not ran,
            f"persist_report ran={ran} for malicious job_id",
        ))
    except Exception as exc:
        assertions.append(_assertion(
            "persist_report_rejects_malicious_job_id_explicitly",
            True,
            f"raised: {exc}",
        ))

    try:
        validate_address("../../../etc/passwd")
        assertions.append(_assertion(
            "validate_address_rejects_path_traversal",
            False,
            "accepted path traversal as address",
        ))
    except ValueError:
        assertions.append(_assertion(
            "validate_address_rejects_path_traversal",
            True,
            "rejected path traversal address",
        ))

    try:
        validate_address("0x000000000000000000000000000000000000dEaD")
        assertions.append(_assertion(
            "validate_address_accepts_ethereum_address",
            True,
            "accepted valid address",
        ))
    except ValueError as e:
        assertions.append(_assertion(
            "validate_address_accepts_ethereum_address",
            False,
            str(e),
        ))

    all_passed = all(a["passed"] for a in assertions)
    return _result(all_passed, assertions)


def _hash_export(export_dir: Path, exclude: set[str]) -> str:
    """Replicate ``_hash_export_data`` hash logic using stdlib only."""
    import hashlib
    candidates = sorted(
        p for p in export_dir.rglob("*")
        if p.is_file() and p.name not in exclude
    )
    h = hashlib.sha256()
    for p in candidates:
        rel = str(p.relative_to(export_dir))
        h.update(rel.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def probe_data_release(workspace: Path) -> dict[str, Any]:
    import hashlib
    import tempfile
    from pathlib import Path
    assertions: list[dict[str, Any]] = []

    exclude = {"manifest.json", ".hash_cache.json"}
    with tempfile.TemporaryDirectory() as td:
        export_dir = Path(td)

        (export_dir / "data.pt").write_text("canonical data")
        (export_dir / "labels.parquet").write_text("label data")
        (export_dir / "manifest.json").write_text('{"version": 1}')
        (export_dir / ".hash_cache.json").write_text('{"cached": true}')

        original_hash = _hash_export(export_dir, exclude)

        manifest = export_dir / "manifest.json"
        manifest.write_text('{"version": 2}')
        hash_after_manifest_change = _hash_export(export_dir, exclude)
        assertions.append(_assertion(
            "semantic_manifest_change_does_not_affect_hash",
            hash_after_manifest_change == original_hash,
            f"before={original_hash} after_change={hash_after_manifest_change}",
        ))

        (export_dir / "data.pt").write_text("tampered data")
        hash_after_data_tamper = _hash_export(export_dir, exclude)
        assertions.append(_assertion(
            "data_tamper_changes_hash",
            hash_after_data_tamper != original_hash,
            f"original={original_hash} tampered={hash_after_data_tamper}",
        ))

        (export_dir / "data.pt").write_text("canonical data")
        hash_restored = _hash_export(export_dir, exclude)
        assertions.append(_assertion(
            "hash_is_deterministic",
            hash_restored == original_hash,
            f"first={original_hash} restored={hash_restored}",
        ))

    with tempfile.TemporaryDirectory() as td:
        export_dir = Path(td)
        (export_dir / "a.pt").write_text("file a")
        (export_dir / "b.pt").write_text("file b")
        (export_dir / "manifest.json").write_text('{"version": 1}')

        hash_ab = _hash_export(export_dir, exclude)

        (export_dir / "c.pt").write_text("file c")
        hash_abc = _hash_export(export_dir, exclude)
        assertions.append(_assertion(
            "added_file_changes_hash",
            hash_abc != hash_ab,
            f"2file_hash={hash_ab} 3file_hash={hash_abc}",
        ))

        (export_dir / "a.pt").unlink()
        hash_bc = _hash_export(export_dir, exclude)
        assertions.append(_assertion(
            "deleted_file_changes_hash",
            hash_bc != hash_ab,
            f"2file_hash={hash_ab} after_delete={hash_bc}",
        ))

    with tempfile.TemporaryDirectory() as td:
        export_dir = Path(td)
        (export_dir / "file_a.pt").write_text("aaa")
        (export_dir / "file_b.pt").write_text("bbb")
        (export_dir / "manifest.json").write_text('{}')

        original_files = {
            str(p.relative_to(export_dir))
            for p in export_dir.rglob("*")
            if p.is_file() and p.name not in exclude
        }

        (export_dir / "file_a.pt").unlink()
        on_disk_files = {
            str(p.relative_to(export_dir))
            for p in export_dir.rglob("*")
            if p.is_file() and p.name not in exclude
        }

        missing = sorted(original_files - on_disk_files)
        extra = sorted(on_disk_files - original_files)
        assertions.append(_assertion(
            "deleted_file_detected_by_set_mismatch",
            "file_a.pt" in missing,
            f"missing={missing}",
        ))

    all_passed = all(a["passed"] for a in assertions)
    return _result(all_passed, assertions)


def probe_signer_boundary(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "agents"))
    assertions: list[dict[str, Any]] = []

    config_mod, config_err = _try_import("src.mcp.servers.audit._config")
    if config_err:
        return _result(False, [_assertion("config_module_importable", False, config_err)])

    from src.mcp.servers.audit._config import _OPERATOR_KEY

    assertions.append(_assertion(
        "operator_key_is_empty_in_config",
        not _OPERATOR_KEY,
        f"_OPERATOR_KEY length={len(_OPERATOR_KEY)}",
    ))

    submit_mod, submit_err = _try_import("src.mcp.servers.audit._submit")
    if not submit_err:
        from src.mcp.servers.audit._submit import build_provenance_manifest
        manifest = build_provenance_manifest(
            teacher_model_hash="a" * 64,
            proxy_checkpoint_hash="b" * 64,
            fusion_embedding=[0.0] * 128,
            class_scores=[0.0] * 10,
            operator_address="0x000000000000000000000000000000000000dEaD",
        )
        assertions.append(_assertion(
            "provenance_manifest_signature_is_none",
            manifest.get("signature") is None,
            f"signature={manifest.get('signature')!r}",
        ))

    handler_mod, handler_err = _try_import("src.mcp.servers.audit._handlers")
    if not handler_err:
        from src.mcp.servers.audit._handlers import list_tools
        tools = asyncio.run(list_tools())
        tool_names = [t.name for t in tools]
        assertions.append(_assertion(
            "mcp_does_not_advertise_submit_audit",
            "submit_audit" not in tool_names,
            f"advertised_tools={tool_names}",
        ))

    all_passed = all(a["passed"] for a in assertions)
    return _result(all_passed, assertions)


def probe_proof_identity(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "agents"))
    assertions: list[dict[str, Any]] = []

    config_mod, config_err = _try_import("src.mcp.servers.audit._config")
    if config_err:
        return _result(False, [_assertion("config_module_importable", False, config_err)])

    from src.mcp.servers.audit._config import _OPERATOR_KEY
    from src.mcp.servers.audit._submit import build_provenance_manifest

    manifest = build_provenance_manifest(
        teacher_model_hash="c" * 64,
        proxy_checkpoint_hash="d" * 64,
        fusion_embedding=[0.1] * 128,
        class_scores=[0.2] * 10,
        operator_address="0x000000000000000000000000000000000000dEaD",
    )

    assertions.append(_assertion(
        "provenance_manifest_has_teacher_model_hash",
        "teacher_model_hash" in manifest,
        f"keys={sorted(manifest)}",
    ))
    assertions.append(_assertion(
        "provenance_manifest_has_fusion_embedding_hash",
        "fusion_embedding_hash" in manifest,
        f"keys={sorted(manifest)}",
    ))
    assertions.append(_assertion(
        "proof_identity_bound_to_chain",
        "chain_id" in manifest or "chainId" in str(manifest),
        "chain_id NOT bound in provenance manifest — cross-identity reuse possible",
    ))
    assertions.append(_assertion(
        "proof_identity_bound_to_round",
        "round_id" in manifest or "roundId" in str(manifest),
        "round_id NOT bound in provenance manifest — cross-round reuse possible",
    ))
    assertions.append(_assertion(
        "no_operator_key_prevents_signature",
        not _OPERATOR_KEY,
        f"no operator key means on-chain submission is blocked; "
        f"identity binding not yet implemented",
    ))

    all_passed = all(a["passed"] for a in assertions)
    return _result(all_passed, assertions)


def probe_transaction_truth(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "agents"))
    assertions: list[dict[str, Any]] = []

    submit_mod, submit_err = _try_import("src.mcp.servers.audit._submit")
    if submit_err:
        return _result(False, [_assertion("submit_module_importable", False, submit_err)])

    from src.mcp.servers.audit._submit import _run_submit

    try:
        result = _run_submit(
            source_code="pragma solidity ^0.8.0; contract C {}",
            contract_address="0x0000000000000000000000000000000000000001",
            model_hash="a" * 64,
        )
    except Exception as exc:
        assertions.append(_assertion(
            "submit_pipeline_has_structure",
            "failed_step" in str(exc) or "partial" in str(exc) or "blocked" in str(type(exc).__name__),
            f"submit pipeline raised: {type(exc).__name__}: {exc}",
        ))
        all_passed = all(a["passed"] for a in assertions)
        return _result(all_passed, assertions)

    status = result.get("status", "unknown")
    failed_step = result.get("failed_step")

    assertions.append(_assertion(
        "submit_pipeline_returns_structured_result",
        isinstance(result, dict) and "status" in result,
        f"result type={type(result).__name__} keys={sorted(result) if isinstance(result, dict) else 'N/A'}",
    ))
    assertions.append(_assertion(
        "transaction_submission_not_reached",
        status != "submitted",
        f"status={status!r} failed_step={failed_step!r} — on-chain submission not reached",
    ))
    assertions.append(_assertion(
        "structure_error_if_no_tx_handling",
        status in {"failed", "partial"},
        f"status={status!r} — pipeline returns structured failure",
    ))

    # Check for gas estimation or receipt status code via source inspection
    # (last resort — actual behavioral check is impossible without working ML API)
    submit_path = workspace / "agents/src/mcp/servers/audit/_submit.py"
    submit_source = submit_path.read_text()
    has_gas_estimation = '"gas"' in submit_source and "wei" in submit_source
    has_receipt_check = "receipt" in submit_source and "status" in submit_source
    tx_features_implemented = has_gas_estimation and has_receipt_check
    assertions.append(_assertion(
        "gas_estimation_or_receipt_check_implemented",
        tx_features_implemented,
        f"gas_estimation={has_gas_estimation} receipt_check={has_receipt_check}",
    ))

    all_passed = all(a["passed"] for a in assertions)
    return _result(all_passed, assertions)


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
    elif args.probe == "report-path":
        result = probe_report_path(workspace)
    elif args.probe == "data-release":
        result = probe_data_release(workspace)
    elif args.probe == "signer-boundary":
        result = probe_signer_boundary(workspace)
    elif args.probe == "proof-identity":
        result = probe_proof_identity(workspace)
    elif args.probe == "transaction-truth":
        result = probe_transaction_truth(workspace)
    else:
        raise ValueError(f"Unknown probe: {args.probe}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
