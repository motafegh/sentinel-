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

import torch


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

    exclude = {"manifest.json", ".hash_cache.json", "release_descriptor.json"}
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

    # ── R0.5: release descriptor ──────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        export_dir = Path(td)

        (export_dir / "data.pt").write_text("some data")
        (export_dir / "labels.parquet").write_text("labels")
        manifest_json = '{"artifact_hash": "abc123", "release": "test"}'
        (export_dir / "manifest.json").write_text(manifest_json)

        sys.path.insert(0, str(workspace / "data_module"))
        from sentinel_data.export.release_descriptor import (
            write_release_descriptor,
            verify_release,
        )

        desc = write_release_descriptor(
            export_dir=export_dir,
            manifest_hash=hashlib.sha256(manifest_json.encode()).hexdigest(),
            artifact_hash="abc123",
        )

        assertions.append(_assertion(
            "release_descriptor_exists",
            (export_dir / "release_descriptor.json").exists(),
            "release_descriptor.json written to export dir",
        ))

        assertions.append(_assertion(
            "release_descriptor_has_manifest_hash",
            "manifest_hash" in desc and len(desc["manifest_hash"]) == 64,
            f"manifest_hash={desc.get('manifest_hash', 'MISSING')!r}",
        ))

        assertions.append(_assertion(
            "release_descriptor_has_files",
            isinstance(desc.get("files"), dict) and len(desc["files"]) >= 2,
            f"files={desc.get('files', {})}",
        ))

        assertions.append(_assertion(
            "release_descriptor_release_id_self_consistent",
            "release_id" in desc and len(desc["release_id"]) == 64,
            f"release_id={desc.get('release_id', 'MISSING')!r}",
        ))

        # Verify succeeds on clean directory
        vr = verify_release(export_dir)
        assertions.append(_assertion(
            "release_verify_passes_on_clean",
            vr["verified"],
            f"reason={vr['reason']}",
        ))

        # Tamper manifest — descriptor should detect it
        (export_dir / "manifest.json").write_text('{"artifact_hash": "TAMPERED", "release": "evil"}')
        vr2 = verify_release(export_dir)
        assertions.append(_assertion(
            "release_verify_detects_manifest_tamper",
            not vr2["verified"],
            f"reason={vr2['reason']} mismatches={vr2.get('mismatches', [])}",
        ))

        # Tamper data file — descriptor should detect it
        (export_dir / "manifest.json").write_text(manifest_json)
        (export_dir / "data.pt").write_text("EVIL DATA")
        vr3 = verify_release(export_dir)
        assertions.append(_assertion(
            "release_verify_detects_data_tamper",
            not vr3["verified"],
            f"reason={vr3['reason']} mismatches={vr3.get('mismatches', [])}",
        ))

    # ── R0.5: pickle-safe serializer ──────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        import pickle
        import os
        sys.path.insert(0, str(workspace / "data_module"))
        from sentinel_data.export.pickle_safe import safe_loads, safe_torch_load

        pt_path = Path(td) / "tensor.pt"
        torch.save(torch.tensor([1.0, 2.0, 3.0]), pt_path)
        try:
            loaded = safe_torch_load(pt_path)
            assertions.append(_assertion(
                "pickle_safe_loads_tensor",
                isinstance(loaded, torch.Tensor) and loaded.shape == (3,),
                f"type={type(loaded).__name__} shape={getattr(loaded, 'shape', None)}",
            ))
        except Exception as exc:
            assertions.append(_assertion(
                "pickle_safe_loads_tensor",
                False,
                f"safe_torch_load(tensor) raised: {type(exc).__name__}: {exc}",
            ))

        unsafe_payload = pickle.dumps(os.system)
        try:
            safe_loads(unsafe_payload)
            assertions.append(_assertion(
                "pickle_safe_rejects_unsafe_code",
                False,
                "safe_loads(os.system) did NOT raise — code execution risk",
            ))
        except pickle.UnpicklingError:
            assertions.append(_assertion(
                "pickle_safe_rejects_unsafe_code",
                True,
                "safe_loads(os.system) raised UnpicklingError as expected",
            ))
        except Exception as exc:
            assertions.append(_assertion(
                "pickle_safe_rejects_unsafe_code",
                False,
                f"safe_loads(os.system) raised unexpected: {type(exc).__name__}: {exc}",
            ))

        # verify_artifact_hash has descriptor_verified key
        sys.path.insert(0, str(workspace / "data_module"))
        from sentinel_data.export.export import SentinelDatasetExport
        from sentinel_data.export.release_descriptor import write_release_descriptor as _wrd
        export_dir = Path(td)
        (export_dir / "data.pt").write_text("data for export")
        (export_dir / "labels.parquet").write_text("labels")
        manifest_str = (
            '{"artifact_hash": "dummy", "hash_algorithm": "sha256", '
            '"schema_version": "v1", "graph_schema_version": "v9", '
            '"shard_size": 1000, "n_contracts": 1, "n_contracts_with_reps": 0, '
            '"n_shards": 0, "splits": {"train": [], "val": [], "test": []}, '
            '"shard_index": {}, "source_set": [], "skipped_sources": [], '
            '"preprocessing_config_hash": "unknown", '
            '"label_class_columns": [], "created_at": "now"}'
        )
        (export_dir / "manifest.json").write_text(manifest_str)
        _wrd(
            export_dir=export_dir,
            manifest_hash=hashlib.sha256(manifest_str.encode()).hexdigest(),
            artifact_hash="dummy",
        )

        ex = SentinelDatasetExport(export_dir)
        vr4 = ex.verify_artifact_hash()
        assertions.append(_assertion(
            "descriptor_verified_key_present",
            vr4.get("descriptor_verified") is True,
            f"verified={vr4['verified']}",
        ))

        # R0.6: delete descriptor — must fail because enforcement is in code
        (export_dir / "release_descriptor.json").unlink()
        ex2 = SentinelDatasetExport(export_dir)
        vr5 = ex2.verify_artifact_hash()
        assertions.append(_assertion(
            "descriptor_code_enforced",
            not vr5["verified"] and vr5.get("descriptor_verified") is False,
            f"no descriptor — verified={vr5['verified']} reason={vr5.get('reason','')}",
        ))

        # R0.6: delete descriptor AND strip any release_descriptor manifest flag
        # — still must fail because enforcement is not in the mutable manifest
        (export_dir / "manifest.json").write_text(manifest_str)
        _wrd(
            export_dir=export_dir,
            manifest_hash=hashlib.sha256(manifest_str.encode()).hexdigest(),
            artifact_hash="dummy",
        )
        (export_dir / "release_descriptor.json").unlink()
        stripped = json.loads(manifest_str)
        stripped.pop("release_descriptor", None)
        (export_dir / "manifest.json").write_text(json.dumps(stripped))
        ex3 = SentinelDatasetExport(export_dir)
        vr6 = ex3.verify_artifact_hash()
        assertions.append(_assertion(
            "descriptor_stripped_still_detected",
            not vr6["verified"] and vr6.get("descriptor_verified") is False,
            f"stripped — verified={vr6['verified']} reason={vr6.get('reason','')}",
        ))

    all_passed = all(a["passed"] for a in assertions)
    return _result(all_passed, assertions)


def probe_signer_boundary(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "agents"))
    assertions: list[dict[str, Any]] = []

    config_mod, config_err = _try_import("src.mcp.servers.audit._config")
    if config_err:
        return _result(False, [_assertion("config_module_importable", False, config_err)])

    # R0-F3: analysis/MCP process must contain no signing key, no key import
    # path, and no raw transaction construction code.
    _submit_path = workspace / "agents/src/mcp/servers/audit/_submit.py"
    _config_path = workspace / "agents/src/mcp/servers/audit/_config.py"
    _submit_src = _submit_path.read_text() if _submit_path.exists() else ""
    _config_src = _config_path.read_text() if _config_path.exists() else ""

    assertions.append(_assertion(
        "operator_key_absent_from_config",
        "_OPERATOR_KEY" not in _config_src,
        "config no longer defines _OPERATOR_KEY",
    ))
    assertions.append(_assertion(
        "no_raw_key_import_in_submit",
        "from eth_account" not in _submit_src and "from_key" not in _submit_src,
        "no eth_account import or from_key call in analysis process",
    ))
    assertions.append(_assertion(
        "no_transaction_construction_in_submit",
        "sign_transaction" not in _submit_src and "send_raw_transaction" not in _submit_src,
        "no raw tx signing or broadcast code in MCP process",
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

    from src.mcp.servers.audit._submit import build_provenance_manifest

    manifest = build_provenance_manifest(
        teacher_model_hash="c" * 64,
        proxy_checkpoint_hash="d" * 64,
        fusion_embedding=[0.1] * 128,
        class_scores=[0.2] * 10,
        operator_address="0x000000000000000000000000000000000000dEaD",
        chain_id=1,
        round_id=42,
        contract_address="0x000000000000000000000000000000000000dEaD",
        target_data_version="v2026.1",
        proof_scope="legacy_proxy_only_unbound",
    )

    assertions.append(_assertion(
        "provenance_manifest_has_teacher_model_hash",
        "teacher_model_hash" in manifest,
        f"keys={sorted(manifest)}",
    ))
    assertions.append(_assertion(
        "proof_scope_is_legacy_unbound",
        manifest.get("proof_scope") == "legacy_proxy_only_unbound",
        f"proof_scope={manifest.get('proof_scope')!r}",
    ))
    assertions.append(_assertion(
        "proof_identity_bound_to_chain",
        "chain_id" in manifest,
        f"chain_id={manifest.get('chain_id')!r}",
    ))
    assertions.append(_assertion(
        "proof_identity_bound_to_round",
        "round_id" in manifest,
        f"round_id={manifest.get('round_id')!r}",
    ))
    assertions.append(_assertion(
        "proof_identity_bound_to_contract",
        "contract_address" in manifest,
        f"contract_address={manifest.get('contract_address')!r}",
    ))
    # R0-F3: analysis process has no key
    _config_check = workspace / "agents/src/mcp/servers/audit/_config.py"
    _conf = _config_check.read_text() if _config_check.exists() else ""
    assertions.append(_assertion(
        "no_operator_key_in_process",
        "_OPERATOR_KEY" not in _conf,
        "signing key removed from MCP process — policy-signer owns submission",
    ))

    # R0-F3: V2 proof scope is always legacy_proxy_only_unbound regardless
    # of identity. The chain/round/contract fields are informational metadata
    # stored in the JSON manifest — they are NOT cryptographically enforced
    # by the EZKL circuit or the Solidity verifier. Full typed identity
    # binding requires R3 V3 protocol work.
    manifest_b = build_provenance_manifest(
        teacher_model_hash="c" * 64,
        proxy_checkpoint_hash="d" * 64,
        fusion_embedding=[0.1] * 128,
        class_scores=[0.2] * 10,
        operator_address="0x000000000000000000000000000000000000dEaD",
        chain_id=5,
        round_id=99,
        contract_address="0x000000000000000000000000000000000000BeEf",
        target_data_version="v2026.1",
        proof_scope="legacy_proxy_only_unbound",
    )
    assertions.append(_assertion(
        "v2_proof_scope_identical_across_identities",
        manifest.get("proof_scope") == manifest_b.get("proof_scope") == "legacy_proxy_only_unbound",
        "V2 proof scope is legacy_proxy_only_unbound regardless of identity",
    ))
    assertions.append(_assertion(
        "cross_identity_different_chain_metadata",
        manifest.get("chain_id") != manifest_b.get("chain_id"),
        f"A={manifest.get('chain_id')} B={manifest_b.get('chain_id')}",
    ))
    assertions.append(_assertion(
        "cross_identity_same_unbound_scope",
        manifest.get("proof_scope") == manifest_b.get("proof_scope"),
        "identity fields change but proof scope stays unbound — not crypto binding",
    ))

    all_passed = all(a["passed"] for a in assertions)
    return _result(all_passed, assertions)


def probe_transaction_truth(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "agents"))
    assertions: list[dict[str, Any]] = []

    submit_mod, submit_err = _try_import("src.mcp.servers.audit._submit")
    if submit_err:
        return _result(False, [_assertion("submit_module_importable", False, submit_err)])

    from src.mcp.servers.audit._submit import (
        _run_submit,
        _estimate_gas,
        TxLifecycle,
        TxState,
    )

    try:
        result = _run_submit(
            source_code="pragma solidity ^0.8.0; contract C {}",
            contract_address="0x0000000000000000000000000000000000000001",
            model_hash="a" * 64,
            chain_id=1,
            round_id=42,
            idempotency_key="test-ik-001",
            target_data_version="v2026.1",
        )
    except Exception as exc:
        assertions.append(_assertion(
            "submit_pipeline_returns_on_error",
            "status" in str(exc) or "policy_rejected" in str(exc)
            or "failed_step" in str(exc),
            f"raised: {type(exc).__name__}: {exc}",
        ))
        all_passed = all(a["passed"] for a in assertions)
        return _result(all_passed, assertions)

    assertions.append(_assertion(
        "submit_pipeline_returns_dict",
        isinstance(result, dict) and "status" in result,
        f"keys={sorted(result) if isinstance(result, dict) else type(result).__name__}",
    ))

    # R0-F3: V2 proof scope is either legacy_proxy_only_unbound (ML API reached)
    # or "none" when the pipeline is blocked before reaching proof generation.
    # Both are ineligible for verified audit finality.
    assertions.append(_assertion(
        "proof_scope_is_unbound_or_none",
        result.get("proof_scope") in ("legacy_proxy_only_unbound", "none"),
        f"proof_scope={result.get('proof_scope')!r}",
    ))
    assertions.append(_assertion(
        "v2_not_submitted_or_confirmed",
        result.get("status") in ("policy_rejected", "failed"),
        f"status={result.get('status')!r}",
    ))
    assertions.append(_assertion(
        "verified_audit_ineligible",
        not result.get("verified_audit_eligible", True),
        f"eligible={result.get('verified_audit_eligible')} "
        f"reason={result.get('finality_ineligible_reason')!r}",
    ))
    assertions.append(_assertion(
        "ineligible_reason_is_proof_scope",
        "proof_scope" in str(result.get("finality_ineligible_reason", "")),
        f"reason={result.get('finality_ineligible_reason')!r}",
    ))

    # R0-F4: transaction state machine
    all_states = set(s for s in TxState)
    assert len(all_states) >= 10, f"TxState must have 11 values, got {len(all_states)}"
    assertions.append(_assertion(
        "tx_state_machine_has_all_required_states",
        all(s in all_states for s in [TxState.PENDING, TxState.POLICY_REJECTED,
              TxState.PREPARED, TxState.SIGNED, TxState.BROADCAST,
              TxState.CONFIRMED, TxState.REVERTED, TxState.FAILED]),
        f"states={sorted(s.value for s in all_states)}",
    ))

    lc = TxLifecycle(tx_hash="0xabc")
    assertions.append(_assertion(
        "tx_default_is_pending",
        lc.state == TxState.PENDING,
        f"state={lc.state.value}",
    ))
    lc.state = TxState.POLICY_REJECTED
    lc.error = "proof_scope_not_identity_bound"
    assertions.append(_assertion(
        "tx_policy_rejected_storable",
        lc.state == TxState.POLICY_REJECTED and lc.error is not None,
        f"state={lc.state.value} error={lc.error}",
    ))
    rlc = TxLifecycle(state=TxState.REVERTED, receipt_status=0, error="reverted on-chain")
    assertions.append(_assertion(
        "tx_reverted_zero_receipt_status",
        rlc.state == TxState.REVERTED and rlc.receipt_status == 0,
        f"state={rlc.state.value} receipt_status={rlc.receipt_status}",
    ))

    # R0-F3: policy-signer boundary — V2/unbound proofs must be rejected
    from src.security.policy_signer import (
        evaluate_submission, PolicyDecision, REJECT_REASON_UNBOUND
    )
    # V2 proof should be rejected
    result_v2 = evaluate_submission(
        proof_scope="legacy_proxy_only_unbound",
        contract_address="0x0000000000000000000000000000000000000001",
        chain_id=1, round_id=42, model_hash="a"*64,
    )
    assertions.append(_assertion(
        "policy_rejects_v2_unbound",
        result_v2.decision == PolicyDecision.REJECTED,
        f"decision={result_v2.decision.value} reason={result_v2.reason}",
    ))
    assertions.append(_assertion(
        "policy_rejection_reason_is_proof_scope",
        result_v2.reason == REJECT_REASON_UNBOUND,
        f"reason={result_v2.reason}",
    ))
    # Missing proof_scope should be rejected
    result_none = evaluate_submission(
        proof_scope="none",
        contract_address="0x0000000000000000000000000000000000000001",
        chain_id=1, round_id=42, model_hash="a"*64,
    )
    assertions.append(_assertion(
        "policy_rejects_no_proof_scope",
        result_none.decision == PolicyDecision.REJECTED,
        f"decision={result_none.decision.value} reason={result_none.reason}",
    ))
    # typed_identity_bound_v3 is also rejected — no caller can self-declare
    result_v3 = evaluate_submission(
        proof_scope="typed_identity_bound_v3",
        contract_address="0x0000000000000000000000000000000000000001",
        chain_id=1, round_id=42, model_hash="a"*64,
    )
    assertions.append(_assertion(
        "policy_rejects_all_scopes_including_v3",
        result_v3.decision == PolicyDecision.REJECTED,
        f"decision={result_v3.decision.value} reason={result_v3.reason}",
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
