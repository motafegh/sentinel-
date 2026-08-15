"""Phase-8 repaired preprocessing pipeline.

This pipeline is intentionally separate from the historical Stage-1 pipeline.
It writes only to a versioned R4-v2 root and never mutates historical
``data/preprocessed`` artifacts.

Execution order per source record:

    raw source -> flatten -> lexical normalize -> compile exact normalized text
               -> segment/metadata -> stage

After all workers finish, the parent process deterministically aggregates
provenance and promotes staged files.  Multiprocessing therefore cannot decide
which source record wins a content-addressed filename.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sentinel_data.preprocessing.compiler import compile_contract
from sentinel_data.preprocessing.deduplicator import Deduplicator
from sentinel_data.preprocessing.flattener import flatten_contract
from sentinel_data.preprocessing.normalizer import NORMALIZER_VERSION, normalize
from sentinel_data.preprocessing.r4_versions import (
    PREPROCESSING_ARTIFACT_VERSION,
    PREPROCESSING_META_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
)
from sentinel_data.preprocessing.r4_raw_verifier import require_manifest_source
from sentinel_data.preprocessing.segmenter import segment_and_bucket


@dataclass(frozen=True)
class PreparedRecord:
    source_name: str
    original_path: str
    source_record_id: str
    raw_sha256: str
    flattened_sha256: str
    normalized_text_sha256: str
    normalized_code_sha256: str
    address_literals: tuple[str, ...]
    pragma: str
    solc_version: str
    attempted_solc_versions: tuple[str, ...]
    command_flags: tuple[str, ...]
    flatten_status: str
    version_bucket: str
    has_unchecked_block: bool
    contract_names: tuple[str, ...]
    n_raw_lines: int
    n_normalized_lines: int
    normalizer_version: str
    staging_path: str
    ingestion_entry: dict[str, Any]


@dataclass(frozen=True)
class RepairResult:
    source: str
    records_seen: int
    records_prepared: int
    records_dropped: int
    artifacts_written: int
    exact_normalized_duplicates_aggregated: int
    normalized_code_groups: int
    address_candidate_groups: int
    output_dir: str
    manifest_records_total: int
    records_requested: int
    complete_source_build: bool


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_record_id(source: str, original_path: str, raw_sha256: str) -> str:
    payload = f"{source}\0{original_path}\0{raw_sha256}"
    return _sha256_text(payload)


def _prepare_one(
    source_name: str,
    sol_path: Path,
    raw_base: Path,
    staging_dir: Path,
    ingestion_entry: dict[str, Any],
) -> tuple[PreparedRecord | None, dict[str, Any] | None]:
    """Prepare one record without promoting it into the repaired artifact root."""

    original_path = str(sol_path.relative_to(raw_base))
    try:
        source = sol_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = sol_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, {
            "source": source_name,
            "original_path": original_path,
            "reason": "read_failed",
            "error": repr(exc),
        }

    raw_sha = _sha256_text(source)
    record_id = _source_record_id(source_name, original_path, raw_sha)
    n_raw_lines = source.count("\n") + 1

    try:
        flat = flatten_contract(sol_path)
        normalized = normalize(flat.content, preserve_line_structure=True)
    except Exception as exc:
        return None, {
            "source": source_name,
            "original_path": original_path,
            "source_record_id": record_id,
            "reason": "flatten_or_normalize_failed",
            "error": repr(exc),
        }

    normalized_sha = _sha256_text(normalized.content)
    dedup_identity = Deduplicator().process(normalized.content, sol_path)

    # Compile the exact bytes that will later be promoted.  The temporary file
    # lives beside the source so relative imports retain their original base.
    compile_tmp = sol_path.parent / (
        f".sentinel_r4_normalized_{sol_path.stem}_{record_id[:12]}.sol"
    )
    try:
        compile_tmp.write_text(normalized.content, encoding="utf-8")
        compile_result = compile_contract(compile_tmp)
    finally:
        try:
            compile_tmp.unlink()
        except OSError:
            pass

    if not compile_result.success:
        return None, {
            "source": source_name,
            "original_path": original_path,
            "source_record_id": record_id,
            "normalized_text_sha256": normalized_sha,
            "pragma": compile_result.pragma_raw,
            "reason": "normalized_compile_failed",
            "error": compile_result.error[:500],
            "attempted_solc_versions": list(compile_result.attempted_versions or []),
            "command_flags": list(compile_result.command_flags or []),
        }

    segment = segment_and_bucket(normalized.content, compile_result.pragma_raw)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / f"{record_id}.sol"
    staging_path.write_text(normalized.content, encoding="utf-8")

    return PreparedRecord(
        source_name=source_name,
        original_path=original_path,
        source_record_id=record_id,
        raw_sha256=raw_sha,
        flattened_sha256=_sha256_text(flat.content),
        normalized_text_sha256=normalized_sha,
        normalized_code_sha256=dedup_identity.normalized_sha256,
        address_literals=dedup_identity.address_literals,
        pragma=compile_result.pragma_raw,
        solc_version=compile_result.solc_version,
        attempted_solc_versions=tuple(compile_result.attempted_versions or []),
        command_flags=tuple(compile_result.command_flags or []),
        flatten_status=flat.flatten_status,
        version_bucket=segment.version_bucket,
        has_unchecked_block=segment.has_unchecked_block,
        contract_names=tuple(segment.contract_names),
        n_raw_lines=n_raw_lines,
        n_normalized_lines=normalized.n_lines_after,
        normalizer_version=normalized.normalizer_version,
        staging_path=str(staging_path),
        ingestion_entry=dict(ingestion_entry),
    ), None


def _prepare_worker(args: tuple[Any, ...]) -> tuple[PreparedRecord | None, dict[str, Any] | None]:
    source_name, sol_path, raw_base, staging_dir, ingestion_entry = args
    try:
        return _prepare_one(
            source_name,
            Path(sol_path),
            Path(raw_base),
            Path(staging_dir),
            ingestion_entry,
        )
    except Exception as exc:  # fail represented, never silently skipped
        return None, {
            "source": source_name,
            "original_path": str(sol_path),
            "reason": "worker_exception",
            "error": repr(exc),
        }


def _source_provenance(record: PreparedRecord) -> dict[str, Any]:
    return {
        "source_name": record.source_name,
        "original_path": record.original_path,
        "source_record_id": record.source_record_id,
        "raw_sha256": record.raw_sha256,
        "flattened_sha256": record.flattened_sha256,
        "ingestion_entry": record.ingestion_entry,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _require_empty_output(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"repaired output root is not empty: {out_dir}. "
            "R4 rebuilds are immutable; use a fresh rebuild directory or "
            "explicitly archive/remove a failed local attempt before rerunning."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _materialize(
    source_name: str,
    records: list[PreparedRecord],
    drops: list[dict[str, Any]],
    out_dir: Path,
    *,
    manifest_records_total: int | None = None,
    records_requested: int | None = None,
    requested_limit: int | None = None,
    raw_verification: dict[str, Any] | None = None,
) -> RepairResult:
    """Deterministically aggregate provenance and promote staged records."""

    records = sorted(records, key=lambda r: r.source_record_id)
    by_artifact: dict[str, list[PreparedRecord]] = {}
    by_normalized_code: dict[str, set[str]] = {}
    address_to_artifacts: dict[str, set[str]] = {}

    for record in records:
        by_artifact.setdefault(record.normalized_text_sha256, []).append(record)
        by_normalized_code.setdefault(record.normalized_code_sha256, set()).add(
            record.normalized_text_sha256
        )
        for address in record.address_literals:
            address_to_artifacts.setdefault(address, set()).add(record.normalized_text_sha256)

    artifact_rows: list[dict[str, Any]] = []
    for artifact_sha in sorted(by_artifact):
        members = sorted(by_artifact[artifact_sha], key=lambda r: r.source_record_id)
        canonical = members[0]
        source_records = [_source_provenance(member) for member in members]
        addresses = sorted({a for member in members for a in member.address_literals})

        src = Path(canonical.staging_path)
        dst = out_dir / f"{artifact_sha}.sol"
        shutil.copyfile(src, dst)

        meta = {
            "sha256": artifact_sha,
            "normalized_text_sha256": artifact_sha,
            "normalized_code_sha256": canonical.normalized_code_sha256,
            "dedup_group_id": f"norm:{canonical.normalized_code_sha256}",
            "leakage_family_seed": f"norm:{canonical.normalized_code_sha256}",
            "source_name": canonical.source_name,
            "original_path": canonical.original_path,
            "source_record_id": canonical.source_record_id,
            "source_records": source_records,
            "source_record_count": len(source_records),
            "address_literals": addresses,
            "pragma": canonical.pragma,
            "solc_version": canonical.solc_version,
            "compile_status": "ok_normalized_bytes",
            "compile_error": "",
            "attempted_solc_versions": list(canonical.attempted_solc_versions),
            "compiler_command_flags": list(canonical.command_flags),
            "flatten_status": canonical.flatten_status,
            "version_bucket": canonical.version_bucket,
            "has_unchecked_block": canonical.has_unchecked_block,
            "contract_names": list(canonical.contract_names),
            "n_raw_lines": canonical.n_raw_lines,
            "n_normalized_lines": canonical.n_normalized_lines,
            "normalizer_version": canonical.normalizer_version,
            "meta_schema_version": PREPROCESSING_META_SCHEMA_VERSION,
            "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        }
        (out_dir / f"{artifact_sha}.meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_rows.append(
            {
                "sha256": artifact_sha,
                "normalized_code_sha256": canonical.normalized_code_sha256,
                "source_record_count": len(source_records),
            }
        )

    normalized_groups = {
        key: sorted(values)
        for key, values in sorted(by_normalized_code.items())
        if len(values) > 1
    }
    address_candidates = {
        address: sorted(values)
        for address, values in sorted(address_to_artifacts.items())
        if len(values) > 1
    }

    requested = (
        len(records) + len(drops)
        if records_requested is None
        else int(records_requested)
    )
    manifest_total = requested if manifest_records_total is None else int(manifest_records_total)
    complete_source_build = requested == manifest_total and requested_limit is None
    manifest = {
        "status": "REPOSITORY_INTERFACE_ONLY_PHYSICAL_ACCEPTANCE_PENDING",
        "source": source_name,
        "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
        "meta_schema_version": PREPROCESSING_META_SCHEMA_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "records_prepared": len(records),
        "records_dropped": len(drops),
        "manifest_records_total": manifest_total,
        "records_requested": requested,
        "requested_limit": requested_limit,
        "complete_source_build": complete_source_build,
        "raw_manifest_verification_passed": bool(
            (raw_verification or {}).get("passed", False)
        ),
        "raw_manifest_sha256": (raw_verification or {}).get("manifest_sha256"),
        "artifacts_written": len(by_artifact),
        "exact_normalized_duplicates_aggregated": len(records) - len(by_artifact),
        "normalized_code_groups_requiring_group_atomic_roles": normalized_groups,
        "address_family_candidates_not_auto_merged": address_candidates,
        "artifacts": artifact_rows,
    }
    (out_dir / "repaired_preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "dropped.csv", sorted(drops, key=lambda r: str(r)))

    return RepairResult(
        source=source_name,
        records_seen=len(records) + len(drops),
        records_prepared=len(records),
        records_dropped=len(drops),
        artifacts_written=len(by_artifact),
        exact_normalized_duplicates_aggregated=len(records) - len(by_artifact),
        normalized_code_groups=len(normalized_groups),
        address_candidate_groups=len(address_candidates),
        output_dir=str(out_dir),
        manifest_records_total=manifest_total,
        records_requested=requested,
        complete_source_build=complete_source_build,
    )


def run_repaired_source(
    source_name: str,
    raw_dir: Path,
    ingestion_manifest: Path,
    out_dir: Path,
    *,
    n_workers: int = 1,
    limit: int | None = None,
) -> RepairResult:
    """Build one source into a fresh repaired/versioned preprocessing root.

    This function needs the local Git-ignored raw corpus.  Repository-only tests
    exercise its deterministic primitives with synthetic fixtures; they do not
    constitute physical DATA acceptance.
    """

    raw_verification = require_manifest_source(
        source_name,
        raw_dir,
        ingestion_manifest,
    )
    manifest = json.loads(ingestion_manifest.read_text(encoding="utf-8"))
    all_entries = sorted(manifest.get("files", []), key=lambda item: item["path"])
    entries = all_entries
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        entries = entries[:limit]
    _require_empty_output(out_dir)

    staging_dir = out_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    args = [
        (
            source_name,
            str(raw_dir / entry["path"]),
            str(raw_dir),
            str(staging_dir),
            dict(entry),
        )
        for entry in entries
    ]

    prepared: list[PreparedRecord] = []
    drops: list[dict[str, Any]] = []
    try:
        if n_workers > 1:
            chunksize = max(1, len(args) // (n_workers * 16)) if args else 1
            with mp.Pool(processes=n_workers) as pool:
                results: Iterable[tuple[PreparedRecord | None, dict[str, Any] | None]] = (
                    pool.imap(_prepare_worker, args, chunksize=chunksize)
                )
                for record, drop in results:
                    if record is not None:
                        prepared.append(record)
                    elif drop is not None:
                        drops.append(drop)
        else:
            for args_one in args:
                record, drop = _prepare_worker(args_one)
                if record is not None:
                    prepared.append(record)
                elif drop is not None:
                    drops.append(drop)

        return _materialize(
            source_name,
            prepared,
            drops,
            out_dir,
            manifest_records_total=len(all_entries),
            records_requested=len(entries),
            requested_limit=limit,
            raw_verification=raw_verification,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
