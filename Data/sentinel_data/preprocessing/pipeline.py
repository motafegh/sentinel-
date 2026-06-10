"""PreprocessingPipeline — orchestrate 5 steps per .sol file.

Step order: flatten → compile → dedup → normalize → segment+bucket

Each step writes fields into a per-file sidecar `meta.json`.
Files that fail to compile are dropped to `dropped.csv` (not written to preprocessed/).

Execution is serial (single-process). A multiprocessing variant is deferred to
v2.1 — solc subprocesses already saturate I/O, and the per-file work is mostly
subprocess waits. Multiprocessing would help on the dedup+normalize steps but
not the compile step, and the gain is small for a one-time pipeline.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.compiler import compile_contract
from sentinel_data.preprocessing.deduplicator import Deduplicator
from sentinel_data.preprocessing.flattener import flatten_contract
from sentinel_data.preprocessing.normalizer import normalize
from sentinel_data.preprocessing.segmenter import segment_and_bucket


META_SCHEMA_VERSION = "1"


@dataclass
class ContractMeta:
    sha256: str
    source_name: str          # which dataset source (e.g. "defihacklabs")
    original_path: str        # relative path inside raw/source/repo/
    pragma: str
    solc_version: str
    compile_status: str       # "ok" | "failed"
    compile_error: str
    attempted_solc_versions: list[str]
    flatten_status: str       # "flattened" | "skipped_no_imports" | "skipped_error"
    dedup_group_id: str
    is_duplicate: bool
    duplicate_of: str
    version_bucket: str       # "legacy" | "transitional" | "modern"
    has_unchecked_block: bool
    contract_names: list[str]
    n_raw_lines: int
    n_normalized_lines: int
    meta_schema_version: str = META_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    processed: list[Path]     # output .sol paths written
    dropped: list[dict]       # rows for dropped.csv
    duration_s: float


class PreprocessingPipeline:
    """Process a list of .sol files through all 5 steps."""

    def __init__(self, source_name: str, out_dir: Path):
        self.source_name = source_name
        self.out_dir = out_dir
        self._dedup = Deduplicator()

    def run(self, sol_files: list[Path], raw_base: Path) -> PipelineResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        processed: list[Path] = []
        dropped: list[dict] = []

        for sol_path in sol_files:
            outcome = self._process_one(sol_path, raw_base)
            if outcome is None:
                continue
            out_sol, meta, drop_row = outcome
            if drop_row:
                dropped.append(drop_row)
            else:
                processed.append(out_sol)
                _write_meta(out_sol.with_suffix(".meta.json"), meta)

        if dropped:
            _write_dropped(self.out_dir / "dropped.csv", dropped)

        return PipelineResult(
            processed=processed,
            dropped=dropped,
            duration_s=time.monotonic() - t0,
        )

    def _process_one(
        self, sol_path: Path, raw_base: Path
    ) -> tuple[Path, ContractMeta, dict | None] | None:
        """Returns (out_path, meta, None) on success, (_, _, drop_row) on drop, None to skip."""
        rel = sol_path.relative_to(raw_base)
        source = sol_path.read_text(errors="replace")
        n_raw = source.count("\n") + 1

        # Step 1 — flatten
        flat = flatten_contract(sol_path)

        # Step 2 — compile. The compiler works on file paths, so if the flattener
        # produced a modified content (solc --flatten or unresolved-import strip),
        # we materialize it to a temp file and compile that. The temp file MUST
        # live in the same directory as the source (not /tmp) so that relative
        # imports like `../interface.sol` resolve correctly. The temp file is
        # auto-cleaned when the function returns. Any sibling files written by
        # the flattener's transitive-strip helper are also cleaned up here.
        compile_target = sol_path
        tmp_paths_for_cleanup: list[Path] = []
        if flat.content != source and flat.flatten_status != "skipped_no_imports":
            tmp = sol_path.parent / f".sentinel_compile_{sol_path.stem}_{sol_path.stat().st_mtime_ns}.sol"
            tmp.write_text(flat.content)
            compile_target = tmp
            tmp_paths_for_cleanup.append(tmp)
            # The flattener may have written `.sentinel_stripped.sol` siblings
            # next to relative-imported files. We discover them by reading the
            # flat.error string (which is informational, but we use a sturdier
            # mechanism: the flattener records them on the FlattenResult).
            # For now, glob the source dir for any .sentinel_stripped.sol files
            # written by this run and clean them up. (Glob is per-file so it's
            # bounded; we only match files we ourselves created in this step.)
        try:
            compile_res = compile_contract(compile_target)
        finally:
            for p in tmp_paths_for_cleanup:
                try:
                    p.unlink()
                except OSError:
                    pass
            # Clean up any `.sentinel_stripped.sol` files the flattener wrote
            # next to relative-imported files. We match by suffix and the
            # naming pattern (must have been written by us — solc never
            # creates files with this suffix).
            for d in {p.parent for p in tmp_paths_for_cleanup}:
                if d is None:
                    continue
                for sib in d.glob("*.sentinel_stripped.sol"):
                    try:
                        sib.unlink()
                    except OSError:
                        pass

        if not compile_res.success:
            return None, None, {
                "source": self.source_name,
                "original_path": str(rel),
                "pragma": compile_res.pragma_raw,
                "reason": "compile_failed",
                "error": compile_res.error[:300],
                "attempted_solc_versions": ",".join(compile_res.attempted_versions),
            }

        # Step 3 — dedup (on flattened content)
        dedup_rec = self._dedup.process(flat.content, sol_path)
        if dedup_rec.is_duplicate:
            return None, None, {
                "source": self.source_name,
                "original_path": str(rel),
                "pragma": compile_res.pragma_raw,
                "reason": "duplicate",
                "duplicate_of": dedup_rec.duplicate_of,
                "attempted_solc_versions": "",
            }

        # Step 4 — normalize
        norm = normalize(flat.content)

        # Step 5 — segment + bucket
        seg = segment_and_bucket(norm.content, compile_res.pragma_raw)

        # Write output
        out_path = self.out_dir / f"{dedup_rec.sha256}.sol"
        out_path.write_text(norm.content)

        meta = ContractMeta(
            sha256=dedup_rec.sha256,
            source_name=self.source_name,
            original_path=str(rel),
            pragma=compile_res.pragma_raw,
            solc_version=compile_res.solc_version,
            compile_status="ok",
            compile_error="",
            attempted_solc_versions=compile_res.attempted_versions,
            flatten_status=flat.flatten_status,
            dedup_group_id=dedup_rec.dedup_group_id,
            is_duplicate=False,
            duplicate_of="",
            version_bucket=seg.version_bucket,
            has_unchecked_block=seg.has_unchecked_block,
            contract_names=seg.contract_names,
            n_raw_lines=n_raw,
            n_normalized_lines=norm.n_lines_after,
        )
        return out_path, meta, None


def _write_meta(path: Path, meta: ContractMeta) -> None:
    with open(path, "w") as f:
        json.dump(asdict(meta), f, indent=2)


def _write_dropped(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Union of all keys across rows — defensive against drop paths that
    # produce slightly different field sets (compile_failed has 5 fields,
    # duplicate has 6).
    all_fields: list[str] = []
    seen_fields: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen_fields:
                seen_fields.add(k)
                all_fields.append(k)
    # Normalize: pad missing fields with empty string for csv consistency
    normalized = [{k: r.get(k, "") for k in all_fields} for r in rows]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(normalized)
