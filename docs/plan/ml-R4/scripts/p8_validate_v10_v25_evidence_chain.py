#!/usr/bin/env python3
"""Validate the protected V10 V2.5 evidence chain before full transition audit.

The bounded V2.5 reproducibility report must be cryptographically bound to both
(1) the original V2 transition audit that produced the 20 unexpected identities
and (2) the exact merged semantic WRITE evidence consumed by the bounded probe.
This is a preflight only; it grants neither physical acceptance nor training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BOUNDED_SCHEMA = "sentinel-r4-v10-v25-reproducibility-probe-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_chain(
    *,
    source_audit: Path,
    bounded_report: Path,
    semantic_evidence: Path,
) -> dict[str, object]:
    bounded = json.loads(bounded_report.read_text(encoding="utf-8"))
    if bounded.get("schema") != BOUNDED_SCHEMA:
        raise ValueError("unexpected bounded V2.5 schema")
    if bounded.get("bounded_v25_reproducibility_passed") is not True:
        raise ValueError("bounded V2.5 reproducibility did not pass")
    if bounded.get("zero_unexplained_drift") is not True:
        raise ValueError("bounded V2.5 report contains unexplained drift")
    if list(bounded.get("blocking_identities") or []):
        raise ValueError("bounded V2.5 report contains blocking identities")

    source_sha = _sha256(source_audit)
    semantic_sha = _sha256(semantic_evidence)
    if bounded.get("source_audit_sha256") != source_sha:
        raise ValueError("bounded report source-audit SHA does not match")
    if bounded.get("semantic_evidence_sha256") != semantic_sha:
        raise ValueError("bounded report semantic-evidence SHA does not match")

    return {
        "schema": "sentinel-r4-v10-v25-evidence-chain-preflight-v1",
        "passed": True,
        "source_audit_sha256": source_sha,
        "bounded_report_sha256": _sha256(bounded_report),
        "semantic_evidence_sha256": semantic_sha,
        "unexpected_identities": bounded.get("unexpected_identities"),
        "index_equivalence_identities": bounded.get("index_equivalence_identities"),
        "semantic_correction_identities": bounded.get("semantic_correction_identities"),
        "repeat_generations": bounded.get("repeat_generations"),
        "physical_acceptance": False,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--bounded-report", type=Path, required=True)
    parser.add_argument("--semantic-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_chain(
        source_audit=args.source_audit,
        bounded_report=args.bounded_report,
        semantic_evidence=args.semantic_evidence,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
