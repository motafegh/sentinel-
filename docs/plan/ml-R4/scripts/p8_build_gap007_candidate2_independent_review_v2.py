#!/usr/bin/env python3
"""Build the self-contained R4-GAP-007 candidate #2 blind-review v2 bundle.

The v1 archive is used only as a hash-bound carrier for the complete Solidity
source. Its old manifest/task are deliberately not copied. The v2 archive
injects the neutral R4-D-016 class definition and generic negative-evidence
policy so a reviewer never needs to inspect the wider Sentinel repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo


REPO_ROOT = Path(__file__).resolve().parents[4]
BUNDLES = REPO_ROOT / "docs/plan/ml-R4/review_bundles"
V1 = BUNDLES / "r4_gap007_candidate2_independent_review_v1.zip"
V2 = BUNDLES / "r4_gap007_candidate2_independent_review_v2.zip"
V2_SHA = BUNDLES / "r4_gap007_candidate2_independent_review_v2.zip.sha256"

V1_EXPECTED_SHA256 = "2e7f48c9648097624406d167266a42a31055f222a0f468a0453b2f353b343f1a"
CANDIDATE_ID = "r4neg-bfe90ef82e33a324d612256a5d4053c6"
CONTRACT_ID = "f7afe9fff9f6c117c6cd9dd4730c0f12e3cc3c8ab98797911de091e240051b93"
SOURCE_MEMBER = f"source/{CONTRACT_ID}.sol"
CLASS_NAME = "CallToUnknown"
CLASS_INDEX = 0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def zip_info(name: str) -> ZipInfo:
    # ZIP_STORED avoids zlib-version-dependent compressed-byte differences.
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.external_attr = 0o644 << 16
    return info


def governing_definition() -> dict:
    return {
        "authority": "R4-D-016",
        "class_index": CLASS_INDEX,
        "class_name": CLASS_NAME,
        "definition": (
            "CallToUnknown is the raw/dynamic low-level unknown-target interaction "
            "class. It covers low-level external-call-family interactions with an "
            "unverified or dynamically selected callee."
        ),
        "included_constructs": [
            "call",
            "delegatecall",
            "legacy callcode",
            "staticcall where applicable",
            "send",
        ],
        "interpretation_rules": [
            (
                "A checked raw low-level call can still satisfy CallToUnknown; "
                "target/callee uncertainty is class-defining."
            ),
            (
                "Unchecked return or failure handling is additional positive "
                "evidence, not the sole defining condition."
            ),
        ],
        "explicit_exclusions_when_destination_uncertainty_is_the_only_reason": [
            "Solidity transfer()",
            "ordinary typed high-level interface calls",
            "same-file or otherwise in-scope library calls",
            "contract creation",
            "selfdestruct/suicide destinations",
        ],
        "scope_note": (
            "An excluded construct may still matter to another vulnerability class; "
            "this definition is class-specific and does not assert global safety."
        ),
    }


def negative_policy() -> dict:
    return {
        "policy": "R4 class-specific confirmed-negative evidence rule",
        "unit_of_truth": "contract x vulnerability_class",
        "verdicts": {
            "CONFIRMED_NEGATIVE": (
                "Complete class-specific review provides affirmative exclusion "
                "evidence sufficient to establish that the class does not apply."
            ),
            "NOT_CONFIRMED": (
                "Credible evidence satisfies or may satisfy the target class, or "
                "specific contradictory evidence prevents class-specific exclusion."
            ),
            "UNKNOWN / insufficient evidence": (
                "Review scope, source completeness, or semantic evidence is "
                "insufficient to establish either exclusion or contradictory class evidence."
            ),
        },
        "requirements": [
            "Inspect the complete supplied source, not only token/search hits.",
            "Identify and assess all constructs plausibly relevant to the target class.",
            "Address credible contradictory positive evidence explicitly.",
            "Preserve uncertainty when source scope or semantics are incomplete.",
        ],
        "prohibitions": [
            "Do not infer a negative from historical labels or historical zero values.",
            "Do not infer a negative from queue membership or unlabeled state.",
            "Do not infer a negative from static-tool silence or absence of findings.",
            "Do not infer global contract safety from a class-specific negative.",
        ],
        "authority_boundary": (
            "An accepted negative is evaluation evidence only. This review does not "
            "authorize a training target, threshold fitting, calibration, M5, or G8."
        ),
    }


def review_task() -> dict:
    return {
        "schema": "sentinel-r4-gap007-independent-review-task-v2",
        "candidate_id": CANDIDATE_ID,
        "contract_id": CONTRACT_ID,
        "target": {"class_index": CLASS_INDEX, "class_name": CLASS_NAME},
        "negative_scope": "CLASS_SPECIFIC_ONLY",
        "allowed_verdicts": [
            "CONFIRMED_NEGATIVE",
            "NOT_CONFIRMED",
            "UNKNOWN / insufficient evidence",
        ],
        "review_boundary": [
            "Use only files contained in this archive for Sentinel project semantics.",
            "Do not inspect CLAUDE.md, repository status files, prior reviews, ADRs, or chat/project memory.",
            "Do not search Sentinel for this candidate or contract.",
            "External Solidity-language documentation may be consulted only to resolve language semantics and must be identified in the rationale.",
        ],
        "blindness_rule": (
            "If the review session already contains any prior candidate-specific "
            "verdict/direction or other candidate-history conclusion, stop and report "
            "PROCEDURALLY_CONTAMINATED without issuing a substantive verdict."
        ),
        "required_output": [
            "evidence inspected",
            "class definition applied",
            "evidence for the class",
            "evidence supporting exclusion",
            "one allowed verdict",
            "confidence and unresolved ambiguity",
            "explicit blindness confirmation",
        ],
        "repository_modification_allowed": False,
    }


def main() -> int:
    if not V1.exists():
        raise FileNotFoundError(f"missing tracked v1 archive: {V1}")

    v1_bytes = V1.read_bytes()
    observed_v1_sha = sha256_bytes(v1_bytes)
    if observed_v1_sha != V1_EXPECTED_SHA256:
        raise RuntimeError(
            "v1 archive SHA-256 mismatch: "
            f"expected {V1_EXPECTED_SHA256}, observed {observed_v1_sha}"
        )

    with ZipFile(V1, "r") as source_zip:
        names = set(source_zip.namelist())
        if SOURCE_MEMBER not in names:
            raise RuntimeError(
                f"v1 archive missing expected complete source member {SOURCE_MEMBER!r}; "
                f"members={sorted(names)}"
            )
        source = source_zip.read(SOURCE_MEMBER)

    observed_source_sha = sha256_bytes(source)
    if observed_source_sha != CONTRACT_ID:
        raise RuntimeError(
            "source SHA-256 mismatch: "
            f"expected {CONTRACT_ID}, observed {observed_source_sha}"
        )

    payloads: dict[str, bytes] = {
        SOURCE_MEMBER: source,
        "governing_material/call_to_unknown_definition.json": json_bytes(
            governing_definition()
        ),
        "governing_material/confirmed_negative_policy.json": json_bytes(
            negative_policy()
        ),
        "review_task.json": json_bytes(review_task()),
    }

    manifest = {
        "schema": "sentinel-r4-gap007-independent-review-bundle-v2",
        "candidate_id": CANDIDATE_ID,
        "contract_id": CONTRACT_ID,
        "target": {"class_index": CLASS_INDEX, "class_name": CLASS_NAME},
        "semantic_authority": "R4-D-016",
        "source_carrier_v1_sha256": observed_v1_sha,
        "source_sha256": observed_source_sha,
        "files": {
            name: {"sha256": sha256_bytes(data), "bytes": len(data)}
            for name, data in sorted(payloads.items())
        },
        "privacy_and_blinding": {
            "contains_primary_review": False,
            "contains_prior_candidate_verdict": False,
            "requires_repository_access": False,
        },
    }
    payloads["bundle_manifest.json"] = json_bytes(manifest)

    V2.parent.mkdir(parents=True, exist_ok=True)
    tmp = V2.with_suffix(V2.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with ZipFile(tmp, "w") as output_zip:
        for name in sorted(payloads):
            output_zip.writestr(zip_info(name), payloads[name])

    tmp.replace(V2)
    archive_sha = sha256_bytes(V2.read_bytes())
    V2_SHA.write_text(f"{archive_sha}  {V2.name}\n", encoding="utf-8")

    # Read-back verification: exact members and every manifest-bound hash.
    with ZipFile(V2, "r") as verify_zip:
        observed_members = sorted(verify_zip.namelist())
        expected_members = sorted(payloads)
        if observed_members != expected_members:
            raise RuntimeError(
                f"v2 member mismatch: expected={expected_members}, observed={observed_members}"
            )
        for name, expected in manifest["files"].items():
            data = verify_zip.read(name)
            observed = sha256_bytes(data)
            if observed != expected["sha256"]:
                raise RuntimeError(
                    f"v2 payload hash mismatch for {name}: {observed} != {expected['sha256']}"
                )
        if json.loads(verify_zip.read("bundle_manifest.json")) != manifest:
            raise RuntimeError("v2 manifest read-back mismatch")

    print(f"PASS: {V2}")
    print(f"v1_sha256={observed_v1_sha}")
    print(f"source_sha256={observed_source_sha}")
    print(f"sha256={archive_sha}")
    print("members:")
    for name in sorted(payloads):
        print(f"  {name}")
    print("candidate_state_unchanged=UNKNOWN/PENDING_REVIEW/target=None")
    print("full_training_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
