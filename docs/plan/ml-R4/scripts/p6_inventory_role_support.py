#!/usr/bin/env python3
"""Inventory Phase-6 role support from the committed Phase-3 ledger.

Read-only with respect to historical/protected data.  This applies the accepted
Phase-5 source/class policy conservatively and reports group-level eligibility;
it does not yet assign final roles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_LEDGER_SHA = "3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7"
EXPECTED_CONTRACTS = 22493
EXPECTED_ROWS = 224930
EXPECTED_CLASSES = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc
    return pq


def group_id(row: dict[str, Any]) -> str:
    project = row.get("project_group_id")
    dedup = row.get("dedup_group_id")
    contract = str(row["contract_id"])
    if project not in (None, ""):
        return f"project:{project}"
    if dedup not in (None, ""):
        return f"dedup:{dedup}"
    return f"contract:{contract}"


def signal_for(row: dict[str, Any], policy: dict[str, Any]) -> tuple[str, str | None]:
    """Return (STRONG|WEAK|NONE, class_name-or-None)."""
    if int(row["historical_target"]) != 1:
        return "NONE", None

    source = str(row["primary_source"])
    class_name = str(row["class_name"])
    class_cfg = policy["class_supervision"][class_name]
    if class_cfg["status"] != "ENABLED":
        return "NONE", None

    if source == "solidifi":
        # Phase-2 reconstruction established SolidiFI's single historical
        # positive as the injected class. Non-target zeros never enter here.
        return "STRONG", class_name

    if source == "smartbugs_curated":
        # Phase-3 rows lost the source-native SmartBugs category. Timestamp is
        # therefore ambiguous between direct time_manipulation and historical
        # bad_randomness->Timestamp and must fail closed to unlabeled.
        if class_name == "Timestamp":
            return "NONE", None
        approved = set(policy["sources"]["smartbugs_curated"]["approved_mappings"].values())
        if class_name in approved:
            return "STRONG", class_name
        return "NONE", None

    if source == "dive":
        if class_name == "TransactionOrderDependence":
            cfg = policy["sources"]["dive"]["mapped_category_policy"]["Front Running"]
            if cfg["training_strength"] == "WEAK" and cfg["target_value"] == 1:
                return "WEAK", class_name
        return "NONE", None

    return "NONE", None


def load_rows(path: Path, pq) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"ledger rows {len(rows)} != {EXPECTED_ROWS}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=Path("docs/plan/ml-R4/ledger/evidence_ledger_v1.parquet"))
    ap.add_argument("--policy", type=Path, default=Path("docs/plan/ml-R4/specs/data_vnext_policy_v1.json"))
    ap.add_argument("--output", type=Path, default=Path("docs/plan/ml-R4/manifests/p6_role_support_inventory.json"))
    ap.add_argument("--groups-output", type=Path, default=Path("docs/plan/ml-R4/manifests/p6_group_eligibility_inventory.jsonl"))
    args = ap.parse_args()

    pq = require_pyarrow()
    if sha256_file(args.ledger) != EXPECTED_LEDGER_SHA:
        raise RuntimeError("Phase-3 ledger SHA mismatch")
    policy = json.loads(args.policy.read_text())
    if policy["status"] != "ACCEPTED_G5":
        raise RuntimeError("Phase-5 policy is not ACCEPTED_G5")
    if policy["class_vocabulary"]["classes"] != EXPECTED_CLASSES:
        raise RuntimeError("class vocabulary/order mismatch")

    rows = load_rows(args.ledger, pq)
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_contract[str(row["contract_id"])].append(row)
    if len(by_contract) != EXPECTED_CONTRACTS:
        raise RuntimeError(f"contracts {len(by_contract)} != {EXPECTED_CONTRACTS}")
    for cid, cr in by_contract.items():
        if len(cr) != 10 or [r["class_name"] for r in sorted(cr, key=lambda r: int(r["class_index"]))] != EXPECTED_CLASSES:
            raise RuntimeError(f"invalid ten-class ledger shape for {cid}")

    groups: dict[str, dict[str, Any]] = {}
    strong_row_counts = Counter()
    weak_row_counts = Counter()
    masked_positive_counts = Counter()
    source_contracts = Counter()
    ambiguous_smartbugs_timestamp_contracts: set[str] = set()

    for cid, cr in by_contract.items():
        first = cr[0]
        gid = group_id(first)
        source = str(first["primary_source"])
        source_contracts[source] += 1
        if any(group_id(r) != gid for r in cr):
            raise RuntimeError(f"group id differs across class rows for {cid}")
        g = groups.setdefault(gid, {
            "group_id": gid,
            "contracts": set(),
            "sources": set(),
            "historical_splits": set(),
            "strong_classes": set(),
            "weak_classes": set(),
            "represented_contracts": 0,
        })
        g["contracts"].add(cid)
        g["sources"].add(source)
        split = first.get("historical_split")
        if split:
            g["historical_splits"].add(str(split))
        g["represented_contracts"] += int(bool(first.get("representation_available")))

        for row in cr:
            strength, cls = signal_for(row, policy)
            if strength == "STRONG":
                g["strong_classes"].add(cls)
                strong_row_counts[cls] += 1
            elif strength == "WEAK":
                g["weak_classes"].add(cls)
                weak_row_counts[cls] += 1
            elif int(row["historical_target"]) == 1:
                masked_positive_counts[str(row["class_name"])] += 1
                if source == "smartbugs_curated" and row["class_name"] == "Timestamp":
                    ambiguous_smartbugs_timestamp_contracts.add(cid)

    group_classification_counts = Counter()
    strong_group_counts_by_class = Counter()
    weak_group_counts_by_class = Counter()
    groups_touching_multiple_historical_splits = 0
    groups_with_multiple_sources = 0
    inventory_rows = []

    for gid in sorted(groups):
        g = groups[gid]
        if g["strong_classes"]:
            classification = "STRONG_ELIGIBLE_GROUP"
        elif g["weak_classes"]:
            classification = "WEAK_ELIGIBLE_GROUP"
        else:
            classification = "UNLABELED_GROUP"
        group_classification_counts[classification] += 1
        if len(g["historical_splits"]) > 1:
            groups_touching_multiple_historical_splits += 1
        if len(g["sources"]) > 1:
            groups_with_multiple_sources += 1
        for cls in g["strong_classes"]:
            strong_group_counts_by_class[cls] += 1
        for cls in g["weak_classes"]:
            weak_group_counts_by_class[cls] += 1
        inventory_rows.append({
            "group_id": gid,
            "classification": classification,
            "contract_ids": sorted(g["contracts"]),
            "contract_count": len(g["contracts"]),
            "sources": sorted(g["sources"]),
            "historical_splits": sorted(g["historical_splits"]),
            "strong_classes": sorted(g["strong_classes"], key=EXPECTED_CLASSES.index),
            "weak_classes": sorted(g["weak_classes"], key=EXPECTED_CLASSES.index),
            "represented_contracts": g["represented_contracts"],
        })

    args.groups_output.parent.mkdir(parents=True, exist_ok=True)
    args.groups_output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in inventory_rows))

    report = {
        "schema": "r4-phase6-role-support-inventory-v1",
        "ledger_sha256": EXPECTED_LEDGER_SHA,
        "policy_version": policy["policy_version"],
        "policy_sha256": sha256_file(args.policy),
        "contracts": len(by_contract),
        "ledger_rows": len(rows),
        "groups": len(groups),
        "source_contract_counts": dict(sorted(source_contracts.items())),
        "group_classification_counts": dict(sorted(group_classification_counts.items())),
        "strong_positive_row_counts_by_class": {c: strong_row_counts[c] for c in EXPECTED_CLASSES},
        "weak_positive_row_counts_by_class": {c: weak_row_counts[c] for c in EXPECTED_CLASSES},
        "masked_historical_positive_rows_by_class": {c: masked_positive_counts[c] for c in EXPECTED_CLASSES},
        "strong_group_counts_by_class": {c: strong_group_counts_by_class[c] for c in EXPECTED_CLASSES},
        "weak_group_counts_by_class": {c: weak_group_counts_by_class[c] for c in EXPECTED_CLASSES},
        "ambiguous_smartbugs_timestamp_contracts": len(ambiguous_smartbugs_timestamp_contracts),
        "groups_touching_multiple_historical_splits": groups_touching_multiple_historical_splits,
        "groups_with_multiple_sources": groups_with_multiple_sources,
        "role_support": {
            "TRAIN_STRONG": "SUPPORTED",
            "TRAIN_WEAK": "SUPPORTED_IF_WEAK_GROUPS_GT_0",
            "TRAIN_UNLABELED": "SUPPORTED",
            "MODEL_SELECTION": "LIMITED_POSITIVE_ONLY_STRONG_GROUPS",
            "INTERNAL_AUDIT": "SUPPORTED_EXPOSED_OR_STRONG_HOLDOUT",
            "THRESHOLD_FIT": "UNSUPPORTED_NO_CONFIRMED_NEGATIVE_SUPPORT",
            "CALIBRATION_FIT": "UNSUPPORTED_NO_CONFIRMED_NEGATIVE_SUPPORT",
            "UNTOUCHED_ACCEPTANCE": "UNSUPPORTED_NO_TRUSTWORTHY_UNEXPOSED_LABELED_CORPUS"
        },
        "limitations": [
            "SmartBugs Timestamp positives are withheld because source-native time_manipulation vs bad_randomness identity is unavailable in the committed ledger.",
            "No historical zero is treated as a negative.",
            "This inventory establishes role eligibility only; final deterministic group roles are assigned in a separate Phase-6 freeze step."
        ],
        "group_inventory_sha256": sha256_file(args.groups_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
