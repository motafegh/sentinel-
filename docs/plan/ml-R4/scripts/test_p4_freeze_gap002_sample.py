#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("p4_freeze_gap002_sample.py")
spec = importlib.util.spec_from_file_location("p4_freezer", SCRIPT)
assert spec and spec.loader
p4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4)


def row(
    contract_id: str,
    class_name: str,
    class_index: int,
    *,
    split: str = "train",
    project: str | None = None,
    dedup: str | None = None,
    target: int = 1,
) -> dict:
    return {
        "ledger_version": p4.LEDGER_VERSION,
        "contract_id": contract_id,
        "class_index": class_index,
        "class_name": class_name,
        "primary_source": "dive",
        "historical_state": "HISTORICAL_POSITIVE" if target else "HISTORICAL_ZERO",
        "historical_target": target,
        "historical_split": split,
        "representation_available": True,
        "dedup_group_id": dedup,
        "project_group_id": project,
    }


def full_population(n: int = 8) -> list[dict]:
    rows: list[dict] = []
    for class_name, class_index, _native in p4.TARGETS:
        for i in range(n):
            rows.append(
                row(
                    f"{class_index:02d}{i:062d}",
                    class_name,
                    class_index,
                    project=f"{class_name}-project-{i}",
                )
            )
    return rows


class Gap002SampleTests(unittest.TestCase):
    LEDGER_SHA = p4.EXPECTED_LEDGER_SHA256

    def test_deterministic_and_balanced(self) -> None:
        rows = full_population(8)
        manifest_a, sample_a = p4.build_population_and_sample(
            rows, ledger_sha=self.LEDGER_SHA, per_stratum=4
        )
        manifest_b, sample_b = p4.build_population_and_sample(
            list(reversed(rows)), ledger_sha=self.LEDGER_SHA, per_stratum=4
        )
        self.assertEqual(sample_a, sample_b)
        self.assertEqual(manifest_a["sample_sha256"], manifest_b["sample_sha256"])
        self.assertEqual(len(sample_a), 20)
        counts = {name: 0 for name, _, _ in p4.TARGETS}
        for item in sample_a:
            counts[item["class_name"]] += 1
            self.assertEqual(item["historical_split"], "train")
        self.assertTrue(all(value == 4 for value in counts.values()))
        self.assertEqual(len({item["review_group_id"] for item in sample_a}), 20)

    def test_group_touching_val_is_excluded(self) -> None:
        rows = full_population(8)
        # Add a DoS-positive training contract in a project that also has a val
        # occurrence. The whole group must be unavailable to Phase-4 review.
        rows.append(
            row(
                "a" * 64,
                "DenialOfService",
                1,
                project="cross-split-project",
            )
        )
        rows.append(
            row(
                "b" * 64,
                "ExternalBug",
                2,
                split="val",
                project="cross-split-project",
            )
        )
        manifest, sample = p4.build_population_and_sample(
            rows, ledger_sha=self.LEDGER_SHA, per_stratum=4
        )
        self.assertNotIn("project:cross-split-project", {x["review_group_id"] for x in sample})
        self.assertGreaterEqual(
            manifest["strata"]["DenialOfService"]["excluded_groups_touching_val_or_test"], 1
        )

    def test_project_group_precedes_dedup_and_contract(self) -> None:
        item = row(
            "c" * 64,
            "IntegerUO",
            4,
            project="project-A",
            dedup="dedup-B",
        )
        self.assertEqual(p4.canonical_group_key(item), "project:project-A")
        item["project_group_id"] = None
        self.assertEqual(p4.canonical_group_key(item), "dedup:dedup-B")
        item["dedup_group_id"] = None
        self.assertEqual(p4.canonical_group_key(item), "contract:" + "c" * 64)

    def test_insufficient_disjoint_groups_fails_closed(self) -> None:
        rows: list[dict] = []
        # Every stratum has only one group. Asking for two must fail rather than
        # silently reusing a project/contract group.
        for class_name, class_index, _native in p4.TARGETS:
            rows.append(
                row(
                    f"{class_index:02d}" + "d" * 62,
                    class_name,
                    class_index,
                    project=f"only-{class_name}",
                )
            )
        with self.assertRaises(RuntimeError):
            p4.build_population_and_sample(rows, ledger_sha=self.LEDGER_SHA, per_stratum=2)

    def test_locked_class_index_mismatch_is_rejected(self) -> None:
        rows = full_population(3)
        rows[0]["class_index"] = 9
        with self.assertRaises(ValueError):
            p4.build_population_and_sample(rows, ledger_sha=self.LEDGER_SHA, per_stratum=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
