"""DIVE published observations: research evidence, never training truth."""
from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict

TOOLS = ("MAIAN", "Mythril", "Semgrep", "Slither", "Solhint", "VeriSmart")
CATEGORIES = ("Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values",
              "DoS", "Bad Randomness", "Front Running", "Time manipulation")
COLUMNS = tuple(f"{tool}_{i}" for tool in TOOLS for i in range(1, 9))
SCHEMA = "s4-dive-observations-v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_table(data: bytes, fields: tuple[str, ...], *, binary_only: bool):
    """Strict literal CSV parser. Ordinal excludes the header."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != ["contractID", *fields]:
        raise ValueError("unexpected CSV header")
    for ordinal, row in enumerate(reader, 1):
        cid = row["contractID"]
        if not cid or not cid.isascii() or not cid.isdecimal() or str(int(cid)) != cid or int(cid) < 1:
            raise ValueError(f"invalid native ID at row {ordinal}")
        if set(row) != {"contractID", *fields}:
            raise ValueError(f"invalid field count at row {ordinal}")
        domain = {"0", "1"} if binary_only else {"", "0", "1"}
        if any(row[k] not in domain for k in fields):
            raise ValueError(f"invalid observation at row {ordinal}")
        yield ordinal, cid, {k: None if row[k] == "" else int(row[k]) for k in fields}


def observation_rows(data: bytes):
    h = digest(data)
    for ordinal, cid, values in parse_table(data, COLUMNS, binary_only=False):
        yield {"row_id": digest(f"{h}:{ordinal}".encode()), "row_ordinal": ordinal,
               "native_id": cid, **values}


def published_labels(data: bytes):
    result = {}
    for _, cid, values in parse_table(data, CATEGORIES, binary_only=True):
        if cid in result:
            raise ValueError("duplicate published label ID")
        result[cid] = values
    return result


def tool_state(values):
    seen = {v for v in values if v is not None}
    if not seen:
        return "MISSING"
    if seen == {0}:
        return "OBSERVED_ZERO"
    if seen == {1}:
        return "OBSERVED_ONE"
    if seen == {0, 1}:
        return "CONFLICTING"
    raise ValueError("invalid observation domain")


def category_view(rows, labels):
    by = defaultdict(list)
    for row in rows:
        by[row["native_id"]].append(row)
    if set(by) != set(labels):
        raise ValueError("tool and published label ID sets differ")
    for cid in sorted(by, key=int):
        rs = sorted(by[cid], key=lambda r: r["row_ordinal"])
        for i, category in enumerate(CATEGORIES, 1):
            states = {t: tool_state(r[f"{t}_{i}"] for r in rs) for t in TOOLS}
            first = {t: tool_state([rs[0][f"{t}_{i}"]]) for t in TOOLS}
            positive = sum(v in ("OBSERVED_ONE", "CONFLICTING") for v in states.values())
            yield {"native_id": cid, "native_category": category, "native_rank": i,
                   "published_annotation": labels[cid][category],
                   "observation_row_ids": [r["row_id"] for r in rs],
                   "first_row_positive_tools": sum(v == "OBSERVED_ONE" for v in first.values()),
                   "positive_observation_tools": positive,
                   "zero_only_tools": sum(v == "OBSERVED_ZERO" for v in states.values()),
                   "missing_tools": sum(v == "MISSING" for v in states.values()),
                   "conflicting_tools": sum(v == "CONFLICTING" for v in states.values()),
                   **{f"{t}_state": states[t] for t in TOOLS}}


def assert_optimizer_reservations(proposed_group_ids, reserved_group_ids):
    """Explicit guard for a prospective optimizer population; not wired into V3."""
    overlap = set(proposed_group_ids) & set(reserved_group_ids)
    if overlap:
        raise ValueError(f"optimizer reservation overlap: {len(overlap)} groups")


def pick_groups(pools, quotas, reserved, salt):
    used = set(reserved)
    selected = []
    for stratum, limit in quotas:
        ordered = sorted(pools.get(stratum, []), key=lambda x: (
            digest(f"{salt}|{stratum}|{x['group_id']}|{x['contract_id']}|{x['native_id']}".encode()),
            x["group_id"], x["contract_id"], x["native_id"]))
        count = 0
        for row in ordered:
            if count >= limit:
                break
            if row["group_id"] in used:
                continue
            used.add(row["group_id"])
            selected.append({**row, "stratum": stratum})
            count += 1
    return selected
