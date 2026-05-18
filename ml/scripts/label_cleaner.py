"""
label_cleaner.py — Structural precondition filter for BCCC OR-label noise.

BCCC uses folder-level OR-labeling: every contract in a vulnerability folder
receives all labels associated with that folder, regardless of whether the
specific contract exhibits the vulnerability. This produces systematic false
positives. This script applies per-class structural plausibility checks and
zeros out labels that fail their check.

Conservative by design: a check can only REMOVE a positive label (false
positive), never CREATE one. Every change is logged to an audit JSON.

Usage:
    # Dry run — print counts only, no files written
    python ml/scripts/label_cleaner.py --dry-run

    # Apply cleaning and write output CSV + audit JSON
    python ml/scripts/label_cleaner.py

    # Custom paths
    python ml/scripts/label_cleaner.py \\
        --csv ml/data/processed/multilabel_index_deduped.csv \\
        --graphs-dir ml/data/graphs \\
        --output ml/data/processed/multilabel_index_cleaned.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants matching graph_schema.py (reproduced to avoid circular imports
# when running this script standalone)
# ---------------------------------------------------------------------------
EDGE_CALLS = 0   # CALLS edge type id

CLASS_NAMES = [
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

# ---------------------------------------------------------------------------
# Per-class structural precondition functions
# ---------------------------------------------------------------------------
# Each function receives a torch_geometric Data object and returns True if the
# graph provides structural evidence consistent with the vulnerability label.
# Returns False → label is implausible → demote to 0.
#
# Feature dim reference (graph_schema.py v7 — 11 dims):
#   [0] type_id/12.0   [1] visibility      [2] uses_block_globals
#   [3] view           [4] payable         [5] complexity
#   [6] loc            [7] return_ignored  [8] call_target_typed
#   [9] has_loop       [10] external_call_count
#   NOTE: in_unchecked (was [9] in v6) dropped — BUG-L2, dead for 87.9% of dataset
# ---------------------------------------------------------------------------

def _has_calls_edge(data) -> bool:
    """True if the graph contains at least one CALLS (type 0) edge."""
    if data.edge_index.size(1) == 0:
        return False
    ea = data.edge_attr
    if ea is None:
        return False
    if ea.dim() > 1:
        ea = ea.squeeze(-1)
    return bool((ea == EDGE_CALLS).any())


def _has_external_call(data) -> bool:
    """True if any node has external_call_count > 0 or a CALLS edge exists."""
    return (
        _has_calls_edge(data)
        or bool((data.x[:, 10] > 0.0).any())  # external_call_count is dim[10] in v7
    )


def check_reentrancy(data) -> bool:
    """Reentrancy requires an external call (Transfer/Send/HL/LL call).
    FIX: CALLS edge (type 0) is internal function calls — NOT external calls.
    External calls are captured in external_call_count (dim[10]).
    Old bug: _has_calls_edge() tested internal calls → 72.8% of removed
    Reentrancy contracts had ext_call_count>0 and were incorrectly stripped."""
    return bool((data.x[:, 10] > 0.0).any())  # external_call_count dim[10]


def check_timestamp(data) -> bool:
    """Timestamp requires at least one node reading block globals.
    KNOWN LIMITATION: the `now` alias (SolidityVariable, pre-Solidity 0.5)
    is not captured by _compute_uses_block_globals in the extractor — it only
    checks SolidityVariableComposed. Contracts using only `now` will have
    x[2]=0 and get incorrectly removed. Fix requires re-extraction.
    Scope: BCCC is predominantly pre-0.5, so this may affect a meaningful
    fraction of the 423 removed Timestamp contracts."""
    return bool((data.x[:, 2] > 0.5).any())


def check_mishandled_exception(data) -> bool:
    """MishandledException requires an ignored return value or an external call
    to an untyped target. Inherits partial benefit from the CallToUnknown fix:
    the OR on external_call_count catches Transfer/Send-only contracts whose
    call_target_typed is incorrectly 1.0 due to extractor gap."""
    return bool(
        (data.x[:, 7] > 0.5).any()      # return_ignored
        or (data.x[:, 8] == 0.0).any()  # call_target_typed=0 → raw address call
        or (data.x[:, 10] > 0.0).any()  # Transfer/Send gap: ext call exists but typed=1 incorrectly
    )


def check_call_to_unknown(data) -> bool:
    """CallToUnknown requires a call to an address whose type is unknown.
    FIX: call_target_typed (dim[8]) misses Transfer and Send — the extractor's
    _compute_call_target_typed only scans func.low_level_calls and
    func.high_level_calls; Transfer/Send are excluded, so contracts using only
    transfer() get call_target_typed=1.0 despite calling an unknown address.
    Fix: accept any contract with external_call_count>0 as a necessary condition
    (all CTU contracts must have ≥1 external call). 82.6% of incorrectly removed
    CTU contracts had external_call_count>0; this restores them.
    Phase-2 fix: add Transfer/Send detection in _compute_call_target_typed."""
    return bool(
        (data.x[:, 8] == 0.0).any()    # untyped call (raw address)
        or (data.x[:, 10] > 0.0).any() # Transfer/Send gap — has external call
    )


def check_unused_return(data) -> bool:
    """UnusedReturn requires at least one ignored external call return value.
    VALID: return_ignored (dim[7]) is reliably computed. No fix needed."""
    return bool((data.x[:, 7] > 0.5).any())


# Mapping: CSV column name → precondition function.
# Classes not listed here have no reliable structural precondition
# (GasException, ExternalBug, TOD, DoS) and are left as-is.
#
# REMOVED: IntegerUO — the has_loop heuristic is wrong. Integer overflow in
# Solidity <0.8 requires only arithmetic operations, not a loop. Removing
# IntegerUO from PRECONDITIONS restores ~9,897 incorrectly stripped labels.
# There is no reliable IntegerUO precondition derivable from the v7 feature
# vector after in_unchecked was dropped (BUG-L2).
PRECONDITIONS: dict[str, callable] = {
    "Reentrancy":          check_reentrancy,
    "Timestamp":           check_timestamp,
    "MishandledException": check_mishandled_exception,
    "CallToUnknown":       check_call_to_unknown,
    "UnusedReturn":        check_unused_return,
}


# ---------------------------------------------------------------------------
# Main cleaning logic
# ---------------------------------------------------------------------------

def clean_labels(
    csv_path: Path,
    graphs_dir: Path,
    output_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Apply structural precondition checks to every row in csv_path.

    Returns a summary dict with per-class removal counts.
    """
    rows_in: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows_in = list(reader)

    if not fieldnames or "md5_stem" not in fieldnames:
        print(f"ERROR: CSV must have 'md5_stem' column. Got: {fieldnames}", file=sys.stderr)
        sys.exit(1)

    changes: list[dict] = []
    removal_counts: dict[str, int] = defaultdict(int)
    missing_graphs = 0
    rows_out: list[dict] = []

    for row in tqdm(rows_in, desc="Cleaning labels", unit="contract"):
        md5 = row["md5_stem"]
        pt_path = graphs_dir / f"{md5}.pt"

        if not pt_path.exists():
            missing_graphs += 1
            rows_out.append(row)
            continue

        try:
            data = torch.load(pt_path, weights_only=False)
        except Exception as e:
            print(f"WARN: Could not load {pt_path}: {e}", file=sys.stderr)
            rows_out.append(row)
            continue

        row_modified = dict(row)
        for class_name, check_fn in PRECONDITIONS.items():
            if class_name not in row_modified:
                continue
            if str(row_modified[class_name]).strip() != "1":
                continue
            if not check_fn(data):
                row_modified[class_name] = "0"
                removal_counts[class_name] += 1
                changes.append({
                    "hash":   md5,
                    "class":  class_name,
                    "old":    1,
                    "new":    0,
                    "reason": f"Failed {check_fn.__name__}",
                })

        rows_out.append(row_modified)

    # Report
    total_removed = sum(removal_counts.values())
    print(f"\n{'DRY RUN — ' if dry_run else ''}Label cleaning summary")
    print(f"  Input rows     : {len(rows_in):,}")
    print(f"  Missing graphs : {missing_graphs:,}")
    print(f"  Labels removed : {total_removed:,}")
    print()
    for cls in CLASS_NAMES:
        n = removal_counts.get(cls, 0)
        if n > 0:
            print(f"    {cls:<30} -{n:,}")

    if dry_run:
        print("\nDry run complete — no files written.")
        return {"total_removed": total_removed, "per_class": dict(removal_counts)}

    # Write cleaned CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nCleaned CSV written: {output_path}")

    # Write audit JSON
    audit_path = output_path.with_suffix(".audit.json")
    with open(audit_path, "w") as f:
        json.dump(changes, f, indent=2)
    print(f"Audit log written : {audit_path}  ({len(changes):,} changes)")

    return {"total_removed": total_removed, "per_class": dict(removal_counts)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structural precondition label cleaner for BCCC OR-label noise."
    )
    parser.add_argument(
        "--csv",
        default="ml/data/processed/multilabel_index_deduped.csv",
        help="Input multilabel CSV (default: ml/data/processed/multilabel_index_deduped.csv)",
    )
    parser.add_argument(
        "--graphs-dir",
        default="ml/data/graphs",
        help="Directory containing graph .pt files (default: ml/data/graphs)",
    )
    parser.add_argument(
        "--output",
        default="ml/data/processed/multilabel_index_cleaned.csv",
        help="Output cleaned CSV path (default: ml/data/processed/multilabel_index_cleaned.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print removal counts without writing any files.",
    )
    args = parser.parse_args()

    clean_labels(
        csv_path=Path(args.csv),
        graphs_dir=Path(args.graphs_dir),
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
