"""Label-aware folderization for sources that distribute labels in a separate file.

Some DAPP vulnerability datasets (DIVE, SmartBugs Wild, Bastet) distribute
contracts and labels in separate files rather than via folder structure.
For these, we want to materialize per-class folder symlinks in
`data/raw/<source>/repo/<class>/` so that downstream tools that assume
folder structure (Stage 3 crosswalk YAMLs, the split-stage stratification)
have a uniform interface.

Multi-label handling: a contract with 3 positive labels appears in 3 folders.

LAYOUT (revised 2026-06-10 to keep root clean):
  data/raw/<source>/repo/__source__/<id>.sol     ← canonical (real) files
  data/raw/<source>/repo/<Class1>/<id>.sol       → ../../__source__/<id>.sol
  data/raw/<source>/repo/<Class2>/<id>.sol       → ../../__source__/<id>.sol
  data/raw/<source>/repo/<Class3>/<id>.sol       → ../../__source__/<id>.sol

The root `data/raw/<source>/repo/` contains only the `__source__/` canonical
dir and the per-class subdirs — no flat .sol files. This makes the
"include_subdirs" config clean: pipelines can either scan all (root +
subdirs) or just one class (e.g. for class-balanced sampling).

When the connector materializes a source, the real .sol files start at
`repo/<id>.sol` (flat, no `__source__/`). The folderize function **moves**
them into `repo/__source__/` first, then creates the per-class symlinks.
This way, both flat-source datasets (DIVE) and already-folderized datasets
(SolidiFI, with `buggy_contracts/<bug_type>/`) work with the same logic.

For folderized sources like SolidiFI where the canonical location is
`repo/buggy_contracts/`, set `source_subdir="buggy_contracts"` and the
symlinks will point to `../../buggy_contracts/<id>.sol` (which is the
real file). For flat sources like DIVE, set `source_subdir="__source__"`
and the move happens first, then symlinks point to `../../__source__/<id>.sol`.

The folderization is **idempotent** — running it twice produces the same
result. It's also **safe to re-run after a re-ingest** since symlinks are
recreated cleanly.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FolderizationResult:
    """Summary of a folderization run.

    Attributes:
        contracts_seen:    number of labeled contracts processed
        symlinks_created:  number of <class>/<id>.sol symlinks created
        classes_present:   sorted list of classes with at least one contract
        multi_label:       number of contracts with >1 positive class
        unparseable_labels: number of rows in the label CSV we couldn't read
        files_moved:       number of .sol files moved into __source__/ (flat-source case)
    """
    contracts_seen: int = 0
    symlinks_created: int = 0
    classes_present: list[str] = None
    multi_label: int = 0
    unparseable_labels: int = 0
    files_moved: int = 0

    def __post_init__(self):
        """Default classes_present to empty list if not provided."""
        if self.classes_present is None:
            self.classes_present = []


def folderize_by_labels(
    repo_dir: Path,
    labels_csv: Path,
    id_column: str,
    class_columns: list[str],
    source_subdir: str = "__source__",
) -> FolderizationResult:
    """Create per-class symlinks in `repo_dir` based on a labels CSV.

    Args:
        repo_dir: The connector's repo/ dir (where the original .sol files live).
        labels_csv: Path to the labels CSV. Must have an ID column and one
                   column per class with `1` (positive) / `0` (negative) / `""` (unknown).
        id_column:  Name of the contract-ID column in labels_csv.
        class_columns: Names of the class columns. Their values `1`/`"1"` are
                      treated as positive. Empty cells are skipped (unknown).
        source_subdir: Subdirectory under repo_dir that holds the canonical
                      .sol files. If flat .sol files exist at repo_dir root
                      (the "flat source" case), they're MOVED into
                      `repo_dir/<source_subdir>/` first.

    Returns:
        FolderizationResult with counts.

    The labels CSV is expected to have **one row per contract**, with the ID
    column matching the source filename. For DIVE, contractID=42 means
    `repo_dir/__source__/42.sol` after the move.
    """
    result = FolderizationResult()
    source_dir = repo_dir / source_subdir

    # Step 1: if there are flat .sol files at repo_dir root and source_subdir
    # is non-empty, move them into source_dir. Idempotent — only moves files
    # that aren't already in source_dir.
    if source_subdir and source_dir.exists() is False:
        # First run: move flat files into source_dir
        flat_files = [p for p in repo_dir.glob("*.sol")]
        if flat_files:
            source_dir.mkdir(parents=True, exist_ok=True)
            for p in flat_files:
                dest = source_dir / p.name
                if not dest.exists():
                    shutil.move(str(p), str(dest))
                    result.files_moved += 1
    elif source_subdir and source_dir.exists():
        # source_dir already exists. Check if there are stragglers at the root
        # (e.g. files added since last run).
        for p in repo_dir.glob("*.sol"):
            dest = source_dir / p.name
            if not dest.exists():
                shutil.move(str(p), str(dest))
                result.files_moved += 1

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source dir {source_dir} does not exist. The connector should "
            f"have materialized the .sol files there first."
        )

    # Step 2: walk the labels CSV, create per-class symlinks
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contract_id = row.get(id_column, "").strip()
            if not contract_id:
                result.unparseable_labels += 1
                continue
            result.contracts_seen += 1

            # Find which classes are positive for this contract
            positive_classes = []
            for c in class_columns:
                v = (row.get(c, "") or "").strip()
                if v in ("1", "true", "True", "yes"):
                    positive_classes.append(c)
            if len(positive_classes) > 1:
                result.multi_label += 1
            if not positive_classes:
                # No labels for this contract — don't create any class folder entry
                continue

            # Source filename (canonical): e.g. "42.sol"
            src_filename = f"{contract_id}.sol"
            src_path = source_dir / src_filename
            if not src_path.exists():
                # Skip silently — there may be more IDs in the labels than files
                continue

            for cls in positive_classes:
                cls_dir = repo_dir / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                link_path = cls_dir / src_filename
                if not link_path.exists():
                    # Symlink to ../../<source_subdir>/<id>.sol
                    link_path.symlink_to(Path("..") / source_subdir / src_filename)
                    result.symlinks_created += 1
                if cls not in result.classes_present:
                    result.classes_present.append(cls)

    result.classes_present.sort()
    return result
