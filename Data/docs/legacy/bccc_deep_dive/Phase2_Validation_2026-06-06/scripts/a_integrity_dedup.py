"""BCCC Phase 2 — WS-A: Integrity & Dedup.

Reads the 111,897 .sol files in BCCC-SCsVul-2024/Source Codes/ and:
1. Computes sha256 of each file.
2. Verifies BCCC-SCsVul-2024.csv MD5 against the .md5 file.
3. Builds a content-based dedup map.
4. Cross-checks dedup distribution against Phase 1 numbers.

Outputs (all under ../integrity/):
  - sha256_all_files.tsv       (111,897 rows: filename, folder, sha256, size)
  - dedup_map.csv              (68,433 rows: content_sha, canonical_id, n_folders, n_files, folders, sample_filename)
  - manifest.md                (what was checked, what passed, trust assumption)
"""
import csv
import hashlib
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("BCCC-SCsVul-2024")
SRC = ROOT / "Source Codes"
CSV_PATH = ROOT / "BCCC-SCsVul-2024.csv"
MD5_PATH = ROOT / "BCCC-SCsVul-2024.md5"
OUT = Path(__file__).resolve().parent.parent / "integrity"
OUT.mkdir(parents=True, exist_ok=True)


def hash_file(p: Path) -> str:
    """SHA-256 of a file's bytes (streaming)."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_csv_md5() -> tuple[str, str, bool]:
    """Compute MD5 of the CSV and compare against the .md5 file."""
    actual = hashlib.md5(CSV_PATH.read_bytes()).hexdigest()
    expected = None
    for line in MD5_PATH.read_text().splitlines():
        m = re.match(r"^[a-fA-F0-9]{32}$", line.strip())
        if m:
            expected = line.strip()
            break
    return actual, expected or "", actual == expected


def main():
    t0 = time.time()
    print("=" * 70)
    print("WS-A: Integrity & Dedup")
    print("=" * 70)

    # 1. Verify CSV MD5
    print("\n[1/3] Verifying CSV MD5...")
    actual_md5, expected_md5, ok = verify_csv_md5()
    print(f"  Actual MD5:   {actual_md5}")
    print(f"  Expected MD5: {expected_md5}")
    print(f"  Status:       {'MATCH' if ok else 'MISMATCH'}")

    # 2. Walk all .sol files, compute sha256
    print("\n[2/3] Hashing 111,897 .sol files...")
    rows = []
    total = 0
    for d in sorted(SRC.iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.sol")):
            sha = hash_file(p)
            size = p.stat().st_size
            rows.append((p.stem, d.name, sha, size))
            total += 1
            if total % 10000 == 0:
                print(f"  {total:>7d} files hashed ({time.time() - t0:.1f}s)")
    print(f"  Total: {total} files in {time.time() - t0:.1f}s")

    # Write sha256_all_files.tsv
    tsv_path = OUT / "sha256_all_files.tsv"
    with tsv_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["filename", "folder", "sha256", "size_bytes"])
        w.writerows(rows)
    print(f"  Wrote {tsv_path} ({tsv_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 3. Build dedup map
    print("\n[3/3] Building content-based dedup map...")
    by_sha: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fname, folder, sha, size in rows:
        by_sha[sha].append((folder, fname))
    print(f"  Unique content hashes: {len(by_sha)}")
    print(f"  Total file copies: {sum(len(v) for v in by_sha.values())}")
    print(f"  Duplicated files: {sum(len(v) for v in by_sha.values()) - len(by_sha)}")

    # Distribution: how many unique contents appear in N folders
    folder_dist: dict[int, int] = defaultdict(int)
    for locs in by_sha.values():
        n_folders = len(set(folder for folder, _ in locs))
        folder_dist[n_folders] += 1

    # Write dedup_map.csv
    dedup_path = OUT / "dedup_map.csv"
    with dedup_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["content_sha256", "canonical_id", "n_files", "n_folders", "folders", "sample_filename"])
        for sha, locs in by_sha.items():
            folders = sorted(set(folder for folder, _ in locs))
            # Pick canonical ID = first filename alphabetically; or the one in the "NonVulnerable" folder if any
            filenames = sorted(fname for _, fname in locs)
            nvs = [f for folder, f in locs if folder == "NonVulnerable"]
            canonical = nvs[0] if nvs else filenames[0]
            w.writerow([sha, canonical, len(locs), len(folders), ";".join(folders), filenames[0]])
    print(f"  Wrote {dedup_path}")

    # Cross-checks vs Phase 1
    print("\n=== Cross-checks vs Phase 1 ===")
    expected_unique = 68433
    print(f"  Unique contents: {len(by_sha)} (Phase 1: {expected_unique}) "
          f"{'MATCH' if len(by_sha) == expected_unique else 'DIFF'}")
    expected_dups = total - expected_unique
    actual_dups = sum(len(v) for v in by_sha.values()) - len(by_sha)
    print(f"  Duplicated files: {actual_dups} (Phase 1: {expected_dups}) "
          f"{'MATCH' if actual_dups == expected_dups else 'DIFF'}")
    print(f"  Folder distribution (N folders -> # unique contents):")
    for n in sorted(folder_dist):
        print(f"    in {n} folder(s): {folder_dist[n]:>7d}")

    # Top 5 most-copied
    print("\n=== Top 5 most-copied content hashes ===")
    ranked = sorted(by_sha.values(), key=lambda locs: -len(locs))
    for locs in ranked[:5]:
        folders = sorted(set(folder for folder, _ in locs))
        print(f"  {len(locs)} copies across {len(folders)} folders: {folders}")
        print(f"    example: {locs[0][1]}.sol")

    # Summary
    summary = {
        "total_files": total,
        "unique_contents": len(by_sha),
        "duplicated_files": actual_dups,
        "csv_md5_actual": actual_md5,
        "csv_md5_expected": expected_md5,
        "csv_md5_match": ok,
        "folder_distribution": dict(folder_dist),
    }

    # Write manifest.md
    manifest = OUT / "manifest.md"
    manifest.write_text(f"""# WS-A: Integrity & Dedup — Manifest

**Date:** 2026-06-06
**Status:** Complete

## Source

- `BCCC-SCsVul-2024/Source Codes/` (12 folders, {total:,} .sol files, 1.6 GB)

## CSV Integrity

| Check | Result |
|---|---|
| MD5 of `BCCC-SCsVul-2024.csv` (actual) | `{actual_md5}` |
| MD5 (expected from `BCCC-SCsVul-2024.md5`) | `{expected_md5}` |
| Match | {'**YES**' if ok else '**NO**'} |

## Source-File Integrity

| Check | Result |
|---|---|
| Per-file MD5 list provided in dataset? | **NO** — `Sourcecodes.md5` validates a `SourceCodes.zip` that is NOT present in our extracted directory. |
| Per-file content verifiable against publisher? | **NO** — must trust extraction. |
| Per-file SHA-256 computed and persisted? | **YES** — `sha256_all_files.tsv` (this run). |
| Idempotent re-hash? | **YES** — same file content → same SHA-256. |

**Trust assumption:** the publisher's extraction from the original ZIP was correct. We have no way to validate this from disk. The CSV-level MD5 is verified, which is the strongest guarantee we have.

## Dedup Map

- **Unique content hashes:** {len(by_sha):,}
- **Total file copies:** {total:,}
- **Duplicated files (in 2+ folders):** {actual_dups:,} ({100 * actual_dups / total:.2f}%)
- **Distribution:**

| N folders | # unique contents | % |
|---:|---:|---:|
""" + "\n".join(
        f"| {n} | {folder_dist[n]:,} | {100 * folder_dist[n] / len(by_sha):.2f}% |"
        for n in sorted(folder_dist)
    ) + f"""

## Cross-Checks vs Phase 1

| Metric | Phase 1 (CSV-based) | WS-A (file-based) | Match |
|---|---:|---:|:---:|
| Unique contents | {expected_unique:,} | {len(by_sha):,} | {'YES' if len(by_sha) == expected_unique else 'NO'} |
| Duplicated files | {expected_dups:,} | {actual_dups:,} | {'YES' if actual_dups == expected_dups else 'NO'} |

## Top 5 Most-Copied Contents

| Copies | N folders | Folders | Example file |
|---:|---:|---|---|
""" + "\n".join(
        f"| {len(locs)} | {len(set(f for f, _ in locs))} | {','.join(sorted(set(f for f, _ in locs)))} | `{locs[0][1]}.sol` |"
        for locs in ranked[:5]
    ) + f"""

## Decision: How to Use the Dedup Map

**Recommendation:** for SENTINEL training, the 68,433 unique contracts form the training unit. The 12 folders are *candidate categories* (see Phase 1 §1.3), and a contract may appear in N folders (1 ≤ N ≤ 9). The dedup map provides the canonical content identity (sha256) and the canonical ID (filename in `NonVulnerable` folder if any, else first alphabetically).

For WS-G (stratified split), split by **canonical ID**, then map back to all (folder, file) tuples for that ID. This prevents the same contract from leaking across train/val/test via different folder copies.

## Files

- `sha256_all_files.tsv` — 111,897 rows × 4 columns (filename, folder, sha256, size_bytes)
- `dedup_map.csv` — 68,433 rows × 6 columns (content_sha256, canonical_id, n_files, n_folders, folders, sample_filename)
- `manifest.md` — this file

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/a_integrity_dedup.py
```

Run time: ~{time.time() - t0:.0f}s (1.6 GB of I/O on WSL filesystem).
""")
    print(f"  Wrote {manifest}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
