#!/usr/bin/env python3
"""Bounded high-signal secret scan for the current tree or reachable Git history.

This is deliberately dependency-free and conservative. It is one engineering
control, not a guarantee that no credential can exist. Provider-side secret
scanning and manual incident response remain complementary.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

ROOT = Path(__file__).resolve().parents[2]
OVERLAP = 512

PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private_key_assignment",
        re.compile(
            rb"(?i)\b(?:private|deployer|operator|signer)[_-]?key\b\s*[=:]\s*[\"']?(?:0x)?[0-9a-f]{64}\b"
        ),
    ),
    (
        "pem_private_key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "github_token",
        re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{24,}\b"),
    ),
    (
        "aws_access_key_id",
        re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    ),
    (
        "credentialed_provider_rpc",
        re.compile(
            rb"https?://[^\s\"'<>]*(?:infura|alchemy)[^\s\"'<>]*/[A-Za-z0-9_-]{20,}",
            re.IGNORECASE,
        ),
    ),
    (
        "mnemonic_assignment",
        re.compile(
            rb"(?i)\b(?:mnemonic|seed_phrase)\b\s*[=:]\s*[\"'][a-z]+(?:\s+[a-z]+){11,23}[\"']"
        ),
    ),
)


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    object_id: str


def _scan_bytes(chunks: Iterable[bytes], *, path: str, object_id: str) -> list[Finding]:
    findings: list[Finding] = []
    tail = b""
    seen: set[str] = set()
    for chunk in chunks:
        window = tail + chunk
        for kind, pattern in PATTERNS:
            if kind in seen:
                continue
            if pattern.search(window):
                findings.append(Finding(kind=kind, path=path, object_id=object_id))
                seen.add(kind)
        tail = window[-OVERLAP:]
    return findings


def _current_paths() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8", errors="surrogateescape") for item in proc.stdout.split(b"\0") if item]


def scan_current() -> tuple[list[Finding], dict[str, int]]:
    findings: list[Finding] = []
    scanned_files = 0
    scanned_bytes = 0
    for raw_path in _current_paths():
        path = ROOT / raw_path
        if not path.is_file():
            continue
        data = path.read_bytes()
        scanned_files += 1
        scanned_bytes += len(data)
        findings.extend(_scan_bytes((data,), path=raw_path, object_id="working-tree"))
    return findings, {"files": scanned_files, "bytes": scanned_bytes}


def _history_objects() -> list[tuple[str, str]]:
    proc = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    unique: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if " " not in line:
            continue
        oid, path = line.split(" ", 1)
        unique.setdefault(oid, path)
    return list(unique.items())


def _read_exact(stream: BinaryIO, size: int) -> Iterable[bytes]:
    remaining = size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError(f"unexpected EOF while reading Git object; {remaining} bytes remain")
        remaining -= len(chunk)
        yield chunk


def scan_history() -> tuple[list[Finding], dict[str, int]]:
    objects = _history_objects()
    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None

    findings: list[Finding] = []
    scanned_blobs = 0
    scanned_bytes = 0
    try:
        for oid, path in objects:
            proc.stdin.write((oid + "\n").encode("ascii"))
            proc.stdin.flush()
            header = proc.stdout.readline().decode("ascii", errors="replace").strip()
            parts = header.split()
            if len(parts) < 3 or parts[1] == "missing":
                raise RuntimeError(f"unexpected git cat-file header for {oid}: {header!r}")
            object_type = parts[1]
            size = int(parts[2])
            chunks = _read_exact(proc.stdout, size)
            if object_type == "blob":
                scanned_blobs += 1
                scanned_bytes += size
                findings.extend(_scan_bytes(chunks, path=path, object_id=oid))
            else:
                for _ in chunks:
                    pass
            terminator = proc.stdout.read(1)
            if terminator != b"\n":
                raise RuntimeError(f"missing git cat-file terminator after {oid}")
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)

    if proc.returncode:
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"git cat-file failed ({proc.returncode}): {stderr}")
    return findings, {"blobs": scanned_blobs, "bytes": scanned_bytes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("current", "history"), default="current")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings, stats = scan_current() if args.scope == "current" else scan_history()
    report = {
        "scope": args.scope,
        "status": "PASS" if not findings else "FAIL",
        "stats": stats,
        "findings": [asdict(item) for item in findings],
        "note": "Bounded high-signal scan; absence of findings is not a proof that no secret exists.",
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"SENTINEL secret scan | scope={args.scope} | status={report['status']} | stats={stats}")
        for item in findings:
            print(f"[FAIL] {item.kind}: {item.path} ({item.object_id})")
        if not findings:
            print("[PASS] no configured high-signal secret shapes found")
        print(report["note"])
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
