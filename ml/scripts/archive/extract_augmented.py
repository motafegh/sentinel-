#!/usr/bin/env python3
"""
extract_augmented.py — Graph + Token Extraction for Augmented Contracts

Bridges generate_safe_variants.py output into the SENTINEL training pipeline:

  1. Extract PyG graphs (v2 schema — 12-dim nodes, 7 edge types) for each
     augmented .sol file via extract_contract_graph().
  2. Tokenize with CodeBERT (MAX_LENGTH=512, MD5-named .pt files).
  3. Append new rows to multilabel_index.csv with caller-supplied labels.
  4. Skip files whose MD5 stem already appears in the index (idempotent).

Data source: BCCC-SCsVul-2024 only (via generate_safe_variants.py output).

IMPORTANT — Must set before running:
    export TRANSFORMERS_OFFLINE=1

USAGE
─────
# Safe variants (all zeros — NonVulnerable)
poetry run python ml/scripts/extract_augmented.py \\
    --input-dir ml/data/augmented/safe \\
    --graphs-dir ml/data/graphs \\
    --tokens-dir ml/data/tokens \\
    --multilabel-csv ml/data/processed/multilabel_index.csv

# With explicit vulnerability labels (e.g. augmented DoS contracts)
poetry run python ml/scripts/extract_augmented.py \\
    --input-dir ml/data/augmented/dos_bounded \\
    --label DenialOfService \\
    --graphs-dir ml/data/graphs \\
    --tokens-dir ml/data/tokens \\
    --multilabel-csv ml/data/processed/multilabel_index.csv

# Multiple labels (multi-label contracts)
poetry run python ml/scripts/extract_augmented.py \\
    --input-dir ml/data/augmented/reentrancy_with_callto \\
    --label Reentrancy --label CallToUnknown \\
    --graphs-dir ml/data/graphs \\
    --tokens-dir ml/data/tokens \\
    --multilabel-csv ml/data/processed/multilabel_index.csv

# Smoke run (5 contracts, no writes)
poetry run python ml/scripts/extract_augmented.py \\
    --input-dir ml/data/augmented/safe \\
    --max-contracts 5 --dry-run

PIPELINE COMMANDS (run after this script)
──────────────────────────────────────────
# Update train split (freeze existing val/test):
poetry run python ml/scripts/create_splits.py \\
    --multilabel-index ml/data/processed/multilabel_index.csv \\
    --splits-dir ml/data/splits \\
    --freeze-val-test

# Validate augmented graphs:
poetry run python ml/scripts/validate_graph_dataset.py \\
    --check-dim 12 \\
    --check-edge-types 7 \\
    --check-contains-edges \\
    --check-control-flow \\
    --check-cfg-subtypes

CLASS_NAMES (must match trainer.py):
  0 CallToUnknown  1 DenialOfService  2 ExternalBug      3 GasException
  4 IntegerUO      5 MishandledException  6 Reentrancy   7 Timestamp
  8 TransactionOrderDependence   9 UnusedReturn
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.src.preprocessing.graph_extractor import (
    GraphExtractionConfig,
    GraphExtractionError,
    extract_contract_graph,
)
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM
from ml.src.data_extraction.ast_extractor import get_solc_binary
from ml.src.utils.hash_utils import get_contract_hash, get_filename_from_hash

logger = logging.getLogger(__name__)


# ── Class definitions ─────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
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

_ZERO_LABEL = {c: 0 for c in CLASS_NAMES}


# ── Label helpers ─────────────────────────────────────────────────────────────

def _build_label(label_args: list[str], label_json: Optional[str]) -> dict[str, int]:
    """Build the label dict {ClassName: 0|1} from CLI arguments."""
    label = dict(_ZERO_LABEL)

    if label_json:
        overrides = json.loads(label_json)
        for cls, val in overrides.items():
            if cls not in label:
                raise ValueError(f"Unknown class in --label-json: {cls!r}. Valid: {CLASS_NAMES}")
            label[cls] = int(val)

    for cls in label_args:
        if cls not in label:
            raise ValueError(f"Unknown class in --label: {cls!r}. Valid: {CLASS_NAMES}")
        label[cls] = 1

    return label


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _load_tokenizer():
    """Load CodeBERT tokenizer from local cache (TRANSFORMERS_OFFLINE=1 required)."""
    from transformers import AutoTokenizer
    cache_dir = str(_PROJECT_ROOT / ".cache" / "huggingface")
    return AutoTokenizer.from_pretrained(
        "microsoft/codebert-base",
        cache_dir=cache_dir,
        use_fast=True,
    )


def _tokenize_source(tokenizer, source: str, sol_path: Path) -> dict:
    """Tokenize source and return a dict matching tokenizer.py's save format."""
    encoded = tokenizer(
        source,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    num_real       = int(attention_mask.sum().item())

    return {
        "input_ids":              input_ids,
        "attention_mask":         attention_mask,
        "contract_hash":          get_contract_hash(sol_path),
        "contract_path":          str(sol_path),
        "num_tokens":             num_real,
        "truncated":              num_real >= 510,
        "tokenizer_name":         "microsoft/codebert-base",
        "max_length":             512,
        "feature_schema_version": "v2",
    }


# ── Graph extraction ──────────────────────────────────────────────────────────

_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+[^;]*?(\d+\.\d+\.\d+)")


def _extract_graph(sol_path: Path, graphs_dir: Path, dry_run: bool) -> Optional[str]:
    """
    Extract a PyG graph and write it to graphs_dir/{md5}.pt.
    Returns the md5 string on success, None on failure.
    """
    source  = sol_path.read_text(encoding="utf-8", errors="replace")
    m       = _PRAGMA_RE.search(source)
    version = m.group(1) if m else None
    config  = GraphExtractionConfig(
        solc_binary=get_solc_binary(version) if version else None,
        solc_version=version,
    )

    try:
        data = extract_contract_graph(str(sol_path), config)
    except GraphExtractionError as exc:
        logger.debug("Extraction failed for %s: %s", sol_path.name, exc)
        return None

    if data.x.shape[1] != NODE_FEATURE_DIM:
        logger.error(
            "%s: node dim=%d, expected %d — skipping",
            sol_path.name, data.x.shape[1], NODE_FEATURE_DIM,
        )
        return None

    md5 = get_contract_hash(sol_path)
    data.contract_hash = md5
    data.contract_path = str(sol_path)
    data.y = torch.tensor([0], dtype=torch.long)

    if not dry_run:
        torch.save(data, graphs_dir / get_filename_from_hash(md5))
    return md5


# ── Label index ───────────────────────────────────────────────────────────────

def _load_known_md5s(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    seen: set[str] = set()
    with csv_path.open(encoding="utf-8") as fh:
        fh.readline()  # skip header
        for line in fh:
            stem = line.split(",")[0].strip()
            if stem:
                seen.add(stem)
    return seen


def _append_row(csv_path: Path, md5: str, label: dict[str, int], dry_run: bool) -> None:
    if dry_run:
        return
    row = [md5] + [str(label[c]) for c in CLASS_NAMES]
    with csv_path.open("a", encoding="utf-8") as fh:
        fh.write(",".join(row) + "\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    input_dir:      Path,
    graphs_dir:     Path,
    tokens_dir:     Path,
    multilabel_csv: Path,
    label:          dict[str, int],
    max_contracts:  Optional[int],
    dry_run:        bool,
) -> dict:
    sol_files = sorted(input_dir.rglob("*.sol"))
    if max_contracts:
        sol_files = sol_files[:max_contracts]

    if not sol_files:
        logger.error("No .sol files found in %s", input_dir)
        return {"total": 0, "accepted": 0, "skipped_duplicate": 0, "failed": 0}

    known_md5s = _load_known_md5s(multilabel_csv)
    logger.info("Existing index: %d entries", len(known_md5s))

    if not dry_run:
        graphs_dir.mkdir(parents=True, exist_ok=True)
        tokens_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CodeBERT tokenizer...")
    try:
        tokenizer = _load_tokenizer()
    except Exception as exc:
        logger.error("Failed to load tokenizer: %s", exc)
        logger.error("Ensure TRANSFORMERS_OFFLINE=1 and CodeBERT is cached.")
        sys.exit(1)

    label_str = ",".join(c for c, v in label.items() if v) or "NonVulnerable"
    logger.info(
        "Processing %d contracts  label=[%s]%s",
        len(sol_files), label_str, "  [DRY RUN]" if dry_run else "",
    )

    accepted = skipped_dup = failed = 0

    for i, sol_path in enumerate(sol_files, 1):
        logger.info("[%d/%d] %s", i, len(sol_files), sol_path.name)

        md5 = get_contract_hash(sol_path)
        if md5 in known_md5s:
            logger.debug("  already in index — skip")
            skipped_dup += 1
            continue

        # Graph
        result = _extract_graph(sol_path, graphs_dir, dry_run)
        if result is None:
            failed += 1
            continue

        # Tokens
        try:
            source     = sol_path.read_text(encoding="utf-8", errors="replace")
            token_data = _tokenize_source(tokenizer, source, sol_path)
        except Exception as exc:
            logger.warning("  tokenization failed: %s", exc)
            failed += 1
            if not dry_run:
                (graphs_dir / get_filename_from_hash(md5)).unlink(missing_ok=True)
            continue

        if not dry_run:
            torch.save(token_data, tokens_dir / get_filename_from_hash(md5))

        _append_row(multilabel_csv, md5, label, dry_run)
        known_md5s.add(md5)
        accepted += 1
        logger.info("  ✓ %s  [%s]", md5[:8], label_str)

    summary = {
        "total":             len(sol_files),
        "accepted":          accepted,
        "skipped_duplicate": skipped_dup,
        "failed":            failed,
        "dry_run":           dry_run,
    }
    logger.info(
        "Done. accepted=%d  skipped_dup=%d  failed=%d",
        accepted, skipped_dup, failed,
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract graphs + tokens for BCCC-augmented contracts; update multilabel_index.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir",       required=True,  type=Path)
    p.add_argument("--graphs-dir",      default="ml/data/graphs",                   type=Path)
    p.add_argument("--tokens-dir",      default="ml/data/tokens",                   type=Path)
    p.add_argument("--multilabel-csv",  default="ml/data/processed/multilabel_index.csv", type=Path)
    p.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="ClassName",
        dest="label_args",
        help=(
            f"Set one class to 1 (repeatable). Default: all-zeros (NonVulnerable). "
            f"Valid: {', '.join(CLASS_NAMES)}"
        ),
    )
    p.add_argument(
        "--label-json",
        default=None,
        metavar="JSON",
        help='Multi-label JSON, e.g. \'{"Reentrancy": 1, "CallToUnknown": 1}\'',
    )
    p.add_argument("--max-contracts",   type=int, default=None, metavar="N")
    p.add_argument("--dry-run",         action="store_true", default=False)
    p.add_argument("--verbose",         action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input_dir.exists():
        logger.error("--input-dir does not exist: %s", args.input_dir)
        sys.exit(1)

    if "TRANSFORMERS_OFFLINE" not in os.environ:
        logger.warning("TRANSFORMERS_OFFLINE not set — HuggingFace may attempt a download.")

    try:
        label = _build_label(args.label_args, args.label_json)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    summary = run(
        input_dir=args.input_dir,
        graphs_dir=args.graphs_dir,
        tokens_dir=args.tokens_dir,
        multilabel_csv=args.multilabel_csv,
        label=label,
        max_contracts=args.max_contracts,
        dry_run=args.dry_run,
    )
    sys.exit(0 if summary["accepted"] > 0 or summary["total"] == 0 else 1)


if __name__ == "__main__":
    main()
