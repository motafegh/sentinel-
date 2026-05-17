#!/usr/bin/env python3
"""
inject_augmented.py — Extract graphs+tokens for augmented contracts and inject
into the deduped CSV + train split.

WHAT IT DOES
────────────
1. Scans ml/data/augmented/*.sol
2. For each file: compute path-MD5, extract graph (.pt), extract tokens (.pt)
3. Build label rows:
     cei_vuln_*.sol  → Reentrancy=1, all other classes=0
     cei_safe_*.sol  → all classes=0
     dos_*.sol       → DenialOfService=1, all other classes=0
     (other patterns extensible)
4. Append new rows to multilabel_index_deduped.csv
5. Append new row indices to train_indices.npy (train-only injection)
6. Print summary

SAFE TO RE-RUN: skips any MD5 already present in the CSV.

USAGE
─────
  source ml/.venv/bin/activate
  PYTHONPATH=. python ml/scripts/inject_augmented.py
  PYTHONPATH=. python ml/scripts/inject_augmented.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from transformers import AutoTokenizer

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from ml.src.utils.hash_utils import get_contract_hash  # noqa: E402
from ml.src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION  # noqa: E402

_TOKENIZER_MODEL = "microsoft/codebert-base"
_WINDOW_SIZE     = 512
_STRIDE          = 256
_MAX_WINDOWS     = 4
_tokenizer       = None  # lazy-loaded on first use
AUG_DIR        = PROJECT_ROOT / "ml" / "data" / "augmented"
GRAPHS_DIR     = PROJECT_ROOT / "ml" / "data" / "graphs"
TOKENS_DIR     = PROJECT_ROOT / "ml" / "data" / "tokens_windowed"
DEDUP_CSV      = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index_deduped.csv"
SPLITS_DIR     = PROJECT_ROOT / "ml" / "data" / "splits" / "deduped"

_SOLC_ARTIFACTS = PROJECT_ROOT / "ml" / ".venv" / ".solc-select" / "artifacts"
_PRAGMA_RE = re.compile(r'pragma\s+solidity\s+[\^~>=<\s]*(\d+\.\d+\.\d+)')
_LATEST_PATCH = {"0.4": "0.4.26", "0.5": "0.5.17", "0.6": "0.6.12",
                 "0.7": "0.7.6",  "0.8": "0.8.31"}

CLASS_NAMES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn",
]


def _md5(sol: Path) -> str:
    return get_contract_hash(sol.relative_to(PROJECT_ROOT))


def _solc_version(sol: Path) -> str:
    try:
        txt = sol.read_text(encoding="utf-8", errors="replace")
        m = _PRAGMA_RE.search(txt)
        if m:
            minor = ".".join(m.group(1).split(".")[:2])
            return _LATEST_PATCH.get(minor, m.group(1))
    except OSError:
        pass
    return "0.8.31"


def _solc_binary(ver: str) -> Path | None:
    b = _SOLC_ARTIFACTS / f"solc-{ver}" / f"solc-{ver}"
    return b if b.exists() else None


def _label_row(sol: Path) -> dict:
    row = {c: 0 for c in CLASS_NAMES}
    name = sol.stem.lower()
    # Match on prefix patterns — "safe" contracts always get all-zeros.
    if name.startswith("cei_vuln"):
        row["Reentrancy"] = 1
    elif name.startswith("dos_vuln"):
        row["DenialOfService"] = 1
    # Any other prefix (cei_safe, dos_safe) → all zeros.
    return row


def _extract_graph(sol: Path, dry_run: bool) -> bool:
    from ml.src.preprocessing.graph_extractor import (
        GraphExtractionConfig, extract_contract_graph,
        EmptyGraphError, SolcCompilationError, SlitherParseError,
    )
    md5 = _md5(sol)
    out = GRAPHS_DIR / f"{md5}.pt"
    ver = _solc_version(sol)
    cfg = GraphExtractionConfig(solc_version=ver, solc_binary=_solc_binary(ver))
    try:
        g = extract_contract_graph(sol, cfg)
    except (EmptyGraphError, SolcCompilationError, SlitherParseError) as e:
        logger.warning(f"  Graph skip {sol.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"  Graph fail {sol.name}: {e}")
        return False
    if not dry_run:
        tmp = out.with_suffix(".tmp")
        torch.save(g, tmp)
        tmp.rename(out)
    logger.debug(f"  Graph OK  {sol.name}  nodes={g.num_nodes}")
    return True


def _tokenize_contract(sol: Path, dry_run: bool) -> bool:
    """Tokenize one augmented contract into tokens_windowed/{md5}.pt."""
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading CodeBERT tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL, cache_dir=".cache/huggingface", use_fast=True
        )

    md5 = _md5(sol)
    out = TOKENS_DIR / f"{md5}.pt"
    if out.exists():
        logger.debug(f"  Token OK  {sol.name} (already exists)")
        return True

    try:
        code = sol.read_text(encoding="utf-8", errors="replace")
        if not code.strip():
            logger.warning(f"  Token skip {sol.name}: empty file")
            return False

        pad_id = _tokenizer.pad_token_id or 0
        encoded = _tokenizer(
            code, max_length=_WINDOW_SIZE, padding="max_length",
            truncation=True, stride=_STRIDE,
            return_overflowing_tokens=True, return_tensors="pt",
        )
        all_ids   = encoded["input_ids"].tolist()
        all_masks = encoded["attention_mask"].tolist()

        # Sub-sample if more than MAX_WINDOWS
        if len(all_ids) > _MAX_WINDOWS:
            import numpy as _np
            idxs = [round(i) for i in _np.linspace(0, len(all_ids) - 1, _MAX_WINDOWS)]
            all_ids   = [all_ids[i]   for i in idxs]
            all_masks = [all_masks[i] for i in idxs]

        while len(all_ids) < _MAX_WINDOWS:
            all_ids.append([pad_id] * _WINDOW_SIZE)
            all_masks.append([0] * _WINDOW_SIZE)

        payload = {
            "input_ids":              torch.tensor(all_ids,   dtype=torch.long),
            "attention_mask":         torch.tensor(all_masks, dtype=torch.long),
            "num_windows":            len([m for m in all_masks if any(m)]),
            "contract_hash":          md5,
            "contract_path":          str(sol),
            "tokenizer_name":         _TOKENIZER_MODEL,
            "max_length":             _WINDOW_SIZE,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }
        if not dry_run:
            tmp = out.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.rename(out)
        logger.debug(f"  Token OK  {sol.name}")
        return True
    except Exception as e:
        logger.error(f"  Token fail {sol.name}: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--aug-dir", type=Path, default=AUG_DIR)
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    if args.dry_run:
        logger.info("DRY RUN — no files written")

    sol_files = sorted(args.aug_dir.glob("*.sol"))
    logger.info(f"Found {len(sol_files)} .sol files in {args.aug_dir}")

    # Load existing CSV to check for already-injected MD5s
    df = pd.read_csv(DEDUP_CSV).rename(columns={"md5_stem": "md5"})
    existing_md5s = set(df["md5"])
    logger.info(f"Existing CSV rows: {len(df):,}")

    new_rows = []
    graph_ok = graph_skip = token_ok = token_skip = 0

    for sol in sol_files:
        md5 = _md5(sol)
        if md5 in existing_md5s:
            logger.debug(f"  Already in CSV: {sol.name}")
            continue

        label = _label_row(sol)
        vuln_str = "VULN" if any(v for v in label.values()) else "safe"
        logger.info(f"  {sol.name}  [{vuln_str}]")

        g_ok = _extract_graph(sol, args.dry_run)
        t_ok = _tokenize_contract(sol, args.dry_run)

        if g_ok:
            graph_ok += 1
        else:
            graph_skip += 1

        if t_ok:
            token_ok += 1
        else:
            token_skip += 1

        if g_ok and t_ok:
            new_rows.append({"md5": md5, **label})
        else:
            logger.warning(f"  Skipping CSV row for {sol.name} (graph={g_ok} token={t_ok})")

    logger.info(f"Graphs: {graph_ok} ok, {graph_skip} skip")
    logger.info(f"Tokens: {token_ok} ok, {token_skip} skip")
    logger.info(f"New CSV rows: {len(new_rows)}")

    if not new_rows:
        logger.info("Nothing new to inject.")
        return

    if not args.dry_run:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([df, new_df], ignore_index=True)
        combined.rename(columns={"md5": "md5_stem"}).to_csv(DEDUP_CSV, index=False)
        logger.info(f"CSV updated: {len(df):,} → {len(combined):,} rows")

        # Append new row indices to train split (train-only injection)
        train_idx = np.load(SPLITS_DIR / "train_indices.npy")
        new_indices = np.arange(len(df), len(combined))
        train_idx_updated = np.concatenate([train_idx, new_indices])
        np.save(SPLITS_DIR / "train_indices.npy", train_idx_updated)
        logger.info(f"Train split: {len(train_idx):,} → {len(train_idx_updated):,}")

        # Label breakdown of injected rows
        logger.info("Injected label breakdown:")
        for cls in CLASS_NAMES:
            n = int(new_df[cls].sum())
            if n:
                logger.info(f"  {cls}: +{n}")

    logger.info("DONE")


if __name__ == "__main__":
    main()
