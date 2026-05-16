#!/usr/bin/env python
"""
Windowed CodeBERT Tokenizer — Multi-Window Production Pipeline
==============================================================

Replaces the single-window (512-token, hard-truncation) approach used in v4/v5.
Each contract is tokenized with a sliding window of 512 tokens and stride 256,
producing W overlapping windows that together cover the full contract source.

WHY WINDOWED:
    Median Solidity contract = 2,469 tokens.
    Single-window sees only ~21% of the median contract.
    4 windows of 512 with stride 256 cover 1,534 tokens = 62% of the median.
    Contracts shorter than 512 tokens produce W=1 (identical to old behaviour).

OUTPUT FORMAT:
    Same .pt file structure as the original tokenizer, but:
        input_ids:      [W, 512] instead of [512]
        attention_mask: [W, 512] instead of [512]
        num_windows:    int — how many windows were produced
        stride:         int — stride used (256)

    Files written to ml/data/tokens_windowed/<hash>.pt
    DualPathDataset and collate_fn already accept [W, 512] tokens as of this change.

WINDOW SELECTION:
    Very long contracts (W > max_windows) are sub-sampled using linspace to pick
    max_windows evenly-spaced windows. This keeps GPU memory bounded while still
    covering beginning + middle + end of long contracts rather than just truncating.

    Default max_windows=4 → max 4×512=2048 token positions per contract.
    At r3070 8GB VRAM this fits batch_size=8 comfortably.
"""

import json
import math
import multiprocessing as mp
import sys
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.hash_utils import get_contract_hash, get_filename_from_hash
from src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("pandas not installed. Run: poetry add pandas") from e

try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError("transformers not installed. Run: poetry add transformers") from e


# ── Configuration ─────────────────────────────────────────────────────────────

TOKENIZER_MODEL = "microsoft/codebert-base"
WINDOW_SIZE     = 512    # CodeBERT max seq length
STRIDE          = 256    # overlap between consecutive windows
MAX_WINDOWS     = 4      # cap for very long contracts (linspace sub-sampling)

DEFAULT_WORKERS        = 11
DEFAULT_CHUNK_SIZE     = 50
CHECKPOINT_INTERVAL    = 500

DEFAULT_INPUT  = "ml/data/processed/multilabel_index_deduped.csv"
DEFAULT_OUTPUT = "ml/data/tokens_windowed"


# ── Worker-level global tokenizer ─────────────────────────────────────────────

_tokenizer = None


def _init_worker() -> None:
    global _tokenizer
    print(f"  Worker {mp.current_process().name}: loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        cache_dir=".cache/huggingface",
        use_fast=True,
    )
    print(f"  Worker {mp.current_process().name}: ready")


# ── Windowed tokenization ─────────────────────────────────────────────────────

def _select_windows(
    all_input_ids:      List[List[int]],
    all_attention_masks: List[List[int]],
    max_windows: int,
) -> tuple:
    """
    If the tokenizer produced more than max_windows windows, sub-sample using
    linspace so that beginning, middle, and end of the contract are all covered.
    """
    W = len(all_input_ids)
    if W <= max_windows:
        return all_input_ids, all_attention_masks

    # linspace: pick max_windows indices spread evenly across 0..W-1
    indices = [round(i) for i in np.linspace(0, W - 1, max_windows)]
    sel_ids   = [all_input_ids[i]       for i in indices]
    sel_masks = [all_attention_masks[i] for i in indices]
    return sel_ids, sel_masks


def tokenize_windowed(contract_path: str, max_windows: int = MAX_WINDOWS) -> Optional[Dict[str, Any]]:
    """
    Tokenize a single contract into exactly [max_windows, 512] tensors.

    Always outputs shape [max_windows, 512] regardless of contract length:
    - Short contracts (< 512 tokens): W=1 real window, remaining (max_windows-1)
      windows are all-zero (attention_mask=0 → CrossAttentionFusion ignores them)
    - Long contracts (> max_windows windows): sub-sampled via linspace
    - Normal contracts (1 < W <= max_windows): padded with zero windows

    Uniform output shape is required for DataLoader collation — torch.stack()
    requires all tensors in a batch to have identical dimensions.

    Runs in worker processes. Uses the module-level _tokenizer.
    """
    global _tokenizer

    try:
        contract_path = Path(contract_path)
        try:
            code = contract_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            code = contract_path.read_text(encoding="utf-8", errors="replace")

        if len(code.strip()) == 0:
            return None

        pad_id = _tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else 0

        # Windowed tokenization — returns one dict per window
        encoded = _tokenizer(
            code,
            max_length=WINDOW_SIZE,
            padding="max_length",
            truncation=True,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

        all_ids   = encoded["input_ids"].tolist()        # list of W lists of 512 ints
        all_masks = encoded["attention_mask"].tolist()   # same

        # Sub-sample if too many windows (linspace preserves start/middle/end)
        all_ids, all_masks = _select_windows(all_ids, all_masks, max_windows)

        num_real_windows = len(all_ids)

        # Pad to exactly max_windows with all-zero windows.
        # attention_mask=0 on padding windows → CrossAttentionFusion's key_padding_mask
        # masks them out; they contribute zero to node→token cross-attention.
        while len(all_ids) < max_windows:
            all_ids.append([pad_id] * WINDOW_SIZE)
            all_masks.append([0] * WINDOW_SIZE)

        input_ids      = torch.tensor(all_ids,   dtype=torch.long)    # [max_windows, 512]
        attention_mask = torch.tensor(all_masks, dtype=torch.long)    # [max_windows, 512]

        # Real token count: sum of attention mask 1s across real windows only
        num_real_tokens = int(attention_mask.sum().item())

        contract_hash = get_contract_hash(contract_path)

        return {
            "input_ids":              input_ids,               # [max_windows, 512]
            "attention_mask":         attention_mask,           # [max_windows, 512]
            "num_windows":            num_real_windows,         # actual windows (not counting padding)
            "stride":                 STRIDE,
            "contract_hash":          contract_hash,
            "contract_path":          str(contract_path),
            "num_tokens":             num_real_tokens,
            "tokenizer_name":         TOKENIZER_MODEL,
            "max_length":             WINDOW_SIZE,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }

    except Exception:
        return None


def _tokenize_with_args(args: tuple) -> Optional[Dict[str, Any]]:
    contract_path, max_windows = args
    return tokenize_windowed(contract_path, max_windows=max_windows)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_token_file(token_data: Dict[str, Any], output_dir: Path) -> bool:
    try:
        filename = get_filename_from_hash(token_data["contract_hash"])
        torch.save(token_data, output_dir / filename)
        return True
    except Exception:
        return False


# ── Batch processing ──────────────────────────────────────────────────────────

def process_batch(
    contracts_df:       "pd.DataFrame",
    output_dir:         Path,
    max_windows:        int = MAX_WINDOWS,
    n_workers:          int = DEFAULT_WORKERS,
    chunk_size:         int = DEFAULT_CHUNK_SIZE,
    checkpoint_every:   int = CHECKPOINT_INTERVAL,
) -> Dict[str, Any]:

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"
    failed_file     = output_dir / "failed_contracts.json"

    # Load checkpoint
    processed_hashes: set = set()
    failed_hashes: list   = []

    if checkpoint_file.exists():
        print("Loading checkpoint...")
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
            processed_hashes = set(ckpt.get("processed", []))
            failed_hashes    = ckpt.get("failed", [])
        print(f"  Found {len(processed_hashes):,} processed, {len(failed_hashes):,} failed")

    # Filter already processed
    if processed_hashes:
        contracts_df = contracts_df.copy()
        contracts_df["_h"] = contracts_df["contract_path"].apply(get_contract_hash)
        contracts_df = contracts_df[~contracts_df["_h"].isin(processed_hashes)].drop(columns=["_h"])
        print(f"  Remaining: {len(contracts_df):,} contracts")

    if len(contracts_df) == 0:
        print("All contracts already processed.")
        return {"total": len(processed_hashes), "successful": len(processed_hashes),
                "failed": len(failed_hashes), "new": 0}

    contract_paths = contracts_df["contract_path"].tolist()
    total = len(contract_paths)
    args_iter = [(p, max_windows) for p in contract_paths]

    print(f"\nStarting windowed tokenization (max_windows={max_windows}, stride={STRIDE})...")
    print(f"  Contracts: {total:,}  Workers: {n_workers}  Chunk: {chunk_size}")

    stats = {"successful": 0, "failed": 0, "w1": 0, "w2": 0, "w3": 0, "w4plus": 0}

    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        results_iter = pool.imap(_tokenize_with_args, args_iter, chunksize=chunk_size)

        for i, result in enumerate(tqdm(results_iter, total=total, desc="Tokenizing")):
            if result is not None:
                if save_token_file(result, output_dir):
                    processed_hashes.add(result["contract_hash"])
                    stats["successful"] += 1
                    W = result["num_windows"]
                    if   W == 1: stats["w1"] += 1
                    elif W == 2: stats["w2"] += 1
                    elif W == 3: stats["w3"] += 1
                    else:        stats["w4plus"] += 1
                else:
                    try:
                        failed_hashes.append(get_contract_hash(contract_paths[i]))
                    except Exception:
                        pass
                    stats["failed"] += 1
            else:
                try:
                    failed_hashes.append(get_contract_hash(contract_paths[i]))
                except Exception:
                    pass
                stats["failed"] += 1

            current_total = len(processed_hashes)
            if current_total % checkpoint_every == 0 and current_total > 0:
                _write_checkpoint(checkpoint_file, failed_file,
                                  processed_hashes, failed_hashes, completed=False)

    _write_checkpoint(checkpoint_file, failed_file,
                      processed_hashes, failed_hashes, completed=True)

    return {
        "total":      len(processed_hashes),
        "successful": stats["successful"],
        "failed":     stats["failed"],
        "w1":         stats["w1"],
        "w2":         stats["w2"],
        "w3":         stats["w3"],
        "w4plus":     stats["w4plus"],
        "truncation_rate": stats["w4plus"] / stats["successful"] if stats["successful"] > 0 else 0,
    }


def _write_checkpoint(ckpt_file, failed_file, processed_hashes, failed_hashes, completed: bool):
    with open(ckpt_file, "w") as f:
        json.dump({
            "processed":  list(processed_hashes),
            "failed":     failed_hashes,
            "total":      len(processed_hashes),
            "timestamp":  datetime.now().isoformat(),
            "completed":  completed,
        }, f, indent=2)
    with open(failed_file, "w") as f:
        json.dump(failed_hashes, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Windowed CodeBERT Tokenizer — Multi-Window Production Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml/scripts/retokenize_windowed.py
  python ml/scripts/retokenize_windowed.py --max-windows 4 --workers 11
  python ml/scripts/retokenize_windowed.py --test      # 100 contracts only
        """,
    )
    parser.add_argument("--input",            default=DEFAULT_INPUT)
    parser.add_argument("--output",           default=DEFAULT_OUTPUT)
    parser.add_argument("--max-windows",      type=int, default=MAX_WINDOWS,
                        help=f"Max windows per contract (default: {MAX_WINDOWS})")
    parser.add_argument("--workers",          type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk-size",       type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--test",             action="store_true",
                        help="Process 100 contracts only")
    args = parser.parse_args()

    print("=" * 70)
    print("Windowed CodeBERT Tokenizer")
    print("=" * 70)
    print(f"Model:       {TOKENIZER_MODEL}")
    print(f"Window size: {WINDOW_SIZE}  Stride: {STRIDE}  Max windows: {args.max_windows}")
    print(f"Output:      {args.output}")
    print("=" * 70)

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    if "success" in df.columns:
        df = df[df["success"] == True].copy()
    print(f"Loaded {len(df):,} contracts")

    if args.test:
        df = df.head(100)
        print(f"TEST MODE: {len(df)} contracts")

    stats = process_batch(
        contracts_df=df,
        output_dir=Path(args.output),
        max_windows=args.max_windows,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        checkpoint_every=args.checkpoint_every,
    )

    print()
    print("=" * 70)
    print("TOKENIZATION COMPLETE")
    print("=" * 70)
    print(f"Total:       {stats['total']:,}")
    print(f"Successful:  {stats['successful']:,}")
    print(f"Failed:      {stats['failed']:,}")
    print(f"W=1 (short): {stats.get('w1', 0):,}")
    print(f"W=2:         {stats.get('w2', 0):,}")
    print(f"W=3:         {stats.get('w3', 0):,}")
    print(f"W=4 (capped):{stats.get('w4plus', 0):,}")
    print(f"Output:      {args.output}")
    print("=" * 70)
