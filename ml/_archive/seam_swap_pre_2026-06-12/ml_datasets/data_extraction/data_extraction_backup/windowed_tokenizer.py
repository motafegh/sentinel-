"""Windowed tokenization for SENTINEL — GraphCodeBERT, [W, 512] output.

Extracted from ml/scripts/retokenize_windowed.py into a proper importable
module. The script's batch orchestration and MD5-based naming remain in the
script; this module exports only the per-file tokenization primitives needed
by the v2 data pipeline.

Why this module (not ml/src/data_extraction/tokenizer.py):
  - tokenizer.py uses microsoft/codebert-base with single-window (512,) output.
  - The training pipeline and all existing .pt files use microsoft/graphcodebert-base
    with [max_windows, 512] output (stride=256, sub-sampled via linspace).
  - The v2 orchestrator (sentinel_data/representation/orchestrator.py) must
    produce the same shape as v1 so the model's DataLoader sees uniform tensors.
  - This module decouples the hash from tokenization: v1 used MD5; v2 uses
    SHA-256 from Stage 1 meta.json. The caller sets the hash; this module
    only produces the token tensors.

Public surface:
  TOKENIZER_MODEL, WINDOW_SIZE, STRIDE, MAX_WINDOWS  — config constants
  init_worker()                                       — call once per process
  tokenize_windowed_contract(path, max_windows, strip_comments) — per-file entry point

A-1 fix (Stage 2 Task 2.5):
  strip_comments=True (default) removes /* */ and // comments before tokenization.
  This reclaims the token budget for actual code tokens rather than documentation.
  NatSpec tags are removed along with all other comment content — they add no
  vulnerability signal and are not part of the AST that the GNN sees anyway.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

TOKENIZER_MODEL: str = "microsoft/graphcodebert-base"
WINDOW_SIZE: int = 512   # CodeBERT max sequence length
STRIDE: int = 256        # overlap between consecutive windows
MAX_WINDOWS: int = 4     # cap; linspace sub-sampling preserves start/mid/end

_tokenizer = None  # process-level global; set by init_worker()


def init_worker() -> None:
    """Load the graphcodebert-base tokenizer into the process-level global.

    Call once per worker process before any tokenize_windowed_contract() call.
    tokenize_windowed_contract() calls this automatically as a fallback for
    single-process callers, but explicit init_worker() is faster in multiprocessing
    pools (avoids repeated auto-init on every call).
    """
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)


def _strip_comments(source: str) -> str:
    """Remove /* ... */ blocks and // line comments from Solidity source.

    Preserves line structure (replaces block comments with a single space so
    adjacent tokens don't merge). Applied before tokenization so the token
    budget is spent on code rather than documentation.
    """
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)
    return source


def _select_windows(
    all_input_ids: list,
    all_attention_masks: list,
    max_windows: int,
) -> tuple[list, list]:
    """Sub-sample to at most max_windows via linspace (preserves start/mid/end)."""
    W = len(all_input_ids)
    if W <= max_windows:
        return all_input_ids, all_attention_masks
    indices = [round(i) for i in np.linspace(0, W - 1, max_windows)]
    return (
        [all_input_ids[i] for i in indices],
        [all_attention_masks[i] for i in indices],
    )


def tokenize_windowed_contract(
    contract_path: str,
    max_windows: int = MAX_WINDOWS,
    strip_comments: bool = True,
) -> Optional[dict[str, Any]]:
    """Tokenize one .sol file into exactly [max_windows, 512] tensors.

    Output is always shape [max_windows, 512] regardless of contract length:
    - Short contracts (< 512 tokens): 1 real window; remaining windows are
      all-zero (attention_mask=0 → CrossAttentionFusion masks them out).
    - Long contracts (> max_windows windows): sub-sampled via linspace so
      beginning, middle, and end of the contract are covered.
    - Normal contracts (1 < W ≤ max_windows): padded with zero windows.

    Uniform shape is required for DataLoader collation — torch.stack() requires
    all tensors in a batch to have identical dimensions.

    Does NOT include a contract hash. The caller (v2 orchestrator) sets the
    hash from Stage 1's SHA-256 in meta.json and names the output file
    accordingly.

    Args:
        contract_path: Absolute path to a preprocessed .sol file.
        max_windows:   Maximum windows to produce. Default MAX_WINDOWS=4.

    Returns:
        dict with keys: input_ids [W,512], attention_mask [W,512], num_windows,
        stride, num_tokens, tokenizer_name, max_length.
        Returns None on empty file or any tokenization error.
    """
    global _tokenizer
    if _tokenizer is None:
        init_worker()

    try:
        path = Path(contract_path)
        try:
            code = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            code = path.read_text(encoding="utf-8", errors="replace")

        if not code.strip():
            return None

        if strip_comments:
            code = _strip_comments(code)
            if not code.strip():
                return None

        pad_id = _tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else 0

        encoded = _tokenizer(
            code,
            max_length=WINDOW_SIZE,
            padding="max_length",
            truncation=True,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

        all_ids: list = encoded["input_ids"].tolist()
        all_masks: list = encoded["attention_mask"].tolist()

        all_ids, all_masks = _select_windows(all_ids, all_masks, max_windows)
        num_real_windows = len(all_ids)

        # Pad to exactly max_windows with all-zero windows.
        while len(all_ids) < max_windows:
            all_ids.append([pad_id] * WINDOW_SIZE)
            all_masks.append([0] * WINDOW_SIZE)

        input_ids      = torch.tensor(all_ids,   dtype=torch.long)  # [max_windows, 512]
        attention_mask = torch.tensor(all_masks, dtype=torch.long)  # [max_windows, 512]
        num_real_tokens = int(attention_mask.sum().item())

        return {
            "input_ids":      input_ids,       # [max_windows, 512]
            "attention_mask": attention_mask,   # [max_windows, 512]
            "num_windows":    num_real_windows,
            "stride":         STRIDE,
            "num_tokens":     num_real_tokens,
            "tokenizer_name": TOKENIZER_MODEL,
            "max_length":     WINDOW_SIZE,
        }

    except Exception:
        return None
