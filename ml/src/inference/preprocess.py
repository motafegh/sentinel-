"""
preprocess.py — Online Inference Preprocessor (Thin Wrapper)

WHAT THIS FILE DOES
───────────────────
Converts one Solidity contract into the two tensors SentinelModel expects:

  graph  : PyG Data object  (AST/CFG structure  → GNNEncoder)
  tokens : dict             (CodeBERT tensors   → TransformerEncoder)

Two public entry points:
  process(sol_path)           — contract already on disk (CLI / batch use)
  process_source(source_code) — raw string from HTTP API or stdin
  process_source_windowed(source_code) — sliding-window variant for long
                                          contracts (see T1-C in improvement plan)

ARCHITECTURE: THIN WRAPPER
───────────────────────────
Graph construction logic now lives in the shared preprocessing package:

  ml/src/preprocessing/graph_schema.py    — constants (NODE_TYPES, VISIBILITY_MAP, …)
  ml/src/preprocessing/graph_extractor.py — extract_contract_graph(), typed exceptions

This file is responsible only for:
  • Slither temp-file management (process_source writes a NamedTemporaryFile
    because solc requires a real path — it cannot read from a pipe)
  • Exception translation: GraphExtractionError → ValueError / RuntimeError
    matching the HTTP 400 / HTTP 500 boundary expected by api.py
  • CodeBERT tokenization (_tokenize, _tokenize_sliding_window)
  • Hashing using hash_utils (replaces the previous inline hashlib.md5 calls)
  • Response logging

WHY WE DON'T USE graph_builder.GraphBuilder HERE
─────────────────────────────────────────────────
graph_builder.py uses one-hot encoding for node features:
  type_onehot(4) + visibility_onehot(4) + mutability_onehot(6) + flags(3) = 17 dims

The training data (68K .pt files) was built by ast_extractor.py, which uses
NODE_FEATURE_DIM raw floats per node (currently 13 in v5, was 8 in v1/v4).
GNNEncoder.conv1 reads in_channels=NODE_FEATURE_DIM from graph_schema —
passing a different-dim feature vector causes a mat-mul shape error at runtime.

graph_builder.py is kept for potential future use but must NOT be called
from this file.

WHY SLITHER REQUIRES A TEMP FILE FOR process_source()
──────────────────────────────────────────────────────
Slither shells out to solc, which requires a real file path — it cannot read
from stdin or a string argument. process_source() writes a NamedTemporaryFile,
calls extract_contract_graph() on it, then deletes it in a finally block.
The tokenizer is called directly on the string — no redundant file read.

SHAPE CONTRACT  (must match training data — do not change without retraining)
──────────────────────────────────────────────────────────────────────────────
  graph.x                  [N, NODE_FEATURE_DIM]  float32  (13 in v5; was 8 in v4)
  graph.edge_index         [2, E]                 int64
  tokens["input_ids"]      [1, 512]   long    (single window)
  tokens["attention_mask"] [1, 512]   long

  process_source_windowed() produces a list of token dicts, each [1, 512].
"""

from __future__ import annotations

import atexit
import glob
import os
import tempfile
from pathlib import Path

import torch
from loguru import logger
from torch_geometric.data import Data
from transformers import AutoTokenizer

from ..preprocessing.graph_extractor import (
    EmptyGraphError,
    GraphExtractionConfig,
    GraphExtractionError,
    SlitherParseError,
    SolcCompilationError,
    extract_contract_graph,
)
from ..preprocessing.graph_schema import FEATURE_SCHEMA_VERSION
from ..utils.hash_utils import get_contract_hash, get_contract_hash_from_content
from .cache import InferenceCache

# ---------------------------------------------------------------------------
# SIGKILL-safe temp file management (Audit #9)
# ---------------------------------------------------------------------------
# SIGKILL cannot be caught. We use two complementary strategies:
#   1. atexit handler — runs on normal exit and SIGTERM; cleans in-flight temps.
#   2. Startup scan   — purges orphaned files left by a prior SIGKILL.
# Both strategies rely on the fixed prefix "sentinel_prep_" to identify files.

_SENTINEL_TMP_PREFIX = "sentinel_prep_"
_active_temp_files: set[str] = set()


def _cleanup_active_temp_files() -> None:
    """atexit handler: remove any temp files that are still registered."""
    for path in list(_active_temp_files):
        try:
            os.unlink(path)
        except OSError:
            pass


atexit.register(_cleanup_active_temp_files)


def _purge_orphaned_sentinel_temps() -> None:
    """
    Delete any sentinel_prep_*.sol files left in the system temp dir by a
    previous process that was killed with SIGKILL.  Called once at startup.
    """
    pattern = os.path.join(tempfile.gettempdir(), f"{_SENTINEL_TMP_PREFIX}*.sol")
    for orphan in glob.glob(pattern):
        try:
            os.unlink(orphan)
            logger.debug(f"Purged orphaned temp file from previous run: {orphan}")
        except OSError:
            pass


class ContractPreprocessor:
    """
    Converts a Solidity contract into (graph, tokens) for SentinelModel.

    Instantiate once per process; reuse across many requests.
    The only expensive initialisation is AutoTokenizer.from_pretrained()
    which downloads/caches the CodeBERT vocab on first call.

    Args:
        cache: Optional InferenceCache. When supplied, process_source() checks
               the cache before running Slither and writes on miss. Eliminates
               the 3-5s Slither cost for repeated contracts (T1-A).

    Public API:
      process(sol_path)            — contract file already on disk
      process_source(source_code)  — raw Solidity string (HTTP API, stdin)
      process_source_windowed(src) — sliding-window for long contracts (T1-C)
    """

    TOKENIZER_NAME   = "microsoft/codebert-base"
    MAX_TOKEN_LENGTH = 512

    # Reject source_code over this size before any Slither/tokenizer work begins.
    # api.py also enforces this at the HTTP boundary; this is a defence-in-depth guard.
    MAX_SOURCE_BYTES = 1 * 1024 * 1024  # 1 MB

    def __init__(self, cache: InferenceCache | None = None) -> None:
        logger.info("ContractPreprocessor initialising...")
        _purge_orphaned_sentinel_temps()  # Audit #9: clean up SIGKILL survivors
        # Load CodeBERT tokenizer once — reused for every request.
        # Must match tokenizer_v1_production.py used during offline preprocessing.
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)
        self._cache    = cache
        logger.info(
            f"ContractPreprocessor ready | cache={'enabled' if cache else 'disabled'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, sol_path: str | Path) -> tuple[Data, dict]:
        """
        Convert a .sol file on disk into (graph, tokens) for SentinelModel.

        Hashing: MD5 of the contract file path via hash_utils.get_contract_hash(),
        matching the offline pipeline (ast_extractor.py → get_contract_hash).

        Args:
            sol_path: Path to the Solidity source file.

        Returns:
            (graph, tokens) — see process_source() for shape details.

        Raises:
            FileNotFoundError: if sol_path does not exist.
            ValueError:        if Solidity compilation or graph extraction fails.
            RuntimeError:      if Slither is not installed.
        """
        sol_path = Path(sol_path)
        if not sol_path.exists():
            raise FileNotFoundError(f"Contract not found: {sol_path}")

        logger.info(f"Preprocessing file: {sol_path.name}")

        contract_hash = f"{get_contract_hash(sol_path)}_{FEATURE_SCHEMA_VERSION}"
        graph  = self._extract_graph(sol_path, contract_hash)
        tokens = self._tokenize(
            sol_path.read_text(encoding="utf-8", errors="ignore"),
            contract_hash,
        )

        self._log_result(graph, tokens)
        return graph, tokens

    def process_source(
        self,
        source_code: str,
        name:        str = "contract",
    ) -> tuple[Data, dict]:
        """
        Convert a raw Solidity string into (graph, tokens) for SentinelModel.

        Writes source to a NamedTemporaryFile, runs graph extraction, deletes
        the file in a finally block, then tokenizes the in-memory string.

        Hashing: MD5 of source content via hash_utils.get_contract_hash_from_content().
        Same source string always produces the same hash (content-addressable),
        enabling the inference cache layer (T1-A) to key on this value.
        Cache key format: "{content_md5}_{FEATURE_SCHEMA_VERSION}"

        Args:
            source_code: Raw Solidity source text.
            name:        Label for log messages and temp-file prefix (sanitised).

        Returns:
            graph: PyG Data object.
                graph.x              [N, NODE_FEATURE_DIM]  float32
                graph.edge_index     [2, E]                 int64
                graph.contract_hash  str
            tokens: dict
                "input_ids"          [1, 512] long
                "attention_mask"     [1, 512] long
                "contract_hash"      str
                "num_tokens"         int
                "truncated"          bool

        Raises:
            ValueError: source_code is empty, too large, or Solidity is invalid.
            RuntimeError: Slither is not installed or infrastructure failure.
        """
        if not source_code or not source_code.strip():
            raise ValueError("source_code is empty")

        source_bytes = len(source_code.encode("utf-8"))
        if source_bytes > self.MAX_SOURCE_BYTES:
            raise ValueError(
                f"source_code too large ({source_bytes:,} bytes > "
                f"{self.MAX_SOURCE_BYTES:,} limit). "
                "Consider splitting or summarising the contract before analysis."
            )

        logger.info(f"Preprocessing source: {name!r} ({len(source_code)} chars)")

        content_hash  = get_contract_hash_from_content(source_code)
        # Full cache key includes schema version so stale cached graphs are never
        # loaded after a feature-engineering change.
        contract_hash = f"{content_hash}_{FEATURE_SCHEMA_VERSION}"

        # ── T1-A: Cache lookup ─────────────────────────────────────────────
        # Hit: skip Slither (3-5 s) and tokenizer; return cached tensors directly.
        # Miss: run full pipeline and write to cache at the end.
        if self._cache is not None:
            cached = self._cache.get(contract_hash)
            if cached is not None:
                graph, tokens = cached
                logger.info(f"Cache hit — skipped Slither for {name!r}")
                return graph, tokens

        # Sanitise name for use as a temp-file prefix.
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name[:32])

        # delete=False: we must close the file before Slither opens it (required
        # on some platforms). The finally block guarantees cleanup.
        # Fixed prefix lets _purge_orphaned_sentinel_temps() find leftovers on restart.
        tmp = tempfile.NamedTemporaryFile(
            suffix=".sol",
            prefix=_SENTINEL_TMP_PREFIX,
            mode="w",
            encoding="utf-8",
            delete=False,
        )
        _active_temp_files.add(tmp.name)
        try:
            tmp.write(source_code)
            tmp.close()
            graph = self._extract_graph(Path(tmp.name), contract_hash)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError as exc:
                logger.warning(f"Failed to delete temp file {tmp.name}: {exc}")
            _active_temp_files.discard(tmp.name)

        tokens = self._tokenize(source_code, contract_hash)

        # ── T1-A: Cache write ──────────────────────────────────────────────
        if self._cache is not None:
            self._cache.put(contract_hash, graph, tokens)

        self._log_result(graph, tokens)
        return graph, tokens

    def process_source_windowed(
        self,
        source_code: str,
        name:        str = "contract",
        stride:      int = 256,
        max_windows: int = 8,
    ) -> tuple[Data, list[dict]]:
        """
        Sliding-window variant of process_source() for long contracts (T1-C).

        Contracts longer than ~400 tokens have their tail silently truncated by
        process_source(). Functions defined late in the file — complex logic,
        withdrawal patterns — become invisible to CodeBERT.

        This method splits the token sequence into overlapping 512-token windows
        with the given stride, producing one token dict per window. The GNN graph
        is built once (it sees the full AST regardless of token length). The
        caller (predictor.py) aggregates per-window probabilities via max pooling.

        Args:
            source_code: Raw Solidity source.
            name:        Log label and temp-file prefix.
            stride:      Token overlap between consecutive windows (default 256
                         = 50% overlap, balancing coverage and compute).
            max_windows: Hard cap on windows to bound inference latency.
                         Contracts exceeding this are truncated at the window level.

        Returns:
            (graph, windows) where:
              graph   — PyG Data (same as process_source; built once)
              windows — list of token dicts, each {"input_ids": [1,512],
                        "attention_mask": [1,512], "contract_hash": str,
                        "num_tokens": int, "truncated": bool, "window_index": int}
                        Length is always ≥ 1 (short contracts return [single_dict]).

        Raises:
            ValueError, RuntimeError — same as process_source().
        """
        # Build the graph via process_source to reuse temp-file management and
        # exception translation. Then replace the single token dict with windows.
        graph, single_tokens = self.process_source(source_code, name)
        windows = self._tokenize_sliding_window(
            source_code,
            single_tokens["contract_hash"],
            stride=stride,
            max_windows=max_windows,
        )
        return graph, windows

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_graph(self, sol_path: Path, contract_hash: str) -> Data:
        """
        Call extract_contract_graph() and translate typed exceptions for the API.

        GraphExtractionError subclass mapping:
          SolcCompilationError → ValueError  (HTTP 400 — user's Solidity is invalid)
          EmptyGraphError      → ValueError  (HTTP 400 — file has no analyzable code)
          SlitherParseError    → RuntimeError (HTTP 500 — infrastructure failure)
          RuntimeError         → re-raised   (HTTP 500 — Slither not installed)

        Attaches inference-specific metadata after extraction:
          graph.contract_hash — content-addressed MD5 + schema version
          graph.contract_path — resolved absolute path (provenance)
          graph.y             — dummy label tensor (never used in forward pass;
                                required by some PyG internals)
        """
        config = GraphExtractionConfig()  # online defaults: system solc, no allow-paths

        try:
            graph = extract_contract_graph(sol_path, config)
        except SolcCompilationError as exc:
            raise ValueError(
                f"Solidity compilation error in '{sol_path.name}': {exc}"
            ) from exc
        except EmptyGraphError as exc:
            raise ValueError(
                f"No analyzable contract nodes in '{sol_path.name}': {exc}"
            ) from exc
        except SlitherParseError as exc:
            raise RuntimeError(
                f"Graph extraction infrastructure failure for '{sol_path.name}': {exc}"
            ) from exc
        except GraphExtractionError as exc:
            # Catch-all for any future GraphExtractionError subclasses.
            raise RuntimeError(
                f"Unexpected graph extraction error for '{sol_path.name}': {exc}"
            ) from exc
        # RuntimeError (Slither not installed) propagates unchanged.

        # Attach inference-specific fields that the shared extractor does not set.
        graph.contract_hash = contract_hash
        graph.contract_path = str(sol_path.resolve())
        graph.y = torch.tensor([0], dtype=torch.long)  # dummy; not used in forward

        return graph

    def _tokenize(self, source_code: str, contract_hash: str) -> dict:
        """
        Tokenize Solidity source with CodeBERT (single window, 512 tokens).

        Pure function — no file I/O. Called by process() and process_source().

        Settings match tokenizer_v1_production.py exactly:
          max_length=512, truncation=True, padding="max_length"
        Changing these requires rebuilding token files and retraining.

        Truncation detection uses a separate encode() call (without truncation)
        to get the true token count. Decoding + comparing is unreliable because
        the tokenizer normalises whitespace on the round-trip.

        Returns:
            dict with keys:
              "input_ids"      [1, 512] long
              "attention_mask" [1, 512] long
              "contract_hash"  str
              "num_tokens"     int   (non-padding tokens in this window)
              "truncated"      bool  (True if source exceeded 512 tokens)
        """
        encoded = self.tokenizer(
            source_code,
            max_length=self.MAX_TOKEN_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        num_tokens: int  = int(encoded["attention_mask"].sum().item())
        true_count:  int = len(self.tokenizer.encode(source_code, add_special_tokens=True))
        truncated:   bool = true_count > self.MAX_TOKEN_LENGTH

        if truncated:
            logger.debug(
                f"Truncated: {true_count} tokens → {self.MAX_TOKEN_LENGTH} "
                f"(lost {true_count - self.MAX_TOKEN_LENGTH} tokens from tail)"
            )

        # Explicit shape guards — RuntimeError (not assert) so -O cannot strip them.
        expected = torch.Size([1, self.MAX_TOKEN_LENGTH])
        if encoded["input_ids"].shape != expected:
            raise RuntimeError(
                f"input_ids shape {encoded['input_ids'].shape} != {expected}. "
                f"Check tokenizer settings: name={self.TOKENIZER_NAME}, "
                f"max_length={self.MAX_TOKEN_LENGTH}."
            )
        if encoded["attention_mask"].shape != expected:
            raise RuntimeError(
                f"attention_mask shape {encoded['attention_mask'].shape} != {expected}."
            )

        return {
            "input_ids":      encoded["input_ids"],       # [1, 512]
            "attention_mask": encoded["attention_mask"],  # [1, 512]
            "contract_hash":  contract_hash,
            "num_tokens":     num_tokens,
            "truncated":      truncated,
        }

    def _tokenize_sliding_window(
        self,
        source_code:    str,
        contract_hash:  str,
        stride:         int = 256,
        max_windows:    int = 8,
    ) -> list[dict]:
        """
        Tokenize source_code into overlapping 512-token windows (T1-C).

        Short contracts (≤ 512 tokens) return a list of exactly one dict —
        identical to _tokenize() output plus a "window_index" key. There is
        no overhead compared to the single-window path.

        Algorithm:
          1. Encode without truncation to get the full token ID sequence.
          2. Slide a window of MAX_TOKEN_LENGTH tokens with the given stride.
          3. Pad each window to MAX_TOKEN_LENGTH.
          4. Cap at max_windows to bound latency (contracts rarely need > 4).

        Args:
            source_code:   Raw Solidity source.
            contract_hash: Shared hash attached to every window dict.
            stride:        Tokens to advance per window (256 = 50% overlap).
            max_windows:   Hard cap on window count.

        Returns:
            List of dicts; each has the same keys as _tokenize() plus
            "window_index" (int, 0-based).
        """
        # Fix E1 (C4): encode content tokens only (no special tokens) so we can
        # reconstruct proper [CLS]…[SEP] framing for EVERY window, not just the first.
        #
        # Bug in original code: tokenizer.encode(src, add_special_tokens=True) gave
        #   [CLS] c1 c2 … cN [SEP]
        # Slicing full_ids[256:768] for window 2 produced c257…c767 — a bare token
        # sequence with no [CLS] at position 0 and no [SEP] at the end.
        # CodeBERT was pre-trained to always see [CLS]…[SEP]; the CLS embedding at
        # position 0 is used by the TransformerEncoder "tf eye" for contract-level
        # representation. Non-first windows in the old code gave a random mid-contract
        # token in the CLS slot, making those windows nearly useless for prediction.
        #
        # Fix: encode without special tokens → slide over pure content → prepend [CLS]
        # and append [SEP] to each window individually.  Content window capacity is
        # MAX_TOKEN_LENGTH - 2 (512 - 2 = 510) to leave room for the two special tokens.
        full_content_ids = self.tokenizer.encode(source_code, add_special_tokens=False)
        _CONTENT_CAP     = self.MAX_TOKEN_LENGTH - 2  # 510 content tokens per window
        total_content    = len(full_content_ids)

        if total_content <= _CONTENT_CAP:
            # Fast path: whole source fits in one window. Delegate to _tokenize()
            # which produces the standard [CLS] c1…c510 [SEP] [PAD]… encoding.
            single = self._tokenize(source_code, contract_hash)
            single["window_index"] = 0
            return [single]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        windows: list[dict] = []
        start = 0
        while start < total_content and len(windows) < max_windows:
            content_chunk = full_content_ids[start : start + _CONTENT_CAP]
            # Wrap each window with [CLS] … [SEP] so every window is a
            # self-contained sequence CodeBERT was trained to handle.
            chunk   = [cls_id] + content_chunk + [sep_id]
            pad_len = self.MAX_TOKEN_LENGTH - len(chunk)

            input_ids      = torch.tensor([chunk + [self.tokenizer.pad_token_id] * pad_len], dtype=torch.long)
            attention_mask = torch.tensor([[1] * len(chunk) + [0] * pad_len],               dtype=torch.long)

            end_content = start + _CONTENT_CAP
            windows.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "contract_hash":  contract_hash,
                "num_tokens":     len(chunk),
                "truncated":      (end_content < total_content),
                "window_index":   len(windows),
            })

            if end_content >= total_content:
                break
            start += stride

        return windows

    def _log_result(self, graph: Data, tokens: dict) -> None:
        logger.info(
            f"Preprocessing complete — "
            f"nodes: {graph.num_nodes}, "
            f"edges: {graph.num_edges}, "
            f"tokens: {tokens['num_tokens']}, "
            f"truncated: {tokens['truncated']}"
        )
