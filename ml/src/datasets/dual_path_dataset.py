"""
Dual-Path Dataset for SENTINEL vulnerability detection.

Loads paired graph (PyG) and token (CodeBERT) files for training.
Handles train/val/test splits via pre-computed index arrays.

Data layout on disk:
    ml/data/graphs/<md5_hash>.pt  →  PyG Data object (x, edge_index, edge_attr, y)
    ml/data/tokens/<md5_hash>.pt  →  dict {input_ids: [512], attention_mask: [512]}

The MD5 hash is the pairing key — graph and token files with the same stem
belong to the same contract.

LABEL MODES
───────────

Binary mode (label_csv=None, default):
    Labels come from graph.y — scalar 0/1 long tensor.
    Collate produces [B] long. Used for binary training and inference with
    old checkpoints.

Multi-label mode (label_csv=Path(...)):
    Labels come from multilabel_index.csv — float32 tensor [10].
    Each position is 0.0 or 1.0 for one of the 10 vulnerability classes:
      0=CallToUnknown  1=DenialOfService  2=ExternalBug    3=GasException
      4=IntegerUO      5=MishandledException  6=Reentrancy  7=Timestamp
      8=TransactionOrderDependence  9=UnusedReturn
    Collate produces [B, 10] float32. Used for Track 3 multi-label retrain.

RAM CACHE
─────────

Pass cache_path=Path("ml/data/cached_dataset.pkl") to __init__ to use a
pre-built pickle that maps each hash to its (graph, token) pair.
If present, __getitem__ reads from the dict instead of individual .pt files,
reducing per-epoch I/O from hours to minutes.
Create the cache once with create_cache.py.

BONUS FIX (not in reviewer list):
    The original code had a hardcoded absolute cache path
    (/home/motafeq/projects/sentinel/...) which silently missed the cache on
    every machine except the original author's. cache_path is now an explicit
    __init__ argument (default None = no cache). Callers opt in deliberately.
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.serialization
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

# Imported for cache schema version validation (Fix D2/H15).
# Only used in __init__ when a cache_path is supplied.
try:
    from ..preprocessing.graph_schema import FEATURE_SCHEMA_VERSION as _FEATURE_SCHEMA_VERSION
except ImportError:
    _FEATURE_SCHEMA_VERSION = None  # fallback: skip schema check if import fails

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch 2.6+ safe-globals allowlist
# ---------------------------------------------------------------------------
# These classes appear inside .pt graph files saved by the AST extractor.
# Registering them here allows weights_only=True (safe deserialization) to
# work without disabling the pickle-security check entirely.
#
# Audit fix #3 (2026-05-01): safe globals were already registered but
# weights_only=False was still being used. Now switched to weights_only=True.
# If a new PyG release adds more wrapper classes that cause UnpicklingError,
# add them to this list rather than reverting to weights_only=False.
torch.serialization.add_safe_globals([
    Data,
    DataEdgeAttr,
    DataTensorAttr,
    GlobalStorage,
])


class DualPathDataset(Dataset):
    """
    Paired graph + token dataset for SENTINEL.

    Each sample is a smart contract represented two ways:
        - Graph  (.pt): PyG Data object — AST/CFG structure → GNNEncoder
        - Tokens (.pt): dict with CodeBERT token tensors   → TransformerEncoder

    Files are matched by MD5 hash (filename stem). Only hashes that exist
    in BOTH directories are included; unmatched files are logged and skipped.

    Loading is lazy — files are read from disk in __getitem__, not __init__.
    This keeps memory usage flat regardless of dataset size.

    Args:
        graphs_dir: Directory containing <hash>.pt graph files.
        tokens_dir: Directory containing <hash>.pt token files.
        indices:    Optional list of integer positions into the full sorted
                    paired-hash list. Used to enforce train/val/test splits.
                    If None, all paired samples are used.
        validate:   If True, loads sample[0] during __init__ to catch
                    file-format issues before training starts.
        label_csv:  Path to multilabel_index.csv for multi-label mode.
                    If None (default), labels come from graph.y (binary mode).
        cache_path: Optional path to a pre-built pickle cache file created by
                    create_cache.py. If provided and the file exists, all
                    __getitem__ calls read from this in-memory dict instead
                    of individual .pt files (much faster per-epoch I/O).
                    Default None = read individual files from disk.
    """

    def __init__(
        self,
        graphs_dir:  str,
        tokens_dir:  str,
        indices:     Optional[List[int]] = None,
        validate:    bool                = True,
        label_csv:   Optional[Path]      = None,
        cache_path:  Optional[Path]      = None,   # explicit opt-in (was hardcoded)
    ) -> None:
        self.graphs_dir = Path(graphs_dir)
        self.tokens_dir = Path(tokens_dir)

        # ── Multi-label mode setup ──────────────────────────────────────────────────
        self._label_map: Optional[Dict[str, torch.Tensor]] = None
        if label_csv is not None:
            label_csv = Path(label_csv)
            logger.info(f"Multi-label mode — loading label CSV: {label_csv}")
            df = pd.read_csv(label_csv)
            class_cols = [c for c in df.columns if c != "md5_stem"]
            label_matrix = torch.tensor(
                df[class_cols].values.astype("float32"), dtype=torch.float32
            )
            stems = df["md5_stem"].tolist()
            self._label_map = {
                stem: label_matrix[i]
                for i, stem in enumerate(stems)
            }
            logger.info(
                f"Label map loaded — {len(self._label_map)} entries, "
                f"{len(class_cols)} classes"
            )

        # ── Discover files and compute paired set ─────────────────────────────────
        graph_files = list(self.graphs_dir.glob("*.pt"))
        token_files = list(self.tokens_dir.glob("*.pt"))
        logger.info(
            f"Found {len(graph_files)} graph files, {len(token_files)} token files"
        )

        graph_hashes   = {f.stem for f in graph_files}
        token_hashes   = {f.stem for f in token_files}
        paired_hashes  = graph_hashes & token_hashes
        unpaired_graphs = len(graph_hashes - token_hashes)
        unpaired_tokens = len(token_hashes - graph_hashes)

        logger.info(f"Paired samples: {len(paired_hashes)}")
        if unpaired_graphs > 0:
            logger.warning(
                f"{unpaired_graphs} graph files have no matching token file — skipped"
            )
        if unpaired_tokens > 0:
            # Phase 2-B4 (2026-05-14): downgraded WARNING → DEBUG.
            # The ~24K extra tokens from the original 68K dataset were moved to
            # ml/data/tokens_orphaned/ — they no longer appear here under normal
            # operation. This log fires only if someone manually adds token files
            # without corresponding graphs.
            logger.debug(
                f"{unpaired_tokens} token files have no matching graph file — skipped. "
                f"(Orphaned tokens should live in ml/data/tokens_orphaned/, not ml/data/tokens/)"
            )

        # Sort for deterministic indexing across runs
        self.paired_hashes = sorted(paired_hashes)

        # Apply split indices if provided
        if indices is not None:
            if len(indices) == 0:
                raise ValueError("indices list is empty — nothing to load")
            max_idx = max(indices)
            if max_idx >= len(self.paired_hashes):
                raise ValueError(
                    f"Index {max_idx} out of range for dataset of size "
                    f"{len(self.paired_hashes)}"
                )
            self.paired_hashes = [self.paired_hashes[i] for i in indices]
            logger.info(f"Split applied: {len(self.paired_hashes)} samples selected")

        # ── RAM Cache ───────────────────────────────────────────────────────────
        # Bonus fix: was a hardcoded absolute path — now an explicit argument.
        # Callers who want the cache pass cache_path=Path("ml/data/cached_dataset.pkl").
        # Callers who don't pass nothing; they will never silently miss a cache.
        self.cached_data = None
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    self.cached_data = pickle.load(f)

                # Type guard.
                if not isinstance(self.cached_data, dict):
                    raise RuntimeError(
                        f"RAM cache at {cache_path} is malformed — "
                        f"expected dict, got {type(self.cached_data).__name__}. "
                        "Delete the cache file and re-run create_cache.py."
                    )

                # Fix D2 (H15): validate schema version stored in the cache.
                # create_cache.py writes "__schema_version__" into the dict so we can
                # detect stale caches after a feature-engineering change without
                # relying on downstream inference errors.
                _cached_schema = self.cached_data.get("__schema_version__")
                if _cached_schema is not None and _FEATURE_SCHEMA_VERSION is not None:
                    if _cached_schema != _FEATURE_SCHEMA_VERSION:
                        raise RuntimeError(
                            f"RAM cache schema mismatch: cache has version={_cached_schema!r} "
                            f"but current FEATURE_SCHEMA_VERSION={_FEATURE_SCHEMA_VERSION!r}. "
                            "Delete the cache file and re-run create_cache.py to rebuild with "
                            "the current schema."
                        )
                elif _cached_schema is None and _FEATURE_SCHEMA_VERSION is not None:
                    # Old cache without version key — warn but don't crash.
                    # The stale-hash check below will catch the most severe corruption.
                    logger.warning(
                        f"RAM cache at {cache_path} has no '__schema_version__' key. "
                        f"It may be stale (current schema: {_FEATURE_SCHEMA_VERSION}). "
                        "Rebuild with create_cache.py to silence this warning."
                    )

                # Fix D1 (H14): random 10-hash integrity sample instead of
                # single spot-check on paired_hashes[0].  A single check can pass
                # even when the majority of the cache is stale or misaligned —
                # the first hash in sorted order tends to be stable across builds
                # (same source, same MD5), so it would survive a partial rebuild.
                # Sampling 10 random hashes makes undetected staleness 10× less likely.
                if self.paired_hashes:
                    _sample_size  = min(10, len(self.paired_hashes))
                    _sample_hashes = random.sample(self.paired_hashes, _sample_size)
                    for _spot in _sample_hashes:
                        if _spot not in self.cached_data:
                            raise RuntimeError(
                                f"RAM cache is stale — hash {_spot!r} not found "
                                f"(sampled 1 of {_sample_size} checks failed). "
                                "Delete the cache file and re-run create_cache.py."
                            )
                        try:
                            _g, _t = self.cached_data[_spot]
                            if not hasattr(_g, "x"):
                                raise ValueError("cached graph missing 'x' attribute")
                            if "input_ids" not in _t:
                                raise ValueError("cached tokens missing 'input_ids'")
                        except (ValueError, TypeError) as _exc:
                            raise RuntimeError(
                                f"RAM cache entry for {_spot!r} is malformed: {_exc}. "
                                "Delete the cache file and re-run create_cache.py."
                            ) from _exc

                logger.info(
                    f"Loaded {len(self.cached_data)} samples from RAM cache "
                    f"({cache_path})"
                )
            else:
                logger.warning(
                    f"cache_path={cache_path} was provided but does not exist. "
                    "Falling back to per-file disk reads. "
                    "Run create_cache.py to build the cache."
                )
        else:
            logger.info("No cache_path provided; reading individual .pt files from disk")

        # ── Eager validation ──────────────────────────────────────────────────────
        if validate and len(self.paired_hashes) > 0:
            try:
                _ = self[0]
                logger.info("Dataset validation passed — first sample loaded OK")
            except Exception as exc:
                logger.error(f"Dataset validation failed on first sample: {exc}")
                raise

    def __len__(self) -> int:
        return len(self.paired_hashes)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Data, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load one (graph, tokens, label) sample.

        Priority:
            1. If RAM cache was provided and loaded, read from self.cached_data.
            2. Otherwise, read the two .pt files from disk.
        """
        hash_id = self.paired_hashes[idx]

        # ── Load graph and tokens ───────────────────────────────────────────────
        if self.cached_data is not None and hash_id in self.cached_data:
            graph, tokens = self.cached_data[hash_id]
        else:
            # Cache miss (hash injected after cache was built) → load from disk
            graph_path = self.graphs_dir / f"{hash_id}.pt"
            token_path = self.tokens_dir / f"{hash_id}.pt"
            # ── Audit fix #3 (2026-05-01): weights_only=True ──────────────────────
            # Safe globals are registered at module level above. This call now
            # uses safe (non-pickle) deserialization. If a future PyG update
            # adds new internal classes that cause UnpicklingError, add them
            # to the add_safe_globals() list above — do NOT revert this flag.
            graph  = torch.load(graph_path, weights_only=True)
            tokens = torch.load(token_path, weights_only=True)

        # ── Extract label ────────────────────────────────────────────────────────
        if self._label_map is not None:
            # Multi-label mode: float32 [10]
            if hash_id not in self._label_map:
                raise KeyError(f"Hash {hash_id} not found in label_csv")
            label = self._label_map[hash_id]
        else:
            # Binary mode: label comes from graph.y
            if not hasattr(graph, "y") or graph.y is None:
                raise KeyError(f"No label (y) for hash {hash_id}")
            label = graph.y
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)
            label = label.view(1).long()  # [1] long

        # ── Validate token shapes ───────────────────────────────────────────────
        if tokens["input_ids"].shape != torch.Size([512]):
            raise ValueError(
                f"input_ids shape {tokens['input_ids'].shape} != [512]"
            )
        if tokens["attention_mask"].shape != torch.Size([512]):
            raise ValueError(
                f"attention_mask shape {tokens['attention_mask'].shape} != [512]"
            )

        # ── Fix #1: Guard against pre-refactor .pt files with edge_attr [E, 1] ──
        # graph_schema.py warns that old files stored edge_attr as [E, 1].
        # GNNEncoder passes edge_attr into nn.Embedding() which strictly requires
        # a 1-D tensor of shape [E]. squeeze(-1) is a no-op on already-correct
        # [E] tensors, so this is safe for both old and new .pt files.
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            if graph.edge_attr.ndim > 1:
                graph.edge_attr = graph.edge_attr.squeeze(-1)

        return graph, tokens, label


# ---------------------------------------------------------------------------
# Collate function — must be module-level for DataLoader multiprocessing
# ---------------------------------------------------------------------------

def dual_path_collate_fn(
    batch: List[Tuple[Data, Dict[str, torch.Tensor], torch.Tensor]],
) -> Tuple[Batch, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Collate a list of (graph, tokens, label) samples into batch tensors.

    PyG graphs are variable-size; Batch.from_data_list() merges them into a
    single disconnected graph with a `batch` index tensor.
    Token tensors are fixed-size [512] and stack normally.
    Labels: multi-label keeps [B, 10] float32; binary squeezes to [B] long.
    """
    graphs = [item[0] for item in batch]
    tokens = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Exclude non-tensor metadata attributes so Batch.from_data_list() doesn't
    # fail when stale graphs (280 v5.0-era .pt files) have extra fields like
    # contract_hash/contract_path/y that freshly-extracted graphs lack.
    # The model only needs x, edge_index, edge_attr — batch is set automatically.
    _EXCLUDE = ["contract_hash", "contract_path", "contract_name",
                "node_metadata", "num_edges", "num_nodes", "y"]
    batched_graphs: Batch = Batch.from_data_list(graphs, exclude_keys=_EXCLUDE)

    batched_tokens: Dict[str, torch.Tensor] = {
        "input_ids":      torch.stack([t["input_ids"]      for t in tokens]),
        "attention_mask": torch.stack([t["attention_mask"] for t in tokens]),
    }

    stacked = torch.stack(labels)
    first_label = labels[0]
    if first_label.dim() == 1 and first_label.shape[0] > 1:
        # Multi-label: [B, num_classes] float32
        batched_labels = stacked
    else:
        # Binary: [B, 1] long → squeeze dim 1 → [B] long
        batched_labels = stacked.squeeze(1)

    return batched_graphs, batched_tokens, batched_labels
